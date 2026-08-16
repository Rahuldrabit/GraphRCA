import os
import re
import csv
import glob
import json
import logging
import statistics
from typing import List, Dict, Any, Optional, Tuple

from swarm_state import AIOpsIncidentState
from tools.scratchpad_client import ScratchpadClient

logger = logging.getLogger(__name__)

# Tokens used as triplet *targets* (never suspects). The diagnoser filters these
# out of the suspect list so only real service nodes get nominated.
_TOKEN_PREFIXES = ("HTTP_", "LOG_", "ERROR", "TIMEOUT", "HIGH_", "POD_", "STATUS_")

# Pod statuses that count as healthy in `kubectl get pods`.
_HEALTHY_POD_STATUS = {"Running", "Completed", "Succeeded"}

# Infra/observability pods that are never an app-level root cause.
_INFRA_SVCS = {
    "jaeger", "prometheus", "loadbalancer", "nginx", "istio", "loki",
    "grafana", "otel", "opentelemetry", "chaos", "kube", "metrics-server",
    "wrk2-job", "wrk", "loadgenerator",  # workload generators
}


def _is_token(node: str) -> bool:
    if not node:
        return True
    return any(node.startswith(p) for p in _TOKEN_PREFIXES)


class ObserverAgent:
    """
    Non-LLM rule engine.
    Parses AIOpsLab telemetry (traces, pod status, metrics, logs) into L1 triplets
    and writes them to the ScratchPad knowledge graph.

    Service names are preserved in their ORIGINAL case/format (e.g. "geo",
    "user-service") so they match AIOpsLab's exact localization scoring. Pod
    replica/hash suffixes are stripped so "search-7b5c6d-x9k2m" canonicalizes to
    "search", merging with the same service observed in traces.
    """

    def __init__(self, scratchpad_client: ScratchpadClient):
        self.client = scratchpad_client

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _canonical_svc(name: str) -> str:
        """Canonical service name: strip k8s pod replica/hash suffix, keep case.

        Handles:
          Deployment pods:  name-<hash9+>-<hash4+>   -> name
          StatefulSet pods: name-<ordinal>           -> name
          Bare service / trace svc:  name            -> name
        """
        if not name:
            return ""
        s = str(name).strip().strip('"').strip("'")
        if not s:
            return ""
        # Deployment: trailing <rs-hash>-<pod-hash>. ReplicaSet hashes can be as
        # short as ~5 chars (e.g. geo-c47ff745-snvqz), so allow {5,} not {9,}.
        m = re.match(r"^(.+)-[a-z0-9]{5,}-[a-z0-9]{4,}$", s)
        if m:
            return m.group(1)
        # StatefulSet: trailing ordinal
        m = re.match(r"^(.+)-(\d+)$", s)
        if m and m.group(1):
            return m.group(1)
        return s

    @staticmethod
    def _rel(count: int, cap: int = 10) -> float:
        """Map a signal count to a [0,1] salience for ScratchPad metadata."""
        try:
            return round(min(1.0, float(count) / float(cap)), 3) if cap > 0 else 1.0
        except Exception:
            return 1.0

    @staticmethod
    def _read_text(path_or_text: str) -> Tuple[str, bool]:
        """If the argument is a path to an existing file, read it; else treat as text.

        Returns (text, was_file)."""
        if not path_or_text:
            return "", False
        s = str(path_or_text).strip()
        if s and ("\n" not in s) and os.path.isfile(s):
            try:
                with open(s, "r", encoding="utf-8", errors="replace") as f:
                    return f.read(), True
            except Exception:
                return "", True
        return path_or_text, False

    # ── parsers (each returns list[triplet dicts] + sets nothing) ─────────

    def parse_traces(self, trace_csv_path_or_text: str) -> List[Dict[str, Any]]:
        """Parse Jaeger trace CSV into triplets.

        Rules:
          1. parent->child service  -> calls   (deduped, weighted by volume/errors)
          2. error spans            -> emits HTTP_<response>
          3. high-duration services -> blocks HIGH_LATENCY
        """
        triplets: List[Dict[str, Any]] = []
        text, _ = self._read_text(trace_csv_path_or_text)
        if not text or "span_id" not in text:
            return triplets

        lines = text.strip().split("\n")
        if len(lines) < 2:
            return triplets

        try:
            reader = csv.DictReader(lines)
            rows = [r for r in reader if r]
        except Exception as e:
            logger.warning(f"[Observer] trace CSV parse failed: {e}")
            return triplets
        if not rows:
            return triplets

        # span_id -> canonical service name
        span_to_svc: Dict[str, str] = {}
        for r in rows:
            sid = (r.get("span_id") or "").strip()
            svc = self._canonical_svc(r.get("service_name", ""))
            if sid and svc:
                span_to_svc[sid] = svc

        # Aggregate call edges + per-service stats.
        calls: Dict[Tuple[str, str], Dict[str, int]] = {}
        svc_err: Dict[str, int] = {}
        svc_dur: Dict[str, List[float]] = {}
        for r in rows:
            svc = self._canonical_svc(r.get("service_name", ""))
            if not svc:
                continue
            parent_id = (r.get("parent_span") or "").strip()
            if parent_id and parent_id in span_to_svc:
                psvc = span_to_svc[parent_id]
                if psvc and psvc != svc:
                    key = (psvc, svc)
                    d = calls.setdefault(key, {"vol": 0, "err": 0})
                    d["vol"] += 1
            has_err = str(r.get("has_error", "")).strip().lower() == "true"
            if has_err:
                svc_err[svc] = svc_err.get(svc, 0) + 1
                if parent_id and parent_id in span_to_svc:
                    psvc = span_to_svc[parent_id]
                    if psvc and psvc != svc:
                        calls.setdefault((psvc, svc), {"vol": 0, "err": 0})["err"] += 1
            dur = r.get("duration")
            try:
                svc_dur.setdefault(svc, []).append(float(dur))
            except (TypeError, ValueError):
                pass

        # Emit deduped calls edges.
        for (psvc, svc), d in calls.items():
            triplets.append({
                "source": psvc, "relationship": "calls", "target": svc,
                "citation_quote": f"trace call volume={d['vol']} errors={d['err']}",
                "relevance": self._rel(d["vol"]),
            })

        # Emit error signals.
        for svc, n in sorted(svc_err.items(), key=lambda kv: kv[1], reverse=True):
            triplets.append({
                "source": svc, "relationship": "emits", "target": "HTTP_ERROR",
                "citation_quote": f"trace error_spans={n}",
                "relevance": self._rel(n),
            })

        # Emit high-latency blocks (services well above the median duration).
        all_dur = [v for vals in svc_dur.values() for v in vals if v is not None]
        if all_dur:
            med = statistics.median(all_dur)
            for svc, vals in svc_dur.items():
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                avg = statistics.mean(vals)
                # >2x the global median AND >100 (ms) guards against noise.
                if avg >= max(100.0, 2.0 * med):
                    triplets.append({
                        "source": svc, "relationship": "blocks", "target": "HIGH_LATENCY",
                        "citation_quote": f"trace avg_duration={avg:.0f} median={med:.0f}",
                        "relevance": self._rel(int(avg / max(med, 1.0)), cap=8),
                    })
        return triplets

    def parse_kubectl(self, pod_status_path_or_text: str) -> List[Dict[str, Any]]:
        """Parse `kubectl get pods [-o wide]` output into triplets.

        Flags non-healthy pods, 0/N ready containers, and high restart counts.
        This is the most reliable signal for pod_failure / pod_kill /
        container_kill / scale_pod_zero / k8s misconfig faults (no traces then).
        """
        triplets: List[Dict[str, Any]] = []
        text, _ = self._read_text(pod_status_path_or_text)
        if not text or "READY" not in text:
            return triplets

        seen: Dict[str, Dict[str, Any]] = {}
        for line in text.strip().split("\n")[1:]:  # skip header
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            ready = parts[1] if len(parts) > 1 else ""
            status = parts[2] if len(parts) > 2 else ""
            restarts = parts[3] if len(parts) > 3 else "0"
            svc = self._canonical_svc(name)
            if not svc or svc.lower() in _INFRA_SVCS:
                continue

            bad = False
            reasons = []
            if status and status not in _HEALTHY_POD_STATUS:
                bad = True
                reasons.append(f"status={status}")
            # 0/N ready containers
            if "/" in ready:
                try:
                    up, tot = ready.split("/")
                    if int(up) == 0 and int(tot) > 0:
                        bad = True
                        reasons.append(f"ready={ready}")
                except Exception:
                    pass
            try:
                if int(restarts) >= 3:
                    bad = True
                    reasons.append(f"restarts={restarts}")
            except Exception:
                pass

            if bad:
                entry = seen.setdefault(svc, {"reasons": [], "n": 0})
                entry["reasons"].extend(reasons)
                entry["n"] += 1

        for svc, entry in seen.items():
            target = "POD_UNHEALTHY"
            triplets.append({
                "source": svc, "relationship": "emits", "target": target,
                "citation_quote": "kubectl pods " + ", ".join(sorted(set(entry["reasons"]))),
                "relevance": self._rel(entry["n"], cap=4),
            })
        return triplets

    def parse_metrics(self, metrics_path_or_text: str) -> List[Dict[str, Any]]:
        """Parse Prometheus metrics CSV(s) into triplets.

        Accepts either the combined summary CSV produced by the swarm fetcher
        (columns: metric,cmdb_id,kpi_name,value) or a metrics directory.
        Flags services with anomalously high CPU / memory values (>2x the
        per-metric mean). Best-effort: returns [] if the format is unexpected.
        """
        triplets: List[Dict[str, Any]] = []
        text, was_file = self._read_text(metrics_path_or_text)
        if not text:
            return triplets

        # Directory of kpi_*.csv files
        if was_file is False and os.path.isdir(metrics_path_or_text.strip()):
            return self._parse_metrics_dir(metrics_path_or_text.strip())

        rows: List[Dict[str, str]] = []
        # Combined summary CSV (metric,cmdb_id,kpi_name,value)
        if "cmdb_id" in text or "kpi_name" in text:
            try:
                reader = csv.DictReader(text.strip().split("\n"))
                rows = [r for r in reader if r]
            except Exception:
                rows = []

        # per-metric aggregated values: {metric_kind: {svc: [values]}}
        agg: Dict[str, Dict[str, List[float]]] = {}
        for r in rows:
            kpi = (r.get("kpi_name") or r.get("metric") or "").lower()
            svc = self._canonical_svc(self._svc_from_cmdb(r.get("cmdb_id", "")))
            try:
                val = float(r.get("value", "nan"))
            except (TypeError, ValueError):
                continue
            if not svc or val != val:  # NaN check
                continue
            kind = "cpu" if "cpu" in kpi else "mem" if "mem" in kpi else ""
            if not kind:
                continue
            agg.setdefault(kind, {}).setdefault(svc, []).append(val)

        for kind, svc_map in agg.items():
            all_vals = [v for vals in svc_map.values() for v in vals]
            if not all_vals:
                continue
            mean = statistics.mean(all_vals)
            for svc, vals in svc_map.items():
                if not vals:
                    continue
                mx = max(vals)
                if mx >= 2.0 * mean and mean > 0:
                    target = "HIGH_CPU" if kind == "cpu" else "HIGH_MEM"
                    triplets.append({
                        "source": svc, "relationship": "emits", "target": target,
                        "citation_quote": f"metric {kind} max={mx:.3f} mean={mean:.3f}",
                        "relevance": self._rel(int(mx / max(mean, 1e-9)), cap=5),
                    })
        return triplets

    def _parse_metrics_dir(self, dirpath: str) -> List[Dict[str, Any]]:
        triplets: List[Dict[str, Any]] = []
        csvs = glob.glob(os.path.join(dirpath, "**", "kpi_*.csv"), recursive=True)
        kind_map = {"cpu": [], "mem": []}
        for c in csvs[:40]:
            metric = os.path.basename(c)[4:-4].lower()
            kind = "cpu" if "cpu" in metric else "mem" if "mem" in metric else None
            if not kind:
                continue
            try:
                with open(c, encoding="utf-8", errors="replace") as f:
                    for r in csv.reader(f):
                        if len(r) >= 4:
                            kind_map[kind].append((r[1], r[2], r[3]))  # cmdb, kpi, value
            except Exception:
                continue

        for kind, recs in kind_map.items():
            agg: Dict[str, List[float]] = {}
            for cmdb, _kpi, val in recs:
                svc = self._canonical_svc(self._svc_from_cmdb(cmdb))
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if svc and v == v:
                    agg.setdefault(svc, []).append(v)
            allv = [v for vs in agg.values() for v in vs]
            if not allv:
                continue
            mean = statistics.mean(allv)
            for svc, vs in agg.items():
                if vs and max(vs) >= 2.0 * mean and mean > 0:
                    target = "HIGH_CPU" if kind == "cpu" else "HIGH_MEM"
                    triplets.append({
                        "source": svc, "relationship": "emits", "target": target,
                        "citation_quote": f"metric {kind} max={max(vs):.3f} mean={mean:.3f}",
                        "relevance": self._rel(int(max(vs) / max(mean, 1e-9)), cap=5),
                    })
        return triplets

    @staticmethod
    def _svc_from_cmdb(cmdb: str) -> str:
        """cmdb_id is 'instance.pod' — the service is the pod name (last segment)."""
        if not cmdb:
            return ""
        return cmdb.split(".")[-1] if "." in cmdb else cmdb

    def parse_logs(self, logs_path_or_text: str) -> List[Dict[str, Any]]:
        """Parse aggregated service logs into triplets.

        Expects the swarm fetcher's sectioned format:
            === <service> ===
            <log lines>
        Flags services emitting ERROR/FATAL/exception/timeout lines, AND flags
        connection failures (refused / failed-to-connect / ECONNREFUSED).

        Connection failures are attributed to the UNREACHABLE CALLEE — extracted
        from the log line — not to `cur_svc` (the caller that logs the failure).
        The unreachable service is the root cause of k8s port-misconfig / network
        faults; attributing its signal to the noisy caller (or dropping it
        entirely, as before) is why the root cause vanished from the graph and
        localization scored 0.0.
        """
        triplets: List[Dict[str, Any]] = []
        text, _ = self._read_text(logs_path_or_text)
        if not text:
            return triplets

        # Connection-failure signatures: a callee is unreachable (port misconfig,
        # network policy, crash). Previously the observer matched only
        # ERROR/FATAL/TIMEOUT, so the connection-refused signature of port /
        # network faults was never captured.
        _CONN_SIGS = ("CONNECTION REFUSED", "CONNECT REFUSED", "CONNECT()",
                      "FAILED TO CONNECT", "COULD NOT CONNECT", "CANNOT CONNECT",
                      "ECONNREFUSED", "CONNREFUSED", "NO ROUTE TO HOST",
                      "CONNECT: CONNECTION", "REFUSED")
        # Extract the unreachable callee host. Two common log shapes:
        #   Thrift:  ... connect() <Host: user-service Port: 9090>: Connection refused
        #   generic: Failed to connect user-service-client
        _CONN_HOST_RE = re.compile(r"Host:\s*([A-Za-z0-9_.-]+?)\s+Port", re.I)
        _CONN_NAME_RE = re.compile(
            r"(?:failed to connect|could not connect|cannot connect to|"
            r"connect to|connecting to)\s+([A-Za-z][\w.-]*?)"
            r"(?:-client)?(?:[\s.:]|$)", re.I)

        cur_svc = ""
        err_counts: Dict[str, int] = {}
        timeout_counts: Dict[str, int] = {}
        conn_counts: Dict[str, int] = {}
        for line in text.split("\n"):
            m = re.match(r"^===\s*(\S.*?)\s*===\s*$", line)
            if m:
                cur_svc = self._canonical_svc(m.group(1))
                continue
            if not cur_svc or cur_svc.lower() in _INFRA_SVCS:
                continue
            up = line.upper()
            if any(tok in up for tok in ("ERROR", "FATAL", "EXCEPTION", "PANIC", "TRACEBACK")):
                err_counts[cur_svc] = err_counts.get(cur_svc, 0) + 1
            if "TIMEOUT" in up or "TIMED OUT" in up or "DEADLINE" in up:
                timeout_counts[cur_svc] = timeout_counts.get(cur_svc, 0) + 1
            # Connection failure -> attribute to the unreachable CALLEE (root
            # cause), not cur_svc (the caller logging the failure).
            if any(sig in up for sig in _CONN_SIGS):
                tgt = None
                mh = _CONN_HOST_RE.search(line)
                if mh:
                    tgt = self._canonical_svc(mh.group(1))
                else:
                    mn = _CONN_NAME_RE.search(line)
                    if mn:
                        # strip a trailing "-client" (e.g. user-service-client)
                        tgt = re.sub(r"-client$", "", mn.group(1), flags=re.I)
                        tgt = self._canonical_svc(tgt)
                if tgt and tgt.lower() not in _INFRA_SVCS and not _is_token(tgt):
                    conn_counts[tgt] = conn_counts.get(tgt, 0) + 1

        for svc, n in err_counts.items():
            triplets.append({
                "source": svc, "relationship": "emits", "target": "LOG_ERROR",
                "citation_quote": f"logs error_lines={n}",
                "relevance": self._rel(n, cap=10),
            })
        for svc, n in timeout_counts.items():
            triplets.append({
                "source": svc, "relationship": "blocks", "target": "TIMEOUT",
                "citation_quote": f"logs timeout_lines={n}",
                "relevance": self._rel(n, cap=6),
            })
        # Connection failures: a definitive root-cause signal (the service is
        # unreachable). One such triplet should outweigh downstream metric
        # symptoms; the diagnoser weights CONNECTION_REFUSED accordingly.
        for svc, n in conn_counts.items():
            triplets.append({
                "source": svc, "relationship": "emits", "target": "CONNECTION_REFUSED",
                "citation_quote": f"logs connection_refused_lines={n}",
                "relevance": self._rel(n, cap=10),
            })
        return triplets

    # ── node entry point ──────────────────────────────────────────────────

    def __call__(self, state: AIOpsIncidentState) -> AIOpsIncidentState:
        """Parse every available telemetry source and commit triplets."""
        session_id = state["scratchpad_session_id"]
        triplets: List[Dict[str, Any]] = []
        breakdown: Dict[str, int] = {}

        for label, key in (
            ("traces", "trace_csv_path"),
            ("pods", "pod_status_path"),
            ("metrics", "metrics_path"),
            ("logs", "logs_path"),
        ):
            src = state.get(key)
            if not src:
                continue
            try:
                parser = {
                    "traces": self.parse_traces,
                    "pods": self.parse_kubectl,
                    ("metrics"): self.parse_metrics,
                    "logs": self.parse_logs,
                }[label]
                got = parser(src)
            except Exception as e:
                logger.warning(f"[Observer] {label} parse failed: {e}")
                got = []
            if got:
                triplets.extend(got)
                breakdown[label] = len(got)

        # Anomaly = any non-calls, non-token signal (errors/latency/bad pods/log errs).
        anomaly = any(
            (t.get("relationship") in ("emits", "blocks"))
            and not _is_token(t.get("source", ""))
            for t in triplets
        )

        # Always record the task marker for context (not an anomaly signal).
        triplets.append({
            "source": "SYSTEM", "relationship": "has_task",
            "target": state.get("task_type", "unknown"),
            "citation_quote": "AIOpsLab task started",
        })

        # Dedupe identical (src,rel,dst) edges but keep the highest-relevance copy.
        dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for t in triplets:
            k = (t["source"], t["relationship"], t["target"])
            if k not in dedup or t.get("relevance", 0) > dedup[k].get("relevance", 0):
                dedup[k] = t
        triplets = list(dedup.values())

        try:
            self.client.init_session(session_id, goal=state.get("problem_id", ""))
            self.client.commit_triplets(session_id, "ObserverAgent", triplets)
        except Exception as e:
            logger.error(f"[Observer] failed to commit {len(triplets)} triplets: {e}")

        state["anomaly_detected"] = bool(anomaly)
        logger.info(
            f"[Observer] committed {len(triplets)} triplets "
            f"(breakdown={breakdown}, anomaly_detected={anomaly})"
        )
        return state
