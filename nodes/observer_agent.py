import os
import re
import csv
import glob
import json
import math
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


# Static cgroup configuration (limit / shares / quota / period) and period
# counters. These are NOT behavioural signals: `container_spec_cpu_period` is
# literally 100000 for every container, and `container_spec_memory_limit_bytes`
# is the configured cap, not consumption. They used to be pooled together with
# real usage metrics into one "cpu"/"mem" distribution, which meant the pooled
# mean sat between the config constants and the usage values, so essentially
# EVERY service cleared the 2x-mean bar and got flagged HIGH_CPU/HIGH_MEM
# (observed: 19/19 hotel-reservation services, 31/31 astronomy-shop services,
# including prometheus/grafana/jaeger). The anomaly signal therefore carried no
# information and the diagnoser ranked on noise. Usage metrics only, and each
# metric is now compared against itself.
_SPEC_METRIC_PREFIXES = ("container_spec_",)
_SKIP_METRIC_SUFFIXES = ("_periods_total",)


# Service roles, written to the ScratchPad `source_type` column (which was
# previously left 'UNKNOWN' on 99.4% of rows). The diagnoser needs these because
# comparing raw resource/latency values ACROSS services measures architecture,
# not fault: a DATASTORE legitimately outweighs an app pod, and an ASYNC_CONSUMER
# legitimately holds spans open for seconds. Without roles those services top
# every ranking and bury the real root cause — measured on astronomy-shop, where
# excluding ASYNC_CONSUMER/OBSERVABILITY moved the true root cause from #3 to #1
# (ad_service_high_cpu) and #4 to #2 (ad_service_manual_gc).
_ROLE_PATTERNS = (
    ("OBSERVABILITY", ("jaeger", "prometheus", "grafana", "otel", "opentelemetry",
                       "opensearch", "opamp", "telemetry", "loki", "chaos",
                       "metrics-server", "mcp", "chatbot")),
    ("LOADGEN", ("load-generator", "loadgenerator", "wrk", "wrk2-job", "locust")),
    ("BROKER", ("kafka", "rabbitmq", "nats", "zookeeper")),
    ("DATASTORE", ("mongodb", "mysql", "postgres", "redis", "valkey", "memcached",
                   "-db", "tidb", "etcd", "consul")),
    ("GATEWAY", ("frontend-proxy", "frontend-web", "nginx", "istio", "envoy",
                 "ingress", "loadbalancer")),
    # Kafka/queue consumers: long-lived spans are their normal mode of operation.
    ("ASYNC_CONSUMER", ("fraud-detection", "accounting", "email")),
)


def _service_role(name: str) -> str:
    """Classify a service into a coarse role for role-aware ranking."""
    n = (name or "").lower()
    if not n:
        return "UNKNOWN"
    for role, needles in _ROLE_PATTERNS:
        if any(x in n for x in needles):
            return role
    return "APP"


def _usage_metric_kind(name: str) -> str:
    """Return 'cpu' | 'mem' for a real usage metric, or '' for one to skip."""
    n = (name or "").lower()
    if n.startswith(_SPEC_METRIC_PREFIXES) or n.endswith(_SKIP_METRIC_SUFFIXES):
        return ""
    if "cpu" in n:
        return "cpu"
    if "mem" in n:
        return "mem"
    return ""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


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
                    ratio = avg / max(med, 1.0)
                    # Log-scaled, not linearly clipped: observed ratios span
                    # 5x-557x on real faults, and a linear cap (old: cap=8)
                    # pinned everything above ~8x to the same relevance=1.0,
                    # which meant the strongest signal in the whole incident
                    # (the actual root cause, often the highest ratio) was
                    # indistinguishable from a mild symptom once ranked --
                    # exactly the "27 rows tied at 1.0" failure that starved
                    # the RCA analyst's token-bounded view of real evidence.
                    latency_saturate = _env_float("GRAPHRCA_LATENCY_RATIO_SATURATE", 1000.0)
                    relevance = round(
                        min(1.0, math.log10(max(ratio, 1.0001)) / math.log10(latency_saturate)),
                        3,
                    )
                    triplets.append({
                        "source": svc, "relationship": "blocks", "target": "HIGH_LATENCY",
                        "citation_quote": f"trace avg_duration={avg:.0f} median={med:.0f} ({ratio:.1f}x)",
                        "relevance": relevance,
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

    @staticmethod
    def _flag_metric_cross_sectional_outliers(
        per_metric: Dict[str, Dict[str, List[Tuple[float, float]]]]
    ) -> List[Dict[str, Any]]:
        """Flag services sitting >2x the mean *of the same metric*, right now.

        Kept alongside `_flag_metric_temporal_outliers` (below) rather than
        replaced by it. Measured against real fault sessions
        (astronomy-shop payment_service_unreachable /
        product_catalog_service_failure / recommendation_service_cache_failure):
        the swarm's single `get_metrics(ns, 5)` call fetches a 5-minute
        window AFTER the fault has already reached steady state -- by the
        time it's scraped, the faulty service is already flat at its new
        (elevated) level for the whole window, so a within-window
        baseline-vs-current comparison sees no transition and misses it
        entirely. Cross-sectional -- "is this bigger than its peers RIGHT
        NOW" -- is the only one of the two that can see an already-elevated
        flat plateau. Verified: temporal-only collapsed to a single false
        positive (`grafana`, whose cache metric happens to always be
        climbing within any 5-minute window, unrelated to any fault) and
        lost the ground truth on all three sessions above.

        Each metric is still compared only against itself (not pooled across
        metrics) to avoid the original config-constant/counter-pooling bug.
        Only the strongest deviation per (service, cpu|mem) is emitted.
        """
        saturate = _env_float("GRAPHRCA_METRIC_RATIO_SATURATE", 1000.0)
        best: Dict[Tuple[str, str], Tuple[float, str, float, float]] = {}
        for metric_name, svc_map in per_metric.items():
            kind = _usage_metric_kind(metric_name)
            if not kind:
                continue
            all_vals = [v for pts in svc_map.values() for _, v in pts]
            if not all_vals:
                continue
            mean = statistics.mean(all_vals)
            if mean <= 0:
                continue
            for svc, pts in svc_map.items():
                vals = [v for _, v in pts]
                if not vals:
                    continue
                mx = max(vals)
                if mx < 2.0 * mean:
                    continue
                ratio = mx / mean
                key = (svc, kind)
                if key not in best or ratio > best[key][0]:
                    best[key] = (ratio, metric_name, mx, mean)

        triplets: List[Dict[str, Any]] = []
        for (svc, kind), (ratio, metric_name, mx, mean) in best.items():
            # Log-scaled, not linearly clipped -- see the HIGH_LATENCY fix in
            # parse_traces for the same reasoning: a linear cap pinned every
            # service above ~5x the mean to the same relevance=1.0, so the
            # ranking carried no magnitude information at all.
            relevance = round(min(1.0, math.log10(max(ratio, 1.0001)) / math.log10(saturate)), 3)
            triplets.append({
                "source": svc,
                "relationship": "emits",
                "target": "HIGH_CPU" if kind == "cpu" else "HIGH_MEM",
                "citation_quote": f"metric {metric_name} max={mx:.3f} mean={mean:.3f} ({ratio:.1f}x, cross-sectional)",
                "relevance": relevance,
            })
        return triplets

    @staticmethod
    def _flag_metric_temporal_outliers(
        per_metric: Dict[str, Dict[str, List[Tuple[float, float]]]]
    ) -> List[Dict[str, Any]]:
        """Flag services whose OWN metric history just changed.

        The old version compared every service's value against the mean of
        ALL services for that metric at one instant. That measures
        architecture, not fault: a mongodb/kafka/opensearch legitimately
        sits at higher CPU/mem than an app pod all the time, so it (and every
        other datastore/broker/observability pod) cleared the "2x the mean"
        bar on nearly every run regardless of whether anything was actually
        wrong -- observed as ~22 of 23 services flagged anomalous on healthy
        and faulty runs alike, which made the anomaly signal roughly
        uninformative and also broke detection (any() over "is anything
        anomalous" was ~always true).

        This version instead compares each (service, metric) time series
        against ITSELF: the values in the tail "current" window vs. the
        values in the earlier "baseline" window, using a z-score. A fault is
        a service's OWN trend changing mid-window, not a service being
        intrinsically bigger than its neighbours. Needs the timestamp column
        (kept in `swarm_agent_aiopslab.py::_collect_metrics`) or, absent
        that, the caller's row order — Prometheus scrape rows are already
        chronological, so row order is an acceptable stand-in for a real
        timestamp.

        Only the strongest |z| per (service, cpu|mem) is emitted, so one
        service isn't scored several times over for correlated metrics all
        moving together.
        """
        z_thresh = _env_float("GRAPHRCA_METRIC_Z_THRESH", 3.0)
        # Above this |z| the relevance score saturates at 1.0. Chosen well
        # above z_thresh so magnitude still discriminates between a
        # marginal outlier (z~3-5) and a severe one (z~15+) instead of
        # every flagged service piling up at relevance=1.0 -- that
        # saturation was a second, independent source of the "8
        # undifferentiated suspects" problem: even when the anomaly WAS
        # real, its strength got clipped away before the diagnoser/RCA
        # ever saw it.
        saturate = _env_float("GRAPHRCA_METRIC_Z_SATURATE", 15.0)
        min_points = 6  # need enough samples to split baseline vs current meaningfully

        best: Dict[Tuple[str, str], Tuple[float, str, float, float]] = {}
        for metric_name, svc_map in per_metric.items():
            kind = _usage_metric_kind(metric_name)
            if not kind:
                continue
            for svc, points in svc_map.items():
                pts = sorted(points, key=lambda p: p[0])
                vals = [v for _, v in pts]
                if len(vals) < min_points:
                    continue
                # Current = last 30% of samples (min 3); baseline = the rest
                # (min 3). A short, recent-weighted current window catches a
                # fault that started partway through the scrape without
                # needing it to dominate the whole window.
                current_n = max(3, int(len(vals) * 0.3))
                current_n = min(current_n, len(vals) - 3)
                if current_n < 3:
                    continue
                baseline = vals[:-current_n]
                current = vals[-current_n:]
                b_mean = statistics.mean(baseline)
                b_std = statistics.pstdev(baseline) if len(baseline) > 1 else 0.0
                # A baseline that is EXACTLY zero for its whole window (many
                # near-idle counters like container_memory_cache legitimately
                # sit at 0 until something first touches the page cache) has
                # no established trend at all -- any tiny nonzero blip in the
                # current window then divides by the 1e-6 floor and produces
                # a z in the billions, which is noise, not signal. Skip it;
                # there's nothing to compare "own history" against.
                if b_mean == 0.0 and b_std == 0.0:
                    continue
                # Floor the denominator so a near-constant NONZERO baseline
                # (b_std ~0) doesn't blow z up unboundedly on a modest move.
                b_std_floor = max(b_std, abs(b_mean) * 0.05, 1e-6)
                c_mean = statistics.mean(current)
                z = (c_mean - b_mean) / b_std_floor
                if abs(z) < z_thresh:
                    continue
                key = (svc, kind)
                if key not in best or abs(z) > abs(best[key][0]):
                    best[key] = (z, metric_name, c_mean, b_mean)

        triplets: List[Dict[str, Any]] = []
        for (svc, kind), (z, metric_name, c_mean, b_mean) in best.items():
            triplets.append({
                "source": svc,
                "relationship": "emits",
                "target": "HIGH_CPU" if kind == "cpu" else "HIGH_MEM",
                "citation_quote": (
                    f"metric {metric_name} current={c_mean:.3f} baseline={b_mean:.3f} "
                    f"(z={z:.1f}, own-history)"
                ),
                "relevance": round(min(1.0, abs(z) / saturate), 3),
            })
        return triplets

    def parse_metrics(self, metrics_path_or_text: str) -> List[Dict[str, Any]]:
        """Parse Prometheus metrics CSV(s) into triplets.

        Accepts either the combined summary CSV produced by the swarm fetcher
        (columns: metric,cmdb_id,kpi_name,value) or a metrics directory.
        Flags services whose value is >2x the mean *of that same metric*.
        Best-effort: returns [] if the format is unexpected.
        """
        triplets: List[Dict[str, Any]] = []
        text, was_file = self._read_text(metrics_path_or_text)
        if not text:
            return triplets

        # Directory of kpi_*.csv files
        if was_file is False and os.path.isdir(metrics_path_or_text.strip()):
            return self._parse_metrics_dir(metrics_path_or_text.strip())

        rows: List[Dict[str, str]] = []
        # Combined summary CSV (metric,cmdb_id,kpi_name,value[,timestamp])
        if "cmdb_id" in text or "kpi_name" in text:
            try:
                reader = csv.DictReader(text.strip().split("\n"))
                rows = [r for r in reader if r]
            except Exception:
                rows = []

        # Keyed by the EXACT metric name so each metric is compared against
        # itself over time: {metric_name: {svc: [(timestamp, value), ...]}}
        # If the CSV predates the timestamp column (or a value is
        # unparseable), fall back to row order -- the source kpi_*.csv rows
        # are already chronological, so enumeration order is still a valid
        # (if less precise) time axis.
        agg: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        for i, r in enumerate(rows):
            kpi = (r.get("kpi_name") or r.get("metric") or "").lower()
            svc = self._canonical_svc(self._svc_from_cmdb(r.get("cmdb_id", "")))
            try:
                val = float(r.get("value", "nan"))
            except (TypeError, ValueError):
                continue
            if not svc or val != val:  # NaN check
                continue
            if not _usage_metric_kind(kpi):
                continue
            try:
                ts = float(r.get("timestamp", "nan"))
                if ts != ts:  # NaN
                    raise ValueError
            except (TypeError, ValueError):
                ts = float(i)
            agg.setdefault(kpi, {}).setdefault(svc, []).append((ts, val))

        return self._flag_metric_outliers(agg)

    @staticmethod
    def _flag_metric_outliers(
        per_metric: Dict[str, Dict[str, List[Tuple[float, float]]]]
    ) -> List[Dict[str, Any]]:
        """Union of the cross-sectional and temporal metric detectors.

        Neither alone is sufficient (see the docstrings on each): the
        cross-sectional channel catches a service already flat at an
        elevated level for the whole scrape window (the common case, since
        metrics are fetched after the fault has settled); the temporal
        channel catches one still actively changing within the window. Emit
        both; when they agree on the same (service, kind), keep whichever
        scored the signal as more relevant.
        """
        combined: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for t in (
            ObserverAgent._flag_metric_cross_sectional_outliers(per_metric)
            + ObserverAgent._flag_metric_temporal_outliers(per_metric)
        ):
            k = (t["source"], t["relationship"], t["target"])
            if k not in combined or t.get("relevance", 0) > combined[k].get("relevance", 0):
                combined[k] = t
        return list(combined.values())

    def _parse_metrics_dir(self, dirpath: str) -> List[Dict[str, Any]]:
        csvs = glob.glob(os.path.join(dirpath, "**", "kpi_*.csv"), recursive=True)
        # Keyed by the EXACT metric name (the kpi_<name>.csv filename) so each
        # metric is compared against itself over time:
        # {metric_name: {svc: [(timestamp, value), ...]}}
        agg: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        for c in csvs[:40]:
            metric = os.path.basename(c)[4:-4].lower()
            if not _usage_metric_kind(metric):
                continue
            try:
                with open(c, encoding="utf-8", errors="replace") as f:
                    for i, r in enumerate(csv.reader(f)):
                        if len(r) < 4:
                            continue
                        svc = self._canonical_svc(self._svc_from_cmdb(r[1]))
                        try:
                            v = float(r[3])
                        except (TypeError, ValueError):
                            continue
                        try:
                            ts = float(r[0])
                        except (TypeError, ValueError):
                            ts = float(i)
                        if svc and v == v:  # NaN check
                            agg.setdefault(metric, {}).setdefault(svc, []).append((ts, v))
            except Exception:
                continue

        return self._flag_metric_outliers(agg)

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

        # Stamp the service role so the diagnoser can rank role-aware (and so the
        # role survives in the ScratchPad for anything reading the graph later).
        for t in triplets:
            if not _is_token(t["source"]) and t["source"] != "SYSTEM":
                t.setdefault("source_type", _service_role(t["source"]))

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
