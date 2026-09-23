import os
import networkx as nx
from typing import List
from swarm_state import AIOpsIncidentState
from tools.scratchpad_client import ScratchpadClient
from nodes.observer_agent import _service_role
import logging

logger = logging.getLogger(__name__)

# Tokens that must never be nominated as suspect services.
_TOKEN_PREFIXES = ("HTTP_", "LOG_", "ERROR", "TIMEOUT", "HIGH_", "POD_", "STATUS_")


def _is_token(node: str) -> bool:
    if not node:
        return True
    return any(node.startswith(p) for p in _TOKEN_PREFIXES) or node in ("SYSTEM", "INCIDENT")


def _env_bool(name: str, default: bool = False) -> bool:
    return str(os.getenv(name, str(default))).strip().lower() in ("1", "true", "yes", "on")


# Confidence multipliers applied to the deterministic anomaly score. Mirrors
# tools/pipeline/llm_scorer.py (the legacy `pipeline` mode's scorer, which the
# swarm graph never reaches). Classification only ever REORDERS candidates —
# nothing is dropped — so a bad LLM answer degrades to the old ranking rather
# than losing the root cause outright.
_RANK_MULTIPLIERS = {"ROOT_CAUSE": 2.0, "SYMPTOM": 0.6, "NORMAL": 0.25}

# Role priors applied to the deterministic anomaly score. Raw resource/latency
# values are only comparable WITHIN a role: a datastore legitimately holds more
# memory than an app pod, an async Kafka consumer legitimately holds spans open
# for seconds, and observability/load-gen pods are never the injected fault.
# Without this, those roles saturate the top of every ranking. Measured on
# astronomy-shop: applying these moved the true root cause #3 -> #1
# (ad_service_high_cpu) and #4 -> #2 (ad_service_manual_gc).
_ROLE_PRIORS = {
    "APP": 1.0,
    "GATEWAY": 0.5,         # entry points surface everyone else's failures
    "DATASTORE": 0.4,       # high memory/CPU is their steady state
    "BROKER": 0.3,
    "ASYNC_CONSUMER": 0.25,  # multi-second spans are normal here
    "OBSERVABILITY": 0.1,   # never the injected fault
    "LOADGEN": 0.1,
    "UNKNOWN": 1.0,
}

_RANK_SYSTEM_PROMPT = (
    "You are an SRE triaging which microservice is the ORIGIN of an incident.\n"
    "For each candidate you are given its observed anomaly evidence.\n"
    "Classify every candidate as exactly one of:\n"
    "  ROOT_CAUSE — the fault most likely originates here\n"
    "  SYMPTOM    — this service looks bad only because a dependency is failing\n"
    "  NORMAL     — the evidence reflects this service's normal role, not a fault\n"
    "\n"
    "Judge evidence against what is NORMAL FOR THAT SERVICE'S ROLE. Databases, "
    "caches, message brokers, search engines, telemetry collectors and load "
    "generators (e.g. mongodb, valkey, kafka, opensearch, otel-collector, "
    "prometheus, grafana, jaeger, load-generator) routinely sit at high memory "
    "or CPU with no fault at all — that is NORMAL, not ROOT_CAUSE. Likewise an "
    "entry-point or gateway (frontend, frontend-proxy, checkout) showing errors "
    "usually reflects a failure further down — that is SYMPTOM.\n"
    "Prefer a service whose OWN behaviour changed (latency far above its peers, "
    "its own errors, pod restarts/crashes).\n"
    "\n"
    "Reply with ONE line per candidate, nothing else:\n"
    "<service> = <ROOT_CAUSE|SYMPTOM|NORMAL>"
)


class TopologicalDiagnoser:
    """
    Zero-token NetworkX node.
    Builds DiGraph from ScratchPad active edges, reverses calls/emits/blocks for
    backward fault propagation, and runs Personalized PageRank. Only services that
    carry an anomaly signal (or sit one hop from one) are nominated, so a healthy
    system yields no suspects (detection -> "No").
    """
    def __init__(self, scratchpad_client: ScratchpadClient):
        self.client = scratchpad_client
        self.alpha = float(os.getenv("SCRATCHPAD_PAGERANK_ALPHA", "0.85"))

    def __call__(self, state: AIOpsIncidentState) -> AIOpsIncidentState:
        session_id = state["scratchpad_session_id"]
        triplets = self.client.get_triplets(session_id)

        if not triplets:
            logger.warning("No triplets found in ScratchPad. Returning empty suspects.")
            state["suspect_nodes"] = []
            return state

        G = nx.DiGraph()
        anomaly_svcs = set()

        # Weight anomalies by how root-cause-like the signal is. Pod crashes,
        # connection/network errors and log FATALs are usually the source fault;
        # HTTP errors / latency spikes are downstream symptoms. Weighting the
        # source service higher biases the nominator toward the cause rather than
        # the loudest symptom. Connection errors in particular are the signature
        # of k8s port-misconfig / network faults — the affected service is the
        # root, not the noisy callers that fail to reach it.
        def _emit_weight(target: str) -> float:
            t = (target or "").upper()
            # A crashed pod is direct fault evidence at that service.
            if t == "POD_UNHEALTHY":
                return 5.0
            # Connection-refused / network errors = the service itself is
            # unreachable (port misconfig, network policy) -> definitive root
            # signal for this fault class. Weighted above stacked resource
            # metrics (a HIGH_CPU+HIGH_MEM datastore can reach 6.0) so the
            # unreachable service is nominated even with a single log line.
            if t.startswith("CONNECTION") or "REFUSED" in t or t.startswith("CONN_"):
                return 8.0
            # A log FATAL/error is root-cause evidence (config fault, crash) —
            # rank it above resource-metric spikes, which are often downstream.
            if t == "LOG_ERROR":
                return 4.5
            if t in ("HIGH_CPU", "HIGH_MEM"):
                return 3.0
            if t.startswith("HTTP_") or t in ("ERROR",):
                return 2.0
            return 2.0

        # Build the graph.
        for t in triplets:
            src = t["source_entity"]
            rel = t["relationship"]
            dst = t["target_entity"]

            if rel == "calls":
                # Reverse: callee -> caller (backward fault propagation).
                G.add_edge(dst, src, weight=1.0)
            elif rel == "emits":
                # Anomaly: error token -> service (service gets heavy rank).
                G.add_edge(dst, src, weight=_emit_weight(dst))
                if not _is_token(src):
                    anomaly_svcs.add(src)
            elif rel == "blocks":
                # Latency: latency token -> service.
                G.add_edge(dst, src, weight=1.5)
                if not _is_token(src):
                    anomaly_svcs.add(src)
            else:
                G.add_edge(src, dst, weight=1.0)

        # A purely-healthy system (calls topology only) has no anomalies -> no suspects.
        if not anomaly_svcs:
            state["suspect_nodes"] = []
            logger.info("No anomaly signals in graph — healthy system, no suspects.")
            return state

        # Candidate suspects = anomaly services plus their 1-hop topology neighbours
        # (a failing caller/callee is the usual root cause).
        candidates = set(anomaly_svcs)
        for svc in list(anomaly_svcs):
            if svc in G:
                candidates |= set(G.successors(svc)) | set(G.predecessors(svc))
        candidates = {c for c in candidates if not _is_token(c)}

        # Rank candidates. Primary key: direct anomaly score (sum of the weights
        # of emits/blocks edges sourced at the service) — this favours the cause
        # (pod crashes / log FATALs) over the loudest symptom (HTTP errors at the
        # entry-point hub). PageRank over the reversed call graph is the tiebreak,
        # capturing how centrally connected each candidate is.
        # Roles come from the ScratchPad `source_type` column; sessions written
        # before roles were stamped fall back to name-based classification.
        roles: dict = {}
        for t in triplets:
            src = t["source_entity"]
            st = (t.get("source_type") or "UNKNOWN").upper()
            if st not in ("UNKNOWN", "SERVICE"):
                roles[src] = st
            elif src not in roles:
                roles[src] = _service_role(src)

        anomaly_score: dict = {}
        for t in triplets:
            if t["relationship"] == "emits":
                w = _emit_weight(t["target_entity"])
            elif t["relationship"] == "blocks":
                # Latency is the primary signature of resource faults (high CPU,
                # GC pauses, network delay), so it must not sit below routine
                # metric spikes. Scaled by the observed deviation so a service
                # 100x over the median outranks one barely past the threshold.
                w = 4.0
            else:
                continue
            src = t["source_entity"]
            try:
                rel = float(t.get("relevance_score") or 1.0)
            except (TypeError, ValueError):
                rel = 1.0
            w *= max(rel, 0.1) * _ROLE_PRIORS.get(roles.get(src, "UNKNOWN"), 1.0)
            anomaly_score[src] = anomaly_score.get(src, 0.0) + w

        pagerank_scores: dict = {}
        try:
            if len(G.nodes) > 0:
                pagerank_scores = nx.pagerank(G, alpha=self.alpha, weight="weight")
        except Exception as e:
            logger.error(f"PageRank computation failed: {e}")

        # How many candidates to nominate. The old hardcoded top-3 was too
        # aggressive: on a port-misconfig fault it kept the loudest symptom
        # services (high-CPU mongodbs, the error-logging entry-point hub) and
        # DROPPED the actual root cause (the service whose port is wrong),
        # because that root carried only one log signal vs many stacked metric
        # signals on the symptoms — so the RCA never even saw the right answer
        # (Localization Accuracy 0.0). A larger top-K keeps the root cause in
        # contention; the RCA/drill loop then discriminates among them.
        top_k = int(os.getenv("GRAPHRCA_DIAGNOSER_TOP_K", "8"))
        suspects: List[str] = sorted(
            candidates,
            key=lambda n: (anomaly_score.get(n, 0.0), pagerank_scores.get(n, 0.0)),
            reverse=True,
        )[:top_k]

        logger.info(f"Topological suspects identified: {suspects} (anomaly_svcs={sorted(anomaly_svcs)})")

        # Optional LLM re-rank. The deterministic score above compares absolute
        # resource values ACROSS services, which mostly measures architecture
        # (a mongodb legitimately outweighs an app pod) rather than fault. An LLM
        # can apply role priors the score cannot encode. Never changes membership.
        if _env_bool("GRAPHRCA_SWARM_LLM_RANK", False) and state.get("task_type") != "detection":
            try:
                suspects = self._llm_rerank(suspects, triplets, anomaly_score)
            except Exception as e:
                logger.warning(f"[Diagnoser] LLM re-rank failed ({e}); keeping deterministic order")

        state["suspect_nodes"] = suspects
        return state

    def _llm_rerank(self, suspects: List[str], triplets: List[dict],
                    anomaly_score: dict) -> List[str]:
        """Reorder suspects with an LLM using per-service evidence.

        Returns the input list reordered — same members, so a wrong or
        unparseable answer can never drop the true root cause.
        """
        from llm import llm_reason

        # Compact evidence table: one line per candidate, its own signals only.
        by_svc: dict = {}
        for t in triplets:
            if t["relationship"] in ("emits", "blocks") and t["source_entity"] in suspects:
                by_svc.setdefault(t["source_entity"], []).append(
                    f"{t['target_entity']} ({t.get('citation_quote', '')})"
                )
        lines = []
        for s in suspects:
            ev = "; ".join(by_svc.get(s, [])[:4]) or "no direct anomaly signal"
            lines.append(f"- {s} = {ev}")

        raw = llm_reason(
            prompt="Candidates and their evidence:\n" + "\n".join(lines)
            + "\n\nClassify each candidate. One line each, no other text.",
            system_prompt=_RANK_SYSTEM_PROMPT,
            max_tokens=int(os.getenv("GRAPHRCA_SWARM_LLM_RANK_TOKENS", "2048")),
            caller="diagnoser_rank",
        )

        # Lenient parse: find "<service> ... <LABEL>" on any line, in any order.
        labels: dict = {}
        for line in (raw or "").splitlines():
            up = line.upper()
            for label in _RANK_MULTIPLIERS:
                if label in up:
                    for s in suspects:
                        if s.lower() in line.lower():
                            labels.setdefault(s, label)
                    break
        if not labels:
            logger.warning("[Diagnoser] LLM re-rank returned no usable labels; keeping order")
            return suspects

        order = {s: i for i, s in enumerate(suspects)}
        reranked = sorted(
            suspects,
            key=lambda s: (
                -anomaly_score.get(s, 0.0) * _RANK_MULTIPLIERS.get(labels.get(s, "SYMPTOM"), 1.0),
                order[s],  # stable: preserve deterministic order within a tier
            ),
        )
        logger.info(f"[Diagnoser] LLM re-rank: {reranked} (labels={labels})")
        return reranked
