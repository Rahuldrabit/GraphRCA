import os
import networkx as nx
from typing import List
from swarm_state import AIOpsIncidentState
from tools.scratchpad_client import ScratchpadClient
import logging

logger = logging.getLogger(__name__)

# Tokens that must never be nominated as suspect services.
_TOKEN_PREFIXES = ("HTTP_", "LOG_", "ERROR", "TIMEOUT", "HIGH_", "POD_", "STATUS_")


def _is_token(node: str) -> bool:
    if not node:
        return True
    return any(node.startswith(p) for p in _TOKEN_PREFIXES) or node in ("SYSTEM", "INCIDENT")


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
        anomaly_score: dict = {}
        for t in triplets:
            if t["relationship"] == "emits":
                w = _emit_weight(t["target_entity"])
            elif t["relationship"] == "blocks":
                w = 1.5
            else:
                continue
            anomaly_score[t["source_entity"]] = anomaly_score.get(t["source_entity"], 0.0) + w

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

        state["suspect_nodes"] = suspects
        logger.info(f"Topological suspects identified: {suspects} (anomaly_svcs={sorted(anomaly_svcs)})")
        return state
