"""Root Cause Analysis Tools (standalone, Stratus-compatible).

The LangGraph `rca` node expects NetworkX graphs and dataclass-like results.

Public API (used by GraphRCA_agent.nodes.rca):
  - backward_bfs_traversal(G, start_service, max_depth=5) -> List[List[str]]
  - score_candidate(service=..., G=..., spans=..., alerts=..., baselines=..., traversal_depth=...) -> RCACandidate
  - infer_silent_failures(spans, G) -> List[Dict[str, Any]]
  - rank_root_causes(candidates) -> List[RCACandidate]
  - generate_fault_tree(ranked_causes, G, error_service) -> FaultTree
  - store_rca_to_neo4j(ranked_causes, error_service, incident_id, connector) -> bool
"""

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)


def _get(span: Any, key: str, default: Any = None) -> Any:
    if hasattr(span, key):
        return getattr(span, key)
    if isinstance(span, dict):
        return span.get(key, default)
    return default


def _service_name(span: Any) -> str:
    return str(_get(span, "service_name") or _get(span, "service") or "unknown").strip() or "unknown"


def _has_error(span: Any) -> bool:
    return bool(_get(span, "has_error", False))


def _duration_ms(span: Any) -> float:
    try:
        return float(_get(span, "duration_ms") or 0.0)
    except Exception:
        return 0.0


def _baseline_mean_std(baseline: Any, fallback_mean: float) -> tuple[float, float]:
    if baseline is None:
        return float(fallback_mean), 0.0
    if hasattr(baseline, "ewma_mean") and hasattr(baseline, "ewma_std"):
        return float(getattr(baseline, "ewma_mean") or fallback_mean), float(getattr(baseline, "ewma_std") or 0.0)
    if isinstance(baseline, dict):
        return float(baseline.get("ewma_mean", fallback_mean) or fallback_mean), float(baseline.get("ewma_std", 0.0) or 0.0)
    return float(fallback_mean), 0.0


@dataclass
class RCACandidate:
    service: str
    confidence: float
    error_rate: float = 0.0
    latency_z_score: float = 0.0
    traversal_depth: int = 0
    is_silent_failure: bool = False
    evidence: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FaultTree:
    root_cause: str
    error_service: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def backward_bfs_traversal(
    G: nx.DiGraph,
    start_service: str,
    max_depth: int = 5,
) -> List[List[str]]:
    """Backward BFS on a directed service graph (predecessor traversal)."""
    if not G or not start_service or start_service not in G:
        return []

    paths: List[List[str]] = []
    queue: deque[tuple[str, List[str], int]] = deque([(start_service, [start_service], 0)])
    max_paths = 200

    while queue and len(paths) < max_paths:
        current, path, depth = queue.popleft()

        if depth >= max_depth:
            if len(path) > 1:
                paths.append(path)
            continue

        preds = list(G.predecessors(current)) if hasattr(G, "predecessors") else []
        preds = [p for p in preds if p and p not in path]

        if not preds:
            if len(path) > 1:
                paths.append(path)
            continue

        for p in preds:
            queue.append((p, path + [p], depth + 1))

    logger.info(f"BFS from {start_service}: found {len(paths)} paths")
    return paths


def _compute_service_error_rate(spans: List[Any], service: str) -> float:
    total = 0
    err = 0
    for s in spans:
        if _service_name(s) != service:
            continue
        total += 1
        err += 1 if _has_error(s) else 0
    return round(err / total, 4) if total else 0.0


def _compute_service_duration_mean(spans: List[Any], service: str) -> float:
    vals = [_duration_ms(s) for s in spans if _service_name(s) == service]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _max_span_count(G: nx.DiGraph) -> int:
    try:
        return max(int(G.nodes[n].get("span_count", 0) or 0) for n in G.nodes()) if G and G.nodes() else 1
    except Exception:
        return 1


def score_candidate(
    service: str,
    G: nx.DiGraph,
    spans: List[Any],
    alerts: List[Any],
    baselines: Dict[str, Any],
    traversal_depth: int = 0,
) -> RCACandidate:
    """Compute confidence that `service` is the root cause."""
    node_attrs: Dict[str, Any] = dict(G.nodes[service]) if (G and service in G) else {}

    # Pull current stats from graph node attrs; fall back to spans if missing.
    error_rate = float(node_attrs.get("error_rate", 0.0) or 0.0)
    if error_rate == 0.0 and spans:
        error_rate = float(_compute_service_error_rate(spans, service))

    current_mean = float(node_attrs.get("duration_mean_ms", 0.0) or 0.0)
    if current_mean == 0.0 and spans:
        current_mean = float(_compute_service_duration_mean(spans, service))

    span_count = int(node_attrs.get("span_count", 0) or 0)
    max_spans = _max_span_count(G)
    volume_score = min(1.0, span_count / max_spans) if max_spans > 0 else 0.0

    baseline = baselines.get(service) if isinstance(baselines, dict) else None
    baseline_mean, baseline_std = _baseline_mean_std(baseline, current_mean)

    if baseline_std <= 0.001:
        latency_z = 10.0 if abs(current_mean - baseline_mean) > 0.001 else 0.0
    else:
        latency_z = (current_mean - baseline_mean) / baseline_std
    latency_z = float(round(latency_z, 3))

    service_alerts = [
        a
        for a in alerts
        if (getattr(a, "service", None) == service)
        or (isinstance(a, dict) and a.get("service") == service)
    ]
    alert_score = 0.0
    if service_alerts:
        try:
            alert_score = max(
                float(
                    getattr(a, "score", 0.0) if not isinstance(a, dict) else a.get("score", 0.0)
                )
                for a in service_alerts
            )
        except Exception:
            alert_score = 0.2

    # Components
    error_component = min(1.0, error_rate * 3.0)  # 0.33 error_rate -> 1.0
    latency_component = min(1.0, abs(latency_z) / 10.0) if abs(latency_z) > 2.0 else 0.0
    depth_boost = min(0.15, max(0.0, traversal_depth) * 0.03)

    confidence = (
        0.35 * error_component
        + 0.35 * latency_component
        + 0.15 * alert_score
        + 0.10 * volume_score
        + depth_boost
    )
    confidence = float(max(0.0, min(1.0, round(confidence, 4))))

    evidence: List[str] = []
    if error_rate > 0:
        evidence.append(f"Error rate={error_rate:.1%}")
    if abs(latency_z) > 0:
        evidence.append(
            f"Latency mean={current_mean:.1f}ms (baseline={baseline_mean:.1f}±{baseline_std:.1f}), z={latency_z:.2f}"
        )
    if service_alerts:
        types = []
        for a in service_alerts[:2]:
            at = getattr(a, "anomaly_type", "") if not isinstance(a, dict) else a.get("anomaly_type", "")
            if at:
                types.append(str(at))
        if types:
            evidence.append(f"Alerts: {', '.join(types)}")
    if traversal_depth:
        evidence.append(f"Upstream depth={traversal_depth}")

    components = {
        "error": round(error_component, 4),
        "latency": round(latency_component, 4),
        "alerts": round(alert_score, 4),
        "volume": round(volume_score, 4),
        "depth_boost": round(depth_boost, 4),
    }

    return RCACandidate(
        service=service,
        confidence=confidence,
        error_rate=float(round(error_rate, 4)),
        latency_z_score=latency_z,
        traversal_depth=int(traversal_depth or 0),
        evidence=evidence,
        components=components,
    )


def infer_silent_failures(spans: List[Any], G: nx.DiGraph) -> List[Dict[str, Any]]:
    """Infer silent bottlenecks via duration absorption on call edges."""
    results: List[Dict[str, Any]] = []
    if not G:
        return results

    for parent, child, edata in G.edges(data=True):
        parent_mean = float(G.nodes[parent].get("duration_mean_ms", 0.0) or 0.0)
        if parent_mean <= 0:
            parent_mean = _compute_service_duration_mean(spans, parent) if spans else 0.0

        edge_avg = float(edata.get("avg_duration_ms", 0.0) or 0.0)
        child_err = float(G.nodes[child].get("error_rate", 0.0) or 0.0)

        if parent_mean <= 0 or edge_avg <= 0:
            continue

        absorption = min(1.5, edge_avg / parent_mean)
        verdict = "OK"
        if absorption >= 0.75 and child_err < 0.05:
            verdict = "SILENT_BOTTLENECK"
        elif absorption >= 0.6:
            verdict = "POSSIBLE_BOTTLENECK"

        if verdict != "OK":
            results.append(
                {
                    "parent_service": parent,
                    "child_service": child,
                    "absorption_ratio": round(absorption, 3),
                    "parent_duration_mean_ms": round(parent_mean, 2),
                    "child_edge_avg_duration_ms": round(edge_avg, 2),
                    "child_error_rate": round(child_err, 4),
                    "verdict": verdict,
                    "evidence": (
                        f"Child absorbs {absorption:.0%} of parent mean latency "
                        f"(edge_avg={edge_avg:.1f}ms, parent_mean={parent_mean:.1f}ms)"
                    ),
                }
            )

    results.sort(key=lambda r: r.get("absorption_ratio", 0.0), reverse=True)
    logger.info(f"Inferred {len(results)} silent failure edges")
    return results


def rank_root_causes(candidates: List[RCACandidate]) -> List[RCACandidate]:
    """Sort candidates by confidence and assign ranks."""
    ranked = sorted(
        candidates or [],
        key=lambda c: float(getattr(c, "confidence", 0.0) or 0.0),
        reverse=True,
    )
    for i, c in enumerate(ranked):
        try:
            c.rank = i + 1
        except Exception:
            pass
    return ranked


def generate_fault_tree(
    ranked_causes: List[RCACandidate],
    G: nx.DiGraph,
    error_service: str,
) -> FaultTree:
    """Generate a compact fault-tree subgraph for reporting."""
    root = ranked_causes[0].service if ranked_causes else (error_service or "unknown")
    top = ranked_causes[:5] if ranked_causes else []

    node_ids = {c.service for c in top if getattr(c, "service", "")}
    if error_service:
        node_ids.add(error_service)
    if root:
        node_ids.add(root)

    nodes: List[Dict[str, Any]] = []
    for c in top:
        nodes.append(
            {
                "id": c.service,
                "confidence": round(float(c.confidence), 4),
                "rank": int(getattr(c, "rank", 0) or 0),
                "is_root": c.service == root,
                "is_silent": bool(getattr(c, "is_silent_failure", False)),
            }
        )
    if error_service and error_service not in {n["id"] for n in nodes}:
        nodes.append(
            {"id": error_service, "confidence": 0.0, "rank": 0, "is_root": False, "is_silent": False}
        )

    edges: List[Dict[str, Any]] = []
    seen = set()

    # Prefer a single propagation chain root → error_service if available
    if G and root in G and error_service in G and root != error_service:
        try:
            path = nx.shortest_path(G, root, error_service)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"source": u, "target": v, "type": "calls"})
        except Exception:
            pass

    # Add any edges among the displayed nodes
    if G:
        for u, v in G.edges():
            if u in node_ids and v in node_ids:
                key = (u, v)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"source": u, "target": v, "type": "calls"})

    return FaultTree(
        root_cause=root,
        error_service=error_service or "",
        nodes=nodes,
        edges=edges,
    )


def _neo4j_write(connector: Any, query: str, parameters: Optional[Dict[str, Any]] = None) -> None:
    if connector is None:
        raise ValueError("neo4j connector is None")
    if hasattr(connector, "execute_write"):
        connector.execute_write(query, parameters or {})
        return
    if hasattr(connector, "run_query"):
        connector.run_query(query, parameters or {})
        return
    raise AttributeError("Neo4j connector missing execute_write/run_query")


def store_rca_to_neo4j(
    ranked_causes: List[RCACandidate],
    error_service: str,
    incident_id: str,
    connector: Any,
) -> bool:
    """Persist RCA outcome to Neo4j (optional)."""
    if connector is None or not getattr(connector, "is_available", lambda: False)():
        logger.warning("No Neo4j connector provided/available, skipping RCA storage")
        return False

    try:
        root = ranked_causes[0].service if ranked_causes else "unknown"
        root_conf = float(ranked_causes[0].confidence) if ranked_causes else 0.0

        _neo4j_write(
            connector,
            """
            MERGE (i:Incident {id: $id})
            SET i.error_service = $error_service,
                i.root_cause = $root_cause,
                i.root_confidence = $root_confidence,
                i.updated_at = datetime()
            """,
            {
                "id": incident_id,
                "error_service": error_service or "unknown",
                "root_cause": root,
                "root_confidence": root_conf,
            },
        )

        # Root-cause relationship
        _neo4j_write(
            connector,
            """
            MERGE (s:Service {name: $svc})
            WITH s
            MATCH (i:Incident {id: $id})
            MERGE (i)-[r:HAS_ROOT_CAUSE]->(s)
            SET r.confidence = $conf,
                r.updated_at = datetime()
            """,
            {"id": incident_id, "svc": root, "conf": root_conf},
        )

        # Top-N candidates
        for c in ranked_causes[:10]:
            _neo4j_write(
                connector,
                """
                MERGE (s:Service {name: $svc})
                WITH s
                MATCH (i:Incident {id: $id})
                MERGE (i)-[r:HAS_CANDIDATE]->(s)
                SET r.rank = $rank,
                    r.confidence = $conf,
                    r.is_silent = $silent,
                    r.updated_at = datetime()
                """,
                {
                    "id": incident_id,
                    "svc": c.service,
                    "rank": int(getattr(c, "rank", 0) or 0),
                    "conf": float(c.confidence),
                    "silent": bool(getattr(c, "is_silent_failure", False)),
                },
            )

        logger.info(f"Stored RCA for incident {incident_id} to Neo4j")
        return True

    except Exception as e:
        logger.error(f"Failed to store RCA to Neo4j: {e}")
        return False
