"""Graph Builder Tools (standalone, Stratus-compatible).

Implements the same function signatures expected by the LangGraph nodes:
  - build_dag(spans) -> nx.DiGraph
  - add_node_with_metrics(G, node_id, stats) -> nx.DiGraph
  - detect_cycles(G) -> List[List[str]]
  - compute_pagerank(G) -> Dict[str, float]
  - export_graph_json(G) -> str
  - store_graph_to_neo4j(G, connector) -> Dict[str, int]
  - store_trace_spans_to_neo4j(spans, connector) -> Dict[str, int]

The implementation is robust to spans being either dicts (GraphRCA ingest)
or Span-like objects (Stratus style).
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

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


def _operation_name(span: Any) -> str:
    return str(_get(span, "operation_name") or _get(span, "operation") or "").strip()


def build_dag(spans: List[Any]) -> nx.DiGraph:
    """Create directed graph from parent-child span links."""
    G = nx.DiGraph()

    # Build span_id -> span lookup
    span_map = {str(_get(s, "span_id") or "").strip(): s for s in spans if str(_get(s, "span_id") or "").strip()}

    edge_data: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "call_count": 0,
            "total_duration_ms": 0.0,
            "error_count": 0,
            "operations": set(),
        }
    )

    for span in spans:
        svc = _service_name(span)
        if not G.has_node(svc):
            G.add_node(svc, node_type="service")

        parent_span = str(_get(span, "parent_span") or _get(span, "parent_id") or "").strip()
        if not parent_span or parent_span == "ROOT":
            continue

        parent = span_map.get(parent_span)
        if not parent:
            continue

        parent_svc = _service_name(parent)
        if parent_svc == svc:
            continue

        edge_key = (parent_svc, svc)
        ed = edge_data[edge_key]
        ed["call_count"] += 1
        ed["total_duration_ms"] += float(_get(span, "duration_ms") or 0.0)
        ed["error_count"] += 1 if bool(_get(span, "has_error", False)) else 0
        op = _operation_name(span)
        if op:
            ed["operations"].add(op)

    for (src, dst), data in edge_data.items():
        call_count = int(data["call_count"] or 0)
        total_dur = float(data["total_duration_ms"] or 0.0)
        err_count = int(data["error_count"] or 0)

        G.add_edge(
            src,
            dst,
            call_count=call_count,
            avg_duration_ms=round(total_dur / call_count, 2) if call_count else 0.0,
            total_duration_ms=round(total_dur, 2),
            error_count=err_count,
            error_rate=round(err_count / call_count, 4) if call_count else 0.0,
            operations=sorted(list(data["operations"])) if data.get("operations") else [],
        )

    logger.info(f"Built DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def add_node_with_metrics(G: nx.DiGraph, node_id: str, stats: Dict[str, Any]) -> nx.DiGraph:
    """Attach service-level metrics to a node in the graph."""
    if not G.has_node(node_id):
        G.add_node(node_id, node_type="service")

    G.nodes[node_id].update(
        {
            "span_count": stats.get("span_count", 0),
            "error_count": stats.get("error_count", 0),
            "error_rate": stats.get("error_rate", 0),
            "unknown_response_pct": stats.get("unknown_response_pct", 0),
            "duration_mean_ms": stats.get("duration_mean_ms", 0),
            "duration_p50_ms": stats.get("duration_p50_ms", 0),
            "duration_p95_ms": stats.get("duration_p95_ms", 0),
            "duration_p99_ms": stats.get("duration_p99_ms", 0),
            "duration_std_ms": stats.get("duration_std_ms", 0),
            "duration_min_ms": stats.get("duration_min_ms", 0),
            "duration_max_ms": stats.get("duration_max_ms", 0),
            "operations": stats.get("operations", []),
        }
    )
    return G


def detect_cycles(G: nx.DiGraph) -> List[List[str]]:
    """Find circular dependencies that would break BFS traversal."""
    try:
        cycles = list(nx.simple_cycles(G))
        if cycles:
            logger.warning(f"Found {len(cycles)} cycles in service dependency graph: {cycles}")
        return cycles
    except Exception as e:
        logger.error(f"Error detecting cycles: {e}")
        return []


def compute_pagerank(G: nx.DiGraph) -> Dict[str, float]:
    """Score nodes by structural centrality using PageRank."""
    try:
        scores = nx.pagerank(G, alpha=0.85)
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        return {k: round(v, 6) for k, v in sorted_scores.items()}
    except Exception as e:
        logger.error(f"PageRank computation failed: {e}")
        return {}


def export_graph_json(G: nx.DiGraph) -> str:
    """Serialize the graph to JSON for other agents to consume."""
    data = nx.node_link_data(G)

    # Clean up sets for JSON serialization
    for node in data.get("nodes", []):
        for key, val in list(node.items()):
            if isinstance(val, set):
                node[key] = list(val)
    for link in data.get("links", []):
        for key, val in list(link.items()):
            if isinstance(val, set):
                link[key] = list(val)

    return json.dumps(data, indent=2, default=str)


def store_graph_to_neo4j(G: nx.DiGraph, neo4j_connector) -> Dict[str, int]:
    """Store the service dependency graph in Neo4j."""
    if not neo4j_connector or not neo4j_connector.is_available():
        logger.warning("Neo4j not available, skipping graph storage")
        return {"nodes_created": 0, "relationships_created": 0}

    nodes_created = 0
    rels_created = 0

    for node_id in G.nodes():
        attrs = dict(G.nodes[node_id])
        clean_attrs: Dict[str, Any] = {}
        for k, v in attrs.items():
            if isinstance(v, (list, set)):
                clean_attrs[k] = str(list(v))
            elif isinstance(v, (int, float, str, bool)):
                clean_attrs[k] = v

        try:
            neo4j_connector.execute_write(
                """
                MERGE (s:Service {name: $name})
                SET s += $props
                SET s.updated_at = datetime()
                """,
                {"name": node_id, "props": clean_attrs},
            )
            nodes_created += 1
        except Exception as e:
            logger.error(f"Failed to create node {node_id}: {e}")

    for src, dst in G.edges():
        edge_attrs = dict(G.edges[src, dst])
        clean_attrs = {}
        for k, v in edge_attrs.items():
            if isinstance(v, (list, set)):
                clean_attrs[k] = str(list(v))
            elif isinstance(v, (int, float, str, bool)):
                clean_attrs[k] = v

        try:
            neo4j_connector.execute_write(
                """
                MATCH (src:Service {name: $src})
                MATCH (dst:Service {name: $dst})
                MERGE (src)-[r:CALLS]->(dst)
                SET r += $props
                SET r.updated_at = datetime()
                """,
                {"src": src, "dst": dst, "props": clean_attrs},
            )
            rels_created += 1
        except Exception as e:
            logger.error(f"Failed to create relationship {src}->{dst}: {e}")

    logger.info(f"Stored to Neo4j: {nodes_created} nodes, {rels_created} relationships")
    return {"nodes_created": nodes_created, "relationships_created": rels_created}


def store_trace_spans_to_neo4j(spans: List[Any], neo4j_connector) -> Dict[str, int]:
    """Store individual trace spans and their parent-child relationships in Neo4j."""
    if not neo4j_connector or not neo4j_connector.is_available():
        logger.warning("Neo4j not available, skipping span storage")
        return {"traces": 0, "spans": 0, "relationships": 0}

    traces_created = 0
    spans_created = 0
    rels_created = 0
    batch_size = 100

    def _span_record(s: Any) -> Dict[str, Any]:
        trace_id = str(_get(s, "trace_id") or "")
        span_id = str(_get(s, "span_id") or "")
        parent_span = str(_get(s, "parent_span") or _get(s, "parent_id") or "")
        svc = _service_name(s)
        op = _operation_name(s)
        start_time = int(_get(s, "start_time") or 0)
        duration = int(_get(s, "duration") or round(float(_get(s, "duration_ms") or 0.0) * 1000.0))
        duration_ms = float(_get(s, "duration_ms") or 0.0)
        has_error = bool(_get(s, "has_error", False))
        response = str(_get(s, "response") or "Unknown")
        is_root = parent_span == "ROOT" or not parent_span

        return {
            "span_id": span_id,
            "trace_id": trace_id,
            "service_name": svc,
            "operation_name": op,
            "start_time": start_time,
            "duration": duration,
            "duration_ms": duration_ms,
            "has_error": has_error,
            "response": response,
            "is_root": is_root,
            "parent_span": parent_span or "ROOT",
        }

    unique_traces = list({str(_get(s, "trace_id") or "") for s in spans if str(_get(s, "trace_id") or "")})
    span_records = [_span_record(s) for s in spans if str(_get(s, "span_id") or "")]

    for i in range(0, len(unique_traces), batch_size):
        batch = unique_traces[i : i + batch_size]
        try:
            neo4j_connector.execute_write(
                """
                UNWIND $traces AS tid
                MERGE (t:Trace {trace_id: tid})
                SET t.updated_at = datetime()
                """,
                {"traces": batch},
            )
            traces_created += len(batch)
        except Exception as e:
            logger.error(f"Failed to batch-create traces: {e}")

    for i in range(0, len(span_records), batch_size):
        batch = span_records[i : i + batch_size]
        try:
            neo4j_connector.execute_write(
                """
                UNWIND $spans AS sp
                MERGE (span:Span {trace_id: sp.trace_id, span_id: sp.span_id})
                SET span.service_name = sp.service_name,
                    span.operation_name = sp.operation_name,
                    span.start_time = sp.start_time,
                    span.duration = sp.duration,
                    span.duration_ms = sp.duration_ms,
                    span.has_error = sp.has_error,
                    span.response = sp.response,
                    span.is_root = sp.is_root
                WITH span, sp
                MATCH (t:Trace {trace_id: sp.trace_id})
                MERGE (t)-[:CONTAINS]->(span)
                WITH span, sp
                MATCH (s:Service {name: sp.service_name})
                MERGE (span)-[:EXECUTED_BY]->(s)
                """,
                {"spans": batch},
            )
            spans_created += len(batch)
            rels_created += len(batch) * 2
        except Exception as e:
            logger.error(f"Failed to batch-create spans: {e}")

    span_ids = {r["span_id"] for r in span_records}
    child_of_pairs = [
        {"trace_id": r["trace_id"], "child_id": r["span_id"], "parent_id": r["parent_span"]}
        for r in span_records
        if r.get("parent_span") and r["parent_span"] != "ROOT" and r["parent_span"] in span_ids
    ]

    for i in range(0, len(child_of_pairs), batch_size):
        batch = child_of_pairs[i : i + batch_size]
        try:
            neo4j_connector.execute_write(
                """
                UNWIND $pairs AS p
                MATCH (child:Span {trace_id: p.trace_id, span_id: p.child_id})
                MATCH (parent:Span {trace_id: p.trace_id, span_id: p.parent_id})
                MERGE (child)-[:CHILD_OF]->(parent)
                """,
                {"pairs": batch},
            )
            rels_created += len(batch)
        except Exception as e:
            logger.error(f"Failed to batch-create CHILD_OF: {e}")

    summary = {"traces": traces_created, "spans": spans_created, "relationships": rels_created}
    logger.info(f"Stored spans to Neo4j: {summary}")
    return summary
