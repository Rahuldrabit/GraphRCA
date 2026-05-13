"""Graph Builder Node — LangGraph agent node.

Builds a service dependency DAG from span parent-child relationships,
enriches nodes with per-service metrics, computes PageRank, and
optionally persists to Neo4j.

Phase 2: When GRAPHRCA_INFRA_LAYER=true, adds a second infrastructure layer
(Pod/K8sNode nodes and DEPLOYED_AS/RUNS_ON/COLOCATED edges) via kubectl.
"""

import logging
import os
import re
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState

# Use GraphRCA's own pipeline tools
from GraphRCA_agent.tools.pipeline.graph_tools import (
    build_dag,
    add_node_with_metrics,
    detect_cycles,
    compute_pagerank,
    export_graph_json,
    store_graph_to_neo4j,
    store_trace_spans_to_neo4j,
)

logger = logging.getLogger(__name__)


def graph_builder_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: build service dependency knowledge graph.

    Reads:  spans, service_stats, use_neo4j, neo4j_connector, store_spans
    Writes: graph, pagerank, graph_summary
    """
    t0 = time.time()
    spans = state.get("spans", [])
    service_stats = state.get("service_stats", {})
    use_neo4j = state.get("use_neo4j", False)
    neo4j_connector = state.get("neo4j_connector")
    store_spans = state.get("store_spans", False)

    logger.info(f"[GraphBuilder] Building DAG from {len(spans)} spans")

    if not spans:
        return {
            "status": "failed",
            "error": "No spans available for graph construction",
            "messages": state.get("messages", []) + ["[GraphBuilder] ERROR: No spans"],
        }

    try:
        # 1. Build base DAG
        G = build_dag(spans)

        # 2. Enrich nodes with service stats
        for svc, stats in service_stats.items():
            G = add_node_with_metrics(G, svc, stats)

        # 3. Detect cycles
        cycles = detect_cycles(G)

        # 4. Compute PageRank (centrality)
        pagerank = compute_pagerank(G)

        # 5. Export to JSON for logging/storage (result stored in graph_summary via neo4j path)
        export_graph_json(G)

        # 6. Phase 2: Optional infrastructure topology layer
        infra_topology: Dict[str, Any] = {}
        noisy_neighbors: list = []
        infra_layer_enabled = os.getenv("GRAPHRCA_INFRA_LAYER", "").lower() in ("1", "true")
        if infra_layer_enabled:
            try:
                from GraphRCA_agent.tools.pipeline.infra_tools import (
                    discover_infra_topology,
                    build_infra_layer,
                    detect_noisy_neighbors,
                )
                from GraphRCA_agent.tools.kube_tools import exec_kubectl_command

                # Extract namespace from additional_context (same pattern as agent_aiopslab.py)
                additional_context = state.get("additional_context", "") or ""
                ns_match = re.search(
                    r"namespace[:\s]+['\"]?([a-z0-9-]+)['\"]?",
                    additional_context,
                    re.IGNORECASE,
                )
                namespace = ns_match.group(1) if ns_match else "default"

                infra_topology = discover_infra_topology(
                    namespace=namespace,
                    kubectl_fn=exec_kubectl_command,
                )
                current_alerts = state.get("alerts", [])
                G = build_infra_layer(G, infra_topology, service_stats, alerts=current_alerts)
                noisy_neighbors = detect_noisy_neighbors(G, infra_topology, alerts=current_alerts)
                logger.info(
                    f"[GraphBuilder] Infra layer added: "
                    f"{len(infra_topology.get('pods', []))} pods, "
                    f"{len(infra_topology.get('nodes', []))} nodes, "
                    f"{len(noisy_neighbors)} noisy-neighbor pairs"
                )
            except Exception as infra_err:
                logger.warning(f"[GraphBuilder] Infra layer failed (skipping): {infra_err}")

        # 7. Persist to Neo4j
        neo4j_summary: Dict[str, Any] = {}
        if use_neo4j and neo4j_connector:
            neo4j_summary = store_graph_to_neo4j(G, neo4j_connector)
            if store_spans:
                span_summary = store_trace_spans_to_neo4j(spans, neo4j_connector)
                neo4j_summary["spans"] = span_summary

        elapsed = round(time.time() - t0, 2)
        graph_summary = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "cycles": len(cycles),
            "top_pagerank": dict(list(pagerank.items())[:5]),
            "neo4j": neo4j_summary,
            "infra_layer_enabled": infra_layer_enabled,
            "_elapsed_seconds": elapsed,
        }

        logger.info(
            f"[GraphBuilder] Done in {elapsed}s — "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        result: Dict[str, Any] = {
            "graph": G,
            "pagerank": pagerank,
            "graph_summary": graph_summary,
            "infra_topology": infra_topology,
            "noisy_neighbors": noisy_neighbors,
            "infra_layer_enabled": infra_layer_enabled,
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[GraphBuilder] DAG: {G.number_of_nodes()} services, {G.number_of_edges()} call edges"
            ],
            "node_timings": {**state.get("node_timings", {}), "graph_builder": elapsed},
        }
        return result

    except Exception as e:
        logger.exception(f"[GraphBuilder] Failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "messages": state.get("messages", []) + [f"[GraphBuilder] ERROR: {e}"],
        }
