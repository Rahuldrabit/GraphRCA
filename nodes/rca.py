"""RCA Node — LangGraph agent node.

Root Cause Analysis via deterministic backward BFS on the
service dependency graph + multi-signal confidence scoring.
"""

import logging
import os
import time
import uuid
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState

# Use GraphRCA's own pipeline tools
from GraphRCA_agent.tools.pipeline.rca_tools import (
    backward_bfs_traversal,
    score_candidate,
    infer_silent_failures,
    rank_root_causes,
    generate_fault_tree,
    store_rca_to_neo4j,
)

logger = logging.getLogger(__name__)


def rca_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: backward BFS root cause analysis.

    Reads:  graph, spans, alerts, baselines, primary_error_service,
            service_stats, neo4j_connector
    Writes: ranked_causes, incident_id, fault_tree, bfs_paths,
            silent_failures, suspect_services
    """
    t0 = time.time()

    G = state.get("graph")
    spans = state.get("spans", [])
    alerts = state.get("alerts", [])
    baselines = state.get("baselines", {})
    error_service = state.get("primary_error_service", "")
    service_stats = state.get("service_stats", {})
    neo4j_connector = state.get("neo4j_connector")
    rollback_count = state.get("rollback_count", 0)

    # Use rollback count in incident ID to track re-analyses
    incident_id = state.get("incident_id") or f"INC-{uuid.uuid4().hex[:8]}"
    if rollback_count > 0:
        incident_id = f"{incident_id}-retry{rollback_count}"

    logger.info(f"[RCA] Starting BFS from '{error_service}' | incident={incident_id}")

    if not G or not spans:
        return {
            "status": "failed",
            "error": "Graph and spans required for RCA",
            "messages": state.get("messages", []) + ["[RCA] ERROR: Missing graph or spans"],
        }

    if not error_service:
        if alerts:
            error_service = alerts[0].service if hasattr(alerts[0], "service") else alerts[0].get("service", "")
        elif service_stats:
            error_service = max(service_stats, key=lambda s: service_stats[s].get("error_rate", 0))
        else:
            error_service = "unknown"

    try:
        # Step 1: Backward BFS traversal
        paths = backward_bfs_traversal(G, error_service, max_depth=5)

        # Collect visited services
        visited = set()
        for path in paths:
            visited.update(path)
        visited.add(error_service)

        # Step 2: Score each candidate
        candidates = []
        for svc in visited:
            depth = 0
            for path in paths:
                if svc in path:
                    depth = path.index(svc)
                    break
            candidate = score_candidate(
                service=svc,
                G=G,
                spans=spans,
                alerts=alerts,
                baselines=baselines,
                traversal_depth=depth,
            )
            candidates.append(candidate)

        # Step 3: Silent failures via duration absorption
        silent_failures = infer_silent_failures(spans, G)
        silent_services = {
            sf["child_service"] for sf in silent_failures
            if sf["verdict"] == "SILENT_BOTTLENECK"
        }
        for c in candidates:
            if c.service in silent_services:
                c.is_silent_failure = True
                c.confidence = min(1.0, c.confidence * 1.15)
                c.evidence.append("Duration absorption: silent bottleneck detected")

        # Step 4: Rank
        ranked = rank_root_causes(candidates)

        # Optional: let the LLM access the knowledge graph for candidate re-ranking.
        # Off by default. Enable via GRAPHRCA_LLM_KG_MODE=a|b (see run_graphrca.sh prompt).
        kg_mode = str(state.get("llm_kg_mode") or os.getenv("GRAPHRCA_LLM_KG_MODE", "")).strip().lower()
        llm_kg_debug: Dict[str, Any] = {}
        if kg_mode in {"a", "b"}:
            try:
                from GraphRCA_agent.tools.pipeline.kg_llm_tools import maybe_llm_rerank_with_kg

                ranked, llm_kg_debug = maybe_llm_rerank_with_kg(
                    kg_mode=kg_mode,
                    G=G,
                    error_service=error_service,
                    ranked_causes=ranked,
                    neo4j_connector=neo4j_connector,
                )

                if llm_kg_debug.get("applied"):
                    logger.info(f"[RCA] LLM-KG({kg_mode}) applied re-ranking")
                else:
                    why = llm_kg_debug.get("reason") or llm_kg_debug.get("skipped") or llm_kg_debug.get("error")
                    logger.info(f"[RCA] LLM-KG({kg_mode}) not applied: {why}")
            except Exception as e:
                logger.warning(f"[RCA] LLM-KG({kg_mode}) failed (non-fatal): {e}")
                llm_kg_debug = {"enabled": True, "mode": kg_mode, "error": str(e)}

        # Step 5: Fault tree
        fault_tree = generate_fault_tree(ranked, G, error_service)

        # Step 6: Store in Neo4j
        if neo4j_connector:
            store_rca_to_neo4j(ranked, error_service, incident_id, neo4j_connector)

        # Suspect services = top 3 ranked (for log analysis)
        suspect_services = [
            c.service if hasattr(c, "service") else c.get("service", "")
            for c in ranked[:3]
        ]

        elapsed = round(time.time() - t0, 2)

        top3_summary = []
        for c in ranked[:3]:
            svc = c.service if hasattr(c, "service") else c.get("service", "")
            conf = c.confidence if hasattr(c, "confidence") else c.get("confidence", 0)
            top3_summary.append(f"{svc}:{conf:.2f}")

        logger.info(
            f"[RCA] Done in {elapsed}s — "
            f"{len(ranked)} candidates | top3={top3_summary}"
        )

        return {
            "ranked_causes": ranked,
            "incident_id": incident_id,
            "fault_tree": fault_tree.to_dict(),
            "bfs_paths": paths[:20],
            "silent_failures": silent_failures[:20],
            "suspect_services": suspect_services,
            "llm_kg": llm_kg_debug,
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[RCA] Root causes: {' | '.join(top3_summary)} | incident={incident_id}"
            ],
            "node_timings": {**state.get("node_timings", {}), "rca": elapsed},
        }

    except Exception as e:
        logger.exception(f"[RCA] Failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "messages": state.get("messages", []) + [f"[RCA] ERROR: {e}"],
        }
