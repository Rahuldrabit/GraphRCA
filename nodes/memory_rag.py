"""Memory RAG Node — LangGraph agent node.

Handles two sub-actions:
  1. memory_search — find similar past incidents before RCA
  2. memory_store  — persist incident record after full analysis

Uses GraphRCA's own SQLite-based MemoryStore.
"""

import logging
import os
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState

# Use GraphRCA's own pipeline tools
from GraphRCA_agent.tools.pipeline.memory_tools import (
    MemoryStore,
    load_similar_cases,
    embed_incident,
    update_confidence_from_outcome,
    store_rca_to_neo4j_memory,
)

logger = logging.getLogger(__name__)

# Shared MemoryStore instance
_memory_store: MemoryStore | None = None


def _get_memory_store(output_dir: str | None = None) -> MemoryStore:
    global _memory_store

    # Prefer explicit env var, else default to a per-run DB under output_dir.
    db_path = os.getenv("SQLITE_DB_PATH") or os.getenv("GRAPHRCA_SQLITE_DB_PATH") or ""
    db_path = db_path.strip().strip('"').strip("'")
    if not db_path and output_dir:
        db_path = os.path.join(output_dir, "memory.sqlite")

    if _memory_store is None:
        _memory_store = MemoryStore(db_path=db_path or None)
    return _memory_store


# ── Search Node ─────────────────────────────────────────────────────────────


def memory_search_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: search past incidents for context.

    Reads:  primary_error_service, alerts
    Writes: similar_cases, false_positive_patterns
    """
    t0 = time.time()
    error_svc = state.get("primary_error_service", "")
    alerts = state.get("alerts", [])

    logger.info(f"[MemorySearch] Searching for incidents similar to {error_svc}")

    try:
        store = _get_memory_store(state.get("output_dir"))

        # Build anomaly type string from alerts
        anomaly_type = "+".join(
            sorted({a.anomaly_type if hasattr(a, "anomaly_type") else a.get("anomaly_type", "") for a in alerts[:3]})
        )

        similar = load_similar_cases(
            error_service=error_svc,
            anomaly_type=anomaly_type,
            evidence=[],
            memory_store=store,
            limit=5,
        )

        fp_patterns = store.get_false_positive_patterns()
        incident_count = store.get_incident_count()

        elapsed = round(time.time() - t0, 2)
        logger.info(f"[MemorySearch] Found {len(similar)} similar cases | DB has {incident_count} incidents")

        return {
            "similar_cases": similar,
            "false_positive_patterns": fp_patterns,
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[MemorySearch] {len(similar)} similar past incidents found"
            ],
            "node_timings": {**state.get("node_timings", {}), "memory_search": elapsed},
        }

    except Exception as e:
        logger.warning(f"[MemorySearch] Failed (non-fatal): {e}")
        return {
            "similar_cases": [],
            "false_positive_patterns": [],
            "messages": state.get("messages", []) + [f"[MemorySearch] WARNING: {e}"],
        }


# ── Store Node ───────────────────────────────────────────────────────────────


def memory_store_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: persist incident record for future RAG retrieval.

    Reads:  incident_id, primary_error_service, ranked_causes,
            alerts, mitigation_actions, neo4j_connector
    Writes: memory_stored, status
    """
    t0 = time.time()
    incident_id = state.get("incident_id", "INC-unknown")
    error_svc = state.get("primary_error_service", "unknown")
    ranked_causes = state.get("ranked_causes", [])
    alerts = state.get("alerts", [])
    mitigation_actions = state.get("mitigation_actions", [])
    neo4j_connector = state.get("neo4j_connector")
    service_stats = state.get("service_stats", {})

    logger.info(f"[MemoryStore] Storing incident {incident_id}")

    try:
        store = _get_memory_store(state.get("output_dir"))

        # Top cause
        top_cause = ranked_causes[0] if ranked_causes else None
        root_svc = top_cause.service if (top_cause and hasattr(top_cause, "service")) else \
                   (top_cause.get("service", "") if isinstance(top_cause, dict) else "")
        root_conf = top_cause.confidence if (top_cause and hasattr(top_cause, "confidence")) else \
                    (top_cause.get("confidence", 0.0) if isinstance(top_cause, dict) else 0.0)

        # Anomaly type from alerts
        anomaly_type = "+".join(
            sorted({a.anomaly_type if hasattr(a, "anomaly_type") else a.get("anomaly_type", "") for a in alerts[:3]})
        ) or "unknown"

        # Mitigation summary
        if mitigation_actions:
            first_action = mitigation_actions[0]
            mit_desc = first_action.get("title", "") if isinstance(first_action, dict) else \
                       getattr(first_action, "title", "")
        else:
            mit_desc = "no_action"

        # Evidence from top cause
        evidence = []
        if top_cause:
            ev = top_cause.evidence if hasattr(top_cause, "evidence") else \
                 (top_cause.get("evidence", []) if isinstance(top_cause, dict) else [])
            evidence = ev if isinstance(ev, list) else [str(ev)]

        embed_incident(
            incident_id=incident_id,
            error_service=error_svc,
            root_cause_service=root_svc,
            root_cause_confidence=root_conf,
            anomaly_type=anomaly_type,
            mitigation_applied=mit_desc,
            mitigation_success=not state.get("rollback_triggered", False),
            resolution_time_seconds=time.time() - state.get("pipeline_start_time", time.time()),
            service_stats=service_stats,
            evidence=evidence,
            memory_store=store,
        )

        # Also store in Neo4j
        if neo4j_connector:
            store_rca_to_neo4j_memory(
                incident_id=incident_id,
                error_service=error_svc,
                root_cause_service=root_svc,
                confidence=root_conf,
                anomaly_type=anomaly_type,
                neo4j_connector=neo4j_connector,
            )

        elapsed = round(time.time() - t0, 2)
        logger.info(f"[MemoryStore] Stored incident {incident_id} in {elapsed}s")

        return {
            "memory_stored": True,
            "status": "complete",
            "messages": state.get("messages", []) + [
                f"[MemoryStore] Incident {incident_id} persisted for future RAG retrieval"
            ],
            "node_timings": {**state.get("node_timings", {}), "memory_store": elapsed},
        }

    except Exception as e:
        logger.warning(f"[MemoryStore] Failed (non-fatal): {e}")
        return {
            "memory_stored": False,
            "status": "complete",
            "messages": state.get("messages", []) + [f"[MemoryStore] WARNING: {e}"],
        }
