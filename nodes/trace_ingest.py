"""Trace Ingest Node — LangGraph agent node.

Parses raw CSV trace files, deduplicates spans, validates schema,
and computes per-service statistics.

Uses GraphRCA's own pipeline tools (follows Stratus methodology).
"""

import logging
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState

# Use GraphRCA's own pipeline tools
from GraphRCA_agent.tools.pipeline.ingest_tools import (
    parse_csv_directory,
    deduplicate_spans,
    validate_schema,
    detect_orphan_spans,
    compute_stats,
    compute_overall_stats,
)

logger = logging.getLogger(__name__)


def trace_ingest_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: parse, validate, and deduplicate trace spans.

    Reads:  trace_dir, output_dir
    Writes: spans, service_stats, ingest_summary, status
    """
    t0 = time.time()
    trace_dir = state.get("trace_dir", "./trace_output")
    logger.info(f"[TraceIngest] Starting — trace_dir={trace_dir}")

    try:
        # 1. Parse all CSV files
        raw_spans = parse_csv_directory(trace_dir)
        if not raw_spans:
            logger.error(f"[TraceIngest] No spans parsed from {trace_dir}")
            return {
                "status": "failed",
                "error": f"No spans found in {trace_dir}",
                "messages": state.get("messages", []) + ["[TraceIngest] ERROR: No spans found"],
            }

        # 2. Deduplicate
        spans, dedup_stats = deduplicate_spans(raw_spans)

        # 3. Validate schema
        validation = validate_schema(spans)

        # 4. Detect orphan spans
        orphans = detect_orphan_spans(spans)

        # 5. Per-service statistics
        service_stats = compute_stats(spans)

        # 6. Overall stats
        overall = compute_overall_stats(spans)

        elapsed = round(time.time() - t0, 2)
        summary = {
            "total_spans": overall.total_spans,
            "unique_spans": overall.unique_spans,
            "unique_services": overall.unique_services,
            "unique_traces": overall.unique_traces,
            "error_spans": overall.error_spans,
            "unknown_response_pct": overall.unknown_response_pct,
            "orphan_spans": len(orphans),
            "services": overall.services,
            "dedup": dedup_stats,
            "validation_rate": validation.get("validation_rate", 100),
            "_elapsed_seconds": elapsed,
        }

        logger.info(
            f"[TraceIngest] Done in {elapsed}s — "
            f"{overall.unique_spans} spans, {overall.unique_services} services"
        )

        return {
            "spans": spans,
            "service_stats": service_stats,
            "ingest_summary": summary,
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[TraceIngest] Ingested {overall.unique_spans} spans from {overall.unique_services} services"
            ],
            "node_timings": {**state.get("node_timings", {}), "trace_ingest": elapsed},
        }

    except Exception as e:
        logger.exception(f"[TraceIngest] Failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "messages": state.get("messages", []) + [f"[TraceIngest] ERROR: {e}"],
        }
