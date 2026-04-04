"""Detection Node — LangGraph agent node.

Computes EWMA baselines and detects latency/error anomalies
across all services. Implements multi-signal anomaly scoring.
"""

import logging
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState

# Use GraphRCA's own pipeline tools
from GraphRCA_agent.tools.pipeline.detection_tools import (
    compute_ewma_baseline,
    detect_all_anomalies,
)

logger = logging.getLogger(__name__)


def detection_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: EWMA anomaly detection across all services.

    Reads:  spans, service_stats
    Writes: alerts, baselines, primary_error_service, health_score_before
    """
    t0 = time.time()
    spans = state.get("spans", [])
    service_stats = state.get("service_stats", {})

    logger.info(f"[Detection] Running EWMA detection on {len(service_stats)} services")

    if not spans:
        return {
            "alerts": [],
            "baselines": {},
            "primary_error_service": "unknown",
            "messages": state.get("messages", []) + ["[Detection] WARNING: No spans to analyze"],
        }

    try:
        # 1. Compute EWMA baselines for each service
        baselines = compute_ewma_baseline(spans, alpha=0.3, window_size=100)

        # 2. Run anomaly detection (multi-signal: z-score + error rate + unknown%)
        alerts = detect_all_anomalies(
            spans=spans,
            service_stats=service_stats,
            baselines=baselines,
            z_threshold=3.0,
            error_threshold=0.05,
        )

        # 3. Identify primary error service (highest anomaly score)
        if alerts:
            primary = max(alerts, key=lambda a: a.score)
            primary_error_service = primary.service
        else:
            # Fallback: service with highest error rate
            if service_stats:
                primary_error_service = max(
                    service_stats.keys(),
                    key=lambda s: service_stats[s].get("error_rate", 0),
                )
            else:
                primary_error_service = "unknown"

        # 4. Compute pre-mitigation health score μ(s)
        from GraphRCA_agent.tools.safety_tools import compute_health_score
        sla_violations = state.get("sla_violations", [])
        unhealthy_nodes = state.get("unhealthy_nodes", [])
        health_before = compute_health_score(alerts, sla_violations, unhealthy_nodes)

        elapsed = round(time.time() - t0, 2)

        logger.info(
            f"[Detection] Done in {elapsed}s — "
            f"{len(alerts)} alerts, primary={primary_error_service}"
        )

        alert_summary = []
        for a in alerts[:5]:
            alert_summary.append(
                f"{a.service}: {a.severity} ({a.anomaly_type}, score={a.score:.3f})"
            )

        return {
            "alerts": alerts,
            "baselines": baselines,
            "primary_error_service": primary_error_service,
            "health_score_before": health_before,
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[Detection] {len(alerts)} anomalies | primary={primary_error_service} | μ(s)={health_before:.4f}"
            ],
            "node_timings": {**state.get("node_timings", {}), "detection": elapsed},
        }

    except Exception as e:
        logger.exception(f"[Detection] Failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "alerts": [],
            "baselines": {},
            "messages": state.get("messages", []) + [f"[Detection] ERROR: {e}"],
        }
