"""Detection Node — LangGraph agent node.

Computes baselines and detects latency/error anomalies across all services.

Detector selection is controlled by GRAPHRCA_DETECTOR env var:
  "ewma"             — always use EWMA (original behaviour, default)
  "isolation_forest" — always use IsolationForest
  "auto"             — use IF when len(spans) >= GRAPHRCA_IF_MIN_SAMPLES, else EWMA

LLM contextual scoring is opt-in via GRAPHRCA_LLM_SCORER=true.
"""

import logging
import os
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "y", "on"}


def detection_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: anomaly detection across all services.

    Reads:  spans, service_stats
    Writes: alerts, baselines, primary_error_service, health_score_before
    """
    t0 = time.time()
    spans = state.get("spans", [])
    service_stats = state.get("service_stats", {})

    detector_mode = os.getenv("GRAPHRCA_DETECTOR", "ewma").lower().strip()
    use_llm_scorer = os.getenv("GRAPHRCA_LLM_SCORER", "").lower().strip() in _TRUTHY

    logger.info(
        f"[Detection] detector={detector_mode}, llm_scorer={use_llm_scorer}, "
        f"services={len(service_stats)}"
    )

    if not spans:
        return {
            "alerts": [],
            "baselines": {},
            "primary_error_service": "unknown",
            "messages": state.get("messages", []) + ["[Detection] WARNING: No spans to analyze"],
        }

    try:
        from GraphRCA_agent.tools.pipeline.anomaly_detectors import get_detector

        detector = get_detector(mode=detector_mode, span_count=len(spans))
        alerts, baselines = detector.detect(spans=spans, service_stats=service_stats)

        # 3. Optional LLM contextual scoring (adjusts scores, never removes alerts)
        if use_llm_scorer and alerts:
            try:
                from GraphRCA_agent.tools.pipeline.llm_scorer import contextual_score_alerts
                graph = state.get("graph")
                pagerank = state.get("pagerank", {})
                alerts = contextual_score_alerts(alerts, graph=graph, pagerank=pagerank)
                logger.info(f"[Detection] LLM scorer applied to {len(alerts)} alerts")
            except Exception as llm_err:
                logger.warning(f"[Detection] LLM scorer failed (skipping): {llm_err}")

        # 4. Identify primary error service (highest anomaly score)
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

        # 5. Compute pre-mitigation health score μ(s)
        from GraphRCA_agent.tools.safety_tools import compute_health_score
        sla_violations = state.get("sla_violations", [])
        unhealthy_nodes = state.get("unhealthy_nodes", [])
        health_before = compute_health_score(alerts, sla_violations, unhealthy_nodes)

        elapsed = round(time.time() - t0, 2)

        logger.info(
            f"[Detection] Done in {elapsed}s — "
            f"{len(alerts)} alerts, primary={primary_error_service}, "
            f"detector={detector_mode}"
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
