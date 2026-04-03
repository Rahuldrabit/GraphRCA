"""Causal Ranker Node — LangGraph agent node (STRATUS Pillar 2).

Upgrades the BFS-ranked results with:
  1. Temporal anomaly ordering (earliest anomaly = higher priority)
  2. Pairwise lagged cross-correlation causal scoring
  3. Combined α·BFS + β·temporal + γ·causal re-ranking
"""

import logging
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState
from GraphRCA_agent.tools.causal_tools import (
    temporal_order_analysis,
    compute_pairwise_causal_scores,
    rerank_with_causality,
)

logger = logging.getLogger(__name__)


def causal_ranker_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: re-rank RCA candidates using causal inference.

    Reads:  ranked_causes, spans, alerts
    Writes: ranked_causes (updated), causal_scores, temporal_order
    """
    t0 = time.time()
    ranked_causes = state.get("ranked_causes", [])
    spans = state.get("spans", [])
    alerts = state.get("alerts", [])

    logger.info(f"[CausalRanker] Re-ranking {len(ranked_causes)} candidates")

    if not ranked_causes or not spans:
        logger.warning("[CausalRanker] Nothing to re-rank — passing through")
        return {
            "temporal_order": [],
            "causal_scores": {},
            "messages": state.get("messages", []) + ["[CausalRanker] Skipped (no candidates)"],
        }

    try:
        # 1. Temporal ordering: sort services by earliest error timestamp
        temporal_order = temporal_order_analysis(spans, alerts)

        # 2. Pairwise causal scoring for top candidates
        candidate_services = []
        for c in ranked_causes[:8]:
            svc = c.service if hasattr(c, "service") else c.get("service", "")
            if svc:
                candidate_services.append(svc)

        causal_scores = compute_pairwise_causal_scores(spans, candidate_services, lag=5)

        # 3. Re-rank using combined score
        reranked = rerank_with_causality(
            ranked_causes=ranked_causes,
            temporal_order=temporal_order,
            causal_scores=causal_scores,
            alpha=0.50,
            beta=0.25,
            gamma=0.25,
        )

        elapsed = round(time.time() - t0, 2)

        top3 = []
        for c in reranked[:3]:
            svc = c.service if hasattr(c, "service") else c.get("service", "")
            conf = c.confidence if hasattr(c, "confidence") else c.get("confidence", 0)
            top3.append(f"{svc}:{conf:.2f}(causal={causal_scores.get(svc, 0):.2f})")

        logger.info(f"[CausalRanker] Done in {elapsed}s | top3={top3}")

        return {
            "ranked_causes": reranked,
            "causal_scores": causal_scores,
            "temporal_order": temporal_order,
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[CausalRanker] Causal re-rank: {' | '.join(top3)}"
            ],
            "node_timings": {**state.get("node_timings", {}), "causal_ranker": elapsed},
        }

    except Exception as e:
        logger.warning(f"[CausalRanker] Failed (non-fatal): {e}")
        return {
            "temporal_order": [],
            "causal_scores": {},
            "messages": state.get("messages", []) + [f"[CausalRanker] WARNING: {e}"],
        }
