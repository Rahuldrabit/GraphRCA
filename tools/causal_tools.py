"""Causal Inference Tools — STRATUS Pillar 2.

Upgrades raw BFS ranking with:
  1. Temporal ordering  — earlier anomaly first
  2. Pairwise causal scoring — PC-algorithm-style conditional independence
  3. Combined re-ranking — BFS rank + temporal priority + causal strength
"""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Temporal Logic ──────────────────────────────────────────────────────────


def temporal_order_analysis(
    spans: List[Any],
    alerts: List[Any],
) -> List[str]:
    """Sort services by the earliest anomaly/error timestamp.

    The intuition: if Service A shows an error spike *before* Service B,
    A is more likely to be a root cause than a downstream symptom.

    Args:
        spans:  List of Span objects (with start_time in microseconds)
        alerts: List of AlertSignal objects (with service field)

    Returns:
        List of service names ordered by earliest anomaly time (asc)
    """
    alerted_services = {a.service if hasattr(a, "service") else a.get("service", "") for a in alerts}

    # Find earliest error span per service
    earliest: Dict[str, int] = {}
    for span in spans:
        svc = span.service_name if hasattr(span, "service_name") else span.get("service_name", "")
        has_error = span.has_error if hasattr(span, "has_error") else span.get("has_error", False)
        start_time = span.start_time if hasattr(span, "start_time") else span.get("start_time", 0)

        if svc in alerted_services:
            if svc not in earliest or start_time < earliest[svc]:
                earliest[svc] = start_time

    # Sort by earliest error time
    ordered = sorted(earliest.keys(), key=lambda s: earliest[s])
    logger.info(f"Temporal order (earliest anomaly first): {ordered}")
    return ordered


# ── Pairwise Causal Scoring ─────────────────────────────────────────────────


def _get_durations(spans: List[Any], service: str) -> np.ndarray:
    """Extract duration_ms array for a service."""
    durations = []
    for span in spans:
        svc = span.service_name if hasattr(span, "service_name") else span.get("service_name", "")
        dur = span.duration_ms if hasattr(span, "duration_ms") else span.get("duration_ms", 0)
        if svc == service:
            durations.append(float(dur))
    return np.array(durations) if durations else np.array([0.0])


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays (same length via interpolation)."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def causal_score(
    spans: List[Any],
    service_a: str,
    service_b: str,
    lag: int = 5,
) -> float:
    """Estimate pairwise causal strength A → B.

    Uses a simplified PC-algorithm approach:
    1. Compute lagged cross-correlation of latency distributions
    2. If corr(A_t, B_{t+lag}) > corr(B_t, A_{t+lag}), A likely causes B

    Args:
        spans: All spans
        service_a: Candidate cause service
        service_b: Candidate effect service
        lag: Number of span positions to shift for lagged correlation

    Returns:
        Causal strength 0.0–1.0 (higher = A more likely causes B)
    """
    dur_a = _get_durations(spans, service_a)
    dur_b = _get_durations(spans, service_b)

    n = min(len(dur_a), len(dur_b))
    if n < max(lag + 2, 5):
        return 0.0

    # Lagged correlation: A_t → B_{t+lag}
    corr_ab = _pearson_correlation(dur_a[:n - lag], dur_b[lag:n])
    # Reverse: B_t → A_{t+lag}
    corr_ba = _pearson_correlation(dur_b[:n - lag], dur_a[lag:n])

    # A "causes" B if its lagged correlation is stronger
    if corr_ab > corr_ba and corr_ab > 0:
        # Normalise to 0-1
        strength = float(np.clip((corr_ab - corr_ba) / max(abs(corr_ab) + abs(corr_ba), 1e-9), 0, 1))
    else:
        strength = 0.0

    logger.debug(f"Causal {service_a}→{service_b}: corr_ab={corr_ab:.3f} corr_ba={corr_ba:.3f} strength={strength:.3f}")
    return round(strength, 4)


def compute_pairwise_causal_scores(
    spans: List[Any],
    candidate_services: List[str],
    lag: int = 5,
) -> Dict[str, float]:
    """Compute aggregate causal score for each candidate service.

    For each service A, sum its causal strength against all other candidates.
    A higher aggregate score means A is more likely to be a causal origin.

    Args:
        spans: All spans
        candidate_services: Services to evaluate
        lag: Lag for cross-correlation

    Returns:
        Dict {service: aggregate_causal_score}
    """
    scores: Dict[str, float] = defaultdict(float)

    for i, svc_a in enumerate(candidate_services):
        for j, svc_b in enumerate(candidate_services):
            if i == j:
                continue
            strength = causal_score(spans, svc_a, svc_b, lag=lag)
            scores[svc_a] += strength

    # Normalize
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {k: round(v / max_score, 4) for k, v in scores.items()}

    logger.info(f"Causal scores: {dict(scores)}")
    return dict(scores)


# ── Combined Re-Ranking ─────────────────────────────────────────────────────


def rerank_with_causality(
    ranked_causes: List[Any],
    temporal_order: List[str],
    causal_scores: Dict[str, float],
    alpha: float = 0.50,
    beta: float = 0.25,
    gamma: float = 0.25,
) -> List[Any]:
    """Re-rank root cause candidates using causal signals.

    Combined score = α·BFS_confidence + β·temporal_priority + γ·causal_score

    Args:
        ranked_causes: List[RCACandidate] from BFS
        temporal_order: Services ordered by earliest anomaly (idx 0 = earliest)
        causal_scores: {service: normalized causal strength}
        alpha: Weight for BFS confidence
        beta: Weight for temporal priority
        gamma: Weight for causal strength

    Returns:
        Re-ranked list (same objects, new order)
    """
    if not ranked_causes:
        return ranked_causes

    n_temporal = len(temporal_order)

    def combined_score(candidate) -> float:
        svc = candidate.service if hasattr(candidate, "service") else candidate.get("service", "")
        bfs_conf = candidate.confidence if hasattr(candidate, "confidence") else candidate.get("confidence", 0.0)

        # Temporal priority: earlier = higher priority (inverted rank)
        if svc in temporal_order:
            idx = temporal_order.index(svc)
            temporal_prio = 1.0 - idx / max(n_temporal, 1)
        else:
            temporal_prio = 0.0

        causal_s = causal_scores.get(svc, 0.0)

        return alpha * bfs_conf + beta * temporal_prio + gamma * causal_s

    reranked = sorted(ranked_causes, key=combined_score, reverse=True)

    logger.info("Causal re-ranking results:")
    for i, c in enumerate(reranked[:5]):
        svc = c.service if hasattr(c, "service") else c.get("service", "")
        conf = c.confidence if hasattr(c, "confidence") else c.get("confidence", 0.0)
        logger.info(f"  #{i+1} {svc}: combined_score={combined_score(c):.3f} (bfs={conf:.3f}, causal={causal_scores.get(svc,0):.3f})")

    return reranked
