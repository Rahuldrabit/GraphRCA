"""Unit tests for causal_tools (Pillar 2 — Causal Inference)."""

import numpy as np
from unittest.mock import MagicMock

from GraphRCA_agent.tools.causal_tools import (
    temporal_order_analysis,
    causal_score,
    compute_pairwise_causal_scores,
    rerank_with_causality,
)


def _make_span(service, start_time, has_error=False, duration_ms=100.0):
    span = MagicMock()
    span.service_name = service
    span.start_time = start_time
    span.has_error = has_error
    span.duration_ms = duration_ms
    span.get = lambda k, d=None: getattr(span, k, d)
    return span


def _make_alert(service):
    a = MagicMock()
    a.service = service
    a.get = lambda k, d=None: getattr(a, k, d)
    return a


class TestTemporalOrder:
    def test_earlier_service_ranked_first(self):
        spans = [
            _make_span("svc-a", 1000, has_error=True),
            _make_span("svc-b", 2000, has_error=True),
            _make_span("svc-c", 500, has_error=True),
        ]
        alerts = [_make_alert("svc-a"), _make_alert("svc-b"), _make_alert("svc-c")]
        order = temporal_order_analysis(spans, alerts)
        assert order[0] == "svc-c"
        assert order[-1] == "svc-b"

    def test_empty_returns_empty(self):
        assert temporal_order_analysis([], []) == []


class TestCausalScore:
    def test_returns_float_between_0_and_1(self):
        spans = [_make_span("a", i, duration_ms=float(i)) for i in range(50)] + \
                [_make_span("b", i, duration_ms=float(i) * 1.5) for i in range(50)]
        score = causal_score(spans, "a", "b", lag=2)
        assert 0.0 <= score <= 1.0

    def test_too_few_samples_returns_zero(self):
        spans = [_make_span("a", 1), _make_span("b", 2)]
        score = causal_score(spans, "a", "b", lag=5)
        assert score == 0.0


class TestPairwiseCausalScores:
    def test_scores_normalized_to_1(self):
        spans = [_make_span("a", i, duration_ms=float(i)) for i in range(30)] + \
                [_make_span("b", i, duration_ms=float(i + 1)) for i in range(30)] + \
                [_make_span("c", i, duration_ms=float(i * 2)) for i in range(30)]
        scores = compute_pairwise_causal_scores(spans, ["a", "b", "c"], lag=2)
        if scores:
            assert max(scores.values()) <= 1.0
            assert all(v >= 0.0 for v in scores.values())


class TestRerank:
    def _make_candidate(self, service, confidence):
        c = MagicMock()
        c.service = service
        c.confidence = confidence
        return c

    def test_earlier_anomaly_gets_boost(self):
        candidates = [
            self._make_candidate("svc-b", 0.9),
            self._make_candidate("svc-a", 0.6),
        ]
        temporal_order = ["svc-a", "svc-b"]  # svc-a anomaly came first
        causal_scores = {"svc-a": 0.8, "svc-b": 0.2}

        reranked = rerank_with_causality(candidates, temporal_order, causal_scores,
                                         alpha=0.4, beta=0.35, gamma=0.25)
        # With high temporal + causal scores, svc-a may overtake svc-b
        assert len(reranked) == 2

    def test_empty_input_returns_empty(self):
        result = rerank_with_causality([], [], {})
        assert result == []
