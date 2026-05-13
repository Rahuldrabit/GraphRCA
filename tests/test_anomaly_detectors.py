"""Tests for Phase 1 — anomaly_detectors.py (factory, EWMA, IsolationForest)."""

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_span(service, duration_ms=100.0, has_error=False, span_id="s1", parent="ROOT"):
    return {
        "span_id": span_id,
        "parent_span": parent,
        "service_name": service,
        "duration_ms": duration_ms,
        "has_error": has_error,
        "response": "200",
    }


def _make_stats(service, duration_mean=100.0, error_rate=0.0, span_count=10):
    return {
        service: {
            "duration_mean_ms": duration_mean,
            "duration_p50_ms": duration_mean,
            "duration_p95_ms": duration_mean * 1.2,
            "error_rate": error_rate,
            "span_count": span_count,
            "unknown_response_pct": 0.0,
        }
    }


# ── Factory tests ──────────────────────────────────────────────────────────────

def test_get_detector_ewma_forced():
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import get_detector, EWMADetector
    d = get_detector(mode="ewma", span_count=100)
    assert isinstance(d, EWMADetector)


def test_get_detector_if_forced():
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import get_detector, IsolationForestDetector
    d = get_detector(mode="isolation_forest", span_count=5)
    assert isinstance(d, IsolationForestDetector)


def test_get_detector_auto_small_sample():
    """Auto mode picks EWMA when span_count < threshold (30)."""
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import get_detector, EWMADetector
    d = get_detector(mode="auto", span_count=10)
    assert isinstance(d, EWMADetector)


def test_get_detector_auto_large_sample():
    """Auto mode picks IsolationForest when span_count >= threshold (30)."""
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import get_detector, IsolationForestDetector
    d = get_detector(mode="auto", span_count=50)
    assert isinstance(d, IsolationForestDetector)


def test_get_detector_unknown_mode_falls_through_to_ewma():
    """Unknown mode string defaults to EWMA via auto path."""
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import get_detector, EWMADetector
    d = get_detector(mode="unknown_xyz", span_count=5)
    assert isinstance(d, EWMADetector)


# ── EWMADetector tests ─────────────────────────────────────────────────────────

def test_ewma_detector_returns_alerts_and_baselines():
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import EWMADetector

    spans = [_make_span("svc-a", duration_ms=50.0, span_id=f"s{i}") for i in range(5)]
    stats = _make_stats("svc-a", duration_mean=50.0, error_rate=0.0, span_count=5)

    d = EWMADetector()
    alerts, baselines = d.detect(spans, stats)

    assert isinstance(alerts, list)
    assert isinstance(baselines, dict)
    assert "svc-a" in baselines


def test_ewma_detector_fires_on_high_error_rate():
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import EWMADetector

    spans = [
        _make_span("bad-svc", duration_ms=100.0, has_error=True, span_id=f"s{i}")
        for i in range(5)
    ]
    stats = _make_stats("bad-svc", duration_mean=100.0, error_rate=1.0, span_count=5)

    d = EWMADetector()
    alerts, _ = d.detect(spans, stats, error_threshold=0.05)

    assert any(a.service == "bad-svc" for a in alerts)


# ── IsolationForestDetector tests ──────────────────────────────────────────────

def test_if_detector_falls_back_to_ewma_on_small_sample():
    """IF detector returns EWMA results when span_count < MIN_SAMPLES."""
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import IsolationForestDetector

    spans = [_make_span("svc-x", span_id=f"s{i}") for i in range(5)]
    stats = _make_stats("svc-x", error_rate=0.0, span_count=5)

    d = IsolationForestDetector()
    d.MIN_SAMPLES = 30  # ensure we're below threshold
    alerts, baselines = d.detect(spans, stats)

    assert isinstance(alerts, list)
    assert isinstance(baselines, dict)


def test_if_detector_falls_back_when_sklearn_missing():
    """IF detector gracefully falls back to EWMA when sklearn not available."""
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import IsolationForestDetector

    spans = [_make_span("svc-y", span_id=f"s{i}") for i in range(40)]
    stats = _make_stats("svc-y", error_rate=0.0, span_count=40)

    d = IsolationForestDetector()
    d.MIN_SAMPLES = 5  # force past the sample check

    with patch.dict("sys.modules", {"sklearn": None, "sklearn.ensemble": None}):
        alerts, baselines = d.detect(spans, stats)

    assert isinstance(alerts, list)
    assert isinstance(baselines, dict)


def test_if_detector_returns_no_alerts_for_empty_service_stats():
    from GraphRCA_agent.tools.pipeline.anomaly_detectors import IsolationForestDetector

    spans = [_make_span("svc-z", span_id=f"s{i}") for i in range(40)]
    d = IsolationForestDetector()
    d.MIN_SAMPLES = 5
    alerts, baselines = d.detect(spans, {})

    assert alerts == []
    assert isinstance(baselines, dict)


# ── LLM scorer tests ───────────────────────────────────────────────────────────

def test_llm_scorer_applies_multipliers():
    """ROOT_CAUSE multiplier raises score; FALSE_POSITIVE lowers it."""
    from GraphRCA_agent.tools.pipeline.llm_scorer import contextual_score_alerts

    mock_alert = MagicMock()
    mock_alert.service = "svc-a"
    mock_alert.score = 0.5
    mock_alert.z_score = 3.0
    mock_alert.anomaly_type = "latency_spike"
    mock_alert.severity = "HIGH"
    mock_alert.details = "test"

    llm_json = '{"classifications": [{"service": "svc-a", "classification": "ROOT_CAUSE", "reason": "high error"}]}'

    with patch("GraphRCA_agent.llm.llm_reason", return_value=llm_json):
        result = contextual_score_alerts([mock_alert], graph=None, pagerank={})

    assert len(result) == 1
    assert result[0].score == round(0.5 * 1.2, 3)


def test_llm_scorer_never_removes_alerts():
    """Even FALSE_POSITIVE alerts are kept (score lowered, not removed)."""
    from GraphRCA_agent.tools.pipeline.llm_scorer import contextual_score_alerts

    alerts = []
    for i in range(3):
        a = MagicMock()
        a.service = f"svc-{i}"
        a.score = 0.4
        a.z_score = 1.0
        a.anomaly_type = "latency_spike"
        a.severity = "LOW"
        a.details = ""
        alerts.append(a)

    llm_json = '{"classifications": [{"service": "svc-0", "classification": "FALSE_POSITIVE", "reason": "noise"}]}'

    with patch("GraphRCA_agent.llm.llm_reason", return_value=llm_json):
        result = contextual_score_alerts(alerts, graph=None, pagerank={})

    assert len(result) == 3  # none removed
    svc0 = next(r for r in result if r.service == "svc-0")
    assert svc0.score == round(0.4 * 0.3, 3)


def test_llm_scorer_returns_original_on_llm_failure():
    """If LLM call raises, original alerts returned unchanged."""
    from GraphRCA_agent.tools.pipeline.llm_scorer import contextual_score_alerts

    alert = MagicMock()
    alert.service = "svc-a"
    alert.score = 0.6
    alert.z_score = 2.0
    alert.anomaly_type = "error_rate_high"
    alert.severity = "HIGH"
    alert.details = ""

    with patch("GraphRCA_agent.llm.llm_reason", side_effect=RuntimeError("API down")):
        result = contextual_score_alerts([alert])

    assert result == [alert]
