"""Detection agent tools (standalone, Stratus-compatible).

The LangGraph `detection` node expects:
  - compute_ewma_baseline(...) -> Dict[str, EWMABaseline]
  - detect_all_anomalies(..., error_threshold=...) -> List[AlertSignal]

Alerts are dataclasses with attributes like `.service`, `.score`, `.severity`, `.anomaly_type`.
"""

import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _service_name(span: Any) -> str:
    if hasattr(span, "service_name"):
        return str(getattr(span, "service_name") or "unknown").strip() or "unknown"
    if isinstance(span, dict):
        return str(span.get("service_name") or span.get("service") or "unknown").strip() or "unknown"
    return "unknown"


def _start_time(span: Any) -> int:
    if hasattr(span, "start_time"):
        try:
            return int(getattr(span, "start_time") or 0)
        except Exception:
            return 0
    if isinstance(span, dict):
        try:
            return int(span.get("start_time") or 0)
        except Exception:
            return 0
    return 0


def _duration_ms(span: Any) -> float:
    if hasattr(span, "duration_ms"):
        try:
            return float(getattr(span, "duration_ms") or 0.0)
        except Exception:
            return 0.0
    if isinstance(span, dict):
        try:
            return float(span.get("duration_ms") or 0.0)
        except Exception:
            return 0.0
    return 0.0


def _operation_name(span: Any) -> str:
    if hasattr(span, "operation_name"):
        return str(getattr(span, "operation_name") or "").strip()
    if isinstance(span, dict):
        return str(span.get("operation_name") or span.get("operation") or "").strip()
    return ""


def _is_request_span(span: Any) -> bool:
    """Best-effort: identify request-handler spans.

    - gRPC server spans often look like: "/rate.Rate/GetRates"
    - HTTP server spans often look like: "HTTP GET /path"

    Internal spans (db/cache) usually don't match these patterns.
    """
    op = _operation_name(span)
    if not op:
        return False
    if op.startswith("/"):
        return True
    if op.upper().startswith("HTTP "):
        return True
    return False


@dataclass
class EWMABaseline:
    service: str
    metric: str
    ewma_mean: float
    ewma_std: float
    sample_count: int
    alpha: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AlertSignal:
    service: str
    severity: str
    anomaly_type: str
    score: float
    z_score: float
    current_value: float
    baseline_mean: float
    baseline_std: float
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_ewma_baseline(
    spans: List[Any],
    alpha: float = 0.3,
    window_size: int = 100,
) -> Dict[str, EWMABaseline]:
    """Calculate EWMA rolling average + std per service."""
    # Prefer request-handler spans for baselines to avoid mixing wildly different
    # internal operations (e.g., cache lookups) into a single service latency metric.
    service_durations_req: Dict[str, List[float]] = defaultdict(list)
    service_durations_all: Dict[str, List[float]] = defaultdict(list)

    for span in sorted(spans, key=_start_time):
        svc = _service_name(span)
        dur = _duration_ms(span)
        service_durations_all[svc].append(dur)
        if _is_request_span(span):
            service_durations_req[svc].append(dur)

    baselines: Dict[str, EWMABaseline] = {}
    # Use request durations when available; otherwise fall back to all spans.
    all_services = set(service_durations_all.keys()) | set(service_durations_req.keys())
    for service in sorted(all_services):
        durations = service_durations_req.get(service, [])
        if len(durations) < 2:
            durations = service_durations_all.get(service, [])
        recent = durations[-window_size:]
        if len(recent) < 2:
            baselines[service] = EWMABaseline(
                service=service,
                metric="duration_ms",
                ewma_mean=round(recent[0], 3) if recent else 0.0,
                ewma_std=0.0,
                sample_count=len(recent),
                alpha=alpha,
            )
            continue

        ewma = recent[0]
        for val in recent[1:]:
            ewma = alpha * val + (1 - alpha) * ewma

        ewma_var = 0.0
        ewma_temp = recent[0]
        for val in recent[1:]:
            ewma_temp = alpha * val + (1 - alpha) * ewma_temp
            diff = val - ewma_temp
            ewma_var = alpha * (diff ** 2) + (1 - alpha) * ewma_var

        ewma_std = math.sqrt(ewma_var) if ewma_var > 0 else 0.0

        baselines[service] = EWMABaseline(
            service=service,
            metric="duration_ms",
            ewma_mean=round(float(ewma), 3),
            ewma_std=round(float(ewma_std), 3),
            sample_count=len(recent),
            alpha=alpha,
        )

    logger.info(f"Computed EWMA baselines for {len(baselines)} services")
    return baselines


def calculate_z_score(current_value: float, baseline: EWMABaseline) -> float:
    """How many standard deviations the current value is from normal."""
    if baseline.ewma_std == 0 or baseline.ewma_std < 0.001:
        if abs(current_value - baseline.ewma_mean) > 0.001:
            return 10.0
        return 0.0

    z = (current_value - baseline.ewma_mean) / baseline.ewma_std
    return round(float(z), 3)


def score_anomaly(z_score: float, error_rate: float, unknown_pct: float = 0.0) -> float:
    """Combine z-score + error rate + unknown response rate into single anomaly score."""
    z_component = min(1.0, abs(z_score) / 10.0) if abs(z_score) > 2.0 else 0.0
    error_component = min(1.0, float(error_rate) * 2.0)
    unknown_penalty = 0.2 if float(unknown_pct) > 80 else 0.0

    score = 0.40 * z_component + 0.45 * error_component + 0.15 * unknown_penalty
    return round(min(1.0, float(score)), 3)


def emit_alert_signal(
    service: str,
    anomaly_type: str,
    score: float,
    z_score: float,
    current_value: float,
    baseline: EWMABaseline,
) -> AlertSignal:
    """Create an anomaly alert signal for a service."""
    if score >= 0.8:
        severity = "CRITICAL"
    elif score >= 0.6:
        severity = "HIGH"
    elif score >= 0.4:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    details = (
        f"{service}: {anomaly_type} detected. "
        f"Current={current_value:.1f}ms, Baseline={baseline.ewma_mean:.1f}ms ± {baseline.ewma_std:.1f}ms, "
        f"Z-score={z_score:.2f}, Score={score:.3f}"
    )

    signal = AlertSignal(
        service=service,
        severity=severity,
        anomaly_type=anomaly_type,
        score=float(score),
        z_score=float(z_score),
        current_value=float(current_value),
        baseline_mean=float(baseline.ewma_mean),
        baseline_std=float(baseline.ewma_std),
        details=details,
    )
    logger.info(f"[ALERT] [{severity}] {details}")
    return signal


def detect_all_anomalies(
    spans: List[Any],
    service_stats: Dict[str, Dict[str, Any]],
    baselines: Dict[str, EWMABaseline],
    z_threshold: float = 3.0,
    error_threshold: float = 0.05,
    current_window_size: int = 10,
) -> List[AlertSignal]:
    """Run full anomaly detection across all services."""
    alerts: List[AlertSignal] = []

    # Precompute per-service duration series for a consistent current metric.
    service_durations_req: Dict[str, List[float]] = defaultdict(list)
    service_durations_all: Dict[str, List[float]] = defaultdict(list)
    for span in sorted(spans, key=_start_time):
        svc = _service_name(span)
        dur = _duration_ms(span)
        service_durations_all[svc].append(dur)
        if _is_request_span(span):
            service_durations_req[svc].append(dur)

    for service, stats in service_stats.items():
        baseline = baselines.get(service)
        if not baseline:
            continue

        # Use recent request-span latency as the current value (falls back to all spans).
        durations = service_durations_req.get(service, [])
        if len(durations) < 2:
            durations = service_durations_all.get(service, [])
        if durations:
            tail_n = max(1, min(int(current_window_size), len(durations)))
            recent_tail = durations[-tail_n:]
            current_mean = float(sum(recent_tail) / len(recent_tail))
        else:
            current_mean = float(stats.get("duration_mean_ms", 0.0) or 0.0)
        z = calculate_z_score(current_mean, baseline)
        error_rate = float(stats.get("error_rate", 0.0) or 0.0)
        unknown_pct = float(stats.get("unknown_response_pct", 0.0) or 0.0)

        anomaly_score = score_anomaly(z, error_rate, unknown_pct)

        anomaly_types: List[str] = []
        if abs(z) >= z_threshold:
            anomaly_types.append("latency_spike")
        if error_rate >= error_threshold:
            anomaly_types.append("error_rate_high")
        if unknown_pct > 80:
            anomaly_types.append("unknown_response_gap")

        # Avoid false positives where the only signal is missing/unknown response
        # classification (common for gRPC spans that don't carry HTTP status codes).
        has_strong_signal = (abs(z) >= z_threshold) or (error_rate >= error_threshold)
        has_meaningful_score = anomaly_score > 0.3
        if (has_strong_signal or has_meaningful_score) and (anomaly_types or has_meaningful_score):
            anomaly_type = "+".join(anomaly_types) if anomaly_types else "multi_signal"
            alerts.append(
                emit_alert_signal(
                    service=service,
                    anomaly_type=anomaly_type,
                    score=anomaly_score,
                    z_score=z,
                    current_value=current_mean,
                    baseline=baseline,
                )
            )

    alerts.sort(key=lambda a: a.score, reverse=True)
    logger.info(f"Detection complete: {len(alerts)} alerts from {len(service_stats)} services")
    return alerts
