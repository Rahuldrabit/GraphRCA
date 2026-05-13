"""Pluggable anomaly detector registry.

Provides two detectors:
  - EWMADetector: thin wrapper around existing compute_ewma_baseline + detect_all_anomalies
  - IsolationForestDetector: sklearn-based multivariate anomaly detection
  - get_detector(mode, span_count): factory that selects the right detector

Usage:
    detector = get_detector(mode="auto", span_count=len(spans))
    alerts, baselines = detector.detect(spans, service_stats)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum spans required to use IsolationForest meaningfully
_IF_MIN_SAMPLES = int(os.getenv("GRAPHRCA_IF_MIN_SAMPLES", "30"))
_IF_CONTAMINATION = float(os.getenv("GRAPHRCA_IF_CONTAMINATION", "0.1"))


class EWMADetector:
    """Thin wrapper around existing EWMA baseline + anomaly detection logic."""

    def detect(
        self,
        spans: List[Any],
        service_stats: Dict[str, Dict[str, Any]],
        alpha: float = 0.3,
        window_size: int = 100,
        z_threshold: float = 3.0,
        error_threshold: float = 0.05,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Run EWMA-based detection.

        Returns:
            (alerts, baselines) — same types as compute_ewma_baseline + detect_all_anomalies
        """
        from GraphRCA_agent.tools.pipeline.detection_tools import (
            compute_ewma_baseline,
            detect_all_anomalies,
        )

        baselines = compute_ewma_baseline(spans, alpha=alpha, window_size=window_size)
        alerts = detect_all_anomalies(
            spans=spans,
            service_stats=service_stats,
            baselines=baselines,
            z_threshold=z_threshold,
            error_threshold=error_threshold,
        )
        return alerts, baselines


class IsolationForestDetector:
    """Multivariate anomaly detection using sklearn IsolationForest.

    Feature vector per service:
        [latency_ratio, error_rate, throughput_share, tail_ratio, unknown_pct]

    Falls back to EWMADetector if fewer than MIN_SAMPLES spans are available
    or if sklearn is not installed.
    """

    MIN_SAMPLES = _IF_MIN_SAMPLES

    def detect(
        self,
        spans: List[Any],
        service_stats: Dict[str, Dict[str, Any]],
        alpha: float = 0.3,
        window_size: int = 100,
        z_threshold: float = 3.0,
        error_threshold: float = 0.05,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Run IsolationForest-based detection with EWMA fallback for baselines.

        Returns:
            (alerts, baselines)
        """
        from GraphRCA_agent.tools.pipeline.detection_tools import (
            compute_ewma_baseline,
            AlertSignal,
            EWMABaseline,
        )

        # Always compute EWMA baselines (needed downstream for z-score evidence)
        baselines = compute_ewma_baseline(spans, alpha=alpha, window_size=window_size)

        if len(spans) < self.MIN_SAMPLES:
            logger.info(
                f"[IFDetector] Only {len(spans)} spans (< {self.MIN_SAMPLES}), "
                "falling back to EWMA"
            )
            from GraphRCA_agent.tools.pipeline.detection_tools import detect_all_anomalies
            alerts = detect_all_anomalies(
                spans=spans,
                service_stats=service_stats,
                baselines=baselines,
                z_threshold=z_threshold,
                error_threshold=error_threshold,
            )
            return alerts, baselines

        try:
            # Lazy import to avoid adding startup latency when IF is not used
            from sklearn.ensemble import IsolationForest
            import numpy as np
        except ImportError:
            logger.warning(
                "[IFDetector] scikit-learn not installed, falling back to EWMA"
            )
            from GraphRCA_agent.tools.pipeline.detection_tools import detect_all_anomalies
            alerts = detect_all_anomalies(
                spans=spans,
                service_stats=service_stats,
                baselines=baselines,
                z_threshold=z_threshold,
                error_threshold=error_threshold,
            )
            return alerts, baselines

        services = list(service_stats.keys())
        if not services:
            return [], baselines

        # Build feature matrix — one row per service
        total_spans = max(1, len(spans))
        feature_rows = []
        for svc in services:
            stats = service_stats[svc]
            baseline = baselines.get(svc)

            latency_mean = float(stats.get("duration_mean_ms", 0.0) or 0.0)
            baseline_mean = float(baseline.ewma_mean) if baseline else latency_mean
            latency_ratio = latency_mean / baseline_mean if baseline_mean > 0 else 1.0

            error_rate = float(stats.get("error_rate", 0.0) or 0.0)
            span_count = float(stats.get("span_count", 0) or 0)
            throughput_share = span_count / total_spans

            p95 = float(stats.get("duration_p95_ms", latency_mean) or latency_mean)
            p50 = float(stats.get("duration_p50_ms", latency_mean) or latency_mean)
            tail_ratio = p95 / p50 if p50 > 0 else 1.0

            unknown_pct = float(stats.get("unknown_response_pct", 0.0) or 0.0) / 100.0

            feature_rows.append([
                latency_ratio,
                error_rate,
                throughput_share,
                tail_ratio,
                unknown_pct,
            ])

        X = np.array(feature_rows, dtype=float)

        # Fit IsolationForest
        model = IsolationForest(
            contamination=_IF_CONTAMINATION,
            n_estimators=100,
            random_state=42,
        )
        model.fit(X)

        # decision_function returns: positive = normal, negative = anomaly
        raw_scores = model.decision_function(X)
        # Invert and normalise to [0, 1] — higher means more anomalous
        anomaly_scores = (-raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-9
        )

        alerts: List[AlertSignal] = []
        for idx, svc in enumerate(services):
            stats = service_stats[svc]
            baseline = baselines.get(svc)
            if baseline is None:
                continue

            anomaly_score = float(anomaly_scores[idx])
            error_rate = float(stats.get("error_rate", 0.0) or 0.0)
            latency_mean = float(stats.get("duration_mean_ms", 0.0) or 0.0)

            # Compute z-score for evidence logging
            if baseline.ewma_std > 0.001:
                z = (latency_mean - baseline.ewma_mean) / baseline.ewma_std
            elif abs(latency_mean - baseline.ewma_mean) > 0.001:
                z = 10.0
            else:
                z = 0.0

            # Only emit an alert if the anomaly score is meaningful
            # or there's a clear error rate signal
            if anomaly_score < 0.3 and error_rate < error_threshold:
                continue

            # Determine severity
            if anomaly_score >= 0.8:
                severity = "CRITICAL"
            elif anomaly_score >= 0.6:
                severity = "HIGH"
            elif anomaly_score >= 0.4:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            # Compose anomaly type from signals
            anomaly_types = []
            if abs(z) >= z_threshold:
                anomaly_types.append("latency_spike")
            if error_rate >= error_threshold:
                anomaly_types.append("error_rate_high")
            anomaly_type = "+".join(anomaly_types) if anomaly_types else "if_multivariate"

            details = (
                f"{svc}: IF anomaly_score={anomaly_score:.3f}, "
                f"error_rate={error_rate:.1%}, z={z:.2f}"
            )

            alert = AlertSignal(
                service=svc,
                severity=severity,
                anomaly_type=anomaly_type,
                score=round(anomaly_score, 3),
                z_score=round(z, 3),
                current_value=latency_mean,
                baseline_mean=baseline.ewma_mean,
                baseline_std=baseline.ewma_std,
                details=details,
            )
            alerts.append(alert)
            logger.info(f"[IFDetector] [{severity}] {details}")

        alerts.sort(key=lambda a: a.score, reverse=True)
        logger.info(
            f"[IFDetector] Detection complete: {len(alerts)} alerts "
            f"from {len(services)} services"
        )
        return alerts, baselines


def get_detector(mode: str = "auto", span_count: int = 0):
    """Factory: return the appropriate detector based on mode and data volume.

    Args:
        mode: "ewma" | "isolation_forest" | "auto"
        span_count: Number of spans available (used by "auto" mode)

    Returns:
        EWMADetector or IsolationForestDetector instance
    """
    mode = mode.lower().strip()

    if mode == "isolation_forest":
        logger.info("[DetectorFactory] Using IsolationForestDetector (forced)")
        return IsolationForestDetector()

    if mode == "ewma":
        logger.info("[DetectorFactory] Using EWMADetector (forced)")
        return EWMADetector()

    # auto: use IF when enough data, else EWMA
    if span_count >= _IF_MIN_SAMPLES:
        logger.info(
            f"[DetectorFactory] Auto-selected IsolationForestDetector "
            f"({span_count} spans >= {_IF_MIN_SAMPLES} threshold)"
        )
        return IsolationForestDetector()
    else:
        logger.info(
            f"[DetectorFactory] Auto-selected EWMADetector "
            f"({span_count} spans < {_IF_MIN_SAMPLES} threshold)"
        )
        return EWMADetector()
