"""eBPF Tools — STRATUS Pillar 3: Observability Fidelity.

Interface for pulling kernel-level metrics via eBPF (Rex framework).
Provides stubs when Rex/eBPF is not available (e.g., local dev).

In production, Rex (xlab-uiuc) exposes a gRPC / HTTP endpoint
that this module queries for:
  - Socket latency (tcp_sendmsg, tcp_recvmsg)
  - Syscall errors (open, read, write, connect)
  - Network drops, retransmissions
  - File descriptor exhaustion
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Env flag to enable real eBPF endpoint
REX_ENDPOINT = os.getenv("REX_EBPF_ENDPOINT", "")


# ── eBPF Metric Fetcher ─────────────────────────────────────────────────────


def get_ebpf_metrics(node_name: str, window_seconds: int = 60) -> Dict[str, Any]:
    """Fetch kernel-level eBPF metrics for a node/pod.

    Queries the Rex eBPF framework endpoint if configured,
    otherwise returns structured stub data for local testing.

    Args:
        node_name: Kubernetes node or pod name
        window_seconds: Observation window in seconds

    Returns:
        Dict with socket_latency_ms, syscall_errors, net_drops,
        tcp_retransmits, fd_exhaustion as keys
    """
    if REX_ENDPOINT:
        return _fetch_from_rex(node_name, window_seconds)
    else:
        logger.debug(f"REX_EBPF_ENDPOINT not set — returning stub metrics for {node_name}")
        return _stub_metrics(node_name)


def _fetch_from_rex(node_name: str, window_seconds: int) -> Dict[str, Any]:
    """Query the Rex eBPF endpoint for real kernel metrics.

    Args:
        node_name: Target node/pod
        window_seconds: Lookback window

    Returns:
        Kernel metrics dict
    """
    try:
        import urllib.request
        import json

        url = f"{REX_ENDPOINT}/metrics/{node_name}?window={window_seconds}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            logger.info(f"eBPF metrics fetched for {node_name}: {list(data.keys())}")
            return data
    except Exception as e:
        logger.warning(f"Rex eBPF fetch failed for {node_name}: {e} — using stubs")
        return _stub_metrics(node_name)


def _stub_metrics(node_name: str) -> Dict[str, Any]:
    """Return stub eBPF metrics for local/CI environments."""
    return {
        "node": node_name,
        "timestamp": time.time(),
        "source": "stub",
        "socket_latency_ms": {
            "p50": 0.5,
            "p95": 1.2,
            "p99": 3.1,
        },
        "syscall_errors": {
            "total": 0,
            "ECONNREFUSED": 0,
            "ETIMEDOUT": 0,
            "EBADF": 0,
        },
        "net_drops": 0,
        "tcp_retransmits": 0,
        "fd_exhaustion": False,
        "open_file_descriptors": 124,
    }


# ── Silent Failure Detection ─────────────────────────────────────────────────


def detect_silent_kernel_failures(
    metrics: Dict[str, Any],
    latency_threshold_ms: float = 50.0,
    error_threshold: int = 10,
) -> List[Dict[str, Any]]:
    """Flag kernel-level anomalies invisible to application tracing.

    "Silent" failures show up in eBPF data (socket timeouts, syscall
    errors) but not in distributed traces because the issue is below
    the instrumentation layer.

    Args:
        metrics: Dict from get_ebpf_metrics()
        latency_threshold_ms: p99 socket latency alert threshold
        error_threshold: Total syscall error count threshold

    Returns:
        List of silent failure signals with type and evidence
    """
    node = metrics.get("node", "unknown")
    signals = []

    # 1. Socket latency spike
    socket_lat = metrics.get("socket_latency_ms", {})
    p99 = socket_lat.get("p99", 0)
    if p99 >= latency_threshold_ms:
        signals.append({
            "type": "kernel_socket_latency",
            "node": node,
            "severity": "HIGH" if p99 > 200 else "MEDIUM",
            "value_ms": p99,
            "threshold_ms": latency_threshold_ms,
            "evidence": f"Socket p99 latency={p99:.1f}ms (threshold={latency_threshold_ms}ms)",
        })

    # 2. Syscall error surge
    syscall = metrics.get("syscall_errors", {})
    total_errors = syscall.get("total", 0)
    if total_errors >= error_threshold:
        dominant = max(
            {k: v for k, v in syscall.items() if k != "total"}.items(),
            key=lambda x: x[1],
            default=("unknown", 0),
        )
        signals.append({
            "type": "syscall_error_spike",
            "node": node,
            "severity": "CRITICAL" if total_errors > 100 else "HIGH",
            "total_errors": total_errors,
            "dominant_error": dominant[0],
            "evidence": f"Syscall errors={total_errors} (dominant={dominant[0]}:{dominant[1]})",
        })

    # 3. TCP retransmissions
    retransmits = metrics.get("tcp_retransmits", 0)
    if retransmits > 50:
        signals.append({
            "type": "tcp_retransmission_spike",
            "node": node,
            "severity": "MEDIUM",
            "count": retransmits,
            "evidence": f"TCP retransmits={retransmits} (possible network congestion)",
        })

    # 4. File descriptor exhaustion
    if metrics.get("fd_exhaustion", False):
        fd_count = metrics.get("open_file_descriptors", 0)
        signals.append({
            "type": "fd_exhaustion",
            "node": node,
            "severity": "CRITICAL",
            "open_fds": fd_count,
            "evidence": f"File descriptor exhaustion detected ({fd_count} open FDs)",
        })

    if signals:
        logger.warning(f"eBPF: {len(signals)} silent kernel failures on {node}")
    else:
        logger.info(f"eBPF: No silent kernel failures on {node}")

    return signals


def get_ebpf_signals_for_suspects(
    suspect_services: List[str],
    spans: List[Any],
) -> List[Dict[str, Any]]:
    """Fetch eBPF signals for all suspect services/nodes.

    Maps service names to node names (uses service name as proxy
    when exact node mapping is unavailable).

    Args:
        suspect_services: Services flagged by RCA
        spans: All trace spans

    Returns:
        Flat list of kernel failure signals across all suspects
    """
    all_signals = []

    for svc in suspect_services:
        metrics = get_ebpf_metrics(svc)
        signals = detect_silent_kernel_failures(metrics)
        for sig in signals:
            sig["service"] = svc
        all_signals.extend(signals)

    logger.info(f"eBPF scan: {len(all_signals)} signals from {len(suspect_services)} services")
    return all_signals
