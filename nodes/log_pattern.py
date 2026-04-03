"""Log Pattern Node — LangGraph agent node (STRATUS Pillar 3).

Collects logs from suspect services (identified by RCA),
then uses the LLM to cluster and summarize error patterns.
Reduces token load by grouping similar errors before main analysis.
"""

import logging
import time
from typing import Any, Dict, List

from GraphRCA_agent.state import PipelineState
from GraphRCA_agent.llm import llm_reason
from GraphRCA_agent.tools.ebpf_tools import get_ebpf_signals_for_suspects

logger = logging.getLogger(__name__)

# Max log lines sent to LLM for clustering
MAX_LOG_LINES = 60


def _extract_log_lines_from_spans(spans: List[Any], services: List[str]) -> List[str]:
    """Extract error-related information from spans as synthetic log lines.

    Since AIOpsLab provides traces (not raw logs), we synthesise log-like
    entries from error spans for LLM clustering.

    Args:
        spans: All trace spans
        services: Suspect service names

    Returns:
        List of log-like strings
    """
    log_lines = []
    for span in spans:
        svc = span.service_name if hasattr(span, "service_name") else span.get("service_name", "")
        if svc not in services:
            continue
        has_error = span.has_error if hasattr(span, "has_error") else span.get("has_error", False)
        if not has_error:
            continue

        op = span.operation_name if hasattr(span, "operation_name") else span.get("operation_name", "")
        resp = span.response if hasattr(span, "response") else span.get("response", "")
        dur_ms = span.duration_ms if hasattr(span, "duration_ms") else span.get("duration_ms", 0)
        trace_id = span.trace_id if hasattr(span, "trace_id") else span.get("trace_id", "")

        log_lines.append(
            f"[{svc}] op={op} response={resp} duration={dur_ms:.0f}ms trace={trace_id[:8]}"
        )

    return log_lines[:MAX_LOG_LINES]


def log_pattern_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: cluster log patterns from suspect services.

    Reads:  spans, suspect_services, ranked_causes
    Writes: log_clusters
    """
    t0 = time.time()
    spans = state.get("spans", [])
    suspect_services = state.get("suspect_services", [])
    ranked_causes = state.get("ranked_causes", [])

    # Derive suspect services from ranked causes if not set
    if not suspect_services and ranked_causes:
        suspect_services = [
            c.service if hasattr(c, "service") else c.get("service", "")
            for c in ranked_causes[:3]
        ]

    if not suspect_services:
        return {
            "log_clusters": [],
            "messages": state.get("messages", []) + ["[LogPattern] Skipped: no suspect services"],
        }

    logger.info(f"[LogPattern] Analysing logs from {suspect_services}")

    try:
        # 1. Extract error log lines from spans
        log_lines = _extract_log_lines_from_spans(spans, suspect_services)

        # 2. Also fetch eBPF signals (Pillar 3 observability)
        ebpf_signals = get_ebpf_signals_for_suspects(suspect_services, spans)
        ebpf_text = ""
        if ebpf_signals:
            ebpf_text = "\n\nKernel-level eBPF signals:\n" + "\n".join(
                f"  [{s['type']}] {s['evidence']}" for s in ebpf_signals
            )

        clusters = []

        if log_lines:
            # 3. Use LLM to cluster error patterns
            log_text = "\n".join(log_lines)
            prompt = (
                f"You are an SRE log analyst. Below are error spans from suspect services "
                f"[{', '.join(suspect_services)}] in a distributed system incident.\n\n"
                f"Log entries:\n{log_text}"
                f"{ebpf_text}\n\n"
                "Group these into at most 5 error pattern clusters. "
                "For each cluster, output:\n"
                "CLUSTER: <pattern name>\n"
                "COUNT: <approximate number of occurrences>\n"
                "SEVERITY: <CRITICAL|HIGH|MEDIUM|LOW>\n"
                "SERVICES: <comma-separated service names>\n"
                "SUMMARY: <one sentence describing the pattern>\n"
                "---\n"
                "Be concise. Output only the clusters."
            )
            raw = llm_reason(prompt, max_tokens=800, caller="log_pattern_node")

            # Parse clusters from raw LLM output
            current = {}
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("CLUSTER:"):
                    if current:
                        clusters.append(current)
                    current = {"pattern": line[8:].strip(), "count": 1, "severity": "MEDIUM",
                               "services": suspect_services, "summary": ""}
                elif line.startswith("COUNT:") and current:
                    try:
                        current["count"] = int(line[6:].strip())
                    except ValueError:
                        pass
                elif line.startswith("SEVERITY:") and current:
                    current["severity"] = line[9:].strip()
                elif line.startswith("SERVICES:") and current:
                    current["services"] = [s.strip() for s in line[9:].split(",")]
                elif line.startswith("SUMMARY:") and current:
                    current["summary"] = line[8:].strip()
            if current:
                clusters.append(current)
        else:
            # No error spans — still report eBPF findings
            for sig in ebpf_signals:
                clusters.append({
                    "pattern": sig["type"],
                    "count": 1,
                    "severity": sig.get("severity", "MEDIUM"),
                    "services": [sig.get("service", "unknown")],
                    "summary": sig.get("evidence", ""),
                })

        elapsed = round(time.time() - t0, 2)
        logger.info(f"[LogPattern] {len(clusters)} clusters from {len(log_lines)} log lines in {elapsed}s")

        return {
            "log_clusters": clusters,
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[LogPattern] {len(clusters)} error patterns from {len(suspect_services)} suspect services"
            ],
            "node_timings": {**state.get("node_timings", {}), "log_pattern": elapsed},
        }

    except Exception as e:
        logger.warning(f"[LogPattern] Failed (non-fatal): {e}")
        return {
            "log_clusters": [],
            "messages": state.get("messages", []) + [f"[LogPattern] WARNING: {e}"],
        }
