"""Triage Agent Node — Multi-Agent Mode (Phase 3).

Reads the initial alert context and uses the LLM to classify the fault
category, then produces an investigation strategy for the Planner Agent.

Reads:  alerts, primary_error_service, service_stats, additional_context
Writes: investigation_strategy
"""

import json
import logging
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior SRE triage analyst. Given anomaly alerts from a microservice
system, classify the fault and specify what evidence to gather.

Fault categories:
  network      — network partitions, timeouts, DNS failures
  resource     — CPU/memory exhaustion, OOM, throttling
  config       — misconfigured services, wrong ports, env vars
  app_level    — application bugs, logic errors, dependency failures
  unknown      — cannot determine from current data

Return ONLY valid JSON (no markdown):
{
  "fault_category": "<network|resource|config|app_level|unknown>",
  "fault_hypothesis": "<1-2 sentence initial theory>",
  "worker_assignments": [
    {
      "mount": "<kubectl|prometheus|jaeger|loki>",
      "tool": "<tool_name>",
      "arguments": {<tool-specific args>},
      "rationale": "<why this query helps>"
    }
  ]
}

Rules:
- Limit worker_assignments to at most 6 entries.
- For kubectl: tool="exec_kubectl_cmd_safely", arguments={"cmd": "kubectl <command>"}
- For prometheus: tool="query_prometheus", arguments={"query": "<PromQL>"}
- For loki: tool="query_loki", arguments={"query": "<LogQL>", "limit": 50}
- For jaeger: tool="get_services", arguments={}  (or get_traces for a specific service)
- Prioritize the primary_error_service in your queries.
- If traces are absent (no spans), favour config/kubectl investigation.
"""


def triage_agent_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: classify fault and produce investigation strategy.

    Reads:  alerts, primary_error_service, service_stats, additional_context
    Writes: investigation_strategy
    """
    t0 = time.time()
    alerts = state.get("alerts", [])
    primary = state.get("primary_error_service", "unknown")
    service_stats = state.get("service_stats", {})
    additional_context = state.get("additional_context", "") or ""
    spans = state.get("spans", [])

    logger.info(f"[TriageAgent] Starting — primary={primary}, alerts={len(alerts)}")

    # Build prompt context
    alert_lines = []
    for a in alerts[:8]:
        svc = getattr(a, "service", "?")
        score = getattr(a, "score", 0.0)
        atype = getattr(a, "anomaly_type", "?")
        severity = getattr(a, "severity", "?")
        alert_lines.append(f"  {svc}: {severity} ({atype}, score={score:.3f})")

    service_list = list(service_stats.keys())[:15]
    span_count = len(spans)

    prompt = (
        f"Primary error service: {primary}\n"
        f"Total spans available: {span_count}\n\n"
        f"Alerts ({len(alerts)} total, showing top 8):\n"
        + ("\n".join(alert_lines) if alert_lines else "  (none)")
        + f"\n\nServices in system: {', '.join(service_list)}\n"
        + (f"\nAdditional context: {additional_context[:500]}\n" if additional_context else "")
        + "\nClassify the fault and specify which MCP queries to run."
    )

    # Default strategy (used if LLM fails)
    default_strategy: Dict[str, Any] = {
        "fault_category": "unknown",
        "fault_hypothesis": f"Anomaly detected on {primary}; investigation needed.",
        "worker_assignments": [
            {
                "mount": "kubectl",
                "tool": "exec_kubectl_cmd_safely",
                "arguments": {"cmd": f"kubectl get pods -o wide"},
                "rationale": "Check pod health and node assignments",
            },
            {
                "mount": "prometheus",
                "tool": "query_prometheus",
                "arguments": {"query": f'rate(http_requests_total{{service="{primary}"}}[5m])'},
                "rationale": "Check request rate for primary service",
            },
        ],
    }

    try:
        from GraphRCA_agent.llm import llm_reason

        raw = llm_reason(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=1000,
            caller="triage_agent",
        )

        # Parse JSON response
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        strategy = json.loads(text)

        # Validate required keys
        if "fault_category" not in strategy or "worker_assignments" not in strategy:
            raise ValueError("Missing required keys in strategy")

        # Cap worker assignments
        max_queries = int(__import__("os").getenv("GRAPHRCA_MAX_PARALLEL_QUERIES", "8"))
        strategy["worker_assignments"] = strategy["worker_assignments"][:max_queries]

        logger.info(
            f"[TriageAgent] Fault={strategy['fault_category']}, "
            f"workers={len(strategy['worker_assignments'])}"
        )

    except Exception as e:
        logger.warning(f"[TriageAgent] LLM failed ({e}), using default strategy")
        strategy = default_strategy

    elapsed = round(time.time() - t0, 2)
    return {
        "investigation_strategy": strategy,
        "status": "running",
        "messages": state.get("messages", []) + [
            f"[TriageAgent] fault={strategy.get('fault_category', '?')}, "
            f"workers={len(strategy.get('worker_assignments', []))}"
        ],
        "node_timings": {**state.get("node_timings", {}), "triage_agent": elapsed},
    }
