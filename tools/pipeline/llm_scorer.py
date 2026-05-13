"""LLM Contextual Alert Scorer.

After anomaly detection produces AlertSignal objects, this module sends the top
alerts to the LLM with service-dependency context and asks it to classify each
alert as ROOT_CAUSE, SYMPTOM, or FALSE_POSITIVE.

Confidence multipliers applied to alert scores:
  ROOT_CAUSE    → 1.2x (more likely to drive BFS ranking)
  SYMPTOM       → 0.8x (propagated from an upstream root cause)
  FALSE_POSITIVE → 0.3x (likely noise)

The LLM scorer NEVER removes alerts — it only adjusts scores. This ensures
the downstream BFS always has candidates to work with.

Usage:
    alerts = contextual_score_alerts(alerts, graph, pagerank)
"""

import json
import logging
import os
from copy import copy
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Max alerts to batch into a single LLM call (keeps token cost bounded)
_MAX_ALERTS_TO_SCORE = int(os.getenv("GRAPHRCA_LLM_SCORER_MAX_ALERTS", "8"))

_MULTIPLIERS = {
    "ROOT_CAUSE": 1.2,
    "SYMPTOM": 0.8,
    "FALSE_POSITIVE": 0.3,
}

_SYSTEM_PROMPT = """\
You are an expert SRE anomaly analyst for distributed microservice systems.
Given a list of anomalous services and their dependency graph, classify each
anomaly as one of:
  ROOT_CAUSE   — this service is likely the origin of the fault
  SYMPTOM      — this service is affected because an upstream dependency is failing
  FALSE_POSITIVE — the anomaly is likely noise and not a real fault

Rules:
- Services that call failing downstream services are upstream callers.
  Their errors are likely SYMPTOM if the downstream has a higher anomaly score.
- Upstream services with independent high error rates are more likely ROOT_CAUSE.
- Services with marginal scores (< 0.35) and no error rate are likely FALSE_POSITIVE.
- If you cannot determine, use ROOT_CAUSE (safer to over-flag than under-flag).

Respond ONLY with valid JSON, no markdown, no prose:
{"classifications": [{"service": "...", "classification": "ROOT_CAUSE|SYMPTOM|FALSE_POSITIVE", "reason": "..."}]}
"""


def _build_graph_context(graph: Any, services: List[str]) -> str:
    """Summarise the call edges involving the alerted services."""
    if graph is None:
        return "No graph available."

    edges = []
    try:
        for u, v in graph.edges():
            if u in services or v in services:
                edge_type = graph.edges[u, v].get("edge_type", "CALLS")
                if edge_type == "CALLS":
                    edges.append(f"  {u} --CALLS--> {v}")
    except Exception:
        pass

    if not edges:
        return "No dependency edges among alerted services."

    return "Service dependency edges (caller → callee):\n" + "\n".join(edges[:30])


def _build_alert_summary(alerts: List[Any], pagerank: Dict[str, float]) -> str:
    """Format alerts for LLM prompt."""
    lines = []
    for i, a in enumerate(alerts, 1):
        svc = getattr(a, "service", "?")
        score = getattr(a, "score", 0.0)
        z = getattr(a, "z_score", 0.0)
        atype = getattr(a, "anomaly_type", "?")
        severity = getattr(a, "severity", "?")
        err_rate_hint = ""
        if "error" in str(atype).lower():
            err_rate_hint = " [has error_rate anomaly]"
        pr = pagerank.get(svc, 0.0)
        lines.append(
            f"{i}. service={svc}, anomaly_score={score:.3f}, z_score={z:.2f}, "
            f"type={atype}, severity={severity}, pagerank={pr:.4f}{err_rate_hint}"
        )
    return "\n".join(lines)


def contextual_score_alerts(
    alerts: List[Any],
    graph: Optional[Any] = None,
    pagerank: Optional[Dict[str, float]] = None,
) -> List[Any]:
    """Classify alerts via LLM and adjust their scores.

    Args:
        alerts:   List of AlertSignal objects from the detector
        graph:    nx.DiGraph (optional) for dependency context
        pagerank: Dict[service, score] for centrality context

    Returns:
        Adjusted list of AlertSignal objects (scores multiplied, never removed)
    """
    if not alerts:
        return alerts

    pagerank = pagerank or {}
    top_alerts = alerts[:_MAX_ALERTS_TO_SCORE]
    alerted_services = [getattr(a, "service", "") for a in top_alerts]

    alert_text = _build_alert_summary(top_alerts, pagerank)
    graph_text = _build_graph_context(graph, alerted_services)

    prompt = (
        f"Alerted services:\n{alert_text}\n\n"
        f"{graph_text}\n\n"
        "Classify each service. Return JSON only."
    )

    try:
        from GraphRCA_agent.llm import llm_reason
        raw = llm_reason(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=800,
            caller="llm_scorer",
        )
    except Exception as e:
        logger.warning(f"[LLMScorer] LLM call failed: {e} — skipping scorer")
        return alerts

    # Parse LLM response
    classifications: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        for item in parsed.get("classifications", []):
            svc = item.get("service", "")
            cls = item.get("classification", "ROOT_CAUSE").upper()
            if cls not in _MULTIPLIERS:
                cls = "ROOT_CAUSE"
            classifications[svc] = cls
            reasons[svc] = item.get("reason", "")
    except Exception as e:
        logger.warning(f"[LLMScorer] Failed to parse LLM response: {e} — skipping")
        return alerts

    # Apply multipliers to copies of top alerts
    adjusted: List[Any] = []
    for a in top_alerts:
        svc = getattr(a, "service", "")
        cls = classifications.get(svc, "ROOT_CAUSE")
        multiplier = _MULTIPLIERS.get(cls, 1.0)
        new_score = round(min(1.0, max(0.01, a.score * multiplier)), 3)

        # Shallow copy to avoid mutating the original
        a_copy = copy(a)
        a_copy.score = new_score
        a_copy.details = (
            getattr(a, "details", "")
            + f" | LLM: {cls} ({reasons.get(svc, '')})"
        )
        adjusted.append(a_copy)

        logger.info(
            f"[LLMScorer] {svc}: {cls} — score {a.score:.3f} → {new_score:.3f} "
            f"(×{multiplier})"
        )

    # Re-sort adjusted top alerts by new score
    adjusted.sort(key=lambda x: x.score, reverse=True)

    # Append remaining alerts (beyond top N) unchanged
    remaining = alerts[_MAX_ALERTS_TO_SCORE:]
    return adjusted + remaining
