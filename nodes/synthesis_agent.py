"""Synthesis Agent Node — Multi-Agent Mode (Phase 3).

Merges live MCP worker evidence with trace-based pipeline results.
Uses the LLM to resolve conflicts and adjust root cause confidence scores.

Reads:  ranked_causes, worker_results, investigation_strategy
Writes: ranked_causes (updated), live_evidence
"""

import json
import logging
import time
from copy import deepcopy
from typing import Any, Dict, List

from GraphRCA_agent.state import PipelineState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an SRE root cause synthesis analyst. You have two sources of evidence:
1. Trace-based analysis — root cause candidates ranked by the pipeline.
2. Live cluster data — results from real-time MCP queries (kubectl, prometheus, etc.).

Your job: evaluate whether the trace-based ranking is consistent with live data,
and produce an updated ranking with adjusted confidence scores.

Return ONLY valid JSON (no markdown):
{
  "updated_ranking": [
    {
      "service": "<service_name>",
      "confidence_delta": <float between -0.3 and +0.3>,
      "reason": "<why you adjusted>"
    }
  ],
  "live_summary": "<2-3 sentence summary of what live evidence showed>",
  "override_root_cause": "<service_name or null if no override>"
}

Rules:
- Only include services you have evidence for in updated_ranking.
- confidence_delta is added to the existing confidence (clamped to [0, 1]).
- If live data strongly contradicts the trace analysis, set override_root_cause.
- If live data is unavailable or inconclusive, return empty updated_ranking.
"""


def _format_worker_results(worker_results: List[Dict]) -> str:
    """Summarise worker results for the LLM prompt."""
    lines = []
    for r in worker_results:
        mount = r.get("mount", "?")
        tool = r.get("tool", "?")
        error = r.get("error")
        result = r.get("result", "")
        if error:
            lines.append(f"[{mount}/{tool}] ERROR: {error}")
        else:
            # Truncate long results
            snippet = str(result)[:800]
            if len(str(result)) > 800:
                snippet += "... [truncated]"
            lines.append(f"[{mount}/{tool}]\n{snippet}")
    return "\n\n".join(lines) if lines else "(no worker results)"


def _format_ranked_causes(ranked_causes: List[Any]) -> str:
    """Summarise trace-based root cause ranking for the LLM prompt."""
    lines = []
    for c in (ranked_causes or [])[:8]:
        svc = getattr(c, "service", None) or (c.get("service") if isinstance(c, dict) else "?")
        conf = getattr(c, "confidence", 0.0) or (c.get("confidence", 0.0) if isinstance(c, dict) else 0.0)
        rank = getattr(c, "rank", 0) or (c.get("rank", 0) if isinstance(c, dict) else 0)
        lines.append(f"  #{rank}: {svc} (confidence={conf:.3f})")
    return "\n".join(lines) if lines else "  (none)"


def synthesis_agent_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: merge live MCP evidence with pipeline results.

    Reads:  ranked_causes, worker_results, investigation_strategy
    Writes: ranked_causes (updated), live_evidence
    """
    t0 = time.time()
    ranked_causes = list(state.get("ranked_causes", []) or [])
    worker_results = list(state.get("worker_results", []) or [])
    strategy = state.get("investigation_strategy", {}) or {}

    logger.info(
        f"[SynthesisAgent] Merging {len(worker_results)} worker results "
        f"with {len(ranked_causes)} ranked causes"
    )

    live_evidence: Dict[str, Any] = {
        "worker_count": len(worker_results),
        "succeeded": sum(1 for r in worker_results if r.get("result") is not None),
        "live_summary": "",
        "override_root_cause": None,
        "adjustments": [],
    }

    # If no worker results or no ranked causes, nothing to merge
    if not worker_results or not ranked_causes:
        logger.info("[SynthesisAgent] Nothing to merge — skipping LLM call")
        elapsed = round(time.time() - t0, 2)
        return {
            "live_evidence": live_evidence,
            "node_timings": {**state.get("node_timings", {}), "synthesis_agent": elapsed},
            "messages": state.get("messages", []) + ["[Synthesis] No live data to merge"],
        }

    # Build prompt
    worker_text = _format_worker_results(worker_results)
    trace_text = _format_ranked_causes(ranked_causes)
    fault_hypothesis = strategy.get("fault_hypothesis", "unknown")

    prompt = (
        f"Fault hypothesis: {fault_hypothesis}\n\n"
        f"Trace-based root cause ranking:\n{trace_text}\n\n"
        f"Live cluster evidence:\n{worker_text}\n\n"
        "Synthesize the evidence. Adjust confidence scores if live data supports "
        "or contradicts the trace-based ranking."
    )

    try:
        from GraphRCA_agent.llm import llm_reason

        raw = llm_reason(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=1000,
            caller="synthesis_agent",
        )

        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        synthesis = json.loads(text)

        live_evidence["live_summary"] = synthesis.get("live_summary", "")
        live_evidence["override_root_cause"] = synthesis.get("override_root_cause")
        live_evidence["adjustments"] = synthesis.get("updated_ranking", [])

        # Apply confidence deltas to ranked_causes (deep copy to avoid mutation)
        adjustment_map: Dict[str, float] = {}
        for adj in synthesis.get("updated_ranking", []):
            svc = adj.get("service", "")
            delta = float(adj.get("confidence_delta", 0.0))
            if svc:
                adjustment_map[svc] = delta

        updated_causes = []
        for c in ranked_causes:
            svc = getattr(c, "service", None) or (c.get("service") if isinstance(c, dict) else None)
            if svc and svc in adjustment_map:
                delta = adjustment_map[svc]
                try:
                    # Dataclass or object
                    c_copy = deepcopy(c)
                    old_conf = float(getattr(c_copy, "confidence", 0.0))
                    new_conf = round(max(0.0, min(1.0, old_conf + delta)), 4)
                    c_copy.confidence = new_conf
                    evidence = list(getattr(c_copy, "evidence", []))
                    evidence.append(f"Synthesis delta={delta:+.3f} (live evidence)")
                    c_copy.evidence = evidence
                    updated_causes.append(c_copy)
                    logger.info(
                        f"[SynthesisAgent] {svc}: confidence {old_conf:.3f} → {new_conf:.3f} "
                        f"(delta={delta:+.3f})"
                    )
                except Exception:
                    updated_causes.append(c)
            else:
                updated_causes.append(c)

        # Handle override_root_cause: move overridden service to rank #1
        override = synthesis.get("override_root_cause")
        if override:
            override_candidates = [
                c for c in updated_causes
                if (getattr(c, "service", None) or "").lower() == override.lower()
            ]
            others = [
                c for c in updated_causes
                if (getattr(c, "service", None) or "").lower() != override.lower()
            ]
            if override_candidates:
                updated_causes = override_candidates + others
                logger.info(f"[SynthesisAgent] Override root cause → {override}")

        # Re-sort by confidence
        updated_causes.sort(
            key=lambda c: float(getattr(c, "confidence", 0.0) if hasattr(c, "confidence")
                                else c.get("confidence", 0.0) if isinstance(c, dict) else 0.0),
            reverse=True,
        )

        # Re-assign ranks
        for i, c in enumerate(updated_causes):
            try:
                c.rank = i + 1
            except Exception:
                pass

        ranked_causes = updated_causes

    except Exception as e:
        logger.warning(f"[SynthesisAgent] LLM synthesis failed ({e}) — keeping original ranking")

    elapsed = round(time.time() - t0, 2)
    logger.info(f"[SynthesisAgent] Done in {elapsed}s")

    return {
        "ranked_causes": ranked_causes,
        "live_evidence": live_evidence,
        "node_timings": {**state.get("node_timings", {}), "synthesis_agent": elapsed},
        "messages": state.get("messages", []) + [
            f"[Synthesis] Live evidence merged — "
            f"summary: {live_evidence.get('live_summary', '')[:100]}"
        ],
    }
