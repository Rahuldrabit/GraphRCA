"""Mitigation Node — LangGraph agent node.

Generates prioritised mitigation actions from ranked RCA causes.
Pushes every state-changing command onto the UndoStack for TNR safety.
"""

import logging
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState
from GraphRCA_agent.tools.safety_tools import UndoStack, build_undo_entry

# Use GraphRCA's own pipeline tools
from GraphRCA_agent.tools.pipeline.mitigation_tools import (
    generate_mitigation_plan,
    request_human_approval,
)

logger = logging.getLogger(__name__)


def mitigation_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: generate and approve mitigation actions.

    Reads:  ranked_causes, service_stats, incident_id, similar_cases
    Writes: mitigation_actions, undo_stack
    """
    t0 = time.time()
    ranked_causes = state.get("ranked_causes", [])
    service_stats = state.get("service_stats", {})
    incident_id = state.get("incident_id", "INC-unknown")
    similar_cases = state.get("similar_cases", [])

    additional_context = state.get("additional_context", "")
    if additional_context:
        logger.warning(f"[Mitigation] RETRY CONTEXT — {additional_context}")

    logger.info(f"[Mitigation] Generating plan for {len(ranked_causes)} candidates | incident={incident_id}")

    if not ranked_causes:
        return {
            "mitigation_actions": [],
            "undo_stack": [],
            "messages": state.get("messages", []) + ["[Mitigation] No ranked causes — no actions generated"],
        }

    try:
        # Log similar case context for awareness
        if similar_cases:
            logger.info(f"[Mitigation] Context: {len(similar_cases)} similar past incidents available")
            for sc in similar_cases[:2]:
                logger.info(
                    f"  Past: {sc.get('incident_id','?')} | "
                    f"root={sc.get('root_cause','?')} | "
                    f"success={sc.get('success','?')}"
                )

        # 1. Generate mitigation plan from Stratus tools
        actions = generate_mitigation_plan(ranked_causes, service_stats)

        # 2. Request approval (auto-approves in automated mode)
        actions = request_human_approval(actions)

        # 3. Build UndoStack — every approved action gets a revert
        undo_stack = UndoStack()
        existing_undo = state.get("undo_stack", [])

        for action in actions:
            if getattr(action, "approved", False) or (isinstance(action, dict) and action.get("approved", False)):
                entry = build_undo_entry(action)
                undo_stack.push(entry)

        elapsed = round(time.time() - t0, 2)

        action_summaries = []
        for a in actions[:3]:
            priority = a.priority if hasattr(a, "priority") else a.get("priority", "P?")
            title = a.title if hasattr(a, "title") else a.get("title", "")
            svc = a.service if hasattr(a, "service") else a.get("service", "")
            action_summaries.append(f"[{priority}] {title} ({svc})")

        logger.info(f"[Mitigation] {len(actions)} actions generated | undo_stack={len(undo_stack)} in {elapsed}s")

        return {
            "mitigation_actions": actions,
            "undo_stack": undo_stack.to_list(),
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[Mitigation] {len(actions)} actions | " + " | ".join(action_summaries)
            ],
            "node_timings": {**state.get("node_timings", {}), "mitigation": elapsed},
        }

    except Exception as e:
        logger.exception(f"[Mitigation] Failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "mitigation_actions": [],
            "undo_stack": [],
            "messages": state.get("messages", []) + [f"[Mitigation] ERROR: {e}"],
        }
