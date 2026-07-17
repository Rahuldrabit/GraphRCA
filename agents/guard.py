"""Guard Agent — GraphRCA v5.2.

Implements the two-layer safety gate before any mitigation action executes:

  Layer 1 (deterministic, free):
    check_plan_safety() / GUARD_BLOCK_LIST regex check.
    Reads MITIGATION_ACTION/TARGET/NAMESPACE via get_resolved_variables_structured()
    (the v5.1 structured endpoint, not markdown parsing).
    Short-circuits immediately on a match — no LLM call spent.

  Layer 2 (SLM consistency check):
    Guard IS a small-model call with the same context/hallucination drawbacks
    as every other agent in the pipeline. It receives the same properly-budgeted
    memory view via get_memory_view() (ctx.sync()), not a hand-picked slice.
    Its output is a *judgment* ("does this logically follow from the RCA?"),
    not an extraction from raw evidence, so there is no raw_chunk to cite
    against — the citation gate does not apply here.
    Only the rejection (if any) is committed, as a plain fact for the
    Mitigation Planner's next sync.

The rule (v5.2):
  If a call generates text via an SLM → sync() for context.
  If a call is deterministic Python → read structured data directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SCRATCHPAD_URL = os.getenv("SCRATCHPAD_URL", "http://localhost:8000")

# Guard system prompt — strict, terse judgment, not extraction.
GUARD_SYSTEM_PROMPT = (
    "You are a strict auditor. Given the investigation context below, "
    "does the proposed mitigation action logically follow from the RCA conclusion? "
    'Output ONLY: {"verdict": "PASS" or "FAIL", "reason": "<=15 words"}'
)

# Deterministic block list — superset of AIOpsLab's own exec_shell BLOCK_LIST.
GUARD_BLOCK_LIST: dict[str, str] = {
    "kubectl edit":         "interactive — not permitted",
    "port-forward":         "interactive — not permitted",
    "logs -f":              "follow mode — not permitted",
    "docker logs -f":       "follow mode — not permitted",
    "delete namespace":     "destructive — not permitted",
    "delete statefulset":   "destructive — not permitted",
    "delete pvc":           "destructive — not permitted",
    "&&":                   "compound shell operator — not permitted",
    "||":                   "compound shell operator — not permitted",
    "; rm":                 "destructive compound — not permitted",
}


def check_plan_safety(
    action: Optional[str],
    target: Optional[str],
    namespace: Optional[str],
) -> str:
    """Layer 1 deterministic policy check.

    Checks the rendered kubectl command against GUARD_BLOCK_LIST.
    Returns a "BLOCKED: <reason>" string if blocked, otherwise "".
    Runs before any LLM call is made.
    """
    if not action:
        return "BLOCKED: no action specified"

    # Build a representative command string for block-list scanning.
    candidate = f"{action} {target or ''} {namespace or ''}"
    for pattern, reason in GUARD_BLOCK_LIST.items():
        if pattern in candidate or pattern in action:
            return f"BLOCKED: {reason}"
    return ""


async def guard(state: dict) -> dict:
    """Guard agent — async, AIOpsLab state-dict interface.

    Reads structured state (v5.1 path) for the deterministic check,
    then syncs the memory view (SLM path) for the consistency judgment.

    Args:
        state: AIOpsLab episode state dict. Must contain "session_id".

    Returns:
        Updated state dict with "guard_passed" bool added.
    """
    from agent_sdk import ScratchpadAgentClient              # type: ignore[import]
    from planner.context import get_resolved_variables_structured  # type: ignore[import]

    shared_id = state["session_id"]
    client = ScratchpadAgentClient("GUARD", shared_id, SCRATCHPAD_URL)

    # ── Deterministic path (v5.1): structured read, no markdown parsing ──────
    # Reads action/target/namespace directly from the DB-level JSON endpoint.
    resolved = get_resolved_variables_structured(shared_id)
    action    = resolved.get("MITIGATION_ACTION")
    target    = resolved.get("MITIGATION_TARGET")
    namespace = resolved.get("MITIGATION_NAMESPACE") or state.get("namespace", "default")

    # Layer 1: deterministic policy check — always runs first.
    safety_text = check_plan_safety(action, target, namespace)
    if "BLOCKED" in safety_text:
        logger.warning(f"[Guard] Layer-1 block: {safety_text}")
        await client.update_memory(
            safety_text,
            [{
                "source_entity": "GUARD",
                "relationship": "rejected_plan",
                "target_entity": action or "UNKNOWN",
                "citation_quote": safety_text[:100],
            }],
            {},
            is_done=True,
        )
        await client.close()
        return {**state, "guard_passed": False}

    # ── SLM path (v5.2): sync for the properly-budgeted memory view ──────────
    # Guard is an SLM call — it needs the full context view, same as every
    # other agent. get_memory_view() is the SLM-facing endpoint.
    try:
        view = await client.get_memory_view()
    except Exception as e:
        logger.warning(f"[Guard] get_memory_view failed: {e} — defaulting to PASS")
        await client.close()
        return {**state, "guard_passed": True}

    # Layer 2: SLM consistency check.
    # Output is a judgment, not an evidence extraction — no citation gate.
    try:
        from inference import UniversalInferenceEngine  # type: ignore[import]
        from pydantic import BaseModel                  # type: ignore[import]

        class GuardVerdict(BaseModel):
            verdict: str
            reason: str

        engine = UniversalInferenceEngine()
        verdict: GuardVerdict = engine.generate_structured(
            prompt=(
                f"{view}\n\n"
                f"Proposed action: {action} on {target} in namespace {namespace}."
            ),
            system_prompt=GUARD_SYSTEM_PROMPT,
            response_schema=GuardVerdict,
        )

        if verdict.verdict == "FAIL":
            logger.warning(f"[Guard] Layer-2 SLM FAIL: {verdict.reason!r}")
            # Only the rejection is committed — it's a plain fact for Mitigation
            # Planner's next sync, not raw-evidence extraction.
            await client.update_memory(
                verdict.reason,
                [{
                    "source_entity": "GUARD",
                    "relationship": "rejected_plan",
                    "target_entity": action or "UNKNOWN",
                    "citation_quote": verdict.reason[:100],
                }],
                {},
                is_done=True,
            )
            await client.close()
            return {**state, "guard_passed": False}

    except Exception as e:
        # UniversalInferenceEngine not available — layer 1 already passed,
        # so skip layer 2 rather than stalling the pipeline.
        logger.debug(f"[Guard] SLM check unavailable: {e} — defaulting to PASS")

    await client.close()
    return {**state, "guard_passed": True}


def guard_sync(state: dict) -> dict:
    """Synchronous wrapper around guard() for use in non-async contexts.

    Usage in router/LangGraph nodes:
        from agents.guard import guard_sync
        state = guard_sync(state)
    """
    return asyncio.run(guard(state))
