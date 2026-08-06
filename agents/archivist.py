"""Archivist Agent — GraphRCA v5.2.

Builds the final submit() call for any task type.

v5.2 fix: Uses get_resolved_variables_structured() — structured JSON read,
not markdown parsing. Archivist is pure template-fill, zero SLM involvement,
so it follows the same v5.1 rule as the router: deterministic-Python
consumers read structured data directly.

The one exception: resolved variable *values* live in the Knowledge Graph
section of the markdown view (the row is deleted from the unresolved matrix).
For value extraction (e.g. "what is ROOT_CAUSE_SERVICE?") we fall back to
ctx.sync() + extract_resolved_value(), which is a read-only KG lookup —
not SLM reasoning. This is still cheaper than calling sync() for every
variable check.

Shared helper `get_resolved_variables_structured()` is also used by:
  - AgentContext.get_unresolved_variables()  (router, v5.1)
  - _need_execution() Guard layer 1          (router, v5.2)
  - Executor pre-flight checks               (§7 of v5 spec)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SCRATCHPAD_URL = os.getenv("SCRATCHPAD_URL", "http://localhost:8000")


def build_submit_call(
    session_id: str,
    task_type: str,
    best_effort: bool = False,
) -> str:
    """Build the exact submit() call for the current task type.

    Per §1 of v5 spec:
      detection:    submit(has_anomaly: str)  →  "Yes" | "No"
      localization: submit(faulty_components: list[str])
      analysis:     submit(analysis: dict[str, str])
      mitigation:   submit()  — no arguments

    v5.2 fix: Reads resolved variable values via extract_resolved_value()
    on the markdown view (KG section lookup, not unresolved matrix parsing).
    The unresolved check is a structured read.

    Args:
        session_id: ScratchPad session ID for this episode.
        task_type:  One of "detection", "localization", "analysis", "mitigation".
        best_effort: If True, fill missing values with best-guess defaults.

    Returns:
        A submit(...) call string ready to be wrapped in a fenced block.
    """
    from planner.context import get_resolved_variables_structured  # type: ignore[import]

    # Structured read — which variables are still unresolved? (v5.1 path)
    # Not used directly here, but available for callers who want to validate
    # completeness before calling build_submit_call().
    _ = get_resolved_variables_structured(session_id)

    # For actual *values* (the entity committed to the KG after resolution),
    # we read the markdown view — extract_resolved_value() is a KG row scan,
    # not SLM reasoning.
    import httpx  # type: ignore[import]
    from graphrca.scratchpad_io import extract_resolved_value  # type: ignore[import]

    try:
        resp = httpx.get(
            f"{SCRATCHPAD_URL}/v1/session/{session_id}/memory",
            timeout=10.0,
        )
        resp.raise_for_status()
        view = resp.text
    except Exception as e:
        logger.warning(f"[Archivist] Could not fetch memory view: {e}")
        view = ""

    if task_type == "detection":
        anomaly_val = extract_resolved_value(view, "ANOMALY_CONFIRMED")
        has_anomaly = "Yes"
        if anomaly_val and anomaly_val.upper() in ("NO", "FALSE", "NONE", "0"):
            has_anomaly = "No"
        return f'submit(has_anomaly="{has_anomaly}")'

    if task_type == "localization":
        svc = extract_resolved_value(view, "ROOT_CAUSE_SERVICE") or "unknown"
        top3_raw = extract_resolved_value(view, "TOP_3_SERVICES")
        if top3_raw:
            svcs = [s.strip() for s in top3_raw.split(",") if s.strip()]
        else:
            svcs = [s.strip() for s in svc.split(",") if s.strip()]
        if not svcs:
            svcs = [svc or "unknown"]
        return f"submit(faulty_components={svcs!r})"

    if task_type == "analysis":
        fault_layer = extract_resolved_value(view, "FAULT_LAYER") or "Unknown"
        fault_type  = extract_resolved_value(view, "FAULT_TYPE")  or "Unknown"
        return (
            f'submit(analysis={{"system_level": "{fault_layer}", '
            f'"fault_type": "{fault_type}"}})'
        )

    if task_type == "mitigation":
        # No arguments — cluster state is the answer.
        return "submit()"

    logger.warning(f"[Archivist] Unknown task_type={task_type!r}, submitting empty.")
    return "submit()"


def get_resolved_variables_structured(session_id: str) -> dict:
    """Re-export for callers that import from this module.

    Delegates to planner.context.get_resolved_variables_structured() —
    the single shared implementation for all deterministic consumers.
    """
    from planner.context import get_resolved_variables_structured as _impl  # type: ignore[import]
    return _impl(session_id)
