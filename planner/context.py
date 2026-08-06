"""AgentContext — Thin wrapper binding ScratchPad to the Variable-Driven Router.

Holds all per-episode state that the router needs between get_action() calls:
  - Which environment responses have arrived and what to do with them.
  - Where we are in the metrics/traces two-step (get-dir → read-file).
  - References to both the shared and per-resolver private ScratchPad clients.

Design principle (§2 of v5 spec):
  Every field here represents state that *cannot* live in ScratchPad —
  it's routing bookkeeping, not incident facts. Incident facts always go
  to ScratchPad via T_commit().

v5.1 — Router / Archivist structured read:
  Deterministic consumers (router, Archivist submit builder, Executor pre-flight)
  call get_resolved_variables_structured() → /v1/session/{id}/variables/raw.
  Only SLM agents call sync() → the markdown view built for context-limited
  readers. The two paths are now fully separated.
"""

import asyncio
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SCRATCHPAD_URL = os.getenv("SCRATCHPAD_URL", "http://localhost:8000")

# ── Task variable map ────────────────────────────────────────────────────────
# Ordered by resolution dependency. The router iterates this order.
TASK_REQUIRED_VARS: dict[str, list[str]] = {
    "detection": [
        "ANOMALY_CONFIRMED",
    ],
    "localization": [
        "ANOMALY_CONFIRMED",
        "ROOT_CAUSE_SERVICE",
    ],
    "analysis": [
        "ANOMALY_CONFIRMED",
        "ROOT_CAUSE_SERVICE",
        "FAULT_TYPE",
        "FAULT_LAYER",
    ],
    "mitigation": [
        "ANOMALY_CONFIRMED",
        "ROOT_CAUSE_SERVICE",
        "FAULT_TYPE",
        "FAULT_LAYER",
        "MITIGATION_ACTION",
        "MITIGATION_EXECUTED",
        "TNR_VERIFIED",
    ],
}


class AgentContext:
    """Per-episode execution context for GraphRCAAgentV5.

    One instance lives for the lifetime of a single AIOpsLab episode.
    get_action() reads and mutates this object on every call.

    ScratchPad client lifecycle:
      - shared_client: bound to the shared incident session (session_id).
        All confirmed facts, resolved variables, and the Unresolved Matrix
        that the router reads all live here.
      - Private clients are created on-demand inside resolver closures
        (e.g. _resolve_fault_type uses private_RCA_{session_id}). They
        are NOT stored here — each resolver manages its own private client.
    """

    def __init__(
        self,
        session_id: str,
        task_type: str,
        namespace: str,
    ) -> None:
        self.session_id = session_id
        self.task_type = task_type
        self.namespace = namespace

        # Lazy import to allow sys.path injection before this module is loaded.
        from agent_sdk import ScratchpadAgentClient  # type: ignore[import]

        self.shared_client = ScratchpadAgentClient(
            agent_id="GRAPHRCA_V5",
            session_id=session_id,
            base_url=SCRATCHPAD_URL,
        )

        # ── Environment response routing state ───────────────────────────
        # Tracks the two-step get→read pattern for metrics and traces.
        self.metrics_dir_path: Optional[str] = None   # None = not yet fetched
        self.metrics_consumed: bool = False            # True = read_metrics done
        self.traces_dir_path: Optional[str] = None
        self.traces_consumed: bool = False

        # Last raw env response text (used by resolvers that parse it inline).
        self.last_env_response: str = ""

        # The action we sent in the previous turn (needed to route the response).
        self._pending_action: Optional[str] = None

        # Health score captured before mitigation (for TNR comparison).
        self.health_score_before: float = 0.0
        # Post-mitigation health CSV text (set after the TNR get+read cycle).
        self.health_csv_text: Optional[str] = None
        # Number of rollback cycles executed this episode.
        self.rollback_count: int = 0

        logger.info(
            f"[AgentContext] Init | session={session_id} "
            f"task={task_type} namespace={namespace}"
        )

    # ── ScratchPad I/O ───────────────────────────────────────────────────────

    def sync(self, max_tokens: int = 6000) -> str:
        """Fetch the current shared markdown view (SLM path, blocking).

        This rendered view is for SLM agents only — token-bounded prose.
        Deterministic consumers (router, Archivist, Executor) must call
        get_resolved_variables_structured() instead.
        """
        from graphrca.scratchpad_io import T_sync  # type: ignore[import]

        return asyncio.run(T_sync(self.shared_client, max_tokens=max_tokens))

    def commit(
        self,
        raw_chunk: str,
        triplets: list[dict],
        mutations: dict[str, str],
    ) -> dict:
        """Write facts + mutations to the shared ScratchPad session (blocking)."""
        from graphrca.scratchpad_io import T_commit  # type: ignore[import]

        return asyncio.run(
            T_commit(self.shared_client, raw_chunk, triplets, mutations)
        )

    # ── Variable resolution helpers ──────────────────────────────────────────

    def get_unresolved_variables(self) -> set[str]:
        """Return the set of required-but-unresolved variables for this task type.

        v5.1: Reads structured JSON from /v1/session/{id}/variables/raw instead
        of parsing the SLM-rendered markdown view. RESOLVED rows are deleted
        server-side, so absence from the JSON == resolved — same semantics,
        no string parsing required.
        """
        raw = get_resolved_variables_structured(self.session_id)
        required = set(TASK_REQUIRED_VARS.get(self.task_type, []))
        # raw contains only *unresolved* rows (RESOLVED deletes the row server-side)
        return required & set(raw.keys())

    def all_required_vars_resolved(self) -> bool:
        """True when the Unresolved Variables Matrix has no required entries left."""
        return len(self.get_unresolved_variables()) == 0

    def get_resolved_value(self, var_name: str) -> Optional[str]:
        """Best-effort extraction of a resolved variable's entity value."""
        from graphrca.scratchpad_io import extract_resolved_value  # type: ignore[import]

        view = self.sync()
        return extract_resolved_value(view, var_name)

    # ── Environment response routing ─────────────────────────────────────────

    def has_pending_observation(self, input_text: str) -> bool:
        """True when we are receiving the env response for a prior action.

        On the very first get_action() call (orchestrator sends the initial
        prompt), _pending_action is None so we skip commit_environment_response.
        """
        return self._pending_action is not None and bool(input_text.strip())

    def commit_environment_response(self, env_text: str) -> None:
        """Route the raw AIOpsLab response to the correct state field.

        The two-step pattern for metrics/traces:
          Step A: get_metrics()  → env_text = directory path
          Step B: read_metrics() → env_text = pandas.to_string() CSV block

        For exec_shell and get_logs, the response is stored in last_env_response
        for resolvers to consume.
        """
        env_text = env_text.strip()
        self.last_env_response = env_text
        pending = self._pending_action or ""

        if pending.startswith("get_metrics"):
            # Response is the directory path; next call will be read_metrics.
            self.metrics_dir_path = env_text
            logger.debug(f"[AgentContext] metrics_dir_path = {env_text[:100]!r}")

        elif pending.startswith("read_metrics"):
            # Response is the fixed-width pandas.to_string() block.
            if self.health_csv_text is None and self.metrics_consumed:
                # Second metrics read: this is the post-mitigation health check.
                self.health_csv_text = env_text
                logger.debug("[AgentContext] Post-mitigation health CSV received.")
            else:
                self.metrics_consumed = True
                logger.debug("[AgentContext] Pre-analysis metrics CSV received.")

        elif pending.startswith("get_traces"):
            self.traces_dir_path = env_text
            logger.debug(f"[AgentContext] traces_dir_path = {env_text[:100]!r}")

        elif pending.startswith("read_traces"):
            self.traces_consumed = True
            logger.debug("[AgentContext] Traces CSV received.")

        elif pending.startswith("exec_shell"):
            logger.debug(f"[AgentContext] exec_shell response: {env_text[:200]!r}")

        elif pending.startswith("get_logs"):
            logger.debug(f"[AgentContext] get_logs response ({len(env_text)} chars).")

        self._pending_action = None

    def record_pending_action(self, action_str: str) -> None:
        """Record the action we are about to return to the orchestrator."""
        self._pending_action = action_str
        logger.debug(f"[AgentContext] pending_action = {action_str[:80]!r}")

    # ── Fresh-data predicates (used by _need_* fns in router) ───────────────

    def has_fresh_metrics(self) -> bool:
        """True when we have already received and consumed a metrics CSV."""
        return self.metrics_consumed

    def has_fresh_traces(self) -> bool:
        """True when we have already received and consumed a traces CSV."""
        return self.traces_consumed

    # ── Submit call builder ──────────────────────────────────────────────────

    def build_submit_call(self, best_effort: bool = False) -> str:
        """Build the exact submit() call for the current task type.

        Per §1 of v5 spec:
          detection:    submit(has_anomaly: str)  →  "Yes" | "No"
          localization: submit(faulty_components: list[str])
          analysis:     submit(analysis: dict[str, str])
          mitigation:   submit()  — no arguments

        v5.2 (Archivist fix): Uses get_resolved_variables_structured() —
        structured JSON read, not markdown parsing. Archivist is pure
        template-fill, zero SLM involvement, so it reads the same structured
        endpoint as the router (v5.1 fix).

        Args:
            best_effort: If True, fill missing values with best-guess defaults
                         rather than raising an error. Used when the internal
                         budget is exhausted before full resolution.

        Returns:
            A submit(...) call string ready to be wrapped in a fenced block.
        """
        # Structured read — Archivist is deterministic template-fill, no SLM.
        # get_resolved_variables_structured() returns the *unresolved* rows;
        # resolved variables are absent (deleted server-side). To read their
        # committed values we still need the KG view — only for the submit
        # builder we fall back to the markdown view for value extraction.
        from graphrca.scratchpad_io import extract_resolved_value  # type: ignore[import]

        view = self.sync()

        if self.task_type == "detection":
            # ANOMALY_CONFIRMED resolved → row absent from structured endpoint.
            # Actual Yes/No value lives in the KG section of the markdown view.
            anomaly_val = extract_resolved_value(view, "ANOMALY_CONFIRMED")
            has_anomaly = "Yes"
            if anomaly_val and anomaly_val.upper() in ("NO", "FALSE", "NONE", "0"):
                has_anomaly = "No"
            return f'submit(has_anomaly="{has_anomaly}")'

        if self.task_type == "localization":
            svc = extract_resolved_value(view, "ROOT_CAUSE_SERVICE") or "unknown"
            # May be a comma-separated list if multiple suspects.
            svcs = [s.strip() for s in svc.split(",") if s.strip()]
            if not svcs:
                svcs = ["unknown"]
            return f"submit(faulty_components={svcs!r})"

        if self.task_type == "analysis":
            fault_layer = extract_resolved_value(view, "FAULT_LAYER") or "Unknown"
            fault_type = extract_resolved_value(view, "FAULT_TYPE") or "Unknown"
            return (
                f'submit(analysis={{"system_level": "{fault_layer}", '
                f'"fault_type": "{fault_type}"}})'
            )

        if self.task_type == "mitigation":
            # No arguments — cluster state is the answer.
            return "submit()"

        # Fallback for unknown task types.
        logger.warning(f"[AgentContext] Unknown task_type={self.task_type!r}, submitting empty.")
        return "submit()"

    async def close(self) -> None:
        """Close the shared ScratchPad client."""
        try:
            await self.shared_client.close()
        except Exception:
            pass


# ── Module-level structured read helper ─────────────────────────────────────
# Shared by: AgentContext.get_unresolved_variables() (router),
#            AgentContext.build_submit_call()       (Archivist/submit),
#            Executor pre-flight checks             (§7 of v5 spec).
#
# v5.1 rule: deterministic-Python consumers read structured JSON directly.
#            Only SLM agents call sync() for the rendered markdown view.


def get_resolved_variables_structured(session_id: str) -> dict:
    """Read the current unresolved_variables rows as structured JSON.

    Calls GET /v1/session/{session_id}/variables/raw — a read-only endpoint
    that returns literal DB rows, NOT rendered markdown. Intended for
    deterministic consumers (router, Archivist, Executor) that need current
    structured state, not a token-bounded prose view built for a
    context-limited reader.

    RESOLVED rows are deleted server-side, so a resolved variable is simply
    *absent* from the returned dict — same semantics as the markdown absence
    check, but without string parsing.

    Returns:
        dict mapping variable_name -> status for all *unresolved* rows.
        Empty dict on network error (caller should treat all required vars
        as unresolved and fall back to sync() if needed).
    """
    try:
        resp = httpx.get(
            f"{SCRATCHPAD_URL}/v1/session/{session_id}/variables/raw",
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json().get("unresolved", {})
    except httpx.HTTPStatusError as e:
        logger.warning(
            f"[Context] /variables/raw HTTP {e.response.status_code} for "
            f"session={session_id!r} — falling back to empty dict."
        )
        return {}
    except Exception as e:
        logger.warning(
            f"[Context] /variables/raw unavailable for session={session_id!r}: {e} "
            f"— falling back to empty dict."
        )
        return {}
