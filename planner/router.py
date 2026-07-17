"""Variable-Driven Router — GraphRCA v5 Planner.

The core insight (§2 of v5 spec):
  ScratchPad-internal reasoning is FREE and UNBOUNDED.
  AIOpsLab environment calls (get_metrics, get_logs, exec_shell...) each
  cost one irreversible step from the episode budget.

  → Resolve everything derivable from data already in the graph before
    asking the environment for more.

Design:
  plan_next_step(ctx) reads the live Unresolved Variables Matrix and
  iterates VARIABLE_ROUTING_TABLE (dependency-ordered). For each missing
  variable required by the current task type:
    - Resolver returns a STRING → environment action needed, return it.
    - Resolver returns None   → resolved internally (committed to ScratchPad),
                                loop again with fresh unresolved set.

  No LLM chooses the action. The router is a plain dict/list lookup.
  SLM extraction only happens inside the resolver closures (e.g.
  _resolve_anomaly, _resolve_fault_type) after math produces a narrative.

All existing GraphRCA math is imported here, not re-implemented:
  - anomaly_detectors.py  (EWMA + IsolationForest + PCA ensemble)
  - rca_tools.py          (backward_bfs_traversal, score_candidate, ...)
  - causal_tools.py       (classify_fault_type)
  - mitigation_tools.py   (generate_mitigation_plan)
  - safety_tools.py       (compute_health_score, health_regressed, UndoStack)

Data ingestion correction (§5 of v5 spec):
  read_metrics() and read_traces() return pandas.to_string() output —
  fixed-width text, NOT CSV. Parse with pd.read_fwf(), not pd.read_csv().
"""

from __future__ import annotations

import io
import logging
import os
import re
from collections import Counter
from typing import TYPE_CHECKING, Callable, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from planner.context import AgentContext

SCRATCHPAD_URL = os.getenv("SCRATCHPAD_URL", "http://localhost:8000")

# ── Action Templates (pre-validated, param-only rendering) ──────────────────
# The SLM NEVER writes a command string. It only picks a rank.
# These templates are rendered to exec_shell("...") strings by the router.

ACTION_TEMPLATES: dict[str, str] = {
    "rollout_restart": "kubectl rollout restart deployment/{deployment} -n {namespace}",
    "rollback":        "kubectl rollout undo deployment/{deployment} -n {namespace}",
    "scale":           "kubectl scale deployment/{deployment} -n {namespace} --replicas={n}",
    "patch_port": (
        "kubectl patch deployment/{deployment} -n {namespace} "
        "--type='json' -p='[{{\"op\":\"replace\",\"path\":\"/spec/template/spec/"
        "containers/0/ports/0/containerPort\",\"value\":{port}}}]'"
    ),
    "set_image": (
        "kubectl set image deployment/{deployment} "
        "{container}={image} -n {namespace}"
    ),
    "wait_ready": (
        "kubectl wait --for=condition=ready pod --all "
        "-n {namespace} --timeout={timeout}"
    ),
}

# ── Guard Block List ────────────────────────────────────────────────────────
# Superset of AIOpsLab's own exec_shell BLOCK_LIST. Checked before spending a step.

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


def _guard_check(cmd: str) -> Optional[str]:
    """Return block reason string if cmd matches the block list, else None."""
    for pattern, reason in GUARD_BLOCK_LIST.items():
        if pattern in cmd:
            return reason
    return None


# ── VARIABLE_ROUTING_TABLE ───────────────────────────────────────────────────
# Ordered by dependency. Each entry: (variable_name, resolver_fn).
# resolver_fn(ctx) -> str | None:
#   str  → an AIOpsLab action string (env call needed)
#   None → resolved internally, no step spent

VARIABLE_ROUTING_TABLE: list[tuple[str, Callable]] = [
    ("ANOMALY_CONFIRMED",   lambda ctx: _need_metrics(ctx) or _resolve_anomaly(ctx)),
    ("ROOT_CAUSE_SERVICE",  lambda ctx: _need_traces(ctx)  or _resolve_localization(ctx)),
    ("FAULT_TYPE",          lambda ctx: _need_logs(ctx)    or _resolve_fault_type(ctx)),
    ("FAULT_LAYER",         lambda ctx: _resolve_fault_layer(ctx)),   # same data, no new env call
    ("MITIGATION_ACTION",   lambda ctx: _resolve_mitigation_plan(ctx)),
    ("MITIGATION_EXECUTED", lambda ctx: _need_execution(ctx)),
    ("TNR_VERIFIED",        lambda ctx: _need_health_check(ctx) or _resolve_tnr(ctx)),
]


def plan_next_step(ctx: "AgentContext") -> Optional[str]:
    """Determine the next action for this turn.

    Iterates VARIABLE_ROUTING_TABLE in dependency order. For each variable
    that is both required for the current task type and currently unresolved:
      - Calls the resolver.
      - If resolver returns a string: return it immediately (env action needed).
      - If resolver returns None: it resolved the variable internally and
        committed to ScratchPad. The loop continues with the (now-updated)
        unresolved set on the next iteration.

    Returns:
        AIOpsLab action string (ready to wrap in fenced block), or
        None if everything required is resolved (caller should submit).
    """
    from planner.context import TASK_REQUIRED_VARS

    required_for_task = set(TASK_REQUIRED_VARS.get(ctx.task_type, []))
    unresolved = ctx.get_unresolved_variables()

    for var_name, resolver in VARIABLE_ROUTING_TABLE:
        if var_name not in required_for_task:
            continue
        if var_name not in unresolved:
            continue  # already resolved — skip

        logger.debug(f"[Router] Resolving {var_name!r} for task={ctx.task_type!r}")
        action = resolver(ctx)

        if action is not None:
            logger.info(f"[Router] Env action needed for {var_name!r}: {action[:80]!r}")
            return action

        # Resolver returned None → resolved internally. Refresh unresolved set.
        unresolved = ctx.get_unresolved_variables()
        logger.debug(f"[Router] {var_name!r} resolved internally. Remaining: {unresolved}")

    return None  # nothing left unresolved for this task


# ── Environment-touching resolvers (return action strings) ──────────────────


def _need_metrics(ctx: "AgentContext") -> Optional[str]:
    """Return get_metrics or read_metrics action if we still need metrics data."""
    if ctx.has_fresh_metrics():
        return None  # already have metrics CSV in last_env_response
    if ctx.metrics_dir_path is None:
        action = f'get_metrics(namespace="{ctx.namespace}", duration=10)'
        return action
    if not ctx.metrics_consumed:
        # We have the directory path — now read the actual file.
        action = f'read_metrics(file_path="{ctx.metrics_dir_path}")'
        return action
    return None


def _need_traces(ctx: "AgentContext") -> Optional[str]:
    """Return get_traces or read_traces action if we still need traces data."""
    if ctx.has_fresh_traces():
        return None
    if ctx.traces_dir_path is None:
        action = f'get_traces(namespace="{ctx.namespace}", duration=10)'
        return action
    if not ctx.traces_consumed:
        action = f'read_traces(file_path="{ctx.traces_dir_path}")'
        return action
    return None


def _need_logs(ctx: "AgentContext") -> Optional[str]:
    """Return get_logs action if we don't have fault-type log evidence yet."""
    from graphrca.scratchpad_io import is_resolved  # type: ignore[import]

    view = ctx.sync()
    # We need logs to classify the fault. Check if ROOT_CAUSE_SERVICE is resolved
    # so we know which service to get logs for.
    from graphrca.scratchpad_io import extract_resolved_value  # type: ignore[import]

    svc = extract_resolved_value(view, "ROOT_CAUSE_SERVICE")
    if not svc:
        return None  # can't fetch logs without a target service; skip for now

    # Check if we already fetched logs this episode (tracked via last_env_response
    # containing typical log format markers).
    if ctx.last_env_response and (
        "ERROR" in ctx.last_env_response
        or "WARN" in ctx.last_env_response
        or "error" in ctx.last_env_response.lower()
        or "exception" in ctx.last_env_response.lower()
    ):
        return None  # already have log data

    return f'get_logs(namespace="{ctx.namespace}", service="{svc}")'


def _need_execution(ctx: "AgentContext") -> Optional[str]:
    """Build and return the exec_shell mitigation command.

    v5.2 Guard fix — two paths, clearly separated:
      Deterministic path: reads MITIGATION_ACTION/TARGET/NAMESPACE from the
        structured /variables/raw endpoint (no markdown parsing).
        Runs GUARD_BLOCK_LIST check before spending an LLM call.
      SLM path: calls ctx.sync() for a properly-budgeted memory view, then
        runs the UniversalInferenceEngine consistency judgment.
        Its output is a judgment (not an extraction from raw evidence), so
        no citation gate applies; only the rejection (if any) is committed.

    Returns None only if we have already executed (MITIGATION_EXECUTED
    is resolved), which is checked by the router before calling this.
    """
    from planner.context import get_resolved_variables_structured  # type: ignore[import]

    # ── Deterministic path: structured read, no markdown parsing ────────────
    resolved = get_resolved_variables_structured(ctx.session_id)
    action_tag = resolved.get("MITIGATION_ACTION")
    target     = resolved.get("MITIGATION_TARGET")
    namespace  = ctx.namespace

    # If the structured endpoint is unavailable, fall back to markdown view.
    if not action_tag:
        from graphrca.scratchpad_io import extract_resolved_value  # type: ignore[import]
        view = ctx.sync()
        action_tag = extract_resolved_value(view, "MITIGATION_ACTION")
        target     = extract_resolved_value(view, "MITIGATION_TARGET")

    if not action_tag:
        logger.warning("[Router] MITIGATION_ACTION resolved but value not found.")
        return None

    # Map action tag to template
    cmd = _render_action_template(action_tag, target, namespace)
    if cmd is None:
        logger.warning(f"[Router] No template found for MITIGATION_ACTION={action_tag!r}")
        # Fallback: restart the root-cause deployment
        deployment = (target or "unknown").replace("/", "-")
        cmd = ACTION_TEMPLATES["rollout_restart"].format(
            deployment=deployment, namespace=namespace
        )

    # Layer 1: deterministic policy check — runs first, short-circuits before
    # spending an LLM call on something regex would have caught for free.
    block_reason = _guard_check(cmd)
    if block_reason:
        logger.error(f"[Router] Mitigation blocked by guard: {block_reason}. cmd={cmd!r}")
        ctx.commit(
            raw_chunk=f"Guard rejected mitigation: {block_reason}. Command: {cmd}",
            triplets=[{
                "source_entity": "GUARD",
                "relationship": "rejected_mitigation",
                "target_entity": action_tag,
                "citation_quote": f"Guard rejected mitigation: {block_reason}.",
            }],
            mutations={},
        )
        return None

    # Layer 2: SLM consistency check (v5.2 Guard fix).
    # Guard IS a small-model call with the same context/hallucination drawbacks
    # as every other agent here — it gets the same properly-budgeted view, not
    # a hand-picked slice. Its output is a judgment ("does this follow logically"),
    # not an extraction from raw evidence, so no citation gate applies.
    try:
        from inference import UniversalInferenceEngine  # type: ignore[import]
        from pydantic import BaseModel                  # type: ignore[import]

        GUARD_SYSTEM_PROMPT = (
            "You are a strict auditor. Given the investigation context below, "
            "does the proposed mitigation action logically follow from the RCA conclusion? "
            "Output ONLY: {\"verdict\": \"PASS\" or \"FAIL\", \"reason\": \"<=15 words\"}"
        )

        class GuardVerdict(BaseModel):
            verdict: str
            reason: str

        # SLM path: sync for the properly-budgeted context view.
        view = ctx.sync()
        engine = UniversalInferenceEngine()
        verdict: GuardVerdict = engine.generate_structured(
            prompt=f"{view}\n\nProposed action: {action_tag} on {target} in namespace {namespace}.",
            system_prompt=GUARD_SYSTEM_PROMPT,
            response_schema=GuardVerdict,
        )

        if verdict.verdict == "FAIL":
            logger.warning(f"[Router] Guard SLM FAIL: {verdict.reason!r}")
            ctx.commit(
                raw_chunk=verdict.reason,
                triplets=[{
                    "source_entity": "GUARD",
                    "relationship": "rejected_plan",
                    "target_entity": action_tag,
                    "citation_quote": verdict.reason[:100],
                }],
                mutations={},
            )
            return None

    except Exception as e:
        # UniversalInferenceEngine not available or call failed — log and continue.
        # Layer 1 deterministic check already passed, so proceeding is safe.
        logger.debug(f"[Router] Guard SLM check skipped: {e}")

    logger.info(f"[Router] Mitigation exec_shell: {cmd!r}")
    return f'exec_shell("{cmd}", timeout=30)'



def _need_health_check(ctx: "AgentContext") -> Optional[str]:
    """Return metrics actions for the post-mitigation TNR health check.

    Uses a separate state flag (health_csv_text) so we don't confuse
    the initial metrics read (for anomaly detection) with the TNR check.
    """
    if ctx.health_csv_text is not None:
        return None  # already have post-mitigation metrics

    # Reuse the two-step pattern but direct it toward the TNR check by
    # resetting the dir path only if both pre- and post-metric reads are done.
    if ctx.metrics_consumed and ctx.metrics_dir_path:
        # Pre-analysis metrics were already consumed. Now we need a fresh read.
        # Reset dir_path so _need_metrics will re-fetch.
        ctx.metrics_dir_path = None
        ctx.metrics_consumed = False

    return _need_metrics(ctx)


# ── Internal resolvers (return None, commit to ScratchPad) ──────────────────


def _resolve_anomaly(ctx: "AgentContext") -> None:
    """Run the anomaly detection ensemble on the freshly-read metrics CSV.

    Parses last_env_response (pandas.to_string() fixed-width) with pd.read_fwf(),
    runs EWMA + IsolationForest + PCA, then asks the SLM to extract facts.
    Commits result to the shared ScratchPad session.

    Returns None always (internal resolution, no env step spent).
    """
    import pandas as pd  # type: ignore[import]

    csv_text = ctx.last_env_response
    if not csv_text:
        logger.warning("[Router] _resolve_anomaly: no metrics CSV in last_env_response.")
        return None

    # Parse fixed-width pandas.to_string() output (NOT csv)
    try:
        df = pd.read_fwf(io.StringIO(csv_text))
    except Exception as e:
        logger.warning(f"[Router] _resolve_anomaly: pd.read_fwf failed: {e}")
        _commit_anomaly_unresolvable(ctx, reason="metrics parse error")
        return None

    if df.empty:
        _commit_anomaly_unresolvable(ctx, reason="empty metrics dataframe")
        return None

    try:
        scores = _run_ensemble(df)
    except Exception as e:
        logger.warning(f"[Router] Ensemble failed: {e}")
        _commit_anomaly_unresolvable(ctx, reason=str(e))
        return None

    narrative = _format_anomaly_narrative(scores)
    _slm_extract_and_commit(ctx, narrative, "DETECTION")
    return None


def _commit_anomaly_unresolvable(ctx: "AgentContext", reason: str) -> None:
    """Commit a 'no anomaly' fact when detection data is unavailable."""
    raw = f"Anomaly detection inconclusive: {reason}."
    ctx.commit(
        raw_chunk=raw,
        triplets=[{
            "source_entity": "DETECTION_ENGINE",
            "relationship": "reports_status",
            "target_entity": "NO_ANOMALY_DETECTED",
            "citation_quote": raw[:100],
        }],
        mutations={"ANOMALY_CONFIRMED": "RESOLVED"},
    )


def _resolve_localization(ctx: "AgentContext") -> None:
    """Run PageRank + backward BFS on the traces to identify root-cause service.

    Parses last_env_response (fixed-width traces CSV) with pd.read_fwf(),
    builds a NetworkX call-graph, runs personalized PageRank, backward BFS,
    and scores candidates using the existing rca_tools logic.

    Returns None always (internal resolution).
    """
    import networkx as nx  # type: ignore[import]
    import pandas as pd    # type: ignore[import]

    csv_text = ctx.last_env_response
    if not csv_text:
        logger.warning("[Router] _resolve_localization: no traces CSV.")
        return None

    try:
        df = pd.read_fwf(io.StringIO(csv_text))
    except Exception as e:
        logger.warning(f"[Router] _resolve_localization: read_fwf failed: {e}")
        return None

    # Build NetworkX call graph from span parent/child edges
    G = _build_call_graph(df)
    if not G or G.number_of_nodes() == 0:
        logger.warning("[Router] _resolve_localization: empty call graph from traces.")
        return None

    # Personalized PageRank — seed on error nodes
    error_services = _extract_error_services(df)
    personalization = {n: (1.0 if n in error_services else 0.0) for n in G.nodes()}
    if all(v == 0.0 for v in personalization.values()):
        personalization = None  # uniform

    try:
        pr = nx.pagerank(G, alpha=0.85, personalization=personalization)
    except Exception as e:
        logger.warning(f"[Router] PageRank failed: {e}")
        pr = {}

    # Backward BFS from highest-PR error node
    from GraphRCA_agent.tools.pipeline.rca_tools import (  # type: ignore[import]
        backward_bfs_traversal,
        score_candidate,
        rank_root_causes,
    )

    if not error_services and pr:
        start_svc = max(pr, key=pr.get)
    elif error_services:
        start_svc = max(error_services, key=lambda s: pr.get(s, 0))
    else:
        logger.warning("[Router] No error service found for BFS.")
        return None

    spans = _df_to_spans(df)
    paths = backward_bfs_traversal(G, start_svc, max_depth=5)
    visited = {s for path in paths for s in path} | {start_svc}

    candidates = []
    for svc in visited:
        depth = next(
            (path.index(svc) for path in paths if svc in path), 0
        )
        c = score_candidate(
            service=svc,
            G=G,
            spans=spans,
            alerts=[],
            baselines={},
            traversal_depth=depth,
        )
        candidates.append(c)

    ranked = rank_root_causes(candidates)
    if not ranked:
        logger.warning("[Router] RCA ranking returned empty list.")
        return None

    root_svc = ranked[0].service if hasattr(ranked[0], "service") else ranked[0].get("service", "unknown")
    top3 = [
        (c.service if hasattr(c, "service") else c.get("service", "?"))
        for c in ranked[:3]
    ]

    narrative = (
        f"Trace-graph root cause analysis results:\n"
        f"Top suspect for root cause is {root_svc}. "
        f"Top 3 suspects: {', '.join(top3)}. "
        f"Personalized PageRank seeded on error services: {sorted(error_services)}."
    )

    _slm_extract_and_commit(
        ctx, narrative, "LOCALIZATION",
        extra_mutations={"ROOT_CAUSE_SERVICE": "RESOLVED"},
    )
    return None


def _resolve_fault_type(ctx: "AgentContext") -> None:
    """Classify fault type from log text using regex + private ScratchPad session.

    Uses the v4.1 dual-memory pattern:
      1. Explore in a private session (private_RCA_{session_id}).
      2. On confirmation, promote only the confirmed FAULT_TYPE/FAULT_LAYER
         triplets to the shared session.

    Returns None always (internal resolution).
    """
    import asyncio

    log_text = ctx.last_env_response
    if not log_text:
        logger.warning("[Router] _resolve_fault_type: no log text available.")
        return None

    service = ctx.get_resolved_value("ROOT_CAUSE_SERVICE") or "unknown"
    narrative = _get_logs_narrative(log_text, service)

    private_id = f"private_RCA_{ctx.session_id}"
    from agent_sdk import ScratchpadAgentClient  # type: ignore[import]
    from graphrca.scratchpad_io import T_init_session, T_commit, T_sync, is_resolved  # type: ignore[import]

    asyncio.run(T_init_session(private_id, user_query=f"RCA exploration for {ctx.session_id}"))
    private_client = ScratchpadAgentClient(
        agent_id="RCA_RESOLVER", session_id=private_id, base_url=SCRATCHPAD_URL
    )

    # Run regex classification first (zero-cost)
    fault_type, fault_layer = _classify_fault_type_regex(log_text, narrative)

    if fault_type and fault_layer:
        # Regex was sufficient — build narrative and commit directly to shared.
        narrative += (
            f"\nFault classification (regex): fault_type={fault_type}, "
            f"fault_layer={fault_layer}."
        )
    else:
        # Fall back to SLM extraction in private session.
        asyncio.run(T_commit(
            private_client, narrative, [], {}
        ))
        private_view = asyncio.run(T_sync(private_client))
        payload = _slm_extract_raw(private_view, narrative, "RCA")
        asyncio.run(T_commit(
            private_client,
            narrative,
            payload.get("extracted_triplets", []),
            payload.get("unresolved_variables_mutations", {}),
        ))
        private_view = asyncio.run(T_sync(private_client))

        fault_type = _parse_var(private_view, "FAULT_TYPE") or "Unknown"
        fault_layer = _parse_var(private_view, "FAULT_LAYER") or "Unknown"

    try:
        asyncio.run(private_client.close())
    except Exception:
        pass

    # Promote confirmed facts to shared session.
    promotion_raw = (
        f"Fault type determined for {service}: "
        f"fault_type={fault_type}, fault_layer={fault_layer}. "
        f"Source: log analysis of {service}."
    )
    ctx.commit(
        raw_chunk=promotion_raw,
        triplets=[
            {
                "source_entity": service,
                "relationship": "has_fault_type",
                "target_entity": fault_type,
                "citation_quote": f"fault_type={fault_type}",
            },
            {
                "source_entity": service,
                "relationship": "has_fault_layer",
                "target_entity": fault_layer,
                "citation_quote": f"fault_layer={fault_layer}",
            },
        ],
        mutations={"FAULT_TYPE": "RESOLVED", "FAULT_LAYER": "RESOLVED"},
    )
    return None


def _resolve_fault_layer(ctx: "AgentContext") -> None:
    """FAULT_LAYER uses the same log evidence — no additional env call needed.

    If FAULT_TYPE was already resolved by _resolve_fault_type, FAULT_LAYER
    was also resolved in the same commit. This resolver is a safety net in
    case they need to be re-evaluated independently.
    """
    from graphrca.scratchpad_io import is_resolved  # type: ignore[import]

    view = ctx.sync()
    if is_resolved(view, "FAULT_TYPE"):
        # FAULT_LAYER should have been resolved in the same commit. If it
        # wasn't for some reason, try the regex on cached log text.
        if ctx.last_env_response:
            _, fault_layer = _classify_fault_type_regex(ctx.last_env_response, "")
            if fault_layer:
                raw = f"Fault layer re-determined from log evidence: {fault_layer}."
                ctx.commit(
                    raw_chunk=raw,
                    triplets=[{
                        "source_entity": "DETECTION_ENGINE",
                        "relationship": "has_fault_layer",
                        "target_entity": fault_layer,
                        "citation_quote": raw[:80],
                    }],
                    mutations={"FAULT_LAYER": "RESOLVED"},
                )
    return None


def _resolve_mitigation_plan(ctx: "AgentContext") -> None:
    """Pick a mitigation action using the existing generate_mitigation_plan logic.

    Uses the v4.1 dual-memory pattern for the mitigation planner's private session.
    Commits the confirmed MITIGATION_ACTION/TARGET/NAMESPACE to shared only after
    safety-checking against GUARD_BLOCK_LIST.

    Returns None always (internal resolution — the actual exec_shell is returned
    by _need_execution on the following router iteration).
    """
    import asyncio

    from graphrca.scratchpad_io import extract_resolved_value  # type: ignore[import]

    view = ctx.sync()
    fault_type = extract_resolved_value(view, "FAULT_TYPE") or "Unknown"
    root_svc = extract_resolved_value(view, "ROOT_CAUSE_SERVICE") or "unknown"

    # Build a minimal ranked_causes list for generate_mitigation_plan
    ranked_causes = [{"service": root_svc, "confidence": 0.9, "error_rate": 0.5}]

    try:
        from GraphRCA_agent.tools.pipeline.mitigation_tools import (  # type: ignore[import]
            generate_mitigation_plan,
        )

        actions = generate_mitigation_plan(ranked_causes, {})
    except Exception as e:
        logger.warning(f"[Router] generate_mitigation_plan failed: {e}")
        actions = []

    # Pick the top remediation action
    remediation = [
        a for a in actions
        if isinstance(a, dict) and a.get("category") == "remediation"
    ]
    if not remediation:
        remediation = actions[:1]

    if not remediation:
        # Fallback: restart root-cause deployment
        action_tag = "rollout_restart"
        target = root_svc
    else:
        top = remediation[0]
        action_tag = top.get("action", "rollout_restart")
        target = top.get("target", root_svc)

    # Render to kubectl command and safety-check
    cmd = _render_action_template(action_tag, target, ctx.namespace)
    if cmd and _guard_check(cmd):
        logger.warning(f"[Router] Mitigation plan blocked by guard: {cmd!r}")
        action_tag = "rollout_restart"
        target = root_svc
        cmd = ACTION_TEMPLATES["rollout_restart"].format(
            deployment=target.replace("/", "-"), namespace=ctx.namespace
        )

    promotion_raw = (
        f"Mitigation plan selected for {root_svc} "
        f"(fault_type={fault_type}): action={action_tag}, target={target}, "
        f"namespace={ctx.namespace}."
    )
    ctx.commit(
        raw_chunk=promotion_raw,
        triplets=[
            {
                "source_entity": root_svc,
                "relationship": "requires_mitigation",
                "target_entity": action_tag,
                "citation_quote": f"action={action_tag}",
            },
            {
                "source_entity": action_tag,
                "relationship": "targets",
                "target_entity": target,
                "citation_quote": f"target={target}",
            },
        ],
        mutations={"MITIGATION_ACTION": "RESOLVED"},
    )
    logger.info(f"[Router] Mitigation plan committed: {action_tag} → {target}")
    return None


def _resolve_tnr(ctx: "AgentContext") -> Optional[str]:
    """Compute post-mitigation health score and decide whether to rollback.

    Reads post-mitigation metrics from ctx.health_csv_text (populated by
    _need_health_check after the second get_metrics + read_metrics cycle).

    Returns:
        exec_shell(rollback_cmd) string if health regressed (costs one env step).
        None if health is OK (TNR_VERIFIED resolved internally).
    """
    import pandas as pd  # type: ignore[import]

    from GraphRCA_agent.tools.safety_tools import (  # type: ignore[import]
        compute_health_score,
        health_regressed,
    )

    health_csv = ctx.health_csv_text
    if not health_csv:
        logger.warning("[Router] _resolve_tnr: no post-mitigation metrics.")
        # Optimistically resolve — better than stalling.
        ctx.commit(
            raw_chunk="TNR check: post-mitigation metrics unavailable, assuming stable.",
            triplets=[{
                "source_entity": "TNR_ENGINE",
                "relationship": "reports_status",
                "target_entity": "HEALTH_CHECK_SKIPPED",
                "citation_quote": "TNR check: post-mitigation metrics unavailable, assuming stable.",
            }],
            mutations={"TNR_VERIFIED": "RESOLVED"},
        )
        return None

    try:
        df = pd.read_fwf(io.StringIO(health_csv))
        # Approximate alerts from error-rate columns
        error_cols = [c for c in df.columns if "error" in c.lower() or "5xx" in c.lower()]
        alert_count = 0
        for col in error_cols:
            try:
                alert_count += int(df[col].fillna(0).astype(float).mean() > 0.01)
            except Exception:
                pass
        health_after = compute_health_score(
            alerts=[None] * alert_count,
            sla_violations=[],
            unhealthy_nodes=[],
        )
    except Exception as e:
        logger.warning(f"[Router] TNR health parse failed: {e}")
        health_after = ctx.health_score_before  # assume no regression on parse error

    regressed = health_regressed(ctx.health_score_before, health_after, tolerance=0.05)

    if not regressed:
        ctx.commit(
            raw_chunk=(
                f"TNR check passed: μ(s) before={ctx.health_score_before:.4f} "
                f"after={health_after:.4f} — within tolerance."
            ),
            triplets=[{
                "source_entity": "TNR_ENGINE",
                "relationship": "reports_status",
                "target_entity": "HEALTH_STABLE",
                "citation_quote": f"TNR check passed: μ(s) before={ctx.health_score_before:.4f}",
            }],
            mutations={"TNR_VERIFIED": "RESOLVED"},
        )
        return None

    # Health regressed → rollback
    ctx.rollback_count += 1
    logger.warning(
        f"[Router] TNR FAIL: μ(s) {ctx.health_score_before:.4f} → {health_after:.4f}. "
        f"Rollback #{ctx.rollback_count}."
    )

    if ctx.rollback_count > 3:
        logger.error("[Router] Max rollback cycles exceeded. Resolving TNR_VERIFIED anyway.")
        ctx.commit(
            raw_chunk="TNR max rollbacks exceeded. Manual intervention required.",
            triplets=[{
                "source_entity": "TNR_ENGINE",
                "relationship": "reports_status",
                "target_entity": "MAX_ROLLBACKS_EXCEEDED",
                "citation_quote": "TNR max rollbacks exceeded.",
            }],
            mutations={"TNR_VERIFIED": "RESOLVED"},
        )
        return None

    # Build rollback command from undo pattern
    from graphrca.scratchpad_io import extract_resolved_value  # type: ignore[import]

    view = ctx.sync()
    target = extract_resolved_value(view, "MITIGATION_TARGET") or "unknown"
    undo_cmd = ACTION_TEMPLATES["rollback"].format(
        deployment=target.replace("/", "-"), namespace=ctx.namespace
    )

    # Re-seed mitigation variables for retry
    ctx.commit(
        raw_chunk=f"Rollback initiated due to health regression. Undoing {target}.",
        triplets=[{
            "source_entity": "TNR_ENGINE",
            "relationship": "triggers_rollback",
            "target_entity": target,
            "citation_quote": f"Rollback initiated due to health regression. Undoing {target}.",
        }],
        mutations={
            "MITIGATION_ACTION": "MISSING",
            "MITIGATION_EXECUTED": "MISSING",
        },
    )
    # Reset health check state for the retry cycle
    ctx.health_csv_text = None
    ctx.metrics_dir_path = None
    ctx.metrics_consumed = False

    return f'exec_shell("{undo_cmd}", timeout=30)'


# ── Data helpers ─────────────────────────────────────────────────────────────


def _run_ensemble(df) -> list[dict]:
    """Run EWMA + IsolationForest + PCA on a metrics DataFrame.

    Delegates to the existing anomaly_detectors module where possible.
    Falls back to a simple error-rate scan if detector import fails.
    """
    try:
        from GraphRCA_agent.tools.pipeline.anomaly_detectors import get_detector  # type: ignore[import]

        spans = _df_to_spans(df)
        service_stats = _compute_service_stats(spans)
        detector = get_detector(mode="auto", span_count=len(spans))
        alerts, _ = detector.detect(spans=spans, service_stats=service_stats)
        return [
            {
                "svc": a.service if hasattr(a, "service") else a.get("service", "?"),
                "score": a.score if hasattr(a, "score") else a.get("score", 0.0),
                "trig": a.anomaly_type if hasattr(a, "anomaly_type") else "unknown",
                "err": a.error_rate if hasattr(a, "error_rate") else 0.0,
            }
            for a in alerts
        ]
    except Exception as e:
        logger.debug(f"[Router] Detector import failed ({e}), fallback to error-rate scan.")
        return _error_rate_scan(df)


def _error_rate_scan(df) -> list[dict]:
    """Fallback: scan for columns with high error rates."""
    results = []
    for col in df.columns:
        if col.lower() in ("timestamp", "time", "index"):
            continue
        try:
            vals = df[col].dropna().astype(float)
            mean = float(vals.mean())
            if mean > 0.01:
                results.append({"svc": col, "score": min(mean, 1.0), "trig": "error_rate", "err": mean})
        except Exception:
            pass
    return sorted(results, key=lambda r: r["score"], reverse=True)


def _format_anomaly_narrative(scores: list[dict]) -> str:
    """Convert ensemble scores to prose for SLM citation extraction."""
    if not scores:
        return "Anomaly detection ensemble: no anomalies detected. All error rates below threshold."

    lines = ["Anomaly detection ensemble results:"]
    for r in scores[:5]:
        lines.append(
            f"{r['svc']} anomaly_score={r['score']:.2f} "
            f"triggered_by={r['trig']} error_rate={r['err']:.3f}."
        )
    if scores[0]["score"] > 0.5:
        lines.append(
            f"Active anomaly confirmed. Primary suspect is {scores[0]['svc']}."
        )
    else:
        lines.append("No active anomaly detected. All scores below 0.5.")
    return "\n".join(lines)


def _build_call_graph(df):
    """Build a NetworkX DiGraph from a traces DataFrame."""
    import networkx as nx  # type: ignore[import]

    G = nx.DiGraph()
    parent_col = next(
        (c for c in df.columns if "parent" in c.lower() or "caller" in c.lower()), None
    )
    child_col = next(
        (c for c in df.columns if "service" in c.lower() or "child" in c.lower()), None
    )
    if parent_col and child_col:
        for _, row in df.iterrows():
            try:
                p = str(row[parent_col]).strip()
                c = str(row[child_col]).strip()
                if p and c and p != "nan" and c != "nan" and p != c:
                    G.add_edge(p, c)
            except Exception:
                pass
    return G


def _extract_error_services(df) -> set[str]:
    """Return service names with errors from traces DataFrame."""
    error_col = next(
        (c for c in df.columns if "error" in c.lower() or "status" in c.lower()), None
    )
    svc_col = next(
        (c for c in df.columns if "service" in c.lower()), None
    )
    if not (error_col and svc_col):
        return set()
    errors = set()
    for _, row in df.iterrows():
        try:
            if str(row[error_col]).lower() in ("true", "1", "error", "5"):
                errors.add(str(row[svc_col]).strip())
        except Exception:
            pass
    return errors


def _df_to_spans(df) -> list[dict]:
    """Convert a traces/metrics DataFrame to a list of span-like dicts."""
    return df.to_dict(orient="records")


def _compute_service_stats(spans: list[dict]) -> dict:
    """Aggregate per-service stats from span records."""
    stats: dict[str, dict] = {}
    for s in spans:
        svc = str(s.get("service_name") or s.get("service") or "unknown").strip()
        if svc not in stats:
            stats[svc] = {"error_count": 0, "total_count": 0, "latencies": []}
        stats[svc]["total_count"] += 1
        if s.get("has_error") or str(s.get("status", "")).lower() in ("error", "true", "1"):
            stats[svc]["error_count"] += 1
        try:
            stats[svc]["latencies"].append(float(s.get("duration_ms", 0)))
        except Exception:
            pass

    for svc, d in stats.items():
        total = d["total_count"] or 1
        d["error_rate"] = d["error_count"] / total
        lats = d["latencies"]
        d["mean_latency"] = sum(lats) / len(lats) if lats else 0.0
    return stats


def _get_logs_narrative(raw_log_text: str, service: str) -> str:
    """Compress and narrativize log text for SLM citation extraction.

    get_logs() output is already AIOpsLab-deduped. We apply a second
    compression layer (top-N counter) to surface dominant error patterns.
    """
    normalised = [
        _normalise_log(l) for l in raw_log_text.splitlines() if l.strip()
    ]
    if not normalised:
        return f"No log entries found for {service}."

    top = Counter(normalised).most_common(5)
    total = len(normalised)
    lines = [
        f"Error analysis for {service} ({total} lines, "
        f"already de-duplicated by AIOpsLab):"
    ]
    for msg, cnt in top:
        lines.append(
            f"Error '{msg[:70]}' occurred {cnt} times "
            f"({cnt / total:.0%})."
        )
    return "\n".join(lines)


def _normalise_log(line: str) -> str:
    """Strip timestamps and IDs to surface the canonical error pattern."""
    line = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\d.:Z+-]*", "", line)
    line = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", line, flags=re.IGNORECASE)
    line = re.sub(r"\b\d+\.\d+\.\d+\.\d+(:\d+)?\b", "<ip>", line)
    return line.strip()


def _classify_fault_type_regex(log_text: str, narrative: str) -> tuple[str, str]:
    """Pure-regex fault classification — zero-cost, no LLM.

    Returns:
        (fault_type, fault_layer) strings, or ("", "") if inconclusive.
    """
    combined = (log_text + "\n" + narrative).lower()

    patterns: list[tuple[str, str, str]] = [
        # (regex, fault_type, fault_layer)
        (r"containerport|targetport|port.*mismatch|connection refused.*port",
         "Misconfiguration", "Application"),
        (r"imagepullbackoff|errimagepull|no such image|image.*not found",
         "ImagePullFailure", "Infrastructure"),
        (r"oomkilled|out of memory|memory limit",
         "ResourceExhaustion", "Application"),
        (r"crashloopbackoff|exit code [1-9]",
         "CrashLoopFailure", "Application"),
        (r"connection refused|connection timed out|dial tcp.*refused",
         "NetworkFailure", "Network"),
        (r"certificate.*expired|x509|tls.*error",
         "CertificateFailure", "Security"),
        (r"database.*connection|sql.*error|pg.*error|mongo.*error",
         "DatabaseFailure", "DataLayer"),
        (r"timeout|deadline exceeded|context deadline",
         "TimeoutFailure", "Application"),
    ]

    for pattern, fault_type, fault_layer in patterns:
        if re.search(pattern, combined):
            return fault_type, fault_layer

    return "", ""


def _parse_var(markdown_view: str, var_name: str) -> Optional[str]:
    """Extract a variable value from the Knowledge Graph section of a markdown view."""
    from graphrca.scratchpad_io import extract_resolved_value  # type: ignore[import]

    return extract_resolved_value(markdown_view, var_name)


def _render_action_template(
    action_tag: str, target: Optional[str], namespace: str
) -> Optional[str]:
    """Render an ACTION_TEMPLATE to a concrete kubectl command string."""
    tpl = ACTION_TEMPLATES.get(action_tag)
    if tpl is None:
        return None
    deployment = (target or "unknown").replace("/", "-")
    try:
        return tpl.format(
            deployment=deployment,
            namespace=namespace,
            n=1,          # default scale target
            port=8080,    # default port patch target
            container=deployment,
            image=f"{deployment}:latest",
            timeout="180s",
        )
    except KeyError:
        return tpl  # return partially-formatted if keys missing


# ── SLM extraction helpers ───────────────────────────────────────────────────


def _slm_extract_and_commit(
    ctx: "AgentContext",
    narrative: str,
    phase: str,
    extra_mutations: Optional[dict] = None,
) -> None:
    """Run SLM structured extraction on narrative and commit to shared ScratchPad."""
    payload = _slm_extract_raw(ctx.sync(), narrative, phase)
    mutations = payload.get("unresolved_variables_mutations", {})
    if extra_mutations:
        mutations.update(extra_mutations)
    ctx.commit(
        raw_chunk=narrative,
        triplets=payload.get("extracted_triplets", []),
        mutations=mutations,
    )


def _slm_extract_raw(view: str, narrative: str, phase: str) -> dict:
    """Call UniversalInferenceEngine.generate_structured() and return raw dict."""
    try:
        from inference import UniversalInferenceEngine  # type: ignore[import]
        from pydantic import BaseModel  # type: ignore[import]
        from typing import List, Dict as DictType

        class SREExtractionPayload(BaseModel):
            extracted_triplets: List[dict]
            unresolved_variables_mutations: DictType[str, str]
            is_chunk_completely_exhausted: bool

        engine = UniversalInferenceEngine()
        result: SREExtractionPayload = engine.generate_structured(
            prompt=f"{view}\n\n[RAW TELEMETRY — {phase}]\n{narrative}",
            system_prompt=_get_system_prompt(phase),
            response_schema=SREExtractionPayload,
        )
        return {
            "extracted_triplets": result.extracted_triplets,
            "unresolved_variables_mutations": result.unresolved_variables_mutations,
        }
    except Exception as e:
        logger.warning(f"[Router] SLM extraction failed for phase={phase}: {e}")
        return {"extracted_triplets": [], "unresolved_variables_mutations": {}}


def _get_system_prompt(phase: str) -> str:
    """Return the phase-specific SLM system prompt."""
    base = (
        "You are an SRE extraction engine. "
        "Extract facts as triplets from the raw telemetry below. "
        "Rules:\n"
        "  1. citation_quote MUST be an exact substring of the RAW TELEMETRY section.\n"
        "  2. Entity names must be UPPERCASE_WITH_UNDERSCORES.\n"
        "  3. Resolve a variable only when the text states it explicitly.\n"
        "  4. Output matches SREExtractionPayload exactly.\n\n"
    )
    phase_instructions = {
        "DETECTION": (
            "Resolve ANOMALY_CONFIRMED when text says 'Active anomaly confirmed' or "
            "'No active anomaly detected'."
        ),
        "LOCALIZATION": (
            "Resolve ROOT_CAUSE_SERVICE only when text says "
            "'Top suspect for root cause is X'."
        ),
        "RCA": (
            "Resolve FAULT_TYPE and FAULT_LAYER when the fault classification "
            "is explicit in the text."
        ),
        "MITIGATION": (
            "Resolve MITIGATION_ACTION, MITIGATION_TARGET, MITIGATION_NAMESPACE "
            "when the mitigation plan is confirmed."
        ),
    }
    return base + phase_instructions.get(phase, "")
