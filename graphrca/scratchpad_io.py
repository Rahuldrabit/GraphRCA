"""ScratchPad I/O — Thin wrappers over agent_sdk for GraphRCA v5.

Maps directly to the verified ScratchPad API surface:
  POST /v1/session/init
  GET  /v1/session/{id}/memory
  POST /v1/agent/update
  POST /v1/middleware/drill-down

Ground truth:
  - GET /v1/session/{id}/memory takes NO agent_id — every agent reads
    the identical shared view. agent_id is used only for write attribution.
  - Posting {"VAR": "RESOLVED"} in unresolved_variables_mutations DELETES
    the row. Resolved = absent from matrix, not present with a status.
  - citation_quote MUST be an exact substring of raw_active_chunk or the
    triplet is silently dropped server-side. All triplets dropped → HTTP 422.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

SCRATCHPAD_URL = os.getenv("SCRATCHPAD_URL", "http://localhost:8000")


# ── Session Lifecycle ────────────────────────────────────────────────────────


async def T_init_session(session_id: str, user_query: str) -> None:
    """Create (or idempotently re-open) a ScratchPad session.

    master_plan is intentionally omitted — compile_bounded_markdown_view()
    never reads it, so it costs tokens without helping agents.
    """
    async with httpx.AsyncClient(base_url=SCRATCHPAD_URL) as c:
        resp = await c.post(
            "/v1/session/init",
            json={"session_id": session_id, "user_query": user_query},
        )
        resp.raise_for_status()
    logger.debug(f"[ScratchPad] Session initialised: {session_id}")


async def T_seed_variables(client, var_names: list[str]) -> dict:
    """Seed the Unresolved Variables Matrix with MISSING entries.

    This is the very first write of the session. Uses a placeholder
    raw_chunk with no triplets — the INSERT OR IGNORE semantics mean
    re-seeding on a retry is safe (already-present vars stay unchanged).

    Args:
        client: ScratchpadAgentClient bound to the shared session.
        var_names: Variable names to mark MISSING.

    Returns:
        Server response dict from update_memory().
    """
    mutations = {v: "MISSING" for v in var_names}
    result = await client.update_memory(
        raw_chunk=f"Investigation initialised for session {client.session_id}.",
        triplets=[],
        variables=mutations,
        is_done=True,
    )
    logger.debug(f"[ScratchPad] Seeded {len(var_names)} variables: {var_names}")
    return result


# ── Per-Turn I/O ─────────────────────────────────────────────────────────────


async def T_sync(client, max_tokens: int = 6000) -> str:
    """Fetch the current shared markdown view.

    Shared across ALL agents — no agent_id filter on the read side.
    The Unresolved Variables Matrix section is guaranteed never-truncated.

    Args:
        client: ScratchpadAgentClient bound to the target session.
        max_tokens: Soft cap on the returned view size.

    Returns:
        Markdown string with Knowledge Graph + Unresolved Variables Matrix.
    """
    return await client.get_memory_view(max_tokens=max_tokens)


async def T_commit(
    client,
    raw_chunk: str,
    triplets: list[dict],
    mutations: dict[str, str],
) -> dict:
    """Write extracted facts + variable mutations to ScratchPad.

    Server-side verification runs automatically:
      - citation_quote must be exact substring of raw_chunk (else triplet dropped).
      - Canonicalization (rapidfuzz, threshold 85) merges near-duplicate entities.
      - If ALL triplets are citation-invalid → HTTP 422.

    Mutations follow binary semantics:
      "MISSING"   → INSERT OR IGNORE (first write wins)
      "RESOLVED"  → DELETE the row (variable disappears from matrix)

    Args:
        client: ScratchpadAgentClient bound to the target session.
        raw_chunk: The source text. All citation_quote values must be substrings.
        triplets: List of {source_entity, relationship, target_entity, citation_quote}.
        mutations: {VAR_NAME: "MISSING" | "RESOLVED"} dict.

    Returns:
        Server response dict. Contains {"status": "rejected"} on HTTP 422.
    """
    try:
        result = await client.update_memory(
            raw_chunk, triplets, mutations, is_done=True
        )
        return result
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 422:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            logger.warning(
                f"[ScratchPad] HTTP 422 — all triplets rejected (citation mismatch). "
                f"detail={detail!r}"
            )
            return {"status": "rejected", "detail": detail}
        raise


async def T_drill_down(client, edge_id: str) -> list[dict]:
    """Expand a compressed node back to its L1 triplets.

    Args:
        client: ScratchpadAgentClient bound to the target session.
        edge_id: The [COMPRESSED | drill-down id: X] edge identifier.

    Returns:
        List of granular history dicts from the server.
    """
    result = await client.drill_down(edge_id)
    if isinstance(result, dict):
        return result.get("granular_history", [])
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────


def is_resolved(markdown_view: str, var_name: str) -> bool:
    """Check whether a variable has been resolved.

    Resolution DELETES the row from the Unresolved Variables Matrix,
    so a resolved variable is simply absent from the matrix section.
    We search only the matrix section (after '## 2.') to avoid false
    positives from variable names that appear in the Knowledge Graph rows.

    Args:
        markdown_view: Full markdown string returned by T_sync().
        var_name: Variable name to check (e.g. "ANOMALY_CONFIRMED").

    Returns:
        True if the variable is absent from the Unresolved Variables Matrix
        (i.e. resolved), False if it is still listed as MISSING.
    """
    if "## 2." in markdown_view:
        matrix_section = markdown_view.split("## 2.")[1]
    else:
        matrix_section = markdown_view
    return f"`{var_name}`" not in matrix_section


def extract_resolved_value(markdown_view: str, var_name: str) -> str | None:
    """Parse a resolved variable's value from the Knowledge Graph rows.

    After a variable is resolved (row deleted from matrix), its value
    lives as a fact in the Knowledge Graph section. This function performs
    a best-effort extraction by scanning for lines containing var_name
    and returning the target entity from the first matching triplet.

    Args:
        markdown_view: Full markdown string returned by T_sync().
        var_name: Resolved variable name (e.g. "ROOT_CAUSE_SERVICE").

    Returns:
        Extracted entity string, or None if not found.
    """
    for line in markdown_view.splitlines():
        if var_name in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            # Typical row: | source | relationship | target | citation |
            if len(parts) >= 4:
                # Target entity is index 3 (after leading empty from split)
                candidate = parts[3] if len(parts) > 3 else parts[-1]
                candidate = candidate.strip("`").strip()
                if candidate and candidate not in ("target_entity", "---", ""):
                    return candidate
    return None
