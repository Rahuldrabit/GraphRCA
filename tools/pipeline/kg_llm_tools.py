"""LLM Knowledge-Graph Access Tools (experimental).

This module provides two optional modes to let the LLM "see" or "query" the
knowledge graph during RCA.

Modes:
  - a: Dump the full NetworkX knowledge graph (node-link JSON) into the LLM prompt.
  - b: Neo4j GraphRAG loop where the LLM proposes Cypher READ queries; we execute
       them and feed the results back for a few rounds.

These are OFF by default and only enabled when GRAPHRCA_LLM_KG_MODE is set to
"a" or "b".
"""

from __future__ import annotations

import json
import logging
import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple

from GraphRCA_agent.llm import llm_reason
from GraphRCA_agent.tools.pipeline.graph_tools import export_graph_json

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()


def _clip_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return head + f"\n...[TRUNCATED {len(text) - max_chars} chars]...\n" + tail


def _extract_json_obj(raw: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM output."""
    if not raw:
        raise ValueError("empty LLM response")

    # Prefer fenced JSON blocks.
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())

    # Fallback: take the outermost {...}
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(raw[start : end + 1].strip())


def _json_safe(obj: Any, *, _depth: int = 0, _max_depth: int = 6) -> Any:
    """Convert arbitrary objects (incl. Neo4j temporal types) into JSON-safe values."""

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        # JSON doesn't support NaN/Infinity.
        if math.isfinite(obj):
            return obj
        return None

    if _depth >= _max_depth:
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _json_safe(v, _depth=_depth + 1, _max_depth=_max_depth) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]

    # Common protocol for date/time-like objects (Neo4j DateTime/Date/Time have iso_format()).
    iso_format = getattr(obj, "iso_format", None)
    if callable(iso_format):
        try:
            return iso_format()
        except Exception:
            pass

    # Python datetime/date/time (and other objects) implement isoformat().
    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    # Neo4j Node/Relationship often have _properties; best-effort flatten.
    props = getattr(obj, "_properties", None)
    if isinstance(props, dict):
        return {
            "_type": obj.__class__.__name__,
            "properties": _json_safe(props, _depth=_depth + 1, _max_depth=_max_depth),
        }

    return str(obj)


def _json_dumps_safe(obj: Any, *, max_chars: Optional[int] = None) -> str:
    text = json.dumps(_json_safe(obj), ensure_ascii=False)
    return _clip_text(text, max_chars) if max_chars else text


def _candidate_dict(c: Any) -> Dict[str, Any]:
    if c is None:
        return {}
    if hasattr(c, "to_dict") and callable(getattr(c, "to_dict")):
        try:
            return c.to_dict()
        except Exception:
            pass
    if isinstance(c, dict):
        return c
    if hasattr(c, "__dict__"):
        try:
            return dict(c.__dict__)
        except Exception:
            pass
    return {"raw": str(c)}


def _reorder_ranked(
    ranked_causes: List[Any],
    ranked_services: List[str],
    root_cause_service: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Reorder ranked_causes based on LLM-provided ordering (best-effort).

    Only services that already exist in ranked_causes are used; unknown services are ignored.
    """
    svc_to_candidate: Dict[str, Any] = {}
    for c in ranked_causes or []:
        svc = getattr(c, "service", None) if not isinstance(c, dict) else c.get("service")
        svc = str(svc or "").strip()
        if svc:
            svc_to_candidate[svc] = c

    desired: List[str] = []
    for s in ranked_services or []:
        s = str(s or "").strip()
        if s and s in svc_to_candidate and s not in desired:
            desired.append(s)

    if not desired:
        rc = str(root_cause_service or "").strip()
        if rc and rc in svc_to_candidate:
            desired = [rc]

    if not desired:
        return ranked_causes, {"applied": False, "reason": "LLM provided no reorderable services"}

    new_ranked: List[Any] = [svc_to_candidate[s] for s in desired]
    for c in ranked_causes or []:
        svc = getattr(c, "service", None) if not isinstance(c, dict) else c.get("service")
        svc = str(svc or "").strip()
        if svc and svc in desired:
            continue
        new_ranked.append(c)

    debug = {
        "applied": True,
        "requested_root": root_cause_service,
        "requested_order": desired,
    }
    return new_ranked, debug


_DISALLOWED_CYPHER = re.compile(
    r"(?i)\b(CREATE|MERGE|SET|DELETE|DETACH|DROP|REMOVE|CALL|LOAD\s+CSV|APOC\.|GDS\.)\b"
)
_ALLOWED_CYPHER_START = re.compile(r"(?i)^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND)\b")


def _is_safe_read_cypher(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if not _ALLOWED_CYPHER_START.search(q):
        return False
    if _DISALLOWED_CYPHER.search(q):
        return False
    # Require a RETURN to avoid accidental long-running queries.
    if not re.search(r"(?i)\bRETURN\b", q):
        return False
    return True


def _ensure_limit(query: str, limit: int) -> str:
    q = (query or "").strip().rstrip(";")
    if not q:
        return q
    if re.search(r"(?i)\bLIMIT\b", q):
        return q
    return f"{q} LIMIT {int(limit)}"


def _records_to_rows(records: Any, max_rows: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not records:
        return out
    for r in records:
        try:
            if hasattr(r, "data") and callable(getattr(r, "data")):
                out.append(_json_safe(r.data()))
            elif isinstance(r, dict):
                out.append(_json_safe(r))
            else:
                out.append({"row": str(r)})
        except Exception:
            out.append({"row": str(r)})
        if len(out) >= max_rows:
            break
    return out


def maybe_llm_rerank_with_kg(
    *,
    kg_mode: str,
    G: Any,
    error_service: str,
    ranked_causes: List[Any],
    neo4j_connector: Any = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Optionally re-rank RCA candidates using LLM KG access.

    Returns (new_ranked_causes, debug_dict).
    """
    mode = str(kg_mode or "").strip().lower()
    if mode not in {"a", "b"}:
        return ranked_causes, {"enabled": False}

    if not ranked_causes:
        return ranked_causes, {"enabled": True, "mode": mode, "skipped": "no ranked_causes"}

    max_candidates = _env_int("GRAPHRCA_LLM_KG_MAX_CANDIDATES", 12)
    max_tokens = _env_int("GRAPHRCA_LLM_KG_MAX_TOKENS", 900)

    candidates = [_candidate_dict(c) for c in (ranked_causes[:max_candidates] if max_candidates > 0 else ranked_causes)]

    if mode == "a":
        max_graph_chars = _env_int("GRAPHRCA_LLM_KG_MAX_CHARS", 60000)
        try:
            graph_json = export_graph_json(G) if G is not None else "{}"
        except Exception as e:
            logger.warning(f"KG dump failed: {e}")
            graph_json = "{}"
        graph_json = _clip_text(graph_json, max_graph_chars)

        prompt = (
            "You are an SRE root-cause analyst.\n"
            "You are given (1) a service dependency knowledge graph as JSON (NetworkX node-link format), "
            "and (2) a shortlist of RCA candidates from a deterministic algorithm.\n\n"
            "Task: pick the single most likely root-cause service and optionally reorder the shortlist.\n\n"
            "Return STRICT JSON only (no markdown):\n"
            "{\n"
            "  \"root_cause_service\": \"<service>\",\n"
            "  \"ranked_services\": [\"<service1>\", \"<service2>\", ...],\n"
            "  \"reason\": \"<one-paragraph reason>\"\n"
            "}\n\n"
            "Rules:\n"
            "- Prefer choosing from the provided candidates.\n"
            "- If you choose a service not in candidates, it MUST appear in the graph nodes.\n"
            "- Use exact service names; do not invent names.\n\n"
            f"primary_error_service: {error_service}\n\n"
            f"candidates_json: {json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"knowledge_graph_json: {graph_json}\n"
        )

        raw = llm_reason(prompt, max_tokens=max_tokens, caller="rca.kg_mode_a")
        debug: Dict[str, Any] = {
            "enabled": True,
            "mode": mode,
            "max_candidates": max_candidates,
            "max_graph_chars": max_graph_chars,
        }
        try:
            parsed = _extract_json_obj(raw)
            debug["llm"] = {
                "root_cause_service": parsed.get("root_cause_service", ""),
                "ranked_services": parsed.get("ranked_services", []),
            }
            new_ranked, apply_debug = _reorder_ranked(
                ranked_causes,
                parsed.get("ranked_services") or [],
                parsed.get("root_cause_service") or "",
            )
            debug.update(apply_debug)
            return new_ranked, debug
        except Exception as e:
            debug["error"] = f"parse_failed: {e}"
            return ranked_causes, debug

    # mode == "b": Neo4j GraphRAG loop
    debug: Dict[str, Any] = {"enabled": True, "mode": mode, "max_candidates": max_candidates}

    if neo4j_connector is None or not getattr(neo4j_connector, "is_available", lambda: False)():
        debug["skipped"] = "neo4j_unavailable"
        return ranked_causes, debug

    max_rounds = _env_int("GRAPHRCA_LLM_KG_RAG_ROUNDS", 3)
    max_rows = _env_int("GRAPHRCA_LLM_KG_RAG_MAX_ROWS", 30)
    max_result_chars = _env_int("GRAPHRCA_LLM_KG_RAG_MAX_RESULT_CHARS", 12000)

    schema = (
        "Neo4j schema (available labels/relationships):\n"
        "- (:Service {name, span_count, error_rate, duration_mean_ms, ...})\n"
        "- (:Service)-[:CALLS]->(:Service) with edge props (call_count, error_rate, avg_duration_ms, operations, ...)\n"
        "- (:Trace {trace_id})\n"
        "- (:Span {trace_id, span_id, service_name, operation_name, start_time, duration_ms, has_error, response, is_root})\n"
        "- (Trace)-[:CONTAINS]->(Span)\n"
        "- (Span)-[:EXECUTED_BY]->(Service)\n"
        "- (Span)-[:CHILD_OF]->(Span)\n"
    )

    history: List[Dict[str, Any]] = []
    last_query = ""
    last_rows: List[Dict[str, Any]] = []
    last_error = ""

    system_prompt = (
        "You are a GraphRCA GraphRAG assistant. "
        "You may request Cypher READ queries to inspect the Neo4j graph. "
        "You must NEVER use write operations (CREATE/MERGE/SET/DELETE/DETACH/REMOVE/DROP/CALL/APOC/GDS). "
        "Always output STRICT JSON only."
    )

    for round_idx in range(max_rounds):
        prompt = (
            f"{schema}\n"
            f"primary_error_service: {error_service}\n\n"
            f"candidates_json: {_json_dumps_safe(candidates)}\n\n"
            "You can do one of two actions by returning JSON:\n"
            "1) Ask for a Cypher query:\n"
            "   {\"action\": \"query\", \"cypher\": \"...\", \"why\": \"...\"}\n"
            "2) Provide a final answer:\n"
            "   {\"action\": \"final\", \"root_cause_service\": \"...\", \"ranked_services\": [..], \"reason\": \"...\"}\n\n"
            "Constraints:\n"
            "- Cypher must be READ-only and include RETURN.\n"
            "- Keep queries small; include LIMIT if possible.\n"
            "- Use exact service names; do not invent.\n\n"
        )

        if history:
            prompt += f"Previous queries/results: {_json_dumps_safe(history)}\n\n"
        if last_error:
            prompt += f"Last query error: {last_error}\n\n"
        if last_query:
            prompt += f"Last cypher: {last_query}\nLast rows: {_json_dumps_safe(last_rows, max_chars=max_result_chars)}\n\n"

        raw = llm_reason(prompt, system_prompt=system_prompt, max_tokens=max_tokens, caller=f"rca.kg_mode_b.r{round_idx+1}")
        try:
            parsed = _extract_json_obj(raw)
        except Exception as e:
            last_error = f"parse_failed: {e}"
            continue

        action = str(parsed.get("action", "")).strip().lower()
        if action == "final":
            debug["llm"] = {
                "root_cause_service": parsed.get("root_cause_service", ""),
                "ranked_services": parsed.get("ranked_services", []),
                "rounds": round_idx + 1,
            }
            new_ranked, apply_debug = _reorder_ranked(
                ranked_causes,
                parsed.get("ranked_services") or [],
                parsed.get("root_cause_service") or "",
            )
            debug.update(apply_debug)
            debug["history"] = history
            return new_ranked, debug

        if action != "query":
            last_error = f"invalid_action: {action}" if action else "missing_action"
            continue

        cypher = str(parsed.get("cypher", "")).strip()
        if not _is_safe_read_cypher(cypher):
            last_error = "unsafe_or_invalid_cypher"
            history.append({"cypher": cypher, "error": last_error})
            continue

        cypher = _ensure_limit(cypher, max_rows)

        try:
            records = None
            if hasattr(neo4j_connector, "execute_read"):
                records = neo4j_connector.execute_read(cypher)
            elif hasattr(neo4j_connector, "run_query"):
                records = neo4j_connector.run_query(cypher)
            rows = _records_to_rows(records, max_rows=max_rows)

            last_query = cypher
            last_rows = rows
            last_error = ""
            rows_preview = _json_dumps_safe(rows, max_chars=max_result_chars)
            history.append({
                "cypher": cypher,
                "row_count": len(rows),
                "rows_preview": rows_preview,
            })
        except Exception as e:
            last_error = str(e)
            history.append({"cypher": cypher, "error": last_error})

    # If we exhausted rounds, fall back to original ordering.
    debug["history"] = history
    debug["skipped"] = debug.get("skipped") or "max_rounds_exhausted"
    return ranked_causes, debug
