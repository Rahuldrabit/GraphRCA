import json
import os
import re
import logging
from typing import Dict, Any, List
from swarm_state import AIOpsIncidentState
from tools.scratchpad_client import ScratchpadClient
from llm import llm_reason

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return str(os.getenv(name, str(default))).strip().lower() in ("1", "true", "yes", "on")


# Completion-token budget for the RCA / drill-down LLM calls. gemma4-graphrca is a
# "thinking" model: on the REAL pipeline (richer graph than the offline harness)
# its reasoning runs well past 4096 tokens, hits the old hardcoded cap mid-trace,
# and emits an EMPTY response -> `_extract_json` raises "empty response" -> the
# drill loop aborts -> falls back to `suspects[0]` (the loudest SYMPTOM service,
# e.g. compose-post-service, not the root cause) -> Localization Accuracy 0.0.
# 8192 gives ~2x headroom over the observed ~4k reasoning trace + the ~300-token
# JSON answer, and sits comfortably inside the num_ctx=32768 baked into
# gemma4-graphrca:12b. Tunable per-run without a code edit.
RCA_MAX_TOKENS = _env_int("GRAPHRCA_RCA_MAX_TOKENS", 8192)


def _balanced_objects(s: str):
    """Yield each top-level balanced {...} substring in s (string/escape-aware)."""
    depth = 0
    in_str = False
    esc = False
    start = None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        yield s[start:i + 1]
                        start = None


def _extract_json(raw: str) -> Dict[str, Any]:
    """Best-effort JSON object extraction from an LLM response.

    Tolerant of what small / reasoning models actually emit:
      - <think>/<reasoning>/<reflection> blocks (deepseek-r1, gemma CoT) — stripped.
      - ```json ... ``` code fences — unwrapped.
      - JSON embedded in prose / with trailing commentary — brace-matched.
      - Whitespace-only or None — raises "empty response".
    Returns the first parsed dict.
    """
    if raw is None:
        raise ValueError("empty response")
    text = str(raw).strip()
    if not text:
        raise ValueError("empty response")

    # 1. Drop reasoning/think blocks (paired first, then any dangling unclosed prefix).
    text = re.sub(
        r"<(?:think|reasoning|reflection)>.*?</(?:think|reasoning|reflection)>",
        "", text, flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<(?:think|reasoning|reflection)>.*", "", text,
                  flags=re.DOTALL | re.IGNORECASE)
    # Gemma4 "thinking" artifacts: the <|think|> enable token and the
    # <|channel|>thought … <|/channel|> reasoning wrapper (empty when thinking is
    # off). Strip paired blocks + the enable token; the brace-matcher below is the
    # real backstop if any control tokens leak through past ollama's own stripping.
    text = re.sub(r"<\|channel\|>\s*thought.*?<\|/channel\|>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<\|/?think\|>", " ", text)
    text = text.strip()
    if not text:
        raise ValueError("empty response (only reasoning)")

    # 2. Unwrap a code fence if present.
    candidates: List[str] = []
    fence = re.search(r"```[a-zA-Z0-9]*\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text)

    # 3. Try direct parse, then each balanced object, of each candidate.
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        for span in _balanced_objects(cand):
            try:
                obj = json.loads(span)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    raise ValueError("no JSON object found")


class RCAAnalystAgent:
    """
    SLM agent that analyzes the ScratchPad markdown view.
    Verifies the root cause and provides a citation quote.

    Two modes (ScratchPad-centered upgrade, gated by GRAPHRCA_SP_DRILLDOWN):
      OFF (default): single-shot — one bounded view, one LLM call. (v1 behaviour,
                     byte-for-byte, so a running v1 eval is never disturbed.)
      ON:            bounded drill-down loop — the SLM reads a query-aware k-hop
                     "radius blast" view, then may drill into COMPRESSED suspects
                     (served from ScratchPad, budget-bounded) before committing.
                     Falls back to single-shot on any error or round exhaustion.

    Hyperparameters (LLM-tunable via env, sensible defaults):
      GRAPHRCA_SP_DRILLDOWN   = 0       (master switch)
      GRAPHRCA_SP_KHOPS       = 3       (radius-blast depth for the bounded view)
      GRAPHRCA_SP_VIEW_TOKENS = 1200    (token budget for the bounded view)
      GRAPHRCA_SP_MAX_ROUNDS  = 3       (drill-down loop cap)
      GRAPHRCA_SP_DRILL_HOPS  = 1       (neighborhood depth per drill-down)
    """
    def __init__(self, scratchpad_client: ScratchpadClient):
        self.client = scratchpad_client

    # ── public node entry ─────────────────────────────────────────────────

    def __call__(self, state: AIOpsIncidentState) -> AIOpsIncidentState:
        session_id = state["scratchpad_session_id"]
        suspects = state.get("suspect_nodes", [])

        if not suspects:
            logger.warning("No suspects provided to RCA Analyst.")
            state["verified_root_cause"] = None
            return state

        # Drill-down RCA only helps where a root cause is actually consumed
        # (localization / analysis / mitigation). Detection just needs the anomaly
        # flag from the guardrail, so skip the extra LLM rounds there.
        if _env_bool("GRAPHRCA_SP_DRILLDOWN", False) and state.get("task_type") != "detection":
            try:
                return self._drill_down_loop(state, session_id, suspects)
            except Exception as e:
                # Never let the upgrade break a task — fall back to v1.
                logger.warning(f"[RCA] drill-down loop failed ({e}); falling back to single-shot")

        return self._single_shot(state, session_id, suspects)

    # ── v1: single-shot (unchanged) ───────────────────────────────────────

    def _single_shot(self, state: AIOpsIncidentState, session_id: str, suspects: List[str]) -> AIOpsIncidentState:
        # Get bounded markdown view
        markdown_view = self.client.get_view(session_id, max_tokens=500)

        system_prompt = (
            "You are an expert Site Reliability Engineer (SRE) performing Root Cause Analysis. "
            "You will be given a list of suspect services and a compact knowledge graph representation "
            "(triplets) of the system state. "
            "Your job is to identify the single root cause service and extract the EXACT citation quote "
            "from the knowledge graph that proves it."
        )

        user_prompt = (
            f"Suspect nodes from topological analysis: {suspects}\n\n"
            f"Knowledge Graph View:\n{markdown_view}\n\n"
            "Based on the above, identify the root cause service.\n"
            "Respond ONLY in valid JSON format:\n"
            "{\n"
            '  "root_cause_service": "SERVICE_NAME",\n'
            '  "relationship": "emits",\n'
            '  "target": "ERROR_STATE",\n'
            '  "citation_quote": "exact quote from the view"\n'
            "}"
        )

        response = llm_reason(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=RCA_MAX_TOKENS,
            caller="rca_analyst"
        )

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                data = json.loads(json_str)

                root_cause = data.get("root_cause_service")
                if root_cause:
                    state["verified_root_cause"] = root_cause

                    # Commit the verified hypothesis back to ScratchPad
                    self.client.commit_triplets(
                        session_id,
                        "RCAAnalyst",
                        [{
                            "source": root_cause,
                            "relationship": data.get("relationship", "causes"),
                            "target": data.get("target", "INCIDENT"),
                            "citation_quote": data.get("citation_quote", ""),
                            "source_type": "SERVICE",
                            "target_type": "FAULT"
                        }]
                    )
                    logger.info(f"RCA Analyst verified root cause: {root_cause}")
                else:
                    state["verified_root_cause"] = suspects[0]  # fallback
            else:
                state["verified_root_cause"] = suspects[0]
        except Exception as e:
            logger.error(f"Failed to parse RCA LLM response: {e}")
            state["verified_root_cause"] = suspects[0] if suspects else None

        return state

    # ── v2: bounded drill-down loop (agent thinks + queries the KG) ───────

    def _drill_down_loop(self, state: AIOpsIncidentState, session_id: str, suspects: List[str]) -> AIOpsIncidentState:
        k_hops = _env_int("GRAPHRCA_SP_KHOPS", 3)
        view_tokens = _env_int("GRAPHRCA_SP_VIEW_TOKENS", 1200)
        max_rounds = _env_int("GRAPHRCA_SP_MAX_ROUNDS", 3)
        drill_hops = _env_int("GRAPHRCA_SP_DRILL_HOPS", 1)

        # Query = the incident description, falling back to the top suspect.
        # This drives the query-aware k-hop blast inside compile_bounded_markdown_view.
        query = (state.get("problem_description") or "").strip() or (suspects[0] if suspects else "")

        # Round 0: the radius blast — a query-biased, token-bounded view.
        view = self.client.get_view(session_id, max_tokens=view_tokens, query=query, k_hops=k_hops)

        system_prompt = (
            "You are an expert SRE performing Root Cause Analysis over a knowledge graph. "
            "You are given suspect services and a token-bounded view of the graph. "
            "Some facts are compressed and marked `[COMPRESSED | drill-down id: <edge_id>]`; "
            "you may expand any of them (or any service) to inspect its neighbourhood before deciding. "
            "Always reply with a SINGLE strict JSON object."
        )

        drilled: List[str] = []
        for round_idx in range(max_rounds):
            prompt = (
                f"Incident / query: {query}\n"
                f"Suspect nodes from topological analysis: {suspects}\n\n"
                f"Knowledge Graph View:\n{view}\n\n"
            )
            if drilled:
                prompt += "Expanded neighbourhoods from your drill-downs:\n" + "\n\n".join(drilled) + "\n\n"
            prompt += (
                "Take ONE action by returning strict JSON only:\n"
                '  drill  : {"action":"drill", "target":"<edge_id or service name>", "why":"..."}\n'
                '  final  : {"action":"final", "root_cause_service":"SERVICE_NAME", '
                '"relationship":"emits", "target":"ERROR_STATE", "citation_quote":"exact quote"}\n'
                "Prefer `final` once you are confident. Use exact service names; do not invent names."
            )

            raw = llm_reason(prompt=prompt, system_prompt=system_prompt, max_tokens=RCA_MAX_TOKENS,
                             caller=f"rca_analyst.drill.r{round_idx}")
            data = None
            for _attempt in (1, 2):  # one retry on parse failure before giving up
                try:
                    data = _extract_json(raw)
                    break
                except Exception as e:
                    if _attempt == 1:
                        logger.debug(
                            f"[RCA] round {round_idx} parse failed ({e}); retrying with stricter instruction")
                        raw = llm_reason(
                            prompt=prompt + "\n\nReply with ONLY the JSON object — "
                            "no prose, no code fence, no reasoning.",
                            system_prompt=system_prompt, max_tokens=RCA_MAX_TOKENS,
                            caller=f"rca_analyst.drill.r{round_idx}.retry",
                        )
                    else:
                        logger.warning(
                            f"[RCA] round {round_idx} parse failed ({e}); ending loop")
            if data is None:
                break

            action = str(data.get("action", "")).strip().lower()
            if action == "final":
                root_cause = (data.get("root_cause_service") or "").strip()
                if root_cause:
                    self._commit(session_id, root_cause,
                                 data.get("relationship", "causes"),
                                 data.get("target", "INCIDENT"),
                                 data.get("citation_quote", ""))
                    state["verified_root_cause"] = root_cause
                    logger.info(f"RCA Analyst (drill-down, {round_idx + 1} round(s)) verified root cause: {root_cause}")
                    return state
                break  # malformed final -> fall through to fallback

            if action == "drill":
                target = (data.get("target") or "").strip()
                if target:
                    expansion = self.client.drill_down(session_id, target, k_hops=drill_hops)
                    drilled.append(f"### drill {target}\n{expansion}")
                    logger.info(f"[RCA] round {round_idx} drilled into '{target}'")
                    continue

            # Unknown/empty action -> stop drilling.
            break

        # Rounds exhausted without a confident final answer: fall back to single-shot,
        # which re-reads the (non-query) view and commits. Keeps v1's safety net.
        logger.info("[RCA] drill-down rounds exhausted; falling back to single-shot")
        return self._single_shot(state, session_id, suspects)

    def _commit(self, session_id: str, root_cause: str, relationship: str, target: str, citation: str) -> None:
        self.client.commit_triplets(
            session_id, "RCAAnalyst",
            [{
                "source": root_cause,
                "relationship": relationship,
                "target": target,
                "citation_quote": citation,
                "source_type": "SERVICE",
                "target_type": "FAULT",
            }]
        )
