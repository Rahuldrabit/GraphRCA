"""LLM Client Factory with comprehensive output logging.

Provides a shared ChatOpenAI instance for LangGraph nodes
and a raw OpenAI client for direct function-calling.
Uses the same env vars as the Stratus pipeline.

All LLM interactions are logged to llm_justification.jsonl for audit.
"""

import json
import os
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "y", "on"}

# ── Singletons ─────────────────────────────────────────────────────────
_openai_client: Optional[OpenAI] = None
_chat_model = None
_llm_log_path: Optional[str] = None
_call_counter: int = 0
_total_prompt_tokens: int = 0
_total_completion_tokens: int = 0


def set_llm_log_dir(output_dir: str):
    """Set the directory for LLM justification logs. Called by run_pipeline.

    Also stamped into an env var, not just the module global: this repo is
    importable under two names (`llm` and `GraphRCA_agent.llm`, because
    `GraphRCA_agent` is a self-symlink to `.`), so a caller that imports
    one name never sees a global set on the other -- every swarm node call
    was silently falling back to the relative GraphRCA/logs/ path instead
    of the per-task output dir. An env var is process-wide, immune to which
    module identity did the writing.
    """
    global _llm_log_path
    os.makedirs(output_dir, exist_ok=True)
    _llm_log_path = os.path.join(output_dir, "llm_justification.jsonl")
    os.environ["GRAPHRCA_LLM_LOG_PATH"] = _llm_log_path
    logger.info(f"LLM justification log: {_llm_log_path}")


def get_token_totals() -> dict:
    """Return accumulated token usage across all LLM calls."""
    return {
        "prompt_tokens": _total_prompt_tokens,
        "completion_tokens": _total_completion_tokens,
        "total_tokens": _total_prompt_tokens + _total_completion_tokens,
    }


def reset_token_totals():
    """Zero out token accumulators (e.g. between retry runs)."""
    global _total_prompt_tokens, _total_completion_tokens
    _total_prompt_tokens = 0
    _total_completion_tokens = 0


def _log_llm_call(caller: str, model: str, prompt: str, system_prompt: str,
                   response: str, tokens_used: dict, elapsed_s: float):
    """Append a structured JSON log entry for every LLM call."""
    global _call_counter, _total_prompt_tokens, _total_completion_tokens
    _call_counter += 1

    # Accumulate token counts
    _total_prompt_tokens += tokens_used.get("prompt", 0)
    _total_completion_tokens += tokens_used.get("completion", 0)

    def _full_enabled() -> bool:
        return os.getenv("GRAPHRCA_LLM_LOG_FULL", "").strip().lower() in _TRUTHY

    def _max_chars(env_name: str, default: int) -> int:
        try:
            return int(os.getenv(env_name, str(default)))
        except Exception:
            return default

    def _clip(text: str, env_name: str, default: int) -> str:
        if not text:
            return ""
        if _full_enabled():
            return text
        limit = max(200, _max_chars(env_name, default))
        if len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-(limit // 2) :]
        return head + f"\n...[TRUNCATED {len(text) - limit} chars]...\n" + tail

    entry = {
        "call_id": _call_counter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "caller": caller,
        "model": model,
        "system_prompt": _clip(system_prompt, "GRAPHRCA_LLM_LOG_MAX_SYSTEM_CHARS", 500),
        "user_prompt": _clip(prompt, "GRAPHRCA_LLM_LOG_MAX_USER_CHARS", 2000),
        "response": _clip(response, "GRAPHRCA_LLM_LOG_MAX_RESPONSE_CHARS", 3000),
        "tokens": tokens_used,
        "elapsed_seconds": round(elapsed_s, 3),
    }

    # Also emit to the structured tracer (if enabled)
    try:
        from GraphRCA_agent.trace_logger import trace_event

        trace_event(
            "llm.call",
            caller=caller,
            model=model,
            system_prompt=system_prompt or "",
            user_prompt=prompt or "",
            response=response or "",
            tokens=tokens_used,
            elapsed_seconds=round(elapsed_s, 3),
        )
    except Exception:
        pass

    # Write to pipeline output dir if configured. Check the env var first
    # (set by set_llm_log_dir on ANY module identity, see its docstring)
    # before this identity's own possibly-unset module global.
    log_path = os.environ.get("GRAPHRCA_LLM_LOG_PATH") or _llm_log_path
    if log_path is None:
        # Fallback to GraphRCA/logs/
        os.makedirs("GraphRCA/logs", exist_ok=True)
        log_path = "GraphRCA/logs/llm_justification.jsonl"

    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write LLM log: {e}")

    # Also log summary to standard logger
    logger.info(
        f"[LLM #{_call_counter}] {caller} | model={model} | "
        f"tokens={tokens_used} | {elapsed_s:.2f}s"
    )


def get_api_key() -> str:
    """Get the API key from environment."""
    return os.getenv("API_KEY_AGENTS") or os.getenv("OPENAI_API_KEY", "")


def get_base_url() -> str:
    """Get the base URL from environment."""
    return os.getenv("URL_AGENTS", "https://api.openai.com/v1")


def get_model_name() -> str:
    """Get the model name from environment."""
    return os.getenv("MODEL_AGENTS", "gpt-4.1-nano")


def get_temperature() -> float:
    """Get temperature from environment."""
    return float(os.getenv("TEMPERATURE_AGENTS", "0.0"))


def get_top_p():
    """Get top_p from environment, or None to use the server default."""
    raw = os.getenv("TOP_P_AGENTS", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def get_top_k():
    """Get top_k from environment, or None to leave unset.

    The OpenAI API has no top_k, but ollama and other local servers accept it;
    callers pass it via the request's passthrough (extra_body).
    """
    raw = os.getenv("TOP_K_AGENTS", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def get_openai_client() -> OpenAI:
    """Return a shared raw OpenAI client (lazy init)."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=get_api_key(),
            base_url=get_base_url(),
        )
        logger.info(f"OpenAI client initialized (base_url={get_base_url()})")
    return _openai_client


def get_chat_model():
    """Return a LangChain ChatOpenAI model for LangGraph nodes.

    Falls back to raw OpenAI client if langchain-openai is not installed.
    """
    global _chat_model
    if _chat_model is not None:
        return _chat_model

    try:
        from langchain_openai import ChatOpenAI

        _chat_model = ChatOpenAI(
            model=get_model_name(),
            temperature=get_temperature(),
            api_key=get_api_key(),
            base_url=get_base_url(),
            max_tokens=4096,
        )
        logger.info(f"ChatOpenAI model initialized (model={get_model_name()})")
    except ImportError:
        logger.warning("langchain-openai not available, using raw OpenAI client")
        _chat_model = get_openai_client()

    return _chat_model


def llm_reason(prompt: str, system_prompt: str = "", max_tokens: int = 2000,
               caller: str = "llm_reason") -> str:
    """Call the LLM for reasoning using the raw OpenAI client.

    All calls are logged to llm_justification.jsonl for audit/justification.

    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        max_tokens: Max response tokens
        caller: Name of the calling node/function for log context

    Returns:
        LLM response text
    """
    client = get_openai_client()
    model = get_model_name()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    t0 = time.time()
    try:
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=get_temperature(),
        )
        top_p = get_top_p()
        if top_p is not None:
            kwargs["top_p"] = top_p
        top_k = get_top_k()
        if top_k is not None:
            # The OpenAI API has no top_k; ollama/local servers accept it via the
            # passthrough body. Required for gemma4's recommended sampling (top_k=64).
            kwargs["extra_body"] = {"top_k": top_k}
        response = client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0
        content = response.choices[0].message.content or ""

        # Extract token usage
        usage = response.usage
        tokens = {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        }

        # Log for justification
        _log_llm_call(caller, model, prompt, system_prompt, content, tokens, elapsed)

        return content
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"LLM call failed: {e}")
        _log_llm_call(caller, model, prompt, system_prompt, f"[ERROR: {e}]", {}, elapsed)
        return f"[LLM Error: {e}]"
