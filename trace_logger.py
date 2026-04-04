"""Lightweight structured tracing for GraphRCA.

When enabled, writes JSONL events into the current run's output directory so
users can inspect *every* tool call / LLM call / Neo4j query end-to-end.

Enable with:
  GRAPHRCA_TRACE=1
Optional:
  GRAPHRCA_TRACE_PRINT=1        # also print a brief summary to stdout
  GRAPHRCA_TRACE_FULL=1         # avoid truncation (still capped by HARD max)
  GRAPHRCA_TRACE_MAX_CHARS=5000 # truncation window when FULL=0
  GRAPHRCA_TRACE_HARD_MAX_CHARS=200000

NOTE: This tracer tries to redact obvious secrets (api keys, passwords).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_trace_log_path: Optional[str] = None


_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY


def trace_enabled() -> bool:
    return _env_truthy("GRAPHRCA_TRACE") or _env_truthy("GRAPHRCA_TRACE_ENABLED")


def trace_print_enabled() -> bool:
    return _env_truthy("GRAPHRCA_TRACE_PRINT")


def trace_print_full_enabled() -> bool:
    return _env_truthy("GRAPHRCA_TRACE_PRINT_FULL")


def trace_full_enabled() -> bool:
    return _env_truthy("GRAPHRCA_TRACE_FULL")


def _trace_max_chars() -> int:
    try:
        return int(os.getenv("GRAPHRCA_TRACE_MAX_CHARS", "5000"))
    except Exception:
        return 5000


def _trace_hard_max_chars() -> int:
    try:
        return int(os.getenv("GRAPHRCA_TRACE_HARD_MAX_CHARS", "200000"))
    except Exception:
        return 200000


def set_trace_log_dir(output_dir: str) -> None:
    """Configure where JSONL trace events should be written for the current run."""
    global _trace_log_path
    os.makedirs(output_dir, exist_ok=True)
    _trace_log_path = os.path.join(output_dir, "trace_events.jsonl")
    if trace_enabled():
        logger.info(f"Trace events log: {_trace_log_path}")


def get_trace_log_path() -> Optional[str]:
    return _trace_log_path


_SECRET_KEY_NAMES = {
    "password",
    "pass",
    "secret",
    "token",
    "authorization",
    "api_key",
    "apikey",
    "openai_api_key",
    "neo4j_password",
}


def _redact_text(text: str) -> str:
    if not text:
        return text

    # Common API key patterns
    text = re.sub(r"sk-[A-Za-z0-9]{16,}", "sk-[REDACTED]", text)

    # key=value style
    text = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,'\"]+)", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(openai[_-]?api[_-]?key\s*[:=]\s*)([^\s,'\"]+)", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(neo4j[_-]?password\s*[:=]\s*)([^\s,'\"]+)", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(password\s*[:=]\s*)([^\s,'\"]+)", r"\1[REDACTED]", text)

    return text


def _truncate_text(text: str) -> str:
    """Truncate long text unless FULL tracing is enabled."""
    if text is None:
        return ""

    text = _redact_text(text)

    hard_max = max(1000, _trace_hard_max_chars())
    if trace_full_enabled():
        if len(text) <= hard_max:
            return text
        return text[:hard_max] + f"\n...[HARD-TRUNCATED {len(text) - hard_max} chars]...\n"

    max_chars = max(200, _trace_max_chars())
    if len(text) <= max_chars:
        return text

    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return head + f"\n...[TRUNCATED {len(text) - max_chars} chars]...\n" + tail


def _redact_obj(obj: Any) -> Any:
    """Recursively redact sensitive keys and truncate large strings."""
    if obj is None:
        return None

    if isinstance(obj, str):
        return _truncate_text(obj)

    if isinstance(obj, (int, float, bool)):
        return obj

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if key.strip().lower() in _SECRET_KEY_NAMES:
                out[key] = "[REDACTED]"
            else:
                out[key] = _redact_obj(v)
        return out

    if isinstance(obj, (list, tuple)):
        return [_redact_obj(x) for x in obj]

    # Fallback: stringify unknown types
    return _truncate_text(str(obj))


def trace_event(event_type: str, **fields: Any) -> None:
    """Append a structured JSONL trace event if tracing is enabled."""
    if not trace_enabled():
        return

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
    }

    for k, v in fields.items():
        entry[k] = _redact_obj(v)

    log_path = _trace_log_path
    if log_path is None:
        # Fallback: local logs folder
        os.makedirs("GraphRCA/logs", exist_ok=True)
        log_path = os.path.join("GraphRCA/logs", "trace_events.jsonl")

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.debug(f"Failed to write trace event: {e}")

    if trace_print_enabled():
        if trace_print_full_enabled():
            logger.info(json.dumps(entry, ensure_ascii=False, default=str))
        else:
            # Keep console printing short to avoid flooding stdout.
            summary_keys = [k for k in ("caller", "tool", "op", "status") if k in entry]
            summary = " ".join(f"{k}={entry[k]}" for k in summary_keys)
            logger.info(f"[TRACE] {event_type} {summary}".rstrip())
