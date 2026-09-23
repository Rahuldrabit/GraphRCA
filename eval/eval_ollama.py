"""eval_ollama.py — Convenience wrapper for running GraphRCA evals with ollama.

Sets the OLLAMA_MODEL env var and delegates to eval.py so there's no
duplicated logic. All eval.py environment variables are honoured.

Usage:
    # Run all tasks with gemma4:12b  (default)
    python eval/eval_ollama.py

    # Run all tasks with deepseek-r1:8b
    python eval/eval_ollama.py --model deepseek-r1:8b

    # Run only detection tasks with gemma4:12b
    python eval/eval_ollama.py --types detection

    # Run detection + localization with deepseek, keep going after failures
    python eval/eval_ollama.py --model deepseek-r1:8b --types detection,localization --continue-on-error

    # Preserve cluster between tasks (faster, but tasks may interfere)
    python eval/eval_ollama.py --preserve

    # Disable Neo4j (faster local runs without Aura)
    python eval/eval_ollama.py --no-neo4j

    # Dry run — show what would run without executing
    python eval/eval_ollama.py --dry-run

Supported models:
    gemma4:12b        good balance of speed and instruction-following
    deepseek-r1:8b    strong for multi-step reasoning / RCA chains

Make sure ollama is running before starting:
    ollama serve
    ollama pull gemma4:12b
    ollama pull deepseek-r1:8b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_PY   = Path(__file__).resolve().parent / "eval.py"

SUPPORTED_MODELS = ["gemma4:12b", "deepseek-r1:8b"]
DEFAULT_MODEL    = "gemma4:12b"
DEFAULT_URL      = "http://localhost:11434/v1"
ALL_TASK_TYPES   = ["detection", "localization", "analysis", "mitigation"]

# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python eval/eval_ollama.py",
        description="Run GraphRCA AIOpsLab evaluation with a local ollama model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model", "-m",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        help=f"ollama model to use (default: {DEFAULT_MODEL}). "
             f"Choices: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_BASE_URL", DEFAULT_URL),
        help=f"ollama API base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--types", "-t",
        default=os.environ.get("EVAL_TASK_TYPES", ",".join(ALL_TASK_TYPES)),
        help="comma-separated task types to run "
             "(default: detection,localization,analysis,mitigation)",
    )
    parser.add_argument(
        "--preserve", "-p",
        action="store_true",
        default=_truthy(os.environ.get("EVAL_PRESERVE_CLUSTER")),
        help="reuse existing kind cluster between tasks",
    )
    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        default=_truthy(os.environ.get("EVAL_NO_NEO4J")),
        help="disable Neo4j for all tasks",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=_truthy(os.environ.get("EVAL_CONTINUE_ON_ERROR")),
        help="keep running after a task failure",
    )
    parser.add_argument(
        "--agent-mode",
        default=os.environ.get("GRAPHRCA_AGENT_MODE", ""),
        help="agent mode: pipeline | multi_agent | scratchpad_swarm",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the task list and exit without running",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="list supported ollama models and exit",
    )

    return parser.parse_args()


def _truthy(val: str | None) -> bool:
    return str(val or "").strip().lower() in {"1", "true", "yes", "y"}


# ── ollama health check ───────────────────────────────────────────────────────

def check_ollama(url: str, model: str) -> None:
    """Warn (not abort) if ollama is not reachable or model is missing."""
    import urllib.request
    import json as _json

    base = url.rstrip("/").removesuffix("/v1")
    tags_url = f"{base}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=3) as resp:
            data = _json.loads(resp.read())
            names = [m.get("name", "") for m in data.get("models", [])]
            if not any(model in n for n in names):
                print(f"[eval_ollama] WARNING: model '{model}' not found in ollama.")
                print(f"[eval_ollama]   Run:  ollama pull {model}")
    except Exception:
        print(f"[eval_ollama] WARNING: cannot reach ollama at {base}")
        print("[eval_ollama]   Start it with:  ollama serve")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if args.list_models:
        print("Supported ollama models:")
        for m in SUPPORTED_MODELS:
            marker = " *" if m == DEFAULT_MODEL else ""
            print(f"  {m}{marker}")
        return 0

    # Warn if --dry-run: just show the task list via eval.py
    if args.dry_run:
        # We print a simple task list ourselves so we don't spin up the
        # full eval machinery.
        from pathlib import Path
        import yaml

        task_types = [t.strip() for t in args.types.split(",")]
        eval_file  = REPO_ROOT / "eval" / "eval_tasks.yaml"
        cfg = yaml.safe_load(eval_file.read_text())

        print(f"\nDRY RUN — model: {args.model}")
        print(f"{'TYPE':<14}  TASK")
        print(f"{'─'*14}  {'─'*50}")
        total = 0
        for tt in task_types:
            for task in (cfg.get(tt) or []):
                print(f"{tt:<14}  {task}")
                total += 1
        print(f"\nTotal: {total} tasks")
        return 0

    # Real run
    check_ollama(args.ollama_url, args.model)

    # Forward settings via env vars — eval.py reads all of these
    os.environ["OLLAMA_MODEL"]      = args.model
    os.environ["OLLAMA_BASE_URL"]   = args.ollama_url
    os.environ["EVAL_TASK_TYPES"]   = args.types
    os.environ["EVAL_PRESERVE_CLUSTER"] = "1" if args.preserve else "0"
    os.environ["EVAL_NO_NEO4J"]     = "1" if args.no_neo4j else "0"
    os.environ["EVAL_CONTINUE_ON_ERROR"] = "1" if args.continue_on_error else "0"
    if args.agent_mode:
        os.environ["GRAPHRCA_AGENT_MODE"] = args.agent_mode

    print(f"[eval_ollama] Starting eval with model: {args.model}")
    print(f"[eval_ollama] Task types: {args.types}")
    print()

    # Run eval.py in-process by exec'ing its main()
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval", EVAL_PY)
    mod  = importlib.util.module_from_spec(spec)        # type: ignore[arg-type]
    spec.loader.exec_module(mod)                         # type: ignore[union-attr]
    mod.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
