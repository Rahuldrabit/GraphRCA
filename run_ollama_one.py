#!/usr/bin/env python3
"""run_ollama_one.py — Pure-Python single-task runner for GraphRCA with ollama.

A self-contained alternative to run_single_task_ollama.sh for environments
where running a bash script is inconvenient (Windows WSL, notebooks, CI).
It injects ollama env vars directly into os.environ, then calls
run_pipeline.run_aiopslab() in-process — no subprocess, no shell required.

Usage:
    python run_ollama_one.py <task_name>
    python run_ollama_one.py <task_name> --model deepseek-r1:8b
    python run_ollama_one.py <task_name> --model gemma4:12b --verbose
    python run_ollama_one.py <task_name> --no-neo4j --mode scratchpad_swarm
    python run_ollama_one.py --list-tasks
    python run_ollama_one.py --list-tasks --types detection

Supported models:
    gemma4:12b        (default — good all-round)
    deepseek-r1:8b    (stronger multi-step reasoning)

Prerequisites:
    ollama serve
    ollama pull gemma4:12b
    ollama pull deepseek-r1:8b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Repo path bootstrap ───────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parent
AIOPSLAB_DIR = REPO_ROOT / "AIOpsLab"
SCRATCHPAD_DIR = REPO_ROOT / "ScratchPad"
EVAL_TASKS_FILE = REPO_ROOT / "eval" / "eval_tasks.yaml"

for p in (str(REPO_ROOT), str(AIOPSLAB_DIR), str(SCRATCHPAD_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ["gemma4:12b", "deepseek-r1:8b"]
DEFAULT_MODEL    = "gemma4:12b"
DEFAULT_URL      = "http://localhost:11434/v1"
TASK_TYPES       = ["detection", "localization", "analysis", "mitigation"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _setup_basic_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)-22s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def _inject_ollama_env(model: str, base_url: str, no_neo4j: bool, agent_mode: str) -> None:
    """Override LLM-related env vars in-process so llm.py picks up ollama."""
    os.environ["PROVIDER_AGENTS"]  = "openai"
    os.environ["MODEL_AGENTS"]     = model
    os.environ["URL_AGENTS"]       = base_url
    os.environ["API_KEY_AGENTS"]   = "ollama"   # ollama ignores the value
    os.environ["OPENAI_API_KEY"]   = "ollama"   # LangChain / CrewAI fallback

    os.environ["PROVIDER_TOOLS"]   = "openai"
    os.environ["MODEL_TOOLS"]      = model
    os.environ["URL_TOOLS"]        = base_url
    os.environ["API_KEY_TOOLS"]    = "ollama"

    if no_neo4j:
        os.environ["NEO4J_ENABLED"] = "False"

    if agent_mode:
        os.environ["GRAPHRCA_AGENT_MODE"] = agent_mode

    # Disable the interactive KG-mode prompt (not suitable for scripted runs)
    os.environ.setdefault("GRAPHRCA_LLM_KG_MODE", "")


def _check_ollama(base_url: str, model: str) -> bool:
    """Return True if ollama is reachable; print warnings otherwise."""
    base = base_url.rstrip("/").removesuffix("/v1")
    tags_url = f"{base}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=4) as resp:
            data = json.loads(resp.read())
            names = [m.get("name", "") for m in data.get("models", [])]
            if not any(model in n for n in names):
                print(f"[run_ollama_one] WARNING: model '{model}' not found in ollama.")
                print(f"[run_ollama_one]   Run:  ollama pull {model}")
            return True
    except Exception:
        print(f"[run_ollama_one] WARNING: ollama not reachable at {base}")
        print("[run_ollama_one]   Start it with:  ollama serve")
        print(f"[run_ollama_one]   Pull model:     ollama pull {model}")
        return False


def _list_tasks(types: list[str]) -> None:
    """Print all tasks from eval_tasks.yaml for the given types."""
    import yaml

    if not EVAL_TASKS_FILE.exists():
        print(f"[run_ollama_one] eval_tasks.yaml not found at {EVAL_TASKS_FILE}")
        return

    cfg = yaml.safe_load(EVAL_TASKS_FILE.read_text())
    total = 0
    print(f"\n{'TYPE':<14}  TASK")
    print(f"{'─'*14}  {'─'*54}")
    for tt in types:
        for task in (cfg.get(tt) or []):
            print(f"{tt:<14}  {task}")
            total += 1
    print(f"\nTotal: {total} tasks across {', '.join(types)}")


def _build_output_dir(task_name: str, model: str) -> Path:
    model_slug = model.replace(":", "-")
    ts = datetime.now().strftime("%m-%d_%H-%M-%S")
    out = REPO_ROOT / "eval" / f"{ts}-{task_name}-{model_slug}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "graphrca_output").mkdir(exist_ok=True)
    return out


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python run_ollama_one.py",
        description="Run a single GraphRCA AIOpsLab task with a local ollama model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "task_name",
        nargs="?",
        help="AIOpsLab task ID (e.g. misconfig_app_hotel_res-detection-1)",
    )
    p.add_argument(
        "--model", "-m",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        help=f"ollama model (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_BASE_URL", DEFAULT_URL),
        help=f"ollama API base URL (default: {DEFAULT_URL})",
    )
    p.add_argument(
        "--mode",
        default=os.environ.get("GRAPHRCA_AGENT_MODE", ""),
        help="agent mode: pipeline | multi_agent | scratchpad_swarm",
    )
    p.add_argument(
        "--no-neo4j",
        action="store_true",
        default=os.environ.get("NEO4J_ENABLED", "True").lower() == "false",
        help="disable Neo4j (useful for local runs without Aura)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="enable debug logging",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=None,
        help="custom output directory (default: eval/<timestamp>-<task>-<model>/)",
    )
    p.add_argument(
        "--list-tasks",
        action="store_true",
        help="list all tasks from eval_tasks.yaml and exit",
    )
    p.add_argument(
        "--types",
        default=",".join(TASK_TYPES),
        help="comma-separated task types for --list-tasks "
             "(default: detection,localization,analysis,mitigation)",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="list supported models and exit",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── Informational exits ────────────────────────────────────────────────────
    if args.list_models:
        print("Supported ollama models:")
        for m in SUPPORTED_MODELS:
            tag = "  (default)" if m == DEFAULT_MODEL else ""
            print(f"  {m}{tag}")
        return 0

    if args.list_tasks:
        types = [t.strip() for t in args.types.split(",")]
        _list_tasks(types)
        return 0

    # ── Validate task name ─────────────────────────────────────────────────────
    if not args.task_name:
        print("[run_ollama_one] ERROR: task_name is required.\n")
        print("Usage: python run_ollama_one.py <task_name> [--model <model>]")
        print()
        print("Example tasks:")
        print("  misconfig_app_hotel_res-detection-1")
        print("  k8s_target_port-misconfig-localization-1")
        print("  auth_miss_mongodb-mitigation-1")
        print("  container_kill-detection")
        print()
        print("Use --list-tasks to see all available tasks.")
        return 1

    task_name = args.task_name
    model     = args.model
    base_url  = args.ollama_url

    # ── Logging ────────────────────────────────────────────────────────────────
    _setup_basic_logging(args.verbose)
    logger = logging.getLogger("run_ollama_one")

    # ── Banner ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  GraphRCA — Single Task Runner (ollama, in-process)")
    print("=" * 66)
    print(f"  Task:       {task_name}")
    print(f"  Model:      {model}")
    print(f"  Ollama URL: {base_url}")
    print(f"  Agent mode: {args.mode or '(from .env)'}")
    print(f"  Neo4j:      {'disabled' if args.no_neo4j else 'enabled'}")
    print("=" * 66)
    print()

    # ── Check ollama ───────────────────────────────────────────────────────────
    _check_ollama(base_url, model)

    # ── Inject env vars BEFORE any imports that read them ─────────────────────
    # Load .env first so we don't clobber user's secrets
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    _inject_ollama_env(
        model=model,
        base_url=base_url,
        no_neo4j=args.no_neo4j,
        agent_mode=args.mode,
    )

    # ── Output directory ───────────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "graphrca_output").mkdir(exist_ok=True)
    else:
        output_dir = _build_output_dir(task_name, model)

    graphrca_output_dir = str(output_dir / "graphrca_output")

    # Propagate output dir via env (run_aiopslab respects these)
    os.environ["TASK_NAME"]           = task_name
    os.environ["OUTPUT_DIRECTORY"]    = graphrca_output_dir

    logger.info(f"Output directory: {output_dir}")

    # ── Run pipeline ───────────────────────────────────────────────────────────
    # Import AFTER env injection so LLM singletons pick up the ollama settings
    from run_pipeline import run_aiopslab

    t_start = time.time()
    try:
        result = run_aiopslab(
            problem_id=task_name,
            output_dir=graphrca_output_dir,
            verbose=args.verbose,
        )
        elapsed = round(time.time() - t_start, 2)
        success = True
    except Exception as exc:
        elapsed = round(time.time() - t_start, 2)
        logger.exception(f"run_aiopslab raised: {exc}")
        result  = {"error": str(exc), "problem_id": task_name}
        success = False

    # ── Write run metadata ─────────────────────────────────────────────────────
    meta = {
        "task_name":       task_name,
        "model":           model,
        "ollama_url":      base_url,
        "agent_mode":      os.environ.get("GRAPHRCA_AGENT_MODE", ""),
        "neo4j_enabled":   os.environ.get("NEO4J_ENABLED", ""),
        "elapsed_seconds": elapsed,
        "success":         success,
        "started_at":      datetime.fromtimestamp(t_start).isoformat(),
        "completed_at":    datetime.now().isoformat(),
        "result_summary":  {
            k: result.get(k)
            for k in ("problem_id", "task_type", "total_elapsed_seconds", "error")
        },
    }
    meta_path = output_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    # ── Final summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 66)
    if success:
        print(f"  DONE  {task_name}")
        print(f"  Model:    {model}")
        print(f"  Elapsed:  {elapsed}s")
        print(f"  Output:   {output_dir}")
        print(f"    graphrca_output/incident_report.json   — RCA report")
        print(f"    graphrca_output/llm_justification.jsonl — LLM call log")
        print(f"    graphrca_output/graphrca_run_stats.json — token usage")
        print(f"    run_meta.json                           — this run's metadata")
    else:
        print(f"  FAILED  {task_name}")
        print(f"  Error:    {result.get('error', 'unknown')}")
        print(f"  Output:   {output_dir}")
    print("=" * 66)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
