"""GraphRCA Evaluation Runner — batch-run all AIOpsLab tasks.

Mirrors stratus/eval/eval.py: reads eval_tasks.yaml, iterates through
each task type, and runs run_graphrca.sh (or run_single_task_ollama.sh
when an ollama model is requested) for each task.

Usage:
    # Original behaviour — uses run_graphrca.sh (OpenAI / .env config)
    python eval/eval.py

    # Ollama override — point LLM at a local ollama model
    OLLAMA_MODEL=gemma4:12b     python eval/eval.py
    OLLAMA_MODEL=deepseek-r1:8b python eval/eval.py

    # Run only specific task types
    EVAL_TASK_TYPES=detection,localization python eval/eval.py

    # Keep going after a failure instead of stopping
    EVAL_CONTINUE_ON_ERROR=1 python eval/eval.py

Environment variables (all optional):
    OLLAMA_MODEL          — if set, routes LLM calls to ollama instead of
                            the provider configured in .env
                            e.g. gemma4:12b  or  deepseek-r1:8b
    OLLAMA_BASE_URL       — ollama base URL (default: http://localhost:11434/v1)
    EVAL_TASK_TYPES       — comma-separated subset of task types to run
                            (default: detection,localization,analysis,mitigation)
    EVAL_CONTINUE_ON_ERROR— set to "1" or "true" to keep running after failures
    EVAL_PRESERVE_CLUSTER — set to "1" to reuse existing kind cluster
    EVAL_NO_NEO4J         — set to "1" to disable Neo4j for all tasks
    GRAPHRCA_AGENT_MODE   — pipeline | multi_agent | scratchpad_swarm
"""

import json
import os
import platform
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

# ── Helpers ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_TASKS_FILE = REPO_ROOT / "eval" / "eval_tasks.yaml"
SINGLE_TASK_SCRIPT = REPO_ROOT / "run_single_task_ollama.sh"
GRAPHRCA_SCRIPT = REPO_ROOT / "run_graphrca.sh"


def _truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in {"1", "true", "yes", "y"}


def _detect_arch() -> str:
    uname = platform.processor()
    machine = platform.machine()
    if machine in ("aarch64", "arm64") or "arm" in machine.lower():
        return "arm"
    return "x86"


def load_tasks(task_types: list[str]) -> list[tuple[str, str]]:
    """Return a flat list of (task_type, task_name) pairs."""
    with open(EVAL_TASKS_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    result = []
    for task_type in task_types:
        tasks = cfg.get(task_type) or []
        if not tasks:
            print(f"[EVAL] no {task_type} tasks found — skipping")
        for t in tasks:
            result.append((task_type, t))
    return result


def build_env(ollama_model: Optional[str], ollama_url: str) -> dict:
    """Build an os.environ copy with ollama overrides applied (if requested)."""
    env = os.environ.copy()
    if ollama_model:
        env["PROVIDER_AGENTS"] = "openai"
        env["MODEL_AGENTS"]    = ollama_model
        env["URL_AGENTS"]      = ollama_url
        env["API_KEY_AGENTS"]  = "ollama"
        env["OPENAI_API_KEY"]  = "ollama"
        env["PROVIDER_TOOLS"]  = "openai"
        env["MODEL_TOOLS"]     = ollama_model
        env["URL_TOOLS"]       = ollama_url
        env["API_KEY_TOOLS"]   = "ollama"
    return env


def run_task(
    task_type: str,
    task_name: str,
    arch: str,
    ollama_model: Optional[str],
    ollama_url: str,
    preserve_cluster: bool,
    no_neo4j: bool,
) -> subprocess.CompletedProcess:
    """Run a single task via the appropriate shell script."""
    if ollama_model:
        # Use the dedicated ollama script which injects env vars cleanly
        cmd = ["/usr/bin/env", "bash", str(SINGLE_TASK_SCRIPT), task_name,
               "--model", ollama_model]
        if preserve_cluster:
            cmd.append("--preserve")
        if no_neo4j:
            cmd.append("--no-neo4j")
    else:
        # Fall back to original run_graphrca.sh behaviour
        cmd = ["/usr/bin/env", "bash", str(GRAPHRCA_SCRIPT),
               "-r", arch, task_name]
        if preserve_cluster:
            cmd.append("--preserve")
        if no_neo4j:
            cmd.append("--no-neo4j")

    env = build_env(ollama_model, ollama_url)

    # Hard per-task cap: a single hung call (ollama stall, or a deploy that
    # never converges) must NOT block the whole batch. Launch the task as its
    # OWN session/process-group and, on timeout, kill the ENTIRE group — bash +
    # kind + docker + python + helm/kubectl — so no orphaned processes leak into
    # the next task (which would corrupt its `kind delete/create` reset). The
    # next task's reset wipes any leftover cluster state regardless.
    try:
        cap = int(os.environ.get("EVAL_TASK_TIMEOUT", "2400"))  # 40 min default
    except ValueError:
        cap = 2400
    try:
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    except Exception as e:
        print(f"[EVAL] failed to launch task '{task_name}': {e}")
        return subprocess.CompletedProcess(cmd, 1)
    try:
        rc = proc.wait(timeout=cap)
        return subprocess.CompletedProcess(cmd, rc)
    except subprocess.TimeoutExpired:
        print(
            f"[EVAL] task '{task_name}' exceeded {cap}s hard cap — "
            f"killing process group, marking failed, continuing"
        )
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        return subprocess.CompletedProcess(cmd, 124)  # 124 = conventional timeout


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Read config from environment ──────────────────────────────────────────
    ollama_model: Optional[str] = os.environ.get("OLLAMA_MODEL") or None
    ollama_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    preserve_cluster = _truthy(os.environ.get("EVAL_PRESERVE_CLUSTER"))
    no_neo4j         = _truthy(os.environ.get("EVAL_NO_NEO4J"))
    continue_on_err  = _truthy(os.environ.get("EVAL_CONTINUE_ON_ERROR"))

    raw_types = os.environ.get("EVAL_TASK_TYPES", "detection,localization,analysis,mitigation")
    task_types = [t.strip() for t in raw_types.split(",") if t.strip()]

    arch = _detect_arch()

    # ── Load task list ─────────────────────────────────────────────────────────
    all_tasks = load_tasks(task_types)
    total = len(all_tasks)

    if total == 0:
        print("[EVAL] No tasks to run. Check eval_tasks.yaml and EVAL_TASK_TYPES.")
        sys.exit(1)

    # ── Banner ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  GraphRCA Eval Runner")
    print("=" * 66)
    print(f"  Model:        {ollama_model or '(from .env)'}")
    if ollama_model:
        print(f"  Ollama URL:   {ollama_url}")
    print(f"  Task types:   {', '.join(task_types)}")
    print(f"  Total tasks:  {total}")
    print(f"  Architecture: {arch}")
    print(f"  Continue on error: {continue_on_err}")
    print("=" * 66)
    print()

    # ── Run loop ───────────────────────────────────────────────────────────────
    results = []
    passed = 0
    failed = 0

    for idx, (task_type, task_name) in enumerate(all_tasks, start=1):
        print(f"\n[EVAL] ({idx}/{total}) [{task_type}] {task_name}")
        t_start = datetime.now(timezone.utc)

        ret = run_task(
            task_type=task_type,
            task_name=task_name,
            arch=arch,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
            preserve_cluster=preserve_cluster,
            no_neo4j=no_neo4j,
        )

        elapsed = round((datetime.now(timezone.utc) - t_start).total_seconds(), 1)
        status = "passed" if ret.returncode == 0 else "failed"

        print(
            f"[EVAL] {task_type} task '{task_name}' → {status.upper()} "
            f"(exit {ret.returncode}, {elapsed}s)"
        )

        results.append({
            "task":            task_name,
            "type":            task_type,
            "status":          status,
            "exit_code":       ret.returncode,
            "elapsed_seconds": elapsed,
        })

        if ret.returncode == 0:
            passed += 1
        else:
            failed += 1
            if not continue_on_err:
                print("[EVAL] Stopping on first failure. Set EVAL_CONTINUE_ON_ERROR=1 to keep going.")
                break

    # ── Summary ────────────────────────────────────────────────────────────────
    skipped = total - passed - failed
    summary = {
        "completed_at":   datetime.now(timezone.utc).isoformat(),
        "model":          ollama_model or os.environ.get("MODEL_AGENTS", "unknown"),
        "task_types":     task_types,
        "total":          total,
        "passed":         passed,
        "failed":         failed,
        "skipped":        skipped,
        "results":        results,
    }

    summary_path = REPO_ROOT / "eval" / f"eval_summary_{datetime.now().strftime('%m-%d_%H-%M-%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 66)
    print("  EVAL COMPLETE")
    print("=" * 66)
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Summary: {summary_path}")
    print("=" * 66)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
