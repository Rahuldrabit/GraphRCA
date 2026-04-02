"""Safety Tools — STRATUS Pillar 1: Transactional No-Regression (TNR).

Implements:
  - μ(s) weighted system health score
  - UndoStack: stores (action_cmd, revert_cmd) pairs
  - generate_revert_command: auto-generates kubectl revert
  - rollback_all: executes full rollback via exec_shell
"""

import logging
import re
import subprocess
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Health Score ────────────────────────────────────────────────────────────


def compute_health_score(
    alerts: List[Any],
    sla_violations: List[Any],
    unhealthy_nodes: List[Any],
    w1: float = 0.40,
    w2: float = 0.35,
    w3: float = 0.25,
) -> float:
    """Compute weighted system health score μ(s).

    μ(s) = w1·|A| + w2·|V| + w3·|L|

    Where:
      A = set of active alerts
      V = set of SLA violations
      L = capacity loss / unhealthy nodes

    Lower score = healthier system.
    A mitigation is safe if μ(s_after) <= μ(s_before).

    Args:
        alerts: List of AlertSignal objects or dicts
        sla_violations: List of SLA violation records
        unhealthy_nodes: List of unhealthy node names
        w1: Weight for alerts (default 0.40)
        w2: Weight for SLA violations (default 0.35)
        w3: Weight for capacity loss (default 0.25)

    Returns:
        Composite health score (lower = healthier)
    """
    score = w1 * len(alerts) + w2 * len(sla_violations) + w3 * len(unhealthy_nodes)
    logger.info(
        f"Health score μ(s) = {score:.4f} "
        f"[alerts={len(alerts)}, sla_v={len(sla_violations)}, unhealthy={len(unhealthy_nodes)}]"
    )
    return round(score, 4)


def health_regressed(before: float, after: float, tolerance: float = 0.05) -> bool:
    """Check if system health got worse after a mitigation.

    Args:
        before: Health score before mitigation
        after: Health score after mitigation
        tolerance: Acceptable increase (default 5%)

    Returns:
        True if health degraded beyond tolerance
    """
    regressed = after > before + tolerance
    if regressed:
        logger.warning(
            f"Health regression detected: μ(s) {before:.4f} → {after:.4f} "
            f"(Δ={after - before:+.4f}, tolerance={tolerance})"
        )
    else:
        logger.info(f"Health OK: μ(s) {before:.4f} → {after:.4f}")
    return regressed


# ── Undo Stack ──────────────────────────────────────────────────────────────


@dataclass
class UndoEntry:
    """One entry on the undo stack."""
    action_cmd: str
    revert_cmd: str
    service: str
    description: str
    executed: bool = False
    reverted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UndoStack:
    """Thread-safe stack of (action_cmd, revert_cmd) pairs.

    Every state-changing command pushed here gets a revert.
    Rollback pops and executes in LIFO order.
    """

    def __init__(self):
        self._stack: List[UndoEntry] = []
        self._lock = threading.Lock()

    def push(self, entry: UndoEntry):
        """Push a new action + revert pair."""
        with self._lock:
            self._stack.append(entry)
        logger.info(f"[UndoStack] Pushed: {entry.description} | revert: {entry.revert_cmd[:80]}")

    def pop(self) -> Optional[UndoEntry]:
        """Pop the last entry without executing."""
        with self._lock:
            return self._stack.pop() if self._stack else None

    def rollback_all(self, dry_run: bool = False) -> List[Dict[str, Any]]:
        """Execute all revert commands in LIFO order.

        Args:
            dry_run: If True, log but don't execute

        Returns:
            List of rollback results
        """
        results = []
        logger.warning(f"[UndoStack] Rolling back {len(self._stack)} actions...")

        with self._lock:
            entries = list(reversed(self._stack))

        for entry in entries:
            if entry.reverted:
                continue
            result = {"cmd": entry.revert_cmd, "service": entry.service, "success": False}
            if dry_run:
                logger.info(f"[DRY RUN] Would execute: {entry.revert_cmd}")
                result["success"] = True
                result["dry_run"] = True
            else:
                success, output = _exec(entry.revert_cmd)
                result["success"] = success
                result["output"] = output[:500]
                if success:
                    entry.reverted = True
                    logger.info(f"[UndoStack] Reverted: {entry.description}")
                else:
                    logger.error(f"[UndoStack] Revert FAILED: {entry.description}\n{output}")
            results.append(result)

        return results

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize stack to list of dicts (for LangGraph state)."""
        with self._lock:
            return [e.to_dict() for e in self._stack]

    def __len__(self) -> int:
        with self._lock:
            return len(self._stack)


# ── Revert Command Generator ────────────────────────────────────────────────


def generate_revert_command(action_cmd: str) -> str:
    """Auto-generate a revert command for common kubectl/helm actions.

    Heuristic rules cover the most common mitigation commands.
    Falls back to a safe no-op comment if no rule matches.

    Args:
        action_cmd: The original mitigation command

    Returns:
        Revert command string
    """
    cmd = action_cmd.strip()

    # kubectl scale → scale back to 2 (conservative)
    m = re.match(r"kubectl scale deployment/(\S+) --replicas=(\d+)", cmd)
    if m:
        svc, new_replicas = m.group(1), int(m.group(2))
        orig = max(1, new_replicas - 2)
        return f"kubectl scale deployment/{svc} --replicas={orig}"

    # kubectl rollout restart → rollout undo
    m = re.match(r"kubectl rollout restart deployment/(\S+)", cmd)
    if m:
        svc = m.group(1)
        return f"kubectl rollout undo deployment/{svc}"

    # kubectl patch configmap → hard to auto-revert, snapshot approach
    m = re.match(r"kubectl patch configmap (\S+)", cmd)
    if m:
        cm = m.group(1)
        return f"# Manual revert required for configmap {cm} — restore from backup"

    # kubectl set env → remove added envs
    m = re.match(r"kubectl set env deployment/(\S+) (.+)", cmd)
    if m:
        svc = m.group(1)
        env_pairs = m.group(2)
        keys = re.findall(r"(\w+)=\S+", env_pairs)
        remove_args = " ".join(f"{k}-" for k in keys)
        return f"kubectl set env deployment/{svc} {remove_args}"

    # helm upgrade → helm rollback
    m = re.match(r"helm upgrade (\S+) (\S+)", cmd)
    if m:
        release = m.group(1)
        return f"helm rollback {release}"

    # kubectl delete pods → restart is already done; undo is a no-op
    m = re.match(r"kubectl delete pods", cmd)
    if m:
        return "# Pod deletion is self-healing; verify with: kubectl get pods -w"

    # kubectl apply (e.g., alert rules) → kubectl delete
    m = re.match(r"kubectl apply -f (.+)", cmd)
    if m:
        manifest = m.group(1)
        return f"kubectl delete -f {manifest}"

    # Unknown — safe no-op
    logger.warning(f"No revert rule for: {cmd[:80]}")
    return f"# No automatic revert for: {cmd[:80]}"


# ── Shell Executor (minimal, used by rollback) ──────────────────────────────


def _exec(cmd: str, timeout: int = 60) -> Tuple[bool, str]:
    """Execute a shell command synchronously.

    Args:
        cmd: Shell command string
        timeout: Timeout in seconds

    Returns:
        (success, combined_output)
    """
    if cmd.startswith("#"):
        return True, f"[SKIP] {cmd}"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"
    except Exception as e:
        return False, str(e)


# ── Convenience: Build UndoEntry from MitigationAction ─────────────────────


def build_undo_entry(action) -> UndoEntry:
    """Create an UndoEntry from a MitigationAction (dict or dataclass).

    Args:
        action: MitigationAction dict or object

    Returns:
        UndoEntry
    """
    if isinstance(action, dict):
        cmd = action.get("command", "")
        svc = action.get("service", "")
        desc = action.get("title", "")
    else:
        cmd = getattr(action, "command", "")
        svc = getattr(action, "service", "")
        desc = getattr(action, "title", "")

    revert = generate_revert_command(cmd)
    return UndoEntry(
        action_cmd=cmd,
        revert_cmd=revert,
        service=svc,
        description=desc,
    )
