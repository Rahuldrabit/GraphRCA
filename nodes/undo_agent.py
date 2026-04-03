"""Undo Agent Node — LangGraph agent node (STRATUS Pillar 1: TNR).

Safety oracle that validates mitigation did not make the system worse.
Computes post-mitigation health score μ(s) and triggers automatic
rollback via UndoStack if health regresses.

Implements the Transactional No-Regression (TNR) contract:
  Every mitigation is a transaction. If μ(s)_after > μ(s)_before,
  the transaction is rolled back.
"""

import logging
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState
from GraphRCA_agent.tools.safety_tools import (
    compute_health_score,
    health_regressed,
    UndoStack,
    UndoEntry,
)

logger = logging.getLogger(__name__)

# Maximum rollback cycles to prevent infinite loops
MAX_ROLLBACK_CYCLES = 3


def undo_agent_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: TNR safety oracle — compute μ(s) and rollback if needed.

    Reads:  alerts, sla_violations, unhealthy_nodes,
            health_score_before, undo_stack, rollback_count
    Writes: health_score_after, rollback_triggered, rollback_count, status
    """
    t0 = time.time()
    alerts = state.get("alerts", [])
    sla_violations = state.get("sla_violations", [])
    unhealthy_nodes = state.get("unhealthy_nodes", [])
    health_before = state.get("health_score_before", 0.0)
    undo_stack_data = state.get("undo_stack", [])
    rollback_count = state.get("rollback_count", 0)

    logger.info(
        f"[UndoAgent] TNR check | μ(s)_before={health_before:.4f} | "
        f"undo_stack_depth={len(undo_stack_data)} | rollbacks_so_far={rollback_count}"
    )

    try:
        # 1. Compute post-mitigation health score
        health_after = compute_health_score(alerts, sla_violations, unhealthy_nodes)

        # 2. Check for health regression
        regressed = health_regressed(health_before, health_after, tolerance=0.05)

        if not regressed:
            elapsed = round(time.time() - t0, 2)
            logger.info(
                f"[UndoAgent] System healthy — no rollback needed. "
                f"μ(s): {health_before:.4f} → {health_after:.4f}"
            )
            return {
                "health_score_after": health_after,
                "rollback_triggered": False,
                "status": "running",
                "messages": state.get("messages", []) + [
                    f"[UndoAgent] ✅ TNR PASS: μ(s) {health_before:.4f}→{health_after:.4f}"
                ],
                "node_timings": {**state.get("node_timings", {}), "undo_agent": elapsed},
            }

        # 3. Health regressed — execute rollback
        if rollback_count >= MAX_ROLLBACK_CYCLES:
            logger.error(
                f"[UndoAgent] Max rollback cycles ({MAX_ROLLBACK_CYCLES}) reached! "
                "Manual intervention required."
            )
            return {
                "health_score_after": health_after,
                "rollback_triggered": False,  # Stop the loop
                "status": "complete",  # Surface to user
                "messages": state.get("messages", []) + [
                    f"[UndoAgent] ⚠️ MAX ROLLBACKS REACHED — manual intervention needed. "
                    f"μ(s): {health_before:.4f}→{health_after:.4f}"
                ],
                "node_timings": {**state.get("node_timings", {}), "undo_agent": round(time.time() - t0, 2)},
            }

        # 4. Rebuild UndoStack and execute rollback
        logger.warning(
            f"[UndoAgent] ❌ TNR FAIL: μ(s) {health_before:.4f}→{health_after:.4f}. "
            f"Executing rollback (cycle {rollback_count + 1}/{MAX_ROLLBACK_CYCLES})"
        )

        undo_stack = UndoStack()
        for entry_data in undo_stack_data:
            entry = UndoEntry(
                action_cmd=entry_data.get("action_cmd", ""),
                revert_cmd=entry_data.get("revert_cmd", ""),
                service=entry_data.get("service", ""),
                description=entry_data.get("description", ""),
                executed=entry_data.get("executed", False),
                reverted=entry_data.get("reverted", False),
            )
            undo_stack._stack.append(entry)

        # Execute rollback (dry_run=True when no real cluster)
        rollback_results = undo_stack.rollback_all(dry_run=False)
        success_count = sum(1 for r in rollback_results if r.get("success", False))

        elapsed = round(time.time() - t0, 2)
        logger.info(
            f"[UndoAgent] Rollback complete: {success_count}/{len(rollback_results)} "
            f"reverts succeeded in {elapsed}s"
        )

        return {
            "health_score_after": health_after,
            "rollback_triggered": True,
            "rollback_count": rollback_count + 1,
            # Clear undo stack after rollback
            "undo_stack": [],
            "status": "running",
            "messages": state.get("messages", []) + [
                f"[UndoAgent] 🔄 ROLLBACK #{rollback_count + 1}: "
                f"{success_count}/{len(rollback_results)} reverts OK. "
                f"Re-entering RCA with new strategy."
            ],
            "node_timings": {**state.get("node_timings", {}), "undo_agent": elapsed},
        }

    except Exception as e:
        logger.exception(f"[UndoAgent] Failed: {e}")
        return {
            "health_score_after": health_before,
            "rollback_triggered": False,
            "status": "complete",
            "messages": state.get("messages", []) + [f"[UndoAgent] ERROR: {e}"],
        }
