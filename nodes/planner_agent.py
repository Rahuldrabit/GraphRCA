"""Planner Agent Node — Multi-Agent Mode (Phase 3).

Reads the investigation_strategy produced by the Triage Agent and
dispatches parallel MCP worker calls using LangGraph's Send() API.

Reads:  investigation_strategy
Writes: messages (node); routes to mcp_worker or aggregate_workers (router)
"""

import logging
import time
from typing import Any, Dict, List, Union

from langgraph.types import Send

from GraphRCA_agent.state import PipelineState

logger = logging.getLogger(__name__)


def planner_agent_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: log dispatch intent.

    Routing to mcp_worker (fan-out) is handled by dispatch_workers_router,
    which is registered as a conditional edge on this node in graph.py.
    """
    strategy = state.get("investigation_strategy") or {}
    assignments = strategy.get("worker_assignments", [])
    valid = [t for t in assignments if t.get("mount") and t.get("tool")]

    logger.info(f"[PlannerAgent] Preparing to dispatch {len(valid)} MCP workers")

    return {
        "messages": state.get("messages", []) + [
            f"[Planner] Dispatching {len(valid)} workers"
        ],
    }


def dispatch_workers_router(state: PipelineState) -> Union[List[Send], str]:
    """Conditional edge router: fan out to mcp_worker nodes via Send().

    Returns a list of Send() objects (one per valid worker assignment) so
    LangGraph executes them in parallel, or "aggregate_workers" directly
    if there are no valid assignments.
    """
    strategy = state.get("investigation_strategy") or {}
    assignments = strategy.get("worker_assignments", [])

    sends: List[Send] = []
    for task in assignments:
        mount = task.get("mount", "")
        tool = task.get("tool", "")
        if not mount or not tool:
            logger.warning(f"[PlannerAgent] Skipping invalid task: {task}")
            continue
        worker_state = {**dict(state), "_worker_task": task}
        sends.append(Send("mcp_worker", worker_state))
        logger.info(f"[PlannerAgent] Queued: {mount}/{tool}")

    if not sends:
        logger.info("[PlannerAgent] No workers to dispatch — routing to aggregate_workers")
        return "aggregate_workers"

    logger.info(f"[PlannerAgent] Fanning out to {len(sends)} workers")
    return sends


def aggregate_workers_node(state: PipelineState) -> Dict[str, Any]:
    """Aggregation node: collect worker results before core pipeline runs.

    This node is a no-op aggregation point — worker results are already
    accumulated in state["worker_results"] by the mcp_worker nodes.
    """
    worker_results = state.get("worker_results", []) or []
    succeeded = sum(1 for r in worker_results if r.get("result") is not None)
    failed = sum(1 for r in worker_results if r.get("error") is not None)

    logger.info(
        f"[AggregateWorkers] {len(worker_results)} workers done: "
        f"{succeeded} succeeded, {failed} failed"
    )

    return {
        "messages": state.get("messages", []) + [
            f"[Planner] {succeeded}/{len(worker_results)} MCP queries succeeded"
        ],
    }
