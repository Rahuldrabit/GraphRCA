"""MCP Worker Node — Multi-Agent Mode (Phase 3).

A generic worker that executes a single MCP tool call and returns the result.
Multiple instances run in parallel (dispatched via LangGraph Send()).

Each worker receives a WorkerTask embedded in state["_worker_task"] and
appends its result to state["worker_results"].
"""

import logging
import time
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState

logger = logging.getLogger(__name__)


def mcp_worker_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node: execute one MCP tool call.

    Reads:  _worker_task (injected by planner via Send())
    Writes: worker_results (appended)
    """
    t0 = time.time()
    task = state.get("_worker_task") or {}

    mount = task.get("mount", "")
    tool = task.get("tool", "")
    arguments = task.get("arguments", {})
    rationale = task.get("rationale", "")

    logger.info(f"[MCPWorker] {mount}/{tool} — {rationale}")

    result_entry: Dict[str, Any] = {
        "mount": mount,
        "tool": tool,
        "arguments": arguments,
        "rationale": rationale,
        "result": None,
        "error": None,
        "elapsed_seconds": 0.0,
    }

    if not mount or not tool:
        result_entry["error"] = "Invalid worker task: missing mount or tool"
        logger.warning(f"[MCPWorker] Invalid task: {task}")
    else:
        try:
            from GraphRCA_agent.tools.mcp_client import call_mcp_tool

            output = call_mcp_tool(mount=mount, tool_name=tool, arguments=arguments)
            result_entry["result"] = output
            logger.info(f"[MCPWorker] {mount}/{tool} → {len(str(output))} chars")
        except Exception as e:
            result_entry["error"] = str(e)
            logger.warning(f"[MCPWorker] {mount}/{tool} failed: {e}")

    elapsed = round(time.time() - t0, 2)
    result_entry["elapsed_seconds"] = elapsed

    # Append to existing worker_results list
    existing = list(state.get("worker_results", []) or [])
    existing.append(result_entry)

    return {
        "worker_results": existing,
        "node_timings": {**state.get("node_timings", {}), f"worker_{mount}_{tool}": elapsed},
    }
