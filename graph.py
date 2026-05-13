"""LangGraph StateGraph Orchestration.

Builds the complete GraphRCA pipeline as a LangGraph StateGraph.

Modes (controlled by GRAPHRCA_AGENT_MODE env var):
  "pipeline"    (default) — existing 11-node deterministic pipeline, zero change
  "multi_agent" — hierarchical multi-agent graph:
                    triage_agent → planner_agent → (parallel mcp_workers)
                    → aggregate_workers → [core 11-node pipeline subgraph]
                    → synthesis_agent → report_generation → END

All 4 STRATUS pillars are wired into the pipeline subgraph:
  Pillar 1 (TNR):           safety_check node + rollback edge
  Pillar 2 (Causal):        causal_ranker node
  Pillar 3 (Observability):  log_analysis node
  Pillar 4 (AIOpsLab):      run_pipeline.py integration
"""

import logging
import os
from typing import Literal

from langgraph.graph import StateGraph, END

from GraphRCA_agent.state import PipelineState
from GraphRCA_agent.nodes.trace_ingest import trace_ingest_node
from GraphRCA_agent.nodes.graph_builder import graph_builder_node
from GraphRCA_agent.nodes.detection import detection_node
from GraphRCA_agent.nodes.memory_rag import memory_search_node, memory_store_node
from GraphRCA_agent.nodes.rca import rca_node
from GraphRCA_agent.nodes.causal_ranker import causal_ranker_node
from GraphRCA_agent.nodes.log_pattern import log_pattern_node
from GraphRCA_agent.nodes.mitigation import mitigation_node
from GraphRCA_agent.nodes.undo_agent import undo_agent_node
from GraphRCA_agent.nodes.report_generation import report_generation_node

logger = logging.getLogger(__name__)


# ── Conditional Router ───────────────────────────────────────────────────────


def route_after_safety_check(
    state: PipelineState,
) -> Literal["rca", "memory_store"]:
    """Conditional edge: after TNR safety check, decide next node.

    If rollback was triggered → re-enter RCA with fresh strategy.
    Otherwise → proceed to memory_store and END.
    """
    if state.get("rollback_triggered", False):
        logger.info("[Router] Rollback triggered — re-entering RCA")
        return "rca"
    return "memory_store"


def route_on_failure(state: PipelineState) -> Literal["trace_ingest", "__end__"]:
    """Early-exit router: if status is 'failed', skip to END."""
    if state.get("status") == "failed":
        logger.error(f"[Router] Pipeline failed: {state.get('error', 'unknown')}")
        return "__end__"
    return "trace_ingest"


# ── Graph Builder ────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and compile the GraphRCA StateGraph.

    Returns:
        Compiled LangGraph CompiledGraph ready for .invoke()
    """
    workflow = StateGraph(PipelineState)

    # ── Register nodes ────────────────────────────────────────────────
    workflow.add_node("trace_ingest", trace_ingest_node)
    workflow.add_node("graph_builder", graph_builder_node)
    workflow.add_node("detection", detection_node)
    workflow.add_node("memory_search", memory_search_node)
    workflow.add_node("rca", rca_node)
    workflow.add_node("causal_ranker", causal_ranker_node)
    workflow.add_node("log_analysis", log_pattern_node)
    workflow.add_node("mitigation", mitigation_node)
    workflow.add_node("safety_check", undo_agent_node)
    workflow.add_node("memory_store", memory_store_node)
    workflow.add_node("report_generation", report_generation_node)

    # ── Linear pipeline edges ─────────────────────────────────────────
    workflow.set_entry_point("trace_ingest")
    workflow.add_edge("trace_ingest", "graph_builder")
    workflow.add_edge("graph_builder", "detection")
    workflow.add_edge("detection", "memory_search")
    workflow.add_edge("memory_search", "rca")
    workflow.add_edge("rca", "causal_ranker")
    workflow.add_edge("causal_ranker", "log_analysis")
    workflow.add_edge("log_analysis", "mitigation")
    workflow.add_edge("mitigation", "safety_check")

    # ── Conditional edge: Pillar 1 TNR rollback loop ──────────────────
    # If μ(s) regressed → loop back to rca (max 3 times, guarded in undo_agent_node)
    # Otherwise → persist incident and end
    workflow.add_conditional_edges(
        "safety_check",
        route_after_safety_check,
        {
            "rca": "rca",
            "memory_store": "memory_store",
        },
    )

    # ── Terminal edge ─────────────────────────────────────────────────
    workflow.add_edge("memory_store", "report_generation")
    workflow.add_edge("report_generation", END)

    return workflow


def build_multi_agent_graph() -> StateGraph:
    """Construct the multi-agent GraphRCA StateGraph.

    Architecture:
        triage_agent
            ↓
        planner_agent  (uses Send() to fan-out to mcp_worker nodes)
            ↓ (parallel)
        mcp_worker     (one per worker assignment)
            ↓
        aggregate_workers
            ↓
        [core pipeline: trace_ingest → ... → report_generation]
            ↓
        synthesis_agent
            ↓
        END

    Returns:
        Uncompiled StateGraph for the multi-agent mode.
    """
    from GraphRCA_agent.nodes.triage_agent import triage_agent_node
    from GraphRCA_agent.nodes.planner_agent import (
        planner_agent_node,
        aggregate_workers_node,
        dispatch_workers_router,
    )
    from GraphRCA_agent.nodes.mcp_worker import mcp_worker_node
    from GraphRCA_agent.nodes.synthesis_agent import synthesis_agent_node

    workflow = StateGraph(PipelineState)

    # ── Multi-agent wrapper nodes ─────────────────────────────────────
    workflow.add_node("triage_agent", triage_agent_node)
    workflow.add_node("planner_agent", planner_agent_node)
    workflow.add_node("mcp_worker", mcp_worker_node)
    workflow.add_node("aggregate_workers", aggregate_workers_node)

    # ── Core pipeline nodes (same 11 nodes as build_graph) ────────────
    workflow.add_node("trace_ingest", trace_ingest_node)
    workflow.add_node("graph_builder", graph_builder_node)
    workflow.add_node("detection", detection_node)
    workflow.add_node("memory_search", memory_search_node)
    workflow.add_node("rca", rca_node)
    workflow.add_node("causal_ranker", causal_ranker_node)
    workflow.add_node("log_analysis", log_pattern_node)
    workflow.add_node("mitigation", mitigation_node)
    workflow.add_node("safety_check", undo_agent_node)
    workflow.add_node("memory_store", memory_store_node)
    workflow.add_node("report_generation", report_generation_node)

    # ── Synthesis node ────────────────────────────────────────────────
    workflow.add_node("synthesis_agent", synthesis_agent_node)

    # ── Edges: multi-agent prefix ─────────────────────────────────────
    workflow.set_entry_point("triage_agent")
    workflow.add_edge("triage_agent", "planner_agent")
    # planner_agent fans out to mcp_worker nodes in parallel via Send(),
    # or skips directly to aggregate_workers if there are no assignments.
    workflow.add_conditional_edges(
        "planner_agent",
        dispatch_workers_router,
        ["mcp_worker", "aggregate_workers"],
    )
    workflow.add_edge("mcp_worker", "aggregate_workers")
    workflow.add_edge("aggregate_workers", "trace_ingest")

    # ── Core pipeline edges (identical to build_graph) ─────────────────
    workflow.add_edge("trace_ingest", "graph_builder")
    workflow.add_edge("graph_builder", "detection")
    workflow.add_edge("detection", "memory_search")
    workflow.add_edge("memory_search", "rca")
    workflow.add_edge("rca", "causal_ranker")
    workflow.add_edge("causal_ranker", "log_analysis")
    workflow.add_edge("log_analysis", "mitigation")
    workflow.add_edge("mitigation", "safety_check")

    # ── Conditional edge: Pillar 1 TNR rollback ────────────────────────
    workflow.add_conditional_edges(
        "safety_check",
        route_after_safety_check,
        {
            "rca": "rca",
            "memory_store": "memory_store",
        },
    )

    # ── Terminal: synthesis then report ───────────────────────────────
    workflow.add_edge("memory_store", "synthesis_agent")
    workflow.add_edge("synthesis_agent", "report_generation")
    workflow.add_edge("report_generation", END)

    return workflow


# ── Compiled singletons ──────────────────────────────────────────────────────

_compiled_pipeline: object = None
_compiled_multi_agent: object = None


def get_graph():
    """Return the compiled LangGraph (lazy init, singleton).

    Mode is controlled by GRAPHRCA_AGENT_MODE env var:
      "pipeline"    (default) — existing deterministic 11-node pipeline
      "multi_agent" — hierarchical multi-agent graph with MCP workers
    """
    global _compiled_pipeline, _compiled_multi_agent

    mode = os.getenv("GRAPHRCA_AGENT_MODE", "pipeline").lower().strip()

    if mode == "multi_agent":
        if _compiled_multi_agent is None:
            logger.info("Compiling GraphRCA Multi-Agent StateGraph...")
            _compiled_multi_agent = build_multi_agent_graph().compile()
            logger.info("GraphRCA Multi-Agent StateGraph compiled successfully")
        return _compiled_multi_agent

    # Default: pipeline mode
    if _compiled_pipeline is None:
        logger.info("Compiling GraphRCA Pipeline StateGraph...")
        _compiled_pipeline = build_graph().compile()
        logger.info("GraphRCA Pipeline StateGraph compiled successfully")
    return _compiled_pipeline
