"""LangGraph StateGraph Orchestration.

Builds the complete GraphRCA pipeline as a LangGraph StateGraph:

  trace_ingest → graph_builder → detection → memory_search
    → rca → causal_ranker → log_analysis → mitigation → safety_check
    → (rollback loop → rca) OR memory_store → END

All 4 STRATUS pillars are wired into this graph:
  Pillar 1 (TNR):         safety_check node + rollback edge
  Pillar 2 (Causal):      causal_ranker node
  Pillar 3 (Observability): log_analysis node
  Pillar 4 (AIOpsLab):    run_pipeline.py integration
"""

import logging
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


# ── Compiled singleton ───────────────────────────────────────────────────────

_compiled_graph = None


def get_graph():
    """Return the compiled LangGraph (lazy init, singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Compiling GraphRCA StateGraph...")
        _compiled_graph = build_graph().compile()
        logger.info("GraphRCA StateGraph compiled successfully")
    return _compiled_graph
