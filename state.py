"""LangGraph Shared State Schema.

Central TypedDict that flows through all LangGraph nodes.
Each node reads what it needs and returns a partial update.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    """Shared state for the GraphRCA LangGraph pipeline.

    All fields are optional (total=False) so nodes can return
    partial updates without specifying every field.
    """

    # ── Input ──────────────────────────────────────────────────────────
    trace_dir: str
    use_neo4j: bool
    neo4j_connector: Any
    store_spans: bool
    additional_context: str         # Reflection text from previous failed runs (VALIDATION_RETRY)
    llm_kg_mode: str                # "a"|"b"|"" (experimental KG access for LLM)

    # ── Trace Ingest ───────────────────────────────────────────────────
    spans: list                     # List[Span]
    service_stats: dict             # {service_name: stats_dict}
    ingest_summary: dict            # Dedup stats, validation report

    # ── Graph Builder ──────────────────────────────────────────────────
    graph: Any                      # nx.DiGraph
    pagerank: dict                  # {service_name: score}
    graph_summary: dict

    # ── Detection (EWMA) ──────────────────────────────────────────────
    alerts: list                    # List[AlertSignal]
    baselines: dict                 # {service_name: EWMABaseline}
    primary_error_service: str

    # ── Memory Search ─────────────────────────────────────────────────
    similar_cases: list
    false_positive_patterns: list

    # ── RCA (Backward BFS) ────────────────────────────────────────────
    ranked_causes: list             # List[RCACandidate]
    incident_id: str
    fault_tree: dict
    bfs_paths: list
    silent_failures: list
    llm_kg: dict                    # Debug info from LLM KG mode (if enabled)

    # ── Causal Ranker (Pillar 2) ──────────────────────────────────────
    causal_scores: dict             # {service: causal_strength}
    temporal_order: list            # Services ordered by anomaly arrival

    # ── Log Pattern Analysis (Pillar 3) ───────────────────────────────
    log_clusters: list              # [{pattern, count, services, severity}]
    suspect_services: list

    # ── Mitigation ────────────────────────────────────────────────────
    mitigation_actions: list        # List[MitigationAction]

    # ── Safety / TNR (Pillar 1) ───────────────────────────────────────
    undo_stack: list                # [(action_cmd, revert_cmd)]
    health_score_before: float      # μ(s) before mitigation
    health_score_after: float       # μ(s) after mitigation
    rollback_triggered: bool
    rollback_count: int
    sla_violations: list
    unhealthy_nodes: list

    # ── Memory Store ──────────────────────────────────────────────────
    memory_stored: bool

    # ── Pipeline Orchestration ────────────────────────────────────────
    status: str                     # running | complete | rollback | failed
    error: str
    output_dir: str
    pipeline_start_time: float
    node_timings: dict              # {node_name: elapsed_seconds}
    messages: list                  # Accumulated log messages
