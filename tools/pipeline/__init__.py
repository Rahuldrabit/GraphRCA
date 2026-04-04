"""GraphRCA Pipeline Tools.

Self-contained implementations of trace ingestion, graph building,
anomaly detection, RCA, memory storage, and mitigation planning.
GraphRCA follows Stratus methodology but is an independent codebase.
"""

from GraphRCA_agent.tools.pipeline.ingest_tools import (
    parse_csv_directory,
    deduplicate_spans,
    validate_schema,
    detect_orphan_spans,
    compute_stats,
    compute_overall_stats,
    OverallStats,
)

from GraphRCA_agent.tools.pipeline.graph_tools import (
    build_dag,
    add_node_with_metrics,
    detect_cycles,
    compute_pagerank,
    export_graph_json,
    store_graph_to_neo4j,
    store_trace_spans_to_neo4j,
)

from GraphRCA_agent.tools.pipeline.detection_tools import (
    compute_ewma_baseline,
    detect_all_anomalies,
)

from GraphRCA_agent.tools.pipeline.rca_tools import (
    backward_bfs_traversal,
    score_candidate,
    infer_silent_failures,
    rank_root_causes,
    generate_fault_tree,
    store_rca_to_neo4j,
)

from GraphRCA_agent.tools.pipeline.memory_tools import (
    MemoryStore,
    load_similar_cases,
    embed_incident,
    update_confidence_from_outcome,
    store_rca_to_neo4j_memory,
)

from GraphRCA_agent.tools.pipeline.mitigation_tools import (
    generate_mitigation_plan,
    request_human_approval,
)

__all__ = [
    # ingest
    "parse_csv_directory",
    "deduplicate_spans",
    "validate_schema",
    "detect_orphan_spans",
    "compute_stats",
    "compute_overall_stats",
    "OverallStats",
    # graph
    "build_dag",
    "add_node_with_metrics",
    "detect_cycles",
    "compute_pagerank",
    "export_graph_json",
    "store_graph_to_neo4j",
    "store_trace_spans_to_neo4j",
    # detection
    "compute_ewma_baseline",
    "detect_all_anomalies",
    # rca
    "backward_bfs_traversal",
    "score_candidate",
    "infer_silent_failures",
    "rank_root_causes",
    "generate_fault_tree",
    "store_rca_to_neo4j",
    # memory
    "MemoryStore",
    "load_similar_cases",
    "embed_incident",
    "update_confidence_from_outcome",
    "store_rca_to_neo4j_memory",
    # mitigation
    "generate_mitigation_plan",
    "request_human_approval",
]
