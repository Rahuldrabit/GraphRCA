from typing import TypedDict, Any, List, Dict, Optional

class AIOpsIncidentState(TypedDict):
    """
    Lightweight state for the 4-agent swarm.
    Does NOT contain conversational history to protect SLMs from context degradation.
    """
    problem_id: str
    task_type: str  # detection, localization, analysis, mitigation
    problem_description: Optional[str]

    # ScratchPad Session Context
    scratchpad_session_id: str

    # Multi-telemetry input paths (observer parses whichever exist).
    # Traces (Jaeger), pod status (kubectl), metrics (Prometheus), logs.
    trace_csv_path: Optional[str]
    pod_status_path: Optional[str]
    metrics_path: Optional[str]
    logs_path: Optional[str]

    # Shared State Pointers
    suspect_nodes: List[str]          # Top-3 candidates from Topological Diagnoser
    verified_root_cause: Optional[str] # Confirmed by RCA Analyst

    # Observer signal: True iff any anomaly (errors / latency / bad pods / log errors)
    # was found across telemetry. Drives detection Yes/No.
    anomaly_detected: bool

    # Task specific submission parameters
    final_submission: Optional[Any]   # Format depends on task_type

    # Internal routing
    retry_count: int
    error: Optional[str]
