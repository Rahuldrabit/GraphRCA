from typing import TypedDict, Any, List, Dict, Optional

class AIOpsIncidentState(TypedDict):
    """
    Lightweight state for the 4-agent swarm.
    Does NOT contain conversational history to protect SLMs from context degradation.
    """
    problem_id: str
    task_type: str  # detection, localization, analysis, mitigation
    
    # ScratchPad Session Context
    scratchpad_session_id: str
    
    # Shared State Pointers
    suspect_nodes: List[str]          # Top-3 candidates from Topological Diagnoser
    verified_root_cause: Optional[str] # Confirmed by RCA Analyst
    
    # Task specific submission parameters
    final_submission: Optional[Any]   # Format depends on task_type
    
    # Internal routing
    retry_count: int
    error: Optional[str]
