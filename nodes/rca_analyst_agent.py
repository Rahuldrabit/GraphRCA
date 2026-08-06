import json
import logging
from typing import Dict, Any, List
from swarm_state import AIOpsIncidentState
from tools.scratchpad_client import ScratchpadClient
from llm import llm_reason

logger = logging.getLogger(__name__)

class RCAAnalystAgent:
    """
    SLM agent that analyzes the ScratchPad markdown view.
    Verifies the root cause and provides a citation quote.
    """
    def __init__(self, scratchpad_client: ScratchpadClient):
        self.client = scratchpad_client

    def __call__(self, state: AIOpsIncidentState) -> AIOpsIncidentState:
        session_id = state["scratchpad_session_id"]
        suspects = state.get("suspect_nodes", [])
        
        if not suspects:
            logger.warning("No suspects provided to RCA Analyst.")
            state["verified_root_cause"] = None
            return state

        # Get bounded markdown view
        markdown_view = self.client.get_view(session_id, max_tokens=500)
        
        system_prompt = (
            "You are an expert Site Reliability Engineer (SRE) performing Root Cause Analysis. "
            "You will be given a list of suspect services and a compact knowledge graph representation "
            "(triplets) of the system state. "
            "Your job is to identify the single root cause service and extract the EXACT citation quote "
            "from the knowledge graph that proves it."
        )
        
        user_prompt = (
            f"Suspect nodes from topological analysis: {suspects}\n\n"
            f"Knowledge Graph View:\n{markdown_view}\n\n"
            "Based on the above, identify the root cause service.\n"
            "Respond ONLY in valid JSON format:\n"
            "{\n"
            '  "root_cause_service": "SERVICE_NAME",\n'
            '  "relationship": "emits",\n'
            '  "target": "ERROR_STATE",\n'
            '  "citation_quote": "exact quote from the view"\n'
            "}"
        )
        
        response = llm_reason(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=300,
            caller="rca_analyst"
        )
        
        try:
            # Simple JSON extraction
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                root_cause = data.get("root_cause_service")
                if root_cause:
                    state["verified_root_cause"] = root_cause
                    
                    # Commit the verified hypothesis back to ScratchPad
                    self.client.commit_triplets(
                        session_id, 
                        "RCAAnalyst", 
                        [{
                            "source": root_cause,
                            "relationship": data.get("relationship", "causes"),
                            "target": data.get("target", "INCIDENT"),
                            "citation_quote": data.get("citation_quote", ""),
                            "source_type": "SERVICE",
                            "target_type": "FAULT"
                        }]
                    )
                    logger.info(f"RCA Analyst verified root cause: {root_cause}")
                else:
                    state["verified_root_cause"] = suspects[0] # fallback
            else:
                state["verified_root_cause"] = suspects[0]
        except Exception as e:
            logger.error(f"Failed to parse RCA LLM response: {e}")
            state["verified_root_cause"] = suspects[0] if suspects else None
            
        return state
