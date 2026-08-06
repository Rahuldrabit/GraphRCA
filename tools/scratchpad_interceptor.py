import json
import logging
from typing import List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# To use ScratchpadClient, we would instantiate it or pass it.
# For simplicity with Langchain tools, we can use a global or factory pattern.
# However, the tools usually need access to the session_id and the client.
# A common pattern is to inject them via kwargs or create a class with methods wrapped as tools.

class ScratchpadToolInterceptor:
    def __init__(self, scratchpad_client, orchestrator=None):
        self.client = scratchpad_client
        self.orchestrator = orchestrator # AIOpsLab orchestrator reference

    def get_tools(self):
        """Returns the list of tools configured for this interceptor."""
        
        @tool
        def compute_pagerank_root_cause(session_id: str) -> str:
            """
            Computes the PageRank root cause over the ScratchPad SQLite triplets.
            Returns the top 3 suspect nodes.
            """
            # Implementation will be inside Topological Diagnoser, 
            # but providing a tool wrapper if an agent needs to trigger it.
            return "Topological Diagnoser should run this."
            
        @tool
        def drill_down_l2_summary(session_id: str, node_id: str) -> str:
            """
            Expands a compacted L1 node into its detailed L2 summary view.
            Use this when the <=500 token markdown view lacks sufficient detail for a specific suspect.
            """
            return self.client.drill_down(session_id, node_id)
            
        @tool
        def commit_verified_hypothesis(session_id: str, agent_id: str, source: str, relationship: str, target: str, citation_quote: str) -> str:
            """
            Commits a verified root cause hypothesis to the ScratchPad.
            Must include the exact citation_quote from the markdown view.
            """
            triplet = {
                "source": source,
                "relationship": relationship,
                "target": target,
                "citation_quote": citation_quote,
                "source_type": "SERVICE",
                "target_type": "FAULT"
            }
            res = self.client.commit_triplets(session_id, agent_id, [triplet])
            if res:
                return f"Successfully committed hypothesis: {source} {relationship} {target}"
            return "Failed to commit hypothesis. Citation quote may not match."

        @tool
        def execute_aiopslab_mitigation(command: str) -> str:
            """
            Executes a mitigation command on the AIOpsLab environment via the Guardrail Actuator.
            """
            # This is intercepted by the Guardrail Actuator.
            # In a real tool call, it just returns an instruction.
            return f"Requested mitigation: {command}"
            
        @tool
        def trigger_louvain_sweeper(session_id: str) -> str:
            """
            Triggers the on-demand Louvain sweeper to compress the ScratchPad graph.
            """
            self.client.run_sweeper(session_id)
            return "Louvain L2 compression triggered."

        return [
            drill_down_l2_summary,
            commit_verified_hypothesis,
            execute_aiopslab_mitigation,
            trigger_louvain_sweeper
        ]
