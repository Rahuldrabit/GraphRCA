import os
import networkx as nx
from typing import List
from swarm_state import AIOpsIncidentState
from tools.scratchpad_client import ScratchpadClient
import logging

logger = logging.getLogger(__name__)

class TopologicalDiagnoser:
    """
    Zero-token NetworkX node.
    Builds DiGraph from ScratchPad active edges, reverses calls, and runs PPR.
    """
    def __init__(self, scratchpad_client: ScratchpadClient):
        self.client = scratchpad_client
        self.alpha = float(os.getenv("SCRATCHPAD_PAGERANK_ALPHA", "0.85"))

    def __call__(self, state: AIOpsIncidentState) -> AIOpsIncidentState:
        session_id = state["scratchpad_session_id"]
        triplets = self.client.get_triplets(session_id)
        
        if not triplets:
            logger.warning("No triplets found in ScratchPad. Returning empty suspects.")
            state["suspect_nodes"] = []
            return state

        G = nx.DiGraph()
        
        # Build the graph
        for t in triplets:
            src = t["source_entity"]
            rel = t["relationship"]
            dst = t["target_entity"]
            
            if rel == "calls":
                # Reverse calls for backward fault propagation
                G.add_edge(dst, src, weight=1.0)
            elif rel == "emits":
                G.add_edge(dst, src, weight=2.0) # faults heavily influence source
            else:
                G.add_edge(src, dst, weight=1.0)
                
        if len(G.nodes) == 0:
            state["suspect_nodes"] = []
            return state

        # Compute PageRank
        try:
            pagerank_scores = nx.pagerank(G, alpha=self.alpha, weight="weight")
            
            # Sort by score descending
            sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Extract top 3 that look like services (not HTTP_500, etc)
            suspects = []
            for node, score in sorted_nodes:
                if not node.startswith("HTTP_") and not node.startswith("LOG_") and not node.startswith("ERROR"):
                    suspects.append(node)
                if len(suspects) >= 3:
                    break
                    
            state["suspect_nodes"] = suspects
            logger.info(f"Topological suspects identified: {suspects}")
        except Exception as e:
            logger.error(f"PageRank computation failed: {e}")
            state["suspect_nodes"] = []

        return state
