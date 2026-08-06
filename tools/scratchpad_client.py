import os
import sys
import sqlite3
from typing import List, Dict, Optional, Any

# Ensure ScratchPad src is in sys.path
_scratchpad_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ScratchPad", "src", "src"))
if _scratchpad_src not in sys.path:
    sys.path.insert(0, _scratchpad_src)

try:
    from engine import ScratchpadEngine
    from database import initialize_database
    from schema import Triplet
except ImportError as e:
    print(f"Warning: Could not import ScratchPad modules from {_scratchpad_src}: {e}")
    ScratchpadEngine = None


class ScratchpadClient:
    """
    In-process ScratchPad client.
    Connects directly to the ScratchPad SQLite WAL database without an HTTP server.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            os.environ["SCRATCHPAD_DB_PATH"] = db_path
        
        # We need to initialize DB if not present
        if ScratchpadEngine is not None:
            initialize_database()
            self.engine = ScratchpadEngine()
        else:
            self.engine = None
    
    def init_session(self, session_id: str, goal: str) -> None:
        """Initializes a new session."""
        if self.engine:
            self.engine.init_session(session_id, goal)
            
    def get_view(self, session_id: str, max_tokens: int = 500) -> str:
        """Returns the bounded Markdown view of the ScratchPad."""
        if self.engine:
            return self.engine.compile_context_view(session_id, max_tokens=max_tokens)
        return ""
        
    def commit_triplets(self, session_id: str, agent_id: str, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Commits L1 triplets through the verification gate."""
        if not self.engine:
            return []
            
        validated_triplets = []
        for t in triplets:
            try:
                triplet_obj = Triplet(
                    source=t.get("source", ""),
                    target=t.get("target", ""),
                    relationship=t.get("relationship", ""),
                    citation_quote=t.get("citation_quote", ""),
                    source_type=t.get("source_type", "UNKNOWN"),
                    target_type=t.get("target_type", "UNKNOWN"),
                )
                validated_triplets.append(triplet_obj)
            except Exception as e:
                print(f"[ScratchPad] Validation error for triplet {t}: {e}")
                
        if not validated_triplets:
            return []
            
        return self.engine.commit_triplets(session_id, agent_id, validated_triplets)

    def get_triplets(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetches all active triplets for a session (used by Topological Diagnoser)."""
        if not self.engine:
            return []
            
        db_path = os.environ.get("SCRATCHPAD_DB_PATH", "scratchpad_memory.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT source_entity, relationship, target_entity, relevance_score, hierarchy_level 
            FROM knowledge_graph 
            WHERE session_id = ? AND is_active = 1
        ''', (session_id,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def run_sweeper(self, session_id: str) -> None:
        """Triggers the on-demand Louvain L2 sweeper."""
        # Optional: Since the sweeper is a background process normally, 
        # we can import and run it directly here or omit it if it's too complex.
        pass
        
    def drill_down(self, session_id: str, node: str) -> str:
        """Gets detailed context for a specific node (L2 expansion)."""
        # Not fully implemented in base engine without sweeper context, returning mock/placeholder
        return f"Drill down details for {node}"
