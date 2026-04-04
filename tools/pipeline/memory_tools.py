"""Memory Storage Tools for GraphRCA.

SQLite-based incident memory store for similar case retrieval
and learning from past incidents.
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryStore:
    """SQLite-based incident memory store."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize memory store.
        
        Args:
            db_path: Path to SQLite database file.
                     If None, uses in-memory database.
        """
        self._conn: Optional[sqlite3.Connection] = None
        self._keep_open: bool = False

        db_path = (db_path or "").strip()
        if db_path:
            # Create parent directory if provided (handle relative paths like "memory.sqlite")
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self.db_path = db_path
        else:
            # In-memory DB is fine for a single connection, but our code opens
            # multiple connections; keep one persistent connection open.
            self.db_path = ":memory:"
            self._keep_open = True
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                root_cause TEXT,
                affected_services TEXT,
                error_signature TEXT,
                resolution TEXT,
                confidence REAL,
                outcome TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_error_signature 
            ON incidents(error_signature)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_root_cause 
            ON incidents(root_cause)
        """)
        
        conn.commit()
        self._maybe_close(conn)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._keep_open:
            if self._conn is None:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            return self._conn
        return sqlite3.connect(self.db_path)

    def _maybe_close(self, conn: sqlite3.Connection) -> None:
        if self._keep_open:
            return
        try:
            conn.close()
        except Exception:
            pass
    
    def store_incident(
        self,
        incident_id: str,
        root_cause: str,
        affected_services: List[str],
        error_signature: str,
        resolution: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store an incident record.
        
        Args:
            incident_id: Unique incident identifier
            root_cause: Identified root cause service
            affected_services: List of affected services
            error_signature: Hash/signature of error pattern
            resolution: Optional resolution description
            confidence: Confidence score (0-1)
            metadata: Optional additional metadata
            
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO incidents 
                (id, timestamp, root_cause, affected_services, error_signature, 
                 resolution, confidence, outcome, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident_id,
                datetime.utcnow().isoformat(),
                root_cause,
                json.dumps(affected_services),
                error_signature,
                resolution,
                confidence,
                "pending",
                json.dumps(metadata or {}),
            ))
            
            conn.commit()
            self._maybe_close(conn)
            
            logger.info(f"Stored incident {incident_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store incident: {e}")
            return False
    
    def find_similar(
        self,
        error_signature: str,
        root_cause: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar past incidents.
        
        Args:
            error_signature: Error signature to match
            root_cause: Optional root cause to filter by
            limit: Maximum results to return
            
        Returns:
            List of similar incident records
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if root_cause:
                cursor.execute("""
                    SELECT * FROM incidents 
                    WHERE error_signature = ? OR root_cause = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (error_signature, root_cause, limit))
            else:
                cursor.execute("""
                    SELECT * FROM incidents 
                    WHERE error_signature = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (error_signature, limit))
            
            rows = cursor.fetchall()
            self._maybe_close(conn)
            
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "root_cause": row[2],
                    "affected_services": json.loads(row[3]) if row[3] else [],
                    "error_signature": row[4],
                    "resolution": row[5],
                    "confidence": row[6],
                    "outcome": row[7],
                    "metadata": json.loads(row[8]) if row[8] else {},
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar incidents: {e}")
            return []
    
    def update_outcome(
        self,
        incident_id: str,
        outcome: str,
        confidence_delta: float = 0.0,
    ) -> bool:
        """Update incident outcome (for learning).
        
        Args:
            incident_id: Incident identifier
            outcome: Outcome status (e.g., "resolved", "failed", "partial")
            confidence_delta: Adjustment to confidence score
            
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE incidents 
                SET outcome = ?, confidence = confidence + ?
                WHERE id = ?
            """, (outcome, confidence_delta, incident_id))
            
            conn.commit()
            self._maybe_close(conn)
            
            logger.info(f"Updated outcome for incident {incident_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update outcome: {e}")
            return False
    
    def get_false_positive_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns that have historically been false positives.
        
        Returns:
            List of false positive pattern records
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT error_signature, root_cause, COUNT(*) as count
                FROM incidents 
                WHERE outcome = 'false_positive'
                GROUP BY error_signature, root_cause
                ORDER BY count DESC
                LIMIT 10
            """)
            
            rows = cursor.fetchall()
            self._maybe_close(conn)
            
            return [
                {"error_signature": row[0], "root_cause": row[1], "count": row[2]}
                for row in rows
            ]
            
        except Exception as e:
            logger.debug(f"Failed to get false positive patterns: {e}")
            return []
    
    def get_incident_count(self) -> int:
        """Get total number of stored incidents.
        
        Returns:
            Total incident count
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM incidents")
            count = cursor.fetchone()[0]
            self._maybe_close(conn)
            
            return count
            
        except Exception as e:
            logger.debug(f"Failed to get incident count: {e}")
            return 0


def load_similar_cases(
    error_service: str = "",
    anomaly_type: str = "",
    evidence: List[Any] = None,
    memory_store: MemoryStore = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Load similar historical cases based on current incident context.
    
    Args:
        error_service: The service experiencing the error
        anomaly_type: Type of anomaly detected
        evidence: List of evidence items
        memory_store: MemoryStore instance
        limit: Maximum cases to return
        
    Returns:
        List of similar past incidents
    """
    if memory_store is None:
        logger.warning("No memory store provided")
        return []
    
    # Generate error signature from the inputs
    signature_parts = []
    if error_service:
        signature_parts.append(f"service:{error_service}")
    if anomaly_type:
        signature_parts.append(f"anomaly:{anomaly_type}")
    
    signature_str = "|".join(signature_parts) if signature_parts else "unknown"
    error_signature = hashlib.md5(signature_str.encode()).hexdigest()
    
    # Find similar cases
    similar = memory_store.find_similar(error_signature, root_cause=error_service, limit=limit)
    
    logger.info(f"Found {len(similar)} similar historical cases")
    return similar


def embed_incident(
    incident_id: str = "",
    error_service: str = "",
    root_cause_service: str = "",
    root_cause_confidence: float = 0.0,
    anomaly_type: str = "",
    mitigation_applied: str = "",
    mitigation_success: bool = False,
    resolution_time_seconds: float = 0.0,
    service_stats: Dict[str, Any] = None,
    evidence: List[Any] = None,
    memory_store: MemoryStore = None,
) -> str:
    """Store an incident embedding for future similarity retrieval.
    
    Args:
        incident_id: Unique incident identifier
        error_service: Service experiencing the error
        root_cause_service: Identified root cause service
        root_cause_confidence: Confidence score (0-1)
        anomaly_type: Type of anomaly detected
        mitigation_applied: Description of mitigation
        mitigation_success: Whether mitigation was successful
        resolution_time_seconds: Time to resolve
        service_stats: Service statistics
        evidence: List of evidence items
        memory_store: MemoryStore instance
        
    Returns:
        Hash string representing the incident signature
    """
    if memory_store is None:
        logger.warning("No memory store provided, skipping incident storage")
        return ""
    
    # Generate error signature
    signature_parts = []
    if error_service:
        signature_parts.append(f"service:{error_service}")
    if anomaly_type:
        signature_parts.append(f"anomaly:{anomaly_type}")
    if root_cause_service:
        signature_parts.append(f"root_cause:{root_cause_service}")
    
    signature_str = "|".join(signature_parts) if signature_parts else "unknown"
    error_signature = hashlib.md5(signature_str.encode()).hexdigest()
    
    # Store the incident
    affected_services = list(service_stats.keys()) if service_stats else [error_service]
    
    memory_store.store_incident(
        incident_id=incident_id or f"INC-{error_signature[:8]}",
        root_cause=root_cause_service or error_service,
        affected_services=affected_services,
        error_signature=error_signature,
        resolution=mitigation_applied,
        confidence=root_cause_confidence,
        metadata={
            "anomaly_type": anomaly_type,
            "mitigation_success": mitigation_success,
            "resolution_time_seconds": resolution_time_seconds,
            "evidence": evidence or [],
        },
    )
    
    logger.info(f"Stored incident {incident_id}")
    return error_signature


def update_confidence_from_outcome(
    memory_store: MemoryStore,
    incident_id: str,
    was_successful: bool,
) -> None:
    """Update confidence based on resolution outcome.
    
    Args:
        memory_store: MemoryStore instance
        incident_id: Incident identifier
        was_successful: Whether resolution was successful
    """
    outcome = "resolved" if was_successful else "failed"
    delta = 0.1 if was_successful else -0.1
    
    memory_store.update_outcome(incident_id, outcome, delta)


def store_rca_to_neo4j_memory(
    incident_id: str = "",
    error_service: str = "",
    root_cause_service: str = "",
    confidence: float = 0.0,
    anomaly_type: str = "",
    neo4j_connector: Any = None,
    similar_cases: List[Dict[str, Any]] = None,
) -> bool:
    """Store memory relationships to Neo4j.
    
    Args:
        incident_id: Current incident ID
        error_service: Service experiencing the error
        root_cause_service: Identified root cause service
        confidence: Confidence score
        anomaly_type: Type of anomaly
        neo4j_connector: Neo4j connector instance
        similar_cases: Similar historical cases (optional)
        
    Returns:
        True if successful
    """
    if neo4j_connector is None:
        logger.warning("No Neo4j connector provided, skipping memory storage")
        return False
    
    try:
        # Create incident node
        neo4j_connector.run_query(
            """
            MERGE (i:Incident {id: $incident_id})
            SET i.error_service = $error_service,
                i.root_cause_service = $root_cause_service,
                i.confidence = $confidence,
                i.anomaly_type = $anomaly_type,
                i.timestamp = datetime()
            """,
            {
                "incident_id": incident_id,
                "error_service": error_service,
                "root_cause_service": root_cause_service,
                "confidence": confidence,
                "anomaly_type": anomaly_type,
            }
        )
        
        # Link to similar cases if provided
        if similar_cases:
            for case in similar_cases:
                neo4j_connector.run_query(
                    """
                    MATCH (i1:Incident {id: $incident_id})
                    MERGE (i2:Incident {id: $case_id})
                    MERGE (i1)-[r:SIMILAR_TO]->(i2)
                    SET r.confidence = $case_confidence
                    """,
                    {
                        "incident_id": incident_id,
                        "case_id": case.get("id", ""),
                        "case_confidence": case.get("confidence", 0),
                    }
                )
        
        logger.info(f"Stored incident {incident_id} to Neo4j memory")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store memory to Neo4j: {e}")
        return False
