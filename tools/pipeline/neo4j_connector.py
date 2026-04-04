"""Neo4j Connector — GraphRCA's own implementation.

Provides connection to Neo4j for knowledge graph storage.
"""

import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Neo4j connection wrapper with lazy initialization."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        # Support both NEO4J_USERNAME (used elsewhere) and NEO4J_USER
        self.user = user or os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver = None
        self._available = None

    def _classify_query(self, query: str) -> str:
        """Classify query roughly as read/write/delete for logging."""
        q = re.sub(r"\s+", " ", (query or "").strip()).upper()
        if not q:
            return "unknown"
        if "DETACH DELETE" in q or re.search(r"\bDELETE\b", q):
            return "delete"
        if re.search(r"\b(CREATE|MERGE|SET|REMOVE|DROP)\b", q):
            return "write"
        return "read"
    
    def _connect(self):
        """Lazily connect to Neo4j."""
        if self._driver is not None:
            return
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")
            self._available = True
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            self._available = False
            self._driver = None
    
    def is_available(self) -> bool:
        """Check if Neo4j connection is available."""
        if self._available is None:
            self._connect()
        return self._available
    
    def get_driver(self):
        """Get the Neo4j driver."""
        if self._driver is None:
            self._connect()
        return self._driver
    
    def close(self):
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._available = None
    
    def run_query(self, query: str, parameters: dict = None):
        """Run a Cypher query."""
        if not self.is_available():
            logger.warning("Neo4j not available, skipping query")
            return None

        params = parameters or {}
        op = self._classify_query(query)

        # Structured tracing (if enabled)
        try:
            from GraphRCA_agent.trace_logger import trace_event

            trace_event("neo4j.query", tool="neo4j", op=op, query=query, parameters=params)
        except Exception:
            pass

        t0 = time.time()
        try:
            with self._driver.session() as session:
                result = session.run(query, params)
                records = list(result)
                summary = result.consume()

            counters = {}
            if summary is not None and getattr(summary, "counters", None) is not None:
                c = summary.counters
                # Expose the counters users care about (ingested/deleted)
                counters = {
                    "nodes_created": getattr(c, "nodes_created", 0),
                    "nodes_deleted": getattr(c, "nodes_deleted", 0),
                    "relationships_created": getattr(c, "relationships_created", 0),
                    "relationships_deleted": getattr(c, "relationships_deleted", 0),
                    "properties_set": getattr(c, "properties_set", 0),
                    "labels_added": getattr(c, "labels_added", 0),
                    "indexes_added": getattr(c, "indexes_added", 0),
                    "constraints_added": getattr(c, "constraints_added", 0),
                }

            elapsed = time.time() - t0

            try:
                from GraphRCA_agent.trace_logger import trace_event

                trace_event(
                    "neo4j.result",
                    tool="neo4j",
                    op=op,
                    record_count=len(records),
                    counters=counters,
                    elapsed_seconds=round(elapsed, 3),
                )
            except Exception:
                pass

            logger.debug(f"Neo4j {op} query completed: records={len(records)} counters={counters}")
            return records
        except Exception as e:
            elapsed = time.time() - t0
            try:
                from GraphRCA_agent.trace_logger import trace_event

                trace_event(
                    "neo4j.error",
                    tool="neo4j",
                    op=op,
                    query=query,
                    parameters=params,
                    error=str(e),
                    elapsed_seconds=round(elapsed, 3),
                )
            except Exception:
                pass
            raise

    # ── Stratus compatibility ─────────────────────────────────────────────

    def execute_write(self, query: str, parameters: dict = None):
        """Stratus-style alias for write queries.

        Some pipeline tools expect a connector with `execute_write(...)`.
        GraphRCA's native method is `run_query(...)`.
        """
        return self.run_query(query, parameters)

    def execute_read(self, query: str, parameters: dict = None):
        """Stratus-style alias for read queries."""
        return self.run_query(query, parameters)
    
    def clear_database(self):
        """Clear all data from Neo4j."""
        if not self.is_available():
            return
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all Neo4j data")


# Singleton instance
_connector: Optional[Neo4jConnector] = None


def get_neo4j_connector() -> Optional[Neo4jConnector]:
    """Get the singleton Neo4j connector."""
    global _connector
    if _connector is None:
        enabled = os.getenv("NEO4J_ENABLED", "False").lower() == "true"
        if not enabled:
            logger.info("Neo4j disabled (NEO4J_ENABLED != True)")
            return None
        _connector = Neo4jConnector()
    return _connector


def reset_connector():
    """Reset the singleton connector (for testing)."""
    global _connector
    if _connector:
        _connector.close()
    _connector = None
