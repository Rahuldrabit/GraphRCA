"""Neo4j Connector — GraphRCA's own implementation.

Connection priority:
  1. Local Neo4j  (bolt://localhost:7687  — Docker container)
  2. Remote Aura  (only when NEO4J_AURA_FALLBACK=true and local is unreachable)
  3. NetworkX     (pure in-memory graph — automatic fallback when both above fail)

Set NEO4J_ENABLED=True  → tries local first, then Aura if configured
Set NEO4J_ENABLED=False → skips Neo4j entirely, uses NetworkX only
"""

import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ── NetworkX in-memory fallback store ────────────────────────────────────────

class NetworkXKGStore:
    """Lightweight NetworkX-based knowledge graph used when Neo4j is unavailable.

    Exposes the same interface as Neo4jConnector so the rest of the pipeline
    never needs to handle a None connector:
        is_available(), run_query(), execute_write(), execute_read(), clear_database()
    Cypher queries are silently ignored — the pipeline builds its own in-memory
    NetworkX DAG via graph_tools anyway; this store just prevents None errors.
    """

    def __init__(self):
        try:
            import networkx as nx
            self._G = nx.DiGraph()
            self._available = True
            logger.info("[KG] Using NetworkX in-memory knowledge graph (Neo4j not available)")
        except ImportError:
            self._available = False
            logger.warning("[KG] NetworkX not installed — knowledge graph store disabled")

    def is_available(self) -> bool:
        return self._available

    def run_query(self, query: str, parameters: dict = None):
        """Cypher queries are no-ops in NetworkX mode."""
        return []

    def execute_write(self, query: str, parameters: dict = None):
        return self.run_query(query, parameters)

    def execute_read(self, query: str, parameters: dict = None):
        return self.run_query(query, parameters)

    def clear_database(self):
        if self._available:
            import networkx as nx
            self._G = nx.DiGraph()

    def get_graph(self):
        """Return the underlying NetworkX DiGraph."""
        return self._G


# ── Neo4j connector ───────────────────────────────────────────────────────────

class Neo4jConnector:
    """Neo4j connection wrapper with lazy init, local-first, and Aura fallback."""

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri      = uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        self.user     = user     or os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        if not self.password:
            raise ValueError(
                "Neo4j password not set. Pass password= explicitly or set the "
                "NEO4J_PASSWORD environment variable."
            )
        self._driver: Optional[object] = None
        self._available: Optional[bool] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify_query(self, query: str) -> str:
        q = re.sub(r"\s+", " ", (query or "").strip()).upper()
        if not q:
            return "unknown"
        if "DETACH DELETE" in q or re.search(r"\bDELETE\b", q):
            return "delete"
        if re.search(r"\b(CREATE|MERGE|SET|REMOVE|DROP)\b", q):
            return "write"
        return "read"

    def _try_connect(self, uri: str, user: str, password: str) -> bool:
        """Try to connect to a specific URI. Returns True on success."""
        try:
            from neo4j import GraphDatabase
            drv = GraphDatabase.driver(uri, auth=(user, password))
            with drv.session() as s:
                s.run("RETURN 1")
            self._driver    = drv
            self._available = True
            self.uri        = uri
            logger.info(f"[Neo4j] Connected at {uri}")
            return True
        except Exception as e:
            logger.warning(f"[Neo4j] {uri!r} unreachable: {e}")
            return False

    def _connect(self):
        """Lazily connect — local first, then Aura fallback."""
        if self._driver is not None:
            return

        # 1. Primary URI (local Docker by default)
        if self._try_connect(self.uri, self.user, self.password):
            return

        # 2. Remote Aura fallback
        if os.getenv("NEO4J_AURA_FALLBACK", "false").lower() == "true":
            aura_uri  = os.getenv("NEO4J_AURA_URI", "")
            aura_pass = os.getenv("NEO4J_AURA_PASSWORD", "")
            aura_user = os.getenv("NEO4J_USERNAME", "neo4j")
            if aura_uri and aura_pass:
                if self._try_connect(aura_uri, aura_user, aura_pass):
                    return

        self._available = False
        self._driver    = None

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self._available is None:
            self._connect()
        return bool(self._available)

    def get_driver(self):
        if self._driver is None:
            self._connect()
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
        self._driver    = None
        self._available = None

    def run_query(self, query: str, parameters: dict = None):
        """Run a Cypher query and return a list of records (or None if unavailable)."""
        if not self.is_available():
            logger.warning("[Neo4j] Not available, skipping query")
            return None

        params = parameters or {}
        op     = self._classify_query(query)

        try:
            from GraphRCA_agent.trace_logger import trace_event
            trace_event("neo4j.query", tool="neo4j", op=op, query=query, parameters=params)
        except Exception:
            pass

        t0 = time.time()
        try:
            with self._driver.session() as session:
                result  = session.run(query, params)
                records = list(result)
                summary = result.consume()

            counters = {}
            if summary and getattr(summary, "counters", None):
                c = summary.counters
                counters = {
                    "nodes_created":         getattr(c, "nodes_created", 0),
                    "nodes_deleted":         getattr(c, "nodes_deleted", 0),
                    "relationships_created": getattr(c, "relationships_created", 0),
                    "relationships_deleted": getattr(c, "relationships_deleted", 0),
                    "properties_set":        getattr(c, "properties_set", 0),
                    "labels_added":          getattr(c, "labels_added", 0),
                }

            elapsed = round(time.time() - t0, 3)
            try:
                from GraphRCA_agent.trace_logger import trace_event
                trace_event("neo4j.result", tool="neo4j", op=op,
                            record_count=len(records), counters=counters,
                            elapsed_seconds=elapsed)
            except Exception:
                pass

            logger.debug(f"[Neo4j] {op} → {len(records)} records  {counters}")
            return records

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            try:
                from GraphRCA_agent.trace_logger import trace_event
                trace_event("neo4j.error", tool="neo4j", op=op,
                            query=query, error=str(e), elapsed_seconds=elapsed)
            except Exception:
                pass
            raise

    # Stratus-compatible aliases
    def execute_write(self, query: str, parameters: dict = None):
        return self.run_query(query, parameters)

    def execute_read(self, query: str, parameters: dict = None):
        return self.run_query(query, parameters)

    def clear_database(self):
        if self.is_available():
            self.run_query("MATCH (n) DETACH DELETE n")
            logger.info("[Neo4j] Database cleared")


# ── Singleton with automatic fallback ────────────────────────────────────────

_connector: Optional[object] = None   # Neo4jConnector | NetworkXKGStore


def get_neo4j_connector() -> object:
    """Return the best available KG store (never returns None).

    Decision:
      NEO4J_ENABLED=False → NetworkXKGStore
      NEO4J_ENABLED=True  → Neo4jConnector (local bolt → Aura)
                            → NetworkXKGStore if both fail
    """
    global _connector
    if _connector is not None:
        return _connector

    if os.getenv("NEO4J_ENABLED", "False").lower() != "true":
        logger.info("[KG] NEO4J_ENABLED=False → using NetworkX in-memory store")
        _connector = NetworkXKGStore()
        return _connector

    neo = Neo4jConnector()
    if neo.is_available():
        logger.info(f"[KG] Neo4j online at {neo.uri}")
        _connector = neo
    else:
        logger.warning("[KG] Neo4j unavailable → falling back to NetworkX in-memory store")
        _connector = NetworkXKGStore()

    return _connector


def reset_connector():
    """Reset the singleton — call between task runs or in tests."""
    global _connector
    if isinstance(_connector, Neo4jConnector):
        _connector.close()
    _connector = None
