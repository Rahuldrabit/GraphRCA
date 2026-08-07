import os
import sys
import sqlite3
import hashlib
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Ensure ScratchPad src is in sys.path.
# Robust to repo layout (folder renamed/copied in various ways): try several
# candidate locations and use the first that actually contains engine.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # tools/ -> GraphRCA root
_candidates = [
    os.path.join(_REPO_ROOT, "ScratchPad", "src"),                # GraphRCA/ScratchPad/src
    os.path.join(_REPO_ROOT, "ScrathPad", "src"),                 # legacy typo
    os.path.join(_REPO_ROOT, "..", "ScratchPad", "src"),          # sibling repo (README layout)
    os.path.join(_REPO_ROOT, "..", "ScratchPad", "src", "src"),   # old double-src layout
]
_scratchpad_src = next(
    (os.path.abspath(c) for c in _candidates if os.path.isfile(os.path.join(c, "engine.py"))),
    None,
)
if _scratchpad_src and _scratchpad_src not in sys.path:
    sys.path.insert(0, _scratchpad_src)

# Best-effort imports of the real ScratchPad in-process API. The bundled
# ScratchPad `engine.py` is function-based (module-level functions), NOT a
# class — there is no ScratchpadEngine. We use the functions directly.
initialize_database = None
get_db_connection = None
compile_bounded_markdown_view = None
try:
    from database import initialize_database, get_db_connection  # type: ignore
except Exception as _e:  # pragma: no cover - degraded path
    logger.debug(f"ScratchPad database module unavailable: {_e}")
try:
    from engine import compile_bounded_markdown_view  # type: ignore
except Exception as _e:  # pragma: no cover - degraded path
    logger.debug(f"ScratchPad engine.compile_bounded_markdown_view unavailable: {_e}")


class ScratchpadClient:
    """
    In-process ScratchPad client.
    Connects directly to the ScratchPad SQLite WAL database without an HTTP server.

    The swarm's agents are themselves deterministic rule engines / SLMs, so we
    write verified triplets straight to the knowledge_graph store (bypassing the
    LLM verification gate, which would reject swarm vocab like 'has_task' /
    'causes'). Reads come from the same table, so observer→diagnoser→RCA share
    one consistent memory. The bounded Markdown view delegates to the real
    ScratchPad `compile_bounded_markdown_view` when available, with a compact
    hand-built fallback.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            os.environ["SCRATCHPAD_DB_PATH"] = db_path
        self.db_path = os.environ.get("SCRATCHPAD_DB_PATH") or "scratchpad_memory.db"
        self._init_db()

    # ── connection / schema ──────────────────────────────────────────────

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Ensure the ScratchPad schema exists. Prefers the real initializer
        (which also runs the type/provenance column migration); falls back to a
        minimal local schema so the client still works without the module."""
        if initialize_database is not None:
            try:
                initialize_database()
            except Exception as e:
                logger.warning(f"ScratchPad initialize_database failed; using minimal schema: {e}")
                self._ensure_minimal_schema()
        else:
            self._ensure_minimal_schema()

        # Introspect actual knowledge_graph columns so commit_triplets only
        # writes columns that exist (handles migrated vs minimal schemas).
        conn = self._conn()
        try:
            rows = conn.execute("PRAGMA table_info(knowledge_graph)").fetchall()
            self._kg_cols = {r[1] for r in rows}
        finally:
            conn.close()

    def _ensure_minimal_schema(self) -> None:
        conn = self._conn()
        try:
            c = conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_query TEXT,
                    master_plan TEXT,
                    global_status TEXT DEFAULT 'PLANNING'
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS knowledge_graph (
                    edge_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_id TEXT,
                    source_entity TEXT NOT NULL,
                    relationship TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    citation_quote TEXT,
                    hierarchy_level INTEGER DEFAULT 1,
                    parent_node_id TEXT DEFAULT NULL,
                    relevance_score REAL DEFAULT 1.0,
                    is_active BOOLEAN DEFAULT 1,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.commit()
        finally:
            conn.close()

    # ── public API ───────────────────────────────────────────────────────

    def init_session(self, session_id: str, goal: Optional[str] = None) -> None:
        """Initializes a new session (idempotent)."""
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, master_plan, global_status) "
                "VALUES (?, ?, 'PLANNING')",
                (session_id, goal),
            )
            conn.commit()
        finally:
            conn.close()

    def commit_triplets(
        self, session_id: str, agent_id: str, triplets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Commits L1 triplets to the knowledge graph (idempotent per edge)."""
        if not triplets:
            return []

        conn = self._conn()
        committed: List[Dict[str, Any]] = []
        try:
            c = conn.cursor()
            # Ensure the session row exists (FK constraint).
            c.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))

            # Base columns always written; optional columns only if present.
            optional_values = {
                "source_type": "UNKNOWN",
                "target_type": "UNKNOWN",
                "extractor": "agent",
                "pass_number": "0",
            }

            for t in triplets:
                try:
                    src = str(t.get("source", "") or "").strip()
                    rel = str(t.get("relationship", "") or "").strip().lower()
                    dst = str(t.get("target", "") or "").strip()
                    if not src or not rel or not dst:
                        continue
                    cite = str(t.get("citation_quote", "") or "")[:240]

                    # Per-source/target type overrides if supplied by the caller.
                    optional_values["source_type"] = str(t.get("source_type") or "UNKNOWN")
                    optional_values["target_type"] = str(t.get("target_type") or "UNKNOWN")

                    # Optional caller-supplied salience in [0,1]; default 1.0.
                    relevance = t.get("relevance")
                    try:
                        relevance = float(relevance) if relevance is not None else 1.0
                    except (TypeError, ValueError):
                        relevance = 1.0

                    edge_id = hashlib.sha1(
                        f"{session_id}|{src}|{rel}|{dst}".encode("utf-8")
                    ).hexdigest()[:16]

                    col_sql = [
                        "edge_id", "session_id", "agent_id",
                        "source_entity", "relationship", "target_entity",
                        "citation_quote", "hierarchy_level", "relevance_score",
                        "is_active",
                    ]
                    vals: List[Any] = [
                        edge_id, session_id, agent_id,
                        src, rel, dst,
                        cite, 1, relevance, 1,
                    ]
                    placeholders = ["?"] * len(col_sql)

                    for opt_col, opt_val in optional_values.items():
                        if opt_col in self._kg_cols:
                            col_sql.append(opt_col)
                            placeholders.append("?")
                            vals.append(opt_val)

                    sql = (
                        "INSERT OR REPLACE INTO knowledge_graph "
                        f"({', '.join(col_sql)}) VALUES ({', '.join(placeholders)})"
                    )
                    c.execute(sql, vals)
                    committed.append(
                        {
                            "source_entity": src,
                            "relationship": rel,
                            "target_entity": dst,
                            "citation_quote": cite,
                        }
                    )
                except Exception as e:
                    logger.warning(f"[ScratchPad] commit triplet rejected {t}: {e}")

            conn.commit()
        finally:
            conn.close()
        return committed

    def get_triplets(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetches all active triplets for a session (used by Topological Diagnoser)."""
        conn = self._conn()
        try:
            rows = conn.execute(
                """SELECT source_entity, relationship, target_entity, citation_quote,
                          relevance_score, hierarchy_level
                   FROM knowledge_graph
                   WHERE session_id = ? AND is_active = 1""",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"[ScratchPad] get_triplets failed: {e}")
            return []
        finally:
            conn.close()

    def get_view(
        self,
        session_id: str,
        max_tokens: int = 500,
        query: Optional[str] = None,
        k_hops: Optional[int] = None,
    ) -> str:
        """Returns the bounded Markdown view of the ScratchPad.

        When `query` is given, the engine runs a query-aware k-hop neighborhood
        blast (`_apply_query_aware_boost`) — facts within k_hops of any entity
        named in the query are promoted to the top of the token-budgeted view.
        This is the ScratchPad-native "radius blast": bounded for the SLM, no
        raw graph dump. Both args are optional and backward compatible.
        """
        if compile_bounded_markdown_view is not None:
            try:
                kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
                if query is not None:
                    kwargs["query"] = query
                if k_hops is not None:
                    kwargs["k_hops"] = k_hops
                view = compile_bounded_markdown_view(session_id, **kwargs)
                if view:
                    return view
            except Exception as e:
                logger.debug(f"[ScratchPad] compile_bounded_markdown_view failed, using fallback: {e}")

        # Fallback: render a compact view straight from the triplets.
        triplets = self.get_triplets(session_id)
        if not triplets:
            return "(ScratchPad knowledge graph is empty for this session)"
        lines = ["## ScratchPad Knowledge Graph", ""]
        for t in triplets:
            cite = t.get("citation_quote", "")
            lines.append(
                f"- `{t['source_entity']}` **{t['relationship']}** `{t['target_entity']}`"
                + (f" — {cite}" if cite else "")
            )
        return "\n".join(lines)

    def run_sweeper(self, session_id: str) -> None:
        """Triggers the on-demand Louvain L2 sweeper (optional / no-op for now)."""
        pass

    def drill_down(self, session_id: str, edge_id_or_node: str, k_hops: int = 1) -> str:
        """Expand a COMPRESSED node or service into its k-hop triplet neighborhood.

        The bounded view marks compressed facts as
        `[COMPRESSED | drill-down id: <edge_id>]`; the SLM passes that edge_id
        (or a service name) and gets the local subgraph back — a ScratchPad-native,
        budget-friendly expansion so the agent can "query the KG" without the
        whole graph being dumped into its context window.
        """
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT source_entity, relationship, target_entity, citation_quote, "
                "relevance_score, edge_id FROM knowledge_graph "
                "WHERE session_id = ? AND is_active = 1",
                (session_id,),
            ).fetchall()
            all_rows = [dict(r) for r in rows]
            if not all_rows:
                return f"(no facts in ScratchPad to expand for '{edge_id_or_node}')"

            # Resolve seed entities: match by edge_id first, else by name.
            seed: set = set()
            needle = str(edge_id_or_node or "").strip()
            by_id = [r for r in all_rows if r.get("edge_id") == needle]
            if by_id:
                for r in by_id:
                    seed.add(r["source_entity"])
                    seed.add(r["target_entity"])
            else:
                nl = needle.lower()
                for r in all_rows:
                    if nl and (nl in (r["source_entity"] or "").lower()
                               or nl in (r["target_entity"] or "").lower()):
                        seed.add(r["source_entity"])
                        seed.add(r["target_entity"])
            if not seed:
                return f"(entity '{edge_id_or_node}' not found in ScratchPad)"

            # k-hop BFS expansion over the in-memory triplet graph.
            neighborhood = set(seed)
            frontier = set(seed)
            for _ in range(max(int(k_hops), 0)):
                nxt = set()
                for r in all_rows:
                    if r["source_entity"] in frontier:
                        nxt.add(r["target_entity"])
                    if r["target_entity"] in frontier:
                        nxt.add(r["source_entity"])
                nxt -= neighborhood
                neighborhood |= nxt
                frontier = nxt
                if not frontier:
                    break

            lines = [
                f"## drill-down: {'/'.join(sorted(seed))} "
                f"({k_hops}-hop, {len(neighborhood)} entities)"
            ]
            shown = 0
            for r in all_rows:
                if r["source_entity"] in neighborhood or r["target_entity"] in neighborhood:
                    cite = (r.get("citation_quote") or "")[:160]
                    lines.append(
                        f"- `{r['source_entity']}` --{r['relationship']}--> "
                        f"`{r['target_entity']}`" + (f" — {cite}" if cite else "")
                    )
                    shown += 1
                    if shown >= 40:
                        lines.append(f"...(showing first {shown} of the neighborhood)")
                        break
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"[ScratchPad] drill_down failed: {e}")
            return f"(drill-down failed for '{edge_id_or_node}')"
        finally:
            conn.close()
