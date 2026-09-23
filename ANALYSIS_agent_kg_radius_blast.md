# Analysis Report — Agent-Driven KG Query + Radius-Blast RCA

**Question:** Can the agent *think* and *query the KG DB*, doing a "radius-blast" analysis like the EWMA → backward-BFS → causal re-rank → safety-gate pipeline?

**Short answer:** **It already exists in this repo** — it is the `pipeline`/`multi_agent` mode ([graph.py:69-117](graph.py#L69-L117)). The batch currently running (PID 113377) uses a *different, simplified* path — `scratchpad_swarm` ([graph.py:215](graph.py#L215)) — which does **none** of it. So the work is not "build it"; it is **bridge the already-built machinery into the swarm path that runs on AIOpsLab**.

> **⚠ Correction (v2) — ScratchPad is the SLM middleware, do not bypass it.**
> The original Phase B below proposed the legacy `kg_llm_tools` (mode a = raw NetworkX JSON dump, mode b = Neo4j Cypher). **Neither uses ScratchPad**, so both defeat the purpose of running ScratchPad as the SLM's bounded-context middleware. ScratchPad already implements its *own* SLM-grade "agent queries KG + radius blast": `compile_bounded_markdown_view(query, k_hops)` does a **k-hop neighborhood blast inside a token budget** ([engine.py:477](ScratchPad/src/engine.py#L477), [542](ScratchPad/src/engine.py#L542)), and **drill-down** on COMPRESSED nodes lets the SLM expand suspects on demand ([engine.py:629](ScratchPad/src/engine.py#L629)). The swarm's RCA node *already* pulls the bounded view ([rca_analyst_agent.py:28](nodes/rca_analyst_agent.py#L28)) — just without `query`/`k_hops` and as a single shot. **See §5B for the ScratchPad-centered recommendation that supersedes §5 Phase B for the SLM case.**

---

## 1. What you described → what already exists (1:1)

| Paper stage | Where it lives in this repo | Status |
|---|---|---|
| **Stage 3** EWMA anomaly (Eq. 2–3) | `detection_node`; adaptive weights in [`_adaptive_weights`](tools/pipeline/rca_tools.py#L186) | ✅ implemented |
| **Stage 5** Backward-BFS (depth 5, cap 200) | [`backward_bfs_traversal`](tools/pipeline/rca_tools.py#L84) — `max_depth=5`, `max_paths=200` | ✅ **this IS the radius blast** |
| **Stage 5** Confidence (Eq. 4: `.35e+.35l+.15a+.10v+d`) | [`score_candidate`](tools/pipeline/rca_tools.py#L214) (0.35/0.35/0.15/0.10 + depth_boost) | ✅ exact match |
| **Stage 5** Silent bottleneck (absorb≥75%, err<5%, ×1.15) | [`infer_silent_failures`](tools/pipeline/rca_tools.py#L351) + ×1.15 in [`rca.py`](nodes/rca.py#L106) | ✅ exact match |
| **Stage 6** Causal re-rank (Eq. 5: `.50c+.25τ+.25κ`, lag 5) | [`causal_ranker_node`](nodes/causal_ranker.py#L23) (α0.50/β0.25/γ0.25, lag=5) | ✅ exact match |
| **Stage 4 / 10** Memory (SQLite/Neo4j) | `memory_search_node` / `memory_store_node` + [`store_rca_to_neo4j`](tools/pipeline/rca_tools.py#L490) | ✅ implemented |
| **Stage 9** TNR safety gate + rollback (Eq. 6–7, max 3) | `undo_agent_node` + [`route_after_safety_check`](graph.py#L44) | ✅ implemented |
| **"agent think + query KG"** | [`maybe_llm_rerank_with_kg`](tools/pipeline/kg_llm_tools.py#L238) — mode **a** = dump NetworkX graph to LLM; mode **b** = Neo4j GraphRAG loop | ✅ **this is "agent queries the KG"** |
| **KG DB (Neo4j)** | [`Neo4jConnector`](tools/pipeline/neo4j_connector.py#L68) — local `bolt://localhost:7687` → Aura → NetworkX fallback | ✅ your container `neo4j-graphrca` |

The numbers in your pasted text (0.35/0.35/0.15/0.10, ×1.15, 0.50/0.25/0.25, lag=5, depth 5, cap 200, rollback max 3) are **literal constants already in the code**. This pipeline was built; it just is not the one running.

---

## 2. "Agent thinks and queries the KG DB" — exactly what it is

[`maybe_llm_rerank_with_kg`](tools/pipeline/kg_llm_tools.py#L238), gated by env `GRAPHRCA_LLM_KG_MODE`, called from [`rca_node`](nodes/rca.py#L112):

- **Mode `a` (cheap, 1 LLM call):** serializes the full NetworkX dependency graph to JSON, puts it in the prompt next to the deterministic candidate shortlist, and the LLM returns `{root_cause_service, ranked_services, reason}`. Results re-order the existing ranked candidates only (no invented names). ([kg_llm_tools.py:262](tools/pipeline/kg_llm_tools.py#L262))
- **Mode `b` (true agentic, ≤3 rounds):** a **bounded GraphRAG loop**. Each round the LLM emits either:
  - `{action:"query", cypher:"..."}` — validated by [`_is_safe_read_cypher`](tools/pipeline/kg_llm_tools.py#L196): must start with `MATCH/OPTIONAL MATCH/WITH/UNWIND`, must `RETURN`, and **rejects** `CREATE/MERGE/SET/DELETE/DETACH/DROP/REMOVE/CALL/APOC/GDS`. We auto-append `LIMIT`. The query runs against Neo4j and the rows are fed back.
  - `{action:"final", root_cause_service, ranked_services}` — ends the loop.
  ([kg_llm_tools.py:349](tools/pipeline/kg_llm_tools.py#L349))

So "agent think + query in KG DB" is built and write-safe. It is **off by default** and never reached from the swarm path.

---

## 3. "Radius-blast analysis" — exactly what it is

[`backward_bfs_traversal(G, error_service, max_depth=5)`](tools/pipeline/rca_tools.py#L84): a **bounded backward BFS from the highest-alerting service**, walking predecessors (upstream dependencies) up to 5 hops, returning up to **200 rootward paths**. It even separates `CALLS` edges (5-hop budget) from infra edges `COLOCATED/RUNS_ON/DEPLOYED_AS` (2-hop budget). Every visited service is then scored by Eq. 4 (`score_candidate`) and re-ranked by Eq. 5 (`rerank_with_causality`). That "expand outward from the symptom toward the cause" traversal **is** the radius blast.

---

## 4. The gap: the running swarm uses none of this

`scratchpad_swarm` (4 nodes: observer → diagnoser → rca_analyst → guardrail):

- **KG** = in-process **SQLite ScratchPad** triplets (`source/relationship/target`), not Neo4j. ([scratchpad_client.py](tools/scratchpad_client.py))
- **Ranking** = simple `anomaly_score` sum + PageRank tiebreak ([topological_diagnoser.py:103](nodes/topological_diagnoser.py#L103)). No backward-BFS, no causal/temporal re-rank, no silent-failure inference, no KG-query LLM, no persistence.
- It scores well on easy faults only because the multi-telemetry observer now collects traces+pods+metrics+logs and the anomaly weighting surfaces the crashed cause (e.g. `geo`). It has **no temporal/causal reasoning**, so cascades, silent bottlenecks, and multi-hop faults will underperform the full pipeline.

---

## 5. Revised recommendation (replaces the earlier "bounded investigator")

**Do not reinvent. Bridge.** My earlier suggestion (a hand-built LLM investigator node) duplicates work that already exists and is better. Instead, layer the existing machinery onto the swarm in new files (so the running batch is untouched):

**Phase A — radius blast on the in-memory graph (deterministic, 0 extra LLM cost, no Neo4j):**
- New node `nodes/swarm_rca_kg.py`. Convert ScratchPad triplets + the observer's per-service stats into a rich `nx.DiGraph` with node attrs (`error_rate`, `duration_mean_ms`, `span_count`) and edge attrs (`avg_duration_ms`, `edge_type=CALLS`). `build_dag`/`add_node_with_metrics` in [graph_tools.py](tools/pipeline/graph_tools.py) already do this from spans.
- Blast start = the diagnoser's #1 anomaly service. Call `backward_bfs_traversal` → `score_candidate` → `infer_silent_failures` → `rerank_with_causality`. Write top-3 back into swarm state as `verified_root_cause`.
- Net effect: upgrades the swarm from a symptom-weight heuristic to the paper's Eq. 4 + Eq. 5 ranking.

**Phase B — agent-thinks-over-KG (the explicit ask):**
- First enable `GRAPHRCA_LLM_KG_MODE=a` (1 LLM call; graph in prompt) — cheap and safe with gemma4:12b.
- Promote to `=b` (Neo4j GraphRAG Cypher loop) **only after** confirming the 12B reliably emits parseable JSON + valid read-only Cypher on a few faults. The safety filter already blocks writes.

**Phase C — Neo4j persistence + memory (Stage 10):**
- Set `NEO4J_ENABLED=True` (container `neo4j-graphrca` is up). Then `store_graph_to_neo4j` + `store_trace_spans_to_neo4j` + `store_rca_to_neo4j` persist each incident, and `memory_search` retrieves historically similar faults.

**Phase D (optional, later) — run the legacy `pipeline` mode directly on AIOpsLab.** Bigger integration (PipelineState wants span objects/alerts/baselines, not the swarm's telemetry files), so the Phase A–C bridge is lower-risk and gets most of the value first.

---

## 5B. Recommendation v2 — ScratchPad-centered (the SLM-correct path)

**Goal restated:** ScratchPad is the SLM's bounded-context middleware (the reason we run it). "Agent thinks + queries KG" and "radius blast" should run **through ScratchPad**, not through the legacy `kg_llm_tools`/Neo4j path that bypasses it. ScratchPad already has the primitives:

- **Radius blast = `compile_bounded_markdown_view(query, k_hops)`** → [`_apply_query_aware_boost`](ScratchPad/src/engine.py#L477) builds an in-memory graph from active triplets, finds entities named in `query`, and **expands a k-hop BFS neighborhood** around them, boosting that subgraph to the top of a token-budgeted view. That is a bounded radius blast.
- **Agent queries KG = drill-down** → COMPRESSED nodes carry `[COMPRESSED | drill-down id: …]`; the SLM asks to expand a suspect and ScratchPad serves the expansion inside the budget ([`agent_sdk.drill_down`](ScratchPad/src/agent_sdk.py#L20), [engine.py:629](ScratchPad/src/engine.py#L629)).

The swarm's RCA node already consumes the bounded view ([rca_analyst_agent.py:28](nodes/rca_analyst_agent.py#L28)) — just in the weakest form. Three small changes turn it into the full ScratchPad-native design:

| # | Change | File |
|---|---|---|
| 1 | Expose `query` + `k_hops` on `ScratchpadClient.get_view` and pass through to `compile_bounded_markdown_view`. Call it with `query=<problem_desc / top suspect>`, `k_hops=3`. → the bounded view **becomes the radius blast**. | [scratchpad_client.py:246](tools/scratchpad_client.py#L246) |
| 2 | Turn the single `get_view` + one `llm_reason` into a **bounded drill-down loop**: SLM reads view → may emit `drill_down(edge_id)` on a COMPRESSED suspect → ScratchPad serves the expansion → SLM commits `root_cause_service`. Cap rounds (e.g. 3) so it can't blow the step budget. | [rca_analyst_agent.py:28](nodes/rca_analyst_agent.py#L28) |
| 3 | Implement the real `ScratchpadClient.drill_down(edge_id)` (expand that node's triplet neighborhood from SQLite; today it's a stub `return f"Drill down details for {node}"`). | [scratchpad_client.py:273](tools/scratchpad_client.py#L273) |

The deterministic backward-BFS ([`backward_bfs_traversal`](tools/pipeline/rca_tools.py#L84)) stays — but as a **ranker that reads the same ScratchPad triplets** (the diagnoser already builds an `nx.DiGraph` from them), not as a replacement for the middleware. **Neo4j becomes optional**, used only for cross-incident memory (Stage 10), never for the live SLM reasoning context.

**Net effect:** ScratchPad remains the single SLM-facing memory; the radius blast and the agent-query loop are both served through its bounded view + drill-down. The SLM (gemma4:12b) never sees an unbounded context.

**Constraint:** changes 1–3 touch files the running batch (PID 113377) imports live, so they wait for the batch. Only a brand-new `nodes/swarm_rca_kg.py` (the deterministic BFS ranker) is safe to prototype now.

---

## 6. Constraints & risks

- **Do NOT touch `scratchpad_swarm` files or `graph.py` while PID 113377 runs** (it imports them; ~12/86 task dirs at time of writing). Phase A goes in a **new** file and is wired only after the batch finishes.
- **gemma4:12b + mode `b`:** the Cypher loop costs 3+ LLM calls and risks malformed JSON/Cypher from a 12B model (mitigated, not eliminated, by `_extract_json_obj` + `_is_safe_read_cypher`). Measure on 5–10 hard faults before adopting. Mode `a` is the safe default.
- **Neo4j write overhead** raises TTD/TTL/TTA (the metrics AIOpsLab scores). Gate persistence behind a flag so accuracy runs are not penalized by latency.
- **Backward BFS is call-edge-driven.** For trace-less faults (pod/container/scale/k8s), the dependency graph is sparse → small blast radius. The multi-telemetry observer mitigates detection, but for these faults the score's volume/error components carry the signal, not the topology.

---

## 7. Suggested validation

After the current batch completes, run a controlled comparison on the same ~6 hard faults (hotel_res misconfig, k8s_target_port-misconfig, one cascade, one silent bottleneck):

1. swarm-baseline (today)
2. swarm + Phase A (radius blast)
3. swarm + Phase A + Phase B mode `a`

Measure localization accuracy, LLM-call count, and TTD/TTL/TTA delta. Promote each phase only if it buys accuracy without wrecking latency.
