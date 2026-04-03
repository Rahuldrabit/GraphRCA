# GraphRCA — LangGraph Autonomous SRE Pipeline

A multi-agent system for root cause analysis, anomaly detection, and autonomous mitigation in distributed systems. Implements the **STRATUS (NeurIPS 2025)** framework pillars using a **LangGraph StateGraph** orchestration.

## Architecture

```
trace_ingest → graph_builder → detection → memory_search
  → rca → causal_ranker → log_analysis → mitigation → safety_check
    → (rollback loop → rca)  OR  memory_store → report_generation → END
```

### 4 Pillars

| Pillar | Description | Node |
|--------|-------------|------|
| 1 — TNR | Transactional No-Regression safety check + auto rollback | `undo_agent` / `safety_check` |
| 2 — Causal | Advanced causal inference & temporal ordering | `causal_ranker` |
| 3 — Observability | eBPF + log pattern deep-dive | `log_pattern` |
| 4 — Benchmark | AIOpsLab integration (detection / localization / analysis / mitigation) | `agent_aiopslab` |

---

## Project Structure

```
GraphRCA/
├── graph.py              # LangGraph StateGraph (11 nodes, rollback loop)
├── state.py              # PipelineState TypedDict — shared state
├── llm.py                # LLM client factory + justification logger
├── run_pipeline.py       # CLI entry point + AIOpsLab integration
├── agent_aiopslab.py     # Threaded AIOpsLab agent (mirrors Stratus pattern)
├── nodes/
│   ├── trace_ingest.py   # CSV parsing, dedup, validation
│   ├── graph_builder.py  # Service dependency DAG + PageRank
│   ├── detection.py      # EWMA anomaly detection
│   ├── memory_rag.py     # Historical incident search & storage
│   ├── rca.py            # Backward BFS root cause analysis
│   ├── causal_ranker.py  # Pillar 2: causal inference ranking
│   ├── log_pattern.py    # Pillar 3: log pattern analysis
│   ├── mitigation.py     # Mitigation action generation
│   ├── undo_agent.py     # Pillar 1: TNR safety check & rollback
│   └── report_generation.py  # Incident report + ITBench output
├── tools/
│   ├── causal_tools.py   # Causal inference utilities
│   ├── ebpf_tools.py     # Pillar 3: eBPF observability
│   ├── grafana_tools.py  # Grafana metrics integration
│   ├── kube_tools.py     # Kubernetes action execution
│   └── safety_tools.py   # Pillar 1: rollback & safety verification
├── .env                  # Secrets (API keys, Neo4j, benchmark config)
├── quickrun.sh           # Quick-start script
├── test_graphrca.sh      # Per-task eval orchestrator (cluster setup + run + log)
└── eval/
    ├── eval_tasks.yaml   # Full AIOpsLab task list (detection/localization/analysis/mitigation)
    ├── eval.py           # Batch eval runner — iterates all tasks from YAML
    └── clean_ansi_from_log.py  # Strip ANSI escape codes from run.log
```

---

## Setup

### 1. Create & activate virtual environment

```bash
cd GraphRCA
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`

Fill in your keys in the `.env` file:

```env
# LLM
PROVIDER_AGENTS=openai
MODEL_AGENTS=gpt-4.1-nano
API_KEY_AGENTS=sk-...
OPENAI_API_KEY=sk-...

# Neo4j (optional)
NEO4J_ENABLED=True
NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>

# Benchmark
BENCHMARK=AIOpsLab             # or ITBench
AIOPSLAB_ROOT=$HOME/AIOpsLab
```

---

## Usage

### Standalone — trace CSV files

```bash
# With Neo4j
PYTHONPATH="stratus/src:$PWD" python -m GraphRCA.run_pipeline \
  --trace-dir ./stratus/trace_output -v

# Without Neo4j
PYTHONPATH="stratus/src:$PWD" python -m GraphRCA.run_pipeline \
  --trace-dir ./stratus/trace_output --no-neo4j -v

# Custom output directory
PYTHONPATH="stratus/src:$PWD" python -m GraphRCA.run_pipeline \
  --trace-dir ./stratus/trace_output --output-dir ./my_output
```

### AIOpsLab Benchmark Mode (Pillar 4)

```bash
PYTHONPATH="AIOpsLab:stratus/src:$PWD" python -m GraphRCA.run_pipeline \
  --aiopslab --problem-id misconfig_app_hotel_res-detection-1 -v
```

Supported task types (auto-detected from problem ID): `detection`, `localization`, `analysis`, `mitigation`.

---

## Evaluation Workflow

### Single-task eval (recommended)

Run from the repo root (`GraphRCA/`):

```bash
# New kind cluster + run
./GraphRCA_agent/test_graphrca.sh misconfig_app_hotel_res-detection-1

# Reuse existing cluster (-p), explicit arch (-r)
./GraphRCA_agent/test_graphrca.sh -p -r x86 misconfig_app_hotel_res-detection-1

# Custom output directory
./GraphRCA_agent/test_graphrca.sh -d my_output/run1 misconfig_app_hotel_res-detection-1

# Only set up the kind cluster without running a task
./GraphRCA_agent/test_graphrca.sh -s
```

Options:

| Flag | Description |
|------|-------------|
| `-p` | Preserve existing cluster (skip delete/create) |
| `-r x86\|arm` | Override architecture detection |
| `-d <dir>` | Custom output directory (default: `eval/<MM-DD_HH-MM-SS>-<task>`) |
| `-s` | Cluster setup only — skip running the task |
| `-h` | Show help |

Output lands in `eval/<MM-DD_HH-MM-SS>-<task_name>/`:

```
eval/04-03_14-30-00-misconfig_app_hotel_res-detection-1/
├── run.log                        # Full stdout+stderr captured via tee
└── graphrca_output/
    ├── graphrca.log               # Python logging output
    ├── incident_report.json       # Main analysis report
    ├── llm_justification.jsonl    # All LLM calls (prompt, response, tokens)
    ├── agent_output_0.json        # Per-attempt submit payload
    ├── run_logs.txt               # Per-attempt summary (start/end time, validation)
    ├── graphrca_run_stats.json    # Token usage + run count stats
    ├── eval_results.json          # AIOpsLab evaluation metrics
    └── reports/
        ├── diagnosis_struct_out.json
        └── remediation_struct_out.json
```

At the end of each run, the terminal prints a Stratus-format evaluation banner:

```
Validation result: {'success': True, 'issues': []}
######### VALIDATION SUCCESSFUL #########
Output written to: eval/.../graphrca_output/agent_output_0.json
== Evaluation ==
Correct detection: Yes
Results:
{'Detection Accuracy': 'Correct', 'TTD': 1205.67, 'steps': 30, 'in_tokens': 236415, 'out_tokens': 382}
== Fault Recovery ==
Recovering for service: frontend | namespace: hotel-reservation
```

### Batch eval (all tasks)

```bash
cd GraphRCA/GraphRCA_agent
python eval/eval.py
```

This reads `eval/eval_tasks.yaml` and calls `test_graphrca.sh` sequentially for each task across all four task types. Each task spins up a fresh kind cluster unless you edit `eval.py` to pass `-p`.

### ANSI log cleaning

`test_graphrca.sh` automatically strips ANSI escape codes from `run.log` after each run. To clean a log manually:

```bash
python eval/clean_ansi_from_log.py path/to/run.log
```

### Quick Run

```bash
bash GraphRCA/quickrun.sh
```

---

## Output

**Standalone runs** produce a timestamped directory under `GraphRCA_output/<timestamp>/`.
**Eval runs** (via `test_graphrca.sh`) produce `eval/<MM-DD_HH-MM-SS>-<task>/graphrca_output/`.

| File | Description |
|------|-------------|
| `graphrca.log` | Full pipeline log (all nodes, timings, debug) |
| `incident_report.json` | Main report (root cause, alerts, actions, health scores) |
| `llm_justification.jsonl` | Every LLM call logged (prompt, response, tokens, timing) |
| `agent_output_N.json` | Submit payload for each attempt (N=0,1,2…) |
| `run_logs.txt` | Per-attempt summary (start/end time, task type, validation, reflection) |
| `graphrca_run_stats.json` | Token totals (prompt/completion/total), run count, elapsed seconds |
| `eval_results.json` | Raw AIOpsLab evaluation metrics (Detection Accuracy, TTD, steps, tokens) |
| `reports/diagnosis_struct_out.json` | ITBench-compatible diagnosis output |
| `reports/remediation_struct_out.json` | ITBench-compatible remediation output |

### LLM Justification Log format

Every call to `llm_reason()` is appended as a JSON line to `llm_justification.jsonl`:

```json
{
  "call_id": 1,
  "timestamp": "2025-03-24T00:30:57Z",
  "caller": "report_gen",
  "model": "gpt-4.1-nano",
  "system_prompt": "...",
  "user_prompt": "...",
  "response": "...",
  "tokens": {"prompt_tokens": 412, "completion_tokens": 238, "total_tokens": 650},
  "elapsed_seconds": 2.14
}
```

---

## Neo4j Management

```bash
# Clear ALL Neo4j data (nodes + relationships)
PYTHONPATH="stratus/src:$PWD" python -m GraphRCA.run_pipeline --clear-neo4j

# Disable Neo4j for a single run
python -m GraphRCA.run_pipeline --trace-dir ./traces --no-neo4j
```

---

## AIOpsLab Integration Pattern

GraphRCA mirrors the Stratus `StratusAgent_AIOpsLab` threaded agent pattern:

```
AIOpsLab Orchestrator
       │
       │  get_action(observation)
       ▼
  GraphRCAAgent (thread)
       │  _communicator() generator
       │  prompt_semaphore / command_semaphore
       │
       ├── _fetch_traces()      → generator.send('get_traces(...)')
       ├── _run_pipeline()      → full LangGraph StateGraph
       └── _submit_results()    → generator.send('submit(...)')
```

Task-type routing:
- `detection`    → `submit("Yes")` / `submit("No")`
- `localization` → `submit(["service-a", "service-b"])`
- `analysis`     → `submit({"system_level": "...", "fault_type": "..."})`
- `mitigation`   → `submit({"fix": "...", "submit": True})`

---

## Key Concepts

### EWMA Anomaly Detection
Exponential Weighted Moving Average baseline per service. Alerts triggered when z-score exceeds threshold. Severity: CRITICAL / HIGH / MEDIUM / LOW.

### Backward BFS RCA
Root cause scored by 3 signals: `error_rate × 0.5 + latency_ratio × 0.3 + call_volume × 0.2`. BFS traversal walks upstream from the error service.

### TNR Rollback Loop (Pillar 1)
After mitigation, health score is computed. If `health_score_after < health_score_before`, rollback is triggered and RCA restarts (up to 3 times).

### Causal Ranking (Pillar 2)
Impact formula: `confidence×0.4 + error_rate×0.3 + downstream_ratio×0.2 + call_volume×0.1`
