# GraphRCA — LangGraph Autonomous SRE Pipeline

A **standalone** multi-agent system for root cause analysis, anomaly detection, and autonomous mitigation in distributed systems. Implements the **STRATUS (NeurIPS 2025)** methodology pillars using a **LangGraph StateGraph** orchestration.

**GraphRCA is fully independent** - it follows the STRATUS methodology but has its own codebase and does not require the STRATUS repository.

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
│   ├── pipeline/          # GraphRCA's own pipeline tools (no stratus dependency)
│   │   ├── ingest_tools.py      # Trace ingestion & validation
│   │   ├── graph_tools.py       # Graph building & Neo4j storage
│   │   ├── detection_tools.py   # EWMA anomaly detection
│   │   ├── rca_tools.py         # Root cause analysis algorithms
│   │   ├── memory_tools.py      # SQLite memory store
│   │   ├── mitigation_tools.py  # Mitigation planning
│   │   └── neo4j_connector.py   # Neo4j connection wrapper
│   ├── causal_tools.py   # Causal inference utilities
│   ├── ebpf_tools.py     # Pillar 3: eBPF observability
│   ├── grafana_tools.py  # Grafana metrics integration
│   ├── kube_tools.py     # Kubernetes action execution
│   └── safety_tools.py   # Pillar 1: rollback & safety verification
├── .env                  # Secrets (API keys, Neo4j, benchmark config)
├── run_graphrca.sh       # Unified runner (recommended)
├── quickrun.sh           # Quick-start script (standalone traces)
├── test_graphrca.sh      # Per-task eval orchestrator (legacy)
└── eval/
    ├── eval_tasks.yaml   # Full AIOpsLab task list (detection/localization/analysis/mitigation)
    ├── eval.py           # Batch eval runner — iterates all tasks from YAML
    └── clean_ansi_from_log.py  # Strip ANSI escape codes from run.log
```

---

## Setup

### 1. Prerequisites

- **Docker Desktop** running with Kubernetes enabled, OR
- **kind** installed for creating Kubernetes clusters
- Python 3.10+ 
- Access to AIOpsLab repository (for benchmark tasks)

### 2. Create & activate virtual environment

```bash
cd GraphRCA_agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure `.env`

Fill in your keys in the `.env` file:

```env
# LLM
PROVIDER_AGENTS=openai
MODEL_AGENTS=gpt-4.1-nano
API_KEY_AGENTS=sk-...
OPENAI_API_KEY=sk-...

# Neo4j (optional - can run without Neo4j)
NEO4J_ENABLED=False            # Set to True if you have Neo4j
NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>

# Benchmark
BENCHMARK=AIOpsLab             # or ITBench
AIOPSLAB_ROOT=../AIOpsLab      # Path to AIOpsLab repository
```

---

## How to Run

### Quick Start — Unified Script (Recommended)

The `run_graphrca.sh` script combines all functionality into one unified interface:

```bash
cd GraphRCA_agent

# Run an AIOpsLab benchmark task (creates fresh kind cluster)
./run_graphrca.sh misconfig_app_hotel_res-detection-1

# Use existing cluster (skip cluster setup - faster for multiple runs)
./run_graphrca.sh -p misconfig_app_hotel_res-detection-1

# Specify architecture explicitly (x86 or arm)
./run_graphrca.sh -r x86 misconfig_app_hotel_res-detection-1

# Custom output directory
./run_graphrca.sh -d ./my_output task_name

# Verbose mode
./run_graphrca.sh -v misconfig_app_hotel_res-detection-1

# Standalone trace analysis (no AIOpsLab - analyze existing traces)
./run_graphrca.sh -t ./trace_output

# Setup cluster only (no task execution)
./run_graphrca.sh -s

# Clear Neo4j database (if using Neo4j)
./run_graphrca.sh --clear-neo4j
```

#### All Options

| Flag | Description |
|------|-------------|
| `-h, --help` | Show help message |
| `-p, --preserve` | Use existing cluster (skip delete/create) |
| `-r, --arch <x86\|arm>` | Override architecture detection |
| `-d, --output-dir <dir>` | Custom output directory |
| `-t, --trace-dir <dir>` | Run standalone trace analysis (skip AIOpsLab) |
| `-s, --setup-only` | Setup cluster only, don't run task |
| `-v, --verbose` | Verbose output |
| `--no-neo4j` | Skip Neo4j operations |
| `--no-spans` | Skip span processing |
| `--clear-neo4j` | Clear all Neo4j data and exit |

---

### Manual Python Commands (Advanced)

#### Standalone — trace CSV files

```bash
# With Neo4j
PYTHONPATH="$PWD" python -m GraphRCA_agent.run_pipeline \
  --trace-dir ./trace_output -v

# Without Neo4j (default if NEO4J_ENABLED=False in .env)
PYTHONPATH="$PWD" python -m GraphRCA_agent.run_pipeline \
  --trace-dir ./trace_output --no-neo4j -v

# Custom output directory
PYTHONPATH="$PWD" python -m GraphRCA_agent.run_pipeline \
  --trace-dir ./trace_output --output-dir ./my_output
```

#### AIOpsLab Benchmark Mode (Pillar 4)

```bash
# Set PYTHONPATH to include AIOpsLab
PYTHONPATH="../AIOpsLab:$PWD" python -m GraphRCA_agent.run_pipeline \
  --aiopslab --problem-id misconfig_app_hotel_res-detection-1 -v
```

**Supported task types** (auto-detected from problem ID): `detection`, `localization`, `analysis`, `mitigation`.

---

## Available AIOpsLab Tasks

You can run any of these tasks with `./run_graphrca.sh <task-name>`:

### Hotel Reservation Application Tasks

**Detection Tasks:**
- `misconfig_app_hotel_res-detection-1` - Detect misconfiguration in hotel-res service
- `pod_failure_hotel_res-detection-1` - Detect pod failure

**Localization Tasks:**
- `misconfig_app_hotel_res-localization-1` - Localize misconfiguration root cause
- `pod_failure_hotel_res-localization-1` - Localize failed pod

**Analysis Tasks:**
- `misconfig_app_hotel_res-analysis-1` - Analyze misconfiguration impact
- `pod_failure_hotel_res-analysis-1` - Analyze pod failure impact

**Mitigation Tasks:**
- `misconfig_app_hotel_res-mitigation-1` - Mitigate misconfiguration
- `pod_failure_hotel_res-mitigation-1` - Mitigate pod failure

### Social Network Application Tasks

**Detection Tasks:**
- `scale_pod_zero_social_net-detection-1` - Detect zero-scaled pods
- `assign_to_non_existent_node_social_net-detection-1` - Detect non-existent node assignment

**Localization Tasks:**
- `scale_pod_zero_social_net-localization-1` - Localize zero-scale issue
- `assign_to_non_existent_node_social_net-localization-1` - Localize node assignment issue

**Analysis Tasks:**
- `scale_pod_zero_social_net-analysis-1` - Analyze zero-scale impact
- `assign_to_non_existent_node_social_net-analysis-1` - Analyze node assignment impact

**Mitigation Tasks:**
- `scale_pod_zero_social_net-mitigation-1` - Mitigate zero-scale issue
- `assign_to_non_existent_node_social_net-mitigation-1` - Mitigate node assignment issue

### MongoDB Authentication Tasks

- `auth_miss_mongodb-detection-1` - Detect missing MongoDB authentication
- `auth_miss_mongodb-localization-1` - Localize auth issue
- `auth_miss_mongodb-analysis-1` - Analyze auth issue
- `auth_miss_mongodb-mitigation-1` - Mitigate auth issue

### Container Kill Tasks

- `container_kill-detection` - Detect container kill events
- `container_kill-localization` - Localize killed container

**Example:**
```bash
# Run detection task
./run_graphrca.sh misconfig_app_hotel_res-detection-1

# Run mitigation task with existing cluster (faster)
./run_graphrca.sh -p misconfig_app_hotel_res-mitigation-1

# Run any social network task
./run_graphrca.sh scale_pod_zero_social_net-detection-1
```

---

## Evaluation Workflow

### Single-task eval

Run from the repo root (`GraphRCA/`):

```bash
# Using unified script (recommended)
./GraphRCA_agent/run_graphrca.sh misconfig_app_hotel_res-detection-1

# With options
./GraphRCA_agent/run_graphrca.sh -p -r x86 -v misconfig_app_hotel_res-detection-1
```

#### Legacy scripts (still available)

```bash
# test_graphrca.sh — original eval script
./GraphRCA_agent/test_graphrca.sh misconfig_app_hotel_res-detection-1

# quickrun.sh — standalone traces only
./GraphRCA_agent/quickrun.sh -t ./traces
```

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

`run_graphrca.sh` and `test_graphrca.sh` automatically strip ANSI escape codes from `run.log` after each run. To clean a log manually:

```bash
python eval/clean_ansi_from_log.py path/to/run.log
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
./run_graphrca.sh --clear-neo4j

# Or manually:
PYTHONPATH="$PWD" python -m GraphRCA_agent.run_pipeline --clear-neo4j

# Disable Neo4j for a single run
./run_graphrca.sh --no-neo4j -t ./traces
```

**Note:** Neo4j is optional. GraphRCA works without it by setting `NEO4J_ENABLED=False` in `.env`.

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

---

## Important Notes

### Standalone Architecture

**GraphRCA is completely independent from STRATUS.** It implements the STRATUS methodology but has its own:
- Pipeline tools in `tools/pipeline/` (ingest, graph, detection, RCA, memory, mitigation, Neo4j)
- No dependency on the STRATUS repository or imports
- Can run entirely standalone with just AIOpsLab for benchmarking

### Cluster Requirements

- First-time runs may take 10-15 minutes as Docker images are pulled
- Use `-p` flag for subsequent runs to reuse the existing cluster (much faster)
- MongoDB pods require persistent volumes - ensure your cluster has storage provisioning
- If deployment times out, try running again with `-p` to use the existing cluster

### Troubleshooting

**Pods stuck in Pending/CrashLoopBackOff:**
```bash
# Check pod status
kubectl get pods -n test-hotel-reservation

# Check specific pod
kubectl describe pod <pod-name> -n test-hotel-reservation

# Delete cluster and start fresh
kind delete cluster
./run_graphrca.sh <task-name>
```

**No module named 'GraphRCA_agent':**
- Make sure you're running from the `GraphRCA_agent/` directory
- Or use `PYTHONPATH="$PWD"` when running manually

**AIOpsLab not found:**
- Check `AIOPSLAB_ROOT` in `.env` points to the correct path
- Default is `../AIOpsLab` (one directory up from GraphRCA_agent/)

### Performance Tips

1. **Reuse clusters**: Use `-p` flag for multiple runs
2. **Skip Neo4j**: Set `NEO4J_ENABLED=False` if not needed
3. **Verbose mode**: Use `-v` for debugging, omit for cleaner output
4. **Architecture**: Explicitly set `-r x86` or `-r arm` to skip auto-detection

---

## License

See LICENSE file in repository root.

## Citation

If you use GraphRCA in your research, please cite the STRATUS paper:

```bibtex
@inproceedings{stratus2025,
  title={STRATUS: Autonomous SRE with Transactional No-Regression},
  booktitle={NeurIPS 2025},
  year={2025}
}
```
