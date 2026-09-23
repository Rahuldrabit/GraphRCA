"""Offline end-to-end validation against REAL telemetry (no kind cluster).

Replays a failed run's exported telemetry (traces/logs/metrics/pods) through the
REAL Observer -> TopologicalDiagnoser -> RCAAnalystAgent, using the real
gemma4-graphrca:12b. Validates the observer/diagnoser/RCA fixes without paying
the ~8 min kind+deploy cost each iteration.
"""
import os, sys, logging, time

TELEMETRY_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "eval/08-09_12-34-11-k8s_target_port-misconfig-localization-1-gemma4-graphrca-12b/graphrca_output/traces"
EXPECTED = (sys.argv[2] if len(sys.argv) > 2 else "user-service").strip().lower()

# --- env BEFORE llm import: force ollama + the fixed config ---
os.environ["MODEL_AGENTS"] = "gemma4-graphrca:12b"
os.environ["URL_AGENTS"] = "http://localhost:11434/v1"
os.environ["API_KEY_AGENTS"] = "ollama"
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["TEMPERATURE_AGENTS"] = "0.0"
os.environ["GRAPHRCA_RCA_MAX_TOKENS"] = "8192"
os.environ["GRAPHRCA_SP_DRILLDOWN"] = "False"      # skip broken-on-gemma4 drill -> single-shot path
os.environ["GRAPHRCA_DIAGNOSER_TOP_K"] = "8"

DB = "/tmp/offline_validate_scratchpad.db"
for p in (DB, DB + "-wal", DB + "-shm"):
    try: os.remove(p)
    except FileNotFoundError: pass
os.environ["SCRATCHPAD_DB_PATH"] = DB

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "GraphRCA_agent"))

from tools.scratchpad_client import ScratchpadClient       # noqa: E402
from nodes.observer_agent import ObserverAgent              # noqa: E402
from nodes.topological_diagnoser import TopologicalDiagnoser  # noqa: E402
from nodes.rca_analyst_agent import RCAAnalystAgent         # noqa: E402
from swarm_state import AIOpsIncidentState                  # noqa: E402

SID = "offline-validate-1"
client = ScratchpadClient(db_path=DB)
client.init_session(SID, goal="localize k8s target_port misconfig")

state = AIOpsIncidentState(
    problem_id="k8s_target_port-misconfig-localization-1",
    task_type="localization",
    problem_description="Errors and latency in the social-network app. Localize the single "
                        "root-cause service (a k8s targetPort misconfiguration makes one service "
                        "unreachable; downstream services log connection failures to it).",
    scratchpad_session_id=SID,
    trace_csv_path=os.path.join(TELEMETRY_DIR, "aiopslab_traces.csv"),
    pod_status_path=os.path.join(TELEMETRY_DIR, "pods.txt"),
    metrics_path=os.path.join(TELEMETRY_DIR, "metrics_summary.csv"),
    logs_path=os.path.join(TELEMETRY_DIR, "logs.txt"),
    suspect_nodes=None, verified_root_cause=None, anomaly_detected=True,
    final_submission=None, retry_count=0, error=None,
)

print(f"\n==== TELEMETRY: {TELEMETRY_DIR} ====", flush=True)
print(f"==== EXPECTED ROOT CAUSE: {EXPECTED} ====\n", flush=True)

# 1) Observer — real parse of REAL telemetry
obs = ObserverAgent(client)
state = obs(state)
print(f"[observer] anomaly_detected={state.get('anomaly_detected')}", flush=True)

# 2) Diagnoser — real ranking
diag = TopologicalDiagnoser(client)
state = diag(state)
suspects = state.get("suspect_nodes") or []
print(f"[diagnoser] SUSPECTS (top-K): {suspects}", flush=True)
print(f"[diagnoser] user-service #1? {bool(suspects) and suspects[0] == 'user-service'}", flush=True)
print(f"[diagnoser] user-service in list? {'user-service' in suspects}", flush=True)

# 3) RCA — real gemma4 (single-shot path)
rca = RCAAnalystAgent(client)
t0 = time.time()
state = rca(state)
dt = time.time() - t0
rc = str(state.get("verified_root_cause") or "").strip().lower()
print(f"\n==== RCA RESULT ({dt:.1f}s) ====", flush=True)
print("verified_root_cause:", state.get("verified_root_cause"), flush=True)
print("EXPECTED:           ", EXPECTED, flush=True)
print("MATCH:", rc == EXPECTED, flush=True)
