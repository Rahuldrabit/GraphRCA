"""Offline end-to-end drill-down confirmation.

Runs the REAL RCAAnalystAgent._drill_down_loop (get_view -> llm_reason ->
_extract_json -> action dispatch -> commit) against a synthetic hotel-reservation
graph with a known root cause (user-service), using the real gemma4-graphrca:12b.
NO kind cluster, NO run_pipeline -> zero conflict with the running batch.
"""
import os, sys, logging, time

# --- MUST precede `llm` import: force ollama + the fixed gemma4 config ---
# (.env defaults to gpt-4.1-nano; load_dotenv(override=False) won't clobber these)
os.environ["MODEL_AGENTS"] = "gemma4-graphrca:12b"
os.environ["URL_AGENTS"] = "http://localhost:11434/v1"
os.environ["API_KEY_AGENTS"] = "ollama"
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["TEMPERATURE_AGENTS"] = "0.0"
os.environ["GRAPHRCA_SP_DRILLDOWN"] = "True"
os.environ["GRAPHRCA_SP_KHOPS"] = "3"
os.environ["GRAPHRCA_SP_VIEW_TOKENS"] = "1200"
os.environ["GRAPHRCA_SP_MAX_ROUNDS"] = "3"

DB = "/tmp/drill_test_scratchpad.db"
for p in (DB, DB + "-wal", DB + "-shm"):
    try: os.remove(p)
    except FileNotFoundError: pass
os.environ["SCRATCHPAD_DB_PATH"] = DB

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.scratchpad_client import ScratchpadClient          # noqa: E402
from nodes.rca_analyst_agent import RCAAnalystAgent           # noqa: E402
from swarm_state import AIOpsIncidentState                    # noqa: E402

SID = "harness-session-1"
client = ScratchpadClient(db_path=DB)
client.init_session(SID, goal="localize latency spike root cause")

# Known root cause = user-service (OOMKilled -> errors propagate downstream).
triplets = [
    {"source": "user-service", "relationship": "emits", "target": "ERROR_RATE_HIGH", "citation_quote": "user-service p95 latency 2840ms, error rate 12%", "source_type": "SERVICE", "target_type": "METRIC"},
    {"source": "user-service", "relationship": "calls", "target": "reservation", "citation_quote": "user-service -> reservation dependency", "source_type": "SERVICE", "target_type": "SERVICE"},
    {"source": "user-service", "relationship": "calls", "target": "profile", "citation_quote": "user-service -> profile dependency", "source_type": "SERVICE", "target_type": "SERVICE"},
    {"source": "reservation", "relationship": "calls", "target": "payment", "citation_quote": "reservation -> payment", "source_type": "SERVICE", "target_type": "SERVICE"},
    {"source": "reservation", "relationship": "calls", "target": "geo", "citation_quote": "reservation -> geo", "source_type": "SERVICE", "target_type": "SERVICE"},
    {"source": "payment", "relationship": "calls", "target": "mongodb", "citation_quote": "payment -> mongodb", "source_type": "SERVICE", "target_type": "DATASTORE"},
    {"source": "profile", "relationship": "calls", "target": "mongodb", "citation_quote": "profile -> mongodb", "source_type": "SERVICE", "target_type": "DATASTORE"},
    {"source": "reservation", "relationship": "emits", "target": "LATENCY_HIGH", "citation_quote": "reservation p95 1900ms (downstream of user-service)", "source_type": "SERVICE", "target_type": "METRIC"},
    {"source": "payment", "relationship": "emits", "target": "OK", "citation_quote": "payment error rate 0.1%", "source_type": "SERVICE", "target_type": "METRIC"},
    {"source": "geo", "relationship": "emits", "target": "OK", "citation_quote": "geo p95 80ms", "source_type": "SERVICE", "target_type": "METRIC"},
    {"source": "mongodb", "relationship": "emits", "target": "OK", "citation_quote": "mongodb connections normal", "source_type": "DATASTORE", "target_type": "METRIC"},
    {"source": "user-service", "relationship": "has", "target": "POD_RESTART", "citation_quote": "user-service pod restarted 3x in 5min", "source_type": "SERVICE", "target_type": "EVENT"},
    {"source": "user-service", "relationship": "logs", "target": "OOM_KILL", "citation_quote": "user-service container OOMKilled", "source_type": "SERVICE", "target_type": "LOG"},
    {"source": "profile", "relationship": "emits", "target": "OK", "citation_quote": "profile healthy", "source_type": "SERVICE", "target_type": "METRIC"},
]
client.commit_triplets(SID, "ObserverAgent", triplets)

state = AIOpsIncidentState(
    problem_id="harness-latency-1",
    task_type="localization",
    problem_description="Latency spike on hotel-reservation checkout. user-service, reservation, payment flagged by topological analysis. Identify the single root cause service.",
    scratchpad_session_id=SID,
    trace_csv_path=None, pod_status_path=None, metrics_path=None, logs_path=None,
    suspect_nodes=["user-service", "reservation", "payment"],
    verified_root_cause=None,
    anomaly_detected=True,
    final_submission=None,
    retry_count=0,
    error=None,
)

agent = RCAAnalystAgent(client)
t0 = time.time()
print("\n==== RUNNING REAL _drill_down_loop against gemma4-graphrca:12b ====", flush=True)
out = agent(state)
dt = time.time() - t0
print(f"\n==== RESULT ({dt:.1f}s) ====", flush=True)
print("verified_root_cause:", out.get("verified_root_cause"), flush=True)
print("EXPECTED:           user-service", flush=True)
print("MATCH:", str(out.get("verified_root_cause", "")).strip().lower() == "user-service", flush=True)
