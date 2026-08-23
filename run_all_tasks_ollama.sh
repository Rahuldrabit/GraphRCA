#!/usr/bin/env bash
# run_all_tasks_ollama.sh — Run every task in eval/eval_tasks.yaml with a local ollama model
#
# Usage:
#   ./run_all_tasks_ollama.sh                              # all task types, gemma4:12b
#   ./run_all_tasks_ollama.sh --model deepseek-r1:8b       # all task types, deepseek
#   ./run_all_tasks_ollama.sh --types detection,localization
#   ./run_all_tasks_ollama.sh --model gemma4:12b --preserve
#   ./run_all_tasks_ollama.sh --model deepseek-r1:8b --no-neo4j --verbose
#   ./run_all_tasks_ollama.sh --dry-run                    # print tasks, don't run
#
# Options:
#   --model  <name>        ollama model (default: gemma4:12b)
#   --types  <list>        comma-separated task types to run
#                          choices: detection,localization,analysis,mitigation
#                          default: all four
#   --mode   <mode>        agent mode: pipeline | multi_agent | scratchpad_swarm
#   --preserve             reuse existing kind cluster (skip teardown between tasks)
#   --no-neo4j             disable Neo4j for all tasks
#   --verbose / -v         verbose logging
#   --dry-run              print the task list and exit (no execution)
#   --continue-on-error    keep running after a task fails (default: stop on first failure)
#   --delay  <seconds>     wait N seconds between tasks (default: 10)
#   -h / --help            show this help
#
# Summary JSON is written to:
#   eval/all_tasks_<timestamp>_<model>/summary.json

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VENV="$SCRIPT_DIR/venv"
AIOPSLAB_ROOT="${AIOPSLAB_ROOT:-$REPO_ROOT/AIOpsLab}"
EVAL_TASKS_FILE="$REPO_ROOT/eval/eval_tasks.yaml"

# ── Colours ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[all-tasks]${NC} $*"; }
success() { echo -e "${GREEN}[all-tasks]${NC} $*"; }
warn()    { echo -e "${YELLOW}[all-tasks]${NC} $*"; }
error()   { echo -e "${RED}[all-tasks]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}$*${NC}\n"; }

# ── Defaults ──────────────────────────────────────────────────────────────────

OLLAMA_MODEL="gemma4:12b"
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_API_KEY="ollama"
TASK_TYPES="detection,localization,analysis,mitigation"
AGENT_MODE=""
PRESERVE_CLUSTER=0
NO_NEO4J=""
VERBOSE=""
DRY_RUN=0
CONTINUE_ON_ERROR=0
INTER_TASK_DELAY=10

# ── Argument Parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)   grep '^#' "$0" | head -40 | sed 's/^# \?//'; exit 0 ;;
        --model)     OLLAMA_MODEL="$2"; shift ;;
        --types)     TASK_TYPES="$2"; shift ;;
        --mode)      AGENT_MODE="$2"; shift ;;
        --preserve|-p) PRESERVE_CLUSTER=1 ;;
        --no-neo4j)  NO_NEO4J="true" ;;
        --verbose|-v) VERBOSE="true" ;;
        --dry-run)   DRY_RUN=1 ;;
        --continue-on-error) CONTINUE_ON_ERROR=1 ;;
        --delay)     INTER_TASK_DELAY="$2"; shift ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ── Validate YAML file ────────────────────────────────────────────────────────

if [[ ! -f "$EVAL_TASKS_FILE" ]]; then
    error "eval_tasks.yaml not found at: $EVAL_TASKS_FILE"
    exit 1
fi

# ── venv + Python ─────────────────────────────────────────────────────────────

if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
fi

PYTHON_BIN=""
if   [[ -x "$VENV/bin/python" ]];         then PYTHON_BIN="$VENV/bin/python"
elif command -v python  >/dev/null 2>&1;   then PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1;   then PYTHON_BIN="python3"
else error "No Python interpreter found"; exit 127
fi

# ── Load base .env ────────────────────────────────────────────────────────────

ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

# ── Override LLM settings → ollama ────────────────────────────────────────────

export PROVIDER_AGENTS="openai"
export MODEL_AGENTS="$OLLAMA_MODEL"
export URL_AGENTS="$OLLAMA_BASE_URL"
export API_KEY_AGENTS="$OLLAMA_API_KEY"
export OPENAI_API_KEY="$OLLAMA_API_KEY"

export PROVIDER_TOOLS="openai"
export MODEL_TOOLS="$OLLAMA_MODEL"
export URL_TOOLS="$OLLAMA_BASE_URL"
export API_KEY_TOOLS="$OLLAMA_API_KEY"

[[ -n "$NO_NEO4J" ]]    && export NEO4J_ENABLED="False"
[[ -n "$AGENT_MODE" ]]  && export GRAPHRCA_AGENT_MODE="$AGENT_MODE"
export GRAPHRCA_LLM_KG_MODE="${GRAPHRCA_LLM_KG_MODE:-}"

# ── PYTHONPATH ────────────────────────────────────────────────────────────────

export PYTHONPATH="$AIOPSLAB_ROOT:$REPO_ROOT:$SCRIPT_DIR:${PYTHONPATH:-}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

# ── Architecture ──────────────────────────────────────────────────────────────

detect_arch() {
    case "$(uname -m)" in
        x86_64) echo 'x86' ;; arm*|aarch64) echo 'arm' ;; *) echo 'x86' ;;
    esac
}
ARCH="$(detect_arch)"

# ── Parse task list from YAML (pure bash, no python dependency here) ──────────
# Uses a tiny Python one-liner so we don't need yq installed.

read_tasks() {
    local types_arg="$1"
    "$PYTHON_BIN" - <<PYEOF
import yaml, sys

with open("$EVAL_TASKS_FILE") as f:
    cfg = yaml.safe_load(f)

types = [t.strip() for t in "$types_arg".split(",")]
for t in types:
    tasks = cfg.get(t, []) or []
    for task in tasks:
        print(f"{t}:{task}")
PYEOF
}

# Collect tasks into an array
mapfile -t TASK_LINES < <(read_tasks "$TASK_TYPES")

if [[ ${#TASK_LINES[@]} -eq 0 ]]; then
    error "No tasks found for types: $TASK_TYPES"
    exit 1
fi

# ── Dry Run ───────────────────────────────────────────────────────────────────

if [[ $DRY_RUN -eq 1 ]]; then
    header "DRY RUN — Task List (model: $OLLAMA_MODEL)"
    printf "%-12s  %s\n" "TYPE" "TASK"
    printf "%-12s  %s\n" "────────────" "──────────────────────────────────────────────"
    for line in "${TASK_LINES[@]}"; do
        type="${line%%:*}"
        task="${line##*:}"
        printf "%-12s  %s\n" "$type" "$task"
    done
    echo ""
    info "Total: ${#TASK_LINES[@]} tasks"
    info "Model: $OLLAMA_MODEL"
    info "Agent mode: ${GRAPHRCA_AGENT_MODE:-scratchpad_swarm (from .env)}"
    exit 0
fi

# ── Check ollama ──────────────────────────────────────────────────────────────

info "Checking ollama at $OLLAMA_BASE_URL ..."
if ! curl -sf "${OLLAMA_BASE_URL%/v1}/api/tags" >/dev/null 2>&1; then
    warn "ollama not reachable at ${OLLAMA_BASE_URL%/v1} — pipeline will fail on LLM calls."
    warn "Start it with:  ollama serve"
    warn "Pull models:    ollama pull $OLLAMA_MODEL"
fi

# ── Run directory ─────────────────────────────────────────────────────────────

MODEL_SLUG="${OLLAMA_MODEL//:/-}"
RUN_TIMESTAMP="$(date +"%m-%d_%H-%M-%S")"
RUN_DIR="$REPO_ROOT/eval/all_tasks_${RUN_TIMESTAMP}_${MODEL_SLUG}"
mkdir -p "$RUN_DIR"

SUMMARY_FILE="$RUN_DIR/summary.json"
RUN_LOG="$RUN_DIR/run_all.log"

# ── Cluster setup helpers ─────────────────────────────────────────────────────

setup_cluster() {
    info "Setting up kind cluster (arch: $ARCH)..."
    kind delete cluster --name kind 2>/dev/null || true
    local cfg="$AIOPSLAB_ROOT/kind/kind-config-${ARCH}.yaml"
    if [[ ! -f "$cfg" ]]; then
        error "Kind config not found: $cfg"; exit 1
    fi
    kind create cluster --config "$cfg"

    # Pre-load app images into the kind node so pods never cold-pull at deploy
    # time. Each task wipes the cluster (kind delete + create), so containerd's
    # image cache is lost every run — without this the DSB microservice images
    # cold-pull over the network and non-deterministically exceed the readiness
    # budget. Images are already in the persistent docker daemon cache, so
    # `kind load` is disk-bound (fast). Gated by app so we don't load the other
    # app's multi-GB images on every task. Best-effort: a miss just falls back
    # to a deploy-time pull (which has the full readiness budget).
    _ensure_pulled() {
        local img="$1"
        [[ -z "$img" ]] && return 0
        if ! docker image inspect "$img" >/dev/null 2>&1; then
            info "Pre-pulling $img ..."
            timeout 120 docker pull "$img" >/dev/null 2>&1 \
                || warn "pull failed/slow for $img (will retry at deploy time)"
        fi
        if docker image inspect "$img" >/dev/null 2>&1; then
            kind load docker-image "$img" >/dev/null 2>&1 || true
        fi
    }

    local HOTEL_IMAGES=(
        "deathstarbench/hotel-reservation:latest"
        "hashicorp/consul:latest"
        "jaegertracing/all-in-one:latest"
    )
    local SOCIAL_IMAGES=(
        "deathstarbench/social-network-microservices:latest"
        # helm chart pins these two to :xenial, not :latest (:latest doesn't
        # exist on Docker Hub — pre-pull always failed/wasted 120s per image).
        "yg397/media-frontend:xenial"
        "yg397/openresty-thrift:xenial"
        # yinfangchen/social-otel:latest is always commented out in every
        # chart's values.yaml (dead ref, never deployed) — only -regress is used.
        "yinfangchen/social-otel-regress:latest"
    )
    local COMMON_IMAGES=(
        "alpine/git:latest"
        # openebs-operator.yaml is applied fresh every task and blocks on
        # wait_for_ready("openebs"); since kind delete+create wipes the node's
        # containerd cache each task, a slow/throttled registry pull here can
        # blow the 1200s readiness budget and take the whole batch down.
        "openebs/provisioner-localpv:3.4.0"
        "openebs/node-disk-manager:2.1.0"
        "openebs/node-disk-exporter:2.1.0"
        "openebs/node-disk-operator:2.1.0"
    )

    case "$TASK_NAME" in
        *hotel_res*)
            for img in "${HOTEL_IMAGES[@]}" "${COMMON_IMAGES[@]}"; do _ensure_pulled "$img"; done ;;
        *social_network*|*social*)
            for img in "${SOCIAL_IMAGES[@]}" "${COMMON_IMAGES[@]}"; do _ensure_pulled "$img"; done ;;
        *)
            for img in "${HOTEL_IMAGES[@]}" "${SOCIAL_IMAGES[@]}" "${COMMON_IMAGES[@]}"; do _ensure_pulled "$img"; done ;;
    esac

    # astronomy-shop: heavy multi-registry deploy (~38 images). Render the
    # remote chart to get the EXACT current image set (robust to appVersion
    # bumps) and pre-load each. Gated on the task name so non-astronomy tasks
    # skip this expensive step.
    if [[ "$TASK_NAME" == *astronomy* ]] && command -v helm >/dev/null 2>&1; then
        helm repo add open-telemetry \
            "https://open-telemetry.github.io/opentelemetry-helm-charts" >/dev/null 2>&1 || true
        helm repo update open-telemetry >/dev/null 2>&1 || true
        local as_render as_count=0 img
        as_render="$(timeout 120 helm template astronomy-shop \
            open-telemetry/opentelemetry-demo -n astronomy-shop 2>/dev/null || true)"
        if [[ -n "$as_render" ]]; then
            while IFS= read -r img; do
                img="${img//\'/}"            # strip helm-template single quotes
                img="${img#\`}"              # drop a leading backtick
                img="$(echo "$img" | xargs)" # trim whitespace
                case "$img" in
                    "") continue ;;          # blank
                    *:*) ;;                  # has a tag → ok
                    *)  continue ;;          # tagless (e.g. bare `busybox`) → skip
                esac
                _ensure_pulled "$img"
                as_count=$(( as_count + 1 ))
            done < <(printf '%s\n' "$as_render" \
                       | grep -E '^[[:space:]]*image:' \
                       | sed -E 's/^[[:space:]]*image:[[:space:]]*//')
            info "astronomy-shop: pre-loaded $as_count rendered images"
        else
            warn "astronomy-shop: helm template failed; skipping dynamic pre-pull"
        fi
    fi

    success "Kind cluster ready"
}

# ── Banner ────────────────────────────────────────────────────────────────────

echo "" | tee -a "$RUN_LOG"
info "================================================================" | tee -a "$RUN_LOG"
info "  GraphRCA — Run ALL Tasks (ollama)"                              | tee -a "$RUN_LOG"
info "================================================================" | tee -a "$RUN_LOG"
info "  Model:        $OLLAMA_MODEL"                                    | tee -a "$RUN_LOG"
info "  Task types:   $TASK_TYPES"                                      | tee -a "$RUN_LOG"
info "  Total tasks:  ${#TASK_LINES[@]}"                                | tee -a "$RUN_LOG"
info "  Agent mode:   ${GRAPHRCA_AGENT_MODE:-scratchpad_swarm}"         | tee -a "$RUN_LOG"
info "  Architecture: $ARCH"                                            | tee -a "$RUN_LOG"
info "  Run dir:      $RUN_DIR"                                         | tee -a "$RUN_LOG"
info "================================================================" | tee -a "$RUN_LOG"
echo "" | tee -a "$RUN_LOG"

# ── JSON summary initialisation ───────────────────────────────────────────────

cat > "$SUMMARY_FILE" <<JSON
{
  "run_timestamp": "$RUN_TIMESTAMP",
  "model": "$OLLAMA_MODEL",
  "agent_mode": "${GRAPHRCA_AGENT_MODE:-scratchpad_swarm}",
  "task_types": "$TASK_TYPES",
  "total_tasks": ${#TASK_LINES[@]},
  "results": []
}
JSON

# ── Tracking counters ─────────────────────────────────────────────────────────

TOTAL=${#TASK_LINES[@]}
PASSED=0
FAILED=0
SKIPPED=0
TASK_IDX=0

# Collect result rows for final JSON update
declare -a RESULT_ROWS=()

# ── Per-task runner ───────────────────────────────────────────────────────────

run_task() {
    local task_type="$1"
    local task_name="$2"
    local task_num="$3"

    header "Task $task_num/$TOTAL  [$task_type]  $task_name"

    TASK_TIMESTAMP="$(date +"%m-%d_%H-%M-%S")"
    TASK_DIR="$RUN_DIR/${task_num}-${task_type}-${task_name}"
    mkdir -p "$TASK_DIR/graphrca_output"

    export TASK_NAME="$task_name"
    export OUTPUT_DIRECTORY="$TASK_DIR/graphrca_output"

    # Setup cluster for this task (unless preserving)
    if [[ $PRESERVE_CLUSTER -eq 0 ]]; then
        setup_cluster
    else
        info "Preserve mode — reusing cluster"
    fi

    # Build command
    CMD=(
        "$PYTHON_BIN" -m GraphRCA_agent.run_pipeline
        --aiopslab
        --problem-id "$task_name"
        --output-dir "$TASK_DIR/graphrca_output"
    )
    [[ -n "$VERBOSE"  ]] && CMD+=("--verbose")
    [[ -n "$NO_NEO4J" ]] && CMD+=("--no-neo4j")

    info "Running: ${CMD[*]}"
    TASK_START=$(date +%s)

    "${CMD[@]}" 2>&1 | tee "$TASK_DIR/run.log"
    TASK_STATUS=${PIPESTATUS[0]}

    TASK_END=$(date +%s)
    TASK_ELAPSED=$(( TASK_END - TASK_START ))

    # Clean ANSI codes from log
    CLEAN="$REPO_ROOT/eval/clean_ansi_from_log.py"
    if [[ -f "$CLEAN" ]]; then
        "$PYTHON_BIN" "$CLEAN" "$TASK_DIR/run.log" 2>/dev/null || true
    fi

    # Record result
    if [[ $TASK_STATUS -eq 0 ]]; then
        success "PASSED  [$task_type] $task_name  (${TASK_ELAPSED}s)"
        RESULT_ROWS+=("{\"task\": \"$task_name\", \"type\": \"$task_type\", \"status\": \"passed\", \"elapsed_seconds\": $TASK_ELAPSED, \"output_dir\": \"$TASK_DIR\"}")
        PASSED=$(( PASSED + 1 ))
    else
        error "FAILED  [$task_type] $task_name  (exit $TASK_STATUS, ${TASK_ELAPSED}s)"
        RESULT_ROWS+=("{\"task\": \"$task_name\", \"type\": \"$task_type\", \"status\": \"failed\", \"exit_code\": $TASK_STATUS, \"elapsed_seconds\": $TASK_ELAPSED, \"output_dir\": \"$TASK_DIR\"}")
        FAILED=$(( FAILED + 1 ))
        return $TASK_STATUS
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────

for line in "${TASK_LINES[@]}"; do
    TASK_TYPE="${line%%:*}"
    TASK_NAME_VAL="${line##*:}"
    TASK_IDX=$(( TASK_IDX + 1 ))

    if ! run_task "$TASK_TYPE" "$TASK_NAME_VAL" "$TASK_IDX" 2>&1 | tee -a "$RUN_LOG"; then
        if [[ $CONTINUE_ON_ERROR -eq 0 ]]; then
            error "Stopping after first failure (use --continue-on-error to keep going)"
            break
        else
            warn "Task failed, continuing (--continue-on-error)"
        fi
    fi

    # Delay between tasks to let the cluster settle
    if [[ $TASK_IDX -lt $TOTAL && $INTER_TASK_DELAY -gt 0 ]]; then
        info "Waiting ${INTER_TASK_DELAY}s before next task..."
        sleep "$INTER_TASK_DELAY"
    fi
done

SKIPPED=$(( TOTAL - PASSED - FAILED ))

# ── Write final summary JSON ──────────────────────────────────────────────────

# Build results array string
RESULTS_JSON=""
for row in "${RESULT_ROWS[@]}"; do
    RESULTS_JSON+="    $row,\n"
done
RESULTS_JSON="${RESULTS_JSON%,\\n}"   # strip trailing comma

cat > "$SUMMARY_FILE" <<JSON
{
  "run_timestamp": "$RUN_TIMESTAMP",
  "completed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "model": "$OLLAMA_MODEL",
  "ollama_url": "$OLLAMA_BASE_URL",
  "agent_mode": "${GRAPHRCA_AGENT_MODE:-scratchpad_swarm}",
  "task_types": "$TASK_TYPES",
  "total_tasks": $TOTAL,
  "passed": $PASSED,
  "failed": $FAILED,
  "skipped": $SKIPPED,
  "results": [
$(printf "%s" "$RESULTS_JSON")
  ]
}
JSON

# ── Final report ──────────────────────────────────────────────────────────────

echo "" | tee -a "$RUN_LOG"
info "================================================================" | tee -a "$RUN_LOG"
info "  ALL TASKS COMPLETE"                                              | tee -a "$RUN_LOG"
info "================================================================" | tee -a "$RUN_LOG"
info "  Model:    $OLLAMA_MODEL"                                         | tee -a "$RUN_LOG"
info "  Total:    $TOTAL"                                                | tee -a "$RUN_LOG"
success "  Passed:   $PASSED"                                            | tee -a "$RUN_LOG"
[[ $FAILED  -gt 0 ]] && error   "  Failed:   $FAILED"   | tee -a "$RUN_LOG" || true
[[ $SKIPPED -gt 0 ]] && warn    "  Skipped:  $SKIPPED"  | tee -a "$RUN_LOG" || true
info "  Summary:  $SUMMARY_FILE"                                         | tee -a "$RUN_LOG"
info "  Full log: $RUN_LOG"                                              | tee -a "$RUN_LOG"
info "================================================================" | tee -a "$RUN_LOG"

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
