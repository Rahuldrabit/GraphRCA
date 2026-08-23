#!/usr/bin/env bash
# run_single_task_ollama.sh — Run one AIOpsLab task using a local ollama model
#
# Usage:
#   ./run_single_task_ollama.sh <task_name>                        # uses gemma4:12b (default)
#   ./run_single_task_ollama.sh <task_name> --model deepseek-r1:8b
#   ./run_single_task_ollama.sh <task_name> --model gemma4:12b
#   ./run_single_task_ollama.sh <task_name> --model deepseek-r1:8b --preserve
#   ./run_single_task_ollama.sh <task_name> --model gemma4:12b --verbose
#   ./run_single_task_ollama.sh <task_name> --mode scratchpad_swarm
#   ./run_single_task_ollama.sh <task_name> --no-neo4j
#
# Options:
#   --model <name>    ollama model to use  (default: gemma4:12b)
#   --mode  <mode>    agent mode: pipeline | multi_agent | scratchpad_swarm
#                     (default: scratchpad_swarm, as set in .env)
#   --preserve        reuse the existing kind cluster (skip cluster setup)
#   --no-neo4j        disable Neo4j knowledge graph
#   --verbose / -v    verbose logging
#   -h / --help       show this message
#
# Supported ollama models:
#   gemma4:12b        (recommended — good balance of speed and quality)
#   deepseek-r1:8b    (good for reasoning-heavy RCA)
#
# Output:  eval/<timestamp>-<task_name>-<model>/

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VENV="$SCRIPT_DIR/venv"
AIOPSLAB_ROOT="${AIOPSLAB_ROOT:-$REPO_ROOT/AIOpsLab}"

# ── Colours ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[GraphRCA-ollama]${NC} $*"; }
success() { echo -e "${GREEN}[GraphRCA-ollama]${NC} $*"; }
warn()    { echo -e "${YELLOW}[GraphRCA-ollama]${NC} $*"; }
error()   { echo -e "${RED}[GraphRCA-ollama]${NC} $*" >&2; }

# ── Defaults ──────────────────────────────────────────────────────────────────

TASK_NAME=""
OLLAMA_MODEL="gemma4:12b"
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_API_KEY="ollama"          # ollama ignores the key but the client requires one
AGENT_MODE=""                    # empty = use whatever is in .env
PRESERVE_CLUSTER=0
NO_NEO4J=""
VERBOSE=""

# ── Argument Parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            grep '^#' "$0" | head -35 | sed 's/^# \?//'; exit 0 ;;
        --model)
            OLLAMA_MODEL="$2"; shift ;;
        --mode)
            AGENT_MODE="$2"; shift ;;
        --preserve|-p)
            PRESERVE_CLUSTER=1 ;;
        --no-neo4j)
            NO_NEO4J="--no-neo4j" ;;
        --verbose|-v)
            VERBOSE="--verbose" ;;
        -*)
            error "Unknown option: $1"; exit 1 ;;
        *)
            TASK_NAME="$1" ;;
    esac
    shift
done

if [[ -z "$TASK_NAME" ]]; then
    error "No task name provided."
    echo ""
    echo "Usage: $0 <task_name> [--model <ollama_model>]"
    echo ""
    echo "Example tasks (from eval/eval_tasks.yaml):"
    echo "  misconfig_app_hotel_res-detection-1"
    echo "  k8s_target_port-misconfig-localization-1"
    echo "  auth_miss_mongodb-mitigation-1"
    echo "  container_kill-detection"
    echo ""
    echo "Available models:"
    echo "  gemma4:12b       (default)"
    echo "  deepseek-r1:8b"
    exit 1
fi

# ── Activate venv ─────────────────────────────────────────────────────────────

if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
    info "Using venv: $VENV"
else
    warn "No venv found at $VENV — using system Python"
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
    info "Loaded: $ENV_FILE"
fi

# ── Override LLM settings → ollama ────────────────────────────────────────────
# ollama exposes an OpenAI-compatible /v1 endpoint, so we just redirect
# URL_AGENTS / MODEL_AGENTS. The rest of the pipeline is untouched.

export PROVIDER_AGENTS="openai"          # keep openai client; ollama is API-compatible
export MODEL_AGENTS="$OLLAMA_MODEL"
export URL_AGENTS="$OLLAMA_BASE_URL"
export API_KEY_AGENTS="$OLLAMA_API_KEY"  # any non-empty string works
export OPENAI_API_KEY="$OLLAMA_API_KEY"  # LangChain / CrewAI fallback

# Also override the "tools" LLM (used by tool-calling nodes)
export PROVIDER_TOOLS="openai"
export MODEL_TOOLS="$OLLAMA_MODEL"
export URL_TOOLS="$OLLAMA_BASE_URL"
export API_KEY_TOOLS="$OLLAMA_API_KEY"

# Optional: disable Neo4j if requested (speeds up local runs without Aura)
if [[ -n "$NO_NEO4J" ]]; then
    export NEO4J_ENABLED="False"
fi

# Override agent mode if specified
if [[ -n "$AGENT_MODE" ]]; then
    export GRAPHRCA_AGENT_MODE="$AGENT_MODE"
fi

# Disable interactive KG mode prompt (non-interactive run)
export GRAPHRCA_LLM_KG_MODE="${GRAPHRCA_LLM_KG_MODE:-}"

# ── Model remap: gemma4:12b → num_ctx=32768 variant ───────────────────────
# gemma4 is a "thinking" model (~3-4K reasoning tokens before it answers). With
# ollama's default num_ctx the RCA/drill prompts left no room for the answer →
# empty responses (the "300 tokens / 0 chars" symptom). The OpenAI-compat /v1
# endpoint ignores options.num_ctx, so the large context is baked into a local
# custom model (Gemma4-GraphRCA.Modelfile). Remap transparently if it exists so
# a running batch started with --model gemma4:12b picks up the fix on its next
# task without needing a restart.
if [[ "$OLLAMA_MODEL" == "gemma4:12b" ]] && ollama list 2>/dev/null | grep -q "^gemma4-graphrca:12b "; then
    info "Remapping gemma4:12b → gemma4-graphrca:12b (num_ctx=32768 baked in)"
    OLLAMA_MODEL="gemma4-graphrca:12b"
fi

# ── PYTHONPATH ────────────────────────────────────────────────────────────────

export PYTHONPATH="$AIOPSLAB_ROOT:$SCRIPT_DIR:$REPO_ROOT:${PYTHONPATH:-}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

# ── Check ollama is reachable ─────────────────────────────────────────────────

info "Checking ollama at $OLLAMA_BASE_URL ..."
if ! curl -sf "${OLLAMA_BASE_URL%/v1}/api/tags" >/dev/null 2>&1; then
    warn "ollama does not appear to be running at ${OLLAMA_BASE_URL%/v1}"
    warn "Start it with:  ollama serve"
    warn "Then pull models:  ollama pull $OLLAMA_MODEL"
    warn "Continuing anyway — pipeline will fail at the first LLM call if ollama is down."
fi

# Check the requested model is pulled
if curl -sf "${OLLAMA_BASE_URL%/v1}/api/tags" >/dev/null 2>&1; then
    if ! curl -sf "${OLLAMA_BASE_URL%/v1}/api/tags" | grep -q "\"${OLLAMA_MODEL}\""; then
        warn "Model '$OLLAMA_MODEL' may not be pulled yet."
        warn "Run:  ollama pull $OLLAMA_MODEL"
    fi
fi

# ── Output directory ──────────────────────────────────────────────────────────

MODEL_SLUG="${OLLAMA_MODEL//:/-}"          # e.g. gemma4-12b
TIMESTAMP="$(date +"%m-%d_%H-%M-%S")"
OUTPUT_DIR="$REPO_ROOT/eval/${TIMESTAMP}-${TASK_NAME}-${MODEL_SLUG}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/graphrca_output"

export TASK_NAME
export OUTPUT_DIRECTORY="${OUTPUT_DIR}/graphrca_output"

# ── Architecture Detection ────────────────────────────────────────────────────

detect_arch() {
    local a; a="$(uname -m)"
    case "$a" in
        x86_64) echo 'x86' ;;
        arm*|aarch64) echo 'arm' ;;
        *) echo 'x86' ;;
    esac
}
ARCH="$(detect_arch)"

# ── Cluster Setup ─────────────────────────────────────────────────────────────

setup_cluster() {
    info "Deleting existing kind cluster..."
    kind delete cluster --name kind 2>/dev/null || true

    local cfg="$AIOPSLAB_ROOT/kind/kind-config-${ARCH}.yaml"
    if [[ ! -f "$cfg" ]]; then
        error "Kind config not found: $cfg"; exit 1
    fi
    info "Creating kind cluster (arch: $ARCH)..."
    kind create cluster --config "$cfg"

    # Pre-load the application + observability images into the kind node so the
    # app pods never hit the network on a cold start. Each task wipes the kind
    # cluster (kind delete + create), so containerd's image cache is lost every
    # run — without this, the DSB microservice images (multi-GB) cold-pull over
    # the network and non-deterministically exceed even the 1200s readiness
    # budget (same task passed once and timed out another — high-variance pulls).
    # Strategy: pull once into the persistent docker daemon cache (reused across
    # tasks), then `kind load` into the fresh node (fast, disk-bound). The first
    # task pays the one-time pull; later tasks hit the cache and just `kind load`.
    # Tags are :latest per the charts' defaultImageVersion. The big DSB images
    # (deathstarbench/*, yg397/*, yinfangchen/*) are the dominant cold-pull cost;
    # bitnami data-stores (mongo/redis/memcached) are smaller and pull more reliably.
    # Pull an image into the docker cache (best-effort, capped), then load it
    # into the kind node. Never aborts setup on a miss — a failed/slow pull just
    # falls back to a deploy-time pull (which has the full readiness budget).
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

    # ── Static set: social-network + hotel-reservation (DeathStarBench) ──
    local APP_IMAGES=(
        "deathstarbench/social-network-microservices:latest"
        "yg397/media-frontend:latest"
        "yg397/openresty-thrift:latest"
        "yinfangchen/social-otel:latest"
        "yinfangchen/social-otel-regress:latest"
        "deathstarbench/hotel-reservation:latest"
        "hashicorp/consul:latest"
        "jaegertracing/all-in-one:latest"
        "alpine/git:latest"
    )
    for img in "${APP_IMAGES[@]}"; do _ensure_pulled "$img"; done

    # ── Dynamic set: astronomy-shop (OpenTelemetry demo) — ASTRONOMY TASKS ONLY ──
    # astronomy-shop is a heavy deploy (~38 images across ghcr/quay/docker.io:
    # the demo microservices + jaeger/prometheus/grafana/opensearch/otel-collector).
    # Cold-pulling that many multi-registry images non-deterministically exceeds
    # even the 1200s readiness budget → deploy timeout → the task dies before the
    # observer runs. Render the remote chart to get the EXACT current image set
    # (robust to chart/appVersion bumps — the app installs `open-telemetry/
    # opentelemetry-demo` unpinned, so we template the same) and pre-load each.
    # GATED on the task name: this is ~38 images, so running it for every (e.g.
    # hotel/social) task would dominate setup. Non-astronomy tasks skip it; their
    # app images are pulled by the kubelet at deploy time (within the readiness
    # budget) if not in the static set above.
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
                    *:*) ;;                  # has a tag (registry[:port]/name:tag) → ok
                    *)  continue ;;          # tagless (e.g. bare `busybox`) → skip
                esac
                _ensure_pulled "$img"
                as_count=$((as_count + 1))
            done < <(printf '%s\n' "$as_render" \
                       | grep -E '^[[:space:]]*image:' \
                       | sed -E 's/^[[:space:]]*image:[[:space:]]*//')
            info "astronomy-shop: pre-loaded $as_count rendered images"
        else
            warn "astronomy-shop: helm template failed; skipping dynamic pre-pull"
        fi
    fi

    success "Kind cluster created"
}

# ── Banner ────────────────────────────────────────────────────────────────────

echo ""
info "========================================================"
info "  GraphRCA — AIOpsLab Single Task (ollama)"
info "========================================================"
info "  Task:         $TASK_NAME"
info "  Model:        $OLLAMA_MODEL"
info "  Ollama URL:   $OLLAMA_BASE_URL"
info "  Agent mode:   ${GRAPHRCA_AGENT_MODE:-scratchpad_swarm}"
info "  Architecture: $ARCH"
info "  Output:       $OUTPUT_DIR"
info "========================================================"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────

if [[ $PRESERVE_CLUSTER -eq 0 ]]; then
    setup_cluster
else
    info "Preserve mode — skipping cluster setup"
fi

CMD=(
    "$PYTHON_BIN" -m GraphRCA_agent.run_pipeline
    --aiopslab
    --problem-id "$TASK_NAME"
    --output-dir "${OUTPUT_DIR}/graphrca_output"
)
[[ -n "$VERBOSE"  ]] && CMD+=("--verbose")
[[ -n "$NO_NEO4J" ]] && CMD+=("--no-neo4j")

info "Running: ${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
STATUS=${PIPESTATUS[0]}

# Clean ANSI from log
CLEAN_SCRIPT="$REPO_ROOT/eval/clean_ansi_from_log.py"
if [[ -f "$CLEAN_SCRIPT" ]]; then
    "$PYTHON_BIN" "$CLEAN_SCRIPT" "${OUTPUT_DIR}/run.log" 2>/dev/null || true
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
if [[ $STATUS -eq 0 ]]; then
    success "Task completed: $TASK_NAME  [model: $OLLAMA_MODEL]"
    info "Output dir: $OUTPUT_DIR"
    info "  run.log                       — full execution log"
    info "  graphrca_output/              — pipeline artefacts"
    info "    incident_report.json        — main RCA report"
    info "    llm_justification.jsonl     — every LLM call logged"
    info "    graphrca_run_stats.json     — token usage / timing"
else
    error "Pipeline exited with status $STATUS"
    error "Check: $OUTPUT_DIR/run.log"
fi

exit $STATUS
