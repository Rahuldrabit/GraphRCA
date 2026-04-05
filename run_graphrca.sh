#!/usr/bin/env bash
# run_graphrca.sh — Unified GraphRCA pipeline runner
#
# Combines quickrun.sh and test_graphrca.sh functionality
#
# Usage:
#   ./run_graphrca.sh <task_name>                  # Run AIOpsLab task (default)
#   ./run_graphrca.sh -t ./my/traces               # Standalone trace analysis
#   ./run_graphrca.sh -p <task_name>               # Use existing cluster
#   ./run_graphrca.sh -s                           # Setup cluster only
#   ./run_graphrca.sh --clear-neo4j                # Wipe Neo4j and exit
#   ./run_graphrca.sh -v <task_name>               # Verbose mode
#
# Options:
#   -h, --help        Show this help message
#   -p, --preserve    Use existing cluster instead of creating a new one
#   -r, --arch <arch> Specify architecture (x86 or arm, auto-detected by default)
#   -d, --output-dir  Specify output directory
#   -t, --trace-dir   Run standalone trace analysis (skip AIOpsLab)
#   -s, --setup-only  Setup cluster only, without running task
#   -v, --verbose     Verbose output
#   --no-neo4j        Skip Neo4j operations
#   --no-spans        Skip span processing
#   --clear-neo4j     Clear all Neo4j data and exit
#
# Output goes to eval/<timestamp>-<task_name>/ or GraphRCA_output/<timestamp>/

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$SCRIPT_DIR/venv"
AIOPSLAB_ROOT="${AIOPSLAB_ROOT:-$REPO_ROOT/AIOpsLab}"

# Prefer an explicit Python interpreter (venv-first) so we don't depend on the
# presence of a system-wide `python` shim.
PYTHON_BIN=""

# ── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[GraphRCA]${NC} $*"; }
success() { echo -e "${GREEN}[GraphRCA]${NC} $*"; }
warn()    { echo -e "${YELLOW}[GraphRCA]${NC} $*"; }
error()   { echo -e "${RED}[GraphRCA]${NC} $*" >&2; }

# ── Architecture Detection ───────────────────────────────────────────────────

detect_architecture() {
    local uname_arch
    uname_arch="$(uname -m)"
    
    if [[ "$uname_arch" = x86_64* ]]; then
        if [[ "$(uname -a)" = *ARM64* ]]; then
            echo 'arm'
        else
            echo 'x86'
        fi
    elif [[ "$uname_arch" = i*86 ]]; then
        echo 'x86'
    elif [[ "$uname_arch" = arm* ]] || [[ "$uname_arch" = aarch64 ]]; then
        echo 'arm'
    else
        echo 'x86'  # Default to x86
    fi
}

# ── Show Help ────────────────────────────────────────────────────────────────

show_help() {
    grep '^#' "$0" | head -30 | sed 's/^# \?//'
}

# ── Defaults ─────────────────────────────────────────────────────────────────

TRACE_DIR=""
OUTPUT_DIR=""
VERBOSE=""
NO_NEO4J=""
NO_SPANS=""
AIOPSLAB_MODE=1
TASK_NAME=""
CLEAR_NEO4J=0
PRESERVE_CLUSTER=0
SETUP_ONLY=0
ARCH=$(detect_architecture)

# ── Parse Arguments ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0 ;;
        -p|--preserve)
            PRESERVE_CLUSTER=1 ;;
        -r|--arch)
            ARCH="$2"
            if [[ "$ARCH" != "x86" && "$ARCH" != "arm" ]]; then
                error "Invalid architecture: $ARCH. Use 'x86' or 'arm'."
                exit 1
            fi
            shift ;;
        -d|--output-dir)
            OUTPUT_DIR="$2"; shift ;;
        -t|--trace-dir)
            TRACE_DIR="$2"
            AIOPSLAB_MODE=0
            shift ;;
        -s|--setup-only)
            SETUP_ONLY=1 ;;
        -v|--verbose)
            VERBOSE="--verbose" ;;
        --no-neo4j)
            NO_NEO4J="--no-neo4j" ;;
        --no-spans)
            NO_SPANS="--no-spans" ;;
        --clear-neo4j)
            CLEAR_NEO4J=1 ;;
        -*)
            error "Unknown option: $1"
            show_help
            exit 1 ;;
        *)
            # Positional argument = task name
            TASK_NAME="$1" ;;
    esac
    shift
done

# ── Activate venv ────────────────────────────────────────────────────────────

if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    info "Using venv: $VENV"
else
    warn "No venv found at $VENV — using system Python"
fi

# Resolve python executable (venv-first, then python/python3)
if [[ -x "$VENV/bin/python" ]]; then
    PYTHON_BIN="$VENV/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    error "No Python interpreter found (tried $VENV/bin/python, python, python3)"
    exit 127
fi

# ── Load .env ────────────────────────────────────────────────────────────────

ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    info "Loading env: $ENV_FILE"
    set -o allexport
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +o allexport
fi

# ── PYTHONPATH ───────────────────────────────────────────────────────────────

export PYTHONPATH="$AIOPSLAB_ROOT:$REPO_ROOT:$SCRIPT_DIR:${PYTHONPATH:-}"
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

# ── Banner ───────────────────────────────────────────────────────────────────

echo ""
info "======================================================="
info "  GraphRCA — LangGraph Autonomous SRE Pipeline"
info "======================================================="
info "Architecture: $ARCH"

# ── Experimental: LLM Knowledge Graph Modes ─────────────────────────────────
#
# For testing/debugging you can let the LLM access the knowledge graph in 2 ways:
#   a) Dump the full NetworkX graph JSON into the LLM prompt
#   b) Neo4j GraphRAG: the LLM writes Cypher READ queries and we feed results back
#
# Selection is interactive (type 'a' or 'b') unless GRAPHRCA_LLM_KG_MODE is set.

prompt_llm_kg_mode() {
    # Respect a pre-set mode (e.g., exported by the user or CI).
    if [[ -n "${GRAPHRCA_LLM_KG_MODE:-}" ]]; then
        return 0
    fi

    # Only prompt when stdin is a TTY.
    if [[ ! -t 0 ]]; then
        return 0
    fi

    echo ""
    info "LLM knowledge graph mode (experimental):"
    echo "  a) Full graph dump into prompt"
    echo "  b) Neo4j GraphRAG (LLM writes Cypher queries)"
    echo "  (Enter) Default (no KG access)"
    read -r -p "Choose [a/b/Enter]: " KG_CHOICE

    case "${KG_CHOICE}" in
        a|A) export GRAPHRCA_LLM_KG_MODE="a" ;;
        b|B) export GRAPHRCA_LLM_KG_MODE="b" ;;
        *) export GRAPHRCA_LLM_KG_MODE="" ;;
    esac
}

prompt_llm_kg_mode

if [[ "${GRAPHRCA_LLM_KG_MODE:-}" == "a" ]]; then
    info "LLM KG mode: a (graph dump)"
elif [[ "${GRAPHRCA_LLM_KG_MODE:-}" == "b" ]]; then
    info "LLM KG mode: b (Neo4j GraphRAG)"
    if [[ "${NO_NEO4J:-}" == "--no-neo4j" ]]; then
        warn "KG mode b selected but --no-neo4j was passed; GraphRAG will be skipped"
    fi
else
    info "LLM KG mode: default (no KG access)"
fi

# ── Clear Neo4j ──────────────────────────────────────────────────────────────

if [[ $CLEAR_NEO4J -eq 1 ]]; then
    warn "Clearing ALL data from Neo4j..."
    "$PYTHON_BIN" -m GraphRCA_agent.run_pipeline --clear-neo4j
    STATUS=$?
    [[ $STATUS -eq 0 ]] && success "Neo4j cleared." || error "Neo4j clear failed."
    exit $STATUS
fi

# ── Setup Cluster Function ───────────────────────────────────────────────────

setup_cluster() {
    info "Deleting existing kind cluster..."
    kind delete cluster --name kind 2>/dev/null || true
    
    info "Creating kind cluster (arch: $ARCH)..."
    local config_file="$AIOPSLAB_ROOT/kind/kind-config-${ARCH}.yaml"
    
    if [[ ! -f "$config_file" ]]; then
        error "Kind config not found: $config_file"
        exit 1
    fi
    
    kind create cluster --config "$config_file"
    success "Kind cluster created successfully"
}

# ── Setup Only Mode ──────────────────────────────────────────────────────────

if [[ $SETUP_ONLY -eq 1 ]]; then
    setup_cluster
    exit 0
fi

# ── AIOpsLab Mode ────────────────────────────────────────────────────────────

if [[ $AIOPSLAB_MODE -eq 1 ]]; then
    if [[ -z "$TASK_NAME" ]]; then
        error "No task name provided."
        echo "Usage: $0 <task_name>"
        echo "Example: $0 misconfig_app_hotel_res-detection-1"
        exit 1
    fi
    
    # Setup output directory
    if [[ -z "$OUTPUT_DIR" ]]; then
        CURRENT_DATE=$(date +"%m-%d_%H-%M-%S")
        OUTPUT_DIR="$REPO_ROOT/eval/${CURRENT_DATE}-${TASK_NAME}"
    fi
    
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}/graphrca_output"
    
    export TASK_NAME
    export OUTPUT_DIRECTORY="${OUTPUT_DIR}/graphrca_output"
    
    # Setup cluster if needed
    if [[ $PRESERVE_CLUSTER -eq 0 ]]; then
        setup_cluster
    else
        info "Using existing cluster (preserve mode)"
    fi
    
    info "Mode:       AIOpsLab benchmark"
    info "Task:       $TASK_NAME"
    info "Output dir: $OUTPUT_DIR"
    echo ""
    
    # Run GraphRCA
    CMD=(python -m GraphRCA_agent.run_pipeline
        --aiopslab
        --problem-id "$TASK_NAME"
        --output-dir "${OUTPUT_DIR}/graphrca_output"
        ${VERBOSE}
    )
    
    info "Running: ${CMD[*]}"
    echo ""
    
    CMD[0]="$PYTHON_BIN"
    "${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
    STATUS=${PIPESTATUS[0]}
    
    # Clean ANSI codes from log if script exists
    CLEAN_SCRIPT="$REPO_ROOT/eval/clean_ansi_from_log.py"
    if [[ -f "$CLEAN_SCRIPT" ]]; then
        info "Cleaning ANSI codes from log..."
        "$PYTHON_BIN" "$CLEAN_SCRIPT" "${OUTPUT_DIR}/run.log" 2>/dev/null || true
    fi

# ── Standalone Trace Mode ────────────────────────────────────────────────────

else
    if [[ -z "$TRACE_DIR" ]]; then
        TRACE_DIR="$SCRIPT_DIR/trace_output"
    fi
    
    info "Mode:       Standalone trace analysis"
    info "Trace dir:  $TRACE_DIR"
    echo ""
    
    if [[ ! -d "$TRACE_DIR" ]]; then
        error "Trace directory not found: $TRACE_DIR"
        error "Pass a valid path with: $0 -t /path/to/traces"
        exit 1
    fi
    
    CSV_COUNT=$(find "$TRACE_DIR" -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
    if [[ "$CSV_COUNT" -eq 0 ]]; then
        error "No CSV files found in: $TRACE_DIR"
        exit 1
    fi
    info "Found $CSV_COUNT trace CSV files"
    
    CMD=("$PYTHON_BIN" -m GraphRCA_agent.run_pipeline
        --trace-dir "$TRACE_DIR"
        ${NO_NEO4J}
        ${NO_SPANS}
        ${VERBOSE}
    )
    [[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")
    
    info "Running: ${CMD[*]}"
    echo ""
    "${CMD[@]}"
    STATUS=$?
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
if [[ $STATUS -eq 0 ]]; then
    success "Pipeline completed successfully"
    if [[ $AIOPSLAB_MODE -eq 1 ]]; then
        info "Output: $OUTPUT_DIR"
        info "  run.log                  — execution log"
        info "  graphrca_output/         — GraphRCA results"
    else
        info "Output: GraphRCA_output/<timestamp>/"
    fi
    info "  incident_report.json     — main report"
    info "  llm_justification.jsonl  — all LLM calls logged"
else
    error "Pipeline exited with status $STATUS"
fi

exit $STATUS
