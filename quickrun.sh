#!/usr/bin/env bash
# quickrun.sh — GraphRCA pipeline quick-run script
#
# Usage:
#   ./quickrun.sh                                  # standalone, trace_output/
#   ./quickrun.sh -t ./my/traces                   # custom trace dir
#   ./quickrun.sh -v                               # verbose
#   ./quickrun.sh --no-neo4j                       # skip Neo4j
#   ./quickrun.sh --clear-neo4j                    # wipe all Neo4j data and exit
#   ./quickrun.sh --aiopslab detection-1           # AIOpsLab benchmark
#   ./quickrun.sh --aiopslab misconfig_app_hotel_res-localization-1
#
# All output goes to GraphRCA_output/<timestamp>/
# LLM calls are logged to llm_justification.jsonl for audit.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$SCRIPT_DIR/venv"
AIOPSLAB_ROOT="${AIOPSLAB_ROOT:-$REPO_ROOT/AIOpsLab}"
STRATUS_SRC="$REPO_ROOT/stratus/src"

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

# ── Parse Arguments ───────────────────────────────────────────────────────────

TRACE_DIR="$REPO_ROOT/stratus/trace_output"
OUTPUT_DIR=""
VERBOSE=""
NO_NEO4J=""
NO_SPANS=""
AIOPSLAB_MODE=0
PROBLEM_ID="misconfig_app_hotel_res-detection-1"
CLEAR_NEO4J=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clear-neo4j)
            CLEAR_NEO4J=1 ;;
        --aiopslab)
            AIOPSLAB_MODE=1
            if [[ $# -gt 1 && "$2" != --* ]]; then
                PROBLEM_ID="$2"
                shift
            fi
            ;;
        -t|--trace-dir)
            TRACE_DIR="$2"; shift ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"; shift ;;
        -v|--verbose)
            VERBOSE="--verbose" ;;
        --no-neo4j)
            NO_NEO4J="--no-neo4j" ;;
        --no-spans)
            NO_SPANS="--no-spans" ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            error "Unknown argument: $1"
            exit 1 ;;
    esac
    shift
done

# ── Activate venv ─────────────────────────────────────────────────────────────

if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    info "Using venv: $VENV"
else
    warn "No venv found at $VENV — using system Python"
    warn "To create: python3 -m venv $VENV && source $VENV/bin/activate && pip install -r requirements.txt"
fi

# ── Load .env ─────────────────────────────────────────────────────────────────

ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    info "Loading env: $ENV_FILE"
    set -o allexport
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +o allexport
else
    warn ".env not found at $ENV_FILE — using existing environment"
fi

# ── PYTHONPATH ────────────────────────────────────────────────────────────────

export PYTHONPATH="$AIOPSLAB_ROOT:$STRATUS_SRC:$REPO_ROOT:${PYTHONPATH:-}"
info "PYTHONPATH includes: AIOpsLab, stratus/src, repo root"

# ── Run ───────────────────────────────────────────────────────────────────────

echo ""
info "======================================================="
info "  GraphRCA — LangGraph Autonomous SRE Pipeline"
info "======================================================="

if [[ $CLEAR_NEO4J -eq 1 ]]; then
    warn "Clearing ALL data from Neo4j..."
    python -m GraphRCA.run_pipeline --clear-neo4j
    STATUS=$?
    [[ $STATUS -eq 0 ]] && success "Neo4j cleared." || error "Neo4j clear failed."
    exit $STATUS
fi

if [[ $AIOPSLAB_MODE -eq 1 ]]; then
    info "Mode:       AIOpsLab benchmark"
    info "Problem ID: $PROBLEM_ID"
    echo ""

    CMD=(python -m GraphRCA.run_pipeline
        --aiopslab
        --problem-id "$PROBLEM_ID"
        ${VERBOSE}
    )
    [[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")

    info "Running: ${CMD[*]}"
    echo ""
    "${CMD[@]}"
else
    info "Mode:       Standalone trace analysis"
    info "Trace dir:  $TRACE_DIR"
    echo ""

    if [[ ! -d "$TRACE_DIR" ]]; then
        error "Trace directory not found: $TRACE_DIR"
        error "Pass a valid path with: ./quickrun.sh -t /path/to/traces"
        exit 1
    fi

    CSV_COUNT=$(find "$TRACE_DIR" -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
    if [[ "$CSV_COUNT" -eq 0 ]]; then
        error "No CSV files found in: $TRACE_DIR"
        exit 1
    fi
    info "Found $CSV_COUNT trace CSV files"

    CMD=(python -m GraphRCA.run_pipeline
        --trace-dir "$TRACE_DIR"
        ${NO_NEO4J}
        ${NO_SPANS}
        ${VERBOSE}
    )
    [[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")

    info "Running: ${CMD[*]}"
    echo ""
    "${CMD[@]}"
fi

STATUS=$?
echo ""
if [[ $STATUS -eq 0 ]]; then
    success "Pipeline completed successfully"
    info "Output: GraphRCA_output/<timestamp>/"
    info "  incident_report.json     — main report"
    info "  llm_justification.jsonl  — all LLM calls logged"
    info "  reports/                 — ITBench-compatible JSON"
else
    error "Pipeline exited with status $STATUS"
fi

exit $STATUS
