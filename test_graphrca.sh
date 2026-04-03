#!/usr/bin/env bash
# test_graphrca.sh — Run GraphRCA against a single AIOpsLab task
#
# Usage: ./test_graphrca.sh [-p] [-r <arch>] [-d <output_dir>] <task_name>
#
# Options:
#   -h              Show this help message
#   -p              Use existing cluster instead of creating a new one
#   -r <arch>       Specify the architecture (x86 or arm)
#   -d <output_dir> Specify the output directory
#   -s              Setup the cluster only, without running the task

set -e

preserve_cluster='false'
setup_cluster_only='false'
arch=''
output_dir=''

function detect_architecture() {
  arch="$(uname -m)"
  if [[ "$arch" = x86_64* ]]; then
    if [[ "$(uname -a)" = *ARM64* ]]; then
      arch='arm'
    else
      arch='x86'
    fi
  elif [[ "$arch" = i*86 ]]; then
    arch='x86'
  elif [[ "$arch" = arm* ]]; then
    arch='arm'
  elif test "$arch" = aarch64; then
    arch='arm'
  else
    arch='unknown'
  fi
}

function show_help() {
  echo "Usage: $0 [-h] [-p] [-r <arch>] [-s] [-d <output_dir>] <task_name>"
  echo "Options:"
  echo "  -h              Show this help message"
  echo "  -p              Use existing cluster instead of creating a new one"
  echo "  -r <arch>       Specify the architecture (x86 or arm)"
  echo "  -s              Setup the cluster only, without running the task"
  echo "  -d <output_dir> Specify the output directory (default: eval/\${date}-\${task_name})"
}

function error() {
  echo -e "\e[31mError: $1\e[0m" >&2
  show_help
  exit 1
}

function set_architecture() {
  case "$1" in
    x86) arch='x86' ;;
    arm) arch='arm' ;;
    *) error "Invalid architecture: $1. Use 'x86' or 'arm'." ;;
  esac
}

function setup_cluster() {
  echo "=== Deleting existing kind cluster ==="
  kind delete cluster --name kind || true
  echo "=== Creating kind cluster ==="
  kind create cluster --config ./AIOpsLab/kind/kind-config-${arch}.yaml
}

detect_architecture

while getopts 'hpr:d:s' flag; do
  case "${flag}" in
    p) preserve_cluster='true' ;;
    r) set_architecture ${OPTARG} ;;
    d) output_dir="${OPTARG}" ;;
    h) show_help && exit 0 ;;
    s) setup_cluster_only='true' ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [[ "$setup_cluster_only" == "true" ]]; then
  setup_cluster
  exit 0
fi

shift $(($OPTIND - 1))
task_name=$1

if [[ -z "$task_name" ]]; then
  error "No task name provided. Usage: $0 <task_name>"
fi

if [[ -z "$output_dir" ]]; then
  current_date_time=$(date +"%m-%d_%H-%M-%S")
  output_dir="eval/${current_date_time}-${task_name}"
fi

if [[ "$preserve_cluster" == "false" ]]; then
  setup_cluster
fi

# ========== GraphRCA setup ==========
CURRENT_PATH=$(pwd)
AIOPSLAB_PATH=$CURRENT_PATH/AIOpsLab
GRAPHRCA_PATH=$CURRENT_PATH/GraphRCA_agent
STRATUS_SRC=$CURRENT_PATH/stratus/src

export TASK_NAME=$task_name
export KUBECONFIG=$HOME/.kube/config
export PYTHONPATH=${CURRENT_PATH}:${AIOPSLAB_PATH}:${GRAPHRCA_PATH}:${STRATUS_SRC}:$PYTHONPATH

mkdir -p ${output_dir}
mkdir -p ${output_dir}/graphrca_output

# ========== GraphRCA execution ==========
export OUTPUT_DIRECTORY=${output_dir}/graphrca_output

echo "=== Running GraphRCA agent ==="
echo "Task:       $task_name"
echo "Output dir: $output_dir"
echo ""

python -m GraphRCA_agent.run_pipeline \
  --aiopslab \
  --problem-id "$task_name" \
  --output-dir "${output_dir}/graphrca_output" \
  2>&1 | tee ${output_dir}/run.log

# ========== Result processing ==========
echo "=== Running log cleaning script ==="
python3 ./eval/clean_ansi_from_log.py ${output_dir}/run.log
