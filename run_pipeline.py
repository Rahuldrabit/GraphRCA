#!/usr/bin/env python3
"""GraphRCA Pipeline Runner — CLI entry point + AIOpsLab integration.

Usage:
  # Standalone (trace CSV files)
    python -m GraphRCA_agent.run_pipeline --trace-dir ./trace_output

  # Without Neo4j
    python -m GraphRCA_agent.run_pipeline --trace-dir ./trace_output --no-neo4j

  # AIOpsLab benchmark mode
    python -m GraphRCA_agent.run_pipeline --aiopslab --problem-id misconfig_app_hotel_res-detection-1

  # Verbose
    python -m GraphRCA_agent.run_pipeline --trace-dir ./trace_output -v
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


# ── Logging ──────────────────────────────────────────────────────────────────


def setup_logging(output_dir: str, verbose: bool = False) -> str:
    """Configure file + console logging."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "graphrca.log")
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(name)-22s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return log_file


# ── Neo4j ────────────────────────────────────────────────────────────────────


def clear_neo4j() -> bool:
    """Delete all nodes and relationships from Neo4j.

    Returns:
        True if cleared successfully, False otherwise.
    """
    uri = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    if not uri or not password:
        logging.error("Neo4j credentials not set (NEO4J_URI / NEO4J_PASSWORD)")
        return False

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            before = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            rels   = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            logging.info(f"Neo4j before clear: {before} nodes, {rels} relationships")

            try:
                from GraphRCA_agent.trace_logger import trace_event

                trace_event(
                    "neo4j.clear.start",
                    tool="neo4j",
                    nodes_before=before,
                    relationships_before=rels,
                )
            except Exception:
                pass

            session.run("MATCH (n) DETACH DELETE n")
            after = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
            logging.info(f"Neo4j cleared: {after} nodes remaining")

            try:
                from GraphRCA_agent.trace_logger import trace_event

                trace_event(
                    "neo4j.clear.end",
                    tool="neo4j",
                    nodes_after=after,
                )
            except Exception:
                pass
        driver.close()
        return True
    except Exception as e:
        logging.error(f"Neo4j clear failed: {e}")
        return False


def get_neo4j_connector():
    """Initialize Neo4j connector if configured."""
    enabled = os.getenv("NEO4J_ENABLED", "False").lower() == "true"
    if not enabled:
        logging.info("Neo4j disabled (NEO4J_ENABLED != True)")
        return None
    try:
        from GraphRCA_agent.tools.pipeline.neo4j_connector import get_neo4j_connector as _get
        connector = _get()
        if connector and connector.is_available():
            logging.info("Neo4j connector available")
            return connector
        return None
    except Exception as e:
        logging.warning(f"Neo4j init failed: {e}")
        return None


# ── Core pipeline runner ─────────────────────────────────────────────────────


def run_pipeline(
    trace_dir: str,
    output_dir: str = None,
    use_neo4j: bool = True,
    store_spans: bool = True,
    verbose: bool = False,
    additional_context: str = "",
) -> dict:
    """Run the GraphRCA LangGraph pipeline on trace CSV files.

    Args:
        trace_dir: Directory containing trace CSV files
        output_dir: Where to write reports
        use_neo4j: Whether to persist to Neo4j
        store_spans: Whether to store individual spans in Neo4j
        verbose: Enable debug logging

    Returns:
        Final pipeline state dict
    """
    if output_dir is None:
        ts = datetime.now().strftime("%m-%d_%H-%M-%S")
        output_dir = os.path.join("GraphRCA_output", ts)

    log_file = setup_logging(output_dir, verbose)

    # Configure LLM justification logging
    from GraphRCA_agent.llm import set_llm_log_dir
    set_llm_log_dir(output_dir)

    # Configure structured trace logging (tool calls, Neo4j, etc.)
    from GraphRCA_agent.trace_logger import set_trace_log_dir, trace_event
    set_trace_log_dir(output_dir)
    trace_event(
        "pipeline.start",
        caller="run_pipeline",
        trace_dir=trace_dir,
        output_dir=output_dir,
        use_neo4j=use_neo4j,
        store_spans=store_spans,
        verbose=verbose,
    )

    logger = logging.getLogger("graphrca.runner")

    logger.info("=" * 70)
    logger.info("  GRAPHRCA — LangGraph Autonomous SRE Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Trace dir:    {trace_dir}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(f"  Neo4j:        {use_neo4j}")
    logger.info(f"  Log:          {log_file}")
    logger.info("=" * 70)

    # Validate trace directory
    if not os.path.isdir(trace_dir):
        logger.error(f"Trace directory not found: {trace_dir}")
        sys.exit(1)
    csv_files = [f for f in os.listdir(trace_dir) if f.endswith(".csv")]
    if not csv_files:
        logger.error(f"No CSV trace files in: {trace_dir}")
        sys.exit(1)
    logger.info(f"Found {len(csv_files)} trace CSV files")

    # Neo4j
    neo4j_connector = None
    if use_neo4j:
        neo4j_connector = get_neo4j_connector()

    # Build initial state
    initial_state = {
        "trace_dir": trace_dir,
        "use_neo4j": use_neo4j and neo4j_connector is not None,
        "neo4j_connector": neo4j_connector,
        "store_spans": store_spans,
        "llm_kg_mode": os.getenv("GRAPHRCA_LLM_KG_MODE", "").strip().lower(),
        "output_dir": output_dir,
        "pipeline_start_time": time.time(),
        "status": "running",
        "rollback_count": 0,
        "rollback_triggered": False,
        "undo_stack": [],
        "sla_violations": [],
        "unhealthy_nodes": [],
        "messages": [],
        "node_timings": {},
        "additional_context": additional_context,
    }

    # Run the LangGraph
    from GraphRCA_agent.graph import get_graph
    app = get_graph()

    logger.info("\n🚀 Launching LangGraph pipeline...\n")
    pipeline_start = time.time()

    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
        final_state = {**initial_state, "status": "failed", "error": str(e)}

    elapsed = round(time.time() - pipeline_start, 2)

    trace_event(
        "pipeline.end",
        caller="run_pipeline",
        status=final_state.get("status", "unknown"),
        elapsed_seconds=elapsed,
        error=final_state.get("error", ""),
    )

    # Print message log
    for msg in final_state.get("messages", []):
        logger.info(f"  {msg}")

    # Build incident report
    ranked = final_state.get("ranked_causes", [])
    top_cause = ranked[0] if ranked else None
    root_svc = (top_cause.service if hasattr(top_cause, "service") else
                top_cause.get("service", "unknown") if top_cause else "unknown")
    root_conf = (top_cause.confidence if hasattr(top_cause, "confidence") else
                 top_cause.get("confidence", 0.0) if top_cause else 0.0)

    actions = final_state.get("mitigation_actions", [])
    top_actions = []
    for a in actions[:3]:
        p = a.priority if hasattr(a, "priority") else a.get("priority", "")
        t = a.title if hasattr(a, "title") else a.get("title", "")
        c = a.command if hasattr(a, "command") else a.get("command", "")
        top_actions.append({"priority": p, "title": t, "command": c[:120]})

    def _action_to_dict(a):
        if a is None:
            return {}
        if isinstance(a, dict):
            return a
        # Pydantic-style
        if hasattr(a, "dict") and callable(getattr(a, "dict")):
            try:
                return a.dict()
            except Exception:
                pass
        # Dataclass / object
        if hasattr(a, "__dict__"):
            try:
                return dict(a.__dict__)
            except Exception:
                pass
        return {"raw": str(a)}

    full_actions = [_action_to_dict(a) for a in actions]

    report = {
        "incident_id": final_state.get("incident_id", "INC-unknown"),
        "timestamp": datetime.now(tz=None).isoformat() + "Z",
        "status": final_state.get("status", "unknown"),
        "pipeline_elapsed_seconds": elapsed,
        "summary": {
            "primary_error_service": final_state.get("primary_error_service", "unknown"),
            "root_cause_service": root_svc,
            "root_cause_confidence": round(root_conf, 3),
            "alerts_detected": len(final_state.get("alerts", [])),
            "rollback_triggered": final_state.get("rollback_triggered", False),
            "rollback_count": final_state.get("rollback_count", 0),
            "health_score_before": final_state.get("health_score_before", 0.0),
            "health_score_after": final_state.get("health_score_after", 0.0),
        },
        "knowledge_graph": final_state.get("graph_summary", {}),
        "detection": {
            "alert_count": len(final_state.get("alerts", [])),
            "primary_service": final_state.get("primary_error_service", ""),
        },
        "rca": {
            "top_3_causes": [
                {
                    "service": c.service if hasattr(c, "service") else c.get("service", ""),
                    "confidence": round(c.confidence if hasattr(c, "confidence") else c.get("confidence", 0), 3),
                    "is_silent": c.is_silent_failure if hasattr(c, "is_silent_failure") else False,
                }
                for c in ranked[:3]
            ],
            "causal_scores": final_state.get("causal_scores", {}),
            "temporal_order": final_state.get("temporal_order", []),
            "llm_kg": final_state.get("llm_kg", {}),
        },
        "log_analysis": {
            "clusters": final_state.get("log_clusters", []),
        },
        "mitigation": {
            "action_count": len(actions),
            "top_actions": top_actions,
            "actions": full_actions,
        },
        "memory": {
            "similar_cases_found": len(final_state.get("similar_cases", [])),
            "stored": final_state.get("memory_stored", False),
        },
        "node_timings": final_state.get("node_timings", {}),
        "error": final_state.get("error", ""),
    }

    # Save report
    report_path = os.path.join(output_dir, "incident_report.json")
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\n📄 Report saved to: {report_path}")
    except Exception as e:
        logger.warning(f"Could not save report: {e}")

    # Summary banner
    logger.info("\n" + "=" * 70)
    if final_state.get("status") in ("complete", "running"):
        logger.info(f"  ✅ Pipeline completed in {elapsed:.2f}s")
    else:
        logger.info(f"  ❌ Pipeline failed: {final_state.get('error', 'unknown')}")
    logger.info(f"  Root cause:  {root_svc} (confidence={root_conf:.1%})")
    logger.info(f"  Alerts:      {len(final_state.get('alerts', []))}")
    logger.info(f"  Actions:     {len(actions)}")
    logger.info(f"  Rollbacks:   {final_state.get('rollback_count', 0)}")
    logger.info(f"  μ(s) before: {final_state.get('health_score_before', 0):.4f}")
    logger.info(f"  μ(s) after:  {final_state.get('health_score_after', 0):.4f}")
    logger.info("=" * 70)

    return report


# ── AIOpsLab Integration (Pillar 4) ──────────────────────────────────────────


def run_aiopslab(problem_id: str = None, output_dir: str = None, verbose: bool = False) -> dict:
    """Run GraphRCA against an AIOpsLab benchmark problem.

    Mirrors the pattern from stratus/src/stratus/main.py exactly:
    - Threaded agent with generator-based communication
    - Fetches traces/logs via AIOpsLab orchestrator
    - Runs full LangGraph pipeline on fetched data
    - Submits results based on task type

    All LLM outputs are logged to llm_justification.jsonl for audit.

    Args:
        problem_id: AIOpsLab problem ID (e.g., "misconfig_app_hotel_res-detection-1").
                    Falls back to TASK_NAME env var.
        output_dir: Output directory for reports.
                    Falls back to OUTPUT_DIRECTORY env var.
        verbose: Debug logging

    Returns:
        Benchmark result dict with TTM and success flag
    """
    # Support env-var fallbacks (for test_graphrca.sh integration)
    if problem_id is None:
        problem_id = os.environ.get("TASK_NAME", "misconfig_app_hotel_res-detection-1")
    if output_dir is None:
        output_dir = os.environ.get("OUTPUT_DIRECTORY")
    if output_dir is None:
        ts = datetime.now().strftime("%m-%d_%H-%M-%S")
        output_dir = os.path.join("GraphRCA_output", f"aiopslab-{problem_id[:30]}-{ts}")

    log_file = setup_logging(output_dir, verbose)

    # Configure LLM justification logging
    from GraphRCA_agent.llm import set_llm_log_dir
    set_llm_log_dir(output_dir)

    # Configure structured trace logging (tool calls, Neo4j, etc.)
    from GraphRCA_agent.trace_logger import set_trace_log_dir, trace_event
    set_trace_log_dir(output_dir)
    trace_event(
        "aiopslab.start",
        caller="run_aiopslab",
        problem_id=problem_id,
        output_dir=output_dir,
        verbose=verbose,
    )

    logger = logging.getLogger("graphrca.aiopslab")

    logger.info("=" * 70)
    logger.info("  GRAPHRCA — AIOpsLab Benchmark Mode")
    logger.info("=" * 70)
    logger.info(f"  Problem ID:   {problem_id}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(f"  Log:          {log_file}")
    logger.info("=" * 70)

    # Import AIOpsLab
    try:
        from aiopslab.orchestrator import Orchestrator
    except ImportError as e:
        logger.error(f"AIOpsLab not installed. Add AIOpsLab to PYTHONPATH. Error: {e}")
        return {"success": False, "error": f"AIOpsLab not available: {e}"}

    # Determine task type from problem_id
    task_type = next(
        (t for t in ["detection", "mitigation", "localization", "analysis"] if t in problem_id),
        "unknown",
    )
    logger.info(f"[AIOpsLab] Task type: {task_type}")

    # Initialize orchestrator (same as Stratus main.py)
    orchestrator = Orchestrator()
    orchestrator.agent_name = "GraphRCA_LangGraph"
    problem_desc, instructions, apis = orchestrator.init_problem(problem_id)

    logger.info(f"[AIOpsLab] Problem:\n{problem_desc[:500]}")
    logger.info(f"[AIOpsLab] Instructions:\n{str(instructions)[:300]}")

    # Create threaded agent (mirrors StratusAgent_AIOpsLab)
    from GraphRCA_agent.agent_aiopslab import GraphRCAAgent
    use_neo4j = os.getenv("NEO4J_ENABLED", "False").lower() == "true"

    agent = GraphRCAAgent(
        problem_desc=problem_desc,
        task_type=task_type,
        output_dir=output_dir,
        verbose=verbose,
        use_neo4j=use_neo4j,
    )

    # Register and start (same pattern as Stratus)
    orchestrator.register_agent(agent, name=orchestrator.agent_name)
    agent.run()

    benchmark_start = time.time()
    logger.info("[AIOpsLab] Starting orchestrator...")
    asyncio.run(orchestrator.start_problem(max_steps=30))
    total_elapsed = round(time.time() - benchmark_start, 2)

    agent.finalize()

    trace_event(
        "aiopslab.end",
        caller="run_aiopslab",
        problem_id=problem_id,
        elapsed_seconds=total_elapsed,
        task_type=task_type,
    )

    # ── Extract AIOpsLab evaluation results ──────────────────────────────
    eval_results = {}
    try:
        session = orchestrator.session
        if hasattr(session, "results") and session.results:
            eval_results = dict(session.results)
    except Exception as e:
        logger.warning(f"[AIOpsLab] Could not extract eval results: {e}")

    # Save eval_results.json
    if eval_results:
        eval_path = os.path.join(output_dir, "eval_results.json")
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        logger.info(f"[AIOpsLab] Eval results saved: {eval_path}")

    # ── Print Stratus-format evaluation output ───────────────────────────
    _print_eval_banner(task_type, eval_results, output_dir, agent)

    # Build result
    result = agent.result or {}
    result["total_elapsed_seconds"] = total_elapsed
    result["problem_id"] = problem_id
    result["task_type"] = task_type
    result["eval_results"] = eval_results

    # Save benchmark result
    bench_path = os.path.join(output_dir, "benchmark_result.json")
    with open(bench_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"[AIOpsLab] Benchmark result saved: {bench_path}")

    # Summary
    logger.info("=" * 70)
    logger.info(f"  Problem:   {problem_id}")
    logger.info(f"  Task:      {task_type}")
    logger.info(f"  Elapsed:   {total_elapsed}s")
    logger.info(f"  Output:    {output_dir}")
    logger.info("=" * 70)

    return result


def _print_eval_banner(task_type: str, eval_results: dict, output_dir: str, agent):
    """Print evaluation results in the same format as Stratus run.log output."""
    # Validation result line
    validation_success = eval_results.get("success", None)
    if validation_success is None:
        # Detection tasks in AIOpsLab typically don't set a `success` key.
        if task_type == "detection":
            validation_success = eval_results.get("Detection Accuracy") == "Correct"
        else:
            # Default to True when the task doesn't define a success criterion.
            validation_success = True
    validation_success = bool(validation_success)
    validation_issues = eval_results.get("issues", [])
    print(f"\nValidation result: {{'success': {validation_success}, 'issues': {validation_issues}}}")
    if validation_success:
        print("######### VALIDATION SUCCESSFUL #########")
    else:
        print("######### VALIDATION FAILED #########")

    # Output written line
    last_output = os.path.join(output_dir, f"agent_output_{max(agent._run_count - 1, 0)}.json")
    print(f"Output written to: {last_output}")

    # Evaluation section
    print("== Evaluation ==")

    # Task-type-specific accuracy
    metric_map = {
        "detection": ("Detection Accuracy", "TTD"),
        "localization": ("Localization Accuracy", "TTL"),
        "analysis": ("system_level_correct", "TTA"),
        "mitigation": ("Mitigation Success", "TTM"),
    }
    accuracy_key, time_key = metric_map.get(task_type, ("Accuracy", "Time"))

    if task_type == "mitigation":
        # Mitigation problems set `success` boolean.
        accuracy_val = bool(eval_results.get("success", False))
    else:
        accuracy_val = eval_results.get(accuracy_key, "N/A")
    if task_type == "detection":
        correct = "Yes" if accuracy_val == "Correct" else "No"
        print(f"Correct detection: {correct}")

    # Compact results dict (mirrors Stratus format)
    results_summary = {
        accuracy_key: accuracy_val,
        time_key: eval_results.get(time_key, eval_results.get("TTD", 0)),
        "steps": eval_results.get("steps", eval_results.get("num_steps_taken", 0)),
        "in_tokens": eval_results.get("in_tokens", 0),
        "out_tokens": eval_results.get("out_tokens", 0),
    }
    print(f"\033[35mResults:\n\033[0m{results_summary}")

    # Fault recovery section
    namespace = getattr(agent, "namespace", "unknown")
    root_svc = "unknown"
    if agent.result and isinstance(agent.result, dict):
        root_svc = agent.result.get("summary", {}).get("root_cause_service", "unknown")
    print(f"== Fault Recovery ==")
    print(f"Recovering for service: {root_svc} | namespace: {namespace}")
    print(f"Service: {root_svc} | Namespace: {namespace}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="GraphRCA — LangGraph Autonomous SRE Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on trace CSV files (no Neo4j)
    python -m GraphRCA_agent.run_pipeline --trace-dir ./trace_output --no-neo4j

  # Run with Neo4j and verbose logging
    python -m GraphRCA_agent.run_pipeline --trace-dir ./trace_output -v

  # AIOpsLab benchmark mode
    python -m GraphRCA_agent.run_pipeline --aiopslab --problem-id misconfig_app_hotel_res-detection-1
        """,
    )

    parser.add_argument("--trace-dir", "-t", default="./trace_output",
                        help="Directory containing trace CSV files")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: GraphRCA_output/<timestamp>)")
    parser.add_argument("--no-neo4j", action="store_true",
                        help="Disable Neo4j storage")
    parser.add_argument("--no-spans", action="store_true",
                        help="Skip storing individual spans in Neo4j")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--aiopslab", action="store_true",
                        help="Run in AIOpsLab benchmark mode")
    parser.add_argument("--problem-id", default="misconfig_app_hotel_res-detection-1",
                        help="AIOpsLab problem ID (used with --aiopslab)")
    parser.add_argument("--clear-neo4j", action="store_true",
                        help="Delete all nodes and relationships from Neo4j, then exit")

    args = parser.parse_args()

    if args.clear_neo4j:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
        ok = clear_neo4j()
        sys.exit(0 if ok else 1)

    if args.aiopslab:
        result = run_aiopslab(
            problem_id=args.problem_id or os.environ.get("TASK_NAME"),
            output_dir=args.output_dir or os.environ.get("OUTPUT_DIRECTORY"),
            verbose=args.verbose,
        )
        print(json.dumps({"status": "done", "problem_id": result.get("problem_id", args.problem_id),
                          "ttm": result.get("ttm_seconds", 0)}, indent=2))
    else:
        result = run_pipeline(
            trace_dir=args.trace_dir,
            output_dir=args.output_dir,
            use_neo4j=not args.no_neo4j,
            store_spans=not args.no_spans,
            verbose=args.verbose,
        )
        status = result.get("status", "unknown")
        sys.exit(0 if status in ("complete", "running") else 1)


if __name__ == "__main__":
    main()
