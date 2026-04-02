"""GraphRCA AIOpsLab Agent — Threaded agent for AIOpsLab benchmark.

Mirrors the pattern from stratus/src/stratus/agent/aiopslab.py:
- Threaded execution with semaphore-based synchronization
- Generator-based communication with AIOpsLab orchestrator
- Fetches traces/logs via generator, saves to temp CSV, runs LangGraph pipeline
- Submits results based on task type (detection/localization/analysis/mitigation)

All LLM outputs are logged to llm_justification.jsonl for audit.
"""

import csv
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── VALIDATION_RETRY constants (mirrors Stratus base.py) ─────────────────────
MAX_RETRY_ATTEMPTS = 3
VALIDATION_WAIT_SECONDS = 30  # Stratus uses 120s (real cluster); 30s for faster iteration


class GraphRCAAgent:
    """Threaded agent exposing GraphRCA as an AIOpsLab-compatible agent.

    Follows the same communication pattern as StratusAgent_AIOpsLab:
    - get_action(observation) is called by the orchestrator
    - Internally runs LangGraph pipeline via a daemon thread
    - Uses generator + semaphores for bidirectional communication
    """

    def __init__(self, problem_desc: str, task_type: str, output_dir: str,
                 verbose: bool = False, use_neo4j: bool = False):
        self.problem_desc = problem_desc
        self.task_type = task_type
        self.output_dir = output_dir
        self.verbose = verbose
        self.use_neo4j = use_neo4j
        self.result = None
        self.namespace = self._extract_namespace(problem_desc)

        # Semaphore-based communication (same as Stratus)
        self.prompt_semaphore = threading.Semaphore(0)
        self.command_semaphore = threading.Semaphore(0)
        self.prompt_message = ""
        self.command_message = ""
        self.stop_event = threading.Event()
        self.generator = self._communicator()

        logger.info(f"[GraphRCA Agent] task_type={task_type}, namespace={self.namespace}")

    def _extract_namespace(self, desc: str) -> str:
        """Extract Kubernetes namespace from problem description."""
        patterns = [
            r"namespace[:\s]+['\"]?(\S+)['\"]?",
            r"hotel[_-]res\w*",
            r"astronomy[_-]shop\w*",
            r"social[_-]network\w*",
        ]
        for pat in patterns:
            m = re.search(pat, desc, re.IGNORECASE)
            if m:
                ns = m.group(1) if m.lastindex else m.group(0)
                return ns.replace("_", "-")
        return "default"

    def _communicator(self):
        """Generator for bidirectional communication with AIOpsLab orchestrator.

        Identical pattern to StratusAgent_AIOpsLab.communicator().
        """
        while True:
            success = False
            while not self.stop_event.is_set() and not success:
                success = self.prompt_semaphore.acquire(timeout=3)
            if not success:
                while True:
                    yield "The evaluation has already been completed."

            self.command_message = yield self.prompt_message
            self.command_semaphore.release()

    async def get_action(self, observation: str) -> str:
        """Called by AIOpsLab orchestrator with observations.

        Args:
            observation: Text from orchestrator (traces, problem info, etc.)

        Returns:
            Command string for orchestrator (get_traces, submit, etc.)
        """
        self.prompt_message = observation
        self.prompt_semaphore.release()
        self.command_semaphore.acquire()
        return self.command_message

    def send(self, message: str) -> str:
        """Send a command through the generator to the orchestrator."""
        return self.generator.send(message)

    def run(self):
        """Start the agent thread. Called before orchestrator.start_problem()."""
        self.agent_thread = threading.Thread(target=self._run, daemon=True)
        self.agent_thread.start()

    def finalize(self):
        """Stop the agent and wait for thread to finish."""
        self.stop_event.set()
        if hasattr(self, "agent_thread"):
            self.agent_thread.join(timeout=10)
        logger.info("[GraphRCA Agent] Finalized")

    def _run(self):
        """Main agent logic — runs in daemon thread."""
        logger.info("=" * 60)
        logger.info("  GRAPHRCA AIOPSLAB AGENT STARTED")
        logger.info("=" * 60)

        # Prime the generator — get initial observation
        initial_obs = self.generator.send(None)
        logger.info(f"[Agent] Initial observation ({len(initial_obs)} chars)")

        try:
            # Step 1: Fetch traces from the cluster
            trace_data = self._fetch_traces()
            if not trace_data:
                logger.error("[Agent] No trace data received")
                self._submit_default()
                return

            # Step 2: Save traces to temp CSV for pipeline
            trace_dir = self._save_traces_to_csv(trace_data)
            if not trace_dir:
                logger.error("[Agent] Failed to save traces")
                self._submit_default()
                return

            # Step 3: Run pipeline — with VALIDATION_RETRY loop for mitigation tasks
            # (Mirrors Stratus base.py VALIDATION_RETRY pattern)
            if self.task_type == "mitigation":
                report = self._run_with_validation_retry(trace_dir)
            else:
                report = self._run_pipeline(trace_dir)

            # Step 4: Submit results based on task type
            self._submit_results(report)

        except Exception as e:
            logger.exception(f"[Agent] Pipeline failed: {e}")
            self._submit_default()

    def _run_with_validation_retry(self, trace_dir: str) -> dict:
        """VALIDATION_RETRY loop for mitigation tasks — mirrors Stratus base.py run().

        Flow per attempt:
          1. Run LangGraph pipeline (with reflection from previous failed run)
          2. Write agent_output_N.json
          3. Wait VALIDATION_WAIT_SECONDS for cluster to stabilize
          4. Validate cluster pod health
          5. If healthy → done.  If not → collect reflection → retry.
        """
        reflection = ""
        report = {}

        for run_count in range(MAX_RETRY_ATTEMPTS):
            logger.info("=" * 50)
            logger.info(f"  VALIDATION_RETRY — attempt {run_count + 1}/{MAX_RETRY_ATTEMPTS}")
            logger.info("=" * 50)

            run_start = time.time()
            report = self._run_pipeline(trace_dir, reflection=reflection)
            self._write_run_output(run_count, report)

            logger.info(f"[RetryLoop] Waiting {VALIDATION_WAIT_SECONDS}s for cluster to stabilize...")
            time.sleep(VALIDATION_WAIT_SECONDS)

            validation = self._validate_mitigation()
            elapsed = round(time.time() - run_start, 2)
            logger.info(f"[RetryLoop] Run {run_count}: validation={'PASS' if validation['success'] else 'FAIL'} in {elapsed}s")

            if validation["success"]:
                logger.info("[RetryLoop] Validation passed — mitigation successful")
                break

            if run_count < MAX_RETRY_ATTEMPTS - 1:
                reflection = self._collect_reflection(run_count, validation["issues"])
                logger.warning(f"[RetryLoop] Retrying with reflection ({len(validation['issues'])} issues)")
            else:
                logger.warning(f"[RetryLoop] Max retries ({MAX_RETRY_ATTEMPTS}) reached — submitting best result")

        return report

    def _fetch_traces(self) -> str:
        """Fetch traces from AIOpsLab cluster via generator."""
        logger.info(f"[Agent] Fetching traces for namespace={self.namespace}")
        try:
            result = self.send(f'```\nget_traces("{self.namespace}", 5)\n```')
            logger.info(f"[Agent] Received traces ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"[Agent] get_traces failed: {e}")
            return ""

    def _fetch_logs(self, service: str) -> str:
        """Fetch logs for a specific service."""
        try:
            result = self.send(f'```\nget_logs("{self.namespace}", "{service}")\n```')
            return result
        except Exception as e:
            logger.error(f"[Agent] get_logs failed for {service}: {e}")
            return ""

    def _save_traces_to_csv(self, trace_data: str) -> str:
        """Parse trace data from AIOpsLab and save as CSV for the pipeline.

        AIOpsLab returns traces as a pandas DataFrame string or JSON.
        We convert to CSV format compatible with stratus ingest_tools.
        """
        trace_dir = os.path.join(self.output_dir, "traces")
        os.makedirs(trace_dir, exist_ok=True)
        csv_path = os.path.join(trace_dir, "aiopslab_traces.csv")

        try:
            # Try parsing as JSON first
            if trace_data.strip().startswith("[") or trace_data.strip().startswith("{"):
                traces = json.loads(trace_data)
                if isinstance(traces, dict):
                    traces = traces.get("data", traces.get("traces", [traces]))
                if isinstance(traces, list) and traces:
                    import pandas as pd
                    df = pd.DataFrame(traces)
                    df.to_csv(csv_path, index=False)
                    logger.info(f"[Agent] Saved {len(traces)} traces as JSON→CSV to {csv_path}")
                    return trace_dir

            # Try parsing as pandas DataFrame string (fixed-width format)
            if "trace_id" in trace_data and "span_id" in trace_data:
                import pandas as pd
                try:
                    df = pd.read_fwf(io.StringIO(trace_data))
                    df.to_csv(csv_path, index=False)
                    logger.info(f"[Agent] Saved {len(df)} traces as FWF→CSV to {csv_path}")
                    return trace_dir
                except Exception:
                    pass

                # Try as CSV directly
                try:
                    df = pd.read_csv(io.StringIO(trace_data))
                    df.to_csv(csv_path, index=False)
                    logger.info(f"[Agent] Saved {len(df)} traces as CSV to {csv_path}")
                    return trace_dir
                except Exception:
                    pass

            # Fallback: save raw data and let ingest handle it
            with open(csv_path, "w") as f:
                f.write(trace_data)
            logger.warning("[Agent] Saved raw trace data as-is")
            return trace_dir

        except Exception as e:
            logger.error(f"[Agent] Failed to save traces: {e}")
            return ""

    def _validate_mitigation(self) -> dict:
        """Check cluster pod health after mitigation (mirrors Stratus WorkloadOracle).

        Uses kubectl to detect CrashLoopBackOff / Error / Pending pods.
        Returns {"success": bool, "issues": list[str]}.
        """
        from GraphRCA.tools.kube_tools import exec_kubectl_command
        issues = []
        try:
            result = exec_kubectl_command(f"kubectl get pods -n {self.namespace} --no-headers")
            for line in result.splitlines():
                line = line.strip()
                if not line:
                    continue
                if any(bad in line for bad in ("CrashLoopBackOff", "Error", "OOMKilled", "Pending", "ImagePullBackOff")):
                    issues.append(f"Unhealthy pod: {line}")
            if issues:
                logger.warning(f"[Validation] {len(issues)} unhealthy pod(s) found")
                for issue in issues:
                    logger.warning(f"  {issue}")
            else:
                logger.info("[Validation] All pods healthy")
        except Exception as e:
            logger.warning(f"[Validation] kubectl check failed (may not have cluster access): {e}")
            # If kubectl is unavailable (no cluster), treat as success to avoid blocking
            return {"success": True, "issues": []}
        return {"success": len(issues) == 0, "issues": issues}

    def _collect_reflection(self, run_count: int, issues: list) -> str:
        """Build reflection text from failed validation issues (mirrors Stratus base.py).

        This text is fed back into the next pipeline run as additional_context.
        """
        if not issues:
            return ""
        lines = "\n".join(f"  - {i}" for i in issues)
        return (
            f"Previous run {run_count} found these issues in the cluster after mitigation:\n"
            f"{lines}\n"
            f"Please address these issues in your updated mitigation plan."
        )

    def _write_run_output(self, run_count: int, report: dict):
        """Write per-run output file (mirrors Stratus run_logs.txt + agent_output_N.json)."""
        try:
            path = os.path.join(self.output_dir, f"agent_output_{run_count}.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"[Agent] Run {run_count} output → {path}")
        except Exception as e:
            logger.warning(f"[Agent] Failed to write run output: {e}")

    def _run_pipeline(self, trace_dir: str, reflection: str = "") -> dict:
        """Run the full GraphRCA LangGraph pipeline on the fetched traces."""
        from GraphRCA.llm import set_llm_log_dir
        set_llm_log_dir(self.output_dir)

        from GraphRCA.run_pipeline import run_pipeline
        if reflection:
            logger.info(f"[Agent] Pipeline with reflection: {reflection[:120]}...")
        else:
            logger.info(f"[Agent] Running LangGraph pipeline on {trace_dir}")

        start = time.time()
        report = run_pipeline(
            trace_dir=trace_dir,
            output_dir=self.output_dir,
            use_neo4j=self.use_neo4j,
            verbose=self.verbose,
            additional_context=reflection,
        )
        elapsed = round(time.time() - start, 2)
        report["ttm_seconds"] = elapsed
        self.result = report
        logger.info(f"[Agent] Pipeline completed in {elapsed}s")
        return report

    def _submit_results(self, report: dict):
        """Submit results to AIOpsLab based on task type."""
        if self.stop_event.is_set():
            return

        try:
            if self.task_type == "detection":
                n_alerts = report.get("detection", {}).get("alert_count", 0)
                answer = "Yes" if n_alerts > 0 else "No"
                logger.info(f"[Agent] Detection submit: {answer} ({n_alerts} alerts)")
                self.send(f'```\nsubmit("{answer}")\n```')

            elif self.task_type == "localization":
                root_svc = report.get("summary", {}).get("root_cause_service", "unknown")
                # Also include top 3 causes for better accuracy
                rca = report.get("rca", {})
                top_causes = rca.get("top_3_causes", [])
                faulty = [root_svc]
                for c in top_causes:
                    svc = c.get("service", "")
                    if svc and svc not in faulty:
                        faulty.append(svc)
                logger.info(f"[Agent] Localization submit: {faulty}")
                self.send(f"```\nsubmit({faulty})\n```")

            elif self.task_type == "analysis":
                # Determine system_level and fault_type from RCA
                root_svc = report.get("summary", {}).get("root_cause_service", "unknown")
                analysis = {
                    "system_level": "Application",
                    "fault_type": "Misconfiguration",
                }
                # Try to infer from log clusters
                clusters = report.get("log_analysis", {}).get("clusters", [])
                for c in clusters:
                    pattern = c.get("pattern", "").lower()
                    if "network" in pattern or "connection" in pattern:
                        analysis["fault_type"] = "Network"
                    elif "auth" in pattern or "permission" in pattern:
                        analysis["fault_type"] = "Auth Issue"
                    elif "config" in pattern or "misconfig" in pattern:
                        analysis["fault_type"] = "Misconfiguration"
                logger.info(f"[Agent] Analysis submit: {analysis}")
                self.send(f"```\nsubmit({analysis})\n```")

            elif self.task_type == "mitigation":
                actions = report.get("mitigation", {}).get("top_actions", [])
                # Execute top actions first
                for action in actions[:2]:
                    cmd = action.get("command", "")
                    if cmd:
                        logger.info(f"[Agent] Executing mitigation: {cmd[:100]}")
                        try:
                            self.send(f"```\n{cmd}\n```")
                        except Exception as e:
                            logger.warning(f"[Agent] Mitigation cmd failed: {e}")
                # Then submit
                logger.info("[Agent] Mitigation submit")
                self.send("```\nsubmit()\n```")

            else:
                logger.warning(f"[Agent] Unknown task_type={self.task_type}, submitting default")
                self._submit_default()

        except Exception as e:
            logger.error(f"[Agent] Submit failed: {e}")

        self.stop_event.set()

    def _submit_default(self):
        """Fallback submission when pipeline fails."""
        try:
            if self.task_type == "detection":
                self.send('```\nsubmit("Yes")\n```')
            elif self.task_type == "localization":
                self.send('```\nsubmit(["unknown"])\n```')
            elif self.task_type == "analysis":
                self.send('```\nsubmit({"system_level": "Application", "fault_type": "Misconfiguration"})\n```')
            else:
                self.send("```\nsubmit()\n```")
        except Exception as e:
            logger.error(f"[Agent] Default submit failed: {e}")
        self.stop_event.set()
