import os
import re
import csv
import glob
import time
import logging
from datetime import datetime

from GraphRCA_agent.graph import get_graph
from GraphRCA_agent.swarm_state import AIOpsIncidentState
from GraphRCA_agent.agent_aiopslab import GraphRCAAgent

logger = logging.getLogger(__name__)

# Cap how many services we pull logs for (each costs one orchestrator step).
_MAX_LOG_SVCS = 6


def _canonical_svc(name: str) -> str:
    """Strip k8s pod replica/hash suffix; preserve case. Mirrors ObserverAgent."""
    if not name:
        return ""
    s = str(name).strip().strip('"').strip("'")
    if not s:
        return ""
    m = re.match(r"^(.+)-[a-z0-9]{5,}-[a-z0-9]{4,}$", s)
    if m:
        return m.group(1)
    m = re.match(r"^(.+)-(\d+)$", s)
    if m and m.group(1):
        return m.group(1)
    return s


class SwarmGraphRCAAgent(GraphRCAAgent):
    """Subclass that uses the 4-agent ScratchPad swarm instead of the legacy pipeline.

    Fetches MULTIPLE telemetry types (traces + pod status + metrics + logs) rather
    than bailing out when Jaeger traces are empty — many AIOpsLab faults (pod kill,
    container kill, scale-to-zero, k8s misconfig, resource saturation) produce no
    traces, and their signal lives in `kubectl get pods`, Prometheus, or logs.
    """

    # ── telemetry collection ─────────────────────────────────────────────

    def _collect_all_telemetry(self) -> tuple:
        """Fetch every available telemetry source into <output_dir>/traces/.

        Returns (trace_dir, telemetry_dict) where telemetry_dict maps source
        labels to on-disk file paths and sets has_signal iff any source had data.
        """
        trace_dir = os.path.join(self.output_dir, "traces")
        os.makedirs(trace_dir, exist_ok=True)
        tel: dict = {"has_signal": False}

        # 1) Traces (orchestrator get_traces) — primary signal for instrumented apps.
        try:
            trace_data = self._fetch_traces()
            if trace_data:
                d = self._save_traces_to_csv(trace_data)
                if d:
                    trace_dir = d
            csv_path = os.path.join(trace_dir, "aiopslab_traces.csv")
            if os.path.exists(csv_path) and self._csv_has_data_rows(csv_path):
                tel["trace_csv"] = csv_path
                tel["has_signal"] = True
                logger.info(f"[SwarmAgent] traces OK: {csv_path}")
            else:
                logger.info("[SwarmAgent] traces empty/absent — relying on other telemetry")
        except Exception as e:
            logger.warning(f"[SwarmAgent] trace collection failed: {e}")

        # 2) Pod status (LOCAL kubectl — free, no orchestrator step, very reliable).
        # This is the primary signal for pod/container/scale/k8s faults.
        try:
            pods = self._run_kubectl("kubectl get pods")
            if pods and "No resources found" not in pods and not pods.startswith("Error"):
                pods_path = os.path.join(trace_dir, "pods.txt")
                with open(pods_path, "w", encoding="utf-8") as f:
                    f.write(pods)
                tel["pods"] = pods_path
                tel["has_signal"] = True
                logger.info("[SwarmAgent] pod status OK (kubectl)")
        except Exception as e:
            logger.warning(f"[SwarmAgent] pod status collection failed: {e}")

        # 3) Metrics (orchestrator get_metrics -> Prometheus). Best-effort.
        try:
            self._collect_metrics(trace_dir, tel)
        except Exception as e:
            logger.warning(f"[SwarmAgent] metrics collection failed: {e}")

        # 4) Logs (orchestrator get_logs, top services). Best-effort.
        try:
            self._collect_logs(trace_dir, tel)
        except Exception as e:
            logger.warning(f"[SwarmAgent] logs collection failed: {e}")

        # Telemetry is now on disk; the observer port-forwards (prometheus/jaeger)
        # are dead weight whose cleanup is unreliable — a lingering child blocks
        # task teardown and hangs the batch. Kill them now (scrape is already done).
        self._kill_observer_port_forwards()
        logger.info(f"[SwarmAgent] telemetry sources: {[k for k in tel if k != 'has_signal']} "
                    f"(has_signal={tel['has_signal']})")
        return trace_dir, tel

    def _kill_observer_port_forwards(self) -> None:
        """Tear down AIOpsLab observer `kubectl port-forward` children.

        get_metrics/get_traces spawn port-forwards (metric_api/trace_api) stored
        as `self.port_forward_process` whose stop-cleanup is unreliable; the
        lingering child keeps run_pipeline (and thus the whole batch) alive after
        the task result is already saved. Safe to call once telemetry is on disk.
        Matches ONLY observer port-forwards — never arbitrary kubectl.
        """
        import subprocess
        for pat in (
            r"kubectl port-forward .*prometheus-server",
            r"kubectl port-forward .*svc/jaeger",
            r"kubectl port-forward .*16686",
        ):
            try:
                subprocess.run(["pkill", "-9", "-f", pat], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.debug(f"[SwarmAgent] pkill {pat!r} failed: {e}")

    def _collect_metrics(self, trace_dir: str, tel: dict) -> None:
        msg = self._clean_aiopslab_text(self.send(f'```\nget_metrics("{self.namespace}", 5)\n```'))
        # The response prints "Metrics data exported to directory: <path>"; locate it.
        dpath = ""
        m = re.search(r"([^\s\n]*metric_\d+_\d+)", msg or "")
        if m:
            dpath = m.group(1).strip()
        if not dpath or not os.path.isdir(dpath):
            return
        csvs = glob.glob(os.path.join(dpath, "**", "kpi_*.csv"), recursive=True)
        if not csvs:
            return
        out = os.path.join(trace_dir, "metrics_summary.csv")
        with open(out, "w", encoding="utf-8") as f:
            f.write("metric,cmdb_id,kpi_name,value\n")
            for c in csvs[:40]:
                metric = os.path.basename(c)[4:-4]
                try:
                    with open(c, encoding="utf-8", errors="replace") as fr:
                        for row in csv.reader(fr):
                            if len(row) >= 4:
                                f.write(f"{metric},{row[1]},{row[2]},{row[3]}\n")
                except Exception:
                    continue
        tel["metrics"] = out
        tel["has_signal"] = True
        logger.info(f"[SwarmAgent] metrics OK: {len(csvs)} kpi files summarized")

    def _collect_logs(self, trace_dir: str, tel: dict) -> None:
        services = self._discover_services(trace_dir)
        if not services:
            return
        sections = []
        for svc in services[:_MAX_LOG_SVCS]:
            try:
                logs = self._clean_aiopslab_text(
                    self.send(f'```\nget_logs("{self.namespace}", "{svc}")\n```')
                )
            except Exception:
                logs = ""
            if logs and "does not exist" not in logs and not logs.startswith("Error"):
                sections.append(f"=== {svc} ===\n" + logs[:4000])
            if len(sections) >= _MAX_LOG_SVCS:
                break
        if sections:
            out = os.path.join(trace_dir, "logs.txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write("\n".join(sections))
            tel["logs"] = out
            tel["has_signal"] = True
            logger.info(f"[SwarmAgent] logs OK: {len(sections)} services")

    def _discover_services(self, trace_dir: str) -> list:
        """Build a deduped service list from pods.txt + traces CSV for log fetching."""
        raw: list = []
        pods_path = os.path.join(trace_dir, "pods.txt")
        if os.path.exists(pods_path):
            try:
                with open(pods_path, encoding="utf-8", errors="replace") as f:
                    for line in f.read().splitlines()[1:]:
                        parts = line.split()
                        if parts:
                            raw.append(parts[0])
            except Exception:
                pass
        csv_path = os.path.join(trace_dir, "aiopslab_traces.csv")
        if os.path.exists(csv_path):
            try:
                with open(csv_path, encoding="utf-8", errors="replace") as f:
                    for row in csv.DictReader(f):
                        if row.get("service_name"):
                            raw.append(row["service_name"])
            except Exception:
                pass

        infra = {"jaeger", "prometheus", "loadbalancer", "nginx", "istio", "loki",
                 "grafana", "otelcollector", "opentelemetry", "chaos", "wrk2-job", "wrk"}
        seen, out = set(), []
        for s in raw:
            c = _canonical_svc(s)
            if not c or c.lower() in infra or c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out

    # ── main daemon-thread entry point (overrides base trace-only _run) ──

    def _run(self):
        logger.info("=" * 60)
        logger.info("  SWARM AIOPSLAB AGENT STARTED (multi-telemetry)")
        logger.info("=" * 60)

        # Prime the generator (get initial observation).
        try:
            self.generator.send(None)
        except Exception as e:
            logger.exception(f"[SwarmAgent] generator prime failed: {e}")
            self._submit_default()
            return

        try:
            trace_dir, tel = self._collect_all_telemetry()

            if not tel.get("has_signal"):
                logger.error("[SwarmAgent] All telemetry sources empty — using fallback")
                report = self._fallback_no_traces(reason="all telemetry sources empty")
                self._run_count = 1
                self._write_run_output(0, report)
                self._submit_results(report)
                return

            if self.task_type == "mitigation":
                report = self._run_with_validation_retry(trace_dir)
            else:
                report = self._run_pipeline(trace_dir)
                self._run_count = 1
                self._write_run_output(0, report)
                self._append_run_log(0, report, {"success": True, "issues": []}, "N/A")

            self._submit_results(report)

        except Exception as e:
            logger.exception(f"[SwarmAgent] pipeline failed: {e}")
            self._submit_default()
        finally:
            # Last-resort cleanup: never let a leaked observer port-forward keep
            # this task (and the batch) alive after we are done.
            self._kill_observer_port_forwards()

    # ── swarm graph invocation ───────────────────────────────────────────

    def _run_pipeline(self, trace_dir: str, reflection: str = "") -> dict:
        """Run the ScratchPad swarm graph over all collected telemetry files."""
        start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"agent_run_{self.namespace}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)

        logger.info(f"Starting SwarmGraphRCAAgent for namespace: {self.namespace}")

        def _path(name):
            p = os.path.join(trace_dir, name)
            return p if os.path.exists(p) else None

        state = AIOpsIncidentState(
            problem_id=self.namespace,
            task_type=self.task_type,
            problem_description=self.problem_desc,
            scratchpad_session_id=f"session_{self.namespace}_{timestamp}",
            trace_csv_path=_path("aiopslab_traces.csv"),
            pod_status_path=_path("pods.txt"),
            metrics_path=_path("metrics_summary.csv"),
            logs_path=_path("logs.txt"),
            suspect_nodes=[],
            verified_root_cause=None,
            anomaly_detected=False,
            final_submission=None,
            retry_count=0,
            error=None,
        )

        try:
            graph = get_graph()  # returns the scratchpad_swarm graph when env is set
            final_state = graph.invoke(state)
            report = {
                "summary": {
                    "root_cause_service": final_state.get("verified_root_cause") or "unknown",
                    "anomaly_detected": bool(final_state.get("anomaly_detected", False)),
                },
                "detection": {
                    "alert_count": 1 if final_state.get("anomaly_detected") else 0,
                    "primary_service": final_state.get("verified_root_cause") or "",
                },
                "final_submission": final_state.get("final_submission"),
                "ttm_seconds": round(time.time() - start, 2),
            }
            logger.info(f"Swarm completed in {report['ttm_seconds']}s "
                        f"(anomaly={report['summary']['anomaly_detected']}, "
                        f"root_cause={report['summary']['root_cause_service']})")
        except Exception as e:
            logger.error(f"Swarm graph failed: {e}")
            report = self._empty_report_template(error=str(e))

        logging.getLogger().removeHandler(file_handler)
        self.result = report
        return report

    # ── submission (use the Guardrail Actuator output directly) ───────────

    def _submit_results(self, report: dict):
        if self.stop_event.is_set():
            return

        try:
            submission = report.get("final_submission", {}) or {}
            action = submission.get("action")

            if action == "exec":
                command = submission.get("command")
                logger.info(f"[SwarmAgent] Executing mitigation: {command}")
                if command:
                    self._run_kubectl(command)
                self.send("```\nsubmit()\n```")
            elif action == "submit":
                val = submission.get("value")
                if isinstance(val, str):
                    self.send(f'```\nsubmit("{val}")\n```')
                elif val is not None:
                    self.send(f'```\nsubmit({val})\n```')
                else:
                    self.send("```\nsubmit()\n```")
            else:
                super()._submit_results(report)
        except Exception as e:
            logger.error(f"[SwarmAgent] Submit failed: {e}")
            self._submit_default()

        self.stop_event.set()
