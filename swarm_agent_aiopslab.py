import os
import time
import logging
from datetime import datetime
from GraphRCA_agent.graph import get_graph
from GraphRCA_agent.swarm_state import AIOpsIncidentState
from GraphRCA_agent.agent_aiopslab import GraphRCAAgent

logger = logging.getLogger(__name__)

class SwarmGraphRCAAgent(GraphRCAAgent):
    """
    Subclass that uses the 4-agent ScratchPad swarm instead of the legacy pipeline.
    """
    def _run_pipeline(self, trace_dir: str, reflection: str = "") -> dict:
        """Override to run the ScratchPad swarm graph."""
        start = time.time()
        
        # Save logs with timestamp as requested by user
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"agent_run_{self.namespace}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Starting SwarmGraphRCAAgent for namespace: {self.namespace}")
        
        trace_csv = os.path.join(trace_dir, "aiopslab_traces.csv")
        kubectl_out = self._run_kubectl(f"kubectl get pods -n {self.namespace}")
        
        # Set up state with raw telemetry
        state = AIOpsIncidentState(
            problem_id=f"incident_{self.namespace}",
            task_type=self.task_type,
            scratchpad_session_id=f"session_{self.namespace}_{timestamp}",
            namespace=self.namespace,
            raw_telemetry={
                "trace_csv_path": trace_csv,
                "kubectl": kubectl_out,
            },
            suspect_nodes=[],
            verified_root_cause=None,
            final_submission=None,
            retry_count=0,
            error=None
        )
        
        # Execute swarm graph
        graph = get_graph() # This will return the scratchpad_swarm if GRAPHRCA_AGENT_MODE is set
        try:
            final_state = graph.invoke(state)
            
            # Format report based on final state
            report = {
                "summary": {
                    "root_cause_service": final_state.get("verified_root_cause", "unknown")
                },
                "final_submission": final_state.get("final_submission"),
                "ttm_seconds": round(time.time() - start, 2)
            }
            logger.info(f"Swarm completed in {report['ttm_seconds']}s")
            
        except Exception as e:
            logger.error(f"Swarm graph failed: {e}")
            report = self._empty_report_template(error=str(e))
            
        logging.getLogger().removeHandler(file_handler)
        
        self.result = report
        return report

    def _submit_results(self, report: dict):
        """Override submission to use the exact output from Guardrail Actuator."""
        if self.stop_event.is_set():
            return
            
        try:
            submission = report.get("final_submission", {})
            action = submission.get("action")
            
            if action == "exec":
                command = submission.get("command")
                logger.info(f"[SwarmAgent] Executing mitigation: {command}")
                self._run_kubectl(command)
                self.send("```\nsubmit()\n```")
            elif action == "submit":
                val = submission.get("value")
                if val is not None:
                    if isinstance(val, str):
                        self.send(f'```\nsubmit("{val}")\n```')
                    else:
                        self.send(f'```\nsubmit({val})\n```')
                else:
                    self.send("```\nsubmit()\n```")
            else:
                super()._submit_results(report)
        except Exception as e:
            logger.error(f"[SwarmAgent] Submit failed: {e}")
            self._submit_default()
            
        self.stop_event.set()
