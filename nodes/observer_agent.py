import re
import csv
import json
import logging
from typing import List, Dict, Any
from swarm_state import AIOpsIncidentState
from tools.scratchpad_client import ScratchpadClient

logger = logging.getLogger(__name__)

class ObserverAgent:
    """
    Non-LLM rule engine.
    Parses AIOpsLab telemetry (traces, metrics, logs, kubectl) and outputs L1 triplets.
    """
    def __init__(self, scratchpad_client: ScratchpadClient):
        self.client = scratchpad_client

    def parse_traces(self, trace_csv_path_or_text: str) -> List[Dict[str, Any]]:
        """
        Parses Jaeger trace CSV data into triplets.
        Rules:
        1. parent->child -> calls
        2. error spans -> emits error
        3. high duration -> blocks
        """
        triplets = []
        if not trace_csv_path_or_text:
            return triplets
            
        # Basic heuristic parsing for CSV lines
        lines = trace_csv_path_or_text.strip().split('\n')
        if len(lines) < 2:
            return triplets
            
        # Map of span_id to service_name
        span_to_svc = {}
        # We need to parse headers or just use a generic regex if it's text
        try:
            reader = csv.DictReader(lines)
            rows = list(reader)
            
            for row in rows:
                span_id = row.get("span_id", "")
                svc = row.get("service_name", "").upper().replace("-", "_")
                span_to_svc[span_id] = svc
                
            for row in rows:
                span_id = row.get("span_id", "")
                parent_id = row.get("parent_span", "")
                svc = row.get("service_name", "").upper().replace("-", "_")
                has_error = row.get("has_error", "false").lower() == "true"
                response = row.get("response", "")
                
                # Rule 1: calls
                if parent_id and parent_id in span_to_svc:
                    parent_svc = span_to_svc[parent_id]
                    if parent_svc != svc:
                        triplets.append({
                            "source": parent_svc,
                            "relationship": "calls",
                            "target": svc,
                            "citation_quote": f"Trace span {span_id} parent {parent_id}"
                        })
                
                # Rule 2: emits error
                if has_error:
                    target_err = f"HTTP_{response}" if response else "ERROR"
                    triplets.append({
                        "source": svc,
                        "relationship": "emits",
                        "target": target_err,
                        "citation_quote": f"has_error=True response={response}"
                    })
        except Exception as e:
            logger.error(f"Error parsing traces: {e}")
            
        return triplets

    def parse_metrics(self, prometheus_json: str) -> List[Dict[str, Any]]:
        """Parses Prometheus metrics into triplets."""
        triplets = []
        try:
            data = json.loads(prometheus_json)
            for result in data.get("result", []):
                metric = result.get("metric", {})
                pod = metric.get("pod", "")
                if pod:
                    svc = "-".join(pod.split("-")[:-2]).upper().replace("-", "_") if "-" in pod else pod.upper()
                    status = metric.get("status", "")
                    if status and status.startswith("5"):
                        triplets.append({
                            "source": svc,
                            "relationship": "emits",
                            "target": f"HTTP_{status}_RATE_HIGH",
                            "citation_quote": f"pod={pod} status={status}"
                        })
        except Exception:
            pass
        return triplets

    def parse_logs(self, log_text: str, service: str) -> List[Dict[str, Any]]:
        """Parses raw logs using regex."""
        triplets = []
        svc = service.upper().replace("-", "_")
        for line in log_text.split('\n'):
            if "ERROR" in line.upper() or "FATAL" in line.upper():
                triplets.append({
                    "source": svc,
                    "relationship": "emits",
                    "target": "LOG_ERROR",
                    "citation_quote": line[:100]  # Verbatim substring
                })
            if "timeout" in line.lower():
                triplets.append({
                    "source": svc,
                    "relationship": "blocks",
                    "target": "TIMEOUT",
                    "citation_quote": line[:100]
                })
        return triplets

    def parse_kubectl(self, kubectl_output: str) -> List[Dict[str, Any]]:
        """Parses kubectl get pods output."""
        triplets = []
        lines = kubectl_output.strip().split('\n')
        for line in lines[1:]: # skip header
            parts = line.split()
            if len(parts) >= 3:
                pod = parts[0]
                status = parts[2]
                svc = "-".join(pod.split("-")[:-2]).upper().replace("-", "_") if "-" in pod else pod.upper()
                
                if status not in ["Running", "Completed"]:
                    triplets.append({
                        "source": svc,
                        "relationship": "emits",
                        "target": status.upper(),
                        "citation_quote": line[:100]
                    })
        return triplets
        
    def __call__(self, state: AIOpsIncidentState) -> AIOpsIncidentState:
        """
        In a real run, this node would execute API calls to AIOpsLab,
        get telemetry, and parse it. Since we intercept telemetry dynamically,
        this acts as a parsing orchestrator if needed.
        """
        # Commit a starting triplet
        self.client.commit_triplets(
            state["scratchpad_session_id"], 
            "ObserverAgent", 
            [{"source": "SYSTEM", "relationship": "has_task", "target": state["task_type"], "citation_quote": "AIOpsLab task started"}]
        )
        return state
