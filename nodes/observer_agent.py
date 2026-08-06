import os
import re
import csv
import json
import logging
from typing import List, Dict, Any
from swarm_state import AIOpsIncidentState
from tools.scratchpad_client import ScratchpadClient

try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None

logger = logging.getLogger(__name__)

class ObserverAgent:
    """
    Non-LLM rule engine.
    Parses AIOpsLab telemetry (traces, metrics, logs, kubectl) and outputs L1 triplets.
    Uses RapidFuzz for entity canonicalization.
    """
    def __init__(self, scratchpad_client: ScratchpadClient):
        self.client = scratchpad_client

    def canonicalize_entity(self, raw_name: str, candidate_pool: List[str] = None) -> str:
        """
        Canonicalizes raw pod/service strings using RapidFuzz if candidates provided,
        otherwise applies standard normalization.
        """
        if not raw_name:
            return "UNKNOWN"
            
        clean_name = raw_name.strip()
        if candidate_pool and process and fuzz:
            match = process.extractOne(clean_name, candidate_pool, scorer=fuzz.WRatio)
            if match and match[1] >= 75:
                return match[0].upper().replace("-", "_")
                
        # Heuristic fallback: strip K8s random hashes (e.g. frontend-6b4594c9-x2z9p -> FRONTEND)
        if "-" in clean_name:
            parts = clean_name.split("-")
            # If suffix looks like pod hash
            if len(parts) >= 3 and len(parts[-1]) >= 4:
                clean_name = "-".join(parts[:-2])
            elif len(parts) >= 2 and len(parts[-1]) >= 4:
                clean_name = "-".join(parts[:-1])
                
        return clean_name.upper().replace("-", "_")

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
            
        content = trace_csv_path_or_text
        if os.path.exists(trace_csv_path_or_text):
            try:
                with open(trace_csv_path_or_text, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read trace file {trace_csv_path_or_text}: {e}")
                return triplets
                
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return triplets
            
        span_to_svc = {}
        try:
            reader = csv.DictReader(lines)
            rows = list(reader)
            
            for row in rows:
                span_id = row.get("span_id", "")
                svc = self.canonicalize_entity(row.get("service_name", ""))
                span_to_svc[span_id] = svc
                
            for row in rows:
                span_id = row.get("span_id", "")
                parent_id = row.get("parent_span", "")
                svc = self.canonicalize_entity(row.get("service_name", ""))
                has_error = str(row.get("has_error", "false")).lower() == "true"
                response = str(row.get("response", ""))
                
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
                    svc = self.canonicalize_entity(pod)
                    status = metric.get("status", "")
                    if status and str(status).startswith("5"):
                        triplets.append({
                            "source": svc,
                            "relationship": "emits",
                            "target": f"HTTP_{status}_RATE_HIGH",
                            "citation_quote": f"pod={pod} status={status}"
                        })
        except Exception:
            pass
        return triplets

    def parse_logs(self, log_text: str, service: str = "") -> List[Dict[str, Any]]:
        """Parses raw logs using regex and canonicalization."""
        triplets = []
        svc = self.canonicalize_entity(service) if service else "UNKNOWN_SERVICE"
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
                svc = self.canonicalize_entity(pod)
                
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
        Dynamically parses raw telemetry passed in state and populates ScratchPad with L1 triplets.
        """
        session_id = state["scratchpad_session_id"]
        all_triplets = [
            {"source": "SYSTEM", "relationship": "has_task", "target": state["task_type"], "citation_quote": "AIOpsLab task started"}
        ]
        
        raw_telemetry = state.get("raw_telemetry") or {}
        trace_path = state.get("trace_csv_path") or raw_telemetry.get("trace_csv_path")
        
        # 1. Traces
        if trace_path:
            all_triplets.extend(self.parse_traces(trace_path))
            
        # 2. Logs
        logs = raw_telemetry.get("logs")
        if logs:
            all_triplets.extend(self.parse_logs(logs, service=state.get("namespace", "")))
            
        # 3. Kubectl
        kubectl_out = raw_telemetry.get("kubectl")
        if kubectl_out:
            all_triplets.extend(self.parse_kubectl(kubectl_out))
            
        # 4. Metrics
        metrics_json = raw_telemetry.get("metrics")
        if metrics_json:
            all_triplets.extend(self.parse_metrics(metrics_json))
            
        self.client.commit_triplets(session_id, "ObserverAgent", all_triplets)
        logger.info(f"[ObserverAgent] Parsed and committed {len(all_triplets)} L1 triplets to session {session_id}")
        return state
