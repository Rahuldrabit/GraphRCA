import re
import json
import logging
from typing import Dict, Any
from swarm_state import AIOpsIncidentState
from llm import llm_reason

logger = logging.getLogger(__name__)

class GuardrailActuator:
    """
    SLM + deterministic safety regex.
    Decides the final AIOpsLab submission or mitigation command.
    """
    def __init__(self):
        # Patterns that are absolutely forbidden
        self.forbidden_patterns = [
            r"rm\s+-rf",
            r"drop\s+database",
            r"delete\s+ns",
            r"delete\s+namespace",
            r"sudo\s+rm"
        ]

    def _is_safe(self, command: str) -> bool:
        """Deterministic safety check."""
        cmd_lower = command.lower()
        for pattern in self.forbidden_patterns:
            if re.search(pattern, cmd_lower):
                logger.error(f"Safety violation: command matched forbidden pattern '{pattern}'")
                return False
        return True

    def __call__(self, state: AIOpsIncidentState) -> AIOpsIncidentState:
        task_type = state["task_type"]
        root_cause = state.get("verified_root_cause")

        # Detection is a binary decision driven by the observer's anomaly flag —
        # no LLM needed, and it correctly answers "No" for noop (no-fault) tasks.
        if task_type == "detection":
            detected = bool(state.get("anomaly_detected", False))
            state["final_submission"] = {"action": "submit", "value": "Yes" if detected else "No"}
            logger.info(f"[Guardrail] detection submission: {'Yes' if detected else 'No'}")
            return state

        if not root_cause:
            logger.warning("No root cause verified. Sending default submission.")
            self._set_default_submission(state)
            return state
            
        system_prompt = (
            f"You are an SRE agent finalizing a {task_type} task for AIOpsLab. "
            f"The verified root cause is: {root_cause}. "
            "Based on the task type, format the final submission strictly as JSON."
        )
        
        user_prompt = ""
        if task_type == "detection":
            user_prompt = "Return JSON: {\"submission\": \"Yes\"}"
        elif task_type == "localization":
            user_prompt = f"Return JSON: {{\"submission\": [\"{root_cause}\"]}}"
        elif task_type == "analysis":
            user_prompt = (
                "Return JSON with 'system_level' (Hardware|Operating System|Virtualization|Application) "
                "and 'fault_type' (Misconfiguration|Code Defect|Authentication Issue|Network/Storage Issue|Operation Error|Dependency Problem). "
                f"Example: {{\"submission\": {{\"system_level\": \"Application\", \"fault_type\": \"Code Defect\"}}}}"
            )
        elif task_type == "mitigation":
            user_prompt = (
                f"The root cause is {root_cause}. Propose a kubectl mitigation command. "
                "Return JSON: {\"command\": \"kubectl ...\"}"
            )
            
        response = llm_reason(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=200,
            caller="guardrail_actuator"
        )
        
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(response[start:end])
                
                if task_type == "mitigation":
                    command = data.get("command", "")
                    if command and self._is_safe(command):
                        # In a real run, this node yields the command for the orchestrator to execute
                        # and then submits empty when done.
                        # We represent this by setting it to the state.
                        state["final_submission"] = {"action": "exec", "command": command}
                    else:
                        logger.warning("Unsafe or missing mitigation command.")
                        state["final_submission"] = {"action": "submit", "value": None}
                else:
                    state["final_submission"] = {"action": "submit", "value": data.get("submission")}
            else:
                self._set_default_submission(state)
        except Exception as e:
            logger.error(f"Guardrail failed to parse response: {e}")
            self._set_default_submission(state)
            
        return state

    def _set_default_submission(self, state: AIOpsIncidentState):
        task_type = state["task_type"]
        root_cause = state.get("verified_root_cause", "unknown")
        
        if task_type == "detection":
            val = "Yes"
        elif task_type == "localization":
            val = [root_cause]
        elif task_type == "analysis":
            val = {"system_level": "Application", "fault_type": "Code Defect"}
        else:
            val = None
            
        state["final_submission"] = {"action": "submit", "value": val}
