import os
import re
import json
import logging
from typing import Dict, Any
from swarm_state import AIOpsIncidentState
from llm import llm_reason

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


# Token budget for the evidence view handed to the analysis classifier.
# Previously this call got NO telemetry at all -- just the bare root-cause
# service name string -- and had to guess system_level/fault_type with no
# evidence to ground the guess in. That's why it echoed the prompt's own
# example ("Application"/"Code Defect") on every single analysis task this
# session regardless of the true fault.
ANALYSIS_VIEW_TOKENS = _env_int("GRAPHRCA_ANALYSIS_VIEW_TOKENS", 3000)

# Completion-token budget for this node's LLM call (analysis classification
# and mitigation command proposal). Verified against GraphRCA/logs/
# llm_justification.jsonl: EVERY guardrail_actuator call this project has
# ever made -- 11/11 analysis, 100% of mitigation -- returned an empty
# response at the old max_tokens=200. gemma4-graphrca is a "thinking" model
# whose reasoning trace burns through a small completion budget before it
# can emit the JSON answer (the same failure mode already fixed once for
# rca_analyst's RCA_MAX_TOKENS). 200 tokens never gave it a chance to
# finish reasoning, so this call has been silently 100% non-functional --
# every analysis/mitigation result on record fell through to
# _set_default_submission, not to the model's own judgement.
GUARDRAIL_MAX_TOKENS = _env_int("GRAPHRCA_GUARDRAIL_MAX_TOKENS", 8192)


class GuardrailActuator:
    """
    SLM + deterministic safety regex.
    Decides the final AIOpsLab submission or mitigation command.
    """
    def __init__(self, scratchpad_client=None):
        self.client = scratchpad_client
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
            # Ground the classification in the same evidence the RCA analyst
            # used, scoped to the root-cause service via the query-aware
            # boost, instead of asking the model to invent a fault_type from
            # a bare service-name string with no telemetry at all.
            evidence = "(no ScratchPad evidence available for this session)"
            session_id = state.get("scratchpad_session_id")
            if self.client and session_id:
                try:
                    evidence = self.client.get_view(
                        session_id, max_tokens=ANALYSIS_VIEW_TOKENS, query=root_cause
                    )
                except Exception as e:
                    logger.warning(f"[Guardrail] failed to fetch evidence view: {e}")
            user_prompt = (
                f"Evidence for {root_cause}:\n{evidence}\n\n"
                "Using ONLY the evidence above, classify this incident.\n"
                "Return JSON with 'system_level' (Hardware|Operating System|Virtualization|Application) "
                "and 'fault_type' (Misconfiguration|Code Defect|Authentication Issue|Network/Storage Issue|Operation Error|Dependency Problem).\n"
                "Respond in this exact shape, replacing each placeholder with "
                "one option from its list above -- do not copy this shape's words verbatim:\n"
                '{"submission": {"system_level": "<system_level choice>", "fault_type": "<fault_type choice>"}}'
            )
        elif task_type == "mitigation":
            # Same evidence-starvation problem as analysis: without the
            # namespace or the actual anomaly evidence, the model can't
            # produce anything but a generic health-check command. Give it
            # both -- `problem_id` carries the k8s namespace here (see
            # swarm_agent_aiopslab.py's `problem_id=self.namespace`).
            namespace = state.get("problem_id", "")
            evidence = "(no ScratchPad evidence available for this session)"
            session_id = state.get("scratchpad_session_id")
            if self.client and session_id:
                try:
                    evidence = self.client.get_view(
                        session_id, max_tokens=ANALYSIS_VIEW_TOKENS, query=root_cause
                    )
                except Exception as e:
                    logger.warning(f"[Guardrail] failed to fetch evidence view: {e}")
            user_prompt = (
                f"Namespace: {namespace}\n"
                f"Evidence for {root_cause}:\n{evidence}\n\n"
                f"Propose ONE kubectl command that corrects the root cause in "
                f"'{root_cause}' (not just a status check) -- e.g. fixing a bad "
                f"image, restoring a deleted resource, scaling a deployment back "
                f"up, or similar, based on what the evidence above actually shows. "
                f"Always target namespace '{namespace}' explicitly with -n.\n"
                'Return JSON: {"command": "kubectl ..."}'
            )
            
        response = llm_reason(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=GUARDRAIL_MAX_TOKENS,
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
