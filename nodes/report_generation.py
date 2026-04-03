"""Report Generation Node — LangGraph agent node.

Generates ITBench-compatible JSON reports (diagnosis_struct_out.json,
remediation_struct_out.json) from the completed pipeline state using LLMs
for strict schema formatting.
"""

import json
import logging
import os
import re
from typing import Any, Dict

from GraphRCA_agent.state import PipelineState
from GraphRCA_agent.llm import llm_reason

logger = logging.getLogger(__name__)


def generate_structured_json(prompt: str, content: str, caller: str = "report_gen") -> str:
    """Helper method to prompt LLM for a strict JSON extraction."""
    full_prompt = f"{prompt}\n\nContent:\n{content}"
    try:
        response = llm_reason(full_prompt, caller=caller)
        # Extract JSON blob inside markdown
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback raw return
        return response.strip()
    except Exception as e:
        logger.error(f"Failed to generate JSON: {e}")
        return "{}"


def report_generation_node(state: PipelineState) -> Dict[str, Any]:
    """LangGraph node to generate structured JSON reports.

    Reads: ranked_causes, mitigation_actions, trace_dir, incident_id
    Writes: Generates output files to directory.
    """
    logger.info("[Report Generation] Generating strict payload for ITBench.")

    output_dir = state.get("output_dir", "./output")
    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)

    ranked_causes = state.get("ranked_causes", [])
    mitigations = state.get("mitigation_actions", [])

    messages = state.get("messages", [])

    # 1. Diagnosis Report
    if ranked_causes:
        causes_str = "\n".join([str(c) if not hasattr(c, "to_dict") else str(c.to_dict()) for c in ranked_causes])

        diag_prompt = (
            "You are tasked with extracting a structured JSON report of the fault propagation chains.\n"
            "Output MUST be strict JSON matching this schema exactly:\n"
            "{\n"
            '  "faults": [{"entity_name": "string", "entity_type": "string", "root_cause": boolean}]\n'
            "}\n"
            "Do not output markdown, just the JSON string."
        )

        diag_json_str = generate_structured_json(diag_prompt, causes_str, caller="report_gen_diagnosis")
        try:
            diag_out_path = os.path.join(report_dir, "diagnosis_struct_out.json")
            with open(diag_out_path, "w") as f:
                json.dump(json.loads(diag_json_str), f, indent=4)
            logger.info(f"Diagnosis report saved to {diag_out_path}")
            messages.append(f"[Report] Generated diagnosis JSON at {diag_out_path}")
        except json.JSONDecodeError:
            logger.error("Generated diagnosis was not valid JSON.")

    # 2. Remediation Report
    if mitigations:
        mits_str = json.dumps(mitigations, default=str)

        rem_prompt = (
            "You are tasked with extracting a structured JSON report of mitigation plans.\n"
            "Output MUST be strict JSON matching this schema exactly:\n"
            "{\n"
            '  "mitigation": [ [ {"action": "string"} ] ]\n'
            "}\n"
            "Do not output markdown, just the JSON string."
        )

        rem_json_str = generate_structured_json(rem_prompt, mits_str, caller="report_gen_remediation")
        try:
            rem_out_path = os.path.join(report_dir, "remediation_struct_out.json")
            with open(rem_out_path, "w") as f:
                json.dump(json.loads(rem_json_str), f, indent=4)
            logger.info(f"Remediation report saved to {rem_out_path}")
            messages.append(f"[Report] Generated remediation JSON at {rem_out_path}")
        except json.JSONDecodeError:
            logger.error("Generated remediation was not valid JSON.")

    return {"status": "complete", "messages": messages}
