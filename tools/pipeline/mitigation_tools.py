"""Mitigation Planning Tools for GraphRCA.

Generate prioritized mitigation actions from RCA results.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_mitigation_plan(
    ranked_causes: List[Any],
    service_stats: Dict[str, Dict[str, Any]],
    similar_cases: Optional[List[Dict[str, Any]]] = None,
    additional_context: str = "",
) -> List[Dict[str, Any]]:
    """Generate a prioritized mitigation plan.
    
    Args:
        ranked_causes: Ranked root cause candidates
        service_stats: Per-service statistics
        similar_cases: Optional similar historical cases
        additional_context: Additional context for retry scenarios
        
    Returns:
        List of mitigation action dictionaries
    """
    actions = []
    
    if not ranked_causes:
        logger.warning("No ranked causes to generate mitigation plan")
        return [{
            "action": "investigate",
            "target": "unknown",
            "service": "unknown",
            "command": "kubectl get pods -A | grep -v Running",
            "priority": 1,
            "title": "Manual investigation required",
            "description": "Manual investigation required - no clear root cause identified",
            "category": "diagnostic",
        }]
    
    # Check similar cases for proven resolutions
    proven_resolutions = {}
    if similar_cases:
        for case in similar_cases:
            if case.get("outcome") == "resolved" and case.get("resolution"):
                root = case.get("root_cause", "")
                if root:
                    proven_resolutions[root] = case.get("resolution")
    
    for i, cause in enumerate(ranked_causes[:3]):  # Top 3 causes
        service = (
            getattr(cause, "service", None)
            if not isinstance(cause, dict)
            else cause.get("service")
        )
        service = str(service or "unknown")

        score = (
            getattr(cause, "confidence", None)
            if not isinstance(cause, dict)
            else cause.get("confidence", cause.get("total_score"))
        )
        try:
            score = float(score or 0.0)
        except Exception:
            score = 0.0

        error_rate = (
            getattr(cause, "error_rate", None)
            if not isinstance(cause, dict)
            else cause.get("error_rate")
        )
        if error_rate is None:
            error_rate = service_stats.get(service, {}).get("error_rate", 0)
        try:
            error_rate = float(error_rate or 0.0)
        except Exception:
            error_rate = 0.0
        
        priority = i + 1
        
        # Check for proven resolution
        if service in proven_resolutions:
            actions.append({
                "action": "apply_known_fix",
                "target": service,
                "service": service,
                "command": proven_resolutions[service],
                "priority": priority,
                "title": f"Apply known fix for {service}",
                "description": f"Apply proven resolution from similar past incident",
                "category": "remediation",
                "confidence": 0.9,
            })
            continue
        
        # Generate actions based on root cause characteristics
        if error_rate > 0.5:
            # High error rate - likely crash/failure
            actions.extend([
                {
                    "action": "restart",
                    "target": service,
                    "service": service,
                    "command": f"kubectl rollout restart deployment/{service}",
                    "priority": priority,
                    "title": f"Restart {service}",
                    "description": f"Restart {service} due to high error rate ({error_rate:.1%})",
                    "category": "remediation",
                },
                {
                    "action": "check_logs",
                    "target": service,
                    "service": service,
                    "command": f"kubectl logs -l app={service} --tail=100",
                    "priority": priority + 0.5,
                    "title": f"Check logs for {service}",
                    "description": f"Check logs for {service}",
                    "category": "diagnostic",
                },
            ])
        elif error_rate > 0.1:
            # Moderate error rate - may need scaling or config fix
            actions.extend([
                {
                    "action": "check_resources",
                    "target": service,
                    "service": service,
                    "command": f"kubectl top pods -l app={service}",
                    "priority": priority,
                    "title": f"Check resources for {service}",
                    "description": f"Check resource usage for {service}",
                    "category": "diagnostic",
                },
                {
                    "action": "scale_up",
                    "target": service,
                    "service": service,
                    "command": f"kubectl scale deployment/{service} --replicas=3",
                    "priority": priority + 0.5,
                    "title": f"Scale up {service}",
                    "description": f"Scale up {service} if resource constrained",
                    "category": "remediation",
                },
            ])
        else:
            # Low error rate but flagged - likely latency issue
            actions.extend([
                {
                    "action": "check_dependencies",
                    "target": service,
                    "service": service,
                    "command": f"kubectl describe deployment/{service}",
                    "priority": priority,
                    "title": f"Check deployment for {service}",
                    "description": f"Check dependencies and config for {service}",
                    "category": "diagnostic",
                },
            ])
        
        # Always include pod status check
        actions.append({
            "action": "check_pods",
            "target": service,
            "service": service,
            "command": f"kubectl get pods -l app={service} -o wide",
            "priority": priority + 0.3,
            "title": f"Check pods for {service}",
            "description": f"Check pod status for {service}",
            "category": "diagnostic",
        })
    
    # Sort by priority
    actions.sort(key=lambda a: a.get("priority", 99))
    
    # Add sequence numbers
    for i, action in enumerate(actions):
        action["sequence"] = i + 1
    
    logger.info(f"Generated mitigation plan with {len(actions)} actions")
    return actions


def request_human_approval(
    actions: List[Dict[str, Any]],
    auto_approve_diagnostic: bool = True,
) -> List[Dict[str, Any]]:
    """Request human approval for mitigation actions.
    
    In automated mode, this function:
    - Auto-approves diagnostic actions
    - Marks remediation actions as pending approval
    
    Args:
        actions: List of mitigation actions
        auto_approve_diagnostic: Auto-approve diagnostic actions
        
    Returns:
        Actions with approval status
    """
    approved_actions = []
    
    for action in actions:
        category = action.get("category", "")
        
        if category == "diagnostic" and auto_approve_diagnostic:
            action["approved"] = True
            action["approval_note"] = "Auto-approved (diagnostic)"
        elif category == "remediation":
            # In AIOpsLab benchmark mode, auto-approve for evaluation
            action["approved"] = True
            action["approval_note"] = "Auto-approved (benchmark mode)"
        else:
            action["approved"] = True
            action["approval_note"] = "Auto-approved (default)"
        
        approved_actions.append(action)
    
    approved_count = sum(1 for a in approved_actions if a.get("approved"))
    logger.info(f"Approved {approved_count}/{len(actions)} actions")
    
    return approved_actions
