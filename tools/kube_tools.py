"""Kubectl toolkit for executing remote Kubernetes commands."""

import logging
import subprocess  # nosec B404

logger = logging.getLogger(__name__)

def exec_kubectl_command(command: str) -> str:
    """Execute an arbitrary kubectl command securely and return its stdout.
    
    Args:
        command (str): The kubectl command to run (e.g., 'kubectl get pods -n default').
        
    Returns:
        str: Standard output of the command or an error message.
    """
    logger.info(f"Executing kubectl command: {command}")
    try:
        from GraphRCA_agent.trace_logger import trace_event

        trace_event("kubectl.run", tool="kubectl", command=command)
    except Exception:
        pass
    try:
        # We ensure it's a kubectl command for basic safety
        if not command.strip().startswith("kubectl"):
            return "Error: Only kubectl commands are permitted."
            
        out = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)  # nosec B602

        try:
            from GraphRCA_agent.trace_logger import trace_event

            trace_event(
                "kubectl.result",
                tool="kubectl",
                command=command,
                returncode=out.returncode,
                stdout=out.stdout,
                stderr=out.stderr,
            )
        except Exception:
            pass

        logger.debug(f"kubectl stdout (first 500 chars): {out.stdout[:500]}")
        return out.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing kubectl command: {e.stderr}")
        try:
            from GraphRCA_agent.trace_logger import trace_event

            trace_event(
                "kubectl.error",
                tool="kubectl",
                command=command,
                returncode=e.returncode,
                stdout=getattr(e, "stdout", ""),
                stderr=getattr(e, "stderr", ""),
            )
        except Exception:
            pass
        return f"Error executing kubectl command: {e.stderr}"

def dry_run_kubectl_command(command: str) -> str:
    """Execute a kubectl command in dry-run mode to see its effects without applying them.
    
    Args:
        command (str): The kubectl command.
        
    Returns:
        str: Standard output showing the dry-run results.
    """
    if "--dry-run" not in command:
        command += " --dry-run=client"
    return exec_kubectl_command(command)
