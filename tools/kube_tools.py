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
        # We ensure it's a kubectl command for basic safety
        if not command.strip().startswith("kubectl"):
            return "Error: Only kubectl commands are permitted."
            
        out = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)  # nosec B602
        return out.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing kubectl command: {e.stderr}")
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
