"""MCP SSE Client Adapter for GraphRCA Multi-Agent Mode.

Provides lightweight HTTP clients that call the SREGym MCP server
(sregym_mcp_server.py) over its Server-Sent Events (SSE) endpoints.

Architecture:
    MCPToolClient  — thin HTTP wrapper for a single MCP mount point
    call_mcp_tool  — stateless helper for one-shot tool calls
    get_all_mcp_tools — returns a dict of available tool names per mount

The MCP server exposes four mounts:
    /kubectl   — kubectl commands
    /jaeger    — Jaeger distributed traces
    /prometheus — Prometheus metrics (PromQL)
    /loki      — Grafana Loki logs (LogQL)

Usage:
    result = call_mcp_tool("kubectl", "exec_kubectl_cmd_safely",
                           {"cmd": "kubectl get pods -n default"})
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

_MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:9954")
_WORKER_TIMEOUT = int(os.getenv("GRAPHRCA_WORKER_TIMEOUT", "30"))

# Tool names exposed by each MCP mount
_MOUNT_TOOLS: Dict[str, List[str]] = {
    "kubectl": ["exec_kubectl_cmd_safely", "rollback_command", "get_previous_rollbackable_cmd"],
    "jaeger": ["get_traces", "get_trace_by_id", "get_services"],
    "prometheus": ["query_prometheus", "query_prometheus_range", "get_metric_names"],
    "loki": ["query_loki", "query_loki_range", "get_loki_labels"],
}


class MCPToolClient:
    """Lightweight HTTP client for a single SREGym MCP server mount.

    Communicates via HTTP POST to the MCP server's message endpoint.
    The SREGym server uses FastMCP with SSE transport; we drive it via
    direct HTTP JSON-RPC calls which is compatible with the protocol.

    Args:
        mount:      One of "kubectl", "jaeger", "prometheus", "loki"
        server_url: Base URL of the MCP server (default from MCP_SERVER_URL env)
        timeout:    Per-call timeout in seconds
        session_id: Optional session ID (auto-generated if not provided)
    """

    def __init__(
        self,
        mount: str,
        server_url: str = _MCP_SERVER_URL,
        timeout: int = _WORKER_TIMEOUT,
        session_id: Optional[str] = None,
    ):
        self.mount = mount
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session_id = session_id or str(uuid4())
        self._messages_url = f"{self.server_url}/{mount}/messages/"

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Args:
            tool_name:  Name of the MCP tool (e.g., "exec_kubectl_cmd_safely")
            arguments:  Dict of tool arguments

        Returns:
            Tool result as a string (JSON or plain text)

        Raises:
            RuntimeError: If the call fails or times out
        """
        try:
            import httpx
        except ImportError:
            raise RuntimeError(
                "httpx is required for MCP client. Install with: pip install httpx"
            )

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "sregym_ssid": self.session_id,
        }

        t0 = time.time()
        try:
            response = httpx.post(
                self._messages_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            elapsed = round(time.time() - t0, 2)

            data = response.json()
            # JSON-RPC result extraction
            result = data.get("result", {})
            if isinstance(result, dict):
                content = result.get("content", [])
                if content and isinstance(content, list):
                    # FastMCP returns [{type: "text", text: "..."}]
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    output = "\n".join(texts)
                else:
                    output = json.dumps(result)
            else:
                output = str(result)

            logger.info(
                f"[MCPClient] {self.mount}/{tool_name} → {len(output)} chars in {elapsed}s"
            )
            return output

        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            logger.warning(
                f"[MCPClient] {self.mount}/{tool_name} failed in {elapsed}s: {e}"
            )
            raise RuntimeError(f"MCP call {self.mount}/{tool_name} failed: {e}") from e

    def list_tools(self) -> List[str]:
        """Return known tool names for this mount."""
        return list(_MOUNT_TOOLS.get(self.mount, []))


def call_mcp_tool(
    mount: str,
    tool_name: str,
    arguments: Dict[str, Any],
    server_url: str = _MCP_SERVER_URL,
    timeout: int = _WORKER_TIMEOUT,
    session_id: Optional[str] = None,
) -> str:
    """Stateless convenience wrapper for a single MCP tool call.

    Args:
        mount:      MCP server mount ("kubectl", "jaeger", "prometheus", "loki")
        tool_name:  Tool to call
        arguments:  Tool arguments dict
        server_url: MCP server base URL
        timeout:    Timeout in seconds
        session_id: Optional session ID

    Returns:
        Tool result string, or error message on failure
    """
    client = MCPToolClient(
        mount=mount,
        server_url=server_url,
        timeout=timeout,
        session_id=session_id,
    )
    try:
        return client.call(tool_name, arguments)
    except RuntimeError as e:
        return f"[MCP Error] {e}"


def get_all_mcp_tools() -> Dict[str, List[str]]:
    """Return a mapping of mount → tool names for all available MCP mounts."""
    return {mount: list(tools) for mount, tools in _MOUNT_TOOLS.items()}
