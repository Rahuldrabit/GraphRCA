"""Infrastructure Topology Discovery and Multi-Layer Graph Construction.

Adds a second layer to the service dependency graph:
  Layer 1: Trace dependencies — Service --CALLS--> Service (existing)
  Layer 2: Infrastructure    — Service --DEPLOYED_AS--> Pod --RUNS_ON--> K8sNode
                               Pod --COLOCATED--> Pod (shares same node)

Usage:
    topology = discover_infra_topology(namespace, kubectl_fn)
    G = build_infra_layer(G, topology, service_stats, alerts)
    noisy = detect_noisy_neighbors(G, topology, alerts)
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Label keys used to match pods to services (tried in order)
_SERVICE_LABEL_KEYS = [
    "app",
    "app.kubernetes.io/name",
    "io.kompose.service",
    "service",
    "component",
]


def _run_kubectl(kubectl_fn: Callable[[str], str], cmd: str) -> Optional[Dict]:
    """Run a kubectl command and return parsed JSON, or None on failure."""
    try:
        raw = kubectl_fn(cmd)
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
        return None
    except Exception as e:
        logger.warning(f"[InfraTools] kubectl '{cmd}' failed: {e}")
        return None


def _extract_service_name(labels: Dict[str, str]) -> Optional[str]:
    """Try standard label keys to map a pod to its service name."""
    for key in _SERVICE_LABEL_KEYS:
        val = labels.get(key)
        if val:
            return str(val).strip()
    return None


def discover_infra_topology(
    namespace: str,
    kubectl_fn: Callable[[str], str],
) -> Dict[str, Any]:
    """Discover pod/node topology via kubectl.

    Args:
        namespace:   Kubernetes namespace to query
        kubectl_fn:  Callable that takes a kubectl command string and returns output

    Returns:
        {
          "pods":          [{"name", "namespace", "node_name", "labels",
                             "cpu_request", "memory_request", "service_name"}],
          "nodes":         [{"name", "capacity_cpu", "capacity_memory", "conditions"}],
          "pod_to_node":   {pod_name: node_name},
          "node_to_pods":  {node_name: [pod_name, ...]},
          "service_to_pods": {service_name: [pod_name, ...]},
        }

    Graceful degradation: returns empty dict if kubectl is unavailable.
    """
    ns = namespace or "default"
    topology: Dict[str, Any] = {
        "pods": [],
        "nodes": [],
        "pod_to_node": {},
        "node_to_pods": {},
        "service_to_pods": {},
    }

    # ── Pods ──────────────────────────────────────────────────────────────────
    pod_data = _run_kubectl(kubectl_fn, f"get pods -n {ns} -o json")
    if not pod_data or pod_data.get("kind") not in ("PodList", "List"):
        logger.warning(f"[InfraTools] No pod data for namespace '{ns}'")
        return topology

    for item in pod_data.get("items", []):
        meta = item.get("metadata", {})
        spec = item.get("spec", {})
        labels = meta.get("labels", {}) or {}

        pod_name = meta.get("name", "")
        node_name = spec.get("nodeName", "")
        if not pod_name:
            continue

        # Extract resource requests from first container
        cpu_req = ""
        mem_req = ""
        containers = spec.get("containers", [])
        if containers:
            resources = containers[0].get("resources", {})
            requests = resources.get("requests", {})
            cpu_req = requests.get("cpu", "")
            mem_req = requests.get("memory", "")

        svc_name = _extract_service_name(labels)

        pod_entry = {
            "name": pod_name,
            "namespace": ns,
            "node_name": node_name,
            "labels": labels,
            "cpu_request": cpu_req,
            "memory_request": mem_req,
            "service_name": svc_name,
        }
        topology["pods"].append(pod_entry)
        topology["pod_to_node"][pod_name] = node_name

        if node_name:
            topology["node_to_pods"].setdefault(node_name, []).append(pod_name)
        if svc_name:
            topology["service_to_pods"].setdefault(svc_name, []).append(pod_name)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    node_data = _run_kubectl(kubectl_fn, "get nodes -o json")
    if node_data and node_data.get("kind") in ("NodeList", "List"):
        for item in node_data.get("items", []):
            meta = item.get("metadata", {})
            status = item.get("status", {})
            capacity = status.get("capacity", {})
            conditions = [
                {"type": c["type"], "status": c["status"]}
                for c in status.get("conditions", [])
                if c.get("status") != "Unknown"
            ]
            topology["nodes"].append({
                "name": meta.get("name", ""),
                "capacity_cpu": capacity.get("cpu", ""),
                "capacity_memory": capacity.get("memory", ""),
                "conditions": conditions,
            })

    logger.info(
        f"[InfraTools] Discovered {len(topology['pods'])} pods across "
        f"{len(topology['nodes'])} nodes in namespace '{ns}'"
    )
    return topology


def build_infra_layer(
    G: Any,
    topology: Dict[str, Any],
    service_stats: Dict[str, Any],
    alerts: Optional[List[Any]] = None,
) -> Any:
    """Add Pod and K8sNode nodes + infra edges to an existing nx.DiGraph.

    New node types added:
        node_type="pod"      — Kubernetes pod
        node_type="k8s_node" — Kubernetes cluster node

    New edge types added:
        DEPLOYED_AS  Service → Pod
        RUNS_ON      Pod → K8sNode
        COLOCATED    Pod ↔ Pod  (same node, both pods show anomalous metrics)

    Args:
        G:              Existing NetworkX DiGraph (trace layer)
        topology:       Output of discover_infra_topology()
        service_stats:  Per-service stats dict from pipeline state
        alerts:         Current alert list (used to gate COLOCATED edges)

    Returns:
        The same DiGraph with infra layer added (modified in place, also returned)
    """
    if not topology or not topology.get("pods"):
        return G

    alerted_services = set()
    if alerts:
        for a in alerts:
            svc = getattr(a, "service", None) or (a.get("service") if isinstance(a, dict) else None)
            if svc:
                alerted_services.add(svc)

    # Track which pods we've added (pod_name → node_name)
    pod_to_node: Dict[str, str] = topology.get("pod_to_node", {})

    # ── Add Pod nodes ────────────────────────────────────────────────────────
    for pod in topology.get("pods", []):
        pod_name = pod["name"]
        G.add_node(
            pod_name,
            node_type="pod",
            namespace=pod.get("namespace", ""),
            node_name=pod.get("node_name", ""),
            cpu_request=pod.get("cpu_request", ""),
            memory_request=pod.get("memory_request", ""),
            service_name=pod.get("service_name", ""),
        )

        # DEPLOYED_AS: Service → Pod (if the service exists in the trace graph)
        svc_name = pod.get("service_name")
        if svc_name and svc_name in G.nodes and G.nodes[svc_name].get("node_type") != "pod":
            G.add_edge(svc_name, pod_name, edge_type="DEPLOYED_AS")

    # ── Add K8sNode nodes ────────────────────────────────────────────────────
    for node in topology.get("nodes", []):
        node_name = node["name"]
        if not node_name:
            continue
        G.add_node(
            node_name,
            node_type="k8s_node",
            capacity_cpu=node.get("capacity_cpu", ""),
            capacity_memory=node.get("capacity_memory", ""),
            conditions=node.get("conditions", []),
        )

    # ── Add RUNS_ON edges: Pod → K8sNode ─────────────────────────────────────
    for pod in topology.get("pods", []):
        pod_name = pod["name"]
        node_name = pod.get("node_name", "")
        if pod_name in G.nodes and node_name and node_name in G.nodes:
            G.add_edge(pod_name, node_name, edge_type="RUNS_ON")

    # ── Add COLOCATED edges: Pod ↔ Pod (conservative — only for anomalous pods) ─
    node_to_pods = topology.get("node_to_pods", {})
    for k8s_node, pod_list in node_to_pods.items():
        if len(pod_list) < 2:
            continue

        # Find pods whose owning service is alerted
        anomalous_pods = []
        for pod_name in pod_list:
            # Find service for this pod
            pod_info = next((p for p in topology.get("pods", []) if p["name"] == pod_name), None)
            if pod_info:
                svc = pod_info.get("service_name")
                if svc and svc in alerted_services:
                    anomalous_pods.append(pod_name)

        # Only add COLOCATED edges if at least 2 pods on this node are anomalous
        # (conservative: avoids false positive noisy-neighbor signals)
        if len(anomalous_pods) >= 2:
            for i, pa in enumerate(anomalous_pods):
                for pb in anomalous_pods[i + 1:]:
                    if pa in G.nodes and pb in G.nodes:
                        G.add_edge(pa, pb, edge_type="COLOCATED", shared_node=k8s_node)
                        G.add_edge(pb, pa, edge_type="COLOCATED", shared_node=k8s_node)

    # Count new nodes/edges
    infra_nodes = sum(
        1 for n in G.nodes
        if G.nodes[n].get("node_type") in ("pod", "k8s_node")
    )
    infra_edges = sum(
        1 for u, v, d in G.edges(data=True)
        if d.get("edge_type") in ("DEPLOYED_AS", "RUNS_ON", "COLOCATED")
    )
    logger.info(
        f"[InfraTools] Added {infra_nodes} infra nodes, {infra_edges} infra edges"
    )
    return G


def detect_noisy_neighbors(
    G: Any,
    topology: Dict[str, Any],
    alerts: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Identify pods sharing a node where at least one has an active alert.

    Returns a list of noisy-neighbor candidates:
        [{"pod_a", "pod_b", "shared_node", "alerted_service", "evidence"}]
    """
    results: List[Dict[str, Any]] = []
    if not topology or not alerts:
        return results

    alerted_services = {}
    for a in alerts:
        svc = getattr(a, "service", None) or (a.get("service") if isinstance(a, dict) else None)
        score = getattr(a, "score", 0.0) or (a.get("score", 0.0) if isinstance(a, dict) else 0.0)
        if svc:
            alerted_services[svc] = float(score)

    node_to_pods = topology.get("node_to_pods", {})
    pod_list_all = topology.get("pods", [])

    for k8s_node, pod_names in node_to_pods.items():
        if len(pod_names) < 2:
            continue

        alerted_on_node = []
        for pod_name in pod_names:
            pod_info = next((p for p in pod_list_all if p["name"] == pod_name), None)
            if pod_info:
                svc = pod_info.get("service_name")
                if svc and svc in alerted_services:
                    alerted_on_node.append((pod_name, svc, alerted_services[svc]))

        if len(alerted_on_node) < 2:
            continue

        for i, (pa, svc_a, score_a) in enumerate(alerted_on_node):
            for pb, svc_b, score_b in alerted_on_node[i + 1:]:
                results.append({
                    "pod_a": pa,
                    "pod_b": pb,
                    "service_a": svc_a,
                    "service_b": svc_b,
                    "shared_node": k8s_node,
                    "alerted_service": svc_a if score_a >= score_b else svc_b,
                    "evidence": (
                        f"Pods '{pa}' (svc={svc_a}, score={score_a:.3f}) and "
                        f"'{pb}' (svc={svc_b}, score={score_b:.3f}) "
                        f"share node '{k8s_node}'"
                    ),
                })

    results.sort(key=lambda r: r.get("service_a", ""))
    logger.info(f"[InfraTools] Detected {len(results)} noisy-neighbor pairs")
    return results
