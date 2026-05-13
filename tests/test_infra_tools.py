"""Tests for Phase 2 — infra_tools.py (build_infra_layer, detect_noisy_neighbors)."""

import networkx as nx
import pytest
from unittest.mock import MagicMock


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _base_graph():
    """Service trace graph with two services."""
    G = nx.DiGraph()
    G.add_node("svc-a", node_type="service", error_rate=0.5)
    G.add_node("svc-b", node_type="service", error_rate=0.0)
    G.add_edge("svc-a", "svc-b", edge_type="CALLS")
    return G


def _topology(n_pods=2, same_node=True):
    """Build a minimal topology dict."""
    node_name = "node-1"
    pods = [
        {
            "name": f"pod-{i}",
            "namespace": "default",
            "node_name": node_name if same_node else f"node-{i}",
            "labels": {"app": f"svc-{'a' if i == 0 else 'b'}"},
            "cpu_request": "100m",
            "memory_request": "128Mi",
            "service_name": f"svc-{'a' if i == 0 else 'b'}",
        }
        for i in range(n_pods)
    ]
    pod_to_node = {p["name"]: p["node_name"] for p in pods}
    node_to_pods: dict = {}
    for p in pods:
        node_to_pods.setdefault(p["node_name"], []).append(p["name"])

    return {
        "pods": pods,
        "nodes": [{"name": node_name, "capacity_cpu": "4", "capacity_memory": "8Gi", "conditions": []}],
        "pod_to_node": pod_to_node,
        "node_to_pods": node_to_pods,
        "service_to_pods": {p["service_name"]: [p["name"]] for p in pods},
    }


def _alert(service, score=0.8):
    a = MagicMock()
    a.service = service
    a.score = score
    return a


# ── build_infra_layer tests ────────────────────────────────────────────────────

def test_build_infra_layer_adds_pod_nodes():
    from GraphRCA_agent.tools.pipeline.infra_tools import build_infra_layer

    G = _base_graph()
    topo = _topology()
    G = build_infra_layer(G, topo, {})

    pod_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "pod"]
    assert len(pod_nodes) == 2


def test_build_infra_layer_adds_k8s_node():
    from GraphRCA_agent.tools.pipeline.infra_tools import build_infra_layer

    G = _base_graph()
    topo = _topology()
    G = build_infra_layer(G, topo, {})

    k8s_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "k8s_node"]
    assert len(k8s_nodes) == 1
    assert "node-1" in G.nodes


def test_build_infra_layer_adds_deployed_as_edges():
    from GraphRCA_agent.tools.pipeline.infra_tools import build_infra_layer

    G = _base_graph()
    topo = _topology()
    G = build_infra_layer(G, topo, {})

    deployed_as_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "DEPLOYED_AS"
    ]
    assert len(deployed_as_edges) == 2


def test_build_infra_layer_adds_runs_on_edges():
    from GraphRCA_agent.tools.pipeline.infra_tools import build_infra_layer

    G = _base_graph()
    topo = _topology()
    G = build_infra_layer(G, topo, {})

    runs_on_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "RUNS_ON"
    ]
    assert len(runs_on_edges) == 2


def test_build_infra_layer_adds_colocated_edges_when_both_alerted():
    from GraphRCA_agent.tools.pipeline.infra_tools import build_infra_layer

    G = _base_graph()
    topo = _topology(same_node=True)
    alerts = [_alert("svc-a"), _alert("svc-b")]
    G = build_infra_layer(G, topo, {}, alerts=alerts)

    colocated_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "COLOCATED"
    ]
    assert len(colocated_edges) >= 2  # bidirectional


def test_build_infra_layer_no_colocated_when_only_one_alerted():
    from GraphRCA_agent.tools.pipeline.infra_tools import build_infra_layer

    G = _base_graph()
    topo = _topology(same_node=True)
    alerts = [_alert("svc-a")]  # only one pod alerted
    G = build_infra_layer(G, topo, {}, alerts=alerts)

    colocated_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "COLOCATED"
    ]
    assert len(colocated_edges) == 0


def test_build_infra_layer_empty_topology_returns_unchanged_graph():
    from GraphRCA_agent.tools.pipeline.infra_tools import build_infra_layer

    G = _base_graph()
    original_nodes = set(G.nodes)
    G2 = build_infra_layer(G, {}, {})
    assert set(G2.nodes) == original_nodes


# ── detect_noisy_neighbors tests ───────────────────────────────────────────────

def test_detect_noisy_neighbors_finds_pair_on_same_node():
    from GraphRCA_agent.tools.pipeline.infra_tools import detect_noisy_neighbors

    G = _base_graph()
    topo = _topology(same_node=True)
    alerts = [_alert("svc-a", score=0.9), _alert("svc-b", score=0.7)]

    results = detect_noisy_neighbors(G, topo, alerts)

    assert len(results) == 1
    r = results[0]
    assert "pod-0" in (r["pod_a"], r["pod_b"])
    assert "pod-1" in (r["pod_a"], r["pod_b"])
    assert r["shared_node"] == "node-1"


def test_detect_noisy_neighbors_no_result_when_different_nodes():
    from GraphRCA_agent.tools.pipeline.infra_tools import detect_noisy_neighbors

    G = _base_graph()
    topo = _topology(same_node=False)
    alerts = [_alert("svc-a"), _alert("svc-b")]

    results = detect_noisy_neighbors(G, topo, alerts)
    assert len(results) == 0


def test_detect_noisy_neighbors_no_result_when_one_alert():
    from GraphRCA_agent.tools.pipeline.infra_tools import detect_noisy_neighbors

    G = _base_graph()
    topo = _topology(same_node=True)
    alerts = [_alert("svc-a")]  # only one alerted

    results = detect_noisy_neighbors(G, topo, alerts)
    assert len(results) == 0


def test_detect_noisy_neighbors_empty_on_no_alerts():
    from GraphRCA_agent.tools.pipeline.infra_tools import detect_noisy_neighbors

    G = _base_graph()
    topo = _topology()
    results = detect_noisy_neighbors(G, topo, [])
    assert results == []


# ── discover_infra_topology tests ──────────────────────────────────────────────

def test_discover_infra_topology_returns_empty_on_kubectl_failure():
    from GraphRCA_agent.tools.pipeline.infra_tools import discover_infra_topology

    def failing_kubectl(cmd):
        raise RuntimeError("kubectl not found")

    result = discover_infra_topology("default", failing_kubectl)
    assert result["pods"] == []
    assert result["nodes"] == []


def test_discover_infra_topology_parses_pod_list():
    from GraphRCA_agent.tools.pipeline.infra_tools import discover_infra_topology
    import json

    pod_list = {
        "kind": "PodList",
        "items": [
            {
                "metadata": {"name": "my-pod", "labels": {"app": "my-svc"}},
                "spec": {
                    "nodeName": "node-1",
                    "containers": [{"resources": {"requests": {"cpu": "200m", "memory": "256Mi"}}}],
                },
            }
        ],
    }
    node_list = {"kind": "NodeList", "items": []}

    call_count = [0]

    def mock_kubectl(cmd):
        call_count[0] += 1
        if "pods" in cmd:
            return json.dumps(pod_list)
        return json.dumps(node_list)

    result = discover_infra_topology("default", mock_kubectl)
    assert len(result["pods"]) == 1
    assert result["pods"][0]["name"] == "my-pod"
    assert result["pods"][0]["service_name"] == "my-svc"
    assert result["pod_to_node"]["my-pod"] == "node-1"
