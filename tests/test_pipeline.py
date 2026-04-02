"""Integration smoke test — verifies the full GraphRCA graph compiles and
  runs end-to-end against synthetic trace data (no Neo4j, no real LLM).
"""

import os
import sys
import json
import pytest
import tempfile
from unittest.mock import patch


# ── Synthetic trace CSV generator ─────────────────────────────────────────────

TRACE_CSV_CONTENT = """\
trace_id,span_id,parent_span,service_name,operation_name,start_time,duration,has_error,response
t1,s1,ROOT,frontend,HTTP GET /,1000000,50000,False,200
t1,s2,s1,api-gateway,GET /api/users,1010000,45000,False,200
t1,s3,s2,user-service,db_query,1020000,40000,True,500
t1,s4,s3,postgres,SELECT users,1025000,39000,True,500
t2,s5,ROOT,frontend,HTTP GET /products,2000000,60000,False,200
t2,s6,s5,api-gateway,GET /api/products,2010000,55000,True,500
t2,s7,s6,product-service,db_query,2020000,50000,True,500
t2,s8,s7,postgres,SELECT products,2025000,48000,True,500
t3,s9,ROOT,frontend,HTTP POST /order,3000000,70000,False,200
t3,s10,s9,api-gateway,POST /api/orders,3010000,65000,False,200
t3,s11,s10,order-service,process_order,3020000,60000,False,200
t3,s12,s11,postgres,INSERT orders,3025000,55000,False,200
"""


def create_trace_dir():
    """Create a temp dir with a CSV trace file."""
    tmpdir = tempfile.mkdtemp()
    fname = os.path.join(tmpdir, "traces_test.csv")
    with open(fname, "w") as f:
        f.write(TRACE_CSV_CONTENT)
    return tmpdir


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_graph_compiles():
    """GraphRCA StateGraph compiles without error."""
    from GraphRCA.graph import build_graph
    workflow = build_graph()
    app = workflow.compile()
    assert app is not None


def test_state_schema_importable():
    from GraphRCA.state import PipelineState
    state: PipelineState = {
        "trace_dir": "./traces",
        "use_neo4j": False,
        "status": "running",
        "messages": [],
        "rollback_count": 0,
    }
    assert state["status"] == "running"


def test_pipeline_end_to_end_no_llm():
    """Run full pipeline with mocked LLM — validates data flow."""
    trace_dir = create_trace_dir()
    output_dir = tempfile.mkdtemp()

    # Mock LLM to avoid real API calls
    mock_llm_response = (
        "CLUSTER: DB connection errors\nCOUNT: 4\nSEVERITY: HIGH\n"
        "SERVICES: postgres, user-service\nSUMMARY: Postgres returning 500 errors\n---"
    )

    with patch("GraphRCA.llm.llm_reason", return_value=mock_llm_response):
        from GraphRCA.run_pipeline import run_pipeline
        report = run_pipeline(
            trace_dir=trace_dir,
            output_dir=output_dir,
            use_neo4j=False,
            store_spans=False,
            verbose=False,
        )

    assert report is not None
    assert "incident_id" in report
    assert report["summary"]["alerts_detected"] >= 0
    assert "root_cause_service" in report["summary"]
    assert report["summary"]["root_cause_service"] != ""

    # Verify report file was written
    report_path = os.path.join(output_dir, "incident_report.json")
    assert os.path.exists(report_path)
    with open(report_path) as f:
        saved = json.load(f)
    assert "rca" in saved
    assert "mitigation" in saved


def test_health_score_computed():
    """Detection node computes pre-mitigation health score."""
    trace_dir = create_trace_dir()
    output_dir = tempfile.mkdtemp()

    with patch("GraphRCA.llm.llm_reason", return_value="CLUSTER: test\nCOUNT: 1\nSEVERITY: LOW\nSERVICES: svc\nSUMMARY: test\n---"):
        from GraphRCA.run_pipeline import run_pipeline
        report = run_pipeline(
            trace_dir=trace_dir,
            output_dir=output_dir,
            use_neo4j=False,
        )

    # health_score_before should be a non-negative float
    assert report["summary"]["health_score_before"] >= 0.0
