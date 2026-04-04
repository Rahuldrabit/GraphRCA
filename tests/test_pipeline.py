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
    from GraphRCA_agent.graph import build_graph
    workflow = build_graph()
    app = workflow.compile()
    assert app is not None


def test_state_schema_importable():
    from GraphRCA_agent.state import PipelineState
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

    with patch("GraphRCA_agent.llm.llm_reason", return_value=mock_llm_response):
        from GraphRCA_agent.run_pipeline import run_pipeline
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

    with patch("GraphRCA_agent.llm.llm_reason", return_value="CLUSTER: test\nCOUNT: 1\nSEVERITY: LOW\nSERVICES: svc\nSUMMARY: test\n---"):
        from GraphRCA_agent.run_pipeline import run_pipeline
        report = run_pipeline(
            trace_dir=trace_dir,
            output_dir=output_dir,
            use_neo4j=False,
        )

    # health_score_before should be a non-negative float
    assert report["summary"]["health_score_before"] >= 0.0


def test_ingest_parses_standard_aiopslab_csv_multi_service():
    """Regression: do not mis-detect normal AIOpsLab CSV as pseudo-CSV."""
    trace_dir = create_trace_dir()

    from GraphRCA_agent.tools.pipeline.ingest_tools import parse_csv_directory, compute_stats

    spans = parse_csv_directory(trace_dir)
    assert len(spans) > 0

    stats = compute_stats(spans)
    # Our synthetic CSV has multiple distinct services.
    assert len(stats) >= 4
    assert "frontend" in stats
    assert "api-gateway" in stats
    assert "postgres" in stats


def test_ingest_parses_aiopslab_pseudocsv_rows():
    """Ensure pseudo-CSV path still works for mixed-format rows."""
    import tempfile

    header = "trace_id,span_id,parent_span,service_name,operation_name,start_time,duration,has_error,response\n"
    # Pseudo-CSV data row format expected by _parse_aiopslab_pseudocsv_row:
    # trace_id, "<span_id> <parent_span> <service_name>", "<operation_name> <start_time>", duration, has_error, response
    row1 = "t1,s1 ROOT nginx-web-server,/wrk2-api/post/compose 1000000,10460,True,500\n"
    row2 = "t1,s2 s1 compose-post-service,compose_post_server 1001000,5786,False,Unknown\n"

    tmpdir = tempfile.mkdtemp()
    fname = os.path.join(tmpdir, "pseudo.csv")
    with open(fname, "w") as f:
        f.write(header)
        f.write(row1)
        f.write(row2)

    from GraphRCA_agent.tools.pipeline.ingest_tools import parse_csv_directory, compute_stats

    spans = parse_csv_directory(tmpdir)
    assert len(spans) == 2

    stats = compute_stats(spans)
    assert "nginx-web-server" in stats
    assert "compose-post-service" in stats


def test_detection_does_not_alert_on_unknown_only_signal():
    """Regression: do not flag services only because response is 'Unknown'.

    AIOpsLab gRPC spans often show response=Unknown even when healthy.
    """
    import tempfile

    header = "trace_id,span_id,parent_span,service_name,operation_name,start_time,duration,has_error,response\n"
    # 2 services, stable durations, no errors, but response is Unknown.
    rows = [
        "t1,s1,ROOT,svc-a,/svc.A/Method,1000000,1000,False,Unknown\n",
        "t1,s2,s1,svc-b,/svc.B/Method,1000100,1000,False,Unknown\n",
        "t2,s3,ROOT,svc-a,/svc.A/Method,2000000,1000,False,Unknown\n",
        "t2,s4,s3,svc-b,/svc.B/Method,2000100,1000,False,Unknown\n",
    ]

    tmpdir = tempfile.mkdtemp()
    fname = os.path.join(tmpdir, "unknown_only.csv")
    with open(fname, "w") as f:
        f.write(header)
        for r in rows:
            f.write(r)

    from GraphRCA_agent.tools.pipeline.ingest_tools import parse_csv_directory, compute_stats
    from GraphRCA_agent.tools.pipeline.detection_tools import compute_ewma_baseline, detect_all_anomalies

    spans = parse_csv_directory(tmpdir)
    assert len(spans) == 4

    stats = compute_stats(spans)
    baselines = compute_ewma_baseline(spans, alpha=0.3, window_size=100)

    alerts = detect_all_anomalies(
        spans=spans,
        service_stats=stats,
        baselines=baselines,
        z_threshold=3.0,
        error_threshold=0.05,
    )

    assert len(alerts) == 0
