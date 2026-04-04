"""Trace Ingest Tools for GraphRCA.

Parse CSV trace files, deduplicate spans, validate schema,
and compute per-service statistics.
"""

import csv
import logging
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def parse_csv_directory(trace_dir: str) -> List[Dict[str, Any]]:
    """Parse all CSV files in a directory into a list of span dicts.
    
    Args:
        trace_dir: Path to directory containing trace CSV files
        
    Returns:
        List of span dictionaries with normalized field names
    """
    spans = []
    
    if not os.path.isdir(trace_dir):
        logger.error(f"Trace directory not found: {trace_dir}")
        return spans
    
    csv_files = [f for f in os.listdir(trace_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {trace_dir}")
        return spans
    
    def _looks_like_aiopslab_header(header_line: str) -> bool:
        """Heuristic for AIOpsLab-provided trace export headers.

        Note: AIOpsLab may provide either a standard comma-delimited CSV or a
        pseudo-CSV where data rows are mixed comma + fixed-width fields.
        We must not decide solely based on the header.
        """
        h = (header_line or "").strip().lower()
        return (
            "trace_id" in h
            and "span_id" in h
            and "parent_span" in h
            and "service_name" in h
            and "operation_name" in h
            and "start_time" in h
            and "duration" in h
            and "has_error" in h
            and "response" in h
        )

    def _choose_aiopslab_parsing_mode(file_obj) -> str:
        """Return 'standard' or 'pseudo' based on sampling the first data rows."""
        try:
            header_line = file_obj.readline()
            if not header_line:
                return "standard"

            if not _looks_like_aiopslab_header(header_line):
                return "standard"

            header_cols = next(csv.reader([header_line.strip()]))
            header_len = len(header_cols)

            # Sample a few non-empty, non-header lines to see if they match the header width.
            sample_pos = file_obj.tell()
            for _ in range(25):
                line = file_obj.readline()
                if not line:
                    break
                s = line.strip()
                if not s or s.lower().startswith("trace_id"):
                    continue

                try:
                    row = next(csv.reader([s]))
                except Exception:
                    continue

                # If the row matches the header length, it's a normal CSV.
                if header_len and len(row) == header_len:
                    file_obj.seek(0)
                    return "standard"

                # The pseudo-CSV row format used by our parser has exactly 6 comma fields.
                if len(row) == 6 and header_len >= 8:
                    file_obj.seek(0)
                    return "pseudo"

            # Default: standard (safer; DictReader will ignore malformed rows rather than corrupt fields).
            file_obj.seek(0)
            return "standard"
        finally:
            try:
                file_obj.seek(0)
            except Exception:
                pass

    def _parse_aiopslab_pseudocsv_row(line: str) -> Dict[str, str] | None:
        s = (line or "").strip()
        if not s or s.lower().startswith("trace_id"):
            return None

        parts = s.split(",", maxsplit=5)
        if len(parts) < 6:
            return None

        trace_id = parts[0].strip()
        span_parent_service = parts[1].strip()
        op_start = parts[2].strip()
        duration = parts[3].strip()
        has_error = parts[4].strip()
        response = parts[5].strip()

        tokens = span_parent_service.split()
        span_id = tokens[0] if len(tokens) >= 1 else ""
        parent_span = tokens[1] if len(tokens) >= 2 else ""
        service_name = " ".join(tokens[2:]) if len(tokens) >= 3 else ""

        m = re.match(r"^(.*)\s+(\d+)$", op_start)
        operation_name = m.group(1).strip() if m else op_start
        start_time = m.group(2).strip() if m else ""

        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span": parent_span,
            "service_name": service_name,
            "operation_name": operation_name,
            "start_time": start_time,
            "duration": duration,
            "has_error": has_error,
            "response": response,
        }

    for csv_file in csv_files:
        filepath = os.path.join(trace_dir, csv_file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                mode = _choose_aiopslab_parsing_mode(f)
                if mode == "pseudo":
                    # AIOpsLab pseudo-CSV (mixed comma + fixed-width)
                    header = f.readline()  # consume header
                    for line in f:
                        row = _parse_aiopslab_pseudocsv_row(line)
                        if not row:
                            continue
                        span = _normalize_span(row)
                        if span:
                            spans.append(span)
                else:
                    # Standard comma-delimited CSV
                    reader = csv.DictReader(f)
                    for row in reader:
                        span = _normalize_span(row)
                        if span:
                            spans.append(span)
        except Exception as e:
            logger.error(f"Error parsing {csv_file}: {e}")
    
    logger.info(f"Parsed {len(spans)} spans from {len(csv_files)} CSV files")
    return spans


def _normalize_span(row: Dict[str, str]) -> Dict[str, Any] | None:
    """Normalize a CSV row into a standard span dict."""
    try:
        def _safe_int(val: Any) -> int:
            try:
                # Avoid parsing floats like "123.0" as 1230.
                return int(str(val).strip())
            except (ValueError, TypeError):
                return 0

        # ── Extract raw fields (accept multiple column conventions) ──
        trace_id = (row.get("trace_id") or row.get("traceId") or row.get("TraceId") or "").strip()
        span_id = (row.get("span_id") or row.get("spanId") or row.get("SpanId") or "").strip()
        parent_span = (
            row.get("parent_span")
            or row.get("parentSpan")
            or row.get("parent_id")
            or row.get("parentId")
            or row.get("ParentId")
            or ""
        ).strip()

        service_name = (
            row.get("service_name")
            or row.get("serviceName")
            or row.get("service")
            or row.get("Service")
            or "unknown"
        ).strip() or "unknown"

        operation_name = (
            row.get("operation_name")
            or row.get("operationName")
            or row.get("operation")
            or row.get("Operation")
            or ""
        ).strip()

        # start_time is expected to be microseconds epoch in Stratus traces
        start_time = _safe_int(row.get("start_time") or row.get("startTime") or row.get("StartTime") or "0")

        # duration: prefer duration_ms if present; else treat 'duration' as microseconds
        duration_ms = 0.0
        if row.get("duration_ms") is not None:
            duration_ms = _safe_float(row.get("duration_ms", "0"))
        elif row.get("duration") is not None:
            duration_us = _safe_int(row.get("duration", "0"))
            duration_ms = float(duration_us) / 1000.0
        elif row.get("Duration") is not None:
            duration_us = _safe_int(row.get("Duration", "0"))
            duration_ms = float(duration_us) / 1000.0

        # has_error / response normalization (Stratus-compatible)
        raw_has_error = row.get("has_error")
        if raw_has_error is None:
            raw_has_error = row.get("hasError")
        has_error = str(raw_has_error).strip().lower() in ("true", "1", "yes", "y") if raw_has_error is not None else False

        response = row.get("response")
        if response is None:
            response = row.get("response_class")
        if response is None:
            response = row.get("http.status_code")
        response = str(response).strip() if response is not None else "Unknown"
        if not response or response.lower() in ("none", "nan"):
            response = "Unknown"

        # status is kept for backwards-compatibility, but downstream should prefer has_error/response.
        status_val = (row.get("status") or row.get("Status") or "").strip()
        if not status_val:
            if has_error:
                status_val = "ERROR"
            elif response.lower() == "unknown":
                status_val = "UNKNOWN"
            else:
                status_val = response

        # Also keep duration_us as integer for convenience (best-effort)
        duration_us = int(round(duration_ms * 1000.0))

        # Handle various column naming conventions
        span = {
            # Canonical Stratus-like fields
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span": parent_span or "ROOT",
            "service_name": service_name,
            "operation_name": operation_name,
            "start_time": start_time,
            "duration": duration_us,
            "duration_ms": duration_ms,
            "has_error": has_error,
            "response": response,
            # Backwards-compatible aliases used by earlier GraphRCA code
            "parent_id": parent_span,
            "service": service_name,
            "operation": operation_name,
            "status": status_val or "OK",
            "end_time": (row.get("end_time") or row.get("endTime") or row.get("EndTime") or "").strip(),
        }
        
        # Parse tags if present
        tags_str = row.get("tags") or row.get("Tags", "{}")
        if isinstance(tags_str, str) and tags_str.startswith("{"):
            try:
                import json
                span["tags"] = json.loads(tags_str)
            except:
                span["tags"] = {}
        else:
            span["tags"] = {}
        
        return span
    except Exception as e:
        logger.debug(f"Failed to normalize span: {e}")
        return None


def _safe_float(value: str) -> float:
    """Safely convert string to float."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def deduplicate_spans(spans: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Remove duplicate spans based on span_id.
    
    Args:
        spans: List of span dictionaries
        
    Returns:
        Tuple of (deduplicated spans, dedup statistics)
    """
    seen = set()
    unique_spans = []
    duplicates = 0
    
    for span in spans:
        trace_id = (span.get("trace_id") or "").strip()
        span_id = (span.get("span_id") or "").strip()

        if span_id:
            key = (trace_id, span_id) if trace_id else span_id
        else:
            # Fallback for malformed/partial inputs: keep spans instead of
            # dropping everything when span_id is missing.
            key = (
                trace_id,
                (span.get("parent_id") or "").strip(),
                (span.get("service") or "").strip(),
                (span.get("operation") or "").strip(),
                (span.get("start_time") or "").strip(),
                float(span.get("duration_ms") or 0.0),
                (span.get("status") or "").strip(),
            )

        if key not in seen:
            seen.add(key)
            unique_spans.append(span)
        else:
            duplicates += 1
    
    stats = {
        "original_count": len(spans),
        "unique_count": len(unique_spans),
        "duplicates_removed": duplicates,
    }
    
    logger.info(f"Deduplication: {stats['original_count']} -> {stats['unique_count']} spans")
    return unique_spans, stats


def validate_schema(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that spans have required fields.
    
    Args:
        spans: List of span dictionaries
        
    Returns:
        Dict with validation results including validation_rate
    """
    required_fields = ["trace_id", "span_id", "service"]
    errors = []
    valid_count = 0
    
    for i, span in enumerate(spans):
        missing = [f for f in required_fields if not span.get(f)]
        if missing:
            errors.append(f"Span {i}: missing fields {missing}")
        else:
            valid_count += 1
    
    is_valid = len(errors) == 0
    validation_rate = (valid_count / len(spans) * 100) if spans else 100.0
    
    if not is_valid:
        logger.warning(f"Schema validation found {len(errors)} errors")
    
    return {
        "is_valid": is_valid,
        "errors": errors[:10],  # Return first 10 errors
        "validation_rate": validation_rate,
        "valid_count": valid_count,
        "total_count": len(spans),
    }


def detect_orphan_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find spans whose parent_id doesn't exist in the trace.
    
    Args:
        spans: List of span dictionaries
        
    Returns:
        List of orphan spans
    """
    span_ids = {s.get("span_id") for s in spans if s.get("span_id")}
    orphans = []
    
    for span in spans:
        parent_id = (span.get("parent_span") or span.get("parent_id") or "").strip()
        # Root spans (no parent) are not orphans
        if parent_id and parent_id != "ROOT" and parent_id not in span_ids:
            orphans.append(span)
    
    if orphans:
        logger.info(f"Found {len(orphans)} orphan spans")
    
    return orphans


def compute_stats(spans: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute per-service statistics from spans.
    
    Args:
        spans: List of span dictionaries
        
    Returns:
        Dict mapping service name to statistics
    """
    service_spans: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for span in spans:
        service = (span.get("service_name") or span.get("service") or "unknown").strip() or "unknown"
        service_spans[service].append(span)

    stats: Dict[str, Dict[str, Any]] = {}
    for service, svc_spans in service_spans.items():
        durations = [float(s.get("duration_ms") or 0.0) for s in svc_spans]

        # Error / unknown accounting
        error_count = 0
        unknown_count = 0
        for s in svc_spans:
            if bool(s.get("has_error", False)):
                error_count += 1
            else:
                # Treat HTTP 5xx as error even if has_error missing
                resp = str(s.get("response") or "").strip()
                if resp.isdigit() and int(resp) >= 500:
                    error_count += 1

            resp = str(s.get("response") or "Unknown").strip()
            if not resp or resp.lower() == "unknown":
                unknown_count += 1

        span_count = len(svc_spans)
        error_rate = round(error_count / span_count, 4) if span_count else 0.0
        unknown_pct = round(unknown_count / span_count * 100.0, 1) if span_count else 0.0

        durations_sorted = sorted(durations) if durations else []
        mean = sum(durations_sorted) / len(durations_sorted) if durations_sorted else 0.0
        var = (
            sum((d - mean) ** 2 for d in durations_sorted) / len(durations_sorted)
            if durations_sorted
            else 0.0
        )
        std = math.sqrt(var) if var > 0 else 0.0

        operations = sorted({str(s.get("operation_name") or s.get("operation") or "").strip() for s in svc_spans if (s.get("operation_name") or s.get("operation"))})

        stats[service] = {
            "span_count": span_count,
            "error_count": error_count,
            "error_rate": error_rate,
            "unknown_response_count": unknown_count,
            "unknown_response_pct": unknown_pct,
            "duration_min_ms": round(durations_sorted[0], 2) if durations_sorted else 0.0,
            "duration_max_ms": round(durations_sorted[-1], 2) if durations_sorted else 0.0,
            "duration_mean_ms": round(mean, 2),
            "duration_std_ms": round(std, 2),
            "duration_p50_ms": round(_percentile(durations_sorted, 50), 2),
            "duration_p95_ms": round(_percentile(durations_sorted, 95), 2),
            "duration_p99_ms": round(_percentile(durations_sorted, 99), 2),
            "operations": operations,
        }

    logger.info(f"Computed stats for {len(stats)} services")
    return stats


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = values if values == sorted(values) else sorted(values)
    # Use nearest-rank method (simple and stable)
    k = int(math.ceil((pct / 100.0) * len(sorted_values))) - 1
    k = max(0, min(k, len(sorted_values) - 1))
    return float(sorted_values[k])


class OverallStats:
    """Overall trace statistics container."""
    
    def __init__(
        self,
        total_spans: int = 0,
        unique_spans: int = 0,
        unique_services: int = 0,
        unique_traces: int = 0,
        error_spans: int = 0,
        unknown_response_pct: float = 0.0,
        services: List[str] = None,
    ):
        self.total_spans = total_spans
        self.unique_spans = unique_spans
        self.unique_services = unique_services
        self.unique_traces = unique_traces
        self.error_spans = error_spans
        self.unknown_response_pct = unknown_response_pct
        self.services = services or []


def compute_overall_stats(spans: List[Dict[str, Any]], service_stats: Dict[str, Dict] = None) -> OverallStats:
    """Compute overall trace statistics.
    
    Args:
        spans: List of span dictionaries
        service_stats: Per-service statistics (optional)
        
    Returns:
        OverallStats object with computed statistics
    """
    trace_ids = {s.get("trace_id") for s in spans if s.get("trace_id")}
    span_ids = {s.get("span_id") for s in spans if s.get("span_id")}
    services = {s.get("service_name") or s.get("service") for s in spans if (s.get("service_name") or s.get("service"))}

    error_spans = [s for s in spans if bool(s.get("has_error", False)) or str(s.get("response") or "").strip().isdigit() and int(str(s.get("response")).strip()) >= 500]
    unknown_response = [s for s in spans if str(s.get("response") or "Unknown").strip().lower() == "unknown"]
    
    return OverallStats(
        total_spans=len(spans),
        unique_spans=len(span_ids),
        unique_services=len(services),
        unique_traces=len(trace_ids),
        error_spans=len(error_spans),
        unknown_response_pct=len(unknown_response) / len(spans) * 100 if spans else 0.0,
        services=list(services),
    )
