"""Final query-only timings on the actual retained completed production caches."""
import asyncio
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
import time

import psutil
import pytest


def summary(values):
    return {
        "samples_ms": values,
        "p50_ms": statistics.median(values),
        "p95_ms": sorted(values)[math.ceil(.95 * len(values)) - 1],
    }


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_final_fix_idle_latency(monkeypatch):
    from types import SimpleNamespace
    from Tests.UI.app_factory import _build_test_app
    from Tests.Agents.test_project_index_tool import provider, invoke
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.participants import _core_cached_connection
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB
    from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner
    from tldw_chatbook.Workspaces.project_index.runtime import (
        get_project_index_service, project_index_key,
        close_project_index_service, project_index_registry_connections,
    )

    base = Path(__file__).resolve().parents[2] / "k5-scale"
    app = _build_test_app(user_data_dir=config.get_user_data_dir())
    service = get_project_index_service(app)
    db = app.workspace_registry_service.db
    original_candidates = WorkspaceProjectIndexDB.lexical_candidates
    original_capture = ProjectScanner.capture_file
    public = {"active": False, "lexical": [], "capture": [], "retirement": []}

    def empty_registry():
        return _core_cached_connection(db, getattr(db._thread_local, "conn", None)) is None

    def measured_candidates(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_candidates(*args, **kwargs)
        finally:
            if public["active"]:
                public["lexical"].append((time.perf_counter() - start) * 1000)

    def measured_capture(scanner, *args, **kwargs):
        start = time.perf_counter()
        result = original_capture(scanner, *args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        proof = {
            "pid": scanner.worker_pid,
            "gone": not psutil.pid_exists(scanner.worker_pid),
            "retired": scanner.retired,
            "pipes_closed": scanner.pipes_closed,
            "returncode": scanner.process_returncode,
        }
        assert proof["gone"] and proof["retired"] and proof["pipes_closed"]
        assert proof["returncode"] is not None
        reports.setdefault("all_capture_retirement", []).append(proof)
        if public["active"]:
            public["capture"].append(elapsed)
            public["retirement"].append(proof)
        return result

    monkeypatch.setattr(WorkspaceProjectIndexDB, "lexical_candidates", measured_candidates)
    monkeypatch.setattr(ProjectScanner, "capture_file", measured_capture)
    reports = {
        "load": "No competing K5 tests/builds/renderers. Light read-only coordination and uncontrolled OS background may overlap.",
        "authority_fixture": "Test-seeded LocalToolProvider roots/guard; actual provider-issued search authority, runtime registry/admission/cache/native Capture. No fabricated search-authority DTO.",
        "candidate_clock": "Admitted sample includes current registry fence acquisition, guard/root/chain/scope/state checks, floor merge, canonical lexical_candidates and fence exit. Store open/read pin and final native/cache/registry close are outside this cached-candidate clock. Canonical decorated lexical p95<=500ms is the original gate. Expanded admitted500ms is an explicitly accepted unmet deviation; both raw metrics stay reported.",
        "public_clock": "Instrumented whole public invoke includes production registry ownership, guards, store/pin admission, lexical lookup, native Capture, final validation and owner close. No test outer ownership scope surrounds it. Native Capture time includes scanner worker/pipes retirement. Whole-tool time also includes probe psutil retirement assertions/bookkeeping after Capture.",
    }

    def measure(shape):
        assert empty_registry()
        root = base / shape
        with project_index_registry_connections(db):
            binding = app.workspace_registry_service.list_folder_bindings("qualification-" + shape)[0]
        assert empty_registry()
        key = project_index_key(app, "qualification-" + shape, binding.binding_id, root)
        tool = provider(SimpleNamespace(service=service, root=root, key=key))
        assert empty_registry()

        issued = []
        search = service.search
        def capture_authority(authority, *args, **kwargs):
            issued.append(authority)
            return search(authority, *args, **kwargs)
        with monkeypatch.context() as capture_patch:
            capture_patch.setattr(service, "search", capture_authority)
            warmup = invoke(tool, {"query": "qualificationdeepmarker"}, call_id="authority-" + shape)
        assert warmup.ok and len(json.loads(warmup.content)["hits"]) == 1
        assert len(issued) == 1 and empty_registry() and not service._searches and not service._operations
        authority = issued[0]
        workloads = {}
        with service.registry_connection_scope(), service._search_store(key) as store:
            for label, query, expected in (
                ("common", "generated file", 100),
                ("rare", "qualificationdeepmarker", 1),
                ("miss", "absentqualificationzzzz", 0),
            ):
                boundary, lexical, counts = [], [], []
                for number in range(55):
                    began = time.perf_counter()
                    assert authority.guard()
                    with service._fence(key) as live:
                        assert live is not None and live.root == authority.root and live.chain == authority.chain
                        status = store.load_status(key)
                        assert status.scope == authority.scope and status.state in {"Ready", "Partial"}
                        admitted = replace(authority, exclusions=tuple(set(authority.exclusions + live.exclusions)), authority_revision=live.authority_revision)
                        inner = time.perf_counter()
                        candidates = original_candidates(store, admitted, query, limit=100)
                        inner_ms = (time.perf_counter() - inner) * 1000
                    elapsed_ms = (time.perf_counter() - began) * 1000
                    assert status.state == "Ready" and status.indexed == 20000
                    assert len(candidates) == expected
                    if number >= 5:
                        boundary.append(elapsed_ms)
                        lexical.append(inner_ms)
                        counts.append(len(candidates))
                workloads[label] = {"query": query, "warmups": 5, "samples": 50, "limit": 100, "counts": counts, "admitted": summary(boundary), "inner_lexical": summary(lexical), "guard_and_scope_checks": 55}
        assert empty_registry() and not service._searches

        public["lexical"].clear()
        public["capture"].clear()
        public["retirement"].clear()
        totals, entry_empty, exit_empty = [], [], []
        public["active"] = True
        try:
            for number in range(20):
                entry_empty.append(empty_registry())
                assert entry_empty[-1]
                began = time.perf_counter()
                result = invoke(tool, {"query": "qualificationdeepmarker"}, call_id=f"idle-{shape}-{number}")
                totals.append((time.perf_counter() - began) * 1000)
                payload = json.loads(result.content)
                assert result.ok and len(payload["hits"]) == 1
                assert "qualificationdeepmarker" in payload["hits"][0]["excerpt"]
                exit_empty.append(empty_registry())
                assert exit_empty[-1] and not service._searches and not service._operations
        finally:
            public["active"] = False
        assert len(public["lexical"]) == len(public["capture"]) == len(public["retirement"]) == 20
        return {
            "key": asdict(key), "scope": asdict(authority.scope),
            "requesting_floor": [asdict(item) for item in authority.exclusions],
            "code_issued_authority_warmup": True,
            "fixture_lookup_and_candidate_registry_retired_before_public": True,
            "candidate_workloads": workloads,
            "public_marker": {"samples":20,"lexical":summary(list(public["lexical"])),"live_native_capture":summary(list(public["capture"])),"whole_tool":summary(totals),"native_retirement":list(public["retirement"]),"registry_empty_at_entry":entry_empty,"registry_retired_at_return":exit_empty},
            "searches_remaining": len(service._searches),
            "operations_remaining": len(service._operations),
        }

    try:
        for shape in ("nested", "wide"):
            reports[shape] = await asyncio.to_thread(measure, shape)
    finally:
        await close_project_index_service(app)
        reports["service_closed"] = service._closed
        reports["searches_remaining_after_close"] = len(service._searches)
        reports["operations_remaining_after_close"] = len(service._operations)
        assert not service._operations
        Path(__file__).with_name("integration-idle-latency.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    assert all(reports[shape]["candidate_workloads"][label]["inner_lexical"]["p95_ms"] <= 500 for shape in ("nested", "wide") for label in ("common", "rare", "miss"))
