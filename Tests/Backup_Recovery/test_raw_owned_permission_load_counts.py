"""Original checked permission loads keep effect checks within one source owner."""

from collections import Counter
import sys
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_related_source_preparation as source_cases
from tldw_chatbook.Backup_Recovery import generation_witnesses
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import console_snapshot, recovery_activation
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)

configured_source = source_cases.configured_source
local_root = source_cases.local_root
installed_source_case = source_cases.installed_source_case


@pytest.fixture
def checked_permission_case(installed_source_case):
    case = installed_source_case
    assert case.kind == "permission"
    catalog = LocalMCPStore(case.path.with_name("local_mcp_store.json"))
    local = LocalMCPControlService(store=catalog, manifest_provider=lambda: {})
    service = UnifiedMCPControlPlaneService(
        target_store=None,
        context_store=None,
        local_service=local,
        server_service=None,
    )
    service._permission_store = case.source
    return SimpleNamespace(
        **vars(case),
        service=service,
        captured=console_snapshot._CapturedSources(service),
    )


def read_checked_payload(case):
    """Exercise the old real read on RED, then the named stock consumer on GREEN."""
    reader = getattr(case.captured, "read_permission_payload", None)
    if reader is not None:
        return reader()
    return case.captured.permission_call(
        lambda owner: case.captured.permission_reader()
    )


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_owned_checked_permission_load_keeps_effect_checks_without_nested_scans(
    checked_permission_case, request
):
    case = checked_permission_case
    expected = case.source.load()
    assert expected["kill_switch"] is False
    assert expected["schema_version"] == 1
    assert expected["profiles"]["default"]["global_default"] == "ask"
    assert expected["profiles"]["default"]["servers"] == {}
    payloads = []

    def read():
        payload = read_checked_payload(case)
        assert payload == expected  # Full original payload precedes bool adaptation.
        payloads.append(payload)
        return payload["kill_switch"]

    measured = SimpleNamespace(**{**vars(case), "read": read})
    bindings = tuple(
        (owner, name, function, function.__code__)
        for owner, name in (
            (recovery_activation, "selected_path"),
            (recovery_activation, "readable"),
            (generation_witnesses, "_witnesses"),
        )
        for function in (getattr(owner, name),)
    )
    selected_code, readable_code, witness_code = (row[3] for row in bindings)
    counts = Counter()
    active_selections, selection_reads = [], []
    before_leases = set(storage._live_leases)
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if frame.f_code is selected_code and frame.f_locals["canonical"] == case.path:
            if event == "call":
                counts["selections"] += 1
                active_selections.append([id(frame), 0])
            elif event == "return":
                frame_id, reads = active_selections.pop()
                assert frame_id == id(frame)
                selection_reads.append(reads)
        elif (
            frame.f_code is witness_code
            and event == "call"
            and frame.f_locals["path"] == case.path
        ):
            counts["witnesses"] += 1
            if active_selections:
                active_selections[-1][1] += 1
        elif (
            frame.f_code is readable_code
            and event == "call"
            and frame.f_locals["store"] is case.source
        ):
            counts["readable"] += 1

    sys.setprofile(observe)
    try:
        observed = source_cases._assert_one_original_read_acquisition(measured, request)
    finally:
        sys.setprofile(previous)

    assert all(
        getattr(owner, name) is function and function.__code__ is code
        for owner, name, function, code in bindings
    )
    assert len(payloads) == 1
    assert not active_selections
    assert selection_reads == [1] * counts["selections"]
    assert counts["readable"] == 1
    assert observed.all_acquisition_count == 2
    assert set(storage._live_leases) == before_leases
    checks = Counter()
    for row in observed.check_callers:
        assert row["route"] == "mcp_store"
        checks[row["caller"][1]] += row["calls"]
    assert checks["_operation"] == checks["_file"] == checks["_checked_read"] == 1
    request.node.user_properties.extend(
        [
            ("original_source_observations", dict(counts)),
            ("original_effect_and_scope_checks", dict(checks)),
            ("each_selection_reads_fresh_witnesses", True),
        ]
    )
    # Original checked load: three scope checks, nine selections, ten witnesses.
    # The reader/file/final checks and all custody proofs above remain required.
    assert checks == {"_scope": 1, "_operation": 1, "_file": 1, "_checked_read": 1}
    assert counts == {"selections": 7, "witnesses": 8, "readable": 1}
