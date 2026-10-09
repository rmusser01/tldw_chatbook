"""Registration consumes selected-source metadata, not another native observation."""

from collections import Counter
import sys

import pytest

from Tests.Backup_Recovery import test_raw_related_source_preparation as source_cases
from tldw_chatbook.Backup_Recovery import generation_witnesses
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import recovery_activation

configured_source = source_cases.configured_source
local_root = source_cases.local_root
installed_source_case = source_cases.installed_source_case


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_installed_registration_keeps_only_boundary_source_observations(
    installed_source_case, request
):
    case = installed_source_case
    bindings = tuple(
        (owner, name, function, function.__code__)
        for owner, name in (
            (recovery_activation, "selected_path"),
            (recovery_activation, "readable"),
            (generation_witnesses, "_witnesses"),
        )
        for function in (getattr(owner, name),)
    )
    selected_code, readable_code, witness_code = (
        code for _owner, _name, _function, code in bindings
    )
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
        # Existing original-body census verifies the actual payload, all member
        # paths and order, live native custody, physical FD closure and retirement.
        observed = source_cases._assert_one_original_read_acquisition(case, request)
    finally:
        sys.setprofile(previous)

    assert all(
        getattr(owner, name) is function and function.__code__ is code
        for owner, name, function, code in bindings
    )
    assert not active_selections
    assert selection_reads == [1] * counts["selections"]
    assert counts["readable"] == 1
    assert observed.all_acquisition_count == 2  # Independent canonical and members.
    assert set(storage._live_leases) == before_leases
    checks = Counter()
    for row in observed.check_callers:
        assert row["route"] == "mcp_store"
        checks[row["caller"][1]] += row["calls"]
    assert checks == {"_scope": 3, "_operation": 1, "_file": 1}
    request.node.user_properties.extend(
        [
            ("original_source_observations", dict(counts)),
            ("original_effect_and_scope_checks", dict(checks)),
            ("each_selection_reads_fresh_witnesses", True),
        ]
    )
    # RED on the unchanged source: 11 selections / 12 witnesses. Registration
    # removes three repeated observations; every independent boundary remains.
    assert counts == {"selections": 8, "witnesses": 9, "readable": 1}, dict(counts)
