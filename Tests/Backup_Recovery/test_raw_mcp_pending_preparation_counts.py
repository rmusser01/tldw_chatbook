"""One installed MCP preparation retains admission, never generation witnesses."""

import inspect
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
def test_installed_permission_preparation_retains_one_canonical_admission(
    installed_source_case, request
):
    case = installed_source_case
    selected = recovery_activation.selected_path
    witness = generation_witnesses._witnesses
    acquire = storage.acquire_storage
    close = storage.StorageLease.close
    bindings = tuple(
        (function, function.__code__)
        for function in (selected, witness, acquire, close)
    )
    selected_code, witness_code, acquire_code, close_code = (
        code for _function, code in bindings
    )
    active_selections, selection_reads, issued, closed = [], [], [], []
    counts = {"selections": 0, "witnesses": 0}
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
        elif frame.f_code is witness_code and frame.f_locals["path"] == case.path:
            if event == "call":
                counts["witnesses"] += 1
                if active_selections:
                    active_selections[-1][1] += 1
        elif frame.f_code is acquire_code and event == "return":
            assert isinstance(result, storage.StorageLease)
            assert result in storage._live_leases
            issued.append(result)
        elif frame.f_code is close_code and event == "return":
            lease = frame.f_locals["self"]
            if any(lease is original for original in issued):
                assert lease not in storage._live_leases
                closed.append(lease)

    sys.setprofile(observe)
    try:
        # Original payload, exact member order, live source custody, actual FD
        # closure and final state/lease retirement are checked by this helper.
        observed = source_cases._assert_one_original_read_acquisition(case, request)
    finally:
        sys.setprofile(previous)

    assert recovery_activation.selected_path is selected
    assert generation_witnesses._witnesses is witness
    assert storage.acquire_storage is acquire
    assert inspect.getattr_static(storage.StorageLease, "close") is close
    assert all(function.__code__ is code for function, code in bindings)
    assert not active_selections
    # Task25 consumes the initial selection for pure registration, removing
    # three metadata-only repeats. Every retained selection remains fresh.
    assert counts == {"selections": 8, "witnesses": 9}
    assert selection_reads == [1] * 8
    assert len(issued) == observed.all_acquisition_count
    assert len({id(lease) for lease in issued}) == len(issued)
    assert len(closed) == len(issued)
    assert all(sum(retired is lease for retired in closed) == 1 for lease in issued)
    assert all(lease not in storage._live_leases for lease in issued)
    assert set(storage._live_leases) == before_leases
    request.node.user_properties.extend(
        [
            ("original_source_selections", counts["selections"]),
            ("original_fresh_witnesses", counts["witnesses"]),
            ("issued_lease_identities", ",".join(str(id(lease)) for lease in issued)),
            ("retired_lease_identities", ",".join(str(id(lease)) for lease in closed)),
            (
                "canonical_observation_acquisitions",
                len(issued) - len(observed.returned),
            ),
        ]
    )
    # RED: six original canonical acquisitions plus one independent member
    # acquisition. GREEN must retain one canonical lease without losing checks.
    assert observed.all_acquisition_count == 2
