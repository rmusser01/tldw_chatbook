"""One fresh control read shares ancestry, while completion stays independent."""

from collections import Counter
import inspect
import os
import sys

import pytest

from Tests.Backup_Recovery import test_generation_witness_observation as witness_cases
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import generation_witnesses as generations
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import windows_files

witness_case = witness_cases.witness_case

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.skipif(
        os.name != "nt", reason="Actual Windows native parent traversal"
    ),
]


def test_one_fresh_control_read_shares_initial_verified_parent(witness_case, request):
    _root, _control, selector = witness_case
    native = windows_files._native()
    members = (
        (bootstrap, "_control_records"),
        (bootstrap, "_registry"),
        (bootstrap, "_read"),
        (windows_files.WindowsLocks, "flock"),
        (windows_files.WindowsOS, "stat_many_for_admission"),
        (windows_files._Native, "open_handle"),
    )
    bindings = tuple(
        (owner, name, function, function.__code__)
        for owner, name in members
        for function in (inspect.getattr_static(owner, name),)
    )
    codes = {code: name for _owner, name, _function, code in bindings}
    observation = bootstrap._control_observation
    observation_body = observation.__wrapped__
    observation_code = observation_body.__code__
    witness = generations._witnesses
    witness_code = witness.__code__
    snapshot_code = windows_files.WindowsOS.stat_many_for_admission.__code__
    open_code = windows_files._Native.open_handle.__code__
    counts, drive_opens, handles = Counter(), Counter(), []
    before_leases = set(storage._live_leases)

    with storage.acquire_storage(selector) as lease:
        expected = witness_cases.legacy_witnesses(selector, lease)
        previous = sys.getprofile()

        def observe(frame, event, result):
            if previous is not None:
                previous(frame, event, result)
            name = codes.get(frame.f_code)
            if name is None or event not in {"call", "return"}:
                return
            phase = "initial"
            ancestor = frame
            while ancestor is not None:
                if ancestor.f_code is snapshot_code:
                    phase = "completion"
                if ancestor.f_code is observation_code:
                    break
                # The observation is suspended during actual ActivationStore
                # work in the yielded witness body; that work is not counted.
                if ancestor.f_code is witness_code:
                    return
                ancestor = ancestor.f_back
            if ancestor is None:
                return
            if event == "call":
                counts[name] += 1
                if frame.f_code is open_code:
                    assert frame.f_locals["self"] is native
                    if frame.f_locals["parent"] is None:
                        assert (
                            frame.f_locals["name"] == "\\??\\" + selector.drive + "\\"
                        )
                        assert frame.f_locals["directory"] is True
                        drive_opens[phase] += 1
                elif name == "flock":
                    assert frame.f_locals["operation"] == bootstrap.fcntl.LOCK_SH
            elif frame.f_code is open_code:
                if type(result) is int:  # noqa: E721 - exact native HANDLE result.
                    handles.append(result)
                else:
                    assert result is None
                    counts["open_handle_failed"] += 1

        sys.setprofile(observe)
        try:
            actual = generations._witnesses(selector, lease)
        finally:
            sys.setprofile(previous)
        # The source lease remains live through the actual witness read.
        assert lease in storage._live_leases
        assert lease.execution_context(selector)[0] == _root

    assert actual == expected
    assert actual[0]["generation"] == "g"
    assert counts["_control_records"] == 1
    assert counts["_registry"] == 1
    assert counts["_read"] == 3  # Profile, activation association, registry.
    assert counts["flock"] == 1
    assert counts["stat_many_for_admission"] == 1
    assert drive_opens["completion"] == 2  # Independent forward/reverse tree visits.
    assert handles
    assert len(handles) + counts["open_handle_failed"] == counts["open_handle"]
    for handle in handles:
        with pytest.raises(OSError) as error:
            native.info(handle)
        assert error.value.winerror == 6  # Original physical HANDLE is closed.
    assert set(storage._live_leases) == before_leases
    assert bootstrap._control_observation is observation
    assert observation.__wrapped__ is observation_body
    assert observation_body.__code__ is observation_code
    assert generations._witnesses is witness and witness.__code__ is witness_code
    assert all(
        inspect.getattr_static(owner, name) is function and function.__code__ is code
        for owner, name, function, code in bindings
    )
    request.node.user_properties.extend(
        [
            ("original_control_read_calls", dict(counts)),
            ("original_control_drive_opens", dict(drive_opens)),
            ("original_control_native_handles_retired", len(handles)),
            ("fresh_witness_payload_matches_legacy", True),
        ]
    )
    # RED: five independent initial drive traversals, plus the unchanged two
    # completion visits. Only the initial ancestry should be consolidated.
    assert drive_opens["initial"] == 1, dict(drive_opens)
