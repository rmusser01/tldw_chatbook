"""Actual native source reads avoid repeated owner-only generation selection."""

import collections
import json
import sys

import pytest

from Tests.Backup_Recovery import test_raw_source_coordinator_io as source_fixtures
from tldw_chatbook.Backup_Recovery import generation_witnesses, raw_participants as raw
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.MCP.recovery_activation import selected_path
from tldw_chatbook.Utils import windows_files

local_root = source_fixtures.local_root
permission = source_fixtures.permission


@pytest.mark.skipif(
    sys.platform != "win32", reason="Counts actual Windows native opens"
)
@pytest.mark.usefixtures("local_root")
def test_seeded_permission_read_avoids_owner_only_generation_queries(
    permission, record_property
):
    permission.set_kill_switch(True)
    permission.set_kill_switch(False)
    assert json.loads(permission.path.read_text())["kill_switch"] is False
    assert raw.mcp_sources.binding(permission)[2]
    codes = {
        windows_files._Native.open_handle.__code__: "native_opens",
        windows_files._Native.security.__code__: "native_security",
        generation_witnesses._witnesses.__code__: "fresh_witnesses",
        selected_path.__code__: "source_selections",
        raw._check.__code__: "full_raw_checks",
    }
    counts = collections.Counter()

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        assert permission.get_kill_switch() is False
    finally:
        sys.setprofile(previous)
    for name, value in counts.items():
        record_property(name, value)
    assert counts["native_opens"] > 0 and counts["native_security"] > 0
    # Outer and two nested entries, then the source-operation and file checks.
    assert counts["full_raw_checks"] >= 5
    assert counts["fresh_witnesses"] >= counts["full_raw_checks"]
    assert counts["source_selections"] <= 13
    assert counts["fresh_witnesses"] <= 14


@pytest.mark.usefixtures("local_root")
def test_same_source_nested_selection_still_refuses_another_path(permission):
    permission.set_kill_switch(True)
    with raw._scope(permission, "mcp_store", writing=True) as operation:
        selected = raw._states[operation].selected
        with pytest.raises(RecoveryRequired, match="raw_source_selection_changed"):
            with raw._scope(
                permission,
                "mcp_store",
                writing=True,
                selected_read=selected.with_name("unselected-policy.json"),
            ):
                pytest.fail("nested source admitted another selected path")
        with raw._scope(
            permission, "mcp_store", writing=True, selected_read=selected
        ) as nested:
            assert nested is operation
            assert permission.get_kill_switch() is True
