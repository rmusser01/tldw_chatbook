"""Accepted TASK34408 ancestor intents remain inputs to warm admission."""

import os
import sys

import pytest

from Tests.Backup_Recovery.test_admission_evidence_reuse import _verdict
from Tests.Backup_Recovery.test_admission_evidence_reuse import (
    reuse_switch as _reuse_switch_fixture,
)
from Tests.Backup_Recovery.test_bootstrap import local_scope as _local_scope_fixture
from tldw_chatbook.Backup_Recovery import control_records
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import bind_profile

# Reuse the exact existing fixture objects; no ownership/setup replacement.
local_scope = _local_scope_fixture
reuse_switch = _reuse_switch_fixture

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="current admission reuse is POSIX-only"
)


@pytest.mark.parametrize("scope", ["bound", "unbound", "drifted"])
def test_warm_reuse_refuses_real_unsettled_ancestor_intent(
    local_scope,
    reuse_switch,
    scope,
    record_property,
):
    root, config, data, _ = local_scope
    if scope != "unbound":
        bind_profile(root, config, ("profile",), root / "admission")
    if scope == "drifted":
        config.write_text("drifted")
    target = data / "store.db"
    marker = root.parent / control_records._DIRECTORY_CREATION_INTENT
    assert marker.parent == data.parent == config.parent
    assert not marker.exists()
    reuse_switch(True)
    startup = storage.acquire_storage()
    hold = storage._holds[(os.getpid(), str(root))]
    original_count = hold.count
    scope_code = storage._scope.__code__
    previous = sys.getprofile()
    derivations = []

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is scope_code:
            derivations.append(True)
        if previous is not None:
            previous(frame, event, arg)

    try:
        for _ in range(3):
            assert _verdict(target)[0] == "allowed"
        assert hold.evidence[str(config)].confirmed
        sys.setprofile(observe)
        try:
            assert _verdict(target)[0] == "allowed"
        finally:
            sys.setprofile(previous)
        assert not derivations, "positive warm route must actually skip derivation"
        assert hold.count == original_count
        # This is the accepted fixed marker, placed only in our private fixture.
        # Ancestor permission/identity is unchanged; malformed content still refuses.
        marker.write_bytes(b"malformed controlled creation intent")
        try:
            reused = _verdict(target)
            reuse_switch(False)
            derived = _verdict(target)
            assert hold.count == original_count
        finally:
            marker.unlink()
        assert _verdict(target)[0] == "allowed"
        record_property(
            "current_marker_verdicts", {"reused": reused, "derived": derived}
        )
    finally:
        sys.setprofile(previous)
        marker.unlink(missing_ok=True)
        startup.close()
    assert hold.count == 0
    assert derived == ("refused", "bootstrap_creation_unsettled")
    assert reused == derived


def test_ancestor_intent_appearing_after_reuse_count_retires_counted_lease(
    local_scope,
    reuse_switch,
):
    root, config, data, _ = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    marker = root.parent / control_records._DIRECTORY_CREATION_INTENT
    assert not marker.exists()
    reuse_switch(True)
    startup = storage.acquire_storage()
    hold = storage._holds[(os.getpid(), str(root))]
    original_count = hold.count
    created = []
    previous = sys.getprofile()
    init = storage.StorageLease.__init__.__code__
    reuse = storage._reuse_evidence.__code__

    def observe(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is init
            and frame.f_back.f_code is reuse
            and not created
        ):
            lease = frame.f_locals["self"]
            assert lease._key == hold.key and lease in storage._live_leases
            assert hold.count == original_count + 1
            marker.write_bytes(b"late controlled creation intent")
            created.append(lease)
        if previous is not None:
            previous(frame, event, arg)

    try:
        for _ in range(3):
            assert _verdict(data / "store.db")[0] == "allowed"
        assert hold.evidence[str(config)].confirmed
        sys.setprofile(observe)
        try:
            outcome = _verdict(data / "store.db")
        finally:
            sys.setprofile(previous)
            marker.unlink(missing_ok=True)
        assert len(created) == 1
        assert created[0] not in storage._live_leases
        assert hold.count == original_count
        assert _verdict(data / "store.db")[0] == "allowed"
    finally:
        sys.setprofile(previous)
        marker.unlink(missing_ok=True)
        startup.close()
    assert hold.count == 0
    assert outcome == ("refused", "bootstrap_creation_unsettled")
