"""Native acquisition counts within existing ordinary generation checks."""

import json
import sys
from collections import Counter

import pytest

from Tests.Backup_Recovery.test_windows_metadata_prerequisites import (
    native_scope as native_scope,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import activation, bootstrap, generation_witnesses
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import register_pending
from tldw_chatbook.Utils import windows_files

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows generation evidence"
)


@pytest.fixture
def warm_generation(native_scope):  # noqa: F811
    _, selector, _, _ = native_scope
    # The bound selector and the exact regular-file path must both be confirmed.
    for _ in range(3):
        with storage.acquire_storage(selector):
            pass
    with storage.acquire_storage(selector) as lease:
        assert lease._warm_execution is not None
        yield selector, lease


@pytest.mark.parametrize("checkpoint", ["activation", "witness"])
def test_each_generation_checkpoint_reads_one_fresh_input_set(
    native_scope,  # noqa: F811
    checkpoint,
    request,
):
    root, selector, _, _ = native_scope
    lease = None
    if checkpoint == "witness":
        _, lease = request.getfixturevalue("warm_generation")
    read_codes = {
        bootstrap._parse_control_records.__code__: "control",
        bootstrap._parse_registry.__code__: "registry",
        windows_files._Native.open_handle.__code__: "opens",
        windows_files._Native.security.__code__: "security",
    }
    counts = Counter()
    previous = sys.getprofile()
    leases = set(storage._live_leases)

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "call" and frame.f_code in read_codes:
            counts[read_codes[frame.f_code]] += 1

    sys.setprofile(observe)
    try:
        if checkpoint == "activation":
            observation = activation._ordinary_activation_observation(root, selector)
            assert observation is not None
        else:
            assert generation_witnesses._witnesses(selector, lease) == []
    finally:
        sys.setprofile(previous)

    request.node.user_properties.append(
        ("generation_checkpoint_native_counts", json.dumps(counts))
    )
    assert set(storage._live_leases) == leases
    assert counts["opens"] and counts["security"]
    assert (counts["control"], counts["registry"]) == (1, 1)


@pytest.mark.parametrize("damage", ["association", "profile"])
@pytest.mark.parametrize("registry_pending", [False, True])
def test_activation_only_damage_respects_startup_registry_order(
    native_scope,  # noqa: F811
    damage,
    registry_pending,
):
    root, selector, _, _ = native_scope
    if damage == "association":
        record = root / f"activation-{bootstrap._key(str(selector))}.json"
        record.write_bytes(b'{"version":1,"version":1}')
        full_error = "duplicate_key"
    else:
        record = root / f"profile-{bootstrap._key(str(selector))}.json"
        profile = json.loads(record.read_bytes())
        profile["activation"] = {}
        record.write_text(json.dumps(profile), encoding="utf-8")
        full_error = "invalid_activation_witness"
    if registry_pending:
        (root / "admission" / "registry.pending.json").write_bytes(b"{}")
        assert bootstrap.startup_permission(selector, root) == (
            False,
            "recovery_scope_uncertain",
        )
        with pytest.raises(
            bootstrap.RecoveryRequired, match="recovery_scope_uncertain"
        ):
            activation._ordinary_activation_observation(root, selector)
    else:
        assert bootstrap.startup_permission(selector, root) == (True, "startup_allowed")
        with pytest.raises(ValueError, match=full_error):
            activation._ordinary_activation_observation(root, selector)


def test_pending_selector_retains_original_startup_reason(native_scope):  # noqa: F811
    root, selector, _, _ = native_scope
    control = root.parent / "restore-control"
    windows_files.WindowsOS().mkdir(control, 0o700)
    register_pending(root, "task11", ("profile",), control, (selector,))
    with pytest.raises(bootstrap.RecoveryRequired, match="recovery_pending"):
        activation._ordinary_activation_observation(root, selector)


def test_substituted_startup_reader_retains_original_refusal(
    native_scope,  # noqa: F811
    monkeypatch,
):
    root, selector, _, _ = native_scope
    calls = []

    def deny(selected, control):
        calls.append((selected, control))
        return False, "custom_startup_refusal"

    monkeypatch.setattr(bootstrap, "startup_permission", deny)
    with pytest.raises(bootstrap.RecoveryRequired, match="custom_startup_refusal"):
        activation._ordinary_activation_observation(root, selector)
    assert calls == [(selector, root)]


def test_custom_low_record_reader_retains_original_control_reads(
    native_scope,  # noqa: F811
    monkeypatch,
):
    root, selector, _, _ = native_scope
    reader = bootstrap._read_record_bytes
    names = []

    def observe(parent, name, *, max_bytes):
        names.append(name)
        return reader(parent, name, max_bytes=max_bytes)

    monkeypatch.setattr(bootstrap, "_read_record_bytes", observe)
    assert activation._ordinary_activation_observation(root, selector) is not None
    assert names.count("registry.json") == 2


def test_late_registry_reader_substitution_refuses_before_call(
    native_scope,  # noqa: F811
    monkeypatch,
):
    root, selector, _, _ = native_scope
    reads = []
    previous = sys.getprofile()

    def substitute(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "return" and frame.f_code is bootstrap._parse_control_records.__code__:
            reads.append(frame.f_locals["ordinary_only"])

            def unexpected(_root):
                pytest.fail(
                    "selected observation called the substituted registry reader"
                )

            monkeypatch.setattr(bootstrap, "_registry", unexpected)

    sys.setprofile(substitute)
    try:
        with pytest.raises(
            bootstrap.RecoveryRequired, match="ordinary_activation_sources_changed"
        ):
            activation._ordinary_activation_observation(root, selector)
    finally:
        sys.setprofile(previous)
    assert reads == [True]


def test_late_instance_shadow_of_lease_context_refuses_before_use(
    warm_generation, monkeypatch
):
    path, lease = warm_generation
    witness = generation_witnesses._witnesses.__code__
    control = bootstrap._control_records.__code__
    changed = []
    leases = set(storage._live_leases)
    previous = sys.getprofile()

    def substitute(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "return" or frame.f_code is not control or changed:
            return
        parent = frame.f_back
        if parent.f_code is not witness:
            return
        assert parent.f_locals["lease"] is lease
        changed.append(True)

        def unexpected(_path):
            pytest.fail("selected witness called the substituted lease context")

        monkeypatch.setattr(lease, "execution_context", unexpected)

    sys.setprofile(substitute)
    try:
        with pytest.raises(
            bootstrap.RecoveryRequired, match="projection_generation_unavailable"
        ):
            generation_witnesses._witnesses(path, lease)
    finally:
        sys.setprofile(previous)
    assert changed == [True] and set(storage._live_leases) == leases


def test_selected_control_parent_close_failure_is_not_retried(
    native_scope,  # noqa: F811
    monkeypatch,
):
    root, selector, _, _ = native_scope
    os = bootstrap.os
    close = os.close
    scans, attempts = [], []
    previous = sys.getprofile()

    def fail_close(fd):
        if fd == scans[0]:
            attempts.append(fd)
            raise OSError("injected close failure before retirement")
        close(fd)

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if (
            event == "call"
            and frame.f_code is bootstrap._native_close.__code__
            and not scans
        ):
            parent = frame.f_back
            if (
                parent.f_code is bootstrap.pinned_directory.__wrapped__.__code__
                and parent.f_locals["root"] == root
            ):
                scans.append(frame.f_locals["fd"])
                monkeypatch.setattr(os, "close", fail_close)

    sys.setprofile(observe)
    try:
        with pytest.raises(
            bootstrap.RecoveryRequired, match="recovery_scope_uncertain"
        ):
            activation._ordinary_activation_observation(root, selector)
        assert len(scans) == 1 and attempts == scans
        assert os.fstat(scans[0]).st_ino == os.stat(root).st_ino
    finally:
        sys.setprofile(previous)
        # This injected failure ran before close, so the test owns the birth.
        for fd in set(attempts):
            close(fd)
