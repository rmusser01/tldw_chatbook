"""ACL fixture cleanup must restore and verify the captured native state."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def _fixture(
    monkeypatch,
    *,
    changed=None,
    restore_error=0,
    control=0x9004,
    initialized=None,
    initialize_error=0,
    read_failure_at=None,
):
    from Tests.Backup_Recovery import windows_acl_fixture as helper

    original = ("owner-sid", control, 1, b"original-dacl")
    state = {"security": original, "released": 0, "restored": [], "reads": 0}

    @contextmanager
    def security(native, handle):
        try:
            state["reads"] += 1
            if state["reads"] == read_failure_at:
                raise RuntimeError("fixture native read failed")
            yield "original-pointer", state["security"]
        finally:
            state["released"] += 1

    def restore(handle, kind, flags, owner, group, dacl, sacl):
        state["restored"].append((handle, kind, flags, owner, group, dacl, sacl))
        initial = len(state["restored"]) == 1
        error = initialize_error if initial else restore_error
        if not error:
            state["security"] = (
                initialized or original
                if initial
                else changed or initialized or original
            )
        return error

    monkeypatch.setattr(helper, "_security", security)
    native = SimpleNamespace(advapi=SimpleNamespace(SetSecurityInfo=restore))
    return helper, native, original, state


def test_original_dacl_restored_after_body_failure(monkeypatch):
    helper, native, original, state = _fixture(monkeypatch)
    error = RuntimeError("fixture body failed")
    with pytest.raises(RuntimeError) as caught, helper._preserve_dacl(native, 7):
        state["security"] = ("owner-sid", 0x9004, 1, b"public-dacl")
        raise error
    assert caught.value is error
    assert state["security"] == original
    assert state["released"] == 3
    assert (
        state["restored"]
        == [(7, 1, 4 | 0x80000000, None, None, "original-pointer", None)] * 2
    )


@pytest.mark.parametrize(
    "changed",
    [
        ("other-owner", 0x9004, 1, b"original-dacl"),
        ("owner-sid", 0x9404, 1, b"original-dacl"),
        ("owner-sid", 0x9004, 2, b"original-dacl"),
        ("owner-sid", 0x9004, 1, b"different-dacl"),
    ],
)
def test_native_restoration_mismatch_is_not_normalized(monkeypatch, changed):
    helper, native, _, state = _fixture(monkeypatch, changed=changed)
    with (
        pytest.raises(AssertionError, match="original native security not restored"),
        helper._preserve_dacl(native, 7),
    ):
        pass
    assert state["released"] == 3


def test_native_restore_error_fails_and_releases_original(monkeypatch):
    helper, native, _, state = _fixture(monkeypatch, restore_error=5)
    with pytest.raises(OSError) as caught, helper._preserve_dacl(native, 7):
        pass
    assert caught.value.errno == 5
    assert state["released"] == 2


@pytest.mark.parametrize("restored_acl", [b"sensitive-restored-acl", None])
def test_restoration_mismatch_reports_only_bounded_metadata(monkeypatch, restored_acl):
    changed = ("sensitive-restored-owner", 0x9404, 2, restored_acl)
    helper, native, original, state = _fixture(monkeypatch, changed=changed)
    with pytest.raises(AssertionError) as caught, helper._preserve_dacl(native, 7):
        pass
    expected = (
        "original native security not restored: owner_equal=False "
        "original_control=36868 restored_control=37892 control_xor=1024 "
        "original_revision=1 restored_revision=2 "
        f"original_acl_length=13 restored_acl_length={len(restored_acl or b'')} "
        "acl_equal=False"
    )
    assert str(caught.value) == expected
    assert original[0] not in expected
    assert changed[0] not in expected
    assert "original-dacl" not in expected
    assert "sensitive-restored-acl" not in expected
    assert state["released"] == 3


def test_unprotected_original_is_not_changed_to_protected(monkeypatch):
    helper, native, original, state = _fixture(monkeypatch, control=0x8004)
    with helper._preserve_dacl(native, 7):
        pass
    assert state["restored"][0][2] == 4 | 0x20000000
    assert state["security"] == original


def test_restore_failure_retains_the_original_body_error(monkeypatch):
    helper, native, _, state = _fixture(monkeypatch, restore_error=5)
    original = RuntimeError("fixture body failed")
    with pytest.raises(OSError) as caught, helper._preserve_dacl(native, 7):
        raise original
    assert caught.value.__context__ is original
    assert state["released"] == 2


def test_windows_inheritance_bookkeeping_is_established_before_body(monkeypatch):
    initialized = ("owner-sid", 0x9404, 1, b"original-dacl")
    helper, native, _, state = _fixture(monkeypatch, initialized=initialized)
    with helper._preserve_dacl(native, 7):
        assert state["security"] == initialized
        state["security"] = ("owner-sid", 0x9404, 1, b"public-dacl")
    assert state["security"] == initialized
    assert state["released"] == 3


@pytest.mark.parametrize(
    "initialized",
    [
        ("other-owner", 0x9404, 1, b"original-dacl"),
        ("owner-sid", 0x8404, 1, b"original-dacl"),
        ("owner-sid", 0x9504, 1, b"original-dacl"),
        ("owner-sid", 0x9404, 2, b"original-dacl"),
        ("owner-sid", 0x9404, 1, b"different-dacl"),
    ],
)
def test_initialization_refuses_security_changes_before_body(monkeypatch, initialized):
    helper, native, _, state = _fixture(monkeypatch, initialized=initialized)
    entered = []
    with (
        pytest.raises(AssertionError, match="fixture security initialization"),
        helper._preserve_dacl(native, 7),
    ):
        entered.append(True)
    assert not entered
    assert state["released"] == 2


def test_initialization_error_prevents_body_and_releases_descriptor(monkeypatch):
    helper, native, _, state = _fixture(monkeypatch, initialize_error=5)
    entered = []
    with pytest.raises(OSError) as caught, helper._preserve_dacl(native, 7):
        entered.append(True)
    assert caught.value.errno == 5
    assert not entered
    assert state["released"] == 1


def test_final_cleanup_does_not_mask_inheritance_bit_loss(monkeypatch):
    helper, native, _, state = _fixture(
        monkeypatch,
        initialized=("owner-sid", 0x9404, 1, b"original-dacl"),
        changed=("owner-sid", 0x9004, 1, b"original-dacl"),
    )
    with (
        pytest.raises(AssertionError, match="original native security not restored"),
        helper._preserve_dacl(native, 7),
    ):
        pass
    assert state["released"] == 3


def test_already_normalized_security_remains_exact(monkeypatch):
    helper, native, original, state = _fixture(monkeypatch, control=0x9404)
    with helper._preserve_dacl(native, 7):
        assert state["security"] == original
    assert state["security"] == original
    assert len(state["restored"]) == 2


def test_initialization_cannot_clear_inheritance_bit(monkeypatch):
    helper, native, _, state = _fixture(
        monkeypatch,
        control=0x9404,
        initialized=("owner-sid", 0x9004, 1, b"original-dacl"),
    )
    entered = []
    with (
        pytest.raises(AssertionError, match="fixture security initialization"),
        helper._preserve_dacl(native, 7),
    ):
        entered.append(True)
    assert not entered
    assert len(state["restored"]) == 2


@pytest.mark.parametrize("cleanup_error", [0, 5])
def test_setup_read_failure_keeps_bounded_cleanup(monkeypatch, cleanup_error):
    helper, native, _, state = _fixture(
        monkeypatch,
        read_failure_at=2,
        restore_error=cleanup_error,
    )
    entered = []
    error = OSError if cleanup_error else RuntimeError
    with pytest.raises(error) as caught, helper._preserve_dacl(native, 7):
        entered.append(True)
    if cleanup_error:
        assert isinstance(caught.value.__context__, RuntimeError)
    assert not entered
    assert len(state["restored"]) == 2
    assert state["released"] == 2
