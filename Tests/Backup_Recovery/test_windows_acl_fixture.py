"""ACL fixture cleanup must restore and verify the captured native state."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def _fixture(monkeypatch, *, changed=None, restore_error=0, control=0x9004):
    from Tests.Backup_Recovery import windows_acl_fixture as helper

    original = ("owner-sid", control, 1, b"original-dacl")
    state = {"security": original, "released": 0, "restored": []}

    @contextmanager
    def security(native, handle):
        try:
            yield "original-pointer", state["security"]
        finally:
            state["released"] += 1

    def restore(handle, kind, flags, owner, group, dacl, sacl):
        state["restored"].append((handle, kind, flags, owner, group, dacl, sacl))
        if not restore_error:
            state["security"] = changed or original
        return restore_error

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
    assert state["released"] == 2
    assert state["restored"] == [
        (7, 1, 4 | 0x80000000, None, None, "original-pointer", None)
    ]


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
    assert state["released"] == 2


def test_native_restore_error_fails_and_releases_original(monkeypatch):
    helper, native, _, state = _fixture(monkeypatch, restore_error=5)
    with pytest.raises(OSError) as caught, helper._preserve_dacl(native, 7):
        pass
    assert caught.value.errno == 5
    assert state["released"] == 1


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
    assert state["released"] == 2


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
    assert state["released"] == 1
