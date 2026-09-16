"""Local recovery reviews are durable and scoped to one generation and owner."""

import json
import os
import subprocess
import sys

import pytest

from tldw_chatbook.Backup_Recovery.activation import ActivationStore


def test_approving_one_owner_does_not_resume_another(tmp_path):
    state = ActivationStore(tmp_path / "activation")
    state.require("generation-1", ("sync", "schedules"))
    state.approve("generation-1", "sync")
    reopened = ActivationStore(tmp_path / "activation")
    assert reopened.allowed("generation-1", "sync") is True
    assert reopened.allowed("generation-1", "schedules") is False


def test_missing_state_refuses_without_creating_control_files(tmp_path):
    root = tmp_path / "absent"
    assert ActivationStore(root).allowed("g", "sync") is False
    assert not root.exists()


def test_require_is_idempotent_without_resetting_review(tmp_path):
    state = ActivationStore(tmp_path / "activation")
    state.require("g", ("sync", "schedules"))
    state.approve("g", "sync")
    before = {p: p.read_bytes() for p in state.root.rglob("*.json")}
    state.require("g", ("schedules", "sync"))
    state.approve("g", "sync")
    assert before == {p: p.read_bytes() for p in state.root.rglob("*.json")}
    with pytest.raises(ValueError, match="activation_requirements_changed"):
        state.require("g", ("sync",))
    assert state.allowed("g", "schedules") is False


def test_unknown_owner_and_generation_cannot_be_approved(tmp_path):
    state = ActivationStore(tmp_path / "activation")
    state.require("g", ("sync",))
    for generation, owner in (("other", "sync"), ("g", "schedules")):
        with pytest.raises(ValueError, match="activation_review_unavailable"):
            state.approve(generation, owner)
        assert state.allowed(generation, owner) is False


@pytest.mark.parametrize(
    "damage",
    ["missing", "truncated", "generation", "extra", "duplicate", "mode", "link"],
)
def test_damaged_requirement_never_grants_or_repairs_review(tmp_path, damage):
    state = ActivationStore(tmp_path / "activation")
    state.require("g", ("sync",))
    state.approve("g", "sync")
    record = next(state.root.rglob("required.json"))
    if damage == "missing":
        record.unlink()
    elif damage == "truncated":
        record.write_bytes(b'{"version":1,')
    elif damage == "duplicate":
        record.write_bytes(b'{"version":1,"version":1}')
    elif damage == "mode":
        record.chmod(0o644)
    elif damage == "link":
        os.link(record, tmp_path / "linked")
    else:
        data = json.loads(record.read_bytes())
        data["generation" if damage == "generation" else "approved"] = "other"
        record.write_text(json.dumps(data))
    assert not state.allowed("g", "sync")
    with pytest.raises(ValueError, match="activation_review_unavailable"):
        state.approve("g", "sync")


def test_approval_from_other_generation_is_not_authority(tmp_path):
    state = ActivationStore(tmp_path / "activation")
    state.require("old", ("sync",))
    state.approve("old", "sync")
    approval = next(state.root.rglob("approved-*.json"))
    state.require("new", ("sync",))
    new_dir = next(
        p.parent
        for p in state.root.rglob("required.json")
        if json.loads(p.read_bytes())["generation"] == "new"
    )
    copied = new_dir / approval.name
    copied.write_bytes(approval.read_bytes())
    copied.chmod(0o600)
    assert not state.allowed("new", "sync")


def test_review_survives_fresh_process(tmp_path):
    root = tmp_path / "activation"
    state = ActivationStore(root)
    state.require("g", ("sync", "schedules"))
    state.approve("g", "sync")
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; from tldw_chatbook.Backup_Recovery.activation import ActivationStore; s=ActivationStore(Path(sys.argv[1])); assert s.allowed('g','sync'); assert not s.allowed('g','schedules')",
            str(root),
        ],
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert child.returncode == 0, child.stderr.decode()


@pytest.mark.parametrize("damage", ["truncated", "owner", "mode", "symlink"])
def test_damaged_approval_cannot_be_used_or_silently_replaced(tmp_path, damage):
    state = ActivationStore(tmp_path / "activation")
    state.require("g", ("sync", "schedules"))
    state.approve("g", "sync")
    record = next(state.root.rglob("approved-*.json"))
    if damage == "truncated":
        record.write_bytes(b"{")
    elif damage == "owner":
        data = json.loads(record.read_bytes())
        data["owner"] = "schedules"
        record.write_text(json.dumps(data))
    elif damage == "mode":
        record.chmod(0o644)
    else:
        other = tmp_path / "other.json"
        record.rename(other)
        record.symlink_to(other)
    assert not state.allowed("g", "sync")
    with pytest.raises(ValueError, match="activation_review_unavailable"):
        state.approve("g", "sync")


@pytest.mark.parametrize("damage", ["public", "symlink"])
def test_unsafe_control_root_is_never_repaired(tmp_path, damage):
    root = tmp_path / "activation"
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    if damage == "public":
        root.mkdir(mode=0o755)
        root.chmod(0o755)
    else:
        root.symlink_to(actual, target_is_directory=True)
    state = ActivationStore(root)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        state.require("g", ("sync",))
    assert not state.allowed("g", "sync")
    assert not list(actual.iterdir())
    if damage == "public":
        assert root.stat().st_mode & 0o777 == 0o755


def test_killed_initial_requirement_is_not_recreated(tmp_path):
    root = tmp_path / "activation"
    script = """
import os, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.activation import ActivationStore, Admission
def killed(parent, name, data):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
    os.write(fd, b'{')
    os.fsync(fd)
    os._exit(73)
Admission._write_new_record = killed
ActivationStore(Path(sys.argv[1])).require('g', ('sync',))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert child.returncode == 73, child.stderr.decode()
    state = ActivationStore(root)
    record = next(root.rglob("required.json"))
    with pytest.raises(ValueError):
        state.require("g", ("sync",))
    assert record.read_bytes() == b"{"
    assert not state.allowed("g", "sync")


@pytest.mark.parametrize("operation", ["require", "approve"])
def test_retry_reestablishes_durability_after_complete_ambiguous_write(
    tmp_path, monkeypatch, operation
):
    from tldw_chatbook.Backup_Recovery import activation

    state = ActivationStore(tmp_path / "activation")
    if operation == "approve":
        state.require("g", ("sync",))
    original = activation.Admission._write_new_record

    def written_then_failed(parent, name, data):
        original(parent, name, data)
        raise OSError("simulated_final_barrier_failure")

    def invoke():
        if operation == "require":
            state.require("g", ("sync",))
        else:
            state.approve("g", "sync")

    monkeypatch.setattr(activation.Admission, "_write_new_record", written_then_failed)
    with pytest.raises((OSError, ValueError)):
        invoke()
    monkeypatch.setattr(activation.Admission, "_write_new_record", original)
    barrier = activation.os.fsync
    attempts = []

    def failing_barrier(fd):
        attempts.append(fd)
        raise OSError("simulated_retry_barrier_failure")

    monkeypatch.setattr(activation.os, "fsync", failing_barrier)
    with pytest.raises((OSError, ValueError)):
        invoke()
    assert attempts
    monkeypatch.setattr(activation.os, "fsync", barrier)
    invoke()
