"""Pending publication flushes changed directories, preserving native barriers."""

import errno
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.subprocess_pipes import popen_with_captured_stderr, read_line
from tldw_chatbook.Backup_Recovery import control_records, native_files, publication
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Utils.platform_files import os


@pytest.fixture
def pending_case(tmp_path):
    selector = tmp_path / "config.toml"
    selector.write_text("[general]\n", encoding="utf-8")
    root = tmp_path / "created" / "nested" / "bootstrap"
    native_files.create_private_directory(tmp_path / "control")
    journal = Journal(tmp_path / "control", "pending-durability")
    context = SimpleNamespace(
        bootstrap_root=str(root), namespaces=["profile"], selectors=[str(selector)]
    )
    control_records.register_pending(
        root, journal.operation_id, ("profile",), journal.root.parent, (selector,)
    )
    return root, journal, context


def test_pending_verification_flushes_only_certified_bootstrap_directory(
    pending_case, monkeypatch
):
    root, journal, context = pending_case
    real_flush = publication.flush_directory
    observed = []

    def observe(fd):
        info = os.fstat(fd)
        observed.append((info.st_dev, info.st_ino))
        real_flush(fd)

    monkeypatch.setattr(publication, "flush_directory", observe)
    publication._pending(journal, context, durable=True)
    info = os.stat(root)
    assert observed == [(info.st_dev, info.st_ino)]


def test_failed_creation_barrier_cannot_be_adopted_on_retry(tmp_path, monkeypatch):
    selector = tmp_path / "config.toml"
    selector.write_text("[general]\n", encoding="utf-8")
    root = tmp_path / "bootstrap"
    real_flush = control_records.flush_directory
    calls = []

    def fail_creation(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError(errno.EIO, "creation barrier failed")
        real_flush(fd)

    monkeypatch.setattr(control_records, "flush_directory", fail_creation)
    with pytest.raises(OSError, match="creation barrier failed"):
        control_records.register_pending(
            root, "failed", ("profile",), tmp_path / "control", (selector,)
        )
    assert root.is_dir()
    assert (tmp_path / control_records._DIRECTORY_CREATION_INTENT).is_file()
    monkeypatch.setattr(control_records, "flush_directory", real_flush)
    with pytest.raises(
        control_records.RecoveryRequired, match="bootstrap_creation_unsettled"
    ):
        control_records.register_pending(
            root, "retry", ("profile",), tmp_path / "control", (selector,)
        )
    assert not list(root.glob("pending-*.json"))


@pytest.mark.parametrize("phase", ["record", "directory"])
def test_pending_required_barrier_failure_remains_a_refusal(
    pending_case, monkeypatch, phase
):
    _, journal, context = pending_case

    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "required pending barrier failed")

    if phase == "record":
        monkeypatch.setattr(journal, "_flush_record", fail)
    else:
        monkeypatch.setattr(publication, "flush_directory", fail)
    with pytest.raises(OSError, match="required pending barrier failed"):
        publication._pending(journal, context, durable=True)


def test_registration_flushes_every_parent_it_changes(tmp_path, monkeypatch):
    selector = tmp_path / "config.toml"
    selector.write_text("[general]\n", encoding="utf-8")
    root = tmp_path / "new" / "nested" / "bootstrap"
    real_flush = control_records.flush_directory
    observed = []

    def observe(fd):
        info = os.fstat(fd)
        observed.append((info.st_dev, info.st_ino))
        real_flush(fd)

    monkeypatch.setattr(control_records, "flush_directory", observe)
    control_records.register_pending(
        root, "new-registration", ("profile",), tmp_path / "control", (selector,)
    )
    expected = [
        (info.st_dev, info.st_ino)
        for info in map(os.stat, (tmp_path, root.parent.parent, root.parent))
    ]
    assert observed[:9] == [identity for identity in expected for _ in range(3)]
    assert observed[9:] == [(os.stat(root).st_dev, os.stat(root).st_ino)] * 2


def test_process_death_after_mkdir_cannot_bypass_parent_barrier(tmp_path, monkeypatch):
    selector = tmp_path / "config.toml"
    selector.write_text("[general]\n", encoding="utf-8")
    native_files.create_private_directory(tmp_path / "control")
    journal = Journal(tmp_path / "control", "crashed-registration")
    root = tmp_path / "bootstrap"
    child = popen_with_captured_stderr(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; from pathlib import Path; "
            "from tldw_chatbook.Backup_Recovery import control_records as c; "
            "root=Path(sys.argv[1]); real=c.flush_directory; calls=[]\n"
            "def pause(fd):\n"
            " calls.append(fd)\n"
            " if len(calls)==2:\n"
            "  print('created-before-flush', flush=True)\n  sys.stdin.readline()\n"
            " real(fd)\n"
            "c.flush_directory=pause\nc._ensure(root)\n",
            str(root),
        ],
        tmp_path / "creator.stderr",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert read_line(child) == "created-before-flush"
    finally:
        child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()
    assert root.is_dir()
    assert (tmp_path / control_records._DIRECTORY_CREATION_INTENT).is_file()

    def revoked_rights(fd):
        raise PermissionError(errno.EACCES, "rights changed after process death")

    monkeypatch.setattr(control_records, "flush_directory", revoked_rights)
    with pytest.raises(
        control_records.RecoveryRequired, match="bootstrap_creation_unsettled"
    ):
        control_records.register_pending(
            root, journal.operation_id, ("profile",), journal.root.parent, (selector,)
        )
    assert not list(root.glob("pending-*.json"))


def test_legacy_pending_keeps_uncertified_ancestor_barriers(pending_case, monkeypatch):
    root, journal, context = pending_case
    (root / control_records._ANCESTRY_RECORD).unlink()
    real_flush = publication.flush_directory
    ancestor = os.stat(root.parent)

    def fail(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (ancestor.st_dev, ancestor.st_ino):
            raise OSError(errno.EIO, "legacy ancestor required")
        real_flush(fd)

    monkeypatch.setattr(publication, "flush_directory", fail)
    with pytest.raises(OSError, match="legacy ancestor required"):
        publication._pending(journal, context, durable=True)


def test_registration_cannot_certify_unresolved_legacy_pending(pending_case):
    root, journal, context = pending_case
    (root / control_records._ANCESTRY_RECORD).unlink()
    with pytest.raises(
        control_records.RecoveryRequired, match="bootstrap_ancestry_unverified"
    ):
        control_records.register_pending(
            root,
            "new-operation",
            ("profile",),
            journal.root.parent,
            tuple(map(Path, context.selectors)),
        )
    assert not (root / control_records._ANCESTRY_RECORD).exists()


@pytest.mark.parametrize("field", ["root_inode", "root_device", "root_path"])
def test_creation_receipt_cannot_exempt_another_directory(pending_case, field):
    import json

    root, journal, context = pending_case
    proof = root / control_records._ANCESTRY_RECORD
    data = json.loads(proof.read_text(encoding="utf-8"))
    data[field] = "another-root" if field == "root_path" else -1
    proof.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_bootstrap_ancestry"):
        publication._pending(journal, context, durable=True)


def test_creation_and_retirement_use_the_same_pinned_parent(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    native_files.create_private_directory(parent)
    moved = tmp_path / "moved"
    root = parent / "bootstrap"
    real_flush = control_records.flush_directory
    calls = []

    def move_after_intent(fd):
        real_flush(fd)
        calls.append(fd)
        if len(calls) == 1:
            parent.rename(moved)
            native_files.create_private_directory(parent)

    monkeypatch.setattr(control_records, "flush_directory", move_after_intent)
    with pytest.raises(FileNotFoundError):
        control_records._ensure(root)
    assert (moved / "bootstrap").is_dir()
    assert not root.exists()
    assert not (parent / control_records._DIRECTORY_CREATION_INTENT).exists()


def test_creation_intent_survives_alternate_case_retry(tmp_path, monkeypatch):
    root = tmp_path / "Bootstrap"
    real_flush = control_records.flush_directory
    calls = []

    def fail_after_mkdir(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError(errno.EIO, "creation interrupted")
        real_flush(fd)

    monkeypatch.setattr(control_records, "flush_directory", fail_after_mkdir)
    with pytest.raises(OSError, match="creation interrupted"):
        control_records._ensure(root)
    monkeypatch.setattr(control_records, "flush_directory", real_flush)
    alias = root.with_name("bootstrap")
    with pytest.raises(
        control_records.RecoveryRequired, match="bootstrap_creation_unsettled"
    ):
        control_records._ensure(alias)
