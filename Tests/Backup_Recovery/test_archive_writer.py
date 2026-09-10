"""Real archive writer publication, reader and crypto round trips."""

from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.capture import CaptureResult
from tldw_chatbook.Backup_Recovery.models import Inventory


def test_existing_backup_is_preserved_by_writer(tmp_path):
    output = tmp_path / "good.tldw-backup.zip"
    output.write_bytes(b"keep")
    capture = CaptureResult(
        tmp_path / "capture", Inventory((), True, "scope", ()), b"{}"
    )
    with pytest.raises(FileExistsError):
        write_archive(capture, output, password=None, cancel=Event())
    assert output.read_bytes() == b"keep"


import hashlib
import json
import stat
import zipfile

from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery.archive_reader import acquire
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.models import StorageItem


def captured(tmp_path, data=b"captured text", *, partial=False):
    root = tmp_path / "capture"
    (root / "payload").mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    (root / "payload/1").write_bytes(data)
    (root / "payload/1").chmod(0o600)
    doc = manifest(data)
    if partial:
        doc["consistency"] = "partial"
        doc["exclusions"] = [{"logical_id": "omitted", "reason": "user_excluded"}]
    return CaptureResult(
        root, Inventory((), not partial, "scope", ()), json.dumps(doc).encode()
    )


@pytest.mark.parametrize("partial", [False, True])
def test_plaintext_roundtrip_uses_actual_reader(tmp_path, partial):
    capture = captured(tmp_path, partial=partial)
    destination = tmp_path / "new.tldw-backup.zip"
    result = write_archive(capture, destination, password=None, cancel=Event())
    assert result.path == destination
    assert result.digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    sealed = acquire(destination, tmp_path / "read", ArchiveLimits(), None, Event())
    assert sealed.manifest_bytes == result.manifest_bytes
    with zipfile.ZipFile(sealed.path) as archive:
        assert archive.read("payload/1") == b"captured text"
        assert all(
            stat.S_IFMT(info.external_attr >> 16) == stat.S_IFREG
            for info in archive.infolist()
        )
    assert json.loads(result.manifest_bytes)["consistency"] == (
        "partial" if partial else "coherent"
    )


@pytest.mark.parametrize("kind", ["source", "capture", "control"])
def test_output_cannot_overlap_authority(tmp_path, kind):
    capture = captured(tmp_path)
    parent = capture.root if kind == "capture" else tmp_path / "protected"
    parent.mkdir(exist_ok=True)
    owner = "recovery.control" if kind == "control" else "notes"
    item = StorageItem(
        owner,
        "authority",
        parent,
        "intentionally_excluded" if kind == "control" else "included",
        (),
    )
    capture = CaptureResult(
        capture.root, Inventory((item,), True, "scope", ()), capture.manifest_bytes
    )
    with pytest.raises(ValueError, match="output_overlap"):
        write_archive(
            capture, parent / "new.tldw-backup.zip", password=None, cancel=Event()
        )
    assert not (parent / "new.tldw-backup.zip").exists()


def test_changed_capture_payload_is_refused(tmp_path):
    capture = captured(tmp_path)
    (capture.root / "payload/1").write_bytes(b"changed")
    destination = tmp_path / "new.tldw-backup.zip"
    with pytest.raises(ValueError):
        write_archive(capture, destination, password=None, cancel=Event())
    assert not destination.exists()


def test_encrypted_roundtrip_uses_real_helper(
    tmp_path, helper_resource_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import crypto

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    capture = captured(tmp_path)
    doc = json.loads(capture.manifest_bytes)
    doc["credential_policy"] = "include"
    capture = CaptureResult(capture.root, capture.inventory, json.dumps(doc).encode())
    destination = tmp_path / "new.tldw-backup.zip.age"
    result = write_archive(
        capture, destination, password=b"test-only password", cancel=Event()
    )
    assert destination.read_bytes().startswith(b"age-encryption.org/")
    assert result.digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    read = acquire(
        destination, tmp_path / "read", ArchiveLimits(), b"test-only password", Event()
    )
    assert read.manifest_bytes == result.manifest_bytes
    with pytest.raises(crypto.CryptoError):
        acquire(destination, tmp_path / "wrong", ArchiveLimits(), b"wrong", Event())


def test_highly_compressible_text_stays_within_normal_reader_limits(tmp_path):
    capture = captured(tmp_path, b"text " * 100000)
    result = write_archive(
        capture, tmp_path / "new.tldw-backup.zip", password=None, cancel=Event()
    )
    with zipfile.ZipFile(result.path) as archive:
        assert archive.getinfo("payload/1").compress_type == zipfile.ZIP_STORED
    acquire(result.path, tmp_path / "read", ArchiveLimits(), None, Event())


def test_racing_destination_is_never_overwritten(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import archive_writer as writer

    capture = captured(tmp_path)
    destination = tmp_path / "new.tldw-backup.zip"
    publish = writer.publish_new

    def race(source, target):
        target.write_bytes(b"race-winner")
        publish(source, target)

    monkeypatch.setattr(writer, "publish_new", race)
    with pytest.raises(FileExistsError):
        write_archive(capture, destination, password=None, cancel=Event())
    assert destination.read_bytes() == b"race-winner"
    assert not list(tmp_path.glob(".backup-write-*"))


@pytest.mark.parametrize("when", ["before", "after", "ambiguous"])
def test_publication_interruption_preserves_real_outcome(tmp_path, monkeypatch, when):
    from tldw_chatbook.Backup_Recovery import archive_writer as writer

    capture = captured(tmp_path)
    destination = tmp_path / "new.tldw-backup.zip"
    cancel = Event()
    publish = writer.publish_new

    def interrupted(source, target):
        if when == "before":
            raise InterruptedError("cancelled")
        publish(source, target)
        cancel.set()
        if when == "ambiguous":
            raise OSError("durability-uncertain")

    monkeypatch.setattr(writer, "publish_new", interrupted)
    if when == "after":
        result = write_archive(capture, destination, password=None, cancel=cancel)
        assert result.path == destination
    else:
        with pytest.raises(OSError):
            write_archive(capture, destination, password=None, cancel=cancel)
    assert destination.exists() == (when != "before")
    if destination.exists():
        acquire(destination, tmp_path / "read", ArchiveLimits(), None, Event())


def test_corrupted_packaging_is_caught_by_actual_reader(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import archive_writer as writer

    capture = captured(tmp_path)
    package = writer._package

    def corrupted(*args, **kwargs):
        result = package(*args, **kwargs)
        with args[2].open("ab") as output:
            output.write(b"trailing corruption")
        return result

    monkeypatch.setattr(writer, "_package", corrupted)
    destination = tmp_path / "new.tldw-backup.zip"
    with pytest.raises(ValueError):
        write_archive(capture, destination, password=None, cancel=Event())
    assert not destination.exists()


def test_runtime_enospc_leaves_no_published_or_temporary_artifact(
    tmp_path, monkeypatch
):
    import errno

    from tldw_chatbook.Backup_Recovery import archive_writer as writer

    capture = captured(tmp_path)

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "full")

    monkeypatch.setattr(writer.zipfile.ZipFile, "writestr", full)
    destination = tmp_path / "new.tldw-backup.zip"
    with pytest.raises(OSError):
        write_archive(capture, destination, password=None, cancel=Event())
    assert not destination.exists()
    assert not list(tmp_path.glob(".backup-write-*"))


def test_explicit_external_output_exclusion_is_honored(tmp_path):
    capture = captured(tmp_path)
    external = tmp_path / "external"
    output_root = external / "outputs"
    output_root.mkdir(parents=True)
    items = (
        StorageItem("external.selected", "external", external, "included", ()),
        StorageItem(
            "recovery.output", "output", output_root, "intentionally_excluded", ()
        ),
    )
    capture = CaptureResult(
        capture.root, Inventory(items, True, "scope", ()), capture.manifest_bytes
    )
    destination = output_root / "new.tldw-backup.zip"
    assert (
        write_archive(capture, destination, password=None, cancel=Event()).path
        == destination
    )


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_capture_payload_alias_refused(tmp_path, kind):
    capture = captured(tmp_path)
    path = capture.root / "payload/1"
    peer = tmp_path / "peer"
    path.rename(peer)
    if kind == "symlink":
        path.symlink_to(peer)
    else:
        path.hardlink_to(peer)
    destination = tmp_path / "new.tldw-backup.zip"
    with pytest.raises((OSError, ValueError)):
        write_archive(capture, destination, password=None, cancel=Event())
    assert peer.read_bytes() == b"captured text"
    assert not destination.exists()


def test_cancel_before_packaging_has_no_output(tmp_path):
    capture = captured(tmp_path)
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        write_archive(
            capture, tmp_path / "new.tldw-backup.zip", password=None, cancel=cancel
        )
    assert not list(tmp_path.glob(".backup-write-*"))


def test_unencrypted_credentials_never_packaged(tmp_path):
    capture = captured(tmp_path)
    doc = json.loads(capture.manifest_bytes)
    doc["credential_policy"] = "rollback"
    capture = CaptureResult(capture.root, capture.inventory, json.dumps(doc).encode())
    with pytest.raises(ValueError, match="encryption_required"):
        write_archive(
            capture, tmp_path / "new.tldw-backup.zip", password=None, cancel=Event()
        )
    assert not list(tmp_path.glob(".backup-write-*"))


def test_source_alias_changed_before_publication_is_rechecked(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import archive_writer as writer

    capture = captured(tmp_path)
    alias = tmp_path / "source-alias"
    target = tmp_path / "other"
    target.mkdir()
    alias.symlink_to(target, target_is_directory=True)
    capture = CaptureResult(
        capture.root,
        Inventory(
            (StorageItem("external.selected", "source", alias, "included", ()),),
            True,
            "scope",
            (),
        ),
        capture.manifest_bytes,
    )
    inspect = writer.reader.acquire

    def changed(*args, **kwargs):
        result = inspect(*args, **kwargs)
        alias.unlink()
        alias.symlink_to(tmp_path, target_is_directory=True)
        return result

    monkeypatch.setattr(writer.reader, "acquire", changed)
    destination = tmp_path / "new.tldw-backup.zip"
    with pytest.raises(ValueError, match="output_overlap"):
        write_archive(capture, destination, password=None, cancel=Event())
    assert not destination.exists()


def test_explicit_directory_metadata_and_media_storage_roundtrip(tmp_path):
    capture = captured(tmp_path)
    doc = json.loads(capture.manifest_bytes)
    doc["directories"].append(
        {
            "logical_id": "folder",
            "root_id": "root",
            "parent_id": "root",
            "relative_path": "folder",
            "metadata": {"version": 1, "mtime_ns": 123456789, "mode": 493},
        }
    )
    doc["files"][0].update(parent_id="folder", relative_path="folder/image.png")
    capture = CaptureResult(capture.root, capture.inventory, json.dumps(doc).encode())
    result = write_archive(
        capture, tmp_path / "new.tldw-backup.zip", password=None, cancel=Event()
    )
    checked = acquire(result.path, tmp_path / "read", ArchiveLimits(), None, Event())
    assert json.loads(checked.manifest_bytes)["directories"] == doc["directories"]
    with zipfile.ZipFile(checked.path) as archive:
        assert archive.getinfo("payload/1").compress_type == zipfile.ZIP_STORED
        assert not any(info.is_dir() for info in archive.infolist())


def test_default_bootstrap_control_root_is_refused(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import archive_writer as writer

    capture = captured(tmp_path)
    control = tmp_path / "bootstrap"
    control.mkdir()
    monkeypatch.setattr(writer, "default_bootstrap_root", lambda: control)
    with pytest.raises(ValueError, match="output_overlap"):
        write_archive(
            capture, control / "new.tldw-backup.zip", password=None, cancel=Event()
        )


@pytest.mark.parametrize("phase", ["before", "after"])
def test_real_process_exit_at_publication_boundary(tmp_path, phase):
    import subprocess
    import sys

    capture = captured(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(capture.manifest_bytes)
    destination = tmp_path / "new.tldw-backup.zip"
    code = """
import os, sys
from pathlib import Path
from threading import Event
from tldw_chatbook.Backup_Recovery import archive_writer as writer
from tldw_chatbook.Backup_Recovery.capture import CaptureResult
from tldw_chatbook.Backup_Recovery.models import Inventory
root, manifest, target = map(Path, sys.argv[1:4])
capture = CaptureResult(root, Inventory((), True, "scope", ()), manifest.read_bytes())
publish = writer.publish_new
def stop(source, destination):
    if sys.argv[4] == "before":
        os._exit(23)
    publish(source, destination)
    os._exit(24)
writer.publish_new = stop
writer.write_archive(capture, target, password=None, cancel=Event())
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(capture.root),
            str(manifest_path),
            str(destination),
            phase,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
        check=False,
    )
    assert child.returncode == (23 if phase == "before" else 24)
    assert destination.exists() == (phase == "after")
    if destination.exists():
        acquire(destination, tmp_path / "read", ArchiveLimits(), None, Event())
