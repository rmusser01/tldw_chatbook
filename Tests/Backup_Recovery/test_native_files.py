"""Real native publication evidence; failures are not skipped."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery.native_files import (
    _rename_new,
    create_private_directory,
    create_private_file,
    flush_directory,
    pinned_directory,
    publish_new,
)
from tldw_chatbook.Backup_Recovery.qualification import native_identity, qualified_for


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_installed_object_metadata_barrier_failure_propagates(
    tmp_path, monkeypatch, kind
):
    from tldw_chatbook.Backup_Recovery import publication

    target = tmp_path / "installed"
    if kind == "directory":
        target.mkdir(mode=0o700)
    else:
        target.write_bytes(b"installed content")
        target.chmod(0o600)
    before = publication.os.stat(target)
    expected = before.st_dev, before.st_ino
    barrier_name = "flush_directory" if kind == "directory" else "flush_file"
    original_barrier = getattr(publication, barrier_name)

    def fail_installed_barrier(fd):
        info = publication.os.fstat(fd)
        if (info.st_dev, info.st_ino) == expected:
            raise OSError("installed_metadata_barrier_failed")
        original_barrier(fd)

    monkeypatch.setattr(publication, barrier_name, fail_installed_barrier)
    with pytest.raises(OSError, match="installed_metadata_barrier_failed"):
        publication._installed_metadata(
            target,
            expected,
            {
                "mode": 0o700 if kind == "directory" else 0o600,
                "mtime_ns": before.st_mtime_ns,
            },
        )


@pytest.mark.parametrize("kind", ["file", "empty_directory", "populated_directory"])
def test_raw_native_no_replace_and_directory_flush(tmp_path, kind):
    source, target = tmp_path / "source", tmp_path / "target"
    if kind == "file":
        source.write_bytes(b"candidate")
        target.write_bytes(b"previous")
    else:
        source.mkdir()
        target.mkdir()
        if kind == "populated_directory":
            (source / "content").write_bytes(b"candidate")
            (target / "content").write_bytes(b"previous")
    with pinned_directory(tmp_path) as parent:
        with pytest.raises(FileExistsError):
            _rename_new(parent, "source", parent, "target")
        assert source.exists() and target.exists()
        _rename_new(parent, "source", parent, "published")
        flush_directory(parent)
    assert not source.exists()
    if kind == "file":
        assert target.read_bytes() == b"previous"
        assert (tmp_path / "published").read_bytes() == b"candidate"
    elif kind == "populated_directory":
        assert (target / "content").read_bytes() == b"previous"
        assert (tmp_path / "published/content").read_bytes() == b"candidate"


def test_raw_native_process_death_after_rename_preserves_evidence(tmp_path):
    (tmp_path / "stage").write_bytes(b"candidate")
    code = """
import os, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.native_files import pinned_directory, _rename_new
with pinned_directory(Path(sys.argv[1])) as parent:
    _rename_new(parent, "stage", parent, "published")
    os._exit(23)
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], timeout=10, check=False
    )
    assert child.returncode == 23
    assert not (tmp_path / "stage").exists()
    assert (tmp_path / "published").read_bytes() == b"candidate"
    with pytest.raises(FileExistsError):
        (tmp_path / "published").open("xb")


def test_private_creation_and_native_identity(tmp_path):
    directory = tmp_path / "private"
    create_private_directory(directory)
    with create_private_file(directory / "candidate") as fd:
        os.write(fd, b"candidate")
    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / "candidate").stat().st_mode & 0o777 == 0o600
    with pinned_directory(directory) as fd:
        assert native_identity(fd)["filesystem"] == (
            "apfs" if sys.platform == "darwin" else "ext4"
        )
    with pytest.raises(FileExistsError):
        create_private_directory(directory)


@pytest.mark.parametrize("kind", ["file", "empty_directory", "populated_directory"])
def test_qualified_publication(tmp_path, kind):
    source = tmp_path / "stage"
    if kind == "file":
        source.write_bytes(b"candidate")
    else:
        source.mkdir()
        if kind == "populated_directory":
            (source / "payload").write_bytes(b"candidate")
    assert qualified_for("publish_new", tmp_path)[0]
    publish_new(source, tmp_path / "published")
    assert not source.exists()
    assert (tmp_path / "published").exists()


def test_unqualified_operations_and_symlink_storage_are_visible(tmp_path):
    for operation in ("replacement", "isolated_restore", "invented"):
        assert qualified_for(operation, tmp_path) == (False, "operation_not_qualified")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    assert qualified_for("publish_new", alias) == (False, "qualification_unavailable")
    source = tmp_path / "source"
    source.write_bytes(b"new")
    with pytest.raises(OSError, match="qualification_unavailable"):
        publish_new(source, alias / "destination")
    assert source.read_bytes() == b"new"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_non_private_staged_objects_are_refused(tmp_path, kind):
    original = tmp_path / "original"
    original.write_bytes(b"original")
    stage = tmp_path / "stage"
    if kind == "symlink":
        stage.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, stage)
    else:
        os.mkfifo(stage)
    with pytest.raises(OSError):
        publish_new(stage, tmp_path / "destination")
    assert original.read_bytes() == b"original"
    assert not (tmp_path / "destination").exists()


def test_publication_race_has_exactly_one_winner(tmp_path):
    code = """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.native_files import publish_new
root = Path(sys.argv[1])
source = root / sys.argv[2]
source.write_text(sys.argv[2])
print("ready", flush=True)
sys.stdin.readline()
try:
    publish_new(source, root / "destination")
    print("published", flush=True)
except FileExistsError:
    print("exists", flush=True)
"""
    children = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", code, str(tmp_path), str(i)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        for i in range(4)
    ]
    try:
        for child in children:
            assert child.stdout.readline().strip() == "ready"
        for child in children:
            child.stdin.write("go\n")
            child.stdin.flush()
        outcomes = [child.communicate(timeout=10)[0].strip() for child in children]
        assert outcomes.count("published") == 1
        assert outcomes.count("exists") == 3
        assert (tmp_path / "destination").read_text() in {"0", "1", "2", "3"}
        assert all(child.returncode == 0 for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


def test_evidence_identifies_only_demonstrated_operations():
    from tldw_chatbook.Backup_Recovery import qualification

    evidence = json.loads(
        Path(qualification.__file__).with_name("native_qualification.json").read_text()
    )
    assert evidence["schema_version"] == 1
    assert evidence["evidence"][0]["operations"] == [
        "publish_new",
        "publish_file",
        "publish_directory",
        "admission",
    ]


def test_directory_publication_refuses_linked_payload(tmp_path):
    external = tmp_path / "external"
    external.write_bytes(b"external")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "payload").symlink_to(external)
    with pytest.raises(OSError):
        publish_new(stage, tmp_path / "destination")
    assert not (tmp_path / "destination").exists()
    assert external.read_bytes() == b"external"


@pytest.mark.parametrize("target_kind", ["symlink", "hardlink", "directory"])
def test_preexisting_target_of_any_type_is_preserved(tmp_path, target_kind):
    original = tmp_path / "original"
    original.write_bytes(b"previous")
    target = tmp_path / "target"
    if target_kind == "symlink":
        target.symlink_to(original)
    elif target_kind == "hardlink":
        os.link(original, target)
    else:
        target.mkdir()
    stage = tmp_path / "stage"
    stage.write_bytes(b"candidate")
    with pytest.raises(FileExistsError):
        publish_new(stage, target)
    assert stage.read_bytes() == b"candidate"
    assert original.read_bytes() == b"previous"
    assert target.exists()


def test_native_directory_full_flush_supported(tmp_path):
    """Qualify the actual post-metadata barrier on this native filesystem."""

    (tmp_path / "entry").write_bytes(b"new")
    with pinned_directory(tmp_path) as parent:
        flush_directory(parent)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_full_flush_occurs_after_publication(tmp_path, monkeypatch, kind):
    from tldw_chatbook.Backup_Recovery import native_files

    source, destination = tmp_path / "stage", tmp_path / "published"
    if kind == "file":
        source.write_bytes(b"candidate")
    else:
        source.mkdir()
        (source / "content").write_bytes(b"candidate")
    calls = []
    native_flush = native_files.flush_directory

    def observe(fd):
        result = native_flush(fd)
        calls.append((os.fstat(fd).st_ino, source.exists(), destination.exists()))
        return result

    monkeypatch.setattr(native_files, "flush_directory", observe)
    publish_new(source, destination)
    parent_inode = tmp_path.stat().st_ino
    assert (parent_inode, False, True) in calls


def test_failed_post_publication_full_flush_preserves_published_evidence(
    tmp_path, monkeypatch
):
    import errno

    from tldw_chatbook.Backup_Recovery import native_files

    source, destination = tmp_path / "stage", tmp_path / "published"
    source.write_bytes(b"candidate")
    native_flush = native_files.flush_directory
    parent_inode = tmp_path.stat().st_ino

    def fail_after_native_metadata_flush(fd):
        result = native_flush(fd)
        if os.fstat(fd).st_ino == parent_inode and destination.exists():
            raise OSError(errno.EIO, "injected_post_publication_flush_failure")
        return result

    monkeypatch.setattr(
        native_files, "flush_directory", fail_after_native_metadata_flush
    )
    with pytest.raises(OSError, match="injected_post_publication_flush_failure"):
        publish_new(source, destination)
    assert not source.exists()
    assert destination.read_bytes() == b"candidate"
    # No cleanup/replace retry is allowed after an ambiguous publication result.
    source.write_bytes(b"another")
    with pytest.raises(FileExistsError):
        publish_new(source, destination)
    assert destination.read_bytes() == b"candidate"


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_protocol",
        "unsupported_protocol",
        "previous_protocol",
        "boolean_protocol",
        "string_operations",
        "mixed_operations",
        "missing_operations",
        "unknown_operation",
        "unknown_field",
        "boolean_schema",
        "wrong_rows_type",
    ],
)
def test_malformed_qualification_evidence_never_grants_capability(
    tmp_path, monkeypatch, invalid
):
    from tldw_chatbook.Backup_Recovery import qualification

    record = json.loads(
        Path(qualification.__file__).with_name("native_qualification.json").read_text()
    )
    row = record["evidence"][0]
    if invalid == "missing_protocol":
        row.pop("protocol")
    elif invalid == "unsupported_protocol":
        row["protocol"] = 999
    elif invalid == "previous_protocol":
        row["protocol"] = 1
    elif invalid == "boolean_protocol":
        row["protocol"] = True
    elif invalid == "string_operations":
        row["operations"] = "publish_new"
    elif invalid == "mixed_operations":
        row["operations"].append(9)
    elif invalid == "missing_operations":
        row.pop("operations")
    elif invalid == "unknown_operation":
        row["operations"].append("unreviewed_operation")
    elif invalid == "unknown_field":
        row["future_policy"] = "automatic"
    elif invalid == "boolean_schema":
        record["schema_version"] = True
    else:
        record["evidence"] = row
    (tmp_path / "native_qualification.json").write_text(json.dumps(record))
    monkeypatch.setattr(qualification, "__file__", str(tmp_path / "qualification.py"))
    assert qualification.qualified_for("publish_new", tmp_path) == (
        False,
        "qualification_evidence_invalid",
    )
