from __future__ import annotations

import os
import hashlib
import shutil
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Notes import notes_sync_filesystem, sync_paths
from tldw_chatbook.Notes.notes_sync_filesystem import (
    NotesSyncFilesystemError,
    NotesSyncFilesystemPartialError,
    PosixNotesSyncFilesystem,
    WindowsNotesSyncObservationFilesystem,
    validate_sync_root_admission,
)
from tldw_chatbook.Notes.note_import_discovery import (
    DiscoveredImportSource,
    ImportDiscovery,
    SourceIdentity,
)
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportBounds,
    ImportSource,
    ImportSourceKind,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="writable lasting-sync filesystem is POSIX-specific",
)


@pytest.mark.parametrize(
    ("payload", "bom", "newline", "final_newline"),
    [
        (b"alpha\nbeta\n", False, "lf", True),
        (b"alpha\nbeta", False, "lf", False),
        (b"\xef\xbb\xbfalpha\r\nbeta\r\n", True, "crlf", True),
        (b"\xef\xbb\xbfalpha\r\nbeta", True, "crlf", False),
    ],
)
def test_observe_and_serialize_round_trip_exact_representation(
    tmp_path: Path,
    payload: bytes,
    bom: bool,
    newline: str,
    final_newline: bool,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.md"
    target.write_bytes(payload)
    target.chmod(0o640)

    with PosixNotesSyncFilesystem(root) as filesystem:
        snapshot = filesystem.observe("note.md")
        serialized = filesystem.serialize(
            snapshot.text, snapshot.observation.serialization
        )

    assert snapshot.raw_bytes == payload
    assert serialized == payload
    assert snapshot.observation.serialization.utf8_bom is bom
    assert snapshot.observation.serialization.newline == newline
    assert snapshot.observation.serialization.final_newline is final_newline
    assert snapshot.observation.serialization.mode == 0o640
    assert snapshot.representation_digest == hashlib.sha256(payload).hexdigest()
    if bom or newline == "crlf":
        assert snapshot.representation_digest != snapshot.observation.content_digest


def test_multiple_final_newlines_round_trip_without_collapse(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = b"body\n\n\n"
    (root / "note.md").write_bytes(payload)

    with PosixNotesSyncFilesystem(root) as filesystem:
        snapshot = filesystem.observe("note.md")
        assert (
            filesystem.serialize(
                snapshot.text,
                snapshot.observation.serialization,
            )
            == payload
        )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"not utf8: \xff", "unsupported_encoding"),
        (b"alpha\r\nbeta\ngamma", "mixed_newlines"),
        (b"alpha\rbeta", "unsupported_newline"),
    ],
)
def test_observe_rejects_lossy_representation(
    tmp_path: Path,
    payload: bytes,
    reason: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_bytes(payload)

    with PosixNotesSyncFilesystem(root) as filesystem:
        with pytest.raises(NotesSyncFilesystemError, match=reason):
            filesystem.observe("note.md")


def test_observe_rejects_root_symlink_directory_symlink_and_hard_link(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    selected = tmp_path / "selected"
    selected.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(NotesSyncFilesystemError, match="root_link_or_reparse"):
        PosixNotesSyncFilesystem(selected)

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_bytes(b"outside")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    os.link(outside / "note.md", root / "hard.md")
    with PosixNotesSyncFilesystem(root) as filesystem:
        with pytest.raises(NotesSyncFilesystemError, match="link_or_reparse"):
            filesystem.observe("linked/note.md")
        with pytest.raises(NotesSyncFilesystemError, match="multiple_links"):
            filesystem.observe("hard.md")


def test_guarded_replace_preserves_profile_and_recovery_bytes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.md"
    target.write_bytes(b"before\r\n")
    target.chmod(0o640)

    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("note.md")
        after = filesystem.replace(
            "note.md",
            "after\nline",
            expected=before,
        )

    assert after.recovery_bytes == b"before\r\n"
    assert target.read_bytes() == b"after\r\nline\r\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_serialize_normalizes_note_newlines_before_applying_file_profile(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_bytes(b"before\r\n")

    with PosixNotesSyncFilesystem(root) as filesystem:
        snapshot = filesystem.observe("note.md")
        assert (
            filesystem.serialize(
                "one\r\ntwo\rthree\n",
                snapshot.observation.serialization,
            )
            == b"one\r\ntwo\r\nthree\r\n"
        )


def test_unsupported_metadata_withholds_writable_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_bytes(b"body")

    with PosixNotesSyncFilesystem(root) as filesystem:
        monkeypatch.setattr(
            filesystem,
            "_metadata_issue",
            lambda _snapshot: "unsupported_metadata",
        )
        with pytest.raises(NotesSyncFilesystemError, match="unsupported_metadata"):
            filesystem.observe("note.md", require_writable=True)


def test_guarded_replace_preserves_extended_attributes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.md"
    target.write_bytes(b"before")
    attribute = "user.notes_sync_test"
    if hasattr(os, "setxattr"):
        os.setxattr(target, attribute, b"kept")

        def read_attribute() -> bytes:
            return os.getxattr(target, attribute)

    elif shutil.which("xattr"):
        attribute = "notes_sync_test"
        subprocess.run(
            ["xattr", "-w", attribute, "kept", os.fspath(target)],
            check=True,
            capture_output=True,
        )

        def read_attribute() -> bytes:
            return subprocess.run(
                ["xattr", "-p", attribute, os.fspath(target)],
                check=True,
                capture_output=True,
            ).stdout.rstrip(b"\n")

    else:
        pytest.skip("descriptor xattr fixture unavailable")

    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("note.md")
        filesystem.replace("note.md", "after", expected=before)

    assert target.read_bytes() == b"after"
    assert read_attribute() == b"kept"


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin extended ACL contract")
def test_nontrivial_darwin_acl_withholds_writable_admission(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.md"
    target.write_bytes(b"body")
    result = subprocess.run(
        ["chmod", "+a", "everyone deny write", os.fspath(target)],
        capture_output=True,
    )
    if result.returncode:
        pytest.skip("extended ACL fixture unavailable")

    with PosixNotesSyncFilesystem(root) as filesystem:
        with pytest.raises(NotesSyncFilesystemError, match="unsupported_metadata"):
            filesystem.observe("note.md", require_writable=True)


def test_public_error_suppresses_private_path_bearing_cause(
    tmp_path: Path,
) -> None:
    root = tmp_path / "PRIVATE-root"
    root.mkdir()

    with PosixNotesSyncFilesystem(root) as filesystem:
        with pytest.raises(NotesSyncFilesystemError) as raised:
            filesystem.observe("PRIVATE-missing.md")

    assert raised.value.__cause__ is None
    assert "PRIVATE" not in str(raised.value)


def test_observe_maps_raw_descriptor_read_error_to_bounded_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "PRIVATE-root"
    root.mkdir()
    (root / "note.md").write_bytes(b"before")

    with PosixNotesSyncFilesystem(root) as filesystem:
        monkeypatch.setattr(
            sync_paths.os,
            "read",
            lambda *_args: (_ for _ in ()).throw(OSError("PRIVATE descriptor path")),
        )
        with pytest.raises(NotesSyncFilesystemError) as raised:
            filesystem.observe("note.md")

    assert raised.value.reason_code == "operation_failed"
    assert raised.value.__cause__ is None
    assert "PRIVATE" not in str(raised.value)


def test_metadata_copy_error_is_bounded_private_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.md"
    target.write_bytes(b"before")

    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("note.md")
        monkeypatch.setattr(
            sync_paths,
            "_write_extended_attributes",
            lambda *_args: (_ for _ in ()).throw(OSError("PRIVATE metadata path")),
        )
        with pytest.raises(NotesSyncFilesystemError) as raised:
            filesystem.replace("note.md", "after", expected=before)

    assert raised.value.reason_code == "unsupported_metadata"
    assert raised.value.__cause__ is None
    assert "PRIVATE" not in str(raised.value)
    assert target.read_bytes() == b"before"
    assert list(root.glob(".note.md.tmp-*")) == []


def test_read_only_snapshot_cannot_authorize_replace_or_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "before.md"
    source.write_bytes(b"before")

    with PosixNotesSyncFilesystem(root) as filesystem:
        observed = filesystem.observe("before.md", require_writable=False)
        monkeypatch.setattr(
            filesystem,
            "_metadata_issue",
            lambda _snapshot: "unsupported_metadata",
        )
        with pytest.raises(NotesSyncFilesystemError, match="unsupported_metadata"):
            filesystem.replace("before.md", "after", expected=observed)
        with pytest.raises(NotesSyncFilesystemError, match="unsupported_metadata"):
            filesystem.move("after.md", expected=observed)

    assert source.read_bytes() == b"before"
    assert not (root / "after.md").exists()


def test_high_level_replace_rejects_divergent_post_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_bytes(b"before")

    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("note.md")
        real_observe = filesystem.observe
        calls = 0

        def divergent(path, *, require_writable=True):
            nonlocal calls
            calls += 1
            observed = real_observe(path, require_writable=require_writable)
            return replace(observed, raw_bytes=b"racer") if calls == 1 else observed

        monkeypatch.setattr(filesystem, "observe", divergent)
        with pytest.raises(
            NotesSyncFilesystemPartialError,
            match="replacement_postcondition_failed",
        ):
            filesystem.replace("note.md", "after", expected=before)


def test_high_level_move_rejects_divergent_post_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "before.md").write_bytes(b"before")

    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("before.md")
        real_observe = filesystem.observe
        monkeypatch.setattr(
            filesystem,
            "observe",
            lambda path, *, require_writable=True: replace(
                real_observe(path, require_writable=require_writable),
                raw_bytes=b"racer",
            ),
        )
        with pytest.raises(
            NotesSyncFilesystemPartialError,
            match="move_postcondition_failed",
        ):
            filesystem.move("after.md", expected=before)


def test_committed_replace_observation_failure_is_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.md"
    target.write_bytes(b"before")

    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("note.md")
        monkeypatch.setattr(
            filesystem,
            "observe",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                NotesSyncFilesystemError("PRIVATE post-commit failure")
            ),
        )
        with pytest.raises(NotesSyncFilesystemPartialError) as raised:
            filesystem.replace("note.md", "after", expected=before)

    assert raised.value.reason_code == "replacement_postcondition_failed"
    assert raised.value.__cause__ is None
    assert target.read_bytes() == b"after"


def test_committed_move_observation_failure_is_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "before.md"
    source.write_bytes(b"before")

    with PosixNotesSyncFilesystem(root) as filesystem:
        before = filesystem.observe("before.md")
        monkeypatch.setattr(
            filesystem,
            "observe",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                NotesSyncFilesystemError("PRIVATE post-commit failure")
            ),
        )
        with pytest.raises(NotesSyncFilesystemPartialError) as raised:
            filesystem.move("after.md", expected=before)

    assert raised.value.reason_code == "move_postcondition_failed"
    assert raised.value.__cause__ is None
    assert not source.exists()
    assert (root / "after.md").read_bytes() == b"before"


def test_observation_rejects_file_before_accumulating_past_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_bytes(b"12345")

    with PosixNotesSyncFilesystem(root, max_file_bytes=4) as filesystem:
        with pytest.raises(NotesSyncFilesystemError, match="max_file_bytes_exceeded"):
            filesystem.observe("note.md")


def test_observation_rejects_unbounded_extended_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "note.md").write_bytes(b"body")
    monkeypatch.setattr(
        sync_paths,
        "_read_extended_attributes",
        lambda _descriptor: (_ for _ in ()).throw(OSError("too large")),
    )

    with PosixNotesSyncFilesystem(root) as filesystem:
        with pytest.raises(NotesSyncFilesystemError, match="unsupported_metadata"):
            filesystem.observe("note.md")


def test_root_admission_rejects_overlap_alias_and_private_ownership(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "notes"
    nested = candidate / "nested"
    private = tmp_path / "private"
    for path in (nested, private, private / "child"):
        path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(NotesSyncFilesystemError, match="root_overlap"):
        validate_sync_root_admission(candidate, sync_roots=(nested,))
    with pytest.raises(NotesSyncFilesystemError, match="file_notes_overlap"):
        validate_sync_root_admission(candidate, file_notes_roots=(candidate,))
    with pytest.raises(NotesSyncFilesystemError, match="private_path_overlap"):
        validate_sync_root_admission(private / "child", private_paths=(private,))


def test_writable_capability_is_not_advertised_on_windows() -> None:
    assert PosixNotesSyncFilesystem.supports_writes(platform="win32") is False
    assert PosixNotesSyncFilesystem.supports_writes(platform="freebsd13") is False


def test_windows_observation_reuses_native_read_only_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "note.md"
    source.write_bytes(b"body\r\n")
    structured = root / "structured.json"
    structured.write_bytes(b"{}")
    identity = SourceIdentity(
        device=1,
        inode=2,
        mode=stat.S_IFREG | 0o640,
        size=6,
        modified_ns=3,
        changed_ns=4,
    )
    candidate = DiscoveredImportSource(
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="root/note.md",
            source_path=source,
        ),
        size_bytes=6,
        identity=identity,
        parent_identities=(identity,),
    )
    structured_candidate = DiscoveredImportSource(
        source=ImportSource(
            kind=ImportSourceKind.DIRECTORY_MEMBER,
            display_path="root/structured.json",
            source_path=structured,
        ),
        size_bytes=6,
        identity=identity,
        parent_identities=(identity,),
    )
    discovery = ImportDiscovery(
        candidates=(candidate, structured_candidate),
        failures=(),
        root_label="root",
        total_bytes=6,
        entry_count=1,
    )
    calls: list[tuple[str, object]] = []

    class ReadOnlyNative:
        @staticmethod
        def absolute(path: Path) -> Path:
            return path.absolute()

    native = ReadOnlyNative()

    def fake_discover(paths, bounds, *, filesystem):
        calls.append(("discover", filesystem))
        assert tuple(paths) == (root,)
        return discovery

    def fake_read(item, bounds, *, filesystem):
        calls.append(("read", filesystem))
        assert item is candidate
        return b"body\r\n"

    monkeypatch.setattr(
        "tldw_chatbook.Notes.notes_sync_filesystem.discover_import_sources",
        fake_discover,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Notes.notes_sync_filesystem.read_discovered_source",
        fake_read,
    )
    bounds = ImportBounds(
        max_files=10,
        max_file_bytes=1000,
        max_total_bytes=5000,
        max_depth=4,
    )
    filesystem = WindowsNotesSyncObservationFilesystem(
        root,
        bounds=bounds,
        filesystem=native,
    )

    snapshots = filesystem.observe()

    assert filesystem.supports_writes() is False
    assert snapshots[0].text == "body\n"
    assert len(snapshots[0].stable_identity_digest) == 64
    assert calls == [("discover", native), ("read", native)]


def test_windows_stable_identity_ignores_mutable_file_metadata() -> None:
    before = SourceIdentity(1, 2, stat.S_IFREG | 0o640, 6, 3, 4)
    edited = SourceIdentity(1, 2, stat.S_IFREG | 0o600, 60, 30, 40)

    assert notes_sync_filesystem._windows_stable_identity_digest(
        before
    ) == notes_sync_filesystem._windows_stable_identity_digest(edited)


def test_windows_relative_root_and_duplicate_identity_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical"
    identity = SourceIdentity(1, 2, stat.S_IFREG | 0o640, 4, 3, 4)

    def candidate(name: str) -> DiscoveredImportSource:
        return DiscoveredImportSource(
            source=ImportSource(
                kind=ImportSourceKind.DIRECTORY_MEMBER,
                display_path=f"canonical/{name}",
                source_path=canonical / name,
            ),
            size_bytes=4,
            identity=identity,
            parent_identities=(identity,),
        )

    discovery = ImportDiscovery(
        candidates=(candidate("first.md"), candidate("second.md")),
        failures=(),
        root_label="canonical",
        total_bytes=8,
        entry_count=2,
    )

    class ReadOnlyNative:
        @staticmethod
        def absolute(path: Path) -> Path:
            assert path == Path("relative-root")
            return canonical

    native = ReadOnlyNative()

    def fake_discover(paths, bounds, *, filesystem):
        assert tuple(paths) == (canonical,)
        assert filesystem is native
        return discovery

    monkeypatch.setattr(notes_sync_filesystem, "discover_import_sources", fake_discover)
    monkeypatch.setattr(
        notes_sync_filesystem,
        "read_discovered_source",
        lambda *_args, **_kwargs: b"body",
    )
    bounds = ImportBounds(10, 1000, 5000, 4)
    filesystem = WindowsNotesSyncObservationFilesystem(
        Path("relative-root"),
        bounds=bounds,
        filesystem=native,
    )

    with pytest.raises(NotesSyncFilesystemError, match="duplicate_stable_identity"):
        filesystem.observe()
