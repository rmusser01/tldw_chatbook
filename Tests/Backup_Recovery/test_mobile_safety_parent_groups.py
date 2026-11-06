"""Local R143 mechanics ONLY: upstream evidence is mocked, never authority proof."""

import hashlib
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, journal
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.archive_models import SealedArchive
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit
from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import _CanonicalCopy
from tldw_chatbook.Backup_Recovery.models import FileMetadata, StorageItem
from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


@contextmanager
def local_members(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import (
        later_rollback,
        mobile_canonical_copy,
        publication,
    )
    from tldw_chatbook.Backup_Recovery import mobile_builtin_preservation as module

    root = tmp_path / "package"
    group = root / "expressions"
    root.mkdir(mode=0o700)
    group.mkdir(mode=0o700)
    files = [group / name for name in ("one.webp", "two-😀.webp", "three.webp")]
    for index, path in enumerate(files):
        path.write_bytes(bytes([index + 1]) * (index + 2))
        path.chmod(0o600)
    authority_root = tmp_path / "bootstrap"
    authority = admission_authority(authority_root)
    authority.register("unit.current-source", (root,))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: authority_root)
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    original = journal.Journal(control, "original-negative-prefix")
    current = journal.Journal(control, "new-negative-prefix")
    cipher = control / "cipher"
    cipher.write_bytes(b"unit-only encrypted placeholder")
    cipher.chmod(0o600)
    manifest = b"unit-only manifest placeholder"
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    cipher_object = journal._Object.model_validate(journal.observe_artifact(cipher))
    proof = _CanonicalCopy(
        SimpleNamespace(archive=cipher_object, manifest_digest=manifest_digest),
        "source-digest",
        False,
        None,
    )
    archive = SealedArchive(control / "unread.zip", "archive-digest", manifest)
    receipt = SimpleNamespace(
        archive_digest=archive.digest, manifest_digest=manifest_digest
    )
    members, records, observations = {}, {}, []
    owner = "persona.visual_identity_builtin"
    for index, path in enumerate((root, group, *files)):
        info = path.stat()
        key = "root" if index == 0 else "group" if index == 1 else f"file{index}"
        parent = None if index == 0 else "root" if index == 1 else "group"
        kind = "directory" if path.is_dir() else "file"
        metadata = FileMetadata(
            1,
            "root",
            path.relative_to(root).as_posix() if index else "",
            parent,
            kind,
            info.st_mode & 0o777,
            info.st_mtime_ns,
            "private",
        )
        members[key] = StorageItem(
            owner,
            key,
            path,
            "included_directory" if kind == "directory" else "included",
            (),
            metadata=metadata,
        )
        records[key] = SimpleNamespace(
            metadata=metadata,
            size=info.st_size if kind == "file" else 0,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            if kind == "file"
            else None,
        )
        observations.append(
            SimpleNamespace(
                logical_id=key,
                owner_id=owner,
                source=journal._Object.model_validate(
                    publication._observe_safety_source(path)
                ),
            )
        )
    plan = SimpleNamespace(
        local_snapshot=LocalSnapshotSource(
            control, original.operation_id, "source-digest"
        ),
        archive_digest=archive.digest,
        restore=(),
        retire=(),
        preserve=tuple((key, item.path) for key, item in members.items()),
        safety_scope=tuple(members),
    )
    prepared = SimpleNamespace(
        safety_sources=tuple(observations),
        publication=SimpleNamespace(namespaces=("unit.current-source",)),
    )
    # Explicitly isolated upstream source/manifest/owner matching, not a positive
    # committed-source test. Real journal identities, session, namespace, parent
    # FDs, leaf stats/frames and custody below are unchanged native mechanisms.
    monkeypatch.setattr(
        later_rollback,
        "_accounted_snapshot_evidence",
        lambda *args, **kwargs: (proof, "source-digest", "mobile_committed"),
    )
    monkeypatch.setattr(journal, "_matches", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        mobile_canonical_copy,
        "_verify_canonical_archive",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "_mobile_source_members",
        lambda *args, **kwargs: (
            members,
            records,
            [],
            bootstrap._registry(authority_root),
            kwargs["_read_budget"],
        ),
    )
    with (
        authority.maintenance(
            ("unit.current-source", "bootstrap.unbound"), 2
        ) as session,
        current._locked(exclusive=True) as parent,
    ):
        yield SimpleNamespace(
            module=module,
            plan=plan,
            journal=current,
            parent=parent,
            prepared=prepared,
            receipt=receipt,
            session=session,
            archive=archive,
            original=original,
            root=root,
            group=group,
            files=files,
            members=members,
            records=records,
        )


@pytest.mark.parametrize(
    "mode",
    [
        "reuse",
        "replace",
        "symlink",
        "wrong_frame",
        "cancel_group",
        "cancel_reopen",
        "zero",
        "group_limit",
        "pin_limit",
        "reopen_limit",
        "close_error",
    ],
)
def test_local_parent_mechanics(tmp_path, monkeypatch, mode):
    from tldw_chatbook.Backup_Recovery import admission, archive_reader
    from tldw_chatbook.Utils import platform_files

    native_os = storage.os
    native_pin = storage._CaptureFileDescriptors.pinned_directory
    native_price = admission._price_paths
    pins, active, leaf_stats, closed, buckets = [], set(), [], [], []
    cancel = Event()
    budget = _TraceReadBudget()
    hold_count = len(storage._failed_capture_holds)
    state = {"last_leaf": False, "injected": False}
    expected = None
    if mode in {"replace", "wrong_frame"}:
        expected = (ValueError, "mobile_safety_source_changed")
    elif mode == "symlink":
        expected = (OSError, None)
    elif mode.startswith("cancel"):
        expected = (InterruptedError, None)
    elif mode in {"zero", "group_limit", "pin_limit", "reopen_limit"}:
        expected = (_TraceReadLimit, "mobile_trace_input_limit")
    elif mode == "close_error":
        expected = (admission.AdmissionError, "capture_resources_not_retired")

    @contextmanager
    def outcome():
        if expected is None:
            yield
        else:
            with pytest.raises(expected[0], match=expected[1]):
                yield

    with outcome(), local_members(tmp_path, monkeypatch) as case:
        native_check = archive_reader._check

        @contextmanager
        def pin(custody, path):
            with native_pin(custody, path) as fd:
                if Path(path) == case.group:
                    if pins:
                        # The second current-name descriptor opens while the
                        # original one remains live, not after inode release.
                        assert active
                    pins.append(fd)
                    active.add(fd)
                try:
                    yield fd
                finally:
                    if Path(path) == case.group:
                        active.remove(fd)

        def check(event):
            native_check(event)
            if mode == "reopen_limit" and state["last_leaf"] and not state["injected"]:
                state["injected"] = True
                budget.reserve(budget._remaining)

        def price(paths, allowance, **kwargs):
            result = native_price(paths, allowance, **kwargs)
            if (
                mode == "pin_limit"
                and paths == (case.root.parent,)
                and not state["injected"]
            ):
                state["injected"] = True
                budget.reserve(budget._remaining)
            return result

        native_contains = storage._contains_owned_path

        def contains(*args, **kwargs):
            result = native_contains(*args, **kwargs)
            if (
                result
                and not state["injected"]
                and mode in {"cancel_group", "group_limit"}
            ):
                state["injected"] = True
                if mode == "cancel_group":
                    cancel.set()
                else:
                    budget.reserve(budget._remaining)
            return result

        class Calls:
            def __getattr__(self, name):
                return getattr(native_os, name)

            def stat(self, path, *args, **kwargs):
                info = native_os.stat(path, *args, **kwargs)
                if kwargs.get("dir_fd") in active and path in {
                    p.name for p in case.files
                }:
                    leaf_stats.append(path)
                    if path == case.files[-1].name:
                        state["last_leaf"] = True
                        if mode in {"replace", "symlink"}:
                            retained = case.root / "retained-original"
                            case.group.rename(retained)
                            if mode == "replace":
                                case.group.mkdir(mode=0o700)
                                (case.group / "foreign").write_bytes(
                                    b"retain foreign payload"
                                )
                            else:
                                case.group.symlink_to(
                                    retained, target_is_directory=True
                                )
                        elif mode == "cancel_reopen":
                            cancel.set()
                return info

            def close(self, fd):
                native_os.close(fd)
                if mode == "close_error" and fd in pins:
                    closed.append(fd)
                    raise OSError("actual parent closed, ambiguous callback failure")

        native_retire = storage._CaptureFileDescriptors.retire

        def retire(bucket):
            if bucket in case.session._scopes and bucket not in buckets:
                buckets.append(bucket)
            return native_retire(bucket)

        monkeypatch.setattr(storage._CaptureFileDescriptors, "pinned_directory", pin)
        monkeypatch.setattr(storage._CaptureFileDescriptors, "retire", retire)
        monkeypatch.setattr(archive_reader, "_check", check)
        monkeypatch.setattr(admission, "_price_paths", price)
        monkeypatch.setattr(storage, "_contains_owned_path", contains)
        monkeypatch.setattr(platform_files, "os", Calls())
        monkeypatch.setattr(storage, "os", Calls())
        if mode == "wrong_frame":
            case.records["file2"].sha256 = "0" * 64
        if mode == "zero":
            budget.reserve(budget._remaining)
        actual = case.module._mobile_safety_members(
            case.plan,
            case.journal,
            case.parent,
            case.prepared,
            case.receipt,
            case.session,
            case.archive,
            case.original,
            budget,
            cancel,
        )
        assert actual == case.members
    assert case is not None and not case.session._active
    if mode == "reuse":
        assert len(pins) == 2 and leaf_stats == [p.name for p in case.files]
        assert budget._remaining < 16 * 1024**2
    if mode in {"replace", "symlink"}:
        assert (
            case.root / "retained-original" / case.files[0].name
        ).read_bytes() == b"\x01" * 2
        if mode == "replace":
            assert (case.group / "foreign").read_bytes() == b"retain foreign payload"
    if mode in {"cancel_reopen", "reopen_limit"}:
        assert len(pins) == 1
    if mode in {"zero", "group_limit", "pin_limit", "cancel_group"}:
        assert pins == []
    if mode == "close_error":
        assert len(closed) == 1
        with pytest.raises(OSError):
            native_os.fstat(closed[0])
        assert len(storage._failed_capture_holds) == hold_count + 1
        assert storage._failed_capture_holds[-1][2] is case.session
        assert any(b.close_failed and not b.fds for b in case.session._scopes)
    else:
        assert len(storage._failed_capture_holds) == hold_count
        assert all(
            not b.fds and not b.directory_iterators and not b.close_failed
            for b in buckets
        )
        assert all(b not in case.session._scopes for b in buckets)
