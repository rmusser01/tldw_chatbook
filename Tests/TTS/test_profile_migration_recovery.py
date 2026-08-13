from __future__ import annotations

import asyncio
import copy
import ctypes
import errno
import importlib
import importlib.util
import io
import json
import os
import pickle
import sqlite3
import stat
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from zlib import crc32

import pytest

from tldw_chatbook.TTS import profile_schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


def test_namespace_import_on_windows_does_not_probe_posix_libc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows imports safely while descriptor rename remains unsupported."""
    module_name = "_test_windows_profile_migration_namespace"
    module_path = Path(profile_schema.__file__).with_name(
        "profile_migration_namespace.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    def forbid_posix_libc(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Windows namespace import must not load POSIX libc")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "CDLL", forbid_posix_libc)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        with pytest.raises(OSError) as caught:
            module.rename_noreplace_at(-1, "source", "destination")
        assert caught.value.errno == errno.ENOTSUP
    finally:
        sys.modules.pop(module_name, None)


def _modules() -> tuple[ModuleType, ModuleType]:
    publication = importlib.import_module(
        "tldw_chatbook.TTS.profile_migration_publication"
    )
    try:
        recovery = importlib.import_module(
            "tldw_chatbook.TTS.profile_migration_recovery"
        )
    except ModuleNotFoundError:
        pytest.fail("profile migration recovery module is missing")
    return publication, recovery


def _store(path: Path, *, version: int, marker: str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    connection = sqlite3.connect(path)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        for source_version in range(version):
            profile_schema.MIGRATIONS[source_version](connection)
        connection.execute(
            f"PRAGMA application_id = {crc32(marker.encode('utf-8')) & 0x7FFF_FFFF}"
        )
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)
    return path.read_bytes()


def _publication_fixture(
    tmp_path: Path,
    *,
    slots: tuple[str, ...] = ("active", "pre_v4"),
) -> tuple[ModuleType, Path, tuple[object, ...], tuple[object, ...]]:
    publication, _ = _modules()
    slot_type = publication.ProfileMigrationPublicationSlot
    slot_values = {
        "active": (slot_type.ACTIVE, 4, 3, "profiles.sqlite3"),
        "pre_v3": (slot_type.PRE_V3, 2, 2, "profiles.pre-v3.sqlite3"),
        "pre_v4": (slot_type.PRE_V4, 3, 3, "profiles.pre-v4.sqlite3"),
    }
    artifacts = []
    destinations = []
    active_path: Path | None = None
    for label in slots:
        slot, candidate_version, prior_version, target_name = slot_values[label]
        target = tmp_path / target_name
        candidate = tmp_path / publication.PROFILE_MIGRATION_CANDIDATE_LEAVES[slot]
        _store(target, version=prior_version, marker=f"old-{label}")
        _store(candidate, version=candidate_version, marker=f"new-{label}")
        artifacts.append(
            publication.prepare_profile_migration_artifact(candidate, slot=slot)
        )
        destinations.append(
            publication.retain_profile_migration_destination(
                target,
                slot=slot,
                must_exist=True,
            )
        )
        if label == "active":
            active_path = target
    assert active_path is not None
    return publication, active_path, tuple(artifacts), tuple(destinations)


def _write_journal(
    publication: ModuleType,
    active_path: Path,
    artifacts: tuple[object, ...],
    destinations: tuple[object, ...],
    *,
    phase: str,
) -> Path:
    authority = publication._journal_authority(artifacts, destinations)
    initial, checksum = publication._encode_initial_journal(authority)
    transitions = {
        "prepared": (),
        "publishing": ("publishing",),
        "restoring": ("publishing", "restoring"),
        "unavailable": ("publishing", "restoring", "unavailable"),
        "complete": ("publishing", "complete"),
    }
    raw = initial + b"".join(
        publication._encode_later_journal(checksum, phase=item)
        for item in transitions[phase]
    )
    journal = active_path.with_name(f".{active_path.name}.migration-publication.json")
    journal.write_bytes(raw)
    journal.chmod(0o600)
    return journal


def _link_then_unlink(source: Path, destination: Path, *, finish: bool) -> None:
    os.link(source, destination, follow_symlinks=False)
    if finish:
        source.unlink()


def _paths(artifact: object, destination: object) -> tuple[Path, Path, Path]:
    publication, _recovery = _modules()
    candidate = artifact._path
    target = destination._path
    rollback = target.with_name(
        publication.PROFILE_MIGRATION_ROLLBACK_LEAVES[artifact._slot]
    )
    return candidate, target, rollback


def test_no_journal_is_an_idempotent_noop(tmp_path: Path) -> None:
    _, recovery = _modules()
    active = tmp_path / "profiles.sqlite3"
    _store(active, version=4, marker="active")

    assert recovery.recover_profile_migration_publication(active) is False
    assert recovery.recover_profile_migration_publication(active) is False


def test_prepared_journal_restores_prior_and_cleans_owned_candidates(
    tmp_path: Path,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(tmp_path)
    before = tuple(item._path.read_bytes() for item in destinations)
    candidates = tuple(item._path for item in artifacts)
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="prepared"
    )

    assert recovery.recover_profile_migration_publication(active) is True
    assert tuple(item._path.read_bytes() for item in destinations) == before
    assert all(not path.exists() for path in candidates)
    assert not journal.exists()
    assert recovery.recover_profile_migration_publication(active) is False


@pytest.mark.parametrize("finish_link", [False, True])
def test_publishing_hardlink_half_move_rolls_back_exactly(
    tmp_path: Path,
    finish_link: bool,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    old = target.read_bytes()
    _write_journal(publication, active, artifacts, destinations, phase="publishing")

    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=finish_link)

    assert recovery.recover_profile_migration_publication(active) is True
    assert target.read_bytes() == old
    assert not candidate.exists()
    assert not rollback.exists()


def test_complete_journal_finishes_cleanup_and_keeps_new_authority(
    tmp_path: Path,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(tmp_path)
    expected = tuple(item._path.read_bytes() for item in artifacts)
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="complete"
    )
    for artifact, destination in zip(artifacts, destinations, strict=True):
        candidate, target, rollback = _paths(artifact, destination)
        _link_then_unlink(target, rollback, finish=True)
        _link_then_unlink(candidate, target, finish=True)

    assert recovery.recover_profile_migration_publication(active) is True
    assert tuple(item._path.read_bytes() for item in destinations) == expected
    assert not journal.exists()
    assert not tuple(tmp_path.glob("*.rollback"))


def test_publishing_with_only_completed_authority_finishes_instead_of_guessing(
    tmp_path: Path,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    expected = candidate.read_bytes()
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=True)
    rollback.unlink()

    assert recovery.recover_profile_migration_publication(active) is True
    assert target.read_bytes() == expected
    assert not journal.exists()


def test_foreign_substitution_is_preserved_and_fails_closed(tmp_path: Path) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, _ = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    retained = tmp_path / "retained-candidate.sqlite3"
    candidate.rename(retained)
    foreign = _store(candidate, version=4, marker="foreign")
    old = target.read_bytes()

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        recovery.recover_profile_migration_publication(active)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert candidate.read_bytes() == foreign
    assert retained.is_file()
    assert target.read_bytes() == old
    assert journal.is_file()


def test_unrecognized_hardlink_is_preserved_and_fails_closed(tmp_path: Path) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, _, _ = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    foreign_link = tmp_path / "foreign-link.sqlite3"
    os.link(candidate, foreign_link, follow_symlinks=False)

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        recovery.recover_profile_migration_publication(active)

    assert candidate.is_file()
    assert foreign_link.is_file()
    assert journal.is_file()


def test_malformed_journal_is_bounded_and_does_not_touch_active(tmp_path: Path) -> None:
    _, recovery = _modules()
    active = tmp_path / "PRIVATE profiles.sqlite3"
    before = _store(active, version=3, marker="old")
    journal = active.with_name(f".{active.name}.migration-publication.json")
    journal.write_bytes(b"PRIVATE malformed journal\n")
    journal.chmod(0o600)

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        recovery.recover_profile_migration_publication(active)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(tmp_path) not in repr(caught.value)
    assert active.read_bytes() == before
    assert journal.read_bytes() == b"PRIVATE malformed journal\n"


@pytest.mark.parametrize(
    "cancel_stage", ["admitted", "repaired", "validated", "settled"]
)
def test_post_admission_cancellation_is_deferred_until_authority_is_safe(
    tmp_path: Path,
    cancel_stage: str,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    old = target.read_bytes()
    _write_journal(publication, active, artifacts, destinations, phase="publishing")
    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=True)
    cancellation = asyncio.CancelledError("PRIVATE cancellation")
    raised = False

    def cancel_after_admission(stage: str) -> None:
        nonlocal raised
        if stage == cancel_stage and not raised:
            raised = True
            raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        recovery.recover_profile_migration_publication(
            active,
            _stage_hook=cancel_after_admission,
        )

    assert caught.value is cancellation
    assert target.read_bytes() == old
    assert not candidate.exists()
    assert not rollback.exists()
    assert not tuple(tmp_path.glob("*.migration-publication.json"))


def test_total_repair_failure_is_unavailable_and_retains_recovery_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=True)

    def fail_move(*_args: object, **_kwargs: object) -> None:
        raise OSError("PRIVATE total storage failure")

    monkeypatch.setattr(recovery, "_move_exact", fail_move)

    with pytest.raises(ProfileRepositoryError, match="unavailable") as caught:
        recovery.recover_profile_migration_publication(active)

    assert caught.value.code == "unavailable"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "PRIVATE" not in repr(caught.value)
    assert journal.is_file()
    assert target.is_file()
    assert rollback.is_file()


def test_recovered_files_are_private_single_link_without_sidecars(
    tmp_path: Path,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(tmp_path)
    _write_journal(publication, active, artifacts, destinations, phase="complete")
    for artifact, destination in zip(artifacts, destinations, strict=True):
        candidate, target, rollback = _paths(artifact, destination)
        _link_then_unlink(target, rollback, finish=True)
        _link_then_unlink(candidate, target, finish=True)

    recovery.recover_profile_migration_publication(active)

    for destination in destinations:
        info = destination._path.stat()
        assert stat.S_ISREG(info.st_mode)
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_uid == os.geteuid()
        assert info.st_nlink == 1
        assert not any(
            destination._path.with_name(destination._path.name + suffix).exists()
            for suffix in ("-wal", "-shm", "-journal")
        )


@pytest.mark.parametrize("phase", ["restoring", "unavailable"])
def test_restoration_phases_converge_to_prior_authority(
    tmp_path: Path,
    phase: str,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(tmp_path)
    before = tuple(item._path.read_bytes() for item in destinations)
    _write_journal(publication, active, artifacts, destinations, phase=phase)
    first_candidate, first_target, first_rollback = _paths(
        artifacts[0], destinations[0]
    )
    _link_then_unlink(first_target, first_rollback, finish=True)
    _link_then_unlink(first_candidate, first_target, finish=True)

    assert recovery.recover_profile_migration_publication(active) is True
    assert tuple(item._path.read_bytes() for item in destinations) == before
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob("*.rollback"))


@pytest.mark.parametrize("phase", ["prepared", "restoring", "unavailable"])
def test_noncomplete_phase_never_infers_completion_from_missing_prior(
    tmp_path: Path,
    phase: str,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(publication, active, artifacts, destinations, phase=phase)
    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=True)
    rollback.unlink()

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        recovery.recover_profile_migration_publication(active)

    assert target.is_file()
    assert journal.is_file()


@pytest.mark.parametrize("leaf_kind", ["journal", "target", "rollback"])
def test_sidecar_or_nonregular_substitution_is_preserved_and_rejected(
    tmp_path: Path,
    leaf_kind: str,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    selected = {"journal": journal, "target": target, "rollback": rollback}[leaf_kind]
    if leaf_kind == "rollback":
        selected.write_bytes(b"foreign")
        selected.chmod(0o600)
    else:
        selected.with_name(selected.name + "-wal").write_bytes(b"foreign")

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        recovery.recover_profile_migration_publication(active)

    assert selected.exists()
    assert candidate.exists()
    assert journal.exists()


def test_parent_replacement_and_active_name_mismatch_fail_before_admission(
    tmp_path: Path,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    alternate = tmp_path / "different.sqlite3"
    alternate_journal = alternate.with_name(
        f".{alternate.name}.migration-publication.json"
    )
    alternate_journal.write_bytes(journal.read_bytes())
    alternate_journal.chmod(0o600)

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        recovery.recover_profile_migration_publication(alternate)

    assert journal.exists()
    assert alternate_journal.exists()
    assert active.exists()


def test_recovery_api_returns_only_bool_and_errors_are_serialization_safe(
    tmp_path: Path,
) -> None:
    _, recovery = _modules()
    active = tmp_path / "PRIVATE profiles.sqlite3"
    _store(active, version=3, marker="old")
    journal = active.with_name(f".{active.name}.migration-publication.json")
    journal.write_bytes(b"PRIVATE malformed\n")
    journal.chmod(0o600)

    with pytest.raises(ProfileRepositoryError) as caught:
        recovery.recover_profile_migration_publication(active)

    error = caught.value
    for operation in (copy.copy, copy.deepcopy):
        reproduced = operation(error)
        assert str(tmp_path) not in repr(reproduced)
        assert "PRIVATE" not in repr(reproduced)
    sink = io.BytesIO()
    pickle.Pickler(sink, protocol=5).dump(error)
    assert str(tmp_path).encode() not in sink.getvalue()
    assert b"PRIVATE" not in sink.getvalue()


@pytest.mark.parametrize("phase", ["publishing", "complete"])
def test_fresh_backup_destination_recovers_without_inventing_prior(
    tmp_path: Path,
    phase: str,
) -> None:
    publication, recovery = _modules()
    slot = publication.ProfileMigrationPublicationSlot
    active = tmp_path / "profiles.sqlite3"
    active_candidate = (
        tmp_path / publication.PROFILE_MIGRATION_CANDIDATE_LEAVES[slot.ACTIVE]
    )
    backup = tmp_path / "profiles.pre-v4.sqlite3"
    backup_candidate = (
        tmp_path / publication.PROFILE_MIGRATION_CANDIDATE_LEAVES[slot.PRE_V4]
    )
    old_active = _store(active, version=3, marker="old-active")
    expected_active = _store(active_candidate, version=4, marker="new-active")
    expected_backup = _store(backup_candidate, version=3, marker="new-backup")
    artifacts = (
        publication.prepare_profile_migration_artifact(
            active_candidate, slot=slot.ACTIVE
        ),
        publication.prepare_profile_migration_artifact(
            backup_candidate, slot=slot.PRE_V4
        ),
    )
    destinations = (
        publication.retain_profile_migration_destination(
            active, slot=slot.ACTIVE, must_exist=True
        ),
        publication.retain_profile_migration_destination(
            backup, slot=slot.PRE_V4, must_exist=False
        ),
    )
    _write_journal(publication, active, artifacts, destinations, phase=phase)
    if phase == "complete":
        active_rollback = active.with_name(
            publication.PROFILE_MIGRATION_ROLLBACK_LEAVES[slot.ACTIVE]
        )
        _link_then_unlink(active, active_rollback, finish=True)
        _link_then_unlink(active_candidate, active, finish=True)
        _link_then_unlink(backup_candidate, backup, finish=True)

    assert recovery.recover_profile_migration_publication(active) is True
    if phase == "publishing":
        assert active.read_bytes() == old_active
        assert not backup.exists()
    else:
        assert active.read_bytes() == expected_active
        assert backup.read_bytes() == expected_backup
    assert not active_candidate.exists()
    assert not backup_candidate.exists()
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob("*.rollback"))


@pytest.mark.parametrize(
    "fault_site",
    [
        "move_after_mutation",
        "file_fsync",
        "parent_fsync",
        "validation",
        "journal_quarantine",
    ],
)
def test_internal_control_flow_is_deferred_until_recovery_reconverges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_site: str,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    old = target.read_bytes()
    _write_journal(publication, active, artifacts, destinations, phase="publishing")
    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=True)
    cancellation = asyncio.CancelledError(f"PRIVATE {fault_site}")
    raised = False
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")

    if fault_site == "move_after_mutation":
        real_rename = namespace._rename_noreplace

        def interrupt_rename(*args: object, **kwargs: object) -> None:
            nonlocal raised
            real_rename(*args, **kwargs)
            if not raised:
                raised = True
                raise cancellation

        monkeypatch.setattr(namespace, "_rename_noreplace", interrupt_rename)
    elif fault_site in {"file_fsync", "parent_fsync"}:
        real_fsync = recovery.os.fsync

        def interrupt_fsync(file_fd: int) -> None:
            nonlocal raised
            real_fsync(file_fd)
            is_directory = stat.S_ISDIR(os.fstat(file_fd).st_mode)
            selected = (
                is_directory if fault_site == "parent_fsync" else not is_directory
            )
            if selected and not raised:
                raised = True
                raise cancellation

        monkeypatch.setattr(recovery.os, "fsync", interrupt_fsync)
    elif fault_site == "validation":
        real_validate = recovery._validate_authoritative_targets

        def interrupt_validation(*args: object, **kwargs: object) -> None:
            nonlocal raised
            real_validate(*args, **kwargs)
            if not raised:
                raised = True
                raise cancellation

        monkeypatch.setattr(
            recovery, "_validate_authoritative_targets", interrupt_validation
        )
    else:
        real_remove = recovery.remove_exact_namespace

        def interrupt_journal_quarantine(*args: object, **kwargs: object) -> None:
            nonlocal raised
            path = args[0]
            real_remove(*args, **kwargs)
            if (
                isinstance(path, Path)
                and path.name.endswith(".migration-publication.json")
                and not raised
            ):
                raised = True
                raise cancellation

        monkeypatch.setattr(
            recovery,
            "remove_exact_namespace",
            interrupt_journal_quarantine,
        )

    with pytest.raises(asyncio.CancelledError) as caught:
        recovery.recover_profile_migration_publication(active)

    assert caught.value is cancellation
    assert target.read_bytes() == old
    assert not candidate.exists()
    assert not rollback.exists()
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    for tombstone in tmp_path.glob(".profile-migration-*.tombstone"):
        assert tombstone.read_bytes()
        assert stat.S_IMODE(tombstone.stat().st_mode) == 0o600


def test_foreign_holding_leaf_is_preserved_and_recovery_fails_closed(
    tmp_path: Path,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, _target, _rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="prepared"
    )
    holding = tmp_path / ".profile-migration-active-candidate.tombstone"
    foreign = b"foreign holding bytes"
    holding.write_bytes(foreign)
    holding.chmod(0o600)

    with pytest.raises(ProfileRepositoryError, match="unavailable"):
        recovery.recover_profile_migration_publication(active)

    assert holding.read_bytes() == foreign
    assert candidate.is_file()
    assert journal.is_file()


def test_foreign_holding_leaf_inserted_at_atomic_gap_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, _target, _rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="prepared"
    )
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    real_rename = namespace._rename_noreplace
    foreign = b"foreign atomic hold"
    inserted = False

    def occupy_hold(parent_fd: int, source: str, destination: str) -> None:
        nonlocal inserted
        if (
            source == candidate.name
            and destination == ".profile-migration-active-candidate.tombstone"
        ):
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                os.write(descriptor, foreign)
            finally:
                os.close(descriptor)
            inserted = True
        real_rename(parent_fd, source, destination)

    monkeypatch.setattr(namespace, "_rename_noreplace", occupy_hold)

    with pytest.raises(ProfileRepositoryError, match="unavailable"):
        recovery.recover_profile_migration_publication(active)

    assert inserted
    assert (
        tmp_path / ".profile-migration-active-candidate.tombstone"
    ).read_bytes() == foreign
    assert candidate.is_file()
    assert journal.is_file()


def test_foreign_child_inserted_after_exact_quarantine_dominates_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, _target, _rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="prepared"
    )
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    real_rename = namespace._rename_noreplace
    injected = False

    def mutate_parent(parent_fd: int, source: str, destination: str) -> None:
        nonlocal injected
        real_rename(parent_fd, source, destination)
        if source == candidate.name and not injected:
            os.mkdir("foreign-child", 0o700, dir_fd=parent_fd)
            injected = True

    monkeypatch.setattr(namespace, "_rename_noreplace", mutate_parent)

    with pytest.raises(ProfileRepositoryError, match="unavailable"):
        recovery.recover_profile_migration_publication(active)

    assert injected
    assert (tmp_path / "foreign-child").is_dir()
    assert journal.is_file()


def test_irrecoverable_failure_dominates_deferred_control_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=True)
    cancellation = asyncio.CancelledError("PRIVATE deferred")
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    real_rename = namespace._rename_noreplace
    calls = 0

    def interrupt_then_fail(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_rename(*args, **kwargs)
            raise cancellation
        raise OSError("PRIVATE storage failure")

    monkeypatch.setattr(namespace, "_rename_noreplace", interrupt_then_fail)

    with pytest.raises(ProfileRepositoryError, match="unavailable") as caught:
        recovery.recover_profile_migration_publication(active)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert journal.is_file()


def test_parent_substitution_mid_move_preserves_displaced_recovery_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, recovery = _modules()
    owned = tmp_path / "owned"
    publication, active, artifacts, destinations = _publication_fixture(
        owned, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=True)
    displaced = tmp_path / "displaced"
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    real_rename = namespace._rename_noreplace
    substituted = False

    def substitute_after_move(*args: object, **kwargs: object) -> None:
        nonlocal substituted
        real_rename(*args, **kwargs)
        if not substituted:
            substituted = True
            owned.rename(displaced)
            owned.mkdir(mode=0o700)

    monkeypatch.setattr(namespace, "_rename_noreplace", substitute_after_move)

    with pytest.raises(ProfileRepositoryError, match="unavailable") as caught:
        recovery.recover_profile_migration_publication(active)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert (displaced / journal.name).is_file()
    assert (displaced / rollback.name).is_file()
    assert any(
        path.name in {candidate.name, target.name} or path.name.endswith(".tombstone")
        for path in displaced.iterdir()
    )


def test_parent_link_authority_change_after_admission_fails_closed(
    tmp_path: Path,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="prepared"
    )

    def add_foreign_child(stage: str) -> None:
        if stage == "admitted":
            (tmp_path / "foreign-child").mkdir(mode=0o700, exist_ok=True)

    with pytest.raises(ProfileRepositoryError, match="unavailable"):
        recovery.recover_profile_migration_publication(
            active,
            _stage_hook=add_foreign_child,
        )

    assert journal.is_file()


def test_sqlite_validation_is_bound_to_pinned_descriptor_during_leaf_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    old = target.read_bytes()
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="publishing"
    )
    _link_then_unlink(target, rollback, finish=True)
    _link_then_unlink(candidate, target, finish=True)
    real_connect = recovery.connect_private_sqlite_descriptor
    foreign = tmp_path / "foreign.sqlite3"
    _store(foreign, version=4, marker="foreign")
    swapped = False

    def swap_during_open(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal swapped
        if not swapped:
            swapped = True
            held = tmp_path / ".swapped-owned.sqlite3"
            target.rename(held)
            foreign.rename(target)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(recovery, "connect_private_sqlite_descriptor", swap_during_open)

    with pytest.raises(ProfileRepositoryError, match="unavailable"):
        recovery.recover_profile_migration_publication(active)

    assert journal.is_file()
    assert target.read_bytes() != old


@pytest.mark.parametrize("operation", ["remove", "move", "journal_remove"])
def test_foreign_leaf_swapped_in_atomic_transition_gap_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    candidate, target, rollback = _paths(artifacts[0], destinations[0])
    journal = _write_journal(
        publication,
        active,
        artifacts,
        destinations,
        phase="prepared" if operation == "remove" else "publishing",
    )
    if operation != "remove":
        _link_then_unlink(target, rollback, finish=True)
        _link_then_unlink(candidate, target, finish=True)
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    real_rename = namespace._rename_noreplace
    foreign = b"foreign-substitution"
    swapped_leaf: str | None = None

    def swap_before_atomic_rename(
        parent_fd: int, source: str, destination: str
    ) -> None:
        nonlocal swapped_leaf
        relevant = (
            operation == "remove"
            and source == candidate.name
            or operation == "move"
            and source == rollback.name
            or operation == "journal_remove"
            and source == journal.name
        )
        if relevant and swapped_leaf is None:
            held = f".foreign-swap-held-{operation}"
            os.rename(source, held, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                os.write(descriptor, foreign)
            finally:
                os.close(descriptor)
            swapped_leaf = source
        real_rename(parent_fd, source, destination)

    monkeypatch.setattr(namespace, "_rename_noreplace", swap_before_atomic_rename)

    with pytest.raises(ProfileRepositoryError, match="unavailable"):
        recovery.recover_profile_migration_publication(active)

    assert swapped_leaf is not None
    assert (tmp_path / swapped_leaf).read_bytes() == foreign


@pytest.mark.parametrize("mutation", ["append", "same_size"])
def test_same_inode_journal_mutation_after_admission_is_preserved(
    tmp_path: Path,
    mutation: str,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    journal = _write_journal(
        publication, active, artifacts, destinations, phase="prepared"
    )
    original = journal.read_bytes()

    def mutate(stage: str) -> None:
        if stage != "validated":
            return
        if mutation == "append":
            with journal.open("ab") as stream:
                stream.write(b"PRIVATE append")
        else:
            changed = bytearray(original)
            changed[0] ^= 1
            journal.write_bytes(bytes(changed))
            journal.chmod(0o600)

    with pytest.raises(ProfileRepositoryError, match="unavailable") as caught:
        recovery.recover_profile_migration_publication(active, _stage_hook=mutate)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert journal.is_file()
    assert journal.read_bytes() != original


def test_recovery_artifact_size_limit_has_exact_boundary() -> None:
    _, recovery = _modules()
    maximum = recovery.MAX_PROFILE_MIGRATION_ARTIFACT_BYTES

    assert recovery._artifact_size_allowed(maximum)
    assert not recovery._artifact_size_allowed(maximum + 1)


def test_observed_artifact_over_limit_is_rejected_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, recovery = _modules()
    artifact = tmp_path / "oversize.sqlite3"
    with artifact.open("wb") as stream:
        stream.truncate(recovery.MAX_PROFILE_MIGRATION_ARTIFACT_BYTES + 1)
    descriptor = os.open(artifact, os.O_RDONLY)

    def should_not_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("oversize artifact must fail before reading")

    monkeypatch.setattr(recovery.os, "pread", should_not_read)
    try:
        with pytest.raises(ValueError):
            recovery._hash_sqlite(descriptor)
    finally:
        os.close(descriptor)


def test_oversize_journal_evidence_is_rejected_before_artifact_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication, recovery = _modules()
    publication, active, artifacts, destinations = _publication_fixture(
        tmp_path, slots=("active",)
    )
    authority = publication._journal_authority(artifacts, destinations)
    raw, _checksum = publication._encode_initial_journal(authority)
    decoded = json.loads(raw)
    evidence = decoded["recovery"]["authority"]["slots"][0]["candidate_evidence"]
    evidence["byte_length"] = recovery.MAX_PROFILE_MIGRATION_ARTIFACT_BYTES + 1
    recovery_payload = json.dumps(
        decoded["recovery"],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    decoded["checksum"] = sha256(recovery_payload).hexdigest()
    journal = active.with_name(f".{active.name}.migration-publication.json")
    journal.write_bytes(
        json.dumps(
            decoded,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    journal.chmod(0o600)

    def should_not_hash(_file_fd: int) -> object:
        raise AssertionError("oversize authority must fail before hashing")

    monkeypatch.setattr(recovery, "_hash_sqlite", should_not_hash)

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        recovery.recover_profile_migration_publication(active)

    assert journal.is_file()


@pytest.mark.parametrize("failure", ["short_read", "same_inode_change"])
def test_artifact_hash_rejects_short_read_or_same_inode_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _, recovery = _modules()
    path = tmp_path / "profiles.sqlite3"
    _store(path, version=4, marker="candidate")
    descriptor = os.open(path, os.O_RDWR)
    real_pread = recovery.os.pread
    calls = 0

    def unstable_pread(file_fd: int, count: int, offset: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            chunk = real_pread(file_fd, max(1, count // 2), offset)
            if failure == "same_inode_change":
                os.pwrite(file_fd, b"X", 100)
            return chunk
        return b"" if failure == "short_read" else real_pread(file_fd, count, offset)

    monkeypatch.setattr(recovery.os, "pread", unstable_pread)
    try:
        with pytest.raises(ValueError):
            recovery._hash_sqlite(descriptor)
    finally:
        os.close(descriptor)
