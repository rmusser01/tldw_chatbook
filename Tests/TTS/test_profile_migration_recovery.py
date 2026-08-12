from __future__ import annotations

import asyncio
import copy
import importlib
import io
import os
import pickle
import sqlite3
import stat
from pathlib import Path
from types import ModuleType
from zlib import crc32

import pytest

from tldw_chatbook.TTS import profile_schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


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
        candidate = tmp_path / f".{label}.candidate.sqlite3"
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
    candidate = artifact._path
    target = destination._path
    rollback = target.with_name(f".{target.name}.{artifact._slot.value}.rollback")
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
    active_candidate = tmp_path / ".active.candidate.sqlite3"
    backup = tmp_path / "profiles.pre-v4.sqlite3"
    backup_candidate = tmp_path / ".pre-v4.candidate.sqlite3"
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
        active_rollback = active.with_name(f".{active.name}.active.rollback")
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
