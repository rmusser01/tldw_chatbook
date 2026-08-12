from __future__ import annotations

import asyncio
import importlib
import json
import os
import sqlite3
import stat
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError


def _publication_module() -> ModuleType:
    try:
        return importlib.import_module(
            "tldw_chatbook.TTS.profile_migration_publication"
        )
    except ModuleNotFoundError:
        pytest.fail("profile migration publication module is missing")


def _store(path: Path, *, version: int, marker: str) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE exact_state (marker TEXT NOT NULL)")
        connection.execute("INSERT INTO exact_state VALUES (?)", (marker,))
        connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)
    return path.read_bytes()


def _validator(version: int, marker: str) -> Callable[[sqlite3.Connection], None]:
    def validate(connection: sqlite3.Connection) -> None:
        assert connection.execute("PRAGMA user_version").fetchone() == (version,)
        assert connection.execute("SELECT marker FROM exact_state").fetchone() == (
            marker,
        )

    return validate


def _prepared(module: ModuleType, path: Path, slot: object, marker: str) -> object:
    version = {
        module.ProfileMigrationPublicationSlot.ACTIVE: 4,
        module.ProfileMigrationPublicationSlot.PRE_V3: 2,
        module.ProfileMigrationPublicationSlot.PRE_V4: 3,
    }[slot]
    return module.prepare_profile_migration_artifact(
        path,
        slot=slot,
        validate=_validator(version, marker),
    )


def _retained(
    module: ModuleType,
    path: Path,
    slot: object,
    marker: str | None,
) -> object:
    version = {
        module.ProfileMigrationPublicationSlot.ACTIVE: 3,
        module.ProfileMigrationPublicationSlot.PRE_V3: 2,
        module.ProfileMigrationPublicationSlot.PRE_V4: 3,
    }[slot]
    return module.retain_profile_migration_destination(
        path,
        slot=slot,
        validate=None if marker is None else _validator(version, marker),
        must_exist=slot is module.ProfileMigrationPublicationSlot.ACTIVE,
    )


@pytest.mark.parametrize("cancel_stage_name", ["PREFLIGHT", "JOURNAL_DURABLE"])
def test_prepublication_cancellation_preserves_authority_and_cleans_candidates(
    tmp_path: Path,
    cancel_stage_name: str,
) -> None:
    module = _publication_module()
    active_path = tmp_path / "profiles.sqlite3"
    pre_v3_path = tmp_path / "profiles.pre-v3.sqlite3"
    candidate_path = tmp_path / ".active.candidate"
    prepared_pre_v3_path = tmp_path / ".pre-v3.candidate"
    active_before = _store(active_path, version=3, marker="old-active")
    backup_before = _store(pre_v3_path, version=2, marker="old-pre-v3")
    _store(candidate_path, version=4, marker="new-active")
    _store(prepared_pre_v3_path, version=2, marker="new-pre-v3")

    active_candidate = _prepared(
        module,
        candidate_path,
        module.ProfileMigrationPublicationSlot.ACTIVE,
        "new-active",
    )
    prepared_backup = _prepared(
        module,
        prepared_pre_v3_path,
        module.ProfileMigrationPublicationSlot.PRE_V3,
        "new-pre-v3",
    )
    active = _retained(
        module,
        active_path,
        module.ProfileMigrationPublicationSlot.ACTIVE,
        "old-active",
    )
    retained_backup = _retained(
        module,
        pre_v3_path,
        module.ProfileMigrationPublicationSlot.PRE_V3,
        "old-pre-v3",
    )
    cancellation = asyncio.CancelledError("private cancellation detail")

    def cancel(stage: object) -> None:
        if stage is getattr(module.ProfileMigrationPublicationStage, cancel_stage_name):
            raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        module.publish_profile_migration(
            active_candidate=active_candidate,
            backup_candidates=(prepared_backup,),
            active_destination=active,
            backup_destinations=(retained_backup,),
            stage_hook=cancel,
        )

    assert caught.value is cancellation
    assert active_path.read_bytes() == active_before
    assert pre_v3_path.read_bytes() == backup_before
    assert not candidate_path.exists()
    assert not prepared_pre_v3_path.exists()
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob("*.rollback"))


def test_durable_journal_has_only_recognized_bounded_relative_slots(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".candidate.sqlite3"
    _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    active_candidate = _prepared(
        module,
        candidate_path,
        module.ProfileMigrationPublicationSlot.ACTIVE,
        "new",
    )
    active = _retained(
        module,
        active_path,
        module.ProfileMigrationPublicationSlot.ACTIVE,
        "old",
    )
    observed: dict[str, object] = {}
    cancellation = asyncio.CancelledError()

    def inspect_then_cancel(stage: object) -> None:
        if stage is module.ProfileMigrationPublicationStage.JOURNAL_DURABLE:
            journal_path = next(tmp_path.glob("*.migration-publication.json"))
            raw = journal_path.read_bytes()
            observed["raw"] = raw
            observed["mode"] = stat.S_IMODE(journal_path.stat().st_mode)
            observed["parsed"] = module.parse_profile_migration_journal(raw)
            raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        module.publish_profile_migration(
            active_candidate=active_candidate,
            backup_candidates=(),
            active_destination=active,
            backup_destinations=(),
            stage_hook=inspect_then_cancel,
        )

    assert caught.value is cancellation
    raw = observed["raw"]
    assert isinstance(raw, bytes)
    assert len(raw) <= 4096
    assert str(tmp_path).encode() not in raw
    assert b"old" not in raw
    assert b"new" not in raw
    assert observed["mode"] == 0o600
    parsed = observed["parsed"]
    assert parsed.version == 1
    assert parsed.phase == "prepared"
    assert parsed.slots == ("active",)
    assert json.loads(raw) == {
        "phase": "prepared",
        "slots": [
            {
                "candidate": ".candidate.sqlite3",
                "had_prior": True,
                "rollback": ".profiles.sqlite3.active.rollback",
                "slot": "active",
                "target": "profiles.sqlite3",
            }
        ],
        "version": 1,
    }


def test_prepublication_failure_is_bounded_and_context_free(tmp_path: Path) -> None:
    module = _publication_module()
    active_path = tmp_path / "PRIVATE profiles.sqlite3"
    candidate_path = tmp_path / "PRIVATE candidate.sqlite3"
    active_before = _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    active_candidate = _prepared(
        module,
        candidate_path,
        module.ProfileMigrationPublicationSlot.ACTIVE,
        "new",
    )
    active = _retained(
        module,
        active_path,
        module.ProfileMigrationPublicationSlot.ACTIVE,
        "old",
    )

    def fail(_stage: object) -> None:
        raise RuntimeError(f"PRIVATE failure at {tmp_path}")

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.publish_profile_migration(
            active_candidate=active_candidate,
            backup_candidates=(),
            active_destination=active,
            backup_destinations=(),
            stage_hook=fail,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(tmp_path) not in repr(caught.value)
    assert active_path.read_bytes() == active_before
    assert not candidate_path.exists()


def test_successfully_publishes_active_then_every_backup_in_slot_order(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    pre_v3_path = tmp_path / "profiles.pre-v3.sqlite3"
    pre_v4_path = tmp_path / "profiles.pre-v4.sqlite3"
    active_candidate_path = tmp_path / ".active.candidate"
    pre_v3_candidate_path = tmp_path / ".pre-v3.candidate"
    pre_v4_candidate_path = tmp_path / ".pre-v4.candidate"
    _store(active_path, version=3, marker="old-active")
    old_pre_v3 = _store(pre_v3_path, version=2, marker="old-pre-v3")
    expected_active = _store(active_candidate_path, version=4, marker="new-active")
    expected_pre_v3 = _store(pre_v3_candidate_path, version=2, marker="new-pre-v3")
    expected_pre_v4 = _store(pre_v4_candidate_path, version=3, marker="new-pre-v4")
    artifacts = (
        _prepared(module, active_candidate_path, slot.ACTIVE, "new-active"),
        _prepared(module, pre_v3_candidate_path, slot.PRE_V3, "new-pre-v3"),
        _prepared(module, pre_v4_candidate_path, slot.PRE_V4, "new-pre-v4"),
    )
    destinations = (
        _retained(module, active_path, slot.ACTIVE, "old-active"),
        _retained(module, pre_v3_path, slot.PRE_V3, "old-pre-v3"),
        _retained(module, pre_v4_path, slot.PRE_V4, None),
    )
    events: list[object] = []

    module.publish_profile_migration(
        active_candidate=artifacts[0],
        backup_candidates=artifacts[1:],
        active_destination=destinations[0],
        backup_destinations=destinations[1:],
        stage_hook=events.append,
    )

    stage = module.ProfileMigrationPublicationStage
    assert events == [
        stage.PREFLIGHT,
        stage.JOURNAL_DURABLE,
        stage.PONR,
        stage.ACTIVE_RETAINED,
        stage.ACTIVE_REPLACED,
        stage.ACTIVE_FSYNCED,
        stage.ACTIVE_REOPENED,
        stage.BACKUP_RETAINED,
        stage.BACKUP_REPLACED,
        stage.BACKUP_FSYNCED,
        stage.BACKUP_REOPENED,
        stage.BACKUP_REPLACED,
        stage.BACKUP_FSYNCED,
        stage.BACKUP_REOPENED,
        stage.FINAL_JOURNAL_DURABLE,
    ]
    assert active_path.read_bytes() == expected_active
    assert pre_v3_path.read_bytes() == expected_pre_v3 != old_pre_v3
    assert pre_v4_path.read_bytes() == expected_pre_v4
    assert not active_candidate_path.exists()
    assert not pre_v3_candidate_path.exists()
    assert not pre_v4_candidate_path.exists()
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob("*.rollback"))
    if os.name == "posix":
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600
            for path in (active_path, pre_v3_path, pre_v4_path)
        )


@pytest.mark.parametrize(
    ("fail_stage_name", "occurrence"),
    [
        ("ACTIVE_REOPENED", 1),
        ("BACKUP_REPLACED", 1),
        ("BACKUP_FSYNCED", 1),
        ("BACKUP_REPLACED", 2),
        ("BACKUP_FSYNCED", 2),
    ],
)
def test_post_replace_failure_restores_active_and_every_prior_backup(
    tmp_path: Path,
    fail_stage_name: str,
    occurrence: int,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "PRIVATE profiles.sqlite3"
    pre_v3_path = tmp_path / "PRIVATE profiles.pre-v3.sqlite3"
    pre_v4_path = tmp_path / "PRIVATE profiles.pre-v4.sqlite3"
    paths = (active_path, pre_v3_path, pre_v4_path)
    before = (
        _store(active_path, version=3, marker="old-active"),
        _store(pre_v3_path, version=2, marker="old-pre-v3"),
        _store(pre_v4_path, version=3, marker="old-pre-v4"),
    )
    candidate_paths = (
        tmp_path / ".PRIVATE active.candidate",
        tmp_path / ".PRIVATE pre-v3.candidate",
        tmp_path / ".PRIVATE pre-v4.candidate",
    )
    _store(candidate_paths[0], version=4, marker="new-active")
    _store(candidate_paths[1], version=2, marker="new-pre-v3")
    _store(candidate_paths[2], version=3, marker="new-pre-v4")
    artifacts = (
        _prepared(module, candidate_paths[0], slot.ACTIVE, "new-active"),
        _prepared(module, candidate_paths[1], slot.PRE_V3, "new-pre-v3"),
        _prepared(module, candidate_paths[2], slot.PRE_V4, "new-pre-v4"),
    )
    destinations = (
        _retained(module, active_path, slot.ACTIVE, "old-active"),
        _retained(module, pre_v3_path, slot.PRE_V3, "old-pre-v3"),
        _retained(module, pre_v4_path, slot.PRE_V4, "old-pre-v4"),
    )
    fail_stage = getattr(module.ProfileMigrationPublicationStage, fail_stage_name)
    observed = 0

    def fail(stage: object) -> None:
        nonlocal observed
        if stage is fail_stage:
            observed += 1
            if observed == occurrence:
                raise RuntimeError(f"PRIVATE injected failure at {tmp_path}")

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.publish_profile_migration(
            active_candidate=artifacts[0],
            backup_candidates=artifacts[1:],
            active_destination=destinations[0],
            backup_destinations=destinations[1:],
            stage_hook=fail,
        )

    assert caught.value.code == "migration_failed"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(tmp_path) not in repr(caught.value)
    assert tuple(path.read_bytes() for path in paths) == before
    assert all(not path.exists() for path in candidate_paths)
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob("*.rollback"))


@pytest.mark.parametrize(
    ("cancel_stage_name", "occurrence"),
    [
        ("PONR", 1),
        ("ACTIVE_RETAINED", 1),
        ("ACTIVE_REPLACED", 1),
        ("ACTIVE_FSYNCED", 1),
        ("ACTIVE_REOPENED", 1),
        ("BACKUP_RETAINED", 1),
        ("BACKUP_REPLACED", 1),
        ("BACKUP_FSYNCED", 1),
        ("BACKUP_REOPENED", 1),
        ("BACKUP_RETAINED", 2),
        ("BACKUP_REPLACED", 2),
        ("BACKUP_FSYNCED", 2),
        ("BACKUP_REOPENED", 2),
        ("FINAL_JOURNAL_DURABLE", 1),
    ],
)
def test_cancellation_at_every_post_ponr_stage_is_deferred_then_redelivered(
    tmp_path: Path,
    cancel_stage_name: str,
    occurrence: int,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    paths = (
        tmp_path / "profiles.sqlite3",
        tmp_path / "profiles.pre-v3.sqlite3",
        tmp_path / "profiles.pre-v4.sqlite3",
    )
    _store(paths[0], version=3, marker="old-active")
    _store(paths[1], version=2, marker="old-pre-v3")
    _store(paths[2], version=3, marker="old-pre-v4")
    candidate_paths = (
        tmp_path / ".active.candidate",
        tmp_path / ".pre-v3.candidate",
        tmp_path / ".pre-v4.candidate",
    )
    expected = (
        _store(candidate_paths[0], version=4, marker="new-active"),
        _store(candidate_paths[1], version=2, marker="new-pre-v3"),
        _store(candidate_paths[2], version=3, marker="new-pre-v4"),
    )
    artifacts = (
        _prepared(module, candidate_paths[0], slot.ACTIVE, "new-active"),
        _prepared(module, candidate_paths[1], slot.PRE_V3, "new-pre-v3"),
        _prepared(module, candidate_paths[2], slot.PRE_V4, "new-pre-v4"),
    )
    destinations = (
        _retained(module, paths[0], slot.ACTIVE, "old-active"),
        _retained(module, paths[1], slot.PRE_V3, "old-pre-v3"),
        _retained(module, paths[2], slot.PRE_V4, "old-pre-v4"),
    )
    cancel_stage = getattr(
        module.ProfileMigrationPublicationStage,
        cancel_stage_name,
    )
    cancellation = asyncio.CancelledError("private control-flow detail")
    observed = 0

    def cancel(stage: object) -> None:
        nonlocal observed
        if stage is cancel_stage:
            observed += 1
            if observed == occurrence:
                raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        module.publish_profile_migration(
            active_candidate=artifacts[0],
            backup_candidates=artifacts[1:],
            active_destination=destinations[0],
            backup_destinations=destinations[1:],
            stage_hook=cancel,
        )

    assert caught.value is cancellation
    assert tuple(path.read_bytes() for path in paths) == expected
    assert all(not path.exists() for path in candidate_paths)
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob("*.rollback"))


def test_completion_and_restoration_failure_retains_recovery_set_and_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "PRIVATE profiles.sqlite3"
    backup_path = tmp_path / "PRIVATE profiles.pre-v3.sqlite3"
    active_candidate_path = tmp_path / ".PRIVATE active.candidate"
    backup_candidate_path = tmp_path / ".PRIVATE pre-v3.candidate"
    _store(active_path, version=3, marker="old-active")
    backup_before = _store(backup_path, version=2, marker="old-backup")
    candidate_before = _store(
        active_candidate_path,
        version=4,
        marker="new-active",
    )
    prepared_backup_before = _store(
        backup_candidate_path,
        version=2,
        marker="new-backup",
    )
    active_candidate = _prepared(
        module,
        active_candidate_path,
        slot.ACTIVE,
        "new-active",
    )
    prepared_backup = _prepared(
        module,
        backup_candidate_path,
        slot.PRE_V3,
        "new-backup",
    )
    active = _retained(module, active_path, slot.ACTIVE, "old-active")
    backup = _retained(module, backup_path, slot.PRE_V3, "old-backup")
    real_rename = module._rename_exact

    def fail_active_rollback(source: object, destination: Path) -> object:
        if str(source._path).endswith(".active.rollback"):
            raise OSError(f"PRIVATE total storage failure at {tmp_path}")
        return real_rename(source, destination)

    monkeypatch.setattr(module, "_rename_exact", fail_active_rollback)

    def fail_completion(stage: object) -> None:
        if stage is module.ProfileMigrationPublicationStage.ACTIVE_REOPENED:
            raise RuntimeError("PRIVATE completion failure")

    with pytest.raises(ProfileRepositoryError, match="unavailable") as caught:
        module.publish_profile_migration(
            active_candidate=active_candidate,
            backup_candidates=(prepared_backup,),
            active_destination=active,
            backup_destinations=(backup,),
            stage_hook=fail_completion,
        )

    assert caught.value.code == "unavailable"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert str(tmp_path) not in repr(caught.value)
    journal_path = next(tmp_path.glob("*.migration-publication.json"))
    journal = module.parse_profile_migration_journal(journal_path.read_bytes())
    assert journal.phase == "unavailable"
    assert active_candidate_path.read_bytes() == candidate_before
    assert backup_candidate_path.read_bytes() == prepared_backup_before
    active_rollback = tmp_path / ".PRIVATE profiles.sqlite3.active.rollback"
    assert active_rollback.is_file()
    assert backup_path.read_bytes() == backup_before
    assert not active_path.exists()


def test_opaque_artifact_constructor_cannot_be_forged(tmp_path: Path) -> None:
    module = _publication_module()
    path = tmp_path / "candidate.sqlite3"
    _store(path, version=4, marker="candidate")

    with pytest.raises(TypeError):
        module.PreparedProfileMigrationArtifact(
            path=path,
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            parent_identity=tmp_path.stat(),
            file_identity=path.stat(),
            validate=_validator(4, "candidate"),
        )


def test_reentrant_double_publication_is_rejected_without_disrupting_owner(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".candidate.sqlite3"
    _store(active_path, version=3, marker="old")
    expected = _store(candidate_path, version=4, marker="new")
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    replay_error: ProfileRepositoryError | None = None

    def replay(stage: object) -> None:
        nonlocal replay_error
        if stage is module.ProfileMigrationPublicationStage.PONR:
            try:
                module.publish_profile_migration(
                    active_candidate=artifact,
                    backup_candidates=(),
                    active_destination=destination,
                    backup_destinations=(),
                )
            except ProfileRepositoryError as error:
                replay_error = error

    module.publish_profile_migration(
        active_candidate=artifact,
        backup_candidates=(),
        active_destination=destination,
        backup_destinations=(),
        stage_hook=replay,
    )

    assert replay_error is not None
    assert replay_error.code == "migration_failed"
    assert active_path.read_bytes() == expected
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob("*.rollback"))


@pytest.mark.parametrize(
    "raw",
    [
        b'PRIVATE {"phase":"publishing","slots":[],"version":1}\n',
        b'{"phase":"unknown","slots":[],"version":1}\n',
        b'{"phase":"prepared","slots":[{"candidate":"../private",'
        b'"had_prior":true,"rollback":"r","slot":"active",'
        b'"target":"t"}],"version":1}\n',
    ],
)
def test_journal_parser_rejects_forgery_context_free(raw: bytes) -> None:
    module = _publication_module()

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.parse_profile_migration_journal(raw)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert b"PRIVATE" not in repr(caught.value).encode()


@pytest.mark.parametrize("substitute", ["candidate", "target"])
def test_substitution_fails_closed_and_preserves_foreign_object(
    tmp_path: Path,
    substitute: str,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".candidate.sqlite3"
    retained_original = tmp_path / f"retained-{substitute}.sqlite3"
    _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    selected = candidate_path if substitute == "candidate" else active_path
    selected.rename(retained_original)
    foreign = _store(
        selected,
        version=4 if substitute == "candidate" else 3,
        marker="foreign",
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
        )

    assert caught.value.code in {"migration_failed", "unavailable"}
    assert selected.read_bytes() == foreign
    assert retained_original.is_file()
