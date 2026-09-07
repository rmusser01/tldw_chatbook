from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, fields, is_dataclass
import errno
from hashlib import sha256
import importlib
import inspect
import io
import json
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
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        for source_version in range(version):
            profile_schema.MIGRATIONS[source_version](connection)
        marker_id = crc32(marker.encode("utf-8")) & 0x7FFF_FFFF
        connection.execute(f"PRAGMA application_id = {marker_id}")
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)
    return path.read_bytes()


def _prepared(module: ModuleType, path: Path, slot: object, marker: str) -> object:
    del marker
    return module.prepare_profile_migration_artifact(
        path,
        slot=slot,
    )


def _retained(
    module: ModuleType,
    path: Path,
    slot: object,
    marker: str | None,
) -> object:
    del marker
    return module.retain_profile_migration_destination(
        path,
        slot=slot,
        must_exist=slot is module.ProfileMigrationPublicationSlot.ACTIVE,
    )


def _encoded_journal(
    module: ModuleType,
    tmp_path: Path,
    rows: tuple[object, ...],
    *phases: str,
) -> bytes:
    rows = tuple(
        module.ProfileMigrationJournalSlot(
            slot=row.slot,
            candidate=module.PROFILE_MIGRATION_CANDIDATE_LEAVES[row.slot],
            target=row.target,
            rollback=module.PROFILE_MIGRATION_ROLLBACK_LEAVES[row.slot],
            had_prior=row.had_prior,
        )
        for row in rows
    )
    identity = tmp_path.stat()
    authority_rows = []
    next_inode = identity.st_ino + 1

    def unique_identity() -> os.stat_result:
        nonlocal next_inode
        result = os.stat_result(
            (
                identity.st_mode,
                next_inode,
                identity.st_dev,
                identity.st_nlink,
                identity.st_uid,
                identity.st_gid,
                identity.st_size,
                identity.st_atime,
                identity.st_mtime,
                identity.st_ctime,
            )
        )
        next_inode += 1
        return result

    for row in rows:
        schema_version = {
            module.ProfileMigrationPublicationSlot.ACTIVE: 4,
            module.ProfileMigrationPublicationSlot.PRE_V3: 2,
            module.ProfileMigrationPublicationSlot.PRE_V4: 3,
        }[row.slot]
        candidate_identity = unique_identity()
        prior_identity = unique_identity() if row.had_prior else None
        authority_rows.append(
            (
                row,
                candidate_identity,
                (1, b"\x01" * 32),
                schema_version,
                prior_identity,
                (1, b"\x02" * 32) if row.had_prior else None,
                schema_version if row.had_prior else None,
            )
        )
    authority = module._new_profile_migration_journal_authority(
        parent_identity=identity,
        rows=tuple(authority_rows),
    )
    initial, checksum = module._encode_initial_journal(authority)
    return initial + b"".join(
        module._encode_later_journal(checksum, phase=phase) for phase in phases
    )


def test_journal_artifact_size_boundary_is_symmetric(tmp_path: Path) -> None:
    module = _publication_module()
    maximum = module.MAX_PROFILE_MIGRATION_ARTIFACT_BYTES
    row = module.ProfileMigrationJournalSlot(
        slot=module.ProfileMigrationPublicationSlot.ACTIVE,
        candidate=".profile-migration-active.candidate.sqlite3",
        target="profiles.sqlite3",
        rollback=".profile-migration-active.rollback.sqlite3",
        had_prior=False,
    )
    parent = tmp_path.stat()
    candidate = os.stat_result(
        (
            stat.S_IFREG | 0o600,
            parent.st_ino + 1,
            parent.st_dev,
            1,
            os.geteuid(),
            parent.st_gid,
            maximum,
            parent.st_atime,
            parent.st_mtime,
            parent.st_ctime,
        )
    )
    authority = module._new_profile_migration_journal_authority(
        parent_identity=parent,
        rows=((row, candidate, (maximum, b"\x01" * 32), 4, None, None, None),),
    )
    encoded, _checksum = module._encode_initial_journal(authority)

    assert (
        module.parse_profile_migration_journal(encoded)
        .recovery_rows[0]
        .evidence_fits(maximum)
    )

    with pytest.raises(ValueError):
        module._new_profile_migration_journal_authority(
            parent_identity=parent,
            rows=(
                (
                    row,
                    candidate,
                    (maximum + 1, b"\x01" * 32),
                    4,
                    None,
                    None,
                    None,
                ),
            ),
        )

    decoded = json.loads(encoded)
    decoded["recovery"]["authority"]["slots"][0]["candidate_evidence"][
        "byte_length"
    ] = maximum + 1
    recovery_payload = json.dumps(
        decoded["recovery"],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    decoded["checksum"] = sha256(recovery_payload).hexdigest()
    forged = (
        json.dumps(
            decoded,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.parse_profile_migration_journal(forged)


def test_oversize_sparse_candidate_is_rejected_before_hash_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _publication_module()
    active = tmp_path / "profiles.sqlite3"
    prior = tmp_path / "profiles.pre-v4.sqlite3"
    candidate = tmp_path / ".profile-migration-active.candidate.sqlite3"
    active_before = _store(active, version=3, marker="old-active")
    prior_before = _store(prior, version=3, marker="old-prior")
    _store(candidate, version=4, marker="new")
    with candidate.open("r+b") as stream:
        stream.truncate(module.MAX_PROFILE_MIGRATION_ARTIFACT_BYTES + 1)
    reads = 0
    real_pread = module.os.pread

    def observe_pread(file_fd: int, count: int, offset: int) -> bytes:
        nonlocal reads
        reads += 1
        return real_pread(file_fd, count, offset)

    monkeypatch.setattr(module.os, "pread", observe_pread)

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.prepare_profile_migration_artifact(
            candidate,
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
        )

    assert reads == 0
    assert active.read_bytes() == active_before
    assert prior.read_bytes() == prior_before
    assert not tuple(tmp_path.glob("*.migration-publication.json"))


@pytest.mark.parametrize("cancel_stage_name", ["PREFLIGHT", "JOURNAL_DURABLE"])
def test_prepublication_cancellation_preserves_authority_and_cleans_candidates(
    tmp_path: Path,
    cancel_stage_name: str,
) -> None:
    module = _publication_module()
    active_path = tmp_path / "profiles.sqlite3"
    pre_v3_path = tmp_path / "profiles.pre-v4.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    prepared_pre_v3_path = tmp_path / ".profile-migration-pre-v4.candidate.sqlite3"
    active_before = _store(active_path, version=3, marker="old-active")
    backup_before = _store(pre_v3_path, version=3, marker="old-pre-v4")
    _store(candidate_path, version=4, marker="new-active")
    _store(prepared_pre_v3_path, version=3, marker="new-pre-v4")

    active_candidate = _prepared(
        module,
        candidate_path,
        module.ProfileMigrationPublicationSlot.ACTIVE,
        "new-active",
    )
    prepared_backup = _prepared(
        module,
        prepared_pre_v3_path,
        module.ProfileMigrationPublicationSlot.PRE_V4,
        "new-pre-v4",
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
        module.ProfileMigrationPublicationSlot.PRE_V4,
        "old-pre-v4",
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
    assert not tuple(tmp_path.glob(".profile-migration-*.rollback.sqlite3"))


def test_durable_journal_has_only_recognized_bounded_relative_slots(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
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
    assert parsed.version == 2
    assert parsed.phase == "prepared"
    assert parsed.slots == ("active",)
    assert parsed.recovery_rows == (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    assert module.parse_profile_migration_journal(raw).recovery_rows == (
        parsed.recovery_rows
    )
    assert ".profile-migration-active.candidate.sqlite3" not in repr(parsed)
    assert "profiles.sqlite3" not in repr(parsed.recovery_rows[0])
    decoded_frame = json.loads(raw)
    assert set(decoded_frame) == {"checksum", "recovery"}
    assert len(decoded_frame["checksum"]) == 64
    recovery = decoded_frame["recovery"]
    assert recovery["phase"] == "prepared"
    assert recovery["version"] == 2
    assert set(recovery["authority"]) == {"parent", "slots"}
    assert (
        recovery["authority"]["slots"][0]["candidate"]
        == ".profile-migration-active.candidate.sqlite3"
    )


def test_initial_journal_authority_detects_substitutions_and_redacts_evidence(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    candidate_bytes = candidate_path.read_bytes()
    prior_bytes = active_path.read_bytes()
    candidate_stat = candidate_path.stat()
    prior_stat = active_path.stat()
    parent_stat = tmp_path.stat()
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    observed: dict[str, object] = {}
    cancellation = asyncio.CancelledError()

    def inspect(stage: object) -> None:
        if stage is module.ProfileMigrationPublicationStage.JOURNAL_DURABLE:
            raw = next(tmp_path.glob("*.migration-publication.json")).read_bytes()
            observed["raw"] = raw
            observed["parsed"] = module.parse_profile_migration_journal(raw)
            raise cancellation

    with pytest.raises(asyncio.CancelledError):
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
            stage_hook=inspect,
        )

    parsed = observed["parsed"]
    row = parsed.recovery_rows[0]
    assert parsed.matches_parent(parent_stat)
    assert row.matches_candidate(
        candidate_stat,
        byte_length=len(candidate_bytes),
        sha256_digest=sha256(candidate_bytes).digest(),
        schema_version=4,
    )
    assert row.matches_prior(
        prior_stat,
        byte_length=len(prior_bytes),
        sha256_digest=sha256(prior_bytes).digest(),
        schema_version=3,
    )
    assert (
        row.classify_artifact(
            candidate_stat,
            byte_length=len(candidate_bytes),
            sha256_digest=sha256(candidate_bytes).digest(),
            schema_version=4,
        )
        == "candidate"
    )
    assert (
        row.classify_artifact(
            prior_stat,
            byte_length=len(prior_bytes),
            sha256_digest=sha256(prior_bytes).digest(),
            schema_version=3,
        )
        == "prior"
    )
    assert (
        row.matches_candidate(
            candidate_stat,
            byte_length=len(candidate_bytes),
            sha256_digest=sha256(candidate_bytes).digest(),
            schema_version=3,
        )
        is False
    )
    substituted = os.stat_result(
        (
            candidate_stat.st_mode,
            candidate_stat.st_ino + 1,
            candidate_stat.st_dev,
            candidate_stat.st_nlink,
            candidate_stat.st_uid,
            candidate_stat.st_gid,
            candidate_stat.st_size,
            candidate_stat.st_atime,
            candidate_stat.st_mtime,
            candidate_stat.st_ctime,
        )
    )
    assert (
        row.matches_candidate(
            substituted,
            byte_length=len(candidate_bytes),
            sha256_digest=sha256(candidate_bytes).digest(),
            schema_version=4,
        )
        is False
    )
    assert (
        row.classify_artifact(
            substituted,
            byte_length=len(candidate_bytes),
            sha256_digest=sha256(candidate_bytes).digest(),
            schema_version=4,
        )
        is None
    )
    parent_substitution = os.stat_result(
        (
            parent_stat.st_mode,
            parent_stat.st_ino + 1,
            parent_stat.st_dev,
            parent_stat.st_nlink,
            parent_stat.st_uid,
            parent_stat.st_gid,
            parent_stat.st_size,
            parent_stat.st_atime,
            parent_stat.st_mtime,
            parent_stat.st_ctime,
        )
    )
    assert parsed.matches_parent(parent_substitution) is False
    private_values = (
        str(parent_stat.st_dev),
        str(parent_stat.st_ino),
        sha256(candidate_bytes).hexdigest(),
        sha256(prior_bytes).hexdigest(),
    )
    assert all(value not in repr(parsed) for value in private_values)
    assert all(value not in repr(row) for value in private_values)


def test_parsed_authority_is_not_dataclass_flattenable_or_introspectable(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    journal_module = importlib.import_module(
        "tldw_chatbook.TTS.profile_migration_journal"
    )
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    parsed = module.parse_profile_migration_journal(
        _encoded_journal(module, tmp_path, rows)
    )
    row = parsed.recovery_rows[0]
    identity = tmp_path.stat()
    evidence = journal_module._ArtifactEvidence(
        identity.st_dev,
        identity.st_ino + 9,
        37,
        b"\xab" * 32,
        4,
    )
    private_attribute_names = {
        "dev",
        "ino",
        "byte_length",
        "sha256_digest",
        "schema_version",
        "_candidate_evidence",
        "_prior_evidence",
        "_parent_dev",
        "_parent_ino",
        "_authority_checksum",
    }
    private_values = {
        str(identity.st_dev),
        str(identity.st_ino),
        str(identity.st_ino + 9),
        (b"\xab" * 32).hex(),
    }

    for value in (parsed, row, evidence):
        assert not is_dataclass(value)
        with pytest.raises(TypeError):
            asdict(value)
        with pytest.raises(TypeError):
            fields(value)
        with pytest.raises(TypeError):
            vars(value)
        assert private_attribute_names.isdisjoint(dir(value))

        structured = {
            name: repr(member)
            for name, member in inspect.getmembers(value)
            if not callable(member)
        }
        rendered = json.dumps(structured, sort_keys=True)
        assert all(private not in rendered for private in private_values)


def test_authority_objects_refuse_pickle_copy_and_state_access(tmp_path: Path) -> None:
    module = _publication_module()
    journal_module = importlib.import_module(
        "tldw_chatbook.TTS.profile_migration_journal"
    )
    parent = tmp_path.stat()
    digest = (b"PRIVATE_AUTHORITY_DIGEST_" * 2)[:32]
    evidence = journal_module._ArtifactEvidence(
        parent.st_dev,
        parent.st_ino + 17,
        41,
        digest,
        4,
    )
    artifact_capsule = journal_module._ArtifactAuthorityCapsule(
        parent.st_dev,
        parent.st_ino + 17,
        41,
        digest,
        4,
    )
    parent_capsule = journal_module._ParentAuthorityCapsule(
        parent.st_dev,
        parent.st_ino,
    )
    row = module.ProfileMigrationJournalSlot._with_authority(
        slot=module.ProfileMigrationPublicationSlot.ACTIVE,
        candidate=".profile-migration-active.candidate.sqlite3",
        target="profiles.sqlite3",
        rollback=".profile-migration-active.rollback.sqlite3",
        had_prior=False,
        candidate_evidence=evidence,
        prior_evidence=None,
    )
    authority = journal_module._JournalAuthority(
        parent.st_dev,
        parent.st_ino,
        (row,),
    )
    parsed = module.ParsedProfileMigrationJournal(
        2,
        "prepared",
        (row,),
        parent.st_dev,
        parent.st_ino,
    )
    objects = (artifact_capsule, evidence, parent_capsule, row, authority, parsed)
    private_canaries = (
        digest,
        digest.hex().encode("ascii"),
        str(parent.st_dev).encode("ascii"),
        str(parent.st_ino).encode("ascii"),
        str(parent.st_ino + 17).encode("ascii"),
    )

    for value in objects:
        for protocol in range(2, 6):
            sink = io.BytesIO()
            with pytest.raises(TypeError, match="private_authority") as caught:
                pickle.Pickler(sink, protocol=protocol).dump(value)
            assert caught.value.__cause__ is None
            assert caught.value.__context__ is None
            assert all(canary not in sink.getvalue() for canary in private_canaries)
            assert all(
                canary.decode("ascii", "ignore") not in repr(caught.value)
                for canary in private_canaries
            )
        for operation in (copy.copy, copy.deepcopy):
            with pytest.raises(TypeError, match="private_authority"):
                operation(value)
        with pytest.raises(TypeError, match="private_authority"):
            value.__getstate__()

    assert parsed.version == 2
    assert parsed.phase == "prepared"
    assert parsed.slots == ("active",)
    assert parsed.matches_parent(parent)
    assert row.matches_candidate(
        os.stat_result(
            (
                parent.st_mode,
                parent.st_ino + 17,
                parent.st_dev,
                1,
                parent.st_uid,
                parent.st_gid,
                41,
                parent.st_atime,
                parent.st_mtime,
                parent.st_ctime,
            )
        ),
        byte_length=41,
        sha256_digest=digest,
        schema_version=4,
    )


def test_later_journal_frames_bind_without_repeating_authority(tmp_path: Path) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    observed: dict[str, bytes] = {}

    def inspect(stage: object) -> None:
        if stage is module.ProfileMigrationPublicationStage.PONR:
            observed["raw"] = next(
                tmp_path.glob("*.migration-publication.json")
            ).read_bytes()
            raise RuntimeError("stop after publishing frame")

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
            stage_hook=inspect,
        )

    initial, later = observed["raw"].splitlines()
    assert b'"authority"' in initial
    assert b'"authority"' not in later
    assert b'"authority_checksum"' in later
    assert b'"sha256"' not in later
    assert b'"byte_length"' not in later


def test_later_frame_with_foreign_authority_binding_fails_context_free(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    initial = _encoded_journal(module, tmp_path, rows)
    foreign = module._encode_later_journal(b"\x99" * 32, phase="publishing")

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.parse_profile_migration_journal(initial + foreign)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert (b"99" * 32) not in repr(caught.value).encode()


@pytest.mark.parametrize(
    ("field", "value"),
    [("dev", -1), ("schema_version", 3)],
)
def test_canonical_forged_invalid_authority_evidence_fails_boundedly(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    module = _publication_module()
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    decoded = json.loads(_encoded_journal(module, tmp_path, rows))
    decoded["recovery"]["authority"]["slots"][0]["candidate_evidence"][field] = value
    recovery = json.dumps(
        decoded["recovery"],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    decoded["checksum"] = sha256(recovery).hexdigest()
    forged = (
        json.dumps(decoded, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
    )

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.parse_profile_migration_journal(forged)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "mutation",
    ["candidate_prior_same", "cross_row_duplicate", "foreign_device"],
)
def test_initial_authority_rejects_duplicate_or_foreign_identities(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=slot.ACTIVE,
            candidate=".active",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
        module.ProfileMigrationJournalSlot(
            slot=slot.PRE_V4,
            candidate=".pre-v4",
            target="profiles.pre-v4.sqlite3",
            rollback=".profile-migration-pre-v4.rollback.sqlite3",
            had_prior=True,
        ),
    )
    decoded = json.loads(_encoded_journal(module, tmp_path, rows))
    authority = decoded["recovery"]["authority"]
    first = authority["slots"][0]
    second = authority["slots"][1]
    if mutation == "candidate_prior_same":
        first["prior_evidence"]["dev"] = first["candidate_evidence"]["dev"]
        first["prior_evidence"]["ino"] = first["candidate_evidence"]["ino"]
    elif mutation == "cross_row_duplicate":
        second["candidate_evidence"]["dev"] = first["candidate_evidence"]["dev"]
        second["candidate_evidence"]["ino"] = first["candidate_evidence"]["ino"]
    else:
        second["candidate_evidence"]["dev"] = authority["parent"]["dev"] + 1
    recovery = json.dumps(
        decoded["recovery"],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    decoded["checksum"] = sha256(recovery).hexdigest()
    forged = (
        json.dumps(decoded, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
    )

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.parse_profile_migration_journal(forged)


def test_classification_rejects_ambiguous_authority_match(tmp_path: Path) -> None:
    module = _publication_module()
    journal_module = importlib.import_module(
        "tldw_chatbook.TTS.profile_migration_journal"
    )
    identity = tmp_path.stat()
    evidence = journal_module._ArtifactEvidence(
        identity.st_dev,
        identity.st_ino,
        1,
        b"\x01" * 32,
        4,
    )
    row = module.ProfileMigrationJournalSlot._with_authority(
        slot=module.ProfileMigrationPublicationSlot.ACTIVE,
        candidate=".candidate",
        target="profiles.sqlite3",
        rollback=".profile-migration-active.rollback.sqlite3",
        had_prior=True,
        candidate_evidence=evidence,
        prior_evidence=evidence,
    )

    assert (
        row.classify_artifact(
            identity,
            byte_length=1,
            sha256_digest=b"\x01" * 32,
            schema_version=4,
        )
        is None
    )


@pytest.mark.parametrize("encoding", ["whitespace", "default_json", "uppercase"])
def test_initial_frame_rejects_noncanonical_bytes(
    tmp_path: Path,
    encoding: str,
) -> None:
    module = _publication_module()
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    canonical = _encoded_journal(module, tmp_path, rows)
    decoded = json.loads(canonical)
    if encoding == "whitespace":
        malformed = b" " + canonical
    elif encoding == "default_json":
        malformed = json.dumps(decoded).encode("ascii") + b"\n"
    else:
        decoded["recovery"]["authority"]["slots"][0]["candidate_evidence"]["sha256"] = (
            "AB" * 32
        )
        recovery = json.dumps(
            decoded["recovery"],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        decoded["checksum"] = sha256(recovery).hexdigest()
        malformed = (
            json.dumps(decoded, separators=(",", ":"), sort_keys=True).encode("ascii")
            + b"\n"
        )

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.parse_profile_migration_journal(malformed)


def test_journal_authority_encoder_is_not_public() -> None:
    module = _publication_module()

    assert not hasattr(module, "encode_profile_migration_journal")


@pytest.mark.parametrize("boundary", ["write", "file_fsync", "dir_fsync"])
def test_initial_journal_failure_removes_exact_partial_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    module = _publication_module()
    journal_path = tmp_path / ".profiles.migration-publication.json"
    payload = _encoded_journal(
        module,
        tmp_path,
        (
            module.ProfileMigrationJournalSlot(
                slot=module.ProfileMigrationPublicationSlot.ACTIVE,
                candidate=".profile-migration-active.candidate.sqlite3",
                target="profiles.sqlite3",
                rollback=".profile-migration-active.rollback.sqlite3",
                had_prior=True,
            ),
        ),
    )
    real_write = module.os.write
    real_fsync = module.os.fsync
    journal_fd = -1
    journal_file_fsynced = False
    failed = False

    def fault_write(file_fd: int, payload: bytes) -> int:
        nonlocal failed, journal_fd
        journal_fd = file_fd
        if boundary == "write" and not failed:
            failed = True
            raise OSError(errno.ENOSPC, "PRIVATE disk full")
        return real_write(file_fd, payload)

    def fault_fsync(file_fd: int) -> None:
        nonlocal failed, journal_fd, journal_file_fsynced
        if stat.S_ISREG(os.fstat(file_fd).st_mode):
            journal_fd = file_fd
        if file_fd == journal_fd:
            if boundary == "file_fsync" and not failed:
                failed = True
                raise OSError(errno.ENOSPC, "PRIVATE file fsync")
            journal_file_fsynced = True
        elif boundary == "dir_fsync" and journal_file_fsynced and not failed:
            failed = True
            raise OSError(errno.ENOSPC, "PRIVATE directory fsync")
        real_fsync(file_fd)

    monkeypatch.setattr(module.os, "write", fault_write)
    monkeypatch.setattr(module.os, "fsync", fault_fsync)

    with pytest.raises(OSError):
        module._write_new_journal(journal_path, payload)

    assert failed
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    tombstone = tmp_path / ".profile-migration-journal.tombstone"
    assert stat.S_IMODE(tombstone.stat().st_mode) == 0o600
    tombstone_identity = tombstone.stat()
    if boundary == "write":
        assert tombstone.read_bytes() == b""
        identity = module._write_new_journal(journal_path, payload)
        assert journal_path.read_bytes() == payload
        assert identity.file.st_ino == tombstone_identity.st_ino
        assert not tombstone.exists()
    else:
        assert tombstone.read_bytes() == payload
        with pytest.raises(Exception):
            module._write_new_journal(journal_path, payload)
        assert tombstone.stat().st_ino == tombstone_identity.st_ino


def test_initial_journal_cleanup_preserves_foreign_holding_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _publication_module()
    journal_path = tmp_path / ".profiles.migration-publication.json"
    payload = b"private partial journal bytes"
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    real_write = module.os.write
    real_rename = namespace._rename_noreplace
    foreign = b"foreign holding bytes"
    failed = False

    def partial_write(file_fd: int, value: bytes) -> int:
        nonlocal failed
        if not failed:
            failed = True
            real_write(file_fd, value[:5])
            raise OSError(errno.ENOSPC, "PRIVATE write failure")
        return real_write(file_fd, value)

    def occupy_hold(parent_fd: int, source: str, destination: str) -> None:
        if (
            source == journal_path.name
            and destination == ".profile-migration-journal.tombstone"
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
        real_rename(parent_fd, source, destination)

    monkeypatch.setattr(module.os, "write", partial_write)
    monkeypatch.setattr(namespace, "_rename_noreplace", occupy_hold)

    with pytest.raises(OSError, match="PRIVATE write failure"):
        module._write_new_journal(journal_path, payload)

    assert journal_path.read_bytes() == payload[:5]
    assert (tmp_path / ".profile-migration-journal.tombstone").read_bytes() == foreign


def test_tombstone_reuse_rename_cancellation_restores_bounded_reusable_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _publication_module()
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    journal_path = tmp_path / ".profiles.migration-publication.json"
    tombstone = tmp_path / ".profile-migration-journal.tombstone"
    tombstone.touch(mode=0o600)
    tombstone.chmod(0o600)
    tombstone_identity = tombstone.stat()
    cancellation = asyncio.CancelledError("PRIVATE tombstone reuse")
    real_rename = namespace._rename_noreplace
    interrupted = False

    def interrupt_after_reuse(parent_fd: int, source: str, destination: str) -> None:
        nonlocal interrupted
        real_rename(parent_fd, source, destination)
        if source == tombstone.name and not interrupted:
            interrupted = True
            raise cancellation

    monkeypatch.setattr(namespace, "_rename_noreplace", interrupt_after_reuse)

    with pytest.raises(asyncio.CancelledError) as caught:
        module._write_new_journal(journal_path, b"journal payload")

    assert caught.value is cancellation
    assert not journal_path.exists()
    assert tombstone.read_bytes() == b""
    assert tombstone.stat().st_ino == tombstone_identity.st_ino

    identity = module._write_new_journal(journal_path, b"journal payload")

    assert journal_path.read_bytes() == b"journal payload"
    assert identity.file.st_ino == tombstone_identity.st_ino
    assert not tombstone.exists()


def test_repeated_arbitrary_preflight_candidates_keep_closed_tombstone_set(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    sources = tuple(tmp_path / f".arbitrary-{index}.sqlite3" for index in range(5))
    before = tuple(
        _store(path, version=4, marker=str(index)) for index, path in enumerate(sources)
    )

    for path in sources:
        with pytest.raises(ProfileRepositoryError, match="migration_failed"):
            module.prepare_profile_migration_artifact(
                path,
                slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            )

    assert tuple(path.read_bytes() for path in sources) == before
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob("*.tombstone"))


@pytest.mark.parametrize("boundary", ["write", "file_fsync"])
def test_append_failure_preserves_last_recognized_journal_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    module = _publication_module()
    journal_path = tmp_path / ".profiles.migration-publication.json"
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    prepared, publishing = _encoded_journal(
        module,
        tmp_path,
        rows,
        "publishing",
    ).splitlines(keepends=True)
    identity = module._write_new_journal(journal_path, prepared)
    real_write = module.os.write
    real_fsync = module.os.fsync
    journal_fd = -1
    failed = False

    def fault_write(file_fd: int, payload: bytes) -> int:
        nonlocal failed, journal_fd
        journal_fd = file_fd
        if boundary == "write" and not failed:
            failed = True
            midpoint = max(1, len(payload) // 2)
            real_write(file_fd, payload[:midpoint])
            raise OSError(errno.ENOSPC, "PRIVATE append disk full")
        return real_write(file_fd, payload)

    def fault_fsync(file_fd: int) -> None:
        nonlocal failed
        if boundary == "file_fsync" and file_fd == journal_fd and not failed:
            failed = True
            raise OSError(errno.ENOSPC, "PRIVATE append fsync")
        real_fsync(file_fd)

    monkeypatch.setattr(module.os, "write", fault_write)
    monkeypatch.setattr(module.os, "fsync", fault_fsync)

    with pytest.raises(OSError):
        module._append_journal(journal_path, identity, publishing)

    parsed = module.parse_profile_migration_journal(journal_path.read_bytes())
    assert parsed.phase in {"prepared", "publishing"}
    assert parsed.recovery_rows == rows


def test_parser_uses_last_valid_prefix_before_crash_suffix(tmp_path: Path) -> None:
    module = _publication_module()
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    raw = (
        _encoded_journal(module, tmp_path, rows, "publishing") + b'{"checksum":"partial'
    )

    parsed = module.parse_profile_migration_journal(raw)

    assert parsed.phase == "publishing"
    assert parsed.recovery_rows == rows


@pytest.mark.parametrize(
    "terminated_suffix",
    [
        b"{}\n",
        b"garbage\n",
    ],
)
def test_parser_rejects_every_invalid_terminated_suffix(
    tmp_path: Path,
    terminated_suffix: bytes,
) -> None:
    module = _publication_module()
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    prepared = _encoded_journal(module, tmp_path, rows)

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.parse_profile_migration_journal(prepared + terminated_suffix)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_parser_rejects_canonical_but_illegal_complete_transition() -> None:
    module = _publication_module()
    rows = (
        module.ProfileMigrationJournalSlot(
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            candidate=".profile-migration-active.candidate.sqlite3",
            target="profiles.sqlite3",
            rollback=".profile-migration-active.rollback.sqlite3",
            had_prior=True,
        ),
    )
    prepared = _encoded_journal(module, Path.cwd(), rows)
    authority_checksum = bytes.fromhex(json.loads(prepared)["checksum"])
    complete = module._encode_later_journal(
        authority_checksum,
        phase="complete",
    )

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.parse_profile_migration_journal(prepared + complete)


def test_cleanup_rejects_replacement_parent_even_for_exact_moved_inode(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    owner = tmp_path / "owner"
    owner.mkdir(mode=0o700)
    authority = owner / "authority.sqlite3"
    authority.write_bytes(b"private authority")
    authority.chmod(0o600)
    file_identity = authority.stat()
    parent_identity = owner.stat()
    displaced = tmp_path / "displaced"
    owner.rename(displaced)
    owner.mkdir(mode=0o700)
    moved_authority = displaced / authority.name
    moved_authority.rename(authority)
    assert authority.stat().st_ino == file_identity.st_ino

    assert (
        module._unlink_exact(
            authority,
            file_identity,
            module.ParentAuthority(parent_identity),
            module.MigrationTombstoneKey.ACTIVE_CANDIDATE,
        )
        is False
    )
    assert authority.read_bytes() == b"private authority"


def test_publication_slot_state_repr_redacts_rollback_path(tmp_path: Path) -> None:
    module = _publication_module()
    private_path = tmp_path / "PRIVATE rollback evidence"

    state = module._PublicationSlotState(
        object(),
        object(),
        private_path,
        module.ParentAuthority(tmp_path.stat()),
    )

    assert str(private_path) not in repr(state)
    assert "PRIVATE" not in repr(state)


def test_prepublication_failure_is_bounded_and_context_free(tmp_path: Path) -> None:
    module = _publication_module()
    active_path = tmp_path / "PRIVATE profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
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
    active_candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    pre_v3_candidate_path = tmp_path / ".profile-migration-pre-v3.candidate.sqlite3"
    pre_v4_candidate_path = tmp_path / ".profile-migration-pre-v4.candidate.sqlite3"
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
    final_journal: list[bytes] = []

    def record(stage: object) -> None:
        events.append(stage)
        if stage is module.ProfileMigrationPublicationStage.FINAL_JOURNAL_DURABLE:
            final_journal.append(
                next(tmp_path.glob("*.migration-publication.json")).read_bytes()
            )

    module.publish_profile_migration(
        active_candidate=artifacts[0],
        backup_candidates=artifacts[1:],
        active_destination=destinations[0],
        backup_destinations=destinations[1:],
        stage_hook=record,
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
    assert len(final_journal[0]) <= module.MAX_PROFILE_MIGRATION_JOURNAL_BYTES
    assert module.parse_profile_migration_journal(final_journal[0]).phase == "complete"
    assert active_path.read_bytes() == expected_active
    assert pre_v3_path.read_bytes() == expected_pre_v3 != old_pre_v3
    assert pre_v4_path.read_bytes() == expected_pre_v4
    assert not active_candidate_path.exists()
    assert not pre_v3_candidate_path.exists()
    assert not pre_v4_candidate_path.exists()
    assert not tuple(tmp_path.glob("*.migration-publication.json"))
    assert not tuple(tmp_path.glob(".profile-migration-*.rollback.sqlite3"))


@pytest.mark.parametrize("cleanup_failure", ["false", "exception"])
def test_completed_rollback_cleanup_failure_retains_complete_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_failure: str,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    expected = _store(candidate_path, version=4, marker="new")
    _store(active_path, version=3, marker="old")
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    real_unlink = module._unlink_exact

    def fail_rollback_cleanup(
        path: Path,
        identity: object,
        parent_identity: object,
    ) -> bool:
        if path.name.endswith(".profile-migration-active.rollback.sqlite3"):
            if cleanup_failure == "exception":
                raise OSError("PRIVATE cleanup failure")
            return False
        return real_unlink(path, identity, parent_identity)

    monkeypatch.setattr(module, "_unlink_exact", fail_rollback_cleanup)

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert active_path.read_bytes() == expected
    assert tuple(tmp_path.glob(".profile-migration-*.rollback.sqlite3"))
    journal = next(tmp_path.glob("*.migration-publication.json"))
    assert (
        module.parse_profile_migration_journal(journal.read_bytes()).phase == "complete"
    )


@pytest.mark.parametrize(
    ("fault_stage", "expected_phase"),
    [("JOURNAL_DURABLE", "prepared"), ("ACTIVE_REOPENED", "restoring")],
)
def test_candidate_cleanup_failure_retains_recovery_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_stage: str,
    expected_phase: str,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    active_before = _store(active_path, version=3, marker="old")
    candidate_before = _store(candidate_path, version=4, marker="new")
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    real_unlink = module._unlink_exact

    def fail_candidate_cleanup(
        path: Path,
        identity: object,
        parent_identity: object,
        tombstone_key: object,
    ) -> bool:
        if path == candidate_path:
            return False
        return real_unlink(path, identity, parent_identity, tombstone_key)

    monkeypatch.setattr(module, "_unlink_exact", fail_candidate_cleanup)

    def fail(stage: object) -> None:
        if stage is getattr(module.ProfileMigrationPublicationStage, fault_stage):
            raise RuntimeError("PRIVATE phase failure")

    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
            stage_hook=fail,
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert active_path.read_bytes() == active_before
    assert candidate_path.read_bytes() == candidate_before
    journal = next(tmp_path.glob("*.migration-publication.json"))
    assert (
        module.parse_profile_migration_journal(journal.read_bytes()).phase
        == expected_phase
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
        tmp_path / ".profile-migration-active.candidate.sqlite3",
        tmp_path / ".profile-migration-pre-v3.candidate.sqlite3",
        tmp_path / ".profile-migration-pre-v4.candidate.sqlite3",
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
    assert not tuple(tmp_path.glob(".profile-migration-*.rollback.sqlite3"))


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
        tmp_path / ".profile-migration-active.candidate.sqlite3",
        tmp_path / ".profile-migration-pre-v3.candidate.sqlite3",
        tmp_path / ".profile-migration-pre-v4.candidate.sqlite3",
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
    assert not tuple(tmp_path.glob(".profile-migration-*.rollback.sqlite3"))


def test_completion_and_restoration_failure_retains_recovery_set_and_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "PRIVATE profiles.sqlite3"
    backup_path = tmp_path / "PRIVATE profiles.pre-v4.sqlite3"
    active_candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    backup_candidate_path = tmp_path / ".profile-migration-pre-v4.candidate.sqlite3"
    _store(active_path, version=3, marker="old-active")
    backup_before = _store(backup_path, version=3, marker="old-backup")
    candidate_before = _store(
        active_candidate_path,
        version=4,
        marker="new-active",
    )
    prepared_backup_before = _store(
        backup_candidate_path,
        version=3,
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
        slot.PRE_V4,
        "new-backup",
    )
    active = _retained(module, active_path, slot.ACTIVE, "old-active")
    backup = _retained(module, backup_path, slot.PRE_V4, "old-backup")
    real_rename = module._rename_exact

    def fail_active_rollback(
        source: object, destination: Path, parent_authority: object
    ) -> object:
        if str(source._path).endswith(".profile-migration-active.rollback.sqlite3"):
            raise OSError(f"PRIVATE total storage failure at {tmp_path}")
        return real_rename(source, destination, parent_authority)

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
    active_rollback = tmp_path / ".profile-migration-active.rollback.sqlite3"
    assert active_rollback.is_file()
    assert backup_path.read_bytes() == backup_before
    assert not active_path.exists()


def test_unavailable_dominates_deferred_cancellation_when_restore_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "PRIVATE profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    cancellation = asyncio.CancelledError("PRIVATE deferred cancellation")
    real_validate = module._immutable_validate
    real_rename = module._rename_exact

    def fail_completion(identity: object) -> None:
        if identity._path == active_path and identity._slot is slot.ACTIVE:
            if identity._schema_version == 4:
                raise OSError("PRIVATE completion failure")
        real_validate(identity)

    def fail_restore(source: object, target: Path, parent_authority: object) -> object:
        if str(source._path).endswith(".profile-migration-active.rollback.sqlite3"):
            raise OSError("PRIVATE rollback failure")
        return real_rename(source, target, parent_authority)

    monkeypatch.setattr(module, "_immutable_validate", fail_completion)
    monkeypatch.setattr(module, "_rename_exact", fail_restore)

    def cancel(stage: object) -> None:
        if stage is module.ProfileMigrationPublicationStage.PONR:
            raise cancellation

    with pytest.raises(ProfileRepositoryError, match="unavailable") as caught:
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
            stage_hook=cancel,
        )

    assert caught.value.code == "unavailable"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "PRIVATE" not in repr(caught.value)
    assert next(tmp_path.glob("*.migration-publication.json")).is_file()
    assert next(tmp_path.glob(".profile-migration-active.rollback.sqlite3")).is_file()


@pytest.mark.parametrize("race_leaf", ["rollback", "destination"])
def test_atomic_publication_race_never_clobbers_foreign_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race_leaf: str,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    rollback_path = tmp_path / ".profile-migration-active.rollback.sqlite3"
    active_before = _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    foreign = b"foreign-race-entry"
    selected = rollback_path if race_leaf == "rollback" else active_path
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    real_rename = namespace._rename_noreplace
    injected = False

    def race_rename(parent_fd: int, source: str, target: str) -> None:
        nonlocal injected
        if not injected and target == selected.name:
            injected = True
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                os.write(descriptor, foreign)
            finally:
                os.close(descriptor)
        real_rename(parent_fd, source, target)

    monkeypatch.setattr(namespace, "_rename_noreplace", race_rename)

    with pytest.raises(ProfileRepositoryError) as caught:
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
        )

    assert injected is True
    assert caught.value.code == (
        "migration_failed" if race_leaf == "rollback" else "unavailable"
    )
    assert selected.read_bytes() == foreign
    if race_leaf == "rollback":
        assert active_path.read_bytes() == active_before
    else:
        assert rollback_path.read_bytes() == active_before


def test_publisher_atomic_move_gap_preserves_foreign_source_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / ".profile-migration-active.candidate.sqlite3"
    active_before = _store(active, version=3, marker="old")
    _store(candidate, version=4, marker="new")
    artifact = _prepared(module, candidate, slot.ACTIVE, "new")
    destination = _retained(module, active, slot.ACTIVE, "old")
    namespace = importlib.import_module("tldw_chatbook.TTS.profile_migration_namespace")
    real_rename = namespace._rename_noreplace
    foreign = b"foreign-source-substitution"
    swapped = False

    def swap_source(parent_fd: int, source: str, target: str) -> None:
        nonlocal swapped
        if source == candidate.name and not swapped:
            swapped = True
            held = ".held-candidate.sqlite3"
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
        real_rename(parent_fd, source, target)

    monkeypatch.setattr(namespace, "_rename_noreplace", swap_source)

    with pytest.raises(ProfileRepositoryError) as caught:
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
        )

    assert swapped
    assert caught.value.code in {"migration_failed", "unavailable"}
    assert str(caught.value) in {
        "TTS profile repository failed: migration_failed",
        "TTS profile repository failed: unavailable",
    }
    assert candidate.read_bytes() == foreign
    assert (active.is_file() and active.read_bytes() == active_before) or any(
        path.read_bytes() == active_before
        for path in tmp_path.iterdir()
        if path.is_file()
    )


def test_publisher_sqlite_validation_is_bound_to_pinned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / ".profile-migration-active.candidate.sqlite3"
    foreign = tmp_path / "foreign.sqlite3"
    active_before = _store(active, version=3, marker="old")
    _store(candidate, version=4, marker="new")
    _store(foreign, version=4, marker="foreign")
    artifact = _prepared(module, candidate, slot.ACTIVE, "new")
    destination = _retained(module, active, slot.ACTIVE, "old")
    real_connect = module.connect_private_sqlite_descriptor
    swapped = False

    def swap_during_open(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal swapped
        if not swapped:
            swapped = True
            candidate.rename(tmp_path / ".held-candidate.sqlite3")
            foreign.rename(candidate)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(module, "connect_private_sqlite_descriptor", swap_during_open)

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
        )

    assert swapped
    assert active.read_bytes() == active_before
    assert candidate.is_file()
    assert not tuple(tmp_path.glob("*.migration-publication.json"))


@pytest.mark.parametrize("mutated_owner", ["candidate", "retained"])
def test_same_inode_sqlite_mutation_is_rejected_by_pinned_content_evidence(
    tmp_path: Path,
    mutated_owner: str,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    active_before = _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    artifact = _prepared(module, candidate_path, slot.ACTIVE, "new")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    selected = candidate_path if mutated_owner == "candidate" else active_path
    inode = selected.stat().st_ino
    connection = sqlite3.connect(selected)
    try:
        mutated_id = 100 if mutated_owner == "candidate" else 200
        connection.execute(f"PRAGMA application_id = {mutated_id}")
        connection.commit()
    finally:
        connection.close()
    assert selected.stat().st_ino == inode
    with pytest.raises(ProfileRepositoryError, match="migration_failed") as caught:
        module.publish_profile_migration(
            active_candidate=artifact,
            backup_candidates=(),
            active_destination=destination,
            backup_destinations=(),
        )

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    if mutated_owner == "candidate":
        assert active_path.read_bytes() == active_before
    else:
        check = sqlite3.connect(active_path)
        try:
            assert check.execute("PRAGMA application_id").fetchone() == (200,)
        finally:
            check.close()


@pytest.mark.parametrize(
    ("slot_name", "wrong_version"),
    [("ACTIVE", 3), ("PRE_V3", 3), ("PRE_V4", 2)],
)
def test_preparer_rejects_wrong_schema_by_slot(
    tmp_path: Path,
    slot_name: str,
    wrong_version: int,
) -> None:
    module = _publication_module()
    wrong = tmp_path / f"wrong-{slot_name}.sqlite3"
    _store(wrong, version=wrong_version, marker="wrong")

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.prepare_profile_migration_artifact(
            wrong,
            slot=getattr(module.ProfileMigrationPublicationSlot, slot_name),
        )


def test_preparer_disallows_validator_injection(tmp_path: Path) -> None:
    module = _publication_module()
    candidate = tmp_path / ".profile-migration-active.candidate.sqlite3"
    _store(candidate, version=4, marker="candidate")

    with pytest.raises(TypeError):
        module.prepare_profile_migration_artifact(
            candidate,
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            validate=lambda _connection: None,
        )


def test_publication_rejects_pre_v3_without_required_pre_v4_boundary(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    pre_v3_path = tmp_path / ".profile-migration-pre-v3.candidate.sqlite3"
    pre_v3_target = tmp_path / "profiles.pre-v3.sqlite3"
    _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    _store(pre_v3_path, version=2, marker="pre-v3")
    active = _prepared(module, candidate_path, slot.ACTIVE, "new")
    boundary = _prepared(module, pre_v3_path, slot.PRE_V3, "pre-v3")
    destination = _retained(module, active_path, slot.ACTIVE, "old")
    boundary_target = _retained(module, pre_v3_target, slot.PRE_V3, None)

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.publish_profile_migration(
            active_candidate=active,
            backup_candidates=(boundary,),
            active_destination=destination,
            backup_destinations=(boundary_target,),
        )


def test_publication_rejects_noncanonical_boundary_order(tmp_path: Path) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    paths = {
        "active": tmp_path / ".profile-migration-active.candidate.sqlite3",
        "pre_v3": tmp_path / ".profile-migration-pre-v3.candidate.sqlite3",
        "pre_v4": tmp_path / ".profile-migration-pre-v4.candidate.sqlite3",
    }
    targets = {
        "active": tmp_path / "profiles.sqlite3",
        "pre_v3": tmp_path / "profiles.pre-v3.sqlite3",
        "pre_v4": tmp_path / "profiles.pre-v4.sqlite3",
    }
    _store(targets["active"], version=3, marker="old")
    _store(paths["active"], version=4, marker="new")
    _store(paths["pre_v3"], version=2, marker="pre-v3")
    _store(paths["pre_v4"], version=3, marker="pre-v4")
    active = _prepared(module, paths["active"], slot.ACTIVE, "new")
    pre_v3 = _prepared(module, paths["pre_v3"], slot.PRE_V3, "pre-v3")
    pre_v4 = _prepared(module, paths["pre_v4"], slot.PRE_V4, "pre-v4")
    active_target = _retained(module, targets["active"], slot.ACTIVE, "old")
    pre_v3_target = _retained(module, targets["pre_v3"], slot.PRE_V3, None)
    pre_v4_target = _retained(module, targets["pre_v4"], slot.PRE_V4, None)

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.publish_profile_migration(
            active_candidate=active,
            backup_candidates=(pre_v4, pre_v3),
            active_destination=active_target,
            backup_destinations=(pre_v4_target, pre_v3_target),
        )


def test_publication_rejects_duplicate_boundary_slot(tmp_path: Path) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    first_path = tmp_path / ".profile-migration-pre-v4.candidate.sqlite3"
    second_path = tmp_path / ".second-pre-v4.candidate"
    _store(active_path, version=3, marker="old")
    _store(candidate_path, version=4, marker="new")
    _store(first_path, version=3, marker="first")
    _store(second_path, version=3, marker="second")
    active = _prepared(module, candidate_path, slot.ACTIVE, "new")
    first = _prepared(module, first_path, slot.PRE_V4, "first")
    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        _prepared(module, second_path, slot.PRE_V4, "second")
    active_target = _retained(module, active_path, slot.ACTIVE, "old")
    first_target = _retained(module, tmp_path / "first.sqlite3", slot.PRE_V4, None)
    second_target = _retained(
        module,
        tmp_path / "second.sqlite3",
        slot.PRE_V4,
        None,
    )

    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        module.publish_profile_migration(
            active_candidate=active,
            backup_candidates=(first, first),
            active_destination=active_target,
            backup_destinations=(first_target, second_target),
        )


def test_opaque_artifact_constructor_cannot_be_forged(tmp_path: Path) -> None:
    module = _publication_module()
    path = tmp_path / ".profile-migration-active.candidate.sqlite3"
    _store(path, version=4, marker="candidate")

    with pytest.raises(TypeError):
        module.PreparedProfileMigrationArtifact(
            path=path,
            slot=module.ProfileMigrationPublicationSlot.ACTIVE,
            parent_identity=tmp_path.stat(),
            file_identity=path.stat(),
            schema_version=4,
        )


def test_reentrant_double_publication_is_rejected_without_disrupting_owner(
    tmp_path: Path,
) -> None:
    module = _publication_module()
    slot = module.ProfileMigrationPublicationSlot
    active_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
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
    assert not tuple(tmp_path.glob(".profile-migration-*.rollback.sqlite3"))


@pytest.mark.parametrize(
    "raw",
    [
        b'PRIVATE {"phase":"publishing","slots":[],"version":1}\n',
        b'{"phase":"unknown","slots":[],"version":1}\n',
        b'{"phase":"prepared","slots":[{"candidate":"../private",'
        b'"had_prior":true,"rollback":"r","slot":"active",'
        b'"target":"t"}],"version":1}\n',
        b'{"phase":"prepared","slots":[{"candidate":"candidate",'
        b'"had_prior":true,"rollback":"not-the-authority-rollback",'
        b'"slot":"active","target":"target"}],"version":1}\n',
        b'{"phase":"prepared","slots":[{"candidate":"target",'
        b'"had_prior":true,"rollback":".target.active.rollback",'
        b'"slot":"active","target":"target"}],"version":1}\n',
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
    candidate_path = tmp_path / ".profile-migration-active.candidate.sqlite3"
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
