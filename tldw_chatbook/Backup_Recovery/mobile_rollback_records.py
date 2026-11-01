"""Fixed journal grammar for restoring authenticated mobile before-images."""

import hashlib
import json
import stat
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from .archive_models import Metadata
from .journal import (
    _MOBILE_OWNERS,
    MAX_EVENTS,
    _Directory,
    _DirectoryState,
    _Evidence,
    _mobile_ref,
    _mobile_referenced,
    _mobile_state,
    _MobileRef,
    _MobileTarget,
    _MobileWitness,
    _MoveParent,
    _Object,
)
from .limits import ArchiveLimits


class _RollbackRecord(_Evidence):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    version: int = Field(ge=1, le=1)


class _RollbackStarted(_RollbackRecord):
    prepared: _MobileRef
    claim: _MobileRef
    canonical_capture: _MobileRef
    source_fingerprint: str = Field(pattern=r"\A[0-9a-f]{64}\z")
    destination_identity: str = Field(pattern=r"\A[0-9a-f]{64}\z")


class _RollbackStage(_RollbackRecord):
    started: _MobileRef
    owner_id: Literal["db.chachanotes.primary", "db.workspaces", "db.agent_runs"]
    logical_id: str = Field(min_length=1, max_length=1024)
    member_path: str = Field(min_length=1)
    payload_sha256: str = Field(pattern=r"\A[0-9a-f]{64}\z")
    payload_size: int = Field(ge=0, le=ArchiveLimits().member_bytes)
    target: str
    temporary: str
    parent: _Directory
    metadata: Metadata

    @field_validator("target", "temporary")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value)

    @field_validator("member_path")
    @classmethod
    def member(cls, value):
        from .archive_reader import _path

        _path(value, ArchiveLimits())
        return value


class _RollbackStageObserved(_RollbackRecord):
    intended: _MobileRef
    staged: _Object


class _Subject(_Evidence):
    owner_id: Literal["db.chachanotes.primary", "db.workspaces", "db.agent_runs"] | None
    component: Literal["main", "wal", "shm", "journal", "history", "created_parent"]
    path: str

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value)


class _RollbackMove(_RollbackRecord):
    started: _MobileRef
    ordinal: int = Field(ge=0, lt=MAX_EVENTS)
    subject: _Subject
    step: Literal["retain_current", "restore_snapshot", "retain_created"]
    source: _Object
    destination: str
    parents: list[_MoveParent] = Field(min_length=1, max_length=2)

    @field_validator("destination")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value)


class _RollbackMoveObserved(_RollbackRecord):
    intended: _MobileRef
    destination: _Object
    parents: list[_DirectoryState] = Field(min_length=1, max_length=2)


class _RollbackMetadata(_RollbackRecord):
    started: _MobileRef
    subject: _Subject
    object: _Object
    previous: Metadata
    applied: Metadata
    parent: _Directory


class _RollbackMetadataObserved(_RollbackRecord):
    intended: _MobileRef
    object: _Object
    metadata: Metadata


class _RestoreFinal(_Evidence):
    owner_id: Literal["db.chachanotes.primary", "db.workspaces", "db.agent_runs"]
    stage: _MobileRef | None
    current: _MobileWitness
    snapshot_sha256: str | None = Field(pattern=r"\A[0-9a-f]{64}\z")


class _RolledBack(_RollbackRecord):
    started: _MobileRef
    source_fingerprint: str = Field(pattern=r"\A[0-9a-f]{64}\z")
    destination_identity: str = Field(pattern=r"\A[0-9a-f]{64}\z")
    targets: list[_MobileTarget] = Field(min_length=3, max_length=3)
    restored: list[_RestoreFinal] = Field(min_length=3, max_length=3)
    retained: list[_MobileRef] = Field(max_length=781)
    pending: _Object
    bootstrap: _Directory


_MODELS = {
    "mobile_rollback_started": _RollbackStarted,
    "mobile_rollback_stage_intended": _RollbackStage,
    "mobile_rollback_stage_observed": _RollbackStageObserved,
    "mobile_rollback_move_intended": _RollbackMove,
    "mobile_rollback_move_observed": _RollbackMoveObserved,
    "mobile_rollback_metadata_intended": _RollbackMetadata,
    "mobile_rollback_metadata_observed": _RollbackMetadataObserved,
    "mobile_rolled_back": _RolledBack,
}


def _stage_name(operation_id, owner_id):
    return (
        ".mobile-rollback-"
        + hashlib.sha256(operation_id.encode()).hexdigest()
        + "-"
        + str(_MOBILE_OWNERS.index(owner_id))
        + ".db"
    )


def _retained_name(operation_id, ordinal):
    return (
        ".mobile-retained-"
        + hashlib.sha256(operation_id.encode()).hexdigest()
        + "-"
        + str(ordinal)
    )


def _subject(value, prepared, prior):
    path = Path(value.path)
    if value.component in {"main", "wal", "shm", "journal"}:
        target = next(
            (row for row in prepared.targets if row.owner_id == value.owner_id), None
        )
        suffix = {"main": "", "wal": "-wal", "shm": "-shm", "journal": "-journal"}[
            value.component
        ]
        if target is None or str(path) != target.witnesses[-1].path + suffix:
            raise ValueError("mobile_rollback_subject_invalid")
    elif value.owner_id is not None:
        raise ValueError("mobile_rollback_subject_invalid")
    elif value.component == "history":
        if [
            row.evidence["publication"]
            for row in prior
            if row.event == "mobile_history_intended"
        ] != [str(path)]:
            raise ValueError("mobile_rollback_subject_invalid")
    elif not any(
        row.path == str(path) and row.inode is None
        for target in prepared.targets
        for row in target.witnesses[:-1]
    ) or not any(
        row.event == "mobile_path_observed"
        and row.evidence["after"]["path"] == str(path)
        and stat.S_ISDIR(row.evidence["after"]["mode"])
        for row in prior
    ):
        raise ValueError("mobile_rollback_subject_invalid")


def _physical(value):
    """Apply new uint64 bounds without changing historical journal domains."""
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key in {"device", "inode"}
                and item is not None
                and (type(item) is not int or not 0 <= item < 2**64)
            ):
                raise ValueError("mobile_rollback_physical_scalar_invalid")
            _physical(item)
    elif isinstance(value, list):
        for item in value:
            _physical(item)


def validate_rollback(event, evidence, prior):
    """Validate only literal same-operation snapshot restoration transitions."""
    value = _MODELS[event].model_validate(evidence)
    _physical(evidence)
    if (
        not prior
        or prior[0].event != "mobile_prepared"
        or any(row.event in {"mobile_committed", "mobile_rolled_back"} for row in prior)
    ):
        raise ValueError("mobile_rollback_unavailable")
    prepared, witnesses = _mobile_state(prior)
    started = [row for row in prior if row.event == "mobile_rollback_started"]
    if event == "mobile_rollback_started":
        if started or prior[-1].event not in {
            "mobile_capture_verified",
            "mobile_owner_retired",
            "mobile_history_verified",
        }:
            raise ValueError("mobile_rollback_unsettled")
        _mobile_referenced(value.prepared, prior, "mobile_prepared")
        _mobile_referenced(value.claim, prior, "mobile_claim_observed")
        capture = _mobile_referenced(
            value.canonical_capture, prior, "mobile_capture_verified"
        )
        claim = next(
            row.evidence for row in prior if row.event == "mobile_claim_intended"
        )
        if (
            capture.evidence["purpose"] != "canonical"
            or value.source_fingerprint != prepared.source_fingerprint
            or value.destination_identity != claim["destination_identity"]
        ):
            raise ValueError("mobile_rollback_scope_changed")
        return value.model_dump()
    if len(started) != 1:
        raise ValueError("mobile_rollback_not_started")
    if hasattr(value, "started") and value.started.model_dump() != _mobile_ref(
        started[0]
    ):
        raise ValueError("mobile_rollback_scope_changed")
    last = prior[-1]
    expected = {
        "mobile_rollback_stage_intended": "mobile_rollback_stage_observed",
        "mobile_rollback_move_intended": "mobile_rollback_move_observed",
        "mobile_rollback_metadata_intended": "mobile_rollback_metadata_observed",
    }
    if last.event in expected:
        if event != expected[last.event] or value.intended.model_dump() != _mobile_ref(
            last
        ):
            raise ValueError("mobile_rollback_unsettled")
        intent = last.evidence
        if event == "mobile_rollback_stage_observed":
            rows = [("", "file", intent["payload_size"], intent["payload_sha256"])]
            digest = hashlib.sha256(
                json.dumps(rows, separators=(",", ":")).encode()
            ).hexdigest()
            if (
                value.staged.path,
                value.staged.kind,
                value.staged.size,
                value.staged.sha256,
            ) != (intent["temporary"], "file", intent["payload_size"], digest):
                raise ValueError("mobile_rollback_stage_changed")
        elif event == "mobile_rollback_move_observed":
            if value.destination.model_dump() != {
                **intent["source"],
                "path": intent["destination"],
            } or [row.path for row in value.parents] != [
                row["state"]["path"] for row in intent["parents"]
            ]:
                raise ValueError("mobile_rollback_move_changed")
            for actual, previous in zip(value.parents, intent["parents"], strict=True):
                if (actual.device, actual.inode, actual.mode) != tuple(
                    previous["state"][key] for key in ("device", "inode", "mode")
                ):
                    raise ValueError("mobile_rollback_parent_changed")
        elif (
            value.object.model_dump() != intent["object"]
            or value.metadata.model_dump() != intent["applied"]
        ):
            raise ValueError("mobile_rollback_metadata_changed")
    elif event == "mobile_rollback_stage_intended":
        stages = [row for row in prior if row.event == event]
        target = next(row for row in prepared.targets if row.owner_id == value.owner_id)
        if (
            len(stages) >= 3
            or any(row.evidence["owner_id"] == value.owner_id for row in stages)
            or target.witnesses[-1].inode is None
            or value.target != target.witnesses[-1].path
            or Path(value.temporary)
            != Path(value.target).parent
            / _stage_name(prior[0].operation_id, value.owner_id)
            or value.parent.path != str(Path(value.target).parent)
            or any(row.event == "mobile_rollback_move_intended" for row in prior)
        ):
            raise ValueError("mobile_rollback_stage_invalid")
    elif event == "mobile_rollback_move_intended":
        _subject(value.subject, prepared, prior)
        required = {
            target.owner_id
            for target in prepared.targets
            if target.witnesses[-1].inode is not None
        }
        staged = {
            prior[row.evidence["intended"]["sequence"]].evidence["owner_id"]
            for row in prior
            if row.event == "mobile_rollback_stage_observed"
        }
        if required != staged:
            raise ValueError("mobile_rollback_stage_missing")
        moves = [row for row in prior if row.event == event]
        if (
            value.ordinal != len(moves)
            or len(moves) >= 784
            or any(
                row.evidence["subject"] == value.subject.model_dump()
                and row.evidence["step"] == value.step
                for row in moves
            )
        ):
            raise ValueError("mobile_rollback_move_invalid")
        if value.step == "restore_snapshot":
            stages = [
                row
                for row in prior
                if row.event == "mobile_rollback_stage_intended"
                and row.evidence["owner_id"] == value.subject.owner_id
            ]
            if (
                value.subject.component != "main"
                or len(stages) != 1
                or value.source.path != stages[0].evidence["temporary"]
                or value.destination != value.subject.path
            ):
                raise ValueError("mobile_rollback_move_invalid")
        elif (
            value.source.path != value.subject.path
            or value.destination
            != str(
                Path(value.subject.path).parent
                / _retained_name(prior[0].operation_id, value.ordinal)
            )
            or (value.step == "retain_created")
            != (value.subject.component == "created_parent")
        ):
            raise ValueError("mobile_rollback_move_invalid")
        if {row.state.path for row in value.parents} != {
            str(Path(value.source.path).parent),
            str(Path(value.destination).parent),
        }:
            raise ValueError("mobile_rollback_parent_changed")
    elif event == "mobile_rollback_metadata_intended":
        _subject(value.subject, prepared, prior)
        stages = [
            row
            for row in prior
            if row.event == "mobile_rollback_stage_intended"
            and row.evidence["owner_id"] == value.subject.owner_id
        ]
        if (
            value.subject.component != "main"
            or value.object.path != value.subject.path
            or len(stages) != 1
            or value.applied.model_dump() != stages[0].evidence["metadata"]
            or value.parent.path != str(Path(value.object.path).parent)
            or any(
                row.event == event
                and row.evidence["subject"] == value.subject.model_dump()
                for row in prior
            )
        ):
            raise ValueError("mobile_rollback_metadata_invalid")
    elif event == "mobile_rolled_back":
        _validate_final(value, prior, prepared, witnesses, started[0])
    else:
        raise ValueError("mobile_rollback_transition_invalid")
    return value.model_dump()


def _validate_final(value, prior, prepared, witnesses, started):
    from .journal import _validate_mobile_sidecars

    if (
        value.source_fingerprint != started.evidence["source_fingerprint"]
        or value.destination_identity != started.evidence["destination_identity"]
        or tuple(row.owner_id for row in value.targets) != _MOBILE_OWNERS
        or tuple(row.owner_id for row in value.restored) != _MOBILE_OWNERS
    ):
        raise ValueError("mobile_rollback_final_invalid")
    for target, original, restored in zip(
        value.targets, prepared.targets, value.restored, strict=True
    ):
        _validate_mobile_sidecars(original.witnesses[-1].path, target.sidecars)
        if [row.model_dump() for row in target.witnesses] != [
            witnesses[row.path] for row in original.witnesses
        ] or restored.current != target.witnesses[-1]:
            raise ValueError("mobile_rollback_final_invalid")
        if original.witnesses[-1].inode is None:
            if (
                restored.stage is not None
                or restored.snapshot_sha256 is not None
                or restored.current.inode is not None
            ):
                raise ValueError("mobile_rollback_absence_invalid")
        else:
            if restored.stage is None:
                raise ValueError("mobile_rollback_stage_missing")
            observed = _mobile_referenced(
                restored.stage, prior, "mobile_rollback_stage_observed"
            )
            stage = prior[observed.evidence["intended"]["sequence"]]
            if (
                stage.evidence["owner_id"] != target.owner_id
                or restored.snapshot_sha256 != stage.evidence["payload_sha256"]
                or restored.current.inode != observed.evidence["staged"]["inode"]
            ):
                raise ValueError("mobile_rollback_stage_changed")
        if any(row.inode is not None for row in target.sidecars):
            raise ValueError("mobile_rollback_sidecars_remaining")
    retained = [
        row
        for row in prior
        if row.event == "mobile_rollback_move_observed"
        and prior[row.evidence["intended"]["sequence"]].evidence["step"]
        != "restore_snapshot"
    ]
    if [row.model_dump() for row in value.retained] != [
        _mobile_ref(row) for row in retained
    ]:
        raise ValueError("mobile_rollback_retention_changed")
    root = Path(prepared.bootstrap_root)
    from .bootstrap import _key

    if value.bootstrap.path != str(root) or value.pending.path != str(
        root / ("pending-" + _key(prior[0].operation_id) + ".json")
    ):
        raise ValueError("mobile_rollback_pending_changed")


def replay_witnesses(records, witnesses):
    """Replay only observed native moves and separately observed metadata."""
    for row in records:
        if row.event == "mobile_rollback_move_observed":
            intent = records[row.evidence["intended"]["sequence"]].evidence
            subject = intent["subject"]
            if subject["path"] not in witnesses:
                continue
            if intent["step"] != "restore_snapshot":
                witnesses[subject["path"]] = {
                    "path": subject["path"],
                    "device": None,
                    "inode": None,
                    "mode": None,
                }
            else:
                source = next(
                    parent["children"][Path(intent["source"]["path"]).name]
                    for parent in intent["parents"]
                    if parent["state"]["path"]
                    == str(Path(intent["source"]["path"]).parent)
                )
                witnesses[subject["path"]] = {
                    "path": subject["path"],
                    "device": row.evidence["destination"]["device"],
                    "inode": row.evidence["destination"]["inode"],
                    "mode": source["mode"],
                }
        elif row.event == "mobile_rollback_metadata_observed":
            intent = records[row.evidence["intended"]["sequence"]].evidence
            path = intent["subject"]["path"]
            if path in witnesses and witnesses[path]["inode"] is not None:
                witnesses[path] = {
                    **witnesses[path],
                    "mode": stat.S_IFREG | row.evidence["metadata"]["mode"],
                }
