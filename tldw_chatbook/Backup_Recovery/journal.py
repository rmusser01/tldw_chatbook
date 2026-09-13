"""Private durable restore intent; record labels never release admission (ADR-126)."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tldw_chatbook.Utils.platform_files import os

from .admission import Admission, fcntl
from .archive_models import Metadata
from .bootstrap import _read
from .native_files import create_private_directory, flush_directory, pinned_directory
from .native_platform import flush_file
from .qualification import qualified_for

MAX_EVENTS = 100_000


class _Evidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class _Object(_Evidence):
    path: str
    device: int = Field(ge=0)
    inode: int = Field(ge=0)
    kind: Literal["file", "directory"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0, le=1024**4)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value):
        if not Path(value).is_absolute() or ".." in Path(value).parts or "\0" in value:
            raise ValueError("artifact_path_invalid")
        return value


class _Directory(_Evidence):
    path: str
    device: int = Field(ge=0)
    inode: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value)


class _DirectoryState(_Directory):
    mode: int = Field(ge=0, le=0o777)
    mtime_ns: int = Field(ge=0)
    ctime_ns: int = Field(ge=0)


class _DirectoryRestore(_Evidence):
    logical_id: str
    owner_id: str
    previous: _DirectoryState
    parent: _Directory
    applied: Metadata


class _DirectoryIntent(_Evidence):
    logical_id: str
    before: _DirectoryState
    applied: Metadata


class _DirectoryProgress(_Evidence):
    logical_id: str
    observed: _DirectoryState


class _Artifact(_Evidence):
    logical_id: str = Field(min_length=1, max_length=1024)
    candidate: _Object | None
    target: str
    previous: _Object | None
    retained: str | None
    previous_metadata: _Object | None = None
    action: Literal["publish", "retire", "container"] = "publish"
    rollback_requires_owner: bool = False
    rollback_projection_roots: list[_Object] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )
    parents: list[_Directory] = Field(default_factory=list, max_length=3)

    @field_validator("target", "retained")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value) if value is not None else value


class _PublicationContext(_Evidence):
    bootstrap_root: str
    namespaces: list[str] = Field(min_length=1, max_length=4096)
    selectors: list[str] = Field(min_length=1, max_length=4096)
    archive_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    descriptor: _Object


class _CandidateReceipt(_Evidence):
    local_plan: _Object | None = None
    stage: _Object
    descriptor: _Object
    archive_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _PrepublicationAbort(_Evidence):
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication: _PublicationContext


class _RollbackSource(_Evidence):
    logical_id: str = Field(min_length=1, max_length=1024)
    owner_id: str = Field(min_length=1, max_length=256)
    source: _Object
    sidecars: dict[str, _Object] = Field(max_length=2)
    artifacts: list[str] = Field(min_length=1, max_length=MAX_EVENTS)


class _SqliteRollback(_RollbackSource):
    schema_version: int = Field(ge=0)
    payload_size: int = Field(ge=0, le=1024**4)
    payload_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _SafetySource(_Evidence):
    logical_id: str = Field(min_length=1, max_length=1024)
    owner_id: str = Field(min_length=1, max_length=256)
    source: _Object


class _Rollback(_Evidence):
    ciphertext: _Object
    sealed_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage: dict[str, str]
    sqlite_groups: list[_SqliteRollback] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )
    projection_groups: list[_Object] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )
    credential_issues: list[str] = Field(default_factory=list, max_length=4096)
    safety_sources: list[_SafetySource] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )


class _Progress(_Evidence):
    logical_id: str
    observed: _Object
    directories: list[_DirectoryState] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )


class _IsolatedProfile(_Evidence):
    source_profile: str = Field(min_length=1, max_length=256)
    profile_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    installation_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    config: str
    data: str

    @field_validator("config", "data")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value)


class _RetainedCredentials(_Evidence):
    ciphertext: _Object
    plaintext_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ReplacementProfile(_Evidence):
    config: str
    installation_id: str = Field(pattern=r"^[0-9a-f]{32}$")

    @field_validator("config")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value)


class _Prepared(_Evidence):
    generation: str = Field(min_length=1, max_length=256)
    mode: Literal["isolated", "replace"]
    isolated_profiles: list[_IsolatedProfile] = Field(
        default_factory=list, max_length=4096
    )
    retained_credentials: _RetainedCredentials | None = None
    replacement_profiles: list[_ReplacementProfile] = Field(
        default_factory=list, max_length=4096
    )
    safety_sources: list[_SafetySource] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )
    credential_scopes: dict[str, str] = Field(default_factory=dict, max_length=10000)
    credential_material: _Object | None = None
    incoming_credentials: _RetainedCredentials | None = None
    artifacts: list[_Artifact] = Field(default_factory=list, max_length=MAX_EVENTS)
    rollback_sources: list[_RollbackSource] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )
    publication: _PublicationContext | None = None
    directory_metadata: list[_DirectoryRestore] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )
    installed_paths: list[_Directory] = Field(
        default_factory=list, max_length=MAX_EVENTS
    )


def observe_artifact(path: Path, *, metadata: bool = False) -> dict:
    """Read bounded no-follow file/tree identity and bytes for local reconciliation.

    Modes are deliberately separate from content identity: native publication
    normalizes private modes before rename, and final metadata is validated later.
    Directory evidence includes every child and empty directory, never just names
    from an intended payload list.
    """
    count, total = 0, 0
    observed = {}

    def identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_mode,
        )

    def recheck(fd, relative):
        expected, names = observed[relative]
        if identity(os.fstat(fd)) != expected:
            raise ValueError("artifact_changed")
        if names is not None:
            if sorted(os.listdir(fd)) != names:
                raise ValueError("artifact_changed")
            for name in names:
                child = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
                try:
                    recheck(child, relative + "/" + name)
                finally:
                    os.close(child)

    def read(fd, relative, device):
        nonlocal count, total
        before = os.fstat(fd)
        names = None
        count += 1
        if (
            count > MAX_EVENTS
            or before.st_dev != device
            or before.st_uid != os.geteuid()
        ):
            raise ValueError("artifact_unverified")
        if stat.S_ISREG(before.st_mode):
            if before.st_nlink != 1 or before.st_size > 256 * 1024**3:
                raise ValueError("artifact_unverified")
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(fd, 64 * 1024):
                size += len(chunk)
                total += len(chunk)
                if size > before.st_size or total > 1024**4:
                    raise ValueError("artifact_limit")
                digest.update(chunk)
            rows = [(relative, "file", size, digest.hexdigest())]
            kind = "file"
        elif stat.S_ISDIR(before.st_mode):
            names = sorted(os.listdir(fd))
            if len(names) > MAX_EVENTS - count:
                raise ValueError("artifact_limit")
            rows = [(relative, "directory")]
            kind = "directory"
            for name in names:
                child = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
                try:
                    _, child_rows = read(child, relative + "/" + name, device)
                    rows.extend(child_rows)
                finally:
                    os.close(child)
            if names != sorted(os.listdir(fd)):
                raise ValueError("artifact_changed")
        else:
            raise ValueError("artifact_unverified")
        if metadata:
            rows[0] += (stat.S_IMODE(before.st_mode), before.st_mtime_ns)
        after = os.fstat(fd)
        if identity(before) != identity(after):
            raise ValueError("artifact_changed")
        observed[relative] = identity(after), names
        return kind, rows

    with pinned_directory(path.parent) as parent:
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(fd)
            kind, rows = read(fd, "", before.st_dev)
            recheck(fd, "")
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("artifact_changed")
        finally:
            os.close(fd)
    return _Object(
        path=str(path),
        device=before.st_dev,
        inode=before.st_ino,
        kind=kind,
        size=total,
        sha256=hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest(),
    ).model_dump()


def _matches(expected: _Object, path: str, *, metadata: bool = False) -> bool:
    try:
        actual = observe_artifact(Path(path), metadata=metadata)
    except (OSError, ValueError, RecursionError):
        return False
    return {
        key: value for key, value in actual.items() if key != "path"
    } == expected.model_dump(exclude={"path"})


def _absent(path: str) -> bool:
    try:
        with pinned_directory(Path(path).parent) as parent:
            os.stat(Path(path).name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        # An absent parent could be a disconnected volume. It is not proof that
        # an artifact is absent under the reviewed mounted parent.
        try:
            with pinned_directory(Path(path).parent):
                return True
        except OSError:
            return False
    except OSError:
        return False
    return False


def _states(prepared: _Prepared, *, logical_id: str | None = None) -> dict[str, str]:
    result = {}
    for item in prepared.artifacts:
        if logical_id is not None and item.logical_id != logical_id:
            continue
        candidate_present = item.candidate is not None and _matches(
            item.candidate, item.candidate.path
        )
        previous = item.previous_metadata or item.previous
        container_present = False
        if item.action == "container" and item.candidate is not None:
            try:
                with pinned_directory(Path(item.target)) as fd:
                    info = os.fstat(fd)
                    children = {
                        Path(other.target).relative_to(item.target).parts[0]
                        for other in prepared.artifacts
                        if Path(item.target) in Path(other.target).parents
                    }
                    container_present = (info.st_dev, info.st_ino) == (
                        item.candidate.device,
                        item.candidate.inode,
                    ) and set(os.listdir(fd)) <= children
            except OSError:
                pass

        def old_at(
            path, previous=previous, metadata=item.previous_metadata is not None
        ):
            return previous is not None and _matches(previous, path, metadata=metadata)

        old_present = old_at(item.target) if item.previous else _absent(item.target)
        if (
            (candidate_present or item.action == "retire")
            and old_present
            and (item.retained is None or _absent(item.retained))
        ):
            state = "staged"
        elif (
            (candidate_present or item.action == "retire")
            and _absent(item.target)
            and item.retained
            and old_at(item.retained)
        ):
            state = "retired"
        elif (
            item.candidate is not None
            and _absent(item.candidate.path)
            and (
                container_present
                if item.action == "container"
                else _matches(item.candidate, item.target)
            )
            and (
                old_at(item.retained)
                if item.previous and item.retained
                else item.previous is None
            )
        ):
            state = "published"
        else:
            state = "uncertain"
        result[item.logical_id] = state
    return result


class _MoveChild(_Directory):
    mode: int = Field(ge=0)
    owner: int = Field(ge=0)
    size: int = Field(ge=0)
    links: int = Field(ge=0)
    mtime_ns: int
    ctime_ns: int


class _MoveParent(_Evidence):
    state: _DirectoryState
    children: dict[str, _MoveChild] = Field(max_length=MAX_EVENTS)


class _MoveIntent(_Evidence):
    logical_id: str
    step: Literal[
        "retire",
        "publish",
        "unpublish",
        "restore",
        "credential_retire",
        "credential_publish",
        "credential_unpublish",
    ]
    source: _Object
    destination: str
    parents: list[_MoveParent] = Field(min_length=1, max_length=2)


class _MoveObserved(_Evidence):
    intent_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    moved: bool
    directories: list[_DirectoryState] = Field(max_length=MAX_EVENTS)


class _RollbackStarted(_Evidence):
    prepared_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    rollback_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: str = Field(pattern=r"^[0-9a-f]{32}$")
    profiles: list[_ReplacementProfile] = Field(min_length=1, max_length=4096)
    retained_credential_scopes: list[str] = Field(max_length=10000)


class _RollbackCredentialScope(_Evidence):
    purpose: str
    material_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    value_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _RollbackCredentialReference(_Evidence):
    record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_id: str
    target: str
    artifact_id: str
    owner: Literal["mcp.targets"]


class _RollbackCredentialArtifact(_Evidence):
    logical_id: str
    candidate: _Object
    metadata: _Object


class _RollbackCredentialPlan(_Evidence):
    rollback_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    material_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    scopes: dict[str, _RollbackCredentialScope] = Field(min_length=1, max_length=10000)
    references: list[_RollbackCredentialReference] = Field(
        min_length=1, max_length=10000
    )
    artifacts: list[_RollbackCredentialArtifact] = Field(
        min_length=1, max_length=MAX_EVENTS
    )


class _RollbackCredentialApplied(_Evidence):
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    purpose: str
    value_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _OriginalsValidated(_Evidence):
    credential_plan_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rollback_started_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: list[_Object] = Field(max_length=MAX_EVENTS)
    directories: list[_DirectoryState] = Field(max_length=MAX_EVENTS)


class _RollbackActivation(_Evidence):
    generation: str = Field(pattern=r"^[0-9a-f]{32}$")
    selectors: list[str] = Field(min_length=1, max_length=4096)
    owners: list[str] = Field(min_length=1, max_length=4096)
    originals_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: list[_Object] = Field(min_length=1, max_length=20000)


class _RolledBack(_Evidence):
    generation: str
    activation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Installed(_Evidence):
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: list[_Object] = Field(max_length=MAX_EVENTS)


def _evidence_digest(evidence: dict) -> str:
    return hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class _ActivationRecorded(_Evidence):
    generation: str = Field(min_length=1, max_length=256)
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    installed_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    selectors: list[str] = Field(min_length=1, max_length=4096)
    owners: list[str] = Field(min_length=1, max_length=4096)
    records: list[_Object] = Field(min_length=3, max_length=MAX_EVENTS)


class _CatalogRecorded(_Evidence):
    generation: str
    activation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: list[_Object] = Field(min_length=1, max_length=4096)


class _CredentialIntent(_Evidence):
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    material: _Object
    scopes: dict[str, str] = Field(min_length=1, max_length=10000)


class _CredentialApplied(_Evidence):
    record_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: Literal["create", "retain"]
    purpose: str
    value_digest: str


class _CredentialsComplete(_Evidence):
    intent_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    applied_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Committed(_Evidence):
    generation: str = Field(min_length=1, max_length=256)
    activation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    credential_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class _Event(_Evidence):
    version: Literal[1] = 1
    operation_id: str
    sequence: int = Field(ge=0, lt=MAX_EVENTS)
    previous: str
    event: Literal[
        "candidate_staged",
        "prepublication_aborted",
        "prepared",
        "rollback_verified",
        "publication_started",
        "artifact_retired",
        "artifact_published",
        "installed_validated",
        "activation_recorded",
        "catalog_registered",
        "credential_intended",
        "credential_applied",
        "credentials_completed",
        "committed",
        "move_intended",
        "move_observed",
        "rollback_started",
        "originals_validated",
        "rollback_activation_recorded",
        "rolled_back",
        "rollback_metadata_started",
        "rollback_metadata_applied",
        "rollback_credentials_planned",
        "rollback_credential_applied",
        "directory_metadata_started",
        "directory_metadata_applied",
    ]
    evidence: dict


def _encoded(record: _Event) -> bytes:
    return json.dumps(
        record.model_dump(), sort_keys=True, separators=(",", ":")
    ).encode()


def _validate(event: str, evidence: Mapping[str, object], prior: list[_Event]) -> dict:
    events = [record.event for record in prior]
    lifecycle = [
        value
        for value in events
        if value
        not in {
            "candidate_staged",
            "credential_intended",
            "credential_applied",
            "credentials_completed",
            "move_intended",
            "move_observed",
        }
    ]
    prepared_record = next((row for row in prior if row.event == "prepared"), None)
    allowed = (
        {"candidate_staged", "prepared"}
        if not prior
        else (
            {"prepared"}
            if events == ["candidate_staged"]
            else {"rollback_verified", "publication_started"}
            if lifecycle == ["prepared"]
            else {"publication_started"}
            if lifecycle == ["prepared", "rollback_verified"]
            else {
                "artifact_retired",
                "artifact_published",
                "installed_validated",
                "directory_metadata_started",
                "directory_metadata_applied",
            }
            if "publication_started" in events
            else set()
        )
    )
    if (
        events
        and events[-1] == "rollback_verified"
        and prepared_record.evidence.get("credential_scopes")
    ):
        allowed = {"credential_intended"}
    if events and events[-1] in {"credential_intended", "credential_applied"}:
        allowed = {"credential_applied", "credentials_completed"}
    if events and events[-1] == "credentials_completed":
        allowed = {"publication_started"}
    if events and events[-1] == "committed":
        allowed = set()
    elif events and events[-1] == "activation_recorded":
        allowed = (
            {"catalog_registered"}
            if prepared_record.evidence.get("isolated_profiles")
            else {"committed"}
        )
    elif events and events[-1] == "catalog_registered":
        allowed = {"committed"}
    elif events and events[-1] == "installed_validated":
        allowed.add("activation_recorded")
    rolling = "rollback_started" in events
    if "publication_started" in events and not rolling and "committed" not in events:
        allowed.add("move_intended")
    if (
        prepared_record is not None
        and prepared_record.evidence["mode"] == "replace"
        and "rollback_verified" in events
        and not rolling
        and "committed" not in events
    ):
        allowed.add("rollback_started")
    if rolling:
        allowed = {
            "move_intended",
            "rollback_metadata_started",
            "rollback_metadata_applied",
            "originals_validated",
        }
        if events[-1] == "originals_validated":
            allowed = {"rollback_activation_recorded"}
        elif events[-1] == "rollback_activation_recorded":
            allowed = {"rolled_back"}
        elif events[-1] == "rolled_back":
            allowed = set()
    if "rollback_credentials_planned" in events and not rolling:
        allowed = {"rollback_started"}
    if events and events[-1] == "move_intended":
        allowed = {"move_observed"}
    if (
        prepared_record is not None
        and prepared_record.evidence["mode"] == "replace"
        and "rollback_verified" in events
        and not any(value in events for value in ("committed", "rolled_back"))
        and events[-1] != "move_intended"
    ):
        allowed.add("rollback_credentials_planned")
        if "rollback_credentials_planned" in events:
            allowed.add("rollback_credential_applied")
    if events in (["candidate_staged"], ["candidate_staged", "prepared"]):
        allowed.add("prepublication_aborted")
    if events and events[-1] == "prepublication_aborted":
        allowed = set()
    if event not in allowed:
        raise ValueError("journal_transition_invalid")
    if (
        event == "publication_started"
        and prepared_record.evidence["mode"] == "replace"
        and "rollback_verified" not in events
    ):
        raise ValueError("rollback_required")
    model = {
        "move_intended": _MoveIntent,
        "move_observed": _MoveObserved,
        "rollback_started": _RollbackStarted,
        "originals_validated": _OriginalsValidated,
        "rollback_activation_recorded": _RollbackActivation,
        "rolled_back": _RolledBack,
        "rollback_credentials_planned": _RollbackCredentialPlan,
        "rollback_credential_applied": _RollbackCredentialApplied,
        "rollback_metadata_started": _DirectoryIntent,
        "rollback_metadata_applied": _DirectoryProgress,
        "candidate_staged": _CandidateReceipt,
        "prepublication_aborted": _PrepublicationAbort,
        "prepared": _Prepared,
        "rollback_verified": _Rollback,
        "artifact_retired": _Progress,
        "artifact_published": _Progress,
        "installed_validated": _Installed,
        "activation_recorded": _ActivationRecorded,
        "catalog_registered": _CatalogRecorded,
        "credential_intended": _CredentialIntent,
        "credential_applied": _CredentialApplied,
        "credentials_completed": _CredentialsComplete,
        "committed": _Committed,
        "directory_metadata_started": _DirectoryIntent,
        "directory_metadata_applied": _DirectoryProgress,
    }.get(event, _Evidence)
    try:
        validated = model.model_validate(dict(evidence))
        if event in {
            "move_intended",
            "move_observed",
            "rollback_started",
            "originals_validated",
            "rollback_activation_recorded",
            "rolled_back",
            "rollback_metadata_started",
            "rollback_metadata_applied",
            "rollback_credentials_planned",
            "rollback_credential_applied",
        }:
            _validate_reverse_evidence(event, validated, prior, prepared_record)
        elif isinstance(validated, _PrepublicationAbort):
            receipt = _CandidateReceipt.model_validate(prior[0].evidence)
            if (
                validated.candidate_digest != _evidence_digest(prior[0].evidence)
                or validated.publication.plan_digest != receipt.plan_digest
                or validated.publication.archive_digest != receipt.archive_digest
                or validated.publication.descriptor != receipt.descriptor
                or validated.prepared_digest
                != (
                    _evidence_digest(prepared_record.evidence)
                    if prepared_record
                    else None
                )
                or (
                    prepared_record is not None
                    and (
                        prepared_record.evidence["mode"] != "replace"
                        or validated.publication
                        != _Prepared.model_validate(
                            prepared_record.evidence
                        ).publication
                    )
                )
            ):
                raise ValueError("prepublication_abort_unverified")
        elif isinstance(validated, _Rollback):
            prepared = _Prepared.model_validate(prepared_record.evidence)
            if validated.safety_sources != prepared.safety_sources:
                raise ValueError("rollback_safety_coverage_mismatch")
            expected_projections = {
                row.path: row
                for artifact in prepared.artifacts
                for row in artifact.rollback_projection_roots
            }
            observed_projections = {
                row.path: row for row in validated.projection_groups
            }
            if (
                len(observed_projections) != len(validated.projection_groups)
                or observed_projections != expected_projections
            ):
                raise ValueError("rollback_projection_coverage_mismatch")
            if validated.sqlite_groups or validated.projection_groups:
                coverage = {
                    row.logical_id
                    for row in prepared.artifacts
                    if row.previous is not None
                } | {row.logical_id for row in prepared.directory_metadata}
                if set(validated.coverage) != coverage:
                    raise ValueError("rollback_coverage_mismatch")
            if validated.sqlite_groups:
                if not prepared.rollback_sources:
                    raise ValueError("rollback_sqlite_coverage_mismatch")
                expected = {row.logical_id: row for row in prepared.rollback_sources}
                observed = {
                    row.logical_id: _RollbackSource.model_validate(
                        {
                            key: value
                            for key, value in row.model_dump().items()
                            if key in _RollbackSource.model_fields
                        }
                    )
                    for row in validated.sqlite_groups
                }
                if (
                    len(observed) != len(validated.sqlite_groups)
                    or observed != expected
                ):
                    raise ValueError("rollback_sqlite_coverage_mismatch")
            elif prepared.rollback_sources or any(
                row.rollback_requires_owner for row in prepared.artifacts
            ):
                raise ValueError("rollback_sqlite_owner_receipt_required")
            if validated.credential_issues != sorted(set(validated.credential_issues)):
                raise ValueError("rollback_credential_issues_invalid")
        elif isinstance(validated, _ActivationRecorded):
            prepared = _Prepared.model_validate(prepared_record.evidence)
            if (
                prepared.publication is None
                or validated.generation != prepared.generation
                or validated.plan_digest != prepared.publication.plan_digest
                or validated.installed_digest != _evidence_digest(prior[-1].evidence)
                or validated.selectors != sorted(set(validated.selectors))
                or not set(validated.selectors) <= set(prepared.publication.selectors)
                or validated.owners != sorted(set(validated.owners))
                or any(
                    not owner or len(owner) > 256 or "\0" in owner
                    for owner in validated.owners
                )
                or len({item.path for item in validated.records})
                != len(validated.records)
                or any(item.kind != "file" for item in validated.records)
            ):
                raise ValueError("activation_context_invalid")
            for selector in validated.selectors:
                _Object.absolute_path(selector)
        elif isinstance(validated, _CatalogRecorded):
            prepared = _Prepared.model_validate(prepared_record.evidence)
            if (
                validated.generation != prepared.generation
                or validated.activation_digest != _evidence_digest(prior[-1].evidence)
                or len(validated.records) != len(prepared.isolated_profiles)
                or len({row.path for row in validated.records})
                != len(validated.records)
            ):
                raise ValueError("catalog_context_invalid")
        elif isinstance(validated, _CredentialIntent):
            prepared = _Prepared.model_validate(prepared_record.evidence)
            if (
                validated.material != prepared.credential_material
                or validated.scopes != prepared.credential_scopes
                or validated.plan_digest != prepared.publication.plan_digest
                or validated.descriptor_digest != prepared.publication.descriptor.sha256
                or "rollback_verified" not in events
            ):
                raise ValueError("credential_intent_invalid")
        elif isinstance(validated, _CredentialApplied):
            intent = next(
                row.evidence for row in prior if row.event == "credential_intended"
            )
            planned = json.loads(intent["scopes"].get(validated.record_id, "{}"))
            if (
                validated.action != planned.get("action")
                or validated.purpose != planned.get("purpose", "")
                or any(
                    row.event == "credential_applied"
                    and row.evidence["record_id"] == validated.record_id
                    for row in prior
                )
                or (validated.action == "retain" and validated.value_digest != "")
                or (
                    validated.action == "create"
                    and (
                        len(validated.value_digest) != 64
                        or any(
                            c not in "0123456789abcdef" for c in validated.value_digest
                        )
                    )
                )
            ):
                raise ValueError("credential_application_invalid")
        elif isinstance(validated, _CredentialsComplete):
            intent = next(
                row.evidence for row in prior if row.event == "credential_intended"
            )
            applied = {
                row.evidence["record_id"]: row.evidence
                for row in prior
                if row.event == "credential_applied"
            }
            if (
                set(applied) != set(intent["scopes"])
                or validated.intent_digest != _evidence_digest(intent)
                or validated.applied_digest != _evidence_digest(applied)
            ):
                raise ValueError("credential_application_incomplete")
        elif isinstance(validated, _Committed):
            activation = next(
                row for row in prior if row.event == "activation_recorded"
            )
            catalog = next(
                (row for row in prior if row.event == "catalog_registered"), None
            )
            credential = next(
                (row for row in prior if row.event == "credentials_completed"), None
            )
            if (
                validated.credential_digest
                != (_evidence_digest(credential.evidence) if credential else None)
                or validated.generation != activation.evidence["generation"]
                or validated.activation_digest != _evidence_digest(activation.evidence)
                or validated.catalog_digest
                != (_evidence_digest(catalog.evidence) if catalog else None)
            ):
                raise ValueError("commit_context_invalid")
        elif isinstance(validated, (_DirectoryIntent, _DirectoryProgress)):
            prepared = _Prepared.model_validate(prepared_record.evidence)
            item = next(
                (
                    item
                    for item in prepared.directory_metadata
                    if item.logical_id == validated.logical_id
                ),
                None,
            )
            if item is None or any(
                row.event == event
                and row.evidence["logical_id"] == validated.logical_id
                for row in prior
            ):
                raise ValueError("directory_metadata_progress_invalid")
            observed = (
                validated.before
                if isinstance(validated, _DirectoryIntent)
                else validated.observed
            )
            if (observed.path, observed.device, observed.inode) != (
                item.previous.path,
                item.previous.device,
                item.previous.inode,
            ):
                raise ValueError("directory_metadata_progress_invalid")
            if isinstance(validated, _DirectoryIntent):
                expected_before = item.previous
                for record in prior:
                    if record.event in {"artifact_retired", "artifact_published"}:
                        for value in record.evidence.get("directories", []):
                            if value["path"] == item.previous.path:
                                expected_before = _DirectoryState.model_validate(value)
                if (
                    validated.applied != item.applied
                    or validated.before != expected_before
                ):
                    raise ValueError("directory_metadata_progress_invalid")
            elif not any(
                row.event == "directory_metadata_started"
                and row.evidence["logical_id"] == validated.logical_id
                for row in prior
            ) or (observed.mode, observed.mtime_ns) != (
                item.applied.mode,
                item.applied.mtime_ns,
            ):
                raise ValueError("directory_metadata_progress_invalid")
        elif isinstance(validated, _Installed):
            receipt = next(
                (row for row in prior if row.event == "candidate_staged"), None
            )
            if receipt is None:
                raise ValueError("installed_context_invalid")
            if (
                any(
                    getattr(validated, key) != receipt.evidence[key]
                    for key in ("plan_digest", "manifest_digest")
                )
                or validated.descriptor_digest
                != receipt.evidence["descriptor"]["sha256"]
            ):
                raise ValueError("installed_context_invalid")
            prepared = _Prepared.model_validate(prepared_record.evidence)
            completed = {
                row.evidence["logical_id"]
                for row in prior
                if row.event == "directory_metadata_applied"
            }
            if (
                not {item.logical_id for item in prepared.directory_metadata}
                <= completed
            ):
                raise ValueError("directory_metadata_progress_invalid")
            expected = {
                item.path: (item.device, item.inode)
                for item in prepared.installed_paths
            }
            observed = {
                item.path: (item.device, item.inode) for item in validated.artifacts
            }
            if (
                not expected
                or expected != observed
                or len(observed) != len(validated.artifacts)
            ):
                raise ValueError("installed_objects_invalid")
        elif isinstance(validated, _Prepared):
            artifacts = validated.artifacts
            metadata = validated.directory_metadata
            if (
                len({item.logical_id for item in metadata}) != len(metadata)
                or len({item.previous.path for item in metadata}) != len(metadata)
                or any(
                    item.parent.path != str(Path(item.previous.path).parent)
                    for item in metadata
                )
            ):
                raise ValueError("directory_metadata_mapping_invalid")
            if len({item.logical_id for item in artifacts}) != len(artifacts) or len(
                {item.target for item in artifacts}
            ) != len(artifacts):
                raise ValueError("duplicate_artifact")
            if any(
                (item.candidate is None) != (item.action == "retire")
                or item.candidate is not None
                and item.candidate.path == item.target
                or item.previous
                and item.previous.path != item.target
                or (item.previous is None) != (item.retained is None)
                for item in artifacts
            ):
                raise ValueError("artifact_mapping_invalid")
        elif isinstance(validated, _Progress):
            items = {
                item.logical_id: item
                for item in _Prepared.model_validate(prepared_record.evidence).artifacts
            }
            item = items.get(validated.logical_id)
            if item is None or any(
                row.event == event
                and row.evidence["logical_id"] == validated.logical_id
                for row in prior
                if row.event in {"artifact_retired", "artifact_published"}
            ):
                raise ValueError("artifact_progress_invalid")
            if (
                event == "artifact_retired"
                and item.previous is None
                or event == "artifact_published"
                and item.candidate is None
            ):
                raise ValueError("artifact_progress_invalid")
            expected_directories = {
                directory.previous.path: (
                    directory.previous.device,
                    directory.previous.inode,
                )
                for directory in _Prepared.model_validate(
                    prepared_record.evidence
                ).directory_metadata
                if directory.previous.path == str(Path(item.target).parent)
            }
            observed_directories = {
                directory.path: (directory.device, directory.inode)
                for directory in validated.directories
            }
            if expected_directories != observed_directories or len(
                validated.directories
            ) != len(observed_directories):
                raise ValueError("directory_metadata_progress_invalid")
            expected = (
                item.previous_metadata
                if event == "artifact_retired"
                else item.candidate
            )
            destination = item.retained if event == "artifact_retired" else item.target
            if (
                expected is None
                or validated.observed.path != destination
                or validated.observed.model_dump(exclude={"path"})
                != expected.model_dump(exclude={"path"})
            ):
                raise ValueError("artifact_progress_invalid")
        return validated.model_dump()
    except (ValidationError, TypeError, ValueError):
        raise ValueError("journal_evidence_invalid") from None


def _validate_reverse_evidence(event, value, prior, prepared_record):
    prepared = _Prepared.model_validate(prepared_record.evidence)
    items = {item.logical_id: item for item in prepared.artifacts}
    rolling = next((row for row in prior if row.event == "rollback_started"), None)
    if event == "move_intended":
        item = items.get(value.logical_id)
        if item is None or (
            value.step in {"unpublish", "restore"}
            or value.step.startswith("credential_")
        ) != (rolling is not None):
            raise ValueError("move_context_invalid")
        if value.step.startswith("credential_"):
            candidates = [
                entry
                for row in prior
                if row.event == "rollback_credentials_planned"
                for entry in _RollbackCredentialPlan.model_validate(
                    row.evidence
                ).artifacts
                if entry.logical_id == item.logical_id
            ]
            latest = next((entry for entry in reversed(candidates)), None)
            if value.step == "credential_retire" and latest is not None:
                source, source_path, target = item.previous, item.target, item.retained
            elif value.step == "credential_publish" and latest is not None:
                source, source_path, target = (
                    latest.candidate,
                    latest.candidate.path,
                    item.target,
                )
            elif value.step == "credential_unpublish":
                source = next(
                    (
                        entry.candidate
                        for entry in candidates
                        if entry.candidate.path == value.destination
                    ),
                    None,
                )
                source_path, target = item.target, value.destination
            else:
                raise ValueError("move_mapping_invalid")
        else:
            source, target = {
                "retire": (item.previous, item.retained),
                "publish": (item.candidate, item.target),
                "unpublish": (
                    item.candidate,
                    item.candidate.path if item.candidate else None,
                ),
                "restore": (item.previous, item.target),
            }[value.step]
            source_path = (
                item.target
                if value.step == "unpublish"
                else item.retained
                if value.step == "restore"
                else source.path
                if source
                else None
            )
        if (
            source is None
            or value.source.model_dump(exclude={"path"})
            != source.model_dump(exclude={"path"})
            or value.source.path != source_path
            or value.destination != target
        ):
            raise ValueError("move_mapping_invalid")
        paths = {str(Path(source_path).parent), str(Path(target).parent)}
        if {p.state.path for p in value.parents} != paths or len(value.parents) != len(
            paths
        ):
            raise ValueError("move_parent_invalid")
        for parent in value.parents:
            for name, child in parent.children.items():
                if Path(name).name != name or child.path != str(
                    Path(parent.state.path) / name
                ):
                    raise ValueError("move_child_invalid")
    elif event == "move_observed":
        if prior[
            -1
        ].event != "move_intended" or value.intent_digest != _evidence_digest(
            prior[-1].evidence
        ):
            raise ValueError("move_context_invalid")
        expected = {
            row.previous.path: (row.previous.device, row.previous.inode)
            for row in prepared.directory_metadata
        }
        if {row.path: (row.device, row.inode) for row in value.directories} != expected:
            raise ValueError("move_directory_invalid")
    elif event == "rollback_credentials_planned":
        rollback = next(row for row in prior if row.event == "rollback_verified")
        if value.rollback_digest != _evidence_digest(
            rollback.evidence
        ) or value.previous_digest != _evidence_digest(prior[-1].evidence):
            raise ValueError("rollback_credential_plan_invalid")
        artifact_ids = {row.logical_id for row in value.artifacts}
        if (
            len(artifact_ids) != len(value.artifacts)
            or {row.record_id for row in value.references} != set(value.scopes)
            or len(value.references) != len(value.scopes)
            or artifact_ids != {row.artifact_id for row in value.references}
        ):
            raise ValueError("rollback_credential_plan_invalid")
        for row in value.artifacts:
            item = items.get(row.logical_id)
            if (
                item is None
                or item.previous is None
                or row.candidate.path != row.metadata.path
                or (row.candidate.device, row.candidate.inode, row.candidate.kind)
                != (row.metadata.device, row.metadata.inode, row.metadata.kind)
                or row.candidate.device != item.previous.device
                or Path(item.retained).parent not in Path(row.candidate.path).parents
                or row.candidate.kind != item.previous.kind
            ):
                raise ValueError("rollback_credential_plan_invalid")
        for row in value.references:
            item = items[row.artifact_id]
            if not (
                Path(row.target) == Path(item.target)
                or item.previous.kind == "directory"
                and Path(item.target) in Path(row.target).parents
            ):
                raise ValueError("rollback_credential_plan_invalid")
        for scope in value.scopes.values():
            import re

            if (
                re.fullmatch(r"recovery_[0-9a-f]{32}_[A-Za-z0-9_.-]+", scope.purpose)
                is None
            ):
                raise ValueError("rollback_credential_plan_invalid")
    elif event == "rollback_credential_applied":
        plan = next(
            row
            for row in reversed(prior)
            if row.event == "rollback_credentials_planned"
        )
        expected = plan.evidence["scopes"].get(value.record_id)
        phase = prior[prior.index(plan) + 1 :]
        if (
            expected is None
            or value.plan_digest != _evidence_digest(plan.evidence)
            or value.purpose != expected["purpose"]
            or value.value_digest != expected["value_digest"]
            or any(
                row.event == event and row.evidence["record_id"] == value.record_id
                for row in phase
            )
        ):
            raise ValueError("rollback_credential_value_invalid")
    elif event == "rollback_started":
        rollback = next(row for row in prior if row.event == "rollback_verified")
        if (
            value.prepared_digest != _evidence_digest(prepared_record.evidence)
            or value.rollback_digest != _evidence_digest(rollback.evidence)
            or value.generation == prepared.generation
            or {row.config for row in value.profiles}
            != {row.config for row in prepared.replacement_profiles}
            or len(value.profiles) != len(prepared.replacement_profiles)
            or {row.installation_id for row in value.profiles}
            & {row.installation_id for row in prepared.replacement_profiles}
            or len({row.installation_id for row in value.profiles})
            != len(value.profiles)
        ):
            raise ValueError("rollback_context_invalid")
        if value.retained_credential_scopes != sorted(prepared.credential_scopes):
            raise ValueError("rollback_credential_context_invalid")
    elif event == "originals_validated":
        expected = {
            item.target: item.previous_metadata
            for item in prepared.artifacts
            if item.previous is not None
        }
        credential = next(
            (
                row
                for row in reversed(prior)
                if row.event == "rollback_credentials_planned"
            ),
            None,
        )
        if value.credential_plan_digest != (
            _evidence_digest(credential.evidence) if credential else None
        ):
            raise ValueError("rollback_credential_plan_invalid")
        if credential:
            for row in _RollbackCredentialPlan.model_validate(
                credential.evidence
            ).artifacts:
                expected[items[row.logical_id].target] = row.metadata.model_copy(
                    update={"path": items[row.logical_id].target}
                )
            phase = prior[prior.index(credential) + 1 :]
            if {
                row.evidence["record_id"]
                for row in phase
                if row.event == "rollback_credential_applied"
            } != set(credential.evidence["scopes"]):
                raise ValueError("rollback_credential_application_incomplete")
        if (
            value.rollback_started_digest != _evidence_digest(rolling.evidence)
            or {row.path: row for row in value.artifacts} != expected
            or len(value.artifacts) != len(expected)
        ):
            raise ValueError("originals_evidence_invalid")
        originals = {
            row.previous.path: row.previous for row in prepared.directory_metadata
        }
        if (
            len(value.directories) != len(originals)
            or {row.path for row in value.directories} != set(originals)
            or any(
                (row.device, row.inode, row.mode, row.mtime_ns)
                != (
                    originals[row.path].device,
                    originals[row.path].inode,
                    originals[row.path].mode,
                    originals[row.path].mtime_ns,
                )
                for row in value.directories
            )
        ):
            raise ValueError("original_metadata_invalid")
    elif event == "rollback_activation_recorded":
        from .owner_registry import install_adapters

        if (
            value.generation != rolling.evidence["generation"]
            or value.originals_digest != _evidence_digest(prior[-1].evidence)
            or len({row.path for row in value.records}) != len(value.records)
            or any(row.kind != "file" for row in value.records)
            or value.selectors
            != sorted(row["config"] for row in rolling.evidence["profiles"])
            or value.owners
            != sorted(
                {row.owner_id for row in install_adapters() if row.activation_required}
            )
        ):
            raise ValueError("rollback_activation_invalid")
    elif event == "rolled_back":
        if value.generation != rolling.evidence[
            "generation"
        ] or value.activation_digest != _evidence_digest(prior[-1].evidence):
            raise ValueError("rollback_terminal_invalid")
    elif event in {"rollback_metadata_started", "rollback_metadata_applied"}:
        credential = next(
            (
                row
                for row in reversed(prior)
                if row.event == "rollback_credentials_planned"
            ),
            None,
        )
        if credential:
            prior = prior[prior.index(credential) + 1 :]
        item = next(
            (
                row
                for row in prepared.directory_metadata
                if row.logical_id == value.logical_id
            ),
            None,
        )
        if item is None or any(
            row.event == event and row.evidence["logical_id"] == value.logical_id
            for row in prior
        ):
            raise ValueError("rollback_metadata_invalid")
        if event == "rollback_metadata_started":
            if (
                value.before.path != item.previous.path
                or value.applied.mode != item.previous.mode
                or value.applied.mtime_ns != item.previous.mtime_ns
                or (value.before.device, value.before.inode)
                != (item.previous.device, item.previous.inode)
            ):
                raise ValueError("rollback_metadata_invalid")
        elif (
            not any(
                row.event == "rollback_metadata_started"
                and row.evidence["logical_id"] == value.logical_id
                for row in prior
            )
            or value.observed.path != item.previous.path
            or (
                value.observed.device,
                value.observed.inode,
                value.observed.mode,
                value.observed.mtime_ns,
            )
            != (
                item.previous.device,
                item.previous.inode,
                item.previous.mode,
                item.previous.mtime_ns,
            )
        ):
            raise ValueError("rollback_metadata_invalid")


class Journal:
    """Serialize local records under a stable lock and retain incomplete writes.

    Reopening existing evidence never repairs files or clears bootstrap pointers.
    The caller supplies a locally chosen private control root, never an archive
    locator. A missing, corrupt, or interrupted sequence remains recovery-required.
    """

    def __init__(self, root: Path, operation_id: str):
        if (
            type(operation_id) is not str
            or not 0 < len(operation_id) <= 256
            or "\0" in operation_id
        ):
            raise ValueError("operation_id_invalid")
        allowed, reason = qualified_for("admission", root)
        if not allowed:
            raise OSError(reason)
        self.operation_id = operation_id
        self.root = root / (
            "operation-" + hashlib.sha256(operation_id.encode()).hexdigest()
        )
        try:
            create_private_directory(self.root)
        except FileExistsError:
            pass
        else:
            with pinned_directory(self.root) as parent:
                Admission._create_lock(parent, "journal.lock")
                flush_directory(parent)
        with pinned_directory(self.root) as parent:
            info = os.fstat(parent)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("journal_not_private")
            self._identity = info.st_dev, info.st_ino

    @contextmanager
    def _locked(self, *, exclusive: bool):
        with pinned_directory(self.root) as parent:
            info = os.fstat(parent)
            if (info.st_dev, info.st_ino) != self._identity or info.st_mode & 0o077:
                raise ValueError("journal_identity_changed")
            lock = os.open(
                "journal.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            try:
                info = os.fstat(lock)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                ):
                    raise ValueError("journal_lock_invalid")
                fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                named = os.stat("journal.lock", dir_fd=parent, follow_symlinks=False)
                if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("journal_lock_changed")
                yield parent
            finally:
                os.close(lock)

    def _records(self, parent: int) -> list[_Event]:
        names = sorted(
            name
            for name in os.listdir(parent)
            if name
            not in {"journal.lock", "verified-manifest.json", "restore-plan.json"}
        )
        if len(names) > MAX_EVENTS:
            raise ValueError("journal_limit")
        records = []
        previous = ""
        for index, name in enumerate(names):
            if name != f"{index:06d}.json":
                raise ValueError("journal_sequence_invalid")
            record = _Event.model_validate(_read(parent, name))
            if (
                record.operation_id != self.operation_id
                or record.sequence != index
                or record.previous != previous
            ):
                raise ValueError("journal_sequence_invalid")
            _validate(record.event, record.evidence, records)
            records.append(record)
            previous = hashlib.sha256(_encoded(record)).hexdigest()
        return records

    def record(self, event: str, evidence: Mapping[str, object]) -> None:
        """Flush one strictly typed exclusive record; failed writes remain evidence."""
        if event in {
            "prepublication_aborted",
            "move_intended",
            "move_observed",
            "rollback_started",
            "originals_validated",
            "rollback_activation_recorded",
            "rolled_back",
            "rollback_metadata_started",
            "rollback_metadata_applied",
            "rollback_credentials_planned",
            "rollback_credential_applied",
        }:
            raise ValueError("recovery_execution_required")
        if event in {
            "credential_intended",
            "credential_applied",
            "credentials_completed",
        }:
            raise ValueError("credential_held_application_required")
        if event == "rollback_verified" and evidence.get("safety_sources"):
            raise ValueError("rollback_held_capture_required")
        if event == "catalog_registered":
            raise ValueError("catalog_finalization_required")
        if event == "rollback_verified" and (
            evidence.get("sqlite_groups") or evidence.get("projection_groups")
        ):
            raise ValueError("rollback_held_capture_required")
        with self._locked(exclusive=True) as parent:
            self._append(parent, event, evidence)

    @staticmethod
    def _flush_record(parent: int, name: str, expected: dict) -> None:
        """Reestablish durability of a complete but possibly unflushed record."""

        def identity(info):
            return (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )

        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if _read(parent, name) != expected:
                raise ValueError("durable_record_changed")
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if identity(named) != identity(before):
                raise ValueError("durable_record_changed")
            flush_file(fd)
            if identity(os.fstat(fd)) != identity(before) or identity(
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            ) != identity(before):
                raise ValueError("durable_record_changed")
        finally:
            os.close(fd)

    def _flush_records(self, parent: int) -> None:
        """Revalidate and flush accepted history under the held journal lock."""
        for row in self._records(parent):
            self._flush_record(parent, f"{row.sequence:06d}.json", row.model_dump())
        flush_directory(parent)

    def _append(self, parent: int, event: str, evidence: Mapping[str, object]) -> None:
        """Append while the caller retains this operation's exclusive lock."""
        records = self._records(parent)
        validated = _validate(event, evidence, records)
        record = _Event(
            operation_id=self.operation_id,
            sequence=len(records),
            previous=hashlib.sha256(_encoded(records[-1])).hexdigest()
            if records
            else "",
            event=event,
            evidence=validated,
        )
        encoded = _encoded(record)
        if len(encoded) > 1048576:
            raise ValueError("journal_record_limit")
        Admission._write_new_record(parent, f"{len(records):06d}.json", encoded)
        flush_directory(parent)

    def prepare_publication(
        self,
        candidate,
        plan,
        *,
        bootstrap_root,
        namespaces,
        selectors,
        generation,
        replacement_profiles=(),
        incoming_credentials=None,
    ):
        """Bind an explicit local publication context; never infer its scope."""
        from .publication import _prepare

        _prepare(
            self,
            candidate,
            plan,
            bootstrap_root,
            namespaces,
            selectors,
            generation,
            replacement_profiles=replacement_profiles,
            incoming_credentials=incoming_credentials,
        )

    def record_candidate(self, stage, plan, archive):
        """Durably bind successful staging to the independently verified archive."""
        from .archive_reader import verify_sealed
        from .publication import _descriptor, _plan_digest
        from .restore_plan import recheck_targets

        verify_sealed(archive)
        if plan.archive_digest != archive.digest:
            raise ValueError("archive_plan_mismatch")
        with self._locked(exclusive=True) as parent:
            recheck_targets(plan)
            _descriptor(stage, plan)
            from .limits import ArchiveLimits

            if len(archive.manifest_bytes) > ArchiveLimits().manifest_bytes:
                raise ValueError("manifest_limit")
            try:
                Admission._write_new_record(
                    parent, "verified-manifest.json", archive.manifest_bytes
                )
            except FileExistsError:
                from .archive_reader import _regular

                with _regular(self.root / "verified-manifest.json") as stream:
                    if (
                        stream.read(ArchiveLimits().manifest_bytes + 1)
                        != archive.manifest_bytes
                    ):
                        raise ValueError("verified_manifest_changed") from None
                    flush_file(stream.fileno())
            flush_directory(parent)
            from .plan_records import save_plan

            local_plan = save_plan(self, parent, plan)
            self._append(
                parent,
                "candidate_staged",
                {
                    "local_plan": local_plan,
                    "stage": observe_artifact(stage),
                    "descriptor": observe_artifact(stage / "candidate.json"),
                    "archive_digest": archive.digest,
                    "manifest_digest": hashlib.sha256(
                        archive.manifest_bytes
                    ).hexdigest(),
                    "plan_digest": _plan_digest(plan),
                },
            )

    def validate_installed(self, candidate, plan):
        """Prove installed bytes and metadata without releasing recovery admission."""
        from .publication import _validate_installed

        return _validate_installed(self, candidate, plan)

    def verify_rollback(self, path, *, password, work_root, cancel, coverage):
        """Authenticate exact raw rollback coverage without persisting its secret."""
        from .publication import _verify_rollback

        _verify_rollback(self, path, password, work_root, cancel, coverage)

    def artifact_states(self) -> dict[str, str]:
        """Classify staged/published/uncertain objects without mutating either name."""
        with self._locked(exclusive=False) as parent:
            records = self._records(parent)
            if not records:
                raise ValueError("journal_preparation_missing")
            prepared = next((row for row in records if row.event == "prepared"), None)
            if prepared is None:
                raise ValueError("journal_preparation_missing")
            return _states(_Prepared.model_validate(prepared.evidence))

    def recover(self) -> str:
        """Read local evidence without cleanup, publication, or fence changes."""
        try:
            with self._locked(exclusive=False) as parent:
                records = self._records(parent)
                if records and records[-1].event == "prepared":
                    states = _states(_Prepared.model_validate(records[-1].evidence))
                    if all(state == "staged" for state in states.values()):
                        return "prepared"
        except (OSError, ValueError, RuntimeError):
            return "recovery_required"
        return "recovery_required"
