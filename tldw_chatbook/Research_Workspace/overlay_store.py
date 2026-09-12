"""Private, bounded persistence for Research Workspace pane preferences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    PrivatePathError,
    atomic_private_write_text,
    lexical_path,
    open_private_binary,
    secure_private_directory,
)

from .contracts import QualifiedWorkspaceRef
from .layout_state import ResearchPanePreferences


OVERLAY_SCHEMA_VERSION = 2
MAX_OVERLAY_FILE_BYTES = 1024 * 1024
MAX_OVERLAY_RECORDS = 512
MAX_IDENTITY_CHARS = 256
MAX_TIMESTAMP_CHARS = 64
MAX_SOURCE_FOLDERS = 128
MAX_SOURCE_ANNOTATIONS = 512
MAX_SOURCE_IDS_PER_FOLDER = 512
MAX_FOLDER_DEPTH = 8
MAX_SOURCE_ID_CHARS = 1024
MAX_FOLDER_NAME_CHARS = 120
MAX_ANNOTATION_TEXT_CHARS = 4000

_ROOT_FIELDS = frozenset({"schema_version", "records"})
_V1_RECORD_FIELDS = frozenset(
    {
        "key",
        "revision",
        "pane_preferences",
        "preferred_companion",
        "created_at",
        "updated_at",
    }
)
_V2_RECORD_FIELDS = _V1_RECORD_FIELDS | frozenset(
    {"source_folders", "source_annotations"}
)
_KEY_FIELDS = frozenset(
    {"data_source", "workspace_id", "server_profile_id", "principal_id"}
)
_PREFERENCE_FIELDS = frozenset({"sources_open", "studio_open"})
_FORBIDDEN_KEY_PARTS = frozenset(
    {"api", "body", "content", "password", "path", "secret", "token"}
)
_SOURCE_FOLDER_FIELDS = frozenset(
    {"folder_id", "name", "parent_folder_id", "source_ids"}
)
_SOURCE_ANNOTATION_FIELDS = frozenset(
    {
        "annotation_id",
        "source_id",
        "quote",
        "note",
        "created_at",
        "updated_at",
    }
)
_PRIVATE_TEXT = re.compile(
    r"(?i)(?:\b(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|password)\s*[:=]|"
    r"\bbearer\s+\S+|[a-z][a-z0-9+.-]*://|(?:^|\s)(?:/Users/|/home/|/root/|"
    r"[A-Za-z]:\\\\|\\\\\\\\))"
)


class OverlayValidationError(ValueError):
    """Raised when the overlay container cannot be validated safely."""


class OverlayLimitError(OverlayValidationError):
    """Raised when the overlay exceeds a declared v1 bound."""


class OverlayConflictError(RuntimeError):
    """Raised when the target overlay revision changed before replacement."""


@dataclass(frozen=True, slots=True)
class ResearchSourceFolder:
    """Device-only organization for qualified source associations."""

    folder_id: str
    name: str
    source_ids: tuple[str, ...] = ()
    parent_folder_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "folder_id", _validated_source_id(self.folder_id, "folder_id")
        )
        object.__setattr__(
            self,
            "name",
            _validated_plain_text(self.name, "folder name", MAX_FOLDER_NAME_CHARS),
        )
        parent = str(self.parent_folder_id or "").strip()
        if parent:
            parent = _validated_source_id(parent, "parent_folder_id")
        if parent == self.folder_id:
            raise OverlayValidationError("folder cannot be its own parent")
        object.__setattr__(self, "parent_folder_id", parent)
        if not isinstance(self.source_ids, tuple):
            raise OverlayValidationError("source_ids must be a tuple")
        if len(self.source_ids) > MAX_SOURCE_IDS_PER_FOLDER:
            raise OverlayLimitError("folder source_ids exceed the declared bound")
        source_ids = tuple(
            _validated_source_id(value, "source_id") for value in self.source_ids
        )
        if len(source_ids) != len(set(source_ids)):
            raise OverlayValidationError("folder source_ids must be unique")
        object.__setattr__(self, "source_ids", source_ids)


@dataclass(frozen=True, slots=True)
class ResearchSourceAnnotation:
    """Bounded device-only annotation tied to one source association."""

    annotation_id: str
    source_id: str
    quote: str
    note: str
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_id",
            _validated_source_id(self.annotation_id, "annotation_id"),
        )
        object.__setattr__(
            self, "source_id", _validated_source_id(self.source_id, "source_id")
        )
        object.__setattr__(
            self, "quote", _validated_annotation_text(self.quote, "quote")
        )
        object.__setattr__(self, "note", _validated_annotation_text(self.note, "note"))
        if not self.quote and not self.note:
            raise OverlayValidationError("annotation requires a quote or note")
        object.__setattr__(
            self, "created_at", _validated_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self, "updated_at", _validated_timestamp(self.updated_at, "updated_at")
        )


@dataclass(frozen=True, slots=True)
class ResearchPresentationOverlay:
    ref: QualifiedWorkspaceRef
    revision: int
    preferences: ResearchPanePreferences
    created_at: str
    updated_at: str
    source_folders: tuple[ResearchSourceFolder, ...] = ()
    source_annotations: tuple[ResearchSourceAnnotation, ...] = ()


@dataclass(frozen=True, slots=True)
class QuarantinedOverlayRecord:
    record_index: int
    reason: str
    raw: object

    def export_json(self) -> str:
        """Return the affected raw record for an explicit recovery export."""

        return json.dumps(self.raw, indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class OverlayLoadResult:
    records: dict[QualifiedWorkspaceRef, ResearchPresentationOverlay]
    quarantined: tuple[QuarantinedOverlayRecord, ...] = ()
    target_precondition: PrivateFileWritePrecondition = (
        PrivateFileWritePrecondition.missing()
    )


class ResearchPresentationOverlayStore:
    """Persist presentation-only records under one application-owned directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = lexical_path(path)
        self.application_owned_directory = self.path.parent

    def load_all(self) -> OverlayLoadResult:
        """Load valid records while independently quarantining invalid ones."""

        raw_payload, target_precondition = self._read_payload()
        if raw_payload is None:
            return OverlayLoadResult({}, target_precondition=target_precondition)
        try:
            payload = raw_payload.decode("utf-8")
            root = json.loads(payload)
            schema_version, records = self._validated_record_list(root)
        except UnicodeDecodeError as exc:
            return OverlayLoadResult(
                {},
                (
                    QuarantinedOverlayRecord(
                        -1,
                        str(exc),
                        raw_payload.decode("utf-8", errors="replace"),
                    ),
                ),
                target_precondition,
            )
        except OverlayLimitError:
            raise
        except (json.JSONDecodeError, OverlayValidationError) as exc:
            return OverlayLoadResult(
                {},
                (QuarantinedOverlayRecord(-1, str(exc), payload),),
                target_precondition,
            )

        decoded: dict[QualifiedWorkspaceRef, ResearchPresentationOverlay] = {}
        quarantined: list[QuarantinedOverlayRecord] = []
        for index, raw in enumerate(records):
            try:
                record = _decode_record(raw, schema_version=schema_version)
                if record.ref in decoded:
                    raise OverlayValidationError("duplicate qualified key")
            except (TypeError, ValueError) as exc:
                quarantined.append(QuarantinedOverlayRecord(index, str(exc), raw))
                continue
            decoded[record.ref] = record
        return OverlayLoadResult(decoded, tuple(quarantined), target_precondition)

    def load(self, ref: QualifiedWorkspaceRef) -> ResearchPresentationOverlay | None:
        """Load one qualified overlay; canonical workspace access is separate."""

        if not isinstance(ref, QualifiedWorkspaceRef):
            raise TypeError("ref must be QualifiedWorkspaceRef")
        return self.load_all().records.get(ref)

    def save(
        self,
        ref: QualifiedWorkspaceRef,
        preferences: ResearchPanePreferences,
        *,
        expected_revision: int,
        source_folders: tuple[ResearchSourceFolder, ...] | None = None,
        source_annotations: tuple[ResearchSourceAnnotation, ...] | None = None,
        timestamp: str | None = None,
    ) -> ResearchPresentationOverlay:
        """Compare the current target revision, then atomically replace the file."""

        if not isinstance(ref, QualifiedWorkspaceRef):
            raise TypeError("ref must be QualifiedWorkspaceRef")
        _validate_ref_lengths(ref)
        if not isinstance(preferences, ResearchPanePreferences):
            raise TypeError("preferences must be ResearchPanePreferences")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        now = _validated_timestamp(
            timestamp
            or datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "timestamp",
        )

        # This fresh bounded read is the compare immediately before replacement.
        loaded = self.load_all()
        current = loaded.records.get(ref)
        current_revision = current.revision if current is not None else 0
        if current_revision != expected_revision:
            raise OverlayConflictError(
                f"overlay revision changed: expected {expected_revision}, "
                f"found {current_revision}"
            )
        if any(item.record_index < 0 for item in loaded.quarantined):
            raise OverlayValidationError(
                "overlay container must be recovered before replacement"
            )
        if current is None and (
            len(loaded.records) + len(loaded.quarantined) >= MAX_OVERLAY_RECORDS
        ):
            raise OverlayLimitError(
                f"overlay records exceed maximum {MAX_OVERLAY_RECORDS}"
            )

        saved = ResearchPresentationOverlay(
            ref=ref,
            revision=current_revision + 1,
            preferences=preferences,
            created_at=current.created_at if current is not None else now,
            updated_at=now,
            source_folders=(
                current.source_folders
                if source_folders is None and current is not None
                else tuple(source_folders or ())
            ),
            source_annotations=(
                current.source_annotations
                if source_annotations is None and current is not None
                else tuple(source_annotations or ())
            ),
        )
        _validate_source_overlay(saved.source_folders, saved.source_annotations)
        records = dict(loaded.records)
        records[ref] = saved
        serialized = _encode_store(records.values(), loaded.quarantined)
        if len(serialized.encode("utf-8")) > MAX_OVERLAY_FILE_BYTES:
            raise OverlayLimitError(
                f"overlay file exceeds maximum {MAX_OVERLAY_FILE_BYTES} bytes"
            )

        secure_private_directory(
            self.application_owned_directory,
            create=True,
            application_owned=True,
        )
        try:
            atomic_private_write_text(
                self.path,
                serialized,
                application_owned_directory=self.application_owned_directory,
                target_precondition=loaded.target_precondition,
            )
        except PrivatePathError as exc:
            if exc.result.reason in {"target_appeared", "target_replaced"}:
                raise OverlayConflictError(
                    "overlay changed at the atomic replacement boundary"
                ) from None
            raise
        return saved

    def _read_payload(
        self,
    ) -> tuple[bytes | None, PrivateFileWritePrecondition]:
        secure_private_directory(
            self.application_owned_directory,
            create=True,
            application_owned=True,
        )
        try:
            with open_private_binary(self.path) as opened:
                target_precondition = PrivateFileWritePrecondition.from_opened(opened)
                raw = opened.stream.read(MAX_OVERLAY_FILE_BYTES + 1)
        except FileNotFoundError:
            return None, PrivateFileWritePrecondition.missing()
        if len(raw) > MAX_OVERLAY_FILE_BYTES:
            raise OverlayLimitError(
                f"overlay file exceeds maximum {MAX_OVERLAY_FILE_BYTES} bytes"
            )
        return raw, target_precondition

    @staticmethod
    def _validated_record_list(root: object) -> tuple[int, list[object]]:
        if not isinstance(root, dict):
            raise OverlayValidationError("overlay root must be an object")
        _require_exact_fields(root, _ROOT_FIELDS, "overlay root")
        schema_version = root["schema_version"]
        if schema_version not in {1, OVERLAY_SCHEMA_VERSION}:
            raise OverlayValidationError("unsupported overlay schema version")
        records = root["records"]
        if not isinstance(records, list):
            raise OverlayValidationError("records must be a list")
        if len(records) > MAX_OVERLAY_RECORDS:
            raise OverlayLimitError(
                f"overlay records exceed maximum {MAX_OVERLAY_RECORDS}"
            )
        return schema_version, records


def _decode_record(raw: object, *, schema_version: int) -> ResearchPresentationOverlay:
    if not isinstance(raw, dict):
        raise OverlayValidationError("record must be an object")
    _reject_forbidden_keys(raw)
    _require_exact_fields(
        raw,
        _V1_RECORD_FIELDS if schema_version == 1 else _V2_RECORD_FIELDS,
        "record",
    )

    raw_key = raw["key"]
    if not isinstance(raw_key, dict):
        raise OverlayValidationError("qualified key must be an object")
    _require_exact_fields(raw_key, _KEY_FIELDS, "qualified key")
    ref = QualifiedWorkspaceRef(**raw_key)
    _validate_ref_lengths(ref)

    revision = raw["revision"]
    if type(revision) is not int or revision < 1:
        raise OverlayValidationError("revision must be a positive integer")

    raw_preferences = raw["pane_preferences"]
    if not isinstance(raw_preferences, dict):
        raise OverlayValidationError("pane_preferences must be an object")
    _require_exact_fields(raw_preferences, _PREFERENCE_FIELDS, "pane_preferences")
    preferences = ResearchPanePreferences(
        sources_open=raw_preferences["sources_open"],
        studio_open=raw_preferences["studio_open"],
        preferred_companion=raw["preferred_companion"],
    )
    created_at = _validated_timestamp(raw["created_at"], "created_at")
    updated_at = _validated_timestamp(raw["updated_at"], "updated_at")
    folders = (
        () if schema_version == 1 else _decode_source_folders(raw["source_folders"])
    )
    annotations = (
        ()
        if schema_version == 1
        else _decode_source_annotations(raw["source_annotations"])
    )
    _validate_source_overlay(folders, annotations)
    return ResearchPresentationOverlay(
        ref=ref,
        revision=revision,
        preferences=preferences,
        created_at=created_at,
        updated_at=updated_at,
        source_folders=folders,
        source_annotations=annotations,
    )


def _encode_store(
    records: Iterable[ResearchPresentationOverlay],
    quarantined: Iterable[QuarantinedOverlayRecord] = (),
) -> str:
    payload = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "records": [
            *(_encode_record(record) for record in records),
            *(item.raw for item in quarantined),
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _encode_record(record: ResearchPresentationOverlay) -> dict[str, object]:
    return {
        "key": {
            "data_source": record.ref.data_source.value,
            "workspace_id": record.ref.workspace_id,
            "server_profile_id": record.ref.server_profile_id,
            "principal_id": record.ref.principal_id,
        },
        "revision": record.revision,
        "pane_preferences": {
            "sources_open": record.preferences.sources_open,
            "studio_open": record.preferences.studio_open,
        },
        "preferred_companion": record.preferences.preferred_companion,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "source_folders": [
            {
                "folder_id": folder.folder_id,
                "name": folder.name,
                "parent_folder_id": folder.parent_folder_id,
                "source_ids": list(folder.source_ids),
            }
            for folder in record.source_folders
        ],
        "source_annotations": [
            {
                "annotation_id": annotation.annotation_id,
                "source_id": annotation.source_id,
                "quote": annotation.quote,
                "note": annotation.note,
                "created_at": annotation.created_at,
                "updated_at": annotation.updated_at,
            }
            for annotation in record.source_annotations
        ],
    }


def _decode_source_folders(value: object) -> tuple[ResearchSourceFolder, ...]:
    if not isinstance(value, list):
        raise OverlayValidationError("source_folders must be a list")
    if len(value) > MAX_SOURCE_FOLDERS:
        raise OverlayLimitError("source_folders exceed the declared bound")
    folders: list[ResearchSourceFolder] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise OverlayValidationError("source folder must be an object")
        _require_exact_fields(raw, _SOURCE_FOLDER_FIELDS, "source folder")
        raw_ids = raw["source_ids"]
        if not isinstance(raw_ids, list):
            raise OverlayValidationError("source_ids must be a list")
        folders.append(
            ResearchSourceFolder(
                folder_id=raw["folder_id"],
                name=raw["name"],
                parent_folder_id=raw["parent_folder_id"],
                source_ids=tuple(raw_ids),
            )
        )
    return tuple(folders)


def _decode_source_annotations(
    value: object,
) -> tuple[ResearchSourceAnnotation, ...]:
    if not isinstance(value, list):
        raise OverlayValidationError("source_annotations must be a list")
    if len(value) > MAX_SOURCE_ANNOTATIONS:
        raise OverlayLimitError("source_annotations exceed the declared bound")
    annotations: list[ResearchSourceAnnotation] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise OverlayValidationError("source annotation must be an object")
        _require_exact_fields(raw, _SOURCE_ANNOTATION_FIELDS, "source annotation")
        annotations.append(ResearchSourceAnnotation(**raw))
    return tuple(annotations)


def _validate_source_overlay(
    folders: tuple[ResearchSourceFolder, ...],
    annotations: tuple[ResearchSourceAnnotation, ...],
) -> None:
    if len(folders) > MAX_SOURCE_FOLDERS:
        raise OverlayLimitError("source_folders exceed the declared bound")
    if len(annotations) > MAX_SOURCE_ANNOTATIONS:
        raise OverlayLimitError("source_annotations exceed the declared bound")
    if any(not isinstance(item, ResearchSourceFolder) for item in folders):
        raise OverlayValidationError("source_folders contain an invalid record")
    if any(not isinstance(item, ResearchSourceAnnotation) for item in annotations):
        raise OverlayValidationError("source_annotations contain an invalid record")
    folder_ids = [item.folder_id for item in folders]
    annotation_ids = [item.annotation_id for item in annotations]
    if len(folder_ids) != len(set(folder_ids)):
        raise OverlayValidationError("folder ids must be unique")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise OverlayValidationError("annotation ids must be unique")
    known_folders = set(folder_ids)
    if any(
        folder.parent_folder_id and folder.parent_folder_id not in known_folders
        for folder in folders
    ):
        raise OverlayValidationError("parent folder does not exist")
    by_folder_id = {folder.folder_id: folder for folder in folders}
    for folder in folders:
        visited: set[str] = set()
        current = folder
        depth = 1
        while current.parent_folder_id:
            if current.folder_id in visited:
                raise OverlayValidationError("folder ancestry contains a cycle")
            visited.add(current.folder_id)
            depth += 1
            if depth > MAX_FOLDER_DEPTH:
                raise OverlayLimitError(
                    f"folder ancestry exceeds maximum depth {MAX_FOLDER_DEPTH}"
                )
            current = by_folder_id[current.parent_folder_id]


def _validated_source_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OverlayValidationError(f"{field_name} must be nonblank text")
    normalized = value.strip()
    if len(normalized) > MAX_SOURCE_ID_CHARS or len(normalized.encode("utf-8")) > 4096:
        raise OverlayLimitError(f"{field_name} exceeds the declared bound")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise OverlayValidationError(f"{field_name} contains control characters")
    return normalized


def _validated_plain_text(value: object, field_name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OverlayValidationError(f"{field_name} must be nonblank text")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise OverlayLimitError(f"{field_name} exceeds the declared bound")
    if _PRIVATE_TEXT.search(normalized):
        raise OverlayValidationError(f"{field_name} contains private material")
    return normalized


def _validated_annotation_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise OverlayValidationError(f"{field_name} must be text")
    normalized = value.strip()
    if len(normalized) > MAX_ANNOTATION_TEXT_CHARS:
        raise OverlayLimitError(f"{field_name} exceeds the declared bound")
    if _PRIVATE_TEXT.search(normalized):
        raise OverlayValidationError(f"{field_name} contains private material")
    return normalized


def _require_exact_fields(
    value: Mapping[str, object], allowed: frozenset[str], owner: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise OverlayValidationError(f"{owner} keys must be text")
    actual = set(value)
    if actual != allowed:
        raise OverlayValidationError(f"{owner} fields are invalid")


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise OverlayValidationError("overlay keys must be text")
            parts = key.lower().replace("-", "_").split("_")
            if _FORBIDDEN_KEY_PARTS.intersection(parts):
                raise OverlayValidationError("forbidden overlay field")
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _validate_ref_lengths(ref: QualifiedWorkspaceRef) -> None:
    for field_name in ("workspace_id", "server_profile_id", "principal_id"):
        value = getattr(ref, field_name)
        if len(value) > MAX_IDENTITY_CHARS:
            raise OverlayValidationError(
                f"{field_name} exceeds maximum {MAX_IDENTITY_CHARS} characters"
            )


def _validated_timestamp(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OverlayValidationError(f"{field_name} must be nonblank text")
    if len(value) > MAX_TIMESTAMP_CHARS:
        raise OverlayValidationError(
            f"{field_name} exceeds maximum {MAX_TIMESTAMP_CHARS} characters"
        )
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise OverlayValidationError(f"{field_name} must be ISO-8601") from None
    return value
