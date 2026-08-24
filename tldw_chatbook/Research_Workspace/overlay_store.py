"""Private, bounded persistence for Research Workspace pane preferences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from tldw_chatbook.Utils.private_paths import (
    atomic_private_write_text,
    lexical_path,
    open_private_binary,
    secure_private_directory,
)

from .contracts import QualifiedWorkspaceRef
from .layout_state import ResearchPanePreferences


OVERLAY_SCHEMA_VERSION = 1
MAX_OVERLAY_FILE_BYTES = 1024 * 1024
MAX_OVERLAY_RECORDS = 512
MAX_IDENTITY_CHARS = 256
MAX_TIMESTAMP_CHARS = 64

_ROOT_FIELDS = frozenset({"schema_version", "records"})
_RECORD_FIELDS = frozenset(
    {
        "key",
        "revision",
        "pane_preferences",
        "preferred_companion",
        "created_at",
        "updated_at",
    }
)
_KEY_FIELDS = frozenset(
    {"data_source", "workspace_id", "server_profile_id", "principal_id"}
)
_PREFERENCE_FIELDS = frozenset({"sources_open", "studio_open"})
_FORBIDDEN_KEY_PARTS = frozenset(
    {"api", "body", "content", "password", "path", "secret", "token"}
)


class OverlayValidationError(ValueError):
    """Raised when the overlay container cannot be validated safely."""


class OverlayLimitError(OverlayValidationError):
    """Raised when the overlay exceeds a declared v1 bound."""


class OverlayConflictError(RuntimeError):
    """Raised when the target overlay revision changed before replacement."""


@dataclass(frozen=True, slots=True)
class ResearchPresentationOverlay:
    ref: QualifiedWorkspaceRef
    revision: int
    preferences: ResearchPanePreferences
    created_at: str
    updated_at: str


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


class ResearchPresentationOverlayStore:
    """Persist presentation-only records under one application-owned directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = lexical_path(path)
        self.application_owned_directory = self.path.parent

    def load_all(self) -> OverlayLoadResult:
        """Load valid records while independently quarantining invalid ones."""

        try:
            payload = self._read_payload()
        except UnicodeDecodeError as exc:
            return OverlayLoadResult(
                {},
                (QuarantinedOverlayRecord(-1, str(exc), None),),
            )
        if payload is None:
            return OverlayLoadResult({})
        try:
            root = json.loads(payload)
            records = self._validated_record_list(root)
        except OverlayLimitError:
            raise
        except (json.JSONDecodeError, OverlayValidationError) as exc:
            return OverlayLoadResult(
                {},
                (QuarantinedOverlayRecord(-1, str(exc), payload),),
            )

        decoded: dict[QualifiedWorkspaceRef, ResearchPresentationOverlay] = {}
        quarantined: list[QuarantinedOverlayRecord] = []
        for index, raw in enumerate(records):
            try:
                record = _decode_record(raw)
                if record.ref in decoded:
                    raise OverlayValidationError("duplicate qualified key")
            except (TypeError, ValueError) as exc:
                quarantined.append(QuarantinedOverlayRecord(index, str(exc), raw))
                continue
            decoded[record.ref] = record
        return OverlayLoadResult(decoded, tuple(quarantined))

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
        if current is None and len(loaded.records) >= MAX_OVERLAY_RECORDS:
            raise OverlayLimitError(
                f"overlay records exceed maximum {MAX_OVERLAY_RECORDS}"
            )

        saved = ResearchPresentationOverlay(
            ref=ref,
            revision=current_revision + 1,
            preferences=preferences,
            created_at=current.created_at if current is not None else now,
            updated_at=now,
        )
        records = dict(loaded.records)
        records[ref] = saved
        serialized = _encode_store(records.values())
        if len(serialized.encode("utf-8")) > MAX_OVERLAY_FILE_BYTES:
            raise OverlayLimitError(
                f"overlay file exceeds maximum {MAX_OVERLAY_FILE_BYTES} bytes"
            )

        secure_private_directory(
            self.application_owned_directory,
            create=True,
            application_owned=True,
        )
        atomic_private_write_text(
            self.path,
            serialized,
            application_owned_directory=self.application_owned_directory,
        )
        return saved

    def _read_payload(self) -> str | None:
        secure_private_directory(
            self.application_owned_directory,
            create=True,
            application_owned=True,
        )
        try:
            with open_private_binary(self.path) as opened:
                raw = opened.stream.read(MAX_OVERLAY_FILE_BYTES + 1)
        except FileNotFoundError:
            return None
        if len(raw) > MAX_OVERLAY_FILE_BYTES:
            raise OverlayLimitError(
                f"overlay file exceeds maximum {MAX_OVERLAY_FILE_BYTES} bytes"
            )
        return raw.decode("utf-8")

    @staticmethod
    def _validated_record_list(root: object) -> list[object]:
        if not isinstance(root, dict):
            raise OverlayValidationError("overlay root must be an object")
        _require_exact_fields(root, _ROOT_FIELDS, "overlay root")
        if root["schema_version"] != OVERLAY_SCHEMA_VERSION:
            raise OverlayValidationError("unsupported overlay schema version")
        records = root["records"]
        if not isinstance(records, list):
            raise OverlayValidationError("records must be a list")
        if len(records) > MAX_OVERLAY_RECORDS:
            raise OverlayLimitError(
                f"overlay records exceed maximum {MAX_OVERLAY_RECORDS}"
            )
        return records


def _decode_record(raw: object) -> ResearchPresentationOverlay:
    if not isinstance(raw, dict):
        raise OverlayValidationError("record must be an object")
    _reject_forbidden_keys(raw)
    _require_exact_fields(raw, _RECORD_FIELDS, "record")

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
    return ResearchPresentationOverlay(
        ref=ref,
        revision=revision,
        preferences=preferences,
        created_at=created_at,
        updated_at=updated_at,
    )


def _encode_store(records: Iterable[ResearchPresentationOverlay]) -> str:
    payload = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "records": [_encode_record(record) for record in records],
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
    }


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
