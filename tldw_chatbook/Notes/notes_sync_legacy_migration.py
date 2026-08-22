"""Read-only canonical capture of legacy Database Notes sync metadata."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


JsonScalar: TypeAlias = str | int | float | bool | None
_DEFAULT_SYNC_DIRECTORY = "~/Documents/Notes"
_DEFAULT_SYNC_DIRECTION = "bidirectional"
_DEFAULT_CONFLICT_RESOLUTION = "newer_wins"
_MAX_PATH_LENGTH = 32_768
_MAX_ID_LENGTH = 256
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_DIRECTION_ALIASES = {
    "disk_to_db": "folder_to_notes",
    "folder_to_notes": "folder_to_notes",
    "db_to_disk": "notes_to_folder",
    "notes_to_folder": "notes_to_folder",
    "bidirectional": "bidirectional",
}
_NOTE_FIELDS = (
    "id",
    "file_path_on_disk",
    "relative_file_path_on_disk",
    "sync_root_folder",
    "last_synced_disk_file_hash",
    "last_synced_disk_file_mtime",
    "is_externally_synced",
    "sync_strategy",
    "sync_excluded",
    "file_extension",
    "version",
    "deleted",
)
_CONFLICT_FIELDS = (
    "id",
    "session_id",
    "note_id",
    "file_path",
    "conflict_type",
    "db_content_hash",
    "disk_content_hash",
    "db_modified_time",
    "disk_modified_time",
    "resolution",
    "resolved_at",
)


class LegacyNotesSyncSourceError(RuntimeError):
    """Report a bounded legacy-source capture failure."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True, repr=False, init=False)
class LegacyNotesSyncSourceSnapshot:
    """One immutable private source-revision input and its canonical digest."""

    _canonical_source: str = field(repr=False)
    digest: str = field(repr=False)

    def __init__(self, source: Mapping[str, object], digest: str) -> None:
        canonical = _canonical_bytes(source)
        observed_digest = hashlib.sha256(canonical).hexdigest()
        if digest != observed_digest:
            raise LegacyNotesSyncSourceError("source_digest_mismatch")
        object.__setattr__(self, "_canonical_source", canonical.decode("utf-8"))
        object.__setattr__(self, "digest", observed_digest)

    @property
    def source(self) -> dict[str, object]:
        """Return a defensive copy of the authority bound by ``digest``."""
        return cast(dict[str, object], json.loads(self._canonical_source))

    def __repr__(self) -> str:
        """Return no paths, note IDs, counts, or digests."""
        return "LegacyNotesSyncSourceSnapshot(redacted=True)"


def _real_value(value: object) -> str | None:
    if value is None:
        return None
    number: float | None = None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        pass
    if number is None:
        raise LegacyNotesSyncSourceError("invalid_source_type")
    return number.hex() if math.isfinite(number) else "invalid_non_finite_real"


def _json_scalar(value: object, *, config: bool = False) -> JsonScalar:
    if value is None or type(value) in (str, int, bool):
        return value  # type: ignore[return-value]
    if type(value) is float and math.isfinite(value):
        return value
    reason = "invalid_config_type" if config else "invalid_source_type"
    raise LegacyNotesSyncSourceError(reason)


def _text(value: object, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str:
        raise LegacyNotesSyncSourceError("invalid_source_type")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise LegacyNotesSyncSourceError("invalid_source_type")
    return value


def _boolean(value: object) -> bool:
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise LegacyNotesSyncSourceError("invalid_source_type")


def _exact_mapping(row: Mapping[str, object], fields: tuple[str, ...]) -> None:
    if len(row) != len(fields) or set(row) != set(fields):
        raise LegacyNotesSyncSourceError("invalid_source_shape")


def _canonical_note(row: Mapping[str, object]) -> dict[str, object]:
    _exact_mapping(row, _NOTE_FIELDS)
    return {
        "id": _text(row["id"]),
        "file_path_on_disk": _text(row["file_path_on_disk"], nullable=True),
        "relative_file_path_on_disk": _text(
            row["relative_file_path_on_disk"], nullable=True
        ),
        "sync_root_folder": _text(row["sync_root_folder"], nullable=True),
        "last_synced_disk_file_hash": _text(
            row["last_synced_disk_file_hash"], nullable=True
        ),
        "last_synced_disk_file_mtime": _real_value(row["last_synced_disk_file_mtime"]),
        "is_externally_synced": _boolean(row["is_externally_synced"]),
        "sync_strategy": _text(row["sync_strategy"], nullable=True),
        "sync_excluded": _boolean(row["sync_excluded"]),
        "file_extension": _text(row["file_extension"], nullable=True),
        "version": _integer(row["version"]),
        "deleted": _boolean(row["deleted"]),
    }


def _canonical_conflict(row: Mapping[str, object]) -> dict[str, object]:
    _exact_mapping(row, _CONFLICT_FIELDS)
    return {
        "id": _integer(row["id"]),
        "session_id": _text(row["session_id"]),
        "note_id": _text(row["note_id"], nullable=True),
        "file_path": _text(row["file_path"]),
        "conflict_type": _text(row["conflict_type"]),
        "db_content_hash": _text(row["db_content_hash"], nullable=True),
        "disk_content_hash": _text(row["disk_content_hash"], nullable=True),
        "db_modified_time": _text(row["db_modified_time"], nullable=True),
        "disk_modified_time": _real_value(row["disk_modified_time"]),
        "resolution": _text(row["resolution"], nullable=True),
        "resolved_at": _text(row["resolved_at"], nullable=True),
    }


def _source_revision(
    config: Mapping[str, object],
    notes: Sequence[Mapping[str, object]],
    conflicts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the exact canonical source-revision value."""
    try:
        canonical_config = {
            key: _json_scalar(config[key], config=True)
            for key in (
                "sync_conflict_resolution",
                "sync_direction",
                "sync_directory",
            )
        }
    except KeyError:
        raise LegacyNotesSyncSourceError("invalid_source_shape") from None
    canonical_notes = tuple(
        sorted(
            (_canonical_note(row) for row in notes),
            key=lambda row: cast(str, row["id"]),
        )
    )
    canonical_conflicts = tuple(
        sorted(
            (_canonical_conflict(row) for row in conflicts),
            key=lambda row: cast(int, row["id"]),
        )
    )
    return {
        "config": canonical_config,
        "conflicts": canonical_conflicts,
        "notes": canonical_notes,
        "type": "tldw_notes_sync_legacy_source_revision",
        "version": 1,
    }


def _config_projection(values: Mapping[str, object]) -> dict[str, JsonScalar]:
    notes = values.get("notes", {})
    if not isinstance(notes, Mapping):
        raise LegacyNotesSyncSourceError("invalid_config_type")
    directory = notes.get("sync_directory", _DEFAULT_SYNC_DIRECTORY)
    direction = notes.get("sync_direction", _DEFAULT_SYNC_DIRECTION)
    if "sync_conflict_resolution" in notes:
        conflict_resolution = notes["sync_conflict_resolution"]
    else:
        conflict_resolution = notes.get(
            "conflict_resolution", _DEFAULT_CONFLICT_RESOLUTION
        )
    return {
        "sync_conflict_resolution": _json_scalar(conflict_resolution, config=True),
        "sync_direction": _json_scalar(direction, config=True),
        "sync_directory": _json_scalar(directory, config=True),
    }


def capture_legacy_source(
    notes_db: CharactersRAGDB,
) -> LegacyNotesSyncSourceSnapshot:
    """Capture one fresh config/ChaChaNotes projection without mutation.

    Args:
        notes_db: Real ChaChaNotes owner used for the bounded source read.

    Returns:
        A redacted immutable snapshot bound to its canonical digest.
    """
    from tldw_chatbook.config import get_atomic_config_snapshot

    config = _config_projection(get_atomic_config_snapshot().values)
    notes, conflicts = notes_db.read_legacy_notes_sync_source_rows()
    source = _source_revision(config, notes, conflicts)
    return LegacyNotesSyncSourceSnapshot(
        source,
        _canonical_digest(source),
    )


def map_legacy_direction(value: object) -> tuple[str, str | None]:
    """Map one exact legacy direction to the v2 vocabulary.

    Args:
        value: Exact stored legacy scalar.

    Returns:
        Mapped direction and optional bounded review reason.
    """
    if type(value) is str and value in _DIRECTION_ALIASES:
        return _DIRECTION_ALIASES[value], None
    return "unspecified", "legacy_direction_invalid"


def legacy_value_digest(value: object) -> str:
    """Hash one exact legacy JSON scalar before path-text admission.

    Args:
        value: Exact legacy scalar.

    Returns:
        Lowercase SHA-256 hexadecimal text.
    """
    scalar = _json_scalar(value)
    return _canonical_digest(
        {"type": "tldw_notes_sync_legacy_value", "value": scalar, "version": 1}
    )


def _bounded_text(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > _MAX_PATH_LENGTH
        or "\x00" in value
    ):
        raise LegacyNotesSyncSourceError(f"invalid_{field_name}")
    return value


def _digest(value: object) -> str:
    if type(value) is not str or _DIGEST_PATTERN.fullmatch(value) is None:
        raise LegacyNotesSyncSourceError("invalid_locator_digest")
    return value


def legacy_root_locator_digest(lexical_root_path: object) -> str:
    """Hash one exact bounded lexical root without path normalization.

    Args:
        lexical_root_path: Stored lexical root text.

    Returns:
        Lowercase SHA-256 hexadecimal text.
    """
    path = _bounded_text(lexical_root_path, field_name="root_path")
    return _canonical_digest(
        {
            "lexical_root_path": path,
            "type": "tldw_notes_sync_legacy_root_locator",
            "version": 1,
        }
    )


def legacy_binding_locator_digest(
    note_id: object,
    lexical_relative_path: object,
    root_locator_digest: object,
) -> str:
    """Hash one exact recognizable legacy binding locator.

    Args:
        note_id: Exact bounded note identity.
        lexical_relative_path: Exact bounded stored relative path.
        root_locator_digest: Canonical root locator digest.

    Returns:
        Lowercase SHA-256 hexadecimal text.
    """
    if (
        type(note_id) is not str
        or not note_id
        or len(note_id) > _MAX_ID_LENGTH
        or "\x00" in note_id
    ):
        raise LegacyNotesSyncSourceError("invalid_note_id")
    relative_path = _bounded_text(lexical_relative_path, field_name="relative_path")
    root_digest = _digest(root_locator_digest)
    return _canonical_digest(
        {
            "lexical_relative_path": relative_path,
            "note_id": note_id,
            "root_locator_digest": root_digest,
            "type": "tldw_notes_sync_legacy_binding_locator",
            "version": 1,
        }
    )


def legacy_item_locator_digest(item_kind: object, legacy_primary_key: object) -> str:
    """Hash one canonical root, binding, conflict, or rejected item identity.

    Args:
        item_kind: Canonical migration item kind.
        legacy_primary_key: Canonical valid or rejected source identity.

    Returns:
        Lowercase SHA-256 hexadecimal text.
    """
    if item_kind not in ("root", "binding", "legacy_conflict"):
        raise LegacyNotesSyncSourceError("invalid_item_kind")
    try:
        if item_kind == "legacy_conflict":
            if type(legacy_primary_key) is not int:
                raise LegacyNotesSyncSourceError("invalid_item_locator")
        elif type(legacy_primary_key) is str:
            _digest(legacy_primary_key)
        elif item_kind == "root" and isinstance(legacy_primary_key, Mapping):
            if (
                set(legacy_primary_key) != {"field", "value_digest"}
                or legacy_primary_key["field"] != "notes.sync_directory"
            ):
                raise LegacyNotesSyncSourceError("invalid_item_locator")
            _digest(legacy_primary_key["value_digest"])
        elif item_kind == "binding" and isinstance(legacy_primary_key, Mapping):
            if set(legacy_primary_key) != {
                "note_id",
                "relative_value_digest",
                "root_value_digest",
            }:
                raise LegacyNotesSyncSourceError("invalid_item_locator")
            note_id = legacy_primary_key["note_id"]
            if (
                type(note_id) is not str
                or not note_id
                or len(note_id) > _MAX_ID_LENGTH
                or "\x00" in note_id
            ):
                raise LegacyNotesSyncSourceError("invalid_item_locator")
            _digest(legacy_primary_key["relative_value_digest"])
            _digest(legacy_primary_key["root_value_digest"])
        else:
            raise LegacyNotesSyncSourceError("invalid_item_locator")
    except LegacyNotesSyncSourceError:
        raise LegacyNotesSyncSourceError("invalid_item_locator") from None
    return _canonical_digest(
        {
            "item_kind": item_kind,
            "legacy_primary_key": legacy_primary_key,
            "type": "tldw_notes_sync_legacy_item_locator",
            "version": 1,
        }
    )


def rejected_root_item_locator_digest(value: object) -> str:
    """Hash a rejected root without retaining its malformed raw value.

    Args:
        value: Exact rejected legacy scalar.

    Returns:
        Canonical migration-item locator digest.
    """
    primary_key = {
        "field": "notes.sync_directory",
        "value_digest": legacy_value_digest(value),
    }
    return legacy_item_locator_digest("root", primary_key)


def rejected_binding_item_locator_digest(
    note_id: object,
    relative_value: object,
    root_value: object,
) -> str:
    """Hash a rejected binding without retaining malformed path values.

    Args:
        note_id: Exact bounded note identity.
        relative_value: Exact rejected relative-path scalar.
        root_value: Exact rejected root scalar.

    Returns:
        Canonical migration-item locator digest.
    """
    if (
        type(note_id) is not str
        or not note_id
        or len(note_id) > _MAX_ID_LENGTH
        or "\x00" in note_id
    ):
        raise LegacyNotesSyncSourceError("invalid_note_id")
    primary_key = {
        "note_id": note_id,
        "relative_value_digest": legacy_value_digest(relative_value),
        "root_value_digest": legacy_value_digest(root_value),
    }
    return legacy_item_locator_digest("binding", primary_key)


__all__ = [
    "LegacyNotesSyncSourceError",
    "LegacyNotesSyncSourceSnapshot",
    "capture_legacy_source",
    "legacy_binding_locator_digest",
    "legacy_item_locator_digest",
    "legacy_root_locator_digest",
    "legacy_value_digest",
    "map_legacy_direction",
    "rejected_binding_item_locator_digest",
    "rejected_root_item_locator_digest",
]
