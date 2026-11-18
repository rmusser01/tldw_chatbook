"""Request-owned Library reads over one parse of the local Chatbook registry."""

from __future__ import annotations

import hashlib
import json
import zipfile
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from copy import deepcopy
from datetime import UTC, datetime
from math import floor
from pathlib import Path
from typing import Any

from tldw_chatbook.Library.library_artifacts_state import (
    MISSING_TIMESTAMP_ORDER,
    ArtifactKey,
    ArtifactOrderKey,
    ArtifactScope,
    ArtifactSourceWindow,
    ArtifactSummary,
    ReadDirection,
    validate_artifact_window,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _lower(value: str) -> str:
    """Match the SQLite LOWER keys used by the other artifact owners."""
    return value.translate(_ASCII_LOWER)


def _created_order(value: Any) -> int:
    try:
        instant = datetime.fromisoformat(value)
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=UTC)
        return -floor(instant.timestamp())
    except (TypeError, ValueError, OverflowError, OSError):
        return MISSING_TIMESTAMP_ORDER


class ChatbookArtifactSnapshot:
    """Detached registry data; no files or source services are read by windows."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records: dict[int, dict[str, Any]] = {}
        self._window_keys: dict[ArtifactScope, tuple[ArtifactOrderKey, ...]] = {}
        for record in records:
            raw_id = record.get("chatbook_id", record.get("id"))
            native_id = self._native_id(raw_id)
            if "id" in record and self._native_id(record["id"]) != native_id:
                raise ValueError("Inconsistent registry artifact identity")
            if native_id in self._records:
                raise ValueError("Duplicate registry artifact identity")
            self._records[native_id] = record

    @staticmethod
    def _native_id(raw_id: Any) -> int:
        if type(raw_id) is int:
            native_id = raw_id
        elif isinstance(raw_id, str) and raw_id.isascii() and raw_id.isdecimal():
            native_id = int(raw_id)
            if str(native_id) != raw_id:
                raise ValueError("Noncanonical registry artifact identity")
        else:
            raise TypeError("Invalid registry artifact identity")
        ArtifactKey("chatbook", native_id)
        return native_id

    @staticmethod
    def _source_label(record: dict[str, Any]) -> str:
        metadata = record.get("metadata") or {}
        if str(metadata.get("artifact_source") or "").strip().lower() == "console":
            return "Console"
        return str(metadata.get("artifact_source") or "Registered Chatbook").strip()

    @staticmethod
    def _title(record: dict[str, Any]) -> str:
        return str(record.get("name") or "").strip(" ") or "Untitled Chatbook"

    @classmethod
    def _matches(cls, record: dict[str, Any], query: str) -> bool:
        if not query:
            return True
        searchable = (
            cls._title(record),
            cls._source_label(record),
            str(record.get("description") or ""),
            *(str(value) for value in record.get("tags") or []),
            *(str(value) for value in record.get("categories") or []),
        )
        return any(query in _lower(value) for value in searchable)

    @classmethod
    def _order_key(
        cls, scope: ArtifactScope, native_id: int, record: dict[str, Any]
    ) -> ArtifactOrderKey:
        order = (
            _lower(cls._title(record))
            if scope.sort == "title"
            else _created_order(record.get("created_at"))
        )
        return (order, "chatbook", native_id)

    def _ordered_keys(self, scope: ArtifactScope) -> tuple[ArtifactOrderKey, ...]:
        if scope.view not in ("chatbooks", "all") or scope.kept_only:
            return ()
        if scope not in self._window_keys:
            query = _lower(scope.query)
            self._window_keys[scope] = tuple(
                sorted(
                    self._order_key(scope, native_id, record)
                    for native_id, record in self._records.items()
                    if self._matches(record, query)
                )
            )
        return self._window_keys[scope]

    def _summary(self, order_key: ArtifactOrderKey) -> ArtifactSummary:
        native_id = order_key[2]
        record = self._records[native_id]
        source = self._source_label(record)
        metadata = record.get("metadata") or {}
        saved = (
            source == "Console"
            and str(metadata.get("artifact_kind") or "").strip().lower()
            == "assistant-response"
        )
        revision_metadata = {
            name: record.get(name)
            for name in (
                "name",
                "description",
                "tags",
                "categories",
                "created_at",
                "updated_at",
                "artifact_revision",
                "file_path",
            )
        }
        revision_metadata["artifact_kind"] = metadata.get("artifact_kind")
        revision_metadata["artifact_source"] = metadata.get("artifact_source")
        revision = hashlib.sha256(
            json.dumps(revision_metadata, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return ArtifactSummary(
            key=ArtifactKey("chatbook", native_id),
            order_key=order_key,
            title=self._title(record),
            source_label=source,
            copy_label="Saved response" if saved else "Registered",
            status="Saved" if saved else "Registered",
            type_label="Chatbook",
            revision=revision,
            created_at=str(record.get("created_at") or ""),
        )

    def read_artifact_window(
        self,
        scope: ArtifactScope,
        *,
        boundary: ArtifactOrderKey | None,
        direction: ReadDirection,
        limit: int,
        inclusive: bool = False,
    ) -> ArtifactSourceWindow:
        """Rank the inventory and construct only the selected summaries."""
        validate_artifact_window(scope, boundary, direction, limit, inclusive)
        keys = self._ordered_keys(scope)
        if boundary is None:
            before = len(keys) if direction == "before" else 0
            equal = 0
            edge = before
        else:
            before = bisect_left(keys, boundary)
            equal = bisect_right(keys, boundary) - before
            edge = before
            if (direction == "before" and inclusive) or (
                direction == "after" and not inclusive
            ):
                edge += equal
        window = (
            keys[max(0, edge - limit) : edge]
            if direction == "before"
            else keys[edge : edge + limit]
        )
        return ArtifactSourceWindow(
            tuple(self._summary(key) for key in window), len(keys), before, equal
        )

    def get_artifact_summary(
        self, scope: ArtifactScope, key: ArtifactKey
    ) -> ArtifactSummary | None:
        """Read one exact admitted identity from this same registry snapshot."""
        validate_artifact_window(scope, None, "after", 1)
        if (
            scope.view not in ("chatbooks", "all")
            or scope.kept_only
            or key.source != "chatbook"
        ):
            return None
        record = self._records.get(key.native_id)
        if record is None or not self._matches(record, _lower(scope.query)):
            return None
        return self._summary(self._order_key(scope, key.native_id, record))

    def get_record(self, native_id: int) -> dict[str, Any] | None:
        """Return an isolated copy of the selected record, including saved content."""
        record = self._records.get(native_id)
        return deepcopy(record) if record is not None else None

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Enumerate isolated record copies for the existing share-review dialog."""
        for record in self._records.values():
            yield deepcopy(record)


def usable_chatbook_bundle(raw_path: Any) -> tuple[bool, str]:
    """Check a registered export for read-only share capability presentation.

    The share owner still validates and stages the chosen file after explicit
    review. This check never imports, exports, extracts, executes, or publishes.
    """
    if not raw_path:
        return False, "No exported bundle; manage Chatbook packs to create an export."
    try:
        path = validate_path_simple(Path(raw_path).expanduser(), probe_existing=False)
        if path.is_symlink():
            return False, "Export is a symlink; sharing is unavailable."
        if not path.is_file():
            return False, "Exported bundle is missing; manage Chatbook packs."
        if not zipfile.is_zipfile(path):
            return False, "Export is not a readable ZIP; manage Chatbook packs."
    except (OSError, TypeError, ValueError, RuntimeError):
        return False, "Exported bundle is unavailable; manage Chatbook packs."
    return True, "Shares the existing exported ZIP snapshot."
