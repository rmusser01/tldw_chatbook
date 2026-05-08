"""Pure display-state contracts for Library Collections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence

from tldw_chatbook.Utils.input_validation import sanitize_string, validate_text_input


LIBRARY_COLLECTIONS_EMPTY_COPY = "Group saved Library items for Search/RAG, Study, and Console."
LIBRARY_COLLECTIONS_NAME_MAX_LENGTH = 120
LIBRARY_COLLECTIONS_DESCRIPTION_MAX_LENGTH = 500
_DANGEROUS_DISPLAY_PATTERN = re.compile(
    r"<script\b|javascript:|onerror=|onclick=",
    re.IGNORECASE,
)


def _value(record: Any, key: str, fallback: Any = "") -> Any:
    if isinstance(record, Mapping):
        return record.get(key, fallback)
    return getattr(record, key, fallback)


def _collapse(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).strip().split())
    return text or fallback


def _safe_display_text(value: Any, fallback: str = "", *, max_length: int = 500) -> str:
    text = sanitize_string(str(value or ""), max_length=max_length).strip()
    text = " ".join(text.split())
    if not text:
        return fallback
    if _DANGEROUS_DISPLAY_PATTERN.search(text):
        return fallback
    if not validate_text_input(text, max_length=max_length, allow_html=False):
        return fallback
    return text


def _coerce_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_sync_status(value: Any) -> str:
    status = _collapse(value, "local-only").lower()
    return status if status in {"local-only", "sync-unavailable"} else "local-only"


def _updated_at_label(value: Any) -> str:
    text = _collapse(value)
    if not text:
        return "Updated unknown"
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return f"Updated {text}"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return f"Updated {parsed:%Y-%m-%d %H:%M} UTC"


def _collection_name_validation(value: Any) -> tuple[str, str]:
    raw = "" if value is None else str(value)
    collapsed = " ".join(raw.strip().split())
    if not collapsed:
        return "", "Enter a Collection name."
    if len(collapsed) > LIBRARY_COLLECTIONS_NAME_MAX_LENGTH:
        return "", "Collection names must be 120 characters or fewer."
    safe = _safe_display_text(
        collapsed,
        "",
        max_length=LIBRARY_COLLECTIONS_NAME_MAX_LENGTH,
    )
    if not safe:
        return "", "Enter a safe Collection name."
    return safe, ""


@dataclass(frozen=True)
class LibraryCollectionActionState:
    """Display state for one Library Collections action."""

    label: str
    enabled: bool
    widget_id: str
    disabled_reason: str = ""

    @property
    def tooltip(self) -> str:
        return "" if self.enabled else self.disabled_reason


@dataclass(frozen=True)
class LibraryCollectionSummary:
    """List-row display state for one Library Collection."""

    collection_id: str
    name: str
    description: str
    item_count: int
    source_authority: str
    sync_status: str
    created_at: str
    updated_at: str
    selected: bool = False

    @property
    def sync_status_label(self) -> str:
        return f"Sync: {self.sync_status}"

    @property
    def item_count_label(self) -> str:
        suffix = "item" if self.item_count == 1 else "items"
        return f"{self.item_count} {suffix}"

    @property
    def updated_at_label(self) -> str:
        return _updated_at_label(self.updated_at)

    @classmethod
    def from_record(cls, record: Any, *, selected: bool = False) -> "LibraryCollectionSummary":
        collection_id = _safe_display_text(_value(record, "collection_id"), "", max_length=200)
        name = _safe_display_text(
            _value(record, "name"),
            "Untitled Collection",
            max_length=LIBRARY_COLLECTIONS_NAME_MAX_LENGTH,
        )
        return cls(
            collection_id=collection_id,
            name=name,
            description=_safe_display_text(
                _value(record, "description"),
                "",
                max_length=LIBRARY_COLLECTIONS_DESCRIPTION_MAX_LENGTH,
            ),
            item_count=_coerce_count(_value(record, "item_count", 0)),
            source_authority=_safe_display_text(
                _value(record, "source_authority"),
                "local",
                max_length=64,
            ),
            sync_status=_normalize_sync_status(_value(record, "sync_status")),
            created_at=_collapse(_value(record, "created_at")),
            updated_at=_collapse(_value(record, "updated_at")),
            selected=selected,
        )


@dataclass(frozen=True)
class LibraryCollectionDetail:
    """Detail display state for the selected Library Collection."""

    collection_id: str
    name: str
    description: str
    item_count: int
    source_authority: str
    sync_status: str
    created_at: str
    updated_at: str

    @property
    def sync_status_label(self) -> str:
        return f"Sync: {self.sync_status}"

    @property
    def item_count_label(self) -> str:
        suffix = "item" if self.item_count == 1 else "items"
        return f"{self.item_count} {suffix}"

    @property
    def updated_at_label(self) -> str:
        return _updated_at_label(self.updated_at)

    @classmethod
    def from_summary(cls, summary: LibraryCollectionSummary) -> "LibraryCollectionDetail":
        return cls(
            collection_id=summary.collection_id,
            name=summary.name,
            description=summary.description,
            item_count=summary.item_count,
            source_authority=summary.source_authority,
            sync_status=summary.sync_status,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )


@dataclass(frozen=True)
class LibraryCollectionsPanelState:
    """Pure display state for the Library Collections management panel."""

    status: str
    collections: tuple[LibraryCollectionSummary, ...]
    selected_collection_id: str | None
    selected_collection: LibraryCollectionDetail | None
    empty_copy: str
    create_action: LibraryCollectionActionState
    rename_action: LibraryCollectionActionState
    delete_action: LibraryCollectionActionState
    error_message: str = ""
    recovery_copy: str = ""

    @classmethod
    def from_values(
        cls,
        *,
        collections: Sequence[Any],
        selected_collection_id: Any = None,
        status: Any = "ready",
        error_message: Any = "",
        create_name: Any = "",
        rename_name: Any = None,
    ) -> "LibraryCollectionsPanelState":
        records = tuple(
            LibraryCollectionSummary.from_record(record)
            for record in collections
            if _safe_display_text(_value(record, "collection_id"), "", max_length=200)
        )
        requested_status = _collapse(status, "ready").lower()
        if requested_status not in {"loading", "ready", "empty", "error"}:
            requested_status = "ready"

        selected_id = _safe_display_text(selected_collection_id, "", max_length=200)
        if not selected_id and records:
            selected_id = records[0].collection_id
        selected_record = next(
            (record for record in records if record.collection_id == selected_id),
            None,
        )
        selected_id = selected_record.collection_id if selected_record is not None else ""
        selected_detail = (
            LibraryCollectionDetail.from_summary(selected_record)
            if selected_record is not None
            else None
        )
        summary_rows = tuple(
            LibraryCollectionSummary(
                collection_id=record.collection_id,
                name=record.name,
                description=record.description,
                item_count=record.item_count,
                source_authority=record.source_authority,
                sync_status=record.sync_status,
                created_at=record.created_at,
                updated_at=record.updated_at,
                selected=record.collection_id == selected_id,
            )
            for record in records
        )

        if requested_status == "ready" and not summary_rows:
            requested_status = "empty"
        if requested_status == "error":
            error_copy = _safe_display_text(
                error_message,
                "Library Collections are unavailable.",
                max_length=500,
            )
            recovery_copy = f"Unavailable: Library Collections.\nWhy: {error_copy}"
        else:
            error_copy = ""
            recovery_copy = ""

        create_action = _create_action(create_name, summary_rows)
        rename_action = _rename_action(rename_name, selected_detail, summary_rows)
        delete_action = _delete_action(selected_detail)

        return cls(
            status=requested_status,
            collections=summary_rows,
            selected_collection_id=selected_id or None,
            selected_collection=selected_detail,
            empty_copy=LIBRARY_COLLECTIONS_EMPTY_COPY,
            create_action=create_action,
            rename_action=rename_action,
            delete_action=delete_action,
            error_message=error_copy,
            recovery_copy=recovery_copy,
        )


def _name_exists(
    name: str,
    collections: Sequence[LibraryCollectionSummary],
    *,
    excluding_collection_id: str | None = None,
) -> bool:
    normalized = name.casefold()
    return any(
        collection.name.casefold() == normalized
        and collection.collection_id != excluding_collection_id
        for collection in collections
    )


def _create_action(
    create_name: Any,
    collections: Sequence[LibraryCollectionSummary],
) -> LibraryCollectionActionState:
    name, reason = _collection_name_validation(create_name)
    if reason:
        return LibraryCollectionActionState(
            label="Create Collection",
            enabled=False,
            widget_id="library-create-collection",
            disabled_reason=reason,
        )
    if _name_exists(name, collections):
        return LibraryCollectionActionState(
            label="Create Collection",
            enabled=False,
            widget_id="library-create-collection",
            disabled_reason="A Collection with this name already exists.",
        )
    return LibraryCollectionActionState(
        label="Create Collection",
        enabled=True,
        widget_id="library-create-collection",
    )


def _rename_action(
    rename_name: Any,
    selected_collection: LibraryCollectionDetail | None,
    collections: Sequence[LibraryCollectionSummary],
) -> LibraryCollectionActionState:
    if selected_collection is None:
        return LibraryCollectionActionState(
            label="Rename Collection",
            enabled=False,
            widget_id="library-rename-collection",
            disabled_reason="Select a Collection before renaming it.",
        )
    proposed_name = selected_collection.name if rename_name is None else rename_name
    name, reason = _collection_name_validation(proposed_name)
    if reason:
        return LibraryCollectionActionState(
            label="Rename Collection",
            enabled=False,
            widget_id="library-rename-collection",
            disabled_reason=reason,
        )
    if _name_exists(name, collections, excluding_collection_id=selected_collection.collection_id):
        return LibraryCollectionActionState(
            label="Rename Collection",
            enabled=False,
            widget_id="library-rename-collection",
            disabled_reason="A Collection with this name already exists.",
        )
    return LibraryCollectionActionState(
        label="Rename Collection",
        enabled=True,
        widget_id="library-rename-collection",
    )


def _delete_action(
    selected_collection: LibraryCollectionDetail | None,
) -> LibraryCollectionActionState:
    if selected_collection is None:
        return LibraryCollectionActionState(
            label="Delete Collection",
            enabled=False,
            widget_id="library-delete-collection",
            disabled_reason="Select a Collection before deleting it.",
        )
    return LibraryCollectionActionState(
        label="Delete Collection",
        enabled=True,
        widget_id="library-delete-collection",
    )
