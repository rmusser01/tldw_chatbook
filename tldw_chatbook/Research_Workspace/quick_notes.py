"""Canonical, authority-qualified Quick Notes contracts."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any
from uuid import RFC_4122, UUID, uuid4

from .contracts import BoundedPageResult, QualifiedWorkspaceRef, WorkspaceDataSource


class ResearchNoteConflictError(RuntimeError):
    """A canonical note owner rejected a stale optimistic mutation."""

    def __init__(self, ref: QualifiedWorkspaceRef, note_id: str) -> None:
        super().__init__(
            "The note changed in its canonical owner. Reload or copy as new."
        )
        self.ref = ref
        self.note_id = str(note_id)


@dataclass(frozen=True, slots=True)
class ResearchNotePageRequest:
    query: str = ""
    limit: int = 20
    offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be text")
        query = self.query.strip()
        if len(query) > 1000:
            raise ValueError("query is too long")
        if type(self.limit) is not int or not 1 <= self.limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(self.offset) is not int or not 0 <= self.offset <= 10_000:
            raise ValueError("offset must be between 0 and 10000")
        object.__setattr__(self, "query", query)


@dataclass(frozen=True, slots=True)
class ResearchNoteSaveRequest:
    title: str
    content: str
    note_id: str | None = None
    tags: tuple[str, ...] = ()
    expected_version: int | None = None
    message_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    operation_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.title, str):
            raise TypeError("title must be text")
        title = self.title.strip() or "Untitled Note"
        if len(title) > 1000:
            raise ValueError("title is invalid")
        if not isinstance(self.content, str):
            raise TypeError("content must be text")
        if len(self.content) > 2_000_000:
            raise ValueError("content is too long")
        note_id = (
            _required_text(self.note_id, "note_id", max_chars=1024)
            if self.note_id is not None
            else None
        )
        if note_id is not None and (
            type(self.expected_version) is not int or self.expected_version < 1
        ):
            raise ValueError("expected_version is required for note updates")
        if note_id is None and self.expected_version is not None:
            raise ValueError("expected_version is only valid for note updates")
        tags = _bounded_values(self.tags, "tags", limit=50, max_chars=200)
        if any(_is_provenance_keyword(tag) for tag in tags):
            raise ValueError("tags cannot use reserved provenance prefixes")
        object.__setattr__(self, "title", title)
        object.__setattr__(
            self,
            "operation_id",
            _required_text(
                self.operation_id or f"research-note-{uuid4().hex}",
                "operation_id",
                max_chars=1024,
            ),
        )
        if re.fullmatch(r"research-note-[0-9a-f]{32}", self.operation_id) is None:
            raise ValueError("operation_id is not an app-minted Quick Note token")
        operation_uuid = UUID(hex=self.operation_id.removeprefix("research-note-"))
        if operation_uuid.version != 4 or operation_uuid.variant != RFC_4122:
            raise ValueError("operation_id is not an app-minted Quick Note token")
        object.__setattr__(self, "note_id", note_id)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(
            self,
            "message_ids",
            _bounded_values(self.message_ids, "message_ids", limit=20, max_chars=1024),
        )
        object.__setattr__(
            self,
            "source_ids",
            _bounded_values(self.source_ids, "source_ids", limit=100, max_chars=1024),
        )


@dataclass(frozen=True, slots=True)
class ResearchQuickNote:
    ref: QualifiedWorkspaceRef
    note_id: str
    title: str
    content: str
    version: int
    tags: tuple[str, ...] = ()
    updated_at: str = ""
    message_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "note_id", _required_text(self.note_id, "note_id", max_chars=1024)
        )
        if not isinstance(self.title, str) or len(self.title) > 1000:
            raise ValueError("title is invalid")
        if not isinstance(self.content, str) or len(self.content) > 2_000_000:
            raise ValueError("content is invalid")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer")
        object.__setattr__(
            self, "tags", _bounded_values(self.tags, "tags", limit=50, max_chars=200)
        )
        object.__setattr__(
            self,
            "message_ids",
            _bounded_values(self.message_ids, "message_ids", limit=20, max_chars=1024),
        )
        object.__setattr__(
            self,
            "source_ids",
            _bounded_values(self.source_ids, "source_ids", limit=100, max_chars=1024),
        )
        if not isinstance(self.updated_at, str):
            raise TypeError("updated_at must be text")
        object.__setattr__(self, "updated_at", self.updated_at.strip())


class ResearchQuickNotesService:
    """Validate note results at the selected authority boundary."""

    def __init__(self, ports: Mapping[WorkspaceDataSource, Any]) -> None:
        self._ports = dict(ports)

    async def list_notes(
        self, ref: QualifiedWorkspaceRef, page: ResearchNotePageRequest
    ) -> ResearchNotePage:
        result = await self._port(ref).list_notes(ref, page)
        if not isinstance(result, BoundedPageResult):
            raise TypeError("Note listing returned an invalid page")
        self._validate_refs(result.items, ref)
        return result

    async def get_note(
        self, ref: QualifiedWorkspaceRef, note_id: str
    ) -> ResearchQuickNote | None:
        result = await self._port(ref).get_note(ref, note_id)
        if result is not None:
            if not isinstance(result, ResearchQuickNote):
                raise TypeError("Note detail returned an invalid note")
            self._validate_refs((result,), ref)
        return result

    async def save_note(
        self, ref: QualifiedWorkspaceRef, request: ResearchNoteSaveRequest
    ) -> ResearchQuickNote:
        result = await self._port(ref).save_note(ref, request)
        if not isinstance(result, ResearchQuickNote):
            raise TypeError("Note save returned an invalid note")
        self._validate_refs((result,), ref)
        if request.note_id is not None and result.note_id != request.note_id:
            raise ValueError("Note save returned a mismatched canonical note id")
        return result

    async def delete_note(
        self,
        ref: QualifiedWorkspaceRef,
        note_id: str,
        expected_version: int,
    ) -> bool:
        result = await self._port(ref).delete_note(ref, note_id, expected_version)
        if type(result) is not bool:
            raise TypeError("Note delete returned an invalid result")
        return result

    def _port(self, ref: QualifiedWorkspaceRef) -> Any:
        port = self._ports.get(ref.data_source)
        if port is None:
            raise RuntimeError(f"No adapter is configured for {ref.data_source.value}")
        return port

    @staticmethod
    def _validate_refs(notes: object, ref: QualifiedWorkspaceRef) -> None:
        for note in notes:
            if not isinstance(note, ResearchQuickNote):
                raise TypeError("Note owner returned an invalid note")
            if note.ref != ref:
                raise ValueError("Request returned a mismatched workspace ref")


ResearchNotePage = BoundedPageResult[ResearchQuickNote]


_PROVENANCE_PREFIXES = {
    "message": "research-message-id:",
    "source": "research-source-id:",
}
_RECEIPT_PROOF_PREFIX = "research-receipt-proof:"


def _required_text(value: object, field_name: str, *, max_chars: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if len(normalized) > max_chars:
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _bounded_values(
    values: object, field_name: str, *, limit: int, max_chars: int
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if len(values) > limit:
        raise ValueError(f"{field_name} exceeds the owner bound")
    normalized = tuple(
        _required_text(value, field_name, max_chars=max_chars) for value in values
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must be unique")
    return normalized


def _is_provenance_keyword(value: str) -> bool:
    return value.startswith(_RECEIPT_PROOF_PREFIX) or any(
        value.startswith(prefix) for prefix in _PROVENANCE_PREFIXES.values()
    )


def encode_note_keywords(request: ResearchNoteSaveRequest) -> list[str]:
    """Encode owner-supported tags and opaque provenance as Notes keywords."""

    keywords = list(request.tags)
    keywords.extend(
        _PROVENANCE_PREFIXES["message"] + _encode_identity(value)
        for value in request.message_ids
    )
    keywords.extend(
        _PROVENANCE_PREFIXES["source"] + _encode_identity(value)
        for value in request.source_ids
    )
    return keywords


def split_note_keywords(
    values: object,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split owner keywords into visible tags and qualified provenance IDs."""

    if not isinstance(values, (list, tuple)):
        return (), (), ()
    tags: list[str] = []
    messages: list[str] = []
    sources: list[str] = []
    for raw in values[:170]:
        if not isinstance(raw, str):
            continue
        value = raw.strip()
        if value.startswith(_PROVENANCE_PREFIXES["message"]):
            decoded = _decode_identity(
                value.removeprefix(_PROVENANCE_PREFIXES["message"])
            )
            if decoded:
                messages.append(decoded)
        elif value.startswith(_PROVENANCE_PREFIXES["source"]):
            decoded = _decode_identity(
                value.removeprefix(_PROVENANCE_PREFIXES["source"])
            )
            if decoded:
                sources.append(decoded)
        elif value.startswith(_RECEIPT_PROOF_PREFIX):
            continue
        elif value and len(value) <= 200:
            tags.append(value)
    return (
        tuple(dict.fromkeys(tags[:50])),
        tuple(dict.fromkeys(messages[:20])),
        tuple(dict.fromkeys(sources[:100])),
    )


def _encode_identity(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_identity(value: str) -> str:
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode(value + padding).decode("utf-8")
        return _required_text(decoded, "provenance identity", max_chars=1024)
    except (UnicodeDecodeError, ValueError):
        return ""
