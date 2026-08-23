"""Tests for the shared LocalLibraryToolService (task-1337, plan Task 5).

The service is the single synchronous core behind the 18 direct Library
tools. These tests use fake backends matching the real service signatures
(keyword-only pagination, notes' leading user_id, async prompt/skill
methods) plus real temporary databases for the cross-backend integration
cases.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import sqlite3
from types import SimpleNamespace
from unittest.mock import ANY

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library import local_library_tool_service as service_module
from tldw_chatbook.Library.library_collections_service import (
    LocalLibraryCollectionsService,
)
from tldw_chatbook.Library.library_tool_contract import (
    LIBRARY_TOOL_DESCRIPTORS,
    MAX_RESULT_BYTES,
    parse_cursor,
    parse_public_id,
    serialized_size,
)
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType


# --------------------------------------------------------------------------
# Item factories (raw backend shapes from the Task 2-4 seams)
# --------------------------------------------------------------------------


def _media_item(index, **overrides):
    item = {
        "id": index,
        "uuid": f"media-uuid-{index}",
        "title": f"Media {index}",
        "media_type": "article",
        "author": "author",
        "ingestion_date": "2026-01-01",
        "last_modified": "2026-08-01",
        "version": 1,
        "preview": f"media preview {index}",
        "keywords": [],
        "keyword_total": 0,
        "keywords_truncated": False,
    }
    item.update(overrides)
    return item


def _note_item(index, **overrides):
    item = {
        "id": f"note-uuid-{index}",
        "title": f"Note {index}",
        "created_at": "2026-08-01T00:00:00Z",
        "last_modified": "2026-08-02T00:00:00Z",
        "version": 1,
        "preview": f"note preview {index}",
        "keywords": [],
        "keyword_total": 0,
        "keywords_truncated": False,
    }
    item.update(overrides)
    return item


def _prompt_item(index, **overrides):
    item = {
        "id": index,
        "uuid": f"prompt-uuid-{index}",
        "name": f"Prompt {index}",
        "author": "author",
        "last_modified": "2026-08-01",
        "version": 1,
        "details_preview": f"prompt details {index}",
        "has_system_prompt": True,
        "has_user_prompt": False,
        "has_prompt_definition": False,
        "keywords": [],
        "keyword_total": 0,
        "keywords_truncated": False,
    }
    item.update(overrides)
    return item


def _skill_item(index, **overrides):
    item = {
        "name": f"skill-{index}",
        "description": f"skill description {index}",
        "trust_blocked": False,
        "trust_status": "trusted",
    }
    item.update(overrides)
    return item


def _conversation_item(index, **overrides):
    item = {
        "id": f"conv-uuid-{index}",
        "title": f"Conversation {index}",
        "created_at": "2026-08-01T00:00:00Z",
        "last_modified": "2026-08-02T00:00:00Z",
        "version": 1,
        "keywords": [],
        "keyword_total": 0,
        "keywords_truncated": False,
    }
    item.update(overrides)
    return item


def _collection_item(index, **overrides):
    item = {
        "collection_id": f"collection-{index}",
        "name": f"Collection {index}",
        "description": f"collection description {index}",
        "item_count": index,
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-02T00:00:00Z",
    }
    item.update(overrides)
    return item


def _text_detail(raw_id, text, *, version=1, **overrides):
    detail = {
        "total_chars": len(text),
        "start": 0,
        "returned_chars": len(text),
        "has_more": False,
        "text": text,
        "version": version,
    }
    detail.update(overrides)
    detail.setdefault("id", raw_id)
    return detail


# --------------------------------------------------------------------------
# Fake backends (signatures mirror the real services)
# --------------------------------------------------------------------------


class _Recorded:
    def __init__(self):
        self.calls = []

    def _record(self, method, kwargs):
        self.calls.append((method, kwargs))


class FakeMediaService(_Recorded):
    def __init__(self, *, items=(), total=0, detail=None):
        super().__init__()
        self._items = list(items)
        self._total = total
        self._detail = detail

    def list_library_media(self, *, limit, offset):
        self._record("list_library_media", {"limit": limit, "offset": offset})
        return {"items": self._items, "total": self._total}

    def search_library_media(self, *, query, limit, offset):
        self._record(
            "search_library_media", {"query": query, "limit": limit, "offset": offset}
        )
        return {"items": self._items, "total": self._total}

    def get_library_media_text(self, media_uuid, *, start, max_chars):
        self._record(
            "get_library_media_text",
            {"uuid": media_uuid, "start": start, "max_chars": max_chars},
        )
        return self._detail


class FakeNotesService(_Recorded):
    def __init__(self, *, items=(), total=0, detail=None, text_source=None):
        super().__init__()
        self._items = list(items)
        self._total = total
        self._detail = detail
        self._text_source = text_source

    def list_library_notes(self, user_id, *, limit, offset):
        self._record(
            "list_library_notes", {"user_id": user_id, "limit": limit, "offset": offset}
        )
        return {"items": self._items, "total": self._total}

    def search_library_notes(self, user_id, *, query, limit, offset):
        self._record(
            "search_library_notes",
            {"user_id": user_id, "query": query, "limit": limit, "offset": offset},
        )
        return {"items": self._items, "total": self._total}

    def get_library_note_text(self, user_id, note_id, *, start, max_chars):
        self._record(
            "get_library_note_text",
            {
                "user_id": user_id,
                "note_id": note_id,
                "start": start,
                "max_chars": max_chars,
            },
        )
        if self._text_source is not None:
            text = self._text_source[start : start + max_chars]
            return _text_detail(
                note_id,
                text,
                start=start,
                returned_chars=len(text),
                has_more=start + len(text) < len(self._text_source),
                total_chars=len(self._text_source),
                title="Source note",
                created_at="2026-08-01",
                last_modified="2026-08-02",
            )
        return self._detail


class FakePromptService(_Recorded):
    def __init__(self, *, items=(), total=0, overview=None, section=None):
        super().__init__()
        self._items = list(items)
        self._total = total
        self._overview = overview
        self._section = section

    async def list_library_prompts(self, *, limit, offset):
        self._record("list_library_prompts", {"limit": limit, "offset": offset})
        return {"items": self._items, "total": self._total}

    async def search_library_prompts(self, query, *, limit, offset):
        self._record(
            "search_library_prompts", {"query": query, "limit": limit, "offset": offset}
        )
        return {"items": self._items, "total": self._total}

    async def get_library_prompt_overview(self, prompt_uuid):
        self._record("get_library_prompt_overview", {"uuid": prompt_uuid})
        return self._overview

    async def get_library_prompt_section(self, prompt_uuid, section, *, start, max_chars):
        self._record(
            "get_library_prompt_section",
            {
                "uuid": prompt_uuid,
                "section": section,
                "start": start,
                "max_chars": max_chars,
            },
        )
        return self._section


class FakeSkillsService(_Recorded):
    def __init__(self, *, items=(), total=0, detail=None, file_segment=None, error=None):
        super().__init__()
        self._items = list(items)
        self._total = total
        self._detail = detail
        self._file_segment = file_segment
        self._error = error

    async def list_library_skills(self, *, limit, offset):
        self._record("list_library_skills", {"limit": limit, "offset": offset})
        return {"items": self._items, "total": self._total}

    async def search_library_skills(self, *, query, limit, offset):
        self._record(
            "search_library_skills", {"query": query, "limit": limit, "offset": offset}
        )
        return {"items": self._items, "total": self._total}

    async def get_library_skill(self, skill_name):
        self._record("get_library_skill", {"skill_name": skill_name})
        if self._error is not None:
            raise self._error
        return self._detail

    async def get_library_skill_file(self, skill_name, file_token, *, start, max_chars):
        self._record(
            "get_library_skill_file",
            {
                "skill_name": skill_name,
                "file_token": file_token,
                "start": start,
                "max_chars": max_chars,
            },
        )
        if self._error is not None:
            raise self._error
        return self._file_segment


class FakeConversationService(_Recorded):
    def __init__(self, *, items=(), total=0, detail=None):
        super().__init__()
        self._items = list(items)
        self._total = total
        self._detail = detail

    def list_library_conversations(self, *, limit, offset):
        self._record("list_library_conversations", {"limit": limit, "offset": offset})
        return {"items": self._items, "total": self._total}

    def search_library_conversations(self, *, query, limit, offset):
        self._record(
            "search_library_conversations",
            {"query": query, "limit": limit, "offset": offset},
        )
        return {"items": self._items, "total": self._total}

    def get_library_conversation_messages(self, conversation_id, **kwargs):
        self._record(
            "get_library_conversation_messages",
            {"conversation_id": conversation_id, **kwargs},
        )
        return self._detail


class FakeCollectionsService(_Recorded):
    def __init__(self, *, items=(), total=0, detail=None):
        super().__init__()
        self._items = list(items)
        self._total = total
        self._detail = detail

    def list_library_collections(self, *, limit, offset):
        self._record("list_library_collections", {"limit": limit, "offset": offset})
        return {"items": self._items, "total": self._total}

    def search_library_collections(self, *, query, limit, offset):
        self._record(
            "search_library_collections",
            {"query": query, "limit": limit, "offset": offset},
        )
        return {"items": self._items, "total": self._total}

    def get_library_collection(self, collection_id, *, limit, offset):
        self._record(
            "get_library_collection",
            {"collection_id": collection_id, "limit": limit, "offset": offset},
        )
        return self._detail


def _backends(**overrides):
    backends = {
        "media_service": FakeMediaService(),
        "notes_service": FakeNotesService(),
        "prompt_service": FakePromptService(),
        "skills_service": FakeSkillsService(),
        "conversation_service": FakeConversationService(),
        "collections_service": FakeCollectionsService(),
        "notes_user_id": "user-1",
    }
    backends.update(overrides)
    return backends


def _service(**overrides):
    return LocalLibraryToolService(**_backends(**overrides))


def _error_code(result):
    return result["error"]["code"]


# --------------------------------------------------------------------------
# Dispatch and runtime validation
# --------------------------------------------------------------------------


def test_descriptor_table_covers_24_tools():
    # 18 task-1337 tools + the 5 chunking-agent-tools siblings (spec §4)
    # + library_save_note (student-workflow spec §4).
    assert len(LIBRARY_TOOL_DESCRIPTORS) == 24


def test_unknown_tool_name_is_invalid_argument():
    result = _service().invoke("library_list_widgets", {})
    assert _error_code(result) == "invalid_argument"


def test_arguments_must_be_a_mapping():
    result = _service().invoke("library_list_notes", ["not", "a", "mapping"])
    assert _error_code(result) == "invalid_argument"


def test_unexpected_argument_key_is_rejected():
    result = _service().invoke("library_list_notes", {"limit": 1, "bogus": True})
    assert _error_code(result) == "invalid_argument"


@pytest.mark.parametrize(
    "arguments",
    [
        {"limit": 0},
        {"limit": -3},
        {"limit": 1.5},
        {"limit": True},
        {"offset": -1},
        {"offset": "2"},
    ],
)
def test_list_page_arguments_fail_closed(arguments):
    result = _service().invoke("library_list_media", arguments)
    assert _error_code(result) == "invalid_argument"


def test_limit_above_maximum_clamps():
    notes = FakeNotesService(items=[_note_item(1)], total=1)
    service = _service(notes_service=notes)
    result = service.invoke("library_list_notes", {"limit": 999})
    assert "error" not in result
    assert notes.calls[0][1]["limit"] == 50
    assert result["limit"] == 50


def test_missing_backend_maps_to_feature_unavailable():
    backends = _backends()
    backends["media_service"] = None
    service = LocalLibraryToolService(**backends)
    result = service.invoke("library_list_media", {})
    assert _error_code(result) == "feature_unavailable"
    assert result["error"]["retryable"] is False


# --------------------------------------------------------------------------
# Media chunk-tool dispatch (chunking-agent-tools Task 3)
# --------------------------------------------------------------------------


class FakeMediaChunkService:
    """Duck-typed stand-in for ``LocalMediaChunkToolService``."""

    def __init__(self, *, payload=None):
        self.calls = []
        self._payload = payload if payload is not None else {"echo": True}

    def invoke(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return self._payload


_CHUNK_TOOL_ARGUMENTS = {
    "library_get_media_structure": {"id": "media:AAAA"},
    "library_get_media_chunk": {"id": "media:AAAA", "chunk_index": 0},
    "library_list_chunk_specs": {"limit": 5},
    "library_save_chunk_spec": {"name": "x", "spec": {"method": "words"}},
    "library_rechunk_media": {"id": "media:AAAA"},
}


@pytest.mark.parametrize("tool_name", list(_CHUNK_TOOL_ARGUMENTS))
def test_media_chunk_tools_route_to_the_chunk_service(tool_name):
    chunk = FakeMediaChunkService()
    service = _service(media_chunk_service=chunk)
    arguments = _CHUNK_TOOL_ARGUMENTS[tool_name]

    result = service.invoke(tool_name, arguments)

    assert result == {"echo": True}
    assert chunk.calls == [(tool_name, arguments)]


def test_media_chunk_tools_without_chunk_service_map_to_feature_unavailable():
    service = _service()
    for tool_name, arguments in _CHUNK_TOOL_ARGUMENTS.items():
        result = service.invoke(tool_name, arguments)
        assert _error_code(result) == "feature_unavailable"


def test_media_chunk_service_error_payloads_pass_through_unchanged():
    chunk = FakeMediaChunkService(
        payload={"error": {"code": "not_found", "message": "m", "retryable": False, "details": {}}}
    )
    service = _service(media_chunk_service=chunk)

    result = service.invoke("library_get_media_structure", {"id": "media:AAAA"})

    assert result is chunk._payload
    assert _error_code(result) == "not_found"


# --------------------------------------------------------------------------
# List operations
# --------------------------------------------------------------------------

LIST_CASES = [
    ("library_list_media", "media_service", "list_library_media", "media"),
    ("library_list_notes", "notes_service", "list_library_notes", "note"),
    ("library_list_prompts", "prompt_service", "list_library_prompts", "prompt"),
    ("library_list_skills", "skills_service", "list_library_skills", "skill"),
    (
        "library_list_conversations",
        "conversation_service",
        "list_library_conversations",
        "conversation",
    ),
    (
        "library_list_collections",
        "collections_service",
        "list_library_collections",
        "collection",
    ),
]

_ITEM_FACTORIES = {
    "media": _media_item,
    "note": _note_item,
    "prompt": _prompt_item,
    "skill": _skill_item,
    "conversation": _conversation_item,
    "collection": _collection_item,
}

_RAW_ID_KEYS = {
    "media": "uuid",
    "note": "id",
    "prompt": "uuid",
    "skill": "name",
    "conversation": "id",
    "collection": "collection_id",
}


@pytest.mark.parametrize(
    "tool_name, service_attr, method, item_type",
    LIST_CASES,
    ids=[case[3] for case in LIST_CASES],
)
def test_list_routes_and_normalizes(tool_name, service_attr, method, item_type):
    factory = _ITEM_FACTORIES[item_type]
    raw_items = [factory(1), factory(2)]
    backends = _backends()
    fake = backends[service_attr]
    fake._items = raw_items
    fake._total = 3
    service = LocalLibraryToolService(**backends)

    result = service.invoke(tool_name, {"limit": 2, "offset": 0})

    assert fake.calls[0][0] == method
    assert fake.calls[0][1]["limit"] == 2
    assert fake.calls[0][1]["offset"] == 0
    assert set(result) == {
        "items",
        "total",
        "limit",
        "offset",
        "has_more",
        "next_offset",
        "response_truncated",
        "omitted_fields",
    }
    assert result["total"] == 3
    assert result["has_more"] is True
    assert result["next_offset"] == 2
    assert result["response_truncated"] is False
    assert result["omitted_fields"] == []
    first = result["items"][0]
    assert first["type"] == item_type
    parsed_type, parsed_raw = parse_public_id(first["id"], expected_type=item_type)
    assert parsed_type == item_type
    assert parsed_raw == str(raw_items[0][_RAW_ID_KEYS[item_type]])
    assert first["keyword_total"] == 0
    assert first["keywords_truncated"] is False


def test_notes_list_envelope_matches_contract_example():
    notes = FakeNotesService(items=[_note_item(1), _note_item(2)], total=3)
    service = _service(notes_service=notes)

    result = service.invoke("library_list_notes", {"limit": 2, "offset": 0})

    assert result == {
        "items": ANY,
        "total": 3,
        "limit": 2,
        "offset": 0,
        "has_more": True,
        "next_offset": 2,
        "response_truncated": False,
        "omitted_fields": [],
    }
    assert notes.calls[0][1]["user_id"] == "user-1"


def test_empty_terminal_page_reports_null_next_offset():
    notes = FakeNotesService(items=[], total=3)
    service = _service(notes_service=notes)

    result = service.invoke("library_list_notes", {"limit": 20, "offset": 3})

    assert result["items"] == []
    assert result["total"] == 3
    assert result["has_more"] is False
    assert result["next_offset"] is None


def test_brief_bounds_titles_keywords_and_previews():
    long_title = "t" * 300
    long_keyword = "k" * 200
    control_title = "has\x00control\x07chars"
    item = _note_item(
        1,
        title=long_title + control_title,
        preview="p" * 500,
        keywords=[long_keyword] * 25,
        keyword_total=25,
        keywords_truncated=True,
    )
    notes = FakeNotesService(items=[item], total=1)
    service = _service(notes_service=notes)

    result = service.invoke("library_list_notes", {})
    brief = result["items"][0]

    assert len(brief["title"].encode("utf-8")) <= 160
    assert brief["title_truncated"] is True
    assert "\x00" not in brief["title"]
    assert len(brief["preview"]) <= 240
    assert len(brief["keywords"]) == 20
    assert all(len(keyword) <= 120 for keyword in brief["keywords"])
    assert brief["keyword_total"] == 25
    assert brief["keywords_truncated"] is True


# --------------------------------------------------------------------------
# Search operations
# --------------------------------------------------------------------------

SEARCH_CASES = [
    ("library_search_media", "media_service", "search_library_media", "media"),
    ("library_search_notes", "notes_service", "search_library_notes", "note"),
    ("library_search_prompts", "prompt_service", "search_library_prompts", "prompt"),
    ("library_search_skills", "skills_service", "search_library_skills", "skill"),
    (
        "library_search_conversations",
        "conversation_service",
        "search_library_conversations",
        "conversation",
    ),
    (
        "library_search_collections",
        "collections_service",
        "search_library_collections",
        "collection",
    ),
]


@pytest.mark.parametrize(
    "tool_name, service_attr, method, item_type",
    SEARCH_CASES,
    ids=[case[3] for case in SEARCH_CASES],
)
def test_search_routes_and_preserves_match_evidence(
    tool_name, service_attr, method, item_type
):
    factory = _ITEM_FACTORIES[item_type]
    raw = factory(
        1, matched_fields=["title", "keywords"], matched_keywords=["needle-kw"]
    )
    backends = _backends()
    fake = backends[service_attr]
    fake._items = [raw]
    fake._total = 1
    service = LocalLibraryToolService(**backends)

    result = service.invoke(tool_name, {"query": "  needle  ", "limit": 5, "offset": 0})

    assert fake.calls[0][0] == method
    # The stripped query is forwarded.
    assert fake.calls[0][1]["query"] == "needle"
    item = result["items"][0]
    assert item["matched_fields"] == ["keywords", "title"]
    assert item["matched_keywords"] == ["needle-kw"]
    parse_public_id(item["id"], expected_type=item_type)


@pytest.mark.parametrize("tool_name", [case[0] for case in SEARCH_CASES])
def test_search_rejects_empty_and_missing_queries(tool_name):
    service = _service()
    for arguments in ({}, {"query": ""}, {"query": "   "}, {"query": 42}):
        result = service.invoke(tool_name, arguments)
        assert _error_code(result) == "invalid_argument", (tool_name, arguments)


def test_search_rejects_overlong_query():
    result = _service().invoke("library_search_notes", {"query": "q" * 1001})
    assert _error_code(result) == "invalid_argument"


# --------------------------------------------------------------------------
# Get operations: text types (notes, media)
# --------------------------------------------------------------------------


def _note_detail(**overrides):
    detail = {
        "id": "note-uuid-1",
        "title": "Note One",
        "created_at": "2026-08-01",
        "last_modified": "2026-08-02",
        "version": 2,
        "total_chars": 10,
        "start": 0,
        "returned_chars": 5,
        "has_more": True,
        "text": "hello",
    }
    detail.update(overrides)
    return detail


def _public_id(item_type, raw):
    from tldw_chatbook.Library.library_tool_contract import make_public_id

    return make_public_id(item_type, raw)


def test_get_note_window_and_continuation_round_trip():
    notes = FakeNotesService(detail=_note_detail())
    service = _service(notes_service=notes)
    public = _public_id("note", "note-uuid-1")

    first = service.invoke("library_get_note", {"id": public, "max_chars": 5})

    assert first["item"]["id"] == public
    assert first["item"]["type"] == "note"
    assert first["item"]["title"] == "Note One"
    content = first["content"]
    assert content["text"] == "hello"
    assert content["start"] == 0
    assert content["end"] == 5
    assert content["total_chars"] == 10
    assert content["requested_max_chars"] == 5
    assert content["returned_chars"] == 5
    assert content["revision"] == "2"
    assert content["has_more"] is True
    state = parse_cursor(content["next_cursor"])
    assert state["item"] == public
    assert state["off"] == 5

    # Continuation passes the cursor offset through to the backend.
    notes._detail = _note_detail(
        start=5, text="world", returned_chars=5, has_more=False
    )
    second = service.invoke(
        "library_get_note", {"id": public, "cursor": content["next_cursor"]}
    )
    assert notes.calls[-1][1]["start"] == 5
    assert second["content"]["text"] == "world"
    assert second["content"]["has_more"] is False
    assert second["content"]["next_cursor"] is None


def test_get_note_cursor_rejects_other_items_and_tampering():
    notes = FakeNotesService(detail=_note_detail())
    service = _service(notes_service=notes)
    public = _public_id("note", "note-uuid-1")
    other = _public_id("note", "note-uuid-2")

    from tldw_chatbook.Library.library_tool_contract import make_cursor

    foreign = make_cursor(item_id=other, revision="2", offset=5)
    result = service.invoke("library_get_note", {"id": public, "cursor": foreign})
    assert _error_code(result) == "invalid_argument"

    tampered = make_cursor(item_id=public, revision="2", offset=5)[:-4] + "AAAA"
    result = service.invoke("library_get_note", {"id": public, "cursor": tampered})
    assert _error_code(result) == "invalid_argument"


def test_get_note_reports_content_changed_on_revision_mismatch():
    notes = FakeNotesService(detail=_note_detail(version=2))
    service = _service(notes_service=notes)
    public = _public_id("note", "note-uuid-1")

    from tldw_chatbook.Library.library_tool_contract import make_cursor

    stale = make_cursor(item_id=public, revision="1", offset=5)
    result = service.invoke("library_get_note", {"id": public, "cursor": stale})

    assert _error_code(result) == "content_changed"
    assert result["error"]["details"]["hint"] == "begin_a_fresh_read"


def test_get_note_missing_returns_not_found():
    notes = FakeNotesService(detail=None)
    service = _service(notes_service=notes)
    result = service.invoke("library_get_note", {"id": _public_id("note", "nope")})
    assert _error_code(result) == "not_found"
    assert result["error"]["retryable"] is False


def test_get_rejects_wrong_type_and_malformed_ids():
    service = _service(notes_service=FakeNotesService(detail=_note_detail()))
    wrong_type = _public_id("media", "media-uuid-1")
    result = service.invoke("library_get_note", {"id": wrong_type})
    assert _error_code(result) == "invalid_argument"

    result = service.invoke("library_get_note", {"id": "not-an-id"})
    assert _error_code(result) == "invalid_argument"

    result = service.invoke("library_get_note", {})
    assert _error_code(result) == "invalid_argument"


def test_get_media_window_includes_metadata_and_never_paths():
    detail = {
        "uuid": "media-uuid-1",
        "title": "Article",
        "media_type": "article",
        "author": "someone",
        "ingestion_date": "2026-01-01",
        "last_modified": "2026-08-01",
        "version": 7,
        "total_chars": 4,
        "start": 0,
        "returned_chars": 4,
        "has_more": False,
        "text": "body",
    }
    media = FakeMediaService(detail=detail)
    service = _service(media_service=media)
    public = _public_id("media", "media-uuid-1")

    result = service.invoke("library_get_media", {"id": public})

    assert result["item"]["media_type"] == "article"
    assert result["item"]["author"] == "someone"
    assert result["content"]["revision"] == "7"
    assert result["content"]["next_cursor"] is None
    assert media.calls[0][1] == {"uuid": "media-uuid-1", "start": 0, "max_chars": 8000}


def test_get_max_chars_validation():
    notes = FakeNotesService(detail=_note_detail())
    service = _service(notes_service=notes)
    public = _public_id("note", "note-uuid-1")

    result = service.invoke("library_get_note", {"id": public, "max_chars": 0})
    assert _error_code(result) == "invalid_argument"

    service.invoke("library_get_note", {"id": public, "max_chars": 999_999})
    assert notes.calls[-1][1]["max_chars"] == 16_000


# --------------------------------------------------------------------------
# Save notes: library_save_note (student-workflow spec §4)
# --------------------------------------------------------------------------


class FakeSaveNotesBackend(_Recorded):
    """In-memory stand-in for the legacy notes interop's write path.

    Mirrors ``NotesInteropService``: ``add_note`` rejects empty titles with
    ``InputError``; ``update_note`` raises the REAL ``ConflictError`` on a
    missing row or a version mismatch (ChaChaNotes_DB.py semantics).
    """

    def __init__(self, notes=None):
        super().__init__()
        self._rows = {  # note_id -> {"title", "content", "version"}
            note_id: dict(row) for note_id, row in (notes or {}).items()
        }

    def add_note(self, user_id, title, content, note_id=None):
        self._record(
            "add_note", {"user_id": user_id, "title": title, "content": content}
        )
        if not isinstance(title, str) or not title.strip():
            raise InputError("Note title cannot be empty.")
        new_id = note_id or f"note-{len(self._rows) + 1}"
        self._rows[new_id] = {"title": title, "content": content, "version": 1}
        return new_id

    def get_note_by_id(self, user_id, note_id):
        self._record("get_note_by_id", {"user_id": user_id, "note_id": note_id})
        row = self._rows.get(note_id)
        return dict(row) if row is not None else None

    def update_note(self, user_id, note_id, update_data, expected_version):
        self._record(
            "update_note",
            {
                "user_id": user_id,
                "note_id": note_id,
                "update_data": dict(update_data),
                "expected_version": expected_version,
            },
        )
        row = self._rows.get(note_id)
        if row is None:
            raise ConflictError(
                "Record not found in notes.", entity="notes", entity_id=note_id
            )
        if row["version"] != expected_version:
            raise ConflictError(
                f"Note ID {note_id} update failed: version mismatch.",
                entity="notes",
                entity_id=note_id,
            )
        row.update(update_data)
        row["version"] = expected_version + 1
        return True


class FakeNotesScopeService:
    """Async scope-seam stand-in over an in-memory folder dict.

    Mirrors the real seams the handler consumes: children-list at the root,
    NON-idempotent create (normalized-path collision -> FolderCollisionError),
    and a safe re-attach. Names are keyed by the same normalize_folder_name
    key the repository uses, so lookups match the real collision semantics.
    """

    def __init__(self):
        self.calls = []
        self._folders = {}  # key -> {"folder_id", "name"}
        self._next_id = 0
        #: When True the next create raises the collision error even if the
        #: folder is absent -- simulating a concurrent create winning the race
        #: (the folder then appears on the re-query).
        self.collision_on_next_create = False
        #: Seam method names whose call raises FolderCapabilityError -- the
        #: deployment-without-folder-support shape (the scope service's own
        #: ``_raise_folder_capability_error`` path).
        self.capability_errors: tuple[str, ...] = ()

    async def list_note_folder_children(
        self, *, scope, parent_id, limit, offset, user_id=None
    ):
        from tldw_chatbook.Notes.note_folder_models import (
            FolderCapabilityError,
            normalize_folder_name,
        )

        self.calls.append(
            ("list_note_folder_children", {"parent_id": parent_id, "user_id": user_id})
        )
        if "list_note_folder_children" in self.capability_errors:
            raise FolderCapabilityError(
                reason_code="folder_list_unsupported",
                user_message="Folder listing is not available for this scope.",
            )
        ordered = sorted(self._folders.values(), key=lambda f: f["name"])
        page = ordered[offset : offset + limit]
        end = offset + len(page)
        return SimpleNamespace(
            folders=tuple(
                SimpleNamespace(
                    folder_id=f["folder_id"], name=f["name"], parent_id=None
                )
                for f in page
            ),
            next_folder_offset=end if page and end < len(ordered) else None,
            _lookup={normalize_folder_name(f["name"]).key: f for f in page},
        )

    async def create_note_folder(self, *, scope, name, parent_id, user_id=None):
        from tldw_chatbook.Notes.note_folder_models import (
            FolderCapabilityError,
            FolderCollisionError,
            normalize_folder_name,
        )

        self.calls.append(("create_note_folder", {"name": name}))
        if "create_note_folder" in self.capability_errors:
            raise FolderCapabilityError(
                reason_code="folder_create_unsupported",
                user_message="Folder creation is not available for this scope.",
            )
        key = normalize_folder_name(name).key
        if key in self._folders:
            raise FolderCollisionError(
                "An active folder already uses the normalized path."
            )
        if self.collision_on_next_create:
            self.collision_on_next_create = False
            # The concurrent winner's folder becomes visible for the re-query.
            self._next_id += 1
            self._folders[key] = {
                "folder_id": f"folder-raced-{self._next_id}",
                "name": name,
            }
            raise FolderCollisionError(
                "An active folder already uses the normalized path."
            )
        self._next_id += 1
        folder = {"folder_id": f"folder-{self._next_id}", "name": name}
        self._folders[key] = folder
        return SimpleNamespace(folder_id=folder["folder_id"], name=name)

    async def attach_note_to_folder(self, *, scope, folder_id, note_id, user_id=None):
        self.calls.append(
            ("attach_note_to_folder", {"folder_id": folder_id, "note_id": note_id})
        )
        return SimpleNamespace(folder_id=folder_id, note_id=note_id)

    def folder_count(self):
        return len(self._folders)


class _DenyingPolicyEnforcer:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.actions = []

    def require_allowed(self, *, action_id):
        self.actions.append(action_id)
        if not self.allowed:
            from tldw_chatbook.runtime_policy.types import PolicyDeniedError

            raise PolicyDeniedError(
                action_id=action_id,
                reason_code="authority_denied",
                user_message="denied by test",
                effective_source="local",
                authority_owner="local",
            )


def _save_service(**overrides):
    backends = _backends(
        notes_service=FakeSaveNotesBackend(),
        notes_scope_service=FakeNotesScopeService(),
    )
    backends.update(overrides)
    return LocalLibraryToolService(**backends)


def test_save_note_note_id_and_version_must_arrive_together():
    # The together-rule (student-workflow spec §4.1): exactly one of
    # note_id/expected_version supplied -> invalid_argument. Pinned FIRST
    # because it is the most-missed edge.
    service = _save_service()
    id_only = service.invoke(
        "library_save_note",
        {"title": "t", "content": "c", "note_id": _public_id("note", "note-1")},
    )
    assert _error_code(id_only) == "invalid_argument"

    version_only = service.invoke(
        "library_save_note", {"title": "t", "content": "c", "expected_version": 1}
    )
    assert _error_code(version_only) == "invalid_argument"


def test_save_note_schema_bounds_match_the_spec():
    schema = LIBRARY_TOOL_DESCRIPTORS["library_save_note"].input_schema
    assert schema["required"] == ["title", "content"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["title"]["maxLength"] == 512
    assert schema["properties"]["title"]["minLength"] == 1
    assert schema["properties"]["content"]["maxLength"] == 100_000
    assert schema["properties"]["content"]["minLength"] == 1
    # 255, not 256: the folder model's normalize_folder_name refuses
    # segments longer than 255 chars -- the contract bound must equal it.
    assert schema["properties"]["folder"]["maxLength"] == 255
    assert schema["properties"]["folder"]["minLength"] == 1
    assert schema["properties"]["note_id"]["maxLength"] == 128
    assert schema["properties"]["expected_version"]["minimum"] == 1
    # The route/item/operation identity the dispatch keys on.
    descriptor = LIBRARY_TOOL_DESCRIPTORS["library_save_note"]
    assert descriptor.item_type == "note"
    assert descriptor.operation == "save"
    assert "Writes local Library data only" in descriptor.description


def test_save_note_rejects_unknown_and_missing_arguments():
    service = _save_service()
    unknown = service.invoke(
        "library_save_note", {"title": "t", "content": "c", "color": "blue"}
    )
    assert _error_code(unknown) == "invalid_argument"

    missing = service.invoke("library_save_note", {"title": "t"})
    assert _error_code(missing) == "invalid_argument"


def test_save_note_description_documents_the_provenance_header():
    description = LIBRARY_TOOL_DESCRIPTORS["library_save_note"].description
    # The header convention rides the DESCRIPTION (convention, not enforced
    # code) -- source and revision are the load-bearing lines.
    assert "source:" in description
    assert "revision:" in description
    assert "chunks:" in description


def test_save_note_create_returns_id_version_and_created_flag():
    notes = FakeSaveNotesBackend()
    service = _save_service(notes_service=notes)

    result = service.invoke(
        "library_save_note", {"title": "Chapter 7", "content": "body text"}
    )

    assert "error" not in result
    parse_public_id(result["item"]["id"], expected_type="note")
    assert result["item"]["type"] == "note"
    assert result["item"]["title"] == "Chapter 7"
    assert "folder" not in result["item"]
    assert result["version"] == 1
    assert result["created"] is True
    assert result["notes"] and all(isinstance(line, str) for line in result["notes"])
    assert notes.calls[0][0] == "add_note"
    assert notes.calls[0][1]["user_id"] == "user-1"


def test_save_note_update_bumps_version_and_reports_not_created():
    notes = FakeSaveNotesBackend(notes={"note-1": {"title": "Old", "content": "old", "version": 1}})
    service = _save_service(notes_service=notes)

    result = service.invoke(
        "library_save_note",
        {
            "title": "New title",
            "content": "new body",
            "note_id": _public_id("note", "note-1"),
            "expected_version": 1,
        },
    )

    assert "error" not in result
    assert result["created"] is False
    assert result["version"] == 2
    assert result["item"]["id"] == _public_id("note", "note-1")
    update_call = next(call for call in notes.calls if call[0] == "update_note")
    assert update_call[1]["update_data"] == {"title": "New title", "content": "new body"}
    assert update_call[1]["expected_version"] == 1


def test_save_note_stale_version_maps_to_content_changed():
    notes = FakeSaveNotesBackend(notes={"note-1": {"title": "T", "content": "c", "version": 7}})
    service = _save_service(notes_service=notes)

    result = service.invoke(
        "library_save_note",
        {
            "title": "T",
            "content": "c2",
            "note_id": _public_id("note", "note-1"),
            "expected_version": 6,
        },
    )

    assert _error_code(result) == "content_changed"
    assert result["error"]["retryable"] is False


def test_save_note_unknown_note_id_maps_to_not_found():
    notes = FakeSaveNotesBackend()
    service = _save_service(notes_service=notes)

    result = service.invoke(
        "library_save_note",
        {
            "title": "T",
            "content": "c",
            "note_id": _public_id("note", "missing-note"),
            "expected_version": 1,
        },
    )

    assert _error_code(result) == "not_found"
    # The update seam was never reached (the existence pre-check refused).
    assert all(call[0] != "update_note" for call in notes.calls)


def test_save_note_failed_update_never_mints_the_folder():
    # Qodo review: for UPDATE calls the not_found pre-check and the version
    # read run BEFORE the folder-ensure, so a deterministically failing
    # update (unknown note id, stale version) cannot mint a folder for a
    # call that then fails. Only the update race residual (the version
    # moving between the read and the row write) may leave a folder behind.
    unknown_scope = FakeNotesScopeService()
    unknown = _save_service(
        notes_service=FakeSaveNotesBackend(), notes_scope_service=unknown_scope
    )
    result = unknown.invoke(
        "library_save_note",
        {
            "title": "T",
            "content": "c",
            "folder": "Study",
            "note_id": _public_id("note", "missing-note"),
            "expected_version": 1,
        },
    )
    assert _error_code(result) == "not_found"
    # No folder minted for a dead note id: the seam was never touched.
    assert unknown_scope.calls == []

    stale_scope = FakeNotesScopeService()
    stale = _save_service(
        notes_service=FakeSaveNotesBackend(
            notes={"note-1": {"title": "T", "content": "c", "version": 7}}
        ),
        notes_scope_service=stale_scope,
    )
    result = stale.invoke(
        "library_save_note",
        {
            "title": "T",
            "content": "c2",
            "folder": "Study",
            "note_id": _public_id("note", "note-1"),
            "expected_version": 6,
        },
    )
    assert _error_code(result) == "content_changed"
    # No folder minted for a stale version: the seam was never touched.
    assert stale_scope.calls == []


def test_save_note_rejects_wrong_type_and_malformed_note_ids():
    service = _save_service()
    wrong_type = service.invoke(
        "library_save_note",
        {
            "title": "t",
            "content": "c",
            "note_id": _public_id("media", "media-uuid-1"),
            "expected_version": 1,
        },
    )
    assert _error_code(wrong_type) == "invalid_argument"

    malformed = service.invoke(
        "library_save_note",
        {"title": "t", "content": "c", "note_id": "not-an-id", "expected_version": 1},
    )
    assert _error_code(malformed) == "invalid_argument"


def test_save_note_over_long_fields_name_the_limit_at_invoke_time():
    # Invoke-time mirrors of the schema maxLength literals (final-review
    # follow-up): a schema-bypassing caller still fails closed, and the
    # refusal names the bound so the agent can self-correct.
    service = _save_service()

    long_title = service.invoke(
        "library_save_note", {"title": "t" * 513, "content": "c"}
    )
    assert _error_code(long_title) == "invalid_argument"
    assert "512" in long_title["error"]["message"]

    long_content = service.invoke(
        "library_save_note", {"title": "t", "content": "c" * 100_001}
    )
    assert _error_code(long_content) == "invalid_argument"
    assert "100000" in long_content["error"]["message"]

    long_folder = service.invoke(
        "library_save_note", {"title": "t", "content": "c", "folder": "f" * 256}
    )
    assert _error_code(long_folder) == "invalid_argument"
    assert "255" in long_folder["error"]["message"]

    # Exactly-at-limit is NOT over-long: the checks are strict '>' and the
    # boundary value still reaches the row-writer.
    at_limit = service.invoke(
        "library_save_note", {"title": "t" * 512, "content": "c" * 100_000}
    )
    assert at_limit.get("error", {}).get("code") != "invalid_argument"


def test_save_note_folder_bound_matches_the_folder_model_segment_limit():
    # Qodo review: the folder model's normalize_folder_name refuses segments
    # over 255 chars, so the schema maxLength and the invoke-time guard must
    # both be 255 -- a 256-char name must never pass the contract only to die
    # at the model, and the model's own 255 boundary must survive end-to-end.
    from tldw_chatbook.Notes.note_folder_models import (
        FolderValidationError,
        normalize_folder_name,
    )

    with pytest.raises(FolderValidationError):
        normalize_folder_name("f" * 256)
    assert normalize_folder_name("f" * 255).display == "f" * 255

    schema = LIBRARY_TOOL_DESCRIPTORS["library_save_note"].input_schema
    assert schema["properties"]["folder"]["maxLength"] == 255

    service = _save_service()
    rejected = service.invoke(
        "library_save_note", {"title": "t", "content": "c", "folder": "f" * 256}
    )
    assert _error_code(rejected) == "invalid_argument"
    assert "255" in rejected["error"]["message"]

    accepted = service.invoke(
        "library_save_note", {"title": "t", "content": "c", "folder": "f" * 255}
    )
    assert "error" not in accepted
    assert accepted["item"]["folder"] == "f" * 255


def test_save_note_folder_ensure_is_idempotent_across_saves():
    scope = FakeNotesScopeService()
    service = _save_service(notes_scope_service=scope)

    first = service.invoke(
        "library_save_note", {"title": "A", "content": "a", "folder": "Study"}
    )
    second = service.invoke(
        "library_save_note", {"title": "B", "content": "b", "folder": "Study"}
    )

    assert "error" not in first
    assert "error" not in second
    assert scope.folder_count() == 1
    creates = [call for call in scope.calls if call[0] == "create_note_folder"]
    assert len(creates) == 1
    attaches = [call for call in scope.calls if call[0] == "attach_note_to_folder"]
    assert len(attaches) == 2
    assert {attaches[0][1]["folder_id"]} == {attaches[1][1]["folder_id"]}
    assert first["item"]["folder"] == "Study"
    assert second["item"]["folder"] == "Study"


def test_save_note_folder_ensure_tolerates_the_create_race():
    scope = FakeNotesScopeService()
    scope.collision_on_next_create = True
    service = _save_service(notes_scope_service=scope)

    result = service.invoke(
        "library_save_note", {"title": "A", "content": "a", "folder": "Study"}
    )

    assert "error" not in result  # the race never raises to the agent
    assert scope.folder_count() == 1
    attach = next(call for call in scope.calls if call[0] == "attach_note_to_folder")
    assert attach[1]["folder_id"] == "folder-raced-1"


def test_save_note_folderless_create_never_touches_the_scope_service():
    scope = FakeNotesScopeService()
    service = _save_service(notes_scope_service=scope)

    result = service.invoke("library_save_note", {"title": "A", "content": "a"})

    assert "error" not in result
    assert scope.calls == []


def test_save_note_folder_without_scope_service_is_feature_unavailable():
    notes = FakeSaveNotesBackend()
    service = _save_service(notes_service=notes, notes_scope_service=None)

    result = service.invoke(
        "library_save_note", {"title": "A", "content": "a", "folder": "Study"}
    )

    assert _error_code(result) == "feature_unavailable"
    # Refused before the row write: no orphan note lands unfiled.
    assert notes.calls == []


def test_save_note_folder_capability_error_maps_to_feature_unavailable():
    # Qodo review: the seam's own capability failure (a deployment whose
    # scope cannot list/create folders raises FolderCapabilityError) is the
    # NAMED feature_unavailable -- not the scrubbed generic storage_error.
    notes = FakeSaveNotesBackend()
    create_blocked = FakeNotesScopeService()
    create_blocked.capability_errors = ("create_note_folder",)
    service = _save_service(notes_service=notes, notes_scope_service=create_blocked)

    result = service.invoke(
        "library_save_note", {"title": "A", "content": "a", "folder": "Study"}
    )

    assert _error_code(result) == "feature_unavailable"
    assert "folder operations are not available" in result["error"]["message"].lower()
    # Refused before the row write: no orphan note lands unfiled.
    assert notes.calls == []

    listing_blocked = FakeNotesScopeService()
    listing_blocked.capability_errors = ("list_note_folder_children",)
    service = _save_service(
        notes_service=FakeSaveNotesBackend(), notes_scope_service=listing_blocked
    )

    result = service.invoke(
        "library_save_note", {"title": "A", "content": "a", "folder": "Study"}
    )

    assert _error_code(result) == "feature_unavailable"
    assert "folder operations are not available" in result["error"]["message"].lower()


def test_save_note_policy_denial_precedes_every_backend_call():
    notes = FakeSaveNotesBackend()
    scope = FakeNotesScopeService()
    service = _save_service(
        notes_service=notes,
        notes_scope_service=scope,
        policy_enforcer=_DenyingPolicyEnforcer(allowed=False),
    )

    result = service.invoke(
        "library_save_note", {"title": "A", "content": "a", "folder": "Study"}
    )

    assert _error_code(result) == "feature_unavailable"
    assert result["error"]["details"]["policy_action"] == "library.notes.save.local"
    # The mutation pin: no note row, no folder, no attach -- nothing ran.
    assert notes.calls == []
    assert scope.calls == []


def test_save_note_policy_enforcement_uses_the_dedicated_action():
    enforcer = _DenyingPolicyEnforcer(allowed=True)
    service = _save_service(policy_enforcer=enforcer)

    result = service.invoke("library_save_note", {"title": "A", "content": "a"})

    assert "error" not in result
    assert enforcer.actions == ["library.notes.save.local"]


def test_save_note_invalid_folder_name_is_invalid_argument():
    service = _save_service()
    # The folder model is a tree of single segments: a slash can never name
    # one folder (the repository's normalize_folder_name refuses it).
    result = service.invoke(
        "library_save_note", {"title": "A", "content": "a", "folder": "Study/Book"}
    )
    assert _error_code(result) == "invalid_argument"


def test_real_save_note_creates_places_and_updates(chacha_db, tmp_path):
    notes = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="test-client",
        global_db_to_use=chacha_db,
    )
    scope = NotesScopeService(
        local_notes_service=notes,
        server_service=None,
        folder_repository=LocalNoteFolderRepository(chacha_db),
    )
    service = _save_service(notes_service=notes, notes_scope_service=scope)
    provenance = (
        "source: media:abc123\nrevision: 4\nchapter: Chapter 7\nchunks: 12-15\n\n"
        "Key points..."
    )

    created = service.invoke(
        "library_save_note",
        {"title": "Chapter 7 notes", "content": provenance, "folder": "Study"},
    )
    assert "error" not in created
    assert created["created"] is True
    assert created["version"] == 1
    assert created["item"]["folder"] == "Study"

    # The row is readable back through the read tool with the header intact.
    read = service.invoke("library_get_note", {"id": created["item"]["id"]})
    assert read["content"]["text"].startswith("source: media:abc123")
    assert "revision: 4" in read["content"]["text"]

    # The folder is visible exactly once where the notes screen lists folders.
    children = asyncio.run(
        scope.list_note_folder_children(
            scope=ScopeType.LOCAL_NOTE,
            parent_id=None,
            limit=50,
            offset=0,
            user_id="user-1",
        )
    )
    assert [folder.name for folder in children.folders] == ["Study"]

    # The update path bumps the version and keeps the placement.
    updated = service.invoke(
        "library_save_note",
        {
            "title": "Chapter 7 notes",
            "content": provenance + "\nMore points",
            "folder": "Study",
            "note_id": created["item"]["id"],
            "expected_version": 1,
        },
    )
    assert "error" not in updated
    assert updated["created"] is False
    assert updated["version"] == 2
    children_after = asyncio.run(
        scope.list_note_folder_children(
            scope=ScopeType.LOCAL_NOTE,
            parent_id=None,
            limit=50,
            offset=0,
            user_id="user-1",
        )
    )
    assert [folder.name for folder in children_after.folders] == ["Study"]

    # Stale version on a real row -> the named conflict error.
    stale = service.invoke(
        "library_save_note",
        {
            "title": "Chapter 7 notes",
            "content": "conflicting",
            "note_id": created["item"]["id"],
            "expected_version": 1,
        },
    )
    assert _error_code(stale) == "content_changed"


# --------------------------------------------------------------------------
# Get operations: prompts (overview + section)
# --------------------------------------------------------------------------


def _prompt_overview():
    return {
        "uuid": "prompt-uuid-1",
        "name": "Prompt One",
        "author": "author",
        "last_modified": "2026-08-01",
        "version": 3,
        "sections": {
            "details": {"total_chars": 500, "preview": "d" * 241},
            "system_prompt": {"total_chars": 40, "preview": "sys"},
        },
    }


def test_get_prompt_overview_bounds_section_previews():
    prompts = FakePromptService(overview=_prompt_overview())
    service = _service(prompt_service=prompts)
    public = _public_id("prompt", "prompt-uuid-1")

    result = service.invoke("library_get_prompt", {"id": public})

    assert result["item"]["name"] == "Prompt One"
    assert result["item"]["type"] == "prompt"
    sections = result["sections"]
    assert sections["details"]["total_chars"] == 500
    assert len(sections["details"]["preview"]) <= 240
    assert sections["system_prompt"] == {"total_chars": 40, "preview": "sys"}
    assert "content" not in result


def test_get_prompt_section_windows_with_cursor():
    section_detail = {
        "uuid": "prompt-uuid-1",
        "name": "Prompt One",
        "section": "system_prompt",
        "version": 3,
        "total_chars": 20,
        "start": 0,
        "returned_chars": 8,
        "has_more": True,
        "text": "abcdefgh",
    }
    prompts = FakePromptService(section=section_detail)
    service = _service(prompt_service=prompts)
    public = _public_id("prompt", "prompt-uuid-1")

    result = service.invoke(
        "library_get_prompt", {"id": public, "section": "system_prompt"}
    )

    assert prompts.calls[0][1]["section"] == "system_prompt"
    content = result["content"]
    assert content["revision"] == "3"
    state = parse_cursor(content["next_cursor"])
    assert state["sec"] == "system_prompt"
    assert state["off"] == 8

    # A cursor minted for one section cannot continue another.
    result = service.invoke(
        "library_get_prompt",
        {"id": public, "section": "details", "cursor": content["next_cursor"]},
    )
    assert _error_code(result) == "invalid_argument"


def test_get_prompt_rejects_unknown_section():
    service = _service(prompt_service=FakePromptService())
    public = _public_id("prompt", "prompt-uuid-1")
    result = service.invoke(
        "library_get_prompt", {"id": public, "section": "version_history"}
    )
    assert _error_code(result) == "invalid_argument"


def test_get_prompt_overview_missing_returns_not_found():
    service = _service(prompt_service=FakePromptService(overview=None))
    result = service.invoke(
        "library_get_prompt", {"id": _public_id("prompt", "gone")}
    )
    assert _error_code(result) == "not_found"


# --------------------------------------------------------------------------
# Get operations: skills (detail + file token)
# --------------------------------------------------------------------------


def _skill_detail(**overrides):
    detail = {
        "name": "skill-one",
        "description": "does things",
        "trust_blocked": False,
        "trust_status": "trusted",
        "body_total_chars": 100,
        "body_preview": "b" * 241,
        "files": [
            {
                "path": "SKILL.md",
                "size": 100,
                "is_text": True,
                "file_token": "tok-main",
            },
            {
                "path": "scripts/helper.py",
                "size": 20,
                "is_text": True,
                "file_token": "tok-helper",
            },
        ],
    }
    detail.update(overrides)
    return detail


def test_get_skill_detail_returns_manifest_and_bounded_preview():
    skills = FakeSkillsService(detail=_skill_detail())
    service = _service(skills_service=skills)
    public = _public_id("skill", "skill-one")

    result = service.invoke("library_get_skill", {"id": public})

    item = result["item"]
    assert item["id"] == public
    assert item["name"] == "skill-one"
    assert item["trust_blocked"] is False
    assert item["body_total_chars"] == 100
    assert len(item["body_preview"]) <= 240
    tokens = [entry["file_token"] for entry in item["files"]]
    assert tokens == ["tok-main", "tok-helper"]
    assert all(not entry["path"].startswith("/") for entry in item["files"])


def test_get_skill_blocked_returns_safe_fields_only():
    blocked = {
        "name": "skill-blocked",
        "description": "safe description",
        "trust_blocked": True,
        "trust_status": "blocked",
    }
    skills = FakeSkillsService(detail=blocked)
    service = _service(skills_service=skills)

    result = service.invoke(
        "library_get_skill", {"id": _public_id("skill", "skill-blocked")}
    )

    item = result["item"]
    assert item["trust_blocked"] is True
    assert "body_preview" not in item
    assert "files" not in item


def test_get_skill_file_windows_with_revision_cursor():
    segment = {
        "path": "SKILL.md",
        "revision": "file-rev-1",
        "total_chars": 20,
        "start": 0,
        "returned_chars": 8,
        "has_more": True,
        "text": "abcdefgh",
    }
    skills = FakeSkillsService(file_segment=segment)
    service = _service(skills_service=skills)
    public = _public_id("skill", "skill-one")

    result = service.invoke(
        "library_get_skill", {"id": public, "file_token": "tok-main", "max_chars": 8}
    )

    assert skills.calls[0][1] == {
        "skill_name": "skill-one",
        "file_token": "tok-main",
        "start": 0,
        "max_chars": 8,
    }
    assert result["file"]["path"] == "SKILL.md"
    content = result["content"]
    assert content["revision"] == "file-rev-1"
    state = parse_cursor(content["next_cursor"])
    assert state["ftok"] == "tok-main"
    assert state["off"] == 8

    # A file cursor cannot be replayed against a different file token.
    result = service.invoke(
        "library_get_skill",
        {"id": public, "file_token": "tok-other", "cursor": content["next_cursor"]},
    )
    assert _error_code(result) == "invalid_argument"


def test_get_skill_unknown_maps_to_not_found():
    service = _service(skills_service=FakeSkillsService(error=ValueError("unknown")))
    result = service.invoke("library_get_skill", {"id": _public_id("skill", "gone")})
    assert _error_code(result) == "not_found"


def test_get_skill_trust_blocked_file_read_maps_to_feature_unavailable():
    from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError

    skills = FakeSkillsService(
        error=SkillTrustBlockedError(
            skill_name="skill-one",
            reason_code="blocked",
            trust_status="untrusted",
        )
    )
    service = _service(skills_service=skills)
    result = service.invoke(
        "library_get_skill",
        {"id": _public_id("skill", "skill-one"), "file_token": "tok-main"},
    )
    assert _error_code(result) == "feature_unavailable"


# --------------------------------------------------------------------------
# Get operations: conversations (message pages + within-message continuation)
# --------------------------------------------------------------------------


def _message(index, text, *, revision=None, total_chars=None, char_start=0):
    total = total_chars if total_chars is not None else len(text)
    return {
        "id": f"msg-{index}",
        "sender": "user",
        "timestamp": f"2026-08-01T10:0{index}:00Z",
        "revision": revision if revision is not None else f"rev-{index}",
        "total_chars": total,
        "char_start": char_start,
        "returned_chars": len(text),
        "has_more": char_start + len(text) < total,
        "text": text,
    }


def _conv_detail(messages, *, total, offset=0, version=1):
    has_more = offset + len(messages) < total
    return {
        "id": "conv-uuid-1",
        "title": "Conv One",
        "version": version,
        "message_total": total,
        "message_offset": offset,
        "returned_message_count": len(messages),
        "has_more": has_more,
        "next_message_offset": offset + len(messages) if has_more else None,
        "include_rag_context": False,
        "messages": messages,
    }


def test_get_conversation_page_envelope_and_cursor():
    detail = _conv_detail([_message(0, "first"), _message(1, "second")], total=3)
    conversations = FakeConversationService(detail=detail)
    service = _service(conversation_service=conversations)
    public = _public_id("conversation", "conv-uuid-1")

    result = service.invoke(
        "library_get_conversation", {"id": public, "message_limit": 7}
    )

    assert conversations.calls[0][1] == {
        "conversation_id": "conv-uuid-1",
        "message_offset": 0,
        "message_limit": 7,
        "max_chars": 8000,
    }
    assert result["item"] == {
        "id": public,
        "type": "conversation",
        "title": "Conv One",
        "title_truncated": False,
    }
    assert result["message_total"] == 3
    assert result["message_offset"] == 0
    assert result["returned_message_count"] == 2
    assert result["has_more"] is True
    assert result["next_message_offset"] == 2
    assert result["include_rag_context"] is False
    state = parse_cursor(result["next_cursor"])
    assert state["moff"] == 2
    assert "mid" not in state

    service.invoke(
        "library_get_conversation",
        {"id": public, "cursor": result["next_cursor"], "message_limit": 7},
    )
    assert conversations.calls[-1][1]["message_offset"] == 2


def test_get_conversation_message_limit_validation():
    service = _service(conversation_service=FakeConversationService())
    public = _public_id("conversation", "conv-uuid-1")
    result = service.invoke(
        "library_get_conversation", {"id": public, "message_limit": 0}
    )
    assert _error_code(result) == "invalid_argument"


def test_get_conversation_within_message_continuation_chain():
    cut_message = _message(1, "abcde", total_chars=10)
    page = _conv_detail([_message(0, "first"), cut_message], total=2)
    conversations = FakeConversationService(detail=page)
    service = _service(conversation_service=conversations)
    public = _public_id("conversation", "conv-uuid-1")

    result = service.invoke("library_get_conversation", {"id": public})

    assert result["returned_message_count"] == 2
    assert result["has_more"] is True
    state = parse_cursor(result["next_cursor"])
    assert state["mid"] == "msg-1"
    assert state["moff"] == 1
    assert state["off"] == 5

    # Continue inside the cut message; it completes, so the next cursor
    # resumes paging after it.
    rest = _message(1, "fghij", char_start=5, total_chars=10)
    conversations._detail = _conv_detail([rest], total=2)
    second = service.invoke(
        "library_get_conversation", {"id": public, "cursor": result["next_cursor"]}
    )
    assert conversations.calls[-1][1] == {
        "conversation_id": "conv-uuid-1",
        "message_id": "msg-1",
        "char_start": 5,
        "max_chars": 8000,
    }
    assert second["messages"][0]["text"] == "fghij"
    assert second["messages"][0]["has_more"] is False
    assert second["has_more"] is False
    assert second["next_cursor"] is None


def test_get_conversation_message_revision_mismatch_is_content_changed():
    page = _conv_detail(
        [_message(0, "abcde", total_chars=10, revision="rev-old")], total=1
    )
    conversations = FakeConversationService(detail=page)
    service = _service(conversation_service=conversations)
    public = _public_id("conversation", "conv-uuid-1")
    first = service.invoke("library_get_conversation", {"id": public})
    state = parse_cursor(first["next_cursor"])
    assert state["mid"] == "msg-0"

    conversations._detail = _conv_detail(
        [
            _message(
                0, "XXXXX", char_start=5, total_chars=10, revision="rev-new"
            )
        ],
        total=1,
    )
    result = service.invoke(
        "library_get_conversation", {"id": public, "cursor": first["next_cursor"]}
    )
    assert _error_code(result) == "content_changed"


def test_get_conversation_missing_returns_not_found():
    service = _service(conversation_service=FakeConversationService(detail=None))
    result = service.invoke(
        "library_get_conversation", {"id": _public_id("conversation", "gone")}
    )
    assert _error_code(result) == "not_found"


# --------------------------------------------------------------------------
# Get operations: collections (membership pages)
# --------------------------------------------------------------------------


def _collection_detail(members, *, total, offset=0):
    has_more = offset + len(members) < total
    return {
        "collection_id": "collection-1",
        "name": "Collection One",
        "description": "d",
        "created_at": "2026-08-01",
        "updated_at": "2026-08-02",
        "member_total": total,
        "offset": offset,
        "limit": 20,
        "has_more": has_more,
        "members": members,
    }


def _member(index, source_type="media"):
    return {
        "membership_id": f"mem-{index}",
        "source_type": source_type,
        "item_id": _public_id("media", f"media-uuid-{index}"),
        "source_ref": None,
        "title": f"Member {index}",
        "title_truncated": False,
    }


def test_get_collection_member_page_and_cursor():
    detail = _collection_detail([_member(1), _member(2)], total=3)
    collections = FakeCollectionsService(detail=detail)
    service = _service(collections_service=collections)
    public = _public_id("collection", "collection-1")

    result = service.invoke("library_get_collection", {"id": public, "limit": 2})

    assert collections.calls[0][1] == {
        "collection_id": "collection-1",
        "limit": 2,
        "offset": 0,
    }
    assert result["item"]["name"] == "Collection One"
    assert result["member_total"] == 3
    assert result["has_more"] is True
    assert result["next_offset"] == 2
    state = parse_cursor(result["next_cursor"])
    assert state["off"] == 2
    member = result["members"][0]
    assert member["membership_id"] == "mem-1"
    parse_public_id(member["item_id"], expected_type="media")

    service.invoke(
        "library_get_collection", {"id": public, "cursor": result["next_cursor"]}
    )
    assert collections.calls[-1][1]["offset"] == 2


def test_get_collection_unsupported_member_has_opaque_ref():
    member = _member(1, source_type="server-doc")
    member["item_id"] = None
    member["source_ref"] = "ref:c2VydmVyLWRvYzptLTE"
    detail = _collection_detail([member], total=1)
    service = _service(collections_service=FakeCollectionsService(detail=detail))

    result = service.invoke(
        "library_get_collection", {"id": _public_id("collection", "collection-1")}
    )

    returned = result["members"][0]
    assert returned["item_id"] is None
    assert returned["source_ref"] == "ref:c2VydmVyLWRvYzptLTE"
    assert result["next_cursor"] is None


def test_get_collection_missing_returns_not_found():
    service = _service(collections_service=FakeCollectionsService(detail=None))
    result = service.invoke(
        "library_get_collection", {"id": _public_id("collection", "gone")}
    )
    assert _error_code(result) == "not_found"


# --------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------


class _RaisingBackend:
    def __init__(self, error):
        self._error = error

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise self._error

        return _raise


def test_sqlite_failure_maps_to_scrubbed_storage_error():
    service = _service(
        notes_service=_RaisingBackend(
            sqlite3.OperationalError("table notes has no column xyz")
        )
    )
    result = service.invoke("library_list_notes", {})
    assert _error_code(result) == "storage_error"
    assert result["error"]["retryable"] is True
    assert "no column" not in result["error"]["message"]


def test_filesystem_failure_maps_to_storage_error():
    service = _service(
        collections_service=_RaisingBackend(OSError("disk went away"))
    )
    result = service.invoke("library_get_collection", {"id": _public_id("collection", "c")})
    assert _error_code(result) == "storage_error"


def test_unexpected_failure_is_scrubbed():
    service = _service(
        media_service=_RaisingBackend(RuntimeError("boom at /secret/path"))
    )
    result = service.invoke("library_list_media", {})
    assert _error_code(result) == "storage_error"
    assert "/secret/path" not in str(result)


@pytest.mark.parametrize(
    "service_attr, tool_name",
    [
        ("media_service", "library_list_media"),
        ("notes_service", "library_list_notes"),
        ("prompt_service", "library_list_prompts"),
        ("skills_service", "library_list_skills"),
        ("conversation_service", "library_list_conversations"),
        ("collections_service", "library_list_collections"),
    ],
    ids=["media", "notes", "prompts", "skills", "conversations", "collections"],
)
def test_every_missing_backend_maps_to_feature_unavailable(service_attr, tool_name):
    backends = _backends()
    backends[service_attr] = None
    result = LocalLibraryToolService(**backends).invoke(tool_name, {})
    assert _error_code(result) == "feature_unavailable"


# --------------------------------------------------------------------------
# No semantic-search routing anywhere in the service
# --------------------------------------------------------------------------

_FORBIDDEN_ROUTING = re.compile(
    r"\b(rag|embedding|embeddings|vector|vectors|semantic)\b", re.IGNORECASE
)


def test_service_source_has_no_semantic_search_routing():
    source = inspect.getsource(service_module)
    assert not _FORBIDDEN_ROUTING.search(source)


def test_backend_call_names_have_no_semantic_search_routing():
    backends = _backends()
    backends["notes_service"] = FakeNotesService(
        items=[_note_item(1)], total=1, detail=_note_detail()
    )
    service = LocalLibraryToolService(**backends)
    service.invoke("library_list_notes", {})
    service.invoke("library_search_notes", {"query": "q"})
    service.invoke("library_get_note", {"id": _public_id("note", "note-uuid-1")})
    for backend in backends.values():
        for method, _kwargs in getattr(backend, "calls", []):
            assert not _FORBIDDEN_ROUTING.search(method)


# --------------------------------------------------------------------------
# Byte ceiling and skip/repeat-free continuation
# --------------------------------------------------------------------------


def test_get_note_fits_multibyte_text_under_the_ceiling():
    full = "🙂" * 16_000  # 4 UTF-8 bytes per character
    notes = FakeNotesService(text_source=full)
    service = _service(notes_service=notes)
    public = _public_id("note", "note-uuid-1")

    result = service.invoke("library_get_note", {"id": public, "max_chars": 16_000})

    assert serialized_size(result) <= MAX_RESULT_BYTES
    content = result["content"]
    assert 0 < content["returned_chars"] < 16_000
    assert content["has_more"] is True
    assert content["text"] == full[: content["returned_chars"]]
    assert content["end"] == content["returned_chars"]
    state = parse_cursor(content["next_cursor"])
    assert state["off"] == content["returned_chars"]

    second = service.invoke(
        "library_get_note", {"id": public, "cursor": content["next_cursor"]}
    )
    assert serialized_size(second) <= MAX_RESULT_BYTES
    assert second["content"]["start"] == content["end"]
    # No skipped or repeated characters across the boundary.
    assert content["text"] + second["content"]["text"] == full


# --------------------------------------------------------------------------
# Cross-backend integration with real temporary databases
# --------------------------------------------------------------------------


@pytest.fixture
def chacha_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield db
    db.close_connection()


class _NotesAdapter:
    """Bridges a real CharactersRAGDB to the notes backend signature."""

    def __init__(self, db):
        self._db = db

    def list_library_notes(self, user_id, *, limit, offset):
        return self._db.list_library_notes_page(limit=limit, offset=offset)

    def search_library_notes(self, user_id, *, query, limit, offset):
        return self._db.search_library_notes_page(
            query=query, limit=limit, offset=offset
        )

    def get_library_note_text(self, user_id, note_id, *, start, max_chars):
        return self._db.get_library_note_text(note_id, start=start, max_chars=max_chars)


def test_real_notes_list_search_get_round_trip(chacha_db):
    note_id = chacha_db.add_note("Integration note", "alpha body text")
    keyword_id = chacha_db.add_keyword("integration-kw")
    chacha_db.link_note_to_keyword(note_id, keyword_id)
    chacha_db.add_note("Other note", "unrelated body")
    service = _service(notes_service=_NotesAdapter(chacha_db))

    listed = service.invoke("library_list_notes", {})
    assert listed["total"] == 2
    assert all(item["type"] == "note" for item in listed["items"])

    found = service.invoke("library_search_notes", {"query": "integration-kw"})
    assert found["total"] == 1
    brief = found["items"][0]
    assert brief["matched_keywords"] == ["integration-kw"]
    assert "keywords" in brief["matched_fields"]

    read = service.invoke("library_get_note", {"id": brief["id"]})
    assert read["content"]["text"] == "alpha body text"
    assert read["content"]["has_more"] is False
    assert read["content"]["next_cursor"] is None


def test_real_note_continuation_detects_content_change(chacha_db):
    note_id = chacha_db.add_note("Long note", "x" * 9_000)
    service = _service(notes_service=_NotesAdapter(chacha_db))
    public = _public_id("note", note_id)

    first = service.invoke("library_get_note", {"id": public})
    assert first["content"]["returned_chars"] == 8_000
    assert first["content"]["has_more"] is True

    assert chacha_db.update_note(
        note_id, {"content": "y" * 100}, expected_version=1
    )
    result = service.invoke(
        "library_get_note", {"id": public, "cursor": first["content"]["next_cursor"]}
    )
    assert _error_code(result) == "content_changed"


def test_real_conversation_list_search_get_round_trip(chacha_db):
    conv_id = chacha_db.add_conversation({"title": "Integration conv"})
    chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "first",
            "timestamp": "2026-08-01T10:00:00.000Z",
        }
    )
    chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": "second",
            "timestamp": "2026-08-01T10:01:00.000Z",
        }
    )
    keyword_id = chacha_db.add_keyword("conv-kw")
    chacha_db.link_conversation_to_keyword(conv_id, keyword_id)
    service = _service(conversation_service=ChatConversationService(chacha_db))

    found = service.invoke("library_search_conversations", {"query": "conv-kw"})
    assert found["total"] == 1
    public = found["items"][0]["id"]

    read = service.invoke("library_get_conversation", {"id": public})
    assert read["message_total"] == 2
    assert read["returned_message_count"] == 2
    assert read["include_rag_context"] is False
    assert [message["text"] for message in read["messages"]] == ["first", "second"]
    assert read["has_more"] is False
    assert read["next_cursor"] is None


def test_real_conversation_long_message_continuation(chacha_db):
    conv_id = chacha_db.add_conversation({"title": "long one"})
    chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "short",
            "timestamp": "2026-08-01T10:00:00.000Z",
        }
    )
    big = "b" * 9_000 + "END"
    big_id = chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "assistant",
            "content": big,
            "timestamp": "2026-08-01T10:01:00.000Z",
        }
    )
    service = _service(conversation_service=ChatConversationService(chacha_db))
    public = _public_id("conversation", conv_id)

    first = service.invoke("library_get_conversation", {"id": public})
    assert first["returned_message_count"] == 2
    state = parse_cursor(first["next_cursor"])
    assert state["mid"] == big_id
    assert state["moff"] == 1
    assert state["off"] == 8_000

    second = service.invoke(
        "library_get_conversation", {"id": public, "cursor": first["next_cursor"]}
    )
    message = second["messages"][0]
    assert message["text"] == "b" * 1_000 + "END"
    assert message["has_more"] is False
    assert second["has_more"] is False
    assert second["next_cursor"] is None


def test_real_collection_round_trip(tmp_path):
    collections = LocalLibraryCollectionsService(
        LibraryCollectionsDB(tmp_path / "library_collections.db")
    )
    collection = collections.create_collection("Integration", description="d")
    collections.add_item_to_collection(
        collection.collection_id, source_type="media", source_id="m-1", title="member one"
    )
    collections.add_item_to_collection(
        collection.collection_id,
        source_type="server-doc",
        source_id="doc-1",
        title="member two",
    )
    service = _service(collections_service=collections)

    found = service.invoke("library_search_collections", {"query": "member one"})
    assert found["total"] == 1
    public = found["items"][0]["id"]

    read = service.invoke("library_get_collection", {"id": public})
    assert read["member_total"] == 2
    # Members are ordered (created_at, membership_id); both were added in the
    # same second, so the membership_id tiebreak makes positions arbitrary.
    members = {member["source_type"]: member for member in read["members"]}
    parse_public_id(members["media"]["item_id"], expected_type="media")
    assert members["server-doc"]["item_id"] is None
    assert members["server-doc"]["source_ref"]


def test_real_gets_reject_wrong_type_and_unknown_ids(chacha_db):
    note_id = chacha_db.add_note("Typed", "body")
    service = _service(
        notes_service=_NotesAdapter(chacha_db),
        conversation_service=ChatConversationService(chacha_db),
    )
    note_public = _public_id("note", note_id)

    wrong_type = service.invoke("library_get_conversation", {"id": note_public})
    assert _error_code(wrong_type) == "invalid_argument"

    unknown = service.invoke(
        "library_get_conversation", {"id": _public_id("conversation", "missing")}
    )
    assert _error_code(unknown) == "not_found"


def _walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_integration_outputs_carry_no_paths_bytes_or_blob_keys(chacha_db, tmp_path):
    note_id = chacha_db.add_note("Scan me", "scan body")
    conv_id = chacha_db.add_conversation({"title": "Scan conv"})
    chacha_db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "scan"}
    )
    collections = LocalLibraryCollectionsService(
        LibraryCollectionsDB(tmp_path / "library_collections.db")
    )
    collection = collections.create_collection("Scan")
    collections.add_item_to_collection(
        collection.collection_id, source_type="note", source_id=note_id, title="t"
    )
    service = _service(
        notes_service=_NotesAdapter(chacha_db),
        conversation_service=ChatConversationService(chacha_db),
        collections_service=collections,
    )
    results = [
        service.invoke("library_list_notes", {}),
        service.invoke("library_get_note", {"id": _public_id("note", note_id)}),
        service.invoke(
            "library_get_conversation", {"id": _public_id("conversation", conv_id)}
        ),
        service.invoke(
            "library_get_collection",
            {"id": _public_id("collection", collection.collection_id)},
        ),
    ]
    for result in results:
        for key, value in _walk(result):
            assert not isinstance(value, (bytes, bytearray))
            lowered = str(key).lower()
            assert "embedding" not in lowered
            assert "image" not in lowered
            if isinstance(value, str):
                assert not value.startswith("/"), (key, value)
