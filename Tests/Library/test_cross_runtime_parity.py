"""Cross-runtime parity: Console provider vs the local MCP runtime delegate.

task-1337 (plan Task 10, step 1). The Console ``LibraryToolProvider`` and the
local MCP ``LocalMCPRuntimeDelegate`` both sit on the same synchronous
``LocalLibraryToolService``. This file pins the spec's parity rule: for the
same tool call, the JSON-decoded Console payload is identical to the
direct-MCP payload; both surfaces derive their schemas from
``LIBRARY_TOOL_DESCRIPTORS``; and all six contract error codes surface
identically on both. Per-runtime behavior is pinned separately in
``Tests/Agents/test_library_tool_provider.py`` and
``Tests/MCP/test_library_tools.py``; the service contract itself in
``Tests/Library/test_local_library_tool_service.py``.

All tests are synchronous: the Console provider must run off the event loop
(the service bridges async backends with ``asyncio.run``), so MCP calls are
driven with one ``asyncio.run`` per call -- the delegate dispatches the
service through ``asyncio.to_thread``, keeping the service off the loop too.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_tool_contract import (
    ERROR_CODES,
    ERROR_INDEX_UNAVAILABLE,
    LIBRARY_TOOL_DESCRIPTORS,
    LibraryToolError,
    make_public_id,
)
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
from tldw_chatbook.MCP.local_runtime_delegate import LocalMCPRuntimeDelegate
from tldw_chatbook.MCP.server import describe_local_mcp_capabilities

MEDIA_UUID = "media-uuid-1"
NOTE_ID = "note-uuid-1"
PROMPT_UUID = "prompt-uuid-1"
SKILL_NAME = "stub-skill"
CONVERSATION_ID = "conv-uuid-1"
COLLECTION_ID = "collection-1"

NOTE_BODY = "n" * 9_000  # past the 8_000-char default window -> two pages


def _window(text, start, max_chars):
    return text[start : start + max_chars]


class StubLibraryBackend:
    """In-memory backend implementing all six Library seam surfaces.

    Method signatures mirror the real services exactly (leading ``user_id``
    for notes, positional ``query`` for prompt search, async prompt/skill
    methods). All reads are deterministic and side-effect-free so the same
    instance can serve the Console call and the MCP call in sequence.
    ``errors`` maps a method name to an exception raised on entry.
    """

    def __init__(self) -> None:
        self.errors: dict[str, BaseException] = {}
        self.note_version = 1
        self.note_body = NOTE_BODY
        self.media_body = "media body text"
        self.prompt_system = "You are a stub assistant."
        self.skill_body = "# Stub Skill\nBody text."
        self.skill_ref = "Reference guide text."
        self.media_rows = [
            {
                "id": 1,
                "uuid": MEDIA_UUID,
                "title": "Media 1",
                "media_type": "article",
                "author": "author",
                "ingestion_date": "2026-01-01",
                "last_modified": "2026-08-01",
                "version": 1,
                "preview": "media preview",
                "keywords": ["media-kw"],
                "keyword_total": 1,
                "keywords_truncated": False,
                "matched_fields": ["keywords"],
                "matched_keywords": ["media-kw"],
            }
        ]
        self.note_rows = [
            {
                "id": NOTE_ID,
                "title": "Note 1",
                "created_at": "2026-08-01T00:00:00Z",
                "last_modified": "2026-08-02T00:00:00Z",
                "version": 1,
                "preview": "note preview",
                "keywords": ["note-kw"],
                "keyword_total": 1,
                "keywords_truncated": False,
                "matched_fields": ["keywords"],
                "matched_keywords": ["note-kw"],
            }
        ]
        self.prompt_rows = [
            {
                "id": 1,
                "uuid": PROMPT_UUID,
                "name": "Prompt 1",
                "author": "author",
                "last_modified": "2026-08-01",
                "version": 1,
                "details_preview": "prompt details",
                "has_system_prompt": True,
                "has_user_prompt": False,
                "has_prompt_definition": False,
                "keywords": ["prompt-kw"],
                "keyword_total": 1,
                "keywords_truncated": False,
                "matched_fields": ["keywords"],
                "matched_keywords": ["prompt-kw"],
            }
        ]
        self.skill_rows = [
            {
                "name": SKILL_NAME,
                "description": "stub skill",
                "trust_blocked": False,
                "trust_status": "trusted",
                "matched_fields": ["name"],
                "matched_keywords": [],
            }
        ]
        self.conversation_rows = [
            {
                "id": CONVERSATION_ID,
                "title": "Conversation 1",
                "created_at": "2026-08-01T00:00:00Z",
                "last_modified": "2026-08-02T00:00:00Z",
                "version": 1,
                "keywords": ["conv-kw"],
                "keyword_total": 1,
                "keywords_truncated": False,
                "matched_fields": ["keywords"],
                "matched_keywords": ["conv-kw"],
            }
        ]
        self.collection_rows = [
            {
                "collection_id": COLLECTION_ID,
                "name": "Collection 1",
                "description": "stub collection",
                "item_count": 2,
                "created_at": "2026-08-01T00:00:00Z",
                "updated_at": "2026-08-02T00:00:00Z",
                "matched_fields": ["name"],
                "matched_keywords": [],
            }
        ]
        self.messages = [
            {"id": "msg-1", "sender": "user", "timestamp": "2026-08-01T10:00:00.000Z", "version": 1, "text": "first"},
            {"id": "msg-2", "sender": "assistant", "timestamp": "2026-08-01T10:01:00.000Z", "version": 1, "text": "second"},
            {"id": "msg-3", "sender": "user", "timestamp": "2026-08-01T10:02:00.000Z", "version": 1, "text": "third"},
        ]
        self.collection_members = [
            {
                "source_type": "media",
                "source_ref": MEDIA_UUID,
                "item_id": make_public_id("media", MEDIA_UUID),
                "title": "member one",
            },
            {
                "source_type": "server-doc",
                "source_ref": "doc-1",
                "item_id": None,
                "title": "member two",
            },
        ]

    def _maybe_raise(self, method: str) -> None:
        error = self.errors.get(method)
        if error is not None:
            raise error

    @staticmethod
    def _page(rows, limit, offset):
        return {"items": list(rows[offset : offset + limit]), "total": len(rows)}

    # -- media -----------------------------------------------------------------

    def list_library_media(self, *, limit, offset):
        self._maybe_raise("list_library_media")
        return self._page(self.media_rows, limit, offset)

    def search_library_media(self, *, query, limit, offset):
        self._maybe_raise("search_library_media")
        return self._page(self.media_rows, limit, offset)

    def get_library_media_text(self, media_uuid, *, start, max_chars):
        self._maybe_raise("get_library_media_text")
        if media_uuid != MEDIA_UUID:
            return None
        return {
            "title": "Media 1",
            "media_type": "article",
            "author": "author",
            "ingestion_date": "2026-01-01",
            "last_modified": "2026-08-01",
            "version": 1,
            "text": _window(self.media_body, start, max_chars),
            "total_chars": len(self.media_body),
        }

    # -- notes -----------------------------------------------------------------

    def list_library_notes(self, user_id, *, limit, offset):
        self._maybe_raise("list_library_notes")
        return self._page(self.note_rows, limit, offset)

    def search_library_notes(self, user_id, *, query, limit, offset):
        self._maybe_raise("search_library_notes")
        return self._page(self.note_rows, limit, offset)

    def get_library_note_text(self, user_id, note_id, *, start, max_chars):
        self._maybe_raise("get_library_note_text")
        if note_id != NOTE_ID:
            return None
        return {
            "title": "Note 1",
            "created_at": "2026-08-01T00:00:00Z",
            "last_modified": "2026-08-02T00:00:00Z",
            "version": self.note_version,
            "text": _window(self.note_body, start, max_chars),
            "total_chars": len(self.note_body),
        }

    # -- prompts ---------------------------------------------------------------

    async def list_library_prompts(self, *, limit, offset):
        self._maybe_raise("list_library_prompts")
        return self._page(self.prompt_rows, limit, offset)

    async def search_library_prompts(self, query, *, limit, offset):
        self._maybe_raise("search_library_prompts")
        return self._page(self.prompt_rows, limit, offset)

    async def get_library_prompt_overview(self, prompt_uuid):
        self._maybe_raise("get_library_prompt_overview")
        if prompt_uuid != PROMPT_UUID:
            return None
        return {
            "name": "Prompt 1",
            "author": "author",
            "last_modified": "2026-08-01",
            "sections": {
                "details": {"total_chars": 14, "preview": "prompt details"},
                "system_prompt": {
                    "total_chars": len(self.prompt_system),
                    "preview": self.prompt_system,
                },
            },
        }

    async def get_library_prompt_section(self, prompt_uuid, section, *, start, max_chars):
        self._maybe_raise("get_library_prompt_section")
        if prompt_uuid != PROMPT_UUID or section != "system_prompt":
            return None
        return {
            "name": "Prompt 1",
            "version": 1,
            "text": _window(self.prompt_system, start, max_chars),
            "total_chars": len(self.prompt_system),
        }

    # -- skills ----------------------------------------------------------------

    async def list_library_skills(self, *, limit, offset):
        self._maybe_raise("list_library_skills")
        return self._page(self.skill_rows, limit, offset)

    async def search_library_skills(self, *, query, limit, offset):
        self._maybe_raise("search_library_skills")
        return self._page(self.skill_rows, limit, offset)

    async def get_library_skill(self, skill_name):
        self._maybe_raise("get_library_skill")
        if skill_name != SKILL_NAME:
            raise ValueError(f"unknown skill: {skill_name}")
        return {
            "name": SKILL_NAME,
            "description": "stub skill",
            "trust_status": "trusted",
            "trust_blocked": False,
            "validation_status": "valid",
            "body_total_chars": len(self.skill_body),
            "body_preview": self.skill_body,
            "files": [
                {
                    "path": "SKILL.md",
                    "size": len(self.skill_body),
                    "is_text": True,
                    "file_token": "tok-skill-md",
                },
                {
                    "path": "refs/guide.md",
                    "size": len(self.skill_ref),
                    "is_text": True,
                    "file_token": "tok-guide",
                },
            ],
        }

    async def get_library_skill_file(self, skill_name, file_token, *, start, max_chars):
        self._maybe_raise("get_library_skill_file")
        bodies = {
            "tok-skill-md": ("SKILL.md", self.skill_body),
            "tok-guide": ("refs/guide.md", self.skill_ref),
        }
        if skill_name != SKILL_NAME or file_token not in bodies:
            raise ValueError("file token is not valid for this skill")
        path, body = bodies[file_token]
        return {
            "revision": "1",
            "path": path,
            "text": _window(body, start, max_chars),
            "total_chars": len(body),
        }

    # -- conversations -----------------------------------------------------------

    def list_library_conversations(self, *, limit, offset):
        self._maybe_raise("list_library_conversations")
        return self._page(self.conversation_rows, limit, offset)

    def search_library_conversations(self, *, query, limit, offset):
        self._maybe_raise("search_library_conversations")
        return self._page(self.conversation_rows, limit, offset)

    @staticmethod
    def _window_message(message, char_start, max_chars):
        text = _window(message["text"], char_start, max_chars)
        total = len(message["text"])
        return {
            "id": message["id"],
            "sender": message["sender"],
            "timestamp": message["timestamp"],
            "revision": str(message["version"]),
            "total_chars": total,
            "char_start": char_start,
            "returned_chars": len(text),
            "has_more": char_start + len(text) < total,
            "text": text,
        }

    def get_library_conversation_messages(self, conversation_id, **kwargs):
        self._maybe_raise("get_library_conversation_messages")
        if conversation_id != CONVERSATION_ID:
            return None
        message_id = kwargs.get("message_id")
        char_start = kwargs.get("char_start", 0)
        max_chars = kwargs.get("max_chars", 8_000)
        message_total = len(self.messages)
        if message_id is not None:
            message = next(
                (item for item in self.messages if item["id"] == message_id), None
            )
            messages = (
                []
                if message is None
                else [self._window_message(message, char_start, max_chars)]
            )
            message_offset = 0
        else:
            message_offset = kwargs.get("message_offset", 0)
            message_limit = kwargs.get("message_limit", 20)
            page = self.messages[message_offset : message_offset + message_limit]
            messages = [
                self._window_message(item, char_start, max_chars) for item in page
            ]
        has_more_pages = (
            message_id is None and message_offset + len(messages) < message_total
        )
        return {
            "id": conversation_id,
            "title": "Conversation 1",
            "version": 1,
            "message_total": message_total,
            "message_offset": message_offset,
            "returned_message_count": len(messages),
            "has_more": has_more_pages,
            "next_message_offset": (
                message_offset + len(messages) if has_more_pages else None
            ),
            "include_rag_context": False,
            "messages": messages,
        }

    # -- collections -------------------------------------------------------------

    def list_library_collections(self, *, limit, offset):
        self._maybe_raise("list_library_collections")
        return self._page(self.collection_rows, limit, offset)

    def search_library_collections(self, *, query, limit, offset):
        self._maybe_raise("search_library_collections")
        return self._page(self.collection_rows, limit, offset)

    def get_library_collection(self, collection_id, *, limit, offset):
        self._maybe_raise("get_library_collection")
        if collection_id != COLLECTION_ID:
            return None
        members = self.collection_members[offset : offset + limit]
        return {
            "name": "Collection 1",
            "description": "stub collection",
            "created_at": "2026-08-01T00:00:00Z",
            "updated_at": "2026-08-02T00:00:00Z",
            "member_total": len(self.collection_members),
            "members": members,
        }


def _service(stub, **overrides):
    backends = {
        "media_service": stub,
        "notes_service": stub,
        "prompt_service": stub,
        "skills_service": stub,
        "conversation_service": stub,
        "collections_service": stub,
    }
    backends.update(overrides)
    return LocalLibraryToolService(**backends)


def _console_payload(provider, tool_name, arguments):
    result = provider.invoke(f"library:{tool_name}", arguments)
    if result.ok:
        assert result.error == ""
        return json.loads(result.content), True
    assert result.content == ""
    return json.loads(result.error), False


def _mcp_payload(delegate, tool_name, arguments):
    payload = asyncio.run(delegate.execute_tool(tool_name, arguments))
    return payload, "error" not in payload


def _assert_parity(service, tool_name, arguments):
    """Run one call through both surfaces; assert identical decoded payloads."""
    provider = LibraryToolProvider(service)
    delegate = LocalMCPRuntimeDelegate(library_service=service)
    console_payload, console_ok = _console_payload(provider, tool_name, arguments)
    mcp_payload, mcp_ok = _mcp_payload(delegate, tool_name, arguments)
    assert console_ok == mcp_ok
    assert console_payload == mcp_payload
    return console_payload


# --------------------------------------------------------------------------
# Catalog and schema parity (all 18 tools)
# --------------------------------------------------------------------------


def test_console_catalog_matches_mcp_manifest_and_descriptors():
    provider = LibraryToolProvider(None)
    catalog = provider.list_catalog()
    manifest_names = [
        entry["name"]
        for entry in describe_local_mcp_capabilities()["tools"]
        if entry["name"] in LIBRARY_TOOL_DESCRIPTORS
    ]

    assert [entry.name for entry in catalog] == list(LIBRARY_TOOL_DESCRIPTORS)
    assert [entry.id for entry in catalog] == [
        f"library:{name}" for name in LIBRARY_TOOL_DESCRIPTORS
    ]
    assert manifest_names == list(LIBRARY_TOOL_DESCRIPTORS)


@pytest.mark.parametrize("tool_name", list(LIBRARY_TOOL_DESCRIPTORS))
def test_schema_parity_console_manifest_descriptor(tool_name):
    descriptor = LIBRARY_TOOL_DESCRIPTORS[tool_name]
    provider = LibraryToolProvider(None)
    schema = provider.load_schema(f"library:{tool_name}")
    manifest_entry = next(
        entry
        for entry in describe_local_mcp_capabilities()["tools"]
        if entry["name"] == tool_name
    )

    assert schema.parameters == descriptor.input_schema
    assert manifest_entry["inputSchema"] == descriptor.input_schema
    assert schema.description == manifest_entry["description"] == descriptor.description


# --------------------------------------------------------------------------
# Success parity: list / search / get across all six Library types
# --------------------------------------------------------------------------

LIST_TOOLS = [
    ("library_list_media", "media"),
    ("library_list_notes", "note"),
    ("library_list_prompts", "prompt"),
    ("library_list_skills", "skill"),
    ("library_list_conversations", "conversation"),
    ("library_list_collections", "collection"),
]

SEARCH_TOOLS = [
    ("library_search_media", "media"),
    ("library_search_notes", "note"),
    ("library_search_prompts", "prompt"),
    ("library_search_skills", "skill"),
    ("library_search_conversations", "conversation"),
    ("library_search_collections", "collection"),
]


@pytest.mark.parametrize(
    "tool_name,item_type", LIST_TOOLS, ids=[case[1] for case in LIST_TOOLS]
)
def test_list_parity(tool_name, item_type):
    payload = _assert_parity(_service(StubLibraryBackend()), tool_name, {"limit": 10})

    assert payload["total"] == 1
    assert payload["items"][0]["type"] == item_type


@pytest.mark.parametrize(
    "tool_name,item_type", SEARCH_TOOLS, ids=[case[1] for case in SEARCH_TOOLS]
)
def test_search_parity(tool_name, item_type):
    payload = _assert_parity(
        _service(StubLibraryBackend()), tool_name, {"query": "kw", "limit": 10}
    )

    assert payload["total"] == 1
    assert payload["items"][0]["type"] == item_type


GET_CASES = [
    ("library_get_media", {"id": make_public_id("media", MEDIA_UUID)}, "media"),
    ("library_get_note", {"id": make_public_id("note", NOTE_ID)}, "note"),
    (
        "library_get_prompt",
        {"id": make_public_id("prompt", PROMPT_UUID)},
        "prompt-overview",
    ),
    (
        "library_get_prompt",
        {"id": make_public_id("prompt", PROMPT_UUID), "section": "system_prompt"},
        "prompt-section",
    ),
    ("library_get_skill", {"id": make_public_id("skill", SKILL_NAME)}, "skill-manifest"),
    (
        "library_get_skill",
        {"id": make_public_id("skill", SKILL_NAME), "file_token": "tok-guide"},
        "skill-file",
    ),
    (
        "library_get_conversation",
        {"id": make_public_id("conversation", CONVERSATION_ID)},
        "conversation",
    ),
    (
        "library_get_collection",
        {"id": make_public_id("collection", COLLECTION_ID)},
        "collection",
    ),
]


@pytest.mark.parametrize(
    "tool_name,arguments,case",
    GET_CASES,
    ids=["manifest" if case[2] == "skill-manifest" else case[2] for case in GET_CASES],
)
def test_get_parity(tool_name, arguments, case):
    payload = _assert_parity(_service(StubLibraryBackend()), tool_name, arguments)

    assert "error" not in payload
    assert payload["item"]["type"] == case.split("-")[0]


def test_continuation_walk_parity_for_long_note():
    stub = StubLibraryBackend()
    service = _service(stub)
    public = make_public_id("note", NOTE_ID)

    cursor = None
    chunks = []
    for _ in range(5):  # generous bound; the walk must terminate in 2 pages
        arguments = {"id": public}
        if cursor is not None:
            arguments["cursor"] = cursor
        payload = _assert_parity(service, "library_get_note", arguments)
        content = payload["content"]
        chunks.append(content["text"])
        if not content["has_more"]:
            break
        cursor = content["next_cursor"]
    else:
        raise AssertionError("continuation walk did not terminate")

    assert "".join(chunks) == stub.note_body
    assert len(chunks) == 2  # 9_000 chars over the 8_000-char default window


# --------------------------------------------------------------------------
# Error parity: all six contract codes surface identically on both runtimes
# --------------------------------------------------------------------------

ERROR_CASES = [
    # Unknown argument key -> runtime validation fails closed.
    ("library_list_media", {"limit": 1, "bogus": True}, None, "invalid_argument"),
    # Well-formed public ID naming an item the backend does not have.
    (
        "library_get_media",
        {"id": make_public_id("media", "missing-uuid")},
        None,
        "not_found",
    ),
    # Backend-raised index failure passes through unchanged.
    (
        "library_search_media",
        {"query": "kw"},
        "index_error",
        "index_unavailable",
    ),
    # A missing backend maps its tools to feature_unavailable.
    ("library_list_media", {}, "no_media_backend", "feature_unavailable"),
    # Operational failures are scrubbed to the storage_error payload.
    ("library_list_media", {}, "storage_error", "storage_error"),
]


@pytest.mark.parametrize(
    "tool_name,arguments,setup,expected_code",
    ERROR_CASES,
    ids=[case[3] for case in ERROR_CASES],
)
def test_error_parity(tool_name, arguments, setup, expected_code):
    stub = StubLibraryBackend()
    overrides = {}
    if setup == "index_error":
        stub.errors["search_library_media"] = LibraryToolError(
            ERROR_INDEX_UNAVAILABLE,
            "The local search index is not available.",
            retryable=False,
        )
    elif setup == "storage_error":
        stub.errors["list_library_media"] = sqlite3.OperationalError(
            "no such table: secrets"
        )
    elif setup == "no_media_backend":
        overrides["media_service"] = None
    service = _service(stub, **overrides)

    payload = _assert_parity(service, tool_name, arguments)

    assert payload["error"]["code"] == expected_code
    serialized = json.dumps(payload)
    assert "secrets" not in serialized
    assert "sqlite" not in serialized


def test_content_changed_error_parity():
    stub = StubLibraryBackend()
    service = _service(stub)
    public = make_public_id("note", NOTE_ID)

    first = _assert_parity(service, "library_get_note", {"id": public})
    assert first["content"]["has_more"] is True

    stub.note_version = 2  # the item changes underneath the minted cursor
    payload = _assert_parity(
        service,
        "library_get_note",
        {"id": public, "cursor": first["content"]["next_cursor"]},
    )

    assert payload["error"]["code"] == "content_changed"


def test_error_parity_covers_every_contract_code():
    exercised = {case[3] for case in ERROR_CASES} | {"content_changed"}
    assert exercised == set(ERROR_CODES)


# --------------------------------------------------------------------------
# Real-backend parity: notes over a real ChaChaNotes database
# --------------------------------------------------------------------------


@pytest.fixture
def chacha_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield db
    db.close_connection()


class _NotesAdapter:
    """Bridge a real CharactersRAGDB to the notes backend signature."""

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


def test_real_notes_backend_round_trip_parity(chacha_db):
    note_id = chacha_db.add_note("Parity note", "parity body " * 40)
    keyword_id = chacha_db.add_keyword("parity-kw")
    chacha_db.link_note_to_keyword(note_id, keyword_id)
    chacha_db.add_note("Other note", "unrelated body")
    service = _service(
        StubLibraryBackend(), notes_service=_NotesAdapter(chacha_db)
    )

    listed = _assert_parity(service, "library_list_notes", {})
    assert listed["total"] == 2

    found = _assert_parity(service, "library_search_notes", {"query": "parity-kw"})
    assert found["total"] == 1
    public = found["items"][0]["id"]

    read = _assert_parity(service, "library_get_note", {"id": public})
    assert read["content"]["text"] == "parity body " * 40
    assert read["content"]["has_more"] is False
    assert read["content"]["next_cursor"] is None
