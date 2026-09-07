"""Max-bound / security regression tests for the 18 direct Library tools.

task-1337 (plan Task 10, step 2). Real temporary backends throughout --
``MediaDatabase``, ``CharactersRAGDB`` (behind ``NotesInteropService`` and
``ChatConversationService``), and ``LocalSkillsService`` -- pinning the
contract's hostile-input guarantees where they actually execute:

- display fields are byte-bounded and control-character-free even for
  multibyte/CJK/emoji/hostile stored values, with exact totals and stable
  paging identities;
- keyword lists are visibly bounded while the exact total is still reported;
- an oversized body reads in bounded continuation pages whose concatenation
  is byte-identical to the original, every page under the 32 KiB ceiling;
- conversation pages never carry binary attachments and a full continuation
  walk neither skips nor repeats a message;
- local URLs/paths and embedding internals never surface in any payload;
- a trust-blocked skill exposes safe fields only -- no body, no file
  manifest, no body-only search hits -- and its file reads fail closed;
- huge offsets stay bounded with exact totals;
- no Library tool ever reaches the embedding/RAG pipeline methods.
"""

from __future__ import annotations

import asyncio
import json
import unicodedata

import pytest

from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_tool_contract import (
    DISPLAY_NAME_MAX_BYTES,
    KEYWORD_VALUE_MAX_CHARS,
    KEYWORDS_PER_ITEM_MAX,
    LIBRARY_TOOL_DESCRIPTORS,
    MAX_RESULT_BYTES,
    make_public_id,
    parse_public_id,
    serialized_size,
)
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.fixture
def media_db(tmp_path):
    db = MediaDatabase(db_path=tmp_path / "media.db", client_id="test-client")
    yield db
    db.close_connection()


@pytest.fixture
def chacha_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield db
    db.close_connection()


def _media_service(media_db):
    return LocalLibraryToolService(media_service=LocalMediaReadingService(media_db))


# --------------------------------------------------------------------------
# Hostile stored values: display bounding + exact totals + stable identities
# --------------------------------------------------------------------------

HOSTILE_TITLES = [
    "plain ascii title",
    "标题 with CJK 字符 mixed in",
    "emoji 🎉🔥💥 title",
    "control\x00chars\x07in\x1fthe\u2028title",
    'quotes "double" and \'single\' with \\ backslashes',
    "long-" + "t" * 300,
    "line\nbreaks\r\ninside\tthe\ttitle",
    "<script>alert('xss')</script>",
    "；！全文パンクチュエーション。",
    "format %s %d {0} ${HOME} `whoami`",
]


def _seed_media(db, count=50):
    uuids = []
    for index in range(count):
        title = f"{HOSTILE_TITLES[index % len(HOSTILE_TITLES)]} #{index:03d}"
        _media_id, uuid, _msg = db.add_media_with_keywords(
            title=title,
            content=f"content body {index}",
            media_type="article",
        )
        assert uuid
        uuids.append(uuid)
    return uuids


def test_hostile_media_titles_are_bounded_sanitized_and_stable(media_db):
    uuids = set(_seed_media(media_db, 50))
    service = _media_service(media_db)

    seen = set()
    briefs = []
    offset = 0
    for _ in range(10):
        page = service.invoke("library_list_media", {"limit": 20, "offset": offset})
        assert "error" not in page
        assert page["total"] == 50
        assert serialized_size(page) <= MAX_RESULT_BYTES
        for brief in page["items"]:
            title = brief["title"]
            assert len(title.encode("utf-8")) <= DISPLAY_NAME_MAX_BYTES
            assert all(
                unicodedata.category(ch) not in ("Cc", "Cf") for ch in title
            )
            _raw_type, raw_id = parse_public_id(brief["id"], expected_type="media")
            seen.add(raw_id)
            briefs.append(brief)
        if not page["has_more"]:
            break
        offset = page["next_offset"]
    else:
        raise AssertionError("media paging did not terminate")

    assert seen == uuids  # 50 unique stable identities, none skipped/repeated
    truncated = [brief for brief in briefs if brief["title_truncated"]]
    assert truncated  # the 300-char titles were shortened
    assert all(brief["title"].endswith("…") for brief in truncated)


def test_keywords_are_visibly_bounded_with_exact_total(media_db):
    keywords = [f"k{index:02d}-" + "k" * 124 for index in range(25)]  # 130 chars
    media_db.add_media_with_keywords(
        title="Keyword probe",
        content="keyword body",
        media_type="article",
        keywords=keywords,
    )
    service = _media_service(media_db)

    page = service.invoke("library_list_media", {})
    brief = page["items"][0]

    assert brief["keyword_total"] == 25  # exact, not the visible count
    assert len(brief["keywords"]) <= KEYWORDS_PER_ITEM_MAX
    assert all(len(value) <= KEYWORD_VALUE_MAX_CHARS for value in brief["keywords"])
    assert brief["keywords_truncated"] is True


def test_large_offsets_stay_bounded_with_exact_totals(media_db):
    _seed_media(media_db, 50)
    service = _media_service(media_db)

    page = service.invoke("library_list_media", {"limit": 25, "offset": 25})
    assert page["total"] == 50
    assert len(page["items"]) == 25
    assert page["has_more"] is False

    edge = service.invoke("library_list_media", {"limit": 25, "offset": 50})
    assert edge["items"] == []
    assert edge["total"] == 50
    assert edge["has_more"] is False
    assert edge["next_offset"] is None

    huge = service.invoke("library_list_media", {"limit": 25, "offset": 10**9})
    assert huge["items"] == []
    assert huge["total"] == 50
    assert huge["has_more"] is False
    assert huge["next_offset"] is None
    assert serialized_size(huge) <= MAX_RESULT_BYTES


# --------------------------------------------------------------------------
# Oversized note: full continuation walk under the byte ceiling
# --------------------------------------------------------------------------


def test_oversized_note_full_continuation_walk(tmp_path, chacha_db):
    notes = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="test-client",
        global_db_to_use=chacha_db,
    )
    body = "".join(f"line {index:05d}: " + "x" * 80 + "\n" for index in range(1_300))
    assert len(body) > 100 * 1024  # the fixture really is oversized
    note_id = notes.add_note("local_library", "Oversized note", body)
    service = LocalLibraryToolService(notes_service=notes)
    public = make_public_id("note", note_id)

    parts = []
    pages = 0
    cursor = None
    for _ in range(200):
        arguments = {"id": public}
        if cursor is not None:
            arguments["cursor"] = cursor
        payload = service.invoke("library_get_note", arguments)
        assert "error" not in payload
        assert serialized_size(payload) <= MAX_RESULT_BYTES
        content = payload["content"]
        assert content["total_chars"] == len(body)
        parts.append(content["text"])
        pages += 1
        if not content["has_more"]:
            break
        cursor = content["next_cursor"]
    else:
        raise AssertionError("note continuation walk did not terminate")

    assert "".join(parts) == body  # byte-identical reassembly
    assert pages >= 10  # genuinely paged, not one oversized payload


# --------------------------------------------------------------------------
# Conversations: binary exclusion + exact walk ordering
# --------------------------------------------------------------------------


def test_conversation_pages_exclude_binary_and_never_skip_or_repeat(chacha_db):
    conv_id = chacha_db.add_conversation({"title": "binary walk"})
    expected_ids = []
    for index in range(60):
        message_id = chacha_db.add_message(
            {
                "conversation_id": conv_id,
                "sender": "user" if index % 2 == 0 else "assistant",
                "content": f"message {index:03d} " + "m" * (3_000 + (index % 5) * 200),
                "timestamp": f"2026-08-01T{10 + index // 60:02d}:{index % 60:02d}:00.000Z",
            }
        )
        expected_ids.append(message_id)
    blob_id = chacha_db.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "has an image attached",
            "image_data": b"\x89PNG-binary-blob",
            "image_mime_type": "image/png",
            "timestamp": "2026-08-01T11:01:00.000Z",
        }
    )
    expected_ids.append(blob_id)
    service = LocalLibraryToolService(
        conversation_service=ChatConversationService(chacha_db)
    )
    public = make_public_id("conversation", conv_id)

    seen = []
    pages = 0
    cursor = None
    for _ in range(100):
        arguments = {"id": public}
        if cursor is not None:
            arguments["cursor"] = cursor
        payload = service.invoke("library_get_conversation", arguments)
        assert "error" not in payload
        assert serialized_size(payload) <= MAX_RESULT_BYTES
        assert payload["message_total"] == 61
        assert payload["include_rag_context"] is False
        serialized = json.dumps(payload)
        assert "PNG-binary-blob" not in serialized
        assert "image_data" not in serialized
        for message in payload["messages"]:
            assert set(message) <= {
                "id",
                "sender",
                "timestamp",
                "revision",
                "total_chars",
                "char_start",
                "returned_chars",
                "has_more",
                "text",
            }
            seen.append(message["id"])
        pages += 1
        if not payload["has_more"]:
            break
        cursor = payload["next_cursor"]
    else:
        raise AssertionError("conversation walk did not terminate")

    assert seen == expected_ids  # insertion order, no skips, no repeats
    assert pages > 1  # 61 multi-KB messages cannot fit in one 32 KiB page


# --------------------------------------------------------------------------
# Local URLs / paths / embedding internals never surface
# --------------------------------------------------------------------------


def test_media_url_and_local_paths_never_leak(media_db):
    media_db.add_media_with_keywords(
        url="file:///Users/secret/hidden.pdf",
        title="Leak probe",
        content="leak body",
        media_type="article",
        keywords=["leak-kw"],
    )
    service = _media_service(media_db)

    listed = service.invoke("library_list_media", {})
    public = listed["items"][0]["id"]
    payloads = [
        listed,
        service.invoke("library_search_media", {"query": "leak-kw"}),
        service.invoke("library_get_media", {"id": public}),
    ]
    for payload in payloads:
        assert "error" not in payload
        serialized = json.dumps(payload)
        assert "file:///" not in serialized
        assert "/Users/secret" not in serialized
        assert "hidden.pdf" not in serialized
        assert '"url"' not in serialized
        assert "embedding" not in serialized


# --------------------------------------------------------------------------
# Trust-blocked skills: safe fields only, file reads fail closed
# --------------------------------------------------------------------------

_BODY_MARKER = "BODY-MARKER-7f3d9a"
_FILE_MARKER = "FILE-MARKER-9b1c2d"


def _blocked_skill_service(tmp_path):
    store_dir = tmp_path / "skills"
    compat = LocalSkillsService(
        store_dir=store_dir, allow_untrusted_without_trust_service=True
    )
    asyncio.run(
        compat.create_skill(
            name="marker-skill",
            # Explicit frontmatter description: without it the parser derives
            # the (safe, always-exposed) description from the body, which would
            # legitimately surface a body marker. The marker must live ONLY in
            # the body below the frontmatter for this fixture to mean anything.
            content=(
                "---\n"
                "description: A benign marker-skill description.\n"
                "---\n"
                "# Marker\n"
                "Benign intro line.\n\n"
                f"{_BODY_MARKER}\n"
            ),
            supporting_files={"refs/secret.md": _FILE_MARKER},
        )
    )
    # No trust service: every skill is trust-blocked (fail-closed default).
    return LocalSkillsService(store_dir=store_dir)


def test_blocked_skill_never_leaks_body_or_files(tmp_path):
    blocked = _blocked_skill_service(tmp_path)
    service = LocalLibraryToolService(skills_service=blocked)
    public = make_public_id("skill", "marker-skill")

    manifest = service.invoke("library_get_skill", {"id": public})
    assert "error" not in manifest
    item = manifest["item"]
    assert item["trust_blocked"] is True
    assert "body_preview" not in item
    assert "body_total_chars" not in item
    assert "files" not in item
    serialized = json.dumps(manifest)
    assert _BODY_MARKER not in serialized
    assert _FILE_MARKER not in serialized
    assert "refs/secret.md" not in serialized


def test_blocked_skill_body_only_search_term_finds_nothing(tmp_path):
    blocked = _blocked_skill_service(tmp_path)
    service = LocalLibraryToolService(skills_service=blocked)

    found = service.invoke("library_search_skills", {"query": _BODY_MARKER})
    assert found["total"] == 0

    # Safe fields still surface the blocked skill with its trust status.
    named = service.invoke("library_search_skills", {"query": "marker-skill"})
    assert named["total"] == 1
    assert named["items"][0]["trust_blocked"] is True
    assert _BODY_MARKER not in json.dumps(named)


def test_blocked_skill_file_reads_fail_closed(tmp_path):
    blocked = _blocked_skill_service(tmp_path)
    service = LocalLibraryToolService(skills_service=blocked)
    public = make_public_id("skill", "marker-skill")

    result = service.invoke(
        "library_get_skill", {"id": public, "file_token": "fabricated-token"}
    )

    assert result["error"]["code"] == "feature_unavailable"
    assert _FILE_MARKER not in json.dumps(result)


# --------------------------------------------------------------------------
# No Library tool ever reaches the embedding/RAG pipeline
# --------------------------------------------------------------------------

EMBEDDING_PIPELINE_METHODS = [
    "process_unvectorized_chunks",
    "get_all_active_media_for_embedding",
    "get_media_by_ids_for_embedding",
    "search_media_by_keyword_for_embedding",
]


def test_library_tools_never_reach_the_embedding_pipeline(media_db, monkeypatch):
    _media_id, uuid, _msg = media_db.add_media_with_keywords(
        title="Spy target",
        content="spy content",
        media_type="article",
        keywords=["spy-kw"],
    )
    calls = []

    def _raiser(name):
        def _method(*args, **kwargs):
            calls.append(name)
            raise RuntimeError(f"embedding pipeline method reached: {name}")

        return _method

    for name in EMBEDDING_PIPELINE_METHODS:
        monkeypatch.setattr(media_db, name, _raiser(name))

    # Only the media backend is wired; the other 15 tools fail closed with
    # feature_unavailable, which is fine -- the assertion is about the spies.
    service = _media_service(media_db)
    media_public = make_public_id("media", uuid)
    for tool_name, descriptor in LIBRARY_TOOL_DESCRIPTORS.items():
        if descriptor.operation == "search":
            arguments = {"query": "spy-kw"}
        elif descriptor.operation == "get":
            raw = uuid if descriptor.item_type == "media" else "missing"
            arguments = {"id": make_public_id(descriptor.item_type, raw)}
        else:
            arguments = {}
        result = service.invoke(tool_name, arguments)
        assert isinstance(result, dict)

    # The media list/search/get genuinely ran (proof the spies had a chance).
    assert "error" not in service.invoke("library_get_media", {"id": media_public})
    assert calls == []
