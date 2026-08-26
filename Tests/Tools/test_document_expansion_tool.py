"""TASK-16174 Phase T: the gated `expand_document` tool's contract.

Since TASK-16071's rank-fair merge, 54% of the rows a top-M consumer is fed
are LABEL-ONLY (`Matched media · pdf`, `Matched conversation · 3 messages`;
`Library/library_local_rag_search_service.py`'s `_media_row`/`_conversation_row`,
both with `chunk_id: ""`). An agent could retrieve them and not see behind
them. These tests pin the tool that does, per source type, against REAL
databases seeded through the production writer APIs -- the same four writers
the RAG eval harness uses (`Tests/RAG_Eval/harness/ingest.py`), so row shapes
and id assignment are production's and not a fixture's guess at them.

The two label-only branches (media, conversation) are first-class cases here,
not afterthoughts: AC#4 is "the contract works from exactly what those rows
carry" -- `source_type` + `source_id`, empty `chunk_id`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

CLIENT_ID = "expand-document-tests"
NOTE_BODY = "Tides are driven by the moon's gravity, not by wind."
MEDIA_BODY = "The Fresnel lens focuses a beam without a solid glass block."
PROMPT_DETAILS = "Use for structured extraction."
PROMPT_SYSTEM = "You are a careful extraction assistant."
HEAD_MARKER = "HEADSTART-ONLY-TOKEN"
DEEP_MARKER = "DEEPBODY-ONLY-TOKEN"
TAIL_MARKER = "TAILMESSAGE-ONLY-TOKEN"


def _long_body() -> str:
    """A body whose head and deep region are separately identifiable."""
    return HEAD_MARKER + ("x" * 3000) + DEEP_MARKER + ("y" * 3000)


@pytest.fixture
def dbs(tmp_path, monkeypatch):
    """Real ChaChaNotes/Media/Prompts databases, wired in as the tool's handles.

    The tool resolves its handles through `config.get_*_db_lazy()` with
    function-local imports (the `note_management_tools` precedent), so
    patching the config module's attributes is the seam that reaches it.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    import tldw_chatbook.config as config_module

    chacha = CharactersRAGDB(tmp_path / "chacha.db", client_id=CLIENT_ID)
    media = MediaDatabase(tmp_path / "media.db", client_id=CLIENT_ID)
    prompts = PromptsDatabase(tmp_path / "prompts.db", client_id=CLIENT_ID)

    monkeypatch.setattr(config_module, "get_chachanotes_db_lazy", lambda: chacha)
    monkeypatch.setattr(config_module, "get_media_db_lazy", lambda: media)
    monkeypatch.setattr(config_module, "get_prompts_db_lazy", lambda: prompts)

    yield SimpleNamespace(chacha=chacha, media=media, prompts=prompts)

    for db in (prompts, media, chacha):
        try:
            db.close_connection()
        except Exception:  # noqa: BLE001 -- teardown must not mask a failure
            pass


def _tool():
    from tldw_chatbook.Tools.document_expansion_tool import ExpandDocumentTool

    return ExpandDocumentTool()


def _seed_note(dbs, body: str = NOTE_BODY, title: str = "Tide note") -> str:
    note_id = dbs.chacha.add_note(title=title, content=body)
    assert note_id, "precondition: the note writer returned an id"
    return str(note_id)


def _seed_media(dbs, body: str = MEDIA_BODY, title: str = "Lighthouse optics") -> str:
    media_id, _uuid, message = dbs.media.add_media_with_keywords(
        title=title, media_type="document", content=body
    )
    assert media_id is not None, f"precondition: media write failed ({message})"
    return str(media_id)


def _seed_conversation(dbs, title: str = "Optics chat") -> str:
    conversation_id = dbs.chacha.add_conversation({"title": title})
    assert conversation_id, "precondition: the conversation writer returned an id"
    for sender, content in (("user", "Why does the beam carry?"), ("assistant", MEDIA_BODY)):
        dbs.chacha.add_message(
            {
                "conversation_id": conversation_id,
                "sender": sender,
                "content": content,
                "timestamp": "2026-01-01T00:00:00Z",
            }
        )
    return str(conversation_id)


def _seed_prompt(dbs, name: str = "Extraction prompt") -> str:
    prompt_id, _uuid, message = dbs.prompts.add_prompt(
        name=name,
        author=None,
        details=PROMPT_DETAILS,
        system_prompt=PROMPT_SYSTEM,
    )
    assert prompt_id is not None, f"precondition: prompt write failed ({message})"
    return str(prompt_id)


_SHAPE_KEYS = {
    "status",
    "source_type",
    "source_id",
    "title",
    "text",
    "total_size",
    "window",
    "truncated",
    "next_offset",
}


def _assert_shape(result: dict) -> None:
    """Every branch returns the SAME keys -- Task 3 consumes this shape."""
    assert _SHAPE_KEYS <= set(result), f"missing keys: {_SHAPE_KEYS - set(result)}"
    assert set(result["window"]) == {"start", "end"}


# --- per-source-type contract ----------------------------------------------


async def test_note_expands_to_full_body(dbs):
    note_id = _seed_note(dbs)

    result = await _tool().execute(source_type="note", source_id=note_id)

    _assert_shape(result)
    assert result["status"] == "ok"
    assert result["source_type"] == "note"
    assert result["source_id"] == note_id
    assert result["title"] == "Tide note"
    assert result["text"] == NOTE_BODY
    assert result["total_size"] == len(NOTE_BODY)
    assert result["window"] == {"start": 0, "end": len(NOTE_BODY)}
    assert result["truncated"] is False
    assert result["next_offset"] is None


async def test_media_label_only_row_expands(dbs):
    """AC#4: a media row ships `source_type` + `source_id` and `chunk_id: ""`.

    That is the ENTIRE input the tool gets for 54% of merged rows, so the
    call is made with exactly those fields and nothing else.
    """
    media_id = _seed_media(dbs)
    stored = dbs.media.get_media_by_id(int(media_id))

    result = await _tool().execute(
        source_type="media", source_id=media_id, chunk_id=""
    )

    _assert_shape(result)
    assert result["status"] == "ok"
    assert result["source_type"] == "media"
    assert result["source_id"] == media_id
    assert result["title"] == "Lighthouse optics"
    assert result["text"] == stored["content"]
    assert MEDIA_BODY in result["text"]
    assert result["truncated"] is False


async def test_conversation_returns_role_prefixed_transcript(dbs):
    """Mirrors `ingestion_indexing.conversation_document`'s `f"{sender}: {content}"`
    rendering, so what the agent reads matches what was indexed."""
    conversation_id = _seed_conversation(dbs)

    result = await _tool().execute(
        source_type="conversation", source_id=conversation_id, chunk_id=""
    )

    _assert_shape(result)
    assert result["status"] == "ok"
    assert result["source_type"] == "conversation"
    assert result["title"] == "Optics chat"
    assert result["text"] == (
        f"user: Why does the beam carry?\nassistant: {MEDIA_BODY}"
    )
    assert result["truncated"] is False


async def test_prompt_expands(dbs):
    """The body is the non-empty `PROMPT_DOCUMENT_COLUMNS` joined -- the same
    rendering the prompts sub-leg indexes (`rag_service._prompt_document_text`)."""
    prompt_id = _seed_prompt(dbs)

    result = await _tool().execute(source_type="prompt", source_id=prompt_id)

    _assert_shape(result)
    assert result["status"] == "ok"
    assert result["source_type"] == "prompt"
    assert result["title"] == "Extraction prompt"
    assert result["text"] == f"{PROMPT_DETAILS}\n\n{PROMPT_SYSTEM}"
    assert result["truncated"] is False


# --- budget, windowing, continuation ---------------------------------------


async def test_over_budget_returns_window_and_next_offset(dbs):
    body = _long_body()
    note_id = _seed_note(dbs, body=body)

    result = await _tool().execute(
        source_type="note", source_id=note_id, max_chars=1000
    )

    _assert_shape(result)
    assert result["status"] == "ok"
    assert result["total_size"] == len(body)
    assert result["window"] == {"start": 0, "end": 1000}
    assert result["text"] == body[:1000]
    assert len(result["text"]) == 1000
    assert result["truncated"] is True
    assert result["next_offset"] == 1000


async def test_chunk_start_centres_the_window(dbs):
    """A semantic row carries chunk lineage; the window follows `chunk_start`.

    The row's `provenance` carries `chunk_start` (written by the indexer at
    `rag_service.py`'s chunk-metadata build). The agent pastes it verbatim and
    gets the matched region, NOT the document head. `chunk_id` is deliberately
    absent from this call: it is an INDEX, it anchors nothing, and the fix
    wave retired it from the schema for exactly that reason.
    """
    body = _long_body()
    note_id = _seed_note(dbs, body=body)
    anchor = body.index(DEEP_MARKER)

    result = await _tool().execute(
        source_type="note",
        source_id=note_id,
        chunk_start=anchor,
        max_chars=1000,
    )

    _assert_shape(result)
    assert result["status"] == "ok"
    assert DEEP_MARKER in result["text"]
    assert HEAD_MARKER not in result["text"]
    assert result["window"]["start"] > 0
    assert result["window"]["start"] <= anchor < result["window"]["end"]
    assert result["truncated"] is True


async def test_offset_continuation_walks_the_document(dbs):
    """Navigation within one document without re-querying."""
    body = _long_body()
    note_id = _seed_note(dbs, body=body)
    tool = _tool()

    first = await tool.execute(source_type="note", source_id=note_id, max_chars=1000)
    second = await tool.execute(
        source_type="note",
        source_id=note_id,
        offset=first["next_offset"],
        max_chars=1000,
    )

    _assert_shape(second)
    assert second["status"] == "ok"
    assert second["window"] == {"start": 1000, "end": 2000}
    assert second["text"] == body[1000:2000]
    assert second["text"] != first["text"]
    assert second["next_offset"] == 2000


async def test_chunk_id_is_accepted_but_is_not_agent_facing_surface(dbs):
    """`chunk_id` is retired from the schema: nothing in `execute` reads it.

    An agent pasting a row's whole provenance still works -- `chunk_id` rides
    the `**_provenance` swallow -- and the window it gets is the document
    HEAD, which is the proof the index never anchored anything. Shipping a
    knob wired to nothing is the exact surface this arc's Phase K retired.
    """
    tool = _tool()
    assert "chunk_id" not in tool.parameters["properties"]
    assert "chunk_id" not in tool.parameters["required"]

    body = _long_body()
    note_id = _seed_note(dbs, body=body)

    result = await tool.execute(
        source_type="note",
        source_id=note_id,
        chunk_id=f"note_{note_id}_chunk_9",
        chunk_index=9,
        media_type="document",
        max_chars=1000,
    )

    _assert_shape(result)
    assert result["status"] == "ok"
    assert result["window"] == {"start": 0, "end": 1000}
    assert HEAD_MARKER in result["text"]
    assert DEEP_MARKER not in result["text"]


def test_tool_description_names_chunk_start_as_the_window_anchor():
    """The retirement must reach the prose too: the description tells the
    agent to pass `chunk_start` -- the field the code actually consumes --
    and never mentions the parameter that was removed."""
    description = " ".join(_tool().description.split())

    assert (
        "pass the row's chunk_start to centre that window on the matched "
        "chunk instead of the document head"
    ) in description
    assert "chunk_id" not in description


# --- the budget promise -----------------------------------------------------


async def test_absurd_budget_is_capped_at_the_hard_max(dbs):
    """"the tool never returns more than the budget regardless of what is
    asked" (spec) -- `HARD_MAX_CHARS` is that promise, and it was untested."""
    from tldw_chatbook.Tools.document_expansion_tool import HARD_MAX_CHARS

    body = "z" * (HARD_MAX_CHARS + 5000)
    note_id = _seed_note(dbs, body=body)

    result = await _tool().execute(
        source_type="note", source_id=note_id, max_chars=10**9
    )

    _assert_shape(result)
    assert result["status"] == "ok"
    assert len(result["text"]) == HARD_MAX_CHARS
    assert result["window"] == {"start": 0, "end": HARD_MAX_CHARS}
    assert result["total_size"] == len(body)
    assert result["truncated"] is True
    assert result["next_offset"] == HARD_MAX_CHARS


@pytest.mark.parametrize("budget", [0, -5, "x", None])
async def test_useless_budget_falls_back_to_the_default(dbs, budget):
    """A non-positive or unparseable budget is floored to the default rather
    than returning zero characters (or raising) -- the other half of
    `_resolve_budget`, likewise untested until the fix wave."""
    from tldw_chatbook.Tools.document_expansion_tool import DEFAULT_MAX_CHARS

    body = "z" * (DEFAULT_MAX_CHARS + 2000)
    note_id = _seed_note(dbs, body=body)

    result = await _tool().execute(
        source_type="note", source_id=note_id, max_chars=budget
    )

    _assert_shape(result)
    assert result["status"] == "ok"
    assert len(result["text"]) == DEFAULT_MAX_CHARS
    assert result["window"] == {"start": 0, "end": DEFAULT_MAX_CHARS}
    assert result["next_offset"] == DEFAULT_MAX_CHARS


# --- the conversation message cap is a PARTIAL read, and says so ------------


async def test_conversation_over_the_message_cap_reports_itself_truncated(dbs):
    """A >500-message conversation must not be reported as a complete read.

    `MAX_TRANSCRIPT_MESSAGES` bounds the DB work, so `total_size` describes
    only the prefix that was read. Returning `truncated: False` and
    `next_offset: None` there asserts a completeness the payload does not
    have -- and does it invisibly, since it looks exactly like a successful
    whole-document read.
    """
    from tldw_chatbook.Tools.document_expansion_tool import MAX_TRANSCRIPT_MESSAGES

    conversation_id = dbs.chacha.add_conversation({"title": "Very long chat"})
    assert conversation_id, "precondition: the conversation writer returned an id"
    for index in range(MAX_TRANSCRIPT_MESSAGES + 1):
        content = (
            f"{TAIL_MARKER} last"
            if index == MAX_TRANSCRIPT_MESSAGES
            else f"message {index}"
        )
        dbs.chacha.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": content,
                "timestamp": (
                    f"2026-01-01T{index // 3600:02d}:"
                    f"{(index // 60) % 60:02d}:{index % 60:02d}Z"
                ),
            }
        )

    result = await _tool().execute(
        source_type="conversation",
        source_id=str(conversation_id),
        max_chars=32000,
    )

    _assert_shape(result)
    assert result["status"] == "ok"
    assert TAIL_MARKER not in result["text"], "the 501st message was never read"
    assert result["window"] == {"start": 0, "end": result["total_size"]}
    assert result["truncated"] is True, "a partial read must say it is partial"
    assert result["next_offset"] is None, "character offsets cannot reach message 501"
    note = result.get("note", "")
    assert str(MAX_TRANSCRIPT_MESSAGES) in note and "message" in note.lower(), (
        "the payload says WHICH boundary it hit, not merely that it hit one"
    )


async def test_a_short_conversation_carries_no_truncation_note(dbs):
    """The note is emitted on exactly the partial reads -- a two-message
    transcript is complete and says nothing."""
    conversation_id = _seed_conversation(dbs)

    result = await _tool().execute(
        source_type="conversation", source_id=conversation_id
    )

    assert result["truncated"] is False
    assert "note" not in result


# --- misses and identity ----------------------------------------------------


async def test_unknown_id_is_not_found(dbs):
    result = await _tool().execute(
        source_type="note", source_id="00000000-0000-0000-0000-000000000000"
    )

    _assert_shape(result)
    assert result["status"] == "not_found"
    assert result["text"] == ""
    assert result["total_size"] == 0
    assert result["truncated"] is False
    assert result["next_offset"] is None
    assert "error" not in result


async def test_semantic_identity_fallbacks(dbs):
    """A semantic row's `source_id` can be a CHROMA POINT ID.

    `_semantic_row` builds `source_id` from metadata `source_id` ||
    `document_id` || the point id, so the real document identity may only be
    in the provenance extras. `doc_id` there is the indexer's prefixed
    document id (`f"note_{id}"`, `ingestion_indexing.note_document`), which
    must resolve as well as a bare `note_id` does.
    """
    note_id = _seed_note(dbs)
    tool = _tool()

    via_note_id = await tool.execute(
        source_type="note",
        source_id="a1b2c3d4-chroma-point-id",
        note_id=note_id,
    )
    assert via_note_id["status"] == "ok"
    assert via_note_id["text"] == NOTE_BODY
    assert via_note_id["source_id"] == note_id, "the RESOLVED identity is reported"

    via_doc_id = await tool.execute(
        source_type="note",
        source_id="a1b2c3d4-chroma-point-id",
        doc_id=f"note_{note_id}",
    )
    assert via_doc_id["status"] == "ok"
    assert via_doc_id["text"] == NOTE_BODY
    assert via_doc_id["source_id"] == note_id

    media_id = _seed_media(dbs)
    via_media_id = await tool.execute(
        source_type="media",
        source_id="a1b2c3d4-chroma-point-id",
        media_id=media_id,
    )
    assert via_media_id["status"] == "ok"
    assert MEDIA_BODY in via_media_id["text"]


# --- catalog: gated off by default, risk-floored to ask ---------------------


@pytest.fixture
def tools_config(monkeypatch):
    """`[tools]` gate reads, deterministic and independent of config.toml.

    Same seam `Tests/Agents/test_gateable_builtin_tools.py` patches: both
    `BuiltinToolProvider.__init__` and `all_tool_gates()` reach
    `tldw_chatbook.config.get_cli_setting` by function-local import.
    """
    values: dict = {}
    import tldw_chatbook.config as config_module

    def fake(section, key=None, default=None):
        if section != "tools" or not isinstance(key, str):
            return default
        return values.get(key, default)

    monkeypatch.setattr(config_module, "get_cli_setting", fake)
    return values


def test_tool_is_gated_off_by_default(tools_config):
    """Registered in the ONE table, absent until its gate is switched on.

    The Settings-side switch is DERIVED from `_GATEABLE_BUILTINS` -- the MCP
    hub's gate affordance enumerates it through `all_tool_gates()` -- so
    asserting the row exists there with `enabled=False` is asserting the
    switch exists and starts off.
    """
    from tldw_chatbook.Agents.builtin_tool_gate import all_tool_gates
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        gateable_builtin_tools,
    )

    entry = next(
        e for e in gateable_builtin_tools() if e.tool_name == "expand_document"
    )
    assert entry.gate_key == "expand_document_enabled"
    assert entry.module_name == "document_expansion_tool"
    assert entry.factory_name == "ExpandDocumentTool"

    off_names = {e.name for e in BuiltinToolProvider().list_catalog()}
    assert "expand_document" not in off_names

    gate = next(g for g in all_tool_gates() if g.tool_name == "expand_document")
    assert (gate.section, gate.key, gate.group) == (
        "tools",
        "expand_document_enabled",
        "builtin",
    )
    assert gate.enabled is False
    assert gate.description, "the switch renders the tool's own description"

    tools_config["expand_document_enabled"] = True
    on_names = {e.name for e in BuiltinToolProvider().list_catalog()}
    assert "expand_document" in on_names


def test_tool_carries_risk_tags():
    """It reads the user's notes, media, conversations and prompts, so an
    inherited `allow` must be FLOORED to `ask` (a per-call approval card)."""
    from tldw_chatbook.Agents.builtin_tool_gate import tool_ref
    from tldw_chatbook.MCP.permission_store import (
        BUILTIN_HIGH_RISK_TAGS,
        resolve_builtin_state,
    )

    tool = _tool()
    assert tool.risk_tags == ("reads",)
    assert set(tool.risk_tags) <= BUILTIN_HIGH_RISK_TAGS

    state = resolve_builtin_state({}, tool_ref(tool))
    assert state.state == "ask"
    assert state.risk_floored is True


# --- the conversation fetch does not read image BLOBs it never renders ------


async def test_conversation_fetch_asks_for_no_image_blobs(dbs, monkeypatch):
    """TASK-16688 AC#4 (TASK-16174 finding 15): `include_image_data=False`.

    The transcript renders `sender`/`content` only, so the reader's default
    (`include_image_data=True`, `ChaChaNotes_DB.get_messages_for_conversation`)
    pulls up to `MAX_TRANSCRIPT_MESSAGES` image BLOBs into memory for text
    that cannot use them -- the task-260 case, one seam later.

    The pin wraps the REAL reader (it still runs; only the kwargs are
    recorded), so it fails both ways: drop the flag and the recorded kwargs
    lose it, and the returned rows carry the BLOB again.
    """
    conversation_id = dbs.chacha.add_conversation({"title": "Optics chat"})
    assert conversation_id, "precondition: the conversation writer returned an id"
    dbs.chacha.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Why does the beam carry?",
            "image_data": b"\x89PNG\r\n\x1a\n" + b"NOT-A-REAL-PNG" * 64,
            "image_mime_type": "image/png",
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )
    dbs.chacha.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": MEDIA_BODY,
            "timestamp": "2026-01-01T00:00:01Z",
        }
    )

    real_reader = dbs.chacha.get_messages_for_conversation
    calls: list[dict] = []

    def _recording_reader(*args, **kwargs):
        calls.append({"args": args, "kwargs": dict(kwargs)})
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(
        dbs.chacha, "get_messages_for_conversation", _recording_reader
    )

    result = await _tool().execute(
        source_type="conversation", source_id=str(conversation_id)
    )

    assert len(calls) == 1, "the transcript is ONE read, not a per-message loop"
    assert calls[0]["kwargs"].get("include_image_data") is False, (
        "the transcript fetch must opt out of the image BLOB column"
    )
    rows = real_reader(str(conversation_id), limit=10, include_image_data=False)
    assert rows[0]["image_data"] is None, (
        "control: the flag is what suppresses the BLOB, and the message "
        "really does carry one"
    )
    assert (
        real_reader(str(conversation_id), limit=10)[0]["image_data"] is not None
    ), "control: the default still returns it, so the pin is a real reading"

    _assert_shape(result)
    assert result["status"] == "ok"
    assert result["text"] == (
        f"user: Why does the beam carry?\nassistant: {MEDIA_BODY}"
    ), "the rendered text is byte-identical with the BLOBs skipped"
