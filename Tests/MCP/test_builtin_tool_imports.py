# Tests/MCP/test_builtin_tool_imports.py
"""Regression coverage for Defect 2 (QA round mcp-hub-phase3-2026-07): most
built-in MCP tools crashed with `ImportError: cannot import name
'ChaChaNotes_DB' from 'tldw_chatbook.DB.ChaChaNotes_DB'` the moment they were
actually executed (Hub -> Tools mode -> Test Tool -> Run for
`chat_with_character`, `search_conversations`, `list_characters`,
`get_conversation_history`, `export_conversation`). `create_note` and
`search_notes` are routed by `local_runtime_delegate.py` straight to
`CharactersRAGDB.add_note()` / `.search_notes()` and never go through this
module at all, so they were unaffected by the ImportError.

Root cause (verified by reading each site): `tldw_chatbook/MCP/tools.py`,
`resources.py`, `prompts.py`, and `server.py` all imported a class named
`ChaChaNotes_DB` that never existed -- the real class in
`tldw_chatbook/DB/ChaChaNotes_DB.py` is `CharactersRAGDB`. Fixing that alone
was not sufficient: once the first `ImportError` cleared, three more of the
same class of bug surfaced underneath it (`tools.py` also imported
`save_conversation_from_messages` from `Chat_Functions.py`, `chat_with_
provider` from `LLM_API_Calls.py`, and `get_character_by_id`/
`list_characters` from `Character_Chat_Lib.py` -- none of which exist
anywhere in the current tree; `resources.py` similarly imported a
non-existent free-function `get_note_by_id`). All of those are fixed here
too (renamed/redirected to the real `CharactersRAGDB` methods where a clean
equivalent exists, or a documented `NotImplementedError` stub where none
does -- see the fix commit for the full list), so these modules genuinely
import and `list_characters` genuinely executes now, not just the literal
`ChaChaNotes_DB` symbol QA's live capture happened to surface first.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


def test_mcp_tools_resources_prompts_server_modules_import_without_importerror():
    """Direct regression net for the literal defect: at HEAD (pre-fix) each
    of these raised `ImportError: cannot import name 'ChaChaNotes_DB' from
    'tldw_chatbook.DB.ChaChaNotes_DB'` (or, once that was fixed in
    isolation, a cascade of further ImportErrors for other dead names --
    see module docstring). All four must import cleanly."""
    import tldw_chatbook.MCP.prompts as prompts_module
    import tldw_chatbook.MCP.resources as resources_module
    import tldw_chatbook.MCP.server as server_module
    import tldw_chatbook.MCP.tools as tools_module

    assert tools_module.MCPTools is not None
    assert resources_module.MCPResources is not None
    assert prompts_module.MCPPrompts is not None
    assert server_module.MCP_AVAILABLE in (True, False)


def _real_dbs(tmp_path: Path) -> tuple[CharactersRAGDB, MediaDatabase]:
    chachanotes_db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "test_client")
    media_db = MediaDatabase(str(tmp_path / "media.sqlite"), "test_client")
    return chachanotes_db, media_db


def test_mcp_tools_constructs_against_real_dbs(tmp_path):
    """`MCPTools.__init__` alone (no method call) is enough to exercise the
    broken module-level imports it depends on -- pins that construction no
    longer raises."""
    from tldw_chatbook.MCP.tools import MCPTools

    chachanotes_db, media_db = _real_dbs(tmp_path)
    tools = MCPTools(chachanotes_db, media_db)
    assert tools.chachanotes_db is chachanotes_db


def test_list_available_characters_executes_to_a_real_result_against_real_db(tmp_path):
    """The exact tool QA verified live (`list_characters`, no required args)
    must now execute to a real, correctly-shaped result instead of
    `Failed * 5ms` with the ImportError. A freshly created `CharactersRAGDB`
    seeds one default character card, so this also proves the fixed
    `list_character_cards()` call site returns real row data (`id`/`name`
    keys), not just that the import resolved."""
    from tldw_chatbook.MCP.tools import MCPTools

    chachanotes_db, media_db = _real_dbs(tmp_path)
    tools = MCPTools(chachanotes_db, media_db)

    result = asyncio.run(tools.list_available_characters())

    assert isinstance(result, list)
    assert result, "expected at least the default seeded character card"
    for entry in result:
        assert "error" not in entry, f"tool-level error, not the crash under test: {entry}"
        assert "id" in entry and "name" in entry


@pytest.mark.asyncio
async def test_local_runtime_delegate_list_characters_executes_without_importerror(tmp_path, monkeypatch):
    """End-to-end regression test through the ACTUAL path the Hub's Test
    Tool runner uses (`LocalMCPControlService.execute_tool()` ->
    `LocalMCPRuntimeDelegate.execute_tool()` -> `_tool_list_characters()` ->
    `MCPTools.list_available_characters()`), mirroring QA's own live
    reproduction exactly. Monkeypatches the lazy DB getters
    `local_runtime_delegate.py` imports by name (`get_chachanotes_db_lazy`/
    `get_media_db_lazy`) to real tmp-backed instances instead of the
    process-global singletons those getters normally cache."""
    import tldw_chatbook.MCP.local_runtime_delegate as delegate_module

    chachanotes_db, media_db = _real_dbs(tmp_path)
    monkeypatch.setattr(delegate_module, "get_chachanotes_db_lazy", lambda: chachanotes_db)
    monkeypatch.setattr(delegate_module, "get_media_db_lazy", lambda: media_db)

    delegate = delegate_module.LocalMCPRuntimeDelegate()
    result = await delegate.execute_tool("list_characters", {})

    assert isinstance(result, list)
    assert result and all("error" not in entry for entry in result)


def test_get_conversation_history_executes_against_real_db(tmp_path):
    """Residual dead-call regression (QA follow-up review of commit
    4fd1e908): `get_conversation_history` called
    `chachanotes_db.get_conversation_messages()`, a method that never
    existed on `CharactersRAGDB` (the real method is
    `get_messages_for_conversation`). Seeds a real conversation + message
    and asserts a genuine result comes back, not a tool-level `{"error":
    ...}` naming the dead method."""
    from tldw_chatbook.MCP.tools import MCPTools

    chachanotes_db, media_db = _real_dbs(tmp_path)
    tools = MCPTools(chachanotes_db, media_db)

    conversation_id = chachanotes_db.add_conversation({"title": "Test Conversation"})
    chachanotes_db.add_message({
        "conversation_id": conversation_id,
        "sender": "user",
        "content": "Hello there",
        "role": "user",
    })

    result = asyncio.run(tools.get_conversation_history(conversation_id))

    assert "get_conversation_messages" not in str(result)
    assert "error" not in result, f"tool-level error, not the crash under test: {result}"
    assert result["title"] == "Test Conversation"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["content"] == "Hello there"
    assert result["messages"][0]["role"] == "user"
