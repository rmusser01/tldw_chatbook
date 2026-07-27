"""TASK-983 module-pass follow-ups: sibling defects of the same shape found
while auditing MCP/server.py's notes tools, in the collaborator modules its
tool handlers delegate to (MCP/tools.py, MCP/resources.py, MCP/prompts.py).

Each of these called a method or config key that does not exist, exactly
the pattern TASK-854/968/983 already fixed elsewhere in this package:

- ``MCPTools.chat_with_character`` read the API key via a direct
  ``get_cli_setting("API", f"{provider}_api_key", "")`` call instead of the
  declared ``config.get_api_key()`` accessor (the same defect TASK-968 fixed
  in ``server.py``'s ``chat_with_llm``, in a different call site).
- ``MCPResources.list_recent_conversations`` / ``.list_recent_notes`` called
  ``get_recent_conversations`` / ``get_recent_notes``, neither of which
  exists on ``CharactersRAGDB``.
- ``MCPResources.get_media_resource`` and ``MCPPrompts.analyze_media_prompt``
  called ``media_db.get_media_transcript(media_id)`` (singular), which never
  existed as an instance method -- the real accessor is the module-level
  ``get_media_transcripts`` (plural).
- ``MCPPrompts.summarize_conversation_prompt`` / ``.generate_document_prompt``
  called ``get_conversation_messages``, which never existed on
  ``CharactersRAGDB`` (the real method is ``get_messages_for_conversation``,
  already fixed at the identical call shape in ``tools.py``/``resources.py``).

A related, wider defect surfaced while fixing the above and writing these
tests: ``MCPResources``/``MCPPrompts`` read several dict keys that do not
match the real row shape at all (not missing methods -- wrong column
names), each silently returning ``KeyError`` or (via ``.get()``) a
permanently empty value:

- ``media['media_type']`` / ``media['created_at']`` -- the ``Media`` table's
  real columns are ``type`` and ``ingestion_date``.
- ``char.get('greeting')`` / ``char.get('example_dialogue')`` /
  ``char.get('updated_at')`` -- ``character_cards``' real columns are
  ``first_message``, ``message_example``, and ``last_modified``.
- ``conv.get('updated_at')`` / ``note.get('updated_at')`` -- both tables
  have ``last_modified``, not ``updated_at``.

All fixed alongside the notes-tool fix since they're the identical
unambiguous shape (a real column exists with the obviously-intended
value); ``tags``/``template`` on notes and ``message_count`` on characters
are a different, non-mechanical case (no such column/aggregation exists at
all) and are called out in TASK-983's Implementation Notes rather than
guessed at here.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


def _real_dbs(tmp_path: Path) -> tuple[CharactersRAGDB, MediaDatabase]:
    chachanotes_db = CharactersRAGDB(
        str(tmp_path / "chachanotes.sqlite"), "test_client"
    )
    media_db = MediaDatabase(str(tmp_path / "media.sqlite"), "test_client")
    return chachanotes_db, media_db


def _seed_transcript(media_db: MediaDatabase, media_id: int, text: str) -> None:
    """Insert a Transcripts row directly -- no public "add transcript"
    method exists on MediaDatabase; ingestion normally writes this table
    through the transcription pipeline, out of scope here."""
    media_db.execute_query(
        """
        INSERT INTO Transcripts
            (media_id, whisper_model, transcription, created_at, uuid,
             last_modified, version, client_id, deleted)
        VALUES (?, 'test-model', ?, datetime('now'), ?, datetime('now'), 1,
                'test_client', 0)
        """,
        (media_id, text, f"transcript-uuid-{media_id}"),
        commit=True,
    )


def test_chat_with_character_uses_the_declared_api_key_accessor():
    """AST check mirroring test_server_character_service.py's identical
    check for server.py's chat_with_llm: this sibling call site in
    MCP/tools.py must go through get_api_key(), not get_cli_setting()."""
    import tldw_chatbook.MCP.tools as tools_module

    source = Path(tools_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    called_names = {
        getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "get_api_key" in called_names
    assert "get_cli_setting" not in called_names

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "get_api_key" in imported_names
    assert "get_cli_setting" not in imported_names


def test_list_recent_conversations_executes_against_real_db(tmp_path):
    from tldw_chatbook.MCP.resources import MCPResources

    chachanotes_db, media_db = _real_dbs(tmp_path)
    resources = MCPResources(chachanotes_db, media_db)

    chachanotes_db.add_conversation({"title": "Recent Conversation"})

    result = asyncio.run(resources.list_recent_conversations(limit=5))

    assert result, "expected the seeded conversation to be listed"
    assert result[0]["name"] == "Recent Conversation"
    assert result[0]["uri"].startswith("conversation://")


def test_list_recent_notes_executes_against_real_db(tmp_path):
    from tldw_chatbook.MCP.resources import MCPResources

    chachanotes_db, media_db = _real_dbs(tmp_path)
    resources = MCPResources(chachanotes_db, media_db)

    chachanotes_db.add_note(title="Recent Note", content="Some content")

    result = asyncio.run(resources.list_recent_notes(limit=5))

    assert result, "expected the seeded note to be listed"
    assert result[0]["name"] == "Recent Note"
    assert result[0]["uri"].startswith("note://")


def test_get_character_resource_uses_the_real_character_card_columns(tmp_path):
    """Regression guard for the wrong-column-name defects found alongside
    the notes-tool fix: `greeting`/`example_dialogue`/`updated_at` never
    matched `character_cards`' real `first_message`/`message_example`/
    `last_modified` columns."""
    from tldw_chatbook.MCP.resources import MCPResources

    chachanotes_db, media_db = _real_dbs(tmp_path)
    resources = MCPResources(chachanotes_db, media_db)

    character_id = chachanotes_db.add_character_card(
        {
            "name": "Test Character",
            "first_message": "Hello, traveler!",
            "message_example": "<START>\n{{user}}: Hi\n{{char}}: Hello!",
        }
    )

    result = asyncio.run(resources.get_character_resource(str(character_id)))

    assert result["name"] != "Error"
    assert "Hello, traveler!" in result["content"]
    assert "<START>" in result["content"]
    assert result["metadata"]["updated"] is not None


def test_get_media_resource_surfaces_the_most_recent_transcript(tmp_path):
    from tldw_chatbook.MCP.resources import MCPResources

    chachanotes_db, media_db = _real_dbs(tmp_path)
    resources = MCPResources(chachanotes_db, media_db)

    media_id, _uuid, _msg = media_db.add_media_with_keywords(
        title="Test Media", media_type="video", content="fallback content"
    )
    _seed_transcript(media_db, media_id, "This is the real transcript text.")

    result = asyncio.run(resources.get_media_resource(str(media_id)))

    assert result["name"] != "Error"
    assert "This is the real transcript text." in result["content"]


def test_analyze_media_prompt_surfaces_the_most_recent_transcript(tmp_path):
    from tldw_chatbook.MCP.prompts import MCPPrompts

    chachanotes_db, media_db = _real_dbs(tmp_path)
    prompts = MCPPrompts(chachanotes_db, media_db)

    media_id, _uuid, _msg = media_db.add_media_with_keywords(
        title="Test Media", media_type="video", content="fallback content"
    )
    _seed_transcript(media_db, media_id, "This is the real transcript text.")

    result = asyncio.run(prompts.analyze_media_prompt(media_id))

    assert len(result) == 1
    assert "This is the real transcript text." in result[0]["content"]


def test_summarize_conversation_prompt_executes_against_real_db(tmp_path):
    from tldw_chatbook.MCP.prompts import MCPPrompts

    chachanotes_db, media_db = _real_dbs(tmp_path)
    prompts = MCPPrompts(chachanotes_db, media_db)

    conversation_id = chachanotes_db.add_conversation({"title": "Test Conversation"})
    chachanotes_db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Hello there",
            "role": "user",
        }
    )

    result = asyncio.run(
        prompts.summarize_conversation_prompt(conversation_id, style="concise")
    )

    assert len(result) == 1
    assert "get_conversation_messages" not in result[0]["content"]
    assert "Hello there" in result[0]["content"]


def test_generate_document_prompt_executes_against_real_db(tmp_path):
    from tldw_chatbook.MCP.prompts import MCPPrompts

    chachanotes_db, media_db = _real_dbs(tmp_path)
    prompts = MCPPrompts(chachanotes_db, media_db)

    conversation_id = chachanotes_db.add_conversation({"title": "Test Conversation"})
    chachanotes_db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Hello there",
            "role": "user",
        }
    )

    result = asyncio.run(
        prompts.generate_document_prompt(conversation_id, doc_type="summary")
    )

    assert len(result) == 1
    assert "get_conversation_messages" not in result[0]["content"]
    assert "Hello there" in result[0]["content"]
