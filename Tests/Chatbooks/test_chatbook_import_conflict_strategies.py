# test_chatbook_import_conflict_strategies.py
# Description: Regression coverage for TASK-38594 (REPLACE duplicated notes
#              and failed on characters; MERGE silently became RENAME)
"""
The import wizard offers one conflict strategy for every content type, but
the importer only implemented SKIP and RENAME (TASK-38594 / bug-sweep
2026-11-02 1.7):

* REPLACE fell through to the plain create path -- a duplicate-titled
  note (``add_note`` has no title uniqueness), a ``ConflictError`` for
  characters (unique name constraint), a duplicate-titled conversation.
* MERGE was silently rewritten to RENAME by the resolver, and
  ``ConflictResolver.merge_notes`` was dead code no caller ever reached.

Pinned behaviour:

* REPLACE updates in place: notes and characters are overwritten through
  the versioned update APIs (single row, imported content, version
  bumped); a conflicting conversation is retired by soft delete and
  recreated from the chatbook, so exactly one ACTIVE conversation with
  the title remains, holding the imported messages.
* SKIP and RENAME keep their existing behaviour (second import skips, or
  lands under a "Title (1)" name).
* MERGE is rejected loudly by ``import_chatbook`` BEFORE any item is
  written -- it is gone from the wizard UI (see the wizard tests below)
  and may no longer be accepted silently by the importer.

``pytestmark = bootstrap_profile``: the fixtures construct real
``CharactersRAGDB`` files through the config-participant admission chain,
which fails closed with ``RecoveryRequired`` under the per-test config
redirect.
"""

import shutil
from datetime import datetime
from pathlib import Path

import pytest
from textual.widgets import RadioSet, RadioButton, Static

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Chatbooks.conflict_resolver import (
    ConflictResolution,
    ConflictResolver,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import ConflictResolutionStep

pytestmark = pytest.mark.bootstrap_profile

NOTE_TITLE = "Packing list"
NOTE_CONTENT = "Boots, map, water"
CHARACTER_NAME = "Field Guide"
CHARACTER_DESCRIPTION = "A helpful guide"
CONVERSATION_TITLE = "Trip planning"
CONVERSATION_MESSAGE = "Where to?"


def _make_db_paths(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "ChaChaNotes": str(root / "ChaChaNotes.db"),
        "Media": str(root / "Client_Media_DB.db"),
        "Prompts": str(root / "Prompts_DB.db"),
    }


@pytest.fixture
def chatbook_zip(tmp_path, chachanotes_template_db) -> Path:
    """A chatbook holding one conversation, one note and one character."""
    source_paths = _make_db_paths(tmp_path / "source")
    shutil.copyfile(chachanotes_template_db, source_paths["ChaChaNotes"])
    chacha = CharactersRAGDB(source_paths["ChaChaNotes"], "test_client")

    char_id = chacha.add_character_card(
        {
            "name": CHARACTER_NAME,
            "description": CHARACTER_DESCRIPTION,
            "personality": "Precise",
            "scenario": "",
            "first_message": "Hello!",
            "tags": ["guide"],
            "version": 1,
        }
    )
    conv_id = chacha.add_conversation(
        {
            "title": CONVERSATION_TITLE,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "root_id": "conv_root",
            "character_id": char_id,
        }
    )
    chacha.add_message(
        {
            "conversation_id": conv_id,
            "sender": "user",
            "content": CONVERSATION_MESSAGE,
        }
    )
    note_id = chacha.add_note(title=NOTE_TITLE, content=NOTE_CONTENT)

    export_path = tmp_path / "bundle.zip"
    success, message, _deps = ChatbookCreator(db_paths=source_paths).create_chatbook(
        name="Strategy Bundle",
        description="Conflict strategy fixture",
        content_selections={
            ContentType.CONVERSATION: [str(conv_id)],
            ContentType.NOTE: [str(note_id)],
            ContentType.CHARACTER: [str(char_id)],
        },
        output_path=export_path,
        author="Tests",
        include_media=False,
    )
    assert success is True, message
    return export_path


@pytest.fixture
def destination(tmp_path, chachanotes_template_db) -> Path:
    paths = _make_db_paths(tmp_path / "destination")
    shutil.copyfile(chachanotes_template_db, paths["ChaChaNotes"])
    return Path(paths["ChaChaNotes"])


def _import(chatbook_zip: Path, db_path: Path, strategy: ConflictResolution):
    status = ImportStatus()
    success, message = ChatbookImporter(
        {"ChaChaNotes": str(db_path)}
    ).import_chatbook(
        chatbook_path=chatbook_zip,
        conflict_resolution=strategy,
        prefix_imported=False,
        import_media=False,
        import_status=status,
    )
    return success, message, status


def _open(db_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(str(db_path), "strategy_assert")


def _active_note_rows(db: CharactersRAGDB, title: str) -> list:
    cursor = db.get_connection().execute(
        "SELECT id, content, version FROM notes WHERE title = ? AND deleted = 0",
        (title,),
    )
    return [dict(row) for row in cursor.fetchall()]


def _active_conversation_rows(db: CharactersRAGDB, title: str) -> list:
    cursor = db.get_connection().execute(
        "SELECT id, deleted FROM conversations WHERE title = ?", (title,)
    )
    return [dict(row) for row in cursor.fetchall()]


def _conversation_message_contents(db: CharactersRAGDB, conversation_id: str) -> list:
    cursor = db.get_connection().execute(
        "SELECT content FROM messages WHERE conversation_id = ? ORDER BY rowid",
        (conversation_id,),
    )
    return [row["content"] for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# REPLACE: update in place, never a duplicate
# ---------------------------------------------------------------------------


def test_replace_note_updates_in_place(tmp_path, chatbook_zip, destination):
    """AC#1 for notes: a conflicting note's content is overwritten through
    the versioned update API; no duplicate-titled row is created."""
    assert _import(chatbook_zip, destination, ConflictResolution.REPLACE)[0]
    db = _open(destination)
    first = _active_note_rows(db, NOTE_TITLE)
    assert len(first) == 1
    # Age the note locally so the second import must REALLY replace it.
    db.update_note(first[0]["id"], {"content": "stale local content"}, first[0]["version"])
    db.close_connection()

    success, _message, status = _import(
        chatbook_zip, destination, ConflictResolution.REPLACE
    )

    assert success is True
    assert status.skipped_items == 0
    db = _open(destination)
    rows = _active_note_rows(db, NOTE_TITLE)
    assert len(rows) == 1, "REPLACE must not create a duplicate-titled note"
    assert rows[0]["content"] == NOTE_CONTENT
    assert rows[0]["id"] == first[0]["id"], "the existing note is updated, not recreated"
    assert rows[0]["version"] > first[0]["version"], "versioned update, not raw SQL"
    db.close_connection()


def test_replace_character_updates_in_place(tmp_path, chatbook_zip, destination):
    """AC#1 for characters: a conflicting card is replaced via
    ``update_character_card`` instead of failing with ConflictError."""
    assert _import(chatbook_zip, destination, ConflictResolution.REPLACE)[0]
    db = _open(destination)
    first = db.get_character_card_by_name(CHARACTER_NAME)
    assert first is not None
    db.update_character_card(
        first["id"], {"description": "stale local description"}, first["version"]
    )
    db.close_connection()

    success, _message, status = _import(
        chatbook_zip, destination, ConflictResolution.REPLACE
    )

    assert success is True
    assert status.failed_items == 0
    db = _open(destination)
    cursor = db.get_connection().execute(
        "SELECT COUNT(*) FROM character_cards WHERE name = ? AND deleted = 0",
        (CHARACTER_NAME,),
    )
    assert cursor.fetchone()[0] == 1, "REPLACE must not duplicate the character"
    card = db.get_character_card_by_name(CHARACTER_NAME)
    assert card["description"] == CHARACTER_DESCRIPTION
    assert card["id"] == first["id"], "the existing card is updated, not recreated"
    assert card["version"] > first["version"], "versioned update, not raw SQL"
    db.close_connection()


def test_replace_conversation_retires_the_old_and_keeps_one_active(
    tmp_path, chatbook_zip, destination
):
    """AC#1 for conversations: replacing a conflicting conversation must not
    leave two ACTIVE conversations with the same title. The existing one is
    soft-deleted (the app-wide non-destructive retire) and the imported one
    lands with the imported messages."""
    assert _import(chatbook_zip, destination, ConflictResolution.REPLACE)[0]
    db = _open(destination)
    first_rows = _active_conversation_rows(db, CONVERSATION_TITLE)
    assert len(first_rows) == 1
    first_id = first_rows[0]["id"]
    db.close_connection()

    success, _message, status = _import(
        chatbook_zip, destination, ConflictResolution.REPLACE
    )

    assert success is True
    assert status.skipped_items == 0
    db = _open(destination)
    rows = _active_conversation_rows(db, CONVERSATION_TITLE)
    active = [row for row in rows if not row["deleted"]]
    retired = [row for row in rows if row["deleted"]]
    assert len(active) == 1, "exactly one ACTIVE conversation may hold the title"
    assert active[0]["id"] != first_id
    assert any(row["id"] == first_id for row in retired), (
        "the replaced conversation is soft-deleted, not destroyed"
    )
    assert _conversation_message_contents(db, active[0]["id"]) == [
        CONVERSATION_MESSAGE
    ]
    db.close_connection()


# ---------------------------------------------------------------------------
# SKIP and RENAME: existing behaviour pinned
# ---------------------------------------------------------------------------


def test_skip_leaves_existing_items_untouched(tmp_path, chatbook_zip, destination):
    assert _import(chatbook_zip, destination, ConflictResolution.SKIP)[0]

    success, _message, status = _import(
        chatbook_zip, destination, ConflictResolution.SKIP
    )

    assert success is True
    assert status.skipped_items == 3
    assert status.successful_items == 0
    db = _open(destination)
    assert len(_active_note_rows(db, NOTE_TITLE)) == 1
    cursor = db.get_connection().execute(
        "SELECT COUNT(*) FROM character_cards WHERE deleted = 0"
    )
    assert cursor.fetchone()[0] >= 1
    active = [
        row
        for row in _active_conversation_rows(db, CONVERSATION_TITLE)
        if not row["deleted"]
    ]
    assert len(active) == 1
    db.close_connection()


def test_rename_lands_conflicting_items_under_a_new_name(
    tmp_path, chatbook_zip, destination
):
    assert _import(chatbook_zip, destination, ConflictResolution.RENAME)[0]

    success, _message, status = _import(
        chatbook_zip, destination, ConflictResolution.RENAME
    )

    assert success is True
    assert status.successful_items == 3
    db = _open(destination)
    note_titles = [
        row["title"]
        for row in db.get_connection()
        .execute("SELECT title FROM notes WHERE deleted = 0")
        .fetchall()
    ]
    assert NOTE_TITLE in note_titles
    assert f"{NOTE_TITLE} (1)" in note_titles
    cursor = db.get_connection().execute(
        "SELECT name FROM character_cards WHERE deleted = 0"
    )
    character_names = [row["name"] for row in cursor.fetchall()]
    assert CHARACTER_NAME in character_names
    assert f"{CHARACTER_NAME} (1)" in character_names
    active = [
        row
        for row in _active_conversation_rows(db, CONVERSATION_TITLE)
        if not row["deleted"]
    ]
    assert len(active) == 1
    cursor = db.get_connection().execute(
        "SELECT title FROM conversations WHERE deleted = 0 AND title LIKE ?",
        (f"{CONVERSATION_TITLE}%",),
    )
    assert sorted(row["title"] for row in cursor.fetchall()) == [
        CONVERSATION_TITLE,
        f"{CONVERSATION_TITLE} (1)",
    ]
    db.close_connection()


# ---------------------------------------------------------------------------
# MERGE: rejected loudly, gone from the wizard
# ---------------------------------------------------------------------------


def test_merge_is_rejected_loudly_before_anything_is_written(
    tmp_path, chatbook_zip, destination
):
    """AC#2/AC#3: 'merge' has no implementation, so the importer must refuse
    the strategy up front -- not silently fall through to create (the old
    resolver behaviour rewrote MERGE to RENAME without a word)."""
    success, message, status = _import(
        chatbook_zip, destination, ConflictResolution.MERGE
    )

    assert success is False
    assert "merge" in message.lower()
    assert status.errors, "the rejection must be recorded, not just returned"
    db = _open(destination)
    assert _active_note_rows(db, NOTE_TITLE) == [], "nothing may be written"
    assert _active_conversation_rows(db, CONVERSATION_TITLE) == []
    cursor = db.get_connection().execute(
        "SELECT COUNT(*) FROM character_cards WHERE name = ?", (CHARACTER_NAME,)
    )
    assert cursor.fetchone()[0] == 0
    db.close_connection()


def test_merge_is_always_rejected_not_only_on_conflict(
    tmp_path, chatbook_zip, destination
):
    """The rejection is about the strategy value itself, so it fires even
    when nothing in the destination conflicts yet."""
    fresh = destination.parent / "fresh-destination.db"
    shutil.copyfile(destination, fresh)

    success, message, _status = _import(chatbook_zip, fresh, ConflictResolution.MERGE)

    assert success is False
    assert "merge" in message.lower()


# ---------------------------------------------------------------------------
# The resolver: no silent rewrites of an unsupported strategy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resolve", ["resolve_note_conflict", "resolve_conversation_conflict",
                "resolve_character_conflict", "resolve_prompt_conflict"]
)
@pytest.mark.parametrize(
    "strategy", [ConflictResolution.SKIP, ConflictResolution.RENAME,
                 ConflictResolution.REPLACE]
)
def test_resolver_passes_supported_strategies_through(resolve, strategy):
    resolver = ConflictResolver()
    method = getattr(resolver, resolve)

    assert (
        method({"title": "A", "name": "A"}, {"title": "A", "name": "A"}, strategy)
        == strategy
    )


@pytest.mark.parametrize(
    "resolve", ["resolve_note_conflict", "resolve_conversation_conflict",
                "resolve_character_conflict", "resolve_prompt_conflict"]
)
def test_resolver_ask_without_callback_falls_back_to_skip(resolve):
    resolver = ConflictResolver()
    method = getattr(resolver, resolve)

    assert (
        method({"title": "A", "name": "A"}, {"title": "A", "name": "A"},
               ConflictResolution.ASK)
        == ConflictResolution.SKIP
    )


@pytest.mark.parametrize(
    "resolve", ["resolve_note_conflict", "resolve_conversation_conflict",
                "resolve_character_conflict", "resolve_prompt_conflict"]
)
def test_resolver_rejects_merge_instead_of_rewriting_it(resolve):
    """The old resolver quietly turned MERGE into RENAME ('for now'); an
    unimplemented strategy must now be an error, never a silent substitute."""
    resolver = ConflictResolver()
    method = getattr(resolver, resolve)

    with pytest.raises(ValueError, match="merge"):
        method({"title": "A", "name": "A"}, {"title": "A", "name": "A"},
               ConflictResolution.MERGE)


def test_resolver_ask_with_callback_consults_the_user():
    def _choose_rename(_conflict_info):
        return ConflictResolution.RENAME

    resolver = ConflictResolver(ask_callback=_choose_rename)

    assert (
        resolver.resolve_note_conflict(
            {"title": "A"}, {"title": "A"}, ConflictResolution.ASK
        )
        == ConflictResolution.RENAME
    )


def test_resolver_exposes_no_merge_notes_helper():
    """AC#3: ``merge_notes`` was dead code with unimplementable semantics
    (keyword union on a table with no keywords column, an ``updated_at``
    field no update API accepts); it is deleted, not wired."""
    assert not hasattr(ConflictResolver, "merge_notes")


# ---------------------------------------------------------------------------
# The wizard: no merge option offered
# ---------------------------------------------------------------------------


class _NullWizard:
    """The slice of WizardContainer ConflictResolutionStep touches."""

    app_instance = None
    wizard_data: dict = {}
    initial_execution_mode = "local"
    can_go_back = True

    def refresh_current_step(self) -> None:
        return None


@pytest.mark.asyncio
async def test_wizard_offers_no_merge_option():
    """AC#2: with merge unimplemented the radio button is gone; the wizard
    must not offer a strategy the importer will reject."""
    from textual.app import App, ComposeResult

    class _StepHost(App):
        def __init__(self, step):
            super().__init__()
            self._step = step

        def compose(self) -> ComposeResult:
            yield self._step

    step = ConflictResolutionStep(
        wizard=_NullWizard(),
        config=WizardStepConfig(
            id="conflict-resolution", title="Conflict Resolution", step_number=3
        ),
    )

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        radio_ids = {
            button.id
            for button in step.query_one(
                "#resolution-strategy", RadioSet
            ).query(RadioButton)
        }
        assert radio_ids == {"strategy-skip", "strategy-rename", "strategy-replace"}
        texts = [str(widget.renderable) for widget in step.query(Static)]
        assert not any("Merge" in text for text in texts), (
            "the merge strategy description must not ship either"
        )
