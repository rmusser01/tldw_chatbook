"""ChaChaNotes read-then-write transactions survive a concurrent writer.

TASK-34000.84. Each case opens its own transaction, reads, and then writes.
Under a DEFERRED ``BEGIN`` SQLite refuses the read-to-write upgrade at once
when another connection committed or restarted the WAL after the read
(SQLITE_BUSY_SNAPSHOT, 517), or still holds the write lock (SQLITE_BUSY, 5),
and the busy timeout never applies. Every case takes the write lock first, so
the other connection is held off or waited for, and the write lands.

A case returns ``(mutate, landed)``: the call under test, and a read-back
that its write committed.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Actor_Packs.test_actor_pack_export import _shared_visual_manifest
from Tests.Chat import test_citation_artifact_ownership as artifact_owners
from Tests.Chat import test_citation_legacy_migration as legacy_citations
from Tests.Chat import test_citation_payload_lifecycle as payload_lifecycle
from Tests.Notes.sqlite_contention_support import (
    CONTENTION_IDS,
    CONTENTION_MODES,
    held_off,
    write_committed_between_read_and_write,
    write_lock_held_elsewhere,
)
from Tests.Sync_Interop import test_note_organization_receipt_finalization as receipts
from Tests.Sync_Interop import test_notes_organization_legacy_inventory as inventory
from Tests.Sync_Interop.test_notes_organization_intent_dispatch import (
    DATASET,
    DEVICE,
    PROFILE,
    _id,
    _intent,
    _record_keyword_intent,
    _scope,
    _set_group_state,
)
from tldw_chatbook.Character_Chat.Character_Chat_Lib import create_conversation
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.citation_artifact_ownership import (
    ArtifactOwnerOutboxState,
    CitationArtifactOwnershipCoordinator,
)
from tldw_chatbook.Chat.citation_legacy_migration import (
    CitationLegacyMigrationService,
    LegacyMigrationState,
)
from tldw_chatbook.Chat.citation_payload_lifecycle import CitationPayloadLifecycle
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleContextRepository,
    ConsoleMemoryRecord,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_persona_assignment import (
    prepare_buddy_persona_assignment,
)
from tldw_chatbook.Chat.console_prompt_mixin_models import MixinPreferences
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Notes.agent_lessons import initialize_agent_lessons_folder
from tldw_chatbook.Notes.note_folder_models import (
    FOLDER_CONFLICT_COPY,
    FolderConflictError,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.note_import_executor import LocalNoteImportTarget
from tldw_chatbook.Notes.note_import_plan_models import ParsedNotePayload
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding
from tldw_chatbook.Sync_Interop.conflict_review import SyncV2ConflictReviewService
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

#: ``case(db, tmp_path) -> (mutate, landed)``, or ``(mutate, landed, other_db)``
#: for a case that needs its own database.
Case = Callable[..., tuple]


@pytest.fixture(scope="module")
def _template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One migrated database per module; each test opens a copy (~60 ms)."""
    path = tmp_path_factory.mktemp("contention") / "template.sqlite"
    CharactersRAGDB(path, client_id="contention").close_connection()
    return path


#: Closers a case registers for stores it opens besides the database.
_CLEANUPS: list[Callable[[], object]] = []


@pytest.fixture(autouse=True)
def _close_case_stores(db: CharactersRAGDB) -> Iterator[None]:
    """Close what a case opened while ``db`` is still open (requesting it
    makes this teardown run first)."""
    yield
    while _CLEANUPS:
        _CLEANUPS.pop()()


@pytest.fixture
def db(_template: Path, tmp_path: Path) -> Iterator[CharactersRAGDB]:
    path = tmp_path / "contention.sqlite"
    shutil.copyfile(_template, path)
    database = CharactersRAGDB(path, client_id="contention")
    try:
        yield database
    finally:
        database.close_connection()


# --- CharactersRAGDB ------------------------------------------------------


def _note_update(db, tmp_path):
    note_id = db.add_note("Title", "Body")
    return (
        lambda: db.update_note(note_id, {"content": "Saved"}, expected_version=1),
        lambda: db.get_note_by_id(note_id)["content"] == "Saved",
    )


def _keyword_add(db, tmp_path):  # _add_generic_item
    return (
        lambda: db.add_keyword("fresh"),
        lambda: db.get_keyword_by_text("fresh") is not None,
    )


def _keyword_rename(db, tmp_path):  # _update_generic_item
    keyword_id = db.add_keyword("old")
    return (
        lambda: db.update_keyword(keyword_id, "new", expected_version=1),
        lambda: db.get_keyword_by_id(keyword_id)["keyword"] == "new",
    )


def _keyword_delete(db, tmp_path):  # _soft_delete_generic_item
    keyword_id = db.add_keyword("gone")
    return (
        lambda: db.soft_delete_keyword(keyword_id, expected_version=1),
        lambda: db.get_keyword_by_id(keyword_id) is None,
    )


def _character_update(db, tmp_path):
    character_id = db.add_character_card({"name": "Ada"})
    return (
        lambda: db.update_character_card(
            character_id, {"description": "Saved"}, expected_version=1
        ),
        lambda: db.get_character_card_by_id(character_id)["description"] == "Saved",
    )


def _character_delete(db, tmp_path):
    character_id = db.add_character_card({"name": "Bea"})
    return (
        lambda: db.soft_delete_character_card(character_id, expected_version=1),
        lambda: db.get_character_card_by_id(character_id) is None,
    )


def _character_conversation_create(db, tmp_path):  # Character_Chat_Lib
    character_id = db.add_character_card({"name": "Cass"})
    created: list = []
    return (
        lambda: created.append(create_conversation(db, character_id=character_id)),
        lambda: (
            created[0] is not None and db.get_conversation_by_id(created[0]) is not None
        ),
    )


def _conversation_keywords(db, tmp_path):
    conversation_id = db.add_conversation({"title": "Talk"})
    keyword_id = db.add_keyword("topic")
    return (
        lambda: db.replace_keywords_for_conversation(conversation_id, [keyword_id]),
        lambda: (
            [row["id"] for row in db.get_keywords_for_conversation(conversation_id)]
            == [keyword_id]
        ),
    )


# --- learning repository (Study) -------------------------------------------


def _deck_and_card(db):
    deck_id = db.learning.create_deck("Deck")
    card_id = db.learning.create_flashcard(
        {"deck_id": deck_id, "front": "Front", "back": "Back"}
    )
    return deck_id, card_id


def _flashcard_create(db, tmp_path):
    deck_id = db.learning.create_deck("Deck")
    created: list[str] = []
    return (
        lambda: created.append(
            db.learning.create_flashcard(
                {"deck_id": deck_id, "front": "Front", "back": "Back"}
            )
        ),
        lambda: db.learning.get_flashcard(created[0]) is not None,
    )


def _flashcard_update(db, tmp_path):
    _, card_id = _deck_and_card(db)
    return (
        lambda: db.learning.update_flashcard(card_id, front="Edited"),
        lambda: db.learning.get_flashcard(card_id)["front"] == "Edited",
    )


def _flashcard_review(db, tmp_path):
    _, card_id = _deck_and_card(db)
    return (
        lambda: db.learning.update_flashcard_review(card_id, 4),
        lambda: db.learning.get_flashcard(card_id)["repetitions"] == 1,
    )


def _deck_update(db, tmp_path):
    deck_id = db.learning.create_deck("Deck")
    return (
        lambda: db.learning.update_deck(deck_id, name="Renamed"),
        lambda: db.learning.get_deck(deck_id)["name"] == "Renamed",
    )


def _flashcard_delete(db, tmp_path):
    _, card_id = _deck_and_card(db)
    return (
        lambda: db.learning.delete_flashcard(card_id),
        lambda: db.learning.get_flashcard(card_id) is None,
    )


def _flashcard_move(db, tmp_path):
    _, card_id = _deck_and_card(db)
    target = db.learning.create_deck("Target")
    return (
        lambda: db.learning.move_flashcard(card_id, target),
        lambda: db.learning.get_flashcard(card_id)["deck_id"] == target,
    )


def _deck_delete(db, tmp_path):
    deck_id = db.learning.create_deck("Deck")
    return (
        lambda: db.learning.delete_deck(deck_id),
        lambda: db.learning.get_deck(deck_id) is None,
    )


def _template_id(db):
    return db.learning.create_flashcard_template(
        name="Basic", front_template="{{Front}}", back_template="{{Back}}"
    )["id"]


def _template_update(db, tmp_path):
    template_id = _template_id(db)
    return (
        lambda: db.learning.update_flashcard_template(template_id, name="Renamed"),
        lambda: db.learning.get_flashcard_template(template_id)["name"] == "Renamed",
    )


def _template_delete(db, tmp_path):
    template_id = _template_id(db)
    return (
        lambda: db.learning.delete_flashcard_template(template_id, expected_version=1),
        lambda: db.learning.get_flashcard_template(template_id) is None,
    )


def _quiz_id(db):
    return db.learning.create_quiz("Quiz")


def _question_id(db, quiz_id):
    return db.learning.create_question(
        quiz_id, "true_false", "Is the sky blue?", "true"
    )


def _quiz_update(db, tmp_path):
    quiz_id = _quiz_id(db)
    return (
        lambda: db.learning.update_quiz(quiz_id, {"name": "Renamed"}),
        lambda: db.learning.get_quiz(quiz_id)["name"] == "Renamed",
    )


def _quiz_delete(db, tmp_path):
    quiz_id = _quiz_id(db)
    return (
        lambda: db.learning.delete_quiz(quiz_id),
        lambda: db.learning.get_quiz(quiz_id) is None,
    )


def _question_create(db, tmp_path):
    quiz_id = _quiz_id(db)
    created: list[str] = []
    return (
        lambda: created.append(_question_id(db, quiz_id)),
        lambda: db.learning.get_question(created[0]) is not None,
    )


def _question_update(db, tmp_path):
    question_id = _question_id(db, _quiz_id(db))
    return (
        lambda: db.learning.update_question(question_id, {"question_text": "Edited?"}),
        lambda: db.learning.get_question(question_id)["question_text"] == "Edited?",
    )


def _question_delete(db, tmp_path):
    question_id = _question_id(db, _quiz_id(db))
    return (
        lambda: db.learning.delete_question(question_id),
        lambda: db.learning.get_question(question_id) is None,
    )


def _attempt_start(db, tmp_path):
    quiz_id = _quiz_id(db)
    _question_id(db, quiz_id)
    started: list[dict] = []
    return (
        lambda: started.append(db.learning.start_attempt(quiz_id)),
        lambda: db.learning.get_attempt(started[0]["id"]) is not None,
    )


def _attempt_submit(db, tmp_path):
    quiz_id = _quiz_id(db)
    question_id = _question_id(db, quiz_id)
    attempt = db.learning.start_attempt(quiz_id)
    answers = [{"question_id": question_id, "user_answer": "true"}]
    return (
        lambda: db.learning.submit_attempt(attempt["id"], answers),
        lambda: db.learning.get_attempt(attempt["id"])["completed_at"] is not None,
    )


# --- Sync v2 (local-first) Notes organization -------------------------------


def _sync(db, tmp_path):
    state = SyncStateRepository(tmp_path / "sync.sqlite", client_id="contention")
    state.set_sync_v2_profile_state(
        **_scope(), profile_mode="local_first", device_id=DEVICE, dataset_id=DATASET
    )
    _set_group_state(db)
    service = NotesOrganizationSyncService(
        notes_repository=NotesOrganizationRepository(db, server_profile_id=PROFILE),
        state_repository=state,
    )
    return service, state, LocalNoteFolderRepository(db)


def _count(db, sql: str, *params) -> int:
    return db.get_connection().execute(sql, params).fetchone()[0]


def _sync_managed_links(db, tmp_path):
    service, _, folders = _sync(db, tmp_path)
    folder = service.create_folder(
        folder_repository=folders, name="Root", parent_id=None, **_scope()
    )
    note_id = db.add_note("Note", "Body", note_id=_id(21))
    return (
        lambda: service.mutate_managed_folder_links(
            folder_repository=folders,
            mutation_method="reconcile_managed",
            owner_id="source-a",
            desired=((folder.folder_id, note_id),),
            **_scope(),
        ),
        lambda: len(folders.list_memberships(note_ids=(note_id,))) == 1,
    )


def _sync_folder_rename(db, tmp_path):  # _mutate_folder: rename/move/delete/restore
    service, _, folders = _sync(db, tmp_path)
    folder = service.create_folder(
        folder_repository=folders, name="Work", parent_id=None, **_scope()
    )
    return (
        lambda: service.rename_folder(
            folder_repository=folders,
            folder_id=folder.folder_id,
            name="Projects",
            expected_version=folder.version,
            **_scope(),
        ),
        lambda: folders.get_folder(folder.folder_id).name == "Projects",
    )


def _sync_subject_keywords(db, tmp_path):
    service, _, _ = _sync(db, tmp_path)
    note_id = db.add_note("Note", "Body", note_id=_id(20))
    return (
        lambda: service.sync_subject_keywords(
            subject_type="note", subject_id=note_id, keywords=("Lesson",), **_scope()
        ),
        lambda: (
            [row["keyword"] for row in db.get_keywords_for_note(note_id)] == ["Lesson"]
        ),
    )


def _sync_keyword_create(db, tmp_path):
    service, _, _ = _sync(db, tmp_path)
    return (
        lambda: service.create_keyword(keyword="Portable", **_scope()),
        lambda: db.get_keyword_by_text("Portable") is not None,
    )


def _sync_keyword_rename(db, tmp_path):
    service, _, _ = _sync(db, tmp_path)
    keyword_id = service.create_keyword(keyword="Old", **_scope())
    return (
        lambda: service.mutate_keyword(
            keyword_id=keyword_id, expected_version=1, keyword="New", **_scope()
        ),
        lambda: db.get_keyword_by_id(keyword_id)["keyword"] == "New",
    )


def _sync_collection_create(db, tmp_path):
    service, _, _ = _sync(db, tmp_path)
    return (
        lambda: service.create_keyword_collection(name="Lessons", **_scope()),
        lambda: (
            _count(
                db, "SELECT COUNT(*) FROM keyword_collections WHERE name = ?", "Lessons"
            )
            == 1
        ),
    )


def _sync_collection_rename(db, tmp_path):
    service, _, _ = _sync(db, tmp_path)
    collection_id = service.create_keyword_collection(name="Lessons", **_scope())
    return (
        lambda: service.mutate_keyword_collection(
            collection_id=collection_id,
            expected_version=1,
            update_data={"name": "Renamed"},
            **_scope(),
        ),
        lambda: (
            _count(
                db, "SELECT COUNT(*) FROM keyword_collections WHERE name = ?", "Renamed"
            )
            == 1
        ),
    )


def _sync_collection_link(db, tmp_path):
    service, _, _ = _sync(db, tmp_path)
    keyword_id = service.create_keyword(keyword="Portable", **_scope())
    collection_id = service.create_keyword_collection(name="Lessons", **_scope())
    return (
        lambda: service.set_collection_keyword_link(
            collection_id=collection_id, keyword_id=keyword_id, linked=True, **_scope()
        ),
        lambda: (
            _count(
                db,
                "SELECT COUNT(*) FROM collection_keywords WHERE collection_id = ?",
                collection_id,
            )
            == 1
        ),
    )


def _sync_enrollment_status(db, tmp_path):
    service, _, _ = _sync(db, tmp_path)
    return (
        lambda: service._record_server_enrollment_status(
            server_profile_id=PROFILE,
            dataset_id=DATASET,
            bootstrap_id="bootstrap-a",
            state="failed",
            captured_count=0,
            expected_count=0,
            error_code="server_failed",
        ),
        lambda: (
            _count(
                db,
                "SELECT COUNT(*) FROM notes_organization_sync_checkpoints "
                "WHERE local_state = 'failed'",
            )
            == 1
        ),
    )


def _accept(state, intent_id: str) -> None:
    state.mark_sync_v2_outbox_push_results(
        **_scope(),
        dataset_id=DATASET,
        accepted=[
            {
                "client_envelope_id": intent_id,
                "server_cursor": 17,
                "object_revision": 1,
                "apply_status": "applied",
            }
        ],
        rejected=[],
        conflicts=[],
    )


def _sync_acknowledgement(db, tmp_path):
    service, state, _ = _sync(db, tmp_path)
    intent_id = _record_keyword_intent(db)
    service.drain_pending_intents(**_scope(), dataset_id=DATASET, device_id=DEVICE)
    _accept(state, intent_id)
    return (
        lambda: service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET),
        lambda: _intent(db, intent_id)["acknowledged_at"] is not None,
    )


def _sync_successor_base(db, tmp_path):  # _bind_predecessor_base_if_ready
    service, state, _ = _sync(db, tmp_path)
    repository = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    intents = []
    with db.transaction() as cursor:
        for version, keyword in enumerate(("First", "Second"), start=1):
            intents.append(
                repository.record_intent(
                    cursor,
                    profile=PROFILE,
                    dataset=DATASET,
                    domain="notes.keyword",
                    object_id=_id(31),
                    operation="upsert",
                    payload={"keyword": keyword},
                    source_version=version,
                )
            )
    service.drain_pending_intents(**_scope(), dataset_id=DATASET, device_id=DEVICE)
    _accept(state, intents[0])
    service.reconcile_acknowledgements(**_scope(), dataset_id=DATASET)
    return (
        lambda: service.drain_pending_intents(
            **_scope(), dataset_id=DATASET, device_id=DEVICE
        ),
        lambda: _intent(db, intents[1])["base_server_cursor"] is not None,
    )


def _adoption_review_resolve(db, tmp_path):  # conflict_review, no receipt
    organization = NotesOrganizationRepository(db, server_profile_id=PROFILE)
    LocalNoteFolderRepository(db).create_folder(name="agent_lessons", parent_id=None)
    initialize_agent_lessons_folder(
        db,
        scope_mode="synchronized",
        profile_id=PROFILE,
        dataset_id=DATASET,
        organization_repository=organization,
    )
    review_id = (
        db.get_connection()
        .execute("SELECT review_id FROM notes_organization_adoption_reviews")
        .fetchone()[0]
    )
    service = SyncV2ConflictReviewService(
        state_repository=object(), notes_repository=organization
    )
    return (
        lambda: service.resolve_notes_organization_adoption(
            review_id=review_id, action="keep_local"
        ),
        lambda: (
            _count(
                db,
                "SELECT COUNT(*) FROM notes_organization_adoption_reviews "
                "WHERE state = 'resolved'",
            )
            == 1
        ),
    )


def _receipts_finalize(db, tmp_path):
    notes, _, library, organization, _ = receipts._services(tmp_path)
    receipts._save_pending(library)
    receipts._make_ready(notes)
    return (
        lambda: organization.finalize_pending_note_organization_receipts(
            server_profile_id=receipts.PROFILE, dataset_id=receipts.DATASET
        ),
        lambda: _count(notes, "SELECT COUNT(*) FROM note_organization_receipts") == 0,
        notes,
    )


def _placement_review_resolve(db, tmp_path):
    notes, _, library, organization, _ = receipts._services(tmp_path)
    LocalNoteFolderRepository(notes).create_folder(name="agent_lessons", parent_id=None)
    receipts._save_pending(library)
    receipts._make_ready(notes)
    organization.finalize_pending_note_organization_receipts(
        server_profile_id=receipts.PROFILE, dataset_id=receipts.DATASET
    )
    review_id = (
        notes.get_connection()
        .execute("SELECT review_id FROM note_organization_receipts")
        .fetchone()[0]
    )
    return (
        lambda: organization.resolve_placement_review(
            review_id=review_id, action="keep_local"
        ),
        lambda: _count(notes, "SELECT COUNT(*) FROM note_organization_receipts") == 0,
        notes,
    )


def _inventory_run(db, tmp_path):
    path = tmp_path / "inventory.sqlite"
    inventory._seed_inventory_db(path)
    own = CharactersRAGDB(path, client_id="inventory-tests")
    runner = inventory._inventory_types()(
        NotesOrganizationRepository(own, server_profile_id=inventory.PROFILE),
        dataset_id=inventory.DATASET,
        enrolled_note_ids={inventory.IDS["note_enrolled"]},
        enrolled_conversation_ids={"conversation-enrolled"},
    )
    results: list = []
    return (
        lambda: results.append(runner.run()),
        lambda: results[0].status == "complete",
        own,
    )


# --- Console -----------------------------------------------------------------


def _memory_if_current(db, tmp_path):
    conversation_id = db.add_conversation({"title": "memory"})
    boundary_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "assistant", "content": "turn"}
    )
    repository = ConsoleContextRepository(db)
    record = ConsoleMemoryRecord(
        memory_id="memory-1",
        conversation_id=conversation_id,
        boundary_message_id=boundary_id,
        captured_leaf_message_id=boundary_id,
        lineage_json=f'["{boundary_id}"]',
        summary_text="A derived recap.",
        provider="openai",
        model="gpt-test",
        prompt_id="console.rewind_summarize",
        prompt_revision=3,
        prompt_digest="prompt-sha256",
        selected_units_json=f'[{{"message_id":"{boundary_id}","version":1}}]',
        summarized_prefix_digest="prefix-sha256",
        input_tokens=1_000,
        output_tokens=120,
        before_tokens=8_000,
        after_tokens=2_000,
        created_at="2026-08-10T12:00:00Z",
    )
    repository.insert_memory(record)
    guarded = ConsoleMemoryRecord(**{**record.__dict__, "memory_id": "memory-guarded"})
    return (
        lambda: repository.insert_memory_if_current(
            guarded, expected_memory_id="memory-1", expected_memory_revision=1
        ),
        lambda: len(repository.list_active_memories(conversation_id)) == 2,
    )


def _mixin_preferences(db, tmp_path):
    conversation_id = db.add_conversation({"title": "mixins"})
    persistence = ChatPersistenceService(db)
    return (
        lambda: persistence.save_conversation_mixin_preferences(
            conversation_id=conversation_id,
            expected_revision=0,
            value=MixinPreferences(),
        ),
        lambda: (
            persistence.get_conversation_mixin_preferences(
                conversation_id
            ).preferences.revision
            == 1
        ),
    )


def _persona_assignment(db, tmp_path):
    """``Tests/Chat/test_console_persona_assignment.py``'s ``context``."""
    workspace_db = WorkspaceDB(tmp_path / "workspace.sqlite", "contention")
    _CLEANUPS.append(workspace_db.close)
    personas = LocalCharacterPersonaService(
        db, persona_store_path=tmp_path / "personas.json"
    )
    personas.create_persona_profile(
        {"id": "guide", "name": "Guide", "system_prompt": "Be guide."}
    )
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=controller),
        local_character_persona_service=personas,
        workspace_registry_service=LocalWorkspaceRegistryService(workspace_db),
    )
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="test-model"),
        assistant_kind="generic",
    )
    store.persist_session_if_needed(session.id)
    assignment = prepare_buddy_persona_assignment(
        app, BuddyBinding.for_session(session), session, "guide"
    )
    return (
        lambda: asyncio.run(assignment.apply()),
        lambda: (
            db.get_conversation_by_id(session.persisted_conversation_id)["assistant_id"]
            == "guide"
        ),
    )


# --- Citation provenance ---------------------------------------------------------


def _artifact_owner_operation(db):
    """``Tests/Chat/test_citation_artifact_ownership.py``'s shared-database setup."""
    repository = artifact_owners._repository(db)
    artifact_owners._persist(db, repository)
    artifact_owners._create_shared_database_artifact_tables(db)
    coordinator = CitationArtifactOwnershipCoordinator(
        artifact_store=artifact_owners._SharedDatabaseArtifactStore(db),
        trace_repository=repository,
    )
    operation = coordinator.prepare_link_operation(
        artifact_owners._owner_request(repository),
        artifact_id="shared-artifact-1",
        artifact_revision=1,
        artifact_body='{"artifact":"shared-artifact-1"}',
    )
    return repository, coordinator, operation


def _citation_shared_owner(db, tmp_path):
    _, coordinator, operation = _artifact_owner_operation(db)
    return (
        lambda: coordinator.apply_shared_database_owner_operation(operation),
        lambda: _count(db, "SELECT COUNT(*) FROM shared_test_artifact_owners") == 1,
    )


def _citation_owner_apply(db, tmp_path):
    repository, _, operation = _artifact_owner_operation(db)
    return (
        lambda: repository.apply_artifact_owner_operation(operation),
        lambda: _count(db, "SELECT COUNT(*) FROM rag_artifact_owner_operations") == 1,
    )


def _citation_owner_acknowledge(db, tmp_path):
    repository, _, operation = _artifact_owner_operation(db)
    repository.apply_artifact_owner_operation(operation)
    acknowledged = operation.model_copy(
        update={
            "state": ArtifactOwnerOutboxState.ACKNOWLEDGED,
            "acknowledged_at": datetime.now(UTC),
        }
    )
    # Already applied above: keep the re-apply's own transaction out of the
    # way, so the forced contention lands in the acknowledgement's.
    repository.apply_artifact_owner_operation = lambda _operation: None
    return (
        lambda: repository.acknowledge_artifact_owner_operation(acknowledged),
        lambda: (
            _count(
                db,
                "SELECT COUNT(*) FROM rag_artifact_owner_operations "
                "WHERE state = 'acknowledged'",
            )
            == 1
        ),
    )


def _legacy_service(db, tmp_path, count: int, batch_size: int = 100, **options):
    conversation_id, message_ids = legacy_citations._conversation_with_messages(
        db, count
    )
    sidecar = tmp_path / "chat_rag_context.json"
    legacy_citations._write_sidecar(sidecar, conversation_id, message_ids)
    service = CitationLegacyMigrationService(
        db=db,
        repository=legacy_citations._repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=legacy_citations.CODEC,
        batch_size=batch_size,
        **options,
    )
    return service, conversation_id, message_ids, sidecar


def _citation_legacy_batch(db, tmp_path):
    service, conversation_id, _, _ = _legacy_service(db, tmp_path, 2)
    results: list = []
    return (
        lambda: results.append(service.migrate_next_batch(conversation_id)),
        lambda: results[0].state is LegacyMigrationState.COMPLETE,
    )


def _legacy_cutover(order):
    """``migrate_next_batch``'s cutover, its second transaction.

    The table's own contention lands on the call's first read-then-write
    transaction (the batch claim), so this case forces ``order`` again from
    the ``before_cutover_hook``, which runs between the two.
    """

    def case(db, tmp_path):
        stack = ExitStack()
        _CLEANUPS.append(stack.close)
        forced: list = []
        service, conversation_id, _, _ = _legacy_service(
            db,
            tmp_path,
            2,
            before_cutover_hook=lambda: forced.append(stack.enter_context(order(db))),
        )
        results: list = []
        return (
            lambda: results.append(service.migrate_next_batch(conversation_id)),
            lambda: (
                results[0].state is LegacyMigrationState.COMPLETE
                and held_off(order, forced[0])
            ),
        )

    case.__name__ = f"_citation_legacy_cutover_{order.__name__.split('_')[1]}"
    return case


_citation_legacy_cutover_committed = _legacy_cutover(
    write_committed_between_read_and_write
)
_citation_legacy_cutover_lock = _legacy_cutover(write_lock_held_elsewhere)


def _diverging_service(db, tmp_path):
    """``test_explicit_retry_deletes_hidden_staging_before_rebuild``'s setup,
    with one-message batches instead of 101 messages."""
    service, conversation_id, message_ids, sidecar = _legacy_service(
        db, tmp_path, 2, batch_size=1
    )
    service.migrate_next_batch(conversation_id)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["conversations"][conversation_id][message_ids[-1]][
        "citation_validation"
    ] = {"valid": False}
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return service, conversation_id


def _citation_legacy_diverge(db, tmp_path):  # _record_terminal_state
    service, conversation_id = _diverging_service(db, tmp_path)
    results: list = []
    return (
        lambda: results.append(service.migrate_next_batch(conversation_id)),
        lambda: results[0].state is LegacyMigrationState.DIVERGED,
    )


def _citation_legacy_refuse_first(db, tmp_path):  # _record_terminal_state, no journal
    service, conversation_id, message_ids, sidecar = _legacy_service(db, tmp_path, 2)
    sidecar.write_text("{not json", encoding="utf-8")
    results: list = []
    return (
        lambda: results.append(service.migrate_next_batch(conversation_id)),
        lambda: _count(db, "SELECT COUNT(*) FROM rag_legacy_migration_journal") == 1,
    )


def _citation_legacy_retry(db, tmp_path):
    service, conversation_id = _diverging_service(db, tmp_path)
    assert (
        service.migrate_next_batch(conversation_id).state
        is LegacyMigrationState.DIVERGED
    )
    return (
        lambda: service.retry_diverged(conversation_id),
        lambda: _count(db, "SELECT COUNT(*) FROM rag_legacy_migration_journal") == 0,
    )


def _citation_package_record(db, tmp_path):
    service, conversation_id, message_ids, _ = _legacy_service(db, tmp_path, 1)
    record = {
        **legacy_citations._record(message_ids[0], answer="Legacy answer [1]."),
        "conversation_id": conversation_id,
        "origin": "imported",
    }
    results: list = []
    return (
        lambda: results.append(
            service.persist_package_record(
                conversation_id=conversation_id,
                message_id=message_ids[0],
                record=record,
            )
        ),
        lambda: results[0].state is LegacyMigrationState.COMPLETE,
    )


def _citation_collect(
    db, tmp_path
):  # CitationPayloadLifecycle._collect_under_artifact_guard
    repository = payload_lifecycle._repository(db)
    payload_lifecycle._persist(db, repository)
    payload_lifecycle._mark_owner_deleted(db, updated_at=payload_lifecycle.NOW)
    lifecycle = CitationPayloadLifecycle(
        repository, retention_policy=payload_lifecycle._policy()
    )
    return (
        lambda: lifecycle.collect(now=payload_lifecycle.NOW),
        lambda: _count(db, "SELECT COUNT(*) FROM rag_citation_traces") == 0,
    )


# --- Note import target --------------------------------------------------------


def _import_target(db):
    return LocalNoteImportTarget(db=db, folder_repository=LocalNoteFolderRepository(db))


def _payload(content: str = "Body", keywords=("alpha",)) -> ParsedNotePayload:
    return ParsedNotePayload(title="Imported", content=content, keywords=keywords)


def _import_create(db, tmp_path):
    note_id = str(uuid.uuid4())
    target = _import_target(db)
    return (
        lambda: target.create_note(note_id=note_id, payload=_payload()),
        lambda: db.get_note_by_id(note_id) is not None,
    )


def _import_replace(db, tmp_path):
    note_id = str(uuid.uuid4())
    target = _import_target(db)
    target.create_note(note_id=note_id, payload=_payload())
    return (
        lambda: target.replace_note(
            note_id=note_id, expected_version=1, payload=_payload("Replaced")
        ),
        lambda: db.get_note_by_id(note_id)["content"] == "Replaced",
    )


def _import_keywords(db, tmp_path):
    note_id = str(uuid.uuid4())
    target = _import_target(db)
    target.create_note(note_id=note_id, payload=_payload())
    return (
        lambda: target.sync_keywords(note_id=note_id, keywords=("beta",)),
        lambda: (
            [row["keyword"] for row in db.get_keywords_for_note(note_id)] == ["beta"]
        ),
    )


# --- Visual identity -----------------------------------------------------------


def _activate(db, actor_id, **guard):
    """``Tests/Actor_Packs/test_actor_pack_export.py``'s shared pack.

    ``require_unbound_actor`` (the built-in seed's guard) is what makes the
    activation read the actor's binding before it writes.
    """
    manifest = _shared_visual_manifest(b"neutral-png", "visual/neutral.png")
    asset = dict(manifest["assets"][0])
    asset["original_expression_key"] = asset.pop("original_label")
    asset["source_filename"] = "neutral.png"
    return VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Export reactions",
            "default_expression_key": "neutral",
            "source_kind": "manual",
            "source_context": {"provenance": "local-authoring"},
        },
        manifest=manifest,
        assets=[asset],
        actor_kind="character",
        actor_id=actor_id,
        **guard,
    )


def _visual_activate(db, tmp_path):
    character_id = db.add_character_card({"name": "Vis"})
    repository = VisualIdentityRepository(db)
    return (
        lambda: _activate(db, character_id, require_unbound_actor=True),
        lambda: repository.get_active_actor_pack("character", character_id) is not None,
    )


def _visual_unbind(db, tmp_path):  # mark_binding_deleted
    character_id = db.add_character_card({"name": "Vis"})
    _activate(db, character_id)
    repository = VisualIdentityRepository(db)
    return (
        lambda: repository.mark_binding_deleted("character", character_id),
        lambda: repository.get_active_actor_pack("character", character_id) is None,
    )


def _visual_publish(db, tmp_path):
    character_id = db.add_character_card({"name": "Vis"})
    pack_id = int(_activate(db, character_id)["pack"]["id"])
    repository = VisualIdentityRepository(db)
    manifest = _shared_visual_manifest(b"smile-png", "visual/smile.png")
    asset = dict(manifest["assets"][0])
    asset["original_expression_key"] = asset.pop("original_label")
    asset["source_filename"] = "smile.png"
    return (
        lambda: repository.publish_version(
            pack_id,
            manifest=manifest,
            assets=[asset],
            actor_kind="character",
            actor_id=character_id,
        ),
        lambda: (
            repository.get_active_actor_pack("character", character_id)["version"][
                "version_number"
            ]
            == 2
        ),
    )


def _visual_archive(db, tmp_path):  # _set_pack_status
    character_id = db.add_character_card({"name": "Vis"})
    pack_id = int(_activate(db, character_id)["pack"]["id"])
    repository = VisualIdentityRepository(db)
    repository.mark_binding_deleted("character", character_id)
    return (
        lambda: repository.archive_pack(pack_id),
        lambda: (
            _count(
                db,
                "SELECT COUNT(*) FROM visual_identity_packs WHERE id = ? "
                "AND status = 'archived'",
                pack_id,
            )
            == 1
        ),
    )


CASES: tuple[Case, ...] = (
    _note_update,
    _keyword_add,
    _keyword_rename,
    _keyword_delete,
    _character_update,
    _character_delete,
    _character_conversation_create,
    _conversation_keywords,
    _flashcard_create,
    _flashcard_update,
    _flashcard_review,
    _deck_update,
    _flashcard_delete,
    _flashcard_move,
    _deck_delete,
    _template_update,
    _template_delete,
    _quiz_update,
    _quiz_delete,
    _question_create,
    _question_update,
    _question_delete,
    _attempt_start,
    _attempt_submit,
    _sync_managed_links,
    _sync_folder_rename,
    _sync_subject_keywords,
    _sync_keyword_create,
    _sync_keyword_rename,
    _sync_collection_create,
    _sync_collection_rename,
    _sync_collection_link,
    _sync_enrollment_status,
    _sync_acknowledgement,
    _sync_successor_base,
    _adoption_review_resolve,
    _receipts_finalize,
    _placement_review_resolve,
    _inventory_run,
    _memory_if_current,
    _mixin_preferences,
    _persona_assignment,
    _citation_shared_owner,
    _citation_owner_apply,
    _citation_owner_acknowledge,
    _citation_legacy_batch,
    _citation_legacy_cutover_committed,
    _citation_legacy_cutover_lock,
    _citation_legacy_diverge,
    _citation_legacy_refuse_first,
    _citation_legacy_retry,
    _citation_package_record,
    _citation_collect,
    _import_create,
    _import_replace,
    _import_keywords,
    _visual_activate,
    _visual_unbind,
    _visual_publish,
    _visual_archive,
)


@pytest.mark.parametrize("mode", CONTENTION_MODES, ids=CONTENTION_IDS)
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.__name__.strip("_"))
def test_a_read_then_write_holds_off_a_concurrent_writer(
    db: CharactersRAGDB, tmp_path: Path, case: Case, mode
) -> None:
    mutate, landed, *own_db = case(db, tmp_path)
    target = own_db[0] if own_db else db
    try:
        with mode(target) as outcomes:
            mutate()

        assert held_off(mode, outcomes), outcomes
        assert landed()
    finally:
        if target is not db:
            target.close_connection()


@contextmanager
def _write_lock_held_past_the_busy_timeout(db: CharactersRAGDB) -> Iterator[None]:
    """Another connection holds the write lock for the whole block.

    This thread's connection stops waiting at once (``busy_timeout = 0``), in
    place of the 15 s the app's connections wait; only this test's own
    connection is changed, and it is closed after the test.
    """
    db.get_connection().execute("PRAGMA busy_timeout = 0")
    holder = sqlite3.connect(db.db_path_str, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        yield
    finally:
        holder.rollback()
        holder.close()


def _folder_mutation(service, folders, db, operation: str):
    folder = service.create_folder(
        folder_repository=folders, name="Root", parent_id=None, **_scope()
    )
    note_id = db.add_note("Note", "Body", note_id=_id(40))
    attached = None
    if operation == "move_note_out":
        attached = service.attach_folder_link(
            folder_repository=folders,
            folder_id=folder.folder_id,
            note_id=note_id,
            **_scope(),
        )
    return {
        "new_folder": lambda: service.create_folder(
            folder_repository=folders, name="Second", parent_id=None, **_scope()
        ),
        "move_note_in": lambda: service.attach_folder_link(
            folder_repository=folders,
            folder_id=folder.folder_id,
            note_id=note_id,
            **_scope(),
        ),
        "move_note_out": lambda: service.detach_folder_link(
            folder_repository=folders,
            folder_id=folder.folder_id,
            note_id=note_id,
            expected_version=attached.version if attached else 0,
            **_scope(),
        ),
    }[operation]


@pytest.mark.parametrize("operation", ["new_folder", "move_note_in", "move_note_out"])
def test_a_local_first_folder_mutation_behind_a_held_lock_reads_as_busy(
    db: CharactersRAGDB, tmp_path: Path, operation: str
) -> None:
    """AC#5: on a Sync v2 (local-first) profile, Move note and New folder go
    through the organization service's own ``BEGIN IMMEDIATE``. A lock held
    past the busy timeout raised a raw OperationalError there, which the Notes
    tree painted as "Could not update folder organization"; it is now the
    tree's ``busy`` conflict, and nothing is written."""
    service, _, folders = _sync(db, tmp_path)
    mutate = _folder_mutation(service, folders, db, operation)
    intents_before = _count(db, "SELECT COUNT(*) FROM notes_organization_sync_intents")

    with _write_lock_held_past_the_busy_timeout(db):
        with pytest.raises(FolderConflictError) as caught:
            mutate()

    assert caught.value.reason == "busy"
    assert FOLDER_CONFLICT_COPY[caught.value.reason] == (
        "The notes database was busy — nothing was written."
    )
    assert (
        _count(db, "SELECT COUNT(*) FROM notes_organization_sync_intents")
        == intents_before
    )


@pytest.mark.parametrize(
    "case", [_note_update, _keyword_rename], ids=["note", "keyword"]
)
def test_a_lock_held_past_the_busy_timeout_is_busy_not_a_version_conflict(
    db: CharactersRAGDB, tmp_path: Path, case: Case
) -> None:
    """AC#4: the write lock is now asked for first, so a lock genuinely held
    elsewhere fails as SQLite's busy error, never as the optimistic-version
    ``ConflictError`` a caller would show as someone else's edit."""
    mutate, landed = case(db, tmp_path)

    with _write_lock_held_past_the_busy_timeout(db):
        with pytest.raises(sqlite3.OperationalError) as caught:
            mutate()

    assert not isinstance(caught.value, ConflictError)
    assert caught.value.sqlite_errorcode & 0xFF == sqlite3.SQLITE_BUSY
    assert not landed()


def _inventory_runner(tmp_path: Path, **seed):
    """``test_notes_organization_legacy_inventory.py``'s seeded enrollment."""
    path = tmp_path / "inventory-scan.sqlite"
    inventory._seed_inventory_db(path, **seed)
    own = CharactersRAGDB(path, client_id="inventory-tests")
    _CLEANUPS.append(own.close_connection)
    runner = inventory._inventory_types()(
        NotesOrganizationRepository(own, server_profile_id=inventory.PROFILE),
        dataset_id=inventory.DATASET,
        enrolled_note_ids={inventory.IDS["note_enrolled"]},
        enrolled_conversation_ids={"conversation-enrolled"},
    )
    return path, runner


def _record_scans(monkeypatch: pytest.MonkeyPatch, path: Path) -> list[bool]:
    """Record, per organization scan, whether the write lock was held."""
    scans: list[bool] = []
    inventory_type = inventory._inventory_types()
    real = inventory_type._snapshot

    def probing(self, cursor):
        probe = sqlite3.connect(path, timeout=0, isolation_level=None)
        try:
            probe.execute("BEGIN IMMEDIATE")
            probe.rollback()
            scans.append(False)
        except sqlite3.OperationalError:
            scans.append(True)
        finally:
            probe.close()
        return real(self, cursor)

    monkeypatch.setattr(inventory_type, "_snapshot", probing)
    return scans


def test_enrollment_inventory_scans_outside_the_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of TASK-34000.84: each inventory step took the write lock and
    then re-read and hashed every organization row (310 ms a step with 9,800
    intents at load 55-65), so a concurrent one-row write waited 10.7 s. The
    scan now runs in a plain read transaction and is reused while no
    connection changed the database; another connection's commit only makes
    the next step scan again, still outside the lock."""
    path, runner = _inventory_runner(tmp_path)
    with sqlite3.connect(path) as setup:
        setup.execute("CREATE TABLE inventory_probe (n INTEGER)")
    scans = _record_scans(monkeypatch, path)
    steps: list[str] = []

    def after_commit(phase: str, _key: str | None) -> None:
        steps.append(phase)
        if len(steps) == 3:
            with sqlite3.connect(path) as writer:
                writer.execute("INSERT INTO inventory_probe (n) VALUES (1)")

    result = runner.run(after_commit=after_commit)

    assert result.status == "complete", result
    assert True not in scans, scans
    assert 2 <= len(scans) < len(steps), (scans, steps)


def test_a_change_between_the_scan_and_the_step_is_seen_by_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step re-checks under the write lock that nothing changed since
    the scan: a keyword renamed by another connection in between makes the
    step scan again and hold the inventory, writing nothing from the stale
    scan."""
    path, runner = _inventory_runner(tmp_path)
    steps: list[str] = []
    inventory_type = inventory._inventory_types()
    real_scan = inventory_type._scan_outside_the_write_lock
    calls = {"n": 0}

    def scan_then_rename(self) -> None:
        real_scan(self)
        calls["n"] += 1
        if calls["n"] == 3:
            with sqlite3.connect(path) as writer:
                renamed = writer.execute(
                    "UPDATE keywords SET keyword = 'Drifted' WHERE sync_id = ?",
                    (inventory.IDS["keyword_deleted"],),
                ).rowcount
            steps.append(f"renamed {renamed}")

    monkeypatch.setattr(
        inventory_type, "_scan_outside_the_write_lock", scan_then_rename
    )

    result = runner.run(after_commit=lambda phase, _key: steps.append(phase))

    assert (result.status, result.reason_code) == (
        "held",
        "inventory_source_changed",
    ), result
    assert steps[-1] == "renamed 1", steps


def _commit_after_each_outside_scan(
    monkeypatch: pytest.MonkeyPatch, path: Path, *, limit: int = 1_000
) -> list[int]:
    """Commit a write to a non-source table after every outside scan.

    Every step then finds the database changed since its scan. ``limit``
    turns a step that never progresses into a failure instead of a hang.
    """
    with sqlite3.connect(path) as setup:
        setup.execute("CREATE TABLE IF NOT EXISTS inventory_probe (n INTEGER)")
    inventory_type = inventory._inventory_types()
    real_scan = inventory_type._scan_outside_the_write_lock
    outside: list[int] = []

    def scan_then_commit(self) -> None:
        real_scan(self)
        outside.append(1)
        if len(outside) > limit:
            raise AssertionError(f"no progress after {limit} outside scans")
        with sqlite3.connect(path) as writer:
            writer.execute("INSERT INTO inventory_probe (n) VALUES (1)")

    monkeypatch.setattr(
        inventory_type, "_scan_outside_the_write_lock", scan_then_commit
    )
    return outside


def test_steady_writes_fall_back_to_one_scan_under_the_lock_per_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another connection commits after every outside scan, so no scan is
    ever current under the write lock. Each step tries the outside scan
    ``_OUTSIDE_SCAN_ATTEMPTS`` times and then scans under the lock once, so
    enrollment still completes, with at most a third of the scans under the
    lock."""
    path, runner = _inventory_runner(tmp_path)
    _commit_after_each_outside_scan(monkeypatch, path)
    scans = _record_scans(monkeypatch, path)

    result = runner.run()

    assert result.status == "complete", result
    assert True in scans, scans
    assert scans.count(True) * 3 <= len(scans), scans


@pytest.mark.parametrize("seed", [{}, {"local_state": "pulling"}], ids=["step", "held"])
def test_a_scan_connection_closed_before_the_step_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: dict
) -> None:
    """Review of TASK-34000.84: ``quiesce_connections`` (trace compaction,
    the mobile service) or ``close_idle_connections`` (quit) can close this
    thread's connection between the outside scan and the step's ``BEGIN
    IMMEDIATE``, which then reopens one. The scan taken on the closed
    connection is dropped, never re-keyed through it (that raised
    ``sqlite3.ProgrammingError``), on a stale step and on a held one."""
    path, runner = _inventory_runner(tmp_path, **seed)
    inventory_type = inventory._inventory_types()
    real_scan = inventory_type._scan_outside_the_write_lock
    closed: list[int] = []

    def scan_then_close(self) -> None:
        real_scan(self)
        if not closed:
            closed.append(1)
            self.repository.db.close_connection()

    monkeypatch.setattr(inventory_type, "_scan_outside_the_write_lock", scan_then_close)

    result = runner.run()

    assert closed == [1]
    expected = "held" if seed else "complete"
    assert result.status == expected, result
