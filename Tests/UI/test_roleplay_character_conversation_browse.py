from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.css.query import QueryError
from textual.screen import Screen
from textual.widgets import Static

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
    CharacterKeywordSnapshot,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.Chat.console_conversation_activation import (
    CharacterConversationActivationRequest,
    ConsoleActivationResultKind,
    ConsoleConversationActivationResult,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    RoleplayDraftNavigationDialog,
)
from tldw_chatbook.UI.Persona_Modules.personas_conversations_controller import (
    PersonasConversationsController,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_inspector_pane import (
    PersonasInspectorPane,
)


@pytest.mark.asyncio
async def test_retained_keyword_snapshot_reaches_visible_copy_without_replacing_live_revision(
    tmp_path,
    monkeypatch,
) -> None:
    """Dropping corpus metadata must not mislabel an older ready result as current."""
    db = CharactersRAGDB(tmp_path / "retained.sqlite", client_id="retained")
    try:
        character_id = db.add_character_card({"name": "Searcher"})
        authority = db.get_local_authority_id()
        db.add_conversation(
            {
                "id": "retained-chat",
                "title": "Needle",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
            }
        )
        service = CharacterConversationNavigationService(db)
        db.add_message(
            {
                "id": "retained-message",
                "conversation_id": "retained-chat",
                "sender": "User",
                "role": "user",
                "content": "Needle evidence",
            }
        )
        db.set_conversation_active_leaf("retained-chat", "retained-message")
        service.ensure_keyword_index()
        ready = service.keyword_search("Needle")
        assert ready.total == 1
        captured = ready.keyword_snapshot
        assert captured is not None
        db.add_character_card({"name": "Unrelated source change"})
        monkeypatch.setattr(
            CharacterConversationNavigationService,
            "ensure_keyword_index",
            lambda _self: None,
        )

        class _Roleplay(Screen):
            state = SimpleNamespace(
                active_mode="characters",
                runtime_source="local",
                selected_entity_kind="character",
                selected_entity_id=str(character_id),
            )

            def compose(self):
                yield PersonasInspectorPane()

            def _character_db(self):
                return db

        app = App()
        async with app.run_test(size=(120, 50)) as pilot:
            screen = _Roleplay()
            await app.push_screen(screen)
            controller = PersonasConversationsController(screen)
            controller._list_character_id = str(character_id)
            controller._conversation_query = "Needle"
            attempt = controller._claim_conversation_page(initial=True)
            inspector = screen.query_one(PersonasInspectorPane)
            inspector.show_selection(name="Searcher", kind="character")
            assert await inspector.show_conversations_loading(attempt)
            await asyncio.to_thread(
                controller._load_conversations_sync,
                str(character_id),
                None,
                True,
                attempt,
            )
            await pilot.pause()
            assert "retained-chat" in controller._conversation_rows
            assert controller._conversation_keyword_snapshot == captured
            request = controller._conversation_activation_requests["retained-chat"]
            assert (
                request.data_revision == db.get_character_conversation_search_revision()
            )
            assert request.data_revision > captured.source_revision
            header = screen.query_one("#personas-conversations-header", Static)
            assert "snapshot" in str(header.renderable).lower()
            assert captured.completed_at in str(header.renderable)
            assert header.display
            header.scroll_visible(immediate=True)
            await pilot.pause()
            painted = "\n".join(
                strip.text for strip in screen._compositor.render_strips()
            )
            assert "Keyword snapshot:" in painted
            assert captured.completed_at in painted
    finally:
        # This fixture owns both the caller and offloaded worker connections.
        # The app and awaited worker have settled before quiescing that owner.
        with db.quiesce_connections(timeout_seconds=2.0):
            pass
        assert db.registered_connection_count() == 0


@pytest.mark.parametrize("size", ((52, 20), (120, 50)))
async def test_real_textual_pilot_exposes_exact_draft_choices(size) -> None:
    """All aggregate draft choices remain reachable at required terminal sizes."""

    app = App()
    async with app.run_test(size=size) as pilot:
        for index, (selector, expected) in enumerate(
            (
                ("#roleplay-draft-save-continue", "save"),
                ("#roleplay-draft-discard-continue", "discard"),
                ("#roleplay-draft-stay", None),
            )
        ):
            results: list[str | None] = []
            await app.push_screen(
                RoleplayDraftNavigationDialog(
                    ("character form", "character visuals", "attachments")
                ),
                results.append,
            )
            await pilot.pause()
            if index == 0 and (qa_root := os.environ.get("TASK_31243_QA_DIR")):
                app.save_screenshot(
                    filename=f"roleplay-draft-{size[0]}x{size[1]}.svg",
                    path=qa_root,
                )
            await pilot.click(selector)
            await pilot.pause()
            assert results == [expected]


def test_keyword_search_is_limited_to_the_selected_exact_character(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "roleplay.sqlite", client_id="roleplay")
    try:
        first_id = db.add_character_card({"name": "First"})
        second_id = db.add_character_card({"name": "Second"})
        assert first_id and second_id
        authority = db.get_local_authority_id()
        for character_id, conversation_id in (
            (first_id, "first-chat"),
            (second_id, "second-chat"),
        ):
            assert db.add_conversation(
                {
                    "id": conversation_id,
                    "character_id": character_id,
                    "assistant_kind": "character",
                    "assistant_id": str(character_id),
                    "assistant_authority_id": authority,
                    "title": "Shared searchable title",
                }
            )
            message_id = f"message-{conversation_id}"
            assert (
                db.add_message(
                    {
                        "id": message_id,
                        "conversation_id": conversation_id,
                        "sender": "user",
                        "role": "user",
                        "content": "selected branch body",
                    }
                )
                == message_id
            )
            db.set_conversation_active_leaf(conversation_id, message_id)
        service = CharacterConversationNavigationService(db)
        service.ensure_keyword_index()

        page = service.keyword_search(
            "Shared searchable title",
            character=ResolvedLocalCharacterKey(authority, first_id),
            limit=20,
        )

        assert page.total == 1
        assert [row.target.conversation_id for row in page.rows if row.target] == [
            "first-chat"
        ]
    finally:
        db.close_connection()


def test_view_all_uses_twenty_row_keyset_pages_for_complete_history(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "roleplay-keyset.sqlite", client_id="roleplay")
    try:
        character_id = db.add_character_card({"name": "Historian"})
        assert character_id
        authority = db.get_local_authority_id()
        for index in range(25):
            assert db.add_conversation(
                {
                    "id": f"history-{index:02d}",
                    "character_id": character_id,
                    "assistant_kind": "character",
                    "assistant_id": str(character_id),
                    "assistant_authority_id": authority,
                    "title": f"History {index:02d}",
                }
            )
        service = CharacterConversationNavigationService(db)
        key = ResolvedLocalCharacterKey(authority, character_id)

        first = service.page_for_character(key, limit=20)
        second = service.page_for_character(key, cursor=first.next_cursor, limit=20)

        assert len(first.rows) == 20
        assert len(second.rows) == 5
        assert first.next_cursor is not None
        assert second.next_cursor is None
        assert len({row.row_key for row in first.rows + second.rows}) == 25
    finally:
        db.close_connection()


def test_keyword_controller_exposes_stable_twenty_row_continuation(
    tmp_path: Path,
) -> None:
    """A 25-match Keyword result paints 20, then 5 without repeats or skips."""

    db = CharactersRAGDB(tmp_path / "roleplay-keyword-pages.sqlite", client_id="ui")
    try:
        character_id = db.add_character_card({"name": "Searcher"})
        assert character_id
        authority = db.get_local_authority_id()
        for index in range(25):
            conversation_id = f"keyword-{index:02d}"
            assert db.add_conversation(
                {
                    "id": conversation_id,
                    "character_id": character_id,
                    "assistant_kind": "character",
                    "assistant_id": str(character_id),
                    "assistant_authority_id": authority,
                    "title": f"Needle result {index:02d}",
                }
            )
            message_id = f"message-{index:02d}"
            assert (
                db.add_message(
                    {
                        "id": message_id,
                        "conversation_id": conversation_id,
                        "sender": "User",
                        "role": "user",
                        "content": "needle transcript",
                    }
                )
                == message_id
            )
            db.set_conversation_active_leaf(conversation_id, message_id)

        CharacterConversationNavigationService(db).ensure_keyword_index()
        calls: list[tuple] = []
        screen = SimpleNamespace(
            _character_db=lambda: db,
            app=SimpleNamespace(
                call_from_thread=lambda callback, *args: calls.append(args)
            ),
        )
        controller = PersonasConversationsController(screen)
        controller._conversation_query = "Needle"
        first_attempt = object()
        controller._load_conversations_sync(
            str(character_id), None, True, first_attempt
        )
        first_records = calls[-1][4]
        assert len(first_records) == 21
        assert first_records[-1] == {"_page_sentinel": True}
        assert all(
            record["target"].character.data_authority_id == authority
            and record["data_revision"] >= 0
            for record in first_records[:-1]
        )
        with db.transaction() as connection:
            source_created = dict(
                connection.execute(
                    "SELECT id, CAST(created_at AS TEXT) FROM conversations"
                ).fetchall()
            )
        assert all(
            record.get("created_at") == source_created[record["id"]]
            for record in first_records[:-1]
        )
        assert isinstance(calls[-1][6], CharacterKeywordSnapshot)

        second_attempt = object()
        controller._load_conversations_sync(
            str(character_id), 20, False, second_attempt
        )
        second_records = calls[-1][4]
        first_ids = {record["id"] for record in first_records[:-1]}
        second_ids = {record["id"] for record in second_records}
        assert len(second_records) == 5
        assert len(first_ids | second_ids) == 25
        assert not first_ids & second_ids
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_keyword_mutation_between_pages_restarts_one_bound_generation(
    tmp_path: Path,
) -> None:
    """A rank mutation restarts paging, then yields each current row once."""

    db = CharactersRAGDB(tmp_path / "keyword-mutation.sqlite", client_id="ui")
    try:
        character_id = db.add_character_card({"name": "Searcher"})
        authority = db.get_local_authority_id()
        for index in range(25):
            conversation_id = f"stable-{index:02d}"
            db.add_conversation(
                {
                    "id": conversation_id,
                    "character_id": character_id,
                    "assistant_kind": "character",
                    "assistant_id": str(character_id),
                    "assistant_authority_id": authority,
                    "title": f"Needle {index:02d}",
                }
            )
            message_id = f"stable-message-{index:02d}"
            db.add_message(
                {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "sender": "User",
                    "role": "user",
                    "content": "needle transcript",
                }
            )
            db.set_conversation_active_leaf(conversation_id, message_id)

        class _Inspector:
            def __init__(self):
                self.visible: list[tuple[str, str]] = []

            async def show_conversations_loading(self, _attempt):
                return True

            async def show_older_conversations_loading(self, _attempt):
                return True

            async def show_conversations(self, rows, **_kwargs):
                self.visible = list(rows)
                return True

            async def append_conversations(self, rows, **_kwargs):
                self.visible.extend(rows)
                return True

            def set_conversation_total(self, _total):
                return None

            def set_conversation_snapshot(self, _completed_at):
                return None

        inspector = _Inspector()
        callbacks = []
        queued = []

        class _Screen:
            is_mounted = True
            state = SimpleNamespace(
                active_mode="characters",
                runtime_source="local",
                selected_entity_kind="character",
                selected_entity_id=str(character_id),
            )
            app = SimpleNamespace(
                call_from_thread=lambda callback, *args: callbacks.append(
                    (callback, args)
                )
            )

            def _character_db(self):
                return db

            def query_one(self, query, *_args):
                if query is PersonasInspectorPane:
                    return inspector
                raise QueryError("not mounted")

            def run_worker(self, work, **_kwargs):
                queued.append(work)

        controller = PersonasConversationsController(_Screen())
        controller._conversation_query = "Needle"
        controller._list_character_id = str(character_id)
        first_attempt = controller._claim_conversation_page(initial=True)
        controller._load_conversations_sync(
            str(character_id), None, True, first_attempt
        )
        callback, args = callbacks.pop()
        await callback(*args)
        first_revision = controller._conversation_page_revision
        assert len(inspector.visible) == 20

        db.add_conversation(
            {
                "id": "new-ranked-row",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
                "title": "Needle new",
            }
        )
        db.add_message(
            {
                "id": "new-ranked-message",
                "conversation_id": "new-ranked-row",
                "sender": "User",
                "role": "user",
                "content": "needle transcript",
            }
        )
        db.set_conversation_active_leaf("new-ranked-row", "new-ranked-message")
        second_attempt = controller._claim_conversation_page(initial=False)
        controller._load_conversations_sync(
            str(character_id), 20, False, second_attempt
        )
        callback, args = callbacks.pop()
        await callback(*args)

        assert controller._conversation_page_revision is None
        assert controller._conversation_rows == {}
        assert queued, "revision mismatch must schedule a fresh page one"
        queued.pop(0)()
        callback, args = callbacks.pop()
        await callback(*args)
        assert controller._conversation_page_revision != first_revision
        await controller.request_older_conversations()
        queued.pop(0)()
        callback, args = callbacks.pop()
        await callback(*args)

        ids = [conversation_id for conversation_id, _title in inspector.visible]
        assert len(ids) == 26
        assert len(set(ids)) == 26
        assert {f"stable-{index:02d}" for index in range(25)} <= set(ids)
        assert "new-ranked-row" in ids
    finally:
        db.close_connection()


def test_preview_revalidates_typed_row_before_reading_transcript(
    tmp_path: Path,
) -> None:
    """A moved conversation becomes unavailable without exposing its transcript."""

    db = CharactersRAGDB(tmp_path / "roleplay-preview-fence.sqlite", client_id="ui")
    try:
        first_id = db.add_character_card({"name": "First"})
        second_id = db.add_character_card({"name": "Second"})
        assert first_id and second_id
        authority = db.get_local_authority_id()
        assert db.add_conversation(
            {
                "id": "preview-target",
                "character_id": first_id,
                "assistant_kind": "character",
                "assistant_id": str(first_id),
                "assistant_authority_id": authority,
                "title": "Private transcript",
            }
        )
        assert (
            db.add_message(
                {
                    "id": "private-message",
                    "conversation_id": "preview-target",
                    "sender": "User",
                    "role": "user",
                    "content": "must not be disclosed after the link moves",
                }
            )
            == "private-message"
        )
        page = CharacterConversationNavigationService(db).page_for_character(
            ResolvedLocalCharacterKey(authority, first_id), limit=20
        )
        row = page.rows[0]
        assert row.target
        request_type = __import__(
            "tldw_chatbook.Chat.console_conversation_activation",
            fromlist=["CharacterConversationActivationRequest"],
        ).CharacterConversationActivationRequest
        request = request_type(row.target, authority, page.data_revision)
        record = db.get_conversation_by_id("preview-target")
        assert record
        assert db.update_conversation(
            "preview-target",
            {
                "character_id": second_id,
                "assistant_id": str(second_id),
                "assistant_authority_id": authority,
            },
            record["version"],
        )

        callbacks: list[tuple[str, tuple]] = []
        screen = SimpleNamespace(
            _character_db=lambda: db,
            app=SimpleNamespace(
                call_from_thread=lambda callback, *args: callbacks.append(
                    (callback.__name__, args)
                )
            ),
        )
        controller = PersonasConversationsController(screen)
        controller._conversation_activation_requests["preview-target"] = request

        controller._load_conversation_messages_sync(
            "preview-target", "First", object(), request
        )

        assert callbacks[0][0] == "show_conversation_unavailable"
        assert "must not be disclosed" not in repr(callbacks)
    finally:
        db.close_connection()


def test_queued_preview_keeps_immutable_request_when_search_resets_generation(
    tmp_path: Path,
) -> None:
    """A queued preview never falls back to an ID-only read after reload."""

    db = CharactersRAGDB(tmp_path / "queued-preview.sqlite", client_id="ui")
    try:
        first_id = db.add_character_card({"name": "First"})
        second_id = db.add_character_card({"name": "Second"})
        authority = db.get_local_authority_id()
        db.add_conversation(
            {
                "id": "queued-target",
                "character_id": first_id,
                "assistant_kind": "character",
                "assistant_id": str(first_id),
                "assistant_authority_id": authority,
                "title": "Queued",
            }
        )
        page = CharacterConversationNavigationService(db).page_for_character(
            ResolvedLocalCharacterKey(authority, first_id), limit=20
        )
        row = next(item for item in page.rows if item.target is not None)
        request_type = __import__(
            "tldw_chatbook.Chat.console_conversation_activation",
            fromlist=["CharacterConversationActivationRequest"],
        ).CharacterConversationActivationRequest
        request = request_type(row.target, authority, page.data_revision)
        queued = []
        callbacks = []

        class _Screen:
            app = SimpleNamespace(
                call_from_thread=lambda callback, *args: callbacks.append(
                    (callback.__name__, args)
                )
            )

            def _character_db(self):
                return db

            def run_worker(self, work, **_kwargs):
                queued.append(work)

            def query_one(self, *_args, **_kwargs):
                raise QueryError("not mounted")

        controller = PersonasConversationsController(_Screen())
        preview_attempt = object()
        controller._preview_attempt = preview_attempt
        controller.load_conversation_messages(
            "queued-target", "First", preview_attempt, request
        )
        controller._reset_conversation_browse(str(first_id))
        record = db.get_conversation_by_id("queued-target")
        assert record
        assert db.update_conversation(
            "queued-target",
            {
                "character_id": second_id,
                "assistant_id": str(second_id),
                "assistant_authority_id": authority,
            },
            record["version"],
        )

        queued[0]()

        assert callbacks[0][0] == "show_conversation_unavailable"
        assert callbacks[0][1][1] is preview_attempt
        assert controller._preview_attempt is None
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_failed_stale_revision_refreshes_results_before_retry(
    tmp_path: Path,
) -> None:
    """A generic stale-generation failure replaces the immutable row request."""

    db = CharactersRAGDB(tmp_path / "resume-refresh.sqlite", client_id="ui")
    try:
        character_id = db.add_character_card({"name": "First"})
        authority = db.get_local_authority_id()
        for conversation_id in ("resume-target", "unrelated"):
            assert db.add_conversation(
                {
                    "id": conversation_id,
                    "character_id": character_id,
                    "assistant_kind": "character",
                    "assistant_id": str(character_id),
                    "assistant_authority_id": authority,
                    "title": conversation_id,
                }
            )
        page = CharacterConversationNavigationService(db).page_for_character(
            ResolvedLocalCharacterKey(authority, character_id), limit=20
        )
        row = next(
            item
            for item in page.rows
            if item.target is not None
            and item.target.conversation_id == "resume-target"
        )
        stale_request = CharacterConversationActivationRequest(
            row.target, authority, page.data_revision
        )
        unrelated = db.get_conversation_by_id("unrelated")
        assert unrelated
        assert db.update_conversation(
            "unrelated", {"title": "mutated"}, unrelated["version"]
        )

        activation_requests = []

        class _ActivationApp:
            async def activate_character_conversation_from_roleplay(
                self, request, _cancellation, _phase_changed
            ):
                activation_requests.append(request)
                return ConsoleConversationActivationResult(
                    ConsoleActivationResultKind.FAILED, request.target, False
                )

        class _Inspector:
            async def show_conversations_loading(self, _attempt):
                return True

            async def show_conversations(self, *_args, **_kwargs):
                return True

            def set_conversation_total(self, _total):
                return None

            def focus_conversation(self, _conversation_id):
                return None

        queued = []
        callbacks = []
        notifications = []
        inspector = _Inspector()

        class _Screen:
            is_mounted = True
            app_instance = _ActivationApp()
            state = SimpleNamespace(
                active_mode="characters",
                runtime_source="local",
                selected_entity_kind="character",
                selected_entity_id=str(character_id),
                selected_entity_name="First",
            )
            app = SimpleNamespace(
                call_from_thread=lambda callback, *args: callbacks.append(
                    (callback, args)
                )
            )

            def _character_db(self):
                return db

            async def confirm_navigation(self):
                return True

            def query_one(self, selector, *_args, **_kwargs):
                if selector is PersonasInspectorPane:
                    return inspector
                raise QueryError("not mounted")

            def run_worker(self, work, **_kwargs):
                queued.append(work)

            def _notify(self, message, severity):
                notifications.append((message, severity))

        controller = PersonasConversationsController(_Screen())
        controller._list_character_id = str(character_id)
        controller._conversation_rows = {"resume-target": "Resume target"}
        controller._conversation_activation_requests = {"resume-target": stale_request}
        controller._open_character_id = str(character_id)
        controller._open_conversation_id = "resume-target"

        await controller._resume_in_console()

        assert activation_requests == [stale_request]
        assert queued, "FAILED recovery must start a fresh results generation"
        assert all(
            "character is unavailable" not in message for message, _ in notifications
        )
        queued.pop(0)()
        callback, args = callbacks.pop(0)
        await callback(*args)

        fresh_request = controller._conversation_activation_requests["resume-target"]
        assert fresh_request.data_revision != stale_request.data_revision
        assert (
            fresh_request.data_revision
            == db.get_character_conversation_search_revision()
        )
    finally:
        db.close_connection()
