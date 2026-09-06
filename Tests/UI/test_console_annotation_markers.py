"""Inline annotation markers on transcript rows (task-17169 slice 2).

The both-homes decision's visible half: a Comment that persisted a
transcript_annotations row renders as a marker row under the anchored
message, mirroring the citation-sources sub-row mechanism (screen-owned
map keyed by NATIVE message id, pushed at the sync tick, derived into a
sub-row by ``_transcript_rows``).
"""

from __future__ import annotations

import pytest
from textual.app import ComposeResult

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_left_rail import make_console_pilot
from Tests.UI.test_console_selection_end_to_end import (
    _RecordingPromptQueue,
    _RecordingStore,
    _run_feedback_request,
    _stub_feedback_store,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleReviewNotesRequested,
    ConsoleTranscript,
)


def _message(message_id: str, role=ConsoleMessageRole.ASSISTANT) -> ConsoleChatMessage:
    return ConsoleChatMessage(role=role, content="body", id=message_id)


def _all_rows(transcript: ConsoleTranscript) -> list[object]:
    """Return top-level and recursively nested transcript rows."""
    rows: list[object] = []

    def visit(row: object) -> None:
        rows.append(row)
        for nested in row.nested_rows:  # type: ignore[attr-defined]
            visit(nested)

    for row in transcript._transcript_rows():
        visit(row)
    return rows


def test_annotated_message_gains_a_marker_row() -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages([_message("m1"), _message("m2")])
    transcript.set_annotation_previews({"m1": ("tighten error paths",)})

    rows = [row for row in _all_rows(transcript) if row.kind == "annotations"]
    assert [row.message.id for row in rows] == ["m1"]
    rendered = str(rows[0].renderable)
    assert "tighten error paths" in rendered


def test_marker_row_lists_every_note_in_order() -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages([_message("m1")])
    transcript.set_annotation_previews({"m1": ("first pass", "second pass")})

    (row,) = [r for r in _all_rows(transcript) if r.kind == "annotations"]
    rendered = str(row.renderable)
    assert rendered.index("first pass") < rendered.index("second pass")


def test_no_previews_means_no_marker_rows() -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages([_message("m1")])

    assert [r for r in transcript._transcript_rows() if r.kind == "annotations"] == []


def test_setter_drops_invalid_entries() -> None:
    transcript = ConsoleTranscript()
    transcript.set_annotation_previews(
        {"m1": (), "": ("x",), "m2": ("kept",), None: ("y",)}  # type: ignore[dict-item]
    )
    assert transcript._annotation_previews == {"m2": ("kept",)}


def test_marker_signature_changes_when_notes_change() -> None:
    """The row cache is signature-keyed: an edited or added note must produce
    a different signature or the mounted marker silently goes stale."""
    transcript = ConsoleTranscript()
    transcript.set_messages([_message("m1")])
    transcript.set_annotation_previews({"m1": ("v1",)})
    (before,) = [r for r in _all_rows(transcript) if r.kind == "annotations"]
    transcript.set_annotation_previews({"m1": ("v1", "v2")})
    (after,) = [r for r in _all_rows(transcript) if r.kind == "annotations"]
    assert before.signature != after.signature


def test_marker_widget_is_a_static_keyed_to_the_message() -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages([_message("m1")])
    transcript.set_annotation_previews({"m1": ("note",)})

    (row,) = [r for r in _all_rows(transcript) if r.kind == "annotations"]
    widget = transcript._build_row_widget(row, track=False)
    assert widget.id == "console-annotations-m1"
    assert widget.has_class("console-transcript-annotations")


# ---------------------------------------------------------------------------
# Screen wiring: live updates and restore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_comment_updates_the_screen_preview_map():
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore()
        _stub_feedback_store(screen, store)

        await _run_feedback_request(
            pilot,
            action="comment",
            quote="q",
            comment="tighten error paths",
            anchor_message_id="msg-42",
        )

        assert screen._console_annotation_previews == {
            "msg-42": ("tighten error paths",)
        }


@pytest.mark.asyncio
async def test_failed_annotation_write_leaves_the_preview_map_alone():
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = _RecordingStore(result=False)
        _stub_feedback_store(screen, store)

        await _run_feedback_request(
            pilot,
            action="comment",
            quote="q",
            comment="note",
            anchor_message_id="msg-42",
        )

        assert screen._console_annotation_previews == {}


@pytest.mark.asyncio
async def test_restore_rekeys_persisted_annotations_to_native_ids(tmp_path):
    """The restore half, unmocked: annotations written in a previous life of
    the conversation come back keyed to the CURRENT native message ids."""
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "marker_restore")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            screen._prompt_queue = _RecordingPromptQueue()
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    db,
                    workspace_registry=screen.app_instance.workspace_registry_service,
                )
            )
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            controller.store = store

            session = store.ensure_session(title="Restore markers")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="ok",
                persist=True,
            )
            db.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{assistant.persisted_message_id}",
                message_id=assistant.persisted_message_id,
                quote_text="ok",
                comment="from a previous run",
            )

            screen._review_selection._sync_console_annotation_discovery(store)
            for _ in range(50):
                await pilot.pause()
                if screen._console_annotation_previews:
                    break

            assert screen._console_annotation_previews == {
                assistant.id: ("from a previous run",)
            }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Task 1: clickable marker + `n` action (task-18515 review-note management)
# ---------------------------------------------------------------------------


class _ReviewNotesTranscriptApp(ConsolidatedCSSApp):
    """One-message transcript harness with app-level request capture.

    ``previews`` is set on the instance *before* ``run_test()`` enters (i.e.
    before ``compose()`` runs at mount) so each test can opt a message into
    having notes without a bespoke App subclass -- the same shape as
    ``_FeedbackTranscriptApp`` in ``test_console_selection_end_to_end.py``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.previews: dict[str, tuple[str, ...]] = {}
        self.review_notes_events: list[ConsoleReviewNotesRequested] = []

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages([_message("m1")])
        if self.previews:
            transcript.set_annotation_previews(self.previews)
        yield transcript

    # Module-level Message classes carry no widget namespace, so the
    # auto-generated handler is ``on_console_review_notes_requested``
    # (matching how ``on_console_selection_note_requested`` works above).
    def on_console_review_notes_requested(
        self, event: ConsoleReviewNotesRequested
    ) -> None:
        self.review_notes_events.append(event)


@pytest.mark.asyncio
async def test_marker_click_requests_notes_not_message_toggle():
    app = _ReviewNotesTranscriptApp()
    app.previews = {"m1": ("note",)}

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        # Pre-select the message so a legacy toggle would be visible as a
        # deselect -- the marker click must leave it alone either way.
        await pilot.click("#console-message-m1")
        await pilot.pause()
        assert transcript.selected_message_id == "m1"

        await pilot.click("#console-annotations-m1")
        await pilot.pause()

        assert [event.anchor_message_id for event in app.review_notes_events] == ["m1"]
        assert transcript.selected_message_id == "m1"


@pytest.mark.asyncio
async def test_n_on_selected_noted_message_requests_notes():
    app = _ReviewNotesTranscriptApp()
    app.previews = {"m1": ("note",)}

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.focus()
        await pilot.click("#console-message-m1")
        await pilot.pause()
        assert transcript.selected_message_id == "m1"

        await pilot.press("n")
        await pilot.pause()

        assert [event.anchor_message_id for event in app.review_notes_events] == ["m1"]


@pytest.mark.asyncio
async def test_n_without_notes_toasts_and_requests_nothing():
    app = _ReviewNotesTranscriptApp()

    async with app.run_test(size=(100, 32)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        notifications: list[tuple[str, str]] = []
        app.notify = lambda message, *, severity="information", **kwargs: (
            notifications.append((message, severity))
        )
        transcript.focus()
        await pilot.click("#console-message-m1")
        await pilot.pause()
        assert transcript.selected_message_id == "m1"

        await pilot.press("n")
        await pilot.pause()

        assert app.review_notes_events == []
        assert len(notifications) == 1
        assert notifications[0][1] == "warning"


# ---------------------------------------------------------------------------
# Task 3: screen wiring + unmocked round trips (task-18515 review-note
# management)
# ---------------------------------------------------------------------------


def _stub_review_notes_modal(screen, resolver):
    """Replace ``app.push_screen_wait`` and capture every pushed modal.

    ``resolver(modal)`` is AWAITED with the REAL ``ConsoleReviewNotesModal``
    the flow built (its ``on_edit``/``on_delete`` are the flow's real
    closures, bound to the real DB) and returns the dismiss result the flow
    should see. The callables are async since the DB writes moved off the UI
    event loop, so resolvers are coroutines.
    """
    pushed: list = []

    async def _resolve(modal, *args, **kwargs):
        pushed.append(modal)
        return await resolver(modal)

    screen.app.push_screen_wait = _resolve  # type: ignore[method-assign]
    return pushed


async def _wait_until(pilot, predicate, attempts: int = 40) -> None:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return
    # Final pause so the last state change (if any) has settled before the
    # caller's own assertion runs and produces a readable failure.
    await pilot.pause()


@pytest.mark.asyncio
async def test_edit_then_delete_round_trip_pins_the_sidecar_row(tmp_path):
    """Unmocked screen->store->SQLite: an edit changes only comment/updated_at
    (quote/row_key byte-identical), a delete soft-deletes the row, and a
    ``user_feedback`` trajectory row written alongside the annotation is
    byte-identical after BOTH operations -- the annotation and sidecar
    tables must never touch each other."""
    import json

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, TrajectoryRowWrite

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "review_notes_e2e")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    db,
                    workspace_registry=screen.app_instance.workspace_registry_service,
                )
            )
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            controller.store = store

            session = store.ensure_session(title="Review notes e2e")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="the assistant's answer",
                persist=True,
            )

            annotation_id = db.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{assistant.persisted_message_id}",
                message_id=assistant.persisted_message_id,
                quote_text="the assistant's answer",
                comment="original comment",
            )
            original = db.get_transcript_annotations(conversation_id)[0]

            # ``append_message(persist=True)`` already wrote its own
            # ``assistant`` trajectory row for this message; the
            # ``user_feedback`` row added here is the one this test pins --
            # identified by event_kind, not by "the only row in the table".
            db.upsert_trajectory_rows(
                [
                    TrajectoryRowWrite(
                        message_id=assistant.persisted_message_id,
                        conversation_id=conversation_id,
                        turn_id=assistant.persisted_message_id,
                        seq=None,
                        event_kind="user_feedback",
                        payload_json=json.dumps(
                            {"action": "comment", "quote": "the assistant's answer"}
                        ),
                    )
                ]
            )

            def _feedback_row(rows):
                matches = [row for row in rows if row.event_kind == "user_feedback"]
                assert len(matches) == 1
                return matches[0]

            sidecar_before = _feedback_row(db.get_trajectory_rows(conversation_id))

            # Edit and delete both run inside the resolver
            # (mirroring a real edit-then-delete session before the modal
            # closes), so the intermediate "after edit, before delete"
            # state has to be captured HERE -- by the time ``pushed`` is
            # observable from outside ``push_screen_wait``, both mutations
            # have already happened.
            snapshots: dict[str, object] = {}

            async def _resolver(modal) -> bool:
                snapshots["edit_ok"] = await modal._on_edit(
                    annotation_id, "edited comment"
                )
                snapshots["after_edit_annotations"] = db.get_transcript_annotations(
                    conversation_id
                )
                snapshots["after_edit_sidecar"] = _feedback_row(
                    db.get_trajectory_rows(conversation_id)
                )
                snapshots["delete_ok"] = await modal._on_delete(annotation_id)
                snapshots["after_delete_annotations"] = db.get_transcript_annotations(
                    conversation_id
                )
                snapshots["after_delete_sidecar"] = _feedback_row(
                    db.get_trajectory_rows(conversation_id)
                )
                return True

            pushed = _stub_review_notes_modal(screen, _resolver)
            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            await _wait_until(pilot, lambda: len(pushed) == 1)

            assert pushed, "expected the review-notes modal to be pushed"
            assert snapshots["edit_ok"] is True
            assert snapshots["delete_ok"] is True

            # Edit: comment + updated_at changed, everything else identical.
            after_edit = snapshots["after_edit_annotations"]
            assert len(after_edit) == 1
            edited = after_edit[0]
            assert edited["comment"] == "edited comment"
            assert edited["updated_at"] != original["updated_at"]
            for key in (
                "annotation_id",
                "conversation_id",
                "row_key",
                "message_id",
                "quote_text",
                "created_at",
            ):
                assert edited[key] == original[key], key

            # Sidecar row is byte-identical after the edit.
            assert snapshots["after_edit_sidecar"] == sidecar_before

            # Delete: soft-deleted, no longer returned by the live read.
            assert snapshots["after_delete_annotations"] == []

            # Sidecar row is STILL byte-identical after the delete.
            assert snapshots["after_delete_sidecar"] == sidecar_before
    finally:
        db.close()


@pytest.mark.asyncio
async def test_delete_of_last_note_removes_the_marker_after_forced_reload(tmp_path):
    """Deleting the only note on a message clears its inline marker: the
    screen's forced reload (``_console_annotation_loaded_conversation =
    None`` + an immediate re-sync) must actually run after a change."""
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "review_notes_delete")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    db,
                    workspace_registry=screen.app_instance.workspace_registry_service,
                )
            )
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            controller.store = store

            session = store.ensure_session(title="Review notes delete")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="ok",
                persist=True,
            )
            annotation_id = db.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{assistant.persisted_message_id}",
                message_id=assistant.persisted_message_id,
                quote_text="ok",
                comment="only note",
            )

            screen._review_selection._sync_console_annotation_discovery(store)
            await _wait_until(pilot, lambda: bool(screen._console_annotation_previews))
            assert screen._console_annotation_previews == {assistant.id: ("only note",)}

            async def _resolver(modal) -> bool:
                assert await modal._on_delete(annotation_id) is True
                return True

            pushed = _stub_review_notes_modal(screen, _resolver)
            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            await _wait_until(
                pilot, lambda: assistant.id not in screen._console_annotation_previews
            )

            assert pushed
            assert assistant.id not in screen._console_annotation_previews
            assert db.get_transcript_annotations(conversation_id) == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_no_review_notes_for_message_toasts_and_never_opens_modal(tmp_path):
    """An empty match (message persisted, but no live annotation rows) toasts
    a warning and never reaches ``push_screen_wait``."""
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "review_notes_empty")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    db,
                    workspace_registry=screen.app_instance.workspace_registry_service,
                )
            )
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            controller.store = store

            session = store.ensure_session(title="Review notes empty")
            store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="no notes here",
                persist=True,
            )

            calls: list = []

            async def _spy(modal, *args, **kwargs):
                calls.append(modal)
                return False

            screen.app.push_screen_wait = _spy  # type: ignore[method-assign]
            toasts: list = []
            screen.notify = lambda *a, **k: toasts.append((a, k))  # type: ignore[method-assign]

            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            await _wait_until(pilot, lambda: bool(toasts))

            assert calls == []
            assert toasts, "expected a warning toast when there are no review notes"
            assert toasts[-1][1].get("severity") == "warning"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_on_edit_and_on_delete_never_raise_on_db_failure(tmp_path):
    """The DB wrappers log and return False instead of propagating -- a
    broken write must never crash the worker (``exit_on_error=False`` is the
    backstop, this is the first line of defense)."""
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "review_notes_boom")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    db,
                    workspace_registry=screen.app_instance.workspace_registry_service,
                )
            )
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            controller.store = store

            session = store.ensure_session(title="Review notes boom")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="ok",
                persist=True,
            )
            annotation_id = db.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{assistant.persisted_message_id}",
                message_id=assistant.persisted_message_id,
                quote_text="ok",
                comment="will fail to edit",
            )

            def _boom_edit(**kwargs):
                raise RuntimeError("upsert boom")

            def _boom_delete(_annotation_id):
                raise RuntimeError("delete boom")

            db.upsert_transcript_annotation = _boom_edit  # type: ignore[method-assign]
            db.soft_delete_transcript_annotation = _boom_delete  # type: ignore[method-assign]

            results: list = []

            async def _resolver(modal) -> bool:
                results.append(await modal._on_edit(annotation_id, "new text"))
                results.append(await modal._on_delete(annotation_id))
                return False

            pushed = _stub_review_notes_modal(screen, _resolver)
            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            await _wait_until(pilot, lambda: len(pushed) == 1)

            assert pushed
            assert results == [False, False]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_double_trigger_pushes_exactly_one_modal_and_reads_once(tmp_path):
    """Rapid double marker-click / double-`n` on the same anchor, before the
    first worker's off-thread DB read resolves, must not stack two
    ConsoleReviewNotesModals with independent DB-bound closures -- the
    handler's inflight latch (mirroring ``_console_selection_feedback_
    inflight``) makes the second request a no-op, not a second flow."""
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "review_notes_double")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    db,
                    workspace_registry=screen.app_instance.workspace_registry_service,
                )
            )
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            controller.store = store

            session = store.ensure_session(title="Review notes double")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="ok",
                persist=True,
            )
            db.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{assistant.persisted_message_id}",
                message_id=assistant.persisted_message_id,
                quote_text="ok",
                comment="only note",
            )

            fetch_calls: list = []
            real_get = db.get_transcript_annotations

            def _counting_get(conv_id, message_id=None):
                fetch_calls.append((conv_id, message_id))
                return real_get(conv_id, message_id)

            db.get_transcript_annotations = _counting_get  # type: ignore[method-assign]

            async def _dismiss_without_changes(_modal) -> bool:
                return False

            pushed = _stub_review_notes_modal(screen, _dismiss_without_changes)

            # Both posted before any pause: the second handler must observe
            # the latch the first one set, entirely synchronously -- no
            # thread-timing race required for this to be deterministic.
            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            await _wait_until(pilot, lambda: len(pushed) == 1)
            # A few extra pauses: if the guard were absent, a second worker
            # would still eventually reach push_screen_wait -- give it every
            # chance to show up before asserting it didn't.
            for _ in range(10):
                await pilot.pause()

            assert len(pushed) == 1
            assert len(fetch_calls) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_inflight_latch_releases_after_the_modal_closes(tmp_path):
    """A latched-forever flag would silently kill the feature after its
    first use -- a request made once the first flow has fully finished
    (modal dismissed) must open a fresh modal, not be swallowed."""
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "review_notes_release")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    db,
                    workspace_registry=screen.app_instance.workspace_registry_service,
                )
            )
            screen._console_chat_store = store
            controller = screen._ensure_console_chat_controller()
            controller.store = store

            session = store.ensure_session(title="Review notes release")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="ok",
                persist=True,
            )
            db.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{assistant.persisted_message_id}",
                message_id=assistant.persisted_message_id,
                quote_text="ok",
                comment="only note",
            )

            pushed = _stub_review_notes_modal(screen, lambda _modal: False)

            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            await _wait_until(pilot, lambda: len(pushed) == 1)
            assert not screen._console_review_notes_inflight, (
                "flag should be released once the first flow's finally runs"
            )

            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            await _wait_until(pilot, lambda: len(pushed) == 2)

            assert len(pushed) == 2
    finally:
        db.close()


@pytest.mark.asyncio
async def test_annotation_previews_join_the_transcript_refresh_key():
    """Live-verification catch (task-18515): the marker survived deleting
    its last note because the refresh key ignored annotation previews, so
    nothing re-rendered while the app was idle. Phase 4 only appeared to
    work because writing a note also dispatches a message (starting a run,
    whose sync ticks refresh anyway)."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        await screen._sync_native_console_transcript()
        await pilot.pause()
        before = screen._last_native_transcript_refresh_key
        assert before is not None

        screen._console_annotation_previews = {"m-any": ("a note",)}
        await screen._sync_native_console_transcript()
        await pilot.pause()

        assert screen._last_native_transcript_refresh_key != before, (
            "an annotation-preview change must invalidate the refresh key"
        )


def test_marker_caps_the_notes_it_lists() -> None:
    """Review finding: the marker rendered one line per note with no cap, so
    a heavily-annotated message grew an unbounded inline row. The modal
    scrolls; the transcript row cannot."""
    from tldw_chatbook.Widgets.Console.console_transcript import (
        _annotation_marker_content,
    )

    notes = tuple(f"note {i}" for i in range(9))
    rendered = str(_annotation_marker_content(notes))

    assert "Review notes (9)" in rendered
    assert rendered.count("\n") <= 4, rendered  # header + 3 notes + overflow
    assert "note 0" in rendered and "note 2" in rendered
    assert "note 3" not in rendered
    assert "+6 more" in rendered


@pytest.mark.asyncio
async def test_oversized_note_edit_is_refused(tmp_path):
    """Qodo PR #1820: the edited comment passes the shared validation
    boundary (like the create-note path) before it can reach storage."""
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "chachanotes.sqlite"), "review_notes_validate")
    try:
        async with make_console_pilot() as pilot:
            screen = pilot.app.screen
            store = ConsoleChatStore(
                persistence=ChatPersistenceService(
                    db,
                    workspace_registry=screen.app_instance.workspace_registry_service,
                )
            )
            screen._console_chat_store = store
            screen._ensure_console_chat_controller().store = store

            session = store.ensure_session(title="Review notes validate")
            conversation_id = store.persist_session_if_needed(session.id)
            assistant = store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="ok",
                persist=True,
            )
            db.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{assistant.persisted_message_id}",
                message_id=assistant.persisted_message_id,
                quote_text="ok",
                comment="short note",
            )

            captured: dict = {}

            async def _resolver(modal) -> bool:
                annotation_id = modal._order[0]
                captured["ok"] = await modal._on_edit(annotation_id, "x" * 40_000)
                return False

            pushed = _stub_review_notes_modal(screen, _resolver)
            screen.post_message(
                ConsoleReviewNotesRequested(anchor_message_id=assistant.id)
            )
            await _wait_until(pilot, lambda: bool(pushed))

            assert captured["ok"] is False
            rows = db.get_transcript_annotations(conversation_id)
            assert [row["comment"] for row in rows] == ["short note"]
    finally:
        db.close()
