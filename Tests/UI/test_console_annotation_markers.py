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


def test_annotated_message_gains_a_marker_row() -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages([_message("m1"), _message("m2")])
    transcript.set_annotation_previews({"m1": ("tighten error paths",)})

    rows = [row for row in transcript._transcript_rows() if row.kind == "annotations"]
    assert [row.message.id for row in rows] == ["m1"]
    rendered = str(rows[0].renderable)
    assert "tighten error paths" in rendered


def test_marker_row_lists_every_note_in_order() -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages([_message("m1")])
    transcript.set_annotation_previews({"m1": ("first pass", "second pass")})

    (row,) = [r for r in transcript._transcript_rows() if r.kind == "annotations"]
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
    (before,) = [r for r in transcript._transcript_rows() if r.kind == "annotations"]
    transcript.set_annotation_previews({"m1": ("v1", "v2")})
    (after,) = [r for r in transcript._transcript_rows() if r.kind == "annotations"]
    assert before.signature != after.signature


def test_marker_widget_is_a_static_keyed_to_the_message() -> None:
    transcript = ConsoleTranscript()
    transcript.set_messages([_message("m1")])
    transcript.set_annotation_previews({"m1": ("note",)})

    (row,) = [r for r in transcript._transcript_rows() if r.kind == "annotations"]
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
            store = ConsoleChatStore(persistence=ChatPersistenceService(db))
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

            screen._sync_console_annotation_discovery(store)
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

        assert [
            event.anchor_message_id for event in app.review_notes_events
        ] == ["m1"]
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

        assert [
            event.anchor_message_id for event in app.review_notes_events
        ] == ["m1"]


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
