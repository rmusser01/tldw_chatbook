"""Selected-turn Library activity review contracts."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_library_activity_buffer import (
    LIBRARY_ACTIVITY_NOT_SAVED_COPY,
    LibraryActivityFlushResult,
)
from tldw_chatbook.Chat.library_activity import (
    LibraryActivityEvent,
    LibraryActivityRecord,
    LibraryActivitySourceRef,
    LibraryActivityView,
)
from tldw_chatbook.UI.Console_Modules.right_rail import (
    ConsoleSelectedTurnActivity,
)


def _view() -> LibraryActivityView:
    event = LibraryActivityEvent(
        version=1,
        event_id="event-1",
        attempt_id="attempt-1",
        run_id="run-child",
        actor_kind="subagent",
        parent_run_id="run-parent",
        library_provider="rag",
        operation="search_library_rag",
        status="succeeded",
        result_count=2,
        query_preview="bounded query",
        source_refs=(
            LibraryActivitySourceRef("note", "note-1", "[red] literal title"),
            LibraryActivitySourceRef("media", "media-2", "研究 🧪"),
        ),
        error_code=None,
        error_summary=None,
    )
    return LibraryActivityView(
        selected_turn_id="turn-1",
        actions=(
            LibraryActivityRecord(
                turn_id="turn-1",
                sequence=4,
                occurred_at=1_725_000_000.0,
                event=event,
            ),
        ),
    )


class _ActivityHarness(App):
    def __init__(
        self,
        view: LibraryActivityView,
        *,
        citation_count: int = 0,
        flush_result: LibraryActivityFlushResult | None = None,
    ) -> None:
        super().__init__()
        self._view = view
        self._citation_count = citation_count
        self._flush_result = flush_result

    def compose(self) -> ComposeResult:
        yield ConsoleSelectedTurnActivity(
            self._view,
            citation_count=self._citation_count,
            flush_result=self._flush_result,
        )


async def test_selected_turn_orders_citations_before_activity_and_renders_facts() -> None:
    app = _ActivityHarness(_view(), citation_count=3)

    async with app.run_test(size=(42, 20)):
        group = app.query_one("#console-selected-turn")
        direct_children = [child.id for child in group.children]
        statics = list(group.query(Static))
        copy = [str(item.renderable) for item in statics]

        assert direct_children.index("console-selected-turn-cited-sources") < (
            direct_children.index("console-selected-turn-library-activity")
        )
        assert "Cited sources (3)" in copy
        assert "Library activity (1 actions)" in copy
        assert any(
            "search_library_rag · subagent · RAG · succeeded · 2 results" in line
            for line in copy
        )
        assert any("note · [red] literal title · note-1" in line for line in copy)
        assert any("media · 研究 🧪 · media-2" in line for line in copy)
        literal = next(item for item in statics if "[red] literal title" in str(item.renderable))
        assert literal._render_markup is False


async def test_selected_turn_has_explicit_empty_activity_state() -> None:
    app = _ActivityHarness(
        LibraryActivityView(selected_turn_id="turn-1", actions=()),
        citation_count=0,
    )

    async with app.run_test():
        assert str(
            app.query_one("#console-library-activity-empty", Static).renderable
        ) == "No Library activity for this turn."
        assert str(
            app.query_one("#console-selected-turn-cited-sources", Static).renderable
        ) == "Cited sources (0)"


async def test_selected_turn_becomes_visible_when_selection_arrives() -> None:
    app = _ActivityHarness(LibraryActivityView(selected_turn_id=None, actions=()))

    async with app.run_test() as pilot:
        selected_turn = app.query_one(
            "#console-selected-turn", ConsoleSelectedTurnActivity
        )
        selected_turn.sync_state(_view(), citation_count=0, flush_result=None)
        await pilot.pause()

        assert selected_turn.styles.display == "block"


async def test_unsaved_activity_state_exposes_retry_without_owning_persistence() -> None:
    app = _ActivityHarness(
        _view(),
        flush_result=LibraryActivityFlushResult(
            "failed",
            saved_count=0,
            pending_count=1,
            error_code="retry_exhausted",
            warning=LIBRARY_ACTIVITY_NOT_SAVED_COPY,
        ),
    )

    async with app.run_test():
        warning = app.query_one("#console-library-activity-save-warning", Static)
        retry = app.query_one("#console-library-activity-retry", Button)

        assert str(warning.renderable) == LIBRARY_ACTIVITY_NOT_SAVED_COPY
        assert retry.label.plain == "Retry"
        assert retry.can_focus


async def test_eight_references_remain_inside_the_selected_turn_section() -> None:
    event = _view().actions[0].event
    refs = tuple(
        LibraryActivitySourceRef("note", f"note-{index}", f"Title {index}")
        for index in range(8)
    )
    view = LibraryActivityView(
        selected_turn_id="turn-1",
        actions=(
            LibraryActivityRecord(
                turn_id="turn-1",
                sequence=1,
                occurred_at=1.0,
                event=replace(event, source_refs=refs),
            ),
        ),
    )
    app = _ActivityHarness(view)

    async with app.run_test(size=(34, 10)):
        group = app.query_one("#console-selected-turn")
        assert len(group.query(".console-library-activity-source-ref")) == 8
        assert not app.query("#console-library-activity-top-level")


async def test_error_copy_is_literal_and_invalid_time_degrades_safely() -> None:
    event = replace(
        _view().actions[0].event,
        status="failed",
        error_code="storage_error",
        error_summary="[bold red]literal error[/]",
    )
    app = _ActivityHarness(
        LibraryActivityView(
            selected_turn_id="turn-1",
            actions=(
                LibraryActivityRecord(
                    turn_id="turn-1",
                    sequence=1,
                    occurred_at=1e300,
                    event=event,
                ),
            ),
        )
    )

    async with app.run_test():
        error = app.query_one("#console-library-activity-error-0", Static)
        action = app.query_one("#console-library-activity-action-0", Static)
        assert str(error.renderable) == "[bold red]literal error[/]"
        assert error._render_markup is False
        assert "time unavailable" in str(action.renderable)


async def test_message_affordance_selects_turn_opens_inspector_and_focuses_activity() -> None:
    from Tests.UI.test_console_right_rail import make_console_pilot
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        store = screen._ensure_console_chat_store()
        session_id = store.active_session_id
        assert session_id is not None
        user = store.append_message(
            session_id,
            role=ConsoleMessageRole.USER,
            content="Question",
        )
        assistant = store.append_message(
            session_id,
            role=ConsoleMessageRole.ASSISTANT,
            content="Answer",
        )
        store.admit_library_activity(session_id, user.id, _view().actions[0].event)
        screen._library_activity.invalidate_projection()
        await screen._sync_native_console_chat_ui()
        await pilot.pause()

        await pilot.click(f"#console-library-activity-{assistant.id}")
        await pilot.pause(0.05)
        await pilot.pause()

        transcript = screen.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        selected_turn = screen.query_one("#console-selected-turn")
        assert transcript.selected_message_id == assistant.id
        assert selected_turn.styles.display == "block"
        assert screen.query_one("#console-right-rail").display
        assert pilot.app.focused is not None
        assert (
            pilot.app.focused.id
            == "console-selected-turn-library-activity-heading"
        )


async def test_inspector_retry_delegates_to_store_owned_callback() -> None:
    from Tests.UI.test_console_right_rail import make_console_pilot

    called = asyncio.Event()

    async def retry() -> None:
        called.set()

    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        screen._reveal_console_inspector_rail()
        await pilot.pause()
        rail = screen.query_one("#console-right-rail")
        rail._library_activity_retry = retry
        rail.sync_library_activity(
            _view(),
            citation_count=0,
            flush_result=LibraryActivityFlushResult(
                "failed",
                saved_count=0,
                pending_count=1,
                error_code="retry_exhausted",
                warning=LIBRARY_ACTIVITY_NOT_SAVED_COPY,
            ),
        )
        await pilot.pause()
        retry_button = rail.query_one("#console-library-activity-retry", Button)

        retry_button.press()
        await asyncio.wait_for(called.wait(), timeout=1)

        assert called.is_set()
