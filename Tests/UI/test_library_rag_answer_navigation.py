"""Keyboard reading of generated answers and citation feedback with production CSS."""

import asyncio

import pytest
from textual.widgets import Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _StaticLibraryRagSearchService,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    _rag_result_fixture,
    _ready_library_rag_provider,  # noqa: F401 - register shared autouse fixture
    _switch_to_rag_mode,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("line_count", [2, 80])
@pytest.mark.parametrize(
    ("citation_status", "marker", "feedback"),
    [
        ("uncited", "", "The answer does not cite available staged evidence."),
        (
            "unverified",
            " [S99]",
            "Some citation markers do not match available staged evidence.",
        ),
        ("validated", " [S1]", "Citations resolve to staged evidence."),
    ],
)
async def test_keyboard_can_read_answer_and_citation_feedback(
    size, theme, line_count, citation_status, marker, feedback
):
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-42", "title": "Incident"}])
    service = _StaticLibraryRagSearchService(_rag_result_fixture())
    app.library_rag_search_service = service
    lines = [
        f"Answer line {i:03d}: an expired credential caused the incident."
        for i in range(1, line_count + 1)
    ]
    reply = "\n".join(lines) + marker
    calls = []

    def answer(**kwargs):
        calls.append(kwargs)
        return reply

    app.library_rag_answer_chat = answer
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _switch_to_rag_mode(screen, pilot)
        field = screen.query_one("#library-rag-query-input", Input)
        query = "Why did the incident happen?"
        field.value = query
        await _wait_for_library_rag_query_ready(screen, pilot, query)
        field.focus()
        await pilot.pause()
        scope = screen._library_rag_panel_state().scope.selected_source_types
        await pilot.press("enter")
        feedback_id = (
            "#library-rag-answer-citation-note"
            if citation_status == "validated"
            else "#library-rag-answer-caution"
        )
        await _wait_for_selector(screen, pilot, feedback_id)
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        state = screen._rag_search_state.answer
        assert state.citation_status == citation_status
        assert state.text == reply
        assert screen.focused is field
        panel = screen.query_one("#library-search-rag-panel")

        def painted_rows(selector):
            # Reassemble actual painted rows by their offset in the widget. A
            # wrapped sentence may straddle two pages; it needn't fit one frame.
            widget_region = screen.query_one(selector).content_region
            region = widget_region.intersection(panel.content_region)
            strips = screen._compositor.render_strips()
            return {
                y - widget_region.y: strips[y].crop(region.x, region.right).text
                for y in range(max(0, region.y), min(region.bottom, len(strips)))
            }

        def reading(rows):
            return " ".join(" ".join(rows[y] for y in sorted(rows)).split())

        for _ in range(15):
            await pilot.press("tab")
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            if getattr(screen.focused, "id", None) == "library-rag-result-card-0":
                break
        card = screen.query_one("#library-rag-result-card-0")
        assert screen.focused is card

        for key, edge in (("pageup", 0), ("pagedown", panel.max_scroll_y)):
            answer_rows, feedback_rows = {}, {}
            for _ in range(40):
                answer_rows.update(painted_rows("#library-rag-answer-text"))
                feedback_rows.update(painted_rows(feedback_id))
                if panel.scroll_y == edge:
                    break
                await pilot.press(key)
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
            assert panel.scroll_y == edge, (key, panel.scroll_y, edge)
            assert reading(answer_rows) == " ".join(reply.split()), (
                key,
                answer_rows,
            )
            assert reading(feedback_rows) == feedback, (key, feedback_rows)
            assert screen.focused is card

        # Scrolling never submits again or changes the answer/input/source choices.
        assert field.value == query
        assert screen._library_rag_panel_state().scope.selected_source_types == scope
        assert screen._rag_search_state.answer is state
        assert state.text == reply
        assert len(service.calls) == len(calls) == 1
        # Tab still reaches the evidence action immediately following the card.
        await pilot.press("tab")
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen.focused is not card
        assert screen.focused in card.walk_children()
        assert reading(painted_rows(f"#{screen.focused.id}"))
