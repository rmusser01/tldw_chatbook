"""Readable Collections controls and search recovery through real keyboard input."""

from dataclasses import replace

import pytest
from textual.widgets import Input, TextArea

from Tests.UI.test_library_collection_browse_journeys import _show_pane
from Tests.UI.test_library_collection_reader_journeys import _open, _seed
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import LibraryProductionCSSHarness, _wait_for_condition
from Tests.UI.test_library_skill_editor_journeys import _activate


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_compact_reader_tab_actions_remain_readable_and_operate(
    theme, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app, scope, _ = await _seed()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open(host, pilot)
        await _show_pane(screen, host, pilot, "items")
        engine = screen._library_collections_capture_controller
        identity = engine.state.selected_identity
        start = screen.query_one("#library-collections-mark-read")
        start.focus()
        await _focus(screen, host, pilot, "#library-collections-mark-read", "Mark Read")
        controls = (
            ("favorite", "Favorite"),
            ("archive", "Move to Archive"),
            ("open-original", "Open Original"),
            ("more", "More"),
            ("mode-read", "Read"),
            ("mode-highlights", "Highlights"),
            ("mode-notes", "Notes"),
            ("mode-info", "Info"),
        )
        for name, label in controls:
            await pilot.press("tab")
            await _focus(screen, host, pilot, f"#library-collections-{name}", label)
        for name, label in reversed(controls[:-1]):
            await pilot.press("shift+tab")
            await _focus(screen, host, pilot, f"#library-collections-{name}", label)
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: engine.state.loaded_detail.capture.favorite,
            message="Keyboard Favorite did not update the loaded capture",
        )
        assert (await scope.get_detail(identity)).capture.favorite
        assert engine.state.selected_identity == identity
        await _focus(screen, host, pilot, "#library-collections-favorite", "Favorite")


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_reader_resize_preserves_visible_mode_focus_and_note_draft(
    theme, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app, _, _ = await _seed()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        await _activate(screen, host, pilot, "#library-collections-mode-notes", "Notes")
        note = screen.query_one("#library-collections-freeform-note", TextArea)
        note.text = "Unsaved note survives compact controls"
        await pilot.pause()
        await pilot.press("tab")
        await _focus(screen, host, pilot, "#library-collections-mode-info", "Info")
        for width, height in ((80, 24), (170, 48)):
            await pilot.resize_terminal(width, height)
            reopened = not screen.query_one(
                "#library-collections-reader-shell"
            ).effective_layout.items_open
            await _show_pane(screen, host, pilot, "items")
            # Opening Items is a separate action; return once, then resize must
            # preserve this same mounted mode control without recreating the form.
            info = screen.query_one("#library-collections-mode-info")
            if reopened:
                info.focus()
            await _focus(screen, host, pilot, "#library-collections-mode-info", "Info")
            assert screen.query_one("#library-collections-freeform-note") is note
            assert note.text == "Unsaved note survives compact controls"
            if width == 170:
                for toolbar in ("primary", "mode"):
                    buttons = screen.query(
                        f"#library-collections-{toolbar}-toolbar Button"
                    )
                    assert len({button.region.y for button in buttons}) == 1
        await pilot.press("shift+tab")
        await _focus(screen, host, pilot, "#library-collections-mode-notes", "Notes")
        await pilot.press("enter")
        assert (
            screen.query_one("#library-collections-freeform-note", TextArea).text
            == "Unsaved note survives compact controls"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("empty", ["", "   "])
@pytest.mark.parametrize(
    ("sort", "expected_sort"), [("relevance", "saved_desc"), ("title_asc", "title_asc")]
)
async def test_empty_search_submission_keeps_valid_sort(
    empty, theme, sort, expected_sort, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    app, _, _ = await _seed()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open(host, pilot)
        engine = screen._library_collections_capture_controller
        request = replace(
            engine.state.requested_scope, search="Alpha", sort=sort, statuses=("saved",)
        )
        await screen._collections_controller._apply_library_collection_capture_request(
            request
        )
        await _show_pane(screen, host, pilot, "items")
        await _wait_for_condition(
            pilot,
            lambda: engine.state.page.total == 1,
            message="Seed query did not settle",
        )
        field = screen.query_one("#library-collections-filter", Input)
        field.focus()
        await pilot.press("home", "shift+end", "backspace")
        if empty:
            await pilot.press(*empty)
        assert field.value == empty
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: engine.state.page.total == 2,
            message="Cleared search did not restore results",
        )
        current = engine.state.applied_scope
        assert current.search == "" and current.sort == expected_sort
        assert current.statuses == ("saved",)
        assert {i.title for i in engine.state.page.items} == {"Alpha", "Beta"}
        await _focus(
            screen, host, pilot, "#library-collections-filter", "Filter captures"
        )
