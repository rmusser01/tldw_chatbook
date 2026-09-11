"""Critique #10 media-row fixes: labelled age, applied-scope line, copy polish.

Covers tasks 32347 (the row age says it is an age), 32350 (a scope line
states the APPLIED filter, never the box's draft) and 32364 (a status word
never prefixes a title; one separator glyph; the import footer names what
Enter does at each step).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_media_state import media_added_age_copy
from Tests.UI.test_library_media_render_fixes import _apply_media_filter, _host
from Tests.UI.test_library_media_side_by_side import _open_media_list
from Tests.UI.test_library_shell import _wait_for_condition

#: ``_host()`` seeds "Interview Recording" (audio) and "Product Demo Video"
#: (video); "interview" is in exactly one title and no other row's content,
#: so a one-row applied scope is provable rather than incidental.
_ONE_HIT_QUERY = "interview"


def test_the_age_label_says_what_the_age_is():
    now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
    assert media_added_age_copy("2026-09-11T11:50:00+00:00", now=now) == "added 10m ago"
    assert media_added_age_copy("2026-09-11T11:59:40+00:00", now=now) == "added just now"
    assert media_added_age_copy("", now=now) == ""


@pytest.mark.asyncio
async def test_the_media_header_states_the_applied_scope_not_the_typed_draft():
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, _ONE_HIT_QUERY)
        box = screen.query_one("#library-media-filter", Input)
        box.value = "a different draft"  # never submitted
        await pilot.pause()
        line = screen.query_one("#library-media-scope-line", Static)
        assert str(line.content).startswith("Media · 1 of "), str(line.content)
        assert f'filter “{_ONE_HIT_QUERY}”' in str(line.content), str(line.content)
        assert "a different draft" not in str(line.content), str(line.content)
        assert screen.query("#library-media-scope-clear")


def _ingest_enter_label(*, start_enabled: bool) -> str:
    """The Enter hint the Ingest footer registers for one Start-gate state."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Library_Modules.library_ingest_controller import (
        LibraryIngestController,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    stub = SimpleNamespace(
        LIBRARY_INGEST_SHORTCUTS=LibraryScreen.LIBRARY_INGEST_SHORTCUTS,
        _build_library_ingest_state=lambda: SimpleNamespace(
            start_enabled=start_enabled
        ),
        _library_ingest_start_consent=None,
        app_instance=None,
    )
    shortcuts = LibraryIngestController._library_ingest_shortcuts_for_current_state(
        stub
    )
    key, description = shortcuts[0]
    assert key == "enter", shortcuts
    return description


def test_the_import_footer_names_what_enter_does_at_each_step():
    """task-32364 AC#3: one label for two different actions taught the wrong
    thing at exactly the moment the user commits -- the first Enter on an
    unvalidated path validates it, only the second starts the import."""
    assert _ingest_enter_label(start_enabled=False) == "check this path"
    assert _ingest_enter_label(start_enabled=True) == "start import"


@pytest.mark.asyncio
async def test_a_prompt_row_does_not_repeat_the_canvas_it_is_already_on():
    """task-32364: "Prompt · " led every row on a canvas titled Prompts."""
    from tldw_chatbook.Library.library_prompts_state import (
        PromptListRow,
        PromptsListState,
    )
    from Tests.UI.test_library_prompts_canvas import _CanvasHost

    state = PromptsListState(
        rows=(
            PromptListRow(
                prompt_id=8,
                name="Outcome first",
                secondary="Reusable structure",
                artifact_type="prompt",
                type_label="Prompt",
                source_label="Local",
                lane_summary="has system and user text",
            ),
        ),
        count=1,
        sort="newest",
    )
    async with _CanvasHost(state).run_test() as pilot:
        label = str(pilot.app.query_one("#library-prompt-row-8", Button).label)
        assert "Local · has system and user text" in label, label
        assert "Prompt · " not in label, label


@pytest.mark.asyncio
async def test_clearing_the_applied_filter_also_clears_the_box():
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, _ONE_HIT_QUERY)
        screen.query_one("#library-media-scope-clear", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query_one("#library-media-filter", Input).value,
            message="The scope line's Clear left a draft in the filter box.",
        )
        line = screen.query_one("#library-media-scope-line", Static)
        assert "filter" not in str(line.content), str(line.content)
        assert not screen.query("#library-media-scope-clear")
