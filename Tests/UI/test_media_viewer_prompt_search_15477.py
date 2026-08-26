"""TASK-15477: media-viewer prompt search was dead code raising per keystroke.

`MediaViewerPanel.search_prompts` imported `get_prompts_db` from
`DB/Prompts_DB` -- a symbol that never existed -- so every keystroke in the
Analysis tab's "Search Prompts" box raised an ``ImportError`` that was
silently swallowed at the bottom of the handler. The handler also called
``self.app.call_from_thread`` from the UI thread (illegal in Textual) and,
as designed, would have run its sqlite search inline with no debounce.

Investigation (see the task's Implementation Plan) found the affordance had
no reachable UX purpose: ``MediaViewerPanel`` is only mounted by
``MediaWindow_v2``, which backs the standalone ``MediaScreen`` route --
permanently aliased to Library (task-2851) with no other entry point. The
fix removes the search/keyword/select widgets and their handlers outright
rather than wiring them to a live DB seam that nothing could ever reach.
The System/User Prompt ``TextArea``s and "Generate Analysis" button are
untouched -- they don't depend on the removed widgets.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from loguru import logger as loguru_logger
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button, TextArea

from tldw_chatbook.Widgets.Media.media_viewer_panel import MediaViewerPanel

# Method/handler names the pre-fix panel defined for the dead prompt-search
# affordance. Asserted absent post-fix.
_REMOVED_METHOD_NAMES = (
    "search_prompts",
    "load_prompt_details",
    "_update_prompt_select",
    "handle_prompt_search",
    "handle_prompt_keyword_change",
    "handle_prompt_selection",
)

# Widget ids the pre-fix panel composed for the dead prompt-search
# affordance. Asserted absent post-fix.
_REMOVED_WIDGET_IDS = (
    "prompt-search-input",
    "prompt-keyword-input",
    "prompt-select",
)

# Widgets that must survive the removal: manual prompt editing and
# "Generate Analysis" never depended on the search/select affordance
# (`prepare_analysis_messages`/`handle_generate_analysis` only read these).
_SURVIVING_WIDGET_IDS = (
    "system-prompt-area",
    "user-prompt-area",
    "generate-analysis-btn",
)


class MediaViewerTestApp(App[None]):
    def __init__(self, panel: MediaViewerPanel):
        super().__init__()
        self.panel = panel

    def compose(self) -> ComposeResult:
        yield self.panel


def _media_app() -> Mock:
    app = Mock()
    app._media_types_for_ui = []
    app.get_authoritative_runtime_source = Mock(return_value="local")
    app.notify = Mock()
    return app


@pytest.mark.asyncio
async def test_prompt_search_widgets_are_gone() -> None:
    """AC1: the dead search/keyword/select widgets no longer compose.

    Pre-fix, `panel.query_one("#prompt-search-input", Input)` succeeds (the
    widget composes fine -- it's the keystroke handler that was broken), so
    this assertion is red against the old code and green once the widgets
    are removed.
    """
    panel = MediaViewerPanel(_media_app())
    app = MediaViewerTestApp(panel)

    async with app.run_test() as pilot:
        await pilot.pause()
        for widget_id in _REMOVED_WIDGET_IDS:
            with pytest.raises(NoMatches):
                panel.query_one(f"#{widget_id}")


@pytest.mark.asyncio
async def test_surviving_prompt_widgets_still_compose() -> None:
    """Guard against over-deleting: manual prompt editing + Generate stay."""
    panel = MediaViewerPanel(_media_app())
    app = MediaViewerTestApp(panel)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert panel.query_one("#system-prompt-area", TextArea) is not None
        assert panel.query_one("#user-prompt-area", TextArea) is not None
        assert panel.query_one("#generate-analysis-btn", Button) is not None


def test_prompt_search_methods_are_removed() -> None:
    """AC1/AC2: the broken handler methods no longer exist on the class.

    `search_prompts` was the method that imported the nonexistent
    `get_prompts_db` symbol and illegally called `call_from_thread` from the
    UI thread; removing it (rather than leaving a dead method around)
    guarantees no code path can still raise per keystroke.
    """
    for name in _REMOVED_METHOD_NAMES:
        assert not hasattr(MediaViewerPanel, name), (
            f"MediaViewerPanel.{name} should have been removed with the "
            "dead prompt-search affordance"
        )


def test_prompts_db_module_still_has_no_get_prompts_db() -> None:
    """Documents the root cause: `get_prompts_db` never existed in
    `DB/Prompts_DB`. This module is not being given that symbol -- the
    caller was removed instead -- so this stays true after the fix too, and
    guards against a future PR re-adding an import of a symbol that was
    never there without re-verifying it first.
    """
    import tldw_chatbook.DB.Prompts_DB as prompts_db_module

    assert not hasattr(prompts_db_module, "get_prompts_db")


@pytest.mark.asyncio
async def test_mounting_and_settling_the_panel_logs_no_prompt_search_error() -> None:
    """AC2 log evidence: mounting the panel (which used to `on_mount` into a
    UI where every keystroke in the now-removed search box logged
    ``Error searching prompts: cannot import name 'get_prompts_db'``) never
    emits that error. Pre-fix the widget existed but this assertion alone
    wouldn't fail without a simulated keystroke; combined with
    `test_prompt_search_widgets_are_gone` (which proves there is no longer
    any widget to type into) and `test_prompt_search_methods_are_removed`
    (which proves the offending method is gone), this closes the loop: no
    code path remains that could ever produce that log line again.
    """
    records: list[dict] = []
    sink_id = loguru_logger.add(lambda message: records.append(message.record), level="ERROR")
    try:
        panel = MediaViewerPanel(_media_app())
        app = MediaViewerTestApp(panel)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
    finally:
        loguru_logger.remove(sink_id)

    assert not any(
        "search" in str(record.get("message", "")).lower()
        and "prompt" in str(record.get("message", "")).lower()
        for record in records
    )
