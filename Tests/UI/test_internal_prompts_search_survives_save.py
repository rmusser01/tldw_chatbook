"""Screen-level regression test: the Internal Prompts panel's search text and
widget identity must survive a Save/Reset (P3 whole-branch review, Fix 1).

Background: ``_on_internal_prompts_modified`` used to assign a
``recompose=True`` reactive (``internal_prompts_dirty``) on every
``InternalPromptsPanel.Modified`` event -- which fires on the FIRST save of an
uncustomized prompt (or reset of a customized one). ``recompose=True``
unmounts and rebuilds the whole SettingsScreen detail pane, replacing the
``InternalPromptsPanel`` with a brand-new instance and wiping its search box
text and scroll position -- defeating the panel's own targeted
``_refresh_row`` design (which exists precisely to avoid this).

This only reproduces at the SCREEN level: the panel in isolation
(``Tests/UI/test_internal_prompts_panel_editing.py``) has no screen-driven
recompose to trigger the bug. So this test drives the real ``SettingsScreen``
through the same ``DestinationHarness`` + ``_open_settings_category`` helpers
``Tests/UI/test_settings_configuration_hub.py`` uses.
"""

import pytest
from textual.widgets import Input

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from tldw_chatbook.Internal_Prompts import authoring
from tldw_chatbook.Widgets.settings_internal_prompts_panel import InternalPromptsPanel

_PROMPT_ID = "agents.subagent_system"


@pytest.mark.asyncio
async def test_search_text_and_panel_identity_survive_save(scratch_config):
    scratch_config("")
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-internal-prompts")
        screen = _active_destination_screen(host)

        panel = screen.query_one(
            "#settings-internal-prompts-panel", InternalPromptsPanel
        )
        search = panel.query_one("#internal-prompts-search", Input)
        search.value = "subagent"
        await pilot.pause()

        # Sanity: the prompt starts uncustomized so this save is the "first
        # save of an uncustomized prompt" case the bug report calls out.
        assert authoring.override_state(_PROMPT_ID).customized is False

        await panel._apply_editor_result(
            _PROMPT_ID, {"action": "save", "text": "SCREEN-LEVEL SAVE"}
        )
        await pilot.pause()
        await pilot.pause()

        panel_after = screen.query_one(
            "#settings-internal-prompts-panel", InternalPromptsPanel
        )
        assert panel_after is panel, (
            "Save recomposed the SettingsScreen detail pane and rebuilt the "
            "InternalPromptsPanel (should be a targeted refresh, not a "
            "recompose)"
        )

        search_after = panel_after.query_one("#internal-prompts-search", Input)
        assert search_after.value == "subagent", (
            "Save wiped the panel's search text -- screen-level recompose "
            "regression"
        )
