"""Providers & Models copy names what a save applies to (TASK-33002.3).

ADR-095 as amended 2026-09-26 (D1): a Settings save reaches new chats and
open chats nobody has used yet; a chat with work keeps its own settings,
changed per chat in Console with Alt+M. The save result, the toast, the
State line and the Provider field's Purpose used to say none of that
("Provider settings saved.", "Shared with Console", "Console generation
defaults"). The save result and toast are pinned on the real writer in
test_settings_configuration_hub.py.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _static_text,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

# The State banner's content width in the real app at 211x44 (detail pane
# interior 124 cells, minus the banner's 1-cell side padding and the pane's
# own), measured live on a scratch profile for TASK-33002.3.
_STATE_BANNER_CELLS_AT_211X44 = 120


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_state_line_names_its_scope_in_one_row(request) -> None:
    screen = SettingsScreen(_build_test_app())
    category = SettingsCategoryId.PROVIDERS_MODELS

    scope = screen._category_state_scope_text(category)
    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s | {scope}"
    )
    assert scope == (
        "Applies to new and unused open chats · used chats keep theirs (Console: Alt+M)"
    )
    # One row with the badge AND the unsaved count the dirty line gains
    # (spec §7(c): "State: {badge} · N unsaved | {scope}").
    assert (
        cell_len(f"State: Draft — save with s · 99 unsaved | {scope}")
        <= _STATE_BANNER_CELLS_AT_211X44
    )


@pytest.mark.asyncio
@private_profile_test
async def test_provider_field_purpose_names_the_new_chat_scope(request) -> None:
    screen = SettingsScreen(_build_test_app())
    screen._active_settings_field_id = "settings-provider-value"

    rows = dict(screen._provider_field_guidance_rows_base())

    assert rows["Focused setting"] == "Provider"
    assert rows["Purpose"] == (
        "Sets the provider new chats start with; open chats nobody has used "
        "yet follow it."
    )


@pytest.mark.asyncio
@private_profile_test
async def test_focusing_the_visible_provider_picker_shows_the_provider_purpose(
    request,
) -> None:
    """AC#3 live: the hidden Select was the only id the Provider rows named.

    The Provider field a user reaches is the search box and its picker list;
    focusing either used to leave the inspector on "None — Tab to a setting",
    so the Purpose row could never be read.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
        await _wait_for_selector(screen, pilot, "#settings-provider-search")
        for selector in ("#settings-provider-search", "#settings-provider-picker"):
            screen.query_one(selector).focus()
            await pilot.pause()
            guide = [
                _static_text(screen.query_one(f"#settings-provider-field-guide-{i}"))
                for i in range(2)
            ]
            assert guide[0] == "Focused setting: Provider", selector
            assert guide[1] == (
                "Purpose: Sets the provider new chats start with; open chats "
                "nobody has used yet follow it."
            ), selector
