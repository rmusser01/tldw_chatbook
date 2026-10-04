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
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _static_text,
    _wait_for_selector,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_state_line_names_its_scope_in_one_row(request) -> None:
    """AC#2 on the rendered banner (production stylesheet, 211x44).

    Plan ruling T3 x T4: the dirty line becomes "State: {badge} · N unsaved
    | {scope}" with no Save/Revert tail (the badge and the footer's s/r
    hints carry that guidance), so the scope must leave room for the count.
    The real-edit check fails if the dirty line ever wraps, whatever T4
    makes it say. The F1 help body repeats the scope under "Scope:".
    """
    category = SettingsCategoryId.PROVIDERS_MODELS
    host = _SettingsCssHarness(_build_test_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        screen._select_category(category.value)
        await _wait_for_selector(screen, pilot, "#settings-model-value")
        banner = screen.query_one("#settings-category-state-banner", Static)
        scope = screen._category_state_scope_text(category)
        assert scope == (
            "Applies to new and unused open chats · used chats keep theirs "
            "(Console: Alt+M)"
        )
        assert _static_text(banner) == f"State: Draft — save with s | {scope}"
        assert banner.content_region.height == 1

        widest = f"State: Draft — save with s · 99 unsaved | {scope}"
        banner.update(widest)
        await pilot.pause()
        assert (_static_text(banner), banner.content_region.height) == (widest, 1)

        screen.query_one("#settings-model-value", Input).value = "scope-review"
        await pilot.pause()
        assert screen._category_has_unsaved_changes(category)
        assert _static_text(banner) == screen._category_state_banner_text(category)
        assert banner.content_region.height == 1

        screen.action_show_workbench_help()
        await pilot.pause()
        body = str(host.screen.query_one("#workbench-help-body", Static).render())
        assert f"Scope: {scope}" in body, body


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
    # Final review (Task 3 rider): the Model row kept "when Console has no
    # narrower override" framing; it names the same scope now.
    screen._active_settings_field_id = "settings-model-value"
    assert dict(screen._provider_field_guidance_rows_base())["Purpose"] == (
        "Sets the model new chats start with; open chats nobody has used yet follow it."
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
        # TASK-33007.2, rewritten on purpose: the list under the Provider
        # control never takes focus, so the control is the one field.
        for selector in ("#settings-provider-search",):
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
