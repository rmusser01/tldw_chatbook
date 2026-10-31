"""'/' can reach Console Behavior's generation fallbacks (TASK-33003.14).

Each fallback row carries exactly the label of a Providers & Models row, and
Providers & Models ranks first on a tie, so '/temperature' and its kin always
landed there and the fallbacks were unreachable by search.
"""

from __future__ import annotations

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import _settle_settings
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.settings_search_index import FIELD_SEARCH_INDEX

_CONSOLE = SettingsCategoryId.CONSOLE_BEHAVIOR
_PROVIDERS = SettingsCategoryId.PROVIDERS_MODELS


def _shared_label_fallbacks() -> list[tuple[str, str]]:
    """Console Behavior's fallback rows whose label a Providers row also has."""
    providers = {label for _fid, label in FIELD_SEARCH_INDEX[_PROVIDERS]}
    return [
        (field_id, label)
        for field_id, label in FIELD_SEARCH_INDEX[_CONSOLE]
        if field_id.startswith("settings-console-default-") and label in providers
    ]


@pytest.mark.asyncio
@private_profile_test
async def test_a_fallback_query_lands_on_each_console_fallback(request) -> None:
    """AC#1 for every fallback and AC#2: 'fallback <label>' opens Console
    Behavior on that fallback, and the bare label still opens Providers &
    Models on its own field."""
    screen = SettingsScreen(_build_test_app())
    rows = _shared_label_fallbacks()
    assert len({field_id for field_id, _label in rows}) == 14

    for field_id, label in rows:
        query = f"fallback {label}".lower()
        assert screen._filtered_category_values(query)[0] == _CONSOLE.value, label
        assert screen._top_field_match(query, _CONSOLE)[0] == field_id, label
        assert screen._filtered_category_values(label)[0] == _PROVIDERS.value, label
        provider_field = screen._top_field_match(label, _PROVIDERS)[0]
        assert provider_field.startswith("settings-model-profile-"), label


@pytest.mark.asyncio
@private_profile_test
async def test_enter_on_a_fallback_query_focuses_the_fallback(request) -> None:
    """The mounted landing: Enter on '/global fallback temperature' focuses
    Console Behavior's Temperature, not Providers & Models'."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        await _settle_settings(pilot)
        screen = _active_destination_screen(host)
        screen._submit_category_search("global fallback temperature")
        for _ in range(12):
            await pilot.pause()
        assert screen.active_category == _CONSOLE.value
        focused = host.focused
        assert focused is not None and focused.id == (
            "settings-console-default-temperature"
        ), f"focused={focused!r}"
