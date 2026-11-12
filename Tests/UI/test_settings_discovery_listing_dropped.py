"""Settings ▸ Providers & Models: a discovery listing never outlives the
endpoint it came from (review of #3057, R2 F3).

The Default model picker's "Served now" rows assume the listing is dropped
on every provider or endpoint change. Revert and a deep link changed both
without dropping it, and the vLLM presentation restore put a listing back
without telling the picker.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, SelectionList

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_settings_default_model_picker import (
    _app,
    _open_providers,
    _rows,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _revert
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

_SIZE = (211, 44)
_SERVED = "o3-mini-new"


def _listed(screen) -> list[str]:
    discovered = screen.query_one("#settings-discovered-models-list", SelectionList)
    return [
        str(discovered.get_option_at_index(index).value)
        for index in range(discovered.option_count)
    ]


async def _served_rows(pilot, screen) -> list[str]:
    """The picker's rows for the served id, as a user who types it sees them."""
    _picker, field, results, _adapter = _widgets(screen)
    field.focus()
    await pilot.pause()
    await pilot.press(*"o3-mini")
    await pilot.pause()
    rows = [row for row in _rows(results) if _SERVED in row]
    await pilot.press("escape")
    await pilot.pause()
    return rows


async def _discover(host, pilot, screen) -> None:
    screen.query_one("#settings-discover-provider-models", Button).press()
    await pilot.pause()  # the press starts the worker _settle waits for
    await _settle(host, pilot)
    assert _listed(screen) == [_SERVED]


@pytest.mark.asyncio
@private_profile_test
async def test_revert_drops_a_listing_discovered_from_a_discarded_endpoint(request):
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",), served=(_SERVED,))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.focus()
        await pilot.pause()
        await pilot.press(*"https://draft.example/v1", "escape")
        await pilot.pause()
        await _discover(host, pilot, screen)
        assert await _served_rows(pilot, screen)

        await _revert(host, pilot, discard=True)

        assert endpoint.value == ""
        assert _listed(screen) == []
        assert await _served_rows(pilot, screen) == []


@pytest.mark.asyncio
@private_profile_test
async def test_a_deep_link_to_another_provider_drops_the_listing(request):
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",), served=(_SERVED,))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _discover(host, pilot, screen)

        screen.apply_navigation_context(
            {
                "category": SettingsCategoryId.PROVIDERS_MODELS.value,
                "provider": "llama_cpp",
                "field": "endpoint",
            }
        )
        await _settle(host, pilot)
        await pilot.pause()
        screen = _active_destination_screen(host)

        assert screen._provider_widget_value() == "llama_cpp"
        assert _listed(screen) == []
        assert screen._model_discovery_models == ()


@pytest.mark.asyncio
@private_profile_test
async def test_the_vllm_presentation_restore_tells_the_picker_its_listing(request):
    """No key reaches this restore without a live vLLM server: it is driven
    through the two methods the vLLM default flow calls."""
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",), served=(_SERVED,))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _discover(host, pilot, screen)
        snapshot = screen._snapshot_vllm_default_presentation()

        screen._reset_provider_model_discovery_state()
        await pilot.pause()
        assert await _served_rows(pilot, screen) == []

        screen._restore_vllm_default_presentation(snapshot)
        await pilot.pause()
        assert _listed(screen) == [_SERVED]
        assert await _served_rows(pilot, screen)
