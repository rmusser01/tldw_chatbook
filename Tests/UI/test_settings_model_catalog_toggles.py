"""Settings UI toggles for automatic model catalog refresh (ADR-019).

The Providers & Models category hosts a Model discovery subsection with
toggles for the ``[model_catalog]`` config section: a master auto-refresh
switch, a staleness window in hours, per-provider auto-refresh opt-outs, and
per-provider write-through opt-ins. Changes persist immediately through
``save_settings_to_cli_config`` and saved values initialize the widgets when
the subsection composes.
"""

import time
from unittest.mock import patch

import pytest
from textual.widgets import Checkbox, Input

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
)
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
    AUTO_REFRESH_PROVIDER_LIST_KEYS,
)

_PROVIDERS_CATEGORY_BUTTON = "#settings-category-providers-models"
_OVERVIEW_CATEGORY_BUTTON = "#settings-category-overview"
_MASTER_SELECTOR = "#settings-model-catalog-auto-refresh"
_HOURS_SELECTOR = "#settings-model-catalog-stale-hours"


def _auto_selector(provider: str) -> str:
    return f"#settings-mc-auto-{provider.lower()}"


def _write_selector(provider: str) -> str:
    return f"#settings-mc-write-{provider.lower()}"


def _build_settings_app():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {
        "openai": {
            "api_base_url": "https://api.openai.com/v1",
            "api_key_env_var": "OPENAI_API_KEY",
        },
    }
    return app


async def _open_providers_category(host, pilot):
    await pilot.click(_PROVIDERS_CATEGORY_BUTTON)
    screen = _active_destination_screen(host)
    await _wait_for_selector(screen, pilot, _MASTER_SELECTOR, timeout=5.0)
    return screen


async def _wait_for_save_calls(save_mock, pilot, count: int, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if save_mock.call_count >= count:
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for {count} save call(s); saw {save_mock.call_count}"
    )


@pytest.mark.asyncio
async def test_model_catalog_toggles_render_for_all_providers_with_defaults():
    app = _build_settings_app()
    host = DestinationHarness(app, "settings")

    with (
        patch.object(settings_screen_module, "load_settings", return_value={}),
        patch.object(
            settings_screen_module,
            "save_settings_to_cli_config",
            return_value=True,
        ) as save_mock,
    ):
        async with host.run_test(size=(180, 50)) as pilot:
            screen = await _open_providers_category(host, pilot)

            master = screen.query_one(_MASTER_SELECTOR, Checkbox)
            assert master.value is True

            hours = screen.query_one(_HOURS_SELECTOR, Input)
            assert hours.value == "24"

            for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
                auto_box = screen.query_one(_auto_selector(provider), Checkbox)
                write_box = screen.query_one(_write_selector(provider), Checkbox)
                assert auto_box.value is True, provider
                assert write_box.value is False, provider

        # Mounting the subsection must not persist anything by itself.
        assert not save_mock.called


@pytest.mark.asyncio
async def test_model_catalog_toggles_initialize_from_saved_config():
    app = _build_settings_app()
    host = DestinationHarness(app, "settings")
    saved_settings = {
        "model_catalog": {
            "auto_refresh_enabled": False,
            "stale_after_hours": 6,
            "auto_refresh_disabled": ["openai", "zai"],
            "write_to_config": ["openrouter"],
        },
    }

    with (
        patch.object(
            settings_screen_module,
            "load_settings",
            return_value=saved_settings,
        ),
        patch.object(
            settings_screen_module,
            "save_settings_to_cli_config",
            return_value=True,
        ),
    ):
        async with host.run_test(size=(180, 50)) as pilot:
            screen = await _open_providers_category(host, pilot)

            assert screen.query_one(_MASTER_SELECTOR, Checkbox).value is False
            assert screen.query_one(_HOURS_SELECTOR, Input).value == "6"

            expected_auto = {
                "OpenAI": False,
                "Anthropic": True,
                "MistralAI": True,
                "Moonshot": True,
                "OpenRouter": True,
                "ZAI": False,
            }
            expected_write = {
                "OpenAI": False,
                "Anthropic": False,
                "MistralAI": False,
                "Moonshot": False,
                "OpenRouter": True,
                "ZAI": False,
            }
            for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
                assert (
                    screen.query_one(_auto_selector(provider), Checkbox).value
                    is expected_auto[provider]
                ), provider
                assert (
                    screen.query_one(_write_selector(provider), Checkbox).value
                    is expected_write[provider]
                ), provider


@pytest.mark.asyncio
async def test_model_catalog_toggle_change_persists_immediately():
    app = _build_settings_app()
    host = DestinationHarness(app, "settings")

    with (
        patch.object(settings_screen_module, "load_settings", return_value={}),
        patch.object(
            settings_screen_module,
            "save_settings_to_cli_config",
            return_value=True,
        ) as save_mock,
    ):
        async with host.run_test(size=(180, 50)) as pilot:
            screen = await _open_providers_category(host, pilot)

            master = screen.query_one(_MASTER_SELECTOR, Checkbox)
            master.value = False
            await _wait_for_save_calls(save_mock, pilot, 1)

            payload = save_mock.call_args[0][0]
            assert payload == {
                "model_catalog": {
                    "auto_refresh_enabled": False,
                    "stale_after_hours": 24,
                    "auto_refresh_disabled": [],
                    "write_to_config": [],
                }
            }

            openai_auto = screen.query_one(_auto_selector("OpenAI"), Checkbox)
            openai_auto.value = False
            await _wait_for_save_calls(save_mock, pilot, 2)
            payload = save_mock.call_args[0][0]
            # Persisted form uses the exact [providers] keys per config.py.
            assert payload["model_catalog"]["auto_refresh_disabled"] == ["OpenAI"]
            assert payload["model_catalog"]["auto_refresh_enabled"] is False

            zai_write = screen.query_one(_write_selector("ZAI"), Checkbox)
            zai_write.value = True
            await _wait_for_save_calls(save_mock, pilot, 3)
            payload = save_mock.call_args[0][0]
            assert payload["model_catalog"]["write_to_config"] == ["ZAI"]

            hours = screen.query_one(_HOURS_SELECTOR, Input)
            hours.value = "6"
            await _wait_for_save_calls(save_mock, pilot, 4)
            payload = save_mock.call_args[0][0]
            assert payload["model_catalog"]["stale_after_hours"] == 6


@pytest.mark.asyncio
async def test_model_catalog_invalid_hours_input_skips_persist():
    app = _build_settings_app()
    host = DestinationHarness(app, "settings")

    with (
        patch.object(settings_screen_module, "load_settings", return_value={}),
        patch.object(
            settings_screen_module,
            "save_settings_to_cli_config",
            return_value=True,
        ) as save_mock,
    ):
        async with host.run_test(size=(180, 50)) as pilot:
            screen = await _open_providers_category(host, pilot)

            hours = screen.query_one(_HOURS_SELECTOR, Input)
            hours.value = "6"
            await _wait_for_save_calls(save_mock, pilot, 1)
            calls_before = save_mock.call_count

            # Programmatic assignment bypasses the integer input restriction;
            # an unparseable value must not overwrite the persisted section.
            hours.value = "abc"
            await pilot.pause()
            await pilot.pause()
            assert save_mock.call_count == calls_before


@pytest.mark.asyncio
async def test_model_catalog_ignores_unrelated_checkbox_events():
    app = _build_settings_app()
    host = DestinationHarness(app, "settings")

    with (
        patch.object(settings_screen_module, "load_settings", return_value={}),
        patch.object(
            settings_screen_module,
            "save_settings_to_cli_config",
            return_value=True,
        ) as save_mock,
    ):
        async with host.run_test(size=(180, 50)) as pilot:
            screen = await _open_providers_category(host, pilot)

            rogue = Checkbox("rogue", id="settings-some-other-checkbox")
            screen.post_message(Checkbox.Changed(rogue, True))
            await pilot.pause()
            await pilot.pause()
            assert not save_mock.called


@pytest.mark.asyncio
async def test_model_catalog_saved_values_load_back_on_recompose():
    app = _build_settings_app()
    host = DestinationHarness(app, "settings")
    current_settings: dict = {}

    def fake_load_settings(*_args, **_kwargs):
        return current_settings

    def fake_save(section_values):
        for section, values in section_values.items():
            current_settings[section] = dict(values)
        return True

    with (
        patch.object(
            settings_screen_module,
            "load_settings",
            side_effect=fake_load_settings,
        ),
        patch.object(
            settings_screen_module,
            "save_settings_to_cli_config",
            side_effect=fake_save,
        ),
    ):
        async with host.run_test(size=(180, 50)) as pilot:
            screen = await _open_providers_category(host, pilot)

            screen.query_one(_MASTER_SELECTOR, Checkbox).value = False
            screen.query_one(_auto_selector("OpenAI"), Checkbox).value = False
            screen.query_one(_write_selector("ZAI"), Checkbox).value = True
            screen.query_one(_HOURS_SELECTOR, Input).value = "12"
            await pilot.pause()
            await pilot.pause()

            # Switching categories recomposes the detail pane; the widgets must
            # come back initialized from the values that were just persisted.
            await pilot.click(_OVERVIEW_CATEGORY_BUTTON)
            await pilot.pause()
            await pilot.click(_PROVIDERS_CATEGORY_BUTTON)
            screen = _active_destination_screen(host)
            await _wait_for_selector(screen, pilot, _MASTER_SELECTOR, timeout=5.0)

            assert screen.query_one(_MASTER_SELECTOR, Checkbox).value is False
            assert screen.query_one(_HOURS_SELECTOR, Input).value == "12"
            for provider in AUTO_REFRESH_PROVIDER_LIST_KEYS:
                expected_auto = provider != "OpenAI"
                expected_write = provider == "ZAI"
                assert (
                    screen.query_one(_auto_selector(provider), Checkbox).value
                    is expected_auto
                ), provider
                assert (
                    screen.query_one(_write_selector(provider), Checkbox).value
                    is expected_write
                ), provider
