"""Screen-level tests for Settings > Image Gen (task 4 of the Settings >
Image Gen plan): category registration (rail + search) and the read-only
panel's population from a scratch config.

Harness mirrors ``Tests/UI/test_internal_prompts_search_survives_save.py``:
``DestinationHarness`` + ``_open_settings_category`` (both from
``test_destination_shells``/``test_settings_configuration_hub``) plus the
shared ``scratch_config`` fixture (``Tests/UI/conftest.py`` re-exports it
from ``Tests/Internal_Prompts/conftest.py``).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Checkbox, Input, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _wait_for_settings_search_focus,
    _wait_for_settings_text,
)
from tldw_chatbook.Image_Generation.config import (
    DEFAULT_SWARMUI_TIMEOUT_SECONDS,
    reset_image_generation_config_cache,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_image_gen_defaults import effective_placeholder
from tldw_chatbook.Widgets.settings_image_gen_panel import ImageGenSettingsPanel


_ALL_SECRET_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "NOVITA_API_KEY",
    "TOGETHER_API_KEY",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "SWARMUI_TOKEN",
)


@pytest.fixture(autouse=True)
def _reset_image_gen_cache():
    reset_image_generation_config_cache()
    yield
    reset_image_generation_config_cache()


async def _open_image_gen(pilot) -> None:
    await _open_settings_category(pilot, "#settings-category-image_generation")


@pytest.mark.asyncio
async def test_image_gen_category_in_rail_and_search(scratch_config):
    scratch_config("")
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)

        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "Image Gen" in str(rail_button.label)

        await pilot.press("/")
        await _wait_for_settings_search_focus(screen, pilot)
        await pilot.press(*"swarmui")
        await _wait_for_settings_text(screen, pilot, "swarmui")

        assert screen.query_one("#settings-category-image_generation").display

        await _open_image_gen(pilot)
        assert screen.active_category == SettingsCategoryId.IMAGE_GENERATION.value
        assert "Image Gen" in _visible_text(screen)


@pytest.mark.asyncio
async def test_panel_populates_from_scratch_config(scratch_config):
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]

[image_generation.openrouter]
default_model = "m-x"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        # openrouter row: enabled + marked default.
        enabled_checkbox = panel.query_one(
            "#settings-imagegen-enabled-openrouter", Checkbox
        )
        assert enabled_checkbox.value is True
        default_marker = panel.query_one(
            "#settings-imagegen-default-marker-openrouter", Static
        )
        assert "Default" in str(default_marker.renderable)

        # A non-enabled, non-default backend stays unchecked with no marker.
        sd_cpp_checkbox = panel.query_one(
            "#settings-imagegen-enabled-stable_diffusion_cpp", Checkbox
        )
        assert sd_cpp_checkbox.value is False
        sd_cpp_marker = panel.query_one(
            "#settings-imagegen-default-marker-stable_diffusion_cpp", Static
        )
        assert str(sd_cpp_marker.renderable) == ""

        # openrouter's default_model field renders the RAW config value.
        openrouter_model = panel.query_one(
            "#settings-imagegen-field-openrouter-default_model", Input
        )
        assert openrouter_model.value == "m-x"

        # swarmui's default_model is unset: input stays empty, placeholder is
        # whatever the resolved effective value would be (task-620 lesson) --
        # here that's "" (swarmui has no baked-default model), but the
        # mechanism must still wire placeholder == effective_placeholder(...).
        from tldw_chatbook.Image_Generation.config import get_image_generation_config

        cfg = get_image_generation_config(reload=True)
        swarmui_model = panel.query_one(
            "#settings-imagegen-field-swarmui-default_model", Input
        )
        assert swarmui_model.value == ""
        assert swarmui_model.placeholder == effective_placeholder(
            cfg, "swarmui", "default_model"
        )

        # config.py's bundled default template (CONFIG_TOML_CONTENT) bakes a
        # literal `timeout_seconds = 120` into every backend's TOML section,
        # and load_cli_config_and_ensure_existence deep-merges that template
        # under ANY existing config file (even an empty scratch one) -- so
        # this field is never actually absent from the raw section the way
        # the string fields above are. The panel must still show the RAW
        # value here (not silently substitute the resolved/placeholder
        # value), proving the value/placeholder split reads from the right
        # source rather than always preferring one or the other.
        swarmui_timeout = panel.query_one(
            "#settings-imagegen-field-swarmui-timeout_seconds", Input
        )
        assert swarmui_timeout.value == str(DEFAULT_SWARMUI_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_secret_input_never_echoes_saved_key(scratch_config):
    saved_key = "sk-should-never-render-in-the-ui"
    scratch_config(
        f"""
[image_generation.openrouter]
api_key = "{saved_key}"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        secret_input = panel.query_one(
            "#settings-imagegen-field-openrouter-api_key", Input
        )
        assert secret_input.value == ""

        source_line = panel.query_one(
            "#settings-imagegen-key-source-openrouter", Static
        )
        assert str(source_line.renderable) == "local config key saved"

        assert saved_key not in _visible_text(screen)


@pytest.mark.asyncio
async def test_env_key_shows_env_source_line(scratch_config, monkeypatch):
    for var in _ALL_SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-secret-value-not-rendered")
    scratch_config("")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        secret_input = panel.query_one(
            "#settings-imagegen-field-openrouter-api_key", Input
        )
        assert secret_input.value == ""

        source_line = panel.query_one(
            "#settings-imagegen-key-source-openrouter", Static
        )
        assert str(source_line.renderable) == "env: OPENROUTER_API_KEY"
        assert "env-secret-value-not-rendered" not in _visible_text(screen)


@pytest.mark.asyncio
async def test_save_revert_test_buttons_present_but_disabled(scratch_config):
    scratch_config("")
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        assert panel.query_one("#settings-imagegen-save", Button).disabled
        assert panel.query_one("#settings-imagegen-revert", Button).disabled
        for backend_id in (
            "stable_diffusion_cpp",
            "swarmui",
            "openrouter",
            "novita",
            "together",
            "modelstudio",
        ):
            assert panel.query_one(
                f"#settings-imagegen-test-{backend_id}", Button
            ).disabled
