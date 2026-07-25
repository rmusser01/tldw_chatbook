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

import tomllib

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
    get_image_generation_config,
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

        # A field WITH a real, non-empty Python-level baked default
        # (swarmui timeout_seconds) proves the placeholder mechanism shows
        # the actual value that will be used, not a degenerate empty string.
        # This only holds because the panel reads the USER'S OWN unmerged
        # config file for the VALUE (load_user_image_generation_table()) --
        # config.py's bundled default template bakes a literal
        # `timeout_seconds = 120` into every backend's TOML section, and
        # load_cli_config_and_ensure_existence deep-merges that template
        # into the loaded config regardless of what's on disk. Reading the
        # MERGED config for the value (the pre-fix bug) would show "120" as
        # the VALUE here even though this scratch file never sets it.
        swarmui_timeout = panel.query_one(
            "#settings-imagegen-field-swarmui-timeout_seconds", Input
        )
        assert swarmui_timeout.value == ""
        assert swarmui_timeout.placeholder == str(DEFAULT_SWARMUI_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_fresh_config_shows_placeholder_not_merged_default_value(scratch_config):
    """Set-vs-default blur regression (Fix Round 1).

    A config file with NO ``[image_generation]`` section at all (the
    freshest possible install state) must render every field EMPTY with its
    resolved-effective value as the placeholder -- never the value baked
    into config.py's bundled default template. Uses the design spec's own
    named example (openrouter's model): pre-fix, ``SettingsConfigAdapter
    .load()``'s deep-merged config made this render as if the user had
    explicitly set ``default_model`` to the template's baked value.
    """
    scratch_config(
        """
[general]
default_theme = "textual-dark"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        openrouter_model = panel.query_one(
            "#settings-imagegen-field-openrouter-default_model", Input
        )
        assert openrouter_model.value == ""
        assert openrouter_model.placeholder == "google/gemini-2.5-flash-image"


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


# ---------------------------------------------------------------------------
# Task 5: draft/dirty editing + Save/Revert.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_marks_dirty_and_save_writes_nested_toml(scratch_config, tmp_path):
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]

[image_generation.openrouter]
default_model = "old-model"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        model_input = panel.query_one(
            "#settings-imagegen-field-openrouter-default_model", Input
        )
        model_input.value = "openai/gpt-5-image-mini"
        await pilot.pause()

        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" in str(rail_button.label)
        assert not panel.query_one("#settings-imagegen-save", Button).disabled

        await pilot.click("#settings-imagegen-save")
        await _wait_for_settings_text(screen, pilot, "Image Gen defaults saved.")

        config_path = tmp_path / "config.toml"
        with open(config_path, "rb") as f:
            saved = tomllib.load(f)
        assert (
            saved["image_generation"]["openrouter"]["default_model"]
            == "openai/gpt-5-image-mini"
        )

        cfg = get_image_generation_config(reload=True)
        assert cfg.openrouter_image_default_model == "openai/gpt-5-image-mini"

        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" not in str(rail_button.label)


@pytest.mark.asyncio
async def test_save_blocked_when_default_disabled(scratch_config, tmp_path):
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]
"""
    )
    config_path = tmp_path / "config.toml"
    before = config_path.read_text(encoding="utf-8")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        enabled_checkbox = panel.query_one(
            "#settings-imagegen-enabled-openrouter", Checkbox
        )
        enabled_checkbox.value = False
        await pilot.pause()

        await pilot.click("#settings-imagegen-save")
        await pilot.pause()
        await pilot.pause()

        assert "Default backend must be enabled" in _visible_text(screen)

    after = config_path.read_text(encoding="utf-8")
    assert after == before


@pytest.mark.asyncio
async def test_revert_discards_draft(scratch_config, tmp_path):
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]

[image_generation.openrouter]
default_model = "original-model"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        model_input = panel.query_one(
            "#settings-imagegen-field-openrouter-default_model", Input
        )
        model_input.value = "typed-but-not-saved"
        await pilot.pause()

        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" in str(rail_button.label)

        await pilot.click("#settings-imagegen-revert")
        await pilot.pause()
        await pilot.pause()

        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" not in str(rail_button.label)

        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        model_input = panel.query_one(
            "#settings-imagegen-field-openrouter-default_model", Input
        )
        assert model_input.value == "original-model"

    config_path = tmp_path / "config.toml"
    with open(config_path, "rb") as f:
        saved = tomllib.load(f)
    assert saved["image_generation"]["openrouter"]["default_model"] == "original-model"


@pytest.mark.asyncio
async def test_clear_key_deletes_not_blanks(scratch_config, tmp_path, monkeypatch):
    for var in _ALL_SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]

[image_generation.openrouter]
api_key = "sk-saved-key"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        source_line = panel.query_one(
            "#settings-imagegen-key-source-openrouter", Static
        )
        assert str(source_line.renderable) == "local config key saved"

        # `.press()` (not a coordinate pilot.click) -- this DestinationHarness
        # has no CSS_PATH, so the Clear button's compact-width TCSS classes
        # never load and it renders at Textual's raw default (min-width 16,
        # height 3), which can visually overlap the neighboring Impact pane
        # column at this terminal size. Posting Button.Pressed directly is
        # the same real message a click sends, without depending on pixel
        # geometry this harness doesn't style.
        panel.query_one(
            "#settings-imagegen-clear-openrouter-api_key", Button
        ).press()
        await pilot.pause()

        source_line = screen.query_one(
            "#settings-imagegen-key-source-openrouter", Static
        )
        assert str(source_line.renderable) == "missing"

        await pilot.click("#settings-imagegen-save")
        await _wait_for_settings_text(screen, pilot, "Image Gen defaults saved.")

    config_path = tmp_path / "config.toml"
    with open(config_path, "rb") as f:
        saved = tomllib.load(f)
    assert "api_key" not in saved["image_generation"].get("openrouter", {})


@pytest.mark.asyncio
async def test_pasted_key_saves_and_input_resets(scratch_config, tmp_path, monkeypatch):
    for var in _ALL_SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        key_input = panel.query_one(
            "#settings-imagegen-field-openrouter-api_key", Input
        )
        key_input.value = "sk-pasted-key"
        await pilot.pause()

        await pilot.click("#settings-imagegen-save")
        await _wait_for_settings_text(screen, pilot, "Image Gen defaults saved.")

        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        key_input = panel.query_one(
            "#settings-imagegen-field-openrouter-api_key", Input
        )
        assert key_input.value == ""

        source_line = panel.query_one(
            "#settings-imagegen-key-source-openrouter", Static
        )
        assert str(source_line.renderable) == "local config key saved"

    config_path = tmp_path / "config.toml"
    with open(config_path, "rb") as f:
        saved = tomllib.load(f)
    assert saved["image_generation"]["openrouter"]["api_key"] == "sk-pasted-key"


@pytest.mark.asyncio
async def test_unedited_baked_field_produces_empty_diff_on_save(
    scratch_config, tmp_path, monkeypatch
):
    """Pin (carry-note 1): a FRESH config (no ``[image_generation]`` section
    at all -- every backend field is baked-default-only) must NOT write any
    unedited field when the user saves after editing exactly ONE unrelated
    field. Guards against a regression that builds the save draft by
    reading ALL input values wholesale instead of tracking real edit
    events -- that shape of bug would spuriously persist baked defaults
    (e.g. ``default_backend``/``enabled_backends``, or another backend's
    already-blank field rendered as an empty string) that the user never
    touched.
    """
    for var in _ALL_SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    scratch_config(
        """
[general]
default_theme = "textual-dark"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        model_input = panel.query_one(
            "#settings-imagegen-field-openrouter-default_model", Input
        )
        model_input.value = "openai/gpt-5-image-mini"
        await pilot.pause()

        await pilot.click("#settings-imagegen-save")
        await _wait_for_settings_text(screen, pilot, "Image Gen defaults saved.")

    config_path = tmp_path / "config.toml"
    with open(config_path, "rb") as f:
        saved = tomllib.load(f)

    image_gen_section = saved.get("image_generation", {})
    assert (
        image_gen_section.get("openrouter", {}).get("default_model")
        == "openai/gpt-5-image-mini"
    )
    # Nothing the user never touched gets written -- neither the
    # baked-default global keys nor any other backend section, and no
    # stray empty-string fields alongside the one real edit.
    assert "default_backend" not in image_gen_section
    assert "enabled_backends" not in image_gen_section
    assert set(image_gen_section.keys()) == {"openrouter"}
    assert set(image_gen_section["openrouter"].keys()) == {"default_model"}


# ---------------------------------------------------------------------------
# Fix Round 1 (coordinator review): revert-shortcut recompose, Select
# refire guard, global int/float inline-error feedback.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revert_action_shortcut_recomposes_panel(scratch_config, tmp_path):
    """IMPORTANT fix: the footer `r` shortcut calls
    ``action_settings_revert_category`` directly (not the panel's own
    Revert button) -- before the fix, IMAGE_GENERATION fell through to the
    generic draft-pop-only path, which cleared the dirty marker without
    ever recomposing the panel, leaving the Input stuck showing the
    discarded unsaved text.
    """
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]

[image_generation.openrouter]
default_model = "original-model"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        model_input = panel.query_one(
            "#settings-imagegen-field-openrouter-default_model", Input
        )
        model_input.value = "typed-but-not-saved"
        await pilot.pause()

        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" in str(rail_button.label)

        # The same action the footer's `r` binding invokes -- NOT the
        # panel's own Revert button.
        screen.action_settings_revert_category()
        await pilot.pause()
        await pilot.pause()

        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" not in str(rail_button.label)

        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        model_input = panel.query_one(
            "#settings-imagegen-field-openrouter-default_model", Input
        )
        assert model_input.value == "original-model"

    config_path = tmp_path / "config.toml"
    with open(config_path, "rb") as f:
        saved = tomllib.load(f)
    assert saved["image_generation"]["openrouter"]["default_model"] == "original-model"


@pytest.mark.asyncio
async def test_default_backend_select_mount_refire_is_suppressed(scratch_config):
    """MINOR 1 fix: constructing the default-backend Select with its own
    current (non-blank) value fires a Select.Changed the moment it mounts
    (verified empirically -- a fresh Select's reactive default IS blank,
    so any non-blank initial value is a real change from Select's own
    point of view). `_queue_image_gen_select_suppression` records the
    exact value the about-to-(re)compose Select will mount with; the
    handler must consume-and-ignore exactly that one arrival. A queue
    entry left over after the mount settles would mean the refired
    message either never arrived as expected or was staged into the
    draft instead of being suppressed (and would incorrectly swallow a
    LATER, genuine user selection that happens to land on that same
    value).
    """
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        await pilot.pause()

        assert screen._image_gen_select_suppress_queue == []
        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" not in str(rail_button.label)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        assert panel.query_one("#settings-imagegen-save", Button).disabled


@pytest.mark.asyncio
async def test_invalid_global_int_field_blocks_save_with_inline_error(
    scratch_config, tmp_path
):
    """MINOR 2 fix: typing non-numeric text into a global int field (e.g.
    ``default_batch``) must behave like the per-backend int fields --
    inline error + no write -- instead of the edit silently vanishing with
    no feedback and (were it to reach diff_to_sections unguarded) writing
    the raw string into config.toml.
    """
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter"]
"""
    )
    config_path = tmp_path / "config.toml"
    before = config_path.read_text(encoding="utf-8")

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        batch_input = panel.query_one("#settings-imagegen-default_batch", Input)
        batch_input.value = "not-a-number"
        await pilot.pause()

        # Staged as dirty (feedback that something changed) rather than
        # silently dropped.
        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" in str(rail_button.label)

        await pilot.click("#settings-imagegen-save")
        await pilot.pause()
        await pilot.pause()

        assert "Default batch must be a whole number" in _visible_text(screen)

    after = config_path.read_text(encoding="utf-8")
    assert after == before
