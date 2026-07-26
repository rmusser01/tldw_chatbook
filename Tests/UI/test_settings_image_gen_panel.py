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

import threading
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
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_image_gen_defaults import (
    ImageGenProbeResult,
    effective_placeholder,
)
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
async def test_save_revert_disabled_test_buttons_enabled(scratch_config):
    """Save/Revert stay disabled with nothing staged, but the six Test
    buttons (task 6: wired to `probe_backend`) are clickable regardless of
    dirty state -- a probe reads the CURRENT form values, so it is useful
    even with no unsaved edits at all."""
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
            assert not panel.query_one(
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
async def test_clear_swarmui_token_also_deletes_legacy_api_key(
    scratch_config, tmp_path, monkeypatch
):
    """Clear on swarmui's token must delete BOTH spellings: the loader
    resolves legacy ``api_key`` as a back-compat fallback, so leaving it
    behind would silently resurrect the credential after Clear+Save with
    no in-UI recovery (final-review residual ruling)."""
    for var in _ALL_SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    scratch_config(
        """
[image_generation]
default_backend = "swarmui"
enabled_backends = ["swarmui"]

[image_generation.swarmui]
swarm_token = "fake-current-token"
api_key = "fake-stale-legacy-token"
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        panel.query_one(
            "#settings-imagegen-clear-swarmui-swarm_token", Button
        ).press()
        await pilot.pause()

        await pilot.click("#settings-imagegen-save")
        await _wait_for_settings_text(screen, pilot, "Image Gen defaults saved.")

    config_path = tmp_path / "config.toml"
    with open(config_path, "rb") as f:
        saved = tomllib.load(f)
    swarmui_section = saved["image_generation"].get("swarmui", {})
    assert "swarm_token" not in swarmui_section
    assert "api_key" not in swarmui_section


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


# ---------------------------------------------------------------------------
# Task 6: Test button probe wiring.
#
# Every test here monkeypatches `probe_backend` at its settings_screen.py
# import site (`tldw_chatbook.UI.Screens.settings_screen.image_gen_probe_backend`
# -- the module-level name that module binds it to), never the real network/
# filesystem probe -- Task 3's probe_backend contract is covered by
# ``Tests/UI/test_settings_image_gen_defaults.py`` instead.
# ---------------------------------------------------------------------------

_PROBE_BACKEND_PATCH_TARGET = (
    "tldw_chatbook.UI.Screens.settings_screen.image_gen_probe_backend"
)

_ALL_BACKEND_IDS = (
    "stable_diffusion_cpp",
    "swarmui",
    "openrouter",
    "novita",
    "together",
    "modelstudio",
)


@pytest.mark.asyncio
async def test_probe_uses_current_form_values(scratch_config, monkeypatch):
    """Test gathers the CURRENT form values: an edited-but-unsaved Input
    wins over the saved config value, and a field the user never touched
    (timeout_seconds) falls back to the resolved effective value rather
    than an empty string."""
    scratch_config(
        """
[image_generation]
default_backend = "swarmui"
enabled_backends = ["swarmui"]

[image_generation.swarmui]
base_url = "http://original-host:7801"
"""
    )
    captured: dict[str, object] = {}

    def fake_probe_backend(backend_id, form_values, secret):
        captured["backend_id"] = backend_id
        captured["form_values"] = dict(form_values)
        captured["secret"] = secret
        return ImageGenProbeResult(ok=True, badge="Reachable")

    monkeypatch.setattr(_PROBE_BACKEND_PATCH_TARGET, fake_probe_backend)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        base_url_input = panel.query_one(
            "#settings-imagegen-field-swarmui-base_url", Input
        )
        base_url_input.value = "http://edited-not-saved:9999"
        await pilot.pause()

        # `.press()` (not a coordinate `pilot.click`) -- this
        # DestinationHarness has no CSS_PATH, so the Test button can land
        # outside the (unstyled, default-sized) scroll viewport at this
        # terminal size. Posting Button.Pressed directly is the same real
        # message a click sends, without depending on pixel geometry this
        # harness doesn't style (see test_clear_key_deletes_not_blanks for
        # the established precedent).
        screen.query_one("#settings-imagegen-test-swarmui", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    assert captured["backend_id"] == "swarmui"
    assert captured["form_values"]["base_url"] == "http://edited-not-saved:9999"
    # Untouched this session -- falls back to the resolved effective value,
    # never a blank string.
    assert captured["form_values"]["timeout_seconds"] == str(
        DEFAULT_SWARMUI_TIMEOUT_SECONDS
    )


@pytest.mark.asyncio
async def test_probe_secret_uses_pasted_unsaved_key(scratch_config, monkeypatch):
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
    captured: dict[str, object] = {}

    def fake_probe_backend(backend_id, form_values, secret):
        captured["secret"] = secret
        return ImageGenProbeResult(ok=True, badge="Reachable")

    monkeypatch.setattr(_PROBE_BACKEND_PATCH_TARGET, fake_probe_backend)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        key_input = panel.query_one(
            "#settings-imagegen-field-openrouter-api_key", Input
        )
        key_input.value = "sk-pasted-this-session"
        await pilot.pause()

        screen.query_one("#settings-imagegen-test-openrouter", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    # The pasted-but-unsaved value wins over the saved "sk-saved-key".
    assert captured["secret"] == "sk-pasted-this-session"


@pytest.mark.asyncio
async def test_probe_secret_falls_back_to_effective_when_nothing_pasted(
    scratch_config, monkeypatch
):
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
    captured: dict[str, object] = {}

    def fake_probe_backend(backend_id, form_values, secret):
        captured["secret"] = secret
        return ImageGenProbeResult(ok=True, badge="Reachable")

    monkeypatch.setattr(_PROBE_BACKEND_PATCH_TARGET, fake_probe_backend)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)

        screen.query_one("#settings-imagegen-test-openrouter", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    assert captured["secret"] == "sk-saved-key"


@pytest.mark.asyncio
async def test_probe_renders_badge_and_reenables_buttons(scratch_config, monkeypatch):
    """All six Test buttons disable for the duration of one probe (gated
    via a real `threading.Event` on the probe's own worker thread) and
    re-enable once it completes; the result badge lands on the right
    backend's status Static."""
    scratch_config(
        """
[image_generation]
default_backend = "swarmui"
enabled_backends = ["swarmui"]

[image_generation.swarmui]
base_url = "http://localhost:7801"
"""
    )
    gate = threading.Event()
    entered = threading.Event()

    def fake_probe_backend(backend_id, form_values, secret):
        entered.set()
        gate.wait(timeout=10.0)
        return ImageGenProbeResult(ok=True, badge="Reachable")

    monkeypatch.setattr(_PROBE_BACKEND_PATCH_TARGET, fake_probe_backend)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        screen.query_one("#settings-imagegen-test-swarmui", Button).press()

        for _ in range(100):
            await pilot.pause(0.02)
            if entered.is_set():
                break
        else:
            raise AssertionError("Probe worker never started.")

        for backend_id in _ALL_BACKEND_IDS:
            assert screen.query_one(
                f"#settings-imagegen-test-{backend_id}", Button
            ).disabled, f"{backend_id} Test button should be disabled mid-probe"

        gate.set()

        for _ in range(150):
            await pilot.pause(0.02)
            if not screen._image_gen_probe_in_flight:
                break
        else:
            raise AssertionError("Probe never completed after gate release.")
        await pilot.pause()

        for backend_id in _ALL_BACKEND_IDS:
            assert not screen.query_one(
                f"#settings-imagegen-test-{backend_id}", Button
            ).disabled, f"{backend_id} Test button should re-enable after the probe"

        badge = panel.query_one("#settings-imagegen-status-swarmui", Static)
        assert str(badge.renderable) == "Reachable"


@pytest.mark.asyncio
async def test_probe_escaped_exception_renders_probe_error_badge(
    scratch_config, monkeypatch
):
    """Any exception `probe_backend` fails to catch itself must degrade to
    the closed-set "Unreachable: probe error" badge -- never propagate
    exception text into the badge or a notify(), and buttons must still
    re-enable via the worker's `finally`."""
    scratch_config(
        """
[image_generation]
default_backend = "swarmui"
enabled_backends = ["swarmui"]

[image_generation.swarmui]
base_url = "http://localhost:7801"
"""
    )

    def fake_probe_backend(backend_id, form_values, secret):
        raise RuntimeError("leaked-exception-text-must-never-render")

    monkeypatch.setattr(_PROBE_BACKEND_PATCH_TARGET, fake_probe_backend)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        screen.query_one("#settings-imagegen-test-swarmui", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        badge = panel.query_one("#settings-imagegen-status-swarmui", Static)
        assert str(badge.renderable) == "Unreachable: probe error"
        assert "leaked-exception-text-must-never-render" not in _visible_text(screen)
        assert not screen.query_one(
            "#settings-imagegen-test-swarmui", Button
        ).disabled


@pytest.mark.asyncio
async def test_probe_badge_resets_on_category_reopen(scratch_config, monkeypatch):
    """Probe state is ephemeral -- leaving Image Gen and coming back must
    show the normal Configured/Not-configured badge again, never a probe
    result from the prior visit."""
    scratch_config(
        """
[image_generation]
default_backend = "swarmui"
enabled_backends = ["swarmui"]

[image_generation.swarmui]
base_url = "http://localhost:7801"
"""
    )

    def fake_probe_backend(backend_id, form_values, secret):
        return ImageGenProbeResult(ok=True, badge="Reachable")

    monkeypatch.setattr(_PROBE_BACKEND_PATCH_TARGET, fake_probe_backend)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        screen.query_one("#settings-imagegen-test-swarmui", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        badge = panel.query_one("#settings-imagegen-status-swarmui", Static)
        assert str(badge.renderable) == "Reachable"

        await _open_settings_category(pilot, "#settings-category-appearance")
        await _open_image_gen(pilot)

        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        badge = panel.query_one("#settings-imagegen-status-swarmui", Static)
        assert str(badge.renderable) == "Configured"


# ---------------------------------------------------------------------------
# Final review fix round: swarmui token config key, empty-value deletion
# semantics, stale Task-4 copy, enabled_backends order, global min-clamps.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emptying_a_saved_field_deletes_not_blanks(scratch_config, tmp_path):
    """Important 1, end-to-end: selecting-all-and-deleting a saved field's
    text (not the dedicated Clear button, which only exists for secrets)
    must delete the key from config.toml, never write "" over it.
    """
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
        model_input.value = ""
        await pilot.pause()

        rail_button = screen.query_one("#settings-category-image_generation", Button)
        assert "*" in str(rail_button.label)

        await pilot.click("#settings-imagegen-save")
        await _wait_for_settings_text(screen, pilot, "Image Gen defaults saved.")

    config_path = tmp_path / "config.toml"
    with open(config_path, "rb") as f:
        saved = tomllib.load(f)
    assert "default_model" not in saved["image_generation"].get("openrouter", {})


@pytest.mark.asyncio
async def test_swarmui_token_saves_and_resolves_end_to_end(scratch_config, tmp_path, monkeypatch):
    """CRITICAL fix, end-to-end: a pasted swarmui token must both persist
    to config.toml AND actually resolve (key_sources == "config", input
    resets, source line updates) through the real save -> cache-reset ->
    recompose path -- not just at the loader-unit-test level.
    """
    for var in _ALL_SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    scratch_config(
        """
[image_generation]
default_backend = "swarmui"
enabled_backends = ["swarmui"]
"""
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_image_gen(pilot)
        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)

        token_input = panel.query_one(
            "#settings-imagegen-field-swarmui-swarm_token", Input
        )
        token_input.value = "fake-swarm-token"
        await pilot.pause()

        await pilot.click("#settings-imagegen-save")
        await _wait_for_settings_text(screen, pilot, "Image Gen defaults saved.")

        panel = screen.query_one("#settings-imagegen-panel", ImageGenSettingsPanel)
        token_input = panel.query_one(
            "#settings-imagegen-field-swarmui-swarm_token", Input
        )
        assert token_input.value == ""

        source_line = panel.query_one(
            "#settings-imagegen-key-source-swarmui", Static
        )
        assert str(source_line.renderable) == "local config key saved"

    config_path = tmp_path / "config.toml"
    with open(config_path, "rb") as f:
        saved = tomllib.load(f)
    assert saved["image_generation"]["swarmui"]["swarm_token"] == "fake-swarm-token"

    cfg = get_image_generation_config(reload=True)
    assert cfg.swarmui_swarm_token == "fake-swarm-token"
    assert cfg.key_sources["swarmui"] == "config"


# ---------------------------------------------------------------------------
# Live-CSS layout regression (whole-branch review): DestinationHarness (used
# by every other test in this file) has no CSS_PATH, so it structurally
# cannot exercise the real app-tier TCSS bundle -- it never caught that a
# bare `Checkbox { width: 100%; height: 2; }` type selector living in
# _conversations.tcss (unscoped, app-wide) was stretching every Enabled
# checkbox to its row's full width, clipping the default marker + Test
# button off the visible/clickable area entirely, or that the whole panel's
# implicit `height: 1fr` (Vertical's own default, uncontested by any
# `#settings-imagegen-panel` rule) starved its nested backend-editor
# VerticalScroll down to ~1 row, hiding Backend settings/Generation
# defaults/Save/Revert below the fold with nothing able to scroll them into
# view. These two tests run against the REAL `TldwCli` app directly (the
# same pattern `test_screen_navigation.py`'s real-app tests use) specifically
# so the real CSS bundle is exercised -- confirmed live via a tmux capture
# at 235 and 120 columns before writing this pin (see task-5-report.md).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(235, 52), (120, 45)])
@pytest.mark.asyncio
async def test_backend_row_controls_visible_and_clickable_under_real_css(
    scratch_config, size
):
    scratch_config(
        """
[image_generation]
default_backend = "openrouter"
enabled_backends = ["openrouter", "swarmui"]
"""
    )
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        # The splash screen (1.5s) must close before handle_screen_
        # navigation's switch_screen call is valid -- calling it while the
        # splash is still the top screen raises IndexError from Textual's
        # own result-callback bookkeeping.
        await pilot.pause(2.0)
        await app.handle_screen_navigation(NavigateToScreen("settings"))
        await pilot.pause(0.3)
        screen = app.screen
        screen._select_category("image_generation")
        await pilot.pause(0.2)

        for backend_id in (
            "stable_diffusion_cpp",
            "swarmui",
            "openrouter",
            "novita",
            "together",
            "modelstudio",
        ):
            row = screen.query_one(f"#settings-imagegen-backend-{backend_id}")
            test_btn = screen.query_one(f"#settings-imagegen-test-{backend_id}", Button)
            cb = screen.query_one(
                f"#settings-imagegen-enabled-{backend_id}", Checkbox
            )
            row_right = row.region.x + row.region.width
            test_right = test_btn.region.x + test_btn.region.width
            assert test_right <= row_right, (
                f"{backend_id} @ {size}: Test button's right edge "
                f"({test_right}) is past its row's right edge ({row_right})"
            )
            assert cb.region.width < row.region.width, (
                f"{backend_id} @ {size}: Enabled checkbox claimed the "
                "row's full width (width:100% leaking in from elsewhere)"
            )
            widget, _offset = screen.screen.get_widget_at(*test_btn.region.center)
            assert widget is test_btn, (
                f"{backend_id} @ {size}: something else intercepts a click "
                f"at Test's own center point (got {widget!r} instead)"
            )

        editor = screen.query_one("#settings-imagegen-editor")
        assert editor.region.height == editor.virtual_size.height, (
            f"@ {size}: Backend settings editor region "
            f"({editor.region.height}) is shorter than its content "
            f"({editor.virtual_size.height}) -- starved by a competing "
            "1fr sibling"
        )

        save_btn = screen.query_one("#settings-imagegen-save", Button)
        save_btn.scroll_visible(animate=False)
        await pilot.pause(0.2)
        widget, _offset = screen.screen.get_widget_at(*save_btn.region.center)
        assert widget is save_btn, (
            f"@ {size}: Save button not reachable/clickable via scroll"
        )
