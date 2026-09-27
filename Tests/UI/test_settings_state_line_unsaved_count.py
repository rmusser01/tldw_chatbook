"""The dirty State line keeps its save-model badge and counts unsaved fields.

TASK-33002.4: with unsaved edits the State line used to drop ADR-033's
persistence badge for "State: Unsaved changes | ...", and nothing said how
much was unsaved. It now reads "State: {badge} · N unsaved | ..." where N is
the number of fields that differ from their saved values.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Input, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    _active_destination_screen,
    _build_test_app,
    _static_text,
    _wait_for_selector,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)
from tldw_chatbook.UI.Screens.settings_screen import (
    GUIDED_SETTINGS_MUTATION_CATEGORIES,
    SettingsScreen,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import load_global_speech_tts_state
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_panel_types import (
    SpeechTTSPanelDraftSnapshot,
    _RealtimeSettingsDraft,
)

_REALTIME = _RealtimeSettingsDraft(
    False,
    "openai",
    "gpt-realtime",
    "",
    "30",
    "auto",
    "semantic_vad",
    "0.5",
    "500",
    "",
    False,
)
_STAGING_CATEGORIES = (
    *sorted(GUIDED_SETTINGS_MUTATION_CATEGORIES, key=lambda c: c.value),
    SettingsCategoryId.IMAGE_GENERATION,
    SettingsCategoryId.VIDEO_GENERATION,
    SettingsCategoryId.ADVANCED_CONFIG,
)


def _speech_snapshot(**changes) -> SpeechTTSPanelDraftSnapshot:
    original = load_global_speech_tts_state({}, environment={})
    state = load_global_speech_tts_state({}, environment={})
    for provider, values in changes.pop("providers", {}).items():
        state.providers[provider].update(values)
    if "speed" in changes:
        state.defaults.speed = changes.pop("speed")
    return SpeechTTSPanelDraftSnapshot(
        state=state,
        original_state=original,
        realtime_draft=replace(_REALTIME, **changes),
        realtime_original=replace(_REALTIME),
        configure_provider="kokoro",
        draft_revision=1,
    )


def test_speech_snapshot_counts_fields_that_differ_from_saved() -> None:
    """Validated values compare, so text that parses to the saved value is clean."""
    assert _speech_snapshot().unsaved_field_count() == 0
    snapshot = _speech_snapshot(
        speed=1.5,
        providers={
            "kokoro": {"max_tokens": 600},
            # Collected from an Input as text; validates to the saved 0.5.
            "elevenlabs": {"stability": "0.5"},
        },
        idle_timeout_minutes="45",
    )
    assert snapshot.unsaved_field_count() == 3


@pytest.mark.asyncio
@private_profile_test
async def test_console_behavior_line_keeps_badge_and_counts_fields(request) -> None:
    screen = SettingsScreen(_build_test_app())
    category = SettingsCategoryId.CONSOLE_BEHAVIOR
    draft = SettingsDraft(category=category)
    draft.set_value("paste_collapse_threshold", 50, 503)
    draft.set_value("max_parallel_runs", 2, 3)
    screen._settings_drafts[category] = draft
    scope = "Changes affect global Console fallbacks after save."

    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s · 2 unsaved · revert with r | {scope}"
    )
    draft.set_value("max_parallel_runs", 2, 2)  # reverted back by hand
    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s · 1 unsaved · revert with r | {scope}"
    )


@pytest.mark.asyncio
@private_profile_test
async def test_providers_models_counts_fields_not_draft_keys(request) -> None:
    """The context window and its reset flag are one field; snapshot
    preferences are two fields that live outside the draft."""
    screen = SettingsScreen(_build_test_app())
    category = SettingsCategoryId.PROVIDERS_MODELS
    draft = SettingsDraft(category=category)
    draft.set_value("provider", "openai", "openai")
    draft.set_value("model_context_window", "", "32768")
    draft.set_value("model_context_window_reset", False, True)
    screen._settings_drafts[category] = draft
    scope = screen._category_state_scope_text(category)

    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s · 1 unsaved | {scope}"
    )
    draft.set_value("model_profile_temperature", "", "0.4")
    loaded = type("Loaded", (), {"enabled": True, "keep_count": 5})()
    screen._snapshot_preferences_loaded = loaded
    screen._snapshot_preferences_raw = (False, "7")
    assert screen._category_state_banner_text(category) == (
        f"State: Draft — save with s · 4 unsaved | {scope}"
    )


@pytest.mark.asyncio
@private_profile_test
async def test_other_staging_lines_keep_their_own_badge(request) -> None:
    screen = SettingsScreen(_build_test_app())
    image = SettingsCategoryId.IMAGE_GENERATION
    draft = SettingsDraft(category=image)
    draft.set_value("default_backend", "a", "b")
    screen._settings_drafts[image] = draft
    assert screen._category_state_banner_text(image) == (
        "State: Draft — save/revert below · 1 unsaved | "
        "Defaults affect future Console image generations."
    )

    advanced = SettingsCategoryId.ADVANCED_CONFIG
    screen._category_has_unsaved_changes = lambda category: category is advanced
    assert screen._category_state_banner_text(advanced) == (
        "State: Validate, then Save · 1 unsaved | Draft kept when you leave; "
        "use raw editor controls."
    )


@pytest.mark.asyncio
@private_profile_test
async def test_categories_without_a_draft_keep_todays_badge(request) -> None:
    """AC#4: the clean line is "State: {badge} | {scope}", as before."""
    screen = SettingsScreen(_build_test_app())
    expected = {
        SettingsCategoryId.NETWORK: "Pending — save with s",
        SettingsCategoryId.WORKSPACES: "Applies immediately",
        SettingsCategoryId.SPLASH_SCREEN: "Auto-saved",
        SettingsCategoryId.INTERNAL_PROMPTS: "Per-item Save/Reset",
        SettingsCategoryId.CONSOLE_BEHAVIOR: "Draft — save with s",
    }
    for category, badge in expected.items():
        text = screen._category_state_banner_text(category)
        assert text == f"State: {badge} | {screen._category_state_scope_text(category)}"
        assert "unsaved" not in text


@pytest.mark.asyncio
@private_profile_test
async def test_dirty_state_line_renders_without_a_new_row_at_211x44(request) -> None:
    """AC#1/AC#2 on the real banner (production stylesheet, 211x44).

    Every staging category's worst-case dirty line (99 unsaved) renders in no
    more rows than its clean line, and a real Console Behavior edit counts up
    and back down as fields are edited and reverted by hand.
    """
    host = _SettingsCssHarness(_build_test_app(), "settings")
    async with host.run_test(size=(211, 44)) as pilot:
        screen = _active_destination_screen(host)
        category = SettingsCategoryId.CONSOLE_BEHAVIOR
        screen._select_category(category.value)
        await _wait_for_selector(
            screen, pilot, "#settings-console-paste-collapse-threshold"
        )
        banner = screen.query_one("#settings-category-state-banner", Static)
        assert banner.content_region.width == 120

        async def rows(text: str) -> int:
            banner.update(text)
            await pilot.pause()
            return banner.content_region.height

        real_dirty = screen._category_has_unsaved_changes
        real_count = screen._category_unsaved_count
        for staged in _STAGING_CATEGORIES:
            clean = screen._category_state_banner_text(staged)
            screen._category_has_unsaved_changes = lambda c, staged=staged: c is staged
            screen._category_unsaved_count = lambda c: 99
            dirty = screen._category_state_banner_text(staged)
            screen._category_has_unsaved_changes = real_dirty
            screen._category_unsaved_count = real_count
            assert "· 99 unsaved" in dirty, dirty
            assert await rows(dirty) <= await rows(clean), (staged, dirty)
        banner.update(screen._category_state_banner_text(category))

        threshold = screen.query_one(
            "#settings-console-paste-collapse-threshold", Input
        )
        runs = screen.query_one("#settings-console-max-parallel-runs", Input)
        saved_threshold, saved_runs = threshold.value, runs.value
        threshold.value = str(int(saved_threshold or 0) + 7)
        await pilot.pause()
        assert "· 1 unsaved · revert with r | " in _static_text(banner)
        runs.value = str(int(saved_runs or 1) + 1)
        await pilot.pause()
        assert "· 2 unsaved · revert with r | " in _static_text(banner)
        threshold.value = saved_threshold
        await pilot.pause()
        assert "· 1 unsaved · revert with r | " in _static_text(banner)
        assert banner.content_region.height == 1
        runs.value = saved_runs
        await pilot.pause()
        assert _static_text(banner) == (
            "State: Draft — save with s | "
            "Changes affect global Console fallbacks after save."
        )
