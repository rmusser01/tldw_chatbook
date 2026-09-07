"""TASK-1341: unify Settings save commit models.

Three commit models coexisted with no way to tell them apart: staged s/r
(guided categories), instant-persist toggles (model-catalog auto-refresh),
and editor-owned saves (Theme, Splash). UAT showed the confusion path:
toggle an auto-save checkbox, press ``s`` out of habit, get "no changes to
save" — the user cannot answer "did that save?".

These tests pin:
* staged stays the default model for the guided categories;
* every intentional instant-apply control is labeled inline
  ("applies immediately - no Save needed") and visually separated from
  staged fields (its own bordered group);
* the focused-field inspector documents the per-field save behavior
  ("Save: staged - press s to save, r to revert" vs
  "Save: applies immediately - no Save needed").
"""

import pytest
from textual.widgets import Checkbox, Input, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _visible_text,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import _click_settings_category
from tldw_chatbook.UI.Screens.settings_screen import (
    MODEL_CATALOG_FIELD_IDS,
    SettingsScreen,
)

INSTANT_APPLY_LABEL = "applies immediately - no Save needed"
STAGED_SAVE_ROW = "Save: staged - press s to save, r to revert"
INSTANT_SAVE_ROW = f"Save: {INSTANT_APPLY_LABEL}"


@pytest.mark.asyncio
async def test_model_catalog_controls_are_labeled_and_visually_separated():
    """Providers pane: the auto-refresh block carries the inline instant-apply
    label and lives in its own bordered group, distinct from staged fields."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(160, 50)) as pilot:
        await _click_settings_category(pilot, "providers-models")
        screen = _active_destination_screen(host)

        hint = screen.query_one("#settings-model-catalog-instant-hint", Static)
        assert INSTANT_APPLY_LABEL in str(hint.content)
        assert hint.has_class("settings-instant-apply-hint")

        group = screen.query_one("#settings-model-catalog-group")
        assert group.has_class("settings-instant-apply-group")
        # The instant controls live inside the separated group...
        assert group.query("#settings-model-catalog-auto-refresh")
        assert group.query("#settings-model-catalog-stale-hours")
        # ...while staged Connect fields stay outside of it.
        assert not group.query("#settings-provider-endpoint-value")


@pytest.mark.asyncio
async def test_staged_fields_document_staged_save_in_the_inspector():
    """AC3: every staged field's inspector rows name the staged model."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(160, 50)) as pilot:
        await _click_settings_category(pilot, "providers-models")
        screen = _active_destination_screen(host)

        screen.query_one("#settings-provider-endpoint-value", Input).focus()
        await pilot.pause()
        text = _visible_text(screen)
        assert "Focused setting: Endpoint" in text
        assert STAGED_SAVE_ROW in text

        await _click_settings_category(pilot, "appearance")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-appearance-font-size", Input).focus()
        await pilot.pause()
        assert STAGED_SAVE_ROW in _visible_text(screen)

        await _click_settings_category(pilot, "storage")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-storage-chachanotes-db-path", Input).focus()
        await pilot.pause()
        assert STAGED_SAVE_ROW in _visible_text(screen)


@pytest.mark.asyncio
async def test_console_display_name_documents_staged_save_in_the_inspector():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(160, 50)) as pilot:
        await _click_settings_category(pilot, "console-behavior")
        screen = _active_destination_screen(host)
        screen.query_one(
            "#settings-console-default-user-display-name", Input
        ).focus()
        await pilot.pause()

        text = _visible_text(screen)
        assert "Purpose: Default speaker label for chats without a per-chat override." in text
        assert "Saved as: chat_defaults." in text
        assert "user_display_name" in text
        assert STAGED_SAVE_ROW in text


@pytest.mark.asyncio
async def test_model_catalog_fields_document_instant_apply_in_the_inspector():
    """AC3: focusing an auto-refresh toggle says it applies immediately."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(160, 50)) as pilot:
        await _click_settings_category(pilot, "providers-models")
        screen = _active_destination_screen(host)

        screen.query_one("#settings-model-catalog-auto-refresh", Checkbox).focus()
        await pilot.pause()
        assert INSTANT_SAVE_ROW in _visible_text(screen)


@pytest.mark.asyncio
async def test_splash_category_labels_its_instant_save_model():
    """Splash auto-save stays, but is labeled inline and in the inspector."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(160, 50)) as pilot:
        await _click_settings_category(pilot, "splash_screen")
        screen = _active_destination_screen(host)

        hint = screen.query_one("#settings-splash-instant-hint", Static)
        assert INSTANT_APPLY_LABEL in str(hint.content)

        text = _visible_text(screen)
        assert INSTANT_SAVE_ROW in text


@pytest.mark.asyncio
async def test_theme_category_documents_its_editor_owned_save_model():
    """Theme keeps its editor-owned Apply/Save/Reset flow; the inspector
    names that model explicitly."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(160, 50)) as pilot:
        await _click_settings_category(pilot, "theme")
        screen = _active_destination_screen(host)

        text = _visible_text(screen)
        assert "Save: editor-owned" in text
        assert "Apply/Save/Reset" in text


def test_guidance_row_builders_keep_a_uniform_row_count():
    """Every guidance branch returns the same row count per category.

    The inspector composes one Static per row index and later refreshes
    them by position (see ``_with_save_behavior_row``'s docstring), so a
    future branch returning a different count would silently drop or
    misalign rows. This is the cheap guard for that contract.
    """
    screen = SettingsScreen(_build_test_app())
    provider_field_ids = (
        "settings-provider-value",
        "settings-provider-manual-value",
        "settings-model-value",
        "settings-provider-endpoint-value",
        "settings-provider-api-mode",
        "settings-provider-api-key",
        "settings-provider-credential-env-var",
        "settings-model-profile-temperature",
        "settings-model-profile-top-p",
        "settings-model-profile-min-p",
        "settings-model-profile-top-k",
        "settings-model-profile-max-tokens",
        "settings-model-profile-seed",
        "settings-model-profile-presence-penalty",
        "settings-model-profile-frequency-penalty",
        "settings-model-profile-reasoning-effort",
        "settings-model-profile-reasoning-summary",
        "settings-model-profile-verbosity",
        "settings-model-profile-thinking-effort",
        "settings-model-profile-thinking-budget-tokens",
        "settings-model-profile-streaming",
    )
    appearance_field_ids = (
        "settings-appearance-theme",
        "settings-appearance-palette-theme-limit",
        "settings-appearance-font-size",
        "settings-appearance-density",
        "settings-appearance-animations-enabled",
        "settings-appearance-smooth-scrolling",
    )
    storage_field_ids = (
        "settings-storage-user-db-base-dir",
        "settings-storage-chachanotes-db-path",
        "settings-storage-prompts-db-path",
        "settings-storage-media-db-path",
        "settings-storage-research-db-path",
        "settings-storage-writing-db-path",
        "settings-storage-library-collections-db-path",
        "settings-storage-workspaces-db-path",
    )

    # 4 content rows + the commit-model row inserted by _with_save_behavior_row.
    expected_count = 5
    counts = set()
    for field_id in (None, *provider_field_ids, *MODEL_CATALOG_FIELD_IDS):
        screen._active_settings_field_id = field_id
        counts.add(len(screen._provider_field_guidance_rows()))
    assert counts == {expected_count}

    counts = set()
    for field_id in (None, *appearance_field_ids):
        screen._active_settings_field_id = field_id
        counts.add(len(screen._appearance_field_guidance_rows()))
    assert counts == {expected_count}

    counts = set()
    for field_id in (None, *storage_field_ids):
        screen._active_settings_field_id = field_id
        counts.add(len(screen._storage_field_guidance_rows()))
    assert counts == {expected_count}


@pytest.mark.asyncio
async def test_model_catalog_stale_hours_has_its_own_inspector_branch():
    """The stale-hours Input is not lumped under the checkbox copy: it gets
    its own field name and numeric validation text."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(160, 50)) as pilot:
        await _click_settings_category(pilot, "providers-models")
        screen = _active_destination_screen(host)

        screen.query_one("#settings-model-catalog-stale-hours", Input).focus()
        await pilot.pause()
        text = _visible_text(screen)
        assert "Focused setting: Refresh after (hours)" in text
        # task-1716 folds long tokens at separators for the narrow inspector
        # rail, so assert the key in two fold-tolerant parts (same pattern as
        # the ollama Saved-as check in test_settings_configuration_hub.py).
        assert "Saved as: model_catalog." in text
        assert "stale_after_hours" in text
        assert "number of hours, 0 or greater" in text
        assert INSTANT_SAVE_ROW in text
