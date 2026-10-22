"""Media Save acknowledges submitted bytes without replacing newer edits.

Mounted keyboard/Button.press journeys use the current admitted private config;
they do not qualify full production pointer geometry or external generation.
"""

import asyncio
import threading
import tomllib
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Collapsible, Input
from textual.widgets._collapsible import CollapsibleTitle

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _wait_for_settings_text,
)
from tldw_chatbook.UI.Screens import settings_screen as sm
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId


async def _entered(event):
    for _ in range(500):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    pytest.fail("The real configuration writer was never entered")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,backend", [("image", "openrouter"), ("video", "minimax")]
)
@pytest.mark.parametrize(
    "outcome", ["later", "moved", "restored", "unchanged", "failed"]
)
@private_profile_test
async def test_media_save_acknowledges_only_submitted_draft(
    request, monkeypatch, kind, backend, outcome
):
    from tldw_chatbook.config import get_cli_config_path

    category = f"{kind}_generation"
    category_id = SettingsCategoryId(category)
    prefix = f"settings-{kind}gen"
    mutation = sm.apply_settings_mutation_to_cli_config(
        {
            category: {"default_backend": backend, "enabled_backends": [backend]},
            f"{category}.{backend}": {"default_model": "original"},
        }
    )
    assert mutation.failure_phase is None
    sm.reset_image_generation_runtime()
    sm.reset_video_generation_runtime()
    entered = threading.Event()
    release = threading.Event()
    fail = outcome == "failed"
    load_entered = threading.Event()
    load_release = threading.Event()
    original_load = getattr(sm, f"get_{kind}_generation_config")

    def held_load(*args, **kwargs):
        if outcome == "later" and entered.is_set() and not load_entered.is_set():
            load_entered.set()
            assert load_release.wait(30)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(sm, f"get_{kind}_generation_config", held_load)
    if kind == "image":
        original = sm.apply_settings_mutation_to_cli_config

        def held(*args, **kwargs):
            entered.set()
            assert release.wait(30)
            if fail:
                return SimpleNamespace(file_replaced=False, failure_phase="write")
            return original(*args, **kwargs)

        monkeypatch.setattr(sm, "apply_settings_mutation_to_cli_config", held)
    else:
        original = sm.SettingsConfigAdapter.save_sections

        def held(self, *args, **kwargs):
            entered.set()
            assert release.wait(30)
            if fail:
                return False
            return original(self, *args, **kwargs)

        monkeypatch.setattr(sm.SettingsConfigAdapter, "save_sections", held)

    def persisted():
        saved = tomllib.loads(get_cli_config_path().read_text(encoding="utf-8"))
        return saved[category][backend]["default_model"]

    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, f"#settings-category-{category}")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert not screen._category_has_unsaved_changes(category_id)
        selector = f"#{prefix}-field-{backend}-default_model"
        field = screen.query_one(selector, Input)
        field.focus()
        await pilot.press("end", *"-submitted")
        await pilot.pause()
        assert field.value == "original-submitted"
        screen.query_one(f"#{prefix}-save", Button).press()
        await _entered(entered)
        try:
            if outcome != "unchanged":
                field.focus()
                await pilot.press("end", *"-new", "shift+left", "shift+left")
                await pilot.pause()
            if outcome == "restored":
                await pilot.press("home", "shift+end", "backspace", *"original")
                await pilot.pause()
                assert not screen._category_has_unsaved_changes(category_id)
            selection = field.selection
            if outcome == "moved":
                screen.query_one("#settings-category-appearance", Button).focus()
                await pilot.pause()
                moved_focus = host.focused
            expected = (
                "original-submitted"
                if outcome == "unchanged"
                else "original-submitted-new"
            )
            if outcome == "restored":
                expected = "original"
            assert field.value == expected
        finally:
            release.set()
        if outcome == "later":
            try:
                await _entered(load_entered)
                # The write has completed, but its real config reload is pending.
                field.focus()
                await pilot.press("end", *"-reload", "shift+left", "shift+left")
                await pilot.pause()
                expected = "original-submitted-new-reload"
                assert field.value == expected
                selection = field.selection
            finally:
                load_release.set()
        await host.workers.wait_for_complete()
        await pilot.pause()
        result = (
            f"Failed to save {kind.title()} Gen defaults."
            if fail
            else f"{kind.title()} Gen defaults saved."
        )
        await _wait_for_settings_text(screen, pilot, result, timeout=30.0)
        fresh = screen.query_one(selector, Input)
        assert persisted() == ("original" if fail else "original-submitted")
        assert fresh.value == expected
        assert screen._category_has_unsaved_changes(category_id) == (
            outcome != "unchanged"
        )
        if outcome == "moved":
            assert host.focused is moved_focus
        else:
            assert fresh.has_focus
            assert fresh.selection == selection
        if outcome == "unchanged":
            return
        # Retry and a second explicit Save both use the real checked writer.
        fail = False
        screen.query_one(f"#{prefix}-save", Button).press()
        # Button.press posts the message: let its worker start before joining.
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        await _wait_for_settings_text(
            screen, pilot, f"{kind.title()} Gen defaults saved.", timeout=30.0
        )
        assert persisted() == expected
        assert not screen._category_has_unsaved_changes(category_id)
        fresh = screen.query_one(selector, Input)
        fresh.focus()
        await pilot.press("end", *"-discard")
        await pilot.pause()
        assert screen._category_has_unsaved_changes(category_id)
        screen.query_one(f"#{prefix}-revert", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one(selector, Input).value == expected
        assert not screen._category_has_unsaved_changes(category_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,backend", [("image", "openrouter"), ("video", "minimax")]
)
@private_profile_test
async def test_media_save_keeps_expanded_nondefault_editor_context(
    request, kind, backend
):
    from tldw_chatbook.config import get_cli_config_path

    category = f"{kind}_generation"
    prefix = f"settings-{kind}gen"
    # The edited backend is enabled but is deliberately not the default.
    mutation = sm.apply_settings_mutation_to_cli_config(
        {
            category: {
                "default_backend": "stable_diffusion_cpp",
                "enabled_backends": ["stable_diffusion_cpp", backend],
            },
            f"{category}.{backend}": {"default_model": "original"},
        }
    )
    assert mutation.failure_phase is None
    getattr(sm, f"reset_{kind}_generation_runtime")()
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, f"#settings-category-{category}")
        await host.workers.wait_for_complete()
        await pilot.pause()
        editor_selector = f"#{prefix}-editor-{backend}"
        editor = screen.query_one(editor_selector, Collapsible)
        assert editor.collapsed
        editor.query_one(CollapsibleTitle).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not editor.collapsed
        selector = f"#{prefix}-field-{backend}-default_model"
        field = screen.query_one(selector, Input)
        field.focus()
        await pilot.press("end", *"-saved", "shift+left", "shift+left")
        await pilot.pause()
        assert field.is_on_screen
        selection = field.selection
        screen.query_one(f"#{prefix}-save", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await _wait_for_settings_text(
            screen, pilot, f"{kind.title()} Gen defaults saved.", timeout=30.0
        )
        fresh = screen.query_one(selector, Input)
        assert not screen.query_one(editor_selector, Collapsible).collapsed
        assert fresh.is_on_screen
        assert fresh.has_focus
        assert fresh.selection == selection
        assert fresh.value == "original-saved"
        saved = tomllib.loads(get_cli_config_path().read_text(encoding="utf-8"))
        assert saved[category][backend]["default_model"] == "original-saved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,backend", [("image", "openrouter"), ("video", "minimax")]
)
@private_profile_test
async def test_media_pending_clear_keeps_explicit_retyped_local_key(
    request, monkeypatch, kind, backend
):
    from tldw_chatbook.config import get_cli_config_path

    category = f"{kind}_generation"
    prefix = f"settings-{kind}gen"
    dummy_key = "explicit-local-test-value"
    mutation = sm.apply_settings_mutation_to_cli_config(
        {
            category: {"default_backend": backend, "enabled_backends": [backend]},
            f"{category}.{backend}": {"api_key": dummy_key},
        }
    )
    assert mutation.failure_phase is None
    getattr(sm, f"reset_{kind}_generation_runtime")()
    entered = threading.Event()
    release = threading.Event()
    original = (
        sm.apply_settings_mutation_to_cli_config
        if kind == "image"
        else sm.SettingsConfigAdapter.delete_values
    )

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return original(*args, **kwargs)

    if kind == "image":
        monkeypatch.setattr(sm, "apply_settings_mutation_to_cli_config", held)
    else:
        monkeypatch.setattr(sm.SettingsConfigAdapter, "delete_values", held)
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, f"#settings-category-{category}")
        await host.workers.wait_for_complete()
        await pilot.pause()
        selector = f"#{prefix}-field-{backend}-api_key"
        field = screen.query_one(selector, Input)
        assert field.value == ""  # Saved/borrowed keys never prefill an Input.
        screen.query_one(f"#{prefix}-clear-{backend}-api_key", Button).press()
        await pilot.pause()
        screen.query_one(f"#{prefix}-save", Button).press()
        await _entered(entered)
        try:
            field.focus()
            await pilot.press(*dummy_key)
            await pilot.pause()
            assert field.value == dummy_key
        finally:
            release.set()
        await host.workers.wait_for_complete()
        await _wait_for_settings_text(
            screen, pilot, f"{kind.title()} Gen defaults saved.", timeout=30.0
        )
        saved = tomllib.loads(get_cli_config_path().read_text(encoding="utf-8"))
        assert "api_key" not in saved[category][backend]
        fresh = screen.query_one(selector, Input)
        assert fresh.value == dummy_key
        assert screen._category_has_unsaved_changes(SettingsCategoryId(category))
        screen.query_one(f"#{prefix}-save", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await _wait_for_settings_text(
            screen, pilot, f"{kind.title()} Gen defaults saved.", timeout=30.0
        )
        await host.workers.wait_for_complete()
        await pilot.pause()
        saved = tomllib.loads(get_cli_config_path().read_text(encoding="utf-8"))
        assert saved[category][backend]["api_key"] == dummy_key
        assert not screen._category_has_unsaved_changes(SettingsCategoryId(category))


def test_equal_global_stage_keeps_explicit_media_field_history():
    from tldw_chatbook.UI.Settings_Modules.media_gen_save import (
        acknowledge_media_gen_save,
        stage_media_gen_value,
    )
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsDraft

    category = SettingsCategoryId.IMAGE_GENERATION
    field_key = "field::openrouter::api_key"
    cleared_key = "cleared::openrouter::api_key"
    draft = SettingsDraft(category=category)
    draft.set_value(field_key, "explicit-local-test-value", "explicit-local-test-value")
    submitted = SettingsDraft(category=category)
    submitted.set_value(cleared_key, False, True)
    screen = SimpleNamespace(
        _settings_drafts={category: draft},
        _update_draft_status_widgets=lambda category: None,
    )
    stage_media_gen_value(screen, category, "context_llm_enabled", False, False)
    acknowledge_media_gen_save(screen, category, submitted)
    assert (
        screen._settings_drafts[category].values[field_key]
        == "explicit-local-test-value"
    )
    assert screen._settings_drafts[category].is_dirty
