"""Canonical Hooks settings stage, save and review the actual saved config."""

import asyncio

import pytest
import toml

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

pytestmark = pytest.mark.bootstrap_profile

hook_file = _hook_file


@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_hooks_edit_is_staged_until_canonical_save(hook_file, size):
    raw = toml.loads(hook_file.read_text())
    raw["hooks"]["hook"][0].pop("timeout_s")
    hook_file.write_text(toml.dumps(raw))
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-enabled")
        assert not screen._category_has_unsaved_changes(SettingsCategoryId.HOOKS)
        assert screen.query_one("#settings-hooks-timeout").value == "10.0"
        assert screen._ownership_record(
            SettingsCategoryId.HOOKS
        ).owns_config_sections == ("hooks",)
        from tldw_chatbook.UI.Screens.settings_search_index import FIELD_SEARCH_INDEX

        for field_id, _label in FIELD_SEARCH_INDEX[SettingsCategoryId.HOOKS]:
            assert screen.query(f"#{field_id}"), field_id
        before = hook_file.read_bytes()
        await host.workers.wait_for_complete()
        await pilot.pause()
        checkbox = screen.query_one("#settings-hooks-enabled")
        checkbox.scroll_visible(animate=False, top=True, immediate=True)
        await pilot.pause()
        offset = (2, min(1, checkbox.region.height - 1))
        point = checkbox.region.offset + offset
        hit, _ = host.get_widget_at(*point)
        assert hit is checkbox, (
            checkbox.region,
            point,
            type(hit).__name__,
            hit.id,
            checkbox.parent.region,
            checkbox.parent.parent.region,
            screen.query_one("#settings-detail-pane-body").region,
        )
        assert await pilot.click(checkbox, offset=offset)
        await pilot.pause()
        assert hook_file.read_bytes() == before
        assert screen._category_has_unsaved_changes(SettingsCategoryId.HOOKS)
        screen.query_one("#settings-hooks-save").focus()
        await pilot.press("enter")
        await host.workers.wait_for_complete()
        await pilot.pause()
        saved = toml.loads(hook_file.read_text())["hooks"]
        assert saved["enabled"] is False
        assert "timeout_s" not in saved["hook"][0]
        assert not screen._category_has_unsaved_changes(SettingsCategoryId.HOOKS)


async def test_saved_review_is_available_before_console_mount(hook_file):
    from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
        ConsoleHooksReviewModal,
    )

    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-enabled")
        button = screen.query_one("#settings-hooks-review")
        button.focus()
        await pilot.press("enter")
        await _wait_for_selector(host.screen, pilot, "#console-hooks-review", timeout=5)
        assert isinstance(host.screen, ConsoleHooksReviewModal)
        await pilot.press("escape")
        await pilot.pause()
        assert host.screen is screen


async def test_editor_preserves_unknown_fields_and_rejects_invalid_argv(hook_file):
    from textual.widgets import TextArea

    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.UI.Screens.settings_hooks import HooksSettingsPanel

    raw = toml.loads(hook_file.read_text())
    raw["hooks"]["future"] = {"keep": True}
    raw["hooks"]["hook"][0]["future_row"] = "keep"
    hook_file.write_text(toml.dumps(raw))
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-command")
        panel = screen.query_one(HooksSettingsPanel)
        editor = panel.query_one("#settings-hooks-command", TextArea)
        editor.load_text('"shell command"')
        await pilot.pause()
        with pytest.raises(ValueError, match="Command"):
            panel.submission()
        before = hook_file.read_bytes()
        screen.query_one("#settings-hooks-save").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert hook_file.read_bytes() == before
        editor.load_text('["python3", "-c", "pass", "a b", "[bold]"]')
        await pilot.pause()
        # Native buttons ignore a second activation during their pressed effect.
        async with asyncio.timeout(3):
            while screen.query_one("#settings-hooks-save").has_class("-active"):
                await pilot.pause(0.01)
        screen.query_one("#settings-hooks-save").focus()
        await pilot.press("enter")
        await host.workers.wait_for_complete()
        await pilot.pause()
        saved = toml.loads(hook_file.read_text())["hooks"]
        assert saved["future"] == {"keep": True}
        assert saved["hook"][0]["future_row"] == "keep"
        assert saved["hook"][0]["command"][-2:] == ["a b", "[bold]"]
        assert not HookPermissions().snapshot().ready


async def test_draft_survives_category_navigation_and_revert_reloads_saved_config(
    hook_file,
):
    from textual.widgets import Input

    from tldw_chatbook.UI.Screens.settings_hooks import HooksSettingsPanel

    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-timeout")
        panel = screen.query_one(HooksSettingsPanel)
        panel.query_one("#settings-hooks-timeout", Input).value = "12"
        await pilot.pause()
        screen.apply_navigation_context({"category": "appearance"})
        await pilot.pause()
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-timeout")
        assert screen.query_one("#settings-hooks-timeout", Input).value == "12.0"
        raw = toml.loads(hook_file.read_text())
        raw["hooks"]["hook"][0]["timeout_s"] = 8
        hook_file.write_text(toml.dumps(raw))
        before = hook_file.read_bytes()
        screen._start_hooks_save()
        await host.workers.wait_for_complete()
        assert hook_file.read_bytes() == before
        assert screen._category_has_unsaved_changes(SettingsCategoryId.HOOKS)
        screen._revert_category(SettingsCategoryId.HOOKS)
        await host.workers.wait_for_complete()
        await _wait_for_selector(screen, pilot, "#settings-hooks-timeout")
        assert screen.query_one("#settings-hooks-timeout", Input).value == "8"
        assert not screen._category_has_unsaved_changes(SettingsCategoryId.HOOKS)


async def test_add_edit_toggle_remove_uses_the_shared_settings_draft(hook_file):
    from textual.widgets import Checkbox, TextArea

    from tldw_chatbook.UI.Screens.settings_hooks import HooksSettingsPanel

    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-add")
        panel = screen.query_one(HooksSettingsPanel)
        before = hook_file.read_bytes()
        panel.query_one("#settings-hooks-add").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert len(panel.section["hook"]) == 2
        assert panel.query_one("#settings-hooks-list").highlighted == 1
        assert panel.section["hook"][1]["enabled"] is False
        panel.query_one("#settings-hooks-command", TextArea).load_text(
            '["python3", "-c", "pass"]'
        )
        panel.query_one("#settings-hooks-row-enabled", Checkbox).value = True
        await pilot.pause()
        section, _mapping = panel.submission()
        assert section["hook"][1]["enabled"] is True
        assert hook_file.read_bytes() == before
        screen._start_hooks_save()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert len(toml.loads(hook_file.read_text())["hooks"]["hook"]) == 2
        panel.query_one("#settings-hooks-remove").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert len(panel.section["hook"]) == 1
        screen._start_hooks_save()
        await host.workers.wait_for_complete()
        assert len(toml.loads(hook_file.read_text())["hooks"]["hook"]) == 1


async def test_save_completion_preserves_a_newer_hook_edit(hook_file, monkeypatch):
    import threading

    from textual.widgets import Input

    from tldw_chatbook.UI.Screens.settings_hooks import HooksSettingsPanel

    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-timeout")
        panel = screen.query_one(HooksSettingsPanel)
        panel.query_one("#settings-hooks-timeout", Input).value = "6"
        await pilot.pause()
        owner = screen._hooks_owner()
        original = owner.save_configuration
        started, release = threading.Event(), threading.Event()

        def paused_save(*args, **kwargs):
            started.set()
            assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, "save_configuration", paused_save)
        screen._start_hooks_save()
        try:
            assert await asyncio.to_thread(started.wait, 3)
            panel.query_one("#settings-hooks-timeout", Input).value = "9"
            await pilot.pause()
            release.set()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert (
                toml.loads(hook_file.read_text())["hooks"]["hook"][0]["timeout_s"] == 6
            )
            assert panel.draft.values["section"]["hook"][0]["timeout_s"] == 9
            assert panel.query_one("#settings-hooks-timeout", Input).value == "9"
            assert screen._category_has_unsaved_changes(SettingsCategoryId.HOOKS)
        finally:
            release.set()


@pytest.mark.parametrize("dirty", [False, True])
async def test_saved_hook_reload_updates_clean_draft_and_preserves_dirty_draft(
    hook_file, dirty
):
    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.UI.Screens.settings_hooks import HooksSettingsPanel

    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-timeout")
        panel = screen.query_one(HooksSettingsPanel)
        if dirty:
            panel.query_one("#settings-hooks-timeout").value = "12"
            await pilot.pause()
        owner = HookPermissions()
        saved = owner.snapshot()
        owner.disable(saved, saved.rows[0].entry.key)
        await screen._load_hooks_settings()
        await pilot.pause()
        assert panel.section["hook"][0].get("enabled", True) is dirty
        assert panel.draft.is_dirty is dirty
        if dirty:
            assert panel.section["hook"][0]["timeout_s"] == 12
        else:
            assert not panel.query_one("#settings-hooks-row-enabled").value
        review = panel.query_one("#settings-hooks-review")
        review.focus()
        await screen._load_hooks_settings()
        await pilot.pause()
        assert panel.query_one("#settings-hooks-review") is review
        assert host.focused is review


async def test_v2_definitions_use_saved_review_and_advanced_editor(hook_file):
    import sys

    from textual.widgets import Static

    from tldw_chatbook.UI.Screens.settings_hooks import HooksSettingsPanel
    from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
        ConsoleHooksReviewModal,
    )

    raw = toml.loads(hook_file.read_text())
    definition = {
        "id": "session-review",
        "event": "SessionStart",
        "type": "command",
        "effects": ["context"],
        "required": True,
        "argv": [sys.executable, "-c", "pass", "[bold]\nargument"],
        "env": {"REVIEW_VALUE": "literal"},
        "timeout_seconds": 7,
    }
    raw["hooks"] = {"handler": [definition]}
    hook_file.write_text(toml.dumps(raw))
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-advanced")
        panel = screen.query_one(HooksSettingsPanel)
        section, _ = panel.submission()
        assert section["handler"] == [definition]
        assert len(panel.snapshot.rows) == 1
        screen.query_one("#settings-hooks-review").focus()
        await pilot.press("enter")
        await _wait_for_selector(host.screen, pilot, "#console-hooks-review", timeout=5)
        modal = host.screen
        assert isinstance(modal, ConsoleHooksReviewModal)
        modal.query_one("#hook-review-details-0").focus()
        await pilot.press("enter")
        await _wait_for_selector(modal, pilot, ".hook-review-detail")
        details = str(modal.query_one(".hook-review-detail", Static).render())
        assert "SessionStart" in details and "REVIEW_VALUE" in details
        assert "timeout_seconds" in details and "[bold]\\nargument" in details
        assert not modal.query("#hook-review-disable-0")
        modal.query_one("#console-hooks-allow-all").focus()
        await pilot.press("enter")
        async with asyncio.timeout(5):
            while not modal.snapshot.ready:
                await pilot.pause(0.01)
        await pilot.press("escape")
        screen.query_one("#settings-hooks-advanced").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert screen.active_category == SettingsCategoryId.ADVANCED_CONFIG.value
