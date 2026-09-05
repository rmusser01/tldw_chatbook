"""Snapshot preference ownership, stale drafts, and canonical category Save."""

import asyncio
import threading

import pytest
from textual.widgets import Button, Checkbox, Collapsible, Input

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.LLM_Management import snapshot_settings as preferences
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


@pytest.fixture
def no_snapshot_splash(monkeypatch):
    from tldw_chatbook.config import get_cli_setting

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", [{"keep_count": 0}, {"keep_count": 1.5}, {"enabled": "yes"}]
)
async def test_malformed_f9_preferences_leave_provider_surface_and_revert_recovery(
    no_snapshot_splash, bad
):
    from tldw_chatbook import config

    assert config.apply_settings_mutation_to_cli_config(
        {"llamacpp_snapshots": bad}
    ).fully_applied
    app = _build_test_app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        field = screen.query_one("#settings-snapshot-keep", Input)
        assert field.disabled and field.value == ""
        assert not screen.query_one("#settings-provider-search", Input).disabled
        assert "Advanced Config" in str(
            screen.query_one("#settings-snapshot-result").render()
        )
        assert not screen.query_one("#settings-revert-category", Button).disabled
        screen.query_one("#settings-revert-category", Button).press()
        await pilot.pause()
        assert field.disabled
        assert preferences.save_snapshot_preferences(
            preferences.SnapshotPreferences(enabled=True, keep_count=23)
        )
        screen.query_one("#settings-revert-category", Button).press()
        await pilot.pause()
        assert not field.disabled and field.value == "23"
        field.value = "24"
        await pilot.pause()
        assert config.apply_settings_mutation_to_cli_config(
            {"llamacpp_snapshots": {"keep_count": "broken-private-value"}}
        ).fully_applied
        screen.action_settings_revert_category(allow_text_entry_focus=True)
        await pilot.pause()
        app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause()
        assert field.disabled
        assert "broken-private-value" not in str(
            screen.query_one("#settings-snapshot-result").render()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["027", "+27", " 27 "])
@pytest.mark.parametrize("combined", [False, True])
async def test_success_canonicalizes_unchanged_f9_draft_and_continues_once(
    no_snapshot_splash, monkeypatch, raw, combined
):
    from tldw_chatbook import config

    app = _build_test_app()
    writes = []
    original_save = preferences.save_snapshot_preferences

    def save(value, **kwargs):
        writes.append(value)
        assert len(writes) == 1, "duplicate snapshot persistence"
        return original_save(value, **kwargs)

    monkeypatch.setattr(preferences, "save_snapshot_preferences", save)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        if combined:
            screen.query_one(
                "#settings-model-value", Input
            ).value = "snapshot-review-model"
        field = screen.query_one("#settings-snapshot-keep", Input)
        field.value = raw
        await pilot.pause()
        continuations = []
        original_action = screen.action_settings_save_category

        def action(**kwargs):
            continuations.append(True)
            original_action(**kwargs)

        monkeypatch.setattr(screen, "action_settings_save_category", action)
        original_action(allow_text_entry_focus=True)
        await pilot.pause(0.2)
        assert len(writes) == 1
        assert preferences.load_snapshot_preferences().keep_count == 27
        assert field.value == "27"
        assert not screen._snapshot_preferences_dirty()
        assert len(continuations) == int(combined)
        if combined:
            assert (
                config.get_cli_setting("chat_defaults", "model")
                == "snapshot-review-model"
            ), screen._provider_save_result


@pytest.mark.asyncio
async def test_f9_save_preserves_new_edits_while_persistence_waits(
    no_snapshot_splash, monkeypatch
):
    app = _build_test_app()
    entered, release = threading.Event(), threading.Event()
    original_save = preferences.save_snapshot_preferences
    writes = []

    def save(value, **kwargs):
        writes.append(value)
        entered.set()
        assert release.wait(5)
        return original_save(value, **kwargs)

    monkeypatch.setattr(preferences, "save_snapshot_preferences", save)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        screen._stage_provider_value("model", "snapshot-review-model")
        field = screen.query_one("#settings-snapshot-keep", Input)
        field.value = "027"
        await pilot.pause()
        continuations = []
        original_action = screen.action_settings_save_category

        def action(**kwargs):
            continuations.append(True)
            original_action(**kwargs)

        monkeypatch.setattr(screen, "action_settings_save_category", action)
        original_action(allow_text_entry_focus=True)
        try:
            async with asyncio.timeout(5):
                while not entered.is_set():
                    await pilot.pause(0.01)
            # Model the interval before delivery of the newer Input.Changed.
            with screen.prevent(Input.Changed):
                field.value = "32"
                assert screen._snapshot_preferences_raw[1] == "027"
                release.set()
                async with asyncio.timeout(5):
                    while screen._snapshot_preferences_saving:
                        await pilot.pause(0.01)
            assert field.value == "32"
            assert continuations == []
            field.post_message(Input.Changed(field, "32"))
            await pilot.pause()
        finally:
            release.set()
        await pilot.pause(0.2)
        assert len(writes) == 1
        assert preferences.load_snapshot_preferences().keep_count == 27
        assert field.value == "32"
        assert screen._snapshot_preferences_dirty()
        assert screen._provider_draft().is_dirty


@pytest.mark.asyncio
async def test_f9_snapshot_draft_uses_category_save_offthread_and_reverts(monkeypatch):
    from tldw_chatbook.config import get_cli_setting

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ),
    )
    app = _build_test_app()
    threads = []
    original = preferences.save_snapshot_preferences

    def save(value, **kwargs):
        threads.append(threading.get_ident())
        return original(value, **kwargs)

    monkeypatch.setattr(preferences, "save_snapshot_preferences", save)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        field = screen.query_one("#settings-snapshot-keep", Input)
        field.value = "27"
        screen.query_one("#settings-snapshot-enabled", Checkbox).value = True
        await pilot.pause()
        assert (
            preferences.load_snapshot_preferences() == preferences.SnapshotPreferences()
        )
        screen.action_settings_save_category(allow_text_entry_focus=True)
        async with asyncio.timeout(5):
            while preferences.load_snapshot_preferences().keep_count != 27:
                await pilot.pause(0.01)
        assert threads and all(value != threading.get_ident() for value in threads)
        field.value = "0"
        await pilot.pause()
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause()
        assert preferences.load_snapshot_preferences().keep_count == 27
        screen.action_settings_revert_category(allow_text_entry_focus=True)
        await pilot.pause()
        app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause()
        assert field.value == "27"


@pytest.mark.asyncio
async def test_f9_stale_snapshot_draft_requires_revert_before_save(monkeypatch):
    from tldw_chatbook.config import get_cli_setting

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ),
    )
    app = _build_test_app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        screen.query_one("#settings-snapshot-keep", Input).value = "31"
        await pilot.pause()
        newer = preferences.SnapshotPreferences(enabled=True, keep_count=14)
        await asyncio.to_thread(preferences.save_snapshot_preferences, newer)
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause(0.2)
        assert preferences.load_snapshot_preferences() == newer
        assert "Revert" in str(screen.query_one("#settings-snapshot-result").render())
        assert screen.query_one("#settings-snapshot-keep", Input).value == "31"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 45)])
async def test_f9_snapshot_controls_paint_and_keyboard_save(size, monkeypatch):
    from Tests.UI.test_llamacpp_snapshot_manager import frame, painted_text
    from tldw_chatbook.config import get_cli_setting

    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ),
    )
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        screen.query_one("#settings-snapshot-controls", Collapsible).collapsed = False
        checkbox = screen.query_one("#settings-snapshot-enabled", Checkbox)
        checkbox.focus()
        await pilot.wait_for_scheduled_animations()
        await pilot.press("space")
        await pilot.pause()
        assert checkbox.value
        assert checkbox in screen._compositor.visible_widgets
        frame(app, f"settings-{size[0]}-snapshot-draft")
        assert "Enable snapshots" in painted_text(app, checkbox)
        assert (
            dict(screen._provider_field_guidance_rows())["Saved as"]
            == "llamacpp_snapshots.enabled"
        )
        assert checkbox.label.plain == "Enable snapshots"
        assert (
            "next launch"
            in screen.query_one("#settings-snapshot-launch-scope").render().plain
        )
        save = screen.query_one("#settings-save-category", Button)
        save.focus()
        await pilot.wait_for_scheduled_animations()
        await pilot.press("enter")
        async with asyncio.timeout(5):
            while not preferences.load_snapshot_preferences().enabled:
                await pilot.pause(0.01)
