"""Mounted staged Saves retain later live input, including queued Changed delivery."""

import asyncio
import json
import threading
import tomllib

import pytest
from textual.widgets import Button, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
)
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from Tests.UI.test_settings_provider_keyboard_journeys import _revert
from tldw_chatbook import config
from tldw_chatbook.UI.Screens import settings_screen as sm
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId


async def _until(check):
    deadline = asyncio.get_running_loop().time() + 15
    while not check():
        assert (
            asyncio.get_running_loop().time() < deadline
        ), "Save never entered its writer"
        await asyncio.sleep(0.01)


async def _replace(pilot, field, value):
    field.focus()
    await pilot.press("home", "shift+end", "backspace", *value)
    await pilot.pause()
    assert field.value == value


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("kind", ["appearance", "storage", "console-behavior"])
@pytest.mark.parametrize(
    "outcome", ["later", "different", "queued", "restored", "unchanged", "failed"]
)
@private_profile_test
async def test_staged_save_acknowledges_only_saved_values(
    request, monkeypatch, tmp_path, kind, outcome
):
    """An unconditional pop/sync loses later edits; stale originals break next Save/Revert."""
    category = SettingsCategoryId(kind)
    selector, section, key, submitted, later, other_selector, other_key, other_value = {
        "appearance": (
            "#settings-appearance-font-size",
            "web_server",
            "font_size",
            "18",
            "19",
            "#settings-appearance-palette-theme-limit",
            "palette_theme_limit",
            "25",
        ),
        "storage": (
            "#settings-storage-workspaces-db-path",
            "database",
            "workspaces_db_path",
            str(tmp_path / "submitted.db"),
            str(tmp_path / "later.db"),
            "#settings-storage-prompts-db-path",
            "prompts_db_path",
            str(tmp_path / "prompts.db"),
        ),
        "console-behavior": (
            "#settings-console-paste-collapse-threshold",
            "console",
            "paste_collapse_threshold",
            "120",
            "121",
            "#settings-console-max-parallel-runs",
            "max_parallel_runs",
            "4",
        ),
    }[kind]
    entered, release = threading.Event(), threading.Event()
    real_writer = sm.SettingsConfigAdapter.save_sections
    real_mutation = config.apply_settings_mutation_to_cli_config
    fail = outcome == "failed"

    def held(adapter, *args, **kwargs):
        entered.set()
        assert release.wait(30)
        return real_writer(adapter, *args, **kwargs)

    def mutation(*args, **kwargs):
        if fail:

            def fail_write(*args, **kwargs):
                raise OSError("injected pre-replacement Settings write failure")

            with monkeypatch.context() as failed_writer:
                failed_writer.setattr(
                    config, "_write_raw_cli_config_unlocked", fail_write
                )
                return real_mutation(*args, **kwargs)
        return real_mutation(*args, **kwargs)

    monkeypatch.setattr(sm.SettingsConfigAdapter, "save_sections", held)
    monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", mutation)
    app = _build_test_app()
    app.app_config.setdefault("console", {})["paste_collapse_threshold"] = 50
    handles = {
        key: value
        for key, value in vars(app).items()
        if key.endswith("_db") and value is not None
    }
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, f"#settings-category-{kind}")
        await host.workers.wait_for_complete()
        await pilot.pause()
        field = screen.query_one(selector, Input)
        other = screen.query_one(other_selector, Input)
        original, other_original = field.value, other.value
        config_path = config.get_cli_config_path()
        before = config_path.read_bytes()
        await _replace(pilot, field, submitted)
        completion_name = {
            "appearance": "_apply_appearance_save_result",
            "storage": "_apply_storage_save_result",
            "console-behavior": "_apply_console_behavior_save_result",
        }[kind]
        complete = getattr(screen, completion_name)
        immediate = []

        def completion(*args, **kwargs):
            if outcome == "queued":
                # The same synchronous edit primitive used by Input's key handler:
                # value is accepted now, its Changed/Edited messages cannot run yet.
                field.replace(later, 0, len(field.value))
                assert field.value == later
            live_selection = field.selection
            result = complete(*args, **kwargs)
            if outcome == "queued":
                assert field.selection == live_selection
            immediate.append(
                (field.value, screen._category_has_unsaved_changes(category))
            )
            return result

        monkeypatch.setattr(screen, completion_name, completion)
        screen.query_one("#settings-save-category", Button).press()
        await _until(entered.is_set)
        expected = submitted
        try:
            if outcome in {"later", "failed", "restored"}:
                expected = original if outcome == "restored" else later
                await _replace(pilot, field, expected)
            elif outcome == "different":
                await _replace(pilot, other, other_value)
            if outcome == "queued":
                expected = later
            current = other if outcome == "different" else field
            current.focus()
            await pilot.press("end", "shift+left")
            selection = current.selection
            if outcome == "different":
                # A later deliberate focus move must win over Save completion.
                screen.query_one("#settings-category-appearance", Button).focus()
                await pilot.pause()
            focused = host.focused
        finally:
            release.set()
        await host.workers.wait_for_complete()
        await pilot.pause()
        dirty = outcome != "unchanged"
        (tmp_path / "completion-receipt.json").write_text(
            json.dumps(
                {
                    "category": kind,
                    "outcome": outcome,
                    "submitted": submitted,
                    "prior_baseline": original,
                    "expected_live": expected,
                    "immediate_value_dirty": immediate,
                    "final_live": field.value,
                    "other_live": other.value,
                    "same_input": screen.query_one(selector) is field,
                    "focus_retained": host.focused is focused,
                    "selection": repr(field.selection),
                    "saved_file_changed": config_path.read_bytes() != before,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        assert immediate == [(expected, dirty)]
        assert screen.query_one(selector, Input) is field
        assert screen.query_one(other_selector, Input) is other
        assert field.value == expected
        assert other.value == (
            other_value if outcome == "different" else other_original
        )
        assert screen._category_has_unsaved_changes(category) is dirty
        assert host.focused is focused
        if outcome != "queued":
            assert current.selection == selection
        assert all(getattr(app, name) is handle for name, handle in handles.items())
        if fail:
            assert config_path.read_bytes() == before
        else:
            assert (
                str(
                    tomllib.loads(config_path.read_text(encoding="utf-8"))[section][key]
                )
                == submitted
            )
        if not dirty:
            return
        if outcome == "restored":
            # The first Save's actual baseline wins over the prior loaded value.
            await _revert(host, pilot, discard=True)
            assert screen.query_one(selector, Input).value == submitted
            await _replace(pilot, field, expected)
        fail = False
        monkeypatch.setattr(screen, completion_name, complete)
        screen.query_one("#settings-save-category", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert (
            str(tomllib.loads(config_path.read_text(encoding="utf-8"))[section][key])
            == expected
        )
        assert not screen._category_has_unsaved_changes(category)
        if outcome == "different":
            saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
            other_section = "general" if kind == "appearance" else section
            assert str(saved[other_section][other_key]) == other_value
        await _replace(pilot, field, later if expected != later else submitted)
        await _revert(host, pilot, discard=True)
        assert screen.query_one(selector, Input).value == expected
        assert not screen._category_has_unsaved_changes(category)
        assert all(getattr(app, name) is handle for name, handle in handles.items())


@pytest.mark.asyncio
@private_profile_test
async def test_unchanged_normalized_console_submission_becomes_clean(request):
    """A raw snapshot must acknowledge unchanged whitespace after writer normalization."""
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        await host.workers.wait_for_complete()
        await pilot.pause()
        field = screen.query_one("#settings-console-sidechat-model", Input)
        await _replace(pilot, field, "  saved-model  ")
        screen.query_one("#settings-save-category", Button).press()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()
        saved = tomllib.loads(config.get_cli_config_path().read_text(encoding="utf-8"))
        assert saved["console"]["sidechat_model"] == "saved-model"
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.CONSOLE_BEHAVIOR
        )
        assert screen.query_one("#settings-console-sidechat-model", Input) is field
