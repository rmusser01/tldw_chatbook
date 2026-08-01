"""Profile-owned Settings path display regressions."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

import tldw_chatbook.Character_Chat.Chat_Dictionary_Lib as dictionary_lib
import tldw_chatbook.Prompt_Management.Prompts_Interop as prompts_interop
import tldw_chatbook.UI.CodeRepoCopyPasteWindow as code_repo_window
import tldw_chatbook.UI.Screens.settings_screen as settings_screen
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId, SettingsScreen


@pytest.mark.parametrize("profile_name", ["alpha", "beta"])
def test_config_children_follow_effective_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_name: str,
) -> None:
    """Config-owned defaults are derived when a selected profile is read."""
    config_path = tmp_path / profile_name / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert dictionary_lib._default_dictionary_import_directory() == config_path.parent
    assert (
        prompts_interop._default_prompt_import_directory()
        == config_path.parent / "prompts"
    )
    assert code_repo_window._github_config_guidance_path() == config_path
    assert settings_screen._theme_save_target() == config_path.parent / "themes"
    assert settings_screen._internal_prompts_save_target() == config_path


def test_config_children_retain_default_profile_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No override retains each default profile child's historical location."""
    config_path = tmp_path / "default" / "config.toml"
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    monkeypatch.setattr("tldw_chatbook.config.DEFAULT_CONFIG_PATH", config_path)

    assert dictionary_lib._default_dictionary_import_directory() == config_path.parent
    assert (
        prompts_interop._default_prompt_import_directory()
        == config_path.parent / "prompts"
    )
    assert code_repo_window._github_config_guidance_path() == config_path
    assert settings_screen._theme_save_target() == config_path.parent / "themes"
    assert settings_screen._internal_prompts_save_target() == config_path


def test_settings_theme_copy_uses_effective_profile_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Theme guidance and ownership copy name the active profile's theme directory."""
    config_path = tmp_path / "alpha" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    window = MagicMock(spec=SettingsScreen)
    window._domain_category_ownership_records.return_value = ()

    guidance = SettingsScreen._inspector_guidance(window, SettingsCategoryId.THEME)
    ownership_records = SettingsScreen._category_ownership_records(window)
    theme_ownership = next(
        record
        for record in ownership_records
        if record.category is SettingsCategoryId.THEME
    )

    expected_theme_target = f"{config_path.parent / 'themes'}{os.sep}"
    assert ("Affected config", f"custom theme files under {expected_theme_target}") in guidance
    assert str(config_path.parent / "themes") in theme_ownership.recovery_copy

    window._active_summary.return_value = SimpleNamespace(
        category=SettingsCategoryId.THEME
    )
    window._ownership_record.return_value = theme_ownership
    window._detail_row.side_effect = lambda label, value, **kwargs: SettingsScreen._detail_row(
        window, label, value, **kwargs
    )
    window._inspector_guidance.side_effect = lambda category: SettingsScreen._inspector_guidance(
        window, category
    )

    rows = list(SettingsScreen._render_impact_pane_body(window))

    assert any(
        str(row.renderable).replace("\n  ", "")
        == f"Save target: {expected_theme_target}"
        for row in rows
    )


def test_settings_internal_prompt_copy_uses_effective_profile_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Internal prompt display identifies the selected config file at render time."""
    config_path = tmp_path / "beta" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    window = MagicMock(spec=SettingsScreen)
    window._active_summary.return_value = SimpleNamespace(
        category=SettingsCategoryId.INTERNAL_PROMPTS
    )
    window._ownership_record.return_value = MagicMock()
    window._get_internal_prompts_customized_count.return_value = 0
    window._detail_row.side_effect = lambda label, value, **kwargs: SettingsScreen._detail_row(
        window, label, value, **kwargs
    )
    window._inspector_guidance.side_effect = lambda category: SettingsScreen._inspector_guidance(
        window, category
    )

    rows = list(SettingsScreen._render_impact_pane_body(window))

    assert any(
        str(row.renderable).replace("\n  ", "")
        == f"Save target: {config_path}  [internal_prompts]"
        for row in rows
    )


@pytest.mark.asyncio
async def test_github_token_guidance_uses_effective_config_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GitHub token guidance points at the selected profile's config file."""
    config_path = tmp_path / "gamma" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *_args: "")
    window = MagicMock(spec=code_repo_window.CodeRepoCopyPasteWindow)
    window.notify = Mock()

    await code_repo_window.CodeRepoCopyPasteWindow.configure_token(window, object())

    assert str(config_path) in window.notify.call_args.args[0]
