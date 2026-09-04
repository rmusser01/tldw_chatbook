"""Saved user themes register at application startup (TASK-31250, PR #2375 review #5).

Drives the real ``TldwCli`` startup path via the shared factory: the loader is
unit-tested elsewhere, this pins the ``on_mount`` wiring and the
``general.default_theme`` selection through the public boundary. Removing the
registration loop in ``app.py`` makes both tests fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tldw_chatbook.config as config_module
from Tests.UI.app_factory import _build_test_app


def _write_theme(themes_dir: Path, name: str, primary: str) -> None:
    themes_dir.mkdir(parents=True, exist_ok=True)
    (themes_dir / f"{name}.toml").write_text(
        f'[theme]\nname = "{name}"\ndark = true\n[colors]\nprimary = "{primary}"\n',
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_startup_registers_saved_user_themes(tmp_path, monkeypatch):
    themes_dir = tmp_path / "themes"
    _write_theme(themes_dir, "ocean_startup_probe", "#9966FF")
    monkeypatch.setattr(config_module, "get_user_themes_dir", lambda: themes_dir)

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert "ocean_startup_probe" in app.available_themes


@pytest.mark.asyncio
async def test_startup_applies_saved_theme_named_as_default(tmp_path, monkeypatch):
    themes_dir = tmp_path / "themes"
    _write_theme(themes_dir, "ocean_default_probe", "#9966FF")
    monkeypatch.setattr(config_module, "get_user_themes_dir", lambda: themes_dir)

    # The factory's per-test config sandbox is what get_cli_setting reads at
    # mount, so write the default through the real setter (see the factory's
    # docstring on config_overrides vs save_setting_to_cli_config).
    assert config_module.save_setting_to_cli_config(
        "general", "default_theme", "ocean_default_probe"
    )

    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.theme == "ocean_default_probe"
