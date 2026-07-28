from __future__ import annotations

import json
import logging

import pytest

import tldw_chatbook.UI.Tools_Settings_Window as tools_settings_module
import tldw_chatbook.app as app_module
from tldw_chatbook.UI.Screens.tools_settings_screen import ToolsSettingsScreen
from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow
from tldw_chatbook.app import TldwCli


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


async def _close_production_app(app: TldwCli) -> None:
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_production_app_backup_publishes_one_complete_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    backup_data_dir = tmp_path / "profile-data"
    backup_data_dir.mkdir(mode=0o700)
    monkeypatch.setattr(
        tools_settings_module,
        "get_user_data_dir",
        lambda: backup_data_dir,
    )
    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False

    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(ToolsSettingsScreen(app))
            await pilot.pause()
            window = app.screen.query_one(ToolsSettingsWindow)

            await window._backup_databases()

            backup_directories = tuple((backup_data_dir / "backups").iterdir())
            assert len(backup_directories) == 1
            manifest_path = backup_directories[0] / "backup_info.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert manifest["databases"]
            assert manifest["databases"][-1]["name"] == "TTS Profiles"
            assert all(
                (backup_directories[0] / entry["path"].rsplit("/", 1)[-1]).exists()
                for entry in manifest["databases"]
            )
    finally:
        await _close_production_app(app)
