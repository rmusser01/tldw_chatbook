"""The named-profile prompt exporter shares the application's storage base."""

from contextlib import closing

import pytest

from Helper_Scripts.Prompts import Prompts_Dump
from tldw_chatbook import config
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


@pytest.mark.parametrize("root_kind", ["fallback", "explicit"])
@pytest.mark.parametrize("filename", ["prompts.db", "tldw_chatbook_prompts.db"])
def test_named_profile_export_uses_selected_storage(
    tmp_path, monkeypatch, root_kind, filename
):
    home = tmp_path / "home"
    home.mkdir()
    base = home / ".tldw_cli-data" if root_kind == "fallback" else tmp_path / "custom"
    profile = base / "alice"
    profile.mkdir(parents=True)
    expected = profile / filename
    with closing(connect_private_sqlite("db.prompts.primary", expected)) as db, db:
        db.execute("CREATE TABLE export_probe (value TEXT)")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(config, "get_user_folder_name", lambda: "current-user")

    def setting(section, key, default=None):
        if section.lower() == "paths" and key == "data_dir" and root_kind == "explicit":
            return str(base)
        return default

    monkeypatch.setattr(config, "get_cli_setting", setting)

    assert Prompts_Dump.get_user_db_path("alice") == expected
    assert not (home / ".local").exists()
