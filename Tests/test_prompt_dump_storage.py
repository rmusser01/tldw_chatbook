"""The named-profile prompt exporter shares the application's storage base."""

import json
from contextlib import closing

import pytest

from Helper_Scripts.Prompts import Prompts_Dump
from Tests.Backup_Recovery.config_test_support import select_config_source
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
    selected = tmp_path / "config" / "config.toml"
    selected.parent.mkdir(mode=0o700)
    settings = '[general]\nusers_name = "current-user"\n'
    if root_kind == "explicit":
        settings += f"\n[paths]\ndata_dir = {json.dumps(str(base))}\n"
    selected.write_text(settings, encoding="utf-8")
    selected.chmod(0o600)
    select_config_source(monkeypatch, selected, vars(Prompts_Dump))

    assert Prompts_Dump.get_user_db_path("alice") == expected
    assert not (home / ".local").exists()
