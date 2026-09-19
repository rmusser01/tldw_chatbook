"""Saved database defaults select the same profile under either path flavor."""

from pathlib import PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import profile_paths


def test_runtime_saved_defaults_match_recovery(monkeypatch):
    from tldw_chatbook import config

    defaults = config.DEFAULT_CONFIG_FROM_TOML["database"]
    selected = {"database": defaults, "general": {"users_name": "saved-profile"}}
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: selected.get(section, {}).get(key, default),
    )
    for _, setting, leaf, legacy in profile_paths.DATABASE_PATHS:
        if legacy is not None:
            assert config._get_custom_database_path(setting) is None
            assert (
                profile_paths.database_path(selected, setting)
                == profile_paths.user_data_dir(selected) / leaf
            )


@pytest.mark.parametrize("flavor", [PurePosixPath, PureWindowsPath])
@pytest.mark.parametrize(
    "setting,leaf,legacy",
    [(row[1], row[2], row[3]) for row in profile_paths.DATABASE_PATHS if row[3]],
)
def test_saved_default_uses_profile_database(
    flavor, setting, leaf, legacy, monkeypatch
):
    root = flavor("/private-profile")
    monkeypatch.setattr(profile_paths, "Path", flavor)
    monkeypatch.setattr(profile_paths, "user_data_dir", lambda config: root)
    config = {"database": {setting: "~/.local/share/tldw_cli/" + legacy}}
    assert profile_paths.database_path(config, setting) == root / leaf


@pytest.mark.parametrize("flavor", [PurePosixPath, PureWindowsPath])
def test_saved_optional_default_remains_unused(flavor, monkeypatch):
    from tldw_chatbook.Backup_Recovery import file_inventory
    from tldw_chatbook.DB import recovery_operations

    monkeypatch.setattr(profile_paths, "Path", flavor)
    monkeypatch.setattr(recovery_operations, "Path", flavor)
    monkeypatch.setattr(
        file_inventory,
        "_inventory_root",
        lambda *args, **kwargs: SimpleNamespace(status="unused"),
    )
    config = {
        "database": {"rag_indexing_db_path": "~/.local/share/tldw_cli/rag_indexing.db"}
    }
    assert (
        recovery_operations._sqlite_inventory_status(
            config,
            flavor("/profile/rag_indexing.db"),
            owner="db.rag_indexing",
            setting_name="rag_indexing_db_path",
            optional_default=True,
        )
        == "unused"
    )
