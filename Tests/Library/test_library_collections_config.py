"""Library Collections config path validation regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook import config


def test_library_collections_custom_db_path_is_validated(monkeypatch, tmp_path: Path) -> None:
    safe_path = tmp_path / "collections.db"
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: str(safe_path),
    )

    assert config.get_library_collections_db_path() == safe_path.resolve()


def test_library_collections_custom_db_path_rejects_dangerous_input(monkeypatch) -> None:
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: "/tmp/collections.db;rm -rf",
    )

    with pytest.raises(ValueError):
        config.get_library_collections_db_path()
