"""The shipped embedding-cache literal is a default value, not a target."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Tests.Backup_Recovery.config_test_support import select_config_source

pytestmark = pytest.mark.unit


def test_embedding_cache_default_retargets_to_each_active_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The unchanged shipped default must resolve beneath each active user."""
    from tldw_chatbook import config

    shipped_default = config.DEFAULT_CONFIG_FROM_TOML["embedding_config"][
        "model_cache_dir"
    ]

    data_root = tmp_path / "data"
    data_root.mkdir(mode=0o700)

    def cache_for(user: str) -> Path:
        path = tmp_path / f"{user}.toml"
        path.write_text(
            f"[general]\nusers_name = {json.dumps(user)}\n"
            f"[paths]\ndata_dir = {json.dumps(str(data_root))}\n",
            encoding="utf-8",
        )
        with monkeypatch.context() as selected:
            source = select_config_source(selected, path)
            assert (
                source.get_cli_setting("embedding_config", "model_cache_dir")
                == shipped_default
            )
            return source.get_model_cache_dir()

    alice_cache = cache_for("alice")
    bob_cache = cache_for("bob")

    assert alice_cache == tmp_path / "data" / "alice" / "models" / "embeddings"
    assert bob_cache == tmp_path / "data" / "bob" / "models" / "embeddings"
    assert alice_cache.is_dir()
    assert bob_cache.is_dir()
    assert str(alice_cache) != shipped_default
    assert str(bob_cache) != shipped_default
