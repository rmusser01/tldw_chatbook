"""The shipped embedding-cache literal is a default value, not a target."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_embedding_cache_default_retargets_to_each_active_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The unchanged shipped default must resolve beneath each active user.

    Returning the shipped value from ``get_cli_setting`` represents an unset
    setting. The observable contract is the actual cache directories created
    for Alice and Bob, not the mocked settings call.
    """
    from tldw_chatbook import config

    shipped_default = config.DEFAULT_CONFIG_FROM_TOML["embedding_config"][
        "model_cache_dir"
    ]

    def get_shipped_default(section: str, key: str, default: object) -> object:
        assert (section, key, default) == (
            "embedding_config",
            "model_cache_dir",
            shipped_default,
        )
        return shipped_default

    active_user = {"path": tmp_path / "data" / "alice"}
    monkeypatch.setattr(config, "get_cli_setting", get_shipped_default)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: active_user["path"])

    alice_cache = config.get_model_cache_dir()
    active_user["path"] = tmp_path / "data" / "bob"
    bob_cache = config.get_model_cache_dir()

    assert alice_cache == tmp_path / "data" / "alice" / "models" / "embeddings"
    assert bob_cache == tmp_path / "data" / "bob" / "models" / "embeddings"
    assert alice_cache.is_dir()
    assert bob_cache.is_dir()
    assert str(alice_cache) != shipped_default
    assert str(bob_cache) != shipped_default
