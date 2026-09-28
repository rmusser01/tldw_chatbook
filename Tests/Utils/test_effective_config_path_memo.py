"""The effective-config-path memo (TASK-24304).

`_get_effective_config_path` re-ran `expanduser` + `abspath` + `normpath` on
every call, and a single warm Console screen entry called it 1,132 times. The
memo behind it must not be observable as staleness: the environment read stays
in the caller, and both environment variables that can change the answer are
part of the cache key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.config import (
    _get_effective_config_path,
    _resolve_effective_config_path,
    get_cli_config_path,
)


def test_repeated_resolution_does_no_repeated_path_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unchanged environment resolves the path once, then serves from the memo."""
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setenv("HOME", str(tmp_path))

    _get_effective_config_path()
    before = _resolve_effective_config_path.cache_info()
    for _ in range(200):
        _get_effective_config_path()
    after = _resolve_effective_config_path.cache_info()

    assert after.misses == before.misses, (
        f"{after.misses - before.misses} path normalisations across 200 calls "
        "with an unchanged environment; the memo is not engaging."
    )
    assert after.hits - before.hits == 200


def test_a_changed_override_is_observed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Moving TLDW_CONFIG_PATH changes the answer -- the memo is not a pin."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "first.toml"))
    first = _get_effective_config_path()

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "second.toml"))
    second = _get_effective_config_path()

    assert first != second
    assert second.name == "second.toml"


def test_a_changed_home_is_observed_for_a_tilde_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The platform home variable invalidates tilde expansion.

    Keying the memo on the override alone would return the first HOME's
    expansion forever. Tests move HOME routinely, so this is the failure the
    key exists to prevent.
    """
    first_home = tmp_path / "home-one"
    second_home = tmp_path / "home-two"
    for home in (first_home, second_home):
        home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TLDW_CONFIG_PATH", "~/config.toml")
    home_key = "USERPROFILE" if os.name == "nt" else "HOME"
    monkeypatch.setenv(home_key, str(first_home))
    first = _get_effective_config_path()

    monkeypatch.setenv(home_key, str(second_home))
    second = _get_effective_config_path()

    assert first != second, (
        "the memo returned the first HOME's expansion after HOME moved; "
        "`~` in the override must re-expand."
    )
    assert second.parent == second_home.resolve() or str(second).startswith(
        str(second_home)
    )


def test_relative_override_tracks_current_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relative override must resolve again after the process changes cwd."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", "config.toml")

    monkeypatch.chdir(first_dir)
    first = _get_effective_config_path()
    monkeypatch.chdir(second_dir)
    second = _get_effective_config_path()

    assert first == first_dir / "config.toml"
    assert second == second_dir / "config.toml"


def test_public_accessor_agrees_with_the_private_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`get_cli_config_path` is a thin wrapper and must not diverge."""
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setenv("HOME", str(tmp_path))

    assert get_cli_config_path() == _get_effective_config_path()


def test_unset_override_falls_back_to_the_default_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no override the module default is used, memo or not."""
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)

    assert _get_effective_config_path() == config_module.lexical_path(
        config_module.DEFAULT_CONFIG_PATH
    )
