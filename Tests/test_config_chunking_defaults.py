"""Task 11 (PR D, AC 40): the ``[chunking]`` config section ships.

Spec §9.1: the ``[chunking]`` section does not exist in the shipped config
before this task, so ``[chunking] default_template`` (the ingest resolution
order's config tier) was dark. It is added to the config template AND the
defaults, with tests asserting the REAL loader emits the section.
"""

from __future__ import annotations

import tomllib

import pytest

from tldw_chatbook import config as config_module  # noqa: E402


def test_config_template_ships_chunking_section():
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert template["chunking"]["default_template"] == ""


def test_default_config_tree_carries_chunking_section():
    assert (
        config_module.DEFAULT_CONFIG_FROM_TOML["chunking"]["default_template"] == ""
    )


def test_real_loader_emits_chunking_section(tmp_path, monkeypatch):
    """A fresh profile (missing config file) still resolves the section."""
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)

    assert settings["chunking"]["default_template"] == ""
    assert config_module.get_cli_setting("chunking", "default_template", None) == ""


def test_user_configured_default_template_wins(tmp_path, monkeypatch):
    """A user-set value survives the loader (the config tier is live)."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[chunking]\ndefault_template = "tiny-words"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["chunking"]["default_template"] == "tiny-words"
