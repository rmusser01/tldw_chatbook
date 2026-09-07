"""Configuration contracts for the Collections capture reader."""

from __future__ import annotations

from pathlib import Path

import tldw_chatbook.config as config_module
from tldw_chatbook.UI.Screens.settings_appearance_defaults import (
    build_appearance_save_sections,
    load_appearance_defaults,
)


def test_collections_reader_defaults_are_source_neutral_and_fixed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))

    library = config_module.load_settings(force_reload=True)["library"]
    appearance = load_appearance_defaults({"library": library})

    assert library["collections_reader"] == {
        "items_open": True,
        "items_width": 40,
    }
    assert appearance.library_collections_items_open is True
    assert appearance.library_collections_items_width == 40
    assert appearance.library_reader_custom_widths_enabled is False


def test_collections_reader_has_own_env_and_no_media_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[library.media_reader]
items_open = false
items_width = 68
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("TLDW_LIBRARY_COLLECTIONS_READER_ITEMS_OPEN", "false")
    monkeypatch.setenv("TLDW_LIBRARY_COLLECTIONS_READER_ITEMS_WIDTH", "54")

    collections = config_module.load_settings(force_reload=True)["library"][
        "collections_reader"
    ]

    assert collections == {"items_open": False, "items_width": 54}


def test_appearance_save_preserves_future_collections_keys() -> None:
    loaded = {
        "library": {
            "collections_reader": {
                "items_open": False,
                "items_width": 52,
                "future_collections": "keep",
            }
        }
    }
    values = load_appearance_defaults(loaded)

    sections = build_appearance_save_sections(loaded, values)

    assert sections["library"]["collections_reader"] == {
        "items_open": False,
        "items_width": 52,
        "future_collections": "keep",
    }
