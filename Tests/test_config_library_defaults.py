"""Library configuration defaults."""

import tldw_chatbook.config as config_module


def test_load_settings_exposes_library_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)

    assert settings["library"]["ingest_directory_scan_limit"] == 1000
    assert settings["library"]["ingest_options"] == {}
    assert settings["library"]["reader"] == {
        "library_open": True,
        "custom_widths_enabled": False,
        "library_width": 28,
    }
    assert settings["library"]["media_reader"] == {
        "items_open": True,
        "items_width": 40,
        "library_open": True,
        "custom_widths_enabled": False,
        "library_width": 28,
    }
    for section in (
        "conversations_reader",
        "notes_reader",
        "prompts_reader",
        "skills_reader",
    ):
        assert settings["library"][section] == {
            "items_open": True,
            "items_width": 40,
        }


def test_load_settings_coerces_library_scan_limit(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[library]\ningest_directory_scan_limit = 2500\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["library"]["ingest_directory_scan_limit"] == 2500


def test_load_settings_rejects_invalid_library_scan_limit(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    for raw_value in ("true", "0", "-5"):
        config_path.write_text(
            f"[library]\ningest_directory_scan_limit = {raw_value}\n",
            encoding="utf-8",
        )

        settings = config_module.load_settings(force_reload=True)

        assert settings["library"]["ingest_directory_scan_limit"] == 1000


def test_load_settings_reads_persisted_ingest_options(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[library.ingest_options.pdf]\npdf_engine = "docling"\nocr = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["library"]["ingest_options"] == {
        "pdf": {"pdf_engine": "docling", "ocr": True}
    }


def test_load_settings_falls_back_from_legacy_media_reader_per_shared_key(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[library.search]
history = ["oceans"]

[library.media_reader]
library_open = false
items_open = true
custom_widths_enabled = true
library_width = 99
items_width = 20
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    library = config_module.load_settings(force_reload=True)["library"]

    assert library["search"] == {"history": ["oceans"]}
    assert library["reader"] == {
        "library_open": False,
        "custom_widths_enabled": True,
        "library_width": 48,
    }
    assert library["media_reader"] == {
        "library_open": False,
        "items_open": True,
        "custom_widths_enabled": True,
        "library_width": 99,
        "items_width": 32,
    }


def test_load_settings_reader_partial_values_win_with_per_key_legacy_fallback(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.toml"
    original = """
[library.reader]
library_open = true

[library.media_reader]
library_open = false
custom_widths_enabled = true
library_width = 36
items_open = false
items_width = 64
future_key = "keep"
"""
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    library = config_module.load_settings(force_reload=True)["library"]

    assert library["reader"] == {
        "library_open": True,
        "custom_widths_enabled": True,
        "library_width": 36,
    }
    assert library["media_reader"] == {
        "library_open": False,
        "custom_widths_enabled": True,
        "library_width": 36,
        "items_open": False,
        "items_width": 64,
        "future_key": "keep",
    }
    assert config_path.read_text(encoding="utf-8") == original


def test_load_settings_preserves_saved_widths_while_custom_widths_are_disabled(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[library.reader]
custom_widths_enabled = false
library_width = 36

[library.media_reader]
items_width = 64
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    library = config_module.load_settings(force_reload=True)["library"]

    assert library["reader"]["custom_widths_enabled"] is False
    assert library["reader"]["library_width"] == 36
    assert library["media_reader"]["items_width"] == 64


def test_load_settings_normalizes_all_destination_item_preferences(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[library.conversations_reader]
items_open = false
items_width = 48

[library.notes_reader]
items_open = true
items_width = 52

[library.prompts_reader]
items_open = false
items_width = 60

[library.skills_reader]
items_open = true
items_width = 68
future_key = "keep"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    library = config_module.load_settings(force_reload=True)["library"]

    assert library["conversations_reader"] == {
        "items_open": False,
        "items_width": 48,
    }
    assert library["notes_reader"] == {"items_open": True, "items_width": 52}
    assert library["prompts_reader"] == {"items_open": False, "items_width": 60}
    assert library["skills_reader"] == {
        "items_open": True,
        "items_width": 68,
        "future_key": "keep",
    }


def test_load_settings_normalizes_destination_values_without_rewriting_legacy_keys(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[library.media_reader]
library_open = "sometimes"
items_open = 2
custom_widths_enabled = "no"
library_width = "wide"
items_width = true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    media_reader = config_module.load_settings(force_reload=True)["library"][
        "media_reader"
    ]

    assert media_reader == {
        "library_open": "sometimes",
        "items_open": True,
        "custom_widths_enabled": "no",
        "library_width": "wide",
        "items_width": 40,
    }

    assert config_module.load_settings(force_reload=True)["library"]["reader"] == {
        "library_open": True,
        "custom_widths_enabled": False,
        "library_width": 28,
    }
