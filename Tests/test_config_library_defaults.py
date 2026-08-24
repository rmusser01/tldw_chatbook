"""Library configuration defaults."""

import tldw_chatbook.config as config_module


def test_load_settings_exposes_library_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)

    assert settings["library"]["ingest_directory_scan_limit"] == 1000
    assert settings["library"]["ingest_options"] == {}
    assert settings["library"]["media_reader"] == {
        "library_open": True,
        "items_open": True,
        "custom_widths_enabled": False,
        "library_width": 28,
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


def test_load_settings_normalizes_media_reader_and_preserves_library_keys(
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
    assert library["media_reader"] == {
        "library_open": False,
        "items_open": True,
        "custom_widths_enabled": True,
        "library_width": 48,
        "items_width": 32,
    }


def test_load_settings_ignores_malformed_media_reader_values(tmp_path, monkeypatch):
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
        "library_open": True,
        "items_open": True,
        "custom_widths_enabled": False,
        "library_width": 28,
        "items_width": 40,
    }
