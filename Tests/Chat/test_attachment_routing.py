"""Routing/picker drift-by-construction tests (TASK-222)."""

from pathlib import Path

import pytest

import tldw_chatbook.config as config_mod
from tldw_chatbook.Chat import attachment_core, console_paste_attach
from tldw_chatbook.Utils.file_handlers import ImageFileHandler


@pytest.fixture
def defaults_config(monkeypatch):
    """Simulate a config with no [chat.images] overrides (never read live config)."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


def test_routing_matches_effective_formats(defaults_config, monkeypatch):
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: True)
    handler = ImageFileHandler()
    for ext in attachment_core.supported_image_formats():
        assert handler.can_handle(Path(f"pic{ext}")) is True
    assert handler.can_handle(Path("pic.xcf")) is False


def test_routing_respects_config_narrowing(monkeypatch):
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: (
            [".png"] if key == "supported_formats" else default
        ),
    )
    handler = ImageFileHandler()
    assert handler.can_handle(Path("pic.png")) is True
    assert handler.can_handle(Path("pic.gif")) is False


def test_svg_routing_gated_by_capability(defaults_config, monkeypatch):
    handler = ImageFileHandler()
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
    assert handler.can_handle(Path("logo.svg")) is False
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: True)
    assert handler.can_handle(Path("logo.svg")) is True


def test_looks_attachable_follows_effective_formats(tmp_path, defaults_config, monkeypatch):
    monkeypatch.setattr(console_paste_attach, "is_safe_path", lambda p, r: True)
    svg = tmp_path / "logo.svg"
    svg.write_text("<svg/>")
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
    assert console_paste_attach.looks_attachable(str(svg)) is False
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: True)
    assert console_paste_attach.looks_attachable(str(svg)) is True


def test_looks_attachable_still_takes_tiff(tmp_path, defaults_config, monkeypatch):
    monkeypatch.setattr(console_paste_attach, "is_safe_path", lambda p, r: True)
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"II*\x00")
    assert console_paste_attach.looks_attachable(str(tiff)) is True


def test_old_module_constants_are_gone():
    assert not hasattr(attachment_core, "ATTACHMENT_FILTER_SPECS")
    assert not hasattr(ImageFileHandler, "SUPPORTED_EXTENSIONS")
