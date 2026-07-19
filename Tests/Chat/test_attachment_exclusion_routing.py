"""Explicit rejection for config-excluded image formats (TASK-230).

With a narrowed ``[chat.images].supported_formats``, a known-image extension
picked via the pickers' "All Files" row used to slide past ImageFileHandler
into DefaultFileHandler, inlining a ``[File: x.tiff (…) - image/tiff]``
placeholder that reads as success while sending useless text (TASK-222 final
review, finding M1). The registry now rejects known-image extensions the
config excluded — the legacy error-toast behavior — while truly unknown
extensions keep the generic-file fallthrough.
"""

from io import BytesIO

import pytest
from PIL import Image as PILImage

import tldw_chatbook.config as config_mod
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Utils.file_handlers import file_handler_registry


@pytest.fixture
def png_only_config(monkeypatch):
    """[chat.images].supported_formats narrowed to [".png"]."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, k=None, default=None: (
            {"supported_formats": [".png"]}
            if (section, k) == ("chat", "images")
            else default
        ),
    )


def _write_png_bytes(path, size=(8, 8)):
    buffer = BytesIO()
    PILImage.new("RGB", size, "red").save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())


@pytest.mark.asyncio
async def test_excluded_image_extension_rejects(tmp_path, png_only_config):
    tiff = tmp_path / "scan.tiff"
    PILImage.new("RGB", (8, 8), "red").save(tiff, format="TIFF")
    with pytest.raises(ValueError, match="Unsupported image format: .tiff"):
        await file_handler_registry.process_file(tiff)


@pytest.mark.asyncio
async def test_excluded_rejection_names_config_key(tmp_path, png_only_config):
    gif = tmp_path / "anim.gif"
    PILImage.new("RGB", (8, 8), "red").save(gif, format="GIF")
    with pytest.raises(ValueError, match=r"supported_formats"):
        await file_handler_registry.process_file(gif)


@pytest.mark.asyncio
async def test_svg_capability_absent_rejects_not_placeholder(tmp_path, monkeypatch):
    """cairosvg-missing machines: .svg must reject, never placeholder."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting", lambda s, k=None, default=None: default
    )
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
    svg = tmp_path / "logo.svg"
    svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    with pytest.raises(ValueError, match="Unsupported image format: .svg"):
        await file_handler_registry.process_file(svg)


@pytest.mark.asyncio
async def test_unknown_extension_keeps_generic_fallthrough(tmp_path, png_only_config):
    blob = tmp_path / "artifact.xyz"
    blob.write_bytes(b"opaque")
    processed = await file_handler_registry.process_file(blob)
    assert processed.file_type == "file"
    assert processed.insert_mode == "inline"
    assert "[File: artifact.xyz" in (processed.content or "")


@pytest.mark.asyncio
async def test_effective_extension_still_routes_to_image_handler(
    tmp_path, png_only_config
):
    png = tmp_path / "pic.png"
    _write_png_bytes(png)
    processed = await file_handler_registry.process_file(png)
    assert processed.file_type == "image"
    assert processed.attachment_data is not None


@pytest.mark.asyncio
async def test_extra_configured_format_routes_not_rejects(tmp_path, monkeypatch):
    """A user-added format beyond the defaults (e.g. .heic) must route to the
    image handler when configured — the rejection only fires for KNOWN image
    formats the config excluded."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, k=None, default=None: (
            {"supported_formats": [".png", ".heic"]}
            if (section, k) == ("chat", "images")
            else default
        ),
    )
    heic = tmp_path / "photo.heic"
    _write_png_bytes(heic)  # PNG bytes with a .heic name: routing is by suffix
    processed = await file_handler_registry.process_file(heic)
    assert processed.file_type == "image"
