"""Payload pipeline: config-driven caps, tiff/svg support, truthful mime (TASK-222)."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image as PILImage

import tldw_chatbook.config as config_mod
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import (
    PAYLOAD_FORMAT_MIME,
    ChatImageHandler,
)


def _svg_ready() -> bool:
    from tldw_chatbook.Utils.optional_deps import ensure_svg_rendering

    return ensure_svg_rendering()


svg_required = pytest.mark.skipif(not _svg_ready(), reason="cairosvg unavailable")

SVG_RED_RECT = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
    b'<rect width="40" height="20" fill="red"/></svg>'
)


@pytest.fixture
def defaults_config(monkeypatch):
    """Simulate a config with no [chat.images] overrides (never read live config)."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


def _write_image(tmp_path: Path, name: str, fmt: str, size=(64, 64), mode="RGB") -> Path:
    path = tmp_path / name
    PILImage.new(mode, size, color="red").save(path, format=fmt)
    return path


@pytest.mark.asyncio
async def test_tiff_end_to_end_transcodes_to_png(tmp_path, defaults_config):
    tiff = _write_image(tmp_path, "photo.tiff", "TIFF")
    data, mime = await ChatImageHandler.process_image_file(str(tiff))
    assert mime == "image/png"
    assert PILImage.open(BytesIO(data)).format == "PNG"


@pytest.mark.asyncio
async def test_small_bmp_transcodes_to_png(tmp_path, defaults_config):
    bmp = _write_image(tmp_path, "icon.bmp", "BMP")
    data, mime = await ChatImageHandler.process_image_file(str(bmp))
    assert mime == "image/png"
    assert PILImage.open(BytesIO(data)).format == "PNG"


@pytest.mark.asyncio
async def test_small_png_passes_through_unchanged(tmp_path, defaults_config):
    png = _write_image(tmp_path, "pic.png", "PNG")
    original = png.read_bytes()
    data, mime = await ChatImageHandler.process_image_file(str(png))
    assert data == original
    assert mime == "image/png"


@pytest.mark.asyncio
async def test_large_gif_resizes_with_matching_mime(tmp_path, defaults_config):
    gif = _write_image(tmp_path, "big.gif", "GIF", size=(3000, 1500))
    data, mime = await ChatImageHandler.process_image_file(str(gif))
    img = PILImage.open(BytesIO(data))
    assert max(img.size) <= 2048
    assert PAYLOAD_FORMAT_MIME[img.format] == mime  # mime matches actual bytes


@pytest.mark.asyncio
async def test_cmyk_tiff_transcodes_without_crash(tmp_path, defaults_config):
    tiff = _write_image(tmp_path, "print.tiff", "TIFF", mode="CMYK")
    data, mime = await ChatImageHandler.process_image_file(str(tiff))
    assert mime == "image/png"
    assert PILImage.open(BytesIO(data)).format == "PNG"


@pytest.mark.asyncio
async def test_custom_resize_dimension_honored(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: (
            {"resize_max_dimension": 512}
            if (section, key) == ("chat", "images") else default
        ),
    )
    png = _write_image(tmp_path, "wide.png", "PNG", size=(1024, 800))
    data, _mime = await ChatImageHandler.process_image_file(str(png))
    assert max(PILImage.open(BytesIO(data)).size) <= 512


@pytest.mark.asyncio
async def test_custom_size_cap_rejects_through_real_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: (
            {"max_size_mb": 0.0001}
            if (section, key) == ("chat", "images") else default
        ),
    )
    png = _write_image(tmp_path, "pic.png", "PNG")
    with pytest.raises(ValueError, match="too large"):
        await ChatImageHandler.process_image_file(str(png))


@pytest.mark.asyncio
async def test_custom_formats_reject_through_real_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: (
            {"supported_formats": [".png"]}
            if (section, key) == ("chat", "images") else default
        ),
    )
    gif = _write_image(tmp_path, "anim.gif", "GIF")
    with pytest.raises(ValueError, match="Unsupported image format"):
        await ChatImageHandler.process_image_file(str(gif))


@pytest.mark.asyncio
async def test_process_attachment_bytes_mime_matches_bytes(defaults_config):
    buffer = BytesIO()
    PILImage.new("RGB", (32, 32), "blue").save(buffer, format="PNG")
    pending = await attachment_core.process_attachment_bytes(
        buffer.getvalue(), display_name="clip.png", mime_type="image/png"
    )
    assert pending.mime_type == "image/png"
    assert pending.data is not None


@pytest.mark.asyncio
async def test_process_attachment_bytes_fallback_probes_mime(defaults_config, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("simulated processing failure")

    monkeypatch.setattr(ChatImageHandler, "prepare_image_payload", boom)
    buffer = BytesIO()
    PILImage.new("RGB", (32, 32), "blue").save(buffer, format="PNG")
    pending = await attachment_core.process_attachment_bytes(
        buffer.getvalue(), display_name="clip.png", mime_type="image/jpeg"  # caller lies
    )
    assert pending.data == buffer.getvalue()  # fallback keeps original bytes
    assert pending.mime_type == "image/png"  # probed from the actual bytes


@pytest.mark.asyncio
async def test_svg_rejected_when_capability_absent(tmp_path, monkeypatch, defaults_config):
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
    svg = tmp_path / "logo.svg"
    svg.write_bytes(SVG_RED_RECT)
    with pytest.raises(ValueError, match="Unsupported image format"):
        await ChatImageHandler.process_image_file(str(svg))


@svg_required
@pytest.mark.asyncio
async def test_svg_end_to_end_rasterizes_to_png(tmp_path, defaults_config):
    svg = tmp_path / "logo.svg"
    svg.write_bytes(SVG_RED_RECT)
    data, mime = await ChatImageHandler.process_image_file(str(svg))
    assert mime == "image/png"
    img = PILImage.open(BytesIO(data))
    assert img.format == "PNG"
    assert img.size == (40, 20)


@svg_required
@pytest.mark.asyncio
async def test_svg_oversize_declaration_is_bounded(tmp_path, defaults_config):
    svg = tmp_path / "bomb.svg"
    svg.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="100000" height="100000">'
        b'<rect width="100000" height="100000" fill="red"/></svg>'
    )
    data, _mime = await ChatImageHandler.process_image_file(str(svg))
    assert max(PILImage.open(BytesIO(data)).size) <= 2048


@svg_required
@pytest.mark.asyncio
async def test_svg_xml_entities_rejected(tmp_path, defaults_config):
    svg = tmp_path / "xxe.svg"
    svg.write_bytes(
        b'<?xml version="1.0"?>'
        b'<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/hosts">]>'
        b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        b"<text>&x;</text></svg>"
    )
    with pytest.raises(ValueError, match="Could not render SVG"):
        await ChatImageHandler.process_image_file(str(svg))


@svg_required
@pytest.mark.asyncio
async def test_svg_viewbox_only_preserves_aspect(tmp_path, defaults_config):
    svg = tmp_path / "vb.svg"
    svg.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150">'
        b'<rect width="300" height="150" fill="blue"/></svg>'
    )
    data, _mime = await ChatImageHandler.process_image_file(str(svg))
    assert PILImage.open(BytesIO(data)).size == (300, 150)


@svg_required
@pytest.mark.asyncio
async def test_svg_unparseable_aspect_hard_bounds(tmp_path, defaults_config):
    svg = tmp_path / "pct.svg"
    svg.write_bytes(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">'
        b'<rect width="10" height="10" fill="red"/></svg>'
    )
    data, _mime = await ChatImageHandler.process_image_file(str(svg))
    assert PILImage.open(BytesIO(data)).size == (2048, 2048)
