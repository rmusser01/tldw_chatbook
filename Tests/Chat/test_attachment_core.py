import asyncio

import pytest
from PIL import Image as PILImage

from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Chat.attachment_core import (
    DEFAULT_MAX_HISTORY_IMAGES,
    PendingAttachment,
    image_content_parts,
    max_history_images,
    process_attachment_path,
    vision_block_reason,
)


def _write_png(path, size=(4, 4)):
    PILImage.new("RGB", size, color=(200, 10, 10)).save(path, format="PNG")


def test_process_attachment_path_rejects_paths_outside_allowed_root(tmp_path):
    outside = tmp_path / "evil.txt"
    outside.write_text("nope")
    with pytest.raises(ValueError, match="outside allowed directories"):
        asyncio.run(
            process_attachment_path(str(outside), allowed_root=str(tmp_path / "inner"))
        )


def test_process_attachment_path_rejects_oversized_files(tmp_path, monkeypatch):
    big = tmp_path / "big.txt"
    big.write_text("x" * 64)
    monkeypatch.setattr(attachment_core, "MAX_ATTACHMENT_BYTES", 16)
    with pytest.raises(ValueError, match="File too large"):
        asyncio.run(process_attachment_path(str(big), allowed_root=str(tmp_path)))


def test_process_attachment_path_inlines_text_files(tmp_path):
    note = tmp_path / "notes.md"
    note.write_text("# hello\nworld")
    attachment = asyncio.run(
        process_attachment_path(str(note), allowed_root=str(tmp_path))
    )
    assert attachment.insert_mode == "inline"
    assert attachment.file_type == "text"
    assert "world" in (attachment.text_content or "")
    assert attachment.data is None
    assert attachment.display_name == "notes.md"
    assert attachment.label.startswith("notes.md · ")


def test_process_attachment_path_attaches_images(tmp_path):
    image = tmp_path / "photo.png"
    _write_png(image)
    attachment = asyncio.run(
        process_attachment_path(str(image), allowed_root=str(tmp_path))
    )
    assert attachment.insert_mode == "attachment"
    assert attachment.file_type == "image"
    assert isinstance(attachment.data, bytes) and attachment.data
    assert attachment.mime_type == "image/png"
    assert attachment.processed_size == len(attachment.data)


def test_vision_block_reason_none_for_vision_model():
    assert vision_block_reason("OpenAI", "gpt-4o") is None


def test_vision_block_reason_names_model_and_override():
    reason = vision_block_reason("llama_cpp", "text-model-7b")
    assert reason is not None
    assert "text-model-7b" in reason
    assert "can't accept images" in reason
    assert "[model_capabilities.models]" in reason


class _StubRegistry:
    def __init__(self, capabilities):
        self._capabilities = capabilities

    def get_model_capabilities(self, provider, model):
        return dict(self._capabilities)


def test_max_history_images_uses_capability_value(monkeypatch):
    monkeypatch.setattr(
        attachment_core,
        "_get_capabilities_registry",
        lambda: _StubRegistry({"vision": True, "max_images": 5}),
    )
    assert max_history_images("Anthropic", "any-vision-model") == 5


def test_max_history_images_defaults_when_capability_absent(monkeypatch):
    monkeypatch.setattr(
        attachment_core,
        "_get_capabilities_registry",
        lambda: _StubRegistry({"vision": True}),
    )
    assert max_history_images("FakeProv", "fake-model") == DEFAULT_MAX_HISTORY_IMAGES
    assert max_history_images("OpenAI", None) == DEFAULT_MAX_HISTORY_IMAGES


def test_image_content_parts_builds_data_url():
    parts = image_content_parts("look", b"\x89PNG", "image/png")
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    only_image = image_content_parts("", b"\x89PNG", "image/png")
    assert [p["type"] for p in only_image] == ["image_url"]


def test_process_attachment_bytes_builds_image_pending():
    import asyncio
    from io import BytesIO

    from PIL import Image as PILImage

    from tldw_chatbook.Chat.attachment_core import process_attachment_bytes

    buffer = BytesIO()
    PILImage.new("RGB", (32, 32), (200, 10, 10)).save(buffer, format="PNG")
    data = buffer.getvalue()

    attachment = asyncio.run(
        process_attachment_bytes(data, display_name="clipboard-20260713-120000.png")
    )
    assert attachment.insert_mode == "attachment"
    assert attachment.file_type == "image"
    assert attachment.file_path == ""
    assert attachment.mime_type == "image/png"
    assert attachment.display_name == "clipboard-20260713-120000.png"
    assert attachment.data and attachment.processed_size == len(attachment.data)


def test_process_attachment_bytes_rejects_corrupt_and_oversized(monkeypatch):
    import asyncio

    import pytest as _pytest

    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat.attachment_core import process_attachment_bytes

    with _pytest.raises(ValueError, match="not a valid image"):
        asyncio.run(process_attachment_bytes(b"junk", display_name="x.png"))

    monkeypatch.setattr(attachment_core, "MAX_IMAGE_BYTES", 4)
    with _pytest.raises(ValueError, match="too large"):
        asyncio.run(
            process_attachment_bytes(b"12345678", display_name="big.png")
        )
