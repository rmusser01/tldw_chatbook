"""The image-gen demo decodes on the worker; the loop only updates widgets."""

import io
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.image_gen_demo_screen import ImageGenDemoScreen

pytestmark = pytest.mark.bootstrap_profile

pytest.importorskip("PIL")
pytest.importorskip("rich_pixels")


class _DecodeScreen(ImageGenDemoScreen):
    def __init__(self) -> None:  # bare shell
        pass


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (160, 90), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_result_produces_pixels_and_meta():
    screen = _DecodeScreen()
    payload = _png_bytes()
    res = SimpleNamespace(
        content=payload, content_type="image/png", bytes_len=len(payload)
    )
    req = SimpleNamespace(backend="test", seed=7, prompt="a red square")

    pixels, meta = screen._decode_result(res, req)

    assert pixels is not None
    assert "backend=test" in meta and "seed=7" in meta


def test_apply_result_only_updates_widgets():
    screen = _DecodeScreen()
    updates = []

    def query_one(selector, cls=None):
        return SimpleNamespace(update=lambda value: updates.append((selector, value)))

    screen.query_one = query_one
    screen._apply_result("pixels", "meta text")

    assert [selector for selector, _ in updates] == [
        "#imagegen-status",
        "#imagegen-image",
        "#imagegen-meta",
    ]
    assert updates[-1][1] == "meta text"
