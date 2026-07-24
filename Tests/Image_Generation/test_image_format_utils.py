import io
import pytest
from PIL import Image


def _png_bytes(size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def ifu():
    from tldw_chatbook.Image_Generation.adapters import image_format_utils as m
    return m


def test_format_from_bytes_detects_png(ifu):
    assert ifu.format_from_bytes(_png_bytes()) == "png"


def test_validate_and_convert_output_roundtrip(ifu):
    data, ctype = ifu.validate_and_convert_image_output(_png_bytes(), "image/png", "png", max_bytes=10_000_000)
    assert ctype == "image/png" and isinstance(data, (bytes, bytearray))


def test_validate_rejects_when_over_max_bytes(ifu):
    with pytest.raises(Exception):
        ifu.validate_and_convert_image_output(_png_bytes((256, 256)), "image/png", "png", max_bytes=10)
