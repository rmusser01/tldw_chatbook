"""Test ModelStudio image adapter."""

import base64
import io

from PIL import Image

from tldw_chatbook.Image_Generation import config as _c
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest


def _b64():
    """Generate a test 8x8 PNG in base64."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 0, 160)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_modelstudio_sync_no_reference_image(monkeypatch):
    """Test ModelStudio sync mode with no reference image.

    Verifies:
    - Image content is extracted from sync response
    - reference_image_data_url is never called when reference_image=None
    """
    from tldw_chatbook.Image_Generation.adapters import modelstudio_image_adapter as m

    _c.reset_image_generation_config_cache()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    # reference_image=None must never call reference_image_data_url
    monkeypatch.setattr(
        m,
        "reference_image_data_url",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("reference_image_data_url must not be called")
        ),
    )
    monkeypatch.setattr(
        m,
        "fetch_json",
        lambda method, url, **kw: {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"image": "data:image/png;base64," + _b64()}]
                        }
                    }
                ]
            }
        },
    )
    req = ImageGenRequest(
        backend="modelstudio",
        prompt="lotus",
        negative_prompt=None,
        width=None,
        height=None,
        steps=None,
        cfg_scale=None,
        seed=None,
        sampler=None,
        model="qwen-image",
        format="png",
        extra_params={"mode": "sync"},
        reference_image=None,
    )
    res = m.ModelStudioImageAdapter().generate(req)
    assert res.bytes_len > 0
