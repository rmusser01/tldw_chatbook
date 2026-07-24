import io
import base64
from PIL import Image


def _b64():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 180, 0)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_openrouter_extracts_image(monkeypatch):
    from tldw_chatbook.Image_Generation import config as _c
    _c.reset_image_generation_config_cache()

    from tldw_chatbook.Image_Generation.adapters import openrouter_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(m, "fetch_json", lambda method, url, **kw: {
        "choices": [{"message": {"images": [{"image_url": {"url": "data:image/png;base64," + _b64()}}]}}]
    })
    req = ImageGenRequest(backend="openrouter", prompt="fox", negative_prompt=None, width=None, height=None,
                          steps=None, cfg_scale=None, seed=None, sampler=None, model="openai/gpt-image-1",
                          format="png", extra_params={})
    res = m.OpenRouterImageAdapter().generate(req)
    assert res.bytes_len > 0
