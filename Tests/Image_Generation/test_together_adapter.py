import base64
import io

from PIL import Image


def _b64():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (180, 120, 0)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_together_extracts_image_and_no_v1_doubling(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import together_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation import config as _c
    _c.reset_image_generation_config_cache()
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    seen = {}
    def fake_fetch_json(method, url, **kw):
        seen["url"] = url
        return {"data": [{"b64_json": _b64()}]}
    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)
    req = ImageGenRequest(backend="together", prompt="owl", negative_prompt=None, width=512, height=512,
                          steps=None, cfg_scale=None, seed=None, sampler=None,
                          model="black-forest-labs/FLUX.1-schnell-Free", format="png", extra_params={})
    res = m.TogetherImageAdapter().generate(req)
    assert res.bytes_len > 0
    # default base_url ends in /v1; the adapter must NOT produce /v1/v1/ (spec verification)
    assert "/v1/v1/" not in seen["url"]
