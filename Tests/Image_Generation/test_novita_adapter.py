import io
import base64
import pytest
from PIL import Image


def _b64():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 120, 200)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_novita_submit_then_poll(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import novita_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation import config as _c

    # Reset config cache to ensure fresh state
    _c.reset_image_generation_config_cache()

    monkeypatch.setenv("NOVITA_API_KEY", "k")
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)  # skip poll delay

    step = {"n": 0}

    def fake_fetch_json(method, url, **kw):
        # Submit phase: POST to async/txt2img returns task_id
        if "async/txt2img" in url or method.upper() == "POST":
            return {"task_id": "t1"}
        # Poll phase: GET task-result returns status and image
        step["n"] += 1
        return {"status": "succeeded",
                "images": [{"image_url": "data:image/png;base64," + _b64()}]}

    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)

    req = ImageGenRequest(backend="novita", prompt="whale", negative_prompt=None, width=512, height=512,
                          steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None, format="png", extra_params={})
    res = m.NovitaImageAdapter().generate(req)
    assert res.bytes_len > 0


def test_novita_blocks_api_returned_private_ip_image_url(monkeypatch):
    # An API-returned image URL (as opposed to a URL the adapter built itself
    # from its configured base_url) must be fully subject to the task-498
    # egress policy -- a malicious/compromised backend cannot use it to probe
    # the operator's private network (SSRF).
    from tldw_chatbook.Image_Generation.adapters import novita_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    from tldw_chatbook.Image_Generation import config as _c

    _c.reset_image_generation_config_cache()
    monkeypatch.setenv("NOVITA_API_KEY", "k")
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)

    def fake_fetch_json(method, url, **kw):
        if method.upper() == "POST":
            return {"task_id": "t1"}
        return {"status": "succeeded", "images": [{"image_url": "http://192.168.1.50/steal.png"}]}

    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)

    req = ImageGenRequest(backend="novita", prompt="whale", negative_prompt=None, width=512, height=512,
                          steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None, format="png", extra_params={})
    with pytest.raises(ImageGenerationError):
        m.NovitaImageAdapter().generate(req)
