import io, pytest
from PIL import Image

def _png_b64():
    import base64; buf = io.BytesIO(); Image.new("RGB", (8, 8), (10, 10, 200)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def test_swarmui_generate_happy_path(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import swarmui_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    calls = []
    def fake_fetch_json(method, url, **kw):
        calls.append(url)
        if url.endswith("/API/GetNewSession"):
            return {"session_id": "sess-1"}
        return {"images": [{"image": _png_b64()}]}
    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)
    req = ImageGenRequest(backend="swarmui", prompt="dragon", negative_prompt=None, width=512,
                          height=512, steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None,
                          format="png", extra_params={})
    res = m.SwarmUIAdapter().generate(req)
    assert res.content_type.startswith("image/") and res.bytes_len > 0
    assert any("GetNewSession" in c for c in calls)
