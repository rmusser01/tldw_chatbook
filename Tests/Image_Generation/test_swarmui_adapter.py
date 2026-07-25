import io

import pytest
from PIL import Image

def _png_b64():
    import base64
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 10, 200)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _fake_fetch_json_factory(calls):
    def fake_fetch_json(method, url, **kw):
        calls.append(url)
        if url.endswith("/API/GetNewSession"):
            return {"session_id": "sess-1"}
        return {"images": [{"image": _png_b64()}]}
    return fake_fetch_json


def test_swarmui_generate_happy_path(monkeypatch):
    from tldw_chatbook.Image_Generation.adapters import swarmui_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    calls = []
    monkeypatch.setattr(m, "fetch_json", _fake_fetch_json_factory(calls))
    req = ImageGenRequest(backend="swarmui", prompt="dragon", negative_prompt=None, width=512,
                          height=512, steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None,
                          format="png", extra_params={})
    res = m.SwarmUIAdapter().generate(req)
    assert res.content_type.startswith("image/") and res.bytes_len > 0
    assert any("GetNewSession" in c for c in calls)


def test_swarmui_threads_trusted_origins_for_configured_base_url(monkeypatch):
    # SwarmUI's default base_url is http://127.0.0.1:7801 (loopback) -- the
    # task-498 egress policy blocks private/loopback IPs by default, so every
    # request the adapter builds from its own base_url must carry
    # trusted_origins={"127.0.0.1"} or local-backend generation would regress.
    from tldw_chatbook.Image_Generation.adapters import swarmui_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    seen_trusted = []

    def fake_fetch_json(method, url, **kw):
        seen_trusted.append(kw.get("trusted_origins"))
        if url.endswith("/API/GetNewSession"):
            return {"session_id": "sess-1"}
        return {"images": [{"image": _png_b64()}]}

    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)
    req = ImageGenRequest(backend="swarmui", prompt="dragon", negative_prompt=None, width=512,
                          height=512, steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None,
                          format="png", extra_params={})
    m.SwarmUIAdapter().generate(req)
    assert seen_trusted, "fetch_json was never called"
    assert all(trust == frozenset({"127.0.0.1"}) for trust in seen_trusted)


def test_swarmui_image_fetch_threads_trusted_origins(monkeypatch):
    # A relative/same-origin image ref resolves against base_url; the byte
    # fetch must also trust that same host, or it would be blocked by the
    # egress policy even though _resolve_image_url already same-origin-gated it.
    from tldw_chatbook.Image_Generation.adapters import swarmui_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest

    def fake_fetch_json(method, url, **kw):
        if url.endswith("/API/GetNewSession"):
            return {"session_id": "sess-1"}
        return {"images": [{"image": "/View/local/raw/img.png"}]}

    seen_kwargs = {}

    def fake_fetch_shared_image_bytes(url, **kw):
        seen_kwargs["trusted_origins"] = kw.get("trusted_origins")
        import base64
        return base64.b64decode(_png_b64().split(",", 1)[1]), "image/png"

    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(m, "fetch_shared_image_bytes", fake_fetch_shared_image_bytes)
    req = ImageGenRequest(backend="swarmui", prompt="dragon", negative_prompt=None, width=512,
                          height=512, steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None,
                          format="png", extra_params={})
    res = m.SwarmUIAdapter().generate(req)
    assert res.bytes_len > 0
    assert seen_kwargs["trusted_origins"] == frozenset({"127.0.0.1"})


def test_resolve_image_url_accepts_absolute_same_origin_ref():
    """SwarmUI accepts an absolute image URL on the exact configured origin."""
    from tldw_chatbook.Image_Generation.adapters.swarmui_adapter import SwarmUIAdapter

    url = SwarmUIAdapter._resolve_image_url(
        "http://127.0.0.1:7801", "http://127.0.0.1:7801/View/local/raw/img.png"
    )
    assert url == "http://127.0.0.1:7801/View/local/raw/img.png"


def test_resolve_image_url_rejects_scheme_mismatch():
    """A scheme mismatch against the configured base_url is rejected (task-568)."""
    from tldw_chatbook.Image_Generation.adapters.swarmui_adapter import SwarmUIAdapter
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    with pytest.raises(ImageGenerationError, match="off-origin"):
        SwarmUIAdapter._resolve_image_url(
            "http://127.0.0.1:7801", "https://127.0.0.1:7801/View/local/raw/img.png"
        )


def test_resolve_image_url_rejects_port_mismatch():
    """A port mismatch against the configured base_url is rejected (task-568)."""
    from tldw_chatbook.Image_Generation.adapters.swarmui_adapter import SwarmUIAdapter
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    with pytest.raises(ImageGenerationError, match="off-origin"):
        SwarmUIAdapter._resolve_image_url(
            "http://127.0.0.1:7801", "http://127.0.0.1:9999/View/local/raw/img.png"
        )


def test_swarmui_generate_reports_resolved_model_from_configured_default(monkeypatch):
    """task-558: when the request carries no explicit model, the card's
    "resolved model" should reflect the configured default SwarmUI actually
    used -- an already client-side-resolved value, not fabricated."""
    from dataclasses import replace as dc_replace
    from tldw_chatbook.Image_Generation.adapters import swarmui_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    calls = []
    monkeypatch.setattr(m, "fetch_json", _fake_fetch_json_factory(calls))
    req = ImageGenRequest(backend="swarmui", prompt="dragon", negative_prompt=None, width=512,
                          height=512, steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None,
                          format="png", extra_params={})
    adapter = m.SwarmUIAdapter()
    adapter._config = dc_replace(
        adapter._config,
        swarmui_default_model="OfficialStableDiffusion/sd_xl_base_1.0",
    )
    res = adapter.generate(req)
    assert res.resolved_model == "OfficialStableDiffusion/sd_xl_base_1.0"
    # SwarmUI's response body carries no seed we can trust without guessing
    # an undocumented filename encoding -- must not fabricate one.
    assert res.resolved_seed is None


def test_swarmui_generate_resolved_model_prefers_explicit_request_model(monkeypatch):
    """An explicit request model wins over the configured default in resolved_model."""
    from dataclasses import replace as dc_replace
    from tldw_chatbook.Image_Generation.adapters import swarmui_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    calls = []
    monkeypatch.setattr(m, "fetch_json", _fake_fetch_json_factory(calls))
    req = ImageGenRequest(backend="swarmui", prompt="dragon", negative_prompt=None, width=512,
                          height=512, steps=20, cfg_scale=7.0, seed=-1, sampler=None,
                          model="custom-checkpoint", format="png", extra_params={})
    adapter = m.SwarmUIAdapter()
    adapter._config = dc_replace(adapter._config, swarmui_default_model="fallback-model")
    res = adapter.generate(req)
    assert res.resolved_model == "custom-checkpoint"


def test_swarmui_generate_resolved_model_none_when_unconfigured(monkeypatch):
    """resolved_model stays None when neither request nor config names a model."""
    from dataclasses import replace as dc_replace
    from tldw_chatbook.Image_Generation.adapters import swarmui_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    calls = []
    monkeypatch.setattr(m, "fetch_json", _fake_fetch_json_factory(calls))
    req = ImageGenRequest(backend="swarmui", prompt="dragon", negative_prompt=None, width=512,
                          height=512, steps=20, cfg_scale=7.0, seed=-1, sampler=None, model=None,
                          format="png", extra_params={})
    adapter = m.SwarmUIAdapter()
    adapter._config = dc_replace(adapter._config, swarmui_default_model=None)
    res = adapter.generate(req)
    assert res.resolved_model is None
