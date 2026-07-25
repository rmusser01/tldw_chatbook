import io
import base64
import httpx
import pytest
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


def test_openrouter_blocks_api_returned_private_ip_image_url(monkeypatch):
    from tldw_chatbook.Image_Generation import config as _c
    _c.reset_image_generation_config_cache()

    from tldw_chatbook.Image_Generation.adapters import openrouter_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(m, "fetch_json", lambda method, url, **kw: {
        "choices": [{"message": {"images": [{"image_url": {"url": "http://192.168.1.50/steal.png"}}]}}]
    })
    req = ImageGenRequest(backend="openrouter", prompt="fox", negative_prompt=None, width=None, height=None,
                          steps=None, cfg_scale=None, seed=None, sampler=None, model="openai/gpt-image-1",
                          format="png", extra_params={})
    with pytest.raises(ImageGenerationError):
        m.OpenRouterImageAdapter().generate(req)


def test_openrouter_payload_uses_new_default_model_when_unconfigured(monkeypatch):
    # task-620: the shipped default must actually reach the request payload
    # when nothing (request.model, env, config) overrides it.
    from tldw_chatbook.Image_Generation import config as _c
    _c.reset_image_generation_config_cache()
    monkeypatch.setattr(_c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(_c, "_keyring_get", lambda backend: None, raising=False)
    monkeypatch.delenv("OPENROUTER_IMAGE_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    from tldw_chatbook.Image_Generation.adapters import openrouter_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest

    seen = {}

    def _capture(method, url, *, json=None, **kw):
        seen["payload"] = json
        return {"choices": [{"message": {"images": [{"image_url": {"url": "data:image/png;base64," + _b64()}}]}}]}

    monkeypatch.setattr(m, "fetch_json", _capture)
    req = ImageGenRequest(backend="openrouter", prompt="fox", negative_prompt=None, width=None, height=None,
                          steps=None, cfg_scale=None, seed=None, sampler=None, model=None,
                          format="png", extra_params={})
    m.OpenRouterImageAdapter().generate(req)
    assert seen["payload"]["model"] == "google/gemini-2.5-flash-image"
    assert seen["payload"]["model"] == m.DEFAULT_OPENROUTER_IMAGE_MODEL


def test_openrouter_404_names_model_and_config_path(monkeypatch):
    # task-620: a bare httpx 404 ("Client error '404 Not Found' for url
    # '...chat/completions'") is undiagnosable on its own -- the adapter must
    # enrich it with the attempted model id and the config key to check.
    from tldw_chatbook.Image_Generation import config as _c
    _c.reset_image_generation_config_cache()

    from tldw_chatbook.Image_Generation.adapters import openrouter_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    def _raise_404(method, url, **kw):
        request = httpx.Request(method, url)
        response = httpx.Response(404, request=request, text="Not Found")
        raise httpx.HTTPStatusError(
            "Client error '404 Not Found' for url '{}'".format(url), request=request, response=response
        )

    monkeypatch.setattr(m, "fetch_json", _raise_404)
    req = ImageGenRequest(backend="openrouter", prompt="fox", negative_prompt=None, width=None, height=None,
                          steps=None, cfg_scale=None, seed=None, sampler=None, model="openai/gpt-image-1",
                          format="png", extra_params={})
    with pytest.raises(ImageGenerationError) as exc_info:
        m.OpenRouterImageAdapter().generate(req)
    message = str(exc_info.value)
    assert "openai/gpt-image-1" in message
    assert "[image_generation.openrouter] default_model" in message
    assert "404" in message


def test_openrouter_non_404_status_keeps_generic_message(monkeypatch):
    # Other statuses must not be re-worded into the 404-specific message.
    from tldw_chatbook.Image_Generation import config as _c
    _c.reset_image_generation_config_cache()

    from tldw_chatbook.Image_Generation.adapters import openrouter_image_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    def _raise_500(method, url, **kw):
        request = httpx.Request(method, url)
        response = httpx.Response(500, request=request, text="Internal Server Error")
        raise httpx.HTTPStatusError(
            "Server error '500 Internal Server Error' for url '{}'".format(url), request=request, response=response
        )

    monkeypatch.setattr(m, "fetch_json", _raise_500)
    req = ImageGenRequest(backend="openrouter", prompt="fox", negative_prompt=None, width=None, height=None,
                          steps=None, cfg_scale=None, seed=None, sampler=None, model="openai/gpt-image-1",
                          format="png", extra_params={})
    with pytest.raises(ImageGenerationError) as exc_info:
        m.OpenRouterImageAdapter().generate(req)
    message = str(exc_info.value)
    assert message.startswith("OpenRouter request failed:")
    assert "default_model" not in message
