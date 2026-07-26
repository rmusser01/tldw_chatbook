"""Tests for the Gemini (AI Studio) image generation adapter (task-4 of the
fal/Gemini/Fireworks image-backends plan).

Pinned behaviors (see .superpowers/sdd/2026-07-26-imagegen-fal-gemini-fireworks/task-4-brief.md):
- URL: {base}/models/{validated_model}:generateContent
- Auth: x-goog-api-key header ONLY -- never in the URL or query params.
- generationConfig.responseModalities == ["TEXT", "IMAGE"] (verified live against
  Google's docs -- gemini-2.5-flash-image silently returns an empty parts array
  if only ["IMAGE"] is requested; see task-4-report.md for sources).
- Reference image (when set) is an inline_data part BEFORE the text part.
- Response parsing iterates every candidates[*].content.parts[*], accepting
  both inlineData and inline_data spellings defensively.
- No-image error mapping: blockReason > candidate finishReason != STOP > generic.
- 400/404 enrichment names the model id and the config key to check.
- trusted_origins is the self-built URL's own origin only.
"""
import base64
import io

import httpx
import pytest
from PIL import Image


def _b64_png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 180, 0)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _gemini_response(mime_type="image/png", data=None, key="inlineData"):
    return {
        "candidates": [
            {
                "content": {"parts": [{key: {"mimeType": mime_type, "data": data or _b64_png()}}]},
                "finishReason": "STOP",
            }
        ]
    }


def _req(**overrides):
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest

    defaults = dict(
        backend="gemini",
        prompt="a fox in a forest",
        negative_prompt=None,
        width=None,
        height=None,
        steps=None,
        cfg_scale=None,
        seed=None,
        sampler=None,
        model=None,
        format="png",
        extra_params={},
    )
    defaults.update(overrides)
    return ImageGenRequest(**defaults)


def _reset_and_import(monkeypatch):
    from tldw_chatbook.Image_Generation import config as _c

    _c.reset_image_generation_config_cache()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from tldw_chatbook.Image_Generation.adapters import gemini_image_adapter as m

    return m


# ---------------------------------------------------------------------------
# Payload shape / headers / URL / trusted_origins
# ---------------------------------------------------------------------------


def test_gemini_url_and_headers_key_only_in_header(monkeypatch):
    m = _reset_and_import(monkeypatch)

    seen = {}

    def _capture(method, url, *, headers=None, json=None, params=None, timeout=None, trusted_origins=frozenset()):
        seen.update(method=method, url=url, headers=headers, json=json, params=params, trusted_origins=trusted_origins)
        return _gemini_response()

    monkeypatch.setattr(m, "fetch_json", _capture)
    req = _req(model="gemini-2.5-flash-image")
    m.GeminiImageAdapter().generate(req)

    assert seen["method"] == "POST"
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    assert "test-key" not in seen["url"]
    assert not seen["params"]
    assert seen["headers"]["x-goog-api-key"] == "test-key"
    assert seen["headers"]["Content-Type"] == "application/json"
    # never in a query string form either
    assert "key=" not in seen["url"]


def test_gemini_trusted_origins_is_self_built_url_origin(monkeypatch):
    m = _reset_and_import(monkeypatch)

    seen = {}

    def _capture(method, url, *, headers=None, json=None, params=None, timeout=None, trusted_origins=frozenset()):
        seen["trusted_origins"] = trusted_origins
        seen["url"] = url
        return _gemini_response()

    monkeypatch.setattr(m, "fetch_json", _capture)
    req = _req(model="gemini-2.5-flash-image")
    m.GeminiImageAdapter().generate(req)

    assert seen["trusted_origins"] == frozenset({"generativelanguage.googleapis.com"})


def test_gemini_response_modalities_pinned(monkeypatch):
    # task-620 lesson: verified live against docs, not guessed. See report.
    m = _reset_and_import(monkeypatch)

    seen = {}

    def _capture(method, url, *, headers=None, json=None, params=None, timeout=None, trusted_origins=frozenset()):
        seen["json"] = json
        return _gemini_response()

    monkeypatch.setattr(m, "fetch_json", _capture)
    req = _req(model="gemini-2.5-flash-image")
    m.GeminiImageAdapter().generate(req)

    assert seen["json"]["generationConfig"] == {"responseModalities": ["TEXT", "IMAGE"]}


def test_gemini_negative_prompt_appended(monkeypatch):
    m = _reset_and_import(monkeypatch)

    seen = {}

    def _capture(method, url, *, headers=None, json=None, params=None, timeout=None, trusted_origins=frozenset()):
        seen["json"] = json
        return _gemini_response()

    monkeypatch.setattr(m, "fetch_json", _capture)
    req = _req(model="gemini-2.5-flash-image", negative_prompt="blurry, low quality")
    m.GeminiImageAdapter().generate(req)

    parts = seen["json"]["contents"][0]["parts"]
    text_part = parts[-1]
    assert "a fox in a forest" in text_part["text"]
    assert "blurry, low quality" in text_part["text"]


def test_gemini_body_shape_no_reference(monkeypatch):
    m = _reset_and_import(monkeypatch)

    seen = {}

    def _capture(method, url, *, headers=None, json=None, params=None, timeout=None, trusted_origins=frozenset()):
        seen["json"] = json
        return _gemini_response()

    monkeypatch.setattr(m, "fetch_json", _capture)
    req = _req(model="gemini-2.5-flash-image")
    m.GeminiImageAdapter().generate(req)

    assert seen["json"] == {
        "contents": [{"parts": [{"text": "a fox in a forest"}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }


# ---------------------------------------------------------------------------
# Reference image part shape + ordering
# ---------------------------------------------------------------------------


def test_gemini_reference_image_part_before_text(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage

    seen = {}

    def _capture(method, url, *, headers=None, json=None, params=None, timeout=None, trusted_origins=frozenset()):
        seen["json"] = json
        return _gemini_response()

    monkeypatch.setattr(m, "fetch_json", _capture)
    ref = ResolvedReferenceImage(
        file_id=1,
        filename="ref.png",
        mime_type="image/png",
        width=8,
        height=8,
        bytes_len=3,
        content=b"abc",
        temp_path=None,
    )
    req = _req(model="gemini-2.5-flash-image", reference_image=ref)
    m.GeminiImageAdapter().generate(req)

    parts = seen["json"]["contents"][0]["parts"]
    assert len(parts) == 2
    assert parts[0] == {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(b"abc").decode("ascii"),
        }
    }
    assert "text" in parts[1]


def test_gemini_reference_image_via_temp_path(monkeypatch, tmp_path):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage

    ref_file = tmp_path / "ref.png"
    ref_file.write_bytes(b"xyz123")

    seen = {}

    def _capture(method, url, *, headers=None, json=None, params=None, timeout=None, trusted_origins=frozenset()):
        seen["json"] = json
        return _gemini_response()

    monkeypatch.setattr(m, "fetch_json", _capture)
    ref = ResolvedReferenceImage(
        file_id=1,
        filename="ref.png",
        mime_type="image/jpeg",
        width=8,
        height=8,
        bytes_len=6,
        content=None,
        temp_path=str(ref_file),
    )
    req = _req(model="gemini-2.5-flash-image", reference_image=ref)
    m.GeminiImageAdapter().generate(req)

    parts = seen["json"]["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/jpeg"
    assert parts[0]["inline_data"]["data"] == base64.b64encode(b"xyz123").decode("ascii")


# ---------------------------------------------------------------------------
# Response parsing: parts iteration across candidates
# ---------------------------------------------------------------------------


def test_gemini_extracts_image_text_then_image_parts(monkeypatch):
    m = _reset_and_import(monkeypatch)

    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Here is your fox:"},
                        {"inlineData": {"mimeType": "image/png", "data": _b64_png()}},
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }
    monkeypatch.setattr(m, "fetch_json", lambda *a, **kw: response)
    req = _req(model="gemini-2.5-flash-image")
    res = m.GeminiImageAdapter().generate(req)
    assert res.bytes_len > 0
    assert res.content_type == "image/png"


def test_gemini_extracts_image_from_second_candidate(monkeypatch):
    m = _reset_and_import(monkeypatch)

    response = {
        "candidates": [
            {"content": {"parts": [{"text": "no image here"}]}, "finishReason": "STOP"},
            {
                "content": {"parts": [{"inline_data": {"mime_type": "image/png", "data": _b64_png()}}]},
                "finishReason": "STOP",
            },
        ]
    }
    monkeypatch.setattr(m, "fetch_json", lambda *a, **kw: response)
    req = _req(model="gemini-2.5-flash-image")
    res = m.GeminiImageAdapter().generate(req)
    assert res.bytes_len > 0


def test_gemini_snake_case_inline_data_spelling_accepted(monkeypatch):
    m = _reset_and_import(monkeypatch)

    response = _gemini_response(key="inline_data")
    monkeypatch.setattr(m, "fetch_json", lambda *a, **kw: response)
    req = _req(model="gemini-2.5-flash-image")
    res = m.GeminiImageAdapter().generate(req)
    assert res.bytes_len > 0


# ---------------------------------------------------------------------------
# No-image error matrix
# ---------------------------------------------------------------------------


def test_gemini_no_image_block_reason(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)
    response = {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}
    monkeypatch.setattr(m, "fetch_json", lambda *a, **kw: response)
    req = _req(model="gemini-2.5-flash-image")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.GeminiImageAdapter().generate(req)
    assert str(exc_info.value) == "Gemini blocked the prompt (SAFETY)"


def test_gemini_no_image_finish_reason_not_stop(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)
    response = {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
    monkeypatch.setattr(m, "fetch_json", lambda *a, **kw: response)
    req = _req(model="gemini-2.5-flash-image")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.GeminiImageAdapter().generate(req)
    assert str(exc_info.value) == "Gemini returned no image (SAFETY)"


def test_gemini_no_image_generic_fallback(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)
    response = {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}
    monkeypatch.setattr(m, "fetch_json", lambda *a, **kw: response)
    req = _req(model="gemini-2.5-flash-image")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.GeminiImageAdapter().generate(req)
    assert str(exc_info.value) == "Gemini returned no image"


def test_gemini_no_image_error_never_leaks_response_text_or_prompt(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)
    marker = "SUPER_SECRET_MARKER_TEXT_1234"
    response = {
        "candidates": [
            {
                "content": {"parts": [{"text": marker}]},
                "finishReason": "SAFETY",
            }
        ]
    }
    monkeypatch.setattr(m, "fetch_json", lambda *a, **kw: response)
    req = _req(model="gemini-2.5-flash-image", prompt="a very unique prompt marker XYZ987")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.GeminiImageAdapter().generate(req)
    message = str(exc_info.value)
    assert marker not in message
    assert "XYZ987" not in message
    assert message == "Gemini returned no image (SAFETY)"


def test_gemini_block_reason_takes_priority_over_finish_reason(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)
    response = {
        "promptFeedback": {"blockReason": "OTHER"},
        "candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}],
    }
    monkeypatch.setattr(m, "fetch_json", lambda *a, **kw: response)
    req = _req(model="gemini-2.5-flash-image")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.GeminiImageAdapter().generate(req)
    assert str(exc_info.value) == "Gemini blocked the prompt (OTHER)"


# ---------------------------------------------------------------------------
# Model-id validation matrix
# ---------------------------------------------------------------------------


def test_validate_model_id_accepts_valid_charset(monkeypatch):
    m = _reset_and_import(monkeypatch)
    assert m._validate_model_id("good-model_1.0") == "good-model_1.0"


@pytest.mark.parametrize("bad_id", ["../evil", "a?b", "a b", "", "a/b"])
def test_validate_model_id_rejects_invalid_charset(monkeypatch, bad_id):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    with pytest.raises(ImageBackendUnavailableError) as exc_info:
        m._validate_model_id(bad_id)
    assert bad_id in str(exc_info.value)


def test_gemini_generate_rejects_invalid_model_id_before_network(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    def _boom(*a, **kw):
        raise AssertionError("fetch_json must not be called for an invalid model id")

    monkeypatch.setattr(m, "fetch_json", _boom)
    req = _req(model="../evil")
    with pytest.raises(ImageBackendUnavailableError):
        m.GeminiImageAdapter().generate(req)


# ---------------------------------------------------------------------------
# 404 / 400 enrichment (task-620 lesson)
# ---------------------------------------------------------------------------


def test_gemini_404_names_model_and_config_path(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)

    def _raise_404(method, url, **kw):
        request = httpx.Request(method, url)
        response = httpx.Response(404, request=request, text="Not Found")
        raise httpx.HTTPStatusError(
            "Client error '404 Not Found' for url '{}'".format(url), request=request, response=response
        )

    monkeypatch.setattr(m, "fetch_json", _raise_404)
    req = _req(model="gemini-nonexistent-model")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.GeminiImageAdapter().generate(req)
    message = str(exc_info.value)
    assert "gemini-nonexistent-model" in message
    assert "[image_generation.gemini] default_model" in message
    assert "404" in message


def test_gemini_400_names_model_and_config_path(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)

    def _raise_400(method, url, **kw):
        request = httpx.Request(method, url)
        response = httpx.Response(400, request=request, text="Bad Request")
        raise httpx.HTTPStatusError(
            "Client error '400 Bad Request' for url '{}'".format(url), request=request, response=response
        )

    monkeypatch.setattr(m, "fetch_json", _raise_400)
    req = _req(model="gemini-nonexistent-model")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.GeminiImageAdapter().generate(req)
    message = str(exc_info.value)
    assert "gemini-nonexistent-model" in message
    assert "[image_generation.gemini] default_model" in message
    assert "400" in message


def test_gemini_non_404_400_status_keeps_generic_message(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)

    def _raise_500(method, url, **kw):
        request = httpx.Request(method, url)
        response = httpx.Response(500, request=request, text="Internal Server Error")
        raise httpx.HTTPStatusError(
            "Server error '500 Internal Server Error' for url '{}'".format(url), request=request, response=response
        )

    monkeypatch.setattr(m, "fetch_json", _raise_500)
    req = _req(model="gemini-2.5-flash-image")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.GeminiImageAdapter().generate(req)
    message = str(exc_info.value)
    assert "default_model" not in message


# ---------------------------------------------------------------------------
# Missing api key
# ---------------------------------------------------------------------------


def test_gemini_missing_api_key_raises_backend_unavailable(monkeypatch):
    from tldw_chatbook.Image_Generation import config as _c

    _c.reset_image_generation_config_cache()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(_c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(_c, "_keyring_get", lambda backend: None, raising=False)

    from tldw_chatbook.Image_Generation.adapters import gemini_image_adapter as m
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    req = _req(model="gemini-2.5-flash-image")
    with pytest.raises(ImageBackendUnavailableError):
        m.GeminiImageAdapter().generate(req)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_gemini_registered_in_default_adapters():
    from tldw_chatbook.Image_Generation.adapter_registry import DEFAULT_ADAPTERS

    assert DEFAULT_ADAPTERS["gemini"] == (
        "tldw_chatbook.Image_Generation.adapters.gemini_image_adapter.GeminiImageAdapter"
    )


def test_gemini_registry_resolves_adapter(monkeypatch):
    from tldw_chatbook.Image_Generation import config as _c
    from tldw_chatbook.Image_Generation import adapter_registry as reg

    _c.reset_image_generation_config_cache()
    reg.reset_registry()
    monkeypatch.setattr(
        reg,
        "get_image_generation_config",
        lambda: type(
            "Cfg",
            (),
            {"default_backend": "gemini", "enabled_backends": ["gemini"]},
        )(),
    )
    registry = reg.ImageAdapterRegistry()
    adapter = registry.get_adapter("gemini")
    assert adapter is not None
    assert adapter.name == "gemini"
    reg.reset_registry()
