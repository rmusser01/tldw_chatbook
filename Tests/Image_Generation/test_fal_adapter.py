"""Tests for the fal.ai queue image generation adapter (task-6 of the
fal/Gemini/Fireworks image-backends plan).

Pinned behaviors (see .superpowers/sdd/2026-07-26-imagegen-fal-gemini-fireworks/task-6-brief.md):
- Submit: POST {base}/{validated_model_path}, Authorization: Key {api_key},
  body {"prompt": ..., "seed": ... when set, "image_size": {"width", "height"}
  when both set, "image_url": data URI when a reference image is set}.
- app_id is the first two "/"-segments of the model path (verified against
  fal's own fal_client SDK -- see task-6-report.md); the poll/result URLs
  are ALWAYS self-built from base_url + app_id + request_id, never taken
  from the API response.
- A submit response's own status_url (when present) is a CROSS-CHECK ONLY:
  a mismatch is a loud, sanitized error and the vendor URL is NEVER
  requested.
- Poll status_url until COMPLETED (IN_QUEUE/IN_PROGRESS continue, anything
  else is a sanitized error) within timeout_seconds; then GET result_url
  and fetch images[0].url via fetch_image_bytes with NO trusted_origins and
  NO Authorization header.
- 404 on submit gets task-620 enrichment naming the model path + the
  [image_generation.fal] default_model config key.
"""
import base64
import io

import httpx
import pytest
from PIL import Image


def _b64_png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 100, 0)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _image_bytes_png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 100, 0)).save(buf, "PNG")
    return buf.getvalue()


def _req(**overrides):
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest

    defaults = dict(
        backend="fal",
        prompt="a whale in space",
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
    monkeypatch.setenv("FAL_KEY", "test-fal-key")
    from tldw_chatbook.Image_Generation.adapters import fal_image_adapter as m

    return m


def _submit_response(request_id="req-1", status_url=None):
    resp = {"request_id": request_id}
    if status_url is not None:
        resp["status_url"] = status_url
    return resp


def _status_response(status):
    return {"status": status}


def _result_response(url="https://cdn.fal.media/files/output.png"):
    return {"images": [{"url": url}]}


def _make_fake_fetch_json(calls, *, submit_response, statuses=("COMPLETED",), result_response=None):
    """A fetch_json fake that distinguishes submit/status/result calls by URL shape."""
    status_iter = iter(statuses)
    if result_response is None:
        result_response = _result_response()

    def fake(method, url, **kw):
        calls.append({"method": method, "url": url, **kw})
        if method.upper() == "POST":
            return submit_response
        if url.endswith("/status"):
            return _status_response(next(status_iter))
        return result_response

    return fake


def _patch_no_op_sleep(monkeypatch, m):
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)


# ---------------------------------------------------------------------------
# Submit payload shape
# ---------------------------------------------------------------------------


def test_fal_submit_url_and_headers(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(calls, submit_response=_submit_response())
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell")
    m.FalImageAdapter().generate(req)

    submit_call = calls[0]
    assert submit_call["method"] == "POST"
    assert submit_call["url"] == "https://queue.fal.run/fal-ai/flux/schnell"
    assert submit_call["headers"]["Authorization"] == "Key test-fal-key"
    assert submit_call["headers"]["Content-Type"] == "application/json"
    assert submit_call["trusted_origins"] == frozenset({"queue.fal.run"})


def test_fal_submit_body_basic_shape(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(calls, submit_response=_submit_response())
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell")
    m.FalImageAdapter().generate(req)

    assert calls[0]["json"] == {"prompt": "a whale in space"}


def test_fal_submit_includes_seed_when_set(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(calls, submit_response=_submit_response())
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell", seed=42)
    m.FalImageAdapter().generate(req)

    assert calls[0]["json"]["seed"] == 42


def test_fal_submit_includes_image_size_when_width_and_height_set(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(calls, submit_response=_submit_response())
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell", width=768, height=512)
    m.FalImageAdapter().generate(req)

    assert calls[0]["json"]["image_size"] == {"width": 768, "height": 512}


def test_fal_submit_omits_image_size_when_only_one_dimension_set(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(calls, submit_response=_submit_response())
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell", width=768, height=None)
    m.FalImageAdapter().generate(req)

    assert "image_size" not in calls[0]["json"]


def test_fal_negative_prompt_appended(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(calls, submit_response=_submit_response())
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell", negative_prompt="blurry, low quality")
    m.FalImageAdapter().generate(req)

    prompt = calls[0]["json"]["prompt"]
    assert "a whale in space" in prompt
    assert "blurry, low quality" in prompt


def test_fal_submit_includes_reference_image_data_url(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage

    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(calls, submit_response=_submit_response())
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

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
    req = _req(model="fal-ai/flux/schnell", reference_image=ref)
    m.FalImageAdapter().generate(req)

    expected = "data:image/png;base64," + base64.b64encode(b"abc").decode("ascii")
    assert calls[0]["json"]["image_url"] == expected


# ---------------------------------------------------------------------------
# app_id derivation
# ---------------------------------------------------------------------------


def test_app_id_drops_third_segment(monkeypatch):
    m = _reset_and_import(monkeypatch)
    assert m._app_id("fal-ai/flux/schnell") == "fal-ai/flux"


def test_app_id_keeps_two_segments_unchanged(monkeypatch):
    m = _reset_and_import(monkeypatch)
    assert m._app_id("fal-ai/flux") == "fal-ai/flux"


def test_app_id_rejects_single_segment(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    with pytest.raises(ImageBackendUnavailableError):
        m._app_id("onlyone")


def test_fal_polls_two_segment_app_id_url_for_three_segment_model(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(
        calls,
        submit_response=_submit_response(request_id="req-xyz"),
        statuses=("IN_QUEUE", "IN_PROGRESS", "COMPLETED"),
    )
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell")
    m.FalImageAdapter().generate(req)

    poll_urls = [c["url"] for c in calls if c["url"].endswith("/status")]
    assert poll_urls
    for url in poll_urls:
        assert url == "https://queue.fal.run/fal-ai/flux/requests/req-xyz/status"
    result_calls = [c for c in calls if c["method"].upper() == "GET" and not c["url"].endswith("/status")]
    assert result_calls[0]["url"] == "https://queue.fal.run/fal-ai/flux/requests/req-xyz"


def test_fal_single_segment_model_refused_before_network(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    def _boom(*a, **kw):
        raise AssertionError("fetch_json must not be called for an unresolvable app id")

    monkeypatch.setattr(m, "fetch_json", _boom)
    req = _req(model="onlyone")
    with pytest.raises(ImageBackendUnavailableError):
        m.FalImageAdapter().generate(req)


# ---------------------------------------------------------------------------
# Self-built poll URL cross-check
# ---------------------------------------------------------------------------


def test_fal_matching_status_url_polls_fine(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    expected_status_url = "https://queue.fal.run/fal-ai/flux/requests/req-1/status"
    calls = []
    fake = _make_fake_fetch_json(
        calls,
        submit_response=_submit_response(request_id="req-1", status_url=expected_status_url),
    )
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell")
    res = m.FalImageAdapter().generate(req)
    assert res.bytes_len > 0
    poll_calls = [c for c in calls if c["url"].endswith("/status")]
    assert len(poll_calls) == 1
    assert poll_calls[0]["url"] == expected_status_url


def test_fal_mismatching_status_url_raises_and_never_polls_vendor_url(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    _patch_no_op_sleep(monkeypatch, m)

    vendor_status_url = "https://evil.example/steal/status"
    calls = []
    fake = _make_fake_fetch_json(
        calls,
        submit_response=_submit_response(request_id="req-1", status_url=vendor_status_url),
    )
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.FalImageAdapter().generate(req)

    message = str(exc_info.value)
    assert "fal queue URL shape changed" in message
    assert "https://queue.fal.run/fal-ai/flux/requests/req-1/status" in message
    # Origin only -- never the full vendor URL/path, never credentials.
    assert "steal" not in message
    assert vendor_status_url not in message
    # Only the submit call happened -- no poll request of any kind (self-built
    # or vendor) was ever issued.
    assert len(calls) == 1
    urls_requested = {c["url"] for c in calls}
    assert vendor_status_url not in urls_requested


def test_fal_mismatching_status_url_different_path_same_host(monkeypatch):
    # Same host, different path -- still a mismatch (path is part of the check).
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    _patch_no_op_sleep(monkeypatch, m)

    vendor_status_url = "https://queue.fal.run/fal-ai/other-app/requests/req-1/status"
    calls = []
    fake = _make_fake_fetch_json(
        calls,
        submit_response=_submit_response(request_id="req-1", status_url=vendor_status_url),
    )
    monkeypatch.setattr(m, "fetch_json", fake)

    req = _req(model="fal-ai/flux/schnell")
    with pytest.raises(ImageGenerationError, match="fal queue URL shape changed"):
        m.FalImageAdapter().generate(req)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Poll lifecycle / timeout / failure
# ---------------------------------------------------------------------------


def test_fal_poll_lifecycle_in_queue_then_in_progress_then_completed(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(
        calls,
        submit_response=_submit_response(),
        statuses=("IN_QUEUE", "IN_PROGRESS", "COMPLETED"),
    )
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell")
    res = m.FalImageAdapter().generate(req)
    assert res.bytes_len > 0
    poll_calls = [c for c in calls if c["url"].endswith("/status")]
    assert len(poll_calls) == 3


def test_fal_poll_timeout(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    fake_now = {"t": 0.0}

    def fake_monotonic():
        return fake_now["t"]

    def fake_sleep(interval):
        fake_now["t"] += interval

    monkeypatch.setattr(m.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(m.time, "sleep", fake_sleep)

    calls = []

    def fake_fetch_json(method, url, **kw):
        calls.append({"method": method, "url": url, **kw})
        if method.upper() == "POST":
            return _submit_response()
        return _status_response("IN_QUEUE")  # never completes

    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)

    req = _req(model="fal-ai/flux/schnell")
    with pytest.raises(ImageGenerationError, match="timed out"):
        m.FalImageAdapter().generate(req)


def test_fal_failed_status_raises_sanitized_error(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    _patch_no_op_sleep(monkeypatch, m)

    secret_marker = "SUPER_SECRET_INTERNAL_DETAIL_9182"

    def fake_fetch_json(method, url, **kw):
        if method.upper() == "POST":
            return _submit_response()
        if url.endswith("/status"):
            return {"status": "FAILED", "error": {"message": secret_marker}}
        raise AssertionError("result URL should never be requested after a FAILED status")

    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)

    req = _req(model="fal-ai/flux/schnell")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.FalImageAdapter().generate(req)
    message = str(exc_info.value)
    assert "FAILED" in message
    assert secret_marker not in message


# ---------------------------------------------------------------------------
# Result image fetch: no auth, no trust
# ---------------------------------------------------------------------------


def test_fal_result_image_fetched_without_auth_or_trust(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(
        calls,
        submit_response=_submit_response(),
        result_response=_result_response(url="https://cdn.fal.media/files/whale.png"),
    )
    monkeypatch.setattr(m, "fetch_json", fake)

    fetch_image_calls = []

    def fake_fetch_image_bytes(url, **kw):
        fetch_image_calls.append({"url": url, **kw})
        return _image_bytes_png(), "image/png"

    monkeypatch.setattr(m, "fetch_image_bytes", fake_fetch_image_bytes)

    req = _req(model="fal-ai/flux/schnell")
    m.FalImageAdapter().generate(req)

    assert len(fetch_image_calls) == 1
    image_call = fetch_image_calls[0]
    assert image_call["url"] == "https://cdn.fal.media/files/whale.png"
    assert image_call.get("headers") is None
    assert image_call.get("trusted_origins", frozenset()) == frozenset()
    assert "Authorization" not in (image_call.get("headers") or {})


def test_fal_result_missing_image_url_raises(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    _patch_no_op_sleep(monkeypatch, m)

    def fake_fetch_json(method, url, **kw):
        if method.upper() == "POST":
            return _submit_response()
        if url.endswith("/status"):
            return _status_response("COMPLETED")
        return {"images": []}

    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)

    req = _req(model="fal-ai/flux/schnell")
    with pytest.raises(ImageGenerationError):
        m.FalImageAdapter().generate(req)


# ---------------------------------------------------------------------------
# request_id validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_request_id", ["abc/def", "abc def", "abc?def", "", "abc#def"])
def test_fal_request_id_charset_refusal(monkeypatch, bad_request_id):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    def fake_fetch_json(method, url, **kw):
        if method.upper() == "POST":
            return _submit_response(request_id=bad_request_id) if bad_request_id else {}
        raise AssertionError("polling must never be reached with an invalid/missing request_id")

    monkeypatch.setattr(m, "fetch_json", fake_fetch_json)

    req = _req(model="fal-ai/flux/schnell")
    with pytest.raises(ImageGenerationError):
        m.FalImageAdapter().generate(req)


def test_fal_request_id_valid_charset_accepted(monkeypatch):
    m = _reset_and_import(monkeypatch)
    _patch_no_op_sleep(monkeypatch, m)

    calls = []
    fake = _make_fake_fetch_json(calls, submit_response=_submit_response(request_id="Req-ID-123"))
    monkeypatch.setattr(m, "fetch_json", fake)
    monkeypatch.setattr(m, "fetch_image_bytes", lambda *a, **kw: (_image_bytes_png(), "image/png"))

    req = _req(model="fal-ai/flux/schnell")
    res = m.FalImageAdapter().generate(req)
    assert res.bytes_len > 0


# ---------------------------------------------------------------------------
# Model-path validation matrix
# ---------------------------------------------------------------------------


def test_validate_model_path_accepts_valid(monkeypatch):
    m = _reset_and_import(monkeypatch)
    assert m._validate_model_path("fal-ai/flux/schnell") == "fal-ai/flux/schnell"


@pytest.mark.parametrize(
    "bad_path",
    [
        "../x",
        "a//b",
        "/fal-ai/flux",
        "fal-ai/flux/",
        "fal-ai/flux?x=1",
        "fal-ai/flux#frag",
        "fal-ai/flux%2e%2e",
        "fal ai/flux",
        "",
        "fal-ai/../flux",
    ],
)
def test_validate_model_path_rejects_invalid(monkeypatch, bad_path):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    with pytest.raises(ImageBackendUnavailableError):
        m._validate_model_path(bad_path)


def test_fal_generate_rejects_invalid_model_path_before_network(monkeypatch):
    m = _reset_and_import(monkeypatch)
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    def _boom(*a, **kw):
        raise AssertionError("fetch_json must not be called for an invalid model path")

    monkeypatch.setattr(m, "fetch_json", _boom)
    req = _req(model="../evil")
    with pytest.raises(ImageBackendUnavailableError):
        m.FalImageAdapter().generate(req)


# ---------------------------------------------------------------------------
# 404 enrichment
# ---------------------------------------------------------------------------


def test_fal_404_names_model_path_and_config_key(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)

    def _raise_404(method, url, **kw):
        request = httpx.Request(method, url)
        response = httpx.Response(404, request=request, text="Not Found")
        raise httpx.HTTPStatusError(
            "Client error '404 Not Found' for url '{}'".format(url), request=request, response=response
        )

    monkeypatch.setattr(m, "fetch_json", _raise_404)
    req = _req(model="fal-ai/nonexistent/model")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.FalImageAdapter().generate(req)
    message = str(exc_info.value)
    assert "fal-ai/nonexistent/model" in message
    assert "[image_generation.fal] default_model" in message
    assert "404" in message


def test_fal_non_404_status_keeps_generic_message(monkeypatch):
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    m = _reset_and_import(monkeypatch)

    def _raise_500(method, url, **kw):
        request = httpx.Request(method, url)
        response = httpx.Response(500, request=request, text="Internal Server Error")
        raise httpx.HTTPStatusError(
            "Server error '500 Internal Server Error' for url '{}'".format(url), request=request, response=response
        )

    monkeypatch.setattr(m, "fetch_json", _raise_500)
    req = _req(model="fal-ai/flux/schnell")
    with pytest.raises(ImageGenerationError) as exc_info:
        m.FalImageAdapter().generate(req)
    assert "default_model" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Missing api key
# ---------------------------------------------------------------------------


def test_fal_missing_api_key_raises_backend_unavailable(monkeypatch):
    from tldw_chatbook.Image_Generation import config as _c

    _c.reset_image_generation_config_cache()
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setattr(_c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(_c, "_keyring_get", lambda backend: None, raising=False)

    from tldw_chatbook.Image_Generation.adapters import fal_image_adapter as m
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    req = _req(model="fal-ai/flux/schnell")
    with pytest.raises(ImageBackendUnavailableError):
        m.FalImageAdapter().generate(req)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_fal_registered_in_default_adapters():
    from tldw_chatbook.Image_Generation.adapter_registry import DEFAULT_ADAPTERS

    assert DEFAULT_ADAPTERS["fal"] == (
        "tldw_chatbook.Image_Generation.adapters.fal_image_adapter.FalImageAdapter"
    )


def test_fal_registry_resolves_adapter(monkeypatch):
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
            {"default_backend": "fal", "enabled_backends": ["fal"]},
        )(),
    )
    registry = reg.ImageAdapterRegistry()
    adapter = registry.get_adapter("fal")
    assert adapter is not None
    assert adapter.name == "fal"
    reg.reset_registry()
