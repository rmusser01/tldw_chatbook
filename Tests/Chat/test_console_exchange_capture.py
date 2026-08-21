"""Capture-core tests: allowlist by construction, stubbing, blob round-trip."""
import json
import zlib

from hypothesis import given, strategies as st

from tldw_chatbook.Chat.console_exchange_capture import (
    CAPTURE_REQUEST_ALLOWLIST,
    EXCHANGE_BLOB_MAX_BYTES,
    ExchangeCapture,
    build_request_capture,
    capture_from_blob,
    capture_to_blob,
    stub_binary_strings,
)


def _kwargs():
    return {
        "api_endpoint": "anthropic",
        "system_message": "You are helpful.",
        "messages_payload": [{"role": "user", "content": "hi"}],
        "api_key": "sk-SECRET",
        "model": "claude-sonnet-5",
        "streaming": True,
        "temp": 0.7,
        "tools": [{"type": "function", "function": {"name": "get_time"}}],
    }


def test_api_key_never_in_capture_and_named_in_omitted():
    request, omitted = build_request_capture(_kwargs())
    assert "sk-SECRET" not in json.dumps(request)
    assert "api_key" not in request
    assert "api_key" in omitted


@given(st.text(min_size=1, max_size=30).filter(lambda k: k not in CAPTURE_REQUEST_ALLOWLIST))
def test_unknown_kwarg_never_persists(key):
    request, omitted = build_request_capture({**_kwargs(), key: "future-secret"})
    assert key not in request
    assert "future-secret" not in json.dumps(request)
    assert key in omitted


def test_allowlisted_content_survives_verbatim():
    request, _ = build_request_capture(_kwargs())
    assert request["system_message"] == "You are helpful."
    assert request["messages_payload"] == [{"role": "user", "content": "hi"}]
    assert request["tools"][0]["function"]["name"] == "get_time"


def test_base64_data_uri_is_stubbed_deterministically():
    blob = "data:image/png;base64," + ("QUJD" * 2000)
    row = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": blob}}]}
    first = stub_binary_strings(row)
    second = stub_binary_strings(row)
    text = json.dumps(first)
    assert "QUJDQUJD" not in text
    assert "image/png" in text and "sha256:" in text
    assert first == second


def test_anthropic_source_b64_is_stubbed():
    row = {"role": "user", "content": [{"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg", "data": "QUJE" * 2000}}]}
    text = json.dumps(stub_binary_strings(row))
    assert "QUJEQUJE" not in text and "image/jpeg" in text


def test_short_strings_untouched():
    row = {"role": "user", "content": "data:image/png;base64,QUJD"}
    assert stub_binary_strings(row) == row


def test_blob_round_trip():
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="2026-08-18T00:00:00Z",
        provider="anthropic", model="claude-sonnet-5", endpoint=None,
        request={"model": "claude-sonnet-5"}, response={"content": "hello"},
        status="complete", usage_json=None, omitted_keys=("api_key",),
    )
    assert capture_from_blob(capture_to_blob(cap)) == cap


def test_blob_is_compressed_json():
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="p", model="m",
        endpoint=None, request={"system_message": "x" * 5000},
        response={}, status="complete", usage_json=None, omitted_keys=(),
    )
    blob = capture_to_blob(cap)
    assert len(blob) < 5000
    assert json.loads(zlib.decompress(blob))["request"]["system_message"] == "x" * 5000


def test_oversize_blob_truncates_with_marker():
    """Review finding M13: the oversize branch must preserve the call's
    REAL outcome in ``status`` (it used to overwrite it with the literal
    string ``"truncated"``, discarding whether the call had completed,
    stopped, or errored) -- truncation is marked separately via a
    ``truncated: True`` key inside the (now stubbed) request/response."""
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="p", model="m",
        endpoint=None,
        request={"messages_payload": [{"role": "user", "content": __import__("os").urandom(15 * 1024 * 1024).hex()}]},
        response={}, status="complete", usage_json=None, omitted_keys=(),
    )
    blob = capture_to_blob(cap)
    assert len(blob) <= EXCHANGE_BLOB_MAX_BYTES
    restored = capture_from_blob(blob)
    assert restored.status == "complete"
    assert restored.request.get("truncated") is True
    assert restored.response.get("truncated") is True


def test_oversize_blob_preserves_error_status():
    """Companion to the above: an oversize capture that was actually an
    "error" (or "stopped") must not be reported as "complete" -- the fix
    keeps whatever status the caller passed in, not a hard-coded value."""
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="p", model="m",
        endpoint=None,
        request={"messages_payload": [{"role": "user", "content": __import__("os").urandom(15 * 1024 * 1024).hex()}]},
        response={}, status="error", usage_json=None, omitted_keys=(),
    )
    restored = capture_from_blob(capture_to_blob(cap))
    assert restored.status == "error"


def test_unserializable_value_degrades_not_raises():
    request, _ = build_request_capture({**_kwargs(), "tools": [object()]})
    json.dumps(request)  # must not raise


def test_capture_from_blob_ignores_unknown_future_fields():
    """Review finding M11: a blob written by a NEWER build with an extra
    field must not raise ``TypeError`` in an older build reading it back --
    ``capture_from_blob`` filters to ``ExchangeCapture``'s own known field
    names before construction."""
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="p", model="m",
        endpoint=None, request={}, response={}, status="complete",
        usage_json=None, omitted_keys=(),
    )
    data = json.loads(zlib.decompress(capture_to_blob(cap)))
    data["a_field_from_the_future"] = "unexpected"
    blob_from_the_future = zlib.compress(json.dumps(data).encode("utf-8"))

    restored = capture_from_blob(blob_from_the_future)

    assert restored.run_tag == "r1"
    assert not hasattr(restored, "a_field_from_the_future")


def test_line_wrapped_base64_is_still_stubbed():
    """Review finding M12: ``_BASE64_RE`` permits embedded whitespace
    (line-wrapped base64), but ``b64decode(..., validate=True)`` rejects
    it outright -- without stripping first, line-wrapped base64 always
    failed validation and was never stubbed, landing in the blob verbatim
    (unredacted size/structure, though still allowlist-filtered content)."""
    raw = "QUJD" * 2000
    wrapped = "\n".join(raw[i : i + 76] for i in range(0, len(raw), 76))
    row = {"role": "user", "content": wrapped}

    stubbed = stub_binary_strings(row)

    text = json.dumps(stubbed)
    assert "QUJDQUJD" not in text
    assert "sha256:" in text
