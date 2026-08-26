"""Capture-core tests: allowlist by construction, stubbing, blob round-trip."""
import json
import zlib

from hypothesis import given, strategies as st

from tldw_chatbook.Chat.console_exchange_capture import (
    CAPTURE_REQUEST_ALLOWLIST,
    CaptureDetail,
    CaptureBudget,
    CapturePolicyResolution,
    CapturePolicySource,
    EXCHANGE_BLOB_MAX_BYTES,
    ExchangeCapture,
    build_request_capture,
    build_response_capture,
    capture_from_blob,
    capture_to_blob,
    resolve_capture_policy,
    stub_binary_strings,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

_PROJECT_INSTRUCTION_SENTINEL = (
    "SENTINEL: this is the automatically-injected project instruction body "
    "and must never reach an export, a display, or a persisted capture."
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


def test_capture_policy_precedence_and_invalid_values_fail_safe():
    resolved = resolve_capture_policy(
        enabled=True,
        next_send="full",
        conversation="safe",
        global_default="full",
    )
    assert resolved == CapturePolicyResolution(
        enabled=True,
        detail=CaptureDetail.FULL,
        source=CapturePolicySource.NEXT_SEND,
        invalid_sources=(),
    )
    invalid = resolve_capture_policy(enabled=True, global_default="future-value")
    assert invalid.detail is CaptureDetail.SAFE
    assert invalid.source is CapturePolicySource.APPLICATION
    assert invalid.invalid_sources == ("global",)


def test_capture_off_wins_without_forgetting_dormant_detail():
    resolved = resolve_capture_policy(enabled=False, conversation="full")
    assert resolved.enabled is False
    assert resolved.detail is CaptureDetail.FULL
    assert resolved.source is CapturePolicySource.CONVERSATION


def test_safe_omits_but_full_retains_tagged_project_instruction_body():
    kwargs = {"messages_payload": [_project_instruction_row("AGENTS BODY")]}
    safe, safe_omitted = build_request_capture(kwargs, capture_detail=CaptureDetail.SAFE)
    full, full_omitted = build_request_capture(kwargs, capture_detail=CaptureDetail.FULL)
    assert "AGENTS BODY" not in json.dumps(safe)
    assert "messages_payload[0].content" in safe_omitted
    assert full["messages_payload"][0]["content"] == "AGENTS BODY"
    assert "messages_payload[0].content" not in full_omitted


def test_endpoint_identity_drops_credentials_query_and_fragment():
    request, _ = build_request_capture(
        {"api_base_url": "https://user:pass@example.test/v1?q=secret#fragment"},
        capture_detail=CaptureDetail.FULL,
    )
    assert request["api_base_url"] == "https://example.test/v1"


def test_api_endpoint_identity_drops_credentials_query_and_fragment():
    request, _ = build_request_capture(
        {"api_endpoint": "https://user:pass@example.test/v1?q=secret#fragment"},
        capture_detail=CaptureDetail.FULL,
    )
    assert request["api_endpoint"] == "https://example.test/v1"


def test_request_tool_argument_json_removes_credentials_and_stubs_binary():
    argument_json = json.dumps(
        {
            "access_token": "request-access-token",
            "nested": {
                "client_secret": "request-client-secret",
                "attachment": {"media_type": "image/png", "data": "QUJD" * 2000},
            },
            "note": "ordinary semantic text",
        }
    )
    request, _ = build_request_capture(
        {
            "messages_payload": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "upload", "arguments": argument_json}}
                    ],
                }
            ]
        }
    )

    arguments = json.loads(
        request["messages_payload"][0]["tool_calls"][0]["function"]["arguments"]
    )
    assert "access_token" not in arguments
    assert "client_secret" not in arguments["nested"]
    assert arguments["nested"]["attachment"]["data"].startswith("[image/png,")
    assert arguments["note"] == "ordinary semantic text"


def test_response_tool_json_removes_credentials_and_stubs_binary():
    argument_json = json.dumps(
        {"access_token": "response-access-token", "note": "ordinary argument text"}
    )
    result_json = json.dumps(
        {
            "client_secret": "response-client-secret",
            "nested": {"payload": "data:image/png;base64," + ("QUJD" * 2000)},
            "note": "ordinary result text",
        }
    )
    response = build_response_capture(
        content="done",
        tool_calls=[
            {
                "function": {"name": "upload", "arguments": argument_json},
                "result": result_json,
            }
        ],
    )

    tool_call = response["tool_calls"][0]
    arguments = json.loads(tool_call["function"]["arguments"])
    result = json.loads(tool_call["result"])
    assert "access_token" not in arguments
    assert arguments["note"] == "ordinary argument text"
    assert "client_secret" not in result
    assert result["nested"]["payload"].startswith("[image/png,")
    assert result["note"] == "ordinary result text"


def test_response_base64_content_is_stubbed_before_budget_retention():
    raw = "QUJD" * 2000
    response = build_response_capture(content=raw, tool_calls=[])
    assert response["content"] != raw
    assert response["content"].startswith("[application/octet-stream,")


def test_response_data_uri_content_is_stubbed_before_budget_retention():
    raw = "data:image/png;base64," + ("QUJD" * 2000)
    response = build_response_capture(content=raw, tool_calls=[])
    assert response["content"] != raw
    assert response["content"].startswith("[image/png,")


def test_request_and_response_share_one_bounded_budget():
    budget = CaptureBudget(limit_bytes=256)
    request, _ = build_request_capture(
        {"messages_payload": [{"role": "user", "content": "x" * 220}]},
        capture_detail=CaptureDetail.FULL,
        budget=budget,
    )
    response = build_response_capture(
        content="y" * 220,
        tool_calls=[{"function": {"arguments": "QUJD" * 2000}}],
        budget=budget,
    )
    assert request["truncation_inventory"] or response["truncation_inventory"]
    assert budget.used_bytes <= budget.limit_bytes


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


def test_wrapped_and_unwrapped_base64_produce_identical_stub():
    """Qodo PR #1883 finding: `stub_binary_strings` promises deterministic
    stubs (same bytes -> same [mime, size, sha256:...]), but the old code
    validated a whitespace-stripped candidate and then hashed/sized the
    ORIGINAL whitespace-preserving value -- so the same underlying bytes
    wrapped at different line lengths produced different sha256 AND
    different reported size. Same bytes, wrapped at 76 columns vs. not
    wrapped at all, must yield the identical stub string."""
    raw = "QUJD" * 2000
    wrapped = "\n".join(raw[i : i + 76] for i in range(0, len(raw), 76))
    row_wrapped = {"role": "user", "content": wrapped}
    row_unwrapped = {"role": "user", "content": raw}

    stubbed_wrapped = stub_binary_strings(row_wrapped)
    stubbed_unwrapped = stub_binary_strings(row_unwrapped)

    assert stubbed_wrapped["content"] == stubbed_unwrapped["content"]
    assert stubbed_wrapped["content"].startswith("[")


def test_wrapped_and_unwrapped_data_uri_produce_identical_stub():
    """Same determinism guarantee for the data-URI branch: line-wrapping
    the base64 payload inside a data URI must not change the hash/size."""
    raw = "QUJD" * 2000
    wrapped_payload = "\n".join(raw[i : i + 76] for i in range(0, len(raw), 76))
    row_wrapped = {"role": "user", "content": "data:image/png;base64," + wrapped_payload}
    row_unwrapped = {"role": "user", "content": "data:image/png;base64," + raw}

    stubbed_wrapped = stub_binary_strings(row_wrapped)
    stubbed_unwrapped = stub_binary_strings(row_unwrapped)

    assert stubbed_wrapped["content"] == stubbed_unwrapped["content"]


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


# ---------------------------------------------------------------------------
# C1 (CRITICAL): a project-instruction row's body must never survive into a
# captured request -- it is on the wire (Chat_Functions.py's
# ``_project_instruction_messages_for_handler`` only strips the ORIGIN TAG
# before the provider call, never the body), but the Console's own captured
# request is a separate, retained artifact (export, display, at rest in
# ``message_exchanges``) the shipped user guide promises never carries it
# outside the Next Send tab's disposable preview.
# ---------------------------------------------------------------------------


def _project_instruction_row(content: str = _PROJECT_INSTRUCTION_SENTINEL) -> dict:
    return {
        "role": "user",
        "content": content,
        EPHEMERAL_ORIGIN_KEY: "project_instructions",
    }


def test_project_instruction_row_body_is_redacted_from_capture():
    kwargs = {
        **_kwargs(),
        "messages_payload": [
            {"role": "user", "content": "hi"},
            _project_instruction_row(),
        ],
    }

    request, omitted = build_request_capture(kwargs)

    text = json.dumps(request)
    assert _PROJECT_INSTRUCTION_SENTINEL not in text
    rows = request["messages_payload"]
    assert len(rows) == 2
    tagged_row = rows[1]
    # Role and the origin tag both survive -- only the body is replaced --
    # so the Inspector can still show that such a row was sent.
    assert tagged_row["role"] == "user"
    assert tagged_row[EPHEMERAL_ORIGIN_KEY] == "project_instructions"
    assert "omitted by capture policy" in tagged_row["content"]
    assert str(len(_PROJECT_INSTRUCTION_SENTINEL)) in tagged_row["content"]
    # The row is not silently emptied -- something legible marks the
    # withholding, and it's surfaced through the same visibility mechanism
    # the Inspector already renders as "Omitted by capture policy: ...".
    assert any("messages_payload[1].content" == entry for entry in omitted)


def test_non_tagged_message_with_identical_text_is_not_redacted():
    """Proves the filter keys off the ``EPHEMERAL_ORIGIN_KEY`` tag, not the
    content -- a user who happens to type the exact same text as an
    automatic project instruction body must see it captured verbatim."""
    kwargs = {
        **_kwargs(),
        "messages_payload": [
            {"role": "user", "content": _PROJECT_INSTRUCTION_SENTINEL},
        ],
    }

    request, omitted = build_request_capture(kwargs)

    assert request["messages_payload"] == [
        {"role": "user", "content": _PROJECT_INSTRUCTION_SENTINEL}
    ]
    assert not any("messages_payload" in entry for entry in omitted)


def test_project_instruction_redaction_survives_blob_round_trip():
    """The redaction happens in ``build_request_capture``, upstream of
    ``capture_to_blob`` -- confirms the persisted-at-rest form (what
    actually lands in ``message_exchanges``) carries the redacted row, not
    the original."""
    request, omitted = build_request_capture(
        {**_kwargs(), "messages_payload": [_project_instruction_row()]}
    )
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="p", model="m",
        endpoint=None, request=request, response={}, status="complete",
        usage_json=None, omitted_keys=omitted,
    )

    restored = capture_from_blob(capture_to_blob(cap))

    text = json.dumps(restored.request)
    assert _PROJECT_INSTRUCTION_SENTINEL not in text
    assert "omitted by capture policy" in text


def test_non_list_messages_payload_is_left_alone_not_raised():
    """Defensive: an unexpected ``messages_payload`` shape (e.g. ``None``,
    already-frozen tuple) must degrade, not raise, same contract as the
    rest of this module."""
    request, _ = build_request_capture({**_kwargs(), "messages_payload": None})
    assert request["messages_payload"] is None


# ---------------------------------------------------------------------------
# M1: the stub-eligibility length gate must measure the same canonical
# (whitespace-stripped) length the hash/size below it already use -- not the
# raw string length -- so line-wrapping alone cannot push otherwise-
# identical content across the threshold in one direction but not the other.
# ---------------------------------------------------------------------------


def test_stub_gate_uses_canonical_length_not_raw_wrapped_length():
    """Boundary case from the review finding: canonical (whitespace-
    stripped) length just under ``_STUB_MIN_CHARS`` (4096), but line-
    wrapping inflates the RAW length past it. Must NOT be stubbed -- the
    content itself doesn't warrant it."""
    from tldw_chatbook.Chat.console_exchange_capture import _STUB_MIN_CHARS

    canonical = "Q" * (_STUB_MIN_CHARS - 8)  # 4088 chars, under the gate
    assert len(canonical) < _STUB_MIN_CHARS
    wrapped = "\n".join(canonical[i : i + 76] for i in range(0, len(canonical), 76))
    assert len(wrapped) >= _STUB_MIN_CHARS  # raw length crosses the old gate

    row = {"role": "user", "content": wrapped}
    stubbed = stub_binary_strings(row)

    assert stubbed["content"] == wrapped  # passed through untouched


def test_stub_gate_still_fires_at_canonical_length_over_threshold():
    """Companion: canonical length AT/OVER the gate must still stub,
    whitespace or not."""
    from tldw_chatbook.Chat.console_exchange_capture import _STUB_MIN_CHARS

    canonical = "QUJD" * ((_STUB_MIN_CHARS // 4) + 1)
    assert len(canonical) >= _STUB_MIN_CHARS
    row = {"role": "user", "content": canonical}

    stubbed = stub_binary_strings(row)

    assert stubbed["content"] != canonical
    assert stubbed["content"].startswith("[")
