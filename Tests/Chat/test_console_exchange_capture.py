"""Capture-core tests: allowlist by construction, stubbing, blob round-trip."""

import hashlib as _hashlib
import json
import zlib

import pytest
from hypothesis import given, strategies as st

from tldw_chatbook.Chat.console_exchange_capture import (
    CAPTURE_HISTORY_ELISION_KIND,
    CAPTURE_HISTORY_ELISION_VERSION,
    CAPTURE_HISTORY_MARKER_KEYS,
    CAPTURE_HISTORY_MARKER_ROLES,
    CAPTURE_REQUEST_ALLOWLIST,
    CAPTURE_SAFE_HISTORY_TAIL_ROWS,
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
    compact_safe_history_rows,
    history_elision_marker,
    resolve_capture_policy,
    sanitize_capture_value,
    stub_binary_strings,
    trim_safe_capture_blob,
)
from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY
from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_SANITIZER_UNAVAILABLE,
    CredentialSanitizationResult,
)

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
    safe, safe_omitted = build_request_capture(
        kwargs, capture_detail=CaptureDetail.SAFE
    )
    full, full_omitted = build_request_capture(
        kwargs, capture_detail=CaptureDetail.FULL
    )
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


@pytest.mark.parametrize("detail", [CaptureDetail.SAFE, CaptureDetail.FULL])
def test_request_and_response_free_text_credentials_are_mandatorily_sanitized(
    detail,
):
    credential = "sk-live-abcdefghijklmnop"
    endpoint = "https://user:pass@example.invalid/v1?token=query#fragment"
    request, omitted = build_request_capture(
        {
            "system_message": f"Authorization: Bearer {credential}",
            "messages_payload": [{"role": "user", "content": f"send to {endpoint}"}],
            "tools": [
                {
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"note": f"token={credential}"}),
                    }
                }
            ],
        },
        capture_detail=detail,
    )
    response = build_response_capture(
        content=f"provider returned {credential} from {endpoint}",
        tool_calls=[{"function": {"arguments": f"Bearer {credential}"}}],
    )

    encoded = json.dumps({"request": request, "response": response})
    assert credential not in encoded
    assert "user:pass" not in encoded
    assert "query" not in encoded
    assert {"system_message", "messages_payload", "tools"}.issubset(omitted)
    assert set(response["credential_omission_inventory"]) == {
        "content",
        "tool_calls",
    }
    assert "[credential omitted]" in encoded


def test_sanitizer_failure_is_content_free_and_named_in_capture_inventory(
    monkeypatch,
):
    secret = "SANITIZER-FAILURE-CONTENT-CANARY"

    def unavailable(_self, _value):
        return CredentialSanitizationResult(
            available=False,
            value=None,
            omission_reason_code=CREDENTIAL_SANITIZER_UNAVAILABLE,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_exchange_capture.CredentialSanitizer.sanitize",
        unavailable,
    )
    request, omitted = build_request_capture({"system_message": secret})
    response = build_response_capture(content=secret, tool_calls=[])

    encoded = json.dumps({"request": request, "response": response})
    assert secret not in encoded
    assert request["system_message"] == {"omitted": True}
    assert "system_message" in omitted
    assert response["content"] == {"omitted": True}
    assert "content" in response["credential_omission_inventory"]


def test_short_known_credential_is_replaced_in_nested_tool_mapping_keys():
    credential = "abc1234"
    response = build_response_capture(
        content="ordinary response",
        tool_calls=[
            {
                "function": {
                    "arguments": {f"prefix{credential}suffix": "ordinary"}
                }
            }
        ],
        known_credentials=(credential,),
    )

    encoded = json.dumps(response)
    assert credential not in encoded
    assert response["tool_calls"] == [
        {"function": {"arguments": {"[credential omitted]": "ordinary"}}}
    ]
    assert "tool_calls" in response["credential_omission_inventory"]


@given(
    st.text(min_size=1, max_size=30).filter(
        lambda k: k not in CAPTURE_REQUEST_ALLOWLIST
    )
)
def test_unknown_kwarg_never_persists(key):
    request, omitted = build_request_capture({**_kwargs(), key: "future-secret"})
    assert key not in request
    assert "future-secret" not in json.dumps(request)
    assert key not in omitted
    assert "unknown_parameter" in omitted


def test_credential_bearing_unknown_kwarg_name_never_enters_durable_blob():
    credential = "abc1234"
    raw_name = f"prefix{credential}suffix"
    request, omitted = build_request_capture(
        {raw_name: "ordinary"},
        known_credentials=(credential,),
    )
    cap = ExchangeCapture(
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="openai",
        model="m",
        endpoint=None,
        request=request,
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=omitted,
    )

    restored = capture_from_blob(capture_to_blob(cap))
    encoded = json.dumps(
        {"request": restored.request, "omitted_keys": restored.omitted_keys}
    )
    assert credential not in encoded
    assert raw_name not in restored.omitted_keys
    assert restored.omitted_keys == ("unknown_parameter",)


def test_allowlisted_content_survives_verbatim():
    request, _ = build_request_capture(_kwargs())
    assert request["system_message"] == "You are helpful."
    assert request["messages_payload"] == [{"role": "user", "content": "hi"}]
    assert request["tools"][0]["function"]["name"] == "get_time"


def test_base64_data_uri_is_stubbed_deterministically():
    blob = "data:image/png;base64," + ("QUJD" * 2000)
    row = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": blob}}],
    }
    first = stub_binary_strings(row)
    second = stub_binary_strings(row)
    text = json.dumps(first)
    assert "QUJDQUJD" not in text
    assert "image/png" in text and "sha256:" in text
    assert first == second


def test_anthropic_source_b64_is_stubbed():
    row = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "QUJE" * 2000,
                },
            }
        ],
    }
    text = json.dumps(stub_binary_strings(row))
    assert "QUJEQUJE" not in text and "image/jpeg" in text


def test_short_data_uri_is_stubbed():
    row = {"role": "user", "content": "data:image/png;base64,QUJD"}
    assert stub_binary_strings(row)["content"].startswith("[image/png,")


def test_short_explicit_base64_is_stubbed():
    row = {"source": {"type": "base64", "data": "QUJD"}}
    assert stub_binary_strings(row)["source"]["data"].startswith(
        "[application/octet-stream,"
    )


def test_public_capture_value_sanitizer_removes_nested_credentials_and_binary():
    value = {
        "messages": [
            {
                "role": "tool",
                "content": {
                    "api_key": "wire-api-key",
                    "nested": {"access_token": "wire-token"},
                    "result": json.dumps(
                        {
                            "client_secret": "wire-client-secret",
                            "image": "data:image/png;base64,QUJD",
                        }
                    ),
                },
            }
        ]
    }

    sanitized = sanitize_capture_value(value)
    rendered = json.dumps(sanitized)

    assert "wire-api-key" not in rendered
    assert "wire-token" not in rendered
    assert "wire-client-secret" not in rendered
    assert "data:image/png;base64,QUJD" not in rendered


def test_blob_round_trip():
    cap = ExchangeCapture(
        run_tag="r1",
        seq=0,
        created_at="2026-08-18T00:00:00Z",
        provider="anthropic",
        model="claude-sonnet-5",
        endpoint=None,
        request={"model": "claude-sonnet-5"},
        response={"content": "hello"},
        status="complete",
        usage_json=None,
        omitted_keys=("api_key",),
    )
    assert capture_from_blob(capture_to_blob(cap)) == cap


def test_blob_serializer_fails_closed_for_recursive_capture_content():
    recursive: list[object] = []
    recursive.append(recursive)
    cap = ExchangeCapture(
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="openai",
        model="m",
        endpoint=None,
        request={"messages_payload": recursive},
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=(),
    )

    restored = capture_from_blob(capture_to_blob(cap))
    assert restored.request == {"omitted": True}
    assert restored.response == {"omitted": True}
    assert restored.omitted_keys == ("capture",)


def test_blob_is_compressed_json():
    cap = ExchangeCapture(
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request={"system_message": "x" * 5000},
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=(),
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
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request={
            "messages_payload": [
                {
                    "role": "user",
                    "content": __import__("os").urandom(15 * 1024 * 1024).hex(),
                }
            ]
        },
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=(),
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
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request={
            "messages_payload": [
                {
                    "role": "user",
                    "content": __import__("os").urandom(15 * 1024 * 1024).hex(),
                }
            ]
        },
        response={},
        status="error",
        usage_json=None,
        omitted_keys=(),
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
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request={},
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=(),
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
    row_wrapped = {
        "role": "user",
        "content": "data:image/png;base64," + wrapped_payload,
    }
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
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request=request,
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=omitted,
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


# ---------------------------------------------------------------------------
# task-23026 / ADR-096: a Safe capture must not persist the whole
# conversation per turn. Every send's payload carries the entire history so
# far; storing it verbatim per turn re-stored the conversation O(n²)
# (21.33 MB measured for one 200-turn conversation through the real gateway
# path), default-on, with no retention path. Under Safe the retained set is
# first-system ∪ last-user ∪ final-eight rows; everything else becomes ONE
# content-free aggregate marker (spec:
# Docs/superpowers/specs/2026-08-27-console-safe-capture-retention-design.md).
# ---------------------------------------------------------------------------

_TAIL = CAPTURE_SAFE_HISTORY_TAIL_ROWS


def _incompressible_filler(seed: str, chars: int) -> str:
    """Semi-incompressible prose-like filler (hex words). A repeated-word
    filler compresses ~100x inside the blob's zlib, which lets a
    disabled-compaction mutant slip under byte-size assertions — proven by
    mutation M1 during task-23026."""
    words = []
    counter = 0
    while sum(len(w) + 1 for w in words) < chars:
        words.append(_hashlib.sha256(f"{seed}:{counter}".encode()).hexdigest()[:10])
        counter += 1
    return " ".join(words)[:chars]


def _history_rows(count: int) -> list[dict]:
    return [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"HISTORY-BODY-{i:03d} " + _incompressible_filler(f"r{i}", 180),
        }
        for i in range(count)
    ]


def _tool_loop_rows() -> list[dict]:
    """A payload whose first system row AND last user row both sit outside
    the final eight physical rows (a long assistant/tool loop) — the two
    retention rules the tail alone cannot satisfy."""
    rows = [
        {"role": "system", "content": "SYS-FRAMING " + _incompressible_filler("s", 80)}
    ]
    rows += _history_rows(10)
    rows.append(
        {
            "role": "user",
            "content": "LAST-USER-REQUEST " + _incompressible_filler("u", 80),
        }
    )
    for i in range(10):
        rows.append(
            {
                "role": "assistant",
                "content": f"TOOL-CALL-{i:02d} " + _incompressible_filler(f"a{i}", 80),
            }
        )
        rows.append(
            {
                "role": "tool",
                "content": f"TOOL-RESULT-{i:02d} "
                + _incompressible_filler(f"t{i}", 80),
            }
        )
    return rows


def test_safe_capture_retains_contract_set_and_inserts_one_marker():
    rows = _tool_loop_rows()
    request, omitted = build_request_capture(
        {**_kwargs(), "messages_payload": rows},
        capture_detail=CaptureDetail.SAFE,
    )

    payload = request["messages_payload"]
    marker = history_elision_marker(payload)
    assert marker is not None
    markers = [row for row in payload if history_elision_marker([row])]
    assert markers == [marker], "exactly one aggregate marker"

    # Retained set: first system row, last user row, final eight rows —
    # deduplicated, original order, values untouched.
    expected_retained = [rows[0], rows[11]] + rows[-_TAIL:]
    assert [
        row for row in payload if not history_elision_marker([row])
    ] == expected_retained
    # Marker sits at the position of the first omitted row (right after
    # the retained system row).
    assert history_elision_marker([payload[1]]) == marker

    # Marker contents: counts + retained positions only.
    total = len(rows)
    retained_positions = sorted({0, 11} | set(range(total - _TAIL, total)))
    assert marker["original_rows"] == total
    assert marker["omitted_rows"] == total - len(retained_positions)
    assert marker["retained_positions"] == retained_positions
    omitted_rows = [
        row for pos, row in enumerate(rows) if pos not in retained_positions
    ]
    for role in ("system", "user", "assistant", "tool"):
        assert marker["omitted_roles"][role] == sum(
            1 for row in omitted_rows if row.get("role") == role
        )

    # No omitted body survives, and the withholding is named through the
    # STABLE inventory path (never an ever-changing range string).
    rendered = json.dumps(payload)
    for row in omitted_rows:
        assert row["content"] not in rendered
    assert "messages_payload.history" in omitted


def test_safe_capture_keeps_first_system_and_last_user_outside_tail():
    rows = _tool_loop_rows()
    compacted, elided = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    assert elided == ("messages_payload.history",)
    kept = [row for row in compacted if not history_elision_marker([row])]
    assert kept[0] == rows[0] and kept[0]["role"] == "system"
    assert kept[1] == rows[11] and kept[1]["role"] == "user"
    assert kept[2:] == rows[-_TAIL:]


def test_full_capture_retains_the_whole_history_verbatim():
    """Full is the explicit, consent-gated, purgeable verbatim mode
    (ADR-092) — compaction must never touch it."""
    rows = _history_rows(_TAIL + 12)
    request, omitted = build_request_capture(
        {**_kwargs(), "messages_payload": rows},
        capture_detail=CaptureDetail.FULL,
    )
    assert request["messages_payload"] == rows
    assert not any("history" in entry for entry in omitted)


def test_fully_retained_safe_payload_is_untouched_with_no_marker():
    # 8 rows: the tail covers everything.
    rows = _history_rows(_TAIL)
    request, omitted = build_request_capture(
        {**_kwargs(), "messages_payload": rows},
        capture_detail=CaptureDetail.SAFE,
    )
    assert request["messages_payload"] == rows
    assert not any("history" in entry for entry in omitted)

    # 10 rows where the two head rows are the first system and last user:
    # union covers everything, so the list is unchanged and no marker is
    # added even though it is longer than the tail.
    rows = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "ask"},
    ] + [{"role": "assistant", "content": f"a{i}"} for i in range(_TAIL)]
    compacted, elided = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    assert compacted == rows
    assert elided == ()


def test_reprojection_of_a_compacted_payload_is_a_fixed_point():
    """The export path re-runs build_request_capture over a STORED request
    (project_exchange_export): the compacted list must map to itself —
    no nested markers, no changed counts, no new inventory entries."""
    rows = _tool_loop_rows()
    first, _ = build_request_capture(
        {"messages_payload": rows}, capture_detail=CaptureDetail.SAFE
    )
    second, omitted = build_request_capture(
        {"messages_payload": first["messages_payload"]},
        capture_detail=CaptureDetail.SAFE,
    )
    assert second["messages_payload"] == first["messages_payload"]
    assert not any("history" in entry for entry in omitted)


def test_input_marker_never_disables_compaction_of_surrounding_rows():
    stale_marker = {
        "kind": CAPTURE_HISTORY_ELISION_KIND,
        "version": CAPTURE_HISTORY_ELISION_VERSION,
        "original_rows": 40,
        "omitted_rows": 30,
        "omitted_roles": {role: 6 for role in CAPTURE_HISTORY_MARKER_ROLES},
        "retained_positions": [0, 1],
    }
    rows = [stale_marker] + _history_rows(_TAIL + 12)
    compacted, elided = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    assert elided == ("messages_payload.history",)
    markers = [row for row in compacted if history_elision_marker([row])]
    assert len(markers) == 1
    fresh = markers[0]
    # The fresh marker describes THIS pass over the real rows, not the
    # stale metadata.
    assert fresh["original_rows"] == _TAIL + 12
    assert fresh is not stale_marker and fresh != stale_marker
    kept = [row for row in compacted if not history_elision_marker([row])]
    assert kept[-_TAIL:] == rows[-_TAIL:]


def test_malformed_marker_lookalike_is_an_ordinary_row():
    lookalike = {
        "kind": CAPTURE_HISTORY_ELISION_KIND,
        "version": CAPTURE_HISTORY_ELISION_VERSION,
        "original_rows": 40,
        "omitted_rows": 30,
        "omitted_roles": {role: 6 for role in CAPTURE_HISTORY_MARKER_ROLES},
        "retained_positions": [0, 1],
        "sha256": "ab12cd34",  # extra key: NOT a valid marker
    }
    assert history_elision_marker([lookalike]) is None
    # In the omitted prefix it is compacted away like any ordinary row and
    # counts toward `other` (no `role` key).
    rows = [lookalike] + _history_rows(_TAIL + 12)
    compacted, _ = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    marker = history_elision_marker(compacted)
    assert marker is not None
    assert marker["omitted_roles"]["other"] >= 1
    assert "ab12cd34" not in json.dumps(compacted)
    # In the tail it is retained verbatim like any ordinary row.
    rows = _history_rows(_TAIL + 12) + [lookalike]
    compacted, _ = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    assert compacted[-1] == lookalike


def test_marker_shape_guard_a_digest_cannot_be_reintroduced_silently():
    """ADR-096 explicitly forbids the marker carrying content, snippets,
    per-row lengths, hashes/digests, IDs, or timestamps — a sha256 prefix
    would let anyone with DB access confirm guesses about omitted private
    text. This guard pins the marker's EXACT shape: any new key, any
    string value beyond the kind discriminator, or any non-count payload
    turns it red."""
    rows = _tool_loop_rows()
    compacted, _ = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    marker = history_elision_marker(compacted)
    assert marker is not None

    # Exact frozen key set — a silently added digest key fails here.
    assert set(marker.keys()) == CAPTURE_HISTORY_MARKER_KEYS

    # The ONLY string anywhere in the marker is the kind discriminator —
    # a digest/snippet VALUE fails here.
    def _strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, nested in value.items():
                yield key
                yield from _strings(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from _strings(nested)

    string_values = (
        set(_strings(marker))
        - CAPTURE_HISTORY_MARKER_KEYS
        - set(CAPTURE_HISTORY_MARKER_ROLES)
    )
    assert string_values == {CAPTURE_HISTORY_ELISION_KIND}
    # Counts are plain ints; positions are plain ints.
    assert type(marker["original_rows"]) is int
    assert type(marker["omitted_rows"]) is int
    assert all(type(v) is int for v in marker["omitted_roles"].values())
    assert all(type(v) is int for v in marker["retained_positions"])
    # And nothing digest-shaped appears anywhere in the compacted output.
    assert "sha256" not in json.dumps(compacted)


def test_unknown_roles_count_only_toward_other_and_never_reach_the_marker():
    rows = [
        {"role": None, "content": "NULL-ROLE-BODY"},
        {"role": 123, "content": "INT-ROLE-BODY"},
        {"role": "wizard", "content": "CUSTOM-ROLE-BODY"},
    ] + _history_rows(_TAIL + 4)
    compacted, _ = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    marker = history_elision_marker(compacted)
    assert marker is not None
    assert marker["omitted_roles"]["other"] == 3
    rendered = json.dumps(compacted)
    assert "wizard" not in rendered
    assert "NULL-ROLE-BODY" not in rendered


def test_non_mapping_rows_are_eligible_only_through_the_tail():
    rows = ["BARE-PREFIX-STRING"] + _history_rows(_TAIL + 4) + ["BARE-TAIL-STRING"]
    compacted, _ = compact_safe_history_rows(rows, CaptureDetail.SAFE)
    rendered = json.dumps(compacted)
    assert "BARE-PREFIX-STRING" not in rendered
    assert compacted[-1] == "BARE-TAIL-STRING"
    marker = history_elision_marker(compacted)
    assert marker is not None and marker["omitted_roles"]["other"] >= 1


def test_project_instruction_body_never_survives_safe_compaction():
    """C1 continuity: an instruction row in the omitted region disappears
    entirely (redaction metadata in omitted_keys remains); one in the tail
    keeps only its redaction marker, never the body."""
    rows = _history_rows(_TAIL + 3)
    rows[0] = _project_instruction_row()
    request, omitted = build_request_capture(
        {"messages_payload": rows}, capture_detail=CaptureDetail.SAFE
    )
    assert _PROJECT_INSTRUCTION_SENTINEL not in json.dumps(request)
    # The redaction step recorded its path before compaction; the design
    # keeps that metadata visible even though the row is now omitted.
    assert "messages_payload[0].content" in omitted
    assert "messages_payload.history" in omitted

    tail_instruction = _history_rows(_TAIL + 3)
    tail_instruction[-1] = _project_instruction_row()
    request, _ = build_request_capture(
        {"messages_payload": tail_instruction},
        capture_detail=CaptureDetail.SAFE,
    )
    assert _PROJECT_INSTRUCTION_SENTINEL not in json.dumps(request)
    assert "omitted by capture policy" in request["messages_payload"][-1]["content"]


def test_non_list_rows_pass_through_compaction():
    assert compact_safe_history_rows(None, CaptureDetail.SAFE) == (None, ())
    assert compact_safe_history_rows("x", CaptureDetail.SAFE) == ("x", ())


def test_per_turn_blob_size_growth_is_marker_sized_not_content_sized():
    """The blob for a deep-history turn grows only by the aggregate
    marker's retained-position list (O(1)), never by re-copied content:
    100 extra history rows of ~190 chars each (~19 KB of content) must add
    almost nothing to the stored blob."""

    def blob_for(row_count: int) -> int:
        request, omitted = build_request_capture(
            {**_kwargs(), "messages_payload": _history_rows(row_count)},
            capture_detail=CaptureDetail.SAFE,
        )
        cap = ExchangeCapture(
            run_tag="r1",
            seq=0,
            created_at="t",
            provider="p",
            model="m",
            endpoint=None,
            request=request,
            response={"content": "pong"},
            status="complete",
            usage_json=None,
            omitted_keys=omitted,
        )
        return len(capture_to_blob(cap))

    shallow = blob_for(20)
    deep = blob_for(120)
    per_row_cost = (deep - shallow) / 100
    assert per_row_cost < 5, (
        f"deep-history turns re-store content: {per_row_cost:.1f} bytes per "
        "omitted row (the aggregate marker should cost ~0)"
    )


def test_trim_safe_capture_blob_compacts_stored_safe_blobs():
    """The v52→v53 migration's pure helper: a pre-compaction stored Safe
    blob is compacted to exactly the shape build_request_capture now
    produces, with everything not deliberately compacted preserved
    value-identical."""
    rows = _history_rows(_TAIL + 12)
    # A pre-task-23026 capture: history verbatim (built at FULL to bypass
    # compaction, then stamped safe — the historical builder retained
    # ordinary rows verbatim under Safe).
    request, omitted = build_request_capture(
        {**_kwargs(), "messages_payload": rows},
        capture_detail=CaptureDetail.FULL,
    )
    legacy = ExchangeCapture(
        run_tag="r1",
        seq=3,
        created_at="t",
        provider="p",
        model="m",
        endpoint="https://example.test/v1",
        request=request,
        response={"content": "pong", "tool_calls": [], "synthetic_fallback": False},
        status="stopped",
        usage_json='{"input": 5}',
        omitted_keys=omitted,
        capture_detail=CaptureDetail.SAFE,
    )
    trimmed_blob = trim_safe_capture_blob(capture_to_blob(legacy))

    assert trimmed_blob is not None
    restored = capture_from_blob(trimmed_blob)
    payload = restored.request["messages_payload"]
    marker = history_elision_marker(payload)
    assert marker is not None
    assert marker["original_rows"] == _TAIL + 12
    kept = [row for row in payload if not history_elision_marker([row])]
    assert kept == rows[-_TAIL:]
    assert "HISTORY-BODY-000" not in json.dumps(payload)
    # Everything not deliberately compacted is value-identical.
    assert restored.response == legacy.response
    assert restored.status == legacy.status
    assert restored.usage_json == legacy.usage_json
    assert restored.run_tag == legacy.run_tag
    assert restored.seq == legacy.seq
    assert restored.provider == legacy.provider
    assert restored.endpoint == legacy.endpoint
    assert restored.capture_detail is CaptureDetail.SAFE
    untouched = {k: v for k, v in restored.request.items() if k != "messages_payload"}
    # JSON-normalized comparison: the blob round-trip renders tuples (e.g.
    # truncation_inventory) as lists; the VALUES must be identical.
    assert json.loads(json.dumps(untouched, default=str)) == json.loads(
        json.dumps(
            {k: v for k, v in legacy.request.items() if k != "messages_payload"},
            default=str,
        )
    )
    # The stable elision path is folded into the stored omission inventory.
    assert "messages_payload.history" in restored.omitted_keys
    assert set(legacy.omitted_keys).issubset(set(restored.omitted_keys))
    # Fixed point: compacting the compacted blob changes nothing.
    assert trim_safe_capture_blob(trimmed_blob) is None


def test_trim_safe_capture_blob_returns_none_when_nothing_to_compact():
    request, omitted = build_request_capture(
        _kwargs(), capture_detail=CaptureDetail.SAFE
    )
    cap = ExchangeCapture(
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request=request,
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=omitted,
    )
    assert trim_safe_capture_blob(capture_to_blob(cap)) is None


def test_trim_safe_capture_blob_never_touches_full_blobs():
    rows = _history_rows(_TAIL + 12)
    request, omitted = build_request_capture(
        {**_kwargs(), "messages_payload": rows},
        capture_detail=CaptureDetail.FULL,
    )
    cap = ExchangeCapture(
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request=request,
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=omitted,
        capture_detail=CaptureDetail.FULL,
    )
    assert trim_safe_capture_blob(capture_to_blob(cap)) is None


def test_trim_safe_capture_blob_compacts_llamacpp_wire_messages():
    wire_rows = _history_rows(_TAIL + 6)
    cap = ExchangeCapture(
        run_tag="r1",
        seq=0,
        created_at="t",
        provider="llama_cpp",
        model="m",
        endpoint=None,
        request={
            "model": "m",
            "wire_payload": {"model": "m", "messages": wire_rows, "stream": True},
            "truncation_inventory": (),
        },
        response={},
        status="complete",
        usage_json=None,
        omitted_keys=(),
    )
    trimmed_blob = trim_safe_capture_blob(capture_to_blob(cap))

    assert trimmed_blob is not None
    restored = capture_from_blob(trimmed_blob)
    wire = restored.request["wire_payload"]
    marker = history_elision_marker(wire["messages"])
    assert marker is not None
    kept = [row for row in wire["messages"] if not history_elision_marker([row])]
    assert kept == wire_rows[-_TAIL:]
    assert "HISTORY-BODY-000" not in json.dumps(wire)
    assert wire["stream"] is True
    assert "wire_payload.messages.history" in restored.omitted_keys


def test_trim_safe_capture_blob_raises_capture_unavailable_on_corrupt_blob():
    from tldw_chatbook.Chat.console_exchange_capture import CaptureUnavailableError

    with pytest.raises(CaptureUnavailableError):
        trim_safe_capture_blob(b"not-a-zlib-blob")
