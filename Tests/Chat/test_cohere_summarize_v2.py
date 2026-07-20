"""task-297: ``summarize_with_cohere`` on the Cohere v2 /chat API.

``chat_with_cohere`` migrated to v2 in task-267 (PR #690, live-gated);
this helper was the repo's last v1 ``/chat`` caller. These tests pin the
migrated behavior contract at the ``requests`` boundary: request shape
(v2 endpoint, messages array, stream flag), non-streaming parts-array
parsing, streaming content-delta extraction, and the unchanged error
string formats callers already rely on.
"""

import json
from unittest.mock import Mock, patch

from tldw_chatbook.LLM_Calls.Summarization_General_Lib import summarize_with_cohere

V2_URL = "https://api.cohere.com/v2/chat"


def _text_response(text="A concise summary."):
    return {
        "id": "resp_1",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "finish_reason": "COMPLETE",
    }


def _mock_post(mock_post, response_json, status_code=200):
    mock_response = Mock()
    mock_response.json.return_value = response_json
    mock_response.status_code = status_code
    mock_response.text = json.dumps(response_json)
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    return mock_response


@patch("requests.Session.post")
def test_non_streaming_hits_v2_with_messages_array(mock_post):
    _mock_post(mock_post, _text_response())

    summary = summarize_with_cohere(
        "test-key",
        "Some article text.",
        "Summarize this.",
        temp=0.2,
        system_message="Be terse.",
        streaming=False,
    )

    assert summary == "A concise summary."
    assert mock_post.call_args[0][0] == V2_URL
    payload = mock_post.call_args[1]["json"]
    assert payload["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Some article text. \n\n\n\nSummarize this."},
    ]
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2
    for v1_only_key in ("preamble", "message", "streaming"):
        assert v1_only_key not in payload


@patch("requests.Session.post")
def test_blank_system_message_is_omitted(mock_post):
    _mock_post(mock_post, _text_response())

    summarize_with_cohere("test-key", "Text.", "Prompt.", system_message="")

    payload = mock_post.call_args[1]["json"]
    # A blank system message must not become an empty system turn; the
    # helper's own default fills None, so only explicit blanks are omitted.
    assert all(
        m["role"] != "system" or m["content"].strip() for m in payload["messages"]
    )


@patch("requests.Session.post")
def test_non_streaming_concatenates_multiple_text_parts(mock_post):
    _mock_post(
        mock_post,
        {
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Part one. "},
                    {"type": "text", "text": "Part two."},
                ],
            },
        },
    )

    summary = summarize_with_cohere("test-key", "Text.", "Prompt.")

    assert summary == "Part one. Part two."


@patch("requests.Session.post")
def test_non_streaming_missing_content_reports_expected_data_error(mock_post):
    _mock_post(mock_post, {"message": {"role": "assistant", "content": []}})

    result = summarize_with_cohere("test-key", "Text.", "Prompt.")

    assert result == "Cohere: Expected data not found in API response."


@patch("requests.Session.post")
def test_non_200_reports_api_request_failed(mock_post):
    _mock_post(mock_post, {"message": "bad request"}, status_code=400)

    result = summarize_with_cohere("test-key", "Text.", "Prompt.")

    assert result.startswith("Cohere: API request failed:")


@patch("requests.Session.post")
def test_streaming_yields_content_delta_text(mock_post):
    events = [
        {"type": "message-start"},
        {
            "type": "content-delta",
            "delta": {"message": {"content": {"text": "Summary "}}},
        },
        {
            "type": "content-delta",
            "delta": {"message": {"content": {"text": "chunks."}}},
        },
        {"type": "message-end", "delta": {"finish_reason": "COMPLETE"}},
    ]
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = [
        f"data: {json.dumps(e)}".encode() for e in events
    ]
    mock_post.return_value = mock_response

    chunks = list(
        summarize_with_cohere(
            "test-key",
            "Text.",
            "Prompt.",
            streaming=True,
        )
    )

    assert chunks == ["Summary ", "chunks."]
    assert mock_post.call_args[0][0] == V2_URL
    assert mock_post.call_args[1]["json"]["stream"] is True


@patch("tldw_chatbook.LLM_Calls.Summarization_General_Lib.get_cli_setting")
@patch("requests.Session.post")
def test_missing_api_key_short_circuits_without_network(mock_post, mock_setting):
    # The config fallback must also come up empty regardless of whether
    # THIS machine's real config carries a cohere key.
    mock_setting.return_value = None
    result = summarize_with_cohere(None, "Text.", "Prompt.")

    assert result == "Cohere: API Key Not Provided/Found in Config file or is empty"
    mock_post.assert_not_called()


@patch("requests.Session.post")
def test_streaming_skips_interleaved_event_lines_without_log_noise(mock_post):
    """task-297 review: Cohere v2 SSE interleaves standalone 'event: <type>'
    lines; they must be skipped before json.loads, not logged as decode
    errors."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = [
        b"event: message-start",
        b'data: {"type": "message-start"}',
        b"event: content-delta",
        b'data: {"type": "content-delta", "delta": {"message": {"content": {"text": "Hi."}}}}',
        b"event: message-end",
        b'data: {"type": "message-end", "delta": {"finish_reason": "COMPLETE"}}',
    ]
    mock_post.return_value = mock_response

    chunks = list(
        summarize_with_cohere(
            "test-key",
            "Text.",
            "Prompt.",
            streaming=True,
        )
    )

    assert chunks == ["Hi."]


@patch("requests.Session.post")
def test_streaming_non_200_reports_the_same_pinned_error_format(mock_post):
    """task-297 review: a streaming-path 4xx must report the same
    'Cohere: API request failed: ...' format as the non-streaming path
    (raise_for_status used to surface a bodyless HTTPError instead)."""
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.text = '{"message": "bad request"}'
    mock_post.return_value = mock_response

    result = summarize_with_cohere(
        "test-key",
        "Text.",
        "Prompt.",
        streaming=True,
    )

    assert result == 'Cohere: API request failed: {"message": "bad request"}'


@patch("requests.Session.post")
def test_streaming_parses_raw_json_lines_without_sse_framing(mock_post):
    """task-297 live-smoke regression: with 'accept: application/json' the
    REAL v2 stream is raw JSON lines (no 'data:' prefix) — a strict
    SSE-only filter dropped every line and yielded zero chunks live."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = [
        b'{"type": "message-start"}',
        b'{"type": "content-delta", "delta": {"message": {"content": {"text": "Raw "}}}}',
        b'{"type": "content-delta", "delta": {"message": {"content": {"text": "lines."}}}}',
        b'{"type": "message-end", "delta": {"finish_reason": "COMPLETE"}}',
    ]
    mock_post.return_value = mock_response

    chunks = list(
        summarize_with_cohere(
            "test-key",
            "Text.",
            "Prompt.",
            streaming=True,
        )
    )

    assert chunks == ["Raw ", "lines."]


@patch("requests.Session.post")
def test_non_200_with_non_json_body_keeps_pinned_error_format(mock_post):
    """Gemini/Qodo #698-2: a gateway/CDN HTML error body must not break the
    pinned failure format via a JSONDecodeError into the outer except."""
    mock_response = Mock()
    mock_response.status_code = 502
    mock_response.text = "<html>Bad gateway</html>"
    mock_response.json.side_effect = json.JSONDecodeError("x", "<html>", 0)
    mock_post.return_value = mock_response

    result = summarize_with_cohere("test-key", "Text.", "Prompt.")

    assert result == "Cohere: API request failed: <html>Bad gateway</html>"


@patch("requests.Session.post")
def test_streaming_skips_non_object_json_and_closes_response(mock_post):
    """Gemini #698-1 + Qodo #698-1: a JSON list/primitive line must not crash
    the generator, and the response is closed when the stream ends."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = [
        b"[1, 2, 3]",
        b'"just a string"',
        b'{"type": "content-delta", "delta": {"message": {"content": {"text": "Still works."}}}}',
    ]
    mock_post.return_value = mock_response

    chunks = list(
        summarize_with_cohere(
            "test-key",
            "Text.",
            "Prompt.",
            streaming=True,
        )
    )

    assert chunks == ["Still works."]
    mock_response.close.assert_called_once()
