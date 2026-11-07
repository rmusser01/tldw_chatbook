"""DeepSeek's documented optional choice fields must survive strict parsing."""

import json
from unittest.mock import patch

import pytest

from Tests.LLM_Calls.test_groq_openrouter_migration_characterization import (
    _nonstreaming_ok_response,
    _sse_stream_response,
)
from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "logprobs",
    [
        None,
        {
            "content": [
                {
                    "token": "hi",
                    "logprob": -0.25,
                    "bytes": [104, 105],
                    "top_logprobs": [],
                }
            ]
        },
    ],
)
def test_deepseek_documented_choice_logprobs(streaming, logprobs):
    common = {
        "id": "test",
        "created": 1,
        "model": "deepseek-chat",
        "system_fingerprint": "test",
    }
    usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    choice = {"index": 0, "logprobs": logprobs, "finish_reason": "stop"}
    if streaming:
        events = [
            {
                **common,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        **choice,
                        "finish_reason": None,
                        "delta": {"role": "assistant", "content": "hi"},
                    }
                ],
            },
            {
                **common,
                "object": "chat.completion.chunk",
                "choices": [{**choice, "delta": {}}],
                "usage": usage,
            },
        ]
        body = (
            "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            + "data: [DONE]\n\n"
        )
        response = _sse_stream_response(body.encode())
    else:
        response = _nonstreaming_ok_response(
            {
                **common,
                "object": "chat.completion",
                "choices": [
                    {**choice, "message": {"role": "assistant", "content": "hi"}}
                ],
                "usage": usage,
            }
        )
    with patch("requests.Session.post", return_value=response):
        result = chat_api_call(
            "deepseek",
            messages_payload=[{"role": "user", "content": "hello"}],
            api_key="fixture-key",
            model="deepseek-chat",
            streaming=streaming,
        )
        if streaming:
            records = [
                json.loads(line[6:])
                for line in result
                if line.startswith("data: ") and "[DONE]" not in line
            ]
            assert (
                "".join(
                    choice.get("delta", {}).get("content", "")
                    for record in records
                    for choice in record.get("choices", [])
                )
                == "hi"
            )
            assert any(
                record.get("usage", {}).get("total_tokens") == 4 for record in records
            )
        else:
            assert result["choices"][0]["message"]["content"] == "hi"
            assert result["usage"]["total_tokens"] == 4


@pytest.mark.parametrize(
    "choice_change", [{"undocumented": "refused"}, {"index": "invalid"}]
)
def test_deepseek_choice_allowance_preserves_strict_shape(choice_change):
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "logprobs": None,
        "message": {"role": "assistant", "content": "hi"},
        **choice_change,
    }
    response = _nonstreaming_ok_response(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [choice],
        }
    )
    with patch("requests.Session.post", return_value=response):
        with pytest.raises(ChatProviderError):
            chat_api_call(
                "deepseek",
                messages_payload=[{"role": "user", "content": "hello"}],
                api_key="fixture-key",
                model="deepseek-chat",
                streaming=False,
            )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_interrupted_empty_answer_preserves_finish_and_usage(streaming, finish_reason):
    """Documented nullable content includes reasoning exhausting the output cap."""
    usage = {
        "prompt_tokens": 12,
        "completion_tokens": 4096,
        "total_tokens": 4108,
        "completion_tokens_details": {"reasoning_tokens": 4096},
    }
    message = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "PRIVATE-THINKING-CANARY",
    }
    choice = {"index": 0, "logprobs": None, "finish_reason": finish_reason}
    if streaming:
        events = [
            {"choices": [{**choice, "finish_reason": None, "delta": message}]},
            {"choices": [{**choice, "delta": {}}], "usage": usage},
        ]
        response = _sse_stream_response(
            (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            ).encode()
        )
    else:
        response = _nonstreaming_ok_response(
            {"choices": [{**choice, "message": message}], "usage": usage}
        )
    with patch("requests.Session.post", return_value=response):
        result = chat_api_call(
            "deepseek",
            messages_payload=[{"role": "user", "content": "Review"}],
            api_key="fixture-key",
            model="deepseek-flash",
            streaming=streaming,
            max_tokens=4096,
        )
        if streaming:
            list(result)
        else:
            assert result["usage"] == usage
        turn = result.terminal_turn
        assert turn.text == "" and not turn.tool_calls
        assert turn.finish_reason == finish_reason
        assert turn.usage == usage
    response.close.assert_called_once()


@pytest.mark.parametrize(
    "finish_reason,tool_calls",
    [
        ("stop", []),
        ("tool_calls", []),
        (
            "length",
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "code_review_submit", "arguments": '{"kind":'},
                }
            ],
        ),
        (
            "length",
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "code_review_submit", "arguments": "{}"},
                }
            ],
        ),
    ],
)
def test_empty_stop_and_interrupted_tool_calls_remain_refused(
    finish_reason, tool_calls
):
    response = _nonstreaming_ok_response(
        {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                }
            ]
        }
    )
    with (
        patch("requests.Session.post", return_value=response),
        pytest.raises(ChatProviderError),
    ):
        chat_api_call(
            "deepseek",
            messages_payload=[{"role": "user", "content": "Review"}],
            api_key="fixture-key",
            model="deepseek-flash",
            streaming=False,
        )
