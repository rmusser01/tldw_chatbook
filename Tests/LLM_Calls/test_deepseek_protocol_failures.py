"""A rejected HTTP-200 body must not masquerade as a retryable HTTP 502."""

import json
from unittest.mock import patch

import pytest

from Tests.LLM_Calls.test_groq_openrouter_migration_characterization import (
    _nonstreaming_ok_response,
    _sse_stream_response,
)
from tldw_chatbook.Agents.agent_models import AgentConfig, ToolResult
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.model_retry import is_transient_model_error
from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Provider_Contract import ChatProviderError

pytestmark = pytest.mark.bootstrap_profile


def malformed_response(streaming=False):
    body = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "code_review_submit",
                                "arguments": "PRIVATE-RESPONSE-CANARY",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    if not streaming:
        return _nonstreaming_ok_response(body)
    choice = body["choices"][0]
    message = choice.pop("message")
    message["tool_calls"][0]["index"] = 0
    choice["delta"] = message
    return _sse_stream_response(
        ("data: " + json.dumps(body) + "\n\ndata: [DONE]\n\n").encode()
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_malformed_success_retains_validation_reason_and_stops_without_retry(streaming):
    response = malformed_response(streaming)
    slept = []
    invoked = []

    def call_model(messages, _schemas):
        result = chat_api_call(
            "deepseek",
            messages_payload=messages,
            model="deepseek-flash",
            api_key="fixture-key",
            streaming=streaming,
        )
        return list(result) if streaming else result

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=invoked.append,
        spawn=lambda _task: ToolResult(ok=False),
        find_tools=lambda _query: [],
        load_schemas=lambda *_args: None,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        sleep=slept.append,
    )
    with (
        patch("requests.Session.post", return_value=response) as post,
        pytest.raises(ChatProviderError) as caught,
    ):
        run_agent_loop(
            AgentConfig(
                provider="deepseek", model="deepseek-flash", system_prompt="Review"
            ),
            [{"role": "user", "content": "Review"}],
            [],
            deps,
        )
    assert post.call_count == 1
    assert not slept and not invoked
    assert caught.value.status_code is None
    assert "tool arguments are malformed" in str(caught.value)
    assert "PRIVATE-RESPONSE-CANARY" not in str(caught.value)
    assert not is_transient_model_error(caught.value)
    response.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_console_preserves_terminal_validation_failure_and_safe_reason(streaming):
    resolution = ConsoleProviderResolution(
        provider="deepseek",
        execution_key="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        api_key="fixture-key",
        ready=True,
        streaming=streaming,
    )
    gateway = ConsoleProviderGateway(environ={}, chat_api_call_fn=chat_api_call)
    response = malformed_response(streaming)
    with (
        patch("requests.Session.post", return_value=response) as post,
        pytest.raises(ChatProviderError) as caught,
    ):
        async for _ in gateway.stream_chat(
            resolution, [{"role": "user", "content": "Review"}]
        ):
            pytest.fail("A malformed response cannot emit content or tools")
    assert post.call_count == 1
    assert caught.value.status_code is None
    assert not is_transient_model_error(caught.value)
    assert "tool arguments are malformed" in caught.value.console_copy
    assert "PRIVATE-RESPONSE-CANARY" not in caught.value.console_copy
    response.close.assert_called_once()


def test_actual_server_failure_remains_retryable():
    from tldw_chatbook.LLM_Calls import deepseek

    failure = ChatProviderError(provider="deepseek", status_code=502)
    with (
        patch.object(deepseek, "hosted_chat_request", side_effect=failure),
        pytest.raises(ChatProviderError) as caught,
    ):
        chat_api_call(
            "deepseek",
            messages_payload=[{"role": "user", "content": "Review"}],
            api_key="fixture-key",
            model="deepseek-flash",
            streaming=False,
        )
    assert caught.value is failure
    assert is_transient_model_error(caught.value)
