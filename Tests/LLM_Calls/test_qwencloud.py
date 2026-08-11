"""Pure request-translation contracts for the QwenCloud adapter."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError, ChatConfigurationError
from tldw_chatbook.LLM_Calls.qwencloud import (
    build_qwencloud_payload,
    normalize_qwencloud_api_mode,
    normalize_qwencloud_base_url,
    resolve_qwencloud_api_key,
)


def test_api_mode_config_then_default_and_exact_values() -> None:
    assert normalize_qwencloud_api_mode(None) == "responses"
    assert (
        normalize_qwencloud_api_mode(
            None, provider_settings={"api_mode": " CHAT_COMPLETIONS "}
        )
        == "chat_completions"
    )
    assert (
        normalize_qwencloud_api_mode(
            " Responses ", provider_settings={"api_mode": "chat_completions"}
        )
        == "responses"
    )
    assert (
        normalize_qwencloud_api_mode(
            "responses",
            provider_settings=7,  # type: ignore[arg-type]
        )
        == "responses"
    )

    for rejected in ("response", "chat", "chat-completions", "unknown", ""):
        with pytest.raises(ChatConfigurationError) as exc_info:
            normalize_qwencloud_api_mode(rejected)
        assert exc_info.value.provider == "qwencloud"


def test_base_url_normalizes_base_and_pasted_endpoints() -> None:
    expected = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert normalize_qwencloud_base_url(None) == expected
    assert normalize_qwencloud_base_url(f"  {expected}///  ") == expected
    assert normalize_qwencloud_base_url(f"{expected}/responses") == expected
    assert normalize_qwencloud_base_url(f"{expected}/chat/completions/") == expected
    assert (
        normalize_qwencloud_base_url("http://gateway.internal:8080/team/qwen/v1/")
        == "http://gateway.internal:8080/team/qwen/v1"
    )


def test_base_url_rejects_unsafe_or_malformed_values() -> None:
    rejected = (
        "dashscope.example/v1",
        "ftp://dashscope.example/v1",
        "https:///v1",
        "https://user:secret@dashscope.example/v1",
        "https://dashscope.example/v1?tenant=a",
        "https://dashscope.example/v1#fragment",
        "https://dashscope.example/v1?",
        "https://dashscope.example/v1#",
        "https://dashscope.example/v1/models",
        "https://dashscope.example/v1/responses/responses",
        "https://dashscope.example/v1/chat/completions/chat/completions",
        "https://dashscope.example/v1/responses/extra",
        "https://dashscope.example/v1/chat/completions/extra",
        "https://dashscope.example//compatible-mode/v1",
        "https://bad host.example/v1",
        "https://dashscope.example:/v1",
        "https://dashscope.example\n.evil/v1",
        "https://dashscope.example/%zz",
        "   ",
    )
    for value in rejected:
        with pytest.raises(ChatConfigurationError) as exc_info:
            normalize_qwencloud_base_url(value)
        assert exc_info.value.provider == "qwencloud"
        assert "secret" not in str(exc_info.value)


def test_base_url_rejects_malformed_authorities() -> None:
    malformed_authorities = (
        "https://good.example\\evil/v1",
        "https://%zz/v1",
        "https://good.example|evil/v1",
        "https://good.example^evil/v1",
        "https://good.example\x00evil/v1",
    )
    for value in malformed_authorities:
        with pytest.raises(ChatConfigurationError) as exc_info:
            normalize_qwencloud_base_url(value)
        assert exc_info.value.provider == "qwencloud"


def test_api_key_precedence_is_provider_isolated() -> None:
    environ = {
        "DASHSCOPE_API_KEY": "default-env-key",
        "QWEN_KEY": "selected-env-key",
        "OPENAI_API_KEY": "other-provider-key",
    }
    settings = {
        "api_key": "modern-key",
        "api_key_env_var": "QWEN_KEY",
        "openai_api_key": "other-provider-setting",
    }

    assert (
        resolve_qwencloud_api_key(
            "trusted-key", provider_settings=settings, environ=environ
        )
        == "trusted-key"
    )
    assert (
        resolve_qwencloud_api_key(
            "trusted-key",
            provider_settings=7,  # type: ignore[arg-type]
            environ=7,  # type: ignore[arg-type]
        )
        == "trusted-key"
    )
    assert (
        resolve_qwencloud_api_key(None, provider_settings=settings, environ=environ)
        == "modern-key"
    )
    assert (
        resolve_qwencloud_api_key(
            None,
            provider_settings={"api_key": "modern-key"},
            environ=7,  # type: ignore[arg-type]
        )
        == "modern-key"
    )
    assert (
        resolve_qwencloud_api_key(
            None,
            provider_settings={"api_key_env_var": "QWEN_KEY"},
            environ=environ,
        )
        == "selected-env-key"
    )
    assert resolve_qwencloud_api_key(None, environ=environ) == "default-env-key"

    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_qwencloud_api_key(
            None,
            provider_settings={"openai_api_key": "do-not-use"},
            environ={"OPENAI_API_KEY": "do-not-use"},
        )
    assert exc_info.value.provider == "qwencloud"
    assert "do-not-use" not in str(exc_info.value)


def test_resolution_helpers_reject_invalid_mapping_shapes() -> None:
    with pytest.raises(ChatConfigurationError) as exc_info:
        normalize_qwencloud_api_mode(
            None,
            provider_settings=7,  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"

    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_qwencloud_api_key(
            None,
            provider_settings=7,
            environ={},  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"

    with pytest.raises(ChatConfigurationError) as exc_info:
        resolve_qwencloud_api_key(
            None,
            provider_settings={},
            environ=7,  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"


def test_responses_payload_has_exact_allowlist_and_stateless_invariants() -> None:
    payload = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message="Be concise.",
        messages_payload=[{"role": "user", "content": "Hello"}],
        streaming=True,
        temp=0.2,
        topp=0.8,
        topk=20,
        max_tokens=128,
        seed=7,
        presence_penalty=0.3,
        stop=["END"],
        response_format={"type": "json_object"},
        n=2,
        logprobs=True,
        top_logprobs=3,
    )

    assert payload == {
        "model": "qwen3.8-max",
        "input": [{"role": "user", "content": "Hello"}],
        "instructions": "Be concise.",
        "stream": True,
        "store": False,
        "temperature": 0.2,
        "top_p": 0.8,
        "max_output_tokens": 128,
    }
    assert "previous_response_id" not in payload
    assert "conversation" not in payload

    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(
            api_mode="responses",
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            max_tokens=15,
        )
    assert exc_info.value.provider == "qwencloud"

    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(
            api_mode="responses",
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            max_tokens="128",  # type: ignore[arg-type]
        )
    assert exc_info.value.provider == "qwencloud"


def test_responses_system_message_maps_to_instructions() -> None:
    kwargs = {
        "api_mode": "responses",
        "model": "qwen3.8-max",
        "streaming": False,
    }

    from_leading_row = build_qwencloud_payload(
        **kwargs,
        system_message=None,
        messages_payload=[
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert from_leading_row["instructions"] == "Be precise."
    assert from_leading_row["input"] == [{"role": "user", "content": "Hello"}]

    duplicate = build_qwencloud_payload(
        **kwargs,
        system_message="Be precise.",
        messages_payload=[
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert duplicate["instructions"] == "Be precise."
    assert duplicate["input"] == [{"role": "user", "content": "Hello"}]

    with pytest.raises(ChatBadRequestError):
        build_qwencloud_payload(
            **kwargs,
            system_message="Be concise.",
            messages_payload=[
                {"role": "system", "content": "Be expansive."},
                {"role": "user", "content": "Hello"},
            ],
        )
    with pytest.raises(ChatBadRequestError):
        build_qwencloud_payload(
            **kwargs,
            system_message=None,
            messages_payload=[
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Too late."},
            ],
        )


def test_leading_system_row_with_tool_calls_is_rejected() -> None:
    messages = [
        {
            "role": "system",
            "content": "Never execute tools.",
            "tool_calls": [
                {
                    "id": "call_system",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "Hello"},
    ]
    for mode in ("responses", "chat_completions"):
        with pytest.raises(ChatBadRequestError) as exc_info:
            build_qwencloud_payload(
                api_mode=mode,
                model="qwen3.8-max",
                system_message=None,
                messages_payload=messages,
                streaming=False,
            )
        assert exc_info.value.provider == "qwencloud"


def test_responses_reasoning_effort_enum_is_exact() -> None:
    base = {
        "api_mode": "responses",
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": [{"role": "user", "content": "Hello"}],
        "streaming": False,
    }
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        payload = build_qwencloud_payload(**base, reasoning_effort=effort)
        assert payload["reasoning"] == {"effort": effort}

    for rejected in ("", "LOW", "ultra", "maximum"):
        with pytest.raises(ChatBadRequestError) as exc_info:
            build_qwencloud_payload(**base, reasoning_effort=rejected)
        assert exc_info.value.provider == "qwencloud"
    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(**base, reasoning_effort=[])  # type: ignore[arg-type]
    assert exc_info.value.provider == "qwencloud"


def test_chat_payload_has_exact_allowlist_and_thinking_invariant() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    payload = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message="Be precise.",
        messages_payload=[{"role": "user", "content": "Hello"}],
        streaming=True,
        tools=tools,
        tool_choice="auto",
        temp=0.2,
        topp=0.8,
        topk=20,
        max_tokens=128,
        seed=7,
        presence_penalty=0.3,
        stop=["END"],
        response_format={"type": "json_object"},
        n=1,
        logprobs=True,
        top_logprobs=3,
        reasoning_effort="high",
    )
    assert payload == {
        "model": "qwen3.8-max",
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": True,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 20,
        "max_completion_tokens": 128,
        "seed": 7,
        "presence_penalty": 0.3,
        "stop": ["END"],
        "response_format": {"type": "json_object"},
        "n": 1,
        "logprobs": True,
        "top_logprobs": 3,
        "tools": tools,
        "tool_choice": "auto",
        "reasoning_effort": "high",
        "preserve_thinking": False,
        "stream_options": {"include_usage": True},
    }

    nonstream = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=[{"role": "user", "content": "Hello"}],
        streaming=False,
        response_format={"type": "text"},
        n=2,
    )
    assert nonstream["n"] == 2
    assert nonstream["preserve_thinking"] is False
    assert "stream_options" not in nonstream

    with pytest.raises(ChatBadRequestError):
        build_qwencloud_payload(
            api_mode="chat_completions",
            model="qwen3.8-max",
            system_message=None,
            messages_payload=[{"role": "user", "content": "Hello"}],
            streaming=False,
            tools=tools,
            n=2,
        )
    for rejected_format in (
        {"type": "json_schema"},
        {"type": "text", "extra": "not-allowed"},
    ):
        with pytest.raises(ChatBadRequestError):
            build_qwencloud_payload(
                api_mode="chat_completions",
                model="qwen3.8-max",
                system_message=None,
                messages_payload=[{"role": "user", "content": "Hello"}],
                streaming=False,
                response_format=rejected_format,
            )


def test_chat_stop_sequence_is_deep_copied() -> None:
    stop = ["END", "DONE"]
    payload = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=[{"role": "user", "content": "Hello"}],
        streaming=False,
        stop=stop,
    )

    assert payload["stop"] == ["END", "DONE"]
    assert payload["stop"] is not stop
    stop.append("MUTATED")
    assert payload["stop"] == ["END", "DONE"]


def test_function_tools_translate_by_mode() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    original = deepcopy(tools)
    common = {
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": [{"role": "user", "content": "Hello"}],
        "streaming": False,
        "tools": tools,
        "tool_choice": "auto",
    }

    chat_payload = build_qwencloud_payload(api_mode="chat_completions", **common)
    assert chat_payload["tools"] == tools
    assert chat_payload["tool_choice"] == "auto"
    assert chat_payload["n"] == 1

    responses_payload = build_qwencloud_payload(api_mode="responses", **common)
    assert responses_payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    assert responses_payload["tool_choice"] == "auto"
    assert tools == original
    assert chat_payload["tools"] is not tools
    assert (
        responses_payload["tools"][0]["parameters"]
        is not tools[0]["function"]["parameters"]
    )

    for mode in ("responses", "chat_completions"):
        for accepted_choice in (None, "auto", "none"):
            payload = build_qwencloud_payload(
                api_mode=mode, **{**common, "tool_choice": accepted_choice}
            )
            if accepted_choice is None:
                assert "tool_choice" not in payload
            else:
                assert payload["tool_choice"] == accepted_choice


def test_invalid_or_builtin_tools_fail_before_network() -> None:
    base = {
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": [{"role": "user", "content": "Hello"}],
        "streaming": False,
    }
    valid_function = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    rejected_tool_sets = (
        [{"type": "web_search"}],
        [{"type": "function", "function": {"name": "", "parameters": {}}}],
        [{"type": "function", "function": {"name": "   ", "parameters": {}}}],
        [
            valid_function,
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            },
        ],
        [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": []},
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "array"},
                },
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "type": "web_search",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    for mode in ("responses", "chat_completions"):
        for rejected_tools in rejected_tool_sets:
            with pytest.raises(ChatBadRequestError) as exc_info:
                build_qwencloud_payload(
                    api_mode=mode,
                    **base,
                    tools=rejected_tools,  # type: ignore[arg-type]
                )
            assert exc_info.value.provider == "qwencloud"

        for rejected_choice in ("required", "lookup", {"type": "function"}):
            with pytest.raises(ChatBadRequestError):
                build_qwencloud_payload(
                    api_mode=mode,
                    **base,
                    tools=[valid_function],
                    tool_choice=rejected_choice,
                )


@pytest.mark.parametrize(
    "invalid_override",
    (
        {"messages_payload": None},
        {"tools": 7},
        {"response_format": 7},
    ),
    ids=("messages-none", "tools-int", "response-format-int"),
)
def test_invalid_public_build_shapes_raise_typed_error(
    invalid_override: dict[str, object],
) -> None:
    kwargs: dict[str, object] = {
        "api_mode": "chat_completions",
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": [{"role": "user", "content": "Hello"}],
        "streaming": False,
    }
    kwargs.update(invalid_override)

    with pytest.raises(ChatBadRequestError) as exc_info:
        build_qwencloud_payload(**kwargs)  # type: ignore[arg-type]
    assert exc_info.value.provider == "qwencloud"


def test_message_content_translation_is_role_safe_and_immutable() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,AAAA",
                        "detail": "auto",
                    },
                },
            ],
        },
    ]
    original = deepcopy(messages)
    common = {
        "model": "qwen3.8-max",
        "system_message": None,
        "messages_payload": messages,
        "streaming": False,
    }

    chat = build_qwencloud_payload(api_mode="chat_completions", **common)
    assert chat["messages"] == [
        {"role": "user", "content": "Hello world"},
        original[1],
    ]
    responses = build_qwencloud_payload(api_mode="responses", **common)
    assert responses["input"] == [
        {"role": "user", "content": "Hello world"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is shown?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                },
            ],
        },
    ]
    assert messages == original
    assert chat["messages"][1]["content"] is not messages[1]["content"]

    empty_assistant_batch = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_empty",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_empty", "content": "ok"},
    ]
    empty_chat = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=empty_assistant_batch,
        streaming=False,
    )
    assert empty_chat["messages"][0]["content"] == ""
    empty_responses = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=empty_assistant_batch,
        streaming=False,
    )
    assert empty_responses["input"] == [
        {
            "type": "function_call",
            "call_id": "call_empty",
            "name": "lookup",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call_empty",
            "output": "ok",
        },
    ]

    rejected_messages = (
        [{"role": "assistant", "content": [original[1]["content"][1]]}],
        [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}],
        [{"role": "user", "content": [{"type": "video", "video": "x"}]}],
        [{"role": "user", "content": [{"type": "file", "file": "x"}]}],
        [{"role": "user", "content": [{"type": "unknown", "value": "x"}]}],
        [{"role": "critic", "content": "No"}],
        [{"role": 42, "content": "No"}],
        [{"role": "user", "content": 42}],
        [{"role": "user", "content": [{"type": "text", "text": 42}]}],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Too late"},
        ],
    )
    for mode in ("responses", "chat_completions"):
        for rejected in rejected_messages:
            with pytest.raises(ChatBadRequestError) as exc_info:
                build_qwencloud_payload(
                    api_mode=mode,
                    model="qwen3.8-max",
                    system_message=None,
                    messages_payload=rejected,  # type: ignore[arg-type]
                    streaming=False,
                )
            assert exc_info.value.provider == "qwencloud"


def test_responses_assistant_text_is_id_free_easy_input_message() -> None:
    payload = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=[
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Prior answer"}],
            },
        ],
        streaming=False,
    )
    assistant_item = payload["input"][1]
    assert assistant_item == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Prior answer"}],
    }
    assert set(assistant_item) == {"role", "content"}
    assert "id" not in assistant_item
    assert "status" not in assistant_item
    assert "type" not in assistant_item


def test_responses_pairs_out_of_order_results_by_call_id() -> None:
    messages = [
        {"role": "user", "content": "Compare both."},
        {
            "role": "assistant",
            "content": "I'll check.",
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {
                        "name": "first_tool",
                        "arguments": '{"value": 1}',
                    },
                },
                {
                    "id": "call_B",
                    "type": "function",
                    "function": {
                        "name": "second_tool",
                        "arguments": '{"value": 2}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_B", "content": "result B"},
        {"role": "tool", "tool_call_id": "call_A", "content": "result A"},
    ]
    original = deepcopy(messages)

    payload = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=messages,
        streaming=False,
    )
    assert payload["input"] == [
        {"role": "user", "content": "Compare both."},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "I'll check."}],
        },
        {
            "type": "function_call",
            "call_id": "call_A",
            "name": "first_tool",
            "arguments": '{"value": 1}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_A",
            "output": "result A",
        },
        {
            "type": "function_call",
            "call_id": "call_B",
            "name": "second_tool",
            "arguments": '{"value": 2}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_B",
            "output": "result B",
        },
    ]
    assert messages == original

    chat = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=messages,
        streaming=False,
    )
    assert chat["messages"] == original


def test_tool_call_arguments_reject_non_finite_json_constants() -> None:
    for arguments in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
        history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_strict_json",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": arguments},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_strict_json",
                "content": "ok",
            },
        ]
        with pytest.raises(ChatBadRequestError) as exc_info:
            build_qwencloud_payload(
                api_mode="responses",
                model="qwen3.8-max",
                system_message=None,
                messages_payload=history,
                streaming=False,
            )
        assert exc_info.value.provider == "qwencloud"


def test_responses_rejects_unpairable_tool_batches_before_network() -> None:
    def call(
        call_id: object = "call_A", name: object = "lookup", arguments: object = "{}"
    ) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    def assistant(*calls: dict, content: object = "") -> dict:
        return {"role": "assistant", "content": content, "tool_calls": list(calls)}

    def result(call_id: object = "call_A", content: object = "ok") -> dict:
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    rejected_histories = (
        [assistant(call())],
        [assistant(call()), result(), result()],
        [result()],
        [assistant(call()), result("call_extra")],
        [assistant(call(), call()), result()],
        [assistant(call("")), result("")],
        [assistant(call(name="")), result()],
        [assistant(call(arguments="{")), result()],
        [assistant(call(arguments=42)), result()],
        [assistant(call(arguments="[]")), result()],
        [assistant(call()), result(content={"not": "a string"})],
        [assistant(call()), {"role": "tool", "content": "missing id"}],
        [
            assistant(call("call_A")),
            result("call_A"),
            assistant(call("call_A")),
            result("call_A"),
        ],
        [
            assistant(call("call_A")),
            assistant(call("call_B")),
            result("call_A"),
            result("call_B"),
        ],
    )
    for history in rejected_histories:
        with pytest.raises(ChatBadRequestError) as exc_info:
            build_qwencloud_payload(
                api_mode="responses",
                model="qwen3.8-max",
                system_message=None,
                messages_payload=history,
                streaming=False,
            )
        assert exc_info.value.provider == "qwencloud"
