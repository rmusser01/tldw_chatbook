"""Canonical adapter payload regressions; marker placement is not cache-hit evidence."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest

from Tests.Chat.test_anthropic_caching_degrade import _ok_response
from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.Chat.console_prepared_request import (
    build_console_request,
    compose_current_mixin_messages,
    prepare_provider_request,
    resolve_request_capacity,
)
from tldw_chatbook.Chat.console_prompt_mixins import (
    MixinAttemptSnapshot,
    MixinReplayRead,
    MixinSnapshot,
    MixinSourceRef,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    LiteralMixinDefinition,
)

pytestmark = pytest.mark.bootstrap_profile

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _snapshot(body):
    return (
        MixinAttemptSnapshot(
            (
                MixinSnapshot(
                    MixinSourceRef(
                        "profile-test", "local", "00000000-0000-4000-8000-000000000001"
                    ),
                    1,
                    "Style",
                    LiteralMixinDefinition(body, "before"),
                ),
            )
        )
        if body
        else MixinAttemptSnapshot()
    )


def _prepare(
    body="{style}",
    *,
    history=None,
    tail=(),
    ceiling=None,
    provider="anthropic",
    content="new",
    attempt_snapshot=None,
):
    rows = [{"role": "system", "content": "stable system"}]
    rows += deepcopy(
        history
        if history is not None
        else [
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "older answer"},
        ]
    )
    rows += [
        {"role": "user", "content": content, "_native_message_id": "active"},
        *tail,
    ]
    composed = compose_current_mixin_messages(
        rows,
        _snapshot(body) if attempt_snapshot is None else attempt_snapshot,
        user_message_id="active",
        attempt_id="attempt",
    )
    semantic = build_console_request(composed, tools=TOOLS)
    capacity = resolve_request_capacity(
        context_window_tokens=None if ceiling is None else ceiling + 522,
        requested_response_tokens=10,
    )
    return prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        provider=provider,
        model="claude-sonnet-4-6",
        capacity=capacity,
        count_fn=lambda messages, model: sum(
            len(str(m.get("content", "")).split()) for m in messages
        ),
    )


def _kwargs(request, *, enabled=True, provider="anthropic", base_url=""):
    return ConsoleProviderGateway._chat_api_kwargs_from_prepared(
        ConsoleProviderResolution(
            provider=provider,
            execution_key=provider,
            base_url=base_url,
            model="claude-sonnet-4-6" if provider == "anthropic" else "gpt-5.6",
            ready=True,
            api_key="test-key",
            streaming=False,
            prompt_caching=enabled if provider == "anthropic" else None,
        ),
        request,
    )


def _send(request, *, settings=None, **kwargs):
    def setting(section, key, default=None):
        return (settings or {}).get(key, default)

    with (
        patch("requests.Session.post", return_value=_ok_response()) as post,
        patch(
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting", side_effect=setting
        ),
    ):
        chat_api_call(**_kwargs(request, **kwargs))
    return post.call_args.kwargs["json"]


def test_two_mixin_turns_keep_clean_history_system_and_tools_stable():
    first, second = (_send(_prepare(body)) for body in ("{style}", "other style"))
    assert first["system"] == second["system"]
    assert first["tools"] == second["tools"]
    assert first["messages"][:-1] == second["messages"][:-1]
    assert first["messages"][-2]["content"][-1] == {
        "type": "text",
        "text": "older answer",
        "cache_control": {"type": "ephemeral"},
    }
    assert first["messages"][-1]["content"][0]["text"] == "{style}\n\nnew"
    assert second["messages"][-1]["content"][0]["text"] == "other style\n\nnew"


def _message_markers(payload):
    return [
        (i, j)
        for i, row in enumerate(payload["messages"])
        for j, block in enumerate(row["content"])
        if "cache_control" in block
    ]


def test_boundary_is_ephemeral_hidden_and_bound_to_exact_wire():
    request = _prepare()
    boundary = request.cache_boundary
    assert request.semantic.clean_history_message_index == 2
    assert boundary.clean_history_message_index == 1  # System moved to its own field.
    assert boundary.matches(_kwargs(request)["messages_payload"])
    assert "fingerprint" not in repr(boundary)
    assert "cache_boundary" not in repr(request)
    assert "clean_history_message_index" not in repr(request.semantic)
    body = _send(request)
    assert "fingerprint" not in str(body) and "_tldw" not in str(body)
    assert _message_markers(body) == [(1, 0), (2, 0)]
    assert len(_message_markers(body)) + 2 == 4  # System + tools + history + active.


def test_windowing_recomputes_boundary_after_removing_oldest_unit():
    history = [
        {"role": "user", "content": "discard " * 50},
        {"role": "assistant", "content": "discarded reply"},
        {"role": "user", "content": "kept"},
        {"role": "assistant", "content": "kept answer"},
    ]
    request = _prepare(history=history, ceiling=12)
    assert request.dropped_units == 1
    assert request.semantic.clean_history_message_index == 2
    payload = _send(request)
    assert payload["messages"][1]["content"][0]["text"] == "kept answer"
    assert _message_markers(payload) == [(1, 0), (2, 0)]
    assert "discard" not in str(payload)


@pytest.mark.parametrize("history,ceiling", [([], None), (None, 5)])
def test_no_remaining_history_emits_only_existing_active_marker(history, ceiling):
    request = _prepare(history=history, ceiling=ceiling)
    assert not request.semantic.compactable
    assert request.semantic.clean_history_message_index is None
    assert request.cache_boundary is None
    assert "console_cache_boundary" not in _kwargs(request)
    assert _message_markers(_send(request)) == [(0, 0)]


@pytest.mark.parametrize("recorded", [False, True])
def test_empty_or_all_off_attempt_does_not_change_original_payload(recorded):
    replay = MixinReplayRead(
        "recorded" if recorded else "legacy", MixinAttemptSnapshot()
    )
    request = _prepare(body="", attempt_snapshot=replay.snapshot)
    assert replay.status == ("recorded" if recorded else "legacy")
    assert request.semantic.clean_history_message_index is None
    assert request.cache_boundary is None
    kwargs = _kwargs(request)
    assert "console_cache_boundary" not in kwargs
    payload = _send(request)
    with (
        patch("requests.Session.post", return_value=_ok_response()) as post,
        patch(
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting",
            side_effect=lambda s, k, d=None: d,
        ),
    ):
        chat_api_call(**kwargs)
    assert payload == post.call_args.kwargs["json"]
    assert _message_markers(payload) == [(2, 0)]


def _tool_echo():
    return {
        "role": "assistant",
        "content": "working",
        "tool_calls": [
            {
                "id": "one",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            },
            {
                "id": "two",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            },
        ],
    }


def test_history_tool_results_coalesce_and_marker_stays_on_exact_last_result():
    request = _prepare(
        history=[
            {"role": "user", "content": "older"},
            _tool_echo(),
            {"role": "tool", "tool_call_id": "one", "content": "first result"},
            {"role": "tool", "tool_call_id": "two", "content": "last result"},
        ]
    )
    payload = _send(request)
    assert len(payload["messages"]) == 4
    assert payload["messages"][2]["content"][1]["content"] == "last result"
    assert _message_markers(payload) == [(2, 1), (3, 0)]


def test_active_tool_loop_reuses_history_boundary_without_marking_mixin_text():
    request = _prepare(
        tail=[
            _tool_echo(),
            {"role": "tool", "tool_call_id": "one", "content": "first result"},
            {"role": "tool", "tool_call_id": "two", "content": "last result"},
        ]
    )
    payload = _send(request)
    assert _message_markers(payload) == [(1, 0), (4, 1)]
    assert payload["messages"][2]["content"] == [
        {"type": "text", "text": "{style}\n\nnew"}
    ]


@pytest.mark.parametrize(
    "history_content,expected",
    [
        (
            [
                {"type": "text", "text": "caption"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
            [(0, 1), (1, 2)],
        ),
        (
            [
                {"type": "text", "text": "caption"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/image.png"},
                },
            ],
            [(1, 2)],
        ),
        ("", [(1, 2)]),
    ],
)
def test_attachments_map_to_true_history_end_or_omit_lossy_conversion(
    history_content, expected
):
    request = _prepare(
        history=[{"role": "user", "content": history_content}],
        content=[
            {"type": "text", "text": "new"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    )
    payload = _send(request)
    assert _message_markers(payload) == expected
    assert payload["messages"][-1]["content"][0]["text"] == "{style}"
    assert payload["messages"][-1]["content"][2]["type"] == "image"


@pytest.mark.parametrize(
    "enabled,settings,expected",
    [
        (False, {}, []),
        (True, {"anthropic_enabled": False}, []),
    ],
)
def test_history_marker_obeys_existing_opt_in_and_global_kill_switch(
    enabled, settings, expected
):
    payload = _send(_prepare(), enabled=enabled, settings=settings)
    assert _message_markers(payload) == expected
    if settings:
        from tldw_chatbook.LLM_Calls.LLM_API_Calls import _contains_cache_control

        assert not _contains_cache_control(payload)
    else:
        assert "cache_control" in payload["system"][0]
        assert "cache_control" in payload["tools"][-1]


@pytest.mark.parametrize("change", ["text", "index", "type"])
def test_stale_or_invalid_boundary_is_omitted_with_body_free_status(change):
    request = _prepare()
    kwargs = _kwargs(request)
    if change == "text":
        kwargs["messages_payload"][1]["content"] = "changed after preparation"
    elif change == "index":
        kwargs["console_cache_boundary"] = replace(
            request.cache_boundary, clean_history_message_index=2
        )
    else:
        kwargs["console_cache_boundary"] = {"clean_history_message_index": 1}
    with (
        patch("requests.Session.post", return_value=_ok_response()) as post,
        patch(
            "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting",
            side_effect=lambda s, k, d=None: d,
        ),
        patch("tldw_chatbook.LLM_Calls.LLM_API_Calls.log_counter") as counters,
    ):
        chat_api_call(**kwargs)
    assert _message_markers(post.call_args.kwargs["json"]) == [(2, 0)]
    assert any(
        c.args[0] == "anthropic_clean_history_cache_unavailable"
        and c.kwargs["labels"] == {"reason": "mapping_or_budget"}
        for c in counters.call_args_list
    )


def test_preserves_existing_breakpoint_budget_before_optional_history_marker():
    native_tools = tuple(
        {
            "name": f"t{i}",
            "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral"},
        }
        for i in range(2)
    )
    semantic = replace(_prepare().semantic, tools=native_tools)
    request = prepare_provider_request(
        semantic,
        wire_style="single_preamble",
        provider="anthropic",
        model="claude-sonnet-4-6",
        capacity=resolve_request_capacity(context_window_tokens=None),
    )
    payload = _send(request)
    assert _message_markers(payload) == [(2, 0)]  # 2 tools + system + active = 4.


@pytest.mark.parametrize("custom", [False, True])
def test_openai_uses_only_existing_automatic_caching_and_stable_key(custom):
    base_url = "https://strict-proxy.example/v1" if custom else ""
    first, second = (
        _send(
            _prepare(body, provider="openai"),
            provider="openai",
            base_url=base_url,
            settings={"openai_cache_key": True},
        )
        for body in ("one style", "two style")
    )
    assert "console_cache_boundary" not in _kwargs(
        _prepare(), provider="openai", base_url=base_url
    )
    assert "prompt_cache_options" not in first
    assert "prompt_cache_breakpoint" not in str(first)
    assert "cache_control" not in str(first)
    if custom:
        assert "prompt_cache_key" not in first
    else:
        assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["messages"][:-1] == second["messages"][:-1]


def test_unsupported_history_tail_is_not_guessed_from_earlier_user_text():
    request = _prepare(
        history=[
            {"role": "user", "content": "looks like {style}"},
            {"role": "developer", "content": "unsupported historical role"},
        ]
    )
    assert _message_markers(_send(request)) == [(1, 0)]


def test_project_guidance_coalescing_preserves_exact_history_block_end():
    from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

    request = _prepare(
        history=[
            {"role": "user", "content": "older"},
            _tool_echo(),
            {"role": "tool", "tool_call_id": "one", "content": "first result"},
            {"role": "tool", "tool_call_id": "two", "content": "last result"},
            {
                "role": "user",
                "content": "runtime guidance",
                EPHEMERAL_ORIGIN_KEY: "project_instructions",
            },
        ]
    )
    payload = _send(request)
    assert _message_markers(payload) == [(2, 2), (3, 0)]
    assert payload["messages"][2]["content"][2]["text"] == "runtime guidance"


@pytest.mark.parametrize(
    "mixed", [False, True], ids=["all-invalid", "mixed-valid-invalid"]
)
def test_discarded_historical_tool_call_tail_omits_history_marker(mixed):
    calls = [{"function": {"name": ""}}]
    if mixed:
        calls.insert(0, _tool_echo()["tool_calls"][0])
    request = _prepare(
        history=[
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "surviving text", "tool_calls": calls},
        ]
    )
    with patch("tldw_chatbook.LLM_Calls.LLM_API_Calls.log_counter") as counters:
        payload = _send(request)
    assert _message_markers(payload) == [(2, 0)]
    assert payload["messages"][1]["content"][0] == {
        "type": "text",
        "text": "surviving text",
    }
    assert payload["messages"][1]["content"][1:] == (
        [
            {"type": "tool_use", "id": "one", "name": "lookup", "input": {}},
        ]
        if mixed
        else []
    )
    assert payload["messages"][-1]["content"] == [
        {
            "type": "text",
            "text": "{style}\n\nnew",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert any(
        c.args[0] == "anthropic_clean_history_cache_unavailable"
        and c.kwargs["labels"] == {"reason": "mapping_or_budget"}
        for c in counters.call_args_list
    )
