"""Reasoning and tool request contracts of existing engine presets.

- TASK-33502: NVIDIA NIM's Qwen3.5 thinks by default and takes no
  ``reasoning_effort``; its only switch is
  ``chat_template_kwargs.enable_thinking``. Effort "none" turns thinking off,
  any other level leaves it on, and every other NVIDIA model keeps the local
  refusal.
- TASK-33503: Fireworks rejects a request carrying both ``thinking`` and
  ``reasoning_effort``; no engine preset can build one.
- TASK-33500: Cerebras documents ``strict`` as optional (default false), so
  its function tools go out without it, exactly as callers pass them.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    build_hosted_chat_payload,
)
from tldw_chatbook.provider_registry import (
    ALL_RECORDS,
    CEREBRAS,
    FIREWORKS,
    NVIDIA,
    ProviderRecord,
    thinking_toggle_key,
)

_QWEN = "qwen/qwen3.5-397b-a17b"
_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers.",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}}},
    },
}


def _payload(record: ProviderRecord, model: str = "doc-model", **kwargs: Any) -> dict[str, Any]:
    resolution = HostedProviderResolution(
        provider=record.key,
        model=model,
        api_key="secret",
        base_url=record.default_base_url or "https://engine.invalid/v1",
        timeout=10.0,
        retries=0,
        retry_delay=0.0,
        streaming=False,
    )
    return build_hosted_chat_payload(
        record,
        resolution=resolution,
        messages_payload=[{"role": "user", "content": "hi"}],
        streaming=False,
        **kwargs,
    )


# --- TASK-33502: NVIDIA Qwen thinking toggle ---


@pytest.mark.parametrize(("effort", "enabled"), [("none", False), ("low", True), ("high", True)])
def test_nvidia_qwen_effort_becomes_the_documented_thinking_switch(effort: str, enabled: bool) -> None:
    """Args:
    effort: Console reasoning-effort level.
    enabled: The ``enable_thinking`` value NIM must receive.
    """
    payload = _payload(NVIDIA, _QWEN, reasoning_effort=effort)
    assert payload["chat_template_kwargs"] == {"enable_thinking": enabled}
    assert "reasoning_effort" not in payload


def test_nvidia_qwen_without_effort_sends_no_thinking_switch() -> None:
    assert "chat_template_kwargs" not in _payload(NVIDIA, _QWEN)


def test_nvidia_qwen_refuses_an_unknown_effort_level() -> None:
    """A boolean switch cannot let the provider reject a bad level, so the
    engine does (Qodo #2915)."""
    with pytest.raises(ChatBadRequestError):
        _payload(NVIDIA, _QWEN, reasoning_effort="garbage")


def test_nvidia_qwen_thinking_switch_survives_chat_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Console's generic dispatch (param map -> engine handler) carries
    reasoning effort all the way to the NIM payload."""
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    from tldw_chatbook.LLM_Calls import hosted_provider_engine

    resolution = HostedProviderResolution(
        provider="nvidia", model=_QWEN, api_key="secret",
        base_url=NVIDIA.default_base_url, timeout=10.0, retries=0,
        retry_delay=0.0, streaming=False,
    )
    monkeypatch.setattr(hosted_provider_engine, "resolve_hosted_request", lambda _r, **_k: resolution)
    captured: dict[str, Any] = {}

    def fake_post(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return {"id": "c", "object": "chat.completion", "created": 1, "model": _QWEN,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}]}

    monkeypatch.setattr(hosted_provider_engine, "owned_json_post", fake_post)
    chat_api_call(
        "nvidia", messages_payload=[{"role": "user", "content": "hi"}], api_key="secret",
        model=_QWEN, streaming=False, reasoning_effort="none",
    )
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_other_nvidia_models_still_refuse_reasoning_effort() -> None:
    with pytest.raises(ChatBadRequestError):
        _payload(NVIDIA, "meta/llama-3.3-70b-instruct", reasoning_effort="none")


def test_thinking_toggle_is_nvidia_qwen_only() -> None:
    declared = {r.key: dict(r.thinking_toggle_models) for r in ALL_RECORDS if r.thinking_toggle_models}
    assert declared == {"nvidia": {"qwen/qwen3.5-*": "enable_thinking"}}
    assert thinking_toggle_key(NVIDIA, _QWEN) == "enable_thinking"
    assert thinking_toggle_key(NVIDIA, "QWEN/QWEN3.5-397B-A17B") is None
    assert thinking_toggle_key(NVIDIA, None) is None


# --- TASK-33503: never thinking + reasoning_effort together ---


@pytest.mark.parametrize("record", [r for r in ALL_RECORDS if r.engine_driven], ids=lambda r: r.key)
def test_no_engine_preset_can_send_thinking_with_reasoning_effort(record: ProviderRecord) -> None:
    """Fireworks fails validation on the pair; no preset may build it.

    Args:
        record: One engine-driven provider record.
    """
    assert not ("thinking" in record.extra_body_fields and record.reasoning_effort)
    try:
        payload = _payload(record, reasoning_effort="high")
    except ChatBadRequestError:
        return  # the record refuses reasoning effort outright
    assert not {"thinking", "reasoning_effort"} <= payload.keys()


def test_fireworks_tool_turn_reasoning_is_kept_and_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fireworks requires reasoning_content back on interleaved tool turns: the
    tool-call response keeps it in the continuation, and the next request
    restores it on the assistant message (Qodo #2915)."""
    from tldw_chatbook.Chat.provider_continuation import parse_provider_continuation_json
    from tldw_chatbook.LLM_Calls import hosted_provider_engine

    resolution = HostedProviderResolution(
        provider="fireworks", model="accounts/fireworks/models/qwen3p8", api_key="secret",
        base_url=FIREWORKS.default_base_url, timeout=10.0, retries=0,
        retry_delay=0.0, streaming=False,
    )
    monkeypatch.setattr(hosted_provider_engine, "resolve_hosted_request", lambda _r, **_k: resolution)
    call = {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": "{}"}}
    monkeypatch.setattr(hosted_provider_engine, "owned_json_post", lambda **_k: {
        "id": "c", "object": "chat.completion", "created": 1, "model": resolution.model,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "", "reasoning_content": "PRIVATE", "tool_calls": [call]}}],
    })
    result = hosted_provider_engine.build_hosted_chat_handler(FIREWORKS)(
        input_data=[{"role": "user", "content": "add"}], api_key="secret", streaming=False, tools=[_TOOL],
    )
    assert "reasoning_content" not in result["choices"][0]["message"]  # private
    checkpoint = result.provider_continuation
    assert checkpoint is not None and list(checkpoint.rounds[0].reasoning_blocks) == ["PRIVATE"]

    completed = parse_provider_continuation_json({
        "schema_version": 1, "checkpoint_revision": 1, "provider": "fireworks",
        "protocol": "chat_completions", "model": resolution.model,
        "api_base_url": resolution.base_url, "state": "complete",
        "rounds": [{"assistant_content": "", "reasoning_blocks": ["PRIVATE"], "calls": [
            {"call_id": "call_1", "name": "add", "arguments": "{}", "state": "completed", "result": "3"}]}],
    })
    history = [
        {"role": "user", "content": "add"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "3"},
    ]
    payload = build_hosted_chat_payload(
        FIREWORKS, resolution=resolution, messages_payload=history,
        tools=[_TOOL], provider_continuations=[completed],
    )
    assert payload["messages"][1]["reasoning_content"] == "PRIVATE"


_CONSOLE_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh")


@pytest.mark.parametrize("level", ["none", "low", "high", "xhigh"])
def test_fireworks_sends_a_documented_reasoning_effort_level(level: str) -> None:
    """Fireworks takes reasoning_effort as-is, and never a thinking field.

    Args:
        level: A level Fireworks documents.
    """
    payload = _payload(FIREWORKS, "accounts/fireworks/models/deepseek-v3p1", reasoning_effort=level)
    assert payload["reasoning_effort"] == level
    assert "thinking" not in payload


@pytest.mark.parametrize("level", ["minimal", "max-ish"])
def test_fireworks_refuses_an_undocumented_level_locally(level: str) -> None:
    """Fireworks has no "minimal"; an unknown level never reaches the wire.

    Args:
        level: A level Fireworks does not document.
    """
    with pytest.raises(ChatBadRequestError):
        _payload(FIREWORKS, "accounts/fireworks/models/deepseek-v3p1", reasoning_effort=level)


def test_fireworks_settings_and_draft_rebase_follow_its_levels() -> None:
    """Settings leaves out "minimal" for Fireworks; the rebase carries the level."""
    from tldw_chatbook.Chat.console_provider_support import (
        reasoning_effort_values_sent,
        supported_generation_fields,
    )

    assert reasoning_effort_values_sent("fireworks", _CONSOLE_LEVELS) == ("none", "low", "medium", "high", "xhigh")
    assert reasoning_effort_values_sent("openai", _CONSOLE_LEVELS) == _CONSOLE_LEVELS
    assert "reasoning_effort" in supported_generation_fields("fireworks", "accounts/fireworks/models/qwen3p8")


def test_only_fireworks_restricts_reasoning_effort_levels() -> None:
    """The allowlist is per record, not a new global rule."""
    restricted = {r.key for r in ALL_RECORDS if r.reasoning_effort_values is not None}
    assert restricted == {"fireworks"}


# --- TASK-33500: Cerebras tools go out without strict ---


def test_cerebras_tools_are_sent_without_strict() -> None:
    payload = _payload(CEREBRAS, tools=[_TOOL])
    assert payload["tools"] == [_TOOL]
    assert "strict" not in payload["tools"][0]["function"]


def test_cerebras_refuses_a_caller_supplied_strict_flag() -> None:
    strict_tool = {"type": "function", "function": {**_TOOL["function"], "strict": True}}
    with pytest.raises(ChatBadRequestError):
        _payload(CEREBRAS, tools=[strict_tool])
