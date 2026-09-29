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


# --- TASK-33500: Cerebras tools go out without strict ---


def test_cerebras_tools_are_sent_without_strict() -> None:
    payload = _payload(CEREBRAS, tools=[_TOOL])
    assert payload["tools"] == [_TOOL]
    assert "strict" not in payload["tools"][0]["function"]


def test_cerebras_refuses_a_caller_supplied_strict_flag() -> None:
    strict_tool = {"type": "function", "function": {**_TOOL["function"], "strict": True}}
    with pytest.raises(ChatBadRequestError):
        _payload(CEREBRAS, tools=[strict_tool])
