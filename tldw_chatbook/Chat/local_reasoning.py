"""Device-local replay policy for canonical Console thinking envelopes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit, urlunsplit

LOCAL_TOOL_RESULT_KEY = "_tldw_local_tool_result"
EXCHANGE_CONTINUATION_KEY = "_tldw_exchange_continuation"
_LOCAL_FAMILIES = {
    "llama_cpp": "llama_cpp",
    "local_llamacpp": "llama_cpp",
    "local-llm": "llama_cpp",
    "local_llamafile": "llama_cpp",
    "vllm": "vllm",
    "local_vllm": "vllm",
    "ollama": "ollama",
    "local_ollama": "ollama",
}


REASONING_HISTORY_OPTIONS = (
    ("Automatic (recommended)", "auto"),
    ("Current exchange", "current"),
    ("All available", "all"),
    ("Off", "off"),
)
# Exact reviewed upstream templates; provenance and render tests live in
# Tests/fixtures/reasoning_templates. Do not infer policy from a model alias.
_REVIEWED_TEMPLATES = {
    "6a1015c47ccfcfa67c3b772385bccee357a4d37c3cda37bd202e9047f391ab82": (
        "Gemma 4",
        "current",
        True,
    ),
    "1d35a24a2a63cc600c0cab43628a6df4cd3318e959b03307cf5f9f7e29ecf783": (
        "Gemma 4",
        "current",
        False,
    ),
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715": (
        "Qwen3.5",
        "current",
        False,
    ),
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259": (
        "Qwen3.6",
        "current",
        True,
    ),
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041": (
        "Qwen3.8",
        "all",
        True,
    ),
}


@dataclass(frozen=True)
class ReasoningReplayPolicy:
    """Immutable optional-reasoning policy pinned to one resolved send."""

    mode: str
    source: str
    template_family: str = ""
    supports_preserve: bool = False
    verified: bool = False
    native_tools: bool = False

    @property
    def label(self) -> str:
        label = {value: label for label, value in REASONING_HISTORY_OPTIONS}.get(
            self.mode, "Server default (unverified template)"
        )
        return f"{self.source} — {label}" + (
            f" ({self.template_family})" if self.template_family else ""
        )


def resolve_reasoning_policy(
    mode: str, *, template: object = None, native_tools: bool = False
) -> ReasoningReplayPolicy:
    """Resolve only reviewed templates; unknown text is never executed."""
    known = None
    if isinstance(template, str):
        digest = hashlib.sha256(
            template.replace("\r\n", "\n").strip().encode()
        ).hexdigest()
        known = _REVIEWED_TEMPLATES.get(digest)
    family, automatic, preserve = known or ("", "server_default", False)
    if mode not in {value for _, value in REASONING_HISTORY_OPTIONS}:
        mode = "auto"
    return ReasoningReplayPolicy(
        automatic if mode == "auto" else mode,
        "Auto" if mode == "auto" else "User override",
        family,
        preserve,
        known is not None,
        native_tools,
    )


def reasoning_override_key(provider: str, endpoint: str, model: str) -> str:
    """Stable local target identity, without persisting credentials in the key."""
    parsed = urlsplit(endpoint.strip())
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/v1"):
        path = path.removesuffix(suffix)
    netloc = parsed.netloc.rsplit("@", 1)[-1].lower()
    normalized = urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))
    identity = [
        _LOCAL_FAMILIES.get(provider.lower(), provider.lower()),
        normalized,
        model,
    ]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def reasoning_mode_setting(
    console: Mapping[str, Any],
    *,
    provider: str = "",
    endpoint: str = "",
    model: str = "",
) -> str:
    """Read a scoped override, with explicit legacy-off migration."""
    modes = {value for _, value in REASONING_HISTORY_OPTIONS}
    overrides = console.get("reasoning_history_overrides", {})
    if provider and model and isinstance(overrides, Mapping):
        value = overrides.get(reasoning_override_key(provider, endpoint, model))
        if isinstance(value, str) and value in modes:
            return value
    value = console.get("reasoning_history")
    if isinstance(value, str) and value in modes:
        return value
    from tldw_chatbook.config import coerce_bool_setting

    return (
        "auto"
        if coerce_bool_setting(console.get("replay_thinking", True), True)
        else "off"
    )


def effective_replay_policy(
    policy: ReasoningReplayPolicy | None,
    conversation_policy: str,
) -> ReasoningReplayPolicy | None:
    """Apply explicit conversation authority before device-local Auto policy."""
    if conversation_policy in {"exclude", "include"}:
        base = policy or ReasoningReplayPolicy("server_default", "Server default")
        return replace(base, mode="off" if conversation_policy == "exclude" else "all")
    return policy


def starts_reasoning_exchange(row: Mapping[str, Any]) -> bool:
    """Only an actual user request closes the previous reasoning exchange."""
    from tldw_chatbook.Agents.agent_models import FENCE_TOOL_RESULT_PREFIX
    from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

    return (
        row.get("role") == "user"
        and row.get(LOCAL_TOOL_RESULT_KEY) is not True
        and row.get(EXCHANGE_CONTINUATION_KEY) is not True
        and EPHEMERAL_ORIGIN_KEY not in row
        and not str(row.get("content") or "").startswith(FENCE_TOOL_RESULT_PREFIX)
    )


def supports_local_reasoning(provider: str, model: str) -> bool:
    """Recognize local transports that accept separate reasoning fields."""
    return provider.lower() in _LOCAL_FAMILIES and bool(model.strip())


def project_reasoning_history(
    messages: Sequence[Mapping[str, Any]],
    *,
    provider: str,
    model: str,
    policy: ReasoningReplayPolicy | None = None,
    enabled: bool = True,
) -> list[dict[str, Any]]:
    """Project ephemeral per-call canonical envelopes through the wire serializer.

    This is the agent accounting path; durable history comes from canonical
    sidecars at the gateway. No separate persisted reasoning body is accepted.
    """
    if not messages:
        return []
    from .console_prepared_request import (
        THINKING_OWNER_KEY,
        build_console_request,
        prepare_provider_request,
        resolve_request_capacity,
        thaw_json,
    )
    from .console_thinking_capture import consume_call_thinking
    from .console_thinking_history import ThinkingReplayTarget, resolve_thinking_history

    rows, sidecars = consume_call_thinking(messages, owner_key=THINKING_OWNER_KEY)
    resolved = resolve_thinking_history(
        target=ThinkingReplayTarget(
            provider,
            model,
            "chat_completions",
            "displayable",
            1,
            reasoning_replay=policy,
        ),
        policy="auto" if enabled else "exclude",
        sidecars=tuple(sidecars),
    )
    eligible = {group.owner_message_id for group in resolved.groups}
    rows = [
        {
            key: value
            for key, value in row.items()
            if key != THINKING_OWNER_KEY or value in eligible
        }
        for row in rows
    ]
    semantic = build_console_request(
        rows,
        thinking_groups=resolved.groups,
        thinking_policy=resolved.saved_policy,
        effective_thinking_policy=resolved.effective_policy,
    )
    prepared = prepare_provider_request(
        semantic,
        wire_style="distinct_roles",
        provider=provider,
        model=model,
        reasoning_replay=policy,
        capacity=resolve_request_capacity(context_window_tokens=None),
        count_fn=lambda _rows, _model: 0,
    )
    return [thaw_json(row) for row in prepared.messages]


def reasoning_template_kwargs(
    provider: str,
    policy: ReasoningReplayPolicy | None,
) -> dict[str, bool]:
    """Expose exact reviewed template options for dispatch and trace admission."""
    if (
        policy is not None
        and policy.supports_preserve
        and _LOCAL_FAMILIES.get(provider.lower()) in {"llama_cpp", "vllm"}
    ):
        return {"preserve_thinking": policy.mode == "all"}
    return {}
