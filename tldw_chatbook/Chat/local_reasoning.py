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
    """Immutable optional-reasoning policy pinned to one resolved send.

    Attributes:
        mode: Effective history mode, including ``server_default`` for an
            unreviewed template's automatic behavior.
        source: User-facing description of where the policy was selected.
        template_family: Reviewed template family, or an empty string.
        supports_preserve: Whether the template accepts preservation options;
            this does not widen the template's own replay eligibility.
        verified: Whether the template matched an exact reviewed fingerprint.
        native_tools: Whether the server independently declares native tool
            support or the user explicitly enabled it for this target.
    """

    mode: str
    source: str
    template_family: str = ""
    supports_preserve: bool = False
    verified: bool = False
    native_tools: bool = False

    @property
    def label(self) -> str:
        """Return a concise description of the resolved mode and template.

        Returns:
            User-facing policy source, mode, and optional template family.
        """
        label = {value: label for label, value in REASONING_HISTORY_OPTIONS}.get(
            self.mode, "Server default (unverified template)"
        )
        return f"{self.source} — {label}" + (
            f" ({self.template_family})" if self.template_family else ""
        )


def resolve_reasoning_policy(
    mode: str, *, template: object = None, native_tools: bool = False
) -> ReasoningReplayPolicy:
    """Resolve only reviewed templates; unknown text is never executed.

    Args:
        mode: Requested mode: ``auto``, ``current``, ``all``, or ``off``.
            Unrecognized values use ``auto``.
        template: Template text to fingerprint, or any non-string value when
            metadata is unavailable. This function never renders the text.
        native_tools: Separately established native-tool capability.

    Returns:
        Frozen effective policy. Automatic mode uses a reviewed template's
        preference when available and otherwise retains the server default.
    """
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
    """Build a stable local target key without retaining endpoint credentials.

    Args:
        provider: Local provider key or supported alias.
        endpoint: Server URL, optionally ending in ``/v1/chat/completions``.
        model: Exact selected model identifier.

    Returns:
        SHA-256 digest of the normalized provider, credential-free endpoint,
        and model identity.

    Raises:
        ValueError: If the endpoint cannot be parsed as a URL.
    """
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
    """Read a scoped override, with an explicit legacy-off fallback.

    Args:
        console: Loaded Console settings mapping.
        provider: Provider key for an optional target-specific lookup.
        endpoint: Endpoint used with the provider and model to identify an
            override.
        model: Exact model identifier; an empty value skips scoped lookup.

    Returns:
        A supported mode from the scoped override or global preference, or
        ``off`` for a legacy opt-out and ``auto`` otherwise.

    Raises:
        ValueError: If an endpoint used for scoped lookup is not parseable.
    """
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
    """Apply explicit conversation authority before device-local Auto policy.

    Args:
        policy: Frozen device-local policy, or None when no policy was resolved.
        conversation_policy: Saved conversation preference. ``exclude`` forces
            Off, ``include`` forces All, and other values retain ``policy``.

    Returns:
        The effective optional replay policy without mutating the original.
        Required provider continuation is managed separately.
    """
    if conversation_policy in {"exclude", "include"}:
        base = policy or ReasoningReplayPolicy("server_default", "Server default")
        return replace(base, mode="off" if conversation_policy == "exclude" else "all")
    return policy


def starts_reasoning_exchange(row: Mapping[str, Any]) -> bool:
    """Identify a user request that starts a new reasoning exchange.

    Args:
        row: Semantic message, including any internal origin annotations.

    Returns:
        True for an actual user request. Tool results, runtime guidance, and
        ephemeral project context continue the existing exchange.
    """
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
    """Recognize local transports that accept separate reasoning fields.

    Args:
        provider: Provider key or supported local alias.
        model: Selected model identifier.

    Returns:
        True for a supported local transport with a nonblank model. This does
        not imply that its template or native-tool behavior is verified.
    """
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

    Args:
        messages: Semantic rows carrying optional ephemeral canonical call
            envelopes. Input rows and envelopes are not mutated.
        provider: Target provider key.
        model: Exact target model identifier.
        policy: Frozen local replay and template policy, when available.
        enabled: Whether optional replay is allowed. False applies Exclude
            independently of the policy's mode.

    Returns:
        Mutable provider-visible message copies with eligible reasoning and
        template control encodings. Empty input returns an empty list.

    Raises:
        ValueError: If a call envelope is invalid, belongs to a non-assistant
            row, or message ownership cannot form a canonical request.
        ThinkingHistorySerializationError: If retained reasoning cannot be
            encoded safely with its visible owner.
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
    """Expose exact reviewed template options for dispatch and trace admission.

    Args:
        provider: Target provider key or supported local alias.
        policy: Frozen replay policy, or None when unavailable.

    Returns:
        ``preserve_thinking`` with the selected retention value for compatible
        llama.cpp/vLLM templates, or an empty mapping otherwise.
    """
    if (
        policy is not None
        and policy.supports_preserve
        and _LOCAL_FAMILIES.get(provider.lower()) in {"llama_cpp", "vllm"}
    ):
        return {"preserve_thinking": policy.mode == "all"}
    return {}
