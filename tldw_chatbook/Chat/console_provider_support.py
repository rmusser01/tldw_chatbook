"""Console provider identity helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from tldw_chatbook.Chat.console_context_policy import ContextCarryForwardMode
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.Chat.provider_readiness import (
    PROVIDERS_REQUIRING_API_KEY_KEYS,
    provider_config_key,
)
from tldw_chatbook.model_capabilities import (
    anthropic_model_rejects_fixed_thinking_budget,
    moonshot_model_supports_reasoning_effort,
    zai_model_supports_reasoning_effort,
)
from tldw_chatbook.provider_registry import RECORDS_BY_KEY, thinking_toggle_key

DIRECT_CONSOLE_PROVIDER_KEYS = frozenset({"llama_cpp", "local_llamacpp"})

ConsoleGenerationControl = Literal[
    "reasoning_effort",
    "reasoning_summary",
    "verbosity",
    "thinking_effort",
    "thinking_budget_tokens",
]
ConsoleControlSupport = Literal["supported", "unsupported", "unknown"]

_READINESS_TO_EXECUTION_ALIASES = {
    "custom": "custom-openai-api",
    "custom_2": "custom-openai-api-2",
    "local_llm": "local-llm",
    "local_mlx_lm": "local_mlx_lm",
    "mistralai": "mistralai",
}

_EXECUTION_TO_READINESS_ALIASES = {
    "custom-openai-api": "custom",
    "custom-openai-api-2": "custom_2",
    "local-llm": "local_llm",
    "mlx_lm": "local_mlx_lm",
}


@dataclass(frozen=True)
class ConsoleProviderIdentity:
    """Resolved Console provider identities for config, readiness, and send.

    Attributes:
        display_key: Normalized provider key used by Console controls.
        readiness_key: Provider key used for configuration/readiness lookup.
        execution_key: Provider key passed to ``chat_api_call``.
        is_supported: Whether Console can send through this provider.
        uses_direct_llama_path: Whether the provider bypasses the generic
            adapter and uses the direct llama.cpp path.
    """

    display_key: str
    readiness_key: str
    execution_key: str
    is_supported: bool
    uses_direct_llama_path: bool = False


@dataclass(frozen=True)
class ConsoleProviderCatalogEntry:
    """Provider option Settings can display for Console-compatible sends."""

    readiness_key: str
    execution_key: str
    display_name: str
    requires_api_key: bool
    uses_direct_llama_path: bool = False


# ADR-066: per-execution-key wire formats for Console thinking controls.
# Level = reasoning_effort; budget = thinking_budget_tokens.
CUSTOM_OPENAI_EXECUTION_KEYS = frozenset(
    {"custom-openai-api", "custom-openai-api-2", "custom-hosted"}
)
"""The custom-endpoint family's execution keys (ADR-179 Phase 2 Task 6).

Every gateway/trace surface keyed on the custom family's EXECUTION keys
must consume THIS constant, never a bare literal set: the engine swap
routes ``openai_compatible`` custom-ep entries through ``custom-hosted``,
and a literal set would drop the swapped key from base-URL forwarding,
credential decisions, and thinking support in one silent step. Identity
surfaces (aliases, readiness maps, dispatch registration) keep their own
spellings -- they key on ``custom``/``custom-ep:<slug>``, which
``family_execution_key`` resolves to the legacy slots
(ADR-179 Phase 2 Task 6 decision 1). Guarded by the literal-grep test
``Tests/Chat/test_custom_openai_execution_keys_constant.py``.
"""
_LLAMA_CPP_THINKING_KEYS = frozenset(
    {"llama_cpp", "local_llamacpp", "local_llamafile", "local-llm"}
)
_VLLM_THINKING_KEYS = frozenset({"vllm", "local_vllm"})
_CUSTOM_OPENAI_THINKING_KEYS = CUSTOM_OPENAI_EXECUTION_KEYS
# MLX-LM: template-kwargs shape pending live verification of mlx_lm.server
# support; if unsupported this row degrades to drop-and-log.
_TEMPLATE_KWARGS_THINKING_KEYS = frozenset({"local_mlx_lm"})
# Live-verified (llama.cpp b10430 + Qwen3.8): strict chat templates such as
# Qwen3.8's validate reasoning_effort and raise on unknown values ("minimal"
# -> HTTP 500). "high" is aliased to "xhigh" by the template and is safe;
# "none" is safe because we pair it with enable_thinking=false which
# short-circuits the template's validation block.
_TEMPLATE_SAFE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "none"})

_LOCAL_REASONING_EXECUTION_KEYS = (
    _LLAMA_CPP_THINKING_KEYS
    | _VLLM_THINKING_KEYS
    | _CUSTOM_OPENAI_THINKING_KEYS
    | _TEMPLATE_KWARGS_THINKING_KEYS
)
_LOCAL_BUDGET_EXECUTION_KEYS = _LLAMA_CPP_THINKING_KEYS
_LOCAL_DROPPED_CONTROLS = frozenset(
    {"reasoning_summary", "verbosity", "thinking_effort"}
)

#: Each generation field's ``chat_api_call`` request key(s): the generic names
#: the Console send builders fill from that field
#: (``ConsoleProviderGateway._chat_api_kwargs_from_prepared`` for sends and
#: tool rounds, ``_auxiliary_chat_api_kwargs`` for auxiliary calls; the table
#: test in Tests/Chat/test_console_provider_support.py measures both).
#: ``top_p`` rides both ``topp`` and ``maxp``, so a provider map carrying
#: either one forwards it (OpenAI's carries only ``maxp``). The only place the
#: support decision's field-to-key mapping is defined.
GENERATION_FIELD_REQUEST_KEYS: dict[str, tuple[str, ...]] = {
    "temperature": ("temp",),
    "top_p": ("topp", "maxp"),
    "min_p": ("minp",),
    "top_k": ("topk",),
    "max_tokens": ("max_tokens",),
    "seed": ("seed",),
    "presence_penalty": ("presence_penalty",),
    "frequency_penalty": ("frequency_penalty",),
    "reasoning_effort": ("reasoning_effort",),
    "reasoning_summary": ("reasoning_summary",),
    "verbosity": ("verbosity",),
    "thinking_effort": ("thinking_effort",),
    "thinking_budget_tokens": ("thinking_budget_tokens",),
    "streaming": ("streaming",),
}


@dataclass(frozen=True)
class ModelConfigField:
    """One model-configuration field's user-facing copy (TASK-33002.1).

    Attributes:
        name: Field name; generation fields use their
            ``GENERATION_FIELD_REQUEST_KEYS`` name.
        label: The one label every editor shows.
        help: A plain-language line saying what the field does. It never
            names a config key.
        valid_range: The values the field accepts.
    """

    name: str
    label: str
    help: str
    valid_range: str

    @property
    def request_keys(self) -> tuple[str, ...]:
        """Request keys read from the one definition.

        Returns:
            The request keys this field writes, or ``()`` for a
            non-generation field.
        """
        return GENERATION_FIELD_REQUEST_KEYS.get(self.name, ())


#: The one field table: every editor's label, help and range for each field
#: (Alt+M popover, Chat settings, Settings model defaults and Console
#: Behavior fallbacks). Labels stay within the modal's 23-cell label column;
#: a model field's help stays within 55 cells, so Settings rows show it
#: whole at 211x44 (Model defaults has 61, Console Behavior 55).
MODEL_CONFIG_FIELDS: dict[str, ModelConfigField] = {
    field.name: field
    for field in (
        ModelConfigField(
            "temperature",
            "Temperature",
            "Lower keeps replies focused; higher makes them varied.",
            "0.0 to 2.0",
        ),
        ModelConfigField(
            "top_p",
            "Top P",
            "Sample only from the top tokens whose chances sum to P.",
            "0.0 to 1.0",
        ),
        ModelConfigField(
            "min_p",
            "Min P",
            "Skip tokens far less likely than the top choice.",
            "0.0 to 1.0",
        ),
        ModelConfigField(
            "top_k",
            "Top K",
            "Sample only from the K most likely tokens.",
            "whole number, 0 or more",
        ),
        ModelConfigField(
            "max_tokens",
            "Max tokens",
            "Longest reply the model may write, in tokens.",
            "whole number, 1 or more",
        ),
        ModelConfigField(
            "seed",
            "Seed",
            "Fixed number for repeatable replies, where supported.",
            "whole number, 0 or more",
        ),
        ModelConfigField(
            "presence_penalty",
            "Presence penalty",
            "Higher values push the model toward new topics.",
            "-2.0 to 2.0",
        ),
        ModelConfigField(
            "frequency_penalty",
            "Frequency penalty",
            "Higher values make the model repeat words less.",
            "-2.0 to 2.0",
        ),
        ModelConfigField(
            "reasoning_effort",
            "Reasoning effort",
            "How much a reasoning model thinks before it answers.",
            "a level from the list",
        ),
        ModelConfigField(
            "reasoning_summary",
            "Reasoning summary",
            "How much of its reasoning the model summarizes for you.",
            "auto, concise, detailed or none",
        ),
        ModelConfigField(
            "verbosity",
            "Verbosity",
            "How long and detailed replies are.",
            "low, medium or high",
        ),
        ModelConfigField(
            "thinking_effort",
            "Thinking",
            "How much extended thinking comes before the answer.",
            "off, low, medium, high, xhigh or max",
        ),
        ModelConfigField(
            "thinking_budget_tokens",
            "Thinking budget",
            "Tokens reserved for thinking when Thinking is on.",
            "whole number, 1,024 or more",
        ),
        ModelConfigField(
            "streaming",
            "Streaming",
            "Show the reply as it is generated.",
            "On or Off",
        ),
        ModelConfigField(
            "endpoint",
            "Endpoint",
            "Server address the requests go to.",
            "an http:// or https:// address",
        ),
        ModelConfigField(
            "conversation_budget_mode",
            "Budget strategy",
            "How much conversation history each request may carry.",
            "Automatic or Custom",
        ),
        ModelConfigField(
            "compaction_mode",
            "When limit nears",
            "What happens when the conversation nears its token limit.",
            "Ask, Automatic or Off",
        ),
        ModelConfigField(
            "compaction_target_ratio",
            "Reduce context to (%)",
            "How full the conversation budget is left after compaction.",
            "a percentage at least 15 below Compact at",
        ),
        ModelConfigField(
            "compaction_carry_forward_mode",
            "Keep after compaction",
            "What stays word for word next to the memory summary.",
            "Memory with recent turns or Memory with latest exchange",
        ),
    )
}
MODEL_FIELD_LABELS = {name: field.label for name, field in MODEL_CONFIG_FIELDS.items()}
#: The Keep after compaction choices, shared by both editors of the field.
CARRY_FORWARD_OPTIONS = (
    (
        "Memory with recent turns",
        ContextCarryForwardMode.MEMORY_WITH_RECENT_TURNS.value,
    ),
    (
        "Memory with latest exchange",
        ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE.value,
    ),
)
_PROVIDER_GATED_GENERATION_FIELDS = frozenset(
    {
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
    }
)
_DIRECT_PROVIDER_GENERATION_FIELDS = {
    "openai": frozenset({"reasoning_effort", "reasoning_summary", "verbosity"}),
    "qwencloud": frozenset({"reasoning_effort"}),
}


def _engine_reasoning_effort_support(
    execution_key: str | None, model: str | None
) -> ConsoleControlSupport | None:
    """Decide reasoning effort for an engine preset whose record refuses it.

    Such a preset fails the send locally, so the Console must not offer the
    control (TASK-33501) -- unless the model has a thinking toggle the engine
    sends instead (TASK-33502).

    Args:
        execution_key: ``chat_api_call`` provider key.
        model: Selected model identifier, if any.

    Returns:
        ``supported`` or ``unsupported`` for such a preset; ``None`` for any
        other provider, whose answer is decided elsewhere.
    """
    record = RECORDS_BY_KEY.get(execution_key or "")
    if record is None or not record.engine_driven or record.reasoning_effort:
        return None
    return "supported" if thinking_toggle_key(record, model) else "unsupported"


def _capability_generation_fields(
    provider: str | None, model: str | None
) -> frozenset[str]:
    """Return the fields the provider/model capability rules allow."""

    provider_key = provider_config_key(provider or "")
    supported = set(GENERATION_FIELD_REQUEST_KEYS) - _PROVIDER_GATED_GENERATION_FIELDS
    if provider_key == "moonshot" and moonshot_model_supports_reasoning_effort(model):
        supported.add("reasoning_effort")
    if provider_key == "zai" and zai_model_supports_reasoning_effort(model):
        supported.add("reasoning_effort")
    supported.update(_DIRECT_PROVIDER_GENERATION_FIELDS.get(provider_key, ()))
    if provider_key == "anthropic":
        supported.add("thinking_effort")
        if not anthropic_model_rejects_fixed_thinking_budget(model):
            supported.add("thinking_budget_tokens")

    execution_key = resolve_console_provider_identity(provider_key).execution_key
    if build_local_thinking_payload_fields(execution_key, "low", None):
        supported.add("reasoning_effort")
    if _engine_reasoning_effort_support(execution_key, model) == "supported":
        supported.add("reasoning_effort")
    engine_record = RECORDS_BY_KEY.get(execution_key or "")
    if engine_record is not None and engine_record.engine_driven and engine_record.reasoning_effort:
        # A preset that sends reasoning effort (Fireworks): the draft rebase
        # must carry the user's level to the request.
        supported.add("reasoning_effort")
    if execution_key in _LOCAL_BUDGET_EXECUTION_KEYS:
        supported.add("thinking_budget_tokens")
    return frozenset(supported)


def _registry_family_provider(
    provider: str | None, app_config: Mapping[str, object] | None
) -> str:
    """Return a ``custom-ep`` id's family key, as the gateway sends it.

    Without ``app_config`` (or for any other id) the provider is returned
    unchanged, so a registry id then has no map entry.
    """
    if app_config is not None:
        # Lazy: the registry imports this module via console_session_settings.
        from tldw_chatbook.Chat.custom_endpoint_registry import (
            entry_for,
            family_execution_key,
        )

        entry = entry_for(app_config, provider)
        if entry is not None:
            return family_execution_key(entry.family)
    return provider or ""


def reasoning_effort_values_sent(
    provider: str | None,
    values: tuple[str, ...],
    app_config: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Return the reasoning-effort ``values`` a request for ``provider`` carries.

    A local strict-template request drops a level its chat template rejects
    ("minimal" on llama.cpp) with only a debug log, so an editor offering it
    would save a value that is never sent (TASK-33002.1). An engine preset
    with ``reasoning_effort_values`` (Fireworks) keeps the levels it sends,
    after its ``reasoning_effort_map`` (Fireworks sends "minimal" as "low").
    Every other provider keeps ``values`` unchanged.

    Args:
        provider: Provider identity, config key or ``custom-ep`` id.
        values: Candidate levels, in display order.
        app_config: Config holding the ADR-146 endpoint registry.

    Returns:
        The subset of ``values`` the request builder forwards, in order.
    """
    execution_key = resolve_console_provider_identity(
        provider_config_key(_registry_family_provider(provider, app_config))
    ).execution_key
    engine_record = RECORDS_BY_KEY.get(execution_key or "")
    if engine_record is not None and engine_record.reasoning_effort_values is not None:
        mapped = engine_record.reasoning_effort_map
        return tuple(
            value for value in values
            if mapped.get(value, value) in engine_record.reasoning_effort_values
        )
    if not build_local_thinking_payload_fields(execution_key, "low", None):
        return values
    return tuple(
        value
        for value in values
        if build_local_thinking_payload_fields(execution_key, value, None)
    )


def supported_generation_fields(
    provider: str | None,
    model: str | None,
    app_config: Mapping[str, object] | None = None,
) -> frozenset[str]:
    """Return the generation fields one provider/model request really carries.

    The single field-support decision for the Console draft rebase, the
    model-default writer and Settings' model-default rows: the capability
    rules intersected with the provider's ``PROVIDER_PARAM_MAP`` entry, so a
    field the request would drop (Anthropic's Min P, Seed and penalties) is
    never reported supported. A provider with no map entry keeps the
    capability answer unchanged (TASK-30012 AC#3).

    Args:
        provider: Provider identity or config key selected by the draft.
        model: Selected model identifier, when one is chosen.
        app_config: Config holding the ADR-146 endpoint registry. With it, a
            ``custom-ep`` id is decided as its entry's family, the way the
            gateway sends it; without it, a registry id has no map entry.

    Returns:
        Names of the supported fields, drawn from
        ``GENERATION_FIELD_REQUEST_KEYS``.
    """
    from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP

    provider_key = provider_config_key(_registry_family_provider(provider, app_config))
    capable = _capability_generation_fields(provider_key, model)
    request_keys = PROVIDER_PARAM_MAP.get(
        resolve_console_provider_identity(provider_key).execution_key
    )
    if request_keys is None:
        return capable
    return frozenset(
        field
        for field in capable
        if any(key in request_keys for key in GENERATION_FIELD_REQUEST_KEYS[field])
    )


def console_generation_control_support(
    provider: str,
    model: str | None,
    control: ConsoleGenerationControl,
    app_config: Mapping[str, object] | None = None,
) -> ConsoleControlSupport:
    """Return existing authoritative support for one generation control.

    The answer describes whether the current Console send path and known model
    family consume the control. An unrecognised model stays ``unknown`` rather
    than inheriting a negative result from a capability predicate whose false
    value also covers names outside that predicate's domain.

    Args:
        provider: Provider selected by the Console draft.
        model: Optional selected model identifier.
        control: Generation control whose support should be projected.
        app_config: Config holding the ADR-146 endpoint registry. With it, a
            ``custom-ep`` id is decided as its entry's family, the way
            ``supported_generation_fields`` and the gateway decide it.

    Returns:
        ``supported``, ``unsupported``, or ``unknown`` for the exact draft.
    """
    identity = resolve_console_provider_identity(
        _registry_family_provider(provider, app_config)
    )
    execution_key = identity.execution_key
    readiness_key = identity.readiness_key

    if execution_key in _LOCAL_REASONING_EXECUTION_KEYS:
        if control in _LOCAL_DROPPED_CONTROLS:
            return "unsupported"
        if control == "thinking_budget_tokens":
            return (
                "supported"
                if execution_key in _LOCAL_BUDGET_EXECUTION_KEYS
                else "unsupported"
            )
        if control == "reasoning_effort":
            # A custom OpenAI-compatible server accepts an arbitrary model and
            # does not provide authoritative model capability metadata.
            if execution_key in (
                _CUSTOM_OPENAI_THINKING_KEYS | _TEMPLATE_KWARGS_THINKING_KEYS
            ):
                return "unknown"
            return "supported"

    if control == "reasoning_effort":
        engine_answer = _engine_reasoning_effort_support(execution_key, model)
        if engine_answer is not None:
            return engine_answer

    if identity.is_supported:
        from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP

        provider_params = PROVIDER_PARAM_MAP.get(execution_key)
        if provider_params is not None and control not in provider_params:
            return "unsupported"

    if readiness_key == "anthropic":
        if anthropic_model_rejects_fixed_thinking_budget(model):
            return "unsupported" if control == "thinking_budget_tokens" else "supported"
        return "unknown"

    if readiness_key == "moonshot":
        return (
            "supported"
            if moonshot_model_supports_reasoning_effort(model)
            else "unknown"
        )

    if readiness_key == "zai":
        return "supported" if zai_model_supports_reasoning_effort(model) else "unknown"

    return "unknown"


def build_local_thinking_payload_fields(
    execution_key: str | None,
    reasoning_effort: str | None,
    thinking_budget_tokens: int | None,
) -> dict[str, Any]:
    """Compose thinking-control payload fragments for a local provider.

    Args:
        execution_key: ``chat_api_call`` provider key (e.g. ``llama_cpp``).
        reasoning_effort: Verbatim user-selected effort level, if any.
        thinking_budget_tokens: Max thinking tokens, if any.

    Returns:
        Fragments to merge into an OpenAI-compatible chat payload. Empty
        dict when the key has no thinking support or no values are set.
    """
    key = str(execution_key or "").strip().lower()
    effort = str(reasoning_effort or "").strip().lower() or None
    budget: int | None = (
        thinking_budget_tokens
        if isinstance(thinking_budget_tokens, int)
        and not isinstance(thinking_budget_tokens, bool)
        else None
    )
    fields: dict[str, Any] = {}
    if key in _LLAMA_CPP_THINKING_KEYS or key in _TEMPLATE_KWARGS_THINKING_KEYS:
        if effort is not None:
            if effort in _TEMPLATE_SAFE_EFFORTS:
                template_kwargs: dict[str, Any] = {"reasoning_effort": effort}
                if effort == "none":
                    template_kwargs["enable_thinking"] = False
                fields["chat_template_kwargs"] = template_kwargs
            else:
                logger.debug(
                    "reasoning effort '{}' is not consumable by strict chat "
                    "templates; dropped from chat_template_kwargs",
                    effort,
                )
        if budget is not None and key in _LLAMA_CPP_THINKING_KEYS:
            fields["reasoning_budget_tokens"] = budget
        if budget is not None and key in _TEMPLATE_KWARGS_THINKING_KEYS:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    elif key in _VLLM_THINKING_KEYS:
        if effort is not None:
            fields["reasoning_effort"] = effort
            if effort in _TEMPLATE_SAFE_EFFORTS:
                fields["chat_template_kwargs"] = {"reasoning_effort": effort}
            else:
                logger.debug(
                    "reasoning effort '{}' is not consumable by strict chat "
                    "templates; dropped from chat_template_kwargs",
                    effort,
                )
        if budget is not None:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    elif key in _CUSTOM_OPENAI_THINKING_KEYS:
        if effort is not None:
            fields["reasoning_effort"] = effort
        if budget is not None:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    return fields


def _handler_keys(handler_keys: Collection[str] | None = None) -> frozenset[str]:
    """Return supported ``chat_api_call`` execution keys."""
    if handler_keys is not None:
        return frozenset(handler_keys)

    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    return frozenset(API_CALL_HANDLERS)


def resolve_console_provider_identity(
    provider: str | None,
    *,
    handler_keys: Collection[str] | None = None,
) -> ConsoleProviderIdentity:
    """Resolve Console provider display, readiness, and execution keys.

    Args:
        provider: Raw provider name from config or Console controls.
        handler_keys: Optional ``chat_api_call`` handler keys for deterministic
            tests or side-effect-free callers.

    Returns:
        Resolved provider identity describing display, readiness, and execution
        keys plus whether the provider is supported.
    """
    raw_provider = (provider or "").strip()
    display_key = provider_config_key(raw_provider)
    exact_key = raw_provider.lower()

    # ADR-146: a custom-ep id addresses a registry entry that only
    # app_config-holding seams can resolve (``entry_for`` in
    # ``custom_endpoint_registry``). Importing that module here would cycle
    # (registry -> console_session_settings -> this module), so this is a
    # deliberately minimal inline prefix check: an unresolvable custom-ep id
    # degrades to the generic OpenAI-compatible family instead of an unknown
    # provider, and registry-aware seams override it with the entry's family.
    if raw_provider.startswith("custom-ep:") and len(raw_provider) > len("custom-ep:"):
        return ConsoleProviderIdentity(
            display_key=raw_provider,
            readiness_key="custom",
            execution_key="custom",
            is_supported=True,
            uses_direct_llama_path=False,
        )

    if (
        exact_key in DIRECT_CONSOLE_PROVIDER_KEYS
        or display_key in DIRECT_CONSOLE_PROVIDER_KEYS
    ):
        direct_key = (
            exact_key if exact_key in DIRECT_CONSOLE_PROVIDER_KEYS else display_key
        )
        return ConsoleProviderIdentity(
            display_key=direct_key,
            readiness_key=direct_key,
            execution_key=direct_key,
            is_supported=True,
            uses_direct_llama_path=True,
        )

    handlers = _handler_keys(handler_keys)
    normalized_handler_keys = {
        provider_config_key(handler_key): handler_key for handler_key in handlers
    }
    handler_exact_key = (
        exact_key
        if exact_key in handlers
        else normalized_handler_keys.get(display_key, exact_key)
    )
    readiness_key = _EXECUTION_TO_READINESS_ALIASES.get(handler_exact_key, display_key)
    execution_key = _READINESS_TO_EXECUTION_ALIASES.get(readiness_key)
    if execution_key is None:
        execution_key = (
            handler_exact_key if handler_exact_key in handlers else readiness_key
        )

    return ConsoleProviderIdentity(
        display_key=display_key,
        readiness_key=readiness_key,
        execution_key=execution_key,
        is_supported=execution_key in handlers,
        uses_direct_llama_path=False,
    )


def supported_console_provider_catalog(
    handler_keys: Collection[str] | None = None,
) -> tuple[ConsoleProviderCatalogEntry, ...]:
    """Return Console-sendable provider catalog entries for Settings.

    Args:
        handler_keys: Optional ``chat_api_call`` handler keys for deterministic
            tests or side-effect-free callers.

    Returns:
        Stable, de-duplicated provider entries keyed by readiness/config key.
    """
    handlers = _handler_keys(handler_keys)
    entries: dict[str, ConsoleProviderCatalogEntry] = {}
    for handler_key in sorted(handlers):
        identity = resolve_console_provider_identity(
            handler_key,
            handler_keys=handlers,
        )
        if not identity.is_supported:
            continue
        entries.setdefault(
            identity.readiness_key,
            ConsoleProviderCatalogEntry(
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
                display_name=provider_display_name(identity.readiness_key),
                requires_api_key=identity.readiness_key
                in PROVIDERS_REQUIRING_API_KEY_KEYS,
                uses_direct_llama_path=identity.uses_direct_llama_path,
            ),
        )
    return tuple(sorted(entries.values(), key=lambda entry: entry.readiness_key))


def supported_console_provider_readiness_keys(
    handler_keys: Collection[str] | None = None,
) -> frozenset[str]:
    """Return readiness keys supported by Console provider execution.

    Args:
        handler_keys: Optional ``chat_api_call`` handler keys for deterministic
            tests or side-effect-free callers.

    Returns:
        Set of normalized readiness keys whose providers can be sent from
        Console.
    """
    handlers = _handler_keys(handler_keys)
    return frozenset(
        resolve_console_provider_identity(
            handler_key,
            handler_keys=handlers,
        ).readiness_key
        for handler_key in handlers
    )
