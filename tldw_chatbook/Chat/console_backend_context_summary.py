"""Original context-rail calculation from the backend's captured display inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.Utils.token_counter import ContextWindowResolution
from tldw_chatbook.Widgets.Console.console_context_controls import (
    ConsoleContextControlState,
    build_console_context_control_state,
    compaction_forecast_rows,
)

from .console_chat_models import ConsoleChatMessage
from .console_context_compaction import EffectiveMemoryResult
from .console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
    merge_context_policy,
)
from .console_execution_context_summary_values import ConsoleContextSummaryProjection
from .console_prepared_request import ConsoleRequestTokenAccounting
from .console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    build_console_context_estimate,
    build_console_settings_summary_state,
)
from .console_thinking_history import EffectiveThinkingHistoryPolicy


@dataclass(frozen=True, slots=True)
class ConsoleBackendContextInputs:
    """Resident values for one actual session; never a native-read authority."""

    settings: ConsoleSessionSettings
    summary_settings: ConsoleSessionSettings
    readiness: ConsoleSettingsReadiness
    messages: tuple[ConsoleChatMessage, ...]
    request_ids: frozenset[str] | None
    active_session: bool
    seeded_greeting: str
    draft_text: str
    staged_source_count: int
    staged_context_summary: str
    staged_text: str
    context_window: ContextWindowResolution | None
    tokenizer_ready: bool
    overrides: ConsoleContextPolicyOverrides
    global_overrides: ConsoleContextPolicyOverrides | None
    memory: EffectiveMemoryResult | None
    accounting: ConsoleRequestTokenAccounting | None
    thinking_history_policy: str
    thinking_history_effective_policy: EffectiveThinkingHistoryPolicy | None
    provider_token_caps: tuple[int | None, int | None] | None
    agent_runtime_enabled: bool
    agent_bridge_present: bool
    has_submit_prefill: bool
    has_pending_attachments: bool


def _settings_estimate(
    inputs: ConsoleBackendContextInputs,
) -> ConsoleSettingsContextEstimate:
    if not inputs.tokenizer_ready:
        return ConsoleSettingsContextEstimate(None, None, "Context unavailable")
    settings = inputs.settings
    active = inputs.active_session
    rows = spend.build_console_context_messages(
        inputs.messages, inputs.request_ids if active else None, ""
    )
    settled = build_console_context_estimate(
        rows,
        settings.provider,
        settings.model,
        staged_source_count=inputs.staged_source_count if active else 0,
        staged_context_summary=inputs.staged_context_summary,
        max_tokens_response=settings.max_tokens,
        system_prompt=spend.fold_system_prompt(
            settings.system_prompt, inputs.seeded_greeting if active else ""
        ),
        staged_text=inputs.staged_text if active else "",
        context_window=inputs.context_window,
    )
    if not active or not inputs.draft_text.strip() or settled.used_tokens is None:
        return settled
    return build_console_context_estimate(
        [{"role": "user", "content": inputs.draft_text}],
        settings.provider,
        settings.model,
        staged_source_count=inputs.staged_source_count,
        staged_context_summary=inputs.staged_context_summary,
        max_tokens_response=settings.max_tokens,
        context_window=inputs.context_window,
        history_used_tokens=settled.used_tokens,
    )


def _context_state(
    inputs: ConsoleBackendContextInputs, estimate: ConsoleSettingsContextEstimate
) -> ConsoleContextControlState:
    settings = inputs.settings
    forecast_tokens, forecast_units = None, 0
    if (
        inputs.tokenizer_ready
        and isinstance(inputs.memory, EffectiveMemoryResult)
        and merge_context_policy(
            global_overrides=inputs.global_overrides,
            conversation_overrides=inputs.overrides,
        ).compaction_mode
        is ContextCompactionMode.ASK
    ):
        admitted = [
            message
            for message in inputs.messages
            if inputs.request_ids is None or message.id in inputs.request_ids
        ]
        projected = compaction_forecast_rows(admitted, inputs.memory)
        if projected is not None:
            rows, forecast_units = projected
            forecast = build_console_context_estimate(
                list(rows),
                settings.provider,
                settings.model,
                context_window=inputs.context_window,
            )
            forecast_tokens = forecast.used_tokens

    mandatory = None
    if (
        estimate.used_tokens is not None
        and inputs.tokenizer_ready
        and not (inputs.agent_runtime_enabled and inputs.agent_bridge_present)
        and not inputs.has_submit_prefill
        and not inputs.has_pending_attachments
        and not any(message.attachments for message in inputs.messages)
    ):
        draft = inputs.draft_text if inputs.active_session else ""
        mandatory = build_console_context_estimate(
            [{"role": "user", "content": draft}] if draft.strip() else [],
            settings.provider,
            settings.model,
            system_prompt=spend.fold_system_prompt(
                settings.system_prompt, inputs.seeded_greeting
            ),
            staged_text=inputs.staged_text if inputs.active_session else "",
            context_window=inputs.context_window,
        ).used_tokens

    caps = inputs.provider_token_caps
    unavailable = ""
    estimated_history = forecast_tokens
    if forecast_units and (mandatory is None or caps is None):
        unavailable = "Current mandatory input or tool overhead is not measured."
        forecast_tokens = None
    state = build_console_context_control_state(
        settings=settings,
        estimate=estimate,
        overrides=inputs.overrides,
        global_overrides=inputs.global_overrides,
        active_memory=inputs.memory,
        busy=inputs.memory is None,
        status_message="Loading context…" if inputs.memory is None else "",
        accounting=inputs.accounting,
        thinking_history_policy=inputs.thinking_history_policy,
        thinking_history_effective_policy=inputs.thinking_history_effective_policy,
        forecast_tokens=forecast_tokens,
        forecast_units=forecast_units,
        request_overhead_tokens=mandatory,
        provider_input_cap_tokens=caps[0] if caps is not None else None,
        provider_output_cap_tokens=caps[1] if caps is not None else None,
        forecast_unavailable_reason=unavailable,
    )
    trigger = state.compaction_trigger_tokens
    if (
        unavailable
        and estimated_history is not None
        and trigger is not None
        and estimated_history >= trigger
    ):
        state = replace(
            state,
            forecast_high_context=True,
            forecast_unavailable_reason=(
                "Estimated history reaches the displayed compaction trigger; "
                "next-send check unavailable."
            ),
        )
    return state


def build_backend_context_summary(
    inputs: ConsoleBackendContextInputs,
) -> ConsoleContextSummaryProjection:
    return build_backend_context_presentation(inputs)[0]


def build_backend_context_presentation(
    inputs: ConsoleBackendContextInputs,
) -> tuple[ConsoleContextSummaryProjection, ConsoleContextControlState | None]:
    """Calculate the original active-session summary after native preparation.

    Args:
        inputs: Captured session/display values, including the actual tokenizer
            preparation result and original context-window resolution. The
            caller owns their currentness and resource-retirement checks.

    Returns:
        Original estimate and summary constructors plus the two advisory flags.

    Raises:
        ValueError: Token counting was declared ready without its captured window.
    """
    if (
        inputs.tokenizer_ready
        and (inputs.settings.model or "").strip()
        and inputs.context_window is None
    ):
        raise ValueError("Captured context window is required for prepared display")
    estimate = _settings_estimate(inputs)
    try:
        state = _context_state(inputs, estimate)
    except (KeyError, ValueError):
        # Same optional-control failure boundary as the original summary owner.
        state = None
    return ConsoleContextSummaryProjection(
        estimate,
        build_console_settings_summary_state(
            inputs.summary_settings, estimate, inputs.readiness
        ),
        state is not None and state.compaction_pause_forecast is not None,
        state is not None and bool(state.forecast_unavailable_reason),
    ), state
