"""Presentation contracts for Console conversation context controls.

The widgets consume this immutable snapshot instead of reading the database or
reconstructing provider requests.  That keeps policy ownership in the Console
store/controller while making the quick and full settings surfaces agree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextCapacity,
    ConsoleContextPolicyDefaults,
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
    ResolvedConsoleContextPolicy,
    application_context_policy_defaults,
    merge_context_policy,
    resolve_context_policy,
)
from tldw_chatbook.Chat.console_context_repository import ConsoleMemoryRecord
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)


def format_context_tokens(value: int | None) -> str:
    """Format a token count without pretending an unknown value is zero."""
    return "unknown" if value is None else f"{value:,}"


@dataclass(frozen=True, slots=True)
class ConsoleContextControlState:
    """One coherent context-and-memory snapshot for the current conversation."""

    request_tokens: int | None
    conversation_tokens: int | None
    request_overhead_tokens: int | None
    model_window_tokens: int | None
    model_window_verified: bool
    model_window_source: str
    safe_input_ceiling_tokens: int | None
    response_max_tokens: int
    safety_margin_tokens: int | None
    resolved_policy: ResolvedConsoleContextPolicy
    inherited_policy: ConsoleContextPolicyDefaults
    overrides: ConsoleContextPolicyOverrides
    active_memory: ConsoleMemoryRecord | None = None
    busy: bool = False
    status_message: str = ""

    @property
    def conversation_budget_tokens(self) -> int | None:
        return self.resolved_policy.effective_conversation_budget_tokens

    @property
    def compaction_trigger_tokens(self) -> int | None:
        budget = self.conversation_budget_tokens
        if budget is None:
            return None
        return int(budget * self.resolved_policy.policy.trigger_ratio)

    @property
    def compaction_target_tokens(self) -> int | None:
        budget = self.conversation_budget_tokens
        if budget is None:
            return None
        return int(budget * self.resolved_policy.policy.target_ratio)

    @property
    def request_row(self) -> str:
        used = format_context_tokens(self.request_tokens)
        ceiling = format_context_tokens(self.safe_input_ceiling_tokens)
        if self.safe_input_ceiling_tokens is None:
            suffix = "limit unknown"
        elif self.model_window_verified:
            suffix = "safe input"
        else:
            suffix = "estimated input; model unverified"
        estimate_prefix = "~" if self.request_tokens is not None else ""
        return f"{estimate_prefix}{used} / {ceiling} {suffix}"

    @property
    def conversation_row(self) -> str:
        used = format_context_tokens(self.conversation_tokens)
        budget = format_context_tokens(self.conversation_budget_tokens)
        estimate_prefix = "~" if self.conversation_tokens is not None else ""
        return f"{estimate_prefix}{used} / {budget} max tokens"


def build_console_context_control_state(
    *,
    settings: ConsoleSessionSettings,
    estimate: ConsoleSettingsContextEstimate,
    overrides: ConsoleContextPolicyOverrides | None = None,
    global_overrides: ConsoleContextPolicyOverrides | None = None,
    active_memory: ConsoleMemoryRecord | None = None,
    conversation_tokens: int | None = None,
    request_overhead_tokens: int | None = None,
    provider_input_cap_tokens: int | None = None,
    safety_margin_tokens: int | None = None,
    busy: bool = False,
    status_message: str = "",
) -> ConsoleContextControlState:
    """Build a UI snapshot from the current estimate and owned policy values.

    The ordinary settings estimate does not yet expose a semantic overhead
    breakdown, so callers may provide exact conversation/overhead counts from
    a prepared request.  Until then, the honest fallback marks overhead as
    unknown and uses the displayed request estimate for conversation usage.
    """
    local_overrides = overrides or ConsoleContextPolicyOverrides()
    inherited = merge_context_policy(global_overrides=global_overrides)
    response_max = max(0, int(settings.max_tokens or 0))
    capacity = ConsoleContextCapacity(
        model_context_window_tokens=estimate.token_limit,
        provider_input_cap_tokens=provider_input_cap_tokens,
        response_reservation_tokens=response_max,
        safety_margin_tokens=max(0, int(safety_margin_tokens or 0)),
        mandatory_input_tokens=max(0, int(request_overhead_tokens or 0)),
    )
    resolved = resolve_context_policy(
        capacity=capacity,
        application_defaults=application_context_policy_defaults(),
        global_overrides=global_overrides,
        conversation_overrides=local_overrides,
    )
    used = estimate.used_tokens
    conversation_used = used if conversation_tokens is None else conversation_tokens
    model_window_verified = (
        estimate.token_limit is not None
        if estimate.token_limit_verified is None
        else estimate.token_limit_verified
    )
    return ConsoleContextControlState(
        request_tokens=used,
        conversation_tokens=conversation_used,
        request_overhead_tokens=request_overhead_tokens,
        model_window_tokens=estimate.token_limit,
        model_window_verified=model_window_verified,
        model_window_source=estimate.token_limit_source,
        safe_input_ceiling_tokens=resolved.safe_input_ceiling_tokens,
        response_max_tokens=response_max,
        safety_margin_tokens=safety_margin_tokens,
        resolved_policy=resolved,
        inherited_policy=inherited,
        overrides=local_overrides,
        active_memory=active_memory,
        busy=busy,
        status_message=status_message,
    )


def build_console_context_cost_state(
    context: ConsoleContextControlState,
    cost: ConsoleCostState,
) -> ConsoleCostState:
    """Add current-request context fullness to the existing spend chip."""
    used = context.request_tokens
    ceiling = context.safe_input_ceiling_tokens
    if used is None or ceiling is None or ceiling <= 0:
        fullness = "unknown"
        context_line = (
            "Context: model context window is unavailable; choose a model "
            "with a known window in Settings."
        )
    else:
        raw_percent = used * 100 / ceiling
        fullness = "100%+" if raw_percent > 100 else f"{round(raw_percent)}%"
        context_line = (
            f"Context: ~{format_context_tokens(used)} / "
            f"{format_context_tokens(ceiling)} safe input ({fullness} full)"
        )

    conversation = format_context_tokens(context.conversation_tokens)
    budget = format_context_tokens(context.conversation_budget_tokens)
    conversation_prefix = "~" if context.conversation_tokens is not None else ""
    conversation_line = (
        f"Conversation: {conversation_prefix}{conversation} / {budget} budget"
    )

    compaction_mode = context.resolved_policy.policy.compaction_mode
    trigger = context.compaction_trigger_tokens
    if compaction_mode is ContextCompactionMode.OFF:
        compaction_line = "Compaction: off."
    elif trigger is None:
        compaction_line = f"Compaction: {compaction_mode.value}; trigger unknown."
    elif compaction_mode is ContextCompactionMode.AUTOMATIC:
        compaction_line = (
            "Compaction: automatic at "
            f"{format_context_tokens(trigger)} conversation tokens."
        )
    else:
        compaction_line = (
            f"Compaction: asks at {format_context_tokens(trigger)} conversation tokens."
        )

    tooltip = "\n".join(
        (
            context_line,
            conversation_line,
            compaction_line,
            cost.tooltip,
            "Open Conversation Inspector for Costs, Exchange, and Next Send.",
        )
    )
    return replace(
        cost,
        label=f"Context {fullness} · {cost.label}",
        compact_label=f"Ctx {fullness} · {cost.compact_label}",
        tooltip=tooltip,
    )


__all__ = [
    "ConsoleContextControlState",
    "build_console_context_control_state",
    "build_console_context_cost_state",
    "format_context_tokens",
]
