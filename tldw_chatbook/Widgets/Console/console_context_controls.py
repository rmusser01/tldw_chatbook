"""Presentation contracts for Console conversation context controls.

The widgets consume this immutable snapshot instead of reading the database or
reconstructing provider requests.  That keeps policy ownership in the Console
store/controller while making the quick and full settings surfaces agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    EffectiveMemoryResult,
)
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextCapacity,
    ConsoleContextPolicyDefaults,
    ConsoleContextPolicyOverrides,
    ResolvedConsoleContextPolicy,
    application_context_policy_defaults,
    merge_context_policy,
    resolve_context_policy,
)
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleMemoryRecord,
    ConsoleMemoryScopeRecord,
    MemoryCoverageKind,
    MemoryOriginKind,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Chat.console_thinking_history import (
    EffectiveThinkingHistoryPolicy,
)
from tldw_chatbook.Chat.thinking_blocks import (
    ThinkingHistoryPolicy,
    normalize_thinking_history_policy,
)


_REQUIRED_THINKING_HISTORY_REASON = (
    "Completed provider continuation history must be replayed for this model."
)


def format_context_tokens(value: int | None) -> str:
    """Format a token count without pretending an unknown value is zero."""
    return "unknown" if value is None else f"{value:,}"


@dataclass(frozen=True, slots=True)
class ThinkingHistoryControlState:
    """Saved optional replay preference and its current effective state."""

    saved_policy: ThinkingHistoryPolicy
    effective_label: Literal["Auto", "Include", "Exclude", "Required"]
    required_reason: str | None = None


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
    effective_memory: EffectiveMemoryResult = EffectiveMemoryResult(
        EffectiveMemoryKind.RAW
    )
    busy: bool = False
    status_message: str = ""
    thinking_history: ThinkingHistoryControlState = ThinkingHistoryControlState(
        "auto", "Auto"
    )

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
    effective_memory: EffectiveMemoryResult | None = None,
    active_memory: ConsoleMemoryRecord | EffectiveMemoryResult | None = None,
    conversation_tokens: int | None = None,
    request_overhead_tokens: int | None = None,
    provider_input_cap_tokens: int | None = None,
    safety_margin_tokens: int | None = None,
    busy: bool = False,
    status_message: str = "",
    thinking_history_policy: object = None,
    thinking_history_effective_policy: EffectiveThinkingHistoryPolicy | None = None,
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
    saved_thinking_policy = normalize_thinking_history_policy(thinking_history_policy)
    effective_thinking_policy = (
        thinking_history_effective_policy or saved_thinking_policy
    )
    effective_thinking_label = effective_thinking_policy.title()
    required_reason = (
        _REQUIRED_THINKING_HISTORY_REASON
        if effective_thinking_policy == "required"
        else None
    )
    typed_memory = effective_memory
    if typed_memory is None and isinstance(active_memory, EffectiveMemoryResult):
        typed_memory = active_memory
    elif typed_memory is None and active_memory is not None:
        typed_memory = EffectiveMemoryResult(
            EffectiveMemoryKind.GENERATED_PREFIX,
            memory=active_memory,
            scope=ConsoleMemoryScopeRecord(
                memory_id=active_memory.memory_id,
                conversation_id=active_memory.conversation_id,
                coverage_kind=MemoryCoverageKind.PREFIX,
                origin_kind=MemoryOriginKind.AUTOMATIC,
                selection_anchor_message_id=None,
            ),
        )
    if typed_memory is None:
        typed_memory = EffectiveMemoryResult(EffectiveMemoryKind.RAW)
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
        effective_memory=typed_memory,
        busy=busy,
        status_message=status_message,
        thinking_history=ThinkingHistoryControlState(
            saved_policy=saved_thinking_policy,
            effective_label=effective_thinking_label,
            required_reason=required_reason,
        ),
    )


__all__ = [
    "ConsoleContextControlState",
    "ThinkingHistoryControlState",
    "build_console_context_control_state",
    "format_context_tokens",
]
