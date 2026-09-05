"""Presentation contracts for Console conversation context controls.

The widgets consume this immutable snapshot instead of reading the database or
reconstructing provider requests.  That keeps policy ownership in the Console
store/controller while making the quick and full settings surfaces agree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_prepared_request import (
        ConsoleRequestTokenAccounting,
    )

from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    EffectiveMemoryResult,
)
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
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleMemoryRecord,
    ConsoleMemoryScopeRecord,
    MemoryCoverageKind,
    MemoryOriginKind,
)
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.cost_display import format_cost_amount
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
    #: TASK-26019: named category rows from the LAST prepared request's
    #: accounting; empty until a send has been prepared this session.
    breakdown_rows: tuple["ContextBreakdownRow", ...] = ()

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


@dataclass(frozen=True, slots=True)
class ConsoleNextSendSpendState:
    """Presentation state for incremental input cost on the next send."""

    label: str
    tooltip: str


def build_console_next_send_spend_state(
    *,
    request_tokens: int | None,
    input_per_mtok: float | None,
    sendable_text: bool,
    has_media: bool,
) -> ConsoleNextSendSpendState:
    """Build an honest uncached-input forecast for the next request.

    Args:
        request_tokens: Estimated next-request text tokens, if known.
        input_per_mtok: Uncached input price per million tokens, if known.
        sendable_text: Whether the validated draft contains text to send.
        has_media: Whether the request includes media with unestimated cost.

    Returns:
        Additional input charge or an explicit empty/unavailable state.
    """
    detail = (
        f"Estimated text input: ~{format_context_tokens(request_tokens)} tokens."
        if request_tokens is not None
        else None
    )
    if has_media:
        lines = ["On next send: unavailable", "Media cost is not estimated."]
        if detail is not None:
            lines.append(detail)
        return ConsoleNextSendSpendState("unavailable", "\n".join(lines))
    if request_tokens is None:
        return ConsoleNextSendSpendState(
            "unavailable",
            "On next send: unavailable\n"
            "The next-request text token estimate is unavailable.",
        )
    if input_per_mtok is None:
        return ConsoleNextSendSpendState(
            "unavailable",
            "On next send: unavailable\n"
            "Selected-model input pricing is unavailable.\n"
            f"{detail}",
        )
    if not sendable_text:
        return ConsoleNextSendSpendState(
            "—",
            "On next send: —\nType a message to estimate the next-send input spend.",
        )
    amount = round(request_tokens * input_per_mtok / 1_000_000, 6)
    label = f"~+${format_cost_amount(amount)}"
    return ConsoleNextSendSpendState(
        label,
        f"On next send: {label} uncached input baseline\n"
        f"{detail}\n"
        "Response/output spend is added after completion.\n"
        "Cache reads may lower it; cache writes may raise it.",
    )


@dataclass(frozen=True)
class ContextBreakdownRow:
    """One named slice of the request window (TASK-26019)."""

    label: str
    tokens: int
    #: The action that shrinks this bucket, when one exists (AC#6).
    hint: str = ""


def build_context_breakdown(
    accounting: "ConsoleRequestTokenAccounting",
) -> tuple[ContextBreakdownRow, ...]:
    """Partition the request accounting into named categories (TASK-26019).

    Every figure is a difference of the SAME cumulative wire counts that
    built the request (AC#2) and the rows sum exactly to
    ``total_input_tokens``; nothing is re-estimated here. Zero rows are
    dropped for scannability -- except the residual, which appears whenever
    it is non-zero because hiding it would be silent folding (AC#3).
    """
    attachments = accounting.attachment_tokens
    conversation = max(
        0,
        accounting.compactable_tokens + accounting.active_request_tokens - attachments,
    )
    rows: list[ContextBreakdownRow] = [
        ContextBreakdownRow("System prompt", accounting.system_tokens),
        ContextBreakdownRow(
            "Tool schemas",
            accounting.tool_schema_tokens,
            hint="Disable unused tools in Settings",
        ),
        ContextBreakdownRow(
            "Memory summary",
            accounting.memory_tokens,
            hint="Adjust compaction_summary_max_tokens",
        ),
    ]
    if accounting.rag_attributed:
        rows.append(
            ContextBreakdownRow(
                "Retrieved context",
                accounting.rag_context_tokens,
                hint="Narrow RAG scope or sources",
            )
        )
        rows.append(
            ContextBreakdownRow(
                "Other instructions",
                max(
                    0,
                    accounting.mandatory_tokens - accounting.rag_context_tokens,
                ),
            )
        )
    else:
        # AC#3: capture is off, so RAG vs other instructions is unknowable
        # -- say so instead of folding it into a named bucket.
        rows.append(
            ContextBreakdownRow(
                "Instructions & context (unattributed)",
                accounting.mandatory_tokens,
                hint="Enable capture to attribute retrieved context",
            )
        )
    rows.append(
        ContextBreakdownRow(
            "Attachments",
            attachments,
            hint="Enable [agents] retire_stale_images to reclaim old images",
        )
    )
    rows.append(
        ContextBreakdownRow(
            "Conversation",
            conversation,
            hint="Summarize older turns (/rewind \u25b8 Summarize)",
        )
    )
    residual = accounting.total_input_tokens - sum(row.tokens for row in rows)
    if residual > 0:
        rows.append(ContextBreakdownRow("Unattributed", residual))
    return tuple(row for row in rows if row.tokens > 0)


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
    accounting: "ConsoleRequestTokenAccounting | None" = None,
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
        breakdown_rows=(
            build_context_breakdown(accounting) if accounting is not None else ()
        ),
    )


def build_console_context_cost_state(
    context: ConsoleContextControlState,
    cost: ConsoleCostState,
    next_send: ConsoleNextSendSpendState | None = None,
) -> ConsoleCostState:
    """Combine context fullness with current and next-send spend states.

    Args:
        context: Estimated request occupancy and model capacity.
        cost: Current settled spend and cache presentation.
        next_send: Additional input-charge forecast, if available.

    Returns:
        Full and compact labels with detailed context and spend tooltips.
    """
    next_send = next_send or ConsoleNextSendSpendState(
        "unavailable",
        "On next send: unavailable\nA next-send spend estimate has not been provided.",
    )
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
    prefix = "~" if context.conversation_tokens is not None else ""
    compaction_mode = context.resolved_policy.policy.compaction_mode
    trigger = context.compaction_trigger_tokens
    if compaction_mode is ContextCompactionMode.OFF:
        compaction_line = "Compaction: off."
    elif trigger is None:
        compaction_line = f"Compaction: {compaction_mode.value}; trigger unknown."
    elif compaction_mode is ContextCompactionMode.AUTOMATIC:
        compaction_line = (
            f"Compaction: automatic at {format_context_tokens(trigger)} "
            "conversation tokens."
        )
    else:
        compaction_line = (
            f"Compaction: asks at {format_context_tokens(trigger)} conversation tokens."
        )
    current_label = cost.compact_label
    for suffix in (" ⚠", " ○", " ●"):
        if current_label.endswith(suffix):
            current_label = current_label.removesuffix(suffix)
            break
    return replace(
        cost,
        label=(
            f"Context {fullness} · Current {current_label} "
            f"· On next send {next_send.label}"
        ),
        compact_label=(
            f"Ctx {fullness} · Now {current_label} · Next {next_send.label}"
        ),
        tooltip="\n".join(
            (
                context_line,
                f"Conversation: {prefix}{conversation} / {budget} budget",
                compaction_line,
                cost.tooltip,
                next_send.tooltip,
                "Open Conversation Inspector for Costs, Exchange, and Next Send.",
            )
        ),
    )


__all__ = [
    "ConsoleContextControlState",
    "ConsoleNextSendSpendState",
    "ThinkingHistoryControlState",
    "build_console_context_control_state",
    "build_console_context_cost_state",
    "build_console_next_send_spend_state",
    "format_context_tokens",
]
