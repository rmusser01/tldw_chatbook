"""Presentation contracts for Console conversation context controls.

The widgets consume this immutable snapshot instead of reading the database or
reconstructing provider requests.  That keeps policy ownership in the Console
store/controller while making the quick and full settings surfaces agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage
    from tldw_chatbook.Chat.console_prepared_request import (
        ConsoleRequestTokenAccounting,
    )

from tldw_chatbook.Chat.console_context_compaction import (
    CompactionDecision,
    EffectiveMemoryKind,
    EffectiveMemoryResult,
    decide_compaction,
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
from tldw_chatbook.Chat.console_prepared_request import (
    resolve_request_capacity,
    tagged_memory_message,
)
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


#: TASK-33621.4 AC#3: the header badge and the Context chip's lead, once the
#: chat is over its Ask trigger (Inspect's Run line spells it out in full).
COMPACTION_PAUSE_FORECAST_COPY = "Next send pauses to compact"


@dataclass(frozen=True, slots=True)
class CompactionPauseForecast:
    """The numbers behind a forecast that the next send will pause (Ask)."""

    used_tokens: int
    trigger_tokens: int
    budget_tokens: int


def compaction_forecast_rows(
    messages: Sequence["ConsoleChatMessage"],
    effective_memory: EffectiveMemoryResult,
) -> tuple[tuple[dict[str, str], ...], int] | None:
    """Rows the Ask trigger counts for the next send, and its complete turns.

    TASK-33621.4 AC#3. Mirrors the send preflight: conversation tokens are the
    memory text plus the turns after its boundary; transcript notices and
    unfinished replies are not history, and the draft is the next request,
    not history. The preflight never asks under a legacy prefix and plans a
    range differently, so those shapes (and a boundary that is not on this
    path) forecast nothing rather than guess.

    Args:
        messages: The active path's admitted transcript rows, oldest first.
        effective_memory: The memory the next send would use.

    Returns:
        ``(rows, complete_turns)`` to count, or None when not modelled.
    """
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    rows: list[dict[str, str]] = []
    start = 0
    if effective_memory.kind is EffectiveMemoryKind.GENERATED_PREFIX:
        memory = effective_memory.memory
        boundary = memory.boundary_message_id if memory is not None else None
        start = next(
            (
                index + 1
                for index, message in enumerate(messages)
                if boundary and message.persisted_message_id == boundary
            ),
            -1,
        )
        if memory is None or start < 0:
            return None
        # Wrapped as the send carries it: the wrapper's safety copy counts.
        rows.append(dict(tagged_memory_message(memory.summary_text)))
    elif effective_memory.kind is not EffectiveMemoryKind.RAW:
        return None
    turns = 0
    for message in messages[start:]:
        if message.role is ConsoleMessageRole.SYSTEM or (
            message.role is ConsoleMessageRole.ASSISTANT
            and message.status != "complete"
        ):
            continue
        rows.append({"role": message.role.value, "content": message.content})
        turns += message.role is ConsoleMessageRole.ASSISTANT
    return tuple(rows), turns


_REQUIRED_THINKING_HISTORY_REASON = (
    "Completed provider continuation history must be replayed for this model."
)


def format_context_tokens(value: int | None) -> str:
    """Format a token count without pretending an unknown value is zero."""
    return "unknown" if value is None else f"{value:,}"


def context_token_row(label: str, value: int | None) -> str:
    """Return one Context view row: the label in 20 cells, then the count.

    Args:
        label: The row's label, e.g. ``"Safety margin"``.
        value: The count, or None when nothing measured it.

    Returns:
        For example ``"Safety margin       640 tokens"``; an unknown count
        reads ``"unknown"``, never "unknown tokens" (TASK-33007 capture
        finding 17).
    """
    count = "unknown" if value is None else f"{value:,} tokens"
    return f"{label:<20}{count}"


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
    system_prompt_delta_tokens: int = 0
    #: TASK-33621.4: the conversation's own tokens and complete turns as the
    #: next send's preflight counts them (``compaction_forecast_rows``);
    #: None when not measured.
    forecast_tokens: int | None = None
    forecast_units: int = 0

    @property
    def compaction_pause_forecast(self) -> CompactionPauseForecast | None:
        """The numbers when the next send will pause for compaction (Ask).

        The same ``decide_compaction`` the send preflight runs, so this
        cannot disagree with the hold about whether a send pauses.
        """
        budget = self.conversation_budget_tokens
        trigger = self.compaction_trigger_tokens
        if self.forecast_tokens is None or budget is None or trigger is None:
            return None
        decision = decide_compaction(
            self.resolved_policy,
            conversation_tokens=self.forecast_tokens,
            compactable_units=self.forecast_units,
        )
        if decision is not CompactionDecision.ASK:
            return None
        return CompactionPauseForecast(self.forecast_tokens, trigger, budget)

    @property
    def conversation_budget_tokens(self) -> int | None:
        return self.resolved_policy.effective_conversation_budget_tokens

    @property
    def tool_schema_tokens(self) -> int:
        """The last prepared request's tool schemas (0 before any send).

        TASK-34100.5.3 AC#5: read from the breakdown rows, which every
        rebuild of this state carries over.
        """
        return next(
            (row.tokens for row in self.breakdown_rows if row.label == "Tool schemas"),
            0,
        )

    @property
    def last_request_overhead_tokens(self) -> int | None:
        """The last prepared request's non-conversation tokens, if known.

        An explicit ``request_overhead_tokens`` wins; otherwise the system
        prompt, tool schemas and instruction rows of the last request's
        breakdown, or ``None`` before any send.
        """
        if self.request_overhead_tokens is not None:
            return self.request_overhead_tokens
        rows = [row for row in self.breakdown_rows if row.label in _OVERHEAD_ROW_LABELS]
        return sum(row.tokens for row in rows) if rows else None

    @property
    def next_request_tokens(self) -> int | None:
        """Current inputs plus remembered tool/prompt overhead, never the old total."""
        if self.request_tokens is None:
            return None
        return max(
            0,
            self.request_tokens
            + self.tool_schema_tokens
            + self.system_prompt_delta_tokens,
        )

    def overhead_label(self) -> str:
        """The "Request overhead" value, naming the tool schemas' share.

        TASK-34100.5.3 AC#5: the overhead comes from the last prepared
        request, whose accounting now counts the tools the send carries.
        Display only: the conversation budget does not subtract it (review
        round 1, F6 -- the Conversation figure already counts the system
        prompt). An unmeasured overhead reads "unknown", never "unknown
        tokens", as ``context_token_row`` does (TASK-33007).
        """
        overhead = self.last_request_overhead_tokens
        if overhead is None:
            return "unknown"
        text = f"{format_context_tokens(overhead)} tokens"
        tools = self.tool_schema_tokens
        return (
            f"{text} (tool schemas {format_context_tokens(tools)})" if tools else text
        )

    def request_estimate_label(self, estimate: ConsoleSettingsContextEstimate) -> str:
        """Keep the current estimate and qualify its remembered overhead."""
        label = estimate.label.strip() or "Not estimated"
        used, limit = estimate.used_tokens, estimate.token_limit
        head = f"{used:,} / {limit:,} tokens" if used is not None and limit else ""
        if not self.breakdown_rows or not head or not label.startswith(head):
            return label
        tools = self.tool_schema_tokens
        tool_note = f" (tool schemas {tools:,})" if tools else ""
        return (
            f"{self.next_request_tokens:,} / {limit:,} tokens{tool_note}"
            f"{label[len(head):]} (remembered overhead estimate)"
        )

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


#: The breakdown rows that are not conversation -- system prompt, tool schemas
#: and instructions -- which "Request overhead" sums (TASK-34100.5.3).
_OVERHEAD_ROW_LABELS = frozenset(
    {
        "System prompt",
        "Tool schemas",
        "Retrieved context",
        "Other instructions",
        "Instructions & context (unattributed)",
    }
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
    forecast_tokens: int | None = None,
    forecast_units: int = 0,
) -> ConsoleContextControlState:
    """Build a UI snapshot from the current estimate and owned policy values.

    The ordinary settings estimate does not yet expose a semantic overhead
    breakdown, so callers may provide exact conversation/overhead counts from
    a prepared request.  Until then, the honest fallback marks overhead as
    unknown and uses the displayed request estimate for conversation usage.
    """
    local_overrides = overrides or ConsoleContextPolicyOverrides()
    inherited = merge_context_policy(global_overrides=global_overrides)
    request_capacity = resolve_request_capacity(
        context_window_tokens=estimate.token_limit,
        requested_response_tokens=settings.max_tokens,
        context_window_verified=estimate.token_limit_verified is not False,
    )
    response_max = request_capacity.effective_response_tokens
    if safety_margin_tokens is None:
        safety_margin_tokens = request_capacity.safety_margin_tokens
    capacity = ConsoleContextCapacity(
        model_context_window_tokens=estimate.token_limit,
        model_window_verified=estimate.token_limit_verified is not False,
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
        system_prompt_delta_tokens=(
            accounting.system_prompt_delta_tokens if accounting is not None else 0
        ),
        forecast_tokens=forecast_tokens,
        forecast_units=forecast_units,
    )


def build_console_context_cost_state(
    context: ConsoleContextControlState,
    cost: ConsoleCostState,
    next_send: ConsoleNextSendSpendState | None = None,
    rate_limit_line: str | None = None,
) -> ConsoleCostState:
    """Combine context fullness with current and next-send spend states.

    Args:
        context: Estimated request occupancy and model capacity.
        cost: Current settled spend and cache presentation.
        next_send: Additional input-charge forecast, if available.
        rate_limit_line: The provider's remaining rate limit from its last
            response, shown last among the info lines; ``None`` when the
            provider reported none (TASK-28229).

    Returns:
        Full and compact labels with detailed context and spend tooltips.
    """
    next_send = next_send or ConsoleNextSendSpendState(
        "unavailable",
        "On next send: unavailable\nA next-send spend estimate has not been provided.",
    )
    # TASK-34100.5.3 review round 1 (B-F1): the tools the send carries count.
    used = context.next_request_tokens
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
        if context.breakdown_rows:
            context_line += " (remembered overhead estimate)"
        if (
            context.model_window_tokens is not None
            and not context.model_window_verified
        ):
            # TASK-33007 #12: a fallback window is unknown, its size assumed.
            context_line += " (assumed; window unknown)"
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
    # TASK-34100.5 AC#9: an unknown window only matters when it blocks a
    # send, and that refusal names the fix; the chip drops the segment.
    context_part = "" if fullness == "unknown" else f"Context {fullness} · "
    compact_part = "" if fullness == "unknown" else f"Ctx {fullness} · "
    # TASK-33621.4 AC#3: lead with it, so a clipped strip still shows it.
    forecast = getattr(context, "compaction_pause_forecast", None)
    if forecast is not None:
        context_part = f"{COMPACTION_PAUSE_FORECAST_COPY} · {context_part}"
        compact_part = f"Pauses to compact · {compact_part}"
        compaction_line += (
            " Next send pauses for compaction approval: "
            f"~{format_context_tokens(forecast.used_tokens)} of "
            f"{format_context_tokens(forecast.budget_tokens)} conversation tokens."
        )
    return replace(
        cost,
        label=(
            f"{context_part}Current {current_label} "
            f"· On next send {next_send.label}"
        ),
        compact_label=(f"{compact_part}Now {current_label} · Next {next_send.label}"),
        tooltip="\n".join(
            (
                context_line,
                f"Conversation: {prefix}{conversation} / {budget} budget",
                compaction_line,
                cost.tooltip,
                next_send.tooltip,
                *((rate_limit_line,) if rate_limit_line else ()),
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
    "context_token_row",
    "format_context_tokens",
]
