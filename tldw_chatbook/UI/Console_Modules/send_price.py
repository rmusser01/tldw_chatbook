"""Pure formatting for Console's next-send price preview."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_chatbook.Chat.cost_display import format_cost_amount
from tldw_chatbook.LLM_Calls.pricing_catalog import ModelPricing


@dataclass(frozen=True, slots=True)
class ConsoleNextSendPrice:
    """The fully rendered next-request price tooltip."""

    tooltip: str


def _format_input_line(
    tokens: int | None, pricing: ModelPricing | None, label: str
) -> str:
    if tokens is None:
        return f"{label}: token estimate unavailable"

    line = f"{label}: ~{tokens:,} tokens"
    if pricing is not None:
        cost = round(tokens * pricing.input_per_mtok / 1_000_000, 6)
        line += f" · ~${format_cost_amount(cost)}"
    return line


def _format_reply_line(
    tokens: int | None, pricing: ModelPricing | None, label: str
) -> str:
    if tokens is None:
        return f"{label}: limit not configured"

    line = f"{label}: up to {tokens:,} tokens"
    if pricing is not None:
        cost = round(tokens * pricing.output_per_mtok / 1_000_000, 6)
        line += f" · ~${format_cost_amount(cost)}"
    return line


def _format_provenance(provider: str, model: str, pricing: ModelPricing | None) -> str:
    identifiers = [identifier for identifier in (provider, model) if identifier.strip()]
    pricing_label = (
        f"rates as of {pricing.as_of}"
        if pricing is not None
        else "pricing not configured"
    )
    return " · ".join([*identifiers, pricing_label])


def build_next_send_price(
    *,
    input_tokens: int | None,
    max_reply_tokens: int | None,
    pricing: ModelPricing | None,
    provider: str,
    model: str,
    attachment_count: int = 0,
    historical_media_count: int = 0,
) -> ConsoleNextSendPrice:
    """Build an honest, text-only estimate for the next Console request."""
    has_media = attachment_count > 0 or historical_media_count > 0
    input_label = "Input text" if has_media else "Input"
    reply_label = "Reply text" if has_media else "Reply"
    total_available = (
        pricing is not None
        and input_tokens is not None
        and max_reply_tokens is not None
        and not has_media
    )

    if total_available:
        input_cost = round(input_tokens * pricing.input_per_mtok / 1_000_000, 6)
        reply_cost = round(max_reply_tokens * pricing.output_per_mtok / 1_000_000, 6)
        total_line = f"Next request: up to ~${format_cost_amount(round(input_cost + reply_cost, 6))}"
    else:
        total_line = "Next request: total unavailable"

    lines = [
        total_line,
        _format_input_line(input_tokens, pricing, input_label),
        _format_reply_line(max_reply_tokens, pricing, reply_label),
    ]
    if attachment_count > 0:
        lines.append(f"Attachments: {attachment_count} · media cost not estimated")
    if historical_media_count > 0:
        item_label = "item" if historical_media_count == 1 else "items"
        lines.append(
            f"Media context: {historical_media_count} {item_label} · media cost not estimated"
        )
    lines.append(_format_provenance(provider, model, pricing))
    return ConsoleNextSendPrice(tooltip="\n".join(lines))
