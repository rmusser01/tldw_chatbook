"""Coordination and formatting for Console's next-send price preview."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tldw_chatbook.Chat.console_chat_models import ConsoleNextSendHistoryProjection
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_cost_tracker import (
    TokenEstimateCache,
    token_estimate_signature,
)
from tldw_chatbook.Chat.console_display_state import console_prompted_evidence_text
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    _estimate_tokens_locally,
)
from tldw_chatbook.Chat.cost_display import format_cost_amount
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.LLM_Calls.pricing_catalog import (
    ModelPricing,
    PricingCatalog,
    get_pricing_catalog,
)
from tldw_chatbook.Utils.input_validation import validate_console_draft


@dataclass(frozen=True, slots=True)
class ConsoleNextSendPrice:
    """The fully rendered next-request price tooltip."""

    tooltip: str


#: The honest degraded copy. Pricing must never block Send, so every failure
#: below renders this instead of raising.
UNAVAILABLE_PRICE = ConsoleNextSendPrice("Next request: cost unavailable")


@dataclass(frozen=True, slots=True)
class _DraftPriceContext:
    """The cheap half of the price decision (TASK-23018).

    Everything that decides *whether* a price tooltip exists at all, settled
    without projecting the session transcript or counting a single token.
    ``ConsoleSendPriceController._resolve_context`` returns one of these when
    -- and only when -- the expensive half is worth running, so
    :meth:`ConsoleSendPriceController.availability_for_draft` can answer the
    Send button's label question on the keystroke path while the rendered
    tooltip is derived on demand.
    """

    session_id: str
    validated_draft: str
    has_draft: bool
    attachment_count: int


class ConsoleSendPriceController:
    """Read-only coordinator for the next Console request price preview."""

    def __init__(
        self,
        *,
        settings_accessor: Callable[[], ConsoleSessionSettings],
        chat_store_accessor: Callable[[], ConsoleChatStore | None],
        provider_history_accessor: Callable[[str], ConsoleNextSendHistoryProjection],
        pending_launch_accessor: Callable[[], ConsoleLiveWorkLaunch | None],
        pricing_catalog_accessor: Callable[[], PricingCatalog] = get_pricing_catalog,
        token_counter: Callable[
            [list[dict[str, str]], str, str], int
        ] = _estimate_tokens_locally,
    ) -> None:
        self._settings_accessor = settings_accessor
        self._chat_store_accessor = chat_store_accessor
        self._provider_history_accessor = provider_history_accessor
        self._pending_launch_accessor = pending_launch_accessor
        self._pricing_catalog_accessor = pricing_catalog_accessor
        self._token_counter = token_counter
        self._token_cache = TokenEstimateCache(max_entries=1)

    def _resolve_context(
        self, draft_text: str
    ) -> "_DraftPriceContext | ConsoleNextSendPrice | None":
        """Settle the availability question without touching the transcript.

        Args:
            draft_text: Current composer text before send validation.

        Returns:
            A :class:`_DraftPriceContext` when a real estimate should be
            derived, or the final answer already (the degraded copy, or
            ``None`` when there is nothing to price).

        Raises:
            Exception: Whatever the store accessor raises; both public
                entry points below own that translation.
        """
        store = self._chat_store_accessor()
        session_id = store.active_session_id if store is not None else None
        raw_has_draft = bool(str(draft_text or "").strip())
        validated_draft, validation_error = validate_console_draft(
            draft_text,
            allow_empty=True,
        )
        if validation_error is not None:
            return UNAVAILABLE_PRICE if raw_has_draft else None
        has_draft = bool(validated_draft.strip())
        if store is None or session_id is None:
            return UNAVAILABLE_PRICE if has_draft else None
        try:
            attachment_count = len(store.pending_attachments(session_id))
        except KeyError:
            return UNAVAILABLE_PRICE if has_draft else None
        if not has_draft and attachment_count == 0:
            return None
        return _DraftPriceContext(
            session_id=session_id,
            validated_draft=validated_draft,
            has_draft=has_draft,
            attachment_count=attachment_count,
        )

    def availability_for_draft(self, draft_text: str) -> bool:
        """Return whether a price tooltip exists for ``draft_text``.

        Answers exactly ``presentation_for_draft(draft_text) is not None``
        -- both route through :meth:`_resolve_context`, so the two cannot
        drift -- but without the whole-session history projection or any
        token counting. TASK-23018: this is the only price question the
        composer is allowed to ask per keystroke, because the Send button's
        ``| $`` label suffix depends on it; the rendered tooltip itself is
        derived on demand when the pointer actually reaches Send.

        Args:
            draft_text: Current composer text before send validation.

        Returns:
            True when hovering Send would show a price tooltip.
        """
        try:
            resolved = self._resolve_context(draft_text)
        except Exception:  # noqa: BLE001 -- mirrors presentation_for_draft
            # presentation_for_draft renders the degraded copy for this, so a
            # tooltip does exist.
            return True
        return resolved is not None

    def presentation_for_draft(self, draft_text: str) -> ConsoleNextSendPrice | None:
        """Return a best-effort next-request estimate for the current draft.

        Expensive: projects the whole session's provider history and counts
        its tokens. Never call this on the keystroke path -- see
        :meth:`availability_for_draft`.

        Args:
            draft_text: Current composer text before send validation.

        Returns:
            The price preview, or ``None`` when there is nothing to send.
        """
        try:
            resolved = self._resolve_context(draft_text)
            if not isinstance(resolved, _DraftPriceContext):
                return resolved

            settings = self._settings_accessor()
            provider = settings.provider
            model = settings.model or ""
            normalized_provider = provider_config_key(provider)
            session_id = resolved.session_id
            try:
                projection = self._provider_history_accessor(session_id)
            except KeyError:
                # `_resolve_context` already decided a tooltip is warranted,
                # so a projection failure degrades to the honest copy rather
                # than silently withdrawing the tooltip the `| $` label
                # suffix has already promised.
                return UNAVAILABLE_PRICE

            row_pairs = list(projection.rows)
            if resolved.has_draft:
                row_pairs.append(("user", resolved.validated_draft))
            staged_text = console_prompted_evidence_text(
                self._pending_launch_accessor()
            )
            if staged_text.strip():
                row_pairs.append(("user", staged_text))
            signature = token_estimate_signature(
                row_pairs,
                model,
                normalized_provider,
            )
            try:
                input_tokens = self._token_cache.estimate(
                    session_id,
                    signature,
                    # Built inside the miss callback: the counter rows are an
                    # O(session) list of dicts that a cache HIT never reads,
                    # and eagerly building them made every hit cost more than
                    # it saved (TASK-23018).
                    lambda: self._token_counter(
                        [
                            {"role": role, "content": content}
                            for role, content in row_pairs
                        ],
                        model,
                        normalized_provider,
                    ),
                )
            except Exception:
                input_tokens = None

            pricing = self._pricing_catalog_accessor().get_pricing(provider, model)
            return build_next_send_price(
                input_tokens=input_tokens,
                max_reply_tokens=settings.max_tokens,
                pricing=pricing,
                provider=provider,
                model=model,
                attachment_count=resolved.attachment_count,
                historical_media_count=projection.historical_media_count,
            )
        except Exception:
            return UNAVAILABLE_PRICE

    def tooltip_for_draft(self, draft_text: str) -> str | None:
        """Return only the rendered tooltip for widget consumption.

        Args:
            draft_text: Current composer text before send validation.

        Returns:
            The rendered tooltip, or ``None`` when there is nothing to send.
        """
        presentation = self.presentation_for_draft(draft_text)
        return presentation.tooltip if presentation is not None else None


def _format_input_line(
    tokens: int | None, pricing: ModelPricing | None, label: str
) -> str:
    if tokens is None:
        return f"{label}: token estimate unavailable"

    token_label = "token" if tokens == 1 else "tokens"
    line = f"{label}: ~{tokens:,} {token_label}"
    if pricing is not None:
        cost = round(tokens * pricing.input_per_mtok / 1_000_000, 6)
        line += f" · ~${format_cost_amount(cost)}"
    return line


def _format_reply_line(
    tokens: int | None, pricing: ModelPricing | None, label: str
) -> str:
    if tokens is None:
        return f"{label}: limit not configured"

    token_label = "token" if tokens == 1 else "tokens"
    line = f"{label}: up to {tokens:,} {token_label}"
    if pricing is not None:
        cost = round(tokens * pricing.output_per_mtok / 1_000_000, 6)
        line += f" · ~${format_cost_amount(cost)}"
    return line


def _format_provenance(provider: str, model: str, pricing: ModelPricing | None) -> str:
    identifiers = [
        identifier.strip() for identifier in (provider, model) if identifier.strip()
    ]
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
    """Build an honest, text-only estimate for the next Console request.

    Args:
        input_tokens: Estimated provider input-token count, if available.
        max_reply_tokens: Configured reply-token ceiling, if available.
        pricing: Matching model price rates, if configured.
        provider: Selected provider name for provenance.
        model: Selected model name for provenance.
        attachment_count: Number of pending media attachments.
        historical_media_count: Number of admitted historical media items.

    Returns:
        A rendered price preview with cost, token, media, and provenance detail.
    """
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
        total_line = "Next request: cost unavailable"

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
