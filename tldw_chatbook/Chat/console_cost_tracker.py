# tldw_chatbook/Chat/console_cost_tracker.py
# Description: Cost math + chip-state formatting for the Console cost chip (PR3).
#
"""Cost math + chip-state formatting for the Console cost chip.

House pattern (see ``ConsoleControlState`` in ``console_display_state.py``):
the state dataclass owns ALL label/tooltip formatting, and a widget only
renders whatever string it is handed. Dollars are never stored -- this
module recomputes a session's cost at read time from ``ProviderUsage`` rows
via the pricing catalog (:mod:`tldw_chatbook.LLM_Calls.pricing_catalog`),
falling back to a local token estimate for rows that have no usage yet.

Two entry points:
    ``build_cost_snapshot``: sums the raw dollar/token totals for a
        sequence of transcript rows.
    ``build_cost_state``: turns a snapshot plus cache/TTL/alert context
        into pre-formatted chip text (full label, compact label, tooltip)
        and the boolean/enum flags a widget uses to pick CSS classes.

Both functions are defensive: any unexpected failure is logged and degrades
to a safe fallback value rather than raising, mirroring
``build_console_context_estimate`` in ``console_session_settings.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from loguru import logger

from tldw_chatbook.Chat.console_session_settings import _estimate_tokens_locally
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog

#
#######################################################################################################################
#
# Enums / dataclasses
#


class ConsoleCacheState(str, Enum):
    """Prompt-cache state for the active Console session, as seen by the chip."""

    NONE = "none"
    WARM = "warm"
    EXPIRED = "expired"


@dataclass(frozen=True)
class ConsoleCostSnapshot:
    """Rolled-up dollar/token totals for a Console session's transcript.

    Attributes:
        total_usd: Summed dollar cost across every priced row, or ``None``
            when there were no rows to price or pricing was unknown for at
            least one of them (never a partial/fabricated total).
        total_tokens: Summed token count across every row, from actual
            ``ProviderUsage`` where available and from the local estimator
            otherwise. Always known, even when ``total_usd`` is not.
        pricing_known: True when every row that contributed to
            ``total_tokens`` also resolved to a known per-model rate, so
            ``total_usd`` reflects the whole transcript rather than a
            partial sum.
        has_estimated_entries: True when at least one row had no recorded
            ``ProviderUsage`` and its token/dollar contribution came from
            the local character-ratio estimator instead.
        row_count: Number of transcript rows that contributed to the
            totals above (priced or estimated); rows with neither usage
            nor content are not counted.
    """

    total_usd: Optional[float]
    total_tokens: int
    pricing_known: bool
    has_estimated_entries: bool
    row_count: int


@dataclass(frozen=True)
class ConsoleCostState:
    """Pre-formatted chip text for the Console cost chip.

    Every string here is display-ready; the widget only renders it (see the
    module docstring's house-pattern note). ``label`` is the full chip
    text, ``compact_label`` is the same information with the projected-
    delta suffix dropped for narrow layouts, and ``tooltip`` is the
    multi-line hover text.

    Attributes:
        label: Full chip text, e.g. ``"$0.48 ⚠ ~+$0.13"``.
        compact_label: Narrow-strip fallback -- same as ``label`` but never
            carries the projected-delta suffix.
        tooltip: Multi-line hover text with the total, token count, cache
            state, and pricing provenance.
        alert: True when the prompt cache is warm and about to break for a
            known reason -- the only condition under which the chip shows
            the alert glyph and projected delta.
        cold: True when the prompt cache has expired.
    """

    label: str
    compact_label: str
    tooltip: str
    alert: bool
    cold: bool


@dataclass(frozen=True)
class PayloadFingerprint:
    """A digest of one provider payload's shape, for cache-break detection.

    Recorded at dispatch time (the baseline: what was actually sent) and
    recomputed on demand (the current: what would be sent right now) so
    :func:`fingerprint_break_reason` can tell the cost chip WHY a warm
    Anthropic prompt cache is about to miss its shared prefix -- a provider/
    model swap, an edited system prompt, or edited/truncated earlier
    history. Appending new turns to the tail is the normal case and is
    never a break (see :func:`fingerprint_break_reason`).

    Attributes:
        provider_model: Digest of ``(provider, model)``.
        system: Digest of the leading system row's content, or the digest
            of ``""`` when there is no leading system row.
        history: Per-row digest of ``(role, content)`` for every row after
            the leading system row, oldest first.
    """

    provider_model: str
    system: str
    history: tuple[str, ...]


#
#######################################################################################################################
#
# Formatting helpers
#

_GLYPH_ALERT = "⚠"  # warning sign
_GLYPH_COLD = "○"  # white circle
_GLYPH_NORMAL = "●"  # black circle


def _format_amount(amount: float) -> str:
    """Format a dollar amount for chip/tooltip display.

    Amounts at or above $1 use 2 decimal places. Amounts under $1 use up to
    4 decimal places, trimmed of trailing zeros down to a 2-decimal floor
    (``0.4821`` -> ``"0.4821"``, ``0.48`` -> ``"0.48"``, ``0.10`` ->
    ``"0.10"``) so a coarse estimate doesn't display false precision while
    a precise one isn't truncated.

    Args:
        amount: The dollar amount to format (assumed non-negative).

    Returns:
        The formatted amount, without a leading ``$``.
    """
    if abs(amount) >= 1:
        return f"{amount:.2f}"
    text = f"{amount:.4f}"
    integer_part, _, frac = text.partition(".")
    frac = frac.rstrip("0")
    if len(frac) < 2:
        frac = frac.ljust(2, "0")
    return f"{integer_part}.{frac}"


def _format_tokens(count: int) -> str:
    """Format a token count as a compact chip-sized string.

    Args:
        count: Total token count.

    Returns:
        ``"12.3k"`` for counts at or above 1,000 (one decimal place), or
        the plain integer string below that.
    """
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def _format_ttl(seconds: float) -> str:
    """Format a TTL countdown as ``"M:SS"`` for the tooltip.

    Args:
        seconds: Remaining seconds until the prompt cache expires.
            Negative values are clamped to zero.

    Returns:
        The countdown as minutes:seconds, e.g. ``"4:00"``.
    """
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes}:{secs:02d}"


def _cache_state_line(
    cache_state: "ConsoleCacheState",
    *,
    cold: bool,
    ttl_remaining_s: Optional[float],
    break_reason: Optional[str],
    projected_delta_usd: Optional[float],
) -> str:
    """Build the tooltip's single cache-state narration line.

    Shared by both the priced and tokens-only branches of
    :func:`build_cost_state` so a session on an unpriced model still
    explains WHY its chip is warm/expired/alerting in the tooltip, rather
    than a bare token count leaving an alert-colored chip unexplained.

    Args:
        cache_state: Current prompt-cache state.
        cold: Whether the cache has expired (``cache_state is EXPIRED``).
        ttl_remaining_s: Seconds remaining before the cache expires, or
            ``None`` when not applicable.
        break_reason: Human-readable reason the cache is about to break,
            or ``None``.
        projected_delta_usd: Estimated extra dollar cost if the cache
            breaks, or ``None`` when not applicable/unknown.

    Returns:
        One line of tooltip text, e.g. ``"Cache: warm (4:00 remaining)"``.
    """
    if cold:
        return "Cache: expired"
    if cache_state == ConsoleCacheState.WARM:
        cache_line = "Cache: warm"
        if ttl_remaining_s is not None:
            cache_line += f" ({_format_ttl(ttl_remaining_s)} remaining)"
        if break_reason:
            cache_line += f" — {break_reason}"
            if projected_delta_usd is not None:
                cache_line += f" (~+${_format_amount(abs(projected_delta_usd))})"
        return cache_line
    return "Cache: none"


#
#######################################################################################################################
#
# Payload fingerprinting (cache-break detection)
#


def _digest(value: Any) -> str:
    """Return a stable sha1 digest of ``value`` via canonical JSON.

    ``sort_keys=True`` makes dict-key order irrelevant and ``default=str``
    covers any non-JSON-native content (e.g. enum roles) without raising,
    so the same logical row always hashes the same way regardless of which
    code path built the dict.
    """
    canonical = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def fingerprint_payload(
    provider: str,
    model: Optional[str],
    provider_messages: Sequence[Mapping[str, Any]],
) -> PayloadFingerprint:
    """Digest a provider payload's shape for later cache-break comparison.

    Only ``role``/``content`` are hashed per row -- any other keys a row
    might carry (e.g. the private native-id thread key used to anchor
    ``/rewind`` compaction) are ignored, so a fingerprint taken from an
    annotated payload compares equal to one taken from the same payload
    stripped of that key.

    Args:
        provider: The provider this payload would be/was sent to.
        model: The model this payload would be/was sent to, or ``None``.
        provider_messages: The message rows, in send order. A leading row
            with ``role == "system"`` is treated as the system prompt and
            excluded from ``history``; every other row contributes to
            ``history`` in order.

    Returns:
        A :class:`PayloadFingerprint` digesting the three components.
    """
    rows = list(provider_messages)
    system_content: Any = ""
    history_rows = rows
    if rows and str(rows[0].get("role", "")) == "system":
        system_content = rows[0].get("content", "")
        history_rows = rows[1:]

    history = tuple(
        _digest({"role": row.get("role"), "content": row.get("content")})
        for row in history_rows
    )

    return PayloadFingerprint(
        provider_model=_digest({"provider": provider, "model": model}),
        system=_digest(system_content),
        history=history,
    )


def fingerprint_break_reason(
    baseline: PayloadFingerprint, current: PayloadFingerprint
) -> Optional[str]:
    """Return why ``current`` would break the prompt cache seeded by ``baseline``.

    Anthropic's prompt cache keys off the payload's shared PREFIX -- so a
    longer ``current.history`` whose leading rows still match
    ``baseline.history`` (the normal case: turns were appended since the
    baseline was recorded) is NOT a break. Only an actual prefix mismatch
    (edited or truncated earlier history) is.

    Checked in priority order -- the first mismatch found is the reason
    reported, even when more than one component changed:
        1. ``provider_model`` (a provider/model switch invalidates the
           cache outright; whatever else changed no longer matters).
        2. ``system`` (the system prompt is always the first bytes of the
           payload, so any edit there breaks every row after it too).
        3. ``history`` prefix (an edit or truncation somewhere in the
           already-sent turns).

    Args:
        baseline: The fingerprint recorded when the cache was last warmed.
        current: The fingerprint of what would be sent right now.

    Returns:
        ``None`` when ``current`` is cache-compatible with ``baseline``
        (identical, or only appended new turns); otherwise one of
        ``"model or provider changed"``, ``"system prompt changed"``, or
        ``"earlier history changed"``.
    """
    if baseline.provider_model != current.provider_model:
        return "model or provider changed"
    if baseline.system != current.system:
        return "system prompt changed"
    if current.history[: len(baseline.history)] != baseline.history:
        return "earlier history changed"
    return None


#
#######################################################################################################################
#
# build_cost_snapshot
#


def build_cost_snapshot(
    messages: Sequence[Any], *, provider: str, model: Optional[str]
) -> ConsoleCostSnapshot:
    """Sum dollar/token totals across a Console session's transcript rows.

    Each row is priced from its own recorded ``usage`` (a ``ProviderUsage``)
    when present -- using THAT usage's own ``provider``/``model``, since a
    session can span a provider/model change mid-transcript. Rows with no
    recorded usage but non-blank ``content`` fall back to a local token
    estimate (:func:`_estimate_tokens_locally`), priced at the CURRENT
    session's ``provider``/``model`` (the caller's arguments) since there is
    no other rate to attribute an unsent/un-run row to.

    A pricing miss on any contributing row (either an unpriced usage row or
    an estimated row whose model has no seeded/configured rate) makes the
    whole snapshot's ``total_usd`` ``None`` -- ``total_tokens`` is still
    reported so the UI can fall back to a tokens-only chip rather than
    losing the row entirely.

    Estimated rows are priced role-aware: an ``assistant`` row (a real,
    documented case -- ``ConsoleChatMessage.usage`` is ``None`` for legacy
    rows and providers that reported nothing, not just for user rows) is
    priced at the model's ``output_per_mtok`` rate, since it stands in for
    a completion; every other role is priced at ``input_per_mtok``. Output
    rates commonly run 4-5x input rates, so collapsing both onto the input
    rate would understate an all-assistant-estimated transcript badly.

    Args:
        messages: Transcript rows (duck-typed: each is read via
            ``.content``, ``.usage``, and ``.role``; rows lacking all three
            attributes are treated as having no contribution).
        provider: Current session provider, used to price estimated rows.
            Normalized once through ``provider_config_key`` (the same
            mapping the rest of the app uses) before being handed to the
            estimator or the catalog, so a display-cased spelling
            ("Google") resolves the same char-ratio/pricing entry as its
            normalized form ("google").
        model: Current session model, used to price estimated rows. May be
            ``None`` when no model is selected yet.

    Returns:
        A :class:`ConsoleCostSnapshot`. Never raises -- an unexpected
        failure is logged and degrades to an empty/unknown snapshot.
    """
    try:
        catalog = get_pricing_catalog()
        provider_key = provider_config_key(provider)
        total_usd_accum = 0.0
        usd_known = True
        has_estimated = False
        total_tokens = 0
        row_count = 0

        for message in messages:
            usage = getattr(message, "usage", None)

            if isinstance(usage, ProviderUsage):
                total_tokens += usage.total_tokens
                row_count += 1
                breakdown = catalog.cost_for_usage(usage)
                if breakdown is None:
                    usd_known = False
                else:
                    total_usd_accum += breakdown.total
                continue

            content = getattr(message, "content", "") or ""
            if not str(content).strip():
                continue
            role = getattr(message, "role", "") or ""

            estimated_tokens = _estimate_tokens_locally(
                [{"role": role, "content": content}], model or "", provider_key
            )
            total_tokens += estimated_tokens
            row_count += 1
            has_estimated = True

            pricing = catalog.get_pricing(provider_key, model or "")
            if pricing is None:
                usd_known = False
            else:
                # role is a plain string or ConsoleMessageRole (a str Enum),
                # both of which compare equal to "assistant" by value.
                rate = (
                    pricing.output_per_mtok
                    if role == "assistant"
                    else pricing.input_per_mtok
                )
                total_usd_accum += estimated_tokens * rate / 1_000_000

        pricing_known = usd_known and row_count > 0
        total_usd = round(total_usd_accum, 6) if pricing_known else None

        return ConsoleCostSnapshot(
            total_usd=total_usd,
            total_tokens=total_tokens,
            pricing_known=pricing_known,
            has_estimated_entries=has_estimated,
            row_count=row_count,
        )
    except Exception:
        logger.warning(
            "console_cost_tracker.build_cost_snapshot: failed to sum transcript rows",
            exc_info=True,
        )
        return ConsoleCostSnapshot(
            total_usd=None,
            total_tokens=0,
            pricing_known=False,
            has_estimated_entries=False,
            row_count=0,
        )


#
#######################################################################################################################
#
# build_cost_state
#


def build_cost_state(
    snapshot: ConsoleCostSnapshot,
    *,
    cache_state: "ConsoleCacheState",
    break_reason: Optional[str],
    projected_delta_usd: Optional[float],
    ttl_remaining_s: Optional[float],
    pricing_as_of: Optional[str],
) -> ConsoleCostState:
    """Build pre-formatted chip text from a cost snapshot and cache context.

    Glyph/alert/cold rules:
        - ``alert`` is True only when ``cache_state`` is WARM AND
          ``break_reason`` is set -- a warm cache with nothing that could
          break it is not alarming, and an expired cache is "cold", not
          "alerting".
        - ``cold`` is True when ``cache_state`` is EXPIRED.
        - Glyph is the alert glyph when ``alert``, the cold glyph when
          ``cold``, otherwise the neutral glyph.

    Label rules:
        - When ``snapshot.pricing_known`` is False (or ``total_usd`` is
          ``None``), the chip shows a tokens-only label with no glyph
          (there's no dollar figure to attach one to).
        - Otherwise the label is ``"$<amount> <glyph>"``, prefixed with
          ``~`` when ``snapshot.has_estimated_entries`` (the total includes
          at least one estimated row), and suffixed with
          ``" ~+$<delta>"`` only when ``alert`` is True and
          ``projected_delta_usd`` is given.
        - ``compact_label`` is always the label WITHOUT the projected-delta
          suffix, for narrow layouts.

    Args:
        snapshot: The session's rolled-up cost totals.
        cache_state: Current prompt-cache state.
        break_reason: Human-readable reason the cache is about to break
            (e.g. "system prompt changed"), or ``None``.
        projected_delta_usd: Estimated extra dollar cost if the cache
            breaks, or ``None`` when not applicable/unknown.
        ttl_remaining_s: Seconds remaining before the cache expires, or
            ``None`` when not applicable.
        pricing_as_of: Human-readable date the pricing rates were last
            verified, or ``None`` when unknown/not applicable.

    Returns:
        A :class:`ConsoleCostState`. Never raises -- an unexpected failure
        is logged and degrades to an "unavailable" state.
    """
    try:
        alert = cache_state == ConsoleCacheState.WARM and bool(break_reason)
        cold = cache_state == ConsoleCacheState.EXPIRED

        if not snapshot.pricing_known or snapshot.total_usd is None:
            label = f"{_format_tokens(snapshot.total_tokens)} tok"
            tooltip_lines = [f"Tokens: {_format_tokens(snapshot.total_tokens)}"]
            if snapshot.has_estimated_entries:
                tooltip_lines.append("Includes estimated (unsent) rows.")
            # F3: narrate cache state even without a dollar total, so a
            # warm/expired/alerting chip's tooltip explains itself instead
            # of showing only a token count.
            tooltip_lines.append(
                _cache_state_line(
                    cache_state,
                    cold=cold,
                    ttl_remaining_s=ttl_remaining_s,
                    break_reason=break_reason,
                    projected_delta_usd=projected_delta_usd,
                )
            )
            tooltip_lines.append(
                "Pricing unknown for this model -- add a [pricing] override "
                "in config to see a dollar total."
            )
            tooltip = "\n".join(tooltip_lines)
            return ConsoleCostState(
                label=label, compact_label=label, tooltip=tooltip, alert=alert, cold=cold,
            )

        amount_text = _format_amount(snapshot.total_usd)
        estimate_prefix = "~" if snapshot.has_estimated_entries else ""
        glyph = _GLYPH_ALERT if alert else (_GLYPH_COLD if cold else _GLYPH_NORMAL)
        compact_label = f"{estimate_prefix}${amount_text} {glyph}"

        if alert and projected_delta_usd is not None:
            delta_text = _format_amount(abs(projected_delta_usd))
            label = f"{compact_label} ~+${delta_text}"
        else:
            label = compact_label

        total_line = f"Total: {estimate_prefix}${amount_text}"
        if snapshot.has_estimated_entries:
            total_line += " (includes estimated rows)"
        tooltip_lines = [total_line, f"Tokens: {_format_tokens(snapshot.total_tokens)}"]
        tooltip_lines.append(
            _cache_state_line(
                cache_state,
                cold=cold,
                ttl_remaining_s=ttl_remaining_s,
                break_reason=break_reason,
                projected_delta_usd=projected_delta_usd,
            )
        )

        if pricing_as_of:
            tooltip_lines.append(f"Prices as of {pricing_as_of}")

        tooltip = "\n".join(tooltip_lines)

        return ConsoleCostState(
            label=label,
            compact_label=compact_label,
            tooltip=tooltip,
            alert=alert,
            cold=cold,
        )
    except Exception:
        logger.warning(
            "console_cost_tracker.build_cost_state: failed to build chip state",
            exc_info=True,
        )
        return ConsoleCostState(
            label="Cost: unavailable",
            compact_label="Cost: unavailable",
            tooltip="Cost data unavailable.",
            alert=False,
            cold=False,
        )


#
# End of console_cost_tracker.py
#######################################################################################################################
