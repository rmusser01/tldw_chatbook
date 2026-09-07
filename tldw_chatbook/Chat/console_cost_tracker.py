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
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence

from loguru import logger

from tldw_chatbook.Chat.console_session_settings import _estimate_tokens_locally
from tldw_chatbook.Chat.cost_display import format_cost_amount, format_token_count
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import get_pricing_catalog

#
#######################################################################################################################
#
# Enums / dataclasses
#


def console_cost_snapshot_messages(messages: Sequence[Any]) -> list[Any]:
    """Keep only finalized provider rows in cost summaries.

    Streaming content is materialized for live transcript display, but the cost
    chip must remain stable until that row reaches a terminal status. Direct
    raw-command markers carry no provider usage and are excluded completely.
    """
    return [
        message
        for message in messages
        if getattr(message, "status", "complete") not in {"pending", "streaming"}
        and getattr(message, "raw_cli_presentation", None) is None
    ]


class ConsoleCacheState(str, Enum):
    """Prompt-cache state for the active Console session, as seen by the chip."""

    NONE = "none"
    WARM = "warm"
    EXPIRED = "expired"


#
#######################################################################################################################
#
# Token-estimate memo (task-15451)
#


#: Entry ceiling for :class:`TokenEstimateCache`. Generous on purpose: past
#: this size a single pass over a longer transcript would evict its own
#: earlier rows and degrade to the uncached cost (never to a WRONG cost), so
#: the cap is set well above any realistic Console transcript rather than
#: tuned for memory. Entries hold the caller's OWN row strings -- the store
#: already owns those objects -- so a live entry costs a tuple, not a copy.
TOKEN_ESTIMATE_CACHE_MAX_ENTRIES = 4096


class TokenEstimateCache:
    """Verified memo for local token estimates over transcript rows (task-15451).

    ``_estimate_tokens_locally`` is a per-character Python loop whenever no
    tokenizer is installed (tiktoken is not a base dependency -- task-2526),
    so re-estimating an unchanged transcript on every Console sync tick cost
    O(transcript chars) on the event loop five times a second. This memo
    makes a repeat pass O(rows).

    Correctness does not depend on the KEY. Every hit is verified against
    the full signature of the estimate -- ``(model, provider, rows)``, where
    ``rows`` is a tuple of ``(role, content)`` pairs -- so a key that
    collides, or is reused by a row whose text changed, misses and
    recomputes. The key only determines the hit RATE. That is why this can
    be reasoned about locally: there is no invalidation protocol to get
    wrong, and no way for a stale estimate to be served.

    Tuple comparison short-circuits on identity per element, so the common
    case (the store hands back the same ``str`` objects pass after pass --
    ``dataclasses.replace`` shares the reference) costs pointer compares;
    a rebuilt-but-equal string (the staged-evidence pseudo-row, joined
    afresh every pass) costs one C-level ``memcmp``.

    Not thread-safe: it is owned by, and only ever touched from, the UI
    thread that builds the chip state.
    """

    def __init__(self, *, max_entries: int = TOKEN_ESTIMATE_CACHE_MAX_ENTRIES) -> None:
        self._entries: "OrderedDict[Any, tuple[tuple[Any, ...], int]]" = OrderedDict()
        self._max_entries = max(1, int(max_entries))

    def estimate(
        self,
        key: Any,
        signature: tuple[Any, ...],
        compute: Callable[[], int],
    ) -> int:
        """Return the memoized estimate for ``signature``, computing on a miss.

        Args:
            key: Cache slot. Any hashable; a message id for a transcript row.
                Collisions only cost a recompute, never a wrong answer.
            signature: Everything the estimate depends on -- by convention
                ``(model, provider, rows)`` built by
                :func:`token_estimate_signature`. Compared for equality
                before any hit is served.
            compute: Called on a miss. Deliberately a callback rather than a
                hard-wired estimator call so each call site keeps its own
                estimator reference (and stays interceptable by its own
                tests).

        Returns:
            The estimated token count for ``signature``.
        """
        entry = self._entries.get(key)
        if entry is not None and entry[0] == signature:
            self._entries.move_to_end(key)
            return entry[1]
        value = compute()
        self._entries[key] = (signature, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop every entry (nothing depends on this for correctness)."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def token_estimate_signature(
    rows: Sequence[tuple[Any, Any]],
    model: str,
    provider: str,
) -> tuple[Any, ...]:
    """Build a :class:`TokenEstimateCache` signature for ``rows``.

    Args:
        rows: ``(role, content)`` pairs exactly as they will be handed to
            the estimator.
        model: Model name the estimate is for (selects the tokenizer).
        provider: Normalized provider key (selects the chars-floor ratio).

    Returns:
        A comparison tuple. Only equality is ever taken on it, never a
        hash, so row content that is unhashable still compares correctly.
    """
    return (model, provider, tuple(rows))


def _estimate_row_tokens(role: Any, content: Any, model: str, provider: str) -> int:
    """Estimate one transcript row's tokens.

    The single call shared by :func:`build_cost_snapshot`'s cached and
    uncached paths, so the two can never drift apart.
    """
    return _estimate_tokens_locally(
        [{"role": role, "content": content}], model, provider
    )


#
#######################################################################################################################
#
# Enums / dataclasses (continued)
#


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
        fleet_tokens: PR2b Task 5 (cost rollup) -- sub-agent fleet token
            spend for the active session, ALREADY folded into
            ``total_tokens`` (see :func:`build_cost_snapshot`'s
            ``fleet_tokens`` parameter) and broken out again here only so
            the chip's tooltip can name it. Never contributes to
            ``total_usd``: a fleet child's measured spend
            (``FleetHandle.total_tokens``) is a single combined
            prompt+completion figure with no input/output split, so there
            is no honest per-model rate to price it at -- shown as an
            unpriced token count rather than either a fabricated dollar
            figure or (the other extreme) silently dropping the primary
            transcript's own already-known pricing to "unknown" just
            because a fleet ran. 0 when the session has no fleet spend to
            report.
    """

    total_usd: Optional[float]
    total_tokens: int
    pricing_known: bool
    has_estimated_entries: bool
    row_count: int
    fleet_tokens: int = 0
    available: bool = True


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
class ConsoleCostRow:
    """One transcript row's per-bucket token/cost breakdown (task-5 modal).

    Built by :func:`build_cost_rows`, one row per contributing message, in
    transcript order -- the per-message counterpart to
    :class:`ConsoleCostSnapshot`'s rolled-up totals. A widget renders these
    directly; nothing here is pre-formatted (unlike :class:`ConsoleCostState`)
    since the modal table needs the raw numbers for column alignment.

    Attributes:
        index: 0-based position of this row's source message within the
            transcript rows :func:`build_cost_rows` was given (rows with no
            contribution -- no usage and blank content -- are skipped, so
            this is not necessarily contiguous with neighboring rows).
        role: The message's role (``"user"``, ``"assistant"``, ...).
        model: The model this row's usage/estimate is attributed to --
            the row's own recorded ``ProviderUsage.model`` when present,
            otherwise the current session model passed to
            :func:`build_cost_rows`.
        uncached_input: Uncached input tokens (0 for an estimated
            ``assistant`` row -- see ``estimated``).
        cache_read: Cache-read input tokens (always 0 for estimated rows;
            the local estimator has no cache concept).
        cache_write: Cache-write input tokens (always 0 for estimated rows).
        output: Output tokens (0 for a non-``assistant`` estimated row).
        cost_usd: Dollar cost for this row, or ``None`` when the row's
            model has no known pricing. Already INCLUDES
            ``audio_input``/``audio_output``/``transcription_seconds``'
            dollar contribution (task-2390) -- ``PricingCatalog.
            cost_for_usage``'s ``total`` folds every bucket together; the
            three fields below exist so the modal can still show audio and
            transcription usage as their own line rather than leaving them
            invisible inside this one figure.
        estimated: True when this row had no recorded ``ProviderUsage`` and
            its tokens/cost came from the local character-ratio estimator.
        audio_input: Of ``uncached_input``+``cache_read``, how many were
            audio tokens (task-2390, realtime only; 0 for every other
            row -- see ``ProviderUsage.audio_input``'s own docstring for
            the subset relationship). Always 0 for an estimated row.
        audio_output: Of ``output``, how many were audio tokens
            (task-2390, realtime only; 0 otherwise, always 0 when
            estimated).
        transcription_seconds: Duration of input audio transcribed for
            this row (task-2390, realtime only; 0.0 otherwise, always 0.0
            when estimated). Not a token count.
    """

    index: int
    role: str
    model: str
    uncached_input: int
    cache_read: int
    cache_write: int
    output: int
    cost_usd: Optional[float]
    estimated: bool
    audio_input: int = 0
    audio_output: int = 0
    transcription_seconds: float = 0.0


@dataclass(frozen=True)
class ConsoleCostRowTotals:
    """Aggregate totals row for a :func:`build_cost_rows` breakdown.

    Attributes:
        total_tokens: Summed tokens across every row.
        total_cost_usd: Summed dollar cost, or ``None`` when at least one
            contributing row has unknown pricing (never a partial total).
        has_estimated_entries: True when at least one row was estimated
            rather than priced from recorded usage.
        row_count: Number of rows the totals were computed from.
    """

    total_tokens: int
    total_cost_usd: Optional[float]
    has_estimated_entries: bool
    row_count: int


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

    Thin delegate to :func:`tldw_chatbook.Chat.cost_display.format_cost_amount`
    -- kept under this name so existing call sites/tests in this module are
    untouched. See that function for the full formatting contract (amounts
    at or above $1 use 2 decimal places; amounts under $1 use up to 4,
    trimmed of trailing zeros down to a 2-decimal floor).

    Args:
        amount: The dollar amount to format (assumed non-negative).

    Returns:
        The formatted amount, without a leading ``$``.
    """
    return format_cost_amount(amount)


def _format_tokens(count: int) -> str:
    """Format a token count as a compact chip-sized string.

    Thin delegate to :func:`tldw_chatbook.Chat.cost_display.format_token_count`
    -- kept under this name so existing call sites/tests in this module are
    untouched.

    Args:
        count: Total token count.

    Returns:
        ``"12.3k"`` for counts at or above 1,000 (one decimal place), or
        the plain integer string below that.
    """
    return format_token_count(count)


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
    messages: Sequence[Any],
    *,
    provider: str,
    model: Optional[str],
    fleet_tokens: int = 0,
    estimate_cache: Optional[TokenEstimateCache] = None,
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
        fleet_tokens: PR2b Task 5 -- sub-agent fleet token spend to fold
            into ``total_tokens`` (see :class:`ConsoleCostSnapshot`.
            ``fleet_tokens``'s docstring for why this contributes tokens
            but never dollars). Defaults to 0, byte-identical to this
            function's pre-Task-5 behavior for every caller that doesn't
            pass it.
        estimate_cache: task-15451 -- optional :class:`TokenEstimateCache`
            memoizing the per-row local estimates, so a caller polling this
            on a timer (the Console cost chip, 5x/s while a run is active)
            stops re-tokenizing rows whose text has not changed. Every hit
            is verified against the row's own ``(model, provider, role,
            content)``, so passing one can only change how LONG this takes,
            never what it returns. ``None`` (the default) estimates every
            row on every call, exactly as before.

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

        for index, message in enumerate(messages):
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

            if estimate_cache is None:
                estimated_tokens = _estimate_row_tokens(
                    role, content, model or "", provider_key
                )
            else:
                # Cache slot: the row's own id when it has one. The staged-
                # evidence pseudo-row (`console_prompted_evidence_text`) has
                # none, so it falls back to its position -- stable across
                # passes, and wrong only in the harmless direction: a hit is
                # still verified against the row's text before it is served.
                row_id = getattr(message, "id", None)
                estimated_tokens = estimate_cache.estimate(
                    row_id if isinstance(row_id, str) and row_id else ("#row", index),
                    token_estimate_signature(
                        ((role, content),), model or "", provider_key
                    ),
                    partial(
                        _estimate_row_tokens, role, content, model or "", provider_key
                    ),
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
        # Real, measured tokens (not estimated) -- always folded into the
        # total, never priced (see ConsoleCostSnapshot.fleet_tokens).
        fleet_tokens = max(0, fleet_tokens)
        total_tokens += fleet_tokens

        return ConsoleCostSnapshot(
            total_usd=total_usd,
            total_tokens=total_tokens,
            pricing_known=pricing_known,
            has_estimated_entries=has_estimated,
            row_count=row_count,
            fleet_tokens=fleet_tokens,
        )
    except Exception:
        logger.opt(exception=True).warning(
            "console_cost_tracker.build_cost_snapshot: failed to sum transcript rows"
        )
        return ConsoleCostSnapshot(
            total_usd=None,
            total_tokens=0,
            pricing_known=False,
            has_estimated_entries=False,
            row_count=0,
            fleet_tokens=0,
            available=False,
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

        if not snapshot.available:
            return ConsoleCostState(
                label="unavailable",
                compact_label="unavailable",
                tooltip="Cost data unavailable.\n"
                + _cache_state_line(
                    cache_state,
                    cold=cold,
                    ttl_remaining_s=ttl_remaining_s,
                    break_reason=break_reason,
                    projected_delta_usd=projected_delta_usd,
                ),
                alert=alert,
                cold=cold,
            )

        if not snapshot.pricing_known or snapshot.total_usd is None:
            estimate_prefix = "~" if snapshot.has_estimated_entries else ""
            label = f"{estimate_prefix}{_format_tokens(snapshot.total_tokens)} tok"
            tooltip_lines = [f"Tokens: {_format_tokens(snapshot.total_tokens)}"]
            if snapshot.has_estimated_entries:
                tooltip_lines.append("Includes locally estimated transcript rows.")
            if snapshot.fleet_tokens:
                tooltip_lines.append(
                    f"Sub-agents: {_format_tokens(snapshot.fleet_tokens)} tok "
                    "(not priced)"
                )
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
                label=label,
                compact_label=label,
                tooltip=tooltip,
                alert=alert,
                cold=cold,
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
        if snapshot.fleet_tokens:
            tooltip_lines.append(
                f"Sub-agents: {_format_tokens(snapshot.fleet_tokens)} tok (not priced)"
            )
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
        logger.opt(exception=True).warning(
            "console_cost_tracker.build_cost_state: failed to build chip state"
        )
        return ConsoleCostState(
            label="Cost: unavailable",
            compact_label="Cost: unavailable",
            tooltip="Cost data unavailable.",
            alert=False,
            cold=False,
        )


#
#######################################################################################################################
#
# build_cost_rows (task-5: per-message breakdown for the cost modal)
#


def build_cost_rows(
    messages: Sequence[Any], *, provider: str, model: Optional[str]
) -> list[ConsoleCostRow]:
    """Build one breakdown row per transcript message, for the cost modal.

    Mirrors :func:`build_cost_snapshot`'s per-row pricing rules -- a row's
    own recorded usage prices at THAT usage's own provider/model, a row
    without usage falls back to the local token estimate priced at the
    CURRENT session provider/model, and an estimated row is priced
    role-aware (``assistant`` at the output rate, everything else at the
    input rate) -- but keeps every row separate instead of summing them, so
    the modal can render a per-message table.

    Args:
        messages: Transcript rows (duck-typed like :func:`build_cost_snapshot`:
            each is read via ``.content``, ``.usage``, and ``.role``).
        provider: Current session provider, used to price estimated rows.
            Normalized once through ``provider_config_key``.
        model: Current session model, used to price estimated rows and as
            the fallback ``ConsoleCostRow.model`` for rows without their
            own recorded usage. May be ``None``.

    Returns:
        One :class:`ConsoleCostRow` per contributing message (rows with
        neither usage nor non-blank content are skipped), in transcript
        order. Never raises -- a catalog-init failure returns an empty
        list, and a single row's failure is logged and that row skipped
        rather than aborting the whole breakdown.
    """
    rows: list[ConsoleCostRow] = []
    try:
        catalog = get_pricing_catalog()
        provider_key = provider_config_key(provider)
    except Exception:
        logger.opt(exception=True).warning(
            "console_cost_tracker.build_cost_rows: failed to init pricing catalog"
        )
        return rows

    for index, message in enumerate(messages):
        try:
            usage = getattr(message, "usage", None)
            role = str(getattr(message, "role", "") or "")

            if isinstance(usage, ProviderUsage):
                breakdown = catalog.cost_for_usage(usage)
                rows.append(
                    ConsoleCostRow(
                        index=index,
                        role=role,
                        model=usage.model or (model or ""),
                        uncached_input=usage.uncached_input,
                        cache_read=usage.cache_read,
                        cache_write=usage.cache_write,
                        output=usage.output,
                        cost_usd=breakdown.total if breakdown is not None else None,
                        estimated=False,
                        audio_input=usage.audio_input,
                        audio_output=usage.audio_output,
                        transcription_seconds=usage.transcription_seconds,
                    )
                )
                continue

            content = getattr(message, "content", "") or ""
            if not str(content).strip():
                continue

            estimated_tokens = _estimate_tokens_locally(
                [{"role": role, "content": content}], model or "", provider_key
            )
            pricing = catalog.get_pricing(provider_key, model or "")
            cost_usd: Optional[float] = None
            if pricing is not None:
                # role is a plain string or ConsoleMessageRole (a str Enum),
                # both of which compare equal to "assistant" by value --
                # same role-aware rule as build_cost_snapshot.
                rate = (
                    pricing.output_per_mtok
                    if role == "assistant"
                    else pricing.input_per_mtok
                )
                cost_usd = round(estimated_tokens * rate / 1_000_000, 6)

            is_assistant = role == "assistant"
            rows.append(
                ConsoleCostRow(
                    index=index,
                    role=role,
                    model=model or "",
                    uncached_input=0 if is_assistant else estimated_tokens,
                    cache_read=0,
                    cache_write=0,
                    output=estimated_tokens if is_assistant else 0,
                    cost_usd=cost_usd,
                    estimated=True,
                )
            )
        except Exception:
            logger.opt(exception=True).warning(
                "console_cost_tracker.build_cost_rows: failed to build row {}",
                index,
            )
            continue

    return rows


def build_cost_rows_totals(rows: Sequence[ConsoleCostRow]) -> ConsoleCostRowTotals:
    """Sum a :func:`build_cost_rows` breakdown into one totals row.

    Args:
        rows: The rows to total (typically the output of
            :func:`build_cost_rows`, but any sequence of
            :class:`ConsoleCostRow` works).

    Returns:
        A :class:`ConsoleCostRowTotals`. ``total_cost_usd`` is ``None``
        when at least one row's cost is unknown or ``rows`` is empty (never
        a partial dollar total), mirroring
        :class:`ConsoleCostSnapshot`'s ``pricing_known`` contract.
    """
    total_tokens = 0
    total_cost_accum = 0.0
    cost_known = True
    has_estimated = False
    for row in rows:
        total_tokens += (
            row.uncached_input + row.cache_read + row.cache_write + row.output
        )
        has_estimated = has_estimated or row.estimated
        if row.cost_usd is None:
            cost_known = False
        else:
            total_cost_accum += row.cost_usd

    pricing_known = cost_known and len(rows) > 0
    return ConsoleCostRowTotals(
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost_accum, 6) if pricing_known else None,
        has_estimated_entries=has_estimated,
        row_count=len(rows),
    )


#
# End of console_cost_tracker.py
#######################################################################################################################
