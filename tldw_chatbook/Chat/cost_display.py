# tldw_chatbook/Chat/cost_display.py
# Description: Console-free money/token display vocabulary shared by Console and Library.
#
"""Shared money/token display vocabulary for Console and Library.

Console's cost chip (:mod:`tldw_chatbook.Chat.console_cost_tracker`) grew a
precise dollar/token formatting contract first; this module promotes that
vocabulary out of Console so a second surface -- the Library's RAG Answer
cost footer -- can speak in exactly the same numbers instead of re-deriving
its own rounding rules. ``console_cost_tracker._format_amount`` and
``_format_tokens`` are now thin delegates to :func:`format_cost_amount` and
:func:`format_token_count` below, kept under their old names so the
tracker's internal call sites and tests are untouched.

Import discipline: this module must stay importable from the Library
without dragging Console in. It depends only on
:mod:`tldw_chatbook.Chat.provider_usage` (a pure dataclass module with no
Console imports of its own) -- never on ``Widgets.Console``,
``UI.Screens.chat_screen``, or any ``Chat.console_*`` module. See
``Tests/Chat/test_cost_display.py::test_cost_display_module_has_no_console_imports``.

Anti-fabrication rule: nothing in this module ever turns an unknown amount
into a printable number. :func:`format_cost_amount` and
:func:`format_token_count` raise on ``None`` rather than silently rendering
a fabricated ``"$0.00"``/``"0"`` -- a caller with an unpriced/uncounted
value must say so (see :func:`build_provenance_line`'s "pricing unknown"
branch), never paper over it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional, Union

from tldw_chatbook.Chat.provider_usage import ProviderUsage

#
#######################################################################################################################
#
# Formatters (byte-identical to console_cost_tracker's former privates)
#


def format_cost_amount(value: Optional[Union[Decimal, float]]) -> str:
    """Format a dollar amount for chip/footer/tooltip display.

    Amounts at or above $1 use 2 decimal places. Amounts under $1 use up to
    4 decimal places, trimmed of trailing zeros down to a 2-decimal floor
    (``0.4821`` -> ``"0.4821"``, ``0.48`` -> ``"0.48"``, ``0.10`` ->
    ``"0.10"``) so a coarse estimate doesn't display false precision while a
    precise one isn't truncated. This is the exact contract Console's cost
    chip has always used (formerly ``console_cost_tracker._format_amount``,
    now a thin delegate to this function).

    Args:
        value: The dollar amount to format (assumed non-negative), as a
            ``Decimal`` or ``float``. Never ``None`` -- see Raises.

    Returns:
        The formatted amount, without a leading ``$``.

    Raises:
        ValueError: If ``value`` is ``None``. An unknown cost must never be
            formatted into a number (that would risk a silently fabricated
            ``"0.00"``) -- callers must branch on unknown pricing before
            reaching this function.
    """
    if value is None:
        raise ValueError(
            "format_cost_amount requires a known amount; got None. "
            "Check pricing/cost availability before formatting."
        )
    amount = value
    if abs(amount) >= 1:
        return f"{amount:.2f}"
    text = f"{amount:.4f}"
    integer_part, _, frac = text.partition(".")
    frac = frac.rstrip("0")
    if len(frac) < 2:
        frac = frac.ljust(2, "0")
    return f"{integer_part}.{frac}"


def format_token_count(n: Optional[int]) -> str:
    """Format a token count as a compact chip-sized string.

    This is the exact contract Console's cost chip has always used
    (formerly ``console_cost_tracker._format_tokens``, now a thin delegate
    to this function): a chip/label-sized abbreviation, not a precise
    receipt -- see :func:`build_provenance_line` for the latter.

    Args:
        n: Total token count. Never ``None`` -- see Raises.

    Returns:
        ``"12.3k"`` for counts at or above 1,000 (one decimal place), or
        the plain integer string below that.

    Raises:
        ValueError: If ``n`` is ``None``. An unknown/uncounted token total
            must never be formatted into a number.
    """
    if n is None:
        raise ValueError(
            "format_token_count requires a known count; got None. "
            "Check usage availability before formatting."
        )
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _format_full_token_count(n: int) -> str:
    """Full, comma-grouped token count for the provenance line.

    Deliberately distinct from :func:`format_token_count`'s "12.3k"-style
    abbreviation: :func:`build_provenance_line` is a single paid moment's
    receipt (exact numbers matter for transparency), not a running-session
    chip label (where compactness matters more than precision).
    """
    return f"{n:,}"


#
#######################################################################################################################
#
# build_provenance_line
#


def build_provenance_line(
    *,
    provider: str,
    model: str,
    usage: Optional[ProviderUsage],
    cost: Optional[Decimal],
    pricing_known: bool,
) -> str:
    """Build the one-line provenance string shared by Console and Library.

    Three shapes, depending on what is actually known -- never a fabricated
    dollar figure when pricing isn't known, and never a silently omitted
    token count when usage IS known:

    - No usage yet (nothing sent/received): ``"provider · model"``.
    - Usage recorded, pricing known: ``"provider · model · $<amount>
      (<tokens> tok)"``.
    - Usage recorded, pricing NOT known (or a cost wasn't actually supplied
      despite ``pricing_known``): ``"provider · model · <tokens> tok ·
      pricing unknown"``.

    Fix-review (Library RAG Answer, PR-3 Task 3): ``model`` can legitimately
    be ``""`` -- a real provider payload can omit its own ``"model"`` key
    (upstream shape, not a caller bug) -- while ``usage`` is still known,
    because the call was still billed. The header is built by joining only
    the NON-EMPTY identifiers with `` · ``, so a blank ``model`` (or, in
    principle, a blank ``provider``) is omitted entirely rather than left as
    a dangling separator with nothing after it: ``"anthropic · 11,251 tok ·
    pricing unknown"``, never ``"anthropic ·  · 11,251 tok · pricing
    unknown"``. Every existing caller that always supplies both identifiers
    sees byte-identical output to before this fix.

    Args:
        provider: Provider identifier (as already displayed elsewhere, not
            re-normalized here). May be ``""`` if a caller genuinely has
            none, though every current caller always supplies one.
        model: Model identifier, or ``""`` when the provider's response
            carried no model name (a real, reachable upstream shape -- not
            treated as "no usage yet"; see the fix-review note above).
        usage: The turn's normalized token usage, or ``None`` when nothing
            has been sent/received yet.
        cost: The turn's computed dollar cost, or ``None`` when pricing is
            unknown for this model.
        pricing_known: Whether ``cost`` reflects a real, known rate. Only
            used together with a non-``None`` ``cost`` -- if a caller passes
            ``pricing_known=True`` with ``cost=None`` (a contradictory
            input), this still falls back to the honest tokens-only form
            rather than fabricating ``"$0.00"``.

    Returns:
        The formatted one-line provenance string.
    """
    header = " · ".join(part for part in (provider, model) if part)
    if usage is None:
        return header

    tokens_text = f"{_format_full_token_count(usage.total_tokens)} tok"
    if pricing_known and cost is not None:
        return f"{header} · ${format_cost_amount(cost)} ({tokens_text})"
    return f"{header} · {tokens_text} · pricing unknown"
