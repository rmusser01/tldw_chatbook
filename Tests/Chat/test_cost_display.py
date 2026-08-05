"""Tests for the shared money/token display vocabulary (Chat/cost_display.py).

This module is Console's ``_format_amount``/``_format_tokens`` promoted to a
Console-free location so the Library's RAG Answer footer (Tasks 2-3) can
speak in the same numbers as Console's cost chip. The formatter tests below
pin the EXACT contract read from ``console_cost_tracker.py:212-249`` and its
own test fixtures (``test_console_cost_tracker.py``) -- every value in the
parametrize tables below traces back to either that docstring or one of that
file's assertions, not a contract invented here.
"""

from __future__ import annotations

import ast
import inspect
from decimal import Decimal

import pytest

from tldw_chatbook.Chat import cost_display
from tldw_chatbook.Chat.cost_display import (
    build_provenance_line,
    format_cost_amount,
    format_token_count,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage


# --- format_cost_amount: byte-identical to console_cost_tracker._format_amount ---
#
# Contract (console_cost_tracker.py:212-234): amounts >= $1 use a fixed 2
# decimals; amounts under $1 use up to 4 decimals, trailing zeros trimmed but
# floored at 2 decimals. No thousands separator on the dollar side.


@pytest.mark.parametrize(
    "amount, expected",
    [
        (0.0, "0.00"),  # zero floors to 2 decimals, never blank/omitted
        (0.0031, "0.0031"),  # sub-cent: full 4-decimal precision preserved
        (0.10, "0.10"),  # trailing-zero trim floors at 2 decimals, not "0.1"
        (0.4821, "0.4821"),  # tracker fixture: test_state_normal_warm
        (0.48, "0.48"),  # tracker fixture: test_state_alert_carries_delta_and_reason
        (0.13, "0.13"),  # tracker fixture: same test's projected delta
        (1.0, "1.00"),  # >=1 boundary switches to fixed 2-decimal
        (18.0, "18.00"),  # tracker fixture: test_snapshot_sums_priced_usage_rows
        (12345.6789, "12345.68"),  # large amount: still plain 2-decimal, no comma
    ],
)
def test_format_cost_amount_matches_tracker_contract(amount, expected):
    assert format_cost_amount(amount) == expected


def test_format_cost_amount_accepts_decimal_and_matches_float_contract():
    # Task 2/3 callers hand this a Decimal cost (never-store-dollars pattern
    # elsewhere in the codebase computes cost as Decimal); the shared
    # formatter must accept it and reproduce the same string a float would.
    assert format_cost_amount(Decimal("0.0031")) == "0.0031"
    assert format_cost_amount(Decimal("1")) == "1.00"


def test_format_cost_amount_none_raises_instead_of_fabricating():
    """The type hint allows None (an as-yet-unpriced cost), but formatting
    None into a dollar string would risk a silent "$0.00" for unknown
    pricing -- the anti-fabrication rule this whole module exists to serve.
    Callers (e.g. build_provenance_line) must branch on unknown pricing
    BEFORE calling this, not rely on it to paper over None.
    """
    with pytest.raises(ValueError):
        format_cost_amount(None)


# --- format_token_count: byte-identical to console_cost_tracker._format_tokens ---
#
# Contract (console_cost_tracker.py:237-249): counts >= 1000 compact to
# "<n/1000 to 1 decimal>k"; below that, the plain integer string.


@pytest.mark.parametrize(
    "count, expected",
    [
        (0, "0"),
        (1, "1"),
        (999, "999"),  # just under the compaction boundary
        (1000, "1.0k"),  # boundary itself compacts
        (1240, "1.2k"),
        (5000, "5.0k"),  # tracker fixture: test_estimated_entries_marked...
        (12000, "12.0k"),  # tracker fixture: test_state_normal_warm
        (12345, "12.3k"),  # tracker fixture: test_state_no_pricing_shows_tokens
        (1_000_000, "1000.0k"),  # large count, still no comma grouping
    ],
)
def test_format_token_count_matches_tracker_contract(count, expected):
    assert format_token_count(count) == expected


def test_format_token_count_none_raises_instead_of_fabricating():
    with pytest.raises(ValueError):
        format_token_count(None)


# --- build_provenance_line: the one-line form shared by Console + Library --------


def test_provenance_line_no_usage_yet():
    """Shape 3: nothing has been sent yet -- provider/model only, no
    fabricated token or dollar figure."""
    line = build_provenance_line(
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=None,
        cost=None,
        pricing_known=False,
    )
    assert line == "anthropic · claude-sonnet-4-6"


def test_provenance_line_priced():
    """Shape 1: usage recorded and pricing known -- dollar figure plus the
    exact token count in parentheses."""
    usage = ProviderUsage(
        uncached_input=1000, output=240, provider="anthropic", model="claude-sonnet-4-6"
    )
    line = build_provenance_line(
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=usage,
        cost=Decimal("0.0031"),
        pricing_known=True,
    )
    assert line == "anthropic · claude-sonnet-4-6 · $0.0031 (1,240 tok)"


def test_provenance_line_pricing_unknown():
    """Shape 2: usage recorded but the model has no known rate -- tokens are
    shown (never omitted), and the missing price is stated, never guessed."""
    usage = ProviderUsage(
        uncached_input=1000, output=240, provider="anthropic", model="mystery-9000"
    )
    line = build_provenance_line(
        provider="anthropic",
        model="mystery-9000",
        usage=usage,
        cost=None,
        pricing_known=False,
    )
    assert line == "anthropic · mystery-9000 · 1,240 tok · pricing unknown"


def test_provenance_line_never_fabricates_dollar_when_cost_missing_despite_flag():
    """Defensive/contract test: even if a caller passes a contradictory
    pricing_known=True with cost=None, the line must NOT render "$0.00" --
    it must fall back to the honest tokens-only + "pricing unknown" form.
    """
    usage = ProviderUsage(uncached_input=100, output=0, provider="anthropic", model="m")
    line = build_provenance_line(
        provider="anthropic", model="m", usage=usage, cost=None, pricing_known=True,
    )
    assert "$" not in line
    assert "0.00" not in line
    assert "pricing unknown" in line
    assert "100 tok" in line


def test_provenance_line_large_token_count_uses_full_comma_grouping():
    """The provenance line is a single-answer receipt, not a running-total
    chip -- it intentionally shows the exact grouped count (e.g. "12,345
    tok") rather than the chip's lossy "12.3k" abbreviation, so a user can
    see precisely what they were charged for.
    """
    usage = ProviderUsage(uncached_input=10_000, output=2_345, provider="anthropic", model="m")
    line = build_provenance_line(
        provider="anthropic", model="m", usage=usage, cost=Decimal("0.50"), pricing_known=True,
    )
    assert "(12,345 tok)" in line
    assert "12.3k" not in line


# --- import isolation: this module must be Console-free ---------------------------


def test_cost_display_module_has_no_console_imports():
    """Library must be able to import this module without dragging in
    Console. Inspect the ACTUAL import statements (via ast), not just
    sys.modules after the fact, so this fails the moment someone adds a
    Console import even if it happens to go unused at runtime.
    """
    source = inspect.getsource(cost_display)
    tree = ast.parse(source)

    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.append(node.module)

    assert imported_names, "expected cost_display to import at least ProviderUsage"
    for name in imported_names:
        lowered = name.lower()
        assert "console" not in lowered, f"cost_display must not import {name!r}"
        assert "chat_screen" not in lowered, f"cost_display must not import {name!r}"
        assert "widgets.console" not in lowered, f"cost_display must not import {name!r}"
