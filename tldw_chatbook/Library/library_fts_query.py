"""Pure FTS5 MATCH query construction for Library keyword search.

SQLite FTS5's default ``unicode61`` tokenizer has no stemming, so a note
containing "feedback loops." is never matched by the query "feedback loop"
(task-185 UAT). This module widens each user term into an OR-group of naive
plural/singular variants at query-build time while keeping the result safe
against FTS5 query-syntax injection: every variant is emitted as a
double-quoted FTS5 string literal (embedded quotes doubled), so user text can
never introduce operators such as ``OR``/``NEAR``/``NOT``, column filters, or
unbalanced parentheses. The only bare syntax in the output is what this
module deliberately emits: parentheses and ``OR`` between variants of one
term, and ``AND`` between terms. The explicit ``AND`` keeps the service's
existing whitespace-implied AND-of-terms semantics -- it must be spelled out
because FTS5 rejects implicit AND next to a parenthesized group.

That AND is strict: one absent term zeroes a seam. TASK-3997 measured what
that costs on the gated corpus (32 of 53 ground-truthed golden queries
returned nothing) and the owner adopted ``and_then_prefix``, so this module
now also exposes the WIDENING form -- ``build_prefix_match_query`` -- that
`library_local_rag_search_service` runs for a sub-leg whose AND primary
found nothing. The two are deliberately separate functions: the AND above
is unchanged by TASK-17755 and is still what every seam runs first, and it
has a second consumer (`Chat/scope_picker_listers.py`) that gets no
fallback.
"""

from __future__ import annotations

from tldw_chatbook.Utils.fts5_match_forms import (
    build_prefix_match_expression,
    fts5_query_tokens,
    quote_fts5_token,
)

# Terms shorter than this are never expanded (articles, initials, "as", ...).
_MIN_EXPANSION_LENGTH = 3
# Variants shorter than this are dropped (e.g. "yes" -> "y").
_MIN_VARIANT_LENGTH = 2
# Endings that plausibly take an "es" plural ("box" -> "boxes").
_ES_PLURAL_ENDINGS = ("s", "x", "z", "ch", "sh")


def _quote_fts_term(term: str) -> str:
    """Return `term` as a literal FTS5 string (embedded quotes doubled).

    A thin alias over `Utils.fts5_match_forms.quote_fts5_token`, the ONE
    implementation of this escape (TASK-19558). Kept as a module-local name
    because this module's docstrings and tests refer to it by that name.
    """
    return quote_fts5_token(term)


def expand_keyword_term(term: str) -> tuple[str, ...]:
    """Expand one keyword term into naive plural/singular variants.

    Rules (deliberately naive -- extra variants that hit no real word are
    harmless because FTS simply never matches them):

    - ``ies`` ending swaps to ``y`` ("stories" -> "story").
    - Otherwise a trailing ``es`` or ``s`` is stripped ("loops" -> "loop").
    - Terms not ending in ``s`` gain ``s`` ("loop" -> "loops"), gain ``es``
      after es-plural endings ("box" -> "boxes"), and swap a trailing ``y``
      for ``ies`` ("story" -> "stories").

    Args:
        term: A single whitespace-delimited query term.

    Returns:
        A deduplicated tuple starting with `term` itself. Terms shorter than
        3 characters or containing any non-alphabetic character are returned
        unchanged as a 1-tuple.
    """
    if len(term) < _MIN_EXPANSION_LENGTH or not term.isalpha():
        return (term,)
    lower = term.lower()
    variants = [term]
    if lower.endswith("ies"):
        variants.append(term[:-3] + "y")
    elif lower.endswith("es"):
        variants.extend((term[:-2], term[:-1]))
    elif lower.endswith("s"):
        variants.append(term[:-1])
    else:
        variants.append(term + "s")
        if lower.endswith(_ES_PLURAL_ENDINGS):
            variants.append(term + "es")
        if lower.endswith("y"):
            variants.append(term[:-1] + "ies")
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower()
        if len(variant) >= _MIN_VARIANT_LENGTH and key not in seen:
            seen.add(key)
            deduped.append(variant)
    return tuple(deduped)


def build_fts_match_query(query: str) -> str:
    """Build a safe FTS5 MATCH string with plural/singular widening.

    Each whitespace-delimited term becomes either a single quoted literal
    (no variants) or a parenthesized OR-group of quoted variant literals;
    groups are joined with explicit ``AND`` -- the same multi-term semantics
    the Library service always used for space-joined plain quoted terms
    (FTS5 rejects implicit AND next to a parenthesized group).

    Args:
        query: Validated user query text (plain natural language).

    Returns:
        An FTS5 MATCH expression, e.g. ``"feedback loop"`` becomes
        ``("feedback" OR "feedbacks") AND ("loop" OR "loops")``.
    """
    groups: list[str] = []
    for term in query.split():
        quoted = [_quote_fts_term(variant) for variant in expand_keyword_term(term)]
        if len(quoted) == 1:
            groups.append(quoted[0])
        else:
            groups.append("(" + " OR ".join(quoted) + ")")
    return " AND ".join(groups)


def build_prefix_match_query(query: str) -> str:
    """Build the PREFIX form -- ``and_then_prefix``'s widening fallback.

    TASK-17755. Run by `library_local_rag_search_service` for a sub-leg
    whose `build_fts_match_query` primary returned zero rows, and never
    otherwise. It reaches documents the primary's plural/singular widening
    cannot: "tension" does not widen to "tensioner", but ``"tension"*``
    matches it.

    A thin adapter over `Utils.fts5_match_forms`, which is the ONE
    definition of this form and of the function-word list it trims with --
    shared with the engine's keyword leg, which has run the same
    construction since TASK-15700. Not re-implemented here on purpose: the
    Library screen's Search tab is this path and its RAG Answer tab is the
    engine, and TASK-3997 documented a user hitting two different matching
    rules on one screen. A second copy of the form would re-open that gap
    the first time either copy was edited.

    Args:
        query: Validated user query text (plain natural language).

    Returns:
        The prefix expression, e.g. ``feedback loop`` becomes
        ``"feedback"* "loop"*``. ``""`` when the query has no content
        tokens -- meaning "no rows", which callers must honour by skipping
        the query rather than widening further (see the shared module for
        why an unbounded stopword prefix is the one thing worse than a
        zero-row seam).
    """
    return build_prefix_match_expression(fts5_query_tokens(query))
