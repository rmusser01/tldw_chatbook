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
"""

from __future__ import annotations

# Terms shorter than this are never expanded (articles, initials, "as", ...).
_MIN_EXPANSION_LENGTH = 3
# Variants shorter than this are dropped (e.g. "yes" -> "y").
_MIN_VARIANT_LENGTH = 2
# Endings that plausibly take an "es" plural ("box" -> "boxes").
_ES_PLURAL_ENDINGS = ("s", "x", "z", "ch", "sh")


def _quote_fts_term(term: str) -> str:
    """Return `term` as a literal FTS5 string (embedded quotes doubled)."""
    return '"' + term.replace('"', '""') + '"'


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
