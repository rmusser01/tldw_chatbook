"""The FTS5 MATCH *forms* shared by the RAG engine and the Library seams.

TASK-17755. Two retrieval paths in this app build FTS5 MATCH expressions:
the engine's keyword leg (``RAG_Search/simplified/rag_service.py``) and the
Library's four-seam plain-Search path (``Library/library_fts_query.py`` +
``Library/library_local_rag_search_service.py``). The engine has shipped the
``and_then_prefix`` construction since TASK-15700 and TASK-17755 adopts the
same construction on the four-seam path, so both now need the **PREFIX
form** and the function-word list it trims with.

This module is that ONE definition. It exists because the alternative --
a second copy of a 67-word stopword list and a second `"tok"*` builder --
is the exact failure mode TASK-17600 shipped a guard against: twin literals
drift apart silently, and a retrieval form that differs between two paths
is indistinguishable from a retrieval *bug* when a query answers in one
place and not the other. Reuse is not a tidiness preference here; the whole
point of TASK-17755 is that the Library's Search tab and its RAG Answer tab
stop disagreeing about what matches.

Deliberately dependency-free (stdlib only, no package imports): its two
consumers are a heavy engine module and a pure Library module, and the pure
one must stay cheap to import. That is why it lives under ``Utils/`` --
whose ``__init__`` is empty -- rather than under ``RAG_Search/``, whose
package ``__init__`` pulls the whole simplified engine (chromadb,
embeddings) in a try/except.

**TASK-19558 widened this module's remit, and the sentence that used to
stand here ("What is NOT here: the AND forms") is now false.** The AND form
IS here -- ``build_and_match_expression``/``build_and_match_query`` -- along
with the PHRASE form and the one string-literal escape they all share,
because that task found six spellings of the escape across the app, two of
them wrong. What is still each path's own is the Library's
OR-of-plural-variants widening (``library_fts_query.build_fts_match_query``),
a genuinely different construction.

The three FORMS this module defines, and the rule for picking one:

* **AND** (``build_and_match_query``) -- every token quoted, space-joined,
  i.e. FTS5's implicit AND. What a plain-text search box wants, and what a
  seam that used to bind its query RAW was already doing.
* **PHRASE** (``build_phrase_match_query``) -- the whole query as one quoted
  literal; words must be adjacent and in order. Only for seams that already
  bound a quoted phrase before TASK-19558.
* **PREFIX** (``quote_fts5_prefix`` / ``build_prefix_match_expression``) --
  a quoted term with the star OUTSIDE the quotes, for type-ahead pickers and
  for ``and_then_prefix``'s widening fallback.
"""

from __future__ import annotations

import re
from typing import FrozenSet, Iterable, List

__all__ = [
    "FTS5_STOPWORDS",
    "build_and_match_expression",
    "build_and_match_query",
    "build_phrase_match_query",
    "build_prefix_match_expression",
    "fts5_query_is_searchable",
    "fts5_query_tokens",
    "fts5_token_runs",
    "is_fts5_stopword",
    "quote_fts5_phrase",
    "quote_fts5_prefix",
    "quote_fts5_token",
]

# A small fixed English function-word list, consulted by every widening FORM
# and by NO primary AND form. Moved here verbatim from
# ``rag_service._FTS5_STOPWORDS`` (TASK-15400/15700), which now imports it
# back under that name so the engine's own tests and probes keep reading the
# same object they always did -- there is exactly one list, not two that
# happen to agree today.
#
# Where it lands on the shipped default paths: the engine's
# ``and_then_prefix`` PRIMARY is the full AND, which does not consult this
# list, and its FALLBACK is the prefix form, which does -- so it is
# consulted only for sub-legs whose primary returned zero rows. TASK-17755
# gives the Library's four seams the same shape, so the same sentence is now
# true there.
#
# The PREFIX form is where this list is MOST load-bearing, by a distance. An
# untrimmed OR admits every document containing "the"; an untrimmed prefix
# term is worse still, because ``"the"*`` matches "the", "then", "there",
# "their", "these" -- i.e. nearly a whole corpus -- and the prefix form ANDs
# its terms, so one such term does not merely add noise, it makes the whole
# expression match almost everything the other terms allow. That is why
# ``build_prefix_match_expression`` answers ``""`` (no rows) when trimming
# empties the token list rather than widening to something useless.
#
# Fixed and small on purpose -- a large list starts deleting content words
# (TASK-15400's census found the real blockers were `template`, `building`,
# `rough`, `turns`, `pulls`, `builds`, which no stopword list removes).
FTS5_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a", "about", "all", "also", "am", "an", "and", "any", "are", "as",
        "at", "be", "been", "but", "by", "can", "do", "does", "for", "from",
        "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
        "its", "me", "my", "no", "not", "of", "on", "or", "our", "out",
        "so", "than", "that", "the", "their", "them", "then", "there",
        "these", "they", "this", "to", "up", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "would",
        "you", "your",
    }
)

#: One alphanumeric run, as FTS5's default ``unicode61`` tokenizer reads it.
_RUN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def fts5_token_runs(token: str) -> List[str]:
    """The alphanumeric runs FTS5 would index a raw token as.

    ``Obsidian-3`` is two runs (``Obsidian``, ``3``); ``About,`` is one
    (``about``, case-folded by the tokenizer). Every comparison in this
    module is made on runs rather than on the raw string, because runs are
    what FTS5 actually matches.

    Args:
        token: One raw whitespace-delimited query token.

    Returns:
        The token's alphanumeric runs, in order; empty for pure punctuation.
    """
    return _RUN_PATTERN.findall(token)


def quote_fts5_token(token: str) -> str:
    """Quote ONE query token as an FTS5 string literal.

    The single place the injection-safety property of TASK-3995 is
    implemented for the widening forms: a bare token FTS5 parses as
    column-filter or operator syntax (``Obsidian-3`` raises
    ``OperationalError('no such column: 3')``; a typed ``OR`` becomes a
    disjunction of the user's own words), while a quoted token is a literal
    string with no operator semantics. An embedded double quote is doubled,
    FTS5's own escape for a literal quote inside a quoted term.

    **TASK-19558 made this the ONE escape for the whole app, not just the
    widening forms.** The FTS5 string-literal rule does not care whether
    what it wraps is one token or a whole phrase -- ``"a b"`` is a phrase
    query and ``"a"`` a single-token one, but both are the same literal
    with the same escaping -- so ``quote_fts5_phrase`` below is a
    same-named alias rather than a second implementation. Before that task
    this repo carried **six** spellings of the escape (four correct, two
    that omitted the doubling entirely and raised ``OperationalError(
    'unterminated string')`` on any query containing a ``"``), plus three
    ``ChaChaNotes_DB`` search methods that computed a ``safe_search_term``
    and then bound the RAW one -- protection that read as protection in
    review and reached no query. ``Tests/Utils/
    test_fts5_quoting_adoption_census.py`` is the guard that keeps a
    seventh spelling from being written: any ``.replace('"', '""')``
    outside this module fails it.

    Args:
        token: One raw token from ``fts5_query_tokens``, or a whole raw
            user-typed search phrase (see ``quote_fts5_phrase``).

    Returns:
        The token as a quoted FTS5 term.
    """
    return '"{}"'.format(token.replace('"', '""'))


#: The whole-phrase reading of the same escape. Bound to the identical
#: function on purpose: a caller quoting a whole user-typed search box
#: value wants "this text, literally", which is exactly what an FTS5
#: string literal is, and giving it a second implementation is how the
#: two-that-omit-the-doubling variants got written in the first place. The
#: name exists so a phrase call site reads honestly instead of claiming to
#: quote a "token" it never tokenized.
quote_fts5_phrase = quote_fts5_token


def quote_fts5_prefix(text: str) -> str:
    """Quote ``text`` as an FTS5 phrase-PREFIX term: ``foo"bar`` -> ``"foo""bar"*``.

    The star goes **outside** the quotes: FTS5 reads ``"tok"*`` as "a
    phrase whose last token is a prefix", while a star inside the quotes is
    an inert character in the literal (the tokenizer drops it) and would
    silently reduce this to a plain phrase match. Four call sites had this
    spelled out longhand as ``f'"{term.replace(chr(34), chr(34) * 2)}"*'``
    before TASK-19558; they now share this one.

    Args:
        text: Raw user-typed search text.

    Returns:
        The FTS5 phrase-prefix MATCH expression.
    """
    return f"{quote_fts5_token(text)}*"


def is_fts5_stopword(token: str) -> bool:
    """Whether a raw query token is a function word.

    Compared on the token's alphanumeric runs, because that is how FTS5
    reads a quoted token: ``About,`` indexes and matches exactly as
    ``about``, so the trimmer must see them as the same word. A token with
    more than one run (``read-only``) is content, never a stopword.

    Args:
        token: One raw token from ``fts5_query_tokens``.

    Returns:
        True when the token is a single alphanumeric run listed in
        ``FTS5_STOPWORDS``.
    """
    runs = fts5_token_runs(token)
    return len(runs) == 1 and runs[0].lower() in FTS5_STOPWORDS


def fts5_query_tokens(query: str) -> List[str]:
    """Tokenize a raw user query for MATCH construction.

    Tokens are whitespace-separated runs containing at least one
    alphanumeric character. FTS5's default tokenizer indexes only
    alphanumeric runs, so a pure-punctuation token ("!!!") can never match
    anything and is dropped rather than carried as a no-op that would make
    the prefix form ``"!!!"*`` -- an expression matching nothing, ANDed with
    the terms that would otherwise have matched.

    Length bounding is deliberately NOT done here: the engine bounds at its
    own configured ``MAX_QUERY_LENGTH`` before calling in, and the Library
    validates at ``LIBRARY_RAG_QUERY_MAX_LENGTH`` at its own boundary. A
    third limit baked in here would silently override whichever caller's
    limit is larger.

    Args:
        query: Raw search query (already length-bounded by the caller).

    Returns:
        The query's searchable tokens, in query order; empty when the query
        is empty, whitespace-only or all punctuation.
    """
    if not query:
        return []
    return [token for token in query.split() if any(ch.isalnum() for ch in token)]


def build_prefix_match_expression(tokens: Iterable[str]) -> str:
    """Build the PREFIX form: content tokens as prefix terms, implicitly ANDed.

    The widening form both ``and_then_prefix`` implementations run for a
    sub-leg whose AND primary returned zero rows. Function words are trimmed
    (see ``FTS5_STOPWORDS`` for why that matters more here than anywhere
    else), each surviving token is quoted, and the star goes **outside** the
    quotes: FTS5 reads ``"tok"*`` as "a phrase whose last token is a
    prefix", while a star inside the quotes is an inert character in the
    literal (the tokenizer drops it) and would silently reduce this to a
    plain trimmed AND.

    Passing already-trimmed tokens is safe: the trim is idempotent, which is
    what lets the engine hand in its ``content_tokens`` and the Library hand
    in its full token list and both get the same form.

    Args:
        tokens: Raw query tokens, e.g. from ``fts5_query_tokens``.

    Returns:
        The prefix expression, or ``""`` when trimming leaves nothing.
        ``""`` means "no rows" and callers must skip the FTS5 query
        entirely -- never fall back to a wider expression, because the only
        query left to run would be an unbounded stopword prefix.
    """
    return " ".join(
        f"{quote_fts5_token(token)}*"
        for token in tokens
        if not is_fts5_stopword(token)
    )


def fts5_query_is_searchable(query: object) -> bool:
    """Whether a raw user query can produce a MATCH expression at all.

    Three rejections, each for a concrete reason rather than defensiveness:

    * **Not a string / empty.** ``None`` reaches these seams from callers
      that pass an unset filter through unchanged. Before TASK-19558 the
      raw value went to ``sqlite3`` and produced ``[]`` or a wrapped DB
      error; quoting it produced a bare ``AttributeError: 'NoneType' object
      has no attribute 'replace'`` instead -- an unwrapped exception type no
      caller of a DB search method is written against.

    * **Contains a NUL.** This is the one that is not obvious.
      ``sqlite3`` hands a bound TEXT parameter to SQLite as a C string, so
      the value is truncated at the first NUL -- **after** we have quoted
      it. ``a\\x00b`` becomes the literal ``"a\\x00b"``, SQLite sees ``"a``,
      and FTS5 raises ``unterminated string``. No amount of correct quoting
      can survive that, because the closing quote is on the far side of the
      truncation point. Raw binds were unaffected only by luck: the
      truncated value ``a`` was still a syntactically valid bareword.
      ``Notes/file_notes_replica.search`` has guarded this since it was
      written (``if "\\x00" in query: return []``); this is that rule, moved
      to where every seam can share it.

    * **No alphanumeric character.** FTS5's ``unicode61`` tokenizer indexes
      only alphanumeric runs, so a pure-punctuation query cannot match
      anything. Answering ``""`` lets the caller skip the query rather than
      run one that is guaranteed to return nothing.

    Args:
        query: The raw value a search seam was handed.

    Returns:
        True when a MATCH expression built from it can be executed.
    """
    if not isinstance(query, str) or not query:
        return False
    if "\x00" in query:
        return False
    return any(ch.isalnum() for ch in query)


def build_and_match_expression(tokens: Iterable[str]) -> str:
    """Build the AND form: every token quoted, joined by FTS5's implicit AND.

    **This is the form a plain-text search box wants**, and getting that
    wrong is the defect TASK-3995 fixed once already and TASK-19558's first
    round re-created. Wrapping a whole multi-word query in ONE pair of
    quotes makes it a PHRASE query, which requires the words to appear as a
    contiguous run -- strictly stronger than "all of these words appear",
    not equivalent to it. Measured on a real corpus: ``dragon lore`` matched
    a record titled "dragon lore adjacent" but not one titled "lore of the
    dragon reversed", halving recall at eight seams while looking like a
    pure safety change.

    Quoting each token INDIVIDUALLY keeps the whole injection-safety
    property -- a quoted token is a string literal with no operator
    semantics, so a typed ``OR``, a typed ``NEAR``, a ``col:`` filter and a
    stray ``"`` are all inert -- while restoring AND-of-terms recall.
    ``RAG_Search/simplified/rag_service._escape_fts5_query`` has built this
    exact expression since TASK-3995 and now delegates the escape here; this
    is the shared definition of the whole form.

    Args:
        tokens: Raw query tokens, normally from ``fts5_query_tokens``.

    Returns:
        The AND expression, or ``""`` when there are no tokens. ``""`` means
        "no rows": callers must skip the query rather than run ``MATCH ''``,
        which FTS5 rejects as a syntax error.
    """
    return " ".join(quote_fts5_token(token) for token in tokens)


def build_and_match_query(query: object) -> str:
    """Raw user text -> the AND form, or ``""`` when it cannot be searched.

    The entry point for a plain-text search seam. Combines
    ``fts5_query_is_searchable`` (None / NUL / punctuation-only) with
    ``fts5_query_tokens`` + ``build_and_match_expression``, so a seam needs
    exactly two lines: build, and return no rows on ``""``.

    Args:
        query: The raw value the search seam was handed; need not be a
            string.

    Returns:
        An executable FTS5 MATCH expression, or ``""`` meaning "no rows".
    """
    if not fts5_query_is_searchable(query):
        return ""
    return build_and_match_expression(fts5_query_tokens(str(query)))


def build_phrase_match_query(query: object) -> str:
    """Raw user text -> ONE quoted phrase, or ``""`` when it cannot be searched.

    The deliberate counterpart to ``build_and_match_query``, for the seams
    whose documented behaviour is phrase matching and always was -- notes,
    keywords, keyword collections, datasets. TASK-19558 kept those as
    phrases on purpose: they matched phrases before the task too, so
    widening them to AND would be an unmeasured behaviour change smuggled in
    beside a security fix. The rule the task applied, stated once here so
    the next reader does not have to infer it from thirteen call sites:
    **a seam that bound its query RAW (FTS5 implicit AND) gets
    ``build_and_match_query``; a seam that already bound a quoted phrase
    keeps ``build_phrase_match_query``.**

    Args:
        query: The raw value the search seam was handed.

    Returns:
        The whole query as one quoted FTS5 phrase, or ``""`` meaning
        "no rows".
    """
    if not fts5_query_is_searchable(query):
        return ""
    return quote_fts5_token(str(query))
