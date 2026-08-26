"""Per-token FTS5 quoting: multi-token queries must match non-contiguous text.

TASK-3995: `RAGService._escape_fts5_query` wrapped the ENTIRE query in a
single pair of double quotes, which FTS5 parses as a *phrase* query --
every token must appear as one contiguous run. That is strictly stronger
than AND-of-terms, not equivalent to it. Verified directly against a real
corpus document and the exact SQL join the engine uses (see task-3995's
description): the phrase form of a multi-token query matched 0 rows
against a document that contains the relevant terms but not as one
contiguous run.

The whole-query quoting was not an accident -- it existed to keep FTS5
from parsing a bare token as column-filter syntax. The documented
incident: the unquoted token `Obsidian-3` raises
`OperationalError('no such column: 3')` because FTS5's query grammar
treats the bareword after the hyphen specially. Any fix must drop phrase
semantics while keeping that safety net.

These tests build a plain sqlite3 stdlib in-memory FTS5 table (no app
DBs) so the MATCH behavior under test is the real FTS5 engine, not a
mock. `RAGService._escape_fts5_query` itself needs no DB at all -- but
`_make_service` seeds a real media DB anyway, and that is load-bearing
for exactly one test. `test_all_punctuation_query_short_circuits` spies
on `_perform_fts5_search` to prove the production short-circuit fires;
with no `media_db_path` configured, the media sub-leg bails on an
unresolvable database BEFORE reaching that call, so the spy could never
fire and the assertion held whether or not the guard existed. Verified:
with the short-circuit neutered (`if False:`) the unseeded file stayed
5/5 green; seeded, the same mutation yields three `_perform_fts5_search`
calls (one per retry attempt) and reds the test. Do not drop the seed.
"""
import asyncio
import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.RAG_Search.simplified.citations import CitationType
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    MAX_CITATION_MATCHES,
    RAGService,
)


def _seed_media_db(tmp_path):
    """A real MediaDatabase with one row (FTS populated by its own triggers).

    Same pattern as `test_keyword_leg_chacha._seed_media`.
    """
    db_path = tmp_path / "tldw_cli_media_v2.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test_fts5_escaping")
    try:
        media_id, _, message = db.add_media_with_keywords(
            title="Lathe Maintenance Log",
            content="The Obsidian-3 lathe shows spindle runout under load.",
            media_type="article",
            author="Tester",
            url="https://example.com/lathe-maintenance-log",
        )
        assert media_id is not None, f"media seed failed: {message}"
    finally:
        db.close_connection()
    return db_path


def _make_service(tmp_path):
    """A RAGService whose media sub-leg can actually reach FTS5.

    `media_db_path` is seeded deliberately -- see the module docstring:
    without it the media sub-leg returns early and the short-circuit spy
    in `test_all_punctuation_query_short_circuits` is unreachable.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    cfg.search.media_db_path = Path(_seed_media_db(tmp_path))
    return RAGService(cfg)


def _fts5_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, content)")
    return conn


def _insert(conn, title, content):
    conn.execute(
        "INSERT INTO docs(title, content) VALUES (?, ?)", (title, content)
    )
    conn.commit()


def _match(conn, query):
    return conn.execute(
        "SELECT rowid FROM docs WHERE docs MATCH ?", (query,)
    ).fetchall()


def test_multi_token_query_matches_non_contiguous_tokens(tmp_path):
    """The relevant tokens are all present but NOT contiguous in the seed
    document -- "Obsidian-3" ... "lathe shows" ... "spindle runout" -- so a
    phrase query over the whole 3-token search string must NOT match it,
    while an AND-of-terms query must.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Lathe Maintenance Log",
        "The Obsidian-3 lathe shows spindle runout under load.",
    )
    query = "Obsidian-3 spindle runout"

    # Sanity check on the OLD behavior: wrapping the whole query in one
    # pair of quotes is a phrase query requiring the 3 tokens contiguous,
    # which they are not in the seed doc.
    old_phrase_form = f'"{query}"'
    assert _match(conn, old_phrase_form) == [], (
        "test setup invalid: the seed doc must NOT satisfy the old "
        "whole-query phrase form"
    )

    service = _make_service(tmp_path)
    escaped = service._escape_fts5_query(query)
    assert escaped, "escaped query must not be empty for a real multi-token query"

    rows = _match(conn, escaped)
    assert len(rows) == 1, (
        f"per-token AND-of-terms form must match the non-contiguous seed "
        f"doc; _escape_fts5_query({query!r}) = {escaped!r}, rows={rows}"
    )
    conn.close()


def test_hyphen_numeric_token_still_safe(tmp_path):
    """A bare hyphenated-numeric token is unsafe unquoted -- FTS5's query
    grammar raises `OperationalError('no such column: 3')` on it (the
    documented incident this task's safety property guards against). The
    escaped form must execute cleanly and still match the seed doc.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Lathe Maintenance Log",
        "The Obsidian-3 lathe was serviced today.",
    )
    query = "Obsidian-3"

    # Sanity check: the raw, unquoted token really is unsafe.
    with pytest.raises(sqlite3.OperationalError):
        _match(conn, query)

    service = _make_service(tmp_path)
    escaped = service._escape_fts5_query(query)
    rows = _match(conn, escaped)
    assert len(rows) == 1, f"escaped={escaped!r} rows={rows}"
    conn.close()


def test_embedded_quotes_are_doubled_and_safe(tmp_path):
    """A query containing a raw double-quote character must not break the
    FTS5 MATCH expression -- the quote is doubled per FTS5's own escaping
    convention for a literal quote inside a quoted term.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Quote Test",
        'The technician wrote "runout confirmed" in the log.',
    )
    query = 'confirmed" runout'

    service = _make_service(tmp_path)
    escaped = service._escape_fts5_query(query)

    # Must execute without raising -- the safety property under test.
    rows = _match(conn, escaped)
    assert rows == [(1,)], f"escaped={escaped!r} rows={rows}"
    conn.close()


def test_single_token_behavior_unchanged(tmp_path):
    """A single-token query is a degenerate case where phrase quoting and
    per-token quoting produce the identical MATCH expression -- this must
    keep working exactly as before.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Wombat Field Notes",
        "Field notes mentioning wombat behavior in the wild.",
    )

    service = _make_service(tmp_path)
    escaped = service._escape_fts5_query("wombat")
    assert escaped == '"wombat"'

    rows = _match(conn, escaped)
    assert len(rows) == 1
    conn.close()


def test_all_punctuation_query_short_circuits(tmp_path):
    """A query with no alphanumeric content (FTS5's default tokenizer
    indexes nothing for it) must escape to "" -- and the keyword search
    must recognize "" as "no results" and skip the FTS5 call entirely
    rather than execute a query that can only ever match nothing.
    """
    service = _make_service(tmp_path)
    assert service._escape_fts5_query("!!! ...") == ""

    calls = []
    original_perform = service._perform_fts5_search

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original_perform(*args, **kwargs)

    service._perform_fts5_search = _spy

    results = asyncio.run(service._keyword_search("!!! ...", top_k=5))
    assert results == []
    assert calls == [], (
        "an all-punctuation query must short-circuit before the FTS5 "
        f"call is ever made; calls={calls}"
    )


# --- Citation spans must use the SAME tokenization as the MATCH expression ---
#
# The per-token AND-of-terms fix above admits documents whose query tokens
# are scattered. The citation builder still looked for the RAW query as one
# contiguous, case-insensitive substring -- the phrase assumption the fix
# had just deleted -- so exactly the rows the fix newly admitted came back
# with `citations=[]`. Both halves now read one shared token list
# (`_fts5_query_tokens`), which is what keeps them from drifting again.


def test_one_tokenization_feeds_both_the_match_expression_and_the_spans(tmp_path):
    """The MATCH expression is built from the shared token list verbatim."""
    service = _make_service(tmp_path)
    query = "  Obsidian-3 !!! runout "

    tokens = service._fts5_query_tokens(query)
    assert tokens == ["Obsidian-3", "runout"], tokens
    # Punctuation-only tokens are dropped identically on both sides, and the
    # MATCH expression is exactly those tokens quoted.
    assert service._escape_fts5_query(query) == '"Obsidian-3" "runout"'


def test_non_contiguous_multi_token_match_still_yields_citations(tmp_path):
    """The regression Qodo flagged: scattered tokens must still cite.

    The seed doc ("The Obsidian-3 lathe shows spindle runout under load.")
    contains all three query tokens but never contiguously, so the old
    raw-query substring lookup found nothing and the row -- newly reachable
    thanks to TASK-3995 -- arrived with no citations at all.
    """
    service = _make_service(tmp_path)
    query = "Obsidian-3 spindle runout"

    results = asyncio.run(
        service._keyword_search(query, top_k=5, include_citations=True)
    )
    assert results, "the AND-of-terms form must match the seeded media doc"

    row = results[0]
    assert row.citations, (
        "a keyword row whose tokens are scattered must still carry citation "
        f"spans; document={row.document!r}"
    )

    sliced = []
    for citation in row.citations:
        assert 0 <= citation.start_char < citation.end_char <= len(row.document), (
            f"citation offsets must index the returned content: "
            f"{citation.start_char}-{citation.end_char} into "
            f"{len(row.document)} chars"
        )
        span_text = row.document[citation.start_char : citation.end_char]
        assert span_text, "a citation span must not slice to empty text"
        sliced.append(span_text.lower())

    # Both ends of the scattered match are evidenced, each at its real offset.
    assert "obsidian-3" in sliced, sliced
    assert any("runout" in text for text in sliced), sliced


def test_contiguous_match_still_cites_the_whole_phrase(tmp_path):
    """Old behavior stays pinned: adjacent tokens cite as ONE span.

    "spindle runout" is contiguous in the seed doc, and the citation for it
    must still be the single whole-phrase span (offsets bracketing both
    tokens, `EXACT` at full confidence) that the pre-TASK-3995 raw-query
    lookup produced -- not two half-citations.
    """
    service = _make_service(tmp_path)

    results = asyncio.run(
        service._keyword_search("spindle runout", top_k=5, include_citations=True)
    )
    assert results, "the seeded media doc must match"

    row = results[0]
    spans = [
        row.document[c.start_char : c.end_char].lower() for c in row.citations
    ]
    assert "spindle runout" in spans, spans

    whole_phrase = next(
        c
        for c in row.citations
        if row.document[c.start_char : c.end_char].lower() == "spindle runout"
    )
    assert whole_phrase.match_type == CitationType.EXACT
    assert whole_phrase.confidence == 1.0


def test_span_cap_still_evidences_a_rare_second_token(tmp_path):
    """The citation cap must not be spent entirely on one repeated token.

    "alpha" occurs four times before "beta" occurs once. Taking the first
    `MAX_CITATION_MATCHES` spans in document order would cite "alpha" three
    times and never show that "beta" -- the token that actually made this
    an AND match -- is present at all.
    """
    service = _make_service(tmp_path)
    item = {
        "id": 11,
        "title": "Repeated Token Doc",
        "content": (
            "alpha one. alpha two. alpha three. alpha four. "
            "and much later, beta."
        ),
    }

    result = asyncio.run(
        service._create_keyword_result_with_citations(item, "alpha beta", None)
    )
    assert result is not None
    assert 0 < len(result.citations) <= MAX_CITATION_MATCHES

    spans = [
        result.document[c.start_char : c.end_char].lower() for c in result.citations
    ]
    assert "beta" in spans, spans
    assert "alpha" in spans, spans
    # Offsets index the returned content, not some other string.
    for citation in result.citations:
        assert result.document[citation.start_char : citation.end_char] == (
            citation.metadata["match_text"]
        )


def test_keyword_row_matching_only_on_title_is_never_citation_empty(tmp_path):
    """`media_fts` indexes title AND content, so a row can match on its
    title alone. Its content then holds no span at all -- the fallback
    citation is what keeps such a keyword-backed row from reaching the
    evidence list with nothing to show.
    """
    service = _make_service(tmp_path)
    item = {
        "id": 7,
        "title": "Wombat Field Notes",
        "content": "Burrow geometry, cross-sections and tunnel branching.",
    }

    result = asyncio.run(
        service._create_keyword_result_with_citations(item, "wombat", None)
    )
    assert result is not None
    assert result.citations, (
        "a keyword-backed row must never come back citation-empty; the "
        "title-only match has no in-content span to point at"
    )
    fallback = result.citations[0]
    assert 0 <= fallback.start_char <= fallback.end_char <= len(result.document)


# --- The same safety property, through every MATCH construction ------------
#
# TASK-15400 puts three more MATCH constructions behind
# `SearchConfig.fts_match_construction` (TASK-15700 adds two more, covered
# in their own section below), and one of them (`and_then_or`) issues a
# SECOND query the tests above never reach. The injection property
# is a property of the EXPRESSION BUILDER, not of the AND join, so every
# case above is re-run through the OR form and through the fallback path:
# an operator-shaped token, a column-filter attempt, a hyphenated numeric,
# and an embedded quote must all stay inert literals no matter which
# construction produced the expression. Mutation check: dropping the
# per-token quoting from the OR join (joining bare tokens with " OR ")
# reds `test_hostile_tokens_stay_inert_in_the_or_form` with
# `OperationalError`.


HOSTILE_QUERIES = [
    # The documented incident: unquoted, FTS5 reads `3` as a column name.
    "Obsidian-3 lathe",
    # A column-filter attempt.
    "content:wombat lathe",
    # FTS5 operators typed as ordinary words.
    "lathe AND OR NOT spindle",
    "lathe NEAR spindle",
    # A prefix-operator attempt and a bare wildcard.
    "lathe* spindle",
    # An embedded quote, and a stray closing parenthesis.
    'confirmed" runout',
    "lathe) OR (spindle",
    # A column filter with braces, FTS5's multi-column syntax.
    "{title content}:lathe spindle",
]


@pytest.mark.parametrize("construction", ["or", "and_then_or", "and_stopword_trim"])
@pytest.mark.parametrize("query", HOSTILE_QUERIES)
def test_hostile_tokens_stay_inert_in_the_or_form(tmp_path, construction, query):
    """Every expression a construction can emit must execute cleanly.

    Executed against a real FTS5 table -- the parse is the thing under
    test. Both halves of the pair are checked, so the `and_then_or`
    FALLBACK expression (the one no pre-arc test ever built) is covered.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Lathe Maintenance Log",
        "The Obsidian-3 lathe shows spindle runout under load.",
    )

    service = _make_service(tmp_path)
    service.config.search.fts_match_construction = construction
    primary, fallback = service._fts5_match_expressions(query)

    for expression in (primary, fallback):
        if not expression:
            continue
        # Must not raise: the safety property. (Whether it matches is a
        # relevance question, not a safety one.)
        _match(conn, expression)
    conn.close()


def test_user_typed_operator_words_never_become_operators(tmp_path):
    """`OR` typed by the user is a search term, not a disjunction.

    Unquoted, `lathe OR NOT spindle` is an FTS5 expression whose meaning is
    the user's words rearranged into boolean logic. The only operators in
    an emitted expression are the ones the BUILDER put there: in an AND form
    the user's words stay quoted literals, and in every content-token form
    they are trimmed as the function words they are. Both are checked here,
    and since 2026-08-13 the shipped default exhibits BOTH AT ONCE.
    """
    conn = _fts5_conn()
    _insert(conn, "Operator Words", "The word OR appears here, and NOT much else.")
    _insert(conn, "Lathe Log", "The lathe spindle was replaced.")

    service = _make_service(tmp_path)
    query = "lathe OR NOT spindle"

    # Sanity check: unquoted, FTS5 parses the user's own words as boolean
    # syntax -- here, all the way to a syntax error.
    with pytest.raises(sqlite3.OperationalError):
        _match(conn, query)

    # DISCLOSED ORACLE FLIP #2 (2026-08-13, TASK-15700 Task 4). Three states
    # now, and the third is not a small move from the second:
    #
    #   * pre-15400 the default was `and`, emitting
    #     '"lathe" "OR" "NOT" "spindle"' -- matching NOTHING, since no
    #     document contains the literal words "OR" and "NOT";
    #   * 2026-08-11 the default became `and_stopword_trim`, whose SINGLE
    #     expression dropped both as the function words they are and matched
    #     the lathe log;
    #   * 2026-08-13 the default became `and_then_prefix` by OWNER RULING
    #     (the pre-registered rule's tie-break selected `prefix`; the owner
    #     overrode it for structural self-displacement immunity). Its
    #     PRIMARY is the FULL AND again -- the untrimmed, unmatched form --
    #     and the trimming moved into the per-sub-leg zero-row FALLBACK.
    #
    # So the trim did not disappear from the default; it moved one stage
    # later, and this test now asserts the PAIR. The safety property is
    # untouched throughout and is the reason the pair is safe to state: the
    # only operators in either emitted expression are the ones the BUILDER
    # put there, and in both there are none -- the user's `OR`/`NOT` survive
    # as quoted literals in the primary and are dropped from the fallback.
    primary, fallback = service._fts5_match_expressions(query)
    assert primary == '"lathe" "OR" "NOT" "spindle"'
    assert _match(conn, primary) == [], (
        "the default's primary is the full AND; it must still match nothing "
        "here, which is exactly what makes the fallback fire"
    )
    assert fallback == '"lathe"* "spindle"*'
    assert _match(conn, fallback) == [(2,)], (
        "the fallback is where the shipped default now finds the lathe log"
    )

    # The pre-arc form, still shipped as the `and` construction and as the
    # fail-safe for an unrecognized value -- asked for explicitly. Since
    # 2026-08-13 it is byte-identical to the DEFAULT's primary above, which
    # is the flip stated as an identity rather than as prose.
    service.config.search.fts_match_construction = "and"
    assert service._fts5_match_expressions(query)[0] == (
        '"lathe" "OR" "NOT" "spindle"'
    )
    assert _match(conn, '"lathe" "OR" "NOT" "spindle"') == []

    # The construction that WAS the default until 2026-08-13: one expression,
    # trimmed, no fallback. The state the flip moved away from.
    service.config.search.fts_match_construction = "and_stopword_trim"
    assert service._fts5_match_expressions(query) == ('"lathe" "spindle"', None)
    assert _match(conn, '"lathe" "spindle"') == [(2,)]

    service.config.search.fts_match_construction = "or"
    primary, _fallback = service._fts5_match_expressions(query)
    assert primary == '"lathe" OR "spindle"'
    assert _match(conn, primary) == [(2,)]
    conn.close()


def test_the_fallback_path_runs_a_hostile_query_without_raising(tmp_path, monkeypatch):
    """End to end through the media sub-leg: a hostile query whose AND finds
    nothing must reach the fallback, and the fallback must actually EXECUTE
    against SQLite.

    Asserting only "returns a list" would be vacuous -- every sub-leg
    swallows its own errors, so a fallback that raised would still produce
    `[]`. The spy records the expressions that reached the pooled FTS5
    execution, and the row it returns is the proof the hostile OR form both
    parsed and matched.
    """
    service = _make_service(tmp_path)
    service.config.search.fts_match_construction = "and_then_or"

    executed = []
    original = RAGService._perform_fts5_search

    def spy(self, pool, query, limit, allowed_ids=None):
        rows = original(self, pool, query, limit, allowed_ids)
        executed.append(self._fts5_match_expressions(query))
        return rows

    monkeypatch.setattr(RAGService, "_perform_fts5_search", spy)

    # No document contains "quokka", so the AND is empty and the OR form
    # runs -- carrying the column-filter attempt and the embedded quote.
    query = 'quokka content:wombat confirmed" lathe'
    results = asyncio.run(service._keyword_search(query, top_k=5))

    assert executed, "the media sub-leg never reached FTS5"
    primary, fallback = executed[0]
    assert " OR " not in primary
    assert fallback == '"quokka" OR "content:wombat" OR "confirmed""" OR "lathe"'
    # The seeded doc contains "lathe" but not "quokka": only the fallback
    # can return it, so a row here means the hostile OR form executed.
    assert [row.metadata["fts_match"] for row in results] == ["or"], results


def test_an_all_stopword_query_never_emits_an_empty_match_expression(tmp_path):
    """An empty MATCH string is an FTS5 syntax error, not "no results"."""
    conn = _fts5_conn()
    _insert(conn, "Anything", "Any content at all.")
    service = _make_service(tmp_path)

    for construction in ("and", "and_stopword_trim", "and_then_or", "and_then_prefix"):
        service.config.search.fts_match_construction = construction
        primary, fallback = service._fts5_match_expressions("what about the")
        assert primary, f"{construction} emitted an empty primary expression"
        _match(conn, primary)
        assert fallback is None, construction

    # `or` and `prefix` are the constructions whose answer is honestly "no
    # rows" -- and "" is the existing skip contract, never a query. (A
    # stopword PREFIX, `"the"*`, is junk that matches most of a corpus, so
    # trimming is not optional under the prefix form either.)
    for construction in ("or", "prefix"):
        service.config.search.fts_match_construction = construction
        assert service._fts5_match_expressions("what about the") == (
            "",
            None,
        ), construction


# --- the same safety property, through the PREFIX form (TASK-15700) --------
#
# The two constructions the 15700 re-run pre-registers (`prefix`,
# `and_then_prefix`) emit a term shape no earlier case covers: the quoted
# literal with FTS5's star appended OUTSIDE the closing quote. The families
# above are re-run through both of them (primary AND fallback), plus the two
# failure modes specific to the star's placement -- inside the quotes it is
# an inert character in the literal, and a bare `*` with no term in front of
# it is a syntax error.


@pytest.mark.parametrize("construction", ["prefix", "and_then_prefix"])
@pytest.mark.parametrize("query", HOSTILE_QUERIES)
def test_hostile_tokens_stay_inert_in_the_prefix_form(tmp_path, construction, query):
    """Every expression a prefix construction can emit must execute cleanly.

    Same discipline as the OR-form case above and the same real FTS5 parse:
    both halves of the pair are checked, so `and_then_prefix`'s FALLBACK
    expression -- the one carrying the stars -- is covered as well as its
    plain-AND primary. Mutation check: dropping the per-token quoting from
    the prefix join (`lathe*` instead of `"lathe"*`) reds this with
    `OperationalError` on the `Obsidian-3` case.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Lathe Maintenance Log",
        "The Obsidian-3 lathe shows spindle runout under load.",
    )

    service = _make_service(tmp_path)
    service.config.search.fts_match_construction = construction
    primary, fallback = service._fts5_match_expressions(query)

    for expression in (primary, fallback):
        if not expression:
            continue
        _match(conn, expression)
    conn.close()


def test_the_prefix_star_goes_outside_the_quotes(tmp_path):
    """Star placement is a BEHAVIOURAL fact, not a formatting preference.

    Outside the closing quote the star is FTS5's prefix operator: `"spind"*`
    is "a phrase whose last token starts with spind". Inside it, the star is
    just a character in the literal -- FTS5's own tokenizer drops it -- so
    `"spind*"` is the plain term `spind`, which matches nothing here. The
    construction would silently degrade to the trimmed AND it is supposed to
    widen, with no error anywhere to notice.
    """
    conn = _fts5_conn()
    _insert(
        conn,
        "Lathe Maintenance Log",
        "The Obsidian-3 lathe shows spindle runout under load.",
    )
    service = _make_service(tmp_path)
    service.config.search.fts_match_construction = "prefix"

    primary, fallback = service._fts5_match_expressions("spind runou")
    assert primary == '"spind"* "runou"*'
    assert fallback is None
    assert _match(conn, primary) == [(1,)]

    # The mutation, spelled out: the same tokens with the star inside the
    # quotes parse fine and match NOTHING.
    assert _match(conn, '"spind*" "runou*"') == []
    conn.close()


def test_a_bare_asterisk_token_never_reaches_the_prefix_expression(tmp_path):
    """`*` alone is FTS5 syntax with nothing to apply it to.

    The tokenizer drops any token with no alphanumeric character, so a typed
    `*` cannot become a term at all -- and a query that is nothing but
    wildcards short-circuits on `""` rather than emitting `*` or `""*`.
    Checked through both prefix constructions, and executed for the case
    that does emit an expression.
    """
    conn = _fts5_conn()
    _insert(conn, "Wombat log", "The wombat burrow was surveyed at dusk.")
    service = _make_service(tmp_path)

    service.config.search.fts_match_construction = "prefix"
    assert service._fts5_match_expressions("* wombat") == ('"wombat"*', None)
    assert _match(conn, '"wombat"*') == [(1,)]
    assert service._fts5_match_expressions("* **") == ("", None)

    service.config.search.fts_match_construction = "and_then_prefix"
    assert service._fts5_match_expressions("* wombat") == (
        '"wombat"',
        '"wombat"*',
    )
    assert service._fts5_match_expressions("* **") == ("", None)

    # ...and the bare wildcard FTS5 would have choked on, for contrast.
    with pytest.raises(sqlite3.OperationalError):
        _match(conn, "*")
    conn.close()


def test_user_typed_operator_words_never_become_operators_in_the_prefix_form(
    tmp_path,
):
    """`OR`/`NOT` typed by the user stay function words under the star too.

    The prefix form trims them as the stopwords they are, so the emitted
    expression contains no operator the builder did not put there -- and the
    surviving terms are quoted literals with the star outside.
    """
    conn = _fts5_conn()
    _insert(conn, "Lathe Log", "The lathe spindle was replaced.")
    service = _make_service(tmp_path)
    service.config.search.fts_match_construction = "prefix"

    primary, fallback = service._fts5_match_expressions("lathe OR NOT spind")
    assert primary == '"lathe"* "spind"*'
    assert fallback is None
    assert _match(conn, primary) == [(1,)]
    conn.close()


def test_the_prefix_fallback_path_runs_a_hostile_query_without_raising(
    tmp_path, monkeypatch
):
    """End to end through the media sub-leg, the `and_then_prefix` composition.

    Mirrors the `and_then_or` case above and for the same reason: every
    sub-leg swallows its own errors, so "returns a list" would be vacuous.
    The query's AND finds nothing (the seed says "spindle", the query says
    "spind") while carrying the documented injection token `Obsidian-3`, so a
    returned row is proof the hostile PREFIX expression both parsed and
    matched -- and its stamp names the form that did it.
    """
    service = _make_service(tmp_path)
    service.config.search.fts_match_construction = "and_then_prefix"

    executed = []
    original = RAGService._perform_fts5_search

    def spy(self, pool, query, limit, allowed_ids=None):
        rows = original(self, pool, query, limit, allowed_ids)
        executed.append(self._fts5_match_expressions(query))
        return rows

    monkeypatch.setattr(RAGService, "_perform_fts5_search", spy)

    query = "Obsidian-3 spind runou"
    results = asyncio.run(service._keyword_search(query, top_k=5))

    assert executed, "the media sub-leg never reached FTS5"
    primary, fallback = executed[0]
    assert primary == '"Obsidian-3" "spind" "runou"'
    assert fallback == '"Obsidian-3"* "spind"* "runou"*'
    assert [row.metadata["fts_match"] for row in results] == ["prefix"], results
