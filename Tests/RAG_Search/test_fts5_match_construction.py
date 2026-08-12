"""The engine keyword leg's MATCH construction seam (TASK-15400, Tasks 1+4).

`_escape_fts5_query` builds an implicit AND over every query token, which
is why the leg returned zero rows for 40 of the 60 golden queries (the
census in TASK-15400's description). This file pins the SEAM that let the
arc's sweep measure four candidate constructions against each other.

**The default flipped in Task 4 (2026-08-11), and these pins moved with
it.** `SearchConfig.fts_match_construction` was `"and"` (the pre-arc
implicit AND over every token) and is now `"and_stopword_trim"`, the
construction the sweep's matrix chose under the arc's pre-registered rule
(row `and_trim`: leg census 20 → 21 of 53, hybrid prompt/recall
0.000 → 0.200, no gated cell down in any mode, zero extra FTS queries; the
two OR-bearing rows scored 28/29 and were disqualified for losing the
vector-blind fixture's hybrid rescue — see `_escape_fts5_query`'s docstring
for the interleave-displacement mechanism). Two pins below name both states
explicitly: `test_the_shipped_default_is_the_sweeps_winner` (the flip
itself) and `test_and_construction_is_byte_identical_to_the_shipped_escaper`
(same property, now asked for explicitly). Every OTHER pin here sets its
construction by hand and is unaffected by which one ships.

Four properties, each with a mutation that reds it:

* **Expression shape per construction.** `_fts5_match_expressions` returns
  `(primary, fallback | None)`. `and` is byte-identical to
  `_escape_fts5_query`; `and_stopword_trim` ANDs the content tokens (and
  falls back to the FULL AND when trimming empties the query, never to an
  empty MATCH expression -- an FTS5 syntax error); `or` ORs the content
  tokens and returns `""` (= "no rows", the existing skip contract) when
  trimming empties them; `and_then_or` returns both forms.
* **The fallback fires ONLY on zero primary rows.** Counted with a spy on
  the SQL-executing helper, per sub-leg: one AND row means exactly one
  query. Dropping the zero-row guard (running the fallback unconditionally)
  reds the count assertions; dropping the fallback loop entirely reds the
  rescue assertions.
* **Provenance.** Every keyword row carries `metadata["fts_match"]`, naming
  the FORM that matched it and nothing else: `"and"` for an implicit-AND
  expression (full or stopword-trimmed), `"or"` for the content-token OR
  form -- whether that form was reached as `and_then_or`'s fallback or run
  as the `or` construction's primary. Task 2's negative-composition counter
  and Task 5's mechanism prose both read this key, so it must name the form,
  not the position: under the `or` construction NO row is a fallback row,
  and fallback-ness is derivable from (construction, form) whenever it is
  wanted.
* **The construction is in the cache key.** A per-service cache plus a
  runtime-mutable construction is exactly the shape that made TASK-4110's
  fusion sweep report "k doesn't matter". `"and"` renders the key
  byte-identically to the pre-arc rendering (reproduced literally below,
  the way `test_no_allowlist_keeps_the_legacy_key_byte_identical` does it);
  every other construction renders a different key.

Real databases throughout for the behavioural half (a real
`PromptsDatabase`/`CharactersRAGDB`/`MediaDatabase`, their own writers
maintaining the FTS indexes), because the thing under test is FTS5's own
parse of the expression -- a mock would pin nothing.
"""
import asyncio
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    FTS_MATCH_AND,
    FTS_MATCH_CONSTRUCTION_AND,
    FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM,
    FTS_MATCH_CONSTRUCTIONS,
    FTS_MATCH_OR,
    _FTS5_STOPWORDS,
    RAGService,
)
from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache

CLIENT_ID = "test_fts_match_construction"


# --- fixtures ---------------------------------------------------------------


def _make_service(
    construction=None,
    media_db_path=None,
    chachanotes_db_path=None,
    prompts_db_path=None,
):
    """A RAGService with the in-memory vector store and mock embeddings."""
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    if construction is not None:
        cfg.search.fts_match_construction = construction
    if media_db_path is not None:
        cfg.search.media_db_path = Path(media_db_path)
    if chachanotes_db_path is not None:
        cfg.search.chachanotes_db_path = Path(chachanotes_db_path)
    if prompts_db_path is not None:
        cfg.search.prompts_db_path = Path(prompts_db_path)
    return RAGService(cfg)


def _seed_prompts(tmp_path, rows, name="prompts.db"):
    """A real PromptsDatabase (its writer maintains `prompts_fts`)."""
    db_path = tmp_path / name
    db = PromptsDatabase(db_path, client_id=CLIENT_ID)
    ids = {}
    try:
        for prompt_name, system_prompt in rows:
            prompt_id, _uuid, message = db.add_prompt(
                name=prompt_name,
                author=None,
                details=None,
                system_prompt=system_prompt,
            )
            assert prompt_id is not None, f"prompt seed failed: {message}"
            ids[prompt_name] = prompt_id
    finally:
        db.close_connection()
    return db_path, ids


def _seed_media(tmp_path, rows, name="tldw_cli_media_v2.db"):
    db_path = tmp_path / name
    db = MediaDatabase(db_path=str(db_path), client_id=CLIENT_ID)
    try:
        for title, content in rows:
            media_id, _, message = db.add_media_with_keywords(
                title=title,
                content=content,
                media_type="article",
                author="Tester",
                url=f"https://example.com/{title.replace(' ', '-').lower()}",
            )
            assert media_id is not None, f"media seed failed: {message}"
    finally:
        db.close_connection()
    return db_path


def _seed_notes(tmp_path, rows, name="chachanotes.db"):
    db_path = tmp_path / name
    db = CharactersRAGDB(db_path, client_id=CLIENT_ID)
    try:
        for title, content in rows:
            db.add_note(title=title, content=content)
    finally:
        db.close_connection()
    return db_path


def _seed_conversation(tmp_path, title, messages, name="chachanotes.db"):
    """A real conversation with real messages (`messages_fts` is the index).

    Same shape as `test_keyword_leg_chacha._add_conversation`.
    """
    db_path = tmp_path / name
    db = CharactersRAGDB(db_path, client_id=CLIENT_ID)
    try:
        conv_id = db.add_conversation({"title": title})
        assert conv_id, "conversation seed failed"
        for sender, content in messages:
            assert db.add_message(
                {"conversation_id": conv_id, "sender": sender, "content": content}
            ), "message seed failed"
    finally:
        db.close_connection()
    return db_path, conv_id


@contextmanager
def _captured_warnings():
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# The seed prompt matches "wombat"/"burrow" but carries neither "template"
# nor "checklist" -- so an AND over the natural-language query is empty
# while the content-token OR still finds it. That gap IS the defect
# TASK-15400 measured, reproduced at fixture scale.
PROMPT_ROWS = [
    (
        "Wombat shift handover",
        "Summarise the wombat burrow inspection log for the incoming supervisor.",
    ),
    (
        "Kiln firing schedule",
        "Draft the ceramic kiln firing schedule for the studio.",
    ),
]
AND_HIT_QUERY = "wombat burrow"                      # every token present
OR_ONLY_QUERY = "how does the wombat template work"  # "template"/"work" absent


# --- expression shape -------------------------------------------------------


def test_the_shipped_default_is_the_sweeps_winner(tmp_path):
    """DISCLOSED ORACLE FLIP (2026-08-11, TASK-15400 Task 4, sweep row
    `and_trim`): the default was `"and"` and is now `"and_stopword_trim"`.

    Both states named on purpose. `"and"` was the pre-arc construction (the
    implicit AND over EVERY token); `"and_stopword_trim"` is what the arc's
    construction matrix chose under its pre-registered rule — census 20 → 21
    of 53, hybrid prompt/recall 0.000 → 0.200, no cell down in any mode,
    zero extra FTS queries. This assertion is the flip: a default reverted
    to `"and"` reds it (and reds the gated prompt pin in
    `Tests/RAG_Eval/test_fixture_authoring_probe.py`).
    """
    service = _make_service()
    assert (
        service.config.search.fts_match_construction
        == FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM
    )

    # ...and the winner's shape at the DEFAULT, not merely its name: the
    # function word goes, the content tokens stay ANDed, no fallback query.
    assert service._fts5_match_expressions("notes about the vendor") == (
        '"notes" "vendor"',
        None,
    )


def test_and_construction_is_byte_identical_to_the_shipped_escaper(tmp_path):
    """`and` still produces exactly the pre-arc MATCH expression.

    DISCLOSED (2026-08-11): this used to be measured at the DEFAULT
    construction, which was `"and"`; it now asks for `"and"` explicitly. The
    property is unchanged and still load-bearing — `"and"` is what
    `_resolved_fts_match_construction` degrades an unknown value to, and
    what `and_stopword_trim` itself falls back to when trimming empties the
    query, so the byte-identity is a live path, not a historical one.
    """
    service = _make_service(construction=FTS_MATCH_CONSTRUCTION_AND)

    for query in ("wombat", "Obsidian-3 spindle runout", "notes about the vendor"):
        primary, fallback = service._fts5_match_expressions(query)
        assert primary == service._escape_fts5_query(query), query
        assert fallback is None, query


def test_and_stopword_trim_ands_only_the_content_tokens():
    """`pm-vendor-chaser`'s shape: one function word blocks the whole AND."""
    service = _make_service(construction="and_stopword_trim")

    primary, fallback = service._fts5_match_expressions("notes about the vendor")
    assert primary == '"notes" "vendor"'
    assert fallback is None


def test_and_stopword_trim_falls_back_to_the_full_and_when_trimming_empties():
    """An all-stopword query must never produce an empty MATCH expression."""
    service = _make_service(construction="and_stopword_trim")

    primary, fallback = service._fts5_match_expressions("what about the")
    assert primary == '"what" "about" "the"'
    assert fallback is None


def test_or_construction_ors_the_content_tokens():
    """Quoted tokens joined by ` OR ` -- FTS5's own disjunction syntax."""
    service = _make_service(construction="or")

    primary, fallback = service._fts5_match_expressions("notes about the vendor")
    assert primary == '"notes" OR "vendor"'
    assert fallback is None


def test_or_construction_returns_no_rows_when_trimming_empties_the_query():
    """Honest emptiness, never a syntax error: `""` is the existing "skip".

    A raw OR over every token would match every document containing "the";
    trimming is what keeps the OR form from flooding fusion with junk. When
    trimming leaves nothing, the answer is no rows.
    """
    service = _make_service(construction="or")

    assert service._fts5_match_expressions("what about the") == ("", None)


def test_and_then_or_returns_the_and_primary_and_the_or_fallback():
    """The data's own suggestion: widen only where AND returns nothing."""
    service = _make_service(construction="and_then_or")

    primary, fallback = service._fts5_match_expressions("notes about the vendor")
    assert primary == '"notes" "about" "the" "vendor"'
    assert fallback == '"notes" OR "vendor"'


def test_and_then_or_has_no_fallback_when_every_token_is_a_stopword():
    """Nothing to widen to -- and an empty MATCH expression is a syntax error."""
    service = _make_service(construction="and_then_or")

    assert service._fts5_match_expressions("what about the") == (
        '"what" "about" "the"',
        None,
    )


def test_and_then_or_has_no_redundant_fallback_when_the_forms_are_identical():
    """A one-token OR IS the one-token AND: re-running it costs a query and
    can only return the same zero rows."""
    service = _make_service(construction="and_then_or")

    assert service._fts5_match_expressions("wombat") == ('"wombat"', None)

    # Stopwords AROUND a single content token are a different case: the
    # trimmed OR is strictly weaker than the AND (it drops the function
    # words the AND still demands), so the fallback is real work --
    # `pm-vendor-chaser`'s exact shape, the one query the census measured as
    # blocked solely by a function word.
    assert service._fts5_match_expressions("what about the wombat") == (
        '"what" "about" "the" "wombat"',
        '"wombat"',
    )


def test_a_repeated_token_is_still_one_term_and_suppresses_the_fallback():
    """Suppression is decided on FTS5 TERMS, not on expression strings.

    `"wombat" "wombat"` and `"wombat" OR "wombat"` are the same single-term
    query to FTS5, so widening is impossible -- but the two expression
    STRINGS differ, and a string comparison would spend one extra FTS query
    per zero-row sub-leg on it (and inflate the sweep's extra-query
    tie-break). Case and trailing punctuation fold the same way.
    """
    service = _make_service(construction="and_then_or")

    assert service._fts5_match_expressions("wombat wombat") == (
        '"wombat" "wombat"',
        None,
    )
    assert service._fts5_match_expressions("Wombat wombat,") == (
        '"Wombat" "wombat,"',
        None,
    )
    # Two DISTINCT content terms still widen -- suppression is not a
    # blanket "single OR term" rule.
    assert service._fts5_match_expressions("wombat burrow")[1] == (
        '"wombat" OR "burrow"'
    )


def test_stopword_trimming_is_case_and_punctuation_insensitive():
    """FTS5 reads `About,` as the term `about`; so must the trimmer."""
    service = _make_service(construction="or")

    primary, _ = service._fts5_match_expressions("About, THE vendor's chaser")
    assert primary == '"vendor\'s" OR "chaser"'


def test_stopword_list_is_lowercase_and_covers_the_measured_blocker():
    """`pm-vendor-chaser` is the one golden query blocked solely by "about"."""
    assert "about" in _FTS5_STOPWORDS
    assert all(word == word.lower() for word in _FTS5_STOPWORDS)
    # A small fixed list of function words -- not a content-word vocabulary.
    # The EXACT size is pinned, not a range: Task 5's prose quotes this
    # number, and a range let an off-by-one claim ("66") survive a review.
    assert len(_FTS5_STOPWORDS) == 67


def test_every_token_stays_individually_quoted_in_every_construction():
    """The load-bearing injection property (TASK-3995), across the seam."""
    query = 'Obsidian-3 content:foo about the "quoted'
    for construction in FTS_MATCH_CONSTRUCTIONS:
        service = _make_service(construction=construction)
        primary, fallback = service._fts5_match_expressions(query)
        for expression in (primary, fallback):
            if not expression:
                continue
            # Tokens never contain whitespace, so stripping the OR
            # operators and splitting on spaces yields exactly the terms --
            # every one of which must be a quoted literal.
            terms = expression.replace(" OR ", " ").split(" ")
            bare = [
                term
                for term in terms
                if not (term.startswith('"') and term.endswith('"'))
            ]
            assert bare == [], (
                f"{construction}: unquoted term(s) {bare} in {expression!r}"
            )


def test_an_invalid_construction_warns_once_and_behaves_as_and():
    """Fail-safe to the shipped behaviour -- never a crash, never silence."""
    service = _make_service(construction="or_of_ands_probably")

    with _captured_warnings() as warnings:
        first = service._fts5_match_expressions("notes about the vendor")
        second = service._fts5_match_expressions("a different query entirely")

    assert first == ('"notes" "about" "the" "vendor"', None)
    assert second[1] is None
    matching = [m for m in warnings if "or_of_ands_probably" in m]
    assert len(matching) == 1, (
        f"an invalid construction must warn exactly once per service: {matching}"
    )


# --- the fallback loop: zero rows only, once, per sub-leg -------------------


def _prompts_fts_spy(monkeypatch):
    """Count the prompts sub-leg's SQL executions, per MATCH expression."""
    calls = []
    original = RAGService._prompts_fts

    def spy(conn, escaped_query, limit, allowed_ids=None):
        calls.append(escaped_query)
        return original(conn, escaped_query, limit, allowed_ids)

    monkeypatch.setattr(RAGService, "_prompts_fts", staticmethod(spy))
    return calls


def test_a_matching_and_never_runs_the_fallback(tmp_path, monkeypatch):
    """`kw-plant-maintenance-record`'s protection, mechanically: a non-empty
    AND result is returned as-is and the OR form is never executed."""
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="and_then_or", prompts_db_path=db_path)
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            AND_HIT_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Wombat shift handover"]
    assert len(calls) == 1, f"the fallback ran on a non-empty AND: {calls}"
    assert " OR " not in calls[0]
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_AND]


def test_a_zero_row_and_falls_back_to_the_or_form_exactly_once(
    tmp_path, monkeypatch
):
    """The rescue: a natural-language query that finds nothing under AND
    reaches the prompt through the content-token OR -- one extra query."""
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="and_then_or", prompts_db_path=db_path)
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Wombat shift handover"]
    assert len(calls) == 2, f"expected primary + ONE fallback, got {calls}"
    assert " OR " not in calls[0]
    assert calls[1] == '"wombat" OR "template" OR "work"'
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]


def test_the_shipped_and_construction_never_runs_a_second_query(
    tmp_path, monkeypatch
):
    """Default behaviour is byte-identical, including the query COUNT."""
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(prompts_db_path=db_path)
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert results == []
    assert len(calls) == 1, f"the shipped construction has no fallback: {calls}"


def test_the_or_construction_stamps_its_rows_as_the_or_form(tmp_path):
    """The stamp names the FORM, not the position: under `or` the OR
    expression IS the primary, and calling those rows `and` would make Task
    2's negative-composition counter read zero for the widest candidate."""
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="or", prompts_db_path=db_path)

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Wombat shift handover"]
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]


def test_notes_sub_leg_falls_back_independently(tmp_path, monkeypatch):
    """The loop wraps every sub-leg's SQL helper, not just the prompts one."""
    db_path = _seed_notes(
        tmp_path,
        [("Saltmarsh hide", "The hide overlooks the wombat burrow at dusk.")],
    )
    service = _make_service(construction="and_then_or", chachanotes_db_path=db_path)

    calls = []
    original = RAGService._chacha_notes_fts

    def spy(conn, escaped_query, limit, allowed_ids=None):
        calls.append(escaped_query)
        return original(conn, escaped_query, limit, allowed_ids)

    monkeypatch.setattr(RAGService, "_chacha_notes_fts", staticmethod(spy))

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"note"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Saltmarsh hide"]
    assert len(calls) == 2, calls
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]


def test_conversations_sub_leg_falls_back_independently(tmp_path, monkeypatch):
    """The conversations helper is the one that issues TWO MATCH statements
    off a single expression (the conversation ranking, then the matched
    message lines). Both must run on whichever expression actually matched,
    or the row comes back with an empty document.
    """
    db_path, conv_id = _seed_conversation(
        tmp_path,
        "Burrow handover",
        [
            ("user", "Where did we leave the wombat burrow inspection?"),
            ("assistant", "The dusk pass is still outstanding."),
        ],
    )
    service = _make_service(construction="and_then_or", chachanotes_db_path=db_path)

    calls = []
    original = RAGService._chacha_conversations_fts

    def spy(conn, escaped_query, limit, allowed_ids=None):
        calls.append(escaped_query)
        return original(conn, escaped_query, limit, allowed_ids)

    monkeypatch.setattr(
        RAGService, "_chacha_conversations_fts", staticmethod(spy)
    )

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"conversation"}
        )
    )

    assert [r.metadata["source_id"] for r in results] == [str(conv_id)]
    assert len(calls) == 2, calls
    assert " OR " not in calls[0]
    assert calls[1] == '"wombat" OR "template" OR "work"'
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]
    # The second statement ran on the FALLBACK expression too: without that,
    # the conversation ranks but carries no message text at all.
    assert "wombat burrow" in results[0].document.lower(), results[0].document


def test_media_sub_leg_falls_back_independently(tmp_path, monkeypatch):
    """Same loop over the media sub-leg's pooled FTS5 execution."""
    db_path = _seed_media(
        tmp_path,
        [("Burrow survey", "Notes on the wombat burrow entrance survey.")],
    )
    service = _make_service(construction="and_then_or", media_db_path=db_path)

    calls = []
    original = RAGService._perform_fts5_search

    def spy(self, pool, query, limit, allowed_ids=None):
        calls.append(query)
        return original(self, pool, query, limit, allowed_ids)

    monkeypatch.setattr(RAGService, "_perform_fts5_search", spy)

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"media"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Burrow survey"]
    # ONE call into the pooled helper; the fallback happens inside it, so
    # the retry wrapper cannot multiply it.
    assert len(calls) == 1, calls
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]


def test_sub_legs_interleave_and_and_fallback_rows_in_one_query(tmp_path):
    """The spec's deliberate mixed mode: one query, two provenances.

    Media matches every token (AND); the prompt only matches through the OR
    fallback. Both rows come back, each stamped with the form that found it
    -- which is what keeps Task 5's mechanism prose table-derived.
    """
    media_path = _seed_media(
        tmp_path,
        [("Template work log", "How does the wombat template work in practice?")],
    )
    prompts_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(
        construction="and_then_or",
        media_db_path=media_path,
        prompts_db_path=prompts_path,
    )

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=10, keyword_source_types={"media", "prompt"}
        )
    )

    stamps = {
        r.metadata["source_type"]: r.metadata["fts_match"] for r in results
    }
    assert stamps == {"media": FTS_MATCH_AND, "prompt": FTS_MATCH_OR}


def test_a_query_with_no_searchable_tokens_still_touches_no_database(
    tmp_path, monkeypatch
):
    """The early exit reads the CONSTRUCTION's primary expression, so the
    `or` construction's all-stopword emptiness short-circuits too."""
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="or", prompts_db_path=db_path)
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            "what about the", top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert results == []
    assert calls == [], f"an empty MATCH expression reached FTS5: {calls}"


def test_a_failing_fallback_degrades_the_sub_leg_like_the_primary(
    tmp_path, monkeypatch
):
    """No new failure modes: the fallback inherits the degrade path."""
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="and_then_or", prompts_db_path=db_path)

    original = RAGService._prompts_fts

    def spy(conn, escaped_query, limit, allowed_ids=None):
        if " OR " in escaped_query:
            raise sqlite3.OperationalError("fallback exploded")
        return original(conn, escaped_query, limit, allowed_ids)

    monkeypatch.setattr(RAGService, "_prompts_fts", staticmethod(spy))

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )
    assert results == []


# --- the cache key ----------------------------------------------------------


def _legacy_hash(key_str):
    """`_make_key`'s hashing tail, reproduced so "byte-identical" is checkable."""
    try:
        import xxhash

        return xxhash.xxh64(key_str.encode()).hexdigest()
    except ImportError:
        import hashlib

        return hashlib.md5(key_str.encode()).hexdigest()


def test_the_and_construction_keeps_the_hybrid_key_byte_identical():
    """The pre-arc rendering, reproduced literally (B1a's discipline)."""
    cache = SimpleRAGCache(enabled=True)

    legacy_parts = [
        "quokka",
        "hybrid",
        "10",
        json.dumps({}, sort_keys=True),
        "fusion:"
        + json.dumps(
            {"alpha": 0.7, "rrf_k": 5, "pool_multiplier": 2}, sort_keys=True
        ),
    ]
    assert cache._make_key(
        "quokka",
        "hybrid",
        10,
        None,
        None,
        None,
        (0.7, 5, 2),
        FTS_MATCH_AND,
    ) == _legacy_hash("|".join(legacy_parts))

    # ... and omitting the argument entirely is the same key again.
    assert cache._make_key(
        "quokka", "hybrid", 10, None, None, None, (0.7, 5, 2)
    ) == _legacy_hash("|".join(legacy_parts))


def test_each_construction_gets_its_own_hybrid_cache_key():
    """A per-service cache plus a sweep that mutates the construction is
    exactly the shape that reported "k doesn't matter" in TASK-4110."""
    cache = SimpleRAGCache(enabled=True)

    keys = {
        construction: cache._make_key(
            "quokka", "hybrid", 10, None, None, None, (0.7, 5, 2), construction
        )
        for construction in FTS_MATCH_CONSTRUCTIONS
    }
    assert len(set(keys.values())) == len(FTS_MATCH_CONSTRUCTIONS), keys


def test_the_keyword_search_type_keys_the_construction_too():
    """A keyword-only search reads the same leg, so it needs the same key
    part -- and `"and"` still renders the pre-arc bytes."""
    cache = SimpleRAGCache(enabled=True)

    legacy = cache._make_key("quokka", "keyword", 10)
    assert cache._make_key(
        "quokka", "keyword", 10, None, None, None, None, FTS_MATCH_AND
    ) == legacy
    assert cache._make_key(
        "quokka", "keyword", 10, None, None, None, None, "and_then_or"
    ) != legacy


def test_the_search_path_passes_the_construction_into_the_cache_key(tmp_path):
    """End to end: two searches identical except for the construction must
    not share a cached entry."""
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = True
    cfg.search.prompts_db_path = Path(db_path)
    service = RAGService(cfg)

    first = asyncio.run(
        service.search(
            OR_ONLY_QUERY,
            top_k=5,
            search_type="keyword",
            keyword_source_types={"prompt"},
        )
    )
    assert first == []

    service.config.search.fts_match_construction = "and_then_or"
    second = asyncio.run(
        service.search(
            OR_ONLY_QUERY,
            top_k=5,
            search_type="keyword",
            keyword_source_types={"prompt"},
        )
    )
    assert [r.metadata["doc_title"] for r in second] == ["Wombat shift handover"], (
        "the second search was served the first construction's cached rows"
    )
