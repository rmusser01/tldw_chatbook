"""The engine and the Library share ONE prefix form and ONE word list.

TASK-17755 gave the Library's four-seam plain-Search path the
``and_then_prefix`` construction the engine's keyword leg has run since
TASK-15700. Both now need the PREFIX form and the function-word list it
trims with, and the obvious way to get there -- paste the 67-word frozenset
and the ``"tok"*`` builder into the Library module -- is the failure this
file exists to prevent.

It is not a hypothetical. The reason TASK-17755 was worth doing at all is
that the Library SCREEN runs both paths: its **Search** tab is the four-seam
one and its **RAG Answer** tab is the engine, and TASK-3997 documented a
user getting two different matching rules on one screen (an inflection miss
answered in one tab and returned nothing in the other). Two copies of the
widening form would re-open exactly that gap the first time either copy was
edited, and it would present as a retrieval bug with no obvious cause --
both code paths look correct in isolation.

So: one definition in `Utils/fts5_match_forms.py`, imported by both. These
pins fail if a future edit re-introduces a second copy, and -- because
identity is asserted, not just equality -- they fail while the two copies
still agree, which is the only moment at which the problem is cheap to fix.

The third pin is the one that would matter if the extraction were done
carelessly: the engine's expression must be BYTE-IDENTICAL to what it built
inline before, since `and_then_prefix` is the shipped default and TASK-15700
measured it in that exact form.
"""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Library.library_fts_query import (
    build_fts_match_query,
    build_prefix_match_query,
)
from tldw_chatbook.Library.library_local_rag_search_service import (
    LibraryLocalRagSearchService,
    SeamState,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.RAG_Search.simplified import rag_service as engine_module
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX,
    RAGConfig,
    RAGService,
)
from tldw_chatbook.Utils.fts5_match_forms import (
    FTS5_STOPWORDS,
    build_prefix_match_expression,
    fts5_query_tokens,
)

#: Queries chosen to exercise every branch the two paths could diverge on:
#: a plain multi-token query, function words in the middle and at both ends,
#: a query that is nothing BUT function words, punctuation-only tokens,
#: multi-run tokens, case, and FTS5 operator syntax typed as user text.
QUERIES = [
    "feedback loop",
    "the tension in the guy wire",
    "of the",
    "!!! ???",
    "read-only Obsidian-3 vault",
    "Wombat WOMBAT wombat,",
    "search OR delete NOT keep",
    'a "quoted" phrase',
    "single",
    "",
]


def _engine_service() -> RAGService:
    """A RAGService under the shipped ``and_then_prefix`` construction."""
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    cfg.search.fts_match_construction = FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX
    return RAGService(cfg)


def test_the_stopword_list_is_one_object_not_two_that_agree():
    """Identity, not equality -- equality passes right up until it doesn't."""
    assert engine_module._FTS5_STOPWORDS is FTS5_STOPWORDS, (
        "the engine has its own copy of the function-word list again; the "
        "Library's prefix fallback and the engine's will drift apart"
    )
    assert len(FTS5_STOPWORDS) == 67, (
        "the list changed size; it is fixed and small on purpose -- a large "
        "list starts deleting the content words TASK-15400's census "
        "measured as the real blockers"
    )
    assert all(word == word.lower() for word in FTS5_STOPWORDS)


def test_the_engine_and_the_library_build_the_same_prefix_form():
    """The two paths' widening expressions are the same string, per query."""
    service = _engine_service()

    for query in QUERIES:
        _primary, engine_fallback = service._fts5_match_expressions(query)
        library_fallback = build_prefix_match_query(query)
        # The engine reports "no widening available" as None and the
        # Library as "" -- both mean "skip the query", and neither path
        # ever runs the expression. Normalize before comparing so the pin
        # is about the FORM, not about two callers' empty conventions.
        assert (engine_fallback or "") == library_fallback, (
            f"{query!r}: engine built {engine_fallback!r}, Library built "
            f"{library_fallback!r}"
        )


def test_the_extracted_builder_is_byte_identical_to_the_inline_construction():
    """The engine's shipped default must not have moved during extraction.

    Reproduces the exact expression TASK-15700's inline code built --
    content tokens, quoted, star OUTSIDE the quotes, space-joined -- from
    the engine's own tokenizer and stopword predicate, and compares it to
    what the shared builder now returns.
    """
    service = _engine_service()

    for query in QUERIES:
        tokens = service._fts5_query_tokens(query)
        quoted = [
            service._quote_fts5_token(token)
            for token in tokens
            if not service._is_fts5_stopword(token)
        ]
        inline = " ".join(f"{token}*" for token in quoted)

        assert build_prefix_match_expression(tokens) == inline, query


def test_the_prefix_form_never_stars_a_function_word():
    """The property the whole trim exists for.

    ``"the"*`` matches "the", "then", "there", "their", "these" -- nearly a
    whole corpus -- and the prefix form ANDs its terms, so one untrimmed
    function word drags the whole expression toward matching everything the
    other terms allow.
    """
    expression = build_prefix_match_query("the tension in the guy wire")

    assert expression == '"tension"* "guy"* "wire"*', expression
    for word in ("the", "in"):
        assert f'"{word}"*' not in expression


@pytest.mark.parametrize(
    "hostile",
    [
        'wombat" OR nonsense:"',
        "wombat NEAR/2 badger",
        "title:secret",
        "wombat) OR (badger",
        '"',
        "*",
        "^wombat",
    ],
)
def test_the_library_prefix_form_is_safe_against_a_real_fts5_table(hostile):
    """User text cannot become operator syntax in the widening form either.

    The AND primary's injection safety is pinned in the Library's own suite;
    the fallback is a second expression reaching the same tables, and an
    escape that only covers the primary covers the path that runs when a
    search finds nothing -- i.e. exactly when a user retypes and escalates.
    Run against a real FTS5 table because the property being asserted is
    "SQLite parses this without error", which only SQLite can answer.
    """
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
        connection.execute("INSERT INTO docs (body) VALUES ('wombat badger')")
        for expression in (
            build_fts_match_query(hostile),
            build_prefix_match_query(hostile),
        ):
            if not expression:
                continue
            connection.execute(
                "SELECT rowid FROM docs WHERE docs MATCH ?", (expression,)
            ).fetchall()
    finally:
        connection.close()


#: The document TASK-3997's divergence example is built on: it carries
#: "guy" and "tensioner", so the query "guy tension" is an INFLECTION MISS
#: for the AND primary (whose widening reaches "tensions", never
#: "tensioner") and a hit for the prefix form.
_INFLECTION_DOC = "The mast guy tensioner was re-seated during the shift."
_INFLECTION_QUERY = "guy tension"


def test_plain_search_and_rag_answer_answer_the_same_inflection_miss(tmp_path):
    """AC#4: the Library screen stops having two matching rules.

    This is the argument from TASK-3997 that is not about metrics. The
    Library's **Search** tab is the four-seam keyword path and its **RAG
    Answer** tab is the engine's keyword leg. Until TASK-17755 the engine
    ran `and_then_prefix` and the four-seam path ran a strict AND, so one
    screen answered an inflection miss in one tab and returned nothing in
    the other -- with no way for a user to tell why.

    Both paths are pointed at ONE seeded prompts database here, because the
    claim is about the same corpus and the same query. Anything less (two
    corpora, or comparing expressions rather than rows) would leave the
    divergence provable only in the direction it was already known.
    """
    db_path = tmp_path / "divergence_prompts.db"
    db = PromptsDatabase(db_path, client_id="test_shared_match_forms")
    try:
        prompt_id, _uuid, message = db.add_prompt(
            name="Guy wire handover",
            author=None,
            details=None,
            system_prompt=_INFLECTION_DOC,
        )
        assert prompt_id is not None, f"seed failed: {message}"

        # The premise: the AND primary genuinely cannot reach this document.
        # Stated as the expression, so a future change to the plural widener
        # that DID reach "tensioner" would red here rather than turning the
        # rest of this test into a tautology.
        assert build_fts_match_query(_INFLECTION_QUERY) == (
            '("guy" OR "guys" OR "guies") AND ("tension" OR "tensions")'
        ), "the widener changed; re-derive whether this is still a miss"

        engine = _engine_service()
        engine.config.search.prompts_db_path = db_path
        engine_rows = asyncio.run(
            engine._keyword_search(
                _INFLECTION_QUERY, top_k=5, keyword_source_types={"prompt"}
            )
        )

        library = LibraryLocalRagSearchService(
            SimpleNamespace(
                prompt_scope_service=PromptScopeService(
                    local_service=LocalPromptService(db), server_service=None
                )
            )
        )
        available, library_rows = asyncio.run(
            library._search_prompts(_INFLECTION_QUERY, 5)
        )

        assert available is SeamState.AVAILABLE
        assert [row.metadata["doc_title"] for row in engine_rows] == [
            "Guy wire handover"
        ], "RAG Answer's keyword leg lost the document it used to find"
        assert [row["title"] for row in library_rows] == ["Guy wire handover"], (
            "plain Search still returns nothing for a query RAG Answer "
            "answers -- the TASK-3997 divergence is not closed"
        )
    finally:
        db.close_connection()


def test_the_shared_tokenizer_drops_tokens_fts5_could_never_match():
    """A pure-punctuation token would AND an unmatchable term into the form.

    ``"!!!"*`` matches nothing, and the prefix form ANDs its terms, so
    carrying it would turn a rescue into a guaranteed zero-row query -- the
    fallback silently doing nothing, which is worse than not having one.
    """
    assert fts5_query_tokens("wombat !!! badger") == ["wombat", "badger"]
    assert build_prefix_match_query("wombat !!! badger") == '"wombat"* "badger"*'
