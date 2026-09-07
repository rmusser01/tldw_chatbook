"""The PRF probe's machinery, pinned before the probe is allowed to run.

TASK-15965 Phase A. The arc probes pseudo-relevance feedback BEFORE any
production code exists, so the probe itself is the only thing standing
between a premise and a verdict. Two pure functions carry the whole
mechanism -- which terms a set of pseudo-relevant documents contributes,
and what expression those terms are handed to the engine as -- and both
are the kind of code that fails silently: a term list that quietly drops
its tie-break is still a term list, and an expression that quietly loses
its per-token quoting still returns rows right up until it returns a
`sqlite3.OperationalError` or, worse, the user's own words re-read as
boolean syntax. These tests exist so the probe's numbers are evidence
about PRF rather than about a bug in the probe.

The properties, each with the mutation that reds it:

* **Stopword and query-term exclusion.** Expansion terms are the fed
  documents' CHARACTERISTIC vocabulary; a list led by "the" and by the
  query's own words expands nothing. Mutation: drop the stopword filter
  from `derive_expansion_terms` and
  `test_expansion_terms_exclude_stopwords` reds.
* **Length normalization.** Weights are `tf/|D|` summed (RM3's form), not
  raw occurrences. The frozen corpus runs 39 to 889 words per document
  (median 58), so raw summing would let one long document in a top-5 feed
  write the expansion list by itself. Mutation: sum raw counts and
  `test_a_long_document_does_not_get_to_write_the_expansion_list` reds.
* **The N cut and total determinism.** Same documents, same terms, in the
  same order -- including when two terms tie on weight, where the
  tie-break is alphabetical rather than whichever document happened to be
  read first. The tied fixture puts its tied terms in SEPARATE documents,
  because tied terms sharing one document keep their insertion order
  under any shuffle and would pin nothing. A grid point that cannot be
  re-derived is not a measurement.
* **The document-side tokenizer agrees with the engine's.** It is the
  module's one copy of the engine's run pattern, so it is pinned equal to
  `RAGService._fts5_term_key(x).split()` rather than trusted.
* **Per-token quoting.** The injection-safety property of TASK-3995 is
  load-bearing in a probe too: the probe composes expressions from
  DOCUMENT text, which is far more hostile than a query box. Mutation:
  join the tokens bare in `compose_prf_expression` and
  `test_hostile_tokens_stay_inert_in_the_prf_expression` reds with
  `OperationalError`.
* **The `""` contract.** An empty MATCH expression is an FTS5 syntax
  error, not "no results" -- the engine's existing skip contract, which
  the probe inherits rather than re-invents.

Hostile-token cases execute against a real stdlib in-memory FTS5 table
(the `Tests/RAG_Search/test_fts5_query_escaping.py` idiom): the parse is
the thing under test, so a mock would pin nothing.

**Nothing in this file is gated.** These are pure-function pins that must
run with no env var set. Task 2's probe RUN is the gated part; when it
lands it must carry its own per-test/`harness_gate()` module rather than a
module-level `pytestmark` here, which would silently take these pins down
with it (`harness/environment.py:harness_gate`'s own docstring names that
trap).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import fields

import pytest

from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    _FTS5_STOPWORDS,
    RAGService,
)
from Tests.RAG_Eval.harness.prf_probe import (
    ProbeQueryResult,
    _index_terms,
    compose_feedback_expression,
    compose_prf_expression,
    derive_expansion_terms,
)


# --------------------------------------------------------------------------
# Scratch FTS5 helpers -- the real engine, no app DBs.
# --------------------------------------------------------------------------


def _fts5_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, content)")
    conn.execute(
        "INSERT INTO docs(title, content) VALUES (?, ?)",
        (
            "Lathe Maintenance Log",
            "The Obsidian-3 lathe shows spindle runout under load.",
        ),
    )
    conn.commit()
    return conn


def _match(conn: sqlite3.Connection, expression: str):
    return conn.execute(
        "SELECT rowid FROM docs WHERE docs MATCH ?", (expression,)
    ).fetchall()


# --------------------------------------------------------------------------
# Term derivation
# --------------------------------------------------------------------------


def test_expansion_terms_exclude_stopwords() -> None:
    """Function words are the loudest terms in any document and the least
    characteristic of it.

    The seed is deliberately stopword-heavy: raw TF over it puts "the" and
    "of" on top, and a PRF expansion led by those two widens the query
    toward the whole corpus. The exclusion is what makes the remainder
    mean anything.
    """
    docs = [
        "The spindle of the lathe and the bed of the lathe.",
        "The lathe of the shop, the spindle of the shop.",
    ]

    terms = derive_expansion_terms(
        docs, query_terms=(), n_terms=8, stopwords=_FTS5_STOPWORDS
    )

    assert "the" not in terms
    assert "of" not in terms
    assert "and" not in terms
    # The characteristic vocabulary survives, and it is what leads.
    assert terms[0] == "lathe"
    assert set(terms) == {"lathe", "spindle", "bed", "shop"}


def test_expansion_terms_exclude_the_querys_own_terms() -> None:
    """PRF's whole point is reaching vocabulary the query does NOT have.

    A term the query already carries is already in the second pass by way
    of the query side of the composition; re-deriving it as an
    "expansion" term would inflate the derived-term count with terms that
    expand nothing.
    """
    docs = ["Lathe spindle runout. Lathe spindle bearing. Lathe carriage."]

    terms = derive_expansion_terms(
        docs,
        # Raw query tokens, exactly as the probe will hand them over --
        # including a hyphenated one, whose alphanumeric runs are what
        # FTS5 indexes and therefore what must be excluded.
        query_terms=("Lathe", "spindle-bearing"),
        n_terms=8,
        stopwords=_FTS5_STOPWORDS,
    )

    assert "lathe" not in terms
    assert "spindle" not in terms
    assert "bearing" not in terms
    assert set(terms) == {"runout", "carriage"}


def test_expansion_terms_are_capped_at_n_terms() -> None:
    """N is a pre-registered grid coordinate, so the cut has to be exact."""
    docs = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]

    for n_terms in (0, 1, 4, 8, 16):
        terms = derive_expansion_terms(
            docs, query_terms=(), n_terms=n_terms, stopwords=_FTS5_STOPWORDS
        )
        assert len(terms) == min(n_terms, 10), f"N={n_terms} -> {terms}"


def test_a_long_document_does_not_get_to_write_the_expansion_list() -> None:
    """Length normalization (RM3's tf/|D|), against the lever that exists.

    The frozen corpus runs 39 to 889 words per document (median 58,
    measured over its 172 docs), so raw occurrence-summing lets one long
    document in a top-5 feed outweigh a median one by ~15x -- the
    expansion list would then be written by whichever long document
    happened to land in the feed, and both outcomes reach the verdict
    silently: a topically-wrong long document manufactures a NULL, a
    dominant on-topic one manufactures a rescue no shorter feed
    reproduces.

    The synthetic states the inversion exactly. Raw counts say `boiler`
    (3 occurrences) beats `carriage` (2); normalized weights say
    `carriage` (1/2 + 1/2 = 1) beats `boiler` (3/60 = 1/20) -- the term
    present in MORE of the fed documents wins, which is what
    "characteristic of the feedback set" is supposed to mean.

    Mutation: sum raw counts instead of `Fraction(count, length)` and
    this reds on the very first assertion.
    """
    long_doc = " ".join(["boiler"] * 3 + [f"filler{i}" for i in range(57)])
    assert len(long_doc.split()) == 60

    terms = derive_expansion_terms(
        [long_doc, "carriage widget", "carriage gadget"],
        query_terms=(),
        n_terms=8,
        stopwords=_FTS5_STOPWORDS,
    )

    assert terms[0] == "carriage"
    assert terms.index("carriage") < terms.index("boiler")
    # ...and the long document is not silenced either, only cut down to
    # its share: `boiler` still outranks the fillers it shares its
    # document with.
    assert terms.index("boiler") < terms.index("filler0")


#: Two terms tied on normalized weight (1) in SEPARATE documents, plus an
#: untied tail. The separation is the whole point: `sorted` is stable, so
#: tied terms living in ONE document keep their within-document insertion
#: order no matter how the DOCUMENTS are shuffled -- a fixture that can
#: never expose a missing tie-break. Split across documents, shuffling the
#: documents swaps which tied term is inserted first, and only the
#: alphabetical key holds the output still.
_TIED_DOCS = [
    "zebra zebra",  # zebra: 2/2 = 1
    "runout runout",  # runout: 2/2 = 1
    "apple banana banana banana",  # banana 3/4, apple 1/4
]


def test_expansion_terms_rank_by_weight_then_alphabetically() -> None:
    """The tie-break is stated, not incidental.

    Weights alone leave the order of equal-weight terms to dictionary
    insertion, i.e. to which document the probe happened to read first --
    which makes a grid point unreproducible in exactly the situation
    (many terms, few documents, small weights) where ties are the rule
    rather than the exception.
    """
    terms = derive_expansion_terms(
        _TIED_DOCS, query_terms=(), n_terms=8, stopwords=_FTS5_STOPWORDS
    )

    # `zebra` is inserted FIRST (it is document 0) and still sorts second.
    assert terms == ("runout", "zebra", "banana", "apple")


def test_expansion_terms_do_not_depend_on_document_order() -> None:
    """The determinism the tie-break buys, asserted end to end.

    The shuffle inverts the insertion order of the tied pair, so this
    agrees only because the ranking key does not consult it.
    """
    shuffled = [_TIED_DOCS[1], _TIED_DOCS[2], _TIED_DOCS[0]]

    assert derive_expansion_terms(
        _TIED_DOCS, query_terms=(), n_terms=8, stopwords=_FTS5_STOPWORDS
    ) == derive_expansion_terms(
        shuffled, query_terms=(), n_terms=8, stopwords=_FTS5_STOPWORDS
    )


def test_terms_are_the_lowercased_alphanumeric_runs_fts5_indexes() -> None:
    """A derived term that is not an index term can never match anything.

    FTS5's default tokenizer indexes alphanumeric runs and folds case, so
    `Read-Only,` is indexed as `read` and `only`. Deriving `Read-Only,`
    verbatim would produce a quoted phrase term that happens to work and a
    count split across three spellings of the same word that does not.
    """
    docs = ["Read-Only mode. READ the read-only notice; Read!"]

    terms = derive_expansion_terms(
        docs, query_terms=(), n_terms=8, stopwords=_FTS5_STOPWORDS
    )

    assert terms[0] == "read"  # 4 occurrences once folded and split
    assert set(terms) == {"read", "only", "mode", "notice"}


def test_index_terms_agree_with_the_engines_own_term_key() -> None:
    """The document-side tokenizer is a COPY of the engine's run pattern.

    The engine writes `r"[^\\W_]+"` inline inside `_fts5_term_key` and
    `_is_fts5_stopword` and exposes no "index terms of this text" helper
    -- it only ever tokenizes queries -- so the probe's document side
    re-declares it. A copy with no pin is a copy that drifts, and a
    derivation that splits words differently from the engine derives
    terms that cannot match the documents they came from.

    Not a claim that either side equals what FTS5 INDEXES: `unicode61`
    folds diacritics, so both answer `café` where the index holds `cafe`.
    The frozen corpus has zero non-ASCII words, so that gap cannot reach
    the verdict (module docstring).
    """
    samples = [
        "Obsidian-3 lathe",
        "content:wombat",
        "{title content}:lathe",
        'confirmed" runout',
        "lathe) OR (spindle",
        "READ the Read-Only notice!",
        "under_score mixed_Case",
        "!!! ---",
        "",
        "trailing   whitespace  ",
        "digits 42 and 3.14",
    ]

    for sample in samples:
        assert _index_terms(sample) == RAGService._fts5_term_key(
            sample
        ).split(), sample


def test_no_documents_yields_no_terms() -> None:
    """A first pass that returned nothing feeds PRF nothing.

    This is the arc's central structural risk, so the probe's floor case
    is a pin rather than an accident: no documents, and documents with no
    content term left after exclusion, both answer with an empty tuple --
    never a crash and never a term.
    """
    assert (
        derive_expansion_terms(
            [], query_terms=(), n_terms=8, stopwords=_FTS5_STOPWORDS
        )
        == ()
    )
    assert (
        derive_expansion_terms(
            ["", "   ", "!!! ---"],
            query_terms=(),
            n_terms=8,
            stopwords=_FTS5_STOPWORDS,
        )
        == ()
    )
    assert (
        derive_expansion_terms(
            ["the of and to"],
            query_terms=(),
            n_terms=8,
            stopwords=_FTS5_STOPWORDS,
        )
        == ()
    )


# --------------------------------------------------------------------------
# Expression composition
# --------------------------------------------------------------------------


def test_prf_expression_ors_the_query_content_terms_with_the_expansion_terms() -> None:
    """The second pass's shape, byte for byte.

    Query side first, in query order, then the expansion terms in their
    derived order -- so a table row's expression can be read against the
    term list that produced it.
    """
    expression = compose_prf_expression(
        "the lathe spindle", ("runout", "carriage")
    )

    assert expression == '"lathe" OR "spindle" OR "runout" OR "carriage"'


def test_prf_expression_drops_the_querys_function_words() -> None:
    """A raw OR over every token matches every document containing "the".

    The same reason the engine's own OR construction trims: on the OR
    side a function word is not noise, it is the whole corpus.
    """
    expression = compose_prf_expression("what is the lathe", ())

    assert expression == '"lathe"'


def test_prf_expression_is_empty_when_nothing_survives() -> None:
    """The engine's skip contract, inherited rather than re-invented.

    `""` means "no rows, do not run the query"; an empty string handed to
    MATCH is an FTS5 syntax error, so a probe that treated "nothing
    survived" as "run it anyway" would report a crash as a result.
    """
    assert compose_prf_expression("", ()) == ""
    assert compose_prf_expression("the of and", ()) == ""
    assert compose_prf_expression("!!! ---", ()) == ""
    # Expansion terms with no alphanumeric run cannot become a term
    # either: `""` inside the expression is a syntax error, not an
    # inert token.
    assert compose_prf_expression("", ("", "  ", "!!")) == ""

    # ...but an all-function-word query with real expansion terms is NOT
    # empty: the expansion is exactly what PRF has to offer there.
    assert compose_prf_expression("the of and", ("runout",)) == '"runout"'


@pytest.mark.parametrize(
    ("query", "expansion_terms"),
    [
        # The documented incident: unquoted, FTS5 reads `3` as a column.
        ("Obsidian-3 lathe", ()),
        # A column-filter attempt, on each side of the composition.
        ("content:wombat lathe", ()),
        ("lathe", ("content:wombat",)),
        ("{title content}:lathe spindle", ()),
        # FTS5 operators as ordinary words. On the QUERY side they are
        # function words and get trimmed, so the expansion side is where
        # a literal operator actually has to survive as a literal.
        ("lathe AND OR NOT spindle", ()),
        ("lathe", ("or", "and", "not", "near")),
        # A prefix-operator attempt and a bare wildcard.
        ("lathe* spindle", ()),
        ("lathe", ("spindle*", "*")),
        # An embedded quote and a stray parenthesis, both sides.
        ('confirmed" runout', ()),
        ("lathe) OR (spindle", ()),
        ("lathe", ('confirmed"', "spindle)")),
    ],
)
def test_hostile_tokens_stay_inert_in_the_prf_expression(
    query: str, expansion_terms: tuple[str, ...]
) -> None:
    """Every expression the composition can emit must execute cleanly.

    Against a real FTS5 table, because the parse is the property. Whether
    a hostile token MATCHES anything is a relevance question; that it
    cannot change the expression's grammar is the safety one, and it is
    the safety one that decides whether the probe's numbers exist at all.

    Both sides are parametrized: expansion terms come from DOCUMENT text,
    which no user-input validation has ever seen.
    """
    with closing(_fts5_conn()) as conn:
        expression = compose_prf_expression(query, expansion_terms)
        if expression:
            _match(conn, expression)  # must not raise


def test_the_only_operator_in_the_expression_is_the_one_the_probe_put_there() -> None:
    """The contrast that makes the inertness pin discriminating.

    The same tokens joined bare are not merely a different query -- they
    are a syntax error or the user's own words rearranged into boolean
    logic. Quoting is what buys the difference, so both halves are
    asserted together.
    """
    with closing(_fts5_conn()) as conn:
        # Query side: the `Obsidian-3` incident.
        with pytest.raises(sqlite3.OperationalError):
            _match(conn, "Obsidian-3 OR lathe")
        assert _match(conn, compose_prf_expression("Obsidian-3 lathe", ())) == [
            (1,)
        ]

        # Expansion side: a literal `OR` derived from a document is a
        # search term, never a disjunction. (It cannot arrive from
        # `derive_expansion_terms`, which excludes it as a stopword --
        # the composition must not depend on that.)
        with pytest.raises(sqlite3.OperationalError):
            _match(conn, "lathe OR OR")
        expression = compose_prf_expression("lathe", ("or",))
        assert expression == '"lathe" OR "or"'
        _match(conn, expression)


# --------------------------------------------------------------------------
# The licensed variant's feedback pass
# --------------------------------------------------------------------------


def test_feedback_expression_is_the_content_token_or() -> None:
    """Step 0's licensed fallback: wide enough to return SOMETHING to feed.

    The shipped first pass ANDs its terms, so a paraphrase query can
    return zero rows and feed PRF nothing. This form exists only to
    select pseudo-relevant documents -- users still see the shipped
    results -- which is why it is a separate function rather than a
    change to anything the product runs.
    """
    assert compose_feedback_expression("the lathe spindle") == (
        '"lathe" OR "spindle"'
    )
    assert compose_feedback_expression("Obsidian-3") == '"Obsidian-3"'


def test_feedback_expression_is_empty_when_trimming_empties_the_query() -> None:
    """Same skip contract, same reason."""
    assert compose_feedback_expression("") == ""
    assert compose_feedback_expression("the of and") == ""
    assert compose_feedback_expression("!!!") == ""


def test_feedback_expression_is_byte_identical_to_the_engines_or_construction() -> None:
    """Probe fidelity (the TASK-15400 lesson): measure the form the engine
    would actually run.

    `compose_feedback_expression` cannot construct a `RAGService` per
    query inside the probe's loops, so it composes from the same
    static helpers the engine's `or` construction composes from. This pin
    is what keeps "the same helpers" from drifting into "a similar
    expression": the engine builds the reference, and the two are
    compared as strings.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    cfg.search.fts_match_construction = "or"
    service = RAGService(cfg)

    try:
        for query in (
            "the lathe spindle",
            "Obsidian-3 lathe",
            "what is a vendor chaser",
            'confirmed" runout',
            "the of and",
            "",
        ):
            primary, fallback = service._fts5_match_expressions(query)
            assert fallback is None
            assert compose_feedback_expression(query) == primary, query
    finally:
        # The constructor opens an 8-worker thread pool; a pin is no
        # reason to leak one per run.
        service.close()


# --------------------------------------------------------------------------
# The per-query table row
# --------------------------------------------------------------------------


def test_probe_query_result_carries_every_column_the_report_owes() -> None:
    """The table row is a contract, not a convenience.

    The spec's report owes fireability, what the first pass returned, how
    many documents were fed, the CONTENT-FETCH PRICE those documents cost
    (the row-content fact: media and conversation rows carry label
    snippets, not text), and the target's rank on BOTH sides -- because
    the arc's discipline is gains AND losses by query id, and a table
    without an `after` column beside a `before` column can only report
    gains.
    """
    assert [field.name for field in fields(ProbeQueryResult)] == [
        "query_id",
        "category",
        "fireable",
        "first_pass_rows",
        "fed_docs",
        "content_fetches",
        "target_rank_before",
        "target_rank_after",
        "rows_after",
    ]

    row = ProbeQueryResult(
        query_id="pm-vendor-chaser",
        category="paraphrase",
        fireable=True,
        first_pass_rows=4,
        fed_docs=4,
        content_fetches=4,
        target_rank_before=None,
        target_rank_after=2,
        rows_after=7,
    )

    assert row.target_rank_before is None  # a miss is None, never 0
    assert row.target_rank_after == 2
