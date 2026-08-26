# Tests/RAG_Eval/test_goldenset_integrity.py
"""Always-on tests for the fixture corpus, the golden query set, and the
integrity validator.

Two jobs, deliberately in one file:

1. **Validator contract** — `load_corpus` / `load_golden` / `validate` reject
   malformed input and name *every* defect in one error, so a broken golden
   set fails fast with the full list instead of one defect per run. Planted
   defects are built inline; the real fixtures are never mutated.
2. **Corpus design invariants** — the properties that make the corpus a
   measuring instrument rather than sample text: paraphrase and
   vocabulary-mismatch queries share no content word with their targets (a
   single shared noun silently downgrades the case to a keyword case that FTS
   can solve), keyword queries own a token unique to their targets, negative
   queries are about topics genuinely absent, every capability group spans all
   three source types. These are cheap and pure, and they are the only thing
   standing between a careless fixture edit and months of baselines that
   measure something other than what their category claims.

The stemming here is deliberately crude and over-eager: it exists to make the
overlap checks *stricter* than a real tokenizer would be (it collapses
sales/sale, increased/increase), so a pair that passes here is not reachable
by keyword matching for a token-shape reason either.
"""
from __future__ import annotations

import hashlib
import re
import tomllib
from collections import Counter

import pytest

from Tests.RAG_Eval.harness.goldenset import (
    CATEGORIES,
    CORPUS_PATH,
    GOLDEN_PATH,
    NEGATIVE_CATEGORY,
    REQUIRED_CATEGORIES,
    SCOPEABLE_SOURCE_TYPES,
    SCOPED_CATEGORY,
    SOURCE_TYPES,
    CorpusDoc,
    GoldenQuery,
    GoldenSetError,
    load_corpus,
    load_golden,
    validate,
)

# --------------------------------------------------------------------------
# tokenization helpers (test-local on purpose: these guard the *authoring*
# rules, they are not part of the harness contract Tasks 5/6 consume)
# --------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "from", "by", "with", "without", "into", "onto", "over", "under",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "done", "has", "have", "had", "can", "could", "should", "would",
    "may", "might", "must", "shall", "it", "its", "this", "that", "these",
    "those", "as", "than", "then", "so", "not", "no", "any", "all", "each",
    "every", "some", "who", "whom", "whose", "what", "which", "when", "where",
    "why", "how", "there", "here", "he", "she", "they", "them", "his", "her",
    "their", "our", "we", "you", "your", "i", "my", "me", "up", "out", "off",
    "about", "after", "before", "during", "while", "because", "since",
    "between", "through", "very", "much", "more", "most", "get", "got",
    "make", "made", "one", "two", "own", "same", "other", "another", "such",
    "only", "also", "still", "just", "now", "per", "both", "either",
}

_SUFFIXES = (
    "iness", "ingly", "edly", "ness", "ing", "ies", "ied", "es", "ed", "ly", "s", "e",
)


def _stem(word: str) -> str:
    """Reduce a word to a canonical form by stripping suffixes to a FIXED POINT.

    Stripping only once is not enough, and the difference is not academic: a
    single pass returns on whichever suffix matches first, so `readings` stops
    at `reading` while `reading` continues to `read`. Two spellings of one word
    then carry two stems, the overlap guard sees no collision, and a pair that
    FTS5's porter tokenizer *would* match sails through as "no overlap" — the
    exact failure the guard exists to prevent.

    Re-stripping until the word stops changing makes the result a function of
    the word family rather than of suffix order. The trailing `"e"` entry is
    what closes `increase`/`increased` (-> `increa`) and `process`/`processes`
    (-> `proc`); without it the fixed point still splits that family. The
    4-character floor stops the reduction before words become fragments
    (`early` must not become `ear`).

    The result is not a real Porter stem and is not meant to be readable. It
    only has to be canonical and at least as aggressive as the tokenizer it
    stands in for, so that "no overlap here" is a claim about keyword
    reachability rather than about spelling.
    """
    while True:
        for suffix in _SUFFIXES:
            if not word.endswith(suffix):
                continue
            base = word[: -len(suffix)]
            if suffix == "ies":
                base += "y"
            if len(base) >= 4:  # never strip down to a fragment ("early" -> "ear")
                word = base
                break
        else:
            return word


def _content_stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS and len(w) > 2}


#: Pairs a real stemmer (FTS5's porter, in practice) collapses. The overlap
#: guard is only an over-approximation of keyword reachability if it collapses
#: them too — otherwise it passes pairs that keyword matching can reach.
_MORPHOLOGICAL_PAIRS = [
    ("readings", "reading"),
    ("bearings", "bearing"),
    ("classes", "class"),
    ("increased", "increase"),
    ("processes", "process"),
]


def _doc_stems(doc: CorpusDoc) -> set[str]:
    # Title and content are both indexed by the real writers, so overlap has
    # to be judged against both.
    return _content_stems(f"{doc.title} {doc.content}")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> list[CorpusDoc]:
    return load_corpus(CORPUS_PATH)


@pytest.fixture(scope="module")
def golden() -> list[GoldenQuery]:
    return load_golden(GOLDEN_PATH)


@pytest.fixture(scope="module")
def by_slug(corpus: list[CorpusDoc]) -> dict[str, CorpusDoc]:
    return {doc.slug: doc for doc in corpus}


def _valid_corpus() -> list[CorpusDoc]:
    """Minimal inline corpus covering every source type.

    Including ``prompt``, which `validate` requires like any other: a prompt
    document has no writer behind it, but a corpus that declares the type and
    ships none of them would leave the prompt category unmeasurable.
    """
    return [
        CorpusDoc("n1", "note", "Note one", "Alpha beta. Gamma delta. Epsilon zeta."),
        CorpusDoc("m1", "media", "Media one", "Eta theta. Iota kappa. Lambda mu."),
        CorpusDoc("c1", "conversation", "Chat one", "Nu xi. Omicron pi. Rho sigma."),
        CorpusDoc("p1", "prompt", "Prompt one", "Tau upsilon. Phi chi. Psi omega."),
    ]


def _valid_golden() -> list[GoldenQuery]:
    """Minimal inline golden set covering every REQUIRED category.

    The first four entries keep their positions: several tests below replace
    ``golden[0]``/``[1]``/``[3]`` to plant a defect, so the P2ab categories
    are appended rather than interleaved.
    """
    return [
        GoldenQuery("q-kw", "alpha beta", "keyword", ("n1",)),
        GoldenQuery("q-pr", "eta theta", "paraphrase", ("m1",)),
        GoldenQuery("q-vm", "nu xi", "vocabulary_mismatch", ("c1",)),
        GoldenQuery("q-neg", "nothing here", "negative", ()),
        GoldenQuery("q-sc", "alpha beta", "scoped", ("n1",), scope_slugs=("n1",)),
        GoldenQuery("q-ng", "gamma but never delta", "negation", ("n1",)),
        GoldenQuery("q-pm", "a saved prompt for something", "prompt", ("p1",)),
    ]


def _scoped_query(
    query_id: str = "q-sc",
    *,
    category: str = SCOPED_CATEGORY,
    relevant: tuple[str, ...] = ("n1",),
    scope: tuple[str, ...] | None = ("n1", "m1"),
) -> GoldenQuery:
    """A scoped golden query, with every rule-relevant part parameterized."""
    return GoldenQuery(
        query_id, "alpha beta", category, relevant, scope_slugs=scope
    )


def _defects(corpus: list[CorpusDoc], golden: list[GoldenQuery]) -> str:
    """Run validate() expecting failure; return the error message."""
    with pytest.raises(GoldenSetError) as excinfo:
        validate(corpus, golden)
    return str(excinfo.value)


def _write_toml(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# the real fixtures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("inflected, root", _MORPHOLOGICAL_PAIRS)
def test_stem_is_canonical_across_inflections(inflected, root):
    """The stemmer must be a function of the *word family*, not of which
    suffix happens to match first.

    A single-pass stripper returns on the first hit and never re-stems, so
    `readings` -> `reading` while `reading` -> `read`: two spellings of one
    word get two stems, and an overlap the real tokenizer would see slips
    through the guard.
    """
    assert _stem(inflected) == _stem(root)


@pytest.mark.parametrize("inflected, root", _MORPHOLOGICAL_PAIRS)
def test_overlap_predicate_treats_inflections_as_shared(inflected, root):
    """The property that actually matters: the guard's predicate, not the
    helper, must flag an inflected repeat as overlap."""
    doc = CorpusDoc(
        "probe",
        "note",
        "Probe document",
        f"The {inflected} were recorded. A second sentence. A third sentence.",
    )
    assert _content_stems(f"a question about the {root}") & _doc_stems(doc)


def test_guard_catches_an_inflected_reword_of_a_shipped_pair(by_slug):
    """The two counterexamples that motivated the canonical stemmer, pinned
    against the real (unmutated) fixtures.

    Both queries below are keyword-reachable — the target literally contains
    the inflected form — and a single-pass stemmer scored both as "no
    overlap", which would have let a broken pair ship under a passing guard.
    """
    # note-hypertension-followup says "...elevated systolic readings..."
    hypertension = by_slug["note-hypertension-followup"]
    assert _content_stems("how dangerous is a high blood pressure reading") & _doc_stems(
        hypertension
    )

    # conv-workout-time says "...the evening classes were always full."
    workout = by_slug["conv-workout-time"]
    assert _content_stems("moved my exercise class to sunrise") & _doc_stems(workout)


def test_real_fixtures_validate_clean(corpus, golden):
    assert validate(corpus, golden) is None


def test_corpus_composition_matches_the_planned_design(corpus):
    counts = Counter(doc.source_type for doc in corpus)
    # 49 from P1 (45 short + 3 long multi-chunk targets + the vector-blind
    # keyword target of TASK-3994 AC #2) + 123 from the P2ab scale-up:
    #   32  the scoped collection (24 pool + 8 estate targets)
    #   18  the negation families (12 norm-assertions + 6 exceptions)
    #   10  the acronym family (6 targets + 4 acronym-noise decoys)
    #   19  the compositional family (6 answers + 13 single-conjunct decoys)
    #    6  prompt fixtures (no writer, no index — invisible by design)
    #   14  general distractors
    #   24  anchor dilution (the documents the probe asked for: a class whose
    #       target is the corpus's only document on its subject measures the
    #       corpus's sparseness, not the pipeline)
    # The acronym and compositional families kept their documents after their
    # queries were rejected — as distractors, which is what a family whose
    # class proved unfailable is good for.
    assert len(corpus) == 172
    assert counts["note"] == 76
    assert counts["media"] == 59
    assert counts["conversation"] == 31
    assert counts["prompt"] == 6
    assert sum(counts.values()) == len(corpus), "a source type escaped the count"
    # Floors, asserted independently of the exact numbers above so a future
    # addition cannot quietly drop a source type below what its report cells
    # need. Prompts are floored at the number their golden queries target.
    assert counts["note"] >= 60
    assert counts["media"] >= 45
    assert counts["conversation"] >= 25
    assert counts["prompt"] >= 5


def test_golden_set_category_quotas(golden):
    counts = Counter(query.category for query in golden)
    # 45 from P1 + 15 admitted by the P2ab probe: 7 scoped, 3 negation,
    # 5 prompt. The fail-first classes' quotas are floors set FROM the probe
    # (a quota set before the measurement would have been a target to hit,
    # and hitting it would have meant force-fitting fixtures the pipeline
    # actually answers):
    #   scoped   >= 6  the spec's hard floor; 7 admitted of 8 probed
    #   prompt   >= 4  the spec's hard floor; 5 admitted of 5 (structural)
    #   negation >= 3  what the class yielded (3 of 6 probed); the other
    #                  three are recorded as measured non-failures in
    #                  golden.toml rather than deleted
    assert len(golden) == 60
    assert counts["keyword"] >= 10
    assert counts["paraphrase"] >= 10
    assert counts["vocabulary_mismatch"] >= 8
    assert counts["negative"] >= 6
    assert counts["scoped"] >= 6
    assert counts["prompt"] >= 4
    assert counts["negation"] >= 3
    # Two-sided: no category outside the declared vocabulary may appear, and
    # every REQUIRED category must be present. Both directions are now the
    # same set — `scoped` stopped being exempt when its fixtures landed.
    assert set(counts) <= set(CATEGORIES)
    assert set(REQUIRED_CATEGORIES) <= set(counts)


def test_every_declared_category_is_required_and_populated(golden):
    """The replacement for the pre-fixture sentinel.

    Task 2 shipped the scope schema against a corpus with no scoped
    fixtures, so `scoped` was declared-but-exempt and a sentinel test pinned
    that state. The exemption is gone: every declared category now has
    fixtures, so `REQUIRED_CATEGORIES` is the whole vocabulary and a
    category that loses its queries fails the validator instead of quietly
    vanishing from the report.

    The direction that still needs saying: a category must not be declared
    ahead of its fixtures again. `compositional` and `acronym` were probed
    and admitted nothing, and they are absent from `CATEGORIES` for exactly
    that reason — an empty cell in the per-capability report is
    indistinguishable from an unmeasured one.
    """
    assert set(REQUIRED_CATEGORIES) == set(CATEGORIES)
    assert SCOPED_CATEGORY in REQUIRED_CATEGORIES
    present = {query.category for query in golden}
    assert present == set(CATEGORIES), (
        f"declared but unpopulated: {sorted(set(CATEGORIES) - present)}"
    )


def test_loaded_records_are_frozen(corpus, golden):
    with pytest.raises(Exception):
        corpus[0].slug = "mutated"  # type: ignore[misc]
    with pytest.raises(Exception):
        golden[0].query = "mutated"  # type: ignore[misc]
    assert isinstance(golden[0].relevant_slugs, tuple)


def test_every_document_has_at_least_three_sentences(corpus):
    thin = {
        doc.slug: len([s for s in re.split(r"[.!?]\s", doc.content.strip()) if s.strip()])
        for doc in corpus
    }
    assert {slug: n for slug, n in thin.items() if n < 3} == {}


def test_documents_are_self_contained_and_timeless(corpus):
    """No absolute dates and no now-relative phrasing: a corpus that decays
    makes committed baselines undiagnosable later."""
    offenders = {
        doc.slug
        for doc in corpus
        if re.search(
            r"\b(19|20)\d{2}\b|\b(yesterday|today|tomorrow|last (week|month|year)|next (week|month|year))\b",
            f"{doc.title} {doc.content}",
            re.IGNORECASE,
        )
    }
    assert offenders == set()


def test_paraphrase_and_vocabulary_queries_share_no_content_word_with_targets(golden, by_slug):
    """The defining property of both vector-advantage groups.

    A single shared content word turns a paraphrase case into a keyword case
    and makes a vocabulary-mismatch case reachable without expansion — the
    measurement would then report success for a capability that never ran.
    """
    overlaps = {}
    for query in golden:
        if query.category not in ("paraphrase", "vocabulary_mismatch"):
            continue
        query_stems = _content_stems(query.query)
        for slug in query.relevant_slugs:
            shared = query_stems & _doc_stems(by_slug[slug])
            if shared:
                overlaps[(query.id, slug)] = sorted(shared)
    assert overlaps == {}


def test_keyword_queries_own_a_token_unique_to_their_targets(golden, by_slug):
    """Keyword cases are only FTS-advantage cases if some query token occurs
    in the target set and nowhere else in the corpus."""
    stems_by_slug = {slug: _doc_stems(doc) for slug, doc in by_slug.items()}
    undiscriminating = []
    for query in golden:
        if query.category != "keyword":
            continue
        targets = set(query.relevant_slugs)
        unique = [
            stem
            for stem in _content_stems(query.query)
            if {slug for slug, stems in stems_by_slug.items() if stem in stems} == targets
        ]
        if not unique:
            undiscriminating.append(query.id)
    assert undiscriminating == []


def test_negative_queries_are_about_topics_absent_from_the_corpus(golden, by_slug):
    """Every content word of a negative query is absent corpus-wide, so a hit
    is unambiguously a false positive rather than a labelling argument."""
    stems_by_slug = {slug: _doc_stems(doc) for slug, doc in by_slug.items()}
    leaks = {}
    for query in golden:
        if query.category != NEGATIVE_CATEGORY:
            continue
        for stem in sorted(_content_stems(query.query)):
            hits = sorted(slug for slug, stems in stems_by_slug.items() if stem in stems)
            if hits:
                leaks[(query.id, stem)] = hits
    assert leaks == {}


def test_the_bare_word_will_appears_nowhere_in_the_corpus(corpus):
    """`vm-no-will` ("who inherits when someone leaves no will") depends on
    the corpus never using "will" as a modal verb — the one query term that
    ordinary prose emits by accident.

    This is not a general style rule; it is a named dependency of one
    vocabulary-mismatch case, and it has been violated twice during authoring
    (both times inside a long document written in future tense). The general
    overlap tests cannot catch it, because they only inspect a query's own
    target.
    """
    offenders = {
        doc.slug
        for doc in corpus
        if re.search(r"\bwill\b", f"{doc.title} {doc.content}", re.IGNORECASE)
    }
    assert offenders == set()


#: The three P1 categories whose targets must cover media, notes AND
#: conversations. They are authored to succeed, so their spread is a choice
#: the author makes; the four-seam keyword mode is measured per source type,
#: and a category that never targets a conversation leaves that seam
#: unmeasured for it.
_ALL_TYPE_SPANNING_CATEGORIES: tuple[str, ...] = (
    "keyword",
    "paraphrase",
    "vocabulary_mismatch",
)

#: What the P2ab categories are allowed to span instead, and why. These are
#: NOT weaker versions of the rule above — each is a construction limit:
#:   scoped   a scope can only name media and notes (conversations are
#:            outside the scope vocabulary, rag_scope spec D5)
#:   prompt   a prompt query can only target a prompt document
#:   negation the class's admitted set is whatever the probe admitted; it
#:            spans two source types today and the floor says two, so a
#:            future admission cannot quietly collapse it to one
_P2AB_SPREAD_RULES: dict[str, set[str]] = {
    SCOPED_CATEGORY: {"media", "note"},
    "prompt": {"prompt"},
}


def test_each_capability_group_spans_all_three_source_types(golden, by_slug):
    """Per-mode per-category report cells must not be empty for a source type
    — the four-seam keyword mode is measured per type."""
    spread: dict[str, set[str]] = {}
    for query in golden:
        if query.category == NEGATIVE_CATEGORY:
            continue
        for slug in query.relevant_slugs:
            spread.setdefault(query.category, set()).add(by_slug[slug].source_type)

    three = {"media", "note", "conversation"}
    assert {
        category: types
        for category, types in spread.items()
        if category in _ALL_TYPE_SPANNING_CATEGORIES
    } == {category: three for category in _ALL_TYPE_SPANNING_CATEGORIES}

    for category, allowed in _P2AB_SPREAD_RULES.items():
        assert spread[category] <= allowed, (
            f"{category}: targets a source type its construction cannot "
            f"reach ({sorted(spread[category] - allowed)})"
        )
    assert spread[SCOPED_CATEGORY] == _P2AB_SPREAD_RULES[SCOPED_CATEGORY], (
        "the scoped class must exercise BOTH scopeable source types: a scope "
        "with one type never exercises the per-type allowlist split"
    )
    assert len(spread["negation"]) >= 2, (
        "the negation class has collapsed to a single source type"
    )


def test_corpus_carries_distractors_that_are_nobody_s_answer(corpus, golden):
    referenced = {slug for query in golden for slug in query.relevant_slugs}
    distractors = {doc.slug for doc in corpus} - referenced
    # Raised from 10 with the P2ab scale-up. Two thirds of the corpus is
    # now nobody's answer, which is the point: at 49 documents a top-10 was
    # a fifth of everything and every recall number was flattered by it.
    assert len(distractors) >= 60


def test_multi_target_queries_exist_so_recall_can_discriminate(golden):
    multi = [q.id for q in golden if len(q.relevant_slugs) > 1]
    assert len(multi) >= 2


def test_the_word_plant_appears_only_in_the_vector_blind_target(corpus):
    """`kw-plant-maintenance-record` rests on ONE token being corpus-unique.

    "plant" is the only stem that resolves that query to
    `note-saltmarsh-hide` (maintenance and record are everywhere), so a new
    document writing "the plant room" — the single most natural phrase in a
    corpus of site-maintenance prose — silently destroys the fixture the
    whole fusion-weighting arc was measured on.

    The general uniqueness test below would catch it, but only by reporting
    that one query "owns no unique token", which reads as a golden-set
    problem rather than as "somebody wrote a normal English word". This one
    names the actual dependency. It has already fired once during the P2ab
    scale-up, on "gaps are planted up with hawthorn".
    """
    offenders = {
        doc.slug
        for doc in corpus
        if doc.slug != "note-saltmarsh-hide"
        and re.search(r"\bplant\w*\b", f"{doc.title} {doc.content}", re.IGNORECASE)
    }
    assert offenders == set()


#: Long documents and the rare identifier planted throughout each one.
LONG_DOCS: dict[str, str] = {
    "note-fennimore-changeover": "Fennimore-3",
    "media-larkspur-turbine": "Larkspur-11",
    "conv-drayton-conveyor": "Drayton-6",
}


def test_long_documents_are_long_enough_to_split(by_slug):
    """Arithmetic guard, independent of the chunker being importable.

    ``Chunk_Lib._chunk_text_by_words`` walks ``range(0, len(words), step)``
    with ``step = chunk_size - chunk_overlap``, so chunks == ceil(words/step).
    At the default 400/100 that is a 300-word step; these documents are sized
    so the split survives any overlap from 0 (step 400) to 300 (step 100).
    """
    from tldw_chatbook.RAG_Search.simplified.config import ChunkingConfig

    config = ChunkingConfig()
    assert config.chunking_method == "words"
    for slug in LONG_DOCS:
        doc = by_slug[slug]
        words = len(f"{doc.title} {doc.content}".split())
        assert words >= 700, f"{slug}: {words} words is not comfortably multi-chunk"
        # worst case for splitting is the largest possible step: no overlap
        assert words > config.chunk_size, f"{slug}: {words} words fits one chunk"


def test_long_documents_really_split_and_repeat_their_identifier(by_slug):
    """The real chunker, not a re-implementation of it.

    Every chunk must contain the identifier: that is what makes one query
    produce several chunk-level hits on one document, which is the input the
    document-level canonicalization contract (dedup to first-hit rank) has to
    handle. If this only produced one matching chunk, the dedup path would
    stay untested by the fixtures.
    """
    chunking_service = pytest.importorskip(
        "tldw_chatbook.RAG_Search.chunking_service",
        reason="chunking extras not installed",
    )
    from tldw_chatbook.RAG_Search.simplified.config import ChunkingConfig

    config = ChunkingConfig()
    service = chunking_service.ChunkingService()

    for slug, identifier in LONG_DOCS.items():
        doc = by_slug[slug]
        chunks = service.chunk_text(
            doc.content,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            method=config.chunking_method,
        )
        assert len(chunks) >= 2, f"{slug} produced {len(chunks)} chunk(s)"
        matching = [
            chunk for chunk in chunks if identifier.lower() in chunk["text"].lower()
        ]
        assert len(matching) == len(chunks), (
            f"{slug}: identifier {identifier} reaches {len(matching)} of "
            f"{len(chunks)} chunks; it must appear in every chunk window"
        )


# --------------------------------------------------------------------------
# the fail-first classes (P2ab): the admission protocol, and the two
# construction properties the probe measured against
# --------------------------------------------------------------------------

#: Categories admitted by measured failure rather than by authoring intent.
#: Each of their fixtures must carry the probe's own `# admitted:` line.
FAIL_FIRST_CATEGORIES: tuple[str, ...] = (SCOPED_CATEGORY, "negation", "prompt")

#: `# admitted: 2026-08-10 hybrid=miss semantic=miss plain=1`
_ADMISSION_RE = re.compile(
    r"^# admitted: \d{4}-\d{2}-\d{2} "
    r"hybrid=(?:miss|\d+) semantic=(?:miss|\d+) plain=(?:miss|\d+)$"
)


def _query_blocks() -> list[tuple[str, str]]:
    """Every `[[query]]` block in golden.toml with the comments above it.

    Returns:
        ``(preamble, block)`` pairs in file order, where ``preamble`` is the
        raw text between the previous block and this one.
    """
    segments = GOLDEN_PATH.read_text(encoding="utf-8").split("[[query]]")
    return list(zip(segments[:-1], segments[1:]))


def _field(block: str, name: str) -> str:
    match = re.search(rf'^{name} = "([^"]*)"', block, re.MULTILINE)
    return match.group(1) if match else ""


def _trailing_comments(preamble: str) -> list[str]:
    lines: list[str] = []
    for line in reversed(preamble.strip("\n").split("\n")):
        if line.startswith("#"):
            lines.append(line)
        else:
            break
    return lines


def test_every_fail_first_fixture_carries_its_admission_comment():
    """THE admission protocol, as a test rather than as a convention.

    A fail-first fixture is only worth its cell if today's pipeline was
    MEASURED to fail it; the `# admitted:` line is that measurement's
    receipt, produced by `fixture_probe.admission_comment` and pasted above
    the fixture. Without it, a fixture that was merely *assumed* hard is
    indistinguishable from one that was probed — and "assumed hard" is
    exactly how P1's categories ended up at the ceiling.

    The shape is checked too, not just the prefix: a comment that has lost
    its ranks records that somebody looked, not what they saw.
    """
    missing: list[str] = []
    malformed: list[str] = []
    checked = 0
    for preamble, block in _query_blocks():
        if _field(block, "category") not in FAIL_FIRST_CATEGORIES:
            continue
        checked += 1
        comments = _trailing_comments(preamble)
        admissions = [line for line in comments if line.startswith("# admitted:")]
        if not admissions:
            missing.append(_field(block, "id"))
        elif not any(_ADMISSION_RE.match(line) for line in admissions):
            malformed.append(f"{_field(block, 'id')}: {admissions[0]!r}")
    assert checked >= 15, f"only {checked} fail-first fixtures found in the file"
    assert missing == []
    assert malformed == []


def test_scoped_queries_are_uniquely_resolvable_by_the_keyword_path_in_scope(
    golden, by_slug
):
    """Each scoped query's terms co-occur in exactly ONE in-scope document.

    This is what makes a scoped fixture a measurement rather than a wish.
    The keyword path is AND-of-terms with plural widening
    (`library_fts_query.build_fts_match_query` — the app's own expansion is
    imported here so the guard tracks the real matcher), so a scoped target
    is reachable by keyword only while the query's whole term set occurs
    together in it and nowhere else in the scope. Break that and the fixture
    stops being "findable in principle": the scoped cell would then be
    measuring a query nothing can answer, and the later scope-aware-hybrid
    flip would have nothing to flip to.
    """
    from tldw_chatbook.Library.library_fts_query import expand_keyword_term

    words_by_slug = {
        slug: set(re.findall(r"[a-z0-9]+", f"{doc.title} {doc.content}".lower()))
        for slug, doc in by_slug.items()
    }
    wrong: dict[str, list[str]] = {}
    for query in golden:
        if query.category != SCOPED_CATEGORY:
            continue
        terms = query.query.lower().split()
        hits = sorted(
            slug
            for slug in query.scope_slugs or ()
            if all(
                any(variant.lower() in words_by_slug[slug] for variant in expand_keyword_term(term))
                for term in terms
            )
        )
        if hits != sorted(query.relevant_slugs):
            wrong[query.id] = hits
    assert wrong == {}


#: The scope every scoped fixture runs under, as shipped. Pinned exactly,
#: not floored — see `test_the_shared_scoped_scope_is_the_size_its_own_
#: measurement_requires` for why a floor is the wrong shape here.
SHIPPED_SCOPE_SIZE = 100


def _scoped_scopes(golden) -> dict[str, tuple[str, ...]]:
    return {
        query.id: tuple(query.scope_slugs or ())
        for query in golden
        if query.category == SCOPED_CATEGORY
    }


def test_every_scoped_fixture_carries_the_same_scope(golden):
    """One scope, seven fixtures — asserted, because TOML cannot share a list.

    The scoped fixtures all run under the same collection ("every document
    the P2ab scale-up added that a scope can name"), and TOML has no way to
    say that once: the slug list is pasted into each fixture. Seven copies of
    a hundred slugs is seven chances to diverge, and a diverged scope is
    invisible in the report — each cell would simply be measured against a
    different haystack, which is the "plausible numbers that mean something
    else" failure in its purest form.

    This is the single definition. An intentional change to the collection
    has to change every copy identically, which is exactly the friction that
    makes it a decision rather than an edit.
    """
    scopes = _scoped_scopes(golden)
    assert len(scopes) >= 6, f"only {len(scopes)} scoped fixtures found"
    distinct = set(scopes.values())
    assert len(distinct) == 1, (
        "scoped fixtures no longer share one scope; the diverging ids are "
        f"{sorted(scopes)} with {len(distinct)} distinct slug lists"
    )


def test_the_shared_scoped_scope_is_the_size_its_own_measurement_requires(golden):
    """The scope size is a MEASURED parameter of the fixture, not a minimum.

    Lever data, from authoring (candidates probed against the shipped
    corpus, k=10, admission = target misses top-10 in hybrid AND semantic):

        scope size    scoped candidates that failed
             32                  1 of 8
             80                  6 of 8
            100                  7 of 8   <- shipped

    and, re-measured at review against the SHIPPED seven fixtures, a scope
    trimmed to 40 documents leaves only 4 of 7 failing
    (`sc-storm-overflow-record` surfaces at rank 7, `sc-sample-point-sign` at
    9, `sc-duty-board-notice` at 5). A floor of 40 would therefore have
    passed a corpus whose scoped before-number had silently risen from 0.000
    to roughly 0.43 — and the scope-aware-hybrid task would then have
    measured a flip that had mostly already happened.

    So the size is pinned exactly. A scope is only a measurement while a
    top-10 can exclude something from it; shrinking this list is not a
    tidy-up, it is a change to the instrument, and the only correct response
    to this test failing is to re-probe every scoped fixture and rewrite its
    `# admitted:` line with what the new scope actually produces.
    """
    scopes = _scoped_scopes(golden)
    sizes = {query_id: len(scope) for query_id, scope in scopes.items()}
    assert set(sizes.values()) == {SHIPPED_SCOPE_SIZE}, (
        f"scoped scope sizes are {sorted(set(sizes.values()))}, not "
        f"{SHIPPED_SCOPE_SIZE}: at 40 documents only 4 of these 7 fixtures "
        "still fail, so a trimmed scope reports a before-number that is not "
        "0.000 while every other test stays green (see this test's docstring)"
    )


#: SHA-256 over the shared scope's membership — the COMPOSITION pin that the
#: size pin above cannot be. Regenerate it only alongside a re-probe of every
#: scoped fixture: `_scope_digest(next(iter(_scoped_scopes(golden).values())))`.
SHIPPED_SCOPE_SHA256 = (
    "9678a923e4c32be3643452dd223867f3355f1f2fd78e95a2e0e77336072fc54a"
)


def _scope_digest(scope: tuple[str, ...]) -> str:
    """Order-independent digest of a scope's membership.

    Sorted, because the slug order inside the TOML list is formatting: a
    reflowed list is the same haystack and must not red the pin. Then
    length-delimited per slug, the `baseline_io._fixture_digest` idiom, so a
    character moved from the end of one slug to the start of the next is not
    hashed as no change at all.
    """
    digest = hashlib.sha256()
    for slug in sorted(scope):
        digest.update(f"{len(slug)}:".encode("ascii"))
        digest.update(slug.encode("utf-8"))
    return digest.hexdigest()


def test_the_shared_scoped_scope_is_the_composition_it_was_measured_on(golden):
    """WHICH hundred documents, not how many.

    The two tests above pin identity (all seven scopes agree) and cardinality
    (all seven are exactly 100). Together they still admit a mutation that
    changes what these fixtures measure: swap a slug for another corpus slug
    in all seven lists at once, and the scopes stay identical to each other,
    stay 100 long, and every guard stays green — while the haystack each
    scoped cell is scored against has silently changed. That gap was found by
    review during the fail-first authoring pass and is what this pin closes.

    It matters because the scoped class is authored ON its haystack: each
    fixture was admitted by probing it against THESE hundred documents (the
    `# admitted:` line in `golden.toml` records the run), and the class's
    whole claim — keyword-findable in scope, invisible to the vector leg — is
    a claim about which documents are competing. Swap the competition and the
    claim is untested, at unchanged numbers.

    The correct response to this failing is never to regenerate the constant.
    It is to re-probe every scoped fixture against the new collection and
    rewrite its `# admitted:` line with what the new scope actually produces —
    then regenerate the constant, because the instrument really did change.
    """
    scopes = _scoped_scopes(golden)
    digests = {query_id: _scope_digest(scope) for query_id, scope in scopes.items()}
    assert set(digests.values()) == {SHIPPED_SCOPE_SHA256}, (
        "the scoped fixtures' shared scope is no longer the collection these "
        "fixtures were probed against: digests "
        f"{sorted(set(digests.values()))} != {SHIPPED_SCOPE_SHA256!r}. Same "
        "size and same list in all seven files is not the same haystack (see "
        "this test's docstring)."
    )


def test_negation_queries_carry_a_cue_their_target_never_uses(golden, by_slug):
    """The mechanism of the negation class, pinned.

    A negation query names the norm ("the standard mains supply") and asks
    for the exception. It is hard for exactly one reason: the cue is carried
    by the documents asserting the norm and is ABSENT from the target, so
    the keyword paths cannot reach the target and the vector leg is pulled
    toward the norm. Write the cue into the exception document and the
    keyword path answers the query immediately; delete the norm documents
    and there is nothing for the query to be wrong about.
    """
    stems_by_slug = {slug: _doc_stems(doc) for slug, doc in by_slug.items()}
    weak: dict[str, str] = {}
    for query in golden:
        if query.category != "negation":
            continue
        target_stems: set[str] = set()
        for slug in query.relevant_slugs:
            target_stems |= stems_by_slug[slug]
        cues = {
            stem: sum(1 for stems in stems_by_slug.values() if stem in stems)
            for stem in _content_stems(query.query) - target_stems
        }
        strong = {stem: count for stem, count in cues.items() if count >= 5}
        if not strong:
            weak[query.id] = (
                f"no cue absent from the target and present in >=5 documents; "
                f"candidates: {cues}"
            )
    assert weak == {}


def test_prompt_fixtures_are_the_source_type_the_vector_index_never_holds(
    golden, by_slug
):
    """The prompt class measures the KEYWORD leg alone, by construction.

    DISCLOSED UPDATE (2026-08-11, TASK-15020/B2). This test used to be
    "the source type nothing can write": prompts had no writer, so the
    class's 0.000 was total absence. B2 shipped the writer and the engine's
    prompts sub-leg, so the queries are now answered — through the FTS leg
    only, because nothing indexes prompts semantically. The coupling worth
    pinning is therefore the surviving one: the prompt queries target only
    prompt documents, and `prompt` is exactly the source type ingestion
    writes-but-never-indexes. Without this, a prompt cell reading 0.000 in
    the semantic column would be read as a retrieval defect instead of as
    the structural fact it is.
    """
    from Tests.RAG_Eval.harness.ingest import UNINDEXED_SOURCE_TYPES

    assert set(UNINDEXED_SOURCE_TYPES) == {"prompt"}
    assert set(UNINDEXED_SOURCE_TYPES) < set(SOURCE_TYPES)

    for query in golden:
        targets = {by_slug[slug].source_type for slug in query.relevant_slugs}
        if query.category == "prompt":
            assert targets == {"prompt"}, f"{query.id}: targets {sorted(targets)}"
        else:
            assert "prompt" not in targets, (
                f"{query.id} ({query.category}) targets a prompt document, which "
                "the semantic leg cannot reach — its cell would blame retrieval "
                "for a missing index"
            )


def test_fixture_files_parse_as_plain_toml_with_the_documented_shape():
    """Guards the file-level contract independently of the loader, so a
    loader bug cannot mask a fixture that no longer matches the schema."""
    raw_corpus = tomllib.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    raw_golden = tomllib.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert set(raw_corpus) == {"doc"}
    assert set(raw_golden) == {"query"}
    assert {key for doc in raw_corpus["doc"] for key in doc} == {
        "slug", "source_type", "title", "content",
    }
    golden_keys = {key for query in raw_golden["query"] for key in query}
    assert {"id", "query", "category", "relevant_slugs"} <= golden_keys
    # `scope_slugs` is optional (scoped queries only), so it is permitted but
    # not required; nothing else may appear.
    assert golden_keys <= {
        "id", "query", "category", "relevant_slugs", "scope_slugs",
    }


# --------------------------------------------------------------------------
# validator: planted defects (inline data — the real fixtures stay untouched)
# --------------------------------------------------------------------------


def test_inline_valid_pair_is_accepted():
    assert validate(_valid_corpus(), _valid_golden()) is None


def test_duplicate_corpus_slug_is_reported():
    corpus = _valid_corpus()
    corpus.append(CorpusDoc("n1", "note", "Clashing", "One. Two. Three."))
    message = _defects(corpus, _valid_golden())
    assert "duplicate corpus slug" in message
    assert "'n1'" in message


def test_duplicate_query_id_is_reported():
    golden = _valid_golden()
    golden.append(GoldenQuery("q-kw", "alpha", "keyword", ("n1",)))
    message = _defects(_valid_corpus(), golden)
    assert "duplicate query id" in message
    assert "'q-kw'" in message


def test_unknown_relevant_slug_is_reported():
    golden = _valid_golden()
    golden[0] = GoldenQuery("q-kw", "alpha beta", "keyword", ("n1", "ghost-doc"))
    message = _defects(_valid_corpus(), golden)
    assert "unknown relevant_slug" in message
    assert "'ghost-doc'" in message
    assert "'q-kw'" in message


def test_non_negative_query_without_relevant_slugs_is_reported():
    golden = _valid_golden()
    golden[1] = GoldenQuery("q-pr", "eta theta", "paraphrase", ())
    message = _defects(_valid_corpus(), golden)
    assert "'q-pr'" in message
    assert "relevant_slugs" in message
    assert "negative" in message


def test_negative_query_with_relevant_slugs_is_reported():
    golden = _valid_golden()
    golden[3] = GoldenQuery("q-neg", "nothing here", "negative", ("n1",))
    message = _defects(_valid_corpus(), golden)
    assert "'q-neg'" in message
    assert "negative" in message


def test_repeated_relevant_slug_in_one_query_is_reported():
    golden = _valid_golden()
    golden[0] = GoldenQuery("q-kw", "alpha beta", "keyword", ("n1", "n1"))
    message = _defects(_valid_corpus(), golden)
    assert "repeats relevant_slug" in message
    assert "'n1'" in message


def test_category_with_no_queries_is_reported():
    golden = [q for q in _valid_golden() if q.category != "paraphrase"]
    message = _defects(_valid_corpus(), golden)
    assert "no queries in category 'paraphrase'" in message


def test_source_type_with_no_documents_is_reported():
    corpus = [doc for doc in _valid_corpus() if doc.source_type != "media"]
    golden = [q for q in _valid_golden() if q.id != "q-pr"]
    golden.append(GoldenQuery("q-pr", "alpha", "paraphrase", ("n1",)))
    message = _defects(corpus, golden)
    assert "no corpus documents with source_type 'media'" in message


def test_unknown_source_type_value_is_reported():
    corpus = _valid_corpus()
    corpus[0] = CorpusDoc("n1", "notes", "Note one", "A. B. C.")
    message = _defects(corpus, _valid_golden())
    assert "unknown source_type" in message
    assert "'notes'" in message


def test_unknown_category_value_is_reported():
    golden = _valid_golden()
    golden[0] = GoldenQuery("q-kw", "alpha beta", "keywords", ("n1",))
    message = _defects(_valid_corpus(), golden)
    assert "unknown category" in message
    assert "'keywords'" in message


def test_empty_corpus_and_empty_golden_set_are_reported():
    message = _defects([], [])
    assert "corpus is empty" in message
    assert "golden set is empty" in message


def test_every_defect_is_reported_in_one_error_not_just_the_first():
    """The whole point of the validator: one run, the complete defect list."""
    corpus = _valid_corpus()
    corpus.append(CorpusDoc("n1", "note", "Clashing", "One. Two. Three."))
    golden = _valid_golden()
    golden[0] = GoldenQuery("q-kw", "alpha", "keyword", ("ghost-doc",))
    golden[1] = GoldenQuery("q-pr", "eta", "paraphrase", ())
    golden[3] = GoldenQuery("q-neg", "nothing", "negative", ("m1",))
    golden.append(GoldenQuery("q-neg", "nothing again", "negative", ()))

    with pytest.raises(GoldenSetError) as excinfo:
        validate(corpus, golden)

    message = str(excinfo.value)
    assert "duplicate corpus slug" in message
    assert "unknown relevant_slug" in message
    assert "'q-pr'" in message
    assert "'q-neg'" in message
    assert "duplicate query id" in message
    assert len(excinfo.value.defects) >= 5
    # Every defect is on its own line under a counted header.
    assert message.splitlines()[0].startswith(f"{len(excinfo.value.defects)} ")


# --------------------------------------------------------------------------
# validator: the scoped category and its `scope_slugs`
#
# A scoped query measures retrieval *under a restriction*, so its scope is
# part of the measurement, not decoration. Every rule below turns a scope
# defect into a loud failure instead of a plausible number: a scope on the
# wrong category silently measures nothing, a missing scope makes a "scoped"
# cell unscoped, and a target outside its own scope scores 0.0 forever and
# reads as a retrieval regression.
# --------------------------------------------------------------------------


def test_a_valid_scoped_query_is_accepted():
    """A second scoped query, scoping BOTH scopeable source types.

    The minimal set already carries a single-type scope; this one exercises
    the union case, which is the shape `build_semantic_allowlists` returns
    one entry per type for.
    """
    golden = _valid_golden()
    golden.append(_scoped_query("q-sc-two-types"))
    assert validate(_valid_corpus(), golden) is None


def test_scope_slugs_on_a_non_scoped_category_is_reported():
    """Only the scoped category may carry a scope.

    A `scope_slugs` on a keyword query would be silently ignored by the
    runner (it scopes by category), so the query would measure unscoped
    retrieval while its fixture claims otherwise.
    """
    golden = _valid_golden()
    golden.append(_scoped_query("q-kw-scoped", category="keyword"))
    message = _defects(_valid_corpus(), golden)
    assert "'q-kw-scoped'" in message
    assert "scope_slugs" in message
    assert "scoped" in message


def test_a_scoped_query_without_scope_slugs_is_reported():
    golden = _valid_golden()
    golden.append(_scoped_query("q-sc-none", scope=None))
    message = _defects(_valid_corpus(), golden)
    assert "'q-sc-none'" in message
    assert "scope_slugs" in message


def test_an_empty_scope_slugs_list_is_reported():
    """Empty is not "unscoped": `EffectiveScope` has no scoped-with-nothing
    state, so an empty scope would resolve to a plain unscoped search."""
    golden = _valid_golden()
    golden.append(_scoped_query("q-sc-empty", scope=()))
    message = _defects(_valid_corpus(), golden)
    assert "'q-sc-empty'" in message
    assert "scope_slugs" in message


def test_unknown_scope_slug_is_reported():
    golden = _valid_golden()
    golden.append(_scoped_query("q-sc-ghost", scope=("n1", "ghost-doc")))
    message = _defects(_valid_corpus(), golden)
    assert "unknown scope_slug" in message
    assert "'ghost-doc'" in message
    assert "'q-sc-ghost'" in message


def test_repeated_scope_slug_in_one_query_is_reported():
    golden = _valid_golden()
    golden.append(_scoped_query("q-sc-dup", scope=("n1", "n1")))
    message = _defects(_valid_corpus(), golden)
    assert "repeats scope_slug" in message
    assert "'n1'" in message


def test_a_scope_slug_naming_an_unscopeable_source_type_is_reported():
    """Conversations are outside the scope vocabulary (rag_scope spec D5).

    Scoping one would build an allowlist the seam cannot honour, and the
    query would quietly measure something else.
    """
    golden = _valid_golden()
    golden.append(_scoped_query("q-sc-conv", relevant=("c1",), scope=("c1",)))
    message = _defects(_valid_corpus(), golden)
    assert "'q-sc-conv'" in message
    assert "'c1'" in message
    assert "source_type" in message


def test_a_scoped_querys_targets_must_lie_inside_its_own_scope():
    """The defect that costs a whole category: a target outside the scope is
    unreachable *by construction*, so the cell reports 0.0 recall forever and
    reads as a retrieval regression rather than as a broken fixture."""
    golden = _valid_golden()
    golden.append(_scoped_query("q-sc-outside", relevant=("c1",), scope=("n1", "m1")))
    message = _defects(_valid_corpus(), golden)
    assert "'q-sc-outside'" in message
    assert "'c1'" in message
    assert "scope" in message


def test_the_scopeable_source_types_match_the_apps_scope_vocabulary():
    """Drift guard: the harness's notion of "scopeable" is the app's.

    `SCOPEABLE_SOURCE_TYPES` is stated locally so the always-on fixture
    module stays stdlib-only, which only stays honest while it agrees with
    `rag_scope`'s own vocabulary.
    """
    from tldw_chatbook.Chat.rag_scope import _KNOWN_SOURCE_TYPES

    assert set(SCOPEABLE_SOURCE_TYPES) == set(_KNOWN_SOURCE_TYPES)
    assert set(SCOPEABLE_SOURCE_TYPES) < set(SOURCE_TYPES)


# --------------------------------------------------------------------------
# loaders: structural defects
# --------------------------------------------------------------------------


def test_load_corpus_rejects_a_file_without_the_doc_array(tmp_path):
    path = _write_toml(tmp_path, "corpus.toml", 'title = "not a corpus"\n')
    with pytest.raises(GoldenSetError) as excinfo:
        load_corpus(path)
    assert "[[doc]]" in str(excinfo.value)


def test_load_corpus_rejects_missing_and_unknown_keys(tmp_path):
    path = _write_toml(
        tmp_path,
        "corpus.toml",
        """
[[doc]]
slug = "n1"
source_type = "note"
content = "A. B. C."

[[doc]]
slug = "n2"
sourcetype = "note"
title = "Typo key"
content = "A. B. C."
""",
    )
    with pytest.raises(GoldenSetError) as excinfo:
        load_corpus(path)
    message = str(excinfo.value)
    assert "missing required key 'title'" in message
    assert "unknown key 'sourcetype'" in message
    assert len(excinfo.value.defects) >= 3  # missing title, unknown key, missing source_type


def test_load_corpus_rejects_blank_and_non_string_fields(tmp_path):
    path = _write_toml(
        tmp_path,
        "corpus.toml",
        """
[[doc]]
slug = "  "
source_type = "note"
title = "Blank slug"
content = "A. B. C."

[[doc]]
slug = "n2"
source_type = "note"
title = 7
content = "A. B. C."
""",
    )
    with pytest.raises(GoldenSetError) as excinfo:
        load_corpus(path)
    message = str(excinfo.value)
    assert "'slug'" in message
    assert "'title'" in message
    assert "non-empty string" in message


def test_load_golden_rejects_bad_relevant_slugs(tmp_path):
    path = _write_toml(
        tmp_path,
        "golden.toml",
        """
[[query]]
id = "q1"
query = "alpha"
category = "keyword"
relevant_slugs = "n1"

[[query]]
id = "q2"
query = "beta"
category = "keyword"
relevant_slugs = ["n1", 3]
""",
    )
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden(path)
    message = str(excinfo.value)
    assert "relevant_slugs" in message
    assert len(excinfo.value.defects) >= 2


def test_load_golden_reads_scope_slugs_and_defaults_to_none(tmp_path):
    """The loader must *accept* the key (it is not an unknown key) and must
    distinguish "absent" from "present but empty"; the validator, not the
    loader, decides which categories may carry it."""
    path = _write_toml(
        tmp_path,
        "golden.toml",
        """
[[query]]
id = "q1"
query = "alpha"
category = "scoped"
relevant_slugs = ["n1"]
scope_slugs = ["n1", " m1 "]

[[query]]
id = "q2"
query = "beta"
category = "keyword"
relevant_slugs = ["n1"]
""",
    )
    queries = load_golden(path)
    assert queries[0].scope_slugs == ("n1", "m1")
    assert queries[1].scope_slugs is None


def test_load_golden_rejects_bad_scope_slugs(tmp_path):
    path = _write_toml(
        tmp_path,
        "golden.toml",
        """
[[query]]
id = "q1"
query = "alpha"
category = "scoped"
relevant_slugs = ["n1"]
scope_slugs = "n1"

[[query]]
id = "q2"
query = "beta"
category = "scoped"
relevant_slugs = ["n1"]
scope_slugs = ["n1", 3]
""",
    )
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden(path)
    message = str(excinfo.value)
    assert "scope_slugs" in message
    assert len(excinfo.value.defects) >= 2


def test_load_golden_rejects_a_file_without_the_query_array(tmp_path):
    path = _write_toml(tmp_path, "golden.toml", "[[doc]]\nslug = 'n1'\n")
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden(path)
    assert "[[query]]" in str(excinfo.value)


def test_loaders_report_a_missing_file_by_path(tmp_path):
    missing = tmp_path / "absent.toml"
    with pytest.raises(GoldenSetError) as excinfo:
        load_corpus(missing)
    assert "absent.toml" in str(excinfo.value)


def test_loaders_report_unparseable_toml_by_path(tmp_path):
    path = _write_toml(tmp_path, "corpus.toml", "[[doc]\nslug = 'n1'\n")
    with pytest.raises(GoldenSetError) as excinfo:
        load_corpus(path)
    assert "corpus.toml" in str(excinfo.value)
