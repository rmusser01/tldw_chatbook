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

import re
import tomllib
from collections import Counter

import pytest

from Tests.RAG_Eval.harness.goldenset import (
    CATEGORIES,
    CORPUS_PATH,
    GOLDEN_PATH,
    NEGATIVE_CATEGORY,
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

_SUFFIXES = ("iness", "ingly", "edly", "ness", "ing", "ies", "ied", "es", "ed", "ly", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if not word.endswith(suffix):
            continue
        base = word[: -len(suffix)]
        if suffix == "ies":
            base += "y"
        if len(base) >= 4:  # never strip down to a fragment ("early" -> "ear")
            return base
    return word


def _content_stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS and len(w) > 2}


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
    """Minimal inline corpus covering all three source types."""
    return [
        CorpusDoc("n1", "note", "Note one", "Alpha beta. Gamma delta. Epsilon zeta."),
        CorpusDoc("m1", "media", "Media one", "Eta theta. Iota kappa. Lambda mu."),
        CorpusDoc("c1", "conversation", "Chat one", "Nu xi. Omicron pi. Rho sigma."),
    ]


def _valid_golden() -> list[GoldenQuery]:
    """Minimal inline golden set covering all four categories."""
    return [
        GoldenQuery("q-kw", "alpha beta", "keyword", ("n1",)),
        GoldenQuery("q-pr", "eta theta", "paraphrase", ("m1",)),
        GoldenQuery("q-vm", "nu xi", "vocabulary_mismatch", ("c1",)),
        GoldenQuery("q-neg", "nothing here", "negative", ()),
    ]


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


def test_real_fixtures_validate_clean(corpus, golden):
    assert validate(corpus, golden) is None


def test_corpus_composition_matches_the_planned_design(corpus):
    counts = Counter(doc.source_type for doc in corpus)
    assert len(corpus) == 45
    assert counts["note"] == 13
    assert counts["media"] == 19
    assert counts["conversation"] == 13
    # The brief's floors, asserted independently of the exact numbers above so
    # a future addition cannot quietly drop a source type below them.
    assert counts["note"] >= 12
    assert counts["media"] >= 18
    assert counts["conversation"] >= 12


def test_golden_set_category_quotas(golden):
    counts = Counter(query.category for query in golden)
    assert len(golden) == 41
    assert counts["keyword"] >= 10
    assert counts["paraphrase"] >= 10
    assert counts["vocabulary_mismatch"] >= 8
    assert counts["negative"] >= 6
    assert set(counts) == set(CATEGORIES)


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


def test_each_capability_group_spans_all_three_source_types(golden, by_slug):
    """Per-mode per-category report cells must not be empty for a source type
    — the four-seam keyword mode is measured per type."""
    spread: dict[str, set[str]] = {}
    for query in golden:
        if query.category == NEGATIVE_CATEGORY:
            continue
        for slug in query.relevant_slugs:
            spread.setdefault(query.category, set()).add(by_slug[slug].source_type)
    assert spread == {
        "keyword": set(SOURCE_TYPES),
        "paraphrase": set(SOURCE_TYPES),
        "vocabulary_mismatch": set(SOURCE_TYPES),
    }


def test_corpus_carries_distractors_that_are_nobody_s_answer(corpus, golden):
    referenced = {slug for query in golden for slug in query.relevant_slugs}
    distractors = {doc.slug for doc in corpus} - referenced
    assert len(distractors) >= 10


def test_multi_target_queries_exist_so_recall_can_discriminate(golden):
    multi = [q.id for q in golden if len(q.relevant_slugs) > 1]
    assert len(multi) >= 2


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
    assert {key for query in raw_golden["query"] for key in query} == {
        "id", "query", "category", "relevant_slugs",
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
