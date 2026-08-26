"""Fixture corpus + golden query set: loading and integrity validation.

The harness measures retrieval quality against hand-authored fixtures, so a
malformed fixture set does not produce an error — it produces *plausible
numbers that mean something else*. A golden query whose `relevant_slugs`
points at a slug that no longer exists scores 0 forever and reads as a
regression; a category that lost all its queries silently disappears from the
per-capability report. Both defects are cheap to detect and expensive to
notice later, so every load path validates before any query runs and reports
**every** defect in one error rather than one per run.

Contract consumed by the ingestion and runner tasks:

    load_corpus(path) -> list[CorpusDoc]
    load_golden(path) -> list[GoldenQuery]
    validate(corpus, golden) -> None          # raises GoldenSetError

`source_type` uses the app's singular ingestion vocabulary ("note" / "media" /
"conversation" / "prompt", the ITEM_TYPE_* values), so a fixture doc names its
writer without translation — except "prompt", which names a writer that does
not exist yet and is skipped at ingestion on purpose (see `SOURCE_TYPES`).

One golden field is optional: `scope_slugs`, which only the `scoped` category
may carry and every scoped query must. It names the corpus documents a scoped
query's retrieval scope allows, and the runner turns it into the production
`EffectiveScope` object (`runner.build_query_scope`). It is validated at least
as strictly as `relevant_slugs`, because a scope defect does not surface as an
error either — it surfaces as a category of numbers that quietly measures
something other than scoped retrieval.
"""
from __future__ import annotations

import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "CATEGORIES",
    "CORPUS_PATH",
    "CorpusDoc",
    "FIXTURES_DIR",
    "GOLDEN_PATH",
    "GoldenQuery",
    "GoldenSetError",
    "NEGATIVE_CATEGORY",
    "REQUIRED_CATEGORIES",
    "SCOPEABLE_SOURCE_TYPES",
    "SCOPED_CATEGORY",
    "SOURCE_TYPES",
    "load_corpus",
    "load_fixtures",
    "load_golden",
    "validate",
]

#: Source types a corpus document may declare (the app's ITEM_TYPE_* values).
#:
#: All four are written through the real writer APIs. ``prompt`` was the odd
#: one out until TASK-15020/B2 — declared, validated and then *skipped* at
#: ingestion, because there was no writer, no keyword sub-leg and no vector
#: index; its before-state was total absence, and a before-number authored
#: after the seam exists is not a before-number. B2 shipped the writer and
#: the sub-leg, so prompts are ingested and scored like everything else. The
#: one asymmetry that survives is the vector index: prompts still never
#: enter it (`ingest.UNINDEXED_SOURCE_TYPES`), so they are reachable through
#: the keyword leg alone.
SOURCE_TYPES: tuple[str, ...] = ("note", "media", "conversation", "prompt")

#: Capability categories a golden query may declare.
#:
#: The first four are P1's. The last three are P2ab's fail-first classes,
#: each admitted only where today's pipeline was MEASURED to fail it (see
#: `harness/fixture_probe.py` and the `# admitted:` comments in golden.toml).
#:
#: This vocabulary is deliberately SHORTER than the spec's candidate list.
#: `compositional` and `acronym` were authored (their documents are in the
#: corpus, now serving as distractors), probed, and admitted nothing —
#: today's pipeline answers every candidate of both at rank 1-4. A category
#: with no queries is not a placeholder for later, it is a report cell that
#: silently reads as unmeasured, so the classes that failed to fail are
#: recorded in golden.toml's measured-outcomes block instead of here.
CATEGORIES: tuple[str, ...] = (
    "keyword",
    "paraphrase",
    "vocabulary_mismatch",
    "negative",
    "scoped",
    "negation",
    "prompt",
)

#: The one category permitted to carry an empty ``relevant_slugs``.
NEGATIVE_CATEGORY = "negative"

#: The one category permitted to carry ``scope_slugs`` -- and required to.
#: A scoped query measures retrieval under a real ``EffectiveScope`` built
#: from those slugs, and is reported in its own cell rather than folded into
#: the cross-mode overall row (see `runner.run_eval`): it is asked over the
#: hundred documents of its scope while every other query is asked over the
#: whole corpus, so averaging the two mixes two haystacks into one number.
#: (Until TASK-15020/B1 the stated reason was routing -- a scope forced a
#: hybrid profile down the semantic path, making a scoped row's "hybrid" and
#: "semantic" columns one measurement. That divert is gone; the exclusion is
#: not, because the haystack reason never depended on it.)
SCOPED_CATEGORY = "scoped"

#: Categories that must have at least one query for the set to be valid.
#: Every declared category, including ``scoped``: its fixtures landed with
#: the fail-first authoring pass, so a scoped set that is authored and then
#: silently lost now fails the validator like any other category. (Task 2
#: shipped the schema with ``scoped`` exempt because requiring a category
#: nobody had authored would have failed the always-on gate for the whole
#: arc; that exemption is what this line retires.)
REQUIRED_CATEGORIES: tuple[str, ...] = CATEGORIES

#: Source types a ``scope_slugs`` entry may name. Conversations are outside
#: the scope vocabulary (rag-scope-narrowing spec D5: a scoped search
#: excludes the conversations seam outright rather than allowlisting it), so
#: scoping a conversation document would build an allowlist the seam cannot
#: honour. Stated here rather than imported so this module stays stdlib-only;
#: `test_goldenset_integrity` pins it against `rag_scope`'s own vocabulary.
SCOPEABLE_SOURCE_TYPES: tuple[str, ...] = ("media", "note")

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CORPUS_PATH = FIXTURES_DIR / "corpus.toml"
GOLDEN_PATH = FIXTURES_DIR / "golden.toml"

_CORPUS_KEYS: tuple[str, ...] = ("slug", "source_type", "title", "content")
#: ``scope_slugs`` is OPTIONAL (scoped queries only) but must be listed:
#: `_check_unknown_keys` rejects anything not named here, so a loader that
#: did not know the key would reject every scoped fixture as malformed.
_GOLDEN_KEYS: tuple[str, ...] = (
    "id", "query", "category", "relevant_slugs", "scope_slugs",
)


class GoldenSetError(Exception):
    """Raised when the corpus or golden set is malformed.

    Carries the full defect list (``.defects``) and renders every defect in
    the message: fixing fixtures one error per run is how a broken golden set
    survives three commits.
    """

    def __init__(self, defects: Iterable[str], context: Any = None) -> None:
        self.defects: tuple[str, ...] = tuple(defects)
        self.context = context
        header = f"{len(self.defects)} defect(s)"
        if context is not None:
            header += f" in {context}"
        body = "\n".join(f"  - {defect}" for defect in self.defects)
        super().__init__(f"{header}:\n{body}" if body else f"{header}.")


@dataclass(frozen=True, slots=True)
class CorpusDoc:
    """One fixture document, as the real writers will ingest it."""

    slug: str
    source_type: str
    title: str
    content: str


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    """One golden query.

    ``relevant_slugs`` is a tuple of **fixture slugs**, never DB ids: the real
    writer APIs assign autoincrement ids per run, so ingestion records the
    slug -> runtime-id mapping and the metric layer resolves through it.

    ``scope_slugs`` carries the same kind of slugs for the same reason, and
    is ``None`` for every category except ``scoped``: ``None`` means "run
    this query unscoped", which is what every pre-scope query means and must
    keep meaning. An empty tuple is NOT that -- it is rejected by `validate`,
    because ``EffectiveScope`` has no scoped-with-nothing state and an empty
    scope would silently resolve to an unrestricted search.
    """

    id: str
    query: str
    category: str
    relevant_slugs: tuple[str, ...]
    scope_slugs: tuple[str, ...] | None = None


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise GoldenSetError(["fixture file does not exist"], context=path) from None
    except tomllib.TOMLDecodeError as exc:
        raise GoldenSetError([f"not valid TOML: {exc}"], context=path) from None


def _clean_string_field(
    entry: dict[str, Any], key: str, label: str, defects: list[str]
) -> str:
    """Return the stripped value of a required string field, recording a
    defect (and returning "") when it is absent, mistyped, or blank."""
    if key not in entry:
        defects.append(f"{label}: missing required key {key!r}")
        return ""
    value = entry[key]
    if not isinstance(value, str) or not value.strip():
        defects.append(
            f"{label}: field {key!r} must be a non-empty string (got {value!r})"
        )
        return ""
    return value.strip()


def _slug_list_field(
    entry: dict[str, Any], key: str, label: str, defects: list[str]
) -> tuple[str, ...] | None:
    """Parse a list-of-slugs field, recording a defect per malformed entry.

    Args:
        entry: The raw query table.
        key: Field name (``relevant_slugs`` or ``scope_slugs``).
        label: Human label for defect messages.
        defects: Accumulator, appended to in place.

    Returns:
        The stripped slugs as a tuple (``()`` for a present-but-empty list),
        or ``None`` when the key is absent or its value was malformed. The
        caller distinguishes those two cases: absence is legal for
        ``scope_slugs`` and a separate defect for ``relevant_slugs``, while a
        malformed value has already been recorded here.
    """
    if key not in entry:
        return None
    raw_slugs = entry[key]
    if not isinstance(raw_slugs, list):
        defects.append(
            f"{label}: field {key!r} must be a list of corpus slugs "
            f"(got {type(raw_slugs).__name__})"
        )
        return None
    bad = False
    for position, slug in enumerate(raw_slugs):
        if not isinstance(slug, str) or not slug.strip():
            defects.append(
                f"{label}: {key}[{position}] must be a non-empty string "
                f"(got {slug!r})"
            )
            bad = True
    if bad:
        return None
    return tuple(slug.strip() for slug in raw_slugs)


def _check_unknown_keys(
    entry: dict[str, Any], allowed: tuple[str, ...], label: str, defects: list[str]
) -> None:
    for key in sorted(set(entry) - set(allowed)):
        defects.append(
            f"{label}: unknown key {key!r} (allowed: {', '.join(allowed)})"
        )


def _entries(raw: dict[str, Any], array_name: str, path: Path) -> list[dict[str, Any]]:
    entries = raw.get(array_name)
    if not isinstance(entries, list) or not entries:
        raise GoldenSetError(
            [
                f"file has no non-empty [[{array_name}]] array "
                f"(expected one [[{array_name}]] table per entry)"
            ],
            context=path,
        )
    return entries


def load_corpus(path: Path | str) -> list[CorpusDoc]:
    """Load the fixture corpus from ``path``.

    Args:
        path: Path to a corpus TOML file (``[[doc]]`` tables).

    Returns:
        The documents in file order.

    Raises:
        GoldenSetError: The file is missing, unparseable, or any entry has a
            missing/unknown/blank field. All structural defects are reported
            together.
    """
    path = Path(path)
    entries = _entries(_read_toml(path), "doc", path)

    defects: list[str] = []
    docs: list[CorpusDoc] = []
    for index, entry in enumerate(entries, start=1):
        label = f"doc #{index}"
        if not isinstance(entry, dict):
            defects.append(f"{label}: expected a table, got {type(entry).__name__}")
            continue
        _check_unknown_keys(entry, _CORPUS_KEYS, label, defects)
        values = {
            key: _clean_string_field(entry, key, label, defects)
            for key in _CORPUS_KEYS
        }
        if all(values.values()):
            docs.append(CorpusDoc(**values))

    if defects:
        raise GoldenSetError(defects, context=path)
    return docs


def load_golden(path: Path | str) -> list[GoldenQuery]:
    """Load the golden query set from ``path``.

    Args:
        path: Path to a golden-set TOML file (``[[query]]`` tables).

    Returns:
        The queries in file order.

    Raises:
        GoldenSetError: The file is missing, unparseable, or any entry has a
            missing/unknown/blank field or a malformed ``relevant_slugs``. All
            structural defects are reported together.
    """
    path = Path(path)
    entries = _entries(_read_toml(path), "query", path)

    defects: list[str] = []
    queries: list[GoldenQuery] = []
    for index, entry in enumerate(entries, start=1):
        label = f"query #{index}"
        if not isinstance(entry, dict):
            defects.append(f"{label}: expected a table, got {type(entry).__name__}")
            continue
        _check_unknown_keys(entry, _GOLDEN_KEYS, label, defects)
        values = {
            key: _clean_string_field(entry, key, label, defects)
            for key in ("id", "query", "category")
        }

        if "relevant_slugs" not in entry:
            defects.append(f"{label}: missing required key 'relevant_slugs'")
        slugs = _slug_list_field(entry, "relevant_slugs", label, defects) or ()
        # Absent `scope_slugs` stays None ("run unscoped"), which is what
        # every non-scoped query means; `validate` decides which categories
        # may carry it.
        scope_slugs = _slug_list_field(entry, "scope_slugs", label, defects)

        if all(values.values()):
            queries.append(
                GoldenQuery(**values, relevant_slugs=slugs, scope_slugs=scope_slugs)
            )

    if defects:
        raise GoldenSetError(defects, context=path)
    return queries


def _check_scope_slugs(
    query: GoldenQuery,
    known_slugs: set[str],
    source_type_by_slug: dict[str, str],
    defects: list[str],
) -> None:
    """Check one query's ``scope_slugs`` against every scope rule.

    Each rule turns a scope defect into a loud failure rather than a
    plausible number: a scope on the wrong category is silently ignored by
    the runner (which scopes by category), a missing or empty scope makes a
    "scoped" cell an ordinary unscoped measurement, an unscopeable source
    type builds an allowlist the seam cannot honour, and a target outside its
    own scope is unreachable by construction -- it scores 0.0 forever and
    reads as a retrieval regression.

    Args:
        query: The query to check.
        known_slugs: Every slug the corpus defines.
        source_type_by_slug: Corpus slug -> source_type.
        defects: Accumulator, appended to in place.
    """
    if query.category != SCOPED_CATEGORY:
        if query.scope_slugs is not None:
            defects.append(
                f"query {query.id!r}: category {query.category!r} must not "
                f"carry scope_slugs (only {SCOPED_CATEGORY!r} may be scoped)"
            )
        return

    if query.scope_slugs is None:
        defects.append(
            f"query {query.id!r}: category {SCOPED_CATEGORY!r} requires "
            "scope_slugs (a scoped query without a scope measures unscoped "
            "retrieval in a scoped cell)"
        )
        return
    if not query.scope_slugs:
        defects.append(
            f"query {query.id!r}: scope_slugs is empty; there is no "
            "scoped-with-nothing state, so an empty scope would search "
            "everything"
        )
        return

    for slug, count in sorted(Counter(query.scope_slugs).items()):
        if count > 1:
            defects.append(f"query {query.id!r}: repeats scope_slug {slug!r}")
        if slug not in known_slugs:
            defects.append(
                f"query {query.id!r}: unknown scope_slug {slug!r} "
                "(no corpus document has that slug)"
            )
        elif source_type_by_slug[slug] not in SCOPEABLE_SOURCE_TYPES:
            defects.append(
                f"query {query.id!r}: scope_slug {slug!r} has source_type "
                f"{source_type_by_slug[slug]!r}, which is outside the scope "
                f"vocabulary (scopeable: {', '.join(SCOPEABLE_SOURCE_TYPES)})"
            )

    # If a future fixture class wants a scoped query with NO in-scope answer
    # (a scoped negative — "does scope suppress the out-of-scope match"), it
    # needs its own category or an explicit flag, NOT a relaxation of this
    # rule: relaxing it would also stop catching the ordinary case this rule
    # exists for, where the target was simply left out of the scope by
    # mistake, and the two are indistinguishable from the fixture alone.
    outside = [
        slug for slug in query.relevant_slugs if slug not in set(query.scope_slugs)
    ]
    if outside:
        defects.append(
            f"query {query.id!r}: relevant_slugs {outside} lie outside its own "
            "scope, so they can never be retrieved and the cell reports 0.0 "
            "forever"
        )


def validate(corpus: Sequence[CorpusDoc], golden: Sequence[GoldenQuery]) -> None:
    """Check the corpus and golden set against every integrity rule.

    Args:
        corpus: Loaded fixture documents.
        golden: Loaded golden queries.

    Returns:
        None, when the pair is internally consistent.

    Raises:
        GoldenSetError: Naming **every** defect found — duplicate slugs or
            query ids, unknown ``source_type``/``category`` values, a
            ``relevant_slugs`` entry with no such document, an empty
            ``relevant_slugs`` on a non-negative query (or a non-empty one on
            a negative query), a REQUIRED category with no queries, a source
            type with no documents, or any broken scope rule (see
            ``_check_scope_slugs``).
    """
    defects: list[str] = []

    if not corpus:
        defects.append("corpus is empty")
    if not golden:
        defects.append("golden set is empty")

    for slug, count in sorted(Counter(doc.slug for doc in corpus).items()):
        if count > 1:
            defects.append(f"duplicate corpus slug: {slug!r} appears {count} times")

    for doc in corpus:
        if doc.source_type not in SOURCE_TYPES:
            defects.append(
                f"corpus doc {doc.slug!r}: unknown source_type "
                f"{doc.source_type!r} (expected one of {', '.join(SOURCE_TYPES)})"
            )

    for query_id, count in sorted(Counter(query.id for query in golden).items()):
        if count > 1:
            defects.append(f"duplicate query id: {query_id!r} appears {count} times")

    known_slugs = {doc.slug for doc in corpus}
    source_type_by_slug = {doc.slug: doc.source_type for doc in corpus}
    for query in golden:
        _check_scope_slugs(query, known_slugs, source_type_by_slug, defects)
        if query.category not in CATEGORIES:
            defects.append(
                f"query {query.id!r}: unknown category {query.category!r} "
                f"(expected one of {', '.join(CATEGORIES)})"
            )

        for slug, count in sorted(Counter(query.relevant_slugs).items()):
            if count > 1:
                defects.append(f"query {query.id!r}: repeats relevant_slug {slug!r}")
            if slug not in known_slugs:
                defects.append(
                    f"query {query.id!r}: unknown relevant_slug {slug!r} "
                    "(no corpus document has that slug)"
                )

        if query.category == NEGATIVE_CATEGORY:
            if query.relevant_slugs:
                defects.append(
                    f"query {query.id!r}: category 'negative' must have empty "
                    f"relevant_slugs, got {list(query.relevant_slugs)}"
                )
        elif not query.relevant_slugs:
            defects.append(
                f"query {query.id!r}: category {query.category!r} has empty "
                "relevant_slugs (empty is allowed only for category 'negative')"
            )

    if golden:
        present_categories = {query.category for query in golden}
        for category in REQUIRED_CATEGORIES:
            if category not in present_categories:
                defects.append(
                    f"no queries in category {category!r} — the per-capability "
                    "report would silently lose that cell"
                )

    if corpus:
        present_types = {doc.source_type for doc in corpus}
        for source_type in SOURCE_TYPES:
            if source_type not in present_types:
                defects.append(
                    f"no corpus documents with source_type {source_type!r} — "
                    "the four-seam keyword mode would go unmeasured for it"
                )

    if defects:
        raise GoldenSetError(defects)


def load_fixtures(
    corpus_path: Path | str = CORPUS_PATH,
    golden_path: Path | str = GOLDEN_PATH,
) -> tuple[list[CorpusDoc], list[GoldenQuery]]:
    """Load and validate both fixture files.

    The fail-fast entry point for the harness: integrity is checked before any
    query runs, so a malformed fixture set can never reach the metric layer.

    Args:
        corpus_path: Corpus TOML path (defaults to the shipped fixture).
        golden_path: Golden-set TOML path (defaults to the shipped fixture).

    Returns:
        The validated ``(corpus, golden)`` pair.

    Raises:
        GoldenSetError: On any structural or integrity defect.
    """
    corpus = load_corpus(corpus_path)
    golden = load_golden(golden_path)
    validate(corpus, golden)
    return corpus, golden
