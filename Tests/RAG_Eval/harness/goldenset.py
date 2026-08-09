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
"conversation", the ITEM_TYPE_* values), so a fixture doc names its writer
without translation.
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
    "SOURCE_TYPES",
    "load_corpus",
    "load_fixtures",
    "load_golden",
    "validate",
]

#: Source types a corpus document may declare (the app's ITEM_TYPE_* values).
SOURCE_TYPES: tuple[str, ...] = ("note", "media", "conversation")

#: Capability categories a golden query may declare.
CATEGORIES: tuple[str, ...] = (
    "keyword",
    "paraphrase",
    "vocabulary_mismatch",
    "negative",
)

#: The one category permitted to carry an empty ``relevant_slugs``.
NEGATIVE_CATEGORY = "negative"

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
CORPUS_PATH = FIXTURES_DIR / "corpus.toml"
GOLDEN_PATH = FIXTURES_DIR / "golden.toml"

_CORPUS_KEYS: tuple[str, ...] = ("slug", "source_type", "title", "content")
_GOLDEN_KEYS: tuple[str, ...] = ("id", "query", "category", "relevant_slugs")


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
    """

    id: str
    query: str
    category: str
    relevant_slugs: tuple[str, ...]


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

        slugs: tuple[str, ...] = ()
        if "relevant_slugs" not in entry:
            defects.append(f"{label}: missing required key 'relevant_slugs'")
        else:
            raw_slugs = entry["relevant_slugs"]
            if not isinstance(raw_slugs, list):
                defects.append(
                    f"{label}: field 'relevant_slugs' must be a list of corpus "
                    f"slugs (got {type(raw_slugs).__name__})"
                )
            else:
                bad = False
                for position, slug in enumerate(raw_slugs):
                    if not isinstance(slug, str) or not slug.strip():
                        defects.append(
                            f"{label}: relevant_slugs[{position}] must be a "
                            f"non-empty string (got {slug!r})"
                        )
                        bad = True
                if not bad:
                    slugs = tuple(slug.strip() for slug in raw_slugs)

        if all(values.values()):
            queries.append(GoldenQuery(**values, relevant_slugs=slugs))

    if defects:
        raise GoldenSetError(defects, context=path)
    return queries


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
            a negative query), a category with no queries, or a source type
            with no documents.
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
    for query in golden:
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
        for category in CATEGORIES:
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
