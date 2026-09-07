# Tests/RAG_Eval/harness/canonicalize.py
"""Seam rows -> fixture slugs, at document level.

Retrieval is measured per **document**, but the seam returns *rows*: a row
is a chunk in semantic/hybrid mode and a whole item in keyword mode, and one
document can occupy several of the top-k slots. Scoring rows directly would
make a three-chunk document count as three retrieved documents and score
precision against a ground truth that only ever names documents once.
`rows_to_doc_ids` collapses rows to the fixture slugs the golden set is
written in, keeping each document's first-hit rank.

**The provenance keys, pinned.** Every row the seam emits is built by one of
five functions in `tldw_chatbook/Library/library_local_rag_search_service.py`
and all five agree on the two keys this module reads —
``row["provenance"]["source_type"]`` and ``row["source_id"]`` — but *not* on
what they put in them:

===================  ==============================  ======================
row builder          provenance["source_type"]       source_id
===================  ==============================  ======================
``_note_row``        ``"note"``                      ``str(note id)``
``_media_row``       ``"media"``                     ``str(media id)``
``_conversation_row````"conversation"``              ``str(conversation id)``
``_prompt_row``      ``"prompt"``                    ``str(prompt local_id)``
``_semantic_row``    whatever the chunk's index      ``metadata["source_id"]``
                     metadata carries under          or ``metadata
                     ``source_type`` /``item_type``  ["document_id"]`` or
                     /``type`` (the app's indexers   ``SearchResult.id``
                     write ITEM_TYPE_* : ``media`` /
                     ``note`` / ``conversation``;
                     the engine's KEYWORD leg also
                     stamps ``prompt``, which no
                     indexer writes — TASK-15020/B2)
===================  ==============================  ======================

Two consequences, both load-bearing:

1. **The hybrid FTS leg's `source_id` is a document id, not a source id.**
   `RAGService._keyword_search` builds its metadata from scratch with
   ``doc_id`` — a key `_semantic_row` never reads — and no ``source_id``, so
   `_semantic_row` falls through to ``SearchResult.id``, which is
   ``f"media_{id}"``. Every hybrid keyword hit would otherwise map to
   nothing, and hybrid mode would score as if its keyword leg did not exist.
   So a lookup miss retries once with a leading ``f"{source_type}_"``
   stripped, and only when *that* hits. The raw id is what survives into an
   unknown id — stripping is a lookup fallback, never a rewrite.
2. **`source_type` arrives in several vocabularies.** The writers use the
   singular ITEM_TYPE_* values; the seam's own post-filter
   (`_SEMANTIC_SOURCE_TYPE_MAP`) also accepts ``notes``/``conversations``/
   ``chat``/``media_chunk``. `canonical_source_type` folds them to the
   singular form the runtime's `slug_to_source` map is keyed by.

**Unmapped rows are kept, not dropped** (they become
``"unknown:<source_type>:<source_id>"``). A canonicalizer that dropped them
would make precision answer "of the documents I recognized, how many were
right" — a number that *improves* when retrieval returns more garbage. A
stray row from another source, a row whose id no fixture claims: both occupy
a top-k slot in the product, so they occupy one here.

A **prompt** hit used to be the standing example of that (nothing wrote
prompts, so any prompt row was by definition unclaimed). TASK-15020/B2 gave
prompts a writer and a keyword sub-leg, so a prompt row now resolves to its
fixture slug like any other and is SCORED rather than counted as noise. The
alias table below already carried ``prompt``/``prompts``, which is why this
module needed no behavioural change for B2 — only this sentence, which had
become false.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SOURCE_TYPE_ALIASES",
    "UNKNOWN_PREFIX",
    "canonical_source_type",
    "rows_to_doc_ids",
    "slug_lookup_from",
]

#: Prefix marking a retrieved row that no fixture document claims.
UNKNOWN_PREFIX = "unknown:"

#: Raw provenance ``source_type`` values -> the singular ITEM_TYPE_*
#: vocabulary `EvalRuntime.slug_to_source` is keyed by. Mirrors the seam's
#: `_SEMANTIC_SOURCE_TYPE_MAP` (which folds the same aliases the other way,
#: to the plural scope-toggle identifiers) plus the prompt seam's value.
SOURCE_TYPE_ALIASES: dict[str, str] = {
    "note": "note",
    "notes": "note",
    "media": "media",
    "media_chunk": "media",
    "conversation": "conversation",
    "conversations": "conversation",
    "chat": "conversation",
    "prompt": "prompt",
    "prompts": "prompt",
}


def canonical_source_type(value: Any) -> str:
    """Fold a raw provenance ``source_type`` to the singular vocabulary.

    Args:
        value: The raw value, which may be missing, non-string, plural, or a
            vocabulary this build does not know.

    Returns:
        The canonical singular type, the stripped lowercase input when it is
        unrecognized (so an unknown id still names what came back), or ``""``
        when there is no usable value at all.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    return SOURCE_TYPE_ALIASES.get(text, text)


def slug_lookup_from(
    slug_to_source: Mapping[str, tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Invert `EvalRuntime.slug_to_source` into the lookup `rows_to_doc_ids` takes.

    Args:
        slug_to_source: Fixture slug -> ``(source_type, source_id)``, as the
            ingestion runtime records it.

    Returns:
        ``(canonical source_type, source_id) -> slug``.

    Raises:
        ValueError: Two slugs claim the same source row. That would make
            every hit on that row scoreable two ways, so it is a fixture or
            ingestion defect, not something to resolve arbitrarily.
    """
    lookup: dict[tuple[str, str], str] = {}
    for slug, (source_type, source_id) in slug_to_source.items():
        key = (canonical_source_type(source_type), str(source_id))
        existing = lookup.get(key)
        if existing is not None:
            raise ValueError(
                f"slugs {existing!r} and {slug!r} both map to "
                f"{key[0]} {key[1]!r}; one source row cannot be two documents"
            )
        lookup[key] = slug
    return lookup


def _resolve(
    source_type: str, source_id: str, lookup: Mapping[tuple[str, str], str]
) -> str | None:
    """Resolve one row's provenance to a slug, or None when nothing claims it."""
    slug = lookup.get((source_type, source_id))
    if slug is not None:
        return slug
    # The hybrid FTS leg's document-id form ("media_7"). Retried only on a
    # miss, and only when it actually hits, so a bare id that happens to look
    # like a prefixed one can never be rewritten into someone else's row.
    prefix = f"{source_type}_"
    if source_type and source_id.startswith(prefix):
        return lookup.get((source_type, source_id[len(prefix) :]))
    return None


def rows_to_doc_ids(
    rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    slug_lookup: Mapping[tuple[str, str], str],
) -> list[str]:
    """Collapse seam rows to ranked, deduplicated document ids.

    Args:
        rows: The seam's ``result["results"]`` list, in rank order.
        slug_lookup: ``(canonical source_type, source_id) -> slug``, from
            `slug_lookup_from`.

    Returns:
        Fixture slugs (and ``"unknown:<type>:<id>"`` ids for rows no fixture
        claims), in first-hit rank order, each appearing once.

    Raises:
        TypeError: A row is not a mapping. The seam's row builders all
            return dicts; anything else is a shape change, and skipping it
            would quietly delete a retrieved result from the measurement.
    """
    doc_ids: list[str] = []
    seen: set[str] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"row {position} is a {type(row).__name__}, not a mapping: "
                f"{row!r}"
            )
        provenance = row.get("provenance")
        raw_type = (
            provenance.get("source_type") if isinstance(provenance, Mapping) else None
        )
        source_type = canonical_source_type(raw_type)
        raw_source_id = row.get("source_id")
        source_id = "" if raw_source_id is None else str(raw_source_id)
        doc_id = _resolve(source_type, source_id, slug_lookup)
        if doc_id is None:
            doc_id = f"{UNKNOWN_PREFIX}{source_type}:{source_id}"
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return doc_ids
