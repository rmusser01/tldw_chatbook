"""Per-row expansion policy for Library retrieval rows (TASK-16174, Phase P).

Since TASK-16071's rank-fair merge, most of what a top-M consumer is fed is
label-only: media rows say ``Matched media · {type}`` and conversation rows
say ``Matched conversation · N messages``
(``library_local_rag_search_service._media_row``/``_conversation_row``), both
with an empty ``chunk_id``. Phase T shipped ``expand_document`` so an agent
can see behind such a row; this module is the half that TELLS it which rows
those are, instead of leaving it to infer them from prose.

The helper is pure -- string and shape inspection only, no I/O, no DB, no
config -- so the policy can be unit-tested per branch and reused by any
payload builder. ``Agents/library_rag_tool_provider._project_row`` is its
first consumer.

**Label detection is a TRIPLE**, never the label text alone:
``provenance.source_type`` is a label-bearing seam AND ``chunk_id`` is empty
AND the snippet opens with that seam's label prefix. Any weaker rule
misfires in both directions -- a note that quotes "Matched media · pdf"
would be called a label, and a real media chunk whose text happens to open
with those words would be too, hiding the content the agent already has.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Seams ``expand_document`` (Phase T) can actually fetch. A row of any other
#: type gets no hint: a hint the agent cannot act on is worse than silence.
#:
#: **Singulars only, deliberately** (TASK-16688 AC#2, from TASK-16174's final
#: review finding 6). ``library_local_rag_search_service.
#: _SEMANTIC_SOURCE_TYPE_MAP`` also treats the VARIANT spellings ``notes``,
#: ``media_chunk``, ``conversations``, ``chat`` and ``prompts`` as live raw
#: provenance values, so a row carrying one would get no hint here and --
#: by the Phase P/T shared precondition -- no identity either. That gap was
#: MEASURED before it was decided: TASK-16588's route probe counted variant
#: rows at **0 across all 340 rows** on four (index x route) arms, carrying a
#: positive control proving its detector fires on every variant and on no
#: singular (``Docs/superpowers/qa/2026-08-16-rag-semantic-identity/report.md``,
#: "Canonicalization-variant rows"), and today's local indexer
#: (``RAG_Search/ingestion_indexing.py``) stamps only singulars. Broadening
#: the allowlist on a measured zero would ship agent-facing surface with no
#: producer, so the exclusion stands; it is pinned (both directions) by
#: ``Tests/Library/test_library_expand_policy.py``'s variant tests. Revisit
#: only if a writer starts emitting a variant into chunk metadata -- e.g.
#: third-party or legacy index entries -- which is the live hazard this note
#: exists to keep discoverable.
EXPANDABLE_SOURCE_TYPES: tuple[str, ...] = ("note", "media", "conversation", "prompt")

#: Seams whose keyword rows are rendered as a LABEL rather than as content.
#: The prefixes are the stable head of the 15700-era label contract; the
#: separator and tail (`· pdf`, `· 7 messages`) are deliberately not matched.
LABEL_SNIPPET_PREFIXES: dict[str, str] = {
    "media": "Matched media",
    "conversation": "Matched conversation",
}

#: Mirrors the Library RAG tool provider's snippet projection cap. Callers
#: pass their own cap (`snippet_cap=`) so the two cannot drift silently; this
#: default only serves callers that project at the same width.
DEFAULT_SNIPPET_CAP = 1200

#: Suffixes an upstream excerpt uses to say "there is more".
TRUNCATION_MARKERS: tuple[str, ...] = ("…", "...")

REASON_LABEL_ONLY = "label_only"
REASON_TRUNCATED_SNIPPET = "truncated_snippet"
REASON_TEXT_BEARING = "text_bearing"


def _source_type(row: Mapping[str, Any]) -> str:
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        return ""
    return str(provenance.get("source_type") or "").strip().lower()


def _is_label_only(source_type: str, chunk_id: str, snippet: str) -> bool:
    """The triple: label-bearing seam + no chunk + the seam's label prefix."""
    prefix = LABEL_SNIPPET_PREFIXES.get(source_type)
    if prefix is None or chunk_id:
        return False
    return snippet.lstrip().startswith(prefix)


def _is_truncated(snippet: str, snippet_cap: int | None) -> bool:
    if snippet.rstrip().endswith(TRUNCATION_MARKERS):
        return True
    return bool(snippet_cap) and len(snippet) > snippet_cap


def expand_hint(
    row: Mapping[str, Any],
    *,
    snippet_cap: int | None = DEFAULT_SNIPPET_CAP,
) -> dict[str, Any] | None:
    """Say whether following this row into its document would add anything.

    Args:
        row: A retrieval row in mapping form -- ``source_id``, ``chunk_id``,
            ``snippet`` and ``provenance`` (whose ``source_type`` names the
            seam). Any other keys are ignored; the row is never mutated.
        snippet_cap: The width the caller projects snippets to, used to spot
            a snippet the payload itself cut. ``None`` disables the
            length test (the marker test still applies).

    Returns:
        ``{"expandable": bool, "reason": "label_only" | "truncated_snippet" |
        "text_bearing"}``, or ``None`` when the row cannot be expanded at
        all (unknown/absent ``source_type``, or no ``source_id`` to expand).
        The hint carries no identity or provenance -- only the verdict.
    """
    if not isinstance(row, Mapping):
        return None
    source_type = _source_type(row)
    if source_type not in EXPANDABLE_SOURCE_TYPES:
        return None
    if not str(row.get("source_id") or "").strip():
        return None

    snippet = str(row.get("snippet") or "")
    chunk_id = str(row.get("chunk_id") or "").strip()

    if _is_label_only(source_type, chunk_id, snippet):
        return {"expandable": True, "reason": REASON_LABEL_ONLY}
    if _is_truncated(snippet, snippet_cap):
        return {"expandable": True, "reason": REASON_TRUNCATED_SNIPPET}
    return {"expandable": False, "reason": REASON_TEXT_BEARING}


__all__ = [
    "DEFAULT_SNIPPET_CAP",
    "EXPANDABLE_SOURCE_TYPES",
    "LABEL_SNIPPET_PREFIXES",
    "REASON_LABEL_ONLY",
    "REASON_TEXT_BEARING",
    "REASON_TRUNCATED_SNIPPET",
    "TRUNCATION_MARKERS",
    "expand_hint",
]
