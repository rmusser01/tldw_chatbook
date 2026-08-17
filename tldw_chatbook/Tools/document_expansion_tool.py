"""Agentic document expansion: follow a retrieval hit into its document.

TASK-16174 Phase T. An agent could already RETRIEVE (the Console Library
tool, `Agents/library_rag_tool_provider.py`) but could not FOLLOW a hit
into the document behind it -- and since TASK-16071's rank-fair merge, 54%
of the rows a top-M consumer is fed are LABEL-ONLY: media rows say
"Matched media · {type}" and conversation rows "Matched conversation ·
N messages" (`Library/library_local_rag_search_service.py`'s `_media_row`
/`_conversation_row`), both with an empty `chunk_id`. The majority of what
an agent saw was a label it could not see behind.

This tool is the pull-based answer: budgeted, gated OFF by default, and
risk-tagged so an inherited `allow` is floored to `ask`. It is deliberately
INDEPENDENT of reranking (AC#6) -- it consumes final result rows whatever
produced them. That independence has since been worth having: TASK-16965
implemented `cross_encoder` (which TASK-3502 had left unimplemented) and
MEASURED it as net harmful on average, so a tool that presumed a working
reranker would now be built on a stage nobody recommends turning on.

**The contract works from exactly what a row carries.** `source_type` +
`source_id` is enough (that is all a label-only row has); `chunk_start`
refines the window when the row is a semantic hit; `offset` walks a long
document without re-querying. `chunk_id` is NOT a parameter: it is an INDEX
(`f"{doc_id}_chunk_{i}"`), not a character offset, so it could never anchor
a window -- it is swallowed by `**_provenance` so a pasted row still works,
and shipping it as agent-facing schema would be the same dead knob this
arc's Phase K retired one layer down.

**Identity, and why the fallbacks exist.** A semantic row's `source_id` is
`metadata["source_id"] || metadata["document_id"] || the chroma point id`
(`_semantic_row`), so for some rows the point id is what surfaces and the
real document identity is only in the provenance extras. `note_id`,
`media_id` and `doc_id` are therefore accepted as optional identity
fallbacks -- an agent can paste a row's provenance verbatim. `doc_id` is
the indexer's PREFIXED document id (`f"note_{id}"`, `f"media_{id}"`,
`f"conversation_{id}"` -- see `RAG_Search/ingestion_indexing.py`), so a
`<source_type>_` prefix is stripped as a second candidate.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

from loguru import logger

from . import Tool

#: Default character budget for one expansion. A module constant, not a new
#: `[tools]` key: this arc's Phase K just RETIRED three config knobs that
#: nothing read, and shipping another user-switchable surface with no
#: consumer would repeat exactly that. A caller that wants a different
#: window passes `max_chars`.
DEFAULT_MAX_CHARS = 8000

#: Ceiling the tool will never exceed regardless of what is asked -- the
#: budget is a promise to the payload the agent is assembling, so an
#: absurd `max_chars` is capped rather than honored.
HARD_MAX_CHARS = 32000

#: The four seams the Library retrieval rows can name (`provenance.
#: source_type`). Anything else returns `status="unsupported"` rather than
#: guessing.
SUPPORTED_SOURCE_TYPES: Tuple[str, ...] = ("note", "media", "conversation", "prompt")

#: Transcript ceiling for one conversation read. Bounds the DB work; the
#: character budget then bounds what is returned.
MAX_TRANSCRIPT_MESSAGES = 500

#: Said out loud when a conversation is longer than that ceiling. Without
#: it the payload's `total_size` -- the length of the RENDERED PREFIX -- is
#: read as the whole document's size, and a partial read is indistinguishable
#: from a complete one.
MESSAGE_CAP_NOTE = (
    f"Only the first {MAX_TRANSCRIPT_MESSAGES} messages of this conversation "
    f"were read; total_size, the window and next_offset describe that prefix, "
    f"not the whole conversation."
)

#: Mirrors `RAG_Search.simplified.rag_service.PROMPT_DOCUMENT_COLUMNS` (the
#: three BODY columns the prompts sub-leg renders as a row's document, in
#: that order) WITHOUT importing it: `rag_service` pulls in the embeddings/
#: vector stack, and this tool is constructed by the Settings-side gate
#: enumerator (`builtin_tool_gate.all_tool_gates`) purely to read its
#: description. A three-string tuple is not worth that import.
PROMPT_BODY_COLUMNS: Tuple[str, ...] = ("details", "system_prompt", "user_prompt")


class ExpansionUnavailableError(RuntimeError):
    """A database this expansion needs could not be opened."""


class _Document(NamedTuple):
    """One fetched document, plus how completely it could be read.

    `note` is empty for the ordinary case. It is non-empty only when the
    FETCH itself was capped (today: a conversation longer than
    `MAX_TRANSCRIPT_MESSAGES`), in which case `text` is a prefix of the real
    document and `total_size` describes only that prefix -- so the payload
    must report `truncated` even when the character window covers everything
    that was read.
    """

    title: str
    text: str
    note: str = ""


def _normalize_candidate(source_type: str, value: Any) -> list[str]:
    """Turn one identity hint into the candidate ids it could mean.

    Args:
        source_type: The seam being expanded, used to strip the indexer's
            document-id prefix.
        value: A raw hint (`source_id`, `note_id`, `doc_id`, `media_id`).

    Returns:
        Zero, one or two candidate id strings, most literal first.
    """
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    candidates = [text]
    prefix = f"{source_type}_"
    if text.startswith(prefix) and len(text) > len(prefix):
        candidates.append(text[len(prefix) :])
    return candidates


def _candidate_ids(
    source_type: str,
    source_id: Any,
    note_id: Any,
    media_id: Any,
    doc_id: Any,
) -> list[str]:
    """Every id worth trying, in resolution order, de-duplicated.

    Explicit `source_id` first (it is right for keyword/label-only rows,
    which are the majority case), then the provenance extras.
    """
    ordered: list[str] = []
    for value in (source_id, note_id, media_id, doc_id):
        for candidate in _normalize_candidate(source_type, value):
            if candidate not in ordered:
                ordered.append(candidate)
    return ordered


def _as_int_id(value: str) -> Optional[int]:
    """Coerce an integer-keyed id, or None when it cannot be one.

    Media and prompt ids are INTEGER primary keys and their readers raise
    on a non-integer, so a chroma point id must be discarded here rather
    than turned into an exception the agent has to read.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require(db: Any, label: str) -> Any:
    if db is None:
        raise ExpansionUnavailableError(f"the {label} database is unavailable")
    return db


def _fetch_note(note_id: str) -> Optional[_Document]:
    from ..config import get_chachanotes_db_lazy

    db = _require(get_chachanotes_db_lazy(), "notes")
    row = db.get_note_by_id(note_id)
    if not row:
        return None
    return _Document(str(row.get("title") or ""), str(row.get("content") or ""))


def _fetch_media(media_id: str) -> Optional[_Document]:
    from ..config import get_media_db_lazy

    numeric = _as_int_id(media_id)
    if numeric is None:
        return None
    db = _require(get_media_db_lazy(), "media")
    row = db.get_media_by_id(numeric)
    if not row:
        return None
    return _Document(str(row.get("title") or ""), str(row.get("content") or ""))


def _fetch_conversation(conversation_id: str) -> Optional[_Document]:
    """Render a conversation as the role-prefixed transcript.

    `f"{sender}: {content}"`, joined by newlines -- byte-identical to
    `ingestion_indexing.conversation_document`, so what the agent reads
    back is the text that was chunked and indexed, not a second rendering
    that happens to differ.

    Reads ONE message past `MAX_TRANSCRIPT_MESSAGES` purely to tell "exactly
    at the cap" (a complete read) from "over it" (a partial one). A partial
    read comes back carrying a note, because reporting it as complete would
    be indistinguishable from a whole-document read.

    `include_image_data=False` (the task-260 precedent): the rendering uses
    `sender`/`content` only, so the reader's default would pull up to
    `MAX_TRANSCRIPT_MESSAGES` image BLOBs into memory for text that cannot
    use them. The rendered transcript is byte-identical either way --
    `image_mime_type` is still returned, so nothing downstream loses the
    fact that an image exists.
    """
    from ..config import get_chachanotes_db_lazy

    db = _require(get_chachanotes_db_lazy(), "conversations")
    conversation = db.get_conversation_by_id(conversation_id)
    if not conversation:
        return None
    messages = list(
        db.get_messages_for_conversation(
            conversation_id,
            limit=MAX_TRANSCRIPT_MESSAGES + 1,
            include_image_data=False,
        )
        or ()
    )
    capped = len(messages) > MAX_TRANSCRIPT_MESSAGES
    lines: list[str] = []
    for message in messages[:MAX_TRANSCRIPT_MESSAGES]:
        content = (message or {}).get("content")
        if not content or not str(content).strip():
            continue
        sender = message.get("sender") or message.get("role") or "unknown"
        lines.append(f"{sender}: {content}")
    title = conversation.get("title") or f"Conversation {conversation_id}"
    note = MESSAGE_CAP_NOTE if capped else ""
    return _Document(str(title), "\n".join(lines), note)


def _fetch_prompt(prompt_id: str) -> Optional[_Document]:
    from ..config import get_prompts_db_lazy

    numeric = _as_int_id(prompt_id)
    if numeric is None:
        return None
    db = _require(get_prompts_db_lazy(), "prompts")
    row = db.get_prompt_by_id(numeric)
    if not row:
        return None
    parts = [
        str(row[column]).strip()
        for column in PROMPT_BODY_COLUMNS
        if row.get(column) and str(row[column]).strip()
    ]
    return _Document(str(row.get("name") or ""), "\n\n".join(parts))


_FETCHERS = {
    "note": _fetch_note,
    "media": _fetch_media,
    "conversation": _fetch_conversation,
    "prompt": _fetch_prompt,
}


def _resolve_budget(max_chars: Any) -> int:
    """Resolve the character budget, capped and never zero or negative."""
    if max_chars is None:
        return DEFAULT_MAX_CHARS
    try:
        value = int(max_chars)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    if value <= 0:
        return DEFAULT_MAX_CHARS
    return min(value, HARD_MAX_CHARS)


def _window_bounds(
    total: int, offset: int, anchor: Optional[int], budget: int
) -> Tuple[int, int]:
    """Pick the half-open character window to return.

    Three cases, in priority order:

    1. **Continuation** (`offset > 0`): start EXACTLY at `offset`. A
       continuation that silently slid backwards would re-serve text the
       agent already paid for and never terminate the walk.
    2. **Chunk-centred** (`anchor` known): centre the budget on the matched
       chunk's character start, then pull back if the document ends first
       so a late chunk still returns a full budget rather than a sliver.
    3. **Head**: the first `budget` characters -- what a label-only row,
       which carries no chunk lineage at all, gets.

    Args:
        total: Length of the whole document text.
        offset: Caller's continuation offset (clamped at 0 and `total`).
        anchor: The matched chunk's character start, when known.
        budget: Resolved character budget.

    Returns:
        A `(start, end)` pair with `0 <= start <= end <= total`.
    """
    if total <= 0:
        return 0, 0
    if offset > 0:
        start = min(offset, total)
        return start, min(start + budget, total)
    if anchor is not None and anchor > 0:
        start = max(0, min(anchor, total) - budget // 2)
        end = min(start + budget, total)
        if end - start < budget:
            start = max(0, end - budget)
        return start, end
    return 0, min(budget, total)


def _empty_result(status: str, source_type: str, source_id: str) -> Dict[str, Any]:
    """The full return shape for a branch that produced no text.

    Every branch returns the SAME keys -- a consumer (the policy hints in
    Phase P, an agent's own parsing) must not have to special-case a miss.
    """
    return {
        "status": status,
        "source_type": source_type,
        "source_id": source_id,
        "title": "",
        "text": "",
        "total_size": 0,
        "window": {"start": 0, "end": 0},
        "truncated": False,
        "next_offset": None,
    }


class ExpandDocumentTool(Tool):
    """Expand one retrieval hit into a bounded window of its document."""

    @property
    def name(self) -> str:
        return "expand_document"

    @property
    def description(self) -> str:
        return (
            "Expand a retrieval hit into its document. Use when a "
            "high-ranked hit is label-only (media/conversation rows) or its "
            "snippet is truncated and the answer needs the content. "
            "Re-query instead if the hit itself looks irrelevant. Never "
            "expand the same source twice — reuse the earlier result. "
            "Stop expanding once your remaining context budget is short — a "
            "window you cannot afford to read is spent for nothing. "
            "Returns a bounded window of the document text plus total_size "
            "and, when more remains, next_offset to continue from; pass the "
            "row's chunk_start to centre that window on the matched chunk "
            "instead of the document head."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source_type": {
                    "type": "string",
                    "enum": list(SUPPORTED_SOURCE_TYPES),
                    "description": "The row's provenance.source_type.",
                },
                "source_id": {
                    "type": "string",
                    "description": "The row's source_id.",
                },
                "chunk_start": {
                    "type": "integer",
                    "description": (
                        "The matched chunk's character start, from the row's "
                        "provenance. Given, the window is centred on the "
                        "match instead of starting at the document head."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Continue from this character offset -- pass the "
                        "next_offset of the previous call to walk a long "
                        "document without re-querying."
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        f"Character budget for this call (default "
                        f"{DEFAULT_MAX_CHARS}, hard cap {HARD_MAX_CHARS})."
                    ),
                },
                "note_id": {
                    "type": "string",
                    "description": (
                        "Identity fallback from the row's provenance, used "
                        "when source_id is a vector-store point id."
                    ),
                },
                "media_id": {
                    "type": "string",
                    "description": "Identity fallback from the row's provenance.",
                },
                "doc_id": {
                    "type": "string",
                    "description": (
                        "Identity fallback from the row's provenance "
                        "(the indexer's prefixed document id)."
                    ),
                },
            },
            "required": ["source_type", "source_id"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Reads the user's notes, media, conversations and prompts."""
        return ("reads",)

    async def execute(
        self,
        source_type: str = "",
        source_id: str = "",
        offset: int = 0,
        max_chars: Optional[int] = None,
        chunk_start: Optional[int] = None,
        note_id: Optional[Any] = None,
        media_id: Optional[Any] = None,
        doc_id: Optional[Any] = None,
        **_provenance: Any,
    ) -> Dict[str, Any]:
        """Return a bounded window of the document behind one retrieval hit.

        Extra keyword arguments are accepted and ignored so an agent can
        paste a row's whole `provenance` mapping without a TypeError
        (`chunk_id`, `chunk_index`, `media_type`, `uuid`, ... all ride along
        there). `chunk_id` is deliberately among them rather than a declared
        parameter: it is an index, nothing here reads it, and a knob wired to
        nothing must not ship as agent-facing surface.

        Args:
            source_type: One of `SUPPORTED_SOURCE_TYPES`.
            source_id: The row's `source_id`.
            offset: Continuation offset from a previous call's
                `next_offset`.
            max_chars: Character budget; defaults to `DEFAULT_MAX_CHARS`
                and is capped at `HARD_MAX_CHARS`.
            chunk_start: The matched chunk's character start, when the
                row's provenance carries it.
            note_id: Identity fallback for a note row.
            media_id: Identity fallback for a media row.
            doc_id: Identity fallback carrying the indexer's prefixed
                document id.

        Returns:
            `status` (`"ok"`/`"not_found"`/`"unsupported"`/`"error"`),
            `source_type`, `source_id` (the RESOLVED id), `title`, `text`,
            `total_size`, `window` (`{"start", "end"}`), `truncated` and
            `next_offset` (`None` when the window reaches the end). An
            `"error"` status additionally carries an `error` key, which is
            what makes the provider report the call as failed; a read the
            SOURCE itself capped (a conversation past
            `MAX_TRANSCRIPT_MESSAGES`) additionally carries a `note` saying
            so, and is always reported `truncated`.
        """
        requested_type = str(source_type or "").strip().lower()
        requested_id = str(source_id or "").strip()

        if requested_type not in SUPPORTED_SOURCE_TYPES:
            logger.debug(f"expand_document: unsupported source_type {requested_type!r}")
            return _empty_result("unsupported", requested_type, requested_id)

        try:
            fetch = _FETCHERS[requested_type]
            resolved_id = ""
            document: Optional[_Document] = None
            for candidate in _candidate_ids(
                requested_type, requested_id, note_id, media_id, doc_id
            ):
                document = fetch(candidate)
                if document is not None:
                    resolved_id = candidate
                    break

            if document is None:
                return _empty_result("not_found", requested_type, requested_id)

            title, text, note = document
            total = len(text)
            budget = _resolve_budget(max_chars)
            start, end = _window_bounds(
                total,
                max(0, _as_int_id(str(offset)) or 0),
                _as_int_id(str(chunk_start)) if chunk_start is not None else None,
                budget,
            )
            result: Dict[str, Any] = {
                "status": "ok",
                "source_type": requested_type,
                "source_id": resolved_id,
                "title": title,
                "text": text[start:end],
                "total_size": total,
                "window": {"start": start, "end": end},
                # A source-capped read is truncated even when the window
                # covers every character that was read: `total` describes
                # the prefix, not the document.
                "truncated": bool(note) or start > 0 or end < total,
                "next_offset": end if end < total else None,
            }
            if note:
                result["note"] = note
            return result
        except Exception as exc:  # noqa: BLE001 -- a tool must not crash the loop
            logger.opt(exception=True).error(
                f"expand_document failed for {requested_type}:{requested_id}: {exc}"
            )
            failure = _empty_result("error", requested_type, requested_id)
            failure["error"] = f"Could not expand {requested_type} {requested_id}: {exc}"
            return failure
