"""The media chunking agent tools' handlers (chunking-agent-tools Tasks 3-4).

``LocalMediaChunkToolService`` implements the descriptor-backed media
chunk-tool operations (spec §4) behind the shared
``LocalLibraryToolService`` dispatch:

* ``library_get_media_structure`` (§4.1) -- the existing navigation tree
  (``LocalMediaReadingService.get_media_navigation``, unchanged) annotated
  per node with the stored-chunk index span overlapping the node's source
  span, plus the item-level chunk summary and the media ``version`` revision
  token. Pagination is BY NODES through the contract's cursor mechanism --
  never a byte slice (§8.11).
* ``library_get_media_chunk`` (§4.2) -- one stored
  ``UnvectorizedMediaChunks`` unit read verbatim through Task 2's
  ``get_library_media_chunks`` (reuse-stored-chunks is THE read path;
  nothing re-chunks), with budget-bounded neighbors (§8.12), family
  disambiguation (§8.10), and the revision check (§8.9).
* ``library_list_chunk_specs`` / ``library_save_chunk_spec`` (§4.3) -- the
  agent view of the v7 chunking-template store (specs ARE templates,
  ruling §8.3): a bounded listing carrying the AC-24a validity/reserved
  decoration, and a create-or-update of CUSTOM templates through the
  interop CRUD's validate-on-write gate. The validator's FULL errors array
  rides the refusal payload (§8.15 -- agents self-correct), and a save runs
  only under the ``library.templates.save.local`` policy action (spec §6).
* ``library_rechunk_media`` (§4.4) -- one item re-chunked NOW through
  Task 1's :func:`rechunk_one_item` (the SAME per-item machinery the
  legacy batch runs): the flat spec override resolves through the #3
  name→dict machinery (an unresolvable template is a named refusal, never
  a silent fallback), the stored chunk rows are replaced in one
  transaction, and the forced vector re-index runs only under
  ``reindex: true`` (ruling §8.4) under the ``library.media.rechunk.local``
  policy action (spec §6).

Error discipline mirrors the sibling services exactly: ``LibraryToolError``
payloads, ``sqlite3.Error``/``OSError`` and unexpected exceptions scrubbed
to the storage-error payload, never a stack trace, SQL, or path.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from loguru import logger

from tldw_chatbook.Library.library_tool_contract import (
    DEFAULT_MAX_NODES,
    DISPLAY_NAME_MAX_BYTES,
    ERROR_CONTENT_CHANGED,
    ERROR_FEATURE_UNAVAILABLE,
    ERROR_INVALID_ARGUMENT,
    ERROR_NOT_FOUND,
    ERROR_STORAGE_ERROR,
    KEYWORDS_PER_ITEM_MAX,
    LIBRARY_TOOL_DESCRIPTORS,
    MAX_CHUNK_CONTEXT,
    MAX_MAX_NODES,
    MAX_RESULT_BYTES,
    SPEC_SAVE_DESCRIPTION_MAX_CHARS,
    SPEC_SAVE_NAME_MAX_CHARS,
    LibraryToolDescriptor,
    LibraryToolError,
    check_cursor_revision,
    fit_page_payload,
    make_cursor,
    normalize_display_text,
    parse_cursor,
    parse_public_id,
    serialized_size,
    validate_page_args,
)
from tldw_chatbook.runtime_policy.types import PolicyDeniedError

from .library_rechunk_service import rechunk_one_item

#: The NULL chunk family's wire label (Task 2's backend uses the same
#: rendering; the string round-trips back as the ``chunk_type`` filter).
_PRIMARY_FAMILY_LABEL = "primary"
_LEGACY_ENGINE_LABEL = "legacy"

#: Spec §8.13's exact degradation hint.
_RECHUNK_HINT = "no stored chunks — use library_rechunk_media to enable unit fetches"

#: Spec §6: the policy action the chunk-spec save tool runs under. Registered
#: in ``runtime_policy/registry.py`` (the ``library.templates`` resource on
#: the local Library agent-tools capability); the MCP local-control mapping
#: resolves this exact id for the tool.
SPEC_SAVE_POLICY_ACTION_ID = "library.templates.save.local"

#: Spec §6: the policy action the re-chunk tool runs under -- the
#: ``library.media`` resource's ``rechunk`` verb on the same local Library
#: agent-tools capability (deliberately NOT ``rag.admin.launch``: that verb
#: owns the RAG-admin bulk action per ADR-003; this is a Library-media item
#: action).
RECHUNK_POLICY_ACTION_ID = "library.media.rechunk.local"

#: Description the CRUD demands on CREATE (it refuses empty ones); the tool
#: schema keeps the field optional, so an unsupplied one on create lands as
#: this stable, honest default rather than a refusal.
_DEFAULT_SPEC_DESCRIPTION = (
    "Custom chunking spec saved through the library_save_chunk_spec tool."
)


def _invalid(message: str, *, details: dict | None = None) -> LibraryToolError:
    return LibraryToolError(ERROR_INVALID_ARGUMENT, message, details=details)


def _not_found() -> LibraryToolError:
    return LibraryToolError(
        ERROR_NOT_FOUND, "The requested Library item was not found."
    )


def _storage_error_payload() -> dict[str, Any]:
    return LibraryToolError(
        ERROR_STORAGE_ERROR,
        "The local Library store could not complete the read.",
        retryable=True,
    ).to_payload()


def _validate_max_nodes(value: Any) -> int:
    """Validate/coerce the node-page bound (default 200, clamped to 500)."""
    if value is None:
        return DEFAULT_MAX_NODES
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid("max_nodes must be an integer")
    if value < 1:
        raise _invalid("max_nodes must be at least 1")
    return min(value, MAX_MAX_NODES)


def _validate_chunk_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid("chunk_index must be an integer")
    if value < 0:
        raise _invalid("chunk_index must be at least 0")
    return value


def _validate_context(value: Any) -> int:
    """Validate/coerce the neighbor window (default 0, clamped to 10)."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid("context must be an integer")
    if value < 0:
        raise _invalid("context must be at least 0")
    return min(value, MAX_CHUNK_CONTEXT)


def _validate_chunk_type(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _invalid("chunk_type must be a non-empty string")
    return value.strip()


def _parse_revision(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _invalid("revision must be a non-empty string token")
    return value.strip()


def _family_label(raw: Any) -> str:
    return raw if raw is not None else _PRIMARY_FAMILY_LABEL


class LocalMediaChunkToolService:
    """The media chunking agent tools (spec §4) behind the shared dispatch.

    Backend-agnostic by construction: the media DB and reading service are
    duck-typed handles, so tests stub them freely. All five descriptor
    names resolve through :meth:`invoke`.
    """

    def __init__(
        self,
        media_db: Any = None,
        media_reading_service: Any = None,
        template_interop: Any = None,
        policy_enforcer: Any = None,
        rag_service: Any = None,
        indexing_db: Any = None,
    ) -> None:
        self._media_db = media_db
        self._reading = media_reading_service
        self._templates = template_interop
        self._policy_enforcer = policy_enforcer
        # The OPT-IN forced re-index handles (spec §4.4 ruling §8.4): when
        # ``rag_service`` is not injected, the handler falls back to the
        # process-wide shared RAG service seam -- resolved ONLY on an
        # opted-in run, off every read/default path.
        self._rag_service = rag_service
        self._indexing_db = indexing_db
        self._template_listing: Any = None

    # -- Entry point ---------------------------------------------------------

    def invoke(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one chunk tool; failures return the structured error payload.

        Args:
            tool_name: One of the five media chunk tool names registered in
                ``LIBRARY_TOOL_DESCRIPTORS``.
            arguments: The tool's JSON-object arguments, validated against
                the descriptor's schema keys before any backend touch.

        Returns:
            The tool's response payload, or the structured
            ``LibraryToolError`` payload for named refusals (invalid
            arguments, not-found, feature-unavailable degrades) and the
            scrubbed storage-error payload for operational failures. Never
            raises.
        """
        try:
            return self._dispatch(tool_name, arguments)
        except LibraryToolError as exc:
            return exc.to_payload()
        except (sqlite3.Error, OSError):
            return _storage_error_payload()
        except Exception:  # noqa: BLE001 — scrubbed, never escapes the tool
            return _storage_error_payload()

    def _dispatch(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        descriptor = LIBRARY_TOOL_DESCRIPTORS.get(tool_name)
        if descriptor is None:
            raise _invalid(f"unknown Library tool: {tool_name!r}")
        if not isinstance(arguments, Mapping):
            raise _invalid("arguments must be a JSON object")
        self._validate_argument_keys(descriptor, arguments)
        if tool_name == "library_get_media_structure":
            return self._structure(arguments)
        if tool_name == "library_get_media_chunk":
            return self._fetch_chunk(arguments)
        if tool_name == "library_list_chunk_specs":
            return self._list_specs(arguments)
        if tool_name == "library_save_chunk_spec":
            return self._save_spec(arguments)
        if tool_name == "library_rechunk_media":
            return self._rechunk(arguments)
        raise _invalid(f"unknown Library tool: {tool_name!r}")

    @staticmethod
    def _validate_argument_keys(
        descriptor: LibraryToolDescriptor, arguments: Mapping[str, Any]
    ) -> None:
        """The house argument discipline (mirrors the shared dispatcher)."""
        allowed = set(descriptor.input_schema.get("properties", ()))
        unknown = sorted(str(key) for key in arguments if key not in allowed)
        if unknown:
            raise _invalid(f"unknown argument(s): {', '.join(unknown)}")
        required = descriptor.input_schema.get("required", ())
        missing = [key for key in required if key not in arguments]
        if missing:
            raise _invalid(f"missing required argument(s): {', '.join(missing)}")

    # -- Shared resolution ----------------------------------------------------

    def _require_media_read_backend(self) -> None:
        """Both read-tool handles, or the named degrade payload.

        The two read tools need the media DB (row/chunk reads) AND the
        reading service (navigation, unit fetch); both wiring sites can
        construct this service with one handle absent (the Console factory
        builds on EITHER, the MCP builder survives a failed media backend),
        so a missing handle degrades THESE tools to the structured
        ``feature_unavailable`` payload -- the sibling ``_backend`` None
        discipline -- instead of letting the eventual ``AttributeError``
        scrub to a misleading storage_error.
        """
        if self._media_db is None or self._reading is None:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "The local media store is not available in this deployment,"
                " so media structure and chunk reads are unavailable.",
            )

    def _resolve_media_row(self, arguments: Mapping[str, Any]) -> tuple[str, dict]:
        """Parse the public media ID and load the ACTIVE media row.

        Returns:
            ``(public_id, row)`` -- the caller-opaque ID and the full
            ``Media`` row (id, version, title, chunking_config, ...).

        Raises:
            LibraryToolError: ``invalid_argument`` for malformed IDs;
                ``not_found`` when no active (non-deleted, non-trashed)
                item matches -- the same posture as ``library_get_media``.
        """
        public_id = arguments.get("id")
        _, media_uuid = parse_public_id(public_id, expected_type="media")
        row = self._media_db.get_media_by_uuid(media_uuid)
        if row is None:
            raise _not_found()
        return public_id, row

    @staticmethod
    def _item_block(public_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
        """The bounded ``item`` metadata block (the sibling get discipline)."""
        title, truncated = normalize_display_text(
            row.get("title"), max_bytes=DISPLAY_NAME_MAX_BYTES
        )
        item: dict[str, Any] = {
            "id": public_id,
            "type": "media",
            "title": title,
            "title_truncated": truncated,
        }
        for key in ("media_type", "author", "ingestion_date", "last_modified"):
            value = row.get(key)
            if value is not None:
                item[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
        return item

    def _chunk_rows(self, media_id: int) -> list[dict[str, Any]]:
        """Every live chunk row for the item (parameterized SQL only)."""
        cursor = self._media_db.get_connection().execute(
            "SELECT chunk_index, chunk_type, start_char, end_char, "
            "chunk_engine_version FROM UnvectorizedMediaChunks "
            "WHERE media_id = ? AND deleted = 0 "
            "ORDER BY chunk_index, chunk_type",
            (media_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _family_index_range(
        self, media_id: int, chunk_type: str | None
    ) -> tuple[int, int] | None:
        """The selected family's ``[min_index, max_index]``, or None.

        Two LITERAL statements (no f-string composition): the NULL family
        cannot be expressed as a bound parameter, so the branch picks one
        of two fixed SQL texts -- every VALUE stays parameterized either
        way, matching the reading service's own queries over this table.
        """
        if chunk_type is None or chunk_type == _PRIMARY_FAMILY_LABEL:
            row = self._media_db.get_connection().execute(
                "SELECT MIN(chunk_index) AS lo, MAX(chunk_index) AS hi "
                "FROM UnvectorizedMediaChunks "
                "WHERE media_id = ? AND deleted = 0 AND chunk_type IS NULL",
                (media_id,),
            ).fetchone()
        else:
            row = self._media_db.get_connection().execute(
                "SELECT MIN(chunk_index) AS lo, MAX(chunk_index) AS hi "
                "FROM UnvectorizedMediaChunks "
                "WHERE media_id = ? AND deleted = 0 AND chunk_type = ?",
                (media_id, chunk_type),
            ).fetchone()
        if row is None or row["lo"] is None:
            return None
        return int(row["lo"]), int(row["hi"])

    # -- library_get_media_structure (spec §4.1) --------------------------------

    def _structure(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self._require_media_read_backend()
        public_id, row = self._resolve_media_row(arguments)
        media_id = int(row["id"])
        revision = str(row["version"])

        max_nodes = _validate_max_nodes(arguments.get("max_nodes"))
        offset = 0
        raw_cursor = arguments.get("node_cursor")
        if raw_cursor is not None:
            state = parse_cursor(raw_cursor)
            if state["item"] != public_id:
                raise _invalid(
                    "continuation cursor belongs to a different Library item"
                )
            check_cursor_revision(state, revision)
            offset = int(state["off"])

        notes: list[str] = []
        nodes: list[dict[str, Any]] = []
        node_total = 0
        truncated = False
        try:
            navigation = self._reading.get_media_navigation(
                media_id, max_nodes=MAX_MAX_NODES
            )
        except ValueError:
            navigation = None
        if navigation is not None:
            node_total = int((navigation.get("stats") or {}).get("node_count") or 0)
            truncated = bool((navigation.get("stats") or {}).get("truncated"))
            nodes = list(navigation.get("nodes") or ())
        else:
            notes.append("source headings unavailable for this item")

        chunk_rows = self._chunk_rows(media_id)
        families = sorted({_family_label(r["chunk_type"]) for r in chunk_rows})
        engine_versions = sorted(
            {
                str(r["chunk_engine_version"])
                if r["chunk_engine_version"] is not None
                else _LEGACY_ENGINE_LABEL
                for r in chunk_rows
            }
        )
        summary: dict[str, Any] = {
            "available": bool(chunk_rows),
            "chunk_count": len(chunk_rows),
            "families": families,
            "engine_versions": engine_versions,
            "stale": _LEGACY_ENGINE_LABEL in engine_versions,
        }
        template_name = self._stored_template_name(row)
        if template_name is not None:
            summary["template_name"] = template_name

        # The span family: the family the fetch tool addresses by default --
        # primary when present, else the sole family. With neither, spans
        # would silently mix index spaces, so they stay off and the families
        # note names the round-trippable ``chunk_type`` strings instead.
        if _PRIMARY_FAMILY_LABEL in families:
            span_family: str | None = _PRIMARY_FAMILY_LABEL
        elif len(families) == 1:
            span_family = families[0]
            notes.append(
                f"chunk_span addresses the sole family {span_family!r}; pass"
                " it as chunk_type when fetching units"
            )
        else:
            span_family = None
            if families:
                notes.append(
                    "chunk families present: "
                    + ", ".join(families)
                    + " — pass one as chunk_type to fetch units"
                )

        span_rows = (
            sorted(
                (
                    r
                    for r in chunk_rows
                    if _family_label(r["chunk_type"]) == span_family
                    and r["start_char"] is not None
                    and r["end_char"] is not None
                ),
                key=lambda r: (int(r["start_char"]), int(r["chunk_index"])),
            )
            if span_family is not None
            else []
        )

        # Paging closes at the fetched window, never past it: navigation is
        # fetched once at the 500-node ceiling, and ``node_total`` counts the
        # WHOLE tree (an 800-node doc reports 800 but delivers a 500-node
        # window). Bounding ``has_more`` by the window is what keeps a walk
        # from degenerating -- otherwise offset 500 pages an empty list yet
        # re-mints a cursor against the unreachable remainder, forever.
        window_total = len(nodes)
        pageable_total = min(node_total, window_total)
        if node_total > window_total:
            truncated = True
            notes.append(
                f"navigation window limited to the first {window_total} of"
                f" {node_total} nodes; deeper nodes are not addressable"
                " through this tool"
            )

        page = nodes[offset : offset + max_nodes]
        payload_nodes = [
            self._structure_node(node, span_rows) for node in page
        ]
        has_more = offset + len(page) < pageable_total

        if not chunk_rows:
            notes.append(_RECHUNK_HINT)

        return {
            "item": self._item_block(public_id, row),
            "revision": revision,
            "node_total": node_total,
            "node_offset": offset,
            "returned_node_count": len(payload_nodes),
            "has_more": has_more,
            "next_cursor": (
                make_cursor(
                    item_id=public_id, revision=revision, offset=offset + len(page)
                )
                if has_more
                else None
            ),
            "truncated": truncated,
            "nodes": payload_nodes,
            "chunk_summary": summary,
            "notes": notes,
        }

    @staticmethod
    def _structure_node(node: Mapping[str, Any], span_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """One navigation node in the structure payload shape (spec §4.1)."""
        title, _ = normalize_display_text(
            node.get("title"), max_bytes=DISPLAY_NAME_MAX_BYTES
        )
        start = node.get("target_start")
        end = node.get("target_end")
        payload: dict[str, Any] = {
            "node_id": str(node.get("id") or ""),
            "title": title,
            "level": int(node.get("level") or 0),
            "span": [
                int(start) if start is not None else None,
                int(end) if end is not None else None,
            ],
        }
        chunk_span = LocalMediaChunkToolService._chunk_span_for(
            start, end, span_rows
        )
        if chunk_span is not None:
            payload["chunk_span"] = chunk_span
        return payload

    @staticmethod
    def _chunk_span_for(
        start: Any, end: Any, span_rows: list[dict[str, Any]]
    ) -> list[int] | None:
        """The ``[first, last]`` chunk-index span overlapping ``[start, end)``.

        One pass over the span-ordered chunk rows (spec §4.1 accepts the
        O(nodes×chunks) scan at these caps). A chunk overlaps the node when
        the intervals intersect with a non-empty overlap; the reported span
        covers every overlapping index (nested nodes naturally span more).
        """
        if start is None or end is None:
            return None
        node_start, node_end = int(start), int(end)
        if node_end <= node_start:
            return None
        first: int | None = None
        last: int | None = None
        for chunk in span_rows:
            chunk_start = int(chunk["start_char"])
            chunk_end = int(chunk["end_char"])
            if chunk_start >= node_end:
                break  # spans are start-ordered; nothing later can overlap
            if chunk_end <= node_start:
                continue
            index = int(chunk["chunk_index"])
            if first is None:
                first = index
            last = index if last is None or index > last else last
        if first is None or last is None:
            return None
        return [first, last]

    @staticmethod
    def _stored_template_name(row: Mapping[str, Any]) -> str | None:
        """The stored per-media chunking config's template name, if any."""
        raw = row.get("chunking_config")
        if isinstance(raw, dict):
            config = raw
        else:
            try:
                config = json.loads(raw) if raw else None
            except (TypeError, ValueError):
                return None
        if not isinstance(config, dict):
            return None
        name = config.get("template")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return None

    # -- library_get_media_chunk (spec §4.2) ------------------------------------

    def _fetch_chunk(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self._require_media_read_backend()
        public_id, row = self._resolve_media_row(arguments)
        media_id = int(row["id"])
        revision = str(row["version"])

        chunk_index = _validate_chunk_index(arguments.get("chunk_index"))
        context = _validate_context(arguments.get("context"))
        chunk_type = _validate_chunk_type(arguments.get("chunk_type"))
        supplied_revision = _parse_revision(arguments.get("revision"))
        if supplied_revision is not None and supplied_revision != revision:
            raise LibraryToolError(
                ERROR_CONTENT_CHANGED,
                "The media item changed since this revision token was issued;"
                " re-fetch the structure for fresh chunk addresses.",
                details={"hint": "refetch_structure"},
            )

        result = self._reading.get_library_media_chunks(
            media_id,
            chunk_index=chunk_index,
            chunk_type=chunk_type,
            context=context,
            budget=MAX_RESULT_BYTES,
        )
        if result is None:
            # Active row but no stored chunk rows (spec §8.13): the named
            # degradation names the tool that enables unit fetches.
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "This media item has no stored chunks, so chunk units cannot"
                f" be fetched; {_RECHUNK_HINT}.",
            )

        families = list(result.get("families") or [])
        if chunk_type is None and len(families) > 1:
            raise _invalid(
                "chunk_index is ambiguous: this item has "
                f"{len(families)} chunk families; pass chunk_type with one of: "
                + ", ".join(families),
                details={"families": families},
            )

        chunks = list(result.get("chunks") or [])
        if not chunks:
            self._raise_absent_chunk(media_id, chunk_index, chunk_type, families)

        requested = next(
            (c for c in chunks if int(c["chunk_index"]) == chunk_index), None
        )
        if requested is None:
            self._raise_absent_chunk(media_id, chunk_index, chunk_type, families)
        neighbors = [c for c in chunks if int(c["chunk_index"]) != chunk_index]

        notes: list[str] = []
        dropped = int(result.get("dropped_neighbors") or 0)
        if dropped:
            notes.append(
                f"{dropped} neighbor(s) omitted — the byte budget was reached"
                " before them (context window truncated, never the chunk itself)"
            )
        payload: dict[str, Any] = {
            "item": self._item_block(public_id, row),
            "chunk": self._json_safe_chunk(requested),
            "neighbors": [self._json_safe_chunk(c) for c in neighbors],
            "notes": notes,
            "revision": revision,
        }
        # The requested unit is ALWAYS returned whole (the review carry):
        # an oversized chunk is noted, never truncated, never refused.
        if serialized_size(payload) > MAX_RESULT_BYTES:
            notes.append(
                "the requested chunk exceeds the standard result byte budget"
                " and was returned whole"
            )
        return payload

    def _raise_absent_chunk(
        self,
        media_id: int,
        chunk_index: int,
        chunk_type: str | None,
        families: list[str],
    ) -> None:
        """The absent-chunk named errors (out-of-range / wrong family)."""
        span = self._family_index_range(media_id, chunk_type)
        label = _family_label(chunk_type)
        if span is None:
            raise _invalid(
                f"chunk family {label!r} has no stored chunks for this item;"
                " families present: " + (", ".join(families) or "none"),
                details={"families": families},
            )
        raise _invalid(
            f"chunk_index {chunk_index} is out of range for family {label!r};"
            f" valid range {span[0]}..{span[1]}",
            details={"valid_range": [span[0], span[1]], "families": families},
        )

    @staticmethod
    def _json_safe_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
        return {key: chunk[key] for key in (
            "chunk_index",
            "chunk_type",
            "text",
            "start_char",
            "end_char",
            "word_count",
            "metadata",
        )}

    # -- library_list_chunk_specs / library_save_chunk_spec (spec §4.3) ------

    def _require_templates(self) -> Any:
        """The interop handle, or the named degrade payload."""
        if self._templates is None:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "The local chunk-spec (template) store is not available in"
                " this deployment.",
            )
        return self._templates

    def _decorated_templates(self) -> list[dict[str, Any]]:
        """The v7 store through the AC-24a decorated listing (no new storage).

        ``LocalRAGAdminService._decorate_template_record`` is the one flag
        surface (``template_valid`` / ``template_validation_errors`` /
        ``name_reserved``) -- consuming it here keeps the agent view from
        drifting from the UI's. Lazy import: ``RAG_Admin`` pulls the local
        admin stack, heavier than this module's own chain.
        """
        if self._template_listing is None:
            from ..RAG_Admin.local_rag_admin_service import LocalRAGAdminService

            self._template_listing = LocalRAGAdminService(
                media_db=None, chunking_service=self._require_templates()
            )
        return list(self._template_listing.list_templates())

    def _decorated_template_by_id(self, template_id: int) -> dict[str, Any]:
        """One decorated record by row id (save re-reads through the same
        AC-24a surface the listing uses)."""
        listing = self._decorated_templates()
        for record in listing:
            if int(record.get("id") or -1) == int(template_id):
                return record
        raise _not_found()

    @staticmethod
    def _spec_method(record: Mapping[str, Any]) -> str | None:
        """``chunking.method`` from the stored body, tolerant of bad JSON."""
        raw = record.get("template_json")
        body = raw if isinstance(raw, dict) else None
        if body is None:
            try:
                body = json.loads(raw) if raw else None
            except (TypeError, ValueError):
                return None
        if not isinstance(body, dict):
            return None
        chunking = body.get("chunking")
        if not isinstance(chunking, dict):
            return None
        method = chunking.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
        return None

    @classmethod
    def _spec_item(cls, record: Mapping[str, Any]) -> dict[str, Any]:
        """One template record in the agent listing shape (spec §4.3).

        The flags come verbatim from the AC-24a decoration; ``error_count``
        condenses ``template_validation_errors`` (the listing carries the
        count; a save refusal carries the full array, §8.15).
        """
        name, truncated = normalize_display_text(
            record.get("name"), max_bytes=DISPLAY_NAME_MAX_BYTES
        )
        raw_tags = record.get("tags") or []
        tags = [str(tag) for tag in raw_tags]
        tags_truncated = len(tags) > KEYWORDS_PER_ITEM_MAX
        errors = list(record.get("template_validation_errors") or [])
        return {
            "name": name,
            "name_truncated": truncated,
            "method": cls._spec_method(record),
            "tags": tags[:KEYWORDS_PER_ITEM_MAX],
            "tags_truncated": tags_truncated,
            "is_builtin": bool(record.get("is_builtin")),
            "template_valid": bool(record.get("template_valid")),
            "error_count": len(errors),
            "name_reserved": bool(record.get("name_reserved")),
        }

    def _list_specs(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """The v7 template store's agent view, bounded like sibling lists."""
        self._require_templates()
        limit, offset = validate_page_args(
            arguments.get("limit"), arguments.get("offset")
        )
        records = self._decorated_templates()
        total = len(records)
        page = records[offset : offset + limit]
        items = [self._spec_item(record) for record in page]
        has_more = offset + len(items) < total
        return fit_page_payload(
            {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "next_offset": offset + len(items) if has_more else None,
            }
        )

    def _enforce_spec_save_policy(self) -> None:
        """Spec §6: the save runs under ``library.templates.save.local``.

        Enforcement precedes EVERY backend touch (denial -> the named error
        payload, no CRUD call, not even the routing read). No-op without an
        enforcer handle -- the scope-service precedent: the MCP runtime gate
        (the re-pointed ``_TOOL_ACTION_IDS`` mapping) stays the always-on
        outer layer, and construction sites wire the enforcer where a
        runtime-policy context exists.
        """
        if self._policy_enforcer is None:
            return
        try:
            self._policy_enforcer.require_allowed(
                action_id=SPEC_SAVE_POLICY_ACTION_ID
            )
        except PolicyDeniedError as exc:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "Saving chunk specs is not permitted by the current runtime"
                f" policy ({SPEC_SAVE_POLICY_ACTION_ID}): {exc.user_message}",
                details={
                    "policy_action": SPEC_SAVE_POLICY_ACTION_ID,
                    "reason_code": str(
                        getattr(exc, "reason_code", "authority_denied")
                    ),
                },
            ) from exc

    @staticmethod
    def _validate_spec_save_arguments(
        arguments: Mapping[str, Any]
    ) -> tuple[str, str | None, list[str] | None]:
        """Type-check the save args; returns ``(name, description, tags)``.

        The body is NOT validated here -- it goes through the template
        validator (and then the CRUD's own gate), never ad-hoc checks.
        The name/description length bounds ARE re-checked here (the
        schema's maxLength literals): a schema-bypassing caller still
        fails closed with the named limit.
        """
        raw_name = arguments.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise _invalid("name must be a non-empty string")
        name = raw_name.strip()
        if len(name) > SPEC_SAVE_NAME_MAX_CHARS:
            raise _invalid(
                f"name must be at most {SPEC_SAVE_NAME_MAX_CHARS} characters"
                f" (got {len(name)})"
            )

        raw_description = arguments.get("description")
        if raw_description is None:
            description: str | None = None
        elif isinstance(raw_description, str) and raw_description.strip():
            description = raw_description.strip()
            if len(description) > SPEC_SAVE_DESCRIPTION_MAX_CHARS:
                raise _invalid(
                    "description must be at most"
                    f" {SPEC_SAVE_DESCRIPTION_MAX_CHARS} characters"
                    f" (got {len(description)})"
                )
        else:
            raise _invalid("description must be a non-empty string when supplied")

        raw_tags = arguments.get("tags")
        tags: list[str] | None
        if raw_tags is None:
            tags = None
        elif isinstance(raw_tags, list) and all(
            isinstance(tag, str) and tag.strip() for tag in raw_tags
        ):
            tags = [tag.strip() for tag in raw_tags]
        else:
            raise _invalid("tags must be a list of non-empty strings")

        return name, description, tags

    @staticmethod
    def _run_template_validator(body: Mapping[str, Any]) -> dict[str, Any]:
        """``RAG_Admin.template_validation.validate_template`` on the body,
        with the SAME §7.1 carve-out the CRUD's gate uses (name/description/
        tags never enter the validated body) so this verdict and the CRUD's
        cannot disagree. The validator never raises (Task-6 contract)."""
        from ..RAG_Admin.template_validation import validate_template

        validated = {
            key: value
            for key, value in body.items()
            if key not in ("name", "description", "tags")
        }
        return validate_template(validated)

    def _save_spec(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Create-or-update one CUSTOM template through the interop CRUD."""
        interop = self._require_templates()
        name, description, tags = self._validate_spec_save_arguments(arguments)
        body = arguments.get("spec")
        if not isinstance(body, Mapping):
            raise _invalid(
                "spec must be a JSON object in the template store's shape"
                " (chunking.method/config, optional preprocessing/postprocessing)"
            )
        body = dict(body)

        # Policy before ANY backend touch (spec §6).
        self._enforce_spec_save_policy()

        # Route: built-in names are never mutated (agents duplicate as custom
        # first); an existing custom name updates; a new name creates.
        existing = interop.get_template_by_name(name)
        if existing is not None and bool(existing.get("is_builtin")):
            raise _invalid(
                f"{name!r} is a built-in chunk spec; built-in specs are never"
                " mutated through this tool. Save your version under a new"
                " (custom) name instead — duplicate the built-in as a custom"
                " spec first, then edit the custom copy."
            )

        # The validator BEFORE the CRUD (ruling §8.15): the refusal carries
        # the validator's FULL errors array so agents can self-correct; the
        # CRUD's own gate stays behind this as the backstop.
        verdict = self._run_template_validator(body)
        if not verdict["valid"]:
            errors = list(verdict["errors"])
            summary = "; ".join(
                f"{issue['field']}: {issue['message']}" for issue in errors[:3]
            )
            more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
            raise LibraryToolError(
                ERROR_INVALID_ARGUMENT,
                f"The chunk spec body failed validation and was refused:"
                f" {summary}{more}",
                details={
                    "errors": errors,
                    "warnings": list(verdict["warnings"]),
                },
            )

        from ..Chunking.chunking_interop_library import (
            BuiltinTemplateError,
            ChunkingTemplateError,
            InputError,
            InvalidTemplateError,
        )

        notes: list[str] = []
        try:
            if existing is None:
                effective_description = description
                if effective_description is None:
                    effective_description = _DEFAULT_SPEC_DESCRIPTION
                    notes.append(
                        "no description supplied; a default description was"
                        " saved with the spec"
                    )
                template_id = int(
                    interop.create_template(
                        name,
                        effective_description,
                        body,
                        tags=tags,
                    )
                )
                created = True
            else:
                interop.update_template(
                    int(existing["id"]),
                    description=description,
                    template_json=body,
                    tags=tags,
                )
                template_id = int(existing["id"])
                created = False
        except InputError as exc:
            raise _invalid(f"The chunk-spec store refused the save: {exc}") from exc
        except InvalidTemplateError as exc:
            # The reserved `auto` sentinel refusal lives here (case-
            # insensitive, auto-selection §4.3/AC 14); body refusals were
            # already surfaced above with the full array.
            raise _invalid(str(exc)) from exc
        except BuiltinTemplateError as exc:  # unreachable: pre-checked above
            raise _invalid(
                f"{name!r} is a built-in chunk spec and is never mutated;"
                " duplicate it as a custom spec first."
            ) from exc
        except ChunkingTemplateError as exc:
            logger.error("chunk-spec save failed: %s", exc)
            raise LibraryToolError(
                ERROR_STORAGE_ERROR,
                "The local chunk-spec store could not complete the save.",
                retryable=True,
            ) from exc

        record = self._decorated_template_by_id(template_id)
        return {
            "created": created,
            "spec": self._spec_item(record),
            "warnings": list(verdict["warnings"]),
            "notes": notes,
        }

    # -- library_rechunk_media (spec §4.4) --------------------------------------

    def _require_media_db(self) -> Any:
        """The media DB handle, or the named degrade payload (the re-chunk
        writes chunk rows through it -- there is no reading-service
        fallback for a write)."""
        if self._media_db is None:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "The local media store is not available in this deployment,"
                " so items cannot be re-chunked.",
            )
        return self._media_db

    def _enforce_rechunk_policy(self) -> None:
        """Spec §6: the re-chunk runs under ``library.media.rechunk.local``.

        Same seam and same ordering as the spec-save gate: enforcement
        precedes EVERY backend touch (denial -> the named error payload,
        no row read, no chunking). No-op without an enforcer handle -- the
        MCP runtime gate (the re-pointed tool-mapping) stays the always-on
        outer layer, and both construction sites wire the app's enforcer.
        """
        if self._policy_enforcer is None:
            return
        try:
            self._policy_enforcer.require_allowed(
                action_id=RECHUNK_POLICY_ACTION_ID
            )
        except PolicyDeniedError as exc:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "Re-chunking media items is not permitted by the current"
                f" runtime policy ({RECHUNK_POLICY_ACTION_ID}):"
                f" {exc.user_message}",
                details={
                    "policy_action": RECHUNK_POLICY_ACTION_ID,
                    "reason_code": str(
                        getattr(exc, "reason_code", "authority_denied")
                    ),
                },
            ) from exc

    @staticmethod
    def _validate_rechunk_spec(spec: Any) -> dict[str, Any] | None:
        """Type-check the FLAT override; returns the pre-resolved spec dict
        Task 1's ``rechunk_one_item`` wants (or ``None`` = stored config).

        The shape is closed (template | method/max_size/overlap): a
        ``template`` name governs its own options, so it never mixes with
        the plain keys (#3's explicit-template semantics); plain keys pass
        straight through with the engine left to default whatever the
        agent omitted -- EXCEPT overlap, whose omitted value Task 1 ruled
        to be 0 inside the one-item function (never the engine's 100).
        """
        if spec is None:
            return None
        if not isinstance(spec, Mapping):
            raise _invalid(
                "spec must be a flat JSON object of override keys"
                " (template, method, max_size, overlap) -- NOT the nested"
                " chunking template body library_save_chunk_spec saves"
            )
        allowed = {"template", "method", "max_size", "overlap"}
        unknown = sorted(str(key) for key in spec if key not in allowed)
        if unknown:
            raise _invalid(f"unknown spec key(s): {', '.join(unknown)}")

        template = spec.get("template")
        if template is not None:
            if not isinstance(template, str) or not template.strip():
                raise _invalid("spec.template must be a non-empty string")
            if any(key in spec for key in ("method", "max_size", "overlap")):
                raise _invalid(
                    "spec.template governs its own options; plain"
                    " method/max_size/overlap keys cannot accompany it"
                )
            return {"template": template.strip()}

        resolved: dict[str, Any] = {}
        method = spec.get("method")
        if method is not None:
            if not isinstance(method, str) or not method.strip():
                raise _invalid("spec.method must be a non-empty string")
            resolved["method"] = method.strip()
        for key in ("max_size", "overlap"):
            value = spec.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise _invalid(f"spec.{key} must be a non-negative integer")
            if key == "max_size" and value < 1:
                raise _invalid("spec.max_size must be at least 1")
            resolved[key] = value
        return resolved

    def _rechunk(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """One item re-chunked NOW through Task 1's per-item machinery."""
        media_db = self._require_media_db()
        spec = self._validate_rechunk_spec(arguments.get("spec"))
        reindex = arguments.get("reindex", False)
        if not isinstance(reindex, bool):
            raise _invalid("reindex must be a boolean")

        # Policy before ANY backend touch (spec §6) -- before the row load.
        self._enforce_rechunk_policy()

        # Minor-1 hardening (Task 1's review): the HANDLER loads the full
        # row and refuses on an unresolvable id -- ``rechunk_one_item`` is
        # never handed a None row (which would degrade to a NULL-keyed
        # silent skip instead of a named refusal).
        public_id, row = self._resolve_media_row(arguments)

        if spec is not None and "template" in spec:
            # The one name→dict hop Task 1 left to callers, PRE-CHECKED
            # here so an unresolvable name is the named tool refusal (a
            # #3 TemplateResolutionError mapped to a payload), never the
            # one-item function's failed-outcome note and never a silent
            # fallback to different chunking.
            template_name = spec["template"]
            from ..Chunking.template_runtime import resolve_template

            if resolve_template(media_db, template_name) is None:
                raise LibraryToolError(
                    ERROR_NOT_FOUND,
                    f"The chunk spec {template_name!r} named in the spec"
                    " override does not resolve (deleted or renamed); the"
                    " re-chunk was refused instead of silently falling"
                    " back to different chunking.",
                )

        rag_service = self._rag_service
        notes: list[str] = []
        if reindex:
            if rag_service is None:
                # Resolved OUTSIDE the transient loop below (the
                # #700-hardened rule); a None here degrades the opt-in
                # re-index to a disclosed skip (§10.2.1's conditional),
                # never a raise -- the chunk-row replacement is the call's
                # own contract.
                rag_service = _shared_rag_service_or_none()
        else:
            notes.append(
                "reindex not requested: chunk rows were replaced only"
                " (pass reindex: true to force the vector re-index)"
            )

        # The sync bridge: this service is a pure-synchronous core (the
        # Console worker thread / MCP's ``asyncio.to_thread``), so the
        # one-item coroutine runs under a transient loop -- the dispatcher's
        # own ``_run`` pattern. The RAG service handle was resolved ABOVE,
        # outside that loop (the #700-hardened rule the panel worker
        # documents: never first-construct the shared service inside a
        # closing loop).
        outcome = asyncio.run(
            rechunk_one_item(
                media_db,
                row,
                spec=spec,
                rag_service=rag_service,
                indexing_db=self._indexing_db,
                reindex=reindex,
            )
        )

        status = str(outcome.get("status") or "failed")
        payload: dict[str, Any] = {
            "item": self._item_block(public_id, row),
            "status": status,
            "notes": notes + [str(note) for note in outcome.get("notes") or []],
        }
        if status == "rechunked":
            summary = outcome.get("chunk_summary")
            payload["chunk_summary"] = (
                dict(summary) if isinstance(summary, Mapping) else {}
            )
            if reindex:
                # The opt-in is ALWAYS answered: the run's own reindexed
                # outcome when it ran, else the disclosed skip (spec §4.4's
                # ``reindexed: {done|skipped|failed}`` shape).
                reindexed = outcome.get("reindexed")
                payload["reindexed"] = (
                    dict(reindexed)
                    if isinstance(reindexed, Mapping)
                    else {
                        "status": "skipped",
                        "reason": "semantic index unavailable",
                    }
                )
        return payload


def _shared_rag_service_or_none() -> Any:
    """The process-wide shared RAG service, only when the semantic index is
    enabled (the panel re-chunk worker's own seam).

    Returns ``None`` on every unavailable/failed shape: the opt-in re-index
    then reports itself skipped (disclosed in the payload), never raises --
    the chunk-row replacement is the call's own contract.
    """
    try:
        from ..RAG_Search.ingestion_indexing import (
            get_shared_rag_service,
            semantic_indexing_available,
        )

        if not semantic_indexing_available():
            return None
        return get_shared_rag_service()
    except Exception as exc:  # noqa: BLE001 — degrade, never sink the re-chunk
        logger.debug(f"shared RAG service unavailable for re-chunk: {exc}")
        return None


__all__ = [
    "RECHUNK_POLICY_ACTION_ID",
    "SPEC_SAVE_POLICY_ACTION_ID",
    "LocalMediaChunkToolService",
]
