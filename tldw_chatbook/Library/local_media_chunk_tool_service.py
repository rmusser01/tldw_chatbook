"""The media chunking agent tools' handlers (chunking-agent-tools Task 3).

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
* ``library_list_chunk_specs`` / ``library_save_chunk_spec`` /
  ``library_rechunk_media`` -- not-yet payloads naming their own landing
  task; the handlers land with Tasks 4-5 of this same change, so the
  surface never ships a dead name that silently no-ops.

Error discipline mirrors the sibling services exactly: ``LibraryToolError``
payloads, ``sqlite3.Error``/``OSError`` and unexpected exceptions scrubbed
to the storage-error payload, never a stack trace, SQL, or path.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from tldw_chatbook.Library.library_tool_contract import (
    DEFAULT_MAX_NODES,
    DISPLAY_NAME_MAX_BYTES,
    ERROR_CONTENT_CHANGED,
    ERROR_FEATURE_UNAVAILABLE,
    ERROR_INVALID_ARGUMENT,
    ERROR_NOT_FOUND,
    ERROR_STORAGE_ERROR,
    LIBRARY_TOOL_DESCRIPTORS,
    MAX_CHUNK_CONTEXT,
    MAX_MAX_NODES,
    MAX_RESULT_BYTES,
    LibraryToolDescriptor,
    LibraryToolError,
    check_cursor_revision,
    make_cursor,
    normalize_display_text,
    parse_cursor,
    parse_public_id,
    serialized_size,
)

#: The NULL chunk family's wire label (Task 2's backend uses the same
#: rendering; the string round-trips back as the ``chunk_type`` filter).
_PRIMARY_FAMILY_LABEL = "primary"
_LEGACY_ENGINE_LABEL = "legacy"

#: Spec §8.13's exact degradation hint.
_RECHUNK_HINT = "no stored chunks — use library_rechunk_media to enable unit fetches"

#: The tools whose handlers land with Tasks 4-5 of this change.
_PENDING_TOOL_LANDINGS = {
    "library_list_chunk_specs": "the chunk-spec listing lands with the spec tools task in this change",
    "library_save_chunk_spec": "saving chunk specs lands with the spec tools task in this change",
    "library_rechunk_media": "the re-chunk tool lands with the re-chunk task in this change",
}


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
    duck-typed handles, so tests stub them freely. All four descriptor
    names (plus the not-yet ``library_rechunk_media`` name, whose descriptor
    lands with its handler) resolve through :meth:`invoke`.
    """

    def __init__(
        self,
        media_db: Any = None,
        media_reading_service: Any = None,
        template_interop: Any = None,
    ) -> None:
        self._media_db = media_db
        self._reading = media_reading_service
        self._templates = template_interop

    # -- Entry point ---------------------------------------------------------

    def invoke(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one chunk tool; failures return the structured error payload."""
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
        landing = _PENDING_TOOL_LANDINGS.get(tool_name)
        if landing is not None:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                f"{tool_name} is not available yet; {landing}.",
            )
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
        """The selected family's ``[min_index, max_index]``, or None."""
        if chunk_type is None or chunk_type == _PRIMARY_FAMILY_LABEL:
            clause, params = "chunk_type IS NULL", (media_id,)
        else:
            clause, params = "chunk_type = ?", (media_id, chunk_type)
        row = self._media_db.get_connection().execute(
            f"SELECT MIN(chunk_index) AS lo, MAX(chunk_index) AS hi "
            f"FROM UnvectorizedMediaChunks "
            f"WHERE media_id = ? AND deleted = 0 AND {clause}",
            params,
        ).fetchone()
        if row is None or row["lo"] is None:
            return None
        return int(row["lo"]), int(row["hi"])

    # -- library_get_media_structure (spec §4.1) --------------------------------

    def _structure(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
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


__all__ = ["LocalMediaChunkToolService"]
