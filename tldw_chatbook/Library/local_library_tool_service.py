"""Shared synchronous core for the direct Library tools (task-1337, ADR-030).

One ``LocalLibraryToolService`` owns the public operation contract and
delegates storage work to the six existing local backend services (media,
notes, prompts, skills, conversations, collections) plus the dedicated
media chunk-tool service (structure/fetch/spec operations), and owns the
note-save write path itself (student-workflow spec §4: rows via the legacy
notes interop, folders/placements via the async ``NotesScopeService``).
Both runtimes -- the Console provider and local MCP registration -- call
this core; the descriptor table, ID/cursor codecs, validation, and byte
fitting all live in ``library_tool_contract`` so the two surfaces cannot
drift.

Pure synchronous core: no Textual, MCP, or agent imports. Local backends whose
methods are declared async but perform local work (prompts, skills) are
bridged with ``asyncio.run`` only when their result is awaitable, which is
safe because this core is never called from an event-loop thread (spec §2):
Console runs it on the agent worker thread and MCP wraps it with
``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
from datetime import date, datetime
from typing import Any, Mapping

from tldw_chatbook.Library.library_tool_contract import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MESSAGE_LIMIT,
    DISPLAY_NAME_MAX_BYTES,
    ERROR_CONTENT_CHANGED,
    ERROR_FEATURE_UNAVAILABLE,
    ERROR_INVALID_ARGUMENT,
    ERROR_NOT_FOUND,
    ERROR_STORAGE_ERROR,
    KEYWORD_VALUE_MAX_CHARS,
    KEYWORDS_PER_ITEM_MAX,
    LIBRARY_TOOL_DESCRIPTORS,
    LibraryToolDescriptor,
    LibraryToolError,
    MAX_MESSAGE_LIMIT,
    MAX_RESULT_BYTES,
    PREVIEW_MAX_CHARS,
    SAVE_NOTE_CONTENT_MAX_CHARS,
    SAVE_NOTE_FOLDER_MAX_CHARS,
    SAVE_NOTE_TITLE_MAX_CHARS,
    check_cursor_revision,
    fit_page_payload,
    make_cursor,
    make_public_id,
    normalize_display_text,
    parse_cursor,
    parse_public_id,
    serialized_size,
    validate_max_chars,
    validate_page_args,
    validate_search_query,
)
from tldw_chatbook.Notes.note_folder_models import (
    FolderCapabilityError,
    FolderCollisionError,
    FolderValidationError,
    normalize_folder_name,
)
from tldw_chatbook.runtime_policy.types import PolicyDeniedError
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError

_LIST_METHODS = {
    "media": "list_library_media",
    "note": "list_library_notes",
    "prompt": "list_library_prompts",
    "skill": "list_library_skills",
    "conversation": "list_library_conversations",
    "collection": "list_library_collections",
}

_SEARCH_METHODS = {
    "media": "search_library_media",
    "note": "search_library_notes",
    "prompt": "search_library_prompts",
    "skill": "search_library_skills",
    "conversation": "search_library_conversations",
    "collection": "search_library_collections",
}

_PROMPT_SECTIONS = ("details", "system_prompt", "user_prompt", "prompt_definition")

#: The media chunking operations (chunking-agent-tools spec §4) routed to
#: ``LocalMediaChunkToolService`` rather than the six item-type backends.
_MEDIA_CHUNK_OPERATIONS = frozenset(
    {"structure", "chunk", "spec_list", "spec_save", "rechunk"}
)

#: Student-workflow (spec §4/§6): the policy action the note-save tool runs
#: under. Registered in ``runtime_policy/registry.py`` (the ``library.notes``
#: resource on the ``library_collections`` capability, local-only); denial
#: precedes every backend call.
SAVE_NOTE_POLICY_ACTION_ID = "library.notes.save.local"

#: The scope the note-save folder seam is pinned to (spec §4.3): the notes
#: UI's own local scope -- any other scope makes folders invisible there.
#: Written as the enum's value (``ScopeType.LOCAL_NOTE`` is a ``str`` enum)
#: because importing ``notes_scope_service`` here is circular: it imports
#: ``Library.library_content_evidence``, whose package ``__init__`` imports
#: this module. The scope service normalizes the string to the same member.
_SAVE_NOTE_FOLDER_SCOPE = "local_note"

#: Root children-page size for the folder ensure lookup (the repository's
#: own ceiling) and the page cap: one-level lookups stop after this many
#: pages even if a pathological library keeps paginating.
_SAVE_NOTE_FOLDER_PAGE_LIMIT = 500
_SAVE_NOTE_FOLDER_MAX_PAGES = 20


def _invalid(message: str) -> LibraryToolError:
    return LibraryToolError(ERROR_INVALID_ARGUMENT, message)


def _not_found(
    message: str = "The requested Library item was not found.",
) -> LibraryToolError:
    return LibraryToolError(ERROR_NOT_FOUND, message)


def _storage_error_payload() -> dict[str, Any]:
    """Scrubbed operational-failure payload: no SQL, paths, or exception text."""
    return LibraryToolError(
        ERROR_STORAGE_ERROR,
        # "operation", not "read": the write paths (save-note) reuse this.
        "The local Library store could not complete the operation.",
        retryable=True,
    ).to_payload()


def _folder_capability_unavailable() -> LibraryToolError:
    """The folder seam's named capability refusal (Qodo review pin).

    ``NotesScopeService`` raises ``FolderCapabilityError`` when the pinned
    scope's folder repository is absent or lacks the operation; mapping it
    here keeps the deployment-shape failure a NAMED ``feature_unavailable``
    instead of the generic scrubbed ``storage_error``.
    """
    return LibraryToolError(
        ERROR_FEATURE_UNAVAILABLE,
        "Folder operations are not available for the local note folder"
        " backend in this deployment.",
    )


def _run(value: Any) -> Any:
    """Await a backend result when (and only when) it is awaitable."""
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _bound_preview(value: Any) -> str:
    """Display-normalize a preview, bounded to 240 characters (spec §6)."""
    text, _ = normalize_display_text(
        str(value)[:PREVIEW_MAX_CHARS], max_bytes=PREVIEW_MAX_CHARS * 4
    )
    return text


def _bound_keywords(values: Any) -> list[str]:
    """Bound keyword values to 20 entries of at most 120 characters (spec §6)."""
    bounded: list[str] = []
    for value in list(values or ())[:KEYWORDS_PER_ITEM_MAX]:
        text, _ = normalize_display_text(
            str(value)[:KEYWORD_VALUE_MAX_CHARS], max_bytes=KEYWORD_VALUE_MAX_CHARS * 4
        )
        bounded.append(text)
    return bounded


def _json_safe(value: Any) -> Any:
    """Coerce backend values (e.g. parsed DATETIME columns) to JSON-safe forms.

    Some stores decode timestamp columns into ``datetime`` objects; those break
    ``serialized_size`` and are not wire-serializable, so every value crossing
    into a response payload passes through here.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _make_brief(
    item_type: str,
    *,
    raw_id: Any,
    display_key: str,
    display_value: Any,
    preview: Any,
    keywords: Any,
    keyword_total: Any,
    keywords_truncated: Any,
    matched_fields: Any,
    matched_keywords: Any,
    metadata: tuple[tuple[str, Any], ...],
) -> dict[str, Any]:
    """Normalize one backend list/search row into the common brief shape."""
    display_text, display_truncated = normalize_display_text(
        display_value, max_bytes=DISPLAY_NAME_MAX_BYTES
    )
    visible_keywords = _bound_keywords(keywords)
    total = int(keyword_total or 0)
    brief: dict[str, Any] = {
        "id": make_public_id(item_type, raw_id),
        "type": item_type,
        display_key: display_text,
        f"{display_key}_truncated": display_truncated,
    }
    if preview is not None:
        brief["preview"] = _bound_preview(preview)
    brief["keywords"] = visible_keywords
    brief["keyword_total"] = total
    brief["keywords_truncated"] = bool(keywords_truncated) or total > len(
        visible_keywords
    )
    for key, value in metadata:
        if value is not None:
            brief[key] = _json_safe(value)
    if matched_fields is not None:
        brief["matched_fields"] = sorted(matched_fields)
        brief["matched_keywords"] = _bound_keywords(matched_keywords)
    return brief


def _metadata_item(
    public_id: str,
    item_type: str,
    display_key: str,
    display_value: Any,
    metadata: tuple[tuple[str, Any], ...],
) -> dict[str, Any]:
    """The bounded ``item`` block of a get response."""
    display_text, display_truncated = normalize_display_text(
        display_value, max_bytes=DISPLAY_NAME_MAX_BYTES
    )
    item: dict[str, Any] = {
        "id": public_id,
        "type": item_type,
        display_key: display_text,
        f"{display_key}_truncated": display_truncated,
    }
    for key, value in metadata:
        if value is not None:
            item[key] = _json_safe(value)
    return item


def _finalize_text_payload(
    *,
    item: dict[str, Any],
    public_id: str,
    revision: str,
    start: int,
    max_chars: int,
    text: str,
    total_chars: int,
    cursor_state: dict[str, Any],
) -> dict[str, Any]:
    """Assemble, byte-fit, and cursor-seal a get-text response (spec §7)."""
    window_end = start + len(text)

    def _sealed(end: int) -> dict[str, Any]:
        content = {
            "text": text[: end - start],
            "start": start,
            "end": end,
            "total_chars": total_chars,
            "requested_max_chars": max_chars,
            "returned_chars": end - start,
            "revision": revision,
            "has_more": end < total_chars,
            "next_cursor": None,
        }
        if content["has_more"]:
            content["next_cursor"] = make_cursor(
                item_id=public_id,
                revision=revision,
                offset=end,
                **cursor_state,
            )
        return {"item": item, "content": content}

    sealed = _sealed(window_end)
    if serialized_size(sealed) <= MAX_RESULT_BYTES:
        return sealed
    # Find the LARGEST prefix whose fully sealed payload -- continuation
    # cursor included -- fits the ceiling. Every candidate below window_end
    # has_more, so the serialized size grows monotonically with end.
    best = start
    lo, hi = start, window_end - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if serialized_size(_sealed(mid)) <= MAX_RESULT_BYTES:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return _sealed(best)


def _validate_message_limit(value: Any) -> int:
    if value is None:
        return DEFAULT_MESSAGE_LIMIT
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid("message_limit must be an integer")
    if value < 1:
        raise _invalid("message_limit must be at least 1")
    return min(value, MAX_MESSAGE_LIMIT)


def _fit_message_page(payload: dict[str, Any]) -> None:
    """Byte-fit a conversation message page in place (spec §7).

    Trailing messages drop first; a lone still-oversized message shortens to
    the largest prefix that fits. ``returned_chars``/``has_more`` of a
    shortened message reflect the actual returned prefix so continuation
    neither skips nor repeats characters.
    """
    messages = payload["messages"]
    while len(messages) > 1 and serialized_size(payload) > MAX_RESULT_BYTES:
        messages.pop()
    if len(messages) != 1 or serialized_size(payload) <= MAX_RESULT_BYTES:
        return
    message = messages[0]
    text = message["text"]
    start = int(message["char_start"])
    total = int(message["total_chars"])
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        message["text"] = text[:mid]
        message["returned_chars"] = mid
        message["has_more"] = start + mid < total
        if serialized_size(payload) <= MAX_RESULT_BYTES:
            lo = mid
        else:
            hi = mid - 1
    message["text"] = text[:lo]
    message["returned_chars"] = lo
    message["has_more"] = start + lo < total


class LocalLibraryToolService:
    """Synchronous dispatcher for the 18 descriptor-defined Library tools.

    All backends are optional: a missing backend maps its tools to
    ``feature_unavailable`` rather than failing construction, so partial
    deployments still serve the remaining Library types.
    """

    def __init__(
        self,
        *,
        media_service: Any = None,
        notes_service: Any = None,
        prompt_service: Any = None,
        skills_service: Any = None,
        conversation_service: Any = None,
        collections_service: Any = None,
        media_chunk_service: Any = None,
        notes_user_id: str = "local_library",
        notes_scope_service: Any = None,
        policy_enforcer: Any = None,
    ) -> None:
        self._media = media_service
        self._notes = notes_service
        self._prompts = prompt_service
        self._skills = skills_service
        self._conversations = conversation_service
        self._collections = collections_service
        self._media_chunk = media_chunk_service
        self._notes_user_id = notes_user_id
        # Student-workflow (spec §4.3): the folder seam. Note rows go through
        # the legacy interop above; folders/placements live only in the async
        # NotesScopeService, pinned to its LOCAL_NOTE scope. Optional: a
        # missing handle degrades folder requests to feature_unavailable
        # (never to a half-written note).
        self._notes_scope = notes_scope_service
        # Student-workflow (spec §6): the WRITING note tool's service-level
        # gate (the chunk-tools precedent) -- the same enforcer handle the
        # MCP runtime gate enforces with; None leaves that outer gate alone.
        self._policy_enforcer = policy_enforcer

    # -- Entry point ---------------------------------------------------------

    def invoke(self, tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one descriptor-defined tool; failures return the error payload."""
        try:
            return self._dispatch(tool_name, arguments)
        except LibraryToolError as exc:
            return exc.to_payload()
        except (sqlite3.Error, OSError):
            return _storage_error_payload()
        except Exception:
            # Backend-specific operational errors (DatabaseError and friends)
            # are scrubbed to the same payload; never BaseException.
            return _storage_error_payload()

    # -- Dispatch ------------------------------------------------------------

    def _dispatch(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        descriptor = LIBRARY_TOOL_DESCRIPTORS.get(tool_name)
        if descriptor is None:
            raise _invalid(f"unknown Library tool: {tool_name!r}")
        if not isinstance(arguments, Mapping):
            raise _invalid("arguments must be a JSON object")
        self._validate_argument_keys(descriptor, arguments)
        if descriptor.operation in _MEDIA_CHUNK_OPERATIONS:
            return self._media_chunk_tool(descriptor, tool_name, arguments)
        backend = self._backend(descriptor.item_type)
        if descriptor.operation == "list":
            return self._list(descriptor, backend, arguments)
        if descriptor.operation == "search":
            return self._search(descriptor, backend, arguments)
        if descriptor.operation == "save":
            return self._save_note(descriptor, backend, arguments)
        return self._get(descriptor, backend, arguments)

    def _media_chunk_tool(
        self,
        descriptor: LibraryToolDescriptor,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Route one media chunking operation to its dedicated service.

        The chunk tools own their backend handles (media DB, reading
        service, template interop), so they do NOT resolve through the six
        item-type backends; a missing chunk service degrades its tools to
        the same structured ``feature_unavailable`` as any missing backend.
        """
        if self._media_chunk is None:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "The local media chunk tool backend is not available in this"
                " deployment.",
            )
        return self._media_chunk.invoke(tool_name, arguments)

    @staticmethod
    def _validate_argument_keys(
        descriptor: LibraryToolDescriptor, arguments: Mapping[str, Any]
    ) -> None:
        allowed = set(descriptor.input_schema.get("properties", ()))
        unknown = sorted(str(key) for key in arguments if key not in allowed)
        if unknown:
            raise _invalid(f"unknown argument(s): {', '.join(unknown)}")
        required = descriptor.input_schema.get("required", ())
        missing = [key for key in required if key not in arguments]
        if missing:
            raise _invalid(f"missing required argument(s): {', '.join(missing)}")

    def _backend(self, item_type: str) -> Any:
        backend = {
            "media": self._media,
            "note": self._notes,
            "prompt": self._prompts,
            "skill": self._skills,
            "conversation": self._conversations,
            "collection": self._collections,
        }[item_type]
        if backend is None:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                f"The local {item_type} backend is not available in this deployment.",
            )
        return backend

    # -- List / search ---------------------------------------------------------

    def _list(
        self,
        descriptor: LibraryToolDescriptor,
        backend: Any,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        limit, offset = validate_page_args(
            arguments.get("limit"), arguments.get("offset")
        )
        method = getattr(backend, _LIST_METHODS[descriptor.item_type])
        if descriptor.item_type == "note":
            payload = method(self._notes_user_id, limit=limit, offset=offset)
        else:
            payload = _run(method(limit=limit, offset=offset))
        return self._page_envelope(
            descriptor.item_type, payload, limit=limit, offset=offset
        )

    def _search(
        self,
        descriptor: LibraryToolDescriptor,
        backend: Any,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        query = validate_search_query(arguments.get("query"))
        limit, offset = validate_page_args(
            arguments.get("limit"), arguments.get("offset")
        )
        method = getattr(backend, _SEARCH_METHODS[descriptor.item_type])
        if descriptor.item_type == "note":
            payload = method(self._notes_user_id, query=query, limit=limit, offset=offset)
        elif descriptor.item_type == "prompt":
            payload = _run(method(query, limit=limit, offset=offset))
        else:
            payload = _run(method(query=query, limit=limit, offset=offset))
        return self._page_envelope(
            descriptor.item_type, payload, limit=limit, offset=offset
        )

    def _page_envelope(
        self, item_type: str, payload: Mapping[str, Any], *, limit: int, offset: int
    ) -> dict[str, Any]:
        raw_items = payload.get("items") or []
        total = int(payload.get("total") or 0)
        items = [self._brief(item_type, raw) for raw in raw_items]
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

    def _brief(self, item_type: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        common = {
            "keywords": raw.get("keywords"),
            "keyword_total": raw.get("keyword_total"),
            "keywords_truncated": raw.get("keywords_truncated"),
            "matched_fields": raw.get("matched_fields"),
            "matched_keywords": raw.get("matched_keywords"),
        }
        if item_type == "media":
            return _make_brief(
                item_type,
                raw_id=raw["uuid"],
                display_key="title",
                display_value=raw.get("title"),
                preview=raw.get("preview"),
                metadata=(
                    ("media_type", raw.get("media_type")),
                    ("author", raw.get("author")),
                    ("ingestion_date", raw.get("ingestion_date")),
                    ("last_modified", raw.get("last_modified")),
                ),
                **common,
            )
        if item_type == "note":
            return _make_brief(
                item_type,
                raw_id=raw["id"],
                display_key="title",
                display_value=raw.get("title"),
                preview=raw.get("preview"),
                metadata=(
                    ("created_at", raw.get("created_at")),
                    ("last_modified", raw.get("last_modified")),
                ),
                **common,
            )
        if item_type == "prompt":
            return _make_brief(
                item_type,
                raw_id=raw["uuid"],
                display_key="name",
                display_value=raw.get("name"),
                preview=raw.get("details_preview"),
                metadata=(
                    ("author", raw.get("author")),
                    ("last_modified", raw.get("last_modified")),
                    ("has_system_prompt", raw.get("has_system_prompt")),
                    ("has_user_prompt", raw.get("has_user_prompt")),
                    ("has_prompt_definition", raw.get("has_prompt_definition")),
                ),
                **common,
            )
        if item_type == "skill":
            return _make_brief(
                item_type,
                raw_id=raw["name"],
                display_key="name",
                display_value=raw.get("name"),
                preview=raw.get("description"),
                keywords=None,
                keyword_total=0,
                keywords_truncated=False,
                matched_fields=raw.get("matched_fields"),
                matched_keywords=raw.get("matched_keywords"),
                metadata=(
                    ("trust_status", raw.get("trust_status")),
                    ("trust_blocked", raw.get("trust_blocked")),
                ),
            )
        if item_type == "conversation":
            return _make_brief(
                item_type,
                raw_id=raw["id"],
                display_key="title",
                display_value=raw.get("title"),
                preview=None,
                metadata=(
                    ("created_at", raw.get("created_at")),
                    ("last_modified", raw.get("last_modified")),
                ),
                **common,
            )
        return _make_brief(
            item_type,
            raw_id=raw["collection_id"],
            display_key="name",
            display_value=raw.get("name"),
            preview=raw.get("description"),
            keywords=None,
            keyword_total=0,
            keywords_truncated=False,
            matched_fields=raw.get("matched_fields"),
            matched_keywords=raw.get("matched_keywords"),
            metadata=(
                ("item_count", raw.get("item_count")),
                ("created_at", raw.get("created_at")),
                ("updated_at", raw.get("updated_at")),
            ),
        )

    # -- Get dispatch ------------------------------------------------------------

    def _get(
        self,
        descriptor: LibraryToolDescriptor,
        backend: Any,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        item_type = descriptor.item_type
        public_id = arguments.get("id")
        _, raw_id = parse_public_id(public_id, expected_type=item_type)
        if item_type == "media":
            return self._get_media(backend, public_id, raw_id, arguments)
        if item_type == "note":
            return self._get_note(backend, public_id, raw_id, arguments)
        if item_type == "prompt":
            return self._get_prompt(backend, public_id, raw_id, arguments)
        if item_type == "skill":
            return self._get_skill(backend, public_id, raw_id, arguments)
        if item_type == "conversation":
            return self._get_conversation(backend, public_id, raw_id, arguments)
        return self._get_collection(backend, public_id, raw_id, arguments)

    @staticmethod
    def _cursor_state(
        arguments: Mapping[str, Any], public_id: str
    ) -> dict[str, Any] | None:
        raw_cursor = arguments.get("cursor")
        if raw_cursor is None:
            return None
        state = parse_cursor(raw_cursor)
        if state["item"] != public_id:
            raise _invalid("continuation cursor belongs to a different Library item")
        return state

    # -- Get: media / notes (single text body) --------------------------------

    def _get_media(
        self, backend: Any, public_id: str, raw_id: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        max_chars = validate_max_chars(arguments.get("max_chars"))
        cursor = self._cursor_state(arguments, public_id)
        start = cursor["off"] if cursor is not None else 0
        detail = backend.get_library_media_text(raw_id, start=start, max_chars=max_chars)
        if detail is None:
            raise _not_found()
        revision = str(detail.get("version"))
        if cursor is not None:
            check_cursor_revision(cursor, revision)
        item = _metadata_item(
            public_id,
            "media",
            "title",
            detail.get("title"),
            (
                ("media_type", detail.get("media_type")),
                ("author", detail.get("author")),
                ("ingestion_date", detail.get("ingestion_date")),
                ("last_modified", detail.get("last_modified")),
            ),
        )
        return _finalize_text_payload(
            item=item,
            public_id=public_id,
            revision=revision,
            start=start,
            max_chars=max_chars,
            text=detail["text"],
            total_chars=int(detail.get("total_chars") or 0),
            cursor_state={},
        )

    def _get_note(
        self, backend: Any, public_id: str, raw_id: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        max_chars = validate_max_chars(arguments.get("max_chars"))
        cursor = self._cursor_state(arguments, public_id)
        start = cursor["off"] if cursor is not None else 0
        detail = backend.get_library_note_text(
            self._notes_user_id, raw_id, start=start, max_chars=max_chars
        )
        if detail is None:
            raise _not_found()
        revision = str(detail.get("version"))
        if cursor is not None:
            check_cursor_revision(cursor, revision)
        item = _metadata_item(
            public_id,
            "note",
            "title",
            detail.get("title"),
            (
                ("created_at", detail.get("created_at")),
                ("last_modified", detail.get("last_modified")),
            ),
        )
        return _finalize_text_payload(
            item=item,
            public_id=public_id,
            revision=revision,
            start=start,
            max_chars=max_chars,
            text=detail["text"],
            total_chars=int(detail.get("total_chars") or 0),
            cursor_state={},
        )

    # -- Save: notes (student-workflow spec §4) ----------------------------------

    def _enforce_save_note_policy(self) -> None:
        """Spec §6: the save runs under ``library.notes.save.local``.

        Enforcement precedes EVERY backend touch (denial -> the named error
        payload, no note row, no folder). No-op without an enforcer handle --
        the chunk-tools precedent: the MCP runtime gate (the re-pointed
        ``_TOOL_ACTION_IDS`` mapping) stays the always-on outer layer, and
        construction sites wire the enforcer where a runtime-policy context
        exists.
        """
        if self._policy_enforcer is None:
            return
        try:
            self._policy_enforcer.require_allowed(
                action_id=SAVE_NOTE_POLICY_ACTION_ID
            )
        except PolicyDeniedError as exc:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "Saving notes is not permitted by the current runtime policy"
                f" ({SAVE_NOTE_POLICY_ACTION_ID}): {exc.user_message}",
                details={
                    "policy_action": SAVE_NOTE_POLICY_ACTION_ID,
                    "reason_code": str(
                        getattr(exc, "reason_code", "authority_denied")
                    ),
                },
            ) from exc

    @staticmethod
    def _validate_save_note_arguments(
        arguments: Mapping[str, Any],
    ) -> tuple[str, str, str | None, str | None, int | None]:
        """Type-check the save args; returns ``(title, content, folder,
        note_id, expected_version)``.

        The bounds themselves are schema-level (the descriptor's maxLength
        literals); this re-checks the emptiness/typing the row-writer would
        trip on anyway, so a schema-bypassing caller still fails closed with
        the named error. The together-rule is pinned here FIRST: exactly one
        of note_id/expected_version is the most-missed invalid shape.
        """
        title = arguments.get("title")
        if not isinstance(title, str) or not title.strip():
            raise _invalid("title must be a non-empty string")
        if len(title) > SAVE_NOTE_TITLE_MAX_CHARS:
            raise _invalid(
                f"title must be at most {SAVE_NOTE_TITLE_MAX_CHARS} characters"
                f" (got {len(title)})"
            )
        content = arguments.get("content")
        if not isinstance(content, str) or not content:
            raise _invalid("content must be a non-empty string")
        if len(content) > SAVE_NOTE_CONTENT_MAX_CHARS:
            raise _invalid(
                f"content must be at most {SAVE_NOTE_CONTENT_MAX_CHARS}"
                f" characters (got {len(content)})"
            )
        folder = arguments.get("folder")
        if folder is not None and (not isinstance(folder, str) or not folder.strip()):
            raise _invalid("folder must be a non-empty string when supplied")
        if folder is not None and len(folder) > SAVE_NOTE_FOLDER_MAX_CHARS:
            raise _invalid(
                f"folder must be at most {SAVE_NOTE_FOLDER_MAX_CHARS} characters"
                f" (got {len(folder)})"
            )
        note_id = arguments.get("note_id")
        expected_version = arguments.get("expected_version")
        if (note_id is None) != (expected_version is None):
            raise _invalid(
                "note_id and expected_version must be supplied together"
                " (both for an update, neither for a create)"
            )
        if expected_version is not None and (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or expected_version < 1
        ):
            raise _invalid("expected_version must be an integer of at least 1")
        return title, content, folder, note_id, expected_version

    def _find_note_folder(self, requested_key: str) -> Any | None:
        """Name-lookup one root folder through the scope service's own
        children-list seam (parent_id=None covers the root -- verified against
        the repository's ``parent_id IS NULL`` predicate).

        The repository's path-getter (``get_folder_by_path``) is NOT
        scope-exposed; reaching the repository from the tool layer is ruled
        out (the ledger's STOP rule), so the children-list route is the seam.
        """
        scope = self._notes_scope
        offset = 0
        for _ in range(_SAVE_NOTE_FOLDER_MAX_PAGES):
            page = _run(
                scope.list_note_folder_children(
                    scope=_SAVE_NOTE_FOLDER_SCOPE,
                    parent_id=None,
                    limit=_SAVE_NOTE_FOLDER_PAGE_LIMIT,
                    offset=offset,
                    user_id=self._notes_user_id,
                )
            )
            for candidate in getattr(page, "folders", None) or ():
                try:
                    if normalize_folder_name(candidate.name).key == requested_key:
                        return candidate
                except FolderValidationError:  # pragma: no cover - stored rows
                    continue  # are pre-normalized; skip pathological ones
            next_offset = getattr(page, "next_folder_offset", None)
            if next_offset is None:
                return None
            offset = int(next_offset)
        return None

    def _ensure_note_folder(self, name: str) -> Any:
        """Idempotently return the ONE root folder for ``name``.

        Lookup -> create on miss -> RE-QUERY on collision (``create_note_folder``
        is not idempotent; a concurrent creator winning the race is tolerated
        by re-reading, never raised to the agent -- spec §4.3).
        """
        if self._notes_scope is None:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "The local note folder backend is not available in this"
                " deployment.",
            )
        try:
            requested = normalize_folder_name(name)
        except FolderValidationError:
            raise _invalid(
                "folder must be a single valid folder name (one level, no"
                " slashes, not '.' or '..')"
            ) from None
        try:
            found = self._find_note_folder(requested.key)
        except FolderCapabilityError:
            # The seam's own capability refusal (a scope without folder
            # support) is the NAMED feature_unavailable, never the scrubbed
            # storage_error the generic handler would return.
            raise _folder_capability_unavailable() from None
        if found is not None:
            return found
        try:
            return _run(
                self._notes_scope.create_note_folder(
                    scope=_SAVE_NOTE_FOLDER_SCOPE,
                    name=name,
                    parent_id=None,
                    user_id=self._notes_user_id,
                )
            )
        except FolderCapabilityError:
            raise _folder_capability_unavailable() from None
        except FolderCollisionError:
            # Lost the create race: the concurrent winner is now visible to
            # the lookup, and the placement target converges on ONE folder.
            found = self._find_note_folder(requested.key)
            if found is not None:
                return found
            raise LibraryToolError(
                ERROR_STORAGE_ERROR,
                "The local note folder could not be ensured; retry the save.",
                retryable=True,
            ) from None

    def _save_note(
        self,
        descriptor: LibraryToolDescriptor,
        backend: Any,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create (default) or version-locked update of one note (spec §4).

        Rows go through the legacy interop (the notes UI's own row-writer);
        folders/placements go through the async scope service (bridged with
        the established ``_run``/``asyncio.run`` pattern). Order matters:
        validate -> policy -> update pre-flight (id/existence/version) ->
        folder-ensure -> row write -> attach, so a folder failure never
        lands an orphaned note row and a failed update never mints a folder.
        """
        del descriptor  # routing already resolved; the operation is singular
        # Deferred import: the DB module is heavy and only the exception
        # types are needed (the personas-screen precedent for their home).
        from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError, InputError

        title, content, folder_name, note_id, expected_version = (
            self._validate_save_note_arguments(arguments)
        )
        # Policy before ANY backend touch (spec §6).
        self._enforce_save_note_policy()

        # Update pre-flight (Qodo review): for UPDATE calls the id parse,
        # the not_found existence check, and the version read all run BEFORE
        # the folder-ensure, so a deterministically failing update (unknown
        # id, stale version) never mints a folder for a call that then
        # fails. The ``update_note`` ConflictError below stays as the race
        # backstop (a version moving between the read and the row write is
        # the one accepted folder-residual window).
        raw_id: str | None = None
        if note_id is not None:
            _, raw_id = parse_public_id(note_id, expected_type="note")
            row = backend.get_note_by_id(self._notes_user_id, raw_id)
            if row is None:
                raise _not_found("The requested Library item was not found.")
            stored_version = row.get("version") if isinstance(row, Mapping) else None
            try:
                stale = int(stored_version) != int(expected_version)
            except (TypeError, ValueError):
                stale = False  # unreadable version: the row write arbitrates
            if stale:
                raise LibraryToolError(
                    ERROR_CONTENT_CHANGED,
                    "The note changed since the expected version was read;"
                    " re-read it (library_get_note) and retry with the"
                    " current version.",
                    details={"hint": "re_read_and_retry"},
                )

        # Folder next: a failed folder-ensure must not orphan a note row.
        folder = (
            self._ensure_note_folder(folder_name) if folder_name is not None else None
        )

        if note_id is None:
            try:
                raw_id = backend.add_note(
                    self._notes_user_id, title=title, content=content
                )
            except InputError as exc:
                raise _invalid(str(exc)) from exc
            version = 1
            created = True
        else:
            try:
                updated = backend.update_note(
                    self._notes_user_id,
                    raw_id,
                    {"title": title, "content": content},
                    expected_version,
                )
            except ConflictError as exc:
                raise LibraryToolError(
                    ERROR_CONTENT_CHANGED,
                    "The note changed since the expected version was read;"
                    " re-read it (library_get_note) and retry with the"
                    " current version.",
                    details={"hint": "re_read_and_retry"},
                ) from exc
            if not updated:
                raise LibraryToolError(
                    ERROR_CONTENT_CHANGED,
                    "The note changed since the expected version was read;"
                    " re-read it (library_get_note) and retry with the"
                    " current version.",
                    details={"hint": "re_read_and_retry"},
                )
            version = int(expected_version) + 1
            created = False

        if folder is not None:
            # Re-attach is safe: the repository's attach_manual revives the
            # latest membership history rather than duplicating it.
            _run(
                self._notes_scope.attach_note_to_folder(
                    scope=_SAVE_NOTE_FOLDER_SCOPE,
                    folder_id=folder.folder_id,
                    note_id=raw_id,
                    user_id=self._notes_user_id,
                )
            )

        item = _metadata_item(
            make_public_id("note", raw_id), "note", "title", title, ()
        )
        if folder is not None:
            item["folder"] = folder.name
        return {
            "item": item,
            "version": version,
            "created": created,
            "notes": [
                (
                    "Hold the returned id and version: an update needs note_id"
                    " and expected_version together (exactly one of them is"
                    " refused)."
                ),
                (
                    "Notes have no unique title; before re-running, search by"
                    " title (library_search_notes) and update the match instead"
                    " of creating a duplicate."
                ),
            ],
        }

    # -- Get: prompts (overview manifest + one section) -------------------------

    def _get_prompt(
        self, backend: Any, public_id: str, raw_id: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        section = arguments.get("section")
        if section is not None and section not in _PROMPT_SECTIONS:
            raise _invalid(f"section must be one of: {', '.join(_PROMPT_SECTIONS)}")
        cursor = self._cursor_state(arguments, public_id)
        if section is None:
            if cursor is not None:
                raise _invalid("a continuation cursor only applies to a section read")
            overview = _run(backend.get_library_prompt_overview(raw_id))
            if overview is None:
                raise _not_found()
            item = _metadata_item(
                public_id,
                "prompt",
                "name",
                overview.get("name"),
                (
                    ("author", overview.get("author")),
                    ("last_modified", overview.get("last_modified")),
                ),
            )
            sections = {}
            for name, info in (overview.get("sections") or {}).items():
                sections[name] = {
                    "total_chars": int(info.get("total_chars") or 0),
                    "preview": _bound_preview(info.get("preview") or ""),
                }
            return {"item": item, "sections": sections}
        if cursor is not None and cursor.get("sec") not in (None, section):
            raise _invalid("continuation cursor belongs to a different section")
        max_chars = validate_max_chars(arguments.get("max_chars"))
        start = cursor["off"] if cursor is not None else 0
        detail = _run(
            backend.get_library_prompt_section(
                raw_id, section, start=start, max_chars=max_chars
            )
        )
        if detail is None:
            raise _not_found()
        revision = str(detail.get("version"))
        if cursor is not None:
            check_cursor_revision(cursor, revision)
        item = _metadata_item(public_id, "prompt", "name", detail.get("name"), ())
        return _finalize_text_payload(
            item=item,
            public_id=public_id,
            revision=revision,
            start=start,
            max_chars=max_chars,
            text=detail["text"],
            total_chars=int(detail.get("total_chars") or 0),
            cursor_state={"section": section},
        )

    # -- Get: skills (safe detail + manifest-token file reads) -------------------

    def _get_skill(
        self, backend: Any, public_id: str, raw_id: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        file_token = arguments.get("file_token")
        cursor = self._cursor_state(arguments, public_id)
        if file_token is None:
            if cursor is not None:
                raise _invalid("a continuation cursor only applies to a file read")
            try:
                detail = _run(backend.get_library_skill(raw_id))
            except ValueError:
                raise _not_found() from None
            item = _metadata_item(
                public_id,
                "skill",
                "name",
                detail.get("name"),
                (("description", detail.get("description")),),
            )
            for key in ("trust_status", "trust_blocked", "validation_status"):
                if detail.get(key) is not None:
                    item[key] = detail[key]
            if not detail.get("trust_blocked"):
                if detail.get("body_total_chars") is not None:
                    item["body_total_chars"] = int(detail["body_total_chars"])
                if detail.get("body_preview") is not None:
                    item["body_preview"] = _bound_preview(detail["body_preview"])
                files = []
                for entry in detail.get("files") or ():
                    files.append(
                        {
                            "path": str(entry.get("path") or ""),
                            "size": int(entry.get("size") or 0),
                            "is_text": bool(entry.get("is_text")),
                            "file_token": entry.get("file_token"),
                        }
                    )
                item["files"] = files
            return {"item": item}
        if cursor is not None and cursor.get("ftok") not in (None, file_token):
            raise _invalid("continuation cursor belongs to a different file")
        max_chars = validate_max_chars(arguments.get("max_chars"))
        start = cursor["off"] if cursor is not None else 0
        try:
            segment = _run(
                backend.get_library_skill_file(
                    raw_id, file_token, start=start, max_chars=max_chars
                )
            )
        except SkillTrustBlockedError:
            raise LibraryToolError(
                ERROR_FEATURE_UNAVAILABLE,
                "The skill is not currently trusted, so its files cannot be read.",
            ) from None
        except ValueError:
            raise _invalid(
                "the file token is not valid for this skill; request the manifest again"
            ) from None
        revision = str(segment.get("revision"))
        if cursor is not None:
            check_cursor_revision(cursor, revision)
        item = _metadata_item(public_id, "skill", "name", raw_id, ())
        payload = _finalize_text_payload(
            item=item,
            public_id=public_id,
            revision=revision,
            start=start,
            max_chars=max_chars,
            text=segment["text"],
            total_chars=int(segment.get("total_chars") or 0),
            cursor_state={"file_token": file_token},
        )
        payload["file"] = {"path": str(segment.get("path") or "")}
        return payload

    # -- Get: conversations (message pages + within-message continuation) ---------

    def _get_conversation(
        self, backend: Any, public_id: str, raw_id: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        message_limit = _validate_message_limit(arguments.get("message_limit"))
        cursor = self._cursor_state(arguments, public_id)
        within_message = cursor is not None and "mid" in cursor
        if within_message:
            detail = backend.get_library_conversation_messages(
                raw_id,
                message_id=cursor["mid"],
                char_start=cursor["off"],
                max_chars=DEFAULT_MAX_CHARS,
            )
        else:
            message_offset = cursor.get("moff", 0) if cursor is not None else 0
            detail = backend.get_library_conversation_messages(
                raw_id,
                message_offset=message_offset,
                message_limit=message_limit,
                max_chars=DEFAULT_MAX_CHARS,
            )
        if detail is None:
            raise _not_found()
        conversation_revision = str(detail.get("version"))
        messages = [
            {key: _json_safe(value) for key, value in message.items()}
            for message in detail.get("messages") or ()
        ]
        if cursor is not None:
            if within_message:
                if not messages:
                    raise _not_found(
                        "The message no longer exists; start the read again."
                    )
                check_cursor_revision(cursor, str(messages[0].get("revision")))
            else:
                check_cursor_revision(cursor, conversation_revision)
        base_offset = (
            int(cursor["moff"])
            if within_message
            else int(detail.get("message_offset") or 0)
        )
        message_total = int(detail.get("message_total") or 0)
        item = _metadata_item(
            public_id, "conversation", "title", detail.get("title"), ()
        )
        payload: dict[str, Any] = {
            "item": item,
            "message_total": message_total,
            "message_offset": base_offset,
            "returned_message_count": len(messages),
            "has_more": False,
            "next_message_offset": None,
            "next_cursor": None,
            "include_rag_context": False,
            "messages": messages,
        }
        for _ in range(8):
            # The minted continuation cursor can itself push the page over the
            # ceiling. Refit WITH the previous round's cursor still in place
            # (so the fitter accounts for it), then re-derive and re-mint;
            # offsets only shrink across rounds, so this converges.
            _fit_message_page(payload)
            messages = payload["messages"]
            returned = len(messages)
            payload["has_more"] = False
            payload["next_message_offset"] = None
            payload["next_cursor"] = None

            if within_message:
                message = messages[0]
                char_end = message["char_start"] + message["returned_chars"]
                if char_end < message["total_chars"]:
                    payload["has_more"] = True
                    payload["next_message_offset"] = base_offset + 1
                    payload["next_cursor"] = make_cursor(
                        item_id=public_id,
                        revision=str(message["revision"]),
                        offset=char_end,
                        message_id=message["id"],
                        message_offset=base_offset,
                    )
                else:
                    resume = base_offset + 1
                    has_more = resume < message_total
                    payload["has_more"] = has_more
                    payload["next_message_offset"] = resume if has_more else None
                    if has_more:
                        payload["next_cursor"] = make_cursor(
                            item_id=public_id,
                            revision=conversation_revision,
                            offset=0,
                            message_offset=resume,
                        )
            else:
                cut_index = next(
                    (
                        index
                        for index, message in enumerate(messages)
                        if message["char_start"] + message["returned_chars"]
                        < message["total_chars"]
                    ),
                    None,
                )
                if cut_index is not None:
                    # The page ends at the first char-windowed message; its cursor
                    # continues inside that message, so no message is skipped and
                    # none repeats.
                    del messages[cut_index + 1 :]
                    returned = len(messages)
                    cut = messages[cut_index]
                    char_end = cut["char_start"] + cut["returned_chars"]
                    payload["has_more"] = True
                    payload["next_message_offset"] = base_offset + cut_index + 1
                    payload["next_cursor"] = make_cursor(
                        item_id=public_id,
                        revision=str(cut["revision"]),
                        offset=char_end,
                        message_id=cut["id"],
                        message_offset=base_offset + cut_index,
                    )
                else:
                    has_more = base_offset + returned < message_total
                    payload["has_more"] = has_more
                    payload["next_message_offset"] = (
                        base_offset + returned if has_more else None
                    )
                    if has_more:
                        payload["next_cursor"] = make_cursor(
                            item_id=public_id,
                            revision=conversation_revision,
                            offset=0,
                            message_offset=base_offset + returned,
                        )
            payload["returned_message_count"] = returned
            if serialized_size(payload) <= MAX_RESULT_BYTES:
                break
        return payload

    # -- Get: collections (direct-membership pages) ----------------------------

    def _get_collection(
        self, backend: Any, public_id: str, raw_id: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        limit, offset = validate_page_args(
            arguments.get("limit"), arguments.get("offset")
        )
        cursor = self._cursor_state(arguments, public_id)
        if cursor is not None:
            offset = cursor["off"]
        detail = backend.get_library_collection(raw_id, limit=limit, offset=offset)
        if detail is None:
            raise _not_found()
        revision = str(detail.get("updated_at"))
        if cursor is not None:
            check_cursor_revision(cursor, revision)
        item = _metadata_item(
            public_id,
            "collection",
            "name",
            detail.get("name"),
            (
                ("description", detail.get("description")),
                ("created_at", detail.get("created_at")),
                ("updated_at", detail.get("updated_at")),
            ),
        )
        members = [
            {key: _json_safe(value) for key, value in member.items()}
            for member in detail.get("members") or ()
        ]
        member_total = int(detail.get("member_total") or 0)
        payload: dict[str, Any] = {
            "item": item,
            "member_total": member_total,
            "offset": offset,
            "limit": limit,
            "has_more": False,
            "next_offset": None,
            "next_cursor": None,
            "members": members,
        }
        for _ in range(8):
            # Pop with the previous round's cursor still in place so the minted
            # cursor cannot push the sealed page over the ceiling.
            while members and serialized_size(payload) > MAX_RESULT_BYTES:
                members.pop()
            returned = len(members)
            has_more = offset + returned < member_total
            payload["has_more"] = has_more
            payload["next_offset"] = offset + returned if has_more else None
            payload["next_cursor"] = (
                make_cursor(
                    item_id=public_id, revision=revision, offset=offset + returned
                )
                if has_more
                else None
            )
            if serialized_size(payload) <= MAX_RESULT_BYTES:
                break
        return payload


__all__ = ["LocalLibraryToolService"]
