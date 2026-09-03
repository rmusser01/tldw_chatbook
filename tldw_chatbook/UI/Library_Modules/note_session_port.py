"""Adapter binding the Library's note services to the portable session port.

Moved verbatim out of ``tldw_chatbook/UI/Screens/library_screen.py`` by PR 0a
of the Library screen decomposition
(``.superpowers/sdd/2026-09-01-library-decomposition-foundation``; see
``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``).
``library_screen.py`` re-exports every name here so its import surface is
unchanged; later decomposition tasks import directly from this module.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from ...DB.ChaChaNotes_DB import ConflictError
from ...Library.library_notes_session import (
    DatabaseNotePortLoadReply,
    DatabaseNotePortSaveReply,
)
from ...Library.library_notes_state import (
    DatabaseNoteSavePayload,
    NormalizedDatabaseNote,
)
from .screen_helpers import library_note_persisted_title


class _LibraryDatabaseNoteSessionPort:
    """Adapt the existing Library note services to the portable session port."""

    def __init__(
        self,
        *,
        run_service_call: Any,
        notes_scope_service: Any,
        notes_service: Any,
        user_id: str,
        clock: Any,
    ) -> None:
        self._run_service_call = run_service_call
        self._notes_scope_service = notes_scope_service
        self._notes_service = notes_service
        self._user_id = user_id
        self._clock = clock

    @staticmethod
    def _keyword_strings(records: Any) -> tuple[str, ...]:
        """Return semantic keyword strings in their service-provided order."""
        if records is None:
            return ()
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("Keyword detail was not a sequence.")
        keywords: list[str] = []
        for record in records:
            value = record.get("keyword") if isinstance(record, Mapping) else record
            if value is None:
                continue
            keyword = value if isinstance(value, str) else str(value)
            if keyword:
                keywords.append(keyword)
        return tuple(keywords)

    async def load_note(self, note_id: str) -> DatabaseNotePortLoadReply:
        """Fetch detail plus keywords and return one coherent typed reply."""
        get_note_detail = getattr(self._notes_scope_service, "get_note_detail", None)
        if not callable(get_note_detail):
            return DatabaseNotePortLoadReply.failed("Note loading is unavailable.")
        try:
            detail = await self._run_service_call(
                get_note_detail,
                scope="local_note",
                note_id=note_id,
                user_id=self._user_id,
                isolate_in_worker=True,
            )
            if detail is None:
                return DatabaseNotePortLoadReply.missing()
            if not isinstance(detail, Mapping):
                return DatabaseNotePortLoadReply.failed(
                    "Note detail was incomplete. Press Retry."
                )

            get_keywords = getattr(self._notes_service, "get_keywords_for_note", None)
            if callable(get_keywords):
                keyword_records = await self._run_service_call(
                    get_keywords,
                    self._user_id,
                    note_id,
                    isolate_in_worker=True,
                )
            elif "keywords" in detail:
                keyword_records = detail["keywords"]
            else:
                return DatabaseNotePortLoadReply.failed(
                    "Keyword loading is unavailable. Press Retry."
                )
            keywords = self._keyword_strings(keyword_records)
            raw_note_id = detail.get("id", detail.get("note_id", note_id))
            raw_title = detail.get("title", "")
            raw_body = detail.get("content", detail.get("body", ""))
            created_at = detail.get("created_at", "")
            modified_at = detail.get(
                "last_modified", detail.get("updated_at", created_at)
            )
            normalized = NormalizedDatabaseNote(
                note_id=str(raw_note_id),
                title=raw_title if isinstance(raw_title, str) else str(raw_title or ""),
                body=raw_body if isinstance(raw_body, str) else str(raw_body or ""),
                keywords=keywords,
                version=int(detail.get("version", 0)),
                created_at=(
                    created_at if isinstance(created_at, str) else str(created_at or "")
                ),
                modified_at=(
                    modified_at
                    if isinstance(modified_at, str)
                    else str(modified_at or "")
                ),
            )
        except Exception as error:
            logger.opt(exception=True).warning(
                f"Failed to load Library note session for {note_id!r}."
            )
            failure_detail = str(error).strip()
            return DatabaseNotePortLoadReply.failed(
                (
                    f"Unable to load note — {failure_detail.rstrip('.')}. Press Retry."
                    if failure_detail
                    else "Unable to load note. Press Retry."
                )
            )
        return DatabaseNotePortLoadReply.loaded(normalized)

    async def save_note(
        self,
        note_id: str,
        expected_version: int,
        payload: DatabaseNoteSavePayload,
    ) -> DatabaseNotePortSaveReply:
        """Persist one exact versioned payload and normalize the service reply."""
        save_note = getattr(self._notes_scope_service, "save_note", None)
        if not callable(save_note):
            return DatabaseNotePortSaveReply.failed("Note saving is unavailable.")
        try:
            result = await self._run_service_call(
                save_note,
                scope="local_note",
                # task-3315: restore the LIB-14 save-seam fallback the
                # coordinator refactor (13cf08f90, notes-adaptive PR #1439)
                # silently dropped -- an emptied-out title persists as the
                # same "Untitled" default the create seam uses, never as a
                # blank row title (task-2858's reviewed decision).
                # (P0) The substitution rule lives in ONE helper now --
                # ``_patch_library_note_list_from_session`` applies the
                # same one to the snapshot, so the row name the session
                # reports and the row name on disk can no longer disagree.
                # (rebase note: task-4021's untouched-blank GC gate depends
                # on this same fallback -- both features now share the one
                # helper instead of two independently-typed literals.)
                title=library_note_persisted_title(payload.title),
                content=payload.body,
                note_id=note_id,
                version=expected_version,
                user_id=self._user_id,
                keywords=list(payload.keywords),
                isolate_in_worker=True,
            )
        except ConflictError:
            return DatabaseNotePortSaveReply.conflict()
        except Exception as error:
            logger.opt(exception=True).warning(
                f"Library note save failed for {note_id!r}."
            )
            return DatabaseNotePortSaveReply.failed(str(error))

        if result is False:
            return DatabaseNotePortSaveReply.conflict()
        modified_at = self._clock().isoformat()
        if result is True:
            return DatabaseNotePortSaveReply.saved(
                version=expected_version + 1,
                modified_at=modified_at,
                keywords=payload.keywords,
            )
        if isinstance(result, Mapping):
            try:
                version = int(result.get("version", expected_version + 1))
                keywords = (
                    self._keyword_strings(result["keywords"])
                    if "keywords" in result
                    else payload.keywords
                )
            except (TypeError, ValueError) as error:
                return DatabaseNotePortSaveReply.failed(
                    f"Save returned invalid note metadata: {error}"
                )
            returned_modified = result.get(
                "last_modified", result.get("updated_at", modified_at)
            )
            return DatabaseNotePortSaveReply.saved(
                version=version,
                modified_at=(
                    returned_modified
                    if isinstance(returned_modified, str)
                    else str(returned_modified or modified_at)
                ),
                keywords=keywords,
            )
        return DatabaseNotePortSaveReply.failed(
            "Save returned an unexpected result — edits kept. Press Save to retry."
        )
