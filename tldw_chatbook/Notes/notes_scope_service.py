"""
Scope-aware routing for local notes, server notes, and workspace notes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Sequence


class ScopeType(str, Enum):
    LOCAL_NOTE = "local_note"
    SERVER_NOTE = "server_note"
    WORKSPACE = "workspace"


class NotesScopeService:
    """Route screen-facing note actions to the correct backing service."""

    def __init__(self, local_notes_service: Any, server_service: Any):
        self.local_notes_service = local_notes_service
        self.server_service = server_service

    def _normalize_scope(self, scope: ScopeType | str) -> ScopeType:
        if isinstance(scope, ScopeType):
            return scope
        return ScopeType(str(scope))

    def _require_user_id(self, user_id: Optional[str]) -> str:
        if not user_id:
            raise ValueError("user_id is required for local note operations.")
        return user_id

    def _require_workspace_id(self, workspace_id: Optional[str]) -> str:
        if not workspace_id:
            raise ValueError("workspace_id is required for workspace note operations.")
        return workspace_id

    async def save_note(
        self,
        *,
        scope: ScopeType | str,
        title: str,
        content: str,
        note_id: Any = None,
        version: Optional[int] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        keywords: Optional[Sequence[str]] = None,
    ) -> Any:
        normalized_scope = self._normalize_scope(scope)
        if normalized_scope == ScopeType.LOCAL_NOTE:
            local_user_id = self._require_user_id(user_id)
            if note_id:
                return self.local_notes_service.update_note(
                    local_user_id,
                    note_id,
                    {"title": title, "content": content},
                    version,
                )
            return self.local_notes_service.add_note(
                local_user_id,
                title,
                content,
                note_id=note_id,
            )

        if normalized_scope == ScopeType.SERVER_NOTE:
            return await self.server_service.save_server_note(
                note_id=note_id,
                title=title,
                content=content,
                keywords=keywords,
                version=version,
            )

        return await self.server_service.save_workspace_note(
            workspace_id=self._require_workspace_id(workspace_id),
            note_id=note_id,
            title=title,
            content=content,
            keywords=keywords,
            version=version,
        )

    async def delete_note(
        self,
        *,
        scope: ScopeType | str,
        note_id: Any,
        version: int,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Any:
        normalized_scope = self._normalize_scope(scope)
        if normalized_scope == ScopeType.LOCAL_NOTE:
            return self.local_notes_service.soft_delete_note(
                self._require_user_id(user_id),
                note_id,
                version,
            )
        if normalized_scope == ScopeType.SERVER_NOTE:
            return await self.server_service.delete_server_note(note_id, version)
        return await self.server_service.delete_workspace_note(
            self._require_workspace_id(workspace_id),
            note_id,
            version,
        )

    async def search_notes(
        self,
        *,
        scope: ScopeType | str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        workspace_notes: Optional[Sequence[dict[str, Any]]] = None,
    ) -> Any:
        normalized_scope = self._normalize_scope(scope)
        if normalized_scope == ScopeType.LOCAL_NOTE:
            return self.local_notes_service.search_notes(
                self._require_user_id(user_id),
                query,
                limit=limit,
            )
        if normalized_scope == ScopeType.SERVER_NOTE:
            return await self.server_service.search_server_notes(
                query=query,
                limit=limit,
                offset=offset,
            )
        return await self.server_service.search_workspace_notes(
            self._require_workspace_id(workspace_id),
            query,
            notes=workspace_notes,
        )

    async def load_workspace_context(
        self,
        *,
        scope: ScopeType | str,
        workspace_id: Optional[str],
    ) -> Any:
        normalized_scope = self._normalize_scope(scope)
        if normalized_scope != ScopeType.WORKSPACE:
            raise ValueError("Workspace context can only be loaded for workspace scope.")
        return await self.server_service.load_workspace_context(
            self._require_workspace_id(workspace_id)
        )
