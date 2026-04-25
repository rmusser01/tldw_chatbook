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


_LOCAL_GRAPH_UNSUPPORTED_CAPABILITY = {
    "operation_id": "notes.graph.local",
    "source": "local",
    "supported": False,
    "reason_code": "local_contract_missing",
    "user_message": "Local/offline notes graph generation and manual graph links are deferred; graph operations are server-backed today.",
    "affected_action_ids": [],
}

_WORKSPACE_GRAPH_UNSUPPORTED_CAPABILITY = {
    "operation_id": "notes.graph.workspace",
    "source": "workspace",
    "supported": False,
    "reason_code": "scope_not_supported",
    "user_message": "Workspace-scoped notes remain isolated from the global notes graph until sync/graph semantics are designed.",
    "affected_action_ids": [],
}


class NotesScopeService:
    """Route screen-facing note actions to the correct backing service."""

    def __init__(self, local_notes_service: Any, server_service: Any, policy_enforcer: Any = None):
        self.local_notes_service = local_notes_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

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

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    def _note_action_id(self, scope: ScopeType, action: str) -> str:
        suffix = {
            ScopeType.LOCAL_NOTE: "local",
            ScopeType.SERVER_NOTE: "server",
            ScopeType.WORKSPACE: "workspace",
        }[scope]
        return f"notes.{action}.{suffix}"

    @staticmethod
    def _graph_action_id(action: str) -> str:
        return f"notes.graph.{action}.server"

    def _require_server_graph_scope(self, scope: ScopeType | str) -> None:
        if self._normalize_scope(scope) != ScopeType.SERVER_NOTE:
            raise ValueError("Notes graph operations are currently server-backed.")

    def list_unsupported_capabilities(self, *, scope: ScopeType | str) -> list[dict[str, Any]]:
        normalized_scope = self._normalize_scope(scope)
        if normalized_scope == ScopeType.LOCAL_NOTE:
            return [dict(_LOCAL_GRAPH_UNSUPPORTED_CAPABILITY)]
        if normalized_scope == ScopeType.WORKSPACE:
            return [dict(_WORKSPACE_GRAPH_UNSUPPORTED_CAPABILITY)]
        return []

    @staticmethod
    def _normalize_keywords(keywords: Optional[Sequence[str]]) -> list[str]:
        if keywords is None:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in keywords:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    def _sync_local_note_keywords(
        self,
        *,
        user_id: str,
        note_id: Any,
        keywords: Optional[Sequence[str]],
    ) -> list[str]:
        normalized_keywords = self._normalize_keywords(keywords)
        service = self.local_notes_service
        required_methods = (
            "get_keywords_for_note",
            "get_keyword_by_text",
            "add_keyword",
            "link_note_to_keyword",
            "unlink_note_from_keyword",
        )
        if service is None or not all(hasattr(service, name) for name in required_methods):
            return normalized_keywords

        requested_keyword_map = {keyword.lower(): keyword for keyword in normalized_keywords}
        existing_keyword_rows = service.get_keywords_for_note(user_id, note_id) or []
        existing_keyword_map = {
            str(row.get("keyword", "")).strip().lower(): row.get("id")
            for row in existing_keyword_rows
            if row.get("id") is not None and str(row.get("keyword", "")).strip()
        }

        for keyword_key, keyword_text in requested_keyword_map.items():
            if keyword_key in existing_keyword_map:
                continue
            keyword_row = service.get_keyword_by_text(user_id, keyword_key)
            keyword_id = keyword_row.get("id") if isinstance(keyword_row, dict) else None
            if keyword_id is None:
                keyword_id = service.add_keyword(user_id, keyword_key)
            if keyword_id is not None:
                service.link_note_to_keyword(user_id, note_id, keyword_id)

        for existing_keyword_key, existing_keyword_id in existing_keyword_map.items():
            if existing_keyword_key in requested_keyword_map:
                continue
            service.unlink_note_from_keyword(user_id, note_id, existing_keyword_id)

        return normalized_keywords

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
        self._enforce_policy(
            self._note_action_id(
                normalized_scope,
                "update" if note_id else "create",
            )
        )
        if normalized_scope == ScopeType.LOCAL_NOTE:
            local_user_id = self._require_user_id(user_id)
            if note_id:
                updated = self.local_notes_service.update_note(
                    local_user_id,
                    note_id,
                    {"title": title, "content": content},
                    version,
                )
                if keywords is None:
                    return updated
                if not updated:
                    return False
                return {
                    "id": note_id,
                    "version": (version + 1) if version is not None else None,
                    "title": title,
                    "content": content,
                    "keywords": self._sync_local_note_keywords(
                        user_id=local_user_id,
                        note_id=note_id,
                        keywords=keywords,
                    ),
                }
            created_note_id = self.local_notes_service.add_note(
                local_user_id,
                title,
                content,
                note_id=note_id,
            )
            if keywords is None:
                return created_note_id
            return {
                "id": created_note_id,
                "version": 1,
                "title": title,
                "content": content,
                "keywords": self._sync_local_note_keywords(
                    user_id=local_user_id,
                    note_id=created_note_id,
                    keywords=keywords,
                ),
            }

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
        self._enforce_policy(self._note_action_id(normalized_scope, "delete"))
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
        self._enforce_policy(self._note_action_id(normalized_scope, "list"))
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

    async def list_notes(
        self,
        *,
        scope: ScopeType | str,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
    ) -> Any:
        normalized_scope = self._normalize_scope(scope)
        self._enforce_policy(self._note_action_id(normalized_scope, "list"))
        if normalized_scope == ScopeType.LOCAL_NOTE:
            return self.local_notes_service.list_notes(
                self._require_user_id(user_id),
                limit=limit,
            )
        if normalized_scope == ScopeType.SERVER_NOTE:
            return await self.server_service.list_server_notes(limit=limit, offset=offset)
        raise ValueError("Workspace notes require a selected workspace context.")

    async def list_workspaces(self) -> Any:
        self._enforce_policy("notes.workspace.list.server")
        return await self.server_service.list_workspaces()

    async def get_note_detail(
        self,
        *,
        scope: ScopeType | str,
        note_id: Any,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        workspace_notes: Optional[Sequence[dict[str, Any]]] = None,
    ) -> Any:
        normalized_scope = self._normalize_scope(scope)
        self._enforce_policy(self._note_action_id(normalized_scope, "detail"))
        if normalized_scope == ScopeType.LOCAL_NOTE:
            return self.local_notes_service.get_note_by_id(
                self._require_user_id(user_id),
                note_id,
            )
        if normalized_scope == ScopeType.SERVER_NOTE:
            return await self.server_service.get_server_note(str(note_id))

        resolved_workspace_id = self._require_workspace_id(workspace_id)
        notes = list(workspace_notes) if workspace_notes is not None else await self.server_service.list_workspace_notes(
            resolved_workspace_id
        )
        for note in notes:
            if str(note.get("id")) == str(note_id):
                return note
        return None

    async def load_workspace_context(
        self,
        *,
        scope: ScopeType | str,
        workspace_id: Optional[str],
    ) -> Any:
        normalized_scope = self._normalize_scope(scope)
        self._enforce_policy("notes.workspace.detail.server")
        if normalized_scope != ScopeType.WORKSPACE:
            raise ValueError("Workspace context can only be loaded for workspace scope.")
        return await self.server_service.load_workspace_context(
            self._require_workspace_id(workspace_id)
        )

    async def get_notes_graph(self, *, scope: ScopeType | str, **kwargs: Any) -> Any:
        self._require_server_graph_scope(scope)
        self._enforce_policy(self._graph_action_id("list"))
        return await self.server_service.get_notes_graph(**kwargs)

    async def get_note_neighbors(self, *, scope: ScopeType | str, note_id: str, **kwargs: Any) -> Any:
        self._require_server_graph_scope(scope)
        self._enforce_policy(self._graph_action_id("detail"))
        return await self.server_service.get_note_neighbors(note_id, **kwargs)

    async def create_note_link(self, *, scope: ScopeType | str, note_id: str, **kwargs: Any) -> Any:
        self._require_server_graph_scope(scope)
        self._enforce_policy(self._graph_action_id("create"))
        return await self.server_service.create_note_link(note_id, **kwargs)

    async def delete_note_link(self, *, scope: ScopeType | str, edge_id: str) -> Any:
        self._require_server_graph_scope(scope)
        self._enforce_policy(self._graph_action_id("delete"))
        return await self.server_service.delete_note_link(edge_id)
