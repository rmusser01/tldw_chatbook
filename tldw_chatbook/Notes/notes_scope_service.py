"""
Scope-aware routing for local notes, server notes, and workspace notes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class ScopeType(str, Enum):
    LOCAL_NOTE = "local_note"
    SERVER_NOTE = "server_note"
    WORKSPACE = "workspace"


_SERVER_GRAPH_ACTION_IDS = [
    "notes.graph.list.server",
    "notes.graph.detail.server",
    "notes.graph.create.server",
    "notes.graph.delete.server",
]

_LOCAL_GRAPH_UNSUPPORTED_CAPABILITY = {
    "operation_id": "notes.graph.local",
    "source": "local",
    "supported": False,
    "reason_code": "local_contract_missing",
    "user_message": "Local/offline notes graph generation and manual graph links are deferred; graph operations are server-backed today.",
    "affected_action_ids": list(_SERVER_GRAPH_ACTION_IDS),
}

_WORKSPACE_GRAPH_UNSUPPORTED_CAPABILITY = {
    "operation_id": "notes.graph.workspace",
    "source": "workspace",
    "supported": False,
    "reason_code": "scope_not_supported",
    "user_message": "Workspace-scoped notes remain isolated from the global notes graph until sync/graph semantics are designed.",
    "affected_action_ids": list(_SERVER_GRAPH_ACTION_IDS),
}


class NotesScopeService:
    """Route screen-facing note actions to the correct backing service."""

    def __init__(
        self,
        local_notes_service: Any,
        server_service: Any,
        policy_enforcer: Any = None,
        sync_scope_service: Any = None,
    ):
        self.local_notes_service = local_notes_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer
        self.sync_scope_service = sync_scope_service

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

    def _require_sync_scope_service(self) -> Any:
        if self.sync_scope_service is None:
            raise ValueError("Sync scope service is unavailable.")
        return self.sync_scope_service

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

    @staticmethod
    def _workspace_action_id(action: str) -> str:
        return f"notes.workspace.{action}.server"

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

    def record_sync_mirror_report(
        self,
        *,
        scope: ScopeType | str,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_id: str | None = None,
        local_records: list[dict[str, Any]] | None = None,
        remote_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_scope = self._normalize_scope(scope)
        if normalized_scope == ScopeType.LOCAL_NOTE:
            raise ValueError("Local note mirror reports require a server or workspace scope.")
        if normalized_scope == ScopeType.WORKSPACE:
            workspace_id = self._require_workspace_id(workspace_id)
            domain = "workspace_notes"
            entity_type = "workspace_note"
        else:
            domain = "notes"
            entity_type = "note"
            workspace_id = None

        return self._require_sync_scope_service().record_dry_run_mirror_report(
            mode="server",
            domain=domain,
            entity_type=entity_type,
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_id,
            local_records=local_records or [],
            remote_records=remote_records or [],
        )

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

    @staticmethod
    def _note_id(note: Mapping[str, Any]) -> str:
        return str(note.get("id") or note.get("uuid") or "")

    @staticmethod
    def _keyword_id(keyword: Mapping[str, Any]) -> str:
        value = keyword.get("id")
        if value is None:
            value = keyword.get("uuid") or keyword.get("keyword") or keyword.get("text")
        return str(value)

    @staticmethod
    def _keyword_label(keyword: Mapping[str, Any]) -> str:
        return str(keyword.get("keyword") or keyword.get("text") or keyword.get("name") or "")

    def _build_local_notes_graph(
        self,
        *,
        user_id: str,
        center_note_id: Optional[str] = None,
        edge_types: Optional[Sequence[str]] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_degree: Optional[int] = None,
        **_: Any,
    ) -> dict[str, Any]:
        service = self.local_notes_service
        node_limit = max(1, int(max_nodes or 50))
        edge_limit = max(0, int(200 if max_edges is None else max_edges))
        degree_limit = max(1, int(max_degree or 50))
        edge_type_filter = {str(edge_type) for edge_type in edge_types or []}
        include_tag_edges = not edge_type_filter or "tag_membership" in edge_type_filter
        include_manual_edges = not edge_type_filter or "manual" in edge_type_filter

        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        truncated_by: set[str] = set()

        def add_node(node: dict[str, Any]) -> bool:
            node_id = str(node.get("id") or "")
            if not node_id:
                return False
            if node_id in nodes:
                return True
            if len(nodes) >= node_limit:
                truncated_by.add("max_nodes")
                return False
            nodes[node_id] = node
            return True

        def add_note_node(note: Mapping[str, Any]) -> bool:
            note_id = self._note_id(note)
            return add_node(
                {
                    "id": note_id,
                    "type": "note",
                    "label": str(note.get("title") or note_id),
                    "deleted": bool(note.get("deleted", False)),
                    "degree": 0,
                }
            )

        def ensure_note_node(note_id: str) -> bool:
            if note_id in nodes:
                return True
            note = service.get_note_by_id(user_id, note_id)
            if not isinstance(note, Mapping):
                return False
            return add_note_node(note)

        def add_tag_node(keyword: Mapping[str, Any]) -> str | None:
            keyword_id = self._keyword_id(keyword)
            label = self._keyword_label(keyword)
            if not keyword_id or not label:
                return None
            tag_node_id = f"tag:{keyword_id}"
            added = add_node(
                {
                    "id": tag_node_id,
                    "type": "tag",
                    "label": label,
                    "degree": 0,
                    "tag_count": 0,
                }
            )
            return tag_node_id if added else None

        def add_edge(edge: dict[str, Any]) -> None:
            edge_id = str(edge.get("id") or "")
            if not edge_id or edge_id in edges:
                return
            if len(edges) >= edge_limit:
                truncated_by.add("max_edges")
                return
            edges[edge_id] = edge
            for endpoint in (edge.get("source"), edge.get("target")):
                if endpoint in nodes:
                    nodes[str(endpoint)]["degree"] = int(nodes[str(endpoint)].get("degree") or 0) + 1
                    if nodes[str(endpoint)].get("type") == "tag":
                        nodes[str(endpoint)]["tag_count"] = int(nodes[str(endpoint)].get("tag_count") or 0) + 1

        if center_note_id:
            center_note = service.get_note_by_id(user_id, center_note_id)
            seed_notes = [center_note] if isinstance(center_note, Mapping) else []
        elif hasattr(service, "list_notes"):
            seed_notes = list(service.list_notes(user_id, limit=node_limit, offset=0) or [])
        else:
            seed_notes = []

        for note in seed_notes:
            if not isinstance(note, Mapping):
                continue
            note_id = self._note_id(note)
            if not add_note_node(note) or not include_tag_edges:
                continue
            keywords = list(service.get_keywords_for_note(user_id, note_id) or [])
            for keyword in keywords:
                if not isinstance(keyword, Mapping):
                    continue
                keyword_id = self._keyword_id(keyword)
                tag_node_id = add_tag_node(keyword)
                if not tag_node_id:
                    continue
                add_edge(
                    {
                        "id": f"local:tag_membership:{note_id}:{keyword_id}",
                        "source": note_id,
                        "target": tag_node_id,
                        "type": "tag_membership",
                        "directed": False,
                        "weight": 1.0,
                        "label": self._keyword_label(keyword),
                    }
                )
                if not center_note_id or not hasattr(service, "get_notes_for_keyword"):
                    continue
                related_notes = service.get_notes_for_keyword(
                    user_id,
                    keyword.get("id"),
                    limit=degree_limit,
                    offset=0,
                )
                for related_note in list(related_notes or []):
                    if not isinstance(related_note, Mapping):
                        continue
                    related_note_id = self._note_id(related_note)
                    if not add_note_node(related_note):
                        continue
                    add_edge(
                        {
                            "id": f"local:tag_membership:{related_note_id}:{keyword_id}",
                            "source": related_note_id,
                            "target": tag_node_id,
                            "type": "tag_membership",
                            "directed": False,
                            "weight": 1.0,
                            "label": self._keyword_label(keyword),
                        }
                    )

        if include_manual_edges and hasattr(service, "list_note_links"):
            manual_links = service.list_note_links(
                user_id,
                center_note_id=center_note_id,
                limit=edge_limit,
            )
            for manual_link in list(manual_links or []):
                if not isinstance(manual_link, Mapping):
                    continue
                source = str(manual_link.get("source") or "")
                target = str(manual_link.get("target") or "")
                if not source or not target:
                    continue
                if not ensure_note_node(source) or not ensure_note_node(target):
                    continue
                add_edge(
                    {
                        "id": str(manual_link.get("id") or f"local:manual:{source}:{target}"),
                        "source": source,
                        "target": target,
                        "type": "manual",
                        "directed": bool(manual_link.get("directed", False)),
                        "weight": float(manual_link.get("weight", 1.0)),
                        "label": str((manual_link.get("metadata") or {}).get("label") or "Manual link"),
                        "metadata": dict(manual_link.get("metadata") or {}),
                    }
                )

        return {
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "truncated": bool(truncated_by),
            "truncated_by": sorted(truncated_by),
            "has_more": bool(truncated_by),
            "cursor": None,
            "limits": {
                "max_nodes": node_limit,
                "max_edges": edge_limit,
                "max_degree": degree_limit,
            },
            "radius_cap_applied": False,
        }

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
        self._enforce_policy(self._workspace_action_id("list"))
        return await self.server_service.list_workspaces()

    async def save_workspace(
        self,
        *,
        workspace_id: Optional[str],
        version: Optional[int] = None,
        **fields: Any,
    ) -> Any:
        self._enforce_policy(
            self._workspace_action_id("update" if version is not None else "create")
        )
        payload = dict(fields)
        if version is not None:
            payload["version"] = version
        return await self.server_service.save_workspace(
            workspace_id=self._require_workspace_id(workspace_id),
            **payload,
        )

    async def delete_workspace(self, *, workspace_id: Optional[str]) -> Any:
        self._enforce_policy(self._workspace_action_id("delete"))
        return await self.server_service.delete_workspace(
            self._require_workspace_id(workspace_id)
        )

    async def list_workspace_sources(self, *, workspace_id: Optional[str]) -> Any:
        self._enforce_policy(self._workspace_action_id("detail"))
        return await self.server_service.list_workspace_sources(
            self._require_workspace_id(workspace_id)
        )

    async def save_workspace_source(
        self,
        *,
        workspace_id: Optional[str],
        source_id: str,
        version: Optional[int] = None,
        **fields: Any,
    ) -> Any:
        self._enforce_policy(self._workspace_action_id("update"))
        payload = dict(fields)
        if version is not None:
            payload["version"] = version
        return await self.server_service.save_workspace_source(
            workspace_id=self._require_workspace_id(workspace_id),
            source_id=source_id,
            **payload,
        )

    async def delete_workspace_source(
        self,
        *,
        workspace_id: Optional[str],
        source_id: str,
    ) -> Any:
        self._enforce_policy(self._workspace_action_id("update"))
        return await self.server_service.delete_workspace_source(
            self._require_workspace_id(workspace_id),
            source_id,
        )

    async def list_workspace_artifacts(self, *, workspace_id: Optional[str]) -> Any:
        self._enforce_policy(self._workspace_action_id("detail"))
        return await self.server_service.list_workspace_artifacts(
            self._require_workspace_id(workspace_id)
        )

    async def save_workspace_artifact(
        self,
        *,
        workspace_id: Optional[str],
        artifact_id: str,
        version: Optional[int] = None,
        **fields: Any,
    ) -> Any:
        self._enforce_policy(self._workspace_action_id("update"))
        payload = dict(fields)
        if version is not None:
            payload["version"] = version
        return await self.server_service.save_workspace_artifact(
            workspace_id=self._require_workspace_id(workspace_id),
            artifact_id=artifact_id,
            **payload,
        )

    async def delete_workspace_artifact(
        self,
        *,
        workspace_id: Optional[str],
        artifact_id: str,
    ) -> Any:
        self._enforce_policy(self._workspace_action_id("update"))
        return await self.server_service.delete_workspace_artifact(
            self._require_workspace_id(workspace_id),
            artifact_id,
        )

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
