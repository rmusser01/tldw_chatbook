"""
Service helpers for server-backed notes and workspace resources.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    NoteCreateRequest,
    NoteGraphRequest,
    NoteLinkCreateRequest,
    NoteUpdateRequest,
    TLDWAPIClient,
    WorkspaceArtifactCreateRequest,
    WorkspaceArtifactUpdateRequest,
    WorkspaceCreateRequest,
    WorkspaceNoteCreateRequest,
    WorkspaceNoteUpdateRequest,
    WorkspaceSourceCreateRequest,
    WorkspaceSourceUpdateRequest,
    WorkspaceUpdateRequest,
)

_UNSET = object()


class ServerNotesWorkspaceService:
    """Thin service around server-backed notes and workspace resources."""

    def __init__(self, client: Optional[TLDWAPIClient], policy_enforcer: Any = None):
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any], policy_enforcer: Any = None) -> "ServerNotesWorkspaceService":
        return cls(client=build_tldw_api_client_from_config(app_config), policy_enforcer=policy_enforcer)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server note and workspace operations.")
        return self.client

    def _coerce_items(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("items", "notes", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value if isinstance(item, Mapping)]
        return []

    def _coerce_resource(self, payload: Any, *keys: str) -> dict[str, Any]:
        if isinstance(payload, Mapping):
            for key in keys:
                value = payload.get(key)
                if isinstance(value, Mapping):
                    return dict(value)
            return dict(payload)
        return {}

    def _normalize_keywords(self, keywords: Any) -> list[str]:
        if keywords is None:
            return []
        if isinstance(keywords, str):
            return [part.strip() for part in keywords.split(",") if part.strip()]

        normalized: list[str] = []
        seen: set[str] = set()
        for item in keywords:
            value: Any = item
            if isinstance(item, Mapping):
                value = (
                    item.get("keyword")
                    or item.get("text")
                    or item.get("name")
                    or item.get("value")
                )
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(cleaned)
        return normalized

    def _normalize_workspace_keywords(self, keywords_json: Any) -> list[str]:
        if keywords_json in (None, ""):
            return []
        if isinstance(keywords_json, list):
            return self._normalize_keywords(keywords_json)
        if isinstance(keywords_json, str):
            try:
                decoded = json.loads(keywords_json)
            except (TypeError, ValueError):
                return []
            return self._normalize_keywords(decoded)
        return []

    def _with_optional_update_fields(self, **fields: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in fields.items():
            if value is _UNSET:
                continue
            payload[key] = value
        return payload

    def _require_value(self, value: Any, field_name: str) -> Any:
        if value is _UNSET or value is None:
            raise ValueError(f"{field_name} is required for create operations.")
        return value

    def normalize_server_note(self, note: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": note.get("id"),
            "title": note.get("title") or "",
            "content": note.get("content") or "",
            "keywords": self._normalize_keywords(note.get("keywords")),
            "version": int(note.get("version") or 1),
        }

    def normalize_workspace_note(self, note: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": note.get("id"),
            "workspace_id": note.get("workspace_id"),
            "title": note.get("title") or "",
            "content": note.get("content") or "",
            "keywords": self._normalize_workspace_keywords(note.get("keywords_json")),
            "version": int(note.get("version") or 1),
        }

    def normalize_workspace(self, workspace: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": workspace.get("id"),
            "name": workspace.get("name") or "",
            "archived": bool(workspace.get("archived", False)),
            "study_materials_policy": workspace.get("study_materials_policy") or "general",
            "audio_provider": workspace.get("audio_provider"),
            "audio_model": workspace.get("audio_model"),
            "audio_voice": workspace.get("audio_voice"),
            "audio_speed": workspace.get("audio_speed"),
            "version": int(workspace.get("version") or 1),
        }

    def normalize_workspace_source(self, source: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": source.get("id"),
            "workspace_id": source.get("workspace_id"),
            "media_id": source.get("media_id"),
            "title": source.get("title") or "",
            "source_type": source.get("source_type") or "",
            "url": source.get("url"),
            "position": int(source.get("position") or 0),
            "selected": bool(source.get("selected", True)),
            "version": int(source.get("version") or 1),
        }

    def normalize_workspace_artifact(self, artifact: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": artifact.get("id"),
            "workspace_id": artifact.get("workspace_id"),
            "artifact_type": artifact.get("artifact_type") or "",
            "title": artifact.get("title") or "",
            "status": artifact.get("status") or "pending",
            "content": artifact.get("content"),
            "version": int(artifact.get("version") or 1),
        }

    def build_server_note_create_payload(
        self,
        *,
        title: str,
        content: str,
        note_id: Optional[str] = None,
        keywords: Optional[Sequence[str]] = None,
    ) -> NoteCreateRequest:
        return NoteCreateRequest(
            id=note_id,
            title=title,
            content=content,
            keywords=self._normalize_keywords(list(keywords or [])),
        )

    def build_server_note_update_payload(
        self,
        *,
        title: Any = _UNSET,
        content: Any = _UNSET,
        keywords: Any = _UNSET,
    ) -> NoteUpdateRequest:
        payload = self._with_optional_update_fields(title=title, content=content)
        if keywords is not _UNSET and keywords is not None:
            payload["keywords"] = self._normalize_keywords(keywords)
        return NoteUpdateRequest(**payload)

    def build_workspace_create_payload(
        self,
        *,
        name: str,
        archived: bool = False,
        study_materials_policy: str = "general",
    ) -> WorkspaceCreateRequest:
        return WorkspaceCreateRequest(
            name=name,
            archived=archived,
            study_materials_policy=study_materials_policy,
        )

    def build_workspace_update_payload(
        self,
        *,
        version: int,
        name: Any = _UNSET,
        archived: Any = _UNSET,
        study_materials_policy: Any = _UNSET,
        banner_title: Any = _UNSET,
        banner_subtitle: Any = _UNSET,
        banner_color: Any = _UNSET,
        audio_provider: Any = _UNSET,
        audio_model: Any = _UNSET,
        audio_voice: Any = _UNSET,
        audio_speed: Any = _UNSET,
    ) -> WorkspaceUpdateRequest:
        return WorkspaceUpdateRequest(
            version=version,
            **self._with_optional_update_fields(
                name=name,
                archived=archived,
                study_materials_policy=study_materials_policy,
                banner_title=banner_title,
                banner_subtitle=banner_subtitle,
                banner_color=banner_color,
                audio_provider=audio_provider,
                audio_model=audio_model,
                audio_voice=audio_voice,
                audio_speed=audio_speed,
            ),
        )

    def build_workspace_note_create_payload(
        self,
        *,
        title: str,
        content: str,
        keywords: Optional[Sequence[str]] = None,
    ) -> WorkspaceNoteCreateRequest:
        return WorkspaceNoteCreateRequest(
            title=title,
            content=content,
            keywords=self._normalize_keywords(list(keywords or [])),
        )

    def build_workspace_note_update_payload(
        self,
        *,
        title: Any = _UNSET,
        content: Any = _UNSET,
        keywords: Any = _UNSET,
        version: int,
    ) -> WorkspaceNoteUpdateRequest:
        payload = self._with_optional_update_fields(title=title, content=content)
        if keywords is not _UNSET and keywords is not None:
            payload["keywords_json"] = json.dumps(self._normalize_keywords(keywords))
        return WorkspaceNoteUpdateRequest(version=version, **payload)

    def build_workspace_source_create_payload(
        self,
        *,
        source_id: str,
        media_id: int,
        title: str,
        source_type: str,
        url: Optional[str] = None,
        position: int = 0,
        selected: bool = True,
    ) -> WorkspaceSourceCreateRequest:
        return WorkspaceSourceCreateRequest(
            id=source_id,
            media_id=media_id,
            title=title,
            source_type=source_type,
            url=url,
            position=position,
            selected=selected,
        )

    def build_workspace_source_update_payload(
        self,
        *,
        version: int,
        title: Any = _UNSET,
        source_type: Any = _UNSET,
        url: Any = _UNSET,
        position: Any = _UNSET,
        selected: Any = _UNSET,
    ) -> WorkspaceSourceUpdateRequest:
        return WorkspaceSourceUpdateRequest(
            version=version,
            **self._with_optional_update_fields(
                title=title,
                source_type=source_type,
                url=url,
                position=position,
                selected=selected,
            ),
        )

    def build_workspace_artifact_create_payload(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        title: str,
        status: str = "pending",
        content: Optional[str] = None,
    ) -> WorkspaceArtifactCreateRequest:
        return WorkspaceArtifactCreateRequest(
            id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            status=status,
            content=content,
        )

    def build_workspace_artifact_update_payload(
        self,
        *,
        version: int,
        title: Any = _UNSET,
        status: Any = _UNSET,
        content: Any = _UNSET,
        total_tokens: Any = _UNSET,
        total_cost_usd: Any = _UNSET,
        completed_at: Any = _UNSET,
    ) -> WorkspaceArtifactUpdateRequest:
        return WorkspaceArtifactUpdateRequest(
            version=version,
            **self._with_optional_update_fields(
                title=title,
                status=status,
                content=content,
                total_tokens=total_tokens,
                total_cost_usd=total_cost_usd,
                completed_at=completed_at,
            ),
        )

    async def list_server_notes(self, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        client = self._require_client()
        response = await client.list_server_notes(limit=limit, offset=offset, include_keywords=True)
        items = [self.normalize_server_note(item) for item in self._coerce_items(response)]
        return {
            "items": items,
            "count": int(response.get("count", len(items))) if isinstance(response, Mapping) else len(items),
        }

    async def search_server_notes(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> dict[str, Any]:
        client = self._require_client()
        response = await client.search_server_notes(
            query=query,
            limit=limit,
            offset=offset,
            include_keywords=True,
        )
        items = [self.normalize_server_note(item) for item in self._coerce_items(response)]
        return {
            "items": items,
            "count": int(response.get("count", len(items))) if isinstance(response, Mapping) else len(items),
        }

    async def get_server_note(self, note_id: str) -> dict[str, Any]:
        client = self._require_client()
        response = await client.get_server_note(note_id)
        return self.normalize_server_note(self._coerce_resource(response, "note", "item"))

    async def save_server_note(
        self,
        *,
        title: Any = _UNSET,
        content: Any = _UNSET,
        note_id: Optional[str] = None,
        keywords: Any = _UNSET,
        version: Optional[int] = None,
    ) -> dict[str, Any]:
        client = self._require_client()
        if note_id:
            if version is None:
                raise ValueError("Version is required when updating a server note.")
            request = self.build_server_note_update_payload(
                title=title,
                content=content,
                keywords=keywords,
            )
            response = await client.update_server_note(
                note_id,
                request,
                expected_version=version,
            )
        else:
            request = self.build_server_note_create_payload(
                title=self._require_value(title, "title"),
                content=self._require_value(content, "content"),
                note_id=note_id,
                keywords=None if keywords is _UNSET else keywords,
            )
            response = await client.create_server_note(request)
        return self.normalize_server_note(self._coerce_resource(response, "note", "item"))

    async def delete_server_note(self, note_id: str, version: int) -> dict[str, Any]:
        client = self._require_client()
        return await client.delete_server_note(note_id, expected_version=version)

    def build_note_graph_request(
        self,
        *,
        center_note_id: Optional[str] = None,
        radius: int = 1,
        edge_types: Optional[Sequence[str]] = None,
        tag: Optional[str] = None,
        source: Optional[str] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_degree: Optional[int] = None,
        format: str = "default",
        cursor: Optional[str] = None,
        allow_heavy: bool = False,
    ) -> NoteGraphRequest:
        return NoteGraphRequest(
            center_note_id=center_note_id,
            radius=radius,
            edge_types=list(edge_types) if edge_types is not None else None,
            tag=tag,
            source=source,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_degree=max_degree,
            format=format,
            cursor=cursor,
            allow_heavy=allow_heavy,
        )

    def build_note_link_create_payload(
        self,
        *,
        to_note_id: str,
        directed: bool = False,
        weight: Optional[float] = 1.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> NoteLinkCreateRequest:
        return NoteLinkCreateRequest(
            to_note_id=to_note_id,
            directed=directed,
            weight=weight,
            metadata=dict(metadata) if metadata is not None else None,
        )

    async def get_notes_graph(self, **kwargs: Any) -> dict[str, Any]:
        client = self._require_client()
        request = self.build_note_graph_request(**kwargs)
        response = await client.get_notes_graph(request)
        return dict(response) if isinstance(response, Mapping) else response

    async def get_note_neighbors(self, note_id: str, **kwargs: Any) -> dict[str, Any]:
        client = self._require_client()
        request = self.build_note_graph_request(**kwargs)
        response = await client.get_note_neighbors(note_id, request)
        return dict(response) if isinstance(response, Mapping) else response

    async def create_note_link(
        self,
        note_id: str,
        *,
        to_note_id: str,
        directed: bool = False,
        weight: Optional[float] = 1.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        client = self._require_client()
        request = self.build_note_link_create_payload(
            to_note_id=to_note_id,
            directed=directed,
            weight=weight,
            metadata=metadata,
        )
        response = await client.create_note_link(note_id, request)
        return dict(response) if isinstance(response, Mapping) else response

    async def delete_note_link(self, edge_id: str) -> dict[str, Any]:
        client = self._require_client()
        response = await client.delete_note_link(edge_id)
        return dict(response) if isinstance(response, Mapping) else response

    async def list_workspaces(self) -> list[dict[str, Any]]:
        client = self._require_client()
        response = await client.list_workspaces()
        return [self.normalize_workspace(item) for item in self._coerce_items(response)]

    async def save_workspace(
        self,
        *,
        workspace_id: str,
        name: Any = _UNSET,
        version: Optional[int] = None,
        archived: Any = _UNSET,
        study_materials_policy: Any = _UNSET,
        banner_title: Any = _UNSET,
        banner_subtitle: Any = _UNSET,
        banner_color: Any = _UNSET,
        audio_provider: Any = _UNSET,
        audio_model: Any = _UNSET,
        audio_voice: Any = _UNSET,
        audio_speed: Any = _UNSET,
    ) -> dict[str, Any]:
        client = self._require_client()
        if version is None:
            request = self.build_workspace_create_payload(
                name=self._require_value(name, "name"),
                archived=False if archived is _UNSET else bool(archived),
                study_materials_policy="general" if study_materials_policy in {_UNSET, None} else study_materials_policy,
            )
            response = await client.create_workspace(workspace_id, request)
        else:
            request = self.build_workspace_update_payload(
                name=name,
                archived=archived,
                study_materials_policy=study_materials_policy,
                banner_title=banner_title,
                banner_subtitle=banner_subtitle,
                banner_color=banner_color,
                audio_provider=audio_provider,
                audio_model=audio_model,
                audio_voice=audio_voice,
                audio_speed=audio_speed,
                version=version,
            )
            response = await client.update_workspace(workspace_id, request)
        return self.normalize_workspace(self._coerce_resource(response, "workspace", "item"))

    async def delete_workspace(self, workspace_id: str) -> dict[str, Any]:
        client = self._require_client()
        return await client.delete_workspace(workspace_id)

    def filter_workspace_notes(
        self,
        notes: Sequence[Mapping[str, Any]],
        *,
        workspace_id: str,
        query: str = "",
    ) -> list[dict[str, Any]]:
        lowered = query.strip().lower()
        filtered: list[dict[str, Any]] = []
        for item in notes:
            if item.get("workspace_id") != workspace_id:
                continue
            normalized = self.normalize_workspace_note(item)
            if not lowered:
                filtered.append(normalized)
                continue
            haystack = " ".join(
                [
                    normalized["title"],
                    normalized["content"],
                    " ".join(normalized["keywords"]),
                ]
            ).lower()
            if lowered in haystack:
                filtered.append(normalized)
        return filtered

    async def search_workspace_notes(
        self,
        workspace_id: str,
        query: str,
        notes: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        working_notes = list(notes) if notes is not None else (await self.list_workspace_notes(workspace_id))
        return self.filter_workspace_notes(working_notes, workspace_id=workspace_id, query=query)

    async def list_workspace_notes(self, workspace_id: str) -> list[dict[str, Any]]:
        client = self._require_client()
        response = await client.list_workspace_notes(workspace_id)
        return [self.normalize_workspace_note(item) for item in self._coerce_items(response)]

    async def save_workspace_note(
        self,
        *,
        workspace_id: str,
        title: Any = _UNSET,
        content: Any = _UNSET,
        note_id: Optional[int] = None,
        keywords: Any = _UNSET,
        version: Optional[int] = None,
    ) -> dict[str, Any]:
        client = self._require_client()
        if note_id is None:
            request = self.build_workspace_note_create_payload(
                title=self._require_value(title, "title"),
                content=self._require_value(content, "content"),
                keywords=None if keywords is _UNSET else keywords,
            )
            response = await client.create_workspace_note(workspace_id, request)
        else:
            if version is None:
                raise ValueError("Version is required when updating a workspace note.")
            request = self.build_workspace_note_update_payload(
                title=title,
                content=content,
                keywords=keywords,
                version=version,
            )
            response = await client.update_workspace_note(workspace_id, note_id, request)
        return self.normalize_workspace_note(self._coerce_resource(response, "note", "item"))

    async def delete_workspace_note(
        self,
        workspace_id: str,
        note_id: int,
        version: int,
    ) -> dict[str, Any]:
        client = self._require_client()
        return await client.delete_workspace_note(workspace_id, note_id)

    async def list_workspace_sources(self, workspace_id: str) -> list[dict[str, Any]]:
        client = self._require_client()
        response = await client.list_workspace_sources(workspace_id)
        return [self.normalize_workspace_source(item) for item in self._coerce_items(response)]

    async def save_workspace_source(
        self,
        *,
        workspace_id: str,
        source_id: str,
        media_id: Any = _UNSET,
        title: Any = _UNSET,
        source_type: Any = _UNSET,
        version: Optional[int] = None,
        url: Any = _UNSET,
        position: Any = _UNSET,
        selected: Any = _UNSET,
    ) -> dict[str, Any]:
        client = self._require_client()
        if version is None:
            request = self.build_workspace_source_create_payload(
                source_id=source_id,
                media_id=self._require_value(media_id, "media_id"),
                title=self._require_value(title, "title"),
                source_type=self._require_value(source_type, "source_type"),
                url=None if url is _UNSET else url,
                position=0 if position is _UNSET else position,
                selected=True if selected is _UNSET else selected,
            )
            response = await client.create_workspace_source(workspace_id, request)
        else:
            request = self.build_workspace_source_update_payload(
                title=title,
                source_type=source_type,
                url=url,
                position=position,
                selected=selected,
                version=version,
            )
            response = await client.update_workspace_source(workspace_id, source_id, request)
        return self.normalize_workspace_source(self._coerce_resource(response, "source", "item"))

    async def delete_workspace_source(self, workspace_id: str, source_id: str) -> dict[str, Any]:
        client = self._require_client()
        return await client.delete_workspace_source(workspace_id, source_id)

    async def list_workspace_artifacts(self, workspace_id: str) -> list[dict[str, Any]]:
        client = self._require_client()
        response = await client.list_workspace_artifacts(workspace_id)
        return [self.normalize_workspace_artifact(item) for item in self._coerce_items(response)]

    async def save_workspace_artifact(
        self,
        *,
        workspace_id: str,
        artifact_id: str,
        artifact_type: Any = _UNSET,
        title: Any = _UNSET,
        version: Optional[int] = None,
        status: Any = _UNSET,
        content: Any = _UNSET,
    ) -> dict[str, Any]:
        client = self._require_client()
        if version is None:
            request = self.build_workspace_artifact_create_payload(
                artifact_id=artifact_id,
                artifact_type=self._require_value(artifact_type, "artifact_type"),
                title=self._require_value(title, "title"),
                status="pending" if status in {_UNSET, None} else status,
                content=None if content is _UNSET else content,
            )
            response = await client.create_workspace_artifact(workspace_id, request)
        else:
            request = self.build_workspace_artifact_update_payload(
                title=title,
                status=status,
                content=content,
                version=version,
            )
            response = await client.update_workspace_artifact(workspace_id, artifact_id, request)
        return self.normalize_workspace_artifact(self._coerce_resource(response, "artifact", "item"))

    async def delete_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, Any]:
        client = self._require_client()
        return await client.delete_workspace_artifact(workspace_id, artifact_id)

    async def load_workspace_context(self, workspace_id: str) -> dict[str, Any]:
        client = self._require_client()
        workspace = self.normalize_workspace(
            self._coerce_resource(await client.get_workspace(workspace_id), "workspace", "item")
        )
        notes = [self.normalize_workspace_note(item) for item in self._coerce_items(await client.list_workspace_notes(workspace_id))]
        sources = [self.normalize_workspace_source(item) for item in self._coerce_items(await client.list_workspace_sources(workspace_id))]
        artifacts = [self.normalize_workspace_artifact(item) for item in self._coerce_items(await client.list_workspace_artifacts(workspace_id))]
        return {
            "workspace": workspace,
            "notes": notes,
            "sources": sources,
            "artifacts": artifacts,
        }
