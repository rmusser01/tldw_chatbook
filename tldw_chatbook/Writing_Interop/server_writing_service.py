"""Server-backed writing-suite service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    ManuscriptAnalysisRequest,
    ManuscriptCharacterCreate,
    ManuscriptCharacterUpdate,
    ManuscriptChapterCreate,
    ManuscriptChapterUpdate,
    ManuscriptCitationCreate,
    ManuscriptPartCreate,
    ManuscriptPartUpdate,
    ManuscriptPlotEventCreate,
    ManuscriptPlotEventUpdate,
    ManuscriptPlotHoleCreate,
    ManuscriptPlotHoleUpdate,
    ManuscriptPlotLineCreate,
    ManuscriptPlotLineUpdate,
    ManuscriptProjectCreate,
    ManuscriptProjectUpdate,
    ManuscriptRelationshipCreate,
    ManuscriptResearchRequest,
    ManuscriptSceneCreate,
    ManuscriptSceneUpdate,
    ManuscriptWorldInfoCreate,
    ManuscriptWorldInfoUpdate,
    ReorderRequest,
    SceneCharacterLink,
    SceneWorldInfoLink,
    TLDWAPIClient,
)
from .writing_markdown_adapter import markdown_to_plain_text, markdown_to_server_content
from .writing_normalizers import normalize_writing_record, normalize_writing_structure


_UNSET = object()

REASON_DIRECT_MANUSCRIPT_SCENE = (
    "Direct manuscript-level scenes are not exposed by the current server writing contract."
)
REASON_VERSION_HISTORY = "Server writing version history is not exposed by the current server writing contract."
REASON_TRASH_RESTORE = "Server writing trash listing and restore are not exposed by the current server writing contract."
REASON_SCENE_REPARENT = "Server scene reparenting is not exposed by the current server writing contract."


class ServerWritingService:
    """Thin wrapper that maps Chatbook writing terms onto server manuscript endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient], *, policy_enforcer: Any | None = None):
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerWritingService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server writing operations.")
        return self.client

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server writing action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _action_id(resource: str, action: str) -> str:
        return f"writing.{resource}.{action}.server"

    @staticmethod
    def _model_to_dict(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    async def create_project(self, *, title: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("projects", "create"))
        response = await self._require_client().create_manuscript_project(
            ManuscriptProjectCreate(title=title, **kwargs)
        )
        return normalize_writing_record("server", "project", self._model_to_dict(response))

    async def list_projects(self, *, limit: int = 100, offset: int = 0, status: str | None = None) -> list[dict[str, Any]]:
        self._enforce(self._action_id("projects", "list"))
        response = await self._require_client().list_manuscript_projects(status=status, limit=limit, offset=offset)
        payload = self._model_to_dict(response)
        return [
            normalize_writing_record("server", "project", item)
            for item in list(payload.get("projects", []))
        ]

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        self._enforce(self._action_id("projects", "detail"))
        response = await self._require_client().get_manuscript_project(project_id)
        return normalize_writing_record("server", "project", self._model_to_dict(response))

    async def update_project(
        self,
        project_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("projects", "update"))
        response = await self._require_client().update_manuscript_project(
            project_id,
            ManuscriptProjectUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "project", self._model_to_dict(response))

    async def delete_project(self, project_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("projects", "delete"))
        return await self._require_client().delete_manuscript_project(
            project_id,
            expected_version=expected_version,
        )

    async def create_manuscript(self, project_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("manuscripts", "create"))
        response = await self._require_client().create_manuscript(
            project_id,
            ManuscriptPartCreate(title=title, **kwargs),
        )
        return normalize_writing_record("server", "manuscript", self._model_to_dict(response))

    async def list_manuscripts(self, project_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("manuscripts", "list"))
        response = await self._require_client().list_manuscripts(project_id)
        return [
            normalize_writing_record("server", "manuscript", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def get_manuscript(self, manuscript_id: str) -> dict[str, Any] | None:
        self._enforce(self._action_id("manuscripts", "detail"))
        response = await self._require_client().get_manuscript(manuscript_id)
        return normalize_writing_record("server", "manuscript", self._model_to_dict(response))

    async def update_manuscript(
        self,
        manuscript_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("manuscripts", "update"))
        response = await self._require_client().update_manuscript(
            manuscript_id,
            ManuscriptPartUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "manuscript", self._model_to_dict(response))

    async def delete_manuscript(self, manuscript_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("manuscripts", "delete"))
        return await self._require_client().delete_manuscript(
            manuscript_id,
            expected_version=expected_version,
        )

    async def create_chapter(
        self,
        project_id: str,
        *,
        title: str,
        manuscript_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("chapters", "create"))
        response = await self._require_client().create_manuscript_chapter(
            project_id,
            ManuscriptChapterCreate(title=title, part_id=manuscript_id, **kwargs),
        )
        return normalize_writing_record("server", "chapter", self._model_to_dict(response))

    async def list_chapters(self, project_id: str, manuscript_id: str | None = None) -> list[dict[str, Any]]:
        self._enforce(self._action_id("chapters", "list"))
        response = await self._require_client().list_manuscript_chapters(project_id, part_id=manuscript_id)
        return [
            normalize_writing_record("server", "chapter", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def get_chapter(self, chapter_id: str) -> dict[str, Any] | None:
        self._enforce(self._action_id("chapters", "detail"))
        response = await self._require_client().get_manuscript_chapter(chapter_id)
        return normalize_writing_record("server", "chapter", self._model_to_dict(response))

    async def update_chapter(
        self,
        chapter_id: str,
        *,
        expected_version: int,
        manuscript_id: Any = _UNSET,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("chapters", "update"))
        if manuscript_id is not _UNSET:
            fields["part_id"] = manuscript_id
        response = await self._require_client().update_manuscript_chapter(
            chapter_id,
            ManuscriptChapterUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "chapter", self._model_to_dict(response))

    async def delete_chapter(self, chapter_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("chapters", "delete"))
        return await self._require_client().delete_manuscript_chapter(
            chapter_id,
            expected_version=expected_version,
        )

    async def create_scene(
        self,
        chapter_id: str | None,
        *,
        title: str,
        content_markdown: str = "",
        manuscript_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("scenes", "create"))
        if chapter_id is None or manuscript_id is not None:
            raise NotImplementedError(REASON_DIRECT_MANUSCRIPT_SCENE)
        response = await self._require_client().create_manuscript_scene(
            chapter_id,
            ManuscriptSceneCreate(
                title=title,
                content=markdown_to_server_content(content_markdown),
                content_plain=markdown_to_plain_text(content_markdown),
                **kwargs,
            ),
        )
        return normalize_writing_record("server", "scene", self._model_to_dict(response))

    async def list_scenes(
        self,
        chapter_id: str | None,
        *,
        manuscript_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce(self._action_id("scenes", "list"))
        if chapter_id is None or manuscript_id is not None:
            raise NotImplementedError(REASON_DIRECT_MANUSCRIPT_SCENE)
        response = await self._require_client().list_manuscript_scenes(chapter_id)
        return [
            normalize_writing_record("server", "scene", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        self._enforce(self._action_id("scenes", "detail"))
        response = await self._require_client().get_manuscript_scene(scene_id)
        return normalize_writing_record("server", "scene", self._model_to_dict(response))

    async def update_scene(
        self,
        scene_id: str,
        *,
        expected_version: int,
        content_markdown: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("scenes", "update"))
        if content_markdown is not None:
            fields["content"] = markdown_to_server_content(content_markdown)
            fields["content_plain"] = markdown_to_plain_text(content_markdown)
        response = await self._require_client().update_manuscript_scene(
            scene_id,
            ManuscriptSceneUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "scene", self._model_to_dict(response))

    async def delete_scene(self, scene_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("scenes", "delete"))
        return await self._require_client().delete_manuscript_scene(
            scene_id,
            expected_version=expected_version,
        )

    async def get_structure(self, project_id: str) -> dict[str, Any]:
        self._enforce(self._action_id("projects", "structure"))
        response = await self._require_client().get_manuscript_structure(project_id)
        return normalize_writing_structure("server", self._model_to_dict(response))

    async def reorder_entities(self, project_id: str, entity_type: str, items: list[dict[str, Any]]) -> bool:
        self._enforce(self._action_id("outline", "reorder"))
        server_entity_type = "parts" if entity_type == "manuscripts" else entity_type
        response = await self._require_client().reorder_manuscript_entities(
            project_id,
            ReorderRequest(entity_type=server_entity_type, items=items),
        )
        return bool(response)

    async def create_character(self, project_id: str, *, name: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("characters", "create"))
        response = await self._require_client().create_manuscript_character(
            project_id,
            ManuscriptCharacterCreate(name=name, **kwargs),
        )
        return normalize_writing_record("server", "character", self._model_to_dict(response))

    async def list_characters(
        self,
        project_id: str,
        *,
        role: str | None = None,
        cast_group: str | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce(self._action_id("characters", "list"))
        response = await self._require_client().list_manuscript_characters(
            project_id,
            role=role,
            cast_group=cast_group,
        )
        return [
            normalize_writing_record("server", "character", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def get_character(self, character_id: str) -> dict[str, Any] | None:
        self._enforce(self._action_id("characters", "detail"))
        response = await self._require_client().get_manuscript_character(character_id)
        return normalize_writing_record("server", "character", self._model_to_dict(response))

    async def update_character(
        self,
        character_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("characters", "update"))
        response = await self._require_client().update_manuscript_character(
            character_id,
            ManuscriptCharacterUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "character", self._model_to_dict(response))

    async def delete_character(self, character_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("characters", "delete"))
        return await self._require_client().delete_manuscript_character(
            character_id,
            expected_version=expected_version,
        )

    async def create_relationship(self, project_id: str, **fields: Any) -> dict[str, Any]:
        self._enforce(self._action_id("relationships", "create"))
        response = await self._require_client().create_manuscript_relationship(
            project_id,
            ManuscriptRelationshipCreate(**fields),
        )
        return normalize_writing_record("server", "relationship", self._model_to_dict(response))

    async def list_relationships(self, project_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("relationships", "list"))
        response = await self._require_client().list_manuscript_relationships(project_id)
        return [
            normalize_writing_record("server", "relationship", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def delete_relationship(self, relationship_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("relationships", "delete"))
        return await self._require_client().delete_manuscript_relationship(
            relationship_id,
            expected_version=expected_version,
        )

    async def create_world_info(self, project_id: str, *, kind: str, name: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("world_info", "create"))
        response = await self._require_client().create_manuscript_world_info(
            project_id,
            ManuscriptWorldInfoCreate(kind=kind, name=name, **kwargs),
        )
        return normalize_writing_record("server", "world_info", self._model_to_dict(response))

    async def list_world_info(self, project_id: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        self._enforce(self._action_id("world_info", "list"))
        response = await self._require_client().list_manuscript_world_info(project_id, kind=kind)
        return [
            normalize_writing_record("server", "world_info", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def get_world_info(self, item_id: str) -> dict[str, Any] | None:
        self._enforce(self._action_id("world_info", "detail"))
        response = await self._require_client().get_manuscript_world_info(item_id)
        return normalize_writing_record("server", "world_info", self._model_to_dict(response))

    async def update_world_info(
        self,
        item_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("world_info", "update"))
        response = await self._require_client().update_manuscript_world_info(
            item_id,
            ManuscriptWorldInfoUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "world_info", self._model_to_dict(response))

    async def delete_world_info(self, item_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("world_info", "delete"))
        return await self._require_client().delete_manuscript_world_info(
            item_id,
            expected_version=expected_version,
        )

    async def create_plot_line(self, project_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("plot_lines", "create"))
        response = await self._require_client().create_manuscript_plot_line(
            project_id,
            ManuscriptPlotLineCreate(title=title, **kwargs),
        )
        return normalize_writing_record("server", "plot_line", self._model_to_dict(response))

    async def list_plot_lines(self, project_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("plot_lines", "list"))
        response = await self._require_client().list_manuscript_plot_lines(project_id)
        return [
            normalize_writing_record("server", "plot_line", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def update_plot_line(
        self,
        plot_line_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("plot_lines", "update"))
        response = await self._require_client().update_manuscript_plot_line(
            plot_line_id,
            ManuscriptPlotLineUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "plot_line", self._model_to_dict(response))

    async def delete_plot_line(self, plot_line_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("plot_lines", "delete"))
        return await self._require_client().delete_manuscript_plot_line(
            plot_line_id,
            expected_version=expected_version,
        )

    async def create_plot_event(self, plot_line_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("plot_events", "create"))
        response = await self._require_client().create_manuscript_plot_event(
            plot_line_id,
            ManuscriptPlotEventCreate(title=title, **kwargs),
        )
        return normalize_writing_record("server", "plot_event", self._model_to_dict(response))

    async def list_plot_events(self, plot_line_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("plot_events", "list"))
        response = await self._require_client().list_manuscript_plot_events(plot_line_id)
        return [
            normalize_writing_record("server", "plot_event", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def update_plot_event(
        self,
        plot_event_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("plot_events", "update"))
        response = await self._require_client().update_manuscript_plot_event(
            plot_event_id,
            ManuscriptPlotEventUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "plot_event", self._model_to_dict(response))

    async def delete_plot_event(self, plot_event_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("plot_events", "delete"))
        return await self._require_client().delete_manuscript_plot_event(
            plot_event_id,
            expected_version=expected_version,
        )

    async def create_plot_hole(self, project_id: str, *, title: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("plot_holes", "create"))
        response = await self._require_client().create_manuscript_plot_hole(
            project_id,
            ManuscriptPlotHoleCreate(title=title, **kwargs),
        )
        return normalize_writing_record("server", "plot_hole", self._model_to_dict(response))

    async def list_plot_holes(self, project_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        self._enforce(self._action_id("plot_holes", "list"))
        response = await self._require_client().list_manuscript_plot_holes(project_id, status=status)
        return [
            normalize_writing_record("server", "plot_hole", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def update_plot_hole(
        self,
        plot_hole_id: str,
        *,
        expected_version: int,
        **fields: Any,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("plot_holes", "update"))
        response = await self._require_client().update_manuscript_plot_hole(
            plot_hole_id,
            ManuscriptPlotHoleUpdate(**fields),
            expected_version=expected_version,
        )
        return normalize_writing_record("server", "plot_hole", self._model_to_dict(response))

    async def delete_plot_hole(self, plot_hole_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("plot_holes", "delete"))
        return await self._require_client().delete_manuscript_plot_hole(
            plot_hole_id,
            expected_version=expected_version,
        )

    async def link_scene_character(
        self,
        scene_id: str,
        *,
        character_id: str,
        is_pov: bool = False,
    ) -> list[dict[str, Any]]:
        self._enforce(self._action_id("scene_characters", "create"))
        response = await self._require_client().link_manuscript_scene_character(
            scene_id,
            SceneCharacterLink(character_id=character_id, is_pov=is_pov),
        )
        return [
            normalize_writing_record("server", "scene_character_link", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def list_scene_characters(self, scene_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("scene_characters", "list"))
        response = await self._require_client().list_manuscript_scene_characters(scene_id)
        return [
            normalize_writing_record("server", "scene_character_link", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def unlink_scene_character(self, scene_id: str, character_id: str) -> bool:
        self._enforce(self._action_id("scene_characters", "delete"))
        return await self._require_client().unlink_manuscript_scene_character(scene_id, character_id)

    async def link_scene_world_info(self, scene_id: str, *, world_info_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("scene_world_info", "create"))
        response = await self._require_client().link_manuscript_scene_world_info(
            scene_id,
            SceneWorldInfoLink(world_info_id=world_info_id),
        )
        return [
            normalize_writing_record("server", "scene_world_info_link", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def list_scene_world_info(self, scene_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("scene_world_info", "list"))
        response = await self._require_client().list_manuscript_scene_world_info(scene_id)
        return [
            normalize_writing_record("server", "scene_world_info_link", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def unlink_scene_world_info(self, scene_id: str, world_info_id: str) -> bool:
        self._enforce(self._action_id("scene_world_info", "delete"))
        return await self._require_client().unlink_manuscript_scene_world_info(scene_id, world_info_id)

    async def create_citation(self, scene_id: str, *, source_type: str, **kwargs: Any) -> dict[str, Any]:
        self._enforce(self._action_id("citations", "create"))
        response = await self._require_client().create_manuscript_citation(
            scene_id,
            ManuscriptCitationCreate(source_type=source_type, **kwargs),
        )
        return normalize_writing_record("server", "citation", self._model_to_dict(response))

    async def list_citations(self, scene_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("citations", "list"))
        response = await self._require_client().list_manuscript_citations(scene_id)
        return [
            normalize_writing_record("server", "citation", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def delete_citation(self, citation_id: str, *, expected_version: int) -> bool:
        self._enforce(self._action_id("citations", "delete"))
        return await self._require_client().delete_manuscript_citation(
            citation_id,
            expected_version=expected_version,
        )

    async def research_scene(self, scene_id: str, *, query: str, top_k: int = 5) -> dict[str, Any]:
        self._enforce(self._action_id("research", "launch"))
        response = await self._require_client().research_manuscript_scene(
            scene_id,
            ManuscriptResearchRequest(query=query, top_k=top_k),
        )
        payload = self._model_to_dict(response)
        payload["results"] = [
            normalize_writing_record("server", "research_result", self._model_to_dict(item))
            for item in list(payload.get("results", []))
        ]
        return payload

    async def analyze_scene(
        self,
        scene_id: str,
        *,
        analysis_types: list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce(self._action_id("analysis", "launch"))
        request_data = ManuscriptAnalysisRequest(analysis_types=analysis_types, provider=provider, model=model)
        response = await self._require_client().analyze_manuscript_scene(scene_id, request_data)
        return [
            normalize_writing_record("server", "analysis", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def analyze_chapter(
        self,
        chapter_id: str,
        *,
        analysis_types: list[str],
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce(self._action_id("analysis", "launch"))
        request_data = ManuscriptAnalysisRequest(analysis_types=analysis_types, provider=provider, model=model)
        response = await self._require_client().analyze_manuscript_chapter(chapter_id, request_data)
        return [
            normalize_writing_record("server", "analysis", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def analyze_project_plot_holes(
        self,
        project_id: str,
        *,
        analysis_types: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce(self._action_id("analysis", "launch"))
        request_data = ManuscriptAnalysisRequest(
            analysis_types=analysis_types or ["plot_holes"],
            provider=provider,
            model=model,
        )
        response = await self._require_client().analyze_manuscript_project_plot_holes(project_id, request_data)
        return [
            normalize_writing_record("server", "analysis", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def analyze_project_consistency(
        self,
        project_id: str,
        *,
        analysis_types: list[str] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        self._enforce(self._action_id("analysis", "launch"))
        request_data = ManuscriptAnalysisRequest(
            analysis_types=analysis_types or ["consistency"],
            provider=provider,
            model=model,
        )
        response = await self._require_client().analyze_manuscript_project_consistency(project_id, request_data)
        return [
            normalize_writing_record("server", "analysis", self._model_to_dict(item))
            for item in list(response or [])
        ]

    async def list_analyses(
        self,
        project_id: str,
        *,
        scope_type: str | None = None,
        analysis_type: str | None = None,
        include_stale: bool = False,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("analysis", "list"))
        response = await self._require_client().list_manuscript_analyses(
            project_id,
            scope_type=scope_type,
            analysis_type=analysis_type,
            include_stale=include_stale,
        )
        payload = self._model_to_dict(response)
        payload["analyses"] = [
            normalize_writing_record("server", "analysis", self._model_to_dict(item))
            for item in list(payload.get("analyses", []))
        ]
        return payload

    async def create_version(self, entity_type: str, entity_id: str, *, label: str | None = None) -> dict[str, Any]:
        self._enforce(self._action_id("versions", "create"))
        raise NotImplementedError("Server writing version history is not exposed by the current server contract.")

    async def list_versions(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        self._enforce(self._action_id("versions", "list"))
        raise NotImplementedError("Server writing version history is not exposed by the current server contract.")

    async def get_version(self, entity_type: str, entity_id: str, version_number: int) -> dict[str, Any]:
        self._enforce(self._action_id("versions", "detail"))
        raise NotImplementedError("Server writing version history is not exposed by the current server contract.")

    async def restore_version(
        self,
        entity_type: str,
        entity_id: str,
        version_number: int,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("versions", "restore"))
        raise NotImplementedError("Server writing version restore is not exposed by the current server contract.")

    async def list_trash(self, *, entity_type: str | None = None) -> list[dict[str, Any]]:
        self._enforce(self._action_id("trash", "list"))
        raise NotImplementedError("Server writing trash listing is not exposed by the current server contract.")

    async def restore_trash(
        self,
        entity_type: str,
        entity_id: str,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        self._enforce(self._action_id("trash", "restore"))
        raise NotImplementedError("Server writing trash restore is not exposed by the current server contract.")
