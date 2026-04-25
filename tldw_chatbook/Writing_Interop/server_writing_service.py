"""Server-backed writing-suite service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    ManuscriptChapterCreate,
    ManuscriptChapterUpdate,
    ManuscriptPartCreate,
    ManuscriptPartUpdate,
    ManuscriptProjectCreate,
    ManuscriptProjectUpdate,
    ManuscriptSceneCreate,
    ManuscriptSceneUpdate,
    ReorderRequest,
    TLDWAPIClient,
)
from .writing_markdown_adapter import markdown_to_plain_text, markdown_to_server_content
from .writing_normalizers import normalize_writing_record, normalize_writing_structure


_UNSET = object()


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
            raise NotImplementedError(
                "Direct manuscript-level scenes are not exposed by the current server contract."
            )
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
            raise NotImplementedError(
                "Direct manuscript-level scenes are not exposed by the current server contract."
            )
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
        self._enforce(self._action_id("projects", "detail"))
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
