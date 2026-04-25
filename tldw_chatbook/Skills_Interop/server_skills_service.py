"""Server-backed SKILL.md management service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    SkillCreate,
    SkillExecuteRequest,
    SkillImportRequest,
    SkillUpdate,
    TLDWAPIClient,
)


class ServerSkillsService:
    """Policy-gated access to server skill APIs."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerSkillsService":
        return cls(
            client=build_runtime_api_client_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server skill operations.")
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
                    user_message=getattr(decision, "user_message", None) or "Server skill action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if isinstance(response, list):
            return [ServerSkillsService._dump(item) for item in response]
        if isinstance(response, (dict, bool)):
            return response
        return dict(response or {})

    async def list_skills(
        self,
        *,
        include_hidden: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce("skills.list.server")
        return self._dump(
            await self._require_client().list_skills(
                include_hidden=include_hidden,
                limit=limit,
                offset=offset,
            )
        )

    async def get_context(self) -> dict[str, Any]:
        self._enforce("skills.context.list.server")
        return self._dump(await self._require_client().get_skills_context())

    async def get_skill(self, skill_name: str) -> dict[str, Any]:
        self._enforce("skills.detail.server")
        return self._dump(await self._require_client().get_skill(skill_name))

    async def create_skill(
        self,
        *,
        name: str,
        content: str,
        supporting_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._enforce("skills.create.server")
        request = SkillCreate(name=name, content=content, supporting_files=supporting_files)
        return self._dump(await self._require_client().create_skill(request))

    async def update_skill(
        self,
        skill_name: str,
        *,
        content: str | None = None,
        supporting_files: dict[str, str | None] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        self._enforce("skills.update.server")
        request = SkillUpdate(content=content, supporting_files=supporting_files)
        return self._dump(
            await self._require_client().update_skill(skill_name, request, expected_version=expected_version)
        )

    async def delete_skill(self, skill_name: str, *, expected_version: int | None = None) -> bool:
        self._enforce("skills.delete.server")
        return bool(await self._require_client().delete_skill(skill_name, expected_version=expected_version))

    async def import_skill(
        self,
        *,
        content: str,
        name: str | None = None,
        supporting_files: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        self._enforce("skills.import.launch.server")
        request = SkillImportRequest(
            name=name,
            content=content,
            supporting_files=supporting_files,
            overwrite=overwrite,
        )
        return self._dump(await self._require_client().import_skill(request))

    async def import_skill_file(
        self,
        file_content: bytes,
        *,
        filename: str = "SKILL.md",
        content_type: str = "text/markdown",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        self._enforce("skills.import.launch.server")
        return self._dump(
            await self._require_client().import_skill_file(
                file_content,
                filename=filename,
                content_type=content_type,
                overwrite=overwrite,
            )
        )

    async def export_skill(self, skill_name: str) -> Any:
        self._enforce("skills.export.launch.server")
        return await self._require_client().export_skill(skill_name)

    async def execute_skill(self, skill_name: str, *, args: str | None = None) -> dict[str, Any]:
        self._enforce("skills.execute.launch.server")
        request = SkillExecuteRequest(args=args)
        return self._dump(await self._require_client().execute_skill(skill_name, request))

    async def seed_builtin_skills(self, *, overwrite: bool = False) -> dict[str, Any]:
        self._enforce("skills.seed.launch.server")
        return self._dump(await self._require_client().seed_builtin_skills(overwrite=overwrite))
