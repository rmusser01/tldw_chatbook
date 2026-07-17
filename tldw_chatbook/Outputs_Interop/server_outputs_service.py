"""Server-backed output templates and artifact service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


class ServerOutputsService:
    """Policy-gated access to server output templates and artifacts."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> None:
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerOutputsService":
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerOutputsService":
        return cls(
            client=None,
            client_provider=provider,
            policy_enforcer=policy_enforcer,
        )

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server output operations.")

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
                    user_message=getattr(decision, "user_message", None) or "Server output action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _dump(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        return dict(response or {})

    async def list_templates(
        self,
        *,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce("outputs.templates.list.server")
        return self._dump(await self._require_client().list_output_templates(q=q, limit=limit, offset=offset))

    async def get_template(self, template_id: int) -> dict[str, Any]:
        self._enforce("outputs.templates.detail.server")
        return self._dump(await self._require_client().get_output_template(template_id))

    async def create_template(
        self,
        *,
        name: str,
        type: str,
        format: str,
        body: str,
        description: str | None = None,
        is_default: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import OutputTemplateCreate

        self._enforce("outputs.templates.create.server")
        request = OutputTemplateCreate(
            name=name,
            type=type,  # type: ignore[arg-type]
            format=format,  # type: ignore[arg-type]
            body=body,
            description=description,
            is_default=is_default,
            metadata=metadata,
        )
        return self._dump(await self._require_client().create_output_template(request))

    async def update_template(self, template_id: int, **payload: Any) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import OutputTemplateUpdate

        self._enforce("outputs.templates.update.server")
        request = OutputTemplateUpdate(**{key: value for key, value in payload.items() if value is not None})
        return self._dump(await self._require_client().update_output_template(template_id, request))

    async def delete_template(self, template_id: int) -> dict[str, Any]:
        self._enforce("outputs.templates.delete.server")
        return self._dump(await self._require_client().delete_output_template(template_id))

    async def preview_template(
        self,
        template_id: int,
        *,
        item_ids: list[int] | None = None,
        run_id: int | None = None,
        limit: int = 50,
        data: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import TemplatePreviewRequest

        self._enforce("outputs.render_jobs.launch.server")
        request = TemplatePreviewRequest(
            template_id=template_id,
            item_ids=item_ids,
            run_id=run_id,
            limit=limit,
            data=data,
        )
        return self._dump(await self._require_client().preview_output_template(template_id, request))

    async def list_artifacts(
        self,
        *,
        page: int = 1,
        size: int = 50,
        job_id: int | None = None,
        run_id: int | None = None,
        type: str | None = None,
        workspace_tag: str | None = None,
        include_deleted: bool | None = None,
    ) -> dict[str, Any]:
        self._enforce("outputs.artifacts.list.server")
        return self._dump(
            await self._require_client().list_outputs(
                page=page,
                size=size,
                job_id=job_id,
                run_id=run_id,
                type=type,
                workspace_tag=workspace_tag,
                include_deleted=include_deleted,
            )
        )

    async def list_deleted_artifacts(self, *, page: int = 1, size: int = 50) -> dict[str, Any]:
        self._enforce("outputs.artifacts.list.server")
        return self._dump(await self._require_client().list_deleted_outputs(page=page, size=size))

    async def get_artifact(self, output_id: int) -> dict[str, Any]:
        self._enforce("outputs.artifacts.detail.server")
        return self._dump(await self._require_client().get_output(output_id))

    async def create_artifact(
        self,
        *,
        template_id: int,
        item_ids: list[int] | None = None,
        run_id: int | None = None,
        title: str | None = None,
        data: dict[str, object] | None = None,
        workspace_tag: str | None = None,
        generate_mece: bool = False,
        mece_template_id: int | None = None,
        generate_tts: bool = False,
        tts_template_id: int | None = None,
        ingest_to_media_db: bool = False,
        tts_model: str | None = None,
        tts_voice: str | None = None,
        tts_speed: float | None = None,
    ) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import OutputCreateRequest

        self._enforce("outputs.artifacts.create.server")
        request = OutputCreateRequest(
            template_id=template_id,
            item_ids=item_ids,
            run_id=run_id,
            title=title,
            data=data,
            workspace_tag=workspace_tag,
            generate_mece=generate_mece,
            mece_template_id=mece_template_id,
            generate_tts=generate_tts,
            tts_template_id=tts_template_id,
            ingest_to_media_db=ingest_to_media_db,
            tts_model=tts_model,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
        )
        return self._dump(await self._require_client().create_output(request))

    async def update_artifact(self, output_id: int, **payload: Any) -> dict[str, Any]:
        # Deferred import: avoid module-scope tldw_api schema import (task-285 phase 2).
        from ..tldw_api import OutputUpdateRequest

        self._enforce("outputs.artifacts.update.server")
        request = OutputUpdateRequest(**{key: value for key, value in payload.items() if value is not None})
        return self._dump(await self._require_client().update_output(output_id, request))

    async def delete_artifact(
        self,
        output_id: int,
        *,
        hard: bool = False,
        delete_file: bool = False,
    ) -> dict[str, Any]:
        self._enforce("outputs.artifacts.delete.server")
        return self._dump(
            await self._require_client().delete_output(output_id, hard=hard, delete_file=delete_file)
        )

    async def purge_artifacts(
        self,
        *,
        delete_files: bool = False,
        soft_deleted_grace_days: int = 30,
        include_retention: bool = True,
    ) -> dict[str, Any]:
        self._enforce("outputs.artifacts.delete.server")
        return self._dump(
            await self._require_client().purge_outputs(
                delete_files=delete_files,
                soft_deleted_grace_days=soft_deleted_grace_days,
                include_retention=include_retention,
            )
        )
