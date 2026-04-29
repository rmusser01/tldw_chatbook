"""Thin server-backed retrieval-admin service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    BatchMediaEmbeddingsRequest,
    ChunkingTemplateApplyRequest,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateLearnRequest,
    ChunkingTemplateUpdateRequest,
    EmbeddingCollectionCreateRequest,
    MediaEmbeddingsBatchRequest,
    MediaEmbeddingsGenerateRequest,
    MediaEmbeddingsSearchRequest,
    ReprocessMediaRequest,
    TLDWAPIClient,
)


class ServerRAGAdminService:
    """Thin wrapper around server chunking-template and collection admin endpoints."""

    def __init__(
        self,
        client: Optional[TLDWAPIClient],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ):
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerRAGAdminService":
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
    ) -> "ServerRAGAdminService":
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
        raise ValueError("TLDW API client is required for server retrieval-admin operations.")

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
                    user_message=getattr(decision, "user_message", None)
                    or "Server retrieval-admin action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @staticmethod
    def _template_action_id(action: str) -> str:
        return f"rag.template.{action}.server"

    @staticmethod
    def _admin_action_id(action: str) -> str:
        return f"rag.admin.{action}.server"

    @staticmethod
    def _media_embeddings_action_id(action: str) -> str:
        return f"rag.media_embeddings.{action}.server"

    @staticmethod
    def _media_embedding_jobs_action_id(action: str) -> str:
        return f"rag.media_embedding_jobs.{action}.server"

    def _dump_model(self, value: Any) -> Any:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [self._dump_model(item) for item in value]
        return value

    async def list_templates(
        self,
        *,
        include_builtin: bool = True,
        include_custom: bool = True,
        tags: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        self._enforce(self._template_action_id("list"))
        response = await self._require_client().list_chunking_templates(
            include_builtin=include_builtin,
            include_custom=include_custom,
            tags=list(tags) if tags is not None else None,
            user_id=user_id,
        )
        payload = self._dump_model(response)
        return list(payload.get("templates", []))

    async def get_template(self, template_name: str) -> dict[str, Any]:
        self._enforce(self._template_action_id("detail"))
        return self._dump_model(await self._require_client().get_chunking_template(template_name))

    async def create_template(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        template: Mapping[str, Any],
        tags: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        self._enforce(self._template_action_id("create"))
        request = ChunkingTemplateCreateRequest(
            name=name,
            description=description,
            template=dict(template),
            tags=list(tags or []),
            user_id=user_id,
        )
        return self._dump_model(await self._require_client().create_chunking_template(request))

    async def update_template(
        self,
        template_name: str,
        *,
        description: Optional[str] = None,
        template: Optional[Mapping[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> dict[str, Any]:
        self._enforce(self._template_action_id("update"))
        request = ChunkingTemplateUpdateRequest(
            description=description,
            template=dict(template) if template is not None else None,
            tags=list(tags) if tags is not None else None,
        )
        return self._dump_model(
            await self._require_client().update_chunking_template(template_name, request)
        )

    async def delete_template(self, template_name: str, *, hard_delete: bool = False) -> None:
        self._enforce(self._template_action_id("delete"))
        await self._require_client().delete_chunking_template(template_name, hard_delete=hard_delete)

    async def apply_template(
        self,
        template_name: str,
        *,
        text: str,
        override_options: Optional[Mapping[str, Any]] = None,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        self._enforce(self._admin_action_id("launch"))
        request = ChunkingTemplateApplyRequest(
            template_name=template_name,
            text=text,
            override_options=dict(override_options) if override_options is not None else None,
        )
        return self._dump_model(
            await self._require_client().apply_chunking_template(
                request,
                include_metadata=include_metadata,
            )
        )

    async def get_template_diagnostics(self) -> dict[str, Any]:
        self._enforce(self._admin_action_id("observe"))
        return self._dump_model(await self._require_client().get_chunking_template_diagnostics())

    async def validate_template_config(self, template_config: Mapping[str, Any]) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().validate_chunking_template(dict(template_config))
        )

    async def match_templates(
        self,
        *,
        media_type: Optional[str] = None,
        title: Optional[str] = None,
        url: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().match_chunking_templates(
                media_type=media_type,
                title=title,
                url=url,
                filename=filename,
            )
        )

    async def learn_template(
        self,
        *,
        name: str,
        example_text: Optional[str] = None,
        description: Optional[str] = None,
        save: bool = False,
        classifier: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        request = ChunkingTemplateLearnRequest(
            name=name,
            example_text=example_text,
            description=description,
            save=save,
            classifier=dict(classifier) if classifier is not None else None,
        )
        return self._dump_model(await self._require_client().learn_chunking_template(request))

    async def list_collections(self) -> list[dict[str, Any]]:
        self._enforce(self._admin_action_id("list"))
        return self._dump_model(await self._require_client().list_embedding_collections())

    async def create_collection(
        self,
        *,
        name: str,
        metadata: Mapping[str, Any] | None = None,
        embedding_model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        self._enforce(self._admin_action_id("configure"))
        request = EmbeddingCollectionCreateRequest(
            name=name,
            metadata=dict(metadata) if metadata is not None else None,
            embedding_model=embedding_model,
            provider=provider,
        )
        return self._dump_model(await self._require_client().create_embedding_collection(request))

    async def get_collection_detail(self, collection_name: str) -> dict[str, Any]:
        self._enforce(self._admin_action_id("observe"))
        return self._dump_model(await self._require_client().get_embedding_collection_stats(collection_name))

    async def delete_collection(self, collection_name: str) -> None:
        self._enforce(self._admin_action_id("configure"))
        await self._require_client().delete_embedding_collection(collection_name)

    async def reprocess_media(
        self,
        media_id: Any,
        *,
        perform_chunking: bool = True,
        generate_embeddings: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        force_regenerate_embeddings: bool = False,
        **options: Any,
    ) -> dict[str, Any]:
        self._enforce(self._admin_action_id("launch"))
        request = ReprocessMediaRequest(
            perform_chunking=perform_chunking,
            generate_embeddings=generate_embeddings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_regenerate_embeddings=force_regenerate_embeddings,
            **options,
        )
        return self._dump_model(await self._require_client().reprocess_media(int(media_id), request))

    async def get_media_embeddings_status(self, media_id: Any) -> dict[str, Any]:
        self._enforce(self._media_embeddings_action_id("status"))
        return self._dump_model(await self._require_client().get_media_embeddings_status(int(media_id)))

    async def generate_media_embeddings(
        self,
        media_id: Any,
        *,
        request_data: MediaEmbeddingsGenerateRequest | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_regenerate: bool = False,
        priority: int = 50,
    ) -> dict[str, Any]:
        self._enforce(self._media_embeddings_action_id("create"))
        request = request_data or MediaEmbeddingsGenerateRequest(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_regenerate=force_regenerate,
            priority=priority,
        )
        return self._dump_model(
            await self._require_client().generate_media_embeddings(int(media_id), request)
        )

    async def generate_media_embeddings_batch(
        self,
        *,
        request_data: MediaEmbeddingsBatchRequest | None = None,
        media_ids: Sequence[Any] | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_regenerate: bool = False,
        priority: int = 50,
    ) -> dict[str, Any]:
        self._enforce(self._media_embeddings_action_id("create"))
        request = request_data or MediaEmbeddingsBatchRequest(
            media_ids=[int(media_id) for media_id in list(media_ids or [])],
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_regenerate=force_regenerate,
            priority=priority,
        )
        return self._dump_model(await self._require_client().generate_media_embeddings_batch(request))

    async def search_media_embeddings(
        self,
        *,
        request_data: MediaEmbeddingsSearchRequest | None = None,
        query: str | None = None,
        top_k: int = 5,
        collection: str | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._enforce(self._media_embeddings_action_id("search"))
        if request_data is None and not query:
            raise ValueError("query is required when request_data is not provided.")
        request = request_data or MediaEmbeddingsSearchRequest(
            query=str(query),
            top_k=top_k,
            collection=collection,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            filters=dict(filters) if filters is not None else None,
        )
        return self._dump_model(await self._require_client().search_media_embeddings(request))

    async def delete_media_embeddings(self, media_id: Any) -> dict[str, Any]:
        self._enforce(self._media_embeddings_action_id("delete"))
        return self._dump_model(await self._require_client().delete_media_embeddings(int(media_id)))

    async def get_media_embedding_job(self, job_id: str) -> dict[str, Any]:
        self._enforce(self._media_embedding_jobs_action_id("detail"))
        return self._dump_model(await self._require_client().get_media_embedding_job(str(job_id)))

    async def list_media_embedding_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        self._enforce(self._media_embedding_jobs_action_id("list"))
        return self._dump_model(
            await self._require_client().list_media_embedding_jobs(
                status=status,
                limit=limit,
                offset=offset,
            )
        )
