"""Thin server-backed retrieval-admin service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    BatchMediaEmbeddingsRequest,
    ChunkingTemplateApplyRequest,
    ChunkingTemplateCreateRequest,
    ChunkingTemplateLearnRequest,
    ChunkingTemplateUpdateRequest,
    GenerateMediaEmbeddingsRequest,
    MediaEmbeddingsSearchRequest,
    ReprocessMediaRequest,
    TLDWAPIClient,
)


class ServerRAGAdminService:
    """Thin wrapper around server chunking-template and collection admin endpoints."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerRAGAdminService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server retrieval-admin operations.")
        return self.client

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
        response = await self._require_client().list_chunking_templates(
            include_builtin=include_builtin,
            include_custom=include_custom,
            tags=list(tags) if tags is not None else None,
            user_id=user_id,
        )
        payload = self._dump_model(response)
        return list(payload.get("templates", []))

    async def get_template(self, template_name: str) -> dict[str, Any]:
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
        request = ChunkingTemplateUpdateRequest(
            description=description,
            template=dict(template) if template is not None else None,
            tags=list(tags) if tags is not None else None,
        )
        return self._dump_model(
            await self._require_client().update_chunking_template(template_name, request)
        )

    async def delete_template(self, template_name: str, *, hard_delete: bool = False) -> None:
        await self._require_client().delete_chunking_template(template_name, hard_delete=hard_delete)

    async def apply_template(
        self,
        template_name: str,
        *,
        text: str,
        override_options: Optional[Mapping[str, Any]] = None,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
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
        return self._dump_model(await self._require_client().list_embedding_collections())

    async def get_collection_detail(self, collection_name: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_embedding_collection_stats(collection_name))

    async def delete_collection(self, collection_name: str) -> None:
        await self._require_client().delete_embedding_collection(collection_name)

    async def get_media_embeddings_status(self, media_id: int) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_media_embeddings_status(media_id))

    async def generate_media_embeddings(
        self,
        media_id: int,
        *,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_regenerate: bool = False,
        priority: int = 50,
    ) -> dict[str, Any]:
        request = GenerateMediaEmbeddingsRequest(
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_regenerate=force_regenerate,
            priority=priority,
        )
        return self._dump_model(
            await self._require_client().generate_media_embeddings(media_id, request)
        )

    async def generate_media_embeddings_batch(
        self,
        *,
        media_ids: Sequence[int],
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_regenerate: bool = False,
        priority: int = 50,
    ) -> dict[str, Any]:
        request = BatchMediaEmbeddingsRequest(
            media_ids=list(media_ids),
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
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        request = MediaEmbeddingsSearchRequest(
            query=query,
            top_k=top_k,
            collection=collection,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            filters=dict(filters) if filters is not None else None,
        )
        return self._dump_model(await self._require_client().search_media_embeddings(request))

    async def delete_media_embeddings(self, media_id: int) -> dict[str, Any]:
        return self._dump_model(await self._require_client().delete_media_embeddings(media_id))

    async def get_media_embedding_job(self, job_id: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_media_embedding_job(job_id))

    async def list_media_embedding_jobs(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().list_media_embedding_jobs(
                status=status,
                limit=limit,
                offset=offset,
            )
        )

    async def reprocess_media(
        self,
        media_id: int,
        *,
        perform_chunking: bool = True,
        generate_embeddings: bool = False,
        chunk_method: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        auto_apply_template: bool = False,
        chunking_template_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        force_regenerate_embeddings: bool = False,
        **extra_options: Any,
    ) -> dict[str, Any]:
        request = ReprocessMediaRequest(
            perform_chunking=perform_chunking,
            generate_embeddings=generate_embeddings,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            auto_apply_template=auto_apply_template,
            chunking_template_name=chunking_template_name,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            force_regenerate_embeddings=force_regenerate_embeddings,
            **extra_options,
        )
        return self._dump_model(await self._require_client().reprocess_media(media_id, request))
