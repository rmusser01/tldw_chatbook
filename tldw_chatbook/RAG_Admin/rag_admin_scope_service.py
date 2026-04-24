"""Mode-aware routing for chunking-template and collection admin surfaces."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

from .rag_admin_normalizers import normalize_collection_record, normalize_template_record


class RAGAdminBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class RAGAdminScopeService:
    """Route retrieval-admin actions to local or server backends and normalize outputs."""

    def __init__(self, *, local_service: Any, server_service: Any):
        self.local_service = local_service
        self.server_service = server_service

    def _normalize_mode(self, mode: RAGAdminBackend | str | None) -> RAGAdminBackend:
        if mode is None:
            return RAGAdminBackend.LOCAL
        if isinstance(mode, RAGAdminBackend):
            return mode
        try:
            return RAGAdminBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid RAG admin backend: {mode}") from exc

    def _service_for_mode(self, mode: RAGAdminBackend) -> Any:
        if mode == RAGAdminBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local retrieval-admin backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server retrieval-admin backend is unavailable.")
        return self.server_service

    def _server_service_for_mode(self, mode: RAGAdminBackend | str | None) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != RAGAdminBackend.SERVER:
            raise ValueError("Server retrieval-admin backend is required for this RAG admin operation.")
        return self._service_for_mode(normalized_mode)

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def list_templates(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        include_builtin: bool = True,
        include_custom: bool = True,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        records = await self._maybe_await(
            service.list_templates(
                include_builtin=include_builtin,
                include_custom=include_custom,
                tags=tags,
                user_id=user_id,
            )
        )
        return [
            normalize_template_record(normalized_mode.value, record)
            for record in list(records or [])
        ]

    async def get_template_detail(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        template_name: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(service.get_template(template_name))
        return normalize_template_record(normalized_mode.value, record)

    async def create_template(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        name: str,
        description: str,
        template: dict[str, Any],
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(
            service.create_template(
                name=name,
                description=description,
                template=template,
                tags=tags,
                user_id=user_id,
            )
        )
        return normalize_template_record(normalized_mode.value, record)

    async def update_template(
        self,
        template_name: str,
        *,
        mode: RAGAdminBackend | str | None = None,
        description: str | None = None,
        template: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(
            service.update_template(
                template_name,
                description=description,
                template=template,
                tags=tags,
            )
        )
        return normalize_template_record(normalized_mode.value, record)

    async def delete_template(
        self,
        template_name: str,
        *,
        mode: RAGAdminBackend | str | None = None,
        hard_delete: bool = False,
    ) -> None:
        service = self._service_for_mode(self._normalize_mode(mode))
        await self._maybe_await(service.delete_template(template_name, hard_delete=hard_delete))

    async def get_template_diagnostics(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        diagnostics = await self._maybe_await(service.get_template_diagnostics())
        payload = dict(diagnostics or {})
        payload.setdefault("backend", normalized_mode.value)
        return payload

    async def apply_template(
        self,
        template_name: str,
        *,
        mode: RAGAdminBackend | str | None = None,
        text: str,
        override_options: dict[str, Any] | None = None,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "apply_template", None)
        if not callable(method):
            raise ValueError(f"{normalized_mode.value.title()} template apply is not available yet.")
        return await self._maybe_await(
            method(
                template_name,
                text=text,
                override_options=override_options,
                include_metadata=include_metadata,
            )
        )

    async def validate_template_config(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        template_config: dict[str, Any],
    ) -> dict[str, Any]:
        service = self._server_service_for_mode(mode)
        return await self._maybe_await(service.validate_template_config(template_config))

    async def match_templates(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_type: str | None = None,
        title: str | None = None,
        url: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        service = self._server_service_for_mode(mode)
        kwargs: dict[str, Any] = {}
        if media_type is not None:
            kwargs["media_type"] = media_type
        if title is not None:
            kwargs["title"] = title
        if url is not None:
            kwargs["url"] = url
        if filename is not None:
            kwargs["filename"] = filename
        return await self._maybe_await(service.match_templates(**kwargs))

    async def learn_template(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        name: str,
        example_text: str | None = None,
        description: str | None = None,
        save: bool = False,
        classifier: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        service = self._server_service_for_mode(mode)
        kwargs: dict[str, Any] = {"name": name}
        if example_text is not None:
            kwargs["example_text"] = example_text
        if description is not None:
            kwargs["description"] = description
        if save:
            kwargs["save"] = save
        if classifier is not None:
            kwargs["classifier"] = classifier
        return await self._maybe_await(service.learn_template(**kwargs))

    async def list_collections(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        records = await self._maybe_await(service.list_collections())
        return [
            normalize_collection_record(normalized_mode.value, record)
            for record in list(records or [])
        ]

    async def get_collection_detail(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        collection_name: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        detail = await self._maybe_await(service.get_collection_detail(collection_name))
        return normalize_collection_record(normalized_mode.value, detail)

    async def delete_collection(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        collection_name: str,
    ) -> None:
        service = self._service_for_mode(self._normalize_mode(mode))
        await self._maybe_await(service.delete_collection(collection_name))

    async def get_media_embeddings_status(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: int,
    ) -> dict[str, Any]:
        service = self._service_for_mode(self._normalize_mode(mode))
        return await self._maybe_await(service.get_media_embeddings_status(media_id))

    async def generate_media_embeddings(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: int,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_regenerate: bool = False,
        priority: int = 50,
    ) -> dict[str, Any]:
        service = self._service_for_mode(self._normalize_mode(mode))
        kwargs: dict[str, Any] = {}
        if embedding_model is not None:
            kwargs["embedding_model"] = embedding_model
        if embedding_provider is not None:
            kwargs["embedding_provider"] = embedding_provider
        if chunk_size != 1000:
            kwargs["chunk_size"] = chunk_size
        if chunk_overlap != 200:
            kwargs["chunk_overlap"] = chunk_overlap
        if force_regenerate:
            kwargs["force_regenerate"] = force_regenerate
        if priority != 50:
            kwargs["priority"] = priority
        return await self._maybe_await(service.generate_media_embeddings(media_id, **kwargs))

    async def generate_media_embeddings_batch(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_ids: list[int],
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force_regenerate: bool = False,
        priority: int = 50,
    ) -> dict[str, Any]:
        service = self._server_service_for_mode(mode)
        return await self._maybe_await(
            service.generate_media_embeddings_batch(
                media_ids=media_ids,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force_regenerate=force_regenerate,
                priority=priority,
            )
        )

    async def search_media_embeddings(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        query: str,
        top_k: int = 5,
        collection: str | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        service = self._service_for_mode(self._normalize_mode(mode))
        return await self._maybe_await(
            service.search_media_embeddings(
                query=query,
                top_k=top_k,
                collection=collection,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                filters=filters,
            )
        )

    async def delete_media_embeddings(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: int,
    ) -> dict[str, Any]:
        service = self._service_for_mode(self._normalize_mode(mode))
        return await self._maybe_await(service.delete_media_embeddings(media_id))

    async def get_media_embedding_job(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        service = self._service_for_mode(self._normalize_mode(mode))
        return await self._maybe_await(service.get_media_embedding_job(job_id))

    async def list_media_embedding_jobs(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        service = self._service_for_mode(self._normalize_mode(mode))
        return await self._maybe_await(
            service.list_media_embedding_jobs(
                status=status,
                limit=limit,
                offset=offset,
            )
        )

    async def reprocess_media(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: int,
        perform_chunking: bool = True,
        generate_embeddings: bool = False,
        chunk_method: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        auto_apply_template: bool = False,
        chunking_template_name: str | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        force_regenerate_embeddings: bool = False,
        **extra_options: Any,
    ) -> dict[str, Any]:
        service = self._service_for_mode(self._normalize_mode(mode))
        kwargs: dict[str, Any] = dict(extra_options)
        kwargs["perform_chunking"] = perform_chunking
        if generate_embeddings:
            kwargs["generate_embeddings"] = generate_embeddings
        if chunk_method is not None:
            kwargs["chunk_method"] = chunk_method
        if chunk_size != 500:
            kwargs["chunk_size"] = chunk_size
        if chunk_overlap != 200:
            kwargs["chunk_overlap"] = chunk_overlap
        if auto_apply_template:
            kwargs["auto_apply_template"] = auto_apply_template
        if chunking_template_name is not None:
            kwargs["chunking_template_name"] = chunking_template_name
        if embedding_model is not None:
            kwargs["embedding_model"] = embedding_model
        if embedding_provider is not None:
            kwargs["embedding_provider"] = embedding_provider
        if force_regenerate_embeddings:
            kwargs["force_regenerate_embeddings"] = force_regenerate_embeddings
        return await self._maybe_await(service.reprocess_media(media_id, **kwargs))
