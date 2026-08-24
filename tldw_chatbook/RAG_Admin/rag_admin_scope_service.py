"""Mode-aware routing for chunking-template and collection admin surfaces."""

from __future__ import annotations

import asyncio
import inspect
from enum import Enum
from typing import Any

from .rag_admin_normalizers import (
    normalize_collection_record,
    normalize_template_record,
)
from .template_validation import validate_template


class RAGAdminBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES: list[dict[str, Any]] = [
    {
        "operation_id": "rag.media_embeddings.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Server-style per-media embedding status, generation, search, deletion, and job tracking are not exposed by the local RAG admin seam yet.",
        "affected_action_ids": [],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "rag.collections.export.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server embedding admin contract does not expose embedding collection export.",
        "affected_action_ids": ["rag.admin.observe.server"],
    },
]


class RAGAdminScopeService:
    """Route retrieval-admin actions to local or server backends and normalize outputs."""

    def __init__(
        self, *, local_service: Any, server_service: Any, policy_enforcer: Any = None
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

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
            raise ValueError(
                "Server retrieval-admin backend is required for this RAG admin operation."
            )
        return self._service_for_mode(normalized_mode)

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _call_off_loop(self, service: Any, method_name: str) -> Any:
        """Invoke a backend method without blocking the event loop.

        (TASK-21126) ``_maybe_await(service.method())`` evaluates its
        argument BEFORE the first suspension point, so a *synchronous*
        backend method runs to completion on the event loop — for
        ``get_template_diagnostics`` that meant the legacy-chunk census
        (measured 119 ms at 200k live chunk rows, 701 ms at 1M) froze the
        UI once per Library Search/RAG panel show.

        The offload is opt-in per backend, never blanket: a backend must
        say so by exposing a truthy ``diagnostics_are_thread_safe()``.
        Anything else — an async backend, a test double, a Mock — keeps the
        pre-existing inline behaviour exactly, so this cannot silently move
        an unknown backend's work onto a thread it was never written for.

        Args:
            service: The resolved local or server backend.
            method_name: Name of the zero-argument backend method to call.

        Returns:
            The method's result, awaited if it was awaitable.
        """
        method = getattr(service, method_name)
        if inspect.iscoroutinefunction(method):
            return await method()
        safe = getattr(service, "diagnostics_are_thread_safe", None)
        if not callable(safe) or not safe():
            return await self._maybe_await(method())
        # `to_thread` propagates the callee's exception into this await, so
        # every existing failure path (an unmigrated media DB, a policy
        # error, a closed connection) reaches callers unchanged.
        return await self._maybe_await(await asyncio.to_thread(method))

    def _enforce_policy(self, action_id: str) -> None:
        """Require a policy action id (no-op without an enforcer).

        task-8: the shadowed 3-arg ``(mode, resource, action)`` variant is
        gone — callers build the id with the ``_*_action_id`` helpers (the
        ``rag.admin.<action>.<mode>`` form), so server-mode paths no longer
        TypeError the moment a policy enforcer is present.
        """
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _template_action_id(mode: RAGAdminBackend, action: str) -> str:
        return f"rag.template.{action}.{mode.value}"

    @staticmethod
    def _admin_action_id(mode: RAGAdminBackend, action: str) -> str:
        return f"rag.admin.{action}.{mode.value}"

    @staticmethod
    def _media_embeddings_action_id(action: str) -> str:
        return f"rag.media_embeddings.{action}.server"

    @staticmethod
    def _media_embedding_jobs_action_id(action: str) -> str:
        return f"rag.media_embedding_jobs.{action}.server"

    def _server_media_embeddings_service(
        self, mode: RAGAdminBackend, operation_name: str
    ) -> Any:
        if mode != RAGAdminBackend.SERVER:
            raise ValueError(f"{operation_name} is server-only.")
        return self._service_for_mode(mode)

    @staticmethod
    def _raise_server_collection_export_unsupported() -> None:
        raise NotImplementedError(
            "The current server embedding admin contract does not expose embedding collection export."
        )

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [RAGAdminScopeService._to_plain(item) for item in value]
        return value

    @classmethod
    def _with_backend(cls, mode: RAGAdminBackend, value: Any) -> dict[str, Any]:
        payload = cls._to_plain(value)
        if isinstance(payload, dict):
            result = dict(payload)
        else:
            result = {"data": payload}
        result.setdefault("backend", mode.value)
        return result

    def list_unsupported_capabilities(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == RAGAdminBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        reports = [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]
        if callable(getattr(self.server_service, "export_collection", None)):
            reports = [
                item
                for item in reports
                if item.get("operation_id") != "rag.collections.export.server"
            ]
        return reports

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
        self._enforce_policy(self._template_action_id(normalized_mode, "list"))
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
        self._enforce_policy(self._template_action_id(normalized_mode, "detail"))
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
        self._enforce_policy(self._template_action_id(normalized_mode, "create"))
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
        self._enforce_policy(self._template_action_id(normalized_mode, "update"))
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
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._template_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        await self._maybe_await(
            service.delete_template(template_name, hard_delete=hard_delete)
        )

    async def get_template_diagnostics(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
    ) -> dict[str, Any]:
        """Return the backend's template diagnostics payload.

        (TASK-21126) Routed through :meth:`_call_off_loop`: the local
        backend's payload carries the legacy-chunk census, whose SELECT used
        to run inline on the event loop once per Library Search/RAG panel
        show. Deliberately NOT cached — the census is re-read per show, and
        that is the whole freshness protocol (an ingest, a re-chunk, a media
        delete or a sync-in all show up on the next visit with nothing to
        invalidate). With the v8 covering index the query costs 23 ms at
        200k live chunk rows and 123 ms at 1M, off the loop, so a cache
        would trade a measured-zero saving for a staleness surface.
        """
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        diagnostics = await self._call_off_loop(service, "get_template_diagnostics")
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
        self._enforce_policy(self._admin_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "apply_template", None)
        if not callable(method):
            raise ValueError(
                f"{normalized_mode.value.title()} template apply is not available yet."
            )
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
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == RAGAdminBackend.LOCAL:
            # Local mode validates with the server-parity validator (spec §7)
            # instead of hard-requiring the server backend. The CRUD layer
            # (task-8) runs the same validator on every create/update.
            self._enforce_policy(self._admin_action_id(normalized_mode, "configure"))
            return validate_template(template_config)
        if normalized_mode != RAGAdminBackend.SERVER:
            raise ValueError(
                "Server retrieval-admin backend is required for this RAG admin operation."
            )
        self._enforce_policy(self._admin_action_id(normalized_mode, "configure"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(
            service.validate_template_config(template_config)
        )

    async def match_templates(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_type: str | None = None,
        title: str | None = None,
        url: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != RAGAdminBackend.SERVER:
            raise ValueError(
                "Server retrieval-admin backend is required for this RAG admin operation."
            )
        self._enforce_policy(self._admin_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
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
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != RAGAdminBackend.SERVER:
            raise ValueError(
                "Server retrieval-admin backend is required for this RAG admin operation."
            )
        self._enforce_policy(self._admin_action_id(normalized_mode, "configure"))
        service = self._service_for_mode(normalized_mode)
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
        self._enforce_policy(self._admin_action_id(normalized_mode, "list"))
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
        self._enforce_policy(self._admin_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        detail = await self._maybe_await(service.get_collection_detail(collection_name))
        return normalize_collection_record(normalized_mode.value, detail)

    async def create_collection(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        name: str,
        metadata: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "configure"))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "create_collection", None)
        if not callable(method):
            raise ValueError(
                f"{normalized_mode.value.title()} collection creation is not available yet."
            )
        record = await self._maybe_await(
            method(
                name=name,
                metadata=metadata,
                embedding_model=embedding_model,
                provider=provider,
            )
        )
        return normalize_collection_record(normalized_mode.value, record)

    async def export_collection(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        collection_name: str,
        include_embeddings: bool = True,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "export_collection", None)
        if not callable(method):
            if normalized_mode == RAGAdminBackend.SERVER:
                self._raise_server_collection_export_unsupported()
            raise ValueError(
                f"{normalized_mode.value.title()} collection export is not available yet."
            )
        options: dict[str, Any] = {"include_embeddings": include_embeddings}
        if limit is not None:
            options["limit"] = limit
        if offset is not None:
            options["offset"] = offset
        return dict(await self._maybe_await(method(collection_name, **options)) or {})

    async def delete_collection(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        collection_name: str,
    ) -> None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "configure"))
        service = self._service_for_mode(normalized_mode)
        await self._maybe_await(service.delete_collection(collection_name))

    async def reprocess_media(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: Any,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "reprocess_media", None)
        if not callable(method):
            raise ValueError(
                f"{normalized_mode.value.title()} media reprocess is not available yet."
            )
        result = await self._maybe_await(method(media_id, **options))
        return dict(result or {})

    async def rechunk_legacy_media(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        rag_service: Any = None,
        indexing_db: Any = None,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Launch the re-chunk of older-engine items (task-13, spec §10.4).

        Policy surface: this REUSES the existing ``rag.admin.launch`` verb
        -- exactly what the backfill-shaped trigger already means -- rather
        than adding a fifth ``rag.admin.*`` action (which would require
        editing the registry AND the exact-equality literal test, and whose
        tempting shortcut, the SHARED
        ``DISCOVER_CONFIGURE_TRIGGER_OBSERVE_ACTIONS`` tuple, would
        silently grant the verb to other capabilities).
        """
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._admin_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        method = getattr(service, "rechunk_legacy_media", None)
        if not callable(method):
            raise ValueError(
                f"{normalized_mode.value.title()} legacy re-chunk is not available yet."
            )
        result = await self._maybe_await(
            method(
                rag_service=rag_service,
                indexing_db=indexing_db,
                progress_callback=progress_callback,
            )
        )
        return dict(result or {})

    async def get_media_embeddings_status(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_media_embeddings_service(
            normalized_mode, "Media embedding status"
        )
        self._enforce_policy(self._media_embeddings_action_id("status"))
        return self._with_backend(
            normalized_mode,
            await self._maybe_await(service.get_media_embeddings_status(media_id)),
        )

    async def generate_media_embeddings(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: Any,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_media_embeddings_service(
            normalized_mode, "Media embedding generation"
        )
        self._enforce_policy(self._media_embeddings_action_id("create"))
        return self._with_backend(
            normalized_mode,
            await self._maybe_await(
                service.generate_media_embeddings(media_id, **options)
            ),
        )

    async def generate_media_embeddings_batch(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_media_embeddings_service(
            normalized_mode, "Media embedding batch generation"
        )
        self._enforce_policy(self._media_embeddings_action_id("create"))
        return self._with_backend(
            normalized_mode,
            await self._maybe_await(service.generate_media_embeddings_batch(**options)),
        )

    async def search_media_embeddings(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_media_embeddings_service(
            normalized_mode, "Media embedding search"
        )
        self._enforce_policy(self._media_embeddings_action_id("search"))
        return self._with_backend(
            normalized_mode,
            await self._maybe_await(service.search_media_embeddings(**options)),
        )

    async def delete_media_embeddings(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        media_id: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_media_embeddings_service(
            normalized_mode, "Media embedding deletion"
        )
        self._enforce_policy(self._media_embeddings_action_id("delete"))
        return self._with_backend(
            normalized_mode,
            await self._maybe_await(service.delete_media_embeddings(media_id)),
        )

    async def get_media_embedding_job(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        job_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_media_embeddings_service(
            normalized_mode, "Media embedding job detail"
        )
        self._enforce_policy(self._media_embedding_jobs_action_id("detail"))
        return self._with_backend(
            normalized_mode,
            await self._maybe_await(service.get_media_embedding_job(job_id)),
        )

    async def list_media_embedding_jobs(
        self,
        *,
        mode: RAGAdminBackend | str | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._server_media_embeddings_service(
            normalized_mode, "Media embedding job listing"
        )
        self._enforce_policy(self._media_embedding_jobs_action_id("list"))
        return self._with_backend(
            normalized_mode,
            await self._maybe_await(service.list_media_embedding_jobs(**options)),
        )
