from __future__ import annotations

import inspect
from typing import Any


class WatchlistsBackendController:
    """Route watchlist operations to the active local/server authority."""

    def __init__(
        self,
        *,
        app_instance: Any,
        scope_service: Any,
        server_service: Any,
        notification_dispatch_service: Any = None,
    ) -> None:
        self.app_instance = app_instance
        self.scope_service = scope_service
        self.server_service = server_service
        self.notification_dispatch_service = notification_dispatch_service

    @staticmethod
    def _normalize_backend(runtime_backend: Any) -> str:
        return str(runtime_backend or "local").strip().lower() or "local"

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def list_sources(self, *, runtime_backend: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.list_watch_items(runtime_backend=backend, **kwargs)
        )
        return [dict(item) for item in list(result or [])]

    async def create_source(self, *, runtime_backend: str | None = None, payload: dict[str, Any]) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.create_watch_item(runtime_backend=backend, payload=payload)
        )
        return dict(result)

    async def update_source(self, *, runtime_backend: str | None = None, item_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.update_watch_item(runtime_backend=backend, item_id=item_id, payload=payload)
        )
        return dict(result)

    async def delete_source(self, *, runtime_backend: str | None = None, item_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.delete_watch_item(runtime_backend=backend, item_id=item_id)
        )
        return dict(result)

    async def list_runs(self, *, runtime_backend: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.list_runs(runtime_backend=backend, **kwargs)
        )
        return [dict(item) for item in list(result or [])]

    async def get_run(self, *, runtime_backend: str | None = None, run_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.get_run(runtime_backend=backend, run_id=run_id)
        )
        return dict(result)

    async def observe_run(self, *, runtime_backend: str | None = None, run_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.observe_run(runtime_backend=backend, run_id=run_id)
        )
        return dict(result)

    async def cancel_run(self, *, runtime_backend: str | None = None, run_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.cancel_run(runtime_backend=backend, run_id=run_id)
        )
        return dict(result)

    async def launch_run(self, *, runtime_backend: str | None = None, source_id: Any = None) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.launch_run(runtime_backend=backend, source_id=source_id)
        )
        return dict(result)

    async def list_alert_rules(self, *, runtime_backend: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.list_alert_rules(runtime_backend=backend, **kwargs)
        )
        return [dict(item) for item in list(result or [])]

    async def save_alert_rule(self, *, runtime_backend: str | None = None, payload: dict[str, Any]) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.save_alert_rule(runtime_backend=backend, payload=payload)
        )
        return dict(result)

    async def delete_alert_rule(self, *, runtime_backend: str | None = None, rule_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.delete_alert_rule(runtime_backend=backend, rule_id=rule_id)
        )
        return dict(result)

    def list_unsupported_capabilities(self, *, runtime_backend: str | None = None) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        method = getattr(self.scope_service, "list_unsupported_capabilities", None)
        if callable(method):
            return list(method(runtime_backend=backend))
        return []
