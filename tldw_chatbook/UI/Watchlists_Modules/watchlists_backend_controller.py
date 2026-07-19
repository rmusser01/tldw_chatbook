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

    async def list_items(self, *, runtime_backend: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        """List watchlist content items through the scope service."""
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.list_items(runtime_backend=backend, **kwargs)
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
        """Cancel an in-progress watchlist run.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            run_id: Identifier of the run to cancel.

        Returns:
            Cancellation result metadata.
        """
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

    async def preview_source(self, *, runtime_backend: str | None = None, source_config: dict[str, Any]) -> dict[str, Any]:
        """Preview a watchlist source through the scope service.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            source_config: Source configuration (URL, parser, etc.).

        Returns:
            Preview result containing items and log text.
        """
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.preview_source(runtime_backend=backend, source_config=source_config)
        )
        return dict(result)

    async def check_now(self, *, runtime_backend: str | None = None, source_id: Any) -> dict[str, Any]:
        """Trigger an immediate check for a watchlist source.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            source_id: Identifier of the source to check.

        Returns:
            Run metadata for the launched check.
        """
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.check_now(runtime_backend=backend, source_id=source_id)
        )
        return dict(result)

    async def import_opml(self, *, runtime_backend: str | None = None, xml_text: str) -> dict[str, Any]:
        """Import watchlist sources from an OPML document.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            xml_text: Raw OPML XML string.

        Returns:
            Summary with the number of created sources and their records.
        """
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.import_opml(runtime_backend=backend, xml_text=xml_text)
        )
        return dict(result)

    async def export_opml(self, *, runtime_backend: str | None = None) -> str:
        """Export watchlist sources as an OPML document.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).

        Returns:
            OPML XML string for the retrieved sources.
        """
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.export_opml(runtime_backend=backend)
        )
        return str(result)

    async def list_alert_rules(self, *, runtime_backend: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.list_alert_rules(runtime_backend=backend, **kwargs)
        )
        return [dict(item) for item in list(result or [])]

    async def save_alert_rule(self, *, runtime_backend: str | None = None, payload: dict[str, Any]) -> dict[str, Any]:
        """Create or update an alert rule through the scope service.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).
            payload: Alert rule fields. Presence of ``id`` or ``rule_id``
                selects the update path.

        Returns:
            Created or updated alert rule record.
        """
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
