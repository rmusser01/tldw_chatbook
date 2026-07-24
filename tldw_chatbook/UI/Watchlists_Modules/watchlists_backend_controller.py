from __future__ import annotations

import inspect
from typing import Any

from loguru import logger


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

    async def delete_run(self, *, runtime_backend: str | None = None, run_id: Any) -> dict[str, Any]:
        """Delete a watchlist run if the backend supports it."""
        backend = self._normalize_backend(runtime_backend)
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        method = getattr(self.scope_service, "delete_run", None)
        if not callable(method):
            raise NotImplementedError("Run deletion is not supported by the current backend.")
        result = await self._maybe_await(method(runtime_backend=backend, run_id=run_id))
        return dict(result)

    async def update_item_status(
        self,
        *,
        runtime_backend: str | None = None,
        item_id: Any,
        status: str,
    ) -> dict[str, Any]:
        """Update the status of a watchlist content item."""
        backend = self._normalize_backend(runtime_backend)
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        for method_name in ("update_item", "update_item_status", "mark_item_status"):
            method = getattr(self.scope_service, method_name, None)
            if callable(method):
                if method_name == "mark_item_status":
                    result = await self._maybe_await(method(item_id=item_id, status=status))
                else:
                    result = await self._maybe_await(
                        method(runtime_backend=backend, item_id=item_id, status=status)
                    )
                return dict(result)
        raise NotImplementedError("Item status updates are not supported by the current backend.")

    async def _safe_list(self, method_name: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Call a scope-service list method if it exists, otherwise return []."""
        if self.scope_service is None:
            return []
        method = getattr(self.scope_service, method_name, None)
        if not callable(method):
            return []
        try:
            result = await self._maybe_await(method(**kwargs))
        except Exception:
            logger.opt(exception=True).debug(
                f"Watchlists overview could not call {method_name}."
            )
            return []
        if isinstance(result, dict):
            return [dict(item) for item in list(result.get("items") or [])]
        return [dict(item) for item in list(result or [])]

    async def get_overview_data(self, *, runtime_backend: str | None = None) -> dict[str, Any]:
        """Return derived metrics for the Watchlists overview dashboard.

        Aggregates counts from sources, items, runs, and alert rules. Keeps
        the query cheap by limiting each list call to 100 records.

        Args:
            runtime_backend: Target backend (``local`` or ``server``).

        Returns:
            Dict with total/active/error source counts, total/new item counts,
            latest run status, recent failed runs, and active alert rule count.
        """
        backend = self._normalize_backend(runtime_backend)
        if self.scope_service is None:
            return {
                "total_sources": 0,
                "active_sources": 0,
                "sources_in_error": 0,
                "total_items": 0,
                "new_items": 0,
                "latest_run_status": "unavailable",
                "failed_runs": [],
                "active_alert_rules": 0,
            }

        sources = await self.list_sources(runtime_backend=backend, limit=100)
        items = await self._safe_list(
            "list_items", runtime_backend=backend, limit=100
        )
        runs = await self._safe_list(
            "list_runs", runtime_backend=backend, limit=100
        )
        rules = await self._safe_list(
            "list_alert_rules", runtime_backend=backend
        )

        total_sources = len(sources)
        active_sources = sum(1 for s in sources if s.get("active"))
        sources_in_error = sum(
            1 for s in sources if str(s.get("status") or "").lower() == "error"
        )
        total_items = len(items)
        new_items = sum(1 for item in items if str(item.get("status") or "").lower() == "new")

        latest_run_status = "unavailable"
        if runs:
            latest = runs[0]
            latest_run_status = str(latest.get("status") or "unknown")

        failed_runs = [
            {
                "id": run.get("id"),
                "source_title": run.get("source_title") or run.get("source_name") or "Untitled",
                "status": run.get("status") or "failed",
                "error_msg": run.get("error_msg") or run.get("error") or "",
            }
            for run in runs
            if str(run.get("status") or "").lower() in {"failed", "error"}
        ][:10]

        active_alert_rules = sum(1 for rule in rules if rule.get("enabled"))

        return {
            "total_sources": total_sources,
            "active_sources": active_sources,
            "sources_in_error": sources_in_error,
            "total_items": total_items,
            "new_items": new_items,
            "latest_run_status": latest_run_status,
            "failed_runs": failed_runs,
            "active_alert_rules": active_alert_rules,
        }

    def list_unsupported_capabilities(self, *, runtime_backend: str | None = None) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        method = getattr(self.scope_service, "list_unsupported_capabilities", None)
        if callable(method):
            return list(method(runtime_backend=backend))
        return []
