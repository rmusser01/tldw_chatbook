"""Backend-aware controller for the subscription window watchlist slice."""

from __future__ import annotations

import inspect
from typing import Any

from textual.widgets import Checkbox

from ...Subscriptions.textual_scheduler_worker import SubscriptionSchedulerWorker


class SubscriptionBackendController:
    """Own the runtime-aware behavior delegated out of SubscriptionWindow."""

    def __init__(
        self,
        *,
        window: Any,
        app_instance: Any,
        scope_service: Any = None,
        notification_dispatch_service: Any = None,
    ) -> None:
        self.window = window
        self.app_instance = app_instance
        self.scope_service = scope_service
        self.notification_dispatch_service = notification_dispatch_service
        self.last_loaded_backend: str | None = None

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def refresh_backend_view(self, *, runtime_backend: str) -> None:
        normalized_backend = str(runtime_backend or "local").strip().lower() or "local"
        await self.stop_active_backend_workers()

        if normalized_backend == "local":
            await self._ensure_local_scheduler()
            self.window._clear_local_only_state(tab_id="review")
            await self._maybe_await(self.window.load_new_items())
        else:
            self.window.scheduler_worker = None
            self.window._render_local_only_state(tab_id="review", message="Local-only in this slice.")

        await self._load_watch_items(runtime_backend=normalized_backend)
        self.last_loaded_backend = normalized_backend

    async def stop_active_backend_workers(self) -> None:
        worker = getattr(self.window, "scheduler_worker", None)
        if worker is None:
            return

        stop_method = getattr(worker, "stop_scheduler", None)
        if callable(stop_method):
            await self._maybe_await(stop_method())
        if hasattr(worker, "is_running"):
            worker.is_running = False
        self.window.scheduler_worker = None

    def snapshot_shell_state(self) -> dict[str, Any]:
        return {
            "runtime_backend": self.window._runtime_backend(),
            "last_loaded_backend": self.last_loaded_backend,
            "scheduler_active": self.window.scheduler_worker is not None,
        }

    async def delete_watch_item(self, item_id: str) -> dict[str, Any]:
        runtime_backend = self.window._runtime_backend()
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")

        result = await self._maybe_await(
            self.scope_service.delete_watch_item(
                runtime_backend=runtime_backend,
                item_id=item_id,
            )
        )

        if runtime_backend == "server" and self.notification_dispatch_service is not None:
            self.notification_dispatch_service.dispatch(
                app=self.app_instance,
                category="watchlists",
                title="Watchlist source deleted",
                message="Server source deleted within restore window.",
                source_backend="server",
                source_entity_id=str(result.get("source_id", "")),
                source_entity_kind="watchlist_source",
                payload={
                    "source_id": result.get("source_id"),
                    "restore_window_seconds": result.get("restore_window_seconds"),
                    "restore_expires_at": result.get("restore_expires_at"),
                },
            )
        return dict(result)

    async def save_watch_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        runtime_backend = self.window._runtime_backend()
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")

        is_update = payload.get("id") not in (None, "") or payload.get("source_id") not in (None, "")
        result = await self._maybe_await(
            self.scope_service.save_watch_item(
                runtime_backend=runtime_backend,
                payload=payload,
            )
        )

        if self.notification_dispatch_service is not None:
            entity_kind = str(result.get("entity_kind") or ("watchlist_source" if runtime_backend == "server" else "subscription"))
            action = "updated" if is_update else "created"
            noun = "Watchlist source" if entity_kind == "watchlist_source" else "Subscription"
            self.notification_dispatch_service.dispatch(
                app=self.app_instance,
                category="watchlists",
                title=f"{noun} {action}",
                message=f"{noun} {action}.",
                source_backend=runtime_backend,
                source_entity_id=str(result.get("id") or result.get("source_id") or ""),
                source_entity_kind=entity_kind,
                payload={"action": action, "source_id": result.get("source_id")},
            )

        return dict(result)

    async def _load_watch_items(self, *, runtime_backend: str) -> None:
        if self.scope_service is None:
            if runtime_backend == "local":
                await self._maybe_await(self.window.refresh_subscription_list())
            return

        payload = await self._maybe_await(
            self.scope_service.list_watch_items(runtime_backend=runtime_backend)
        )
        items = [dict(item) for item in list(payload or [])]
        await self.window._render_watch_item_list(items)

    async def _ensure_local_scheduler(self) -> None:
        if getattr(self.window, "db", None) is None:
            return
        if getattr(self.window, "scheduler_worker", None) is None:
            self.window.scheduler_worker = SubscriptionSchedulerWorker(
                self.app_instance,
                self.window.db,
                max_concurrent=10,
                check_interval=60,
            )

        try:
            enable_scheduler = self.window.query_one("#enable-scheduler", Checkbox)
        except Exception:
            return

        if getattr(enable_scheduler, "value", False):
            self.window.run_worker(self.window.scheduler_worker.start_scheduler())
