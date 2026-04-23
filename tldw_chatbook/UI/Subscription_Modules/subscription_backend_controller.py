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
        server_notifications_scope_service: Any = None,
        notification_dispatch_service: Any = None,
    ) -> None:
        self.window = window
        self.app_instance = app_instance
        self.scope_service = scope_service
        self.server_notifications_scope_service = server_notifications_scope_service
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
        await self._refresh_control_plane(runtime_backend=normalized_backend)
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

    async def restore_watch_item(self, item_id: str) -> dict[str, Any]:
        runtime_backend = self.window._runtime_backend()
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")

        result = await self._maybe_await(
            self.scope_service.restore_watch_item(
                runtime_backend=runtime_backend,
                item_id=item_id,
            )
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

    async def _refresh_control_plane(self, *, runtime_backend: str) -> None:
        if runtime_backend != "server":
            message = "Server watchlist jobs, runs, and alert rules are remote-only here. Use the local subscriptions scheduler, review queue, and notifications inbox for offline watchlist operations."
            self.window._render_local_only_state(tab_id="watchlist-jobs", message=message)
            self.window._render_local_only_state(tab_id="watchlist-runs", message=message)
            self.window._render_local_only_state(tab_id="watchlist-alert-rules", message=message)
            reminder_message = "Server reminders and notification feed are remote-only. Use the local Notifications inbox for offline client-owned notifications."
            self.window._render_local_only_state(tab_id="server-reminders", message=reminder_message)
            self.window._render_local_only_state(tab_id="server-feed", message=reminder_message)
            await self.window._render_watchlist_jobs([])
            await self.window._render_watchlist_runs([])
            await self.window._render_watchlist_alert_rules([])
            await self.window._render_server_reminders([])
            await self.window._render_server_feed([])
            return

        self.window._clear_local_only_state(tab_id="watchlist-jobs")
        self.window._clear_local_only_state(tab_id="watchlist-runs")
        self.window._clear_local_only_state(tab_id="watchlist-alert-rules")
        self.window._clear_local_only_state(tab_id="server-reminders")
        self.window._clear_local_only_state(tab_id="server-feed")
        await self.load_watchlist_jobs()
        await self.load_watchlist_runs()
        await self.load_watchlist_alert_rules()
        await self.load_server_reminders()
        await self.load_server_feed()

    async def load_watchlist_jobs(self) -> dict[str, Any]:
        if self.scope_service is None:
            return {"items": [], "total": 0}
        payload = await self._maybe_await(
            self.scope_service.list_jobs(runtime_backend=self.window._runtime_backend())
        )
        items = [dict(item) for item in list((payload or {}).get("items") or [])]
        await self.window._render_watchlist_jobs(items)
        return dict(payload or {"items": [], "total": 0})

    async def save_watchlist_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        result = await self._maybe_await(
            self.scope_service.save_job(runtime_backend=self.window._runtime_backend(), payload=payload)
        )
        return dict(result)

    async def delete_watchlist_job(self, job_id: str) -> dict[str, Any]:
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        result = await self._maybe_await(
            self.scope_service.delete_job(runtime_backend=self.window._runtime_backend(), job_id=job_id)
        )
        return dict(result)

    async def restore_watchlist_job(self, job_id: str) -> dict[str, Any]:
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        result = await self._maybe_await(
            self.scope_service.restore_job(runtime_backend=self.window._runtime_backend(), job_id=job_id)
        )
        return dict(result)

    async def trigger_watchlist_job(self, job_id: str) -> dict[str, Any]:
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        result = await self._maybe_await(
            self.scope_service.trigger_job(runtime_backend=self.window._runtime_backend(), job_id=job_id)
        )
        return dict(result)

    async def load_watchlist_runs(self) -> dict[str, Any]:
        if self.scope_service is None:
            return {"items": [], "total": 0}
        payload = await self._maybe_await(
            self.scope_service.list_runs(runtime_backend=self.window._runtime_backend())
        )
        items = [dict(item) for item in list((payload or {}).get("items") or [])]
        await self.window._render_watchlist_runs(items)
        return dict(payload or {"items": [], "total": 0})

    async def get_watchlist_run_detail(self, run_id: str) -> dict[str, Any]:
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        result = await self._maybe_await(
            self.scope_service.get_run_detail(runtime_backend=self.window._runtime_backend(), run_id=run_id)
        )
        return dict(result)

    async def cancel_watchlist_run(self, run_id: str) -> dict[str, Any]:
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        result = await self._maybe_await(
            self.scope_service.cancel_run(runtime_backend=self.window._runtime_backend(), run_id=run_id)
        )
        return dict(result)

    async def load_watchlist_alert_rules(self) -> dict[str, Any]:
        if self.scope_service is None:
            return {"items": [], "total": 0}
        payload = await self._maybe_await(
            self.scope_service.list_alert_rules(runtime_backend=self.window._runtime_backend())
        )
        items = [dict(item) for item in list((payload or {}).get("items") or [])]
        await self.window._render_watchlist_alert_rules(items)
        return dict(payload or {"items": [], "total": 0})

    async def save_watchlist_alert_rule(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        result = await self._maybe_await(
            self.scope_service.save_alert_rule(runtime_backend=self.window._runtime_backend(), payload=payload)
        )
        return dict(result)

    async def delete_watchlist_alert_rule(self, rule_id: str) -> dict[str, Any]:
        if self.scope_service is None:
            raise ValueError("Watchlist scope service is unavailable.")
        result = await self._maybe_await(
            self.scope_service.delete_alert_rule(runtime_backend=self.window._runtime_backend(), rule_id=rule_id)
        )
        return dict(result)

    async def load_server_reminders(self) -> dict[str, Any]:
        if self.server_notifications_scope_service is None:
            return {"items": [], "total": 0}
        payload = await self._maybe_await(
            self.server_notifications_scope_service.list_reminders(runtime_backend=self.window._runtime_backend())
        )
        items = [dict(item) for item in list((payload or {}).get("items") or [])]
        await self.window._render_server_reminders(items)
        return dict(payload or {"items": [], "total": 0})

    async def save_server_reminder(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.server_notifications_scope_service is None:
            raise ValueError("Server notification scope service is unavailable.")
        result = await self._maybe_await(
            self.server_notifications_scope_service.save_reminder(
                runtime_backend=self.window._runtime_backend(),
                payload=payload,
            )
        )
        return dict(result)

    async def delete_server_reminder(self, task_id: str) -> dict[str, Any]:
        if self.server_notifications_scope_service is None:
            raise ValueError("Server notification scope service is unavailable.")
        result = await self._maybe_await(
            self.server_notifications_scope_service.delete_reminder(
                runtime_backend=self.window._runtime_backend(),
                task_id=task_id,
            )
        )
        return dict(result)

    async def load_server_feed(self) -> dict[str, Any]:
        if self.server_notifications_scope_service is None:
            return {"items": [], "total": 0}
        payload = await self._maybe_await(
            self.server_notifications_scope_service.list_feed(runtime_backend=self.window._runtime_backend())
        )
        items = [dict(item) for item in list((payload or {}).get("items") or [])]
        await self.window._render_server_feed(items)
        return dict(payload or {"items": [], "total": 0})

    async def mark_server_notification_read(self, notification_id: str) -> dict[str, Any]:
        if self.server_notifications_scope_service is None:
            raise ValueError("Server notification scope service is unavailable.")
        result = await self._maybe_await(
            self.server_notifications_scope_service.mark_notification_read(
                runtime_backend=self.window._runtime_backend(),
                notification_id=notification_id,
            )
        )
        return dict(result)

    async def dismiss_server_notification(self, notification_id: str) -> dict[str, Any]:
        if self.server_notifications_scope_service is None:
            raise ValueError("Server notification scope service is unavailable.")
        result = await self._maybe_await(
            self.server_notifications_scope_service.dismiss_notification(
                runtime_backend=self.window._runtime_backend(),
                notification_id=notification_id,
            )
        )
        return dict(result)

    async def snooze_server_notification(self, notification_id: str, *, minutes: int = 30) -> dict[str, Any]:
        if self.server_notifications_scope_service is None:
            raise ValueError("Server notification scope service is unavailable.")
        result = await self._maybe_await(
            self.server_notifications_scope_service.snooze_notification(
                runtime_backend=self.window._runtime_backend(),
                notification_id=notification_id,
                minutes=minutes,
            )
        )
        return dict(result)

    async def cancel_server_notification_snooze(self, notification_id: str) -> dict[str, Any]:
        if self.server_notifications_scope_service is None:
            raise ValueError("Server notification scope service is unavailable.")
        result = await self._maybe_await(
            self.server_notifications_scope_service.cancel_notification_snooze(
                runtime_backend=self.window._runtime_backend(),
                notification_id=notification_id,
            )
        )
        return dict(result)

    async def stream_server_feed_events(self, *, after: int = 0):
        if self.server_notifications_scope_service is None:
            raise ValueError("Server notification scope service is unavailable.")
        async for event in self.server_notifications_scope_service.stream_feed_events(
            runtime_backend=self.window._runtime_backend(),
            after=after,
        ):
            yield event

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
