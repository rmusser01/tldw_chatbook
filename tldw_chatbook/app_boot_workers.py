"""Finite App-owned native boot work; policy and shutdown stay with the App."""

from __future__ import annotations

from typing import Any, Callable


def _start_subscriptions_fts_backfill_worker(
    self: Any, path: Any, admission: Callable[..., bool]
) -> Any:
    """Keep subscription admission and backfill in one finite App-owned call."""

    async def backfill() -> Any:
        from contextvars import Context

        from textual.worker import active_worker, get_current_worker

        from .Backup_Recovery.bootstrap import RecoveryRequired
        from .Chat.console_preparation_reads import run_preparation_read

        gate = getattr(self, "_boot_worker_gate", None)
        if (
            self._exit
            or getattr(self, "_shutting_down", False)
            or (gate is not None and gate.is_closed)
        ):
            return None
        worker = get_current_worker()
        admission_error: Exception | None = None

        def invoke() -> Any:
            def call() -> Any:
                nonlocal admission_error
                token = active_worker.set(worker)
                try:
                    try:
                        allowed = admission(("db.subscriptions",), path)
                    except Exception as error:
                        admission_error = error
                        return None
                    if allowed:
                        # Resolve after admission, preserving custom results
                        # and the body's own fresh execution_scope/error policy.
                        return self._backfill_subscription_items_fts()
                    return None
                finally:
                    active_worker.reset(token)

            return Context().run(call)

        result = await run_preparation_read(
            invoke,
            creator=self,
            session_id=None,
            # Shared finite boot owner: both App shutdown entries already
            # drain this set before workflow and database creators retire.
            reads=self._actor_pack_recovery_reads,
            require_current=lambda: None,
        )
        if admission_error is not None:
            key = "subscriptions_fts_backfill"
            if (
                isinstance(admission_error, RecoveryRequired)
                and str(admission_error) == "storage_locally_paused"
                and gate is not None
            ):
                if self._boot_worker_handles.get(key) is worker:
                    self._boot_worker_handles.pop(key)
                gate.defer((key,))
                self._arm_boot_worker_reconcile()
            else:
                self.loguru_logger.opt(exception=admission_error).warning(
                    f"Staggered boot worker {key!r} failed to start"
                )
        return result

    return self.run_worker(
        backfill,
        name="_backfill_subscription_items_fts",
        group="subscriptions-fts-backfill",
        exclusive=True,
        exit_on_error=True,
    )
