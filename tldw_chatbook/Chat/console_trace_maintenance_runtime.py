"""Post-readiness legacy trace maintenance on its existing runtime owner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


async def run_legacy_trace_maintenance(
    self: Any,
    database: Any,
    normalizer_factory: Callable[[], Any],
    *,
    _source: Any,
) -> None:
    def provider_active() -> bool:
        controller = self._chat_controller
        tasks = getattr(controller, "_active_stream_tasks", None)
        return bool(tasks)

    from tldw_chatbook.Chat.console_trace_maintenance import (
        LegacyTraceMaintenance,
    )

    maintenance = LegacyTraceMaintenance(
        database,
        normalizer=normalizer_factory(),
        provider_active=provider_active,
    )
    from tldw_chatbook.Chat.chat_persistence_service import (
        trace_maintenance_work_generation,
    )

    last_provider_activity = _source.time.monotonic()
    last_physical_attempt = 0.0
    last_collected_epoch: int | None = None
    pending_gc_result: Any | None = None
    # PERF-10 (TASK-33269): once a pass finds nothing to normalize and
    # the GC interval has not elapsed, park. Parked, the loop does no
    # database work (each run_batch was a write transaction, an
    # admission and a helper-spawning connection, once a second
    # forever). It wakes when an exchange row is written, and once per
    # GC interval regardless: trace-call state, retention roots,
    # semantic revisions and other processes advance the graph epoch
    # without signalling, and a failed GC attempt must be retried.
    parked = False
    seen_work = trace_maintenance_work_generation()
    while not self._disposed:
        if parked:
            await _source.asyncio.sleep(
                _source.LEGACY_TRACE_MAINTENANCE_PARK_POLL_SECONDS
            )
            if trace_maintenance_work_generation() != seen_work or (
                _source.time.monotonic() - last_physical_attempt
                >= _source.TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS
            ):
                parked = False
            continue
        # TASK-33801: an exchange written since the last pass -- while
        # parked, during that pass or its cleanup -- means this pass has
        # work, so it skips the idle check's extra admission.
        current_work = trace_maintenance_work_generation()
        if current_work != seen_work:
            maintenance.expect_work = True
        seen_work = current_work
        try:
            result = await _source.run_owned_db_call(database, maintenance.run_batch)
        except Exception as exc:  # noqa: BLE001 - retry remains restart-safe
            _source.logger.warning(  # noqa: PLE1205 -- live Loguru owner uses brace formatting
                "legacy trace maintenance paused after {}",
                type(exc).__name__,
            )
            await _source.asyncio.sleep(
                _source.LEGACY_TRACE_MAINTENANCE_RETRY_DELAY_SECONDS
            )
            continue
        if result.logical_complete:
            now = _source.time.monotonic()
            if provider_active():
                last_provider_activity = now
                await _source.asyncio.sleep(1.0)
                continue
            if (
                now - last_physical_attempt
                < _source.TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS
            ):
                parked = True
                continue
            last_physical_attempt = now
            try:
                from tldw_chatbook.Chat.console_trace_maintenance import (
                    PhysicalTraceCompactor,
                    TraceGarbageCollector,
                )
                from tldw_chatbook.Chat.console_trace_models import (
                    new_opaque_id,
                )
                from tldw_chatbook.Chat.console_capture_policy import resolve_trace_compaction_policy

                controller = self._chat_controller
                pause = getattr(
                    controller,
                    "pause_trace_maintenance_dispatch",
                    lambda: None,
                )
                resume = getattr(
                    controller,
                    "resume_trace_maintenance_dispatch",
                    lambda: None,
                )
                app_config = getattr(self._app, "app_config", {}) or {}
                console_config = (
                    app_config.get("console", {})
                    if isinstance(app_config, _source.Mapping)
                    else {}
                )
                controller_idle = getattr(
                    controller,
                    "trace_maintenance_idle_seconds",
                    None,
                )
                collector = TraceGarbageCollector(database)
                current_epoch = await _source.run_owned_db_call(
                    database, collector.current_graph_epoch
                )
                if (
                    pending_gc_result is not None
                    and getattr(pending_gc_result, "marked_epoch", None)
                    != current_epoch
                ):
                    # A deferred physical pass cannot pin a stale logical
                    # graph after admission, settlement or revocation.
                    pending_gc_result = None
                if pending_gc_result is None:
                    if current_epoch == last_collected_epoch:
                        await _source.asyncio.sleep(1.0)
                        continue
                    pending_gc_result = await _source.run_owned_db_call(
                        database,
                        collector.collect,
                        request_id=f"auto-{new_opaque_id()}",
                    )
                    last_collected_epoch = int(
                        getattr(
                            pending_gc_result,
                            "marked_epoch",
                            current_epoch,
                        )
                    )
                compactor = PhysicalTraceCompactor(
                    database,
                    policy=resolve_trace_compaction_policy(console_config),
                    provider_active=provider_active,
                    idle_seconds=(
                        controller_idle
                        if callable(controller_idle)
                        else lambda: max(
                            0.0, _source.time.monotonic() - last_provider_activity
                        )
                    ),
                    pause_dispatch=pause,
                    resume_dispatch=resume,
                    cancel_requested=lambda: self._disposed,
                )
                outcome = await _source.run_owned_db_call(
                    database,
                    compactor.run_after_gc,
                    pending_gc_result,
                )
                if outcome.reason_code == "logical_gc_unavailable":
                    pending_gc_result = None
                    last_collected_epoch = None
                elif outcome.completed or (
                    outcome.reason_code
                    not in _source.TRACE_PHYSICAL_MAINTENANCE_RETRYABLE_REASONS
                ):
                    pending_gc_result = None
            except ImportError:
                # Narrow test doubles may provide only the legacy worker.
                pass
            except Exception as exc:  # noqa: BLE001 - durable retry state
                _source.logger.warning(  # noqa: PLE1205 -- live Loguru owner uses brace formatting
                    "trace physical maintenance paused after {}",
                    type(exc).__name__,
                )
            await _source.asyncio.sleep(1.0)
            continue
        if not result.admitted:
            await _source.asyncio.sleep(1.0)
            continue
        await _source.asyncio.sleep(0)
