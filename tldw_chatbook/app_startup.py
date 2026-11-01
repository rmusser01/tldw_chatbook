"""App startup and footer wiring, delegated after composition begins.

The app retains its original methods and owns all timers, services and state.
Named dependencies preserve its live module patch routes without importing
screens, constructing new owners or changing the pure boot-worker policy.
"""

from __future__ import annotations

from typing import Any


def _active_footer_status(
    self: Any, *, footer_type: Any, query_error: Any, screen_stack_error: Any
) -> Any:
    """The visible screen's footer, falling back to the default-screen one.

    Every ``BaseAppScreen`` mounts its own ``AppFooterStatus`` (task-264),
    so per-tick updates (DB sizes, word/token counts) must resolve the
    currently active screen's instance rather than the cached
    ``_db_size_status_widget`` acquired once from the default screen at
    startup -- that cached widget is occluded as soon as any screen is
    pushed. The cache is kept as a fallback for the brief window before
    the first screen is pushed (or if the active screen has no footer
    for some reason).

    ``ScreenStackError`` is caught alongside ``QueryError`` because this
    runs from ``set_interval`` timers (DB-size/token ticks) that can fire
    during app shutdown, after the screen stack has already been drained
    -- ``App.screen`` raises then, and the fallback cache is the right
    answer (its update methods are themselves teardown-safe no-ops).
    """
    try:
        return self.screen.query_one(footer_type)
    except (screen_stack_error, query_error):
        return self._db_size_status_widget


def _schedule_deferred_startup_work(
    self: Any,
    *,
    db_size_delay: float,
    audio_delay: float,
    collections_delay: float,
    workspace_delay: float,
    screen_delay: float,
    notes_delay: float,
    install_notes_facades: Any,
) -> Any:
    """Start nonessential services after the first interactive UI frame."""

    # TASK-22215: the boot-time thread fleet starts here, under the
    # explicit order/concurrency policy in `Utils/boot_worker_policy.py`,
    # rather than all at once (and rather than partly from `on_mount`,
    # ahead of first paint, which is where the two FTS backfills used to
    # start).
    self._start_staggered_boot_workers()
    self.set_timer(
        db_size_delay,
        self._schedule_footer_status_updates,
    )
    self.set_timer(
        audio_delay,
        self._start_deferred_audio_service_initialization,
    )
    self.set_timer(
        collections_delay,
        self._deferred_wire_collections_capture_services,
    )
    # Workspace agent provisioning (task-8): best-effort hook attach +
    # startup backfill, deferred past `_ui_ready` so the provisioning
    # module stays out of the UI-ready module census (ADR-097).
    self.set_timer(
        workspace_delay,
        self._deferred_wire_workspace_agent_provisioning,
    )
    self.set_timer(
        screen_delay,
        self._schedule_screen_preimport,
    )
    self.schedule_media_cleanup()
    self._create_deferred_startup_task(
        self._reconcile_interrupted_subscription_work(),
        name="deferred_subscription_interrupt_reconcile",
    )
    coordinator = getattr(
        self,
        "citation_artifact_ownership_coordinator",
        None,
    )
    if coordinator is not None and coordinator.writes_enabled:
        self._create_deferred_startup_task(
            self._reconcile_citation_artifact_ownership(),
            name="deferred_citation_artifact_reconciliation",
        )
    migration = getattr(
        self,
        "citation_legacy_migration_service",
        None,
    )
    if migration is not None and migration.writes_enabled:  # key: first use
        self._create_deferred_startup_task(
            self._migrate_legacy_citations_idle_unit(),
            name="deferred_legacy_citation_migration",
        )
    self._schedule_launch_wake()
    # Schedule Notes/Sync last.  Earlier placement let the nominal 0.1 s
    # delay expire while the remaining synchronous setup below it was
    # still running, so its import graph could win the race against the
    # first-interactive-frame census on slower starts.
    install_notes_facades(self)
    self.set_timer(
        notes_delay,
        self._deferred_wire_notes_sync_services,
    )


def _schedule_footer_status_updates(
    self: Any, *, footer_type: Any, query_error: Any, db_size_delay: float
) -> Any:
    """Wire the status-line DB-size updates after UI readiness.

    task-21133: this used to arm a second pair of timers -- a 0.5 s
    one-shot and a 10 s interval -- for a token counter whose entire
    consumer surface task-17653 removed. Nothing armed the footer's
    ``#footer-token-count`` chip any more (``BaseAppScreen`` composes
    every ``AppFooterStatus`` with ``show_token_count=False``, and that
    is the only construction site in the package), so each tick resolved
    the active footer, ran three ``query_one`` selectors that no live
    screen composes, and threw the answer away in a debug log. The
    interval, its handle, and the whole chain behind it are gone; the
    DB-size timers below are unchanged.
    """

    def record_footer_timer(name: str) -> None:
        record_timer = getattr(self, "_record_footer_timer_created", None)
        try:
            if callable(record_timer):
                record_timer(name)
                return
            monitor = getattr(self, "ui_responsiveness_monitor", None)
            if monitor is not None:
                monitor.record_timer_created(name)
        except Exception:  # noqa: BLE001 -- timer telemetry must not abort startup
            return

    try:
        # The cache is only a pre-first-screen fallback: per-tick updates
        # resolve the ACTIVE screen's footer via `_active_footer_status`.
        # Splash and the first-run wizard mount no AppFooterStatus, so a
        # miss here must not abort timer setup (task-2721: it previously
        # logged two tracebacks per fresh install and left the DB-size
        # timers never started for the whole session).
        try:
            self._db_size_status_widget = self.query_one(footer_type)
            self.loguru_logger.info("AppFooterStatus widget instance acquired.")
        except query_error:
            self._db_size_status_widget = None
            self.loguru_logger.debug(
                "Active screen has no AppFooterStatus; footer timers start "
                "anyway and each tick resolves the active screen's footer."
            )

        self.set_timer(
            db_size_delay,
            self.update_db_sizes,
        )
        self.db_status_manager.start_periodic_updates(120)
        record_footer_timer("footer-db-size-periodic")
        self.loguru_logger.info(
            "DB size update timer started for the shell status line (interval: 2 minutes)."
        )
    except Exception as e_db_size:  # noqa: BLE001 -- preserve footer setup recovery
        self.loguru_logger.opt(exception=True).error(
            f"Error setting up DB size indicator for the shell status line: {e_db_size}",
        )
