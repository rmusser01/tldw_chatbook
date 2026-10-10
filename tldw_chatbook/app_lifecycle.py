"""TldwCli's lifecycle, shutdown and quit flow: ``LifecycleMixin``.

Moved verbatim from ``app.py`` (TASK-33011, PR-F): recovery and backup
maintenance, the shutdown request and worker settling, the app-owned
lifecycle shutdown (``_shutdown_app_owned_lifecycles``, the ``_shutdown`` and
``_handle_exception`` overrides), artifact share, ``on_unmount``, worker-state
and media cleanup, the workbench actions and workflow session, and the quit
flow (``action_quit`` / ``_confirm_and_quit`` and the quit persistence
helpers). ``TldwCli`` mixes the class in before ``App``, so its overrides of
``App`` methods still win; the two members Textual dispatches by decorator or
that tests require on ``TldwCli`` itself (``on_splash_screen_closed``,
``on_app_focus``) stay in ``app.py``.

Patch the names this code reads (``persist_event``, ``get_cli_setting`` and so
on) HERE: the bodies resolve free names through this module's globals, so a
patch on ``tldw_chatbook.app`` alone no longer reaches them. Where ``app.py``
still reads the same name, patch both modules (``Tests/app_module_patches.py``).
``Tests/Architecture/test_app_extracted_patch_targets.py`` fails on an
app-module patch that can only have been meant for code that moved out.
"""

import asyncio
import contextlib
import inspect
import logging
import os
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from loguru import logger
from loguru import logger as loguru_logger
from textual import work
from textual.message_pump import active_message_pump
from textual.worker import Worker, WorkerCancelled, WorkerState

from tldw_chatbook.app_service_wiring import TldwCli  # class proxy (see its docstring)
from tldw_chatbook.Chat.console_runtime import dispose_console_runtime
from tldw_chatbook.Chat.console_settings_durability import (
    ConsoleSettingsDurabilityOwner,
)
from tldw_chatbook.config import (
    get_cli_config_path,
    get_cli_setting,
    persist_cli_config_for_shutdown,
)
from tldw_chatbook.Constants import TAB_ARTIFACTS
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.shell_destinations import get_shell_destination
from tldw_chatbook.Utils.app_shutdown import arm_exit_watchdog, unregister_running_app
from tldw_chatbook.Utils.persistent_diagnostics import persist_event
from tldw_chatbook.Widgets.confirmation_dialog import (
    ConfirmationDialog,
    await_quit_prompt,
    confirm_quit_screens,
    prepare_quit_screens,
    quit_confirmation_screens,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.Chunking.lab_coordinator import LabCoordinator
    from tldw_chatbook.Workflows.session import WorkflowSession


def retire_dead_pump(
    app: Any, pump: Any, frames: Sequence[tuple[str, str, int | None]]
) -> str | None:
    """Load the original recovery helper only when handling a pump error."""
    from tldw_chatbook.app_keep_alive import retire_dead_pump as retire

    return retire(app, pump, frames)


def keep_alive_notice(
    site: tuple[str, str, int | None], pump: Any, raised: BaseException, kind: str
) -> str:
    """Preserve the lifecycle alias without loading error recovery at boot."""
    from tldw_chatbook.app_keep_alive import keep_alive_notice as notice

    return notice(site, pump, raised, kind)


DEFERRED_MEDIA_CLEANUP_DELAY_SECONDS = 5.0

# task-19561: how long a cancelled worker gets to settle before shutdown
# stops waiting on it and says so. Replaces a flat `asyncio.sleep(0.1)` that
# waited on nothing in particular. Sized to be invisible on a quiet exit
# (the wait ends the moment the last worker finishes) while still bounding a
# thread worker that will not notice cancellation at all.
WORKER_CANCELLATION_GRACE_SECONDS = 3.0

# TASK-1240. The `component` this module passes to `persist_event`. It is a
# bounded metadata token (`persist_event` raises `ValueError` otherwise) and is
# used raw to build the diagnostics logger name, so the four emit sites in this
# file must agree on one spelling. Private to `app.py`: every event emitted
# here belongs to the application lifecycle.
_DIAGNOSTICS_COMPONENT_APP = "app"

# TASK-32533. The three `textual.message_pump` frames that mean "one pump's own
# handler raised": `_dispatch_message` (a message handler, and the compose/mount
# dispatch in `_pre_process` -- the P0's own path), `_flush_next_callbacks`
# (`call_after_refresh` / `call_later`) and `_process_messages_loop` (the
# `on_idle` dispatch inlined there). Workers, the compositor and the driver
# carry none of these and still exit.
#
# The application's OWN run loop is the exception: it runs through
# `_process_messages_loop` too, so this frame set alone would make run-loop
# errors non-fatal. It is excluded instead by `keep_alive`'s `pump is not self`
# clause. Neither clause works without the other -- do not remove one because
# the other "already covers it".
_PUMP_DISPATCH_FRAMES = frozenset(
    {"_dispatch_message", "_flush_next_callbacks", "_process_messages_loop"}
)

def _exception_frames(error: BaseException) -> list[tuple[str, str, int | None]]:
    """Return ``(module, function, line)`` for each traceback frame, outermost first.

    TASK-32533. Identifiers only -- module ``__name__``, ``co_name`` and the
    line number -- so the result can be persisted through the metadata-only
    diagnostics schema without ever carrying the message or a file path.
    """
    frames: list[tuple[str, str, int | None]] = []
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        frames.append(
            (
                str(frame.f_globals.get("__name__", "")),
                frame.f_code.co_name,
                tb.tb_lineno,
            )
        )
        tb = tb.tb_next
    return frames


class LifecycleMixin:
    """``TldwCli``'s lifecycle, shutdown and quit members (moved from ``app.py``)."""

    @property
    def recovery_service(self):
        """Retain accepted recovery work independently of any navigation view."""
        service = getattr(self, "_recovery_service", None)
        if service is None:
            from .Backup_Recovery.recovery_service import (
                RecoveryService,
                default_control_root,
            )

            service = self._recovery_service = RecoveryService(default_control_root())
        return service

    def action_backup_restore(self, mode: str = "home") -> None:
        """Open recovery (setup's Restore entry passes ``mode="inspect"``) for all profiles."""
        from .UI.Screens.backup_restore_screen import BackupRestoreScreen

        self.push_screen(
            BackupRestoreScreen(
                self.recovery_service, config_paths=(get_cli_config_path(),),
                include_known_profiles=True, initial_mode=mode,
            )
        )

    def request_recovery_restart(
        self, archive: Path | None, target: Path, *, recovery_copies: bool = False
    ) -> Worker[None] | None:
        """Use ordinary guarded shutdown before the CLI starts recovery alone."""
        from .Backup_Recovery.recovery_restart import RecoveryRestart
        from .Backup_Recovery.runtime_maintenance import RuntimeMaintenance

        if not getattr(self, "_recovery_restart_available", False) or self._quit_in_progress:
            self.notify("Open Chatbook from its CLI to continue in recovery mode.", severity="warning")
            return
        try:
            current = self.recovery_service.current()
            if current is not None and current["state"] == "running":
                raise ValueError("recovery_operation_running")
            if RuntimeMaintenance(self).unsaved_editors():
                self.notify("Save or discard unsaved work before continuing in recovery mode.", severity="warning")
                return
            request = RecoveryRestart(archive, target, recovery_copies=recovery_copies)
        except (OSError, ValueError, RuntimeError):
            self.notify("Recovery mode is unavailable while current work is unsettled.", severity="warning")
            return
        self._quit_in_progress = True

        async def quit_for_recovery() -> None:
            try:
                await self._confirm_and_quit()
                if self._shutting_down:
                    self._recovery_restart_request = request
            finally:
                if not self._shutting_down:
                    self._quit_in_progress = False

        quit_flow = quit_for_recovery()
        try:
            return self.run_worker(
                quit_flow,
                group="application-quit",
                exclusive=True,
                exit_on_error=False,
            )
        except RuntimeError:
            quit_flow.close()
            self._quit_in_progress = False
            loguru_logger.warning(
                "Recovery quit worker could not start; staying in the app"
            )
            return None

    async def _shutdown_recovery_service(self) -> asyncio.CancelledError | None:
        """Settle native recovery while the app maintenance monitor is available."""
        service = getattr(self, "_recovery_service", None)
        if service is None:
            return None
        task = getattr(self, "_recovery_service_shutdown_task", None)
        if task is None:
            task = self._recovery_service_shutdown_task = asyncio.create_task(
                asyncio.to_thread(service.close), name="shutdown-recovery-service"
            )
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        task.result()
        return cancellation

    @work(group="recovery-profile-launch")
    async def open_recovery_profile(self, profile_id: str) -> None:
        """Keep the parent terminal suspended until the actual child exits."""
        from textual.app import SuspendNotSupported

        service = self.recovery_service
        current = service.current()
        if current is not None and current["state"] == "running":
            self.notify("Another recovery operation is running.", severity="warning")
            return
        cancellation = None
        failure = None
        try:
            # Keep redraw paused until the terminal resumes; catch service errors
            # inside suspend so synchronous failures also reach its resume step.
            with self.batch_update(), self.suspend():
                try:
                    operation = service.start_open_profile(profile_id)
                    settling = asyncio.create_task(asyncio.to_thread(service.wait, operation))
                    while not settling.done():
                        try:
                            await asyncio.shield(settling)
                        except asyncio.CancelledError as error:
                            cancellation = cancellation or error
                    settling.result()
                except (OSError, RuntimeError, ValueError) as error:
                    failure = error
        except (OSError, RuntimeError, ValueError, SuspendNotSupported) as error:
            failure = error
        if failure is not None:
            self.notify("Profile opening failed: " + service.issue_code(failure), severity="error")
        if cancellation is not None:
            raise cancellation

    def _start_backup_maintenance_monitor(self) -> None:
        """Retain the installed live maintenance monitor for this app lifetime."""
        if getattr(self, "_backup_maintenance_monitor_task", None) is not None:
            return
        from .Backup_Recovery.runtime_maintenance import monitor_app

        self._backup_maintenance_monitor_task = asyncio.create_task(
            monitor_app(self), name="backup-maintenance-monitor"
        )

    async def _stop_backup_maintenance_monitor(self) -> asyncio.CancelledError | None:
        """Join native readmission before app shutdown retires ordinary owners."""
        task = getattr(self, "_backup_maintenance_monitor_task", None)
        if task is None:
            return None
        if not task.done():
            task.cancel()
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if not task.done() or asyncio.current_task().cancelling():
                    cancellation = cancellation or error
        if not task.cancelled():
            task.result()
        self._backup_maintenance_monitor_task = None
        return cancellation

    async def on_shutdown_request(self) -> None:  # Use the imported ShutdownRequest
        logging.info("--- App Shutdown Requested ---")

        # Set shutdown flag to prevent new operations
        self._shutting_down = True

        # TASK-22215: stop admitting staggered boot workers before cancelling
        # the live ones, so a completion arriving mid-teardown cannot start a
        # fresh thread worker behind the cancel sweep.
        self._close_boot_worker_gate("shutdown request")

        # Cancel all active workers first
        await self._cancel_and_settle_workers("shutdown request")

        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

        # --- Stop DB Size Update Timer ---
        self.db_status_manager.stop_periodic_updates()
        self._stop_footer_status_timers()
        self.loguru_logger.info("DB size update timer stopped.")
        # --- End Stop DB Size Update Timer ---

    async def _cancel_and_settle_workers(self, phase: str) -> None:
        """Cancel every live worker and actually wait for them, bounded.

        task-19561. Both shutdown hooks used to cancel their workers and
        then ``await asyncio.sleep(0.1)`` -- a flat wait that is
        simultaneously too long (nothing to wait for on a quiet exit) and
        far too short (a worker mid-``await`` gets one tick, and a thread
        worker gets nothing at all), and which never observed the outcome
        either way. Waiting on the workers themselves is both faster in the
        common case and honest in the uncommon one; the timeout keeps a
        worker that ignores cancellation from turning quit into a hang, and
        says which ones they were.
        """
        try:
            active_workers = [w for w in self.workers if not w.is_finished]
            if not active_workers:
                return
            self.loguru_logger.info(
                f"Cancelling {len(active_workers)} active workers ({phase})"
            )
            for worker in active_workers:
                worker.cancel()
            # `asyncio.wait`, NOT `wait_for(...)`: on expiry `wait_for`
            # cancels what it is waiting on and awaits that cancellation, so
            # anything that does not honour a cancel hangs the very call
            # meant to bound it. `wait` returns the stragglers instead.
            waiters = [asyncio.ensure_future(w.wait()) for w in active_workers]
            _, unsettled = await asyncio.wait(
                waiters, timeout=WORKER_CANCELLATION_GRACE_SECONDS
            )
            for waiter in waiters:
                if waiter.done() and not waiter.cancelled():
                    # WorkerCancelled/WorkerFailed are the expected outcomes
                    # of cancelling at shutdown; retrieve them so they do not
                    # resurface as "exception was never retrieved".
                    waiter.exception()
                else:
                    waiter.cancel()
            if unsettled:
                stragglers = [w.name for w in active_workers if not w.is_finished]
                self.loguru_logger.warning(
                    f"{len(stragglers)} worker(s) did not settle within "
                    f"{WORKER_CANCELLATION_GRACE_SECONDS}s of cancellation "
                    f"({phase}): {stragglers}"
                )
        except Exception as e:
            self.loguru_logger.error(f"Error cancelling workers ({phase}): {e}")

    async def _close_server_context_provider_cached_client(self) -> None:
        server_context_provider = getattr(self, "server_context_provider", None)
        close_cached_client = getattr(
            server_context_provider, "close_cached_client", None
        )
        if callable(close_cached_client):
            await close_cached_client()

    async def _disconnect_local_mcp_client(self) -> None:
        """Best-effort teardown of local MCP client sessions (P5-T6).

        ``local_mcp_control_service.client`` (``LocalMCPControlService.
        client``) stays ``None`` until a local external MCP profile is
        actually connected during this process's lifetime (see
        ``LocalMCPControlService._get_client``'s lazy-init) -- a session-
        free app quit is a no-op here, matching the sibling teardown
        blocks' own guarded style.
        """
        local_mcp_control_service = getattr(self, "local_mcp_control_service", None)
        client = getattr(local_mcp_control_service, "client", None)
        if client is not None and getattr(client, "sessions", None):
            await client.disconnect_all()

    async def _close_local_writing_service(self) -> None:
        """Release the writing suite's held SQLite connections (TASK-21125).

        Peeks the slot rather than reading through any accessor: a service that
        was never wired must not be constructed purely to close it. A close
        failure is logged (type name only) and never allowed to abort the rest
        of unmount.

        Runs on a thread, NOT inline. ``close()`` waits for an autosave still
        running on a worker thread, and a synchronous call here froze the event
        loop for the whole settle timeout (measured: 5.00 s during which a 50 ms
        ticker fired zero times) -- which also starved the very operation it was
        waiting for.
        """
        service = getattr(self, "local_writing_service", None)
        if service is None:
            return
        try:
            await asyncio.to_thread(service.close)
        except Exception as exc:
            self.loguru_logger.error(
                f"Error closing local writing service: {type(exc).__name__}"
            )

    async def _close_local_research_service(self) -> None:
        """Release the research store's held SQLite connections (TASK-21127).

        Peeks the slot rather than reading through any accessor: a service that
        was never wired must not be constructed purely to close it. A close
        failure is logged (type name only) and never allowed to abort the rest
        of unmount.

        Runs on a thread, NOT inline. ``close()`` waits for an operation still
        running on the research backend thread (a run's progress write, say),
        and a synchronous call here would freeze the event loop for the whole
        settle timeout -- which also starves the very operation it is waiting
        for (the TASK-21125 review's MAJOR-3 finding).
        """
        service = getattr(self, "local_research_service", None)
        if service is None:
            return
        close = getattr(service, "close", None)
        if not callable(close):
            return
        try:
            await asyncio.to_thread(close)
        except Exception as exc:
            self.loguru_logger.error(
                f"Error closing local research service: {type(exc).__name__}"
            )

    async def _shutdown_file_notes_session_owner(self) -> None:
        """Settle the process-owned File Notes Git lifecycle exactly once."""
        owner = getattr(self, "file_notes_session_owner", None)
        if owner is None:
            return
        task = getattr(self, "_file_notes_session_owner_shutdown_task", None)
        if task is None:
            task = asyncio.create_task(
                owner.shutdown_async(),
                name="shutdown_file_notes_session_owner",
            )
            self._file_notes_session_owner_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_notes_sync_runtime(self) -> None:
        """Settle the application-owned lasting-sync runtime exactly once."""

        # TASK-21108: a runtime that was never built was never started, so
        # there is nothing to settle -- and reading the lazy property here
        # would construct one purely to shut it down.
        if getattr(self, "_notes_sync_runtime_owner", None) is None:
            return
        task = getattr(self, "_notes_sync_runtime_shutdown_task", None)
        if task is None:
            task = asyncio.create_task(
                self.notes_sync_runtime_owner.shutdown(),
                name="shutdown_notes_sync_runtime",
            )
            self._notes_sync_runtime_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_console_image_edits(self) -> None:
        """Cancel and settle app-owned H3 edits exactly once before teardown."""
        task = self._console_image_edit_shutdown_task
        if task is None:
            task = asyncio.create_task(
                self.console_image_edit_operations.shutdown(),
                name="shutdown_console_image_edits",
            )
            self._console_image_edit_shutdown_task = task
        await asyncio.shield(task)

    async def _flush_persona_buddy_geometry(self) -> None:
        """Close presentation admission and drain the app-owned view."""
        owner = getattr(self, "_persona_buddy_overlay", None)
        if owner is not None:
            await owner.shutdown()

    async def _shutdown_persona_buddy(self) -> None:
        """Drain the app-owned Buddy before profile database teardown.

        Peeks the lazy controller slot (TASK-21103): a controller that was
        never built has nothing to drain, and going through the property
        here could CONSTRUCT one (importing Persona_Visual + PIL) purely to
        shut it down.
        """
        task = self._persona_buddy_shutdown_task
        if task is None:
            controller = self._persona_buddy_controller
            if controller is None:
                return
            # Debounced geometry must land while the controller still
            # accepts writes (TASK-21122).
            await self._flush_persona_buddy_geometry()
            task = asyncio.create_task(
                controller.shutdown(),
                name="shutdown_persona_buddy",
            )
            self._persona_buddy_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_actor_pack_export(self) -> None:
        """Cancel and drain Actor Pack export before profile teardown."""

        controller = getattr(self, "actor_pack_export_controller", None)
        if controller is None:
            return
        task = getattr(self, "_actor_pack_export_shutdown_task", None)
        if task is None:
            task = asyncio.create_task(
                controller.shutdown(),
                name="shutdown_actor_pack_export",
            )
            self._actor_pack_export_shutdown_task = task
        await asyncio.shield(task)

    def _refresh_after_actor_pack_import(self, result: object) -> None:
        """Fence mounted Persona Buddy state after a committed Persona import."""

        if getattr(result, "actor_kind", None) == "persona":
            self.persona_buddy_controller.invalidate_profile()

    async def _shutdown_actor_pack_import(self) -> None:
        """Cancel and drain Actor Pack import before profile teardown."""

        controller = getattr(self, "actor_pack_import_controller", None)
        if controller is None:
            return
        task = getattr(self, "_actor_pack_import_shutdown_task", None)
        if task is None:
            task = asyncio.create_task(
                controller.shutdown(),
                name="shutdown_actor_pack_import",
            )
            self._actor_pack_import_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_console_runtime(self) -> None:
        """Destroy the app-owned Console runtime exactly once, at exit.

        task-15860: the runtime survives every navigation away from
        Console, so the unmount Textual performs at exit is no longer what
        ends it -- this is. `ConsoleRuntime.dispose` runs the permanent
        teardown in the order `ChatScreen.on_unmount` used to:
        `controller.shutdown()`, then `gateway.aclose()`. Idempotent: the
        runtime detaches itself from the app on the way out.
        """
        task = self._console_runtime_shutdown_task
        if task is None:
            runtime = getattr(self, "console_runtime", None)
            view = getattr(runtime, "view", None)
            image = getattr(view, "_image", None)
            if view is not None:
                view._console_chat_tearing_down = True
            if image is not None:
                image._recovered_images_close_admission()

            async def settle_view_and_dispose() -> None:
                from .UI.Console_Modules.view_workers import (
                    capture_console_view_workers,
                    drain_console_view_workers,
                )

                # Preserve the original App manager-wide selected group scope;
                # Runtime.detach_view can precede a retired view's awaited cleanup.
                captured = capture_console_view_workers(self)
                try:
                    await drain_console_view_workers(captured)
                finally:
                    if image is not None:
                        await image._recovered_images_close()
                await dispose_console_runtime(self)

            task = asyncio.create_task(
                settle_view_and_dispose(),
                name="shutdown_console_runtime",
            )
            self._console_runtime_shutdown_task = task
        cancellation = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        task.result()
        if cancellation is not None:
            raise cancellation
        plugin_service = getattr(self, "_plugin_service", None)
        if plugin_service is not None:
            await plugin_service.aclose()

    async def _shutdown_raw_cli_runtime(self) -> None:
        """Disarm and boundedly drain the app-owned raw CLI runtime once."""
        task = self._raw_cli_runtime_shutdown_task
        if task is None:
            task = asyncio.create_task(
                asyncio.to_thread(self.raw_cli_runtime.shutdown),
                name="shutdown_raw_cli_runtime",
            )
            self._raw_cli_runtime_shutdown_task = task
        await asyncio.shield(task)

    async def _run_terminal_session_manager_shutdown(self) -> None:
        """Drain Terminal once, then close remaining parent-owned handles."""
        # Do not instantiate an unused first-use owner merely to shut it down.
        manager = getattr(self, "_terminal_session_manager", None)
        if manager is None:
            return
        try:
            await manager.shutdown(deadline_seconds=5.0)
        finally:
            manager.finalize_shutdown()

    async def _shutdown_terminal_session_manager(self) -> None:
        """Share one cancellation-resistant Terminal shutdown task."""
        task = getattr(self, "_terminal_session_manager_shutdown_task", None)
        if task is None:
            task = asyncio.create_task(
                self._run_terminal_session_manager_shutdown(),
                name="shutdown_terminal_session_manager",
            )
            self._terminal_session_manager_shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_console_settings_durability(self) -> None:
        """Drain admitted settings writes without cancelling thread work.

        The coordinator tasks can be awaiting ``asyncio.to_thread`` writes,
        which cannot be recalled once admitted. Shielding preserves those
        writes if application shutdown is cancelled; ``_shutdown`` retries
        this lifecycle pass and does not dispose the Console runtime until the
        registry is empty.
        """

        owner = getattr(self, "console_settings_durability_owner", None)
        if not isinstance(owner, ConsoleSettingsDurabilityOwner):
            return
        await owner.close_and_drain()

    @property
    def meeting_session_owner(self):
        """App-owned meeting session owner, built on first use.

        Deferred so ``Audio.meeting_owner`` and the session/tap/wav modules it
        pulls in are not resident at ``_ui_ready``: the UI-ready module census
        (``Tests/Performance/test_ui_ready_module_census.py``) ratchets that
        count, and these four modules only matter once someone opens Meetings
        or presses a Console voice control.
        """
        owner = self._meeting_session_owner
        if owner is None:
            from .Audio.meeting_owner import build_meeting_session_owner

            owner = build_meeting_session_owner(self)
            self._meeting_session_owner = owner
        return owner

    @meeting_session_owner.setter
    def meeting_session_owner(self, owner) -> None:
        """Inject an owner (tests use a fake); ``None`` returns to lazy building."""
        self._meeting_session_owner = owner

    def on_chunking_templates_changed(self, event) -> None:
        """Refresh local ingest consumers without changing their selection/defaults."""
        from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
            LibraryIngestCanvas,
        )

        event.stop()
        for screen in self.screen_stack:
            for canvas in screen.query(LibraryIngestCanvas):
                canvas.invalidate_chunk_templates()

    async def get_chunking_lab_coordinator(self) -> "LabCoordinator":
        """Load one local profile owner; failed reads never grant write authority."""
        from tldw_chatbook import config as lab_config
        from tldw_chatbook.Chunking.lab_autosave import AutosaveWriter
        from tldw_chatbook.Chunking.lab_coordinator import LabCoordinator
        from tldw_chatbook.Chunking.lab_runner import LocalPreviewRunner, PreviewLimits
        from tldw_chatbook.DB.Chunking_Lab_DB import CheckpointStore

        lock = getattr(self, "_chunking_lab_owner_lock", None)
        if lock is None:
            lock = self._chunking_lab_owner_lock = asyncio.Lock()
        async with lock:
            directory = await asyncio.to_thread(lab_config.get_user_data_dir)
            profile_key = str(directory)
            owner = getattr(self, "_chunking_lab_coordinator", None)
            if owner is not None and owner.session.profile_key == profile_key:
                return owner
            if owner is not None:
                # Retain the owner if close fails, including its export/retry authority.
                await owner.close()
                self._chunking_lab_coordinator = None
            writer = AutosaveWriter(
                CheckpointStore(directory / "chunking_lab.sqlite3", profile_key)
            )
            runner = LocalPreviewRunner(PreviewLimits())
            try:
                owner = await LabCoordinator.load(profile_key, writer, runner)
            except BaseException:
                try:
                    await writer.close()
                except Exception as cleanup_error:  # noqa: BLE001 - retain the original load failure after releasing writer resources.
                    logger.debug(
                        "Chunking Lab writer cleanup failed: {}",
                        type(cleanup_error).__name__,
                    )
                await runner.close()
                raise
            self._chunking_lab_coordinator = owner
            return owner

    async def _shutdown_app_owned_lifecycles(self) -> None:
        """Drain durable app-owned work before Textual closes screen state."""
        actor_recovery_cancellation = await TldwCli._shutdown_actor_pack_recovery(self)
        await self._shutdown_workflow_session()
        self._mcp_local_config_saves_closed = True
        self._tool_profile_operations_closed = True
        local_config_saves = getattr(self, "_mcp_local_config_saves", None)
        tool_profiles = getattr(self, "_tool_profile_operations", None)
        if tool_profiles is not None:
            tool_profiles.close_admission()
        if local_config_saves is not None:
            await local_config_saves.close_and_drain()
        if tool_profiles is not None:
            # Settle these writes before another owner's failure can advance
            # teardown to the workspace databases used by their final checks.
            await tool_profiles.close_and_drain()
        recovery_cancellation = await TldwCli._shutdown_recovery_service(self)
        monitor_cancellation = await TldwCli._stop_backup_maintenance_monitor(self)
        recovery_cancellation = (
            recovery_cancellation or monitor_cancellation or actor_recovery_cancellation
        )
        workflow_error = None
        workflow_authoring = getattr(self, "_workflow_authoring", None)
        if workflow_authoring is not None:
            try:
                await workflow_authoring.close()
            except (OSError, RuntimeError, sqlite3.Error) as exc:
                workflow_error = exc
        lab_owner = getattr(self, "_chunking_lab_coordinator", None)
        lab_error = None
        if lab_owner is not None:
            try:
                await lab_owner.close()
            except Exception as exc:  # noqa: BLE001 - complete unrelated lifecycle cleanup before re-raising.
                lab_error = exc
        coordinator = getattr(self, "watchlists_operation_coordinator", None)
        if coordinator is not None:
            await coordinator.shutdown()
        await self._shutdown_collections_capture_runtime()
        await self._shutdown_notes_sync_runtime()
        await self._shutdown_actor_pack_import()
        await self._shutdown_actor_pack_export()
        # Console shutdown terminally fences every trusted Buddy producer
        # before Buddy itself closes admission and drains owned work.
        await self._shutdown_raw_cli_runtime()
        await self._shutdown_terminal_session_manager()
        await self._shutdown_console_settings_durability()
        await self._shutdown_console_runtime()
        change_review = getattr(self, "change_review_consent_service", None)
        if change_review is not None:
            await asyncio.to_thread(change_review.shutdown, timeout=1.0)
        await self._shutdown_persona_buddy()
        snapshot_owner = getattr(self, "_llamacpp_snapshot_service", None)
        if snapshot_owner is not None:
            await snapshot_owner.shutdown()
            snapshot_setup = getattr(self, "_llamacpp_snapshot_setup_task", None)
            if snapshot_setup is not None:
                await asyncio.shield(snapshot_setup)
        coordinator = getattr(self, "_audio_cpp_artifact_lease_coordinator", None)
        if coordinator is not None:
            await coordinator.shutdown()
        await self.audio_cpp_model_install_owner.shutdown()
        meeting_session_owner = self._meeting_session_owner
        if meeting_session_owner is not None:
            # Only an owner that was actually built can hold a live meeting.
            await asyncio.to_thread(meeting_session_owner.shutdown)
        await self._shutdown_console_image_edits()
        await self._shutdown_file_notes_session_owner()
        if recovery_cancellation is not None:
            raise recovery_cancellation
        if lab_error is not None:
            raise lab_error
        if workflow_error is not None:
            raise workflow_error

    async def _shutdown(self) -> None:
        """Settle app-owned durable work before Textual closes screens."""
        # App.exit normally closes mount admission first. Direct shutdown
        # (including run_test teardown) must use that same Textual fence so
        # queued rebuilds cannot register children after message pumps stop.
        self._exit = True
        # Ordinary Quit reaches these drains before on_unmount. Use the same
        # process-owned, idempotent watchdog here so the drains are bounded too.
        arm_exit_watchdog(reason="app shutdown")
        cancellation: asyncio.CancelledError | None = None
        owner_error: BaseException | None = None
        shutdown_task = asyncio.current_task()
        cancellation_requests = (
            shutdown_task.cancelling() if shutdown_task is not None else 0
        )
        while True:
            try:
                await self._shutdown_app_owned_lifecycles()
            except asyncio.CancelledError as error:
                next_cancellation_requests = (
                    shutdown_task.cancelling() if shutdown_task is not None else 0
                )
                if next_cancellation_requests > cancellation_requests:
                    cancellation = cancellation or error
                    cancellation_requests = next_cancellation_requests
                    continue
                owner_error = error
            except BaseException as error:
                owner_error = error
            break

        shutdown_error: BaseException | None = None
        try:
            await super()._shutdown()
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
        except BaseException as error:
            shutdown_error = error

        if shutdown_error is not None:
            if owner_error is not None:
                shutdown_error.add_note(
                    "App-owned lifecycle shutdown also failed before "
                    "Textual screen teardown"
                )
            if cancellation is not None:
                shutdown_error.add_note(
                    "Application shutdown cancellation was also requested"
                )
            raise shutdown_error
        if owner_error is not None:
            if cancellation is not None:
                owner_error.add_note(
                    "Application shutdown cancellation was delayed while "
                    "preserving the lifecycle shutdown failure"
                )
            raise owner_error
        if cancellation is not None:
            raise cancellation

    def _handle_exception(self, error: Exception) -> None:
        """Record the crash site, then keep the screen alive if it is survivable.

        TASK-1240. Names the exception class only -- never the message, which is
        caller-supplied text and may quote user or model content.

        TASK-32533 adds the raising frame to that record and stops calling
        super() for the one case where the default (exit the whole app) is worse
        than the bug: an exception raised inside a widget's own message handler.
        Every other path -- workers, the run loop, the compositor, the driver --
        still goes to super(), which sets the return code; swallowing those
        would turn a crash into a hang.

        `WorkerFailed` is unwrapped. When a worker raises and `exit_on_error` is
        true (the default), `Worker._run` sets `WorkerState.ERROR` -- posting
        `StateChanged` *asynchronously* -- and then calls this method
        *synchronously* with `WorkerFailed(self._error)`. So this override fires
        first and, without unwrapping, would persist
        `exception_type=WorkerFailed` for every worker crash in the app, while
        `_fatal_error()` -> `_close_messages_no_wait()` races the queued
        `StateChanged` so the `worker_failed` event that carries the real type
        and `operation` may never be delivered. A crashed session's log would
        then read `event=unhandled_exception exception_type=WorkerFailed` and
        nothing else. `WorkerFailed.error` holds the real exception.
        """
        from textual.worker import WorkerFailed

        underlying = (
            getattr(error, "error", None) if isinstance(error, WorkerFailed) else None
        )
        raised = underlying if underlying is not None else error
        # TASK-32533: the type alone left critique #3's P0 unrecoverable from
        # the profile log (the traceback went to the dead pane's stderr).
        # Record the raising frame as identifiers only -- module, function,
        # line -- never the message, never a file path.
        frames = _exception_frames(raised)
        site = next(
            (
                frame
                for frame in reversed(frames)
                if frame[0].startswith("tldw_chatbook.")
            ),
            frames[-1] if frames else ("", "", None),
        )
        frame_fields: dict[str, object] = {}
        if frames:
            raise_module, raise_function, raise_line = frames[-1]
            site_module, site_function, site_line = site
            frame_fields = {
                "raise_module": raise_module,
                "raise_function": raise_function,
                "raise_line": raise_line,
                "site_module": site_module,
                "site_function": site_function,
                "site_line": site_line,
            }
        # The pump whose handler raised: Textual calls this method from inside
        # that pump's own context, so the ContextVar names it exactly. A Select
        # that fails while mounting leaves no Chatbook frame on the stack, so
        # its DOM id is the field that makes the site greppable.
        pump = None
        if underlying is None:
            try:
                pump = active_message_pump.get()
            except LookupError:
                pump = None
        if pump is not None:
            frame_fields["widget_type"] = type(pump).__name__
            pump_id = getattr(pump, "id", None)
            if pump_id:
                frame_fields["widget_id"] = pump_id
        try:
            persist_event(
                _DIAGNOSTICS_COMPONENT_APP,
                "unhandled_exception",
                level=logging.ERROR,
                exception_type=type(raised).__name__,
                **frame_fields,
            )
        except Exception:
            # Diagnostics must never be the reason a crash handler fails.
            pass
        # TASK-32533: a widget or screen pump that raises inside its own
        # dispatch reaches here through one of `_PUMP_DISPATCH_FRAMES`;
        # Textual's default then exits the whole app for one panel's bug. Keep
        # the screen alive for that case only -- not for a worker
        # (`underlying`), not for the compositor or driver, and only outside
        # headless `run_test` so the suite keeps its exception signal. Where the
        # raise came from the handler dispatch itself, Textual has already
        # broken that widget's message loop (`_process_messages_loop` breaks
        # after this call), which is why the notification warns that the panel
        # may stop responding.
        #
        # TWO clauses do the pump filtering and BOTH are load-bearing. `pump` is
        # Textual's `active_message_pump`: `App._context()` sets it to the app
        # around the application loop, `MessagePump._context()` sets it to the
        # widget around each widget task (including `_pre_process`, so a
        # mount-time widget failure is still kept alive and still names the
        # widget). `pump is not self` therefore excludes exactly the application
        # loop -- which also runs through `_process_messages_loop` and so is
        # matched by the frame set. Without it, breaking out of
        # `App._process_messages` without `super()` unwinds with no return code
        # and no `panic()`: the app vanishes on exit 0 with nothing in the log,
        # the P0's symptom with LESS evidence than before.
        keep_alive = (
            underlying is None
            and pump is not None
            and pump is not self
            and bool(
                getattr(
                    self, "_keep_screen_alive_on_handler_error", not self.is_headless
                )
            )
            and any(
                module == "textual.message_pump" and function in _PUMP_DISPATCH_FRAMES
                for module, function, _line in frames
            )
        )
        # TASK-33621.13 (GAP4-01): a pump whose loop the error ENDED must not
        # stay in charge of input: `retire_dead_pump` pops a dead SCREEN (and
        # above), or refocuses off a dead widget; None (nothing live left) exits.
        if keep_alive and (kind := retire_dead_pump(self, pump, frames)) is not None:
            try:
                self.bell()
                self.notify(
                    keep_alive_notice(site, pump, raised, kind),
                    severity="error",
                    timeout=12,
                    markup=False,
                )
            except Exception:
                # Telling the user failed, so keeping the app alive would leave
                # them with a silently broken panel: take the old exit instead.
                pass
            else:
                return
        super()._handle_exception(error)

    def _get_artifact_share_controller(self):
        """Return the app-owned share controller, creating it on first use.

        Creation (and the stale-share startup sweep) is deferred to the first
        share interaction so the Web_Server import chain stays off the boot
        path and inside the UI-ready module census budget (ADR-097).
        """
        controller = getattr(self, "artifact_share_controller", None)
        if controller is None:
            from .Web_Server.artifact_share import ArtifactShareController

            controller = ArtifactShareController()
            self.artifact_share_controller = controller
            try:
                controller.startup_sweep()
            except Exception as exc:
                logger.warning(f"Artifact share startup sweep failed: {exc}")
        return controller

    def _shutdown_artifact_share(self) -> None:
        """Stop any running artifact share; safe to call repeatedly."""
        controller = getattr(self, "artifact_share_controller", None)
        if controller is None:
            return
        try:
            controller.stop_share()
        except Exception as exc:
            logger.warning(f"Artifact share shutdown failed: {exc}")
        finally:
            # Drop the reference so a second call (or a late on_unmount
            # re-entry) never re-issues stop_share -- shutdown is strictly
            # once per controller.
            self.artifact_share_controller = None

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""

        # Do not close Notes or arm forced-exit cleanup while its worker is live.
        await self._shutdown_workflow_session()
        recovery_cleanup_cancellation = await TldwCli._shutdown_recovery_service(self)
        monitor_cleanup_cancellation = await TldwCli._stop_backup_maintenance_monitor(self)
        self._speech_initialization_closed = True
        speech_cleanup_cancellation = await self._settle_speech_initialization()
        speech_cleanup_cancellation = (
            recovery_cleanup_cancellation
            or monitor_cleanup_cancellation
            or speech_cleanup_cancellation
        )
        logging.info("--- App Unmounting ---")
        # task-19561: from here to process death, everything is teardown.
        # Arm the bound now rather than at the entry point, so the deadline
        # covers this method too -- and so a quit that wedges inside cleanup
        # is bounded exactly like a SIGTERM that does. Idempotent and
        # monotonic: a signal-armed watchdog already holds a tighter
        # deadline and this call leaves it alone.
        arm_exit_watchdog(reason="app unmount")
        # TASK-32806.5: nothing stopped local LLM servers on the way out, so
        # quitting orphaned every one of them with its port still held and
        # the next launch unable to bind. Inside the watchdog's deadline,
        # and off the loop because each stop waits on a subprocess.
        try:
            from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
                stop_all_server_processes,
            )

            stopped_servers = await asyncio.to_thread(
                stop_all_server_processes, self
            )
            if stopped_servers:
                logging.info(
                    "Stopped local LLM servers on shutdown: %s",
                    ", ".join(sorted(stopped_servers)),
                )
        except Exception as error:
            self.loguru_logger.warning(
                "Stopping local LLM servers on shutdown failed type={}",
                type(error).__name__,
            )
        try:
            await self._stop_served_canvas_control()
        except Exception as error:
            self.loguru_logger.warning(
                "Served Canvas control close failed type={} code=close_failed",
                type(error).__name__,
            )
        # TASK-1240. Distinguishes a clean exit from a kill: a log whose last
        # line is app_started ended abruptly. Wrapped, and deliberately so:
        # this line sits ABOVE the entire shutdown sequence -- DB closes,
        # worker cancellation, ingest pool teardown. An exception escaping here
        # would skip all of it. Diagnostics must never break the thing they
        # observe.
        try:
            # Session totals ride the app_stopping event (issue #365
            # deferred idea): post-hoc visibility even when the summary is
            # disabled. Aggregate counters only -- no user content. The
            # import is quit-time only (boot census, ADR-097).
            from tldw_chatbook.Chat.session_usage import session_usage

            snap = session_usage().snapshot()
            persist_event(
                _DIAGNOSTICS_COMPONENT_APP,
                "app_stopping",
                session_exact_tokens=snap.exact_tokens,
                session_estimated_tokens=snap.estimated_tokens,
                session_embedding_tokens=snap.embeddings_tokens,
                session_llm_calls=snap.calls,
            )
        except Exception:
            pass
        try:
            await self._shutdown_app_owned_lifecycles()
        except Exception as error:
            self.loguru_logger.warning(
                "App-owned lifecycle fallback shutdown failed "
                f"type={type(error).__name__}"
            )
        self._ui_ready = False
        self._stop_ui_responsiveness_monitor()
        monitor = self.ui_responsiveness_monitor
        if monitor is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(monitor.close)

        # F3/TASK-601: shut down both Library ingest worker boundaries. Final
        # shutdown order, explicit:
        #   1. `_ingest_shutdown = True` + executor/pool references detached
        #      (synchronous, inside `_shutdown_ingest_parse_pool`) -- their
        #      callbacks short-circuit before marshaling from this point on.
        #   2. Executor close, then a bounded `pool.terminate()` +
        #      `pool.join()` wait on detached daemon threads, NEVER this (loop)
        #      thread -- terminating
        #      inline here could deadlock against a result-handler thread
        #      parked inside `call_from_thread` (see that method's docstring).
        #      `terminate()` kills every in-flight light parse worker process
        #      immediately -- no waiting on a possibly-long OCR job.
        #   3. The writer (the exclusive `library_ingest_queue` thread
        #      worker) is swept up by the generic worker cancellation
        #      below, same as every other worker.
        # The spec words the quit contract writer-then-pool; here pool
        # teardown is *initiated* first but runs concurrently with the
        # writer sweep, which is equivalent and safe because the two stages
        # share no resources: parse workers never touch `media_db`, the
        # writer never touches the pool, and any late parse completion
        # no-ops via the flag from step 1. The writer's in-flight DB write
        # still completes (see Library/library_ingest_jobs.py's module
        # docstring: quitting joins the writer's in-flight DB write; parses
        # in flight are not waited for symmetrically).
        try:
            self._shutdown_ingest_parse_pool()
        except Exception as e:
            self.loguru_logger.error(
                f"Error shutting down Library ingest parse pool: {e}"
            )

        # Stop all background services and threads
        service_cleanup_primary: BaseException | None = speech_cleanup_cancellation
        try:
            deferred_tasks = [
                task
                for task in getattr(self, "_deferred_startup_tasks", set())
                if not task.done()
            ]
            for task in deferred_tasks:
                task.cancel()
            if deferred_tasks:
                await asyncio.gather(*deferred_tasks, return_exceptions=True)

            # Stop audio player if it exists
            if hasattr(self, "audio_player"):
                try:
                    await self.audio_player.cleanup()
                    self.loguru_logger.info("Audio player cleaned up")
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up audio player: {e}")

            # Clean up handler-owned TTS tasks and files if initialized.
            if hasattr(self, "_tts_handler") and self._tts_handler:
                try:
                    await self._tts_handler.cleanup_tts_resources()
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up TTS handler: {e}")

            # Clean up handler-owned S/TT/S tasks and files if initialized.
            if hasattr(self, "_stts_handler") and self._stts_handler:
                try:
                    if hasattr(self._stts_handler, "cleanup_tts_resources"):
                        await self._stts_handler.cleanup_tts_resources()
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up STTS handler: {e}")

            # Stop the background scheduler loop cleanly.
            scheduler_loop = getattr(self, "scheduler_loop", None)
            scheduler_worker = getattr(self, "scheduler_worker", None)
            if scheduler_loop is not None:
                scheduler_loop.stop()
            if scheduler_worker is not None:
                try:
                    if not scheduler_worker.is_finished:
                        # Textual's public cancellation contract cancels the
                        # underlying asyncio task. Worker.wait() has no timeout
                        # parameter, so request cancellation before observing it.
                        scheduler_worker.cancel()
                    await scheduler_worker.wait()
                except WorkerCancelled:
                    # Cancellation is the expected public Textual shutdown
                    # contract for a loop that may be sleeping between polls.
                    pass
                except Exception as e:
                    self.loguru_logger.error(f"Error stopping scheduler worker: {e}")

            # task-19561: stopping the scheduler worker does NOT stop the
            # generations it dispatched. `BriefingJobHandler.handle` spawns
            # each one as a bare `asyncio.Task` (Locked Decision 3 -- a
            # multi-minute LLM call must not stall the tick), so they are
            # absent from `App.workers` and survived every cancellation
            # above, only to be destroyed mid-flight when the loop closed.
            # Cancel them here, while the loop is still alive to deliver it.
            briefing_handler = getattr(self, "_briefing_job_handler", None)
            if briefing_handler is not None:
                try:
                    cancelled = await briefing_handler.shutdown()
                    if cancelled:
                        self.loguru_logger.info(
                            f"Cancelled {cancelled} in-flight scheduled briefing "
                            "generation(s)"
                        )
                except Exception as e:
                    self.loguru_logger.error(
                        f"Error stopping scheduled briefing generations: {e}"
                    )

            # dreams phase 1: the same seam for scheduled Dreams cycles --
            # `DreamsCycleHandler` spawns them as bare `asyncio.Task`s too,
            # held in the handler module's own set.
            try:
                from .Scheduling.scheduler.handlers.dreams_handler import (
                    shutdown as dreams_shutdown,
                )

                cancelled = await dreams_shutdown()
                if cancelled:
                    self.loguru_logger.info(
                        f"Cancelled {cancelled} in-flight Dreams cycle(s)"
                    )
            except Exception as e:
                self.loguru_logger.error(
                    f"Error stopping scheduled Dreams cycles: {e}"
                )

            # Disconnect local MCP client sessions (P5-T6), if any were ever
            # established this run.
            try:
                await self._disconnect_local_mcp_client()
                self.loguru_logger.info("Local MCP client sessions disconnected")
            except Exception as e:
                self.loguru_logger.error(
                    f"Error disconnecting local MCP client sessions: {e}"
                )

            # Stop any running artifact share (child web server) before the
            # process goes away; idempotent and failure-tolerant.
            try:
                self._shutdown_artifact_share()
                self.loguru_logger.info("Artifact share stopped (if running)")
            except Exception as e:
                self.loguru_logger.error(f"Error stopping artifact share: {e}")

            # Cancel any pending workers and wait for them, bounded.
            await self._cancel_and_settle_workers("unmount")

            # SSH ControlMaster cleanup (Phase 2a): `ssh -O exit` for every
            # master this process started, only after in-flight calls have
            # settled above so closing a master cannot cut a live call.
            # Best-effort and bounded (5s per host, off the loop);
            # ControlPersist remains the crash backstop, so a failure here
            # degrades to an eventually-expiring master and must never
            # block the quit.
            # Session workers close first: closing their stdin lets the
            # remote parents exit before the masters go away.
            try:
                from tldw_chatbook.Tools.remote_session_registry import (
                    close_all_remote_sessions,
                )
                from tldw_chatbook.Tools.remote_workspace_transport import (
                    get_master_manager,
                )

                await asyncio.to_thread(close_all_remote_sessions)
                await asyncio.to_thread(get_master_manager().close_all)
            except Exception as error:
                self.loguru_logger.warning(
                    "Closing SSH control masters on shutdown failed type={}",
                    type(error).__name__,
                )

            # Stop media cleanup timer
            if hasattr(self, "_media_cleanup_timer") and self._media_cleanup_timer:
                self._media_cleanup_timer.stop()
                self.loguru_logger.info("Media cleanup timer stopped")

            try:
                await self._close_server_context_provider_cached_client()
                self.loguru_logger.info("Server context provider cached client closed")
            except Exception as e:
                self.loguru_logger.error(
                    f"Error closing server context provider cached client: {e}"
                )

        except asyncio.CancelledError as error:
            service_cleanup_primary = error
        except Exception as e:
            self.loguru_logger.error(f"Error during service cleanup: {e}")
        except BaseException as error:
            service_cleanup_primary = error
        finally:
            try:
                await self._close_owned_tts_resources()
                self.loguru_logger.info("TTS resources cleaned up properly")
            except BaseException as cleanup_error:
                if service_cleanup_primary is not None:
                    service_cleanup_primary.add_note(
                        "TTS cleanup also failed while preserving the primary "
                        "shutdown error"
                    )
                    self.loguru_logger.warning(
                        "TTS owner cleanup failed while preserving shutdown "
                        f"type={type(cleanup_error).__name__} "
                        "code=operation_failed"
                    )
                elif isinstance(cleanup_error, Exception):
                    self.loguru_logger.warning(
                        "TTS owner cleanup phase=unmount failed "
                        f"type={type(cleanup_error).__name__} "
                        "code=operation_failed"
                    )
                else:
                    raise
        if service_cleanup_primary is not None:
            raise service_cleanup_primary

        # Original cleanup code
        if self._rich_log_handler:  # Ensure it's removed if it exists
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed.")

        # Stop DB size update timer on unmount as well, if not already handled by shutdown_request
        self.db_status_manager.stop_periodic_updates()
        self._stop_footer_status_timers()
        self.loguru_logger.info("DB size update timer stopped during unmount.")

        # Find and remove file handler (more robustly)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    handler.close()
                    root_logger.removeHandler(handler)
                    logging.info("RotatingFileHandler removed and closed.")
                except Exception as e_fh_close:
                    logging.error(f"Error removing/closing file handler: {e_fh_close}")

        # Force cleanup of any remaining threads and processes
        try:
            import platform

            # On macOS, force kill any afplay processes
            if platform.system() == "Darwin":
                try:
                    # Find and kill any afplay processes spawned by this app
                    import psutil

                    current_pid = os.getpid()
                    for proc in psutil.process_iter(["pid", "name", "ppid"]):
                        try:
                            if (
                                proc.info["name"] == "afplay"
                                and proc.info["ppid"] == current_pid
                            ):
                                self.loguru_logger.info(
                                    f"Killing orphaned afplay process: {proc.info['pid']}"
                                )
                                proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except ImportError:
                    # Fallback if psutil not available - run in background

                    @work(thread=True)
                    def kill_afplay_processes():
                        try:
                            # Kill all afplay processes (less precise but works)
                            subprocess.run(
                                ["killall", "afplay"], capture_output=True, timeout=1
                            )
                            self.loguru_logger.info("Killed all afplay processes")
                        except Exception as e:
                            self.loguru_logger.debug(
                                f"Could not kill afplay processes: {e}"
                            )

                    # Run in background to avoid blocking
                    self.run_worker(kill_afplay_processes, name="kill_afplay")
            # task-19561: this used to reach into `loop._default_executor`,
            # call `shutdown(wait=False)` and then set the private attribute
            # to `None`. Nulling it is what made the situation worse, not
            # better: `asyncio.run`'s `Runner.close()` ends with `await
            # loop.shutdown_default_executor(THREAD_JOIN_TIMEOUT)`, which
            # JOINS the worker threads while the loop is still alive. With
            # `_default_executor` set to `None` that coroutine returns at its
            # second line, and the very threads this block was trying to
            # hurry along were instead left for `threading._shutdown()` to
            # join with no bound at all. `run_worker(..., thread=True)` runs
            # on that same default executor (Textual's `Worker._run_threaded`
            # ends in `loop.run_in_executor(None, ...)`), so this is not a
            # corner case.
            #
            # Precise about the other half, because it is easy to overclaim:
            # `shutdown_default_executor` sets `_executor_shutdown_called`
            # BEFORE its `if self._default_executor is None: return`, so the
            # "a stray late `run_in_executor` raises" fence applied at the
            # merge base too. What nulling actually cost was the join -- plus
            # a window between this block and `Runner.close()` in which a late
            # `run_in_executor` would build a brand-new pool (that one IS
            # real, `BaseEventLoop.run_in_executor` creates one when
            # `_default_executor` is None and the fence is not yet set).
            #
            # Doing nothing here is the fix: the public, bounded shutdown
            # runs a few milliseconds later, on its own. Verified on CPython
            # 3.12.11, where `constants.THREAD_JOIN_TIMEOUT` is 300s -- far
            # looser than the exit watchdog armed at the top of this method,
            # which is what actually bounds the wait.

            # Clean up any lingering subprocess
            for proc in (
                (subprocess._active or []).copy()
            ):  # Make a copy to avoid modification during iteration
                try:
                    if proc.poll() is None:  # Process is still running
                        self.loguru_logger.warning(
                            f"Terminating lingering subprocess PID: {proc.pid}"
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)  # Give it 1 second to terminate
                        except subprocess.TimeoutExpired:
                            proc.kill()  # Force kill if it doesn't terminate
                            proc.wait()
                except Exception as e:
                    self.loguru_logger.error(f"Error terminating subprocess: {e}")

            # task-19561: a loop that force-set `thread.daemon = True` on
            # every live `ThreadPoolExecutor*`/`AudioPlayer*` thread used to
            # sit here. CPython raises `RuntimeError: cannot set daemon
            # status of active thread` for every one of them, so it changed
            # nothing and logged an ERROR per thread while doing it. The
            # "Active non-daemon threads remaining" warning that followed
            # reported the same threads a moment before the process was
            # going to wait on them anyway, with no way to act on it.
            # Both are gone. What replaces them is the exit watchdog armed
            # at the top of this method: it names the threads still alive
            # at the moment the wait actually becomes a hang, and ends the
            # process rather than merely describing it.

            # Threads that expose a cooperative stop() still get asked --
            # that half was never dead code.
            for thread in threading.enumerate():
                if thread is threading.main_thread() or not thread.is_alive():
                    continue
                stop = getattr(thread, "stop", None)
                if callable(stop):
                    try:
                        stop()
                        self.loguru_logger.info(f"Stopped thread: {thread.name}")
                    except Exception as e:
                        self.loguru_logger.error(
                            f"Error stopping thread {thread.name}: {e}"
                        )
        except Exception as e:
            self.loguru_logger.error(f"Error checking active threads: {e}")

        # Close the persisted Library ingest job history store (after pool
        # shutdown, above -- no more job writes are in flight by this point).
        store = getattr(self, "_library_ingest_jobs_store", None)
        if store is not None:
            store.close()

        # Release the writing suite's held SQLite connections (TASK-21125).
        await self._close_local_writing_service()

        # Release the research store's held SQLite connections (TASK-21127).
        await self._close_local_research_service()

        # Nothing this app owns is left to ask; a signal from here on has
        # no orderly path to offer and should unwind the main thread.
        unregister_running_app(self)

        logging.shutdown()
        self.loguru_logger.info("--- App Unmounted (Loguru) ---")

    #####################################################################
    # --- Event Handlers for Worker State Changes ---
    #####################################################################
    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """
        Handle worker state changes by delegating to the appropriate handler.

        This method has been refactored to use a handler registry pattern,
        significantly reducing complexity and improving maintainability.
        """
        worker_name = event.worker.name
        worker_group = event.worker.group

        # Log the state change (formatted only when DEBUG is on: PERF-03)
        self.loguru_logger.debug(
            "on_worker_state_changed: Worker '{}' (Group: {}, State: {})",
            worker_name,
            worker_group,
            event.state,
        )

        # TASK-22215. The same "one hook sees every transition" property the
        # diagnostics below rely on is what advances the staggered boot fleet:
        # a terminal state frees that worker's admission slot and lets the next
        # member start. Non-members return immediately (one dict lookup), and
        # the whole thing is best-effort -- a boot stagger must never be able
        # to break the app-wide worker hook.
        if event.state in (
            WorkerState.SUCCESS,
            WorkerState.ERROR,
            WorkerState.CANCELLED,
        ):
            try:
                self._release_boot_worker_slot(event.worker)
            except Exception:
                self.loguru_logger.opt(exception=True).debug(
                    "Staggered boot worker slot release failed"
                )

        # TASK-1240. One hook already sees every worker transition, so failures
        # are recorded without touching any of the 398 run_worker call sites.
        # Only ERROR persists: a start or success event here would emit a line
        # per keystroke-triggered search and per timer tick.
        if event.state is WorkerState.ERROR:
            error = getattr(event.worker, "error", None)
            # DO NOT "improve" `operation` to `event.worker.description`.
            # `Worker.name` is code-side -- the method or literal name given at
            # the `run_worker`/`@work` site. `Worker.description` is built by
            # textual's `_work_decorator` as `f"{name}={value!r}"` over the
            # worker's *actual arguments*, so for a chat, tool or provider
            # worker it contains prompts, API keys and tool values verbatim.
            # Persisting it would put exactly what ADR-029 excludes on disk.
            #
            # `else "unknown"` stays. `Worker._run` assigns `self.state =
            # WorkerState.ERROR` -- whose setter posts `StateChanged` -- one
            # line *before* `self._error = error`. Delivery is via the message
            # queue, so `_error` has landed by the time this handler runs in
            # every real interleaving; the branch is a total-function guard for
            # the ordering itself and for duck-typed workers, and it costs one
            # comparison on a path that only runs when something already broke.
            try:
                persist_event(
                    _DIAGNOSTICS_COMPONENT_APP,
                    "worker_failed",
                    level=logging.ERROR,
                    operation=str(worker_name or "unknown"),
                    exception_type=(
                        type(error).__name__ if error is not None else "unknown"
                    ),
                )
            except Exception:
                # Diagnostics must never break the worker hook every worker
                # transition in the app passes through.
                pass

        # Delegate to the handler registry; it reports unhandled workers, at
        # WARNING only when one failed (PERF-03).
        await self.worker_handler_registry.handle_event(event)

    def chat_wrapper(self, strip_thinking_tags: bool = True, **kwargs: Any) -> Any:
        """Delegate a retained non-streaming media call.

        Args:
            strip_thinking_tags: Whether the core chat call removes thinking
                tags.
            **kwargs: Arguments forwarded through the retained worker adapter.

        Returns:
            The non-streaming core chat result.

        Raises:
            ValueError: If a caller requests streaming, which is owned by the
                native Console provider gateway.
        """
        from .Event_Handlers import worker_events

        return worker_events.chat_wrapper_function(
            self, strip_thinking_tags=strip_thinking_tags, **kwargs
        )

    def schedule_media_cleanup(self) -> None:
        """Schedule periodic media cleanup based on configuration."""
        from tldw_chatbook.Backup_Recovery.activation import execution_allowed

        # TASK-1975: change-review snapshot retention rides the same
        # maintenance path but has its OWN knob ([change_review]
        # retention_days; <=0 disables inside the pass) -- disabling media
        # cleanup must not silently disable snapshot retention.
        try:
            if execution_allowed(("db.agent_runs",)):
                self._change_review_retention_startup_timer = self.set_timer(
                    DEFERRED_MEDIA_CLEANUP_DELAY_SECONDS + 60,
                    self._perform_change_review_retention,
                )
                self._change_review_retention_timer = self.set_interval(
                    24 * 3600, self._perform_change_review_retention
                )
        except Exception:  # noqa: BLE001 -- maintenance must never block boot
            self.loguru_logger.opt(exception=True).warning(
                "Could not schedule change-review retention"
            )
        try:
            if not execution_allowed(("db.media.primary",)):
                return
            # Get cleanup configuration
            cleanup_config = get_cli_setting("media_cleanup", "enabled", True)
            if not cleanup_config:
                self.loguru_logger.info("Media cleanup is disabled in configuration")
                return

            cleanup_interval_hours = get_cli_setting(
                "media_cleanup", "cleanup_interval_hours", 24
            )
            cleanup_on_startup = get_cli_setting(
                "media_cleanup", "cleanup_on_startup", True
            )

            # Run cleanup on startup if configured
            if cleanup_on_startup:
                self.loguru_logger.info(
                    "Scheduling media cleanup after startup idle delay"
                )
                self._media_cleanup_startup_timer = self.set_timer(
                    DEFERRED_MEDIA_CLEANUP_DELAY_SECONDS,
                    self.perform_media_cleanup,
                )

            # Schedule periodic cleanup
            cleanup_interval_seconds = cleanup_interval_hours * 3600
            self._media_cleanup_timer = self.set_interval(
                cleanup_interval_seconds, self.perform_media_cleanup
            )
            self.loguru_logger.info(
                f"Scheduled media cleanup every {cleanup_interval_hours} hours"
            )

        except Exception as e:
            self.loguru_logger.opt(exception=True).error(
                f"Error scheduling media cleanup: {e}"
            )

    async def _perform_change_review_retention(self) -> None:
        """Run one change-review retention pass off the UI thread (TASK-1975)."""
        try:
            db = getattr(self, "chachanotes_db", None)
            db_path = getattr(db, "db_path", None) if db is not None else None
            if not db_path or str(db_path) == ":memory:":
                return
            from tldw_chatbook.Workspaces.change_retention import (
                run_retention_for_app,
            )

            def retained_cleanup():
                from tldw_chatbook.Backup_Recovery.activation import execution_scope

                with execution_scope(
                    ("db.agent_runs",), Path(db_path).parent / "agent_runs.db"
                ) as allowed:
                    if allowed:
                        run_retention_for_app(db_path)

            await asyncio.to_thread(retained_cleanup)
        except Exception:  # noqa: BLE001 -- retention must never surface to the UI
            self.loguru_logger.opt(exception=True).warning(
                "Change-review retention pass failed"
            )

    async def perform_media_cleanup(self) -> None:
        """Perform media cleanup based on configuration settings."""
        try:
            if not self.media_db:
                self.loguru_logger.warning("Media database not available for cleanup")
                return
            db = self.media_db

            def run_cleanup_method(method, days):
                from tldw_chatbook.Backup_Recovery.activation import execution_scope

                with execution_scope(
                    ("db.media.primary",), Path(db.db_path)
                ) as allowed:
                    if not allowed:
                        return None
                    try:
                        return method(days)
                    finally:
                        if type(db) is MediaDatabase and not db.is_memory_db:
                            db.close_connection()

            # Get cleanup configuration
            cleanup_days = get_cli_setting("media_cleanup", "cleanup_days", 30)
            max_items = get_cli_setting("media_cleanup", "max_items_per_cleanup", 100)
            notify_before = get_cli_setting(
                "media_cleanup", "notify_before_cleanup", True
            )

            # Check for candidates first
            candidates = await asyncio.to_thread(
                run_cleanup_method, db.get_deletion_candidates, cleanup_days
            )

            if not candidates:
                self.loguru_logger.info("No media items eligible for cleanup")
                return

            candidate_count = len(candidates)
            items_to_delete = min(candidate_count, max_items)

            # Notify user if configured
            if notify_before and candidate_count > 0:
                self.notify(
                    f"Found {candidate_count} media items eligible for permanent deletion "
                    f"(soft-deleted over {cleanup_days} days ago). "
                    f"Will delete up to {items_to_delete} items.",
                    title="Media Cleanup",
                    severity="information",
                    timeout=5,
                )

            # Perform the cleanup
            deleted_count = await asyncio.to_thread(
                run_cleanup_method, db.hard_delete_old_media, cleanup_days
            )

            if deleted_count is not None and deleted_count > 0:
                self.loguru_logger.info(
                    f"Media cleanup completed: {deleted_count} items permanently deleted"
                )
                self.notify(
                    f"Media cleanup completed: {deleted_count} items permanently deleted",
                    severity="information",
                    timeout=3,
                )

        except Exception as e:
            self.loguru_logger.opt(exception=True).error(
                f"Error during media cleanup: {e}"
            )
            self.notify(
                f"Error during media cleanup: {str(e)}", severity="error", timeout=5
            )

    async def action_show_workbench_help(self) -> None:
        """Delegate contextual help to the active Workbench screen.

        Screens without a custom handler get a generic help panel generated
        from their own BINDINGS (falling back to the app-level bindings when
        the screen declares none), so F1 always shows truthful help.
        """
        handler = getattr(self.screen, "action_show_workbench_help", None)
        if callable(handler):
            result = handler()
            if inspect.isawaitable(result):
                await result
            return
        self._show_generic_screen_help()

    def action_library_artifacts(self) -> None:
        """Preserve Ctrl+6 as a Library route outside permanent shell destinations."""
        self.post_message(NavigateToScreen(TAB_ARTIFACTS))

    def action_shell_destination(self, destination_id: str) -> None:
        """Navigate to the shell destination identified by a stable ID.

        Args:
            destination_id: Shell destination ID from the Textual binding.
        """
        if destination_id == TAB_ARTIFACTS:
            self.post_message(NavigateToScreen(TAB_ARTIFACTS))
            return
        try:
            destination = get_shell_destination(destination_id)
        except KeyError:
            return
        self.post_message(NavigateToScreen(destination.primary_route))

    async def ensure_workflow_authoring(self) -> None:
        """Supply the one app-owned document/draft pair on first entry."""
        if self._workflow_authoring is None:
            from .config import get_workflows_db_path
            from .Workflows.authoring import WorkflowAuthoring

            def workflow_path():
                self._workflow_database_path = get_workflows_db_path()
                return self._workflow_database_path

            self._workflow_authoring = WorkflowAuthoring(workflow_path)
        await self._workflow_authoring.open()
        self.workflow_documents = self._workflow_authoring.documents
        self.workflow_drafts = self._workflow_authoring.drafts

    def ensure_workflow_session(self) -> "WorkflowSession":
        """Compose one lazy session using the existing Notes and permission owners."""
        from .Agents.builtin_tool_gate import BuiltinToolGate
        from .Workflows.session import SessionError, WorkflowSession
        from .Workflows.session_permissions import WorkflowPermissions

        if self._workflow_session is None:
            if (
                getattr(self, "notes_scope_service", None) is None
                or not getattr(self, "notes_user_id", None)
                or getattr(self, "unified_mcp_service", None) is None
            ):
                raise SessionError("services_unavailable")
            self._workflow_session = WorkflowSession(
                WorkflowPermissions(
                    BuiltinToolGate(self.unified_mcp_service, profile_id="default")
                ),
                notes_scope=lambda: self.notes_scope_service,
                notes_user=lambda: self.notes_user_id,
            )
        return self._workflow_session

    async def _confirm_workflow_session_quit(self) -> bool:
        """Pin the current session projection, including off-screen pending review."""

        self._workflow_quit_approved_view = None
        owner = getattr(self, "_workflow_session", None)
        if owner is None:
            return True
        while True:
            view = owner.view()
            if view is None or view.state in {
                "completed",
                "cancelled",
                "failed",
                "rejected",
                "uncertain",
            }:
                self._workflow_quit_approved_view = view
                return True
            decision = await self._await_quit_prompt(
                ConfirmationDialog(
                    title="Quit with a workflow in progress?",
                    message=(
                        f"Workflow {view.workflow_id}\nRevision {view.revision_id}\n"
                        "Session only: leaving this screen keeps the run; "
                        "quitting loses pending review and intermediate results. Saved Notes remain."
                    ),
                    confirm_label="Cancel run and quit",
                    cancel_label="Stay",
                )
            )
            if not decision:
                return False
            if owner.view() == view:
                self._workflow_quit_approved_view = view
                return True
            self.notify(
                "Workflow activity changed; review quitting again.", severity="warning"
            )

    async def _shutdown_workflow_session(self) -> None:
        """Fence, flush drafts, then physically drain before dependent services close."""
        owner = getattr(self, "_workflow_session", None)
        if owner is None:
            return
        owner.begin_close()
        try:
            authoring = getattr(self, "_workflow_authoring", None)
            if authoring is not None:
                await authoring.flush()
        except (Exception, asyncio.CancelledError):
            owner.abort_close()
            raise
        # close retains its settlement task and fence even if this waiter cancels.
        await owner.close()

    async def action_focus_next_workbench_pane(self) -> None:
        """Delegate pane focus cycling to the active Workbench screen."""
        handler = getattr(self.screen, "action_focus_next_workbench_pane", None)
        if callable(handler):
            result = handler()
            if inspect.isawaitable(result):
                await result
            return
        self.notify(
            "No workbench pane focus target is available.",
            severity="information",
        )

    def action_quit(self) -> None:
        """Dispatch one guarded asynchronous pre-quit confirmation worker."""

        if self._quit_in_progress:
            return
        self._quit_in_progress = True
        quit_flow = self._confirm_and_quit()
        try:
            self.run_worker(
                quit_flow,
                group="application-quit",
                exclusive=True,
                exit_on_error=False,
            )
        except Exception:
            quit_flow.close()
            self._quit_in_progress = False
            loguru_logger.warning(
                "Application quit worker could not start; staying in the app"
            )

    async def _confirm_and_quit(self) -> None:
        """Confirm the active screen, then execute one approved cleanup pass."""

        loguru_logger.info("Application quit initiated")
        runtime = getattr(self, "console_runtime", None)
        promotion_owner = getattr(runtime, "_voice_promotion_owner", None)
        promotion_token = None
        promotion_permit = None
        promotion_permit_consumed = False
        workflow_owner = getattr(self, "_workflow_session", None)
        workflow_close_accepted = False
        workflow_close_settled = False
        workflow_authoring = None
        try:
            try:
                begin_quit = getattr(promotion_owner, "begin_quit", None)
                if callable(begin_quit):
                    promotion_token = begin_quit()
                quit_screens = quit_confirmation_screens(self)
                if not await confirm_quit_screens(quit_screens):
                    self._quit_in_progress = False
                    return
                if not await self._confirm_console_runtime_quit():
                    self._quit_in_progress = False
                    return
                if not await self._confirm_workflow_session_quit():
                    return
                if promotion_token is not None:
                    wait_for_quiescence = getattr(
                        promotion_owner,
                        "wait_for_quiescence",
                        None,
                    )
                    if not callable(
                        wait_for_quiescence
                    ) or not await wait_for_quiescence(
                        promotion_token,
                        2.0,
                    ):
                        self._quit_in_progress = False
                        self.notify(
                            "A voice response is still being saved; staying in Chatbook.",
                            severity="warning",
                        )
                        return
                    promotion_permit = promotion_owner.seal_quiescent(promotion_token)
            except Exception:
                loguru_logger.warning(
                    "Pre-quit confirmation failed; staying in the app"
                )
                self._quit_in_progress = False
                try:
                    self.notify(
                        "Couldn't confirm quitting; staying in Chatbook.",
                        severity="warning",
                    )
                except Exception:
                    pass
                return

            try:
                workflow_owner = getattr(self, "_workflow_session", None)
                if workflow_owner is not None:
                    if workflow_owner.view() != self._workflow_quit_approved_view:
                        self.notify(
                            "Workflow activity changed; quit again to review it.",
                            severity="warning",
                        )
                        return
                    workflow_owner.begin_close()
                workflow_authoring = getattr(self, "_workflow_authoring", None)
                if workflow_authoring is not None:
                    await workflow_authoring.prepare_quit()
                await prepare_quit_screens(quit_screens)
            except Exception:
                loguru_logger.warning(
                    "Pre-quit shutdown guard failed; staying in the app"
                )
                self._quit_in_progress = False
                try:
                    self.notify(
                        "Couldn't prepare a safe shutdown; staying in Chatbook.",
                        severity="warning",
                    )
                except Exception:
                    pass
                return

            # Keep the reversible promotion permit unconsumed through fallible
            # workflow settlement. Permanent Console disposal cannot be undone.
            if workflow_owner is not None:
                try:
                    workflow_close_accepted = True
                    await workflow_owner.close()
                    workflow_close_settled = True
                except Exception:  # noqa: BLE001 - failed physical drain must never enter unconditional exit cleanup.
                    self.notify(
                        "Workflow is stopping; physical drain failed. Staying in Chatbook.",
                        severity="error",
                    )
                    return
            fence_console = getattr(runtime, "begin_dispose", None)
            from .Chat.console_chat_models import ConsoleLifecycleRevisionChanged

            while callable(fence_console):
                try:
                    dispose_kwargs = {
                        "expected_revision": getattr(
                            self,
                            "_console_quit_approved_revision",
                            None,
                        )
                    }
                    if promotion_permit is not None:
                        dispose_kwargs["voice_promotion_permit"] = promotion_permit
                    fence_console(**dispose_kwargs)
                    promotion_permit_consumed = promotion_permit is not None
                    break
                except ConsoleLifecycleRevisionChanged:
                    self.notify(
                        "Console activity changed; review the updated impact.",
                        severity="warning",
                    )
                    if await self._confirm_console_runtime_quit():
                        continue
                    self._quit_in_progress = False
                    return
                except Exception:
                    loguru_logger.warning(
                        "Console shutdown fence failed; staying in the app"
                    )
                    self._quit_in_progress = False
                    try:
                        self.notify(
                            "Couldn't prepare a safe shutdown; staying in Chatbook.",
                            severity="warning",
                        )
                    except Exception:
                        pass
                    return
            self._shutting_down = True
            # TASK-22215: the user has approved the quit -- nothing further from
            # the staggered boot fleet may start (idempotent with the same call in
            # `on_shutdown_request`, which the quit path reaches later).
            self._close_boot_worker_gate("quit")
            await self._run_approved_quit_cleanup()
        finally:
            if promotion_token is not None and not promotion_permit_consumed:
                abort_quit = getattr(promotion_owner, "abort_quit", None)
                if callable(abort_quit):
                    abort_quit(promotion_permit or promotion_token)
            if not promotion_permit_consumed and not getattr(
                self,
                "_shutting_down",
                False,
            ):
                self._quit_in_progress = False
            if workflow_authoring is not None and not getattr(
                self, "_shutting_down", False
            ):
                workflow_authoring.abort_quit()
            workflow_owner = getattr(self, "_workflow_session", None)
            if workflow_owner is not None and not getattr(
                self, "_shutting_down", False
            ):
                self._quit_in_progress = False
                if workflow_close_settled:
                    workflow_owner.reopen_after_drained_quit()
                elif not workflow_close_accepted:
                    # Publish reopened controls only after the app quit flag clears.
                    workflow_owner.abort_close()

    async def _await_console_quit_confirmation(self, dialog: Any) -> bool:
        """Await one app-level Console-loss dialog from the quit worker."""

        return await self._await_quit_prompt(dialog)

    async def _await_quit_prompt(self, dialog: Any) -> bool:
        """Await one app-level quit dialog; unanswered means Stay (TASK-33622.10)."""
        return bool(await await_quit_prompt(self, dialog, no_answer=False))

    async def _confirm_console_runtime_quit(self) -> bool:
        """Revision-pin Console loss even when a non-Console screen is mounted."""

        self._console_quit_approved_revision = None
        runtime = getattr(self, "console_runtime", None)
        controller = getattr(runtime, "chat_controller", None)
        if controller is None:
            return True

        while True:
            impact = controller.lifecycle_impact()
            if not impact.has_loss_risk:
                self._console_quit_approved_revision = impact.revision
                return True
            dialog = ConfirmationDialog(
                title="Quit Chatbook?",
                message=(
                    "Quitting Chatbook will cancel or discard:\n\n"
                    f"Live agent runs: {impact.live_run_count}\n"
                    f"Delegated agents: {impact.delegated_child_count}\n"
                    f"Sessions with queued prompts: {impact.queued_session_count}\n"
                    f"Unsent queued prompts: {impact.unsent_prompt_count}\n\n"
                    "Quit Chatbook?"
                ),
                confirm_label="Quit",
                cancel_label="Stay",
            )
            if not await self._await_console_quit_confirmation(dialog):
                return False
            if controller.lifecycle_impact() == impact:
                self._console_quit_approved_revision = impact.revision
                return True
            self.notify(
                "Console activity changed; review the updated impact.",
                severity="warning",
            )

    def _session_summary_duration_seconds(self) -> int:
        """Configured quit-summary duration, clamped to 1..30 (default 3)."""
        raw = get_cli_setting("session_summary", "duration_seconds", 3)
        try:
            value = int(round(float(raw)))
        except (TypeError, ValueError, OverflowError):
            return 3
        return max(1, min(30, value))

    async def _show_session_summary_before_exit(self) -> None:
        """Show the optional quit-time usage summary, hard-capped so exit
        always proceeds (issue #365; spec "Quit-Flow Integration")."""
        # Quit-only imports, deferred off the boot path: the UI-ready
        # module census ratchets down and never rises
        # (Tests/Performance/test_ui_ready_module_census.py).
        from tldw_chatbook.Chat.session_usage import session_usage
        from tldw_chatbook.Widgets.session_summary_dialog import SessionSummaryDialog

        try:
            duration = self._session_summary_duration_seconds()
            dialog = SessionSummaryDialog(
                session_usage().snapshot(),
                started_at=self._startup_start_time,
                duration_seconds=duration,
            )
            # The quit flow's prompt choke point (TASK-33622.10, ADR-031):
            # direct push_screen_wait can hang when a covered modal's
            # dismiss() pops the top screen. No vanish toast -- the summary
            # vanishing still means "exit now". The wait_for stays as a
            # belt-and-braces cap.
            await asyncio.wait_for(
                await_quit_prompt(
                    self, dialog, no_answer=None, vanished_notice=None
                ),
                timeout=duration + 2.0,
            )
        except asyncio.TimeoutError:
            loguru_logger.warning(
                "Session summary dialog did not dismiss in time; exiting anyway."
            )
        except Exception:
            # Deliberately narrow beyond TimeoutError for robustness, but
            # CancelledError is BaseException on 3.12 -- it propagates and
            # must never be swallowed on the quit path (lessons-textual).
            loguru_logger.warning("Session summary display failed; exiting anyway.")

    async def _run_approved_quit_cleanup(self) -> None:
        """Preserve quit ordering without blocking the Textual event loop."""

        try:
            await self._cleanup_audio_for_quit()
            media_timer = getattr(self, "_media_cleanup_timer", None)
            if media_timer is not None:
                try:
                    media_timer.stop()
                except Exception:
                    loguru_logger.warning(
                        "Media cleanup timer could not stop during quit"
                    )
            persistence_ok = True
            try:
                await asyncio.to_thread(self._run_blocking_quit_persistence)
            except Exception:
                persistence_ok = False
                loguru_logger.warning("Blocking quit persistence failed")
            # Fail closed: a config-read failure on the quit path must
            # degrade to "no summary", never to a failed quit (issue #365);
            # a failed persistence also skips the summary (spec: exit
            # reliability wins over the farewell screen).
            summary_enabled = False
            if persistence_ok:
                try:
                    summary_enabled = bool(
                        get_cli_setting("session_summary", "enabled", False)
                    )
                except Exception:
                    summary_enabled = False
            if summary_enabled:
                await self._show_session_summary_before_exit()
        finally:
            self.exit()

    async def _cleanup_audio_for_quit(self) -> None:
        """Stop and release app-owned audio before the final exit."""

        audio_player = getattr(self, "audio_player", None)
        if audio_player is None:
            return
        try:
            await asyncio.wait_for(audio_player.stop(), timeout=0.5)
        except asyncio.TimeoutError:
            loguru_logger.warning("Audio stop timed out")
        except Exception:
            loguru_logger.warning("Audio stop failed during quit")
        try:
            await asyncio.wait_for(audio_player.cleanup(), timeout=0.5)
        except asyncio.TimeoutError:
            loguru_logger.warning("Audio cleanup timed out")
        except Exception:
            loguru_logger.warning("Audio cleanup failed during quit")

    @staticmethod
    def _save_shutdown_caches_with_timeout() -> None:
        """Retain the existing bounded cache-save compatibility pass."""
        loguru_logger.debug("Cache saving skipped - handled by simplified RAG service")

    def _run_blocking_quit_persistence(self) -> None:
        """Run timed joins and configuration persistence off the app loop."""
        from .css.Themes.theme_catalog import wait_for_theme_quit_work
        wait_for_theme_quit_work(self)  # TASK-33121/review M-2: one deadline, THEME_QUIT_WAIT_SECONDS
        try:
            save_thread = threading.Thread(
                target=self._save_shutdown_caches_with_timeout,
                name="chatbook-quit-cache-save",
                daemon=True,
            )
            save_thread.start()
            save_thread.join(timeout=2.0)
            if save_thread.is_alive():
                loguru_logger.warning("Cache save timed out - proceeding with quit")
        except Exception:
            loguru_logger.warning("Error in quit cache handler")

        try:
            persisted = persist_cli_config_for_shutdown()
        except Exception:
            loguru_logger.warning("Configuration shutdown persistence raised an error")
        else:
            if not persisted:
                loguru_logger.warning("Configuration shutdown persistence failed")
