"""Presentation-owned service drain and bounded worker settlement.

LifecycleMixin retains the existing call and patch points; this module loads
only when shutdown actually reaches those phases.
"""

from __future__ import annotations


async def cancel_and_settle_workers(self, phase: str) -> None:
    """Cancel every live worker and actually wait for them, bounded.

    task-19561. Both shutdown hooks used to cancel their workers and
    then ``await _owner.asyncio.sleep(0.1)`` -- a flat wait that is
    simultaneously too long (nothing to wait for on a quiet exit) and
    far too short (a worker mid-``await`` gets one tick, and a thread
    worker gets nothing at all), and which never observed the outcome
    either way. Waiting on the workers themselves is both faster in the
    common case and honest in the uncommon one; the timeout keeps a
    worker that ignores cancellation from turning quit into a hang, and
    says which ones they were.
    """
    from tldw_chatbook import app_lifecycle as _owner

    try:
        active_workers = [w for w in self.workers if not w.is_finished]
        if not active_workers:
            return
        self.loguru_logger.info(
            f"Cancelling {len(active_workers)} active workers ({phase})"
        )
        for worker in active_workers:
            worker.cancel()
        # `_owner.asyncio.wait`, NOT `wait_for(...)`: on expiry `wait_for`
        # cancels what it is waiting on and awaits that cancellation, so
        # anything that does not honour a cancel hangs the very call
        # meant to bound it. `wait` returns the stragglers instead.
        waiters = [_owner.asyncio.ensure_future(w.wait()) for w in active_workers]
        _, unsettled = await _owner.asyncio.wait(
            waiters, timeout=_owner.WORKER_CANCELLATION_GRACE_SECONDS
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
                f"{_owner.WORKER_CANCELLATION_GRACE_SECONDS}s of cancellation "
                f"({phase}): {stragglers}"
            )
    except Exception as e:
        self.loguru_logger.error(f"Error cancelling workers ({phase}): {e}")


async def _drain_app_unmount_services(
    self, primary_error: BaseException | None
) -> None:
    """Drain the app's service phase before its remaining unmount cleanup."""
    from tldw_chatbook import app_lifecycle as _owner

    service_cleanup_primary: BaseException | None = primary_error
    try:
        deferred_tasks = [
            task
            for task in getattr(self, "_deferred_startup_tasks", set())
            if not task.done()
        ]
        for task in deferred_tasks:
            task.cancel()
        if deferred_tasks:
            await _owner.asyncio.gather(*deferred_tasks, return_exceptions=True)

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
            except _owner.WorkerCancelled:
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
            from tldw_chatbook.Scheduling.scheduler.handlers.dreams_handler import (
                shutdown as dreams_shutdown,
            )

            cancelled = await dreams_shutdown()
            if cancelled:
                self.loguru_logger.info(
                    f"Cancelled {cancelled} in-flight Dreams cycle(s)"
                )
        except Exception as e:
            self.loguru_logger.error(f"Error stopping scheduled Dreams cycles: {e}")

        # dreams phase 2: the same seam for scheduled Dreams track
        # checks -- `DreamTrackHandler` spawns them as bare
        # `asyncio.Task`s too, held in its handler module's own set
        # (ruling P9: the call is what makes that set's documented
        # on_unmount reachability true).
        try:
            from tldw_chatbook.Scheduling.scheduler.handlers.dream_track_handler import (
                shutdown as dream_track_shutdown,
            )

            cancelled = await dream_track_shutdown()
            if cancelled:
                self.loguru_logger.info(
                    f"Cancelled {cancelled} in-flight Dreams track " f"check(s)"
                )
        except Exception as e:
            self.loguru_logger.error(
                f"Error stopping scheduled Dreams track checks: {e}"
            )

        # Drain admitted Guardian SQLite callbacks before stores are closed.
        try:
            from tldw_chatbook.Scheduling.scheduler.handlers.guardian_trend_handler import (
                shutdown as guardian_trend_shutdown,
            )

            await guardian_trend_shutdown()
        except Exception as e:
            self.loguru_logger.error(
                f"Error stopping Guardian trend work: {type(e).__name__}"
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

            await _owner.asyncio.to_thread(close_all_remote_sessions)
            await _owner.asyncio.to_thread(get_master_manager().close_all)
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

    except _owner.asyncio.CancelledError as error:
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
