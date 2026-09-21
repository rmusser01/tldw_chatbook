"""
Database Status Manager - Centralized management for database size telemetry.

Computes the local database file sizes on a timer and caches them on the
app (``db_sizes_status``) for the Library Details disclosure, plus logs
them.

task-21133: footer token-count updates used to route through here too. Their
whole consumer surface was retired by task-17653 (no screen composes an armed
``AppFooterStatus``), so the periodic producer and this manager's half of it
are gone; only the DB-size telemetry remains.
"""

import asyncio
import time
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from textual.app import App


class DBStatusManager:
    """Manages database size telemetry.

    Footer token-count updates used to live here too; task-21133 retired
    that half after task-17653 removed its whole consumer surface (see the
    module docstring).
    """

    def __init__(self, app: "App"):
        """
        Initialize the database status manager.

        Args:
            app: The Textual app instance
        """
        self.app = app
        self._update_timer = None
        # task-22220: last sizes the periodic INFO line reported, so an
        # unchanged 120 s fire stays out of the log.
        self._last_logged_sizes: Optional[Dict[str, str]] = None
        self._maintenance_closed = False
        self._maintenance_calls = 0
        self._maintenance_workers: set[asyncio.Task] = set()
        self._maintenance_changed = asyncio.Event()

    def _maintenance_close_admission(self) -> None:
        """Ignore queued telemetry updates until backup releases maintenance."""
        self._maintenance_closed = True

    async def _maintenance_drain(self, deadline: float) -> bool:
        """Keep accepted callers and their native threads visible until settled."""
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        if not self._maintenance_closed:
            raise RecoveryRequired("runtime_producer_not_closed")
        while self._maintenance_calls or self._maintenance_workers:
            self._maintenance_changed.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self._maintenance_changed.wait(), remaining)
            except TimeoutError:
                return False
        return True

    def _maintenance_resume(self) -> None:
        """Reopen telemetry only after every accepted collection completes."""
        from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

        if self._maintenance_calls or self._maintenance_workers:
            raise RecoveryRequired("runtime_work_not_settled")
        self._maintenance_closed = False

    async def update_db_sizes(self) -> None:
        """
        Compute the local database sizes for telemetry consumers.

        F-014: the sizes no longer render in the app footer (a fresh
        install's "Prompts: N/A | Chats/Notes: N/A | Media: N/A" triplet
        read as "broken" in user chrome). They are cached on the app as
        ``db_sizes_status`` -- the Library rail's Details disclosure
        renders them from there -- and logged on change (the Logs
        destination is the other telemetry home).

        task-22220: the collection (~15 stat/exists syscalls across three
        DBs plus their WAL/SHM sidecars) runs off the event loop via
        ``asyncio.to_thread``; only the ``db_sizes_status`` assignment
        happens back on the loop. The INFO triple is change-gated -- an
        unchanged fire updates the cache silently.
        """
        if self._maintenance_closed:
            return
        self._maintenance_calls += 1

        try:
            logger.debug("Computing DB sizes for the db_sizes_status cache.")
            worker = asyncio.create_task(asyncio.to_thread(self._collect_db_sizes))
            self._maintenance_workers.add(worker)

            def finished(completed):
                self._maintenance_workers.discard(completed)
                self._maintenance_changed.set()
                if not completed.cancelled():
                    # A cancelled caller cannot abandon a native worker error.
                    # Ordinary awaiting callers still receive the same exception.
                    completed.exception()

            worker.add_done_callback(finished)
            db_sizes = await asyncio.shield(worker)

            self.app.db_sizes_status = db_sizes
            if db_sizes != self._last_logged_sizes:
                self._last_logged_sizes = db_sizes
                # task-1714: spell the labels -- single letters decoded only
                # by a hover tooltip fail keyboard-first/low-vision users
                # (critique r4).
                logger.info(
                    "DB sizes: "
                    f"Prompts: {db_sizes['prompts']} | "
                    f"Chats/Notes: {db_sizes['chachanotes']} | "
                    f"Media: {db_sizes['media']}"
                )

        except Exception as e:
            logger.opt(exception=True).error(f"Error computing DB sizes: {e}")
        finally:
            self._maintenance_calls -= 1
            self._maintenance_changed.set()

    def _collect_db_sizes(self) -> Dict[str, str]:
        """Stat the DB files (+ WAL/SHM sidecars) -- runs OFF the event loop.

        Called via ``asyncio.to_thread`` from :meth:`update_db_sizes`; keep
        it free of any UI mutation (the caller publishes the result back on
        the loop). WAL-inclusive (task-2859 item 5): a busy, uncheckpointed
        DB can hold most of its recent writes in its ``-wal`` sidecar, so
        the main file's size alone understates the real footprint.
        """
        # Import here to avoid circular imports
        from tldw_chatbook.config import (
            get_prompts_db_path,
            get_chachanotes_db_path,
            get_media_db_path,
        )
        from tldw_chatbook.Utils.Utils import get_formatted_db_size_with_wal

        return {
            "prompts": self._get_db_size(
                get_prompts_db_path, get_formatted_db_size_with_wal
            ),
            "chachanotes": self._get_db_size(
                get_chachanotes_db_path, get_formatted_db_size_with_wal
            ),
            "media": self._get_db_size(
                get_media_db_path, get_formatted_db_size_with_wal
            ),
        }

    def start_periodic_updates(self, interval_seconds: float = 5.0) -> None:
        """
        Start periodic database size updates.

        Args:
            interval_seconds: Update interval in seconds
        """
        if self._update_timer:
            self.stop_periodic_updates()

        self._update_timer = self.app.set_interval(
            interval_seconds, lambda: self.app.call_later(self.update_db_sizes)
        )
        logger.info(
            f"Started periodic DB size updates every {interval_seconds} seconds"
        )

    def stop_periodic_updates(self) -> None:
        """Stop periodic database size updates."""
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None
            logger.info("Stopped periodic DB size updates")

    def _get_db_size(self, path_func: callable, formatter_func: callable) -> str:
        """
        Get the formatted size of a database file.

        Args:
            path_func: Function to get the database path
            formatter_func: Function to format the file size

        Returns:
            Formatted size string or "N/A" if unavailable
        """
        try:
            db_path = path_func()
            size_str = formatter_func(db_path)
            return size_str if size_str is not None else "N/A"
        except Exception as e:
            logger.error(f"Error getting DB size: {e}")
            return "Error"
