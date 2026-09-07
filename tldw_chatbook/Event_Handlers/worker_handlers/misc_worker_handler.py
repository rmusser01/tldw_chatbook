"""
Miscellaneous Worker Handler - Handles various other worker types.

This module manages state changes for workers that don't fit into the main
categories, including:
- Ollama API operations
- Model downloads
"""

from typing import Optional
from textual.worker import Worker, WorkerState

from tldw_chatbook.Constants import MODEL_CATALOG_REFRESH_WORKER_GROUP
from tldw_chatbook.Utils.boot_worker_policy import BOOT_WORKER_POLICY

from .base_handler import BaseWorkerHandler

#: Every group in the declarative boot fleet (``scheduling``,
#: ``ingest_restore``, the actor-pack prefetches, the FTS backfills). DERIVED
#: from BOOT_WORKER_POLICY, not hand-listed, so a future BootWorkerSpec can
#: never silently reintroduce the per-transition "No handler found for worker"
#: warning these one-shots emit at boot (TASK-31806). Each is fire-and-forget:
#: its failure is already persisted by the diagnostics hook in
#: on_worker_state_changed before registry delegation runs.
_BOOT_WORKER_GROUPS = frozenset(spec.group for spec in BOOT_WORKER_POLICY)

#: Fire-and-forget research/library startup reconciliation sweeps spawned
#: directly (TldwCli, not via the boot policy). Same rationale as the boot
#: fleet: self-contained coroutines that catch-and-log their own outcome and
#: need no per-transition handler (TASK-31806). Data-gated ones
#: (``research_source_held_startup``) only fire on some profiles, but warn
#: identically when they do.
_RESEARCH_STARTUP_GROUPS = frozenset(
    {
        "research_source_association_startup",
        "research_source_held_startup",
        "research_paste_staging_startup",
        "research-quick-notes-startup-reconciliation",
    }
)


class MiscWorkerHandler(BaseWorkerHandler):
    """Handles miscellaneous worker state changes."""

    # Worker groups this handler manages
    HANDLED_GROUPS = {
        "ollama_api",
        "model_download",
        # Fire-and-forget groups (task-2726): their failures are already
        # persisted by the diagnostics hook before registry delegation, so
        # acknowledging them here only silences the per-transition
        # "No handler found" warning — one per tab switch for
        # screen-navigation (TASK-1230), several at startup for the others.
        "screen-navigation",
        MODEL_CATALOG_REFRESH_WORKER_GROUP,
    } | _BOOT_WORKER_GROUPS | _RESEARCH_STARTUP_GROUPS

    def can_handle(self, worker_name: str, worker_group: Optional[str] = None) -> bool:
        """
        Check if this handler can process the given worker.

        Args:
            worker_name: The name attribute of the worker
            worker_group: The group attribute of the worker

        Returns:
            True if this handler manages this worker group
        """
        return worker_group in self.HANDLED_GROUPS

    async def handle(self, event: Worker.StateChanged) -> None:
        """
        Handle the worker state change event.

        Args:
            event: The worker state changed event
        """
        worker_info = self.get_worker_info(event)
        self.log_state_change(worker_info, f"{worker_info['group']}: ")

        if worker_info["group"] == "ollama_api":
            # Ollama operations now use asyncio.to_thread instead of workers
            self.logger.info(
                f"Ollama API worker '{worker_info['name']}' finished with state {worker_info['state']}"
            )

        elif worker_info["group"] == "model_download":
            await self._handle_model_download(event, worker_info)

    async def _handle_model_download(
        self, event: Worker.StateChanged, worker_info: dict
    ) -> None:
        """Handle model download workers."""
        if worker_info["state"] == WorkerState.PENDING:
            self.logger.info("Model download worker is PENDING")

        elif worker_info["state"] == WorkerState.RUNNING:
            self.logger.info("Model download worker is RUNNING")
            self.app.notify("Model download in progress...", title="Download Status")

        elif worker_info["state"] == WorkerState.SUCCESS:
            self.logger.info("Model download completed successfully")
            self.app.notify(
                "Model downloaded successfully!",
                title="Download Complete",
                severity="information",
            )

            # Re-enable download button if exists
            await self.update_button_state("download-model-button", disabled=False)

        elif worker_info["state"] == WorkerState.ERROR:
            error_msg = (
                str(event.worker.error)[:100] if event.worker.error else "Unknown error"
            )
            self.logger.error(f"Model download failed: {error_msg}")

            self.app.notify(
                f"Model download failed: {error_msg}",
                title="Download Error",
                severity="error",
            )

            # Re-enable download button if exists
            await self.update_button_state("download-model-button", disabled=False)
