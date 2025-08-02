"""
Miscellaneous Worker Handler - Handles various other worker types.

This module manages state changes for workers that don't fit into the main
categories, including:
- TLDW API calls
- Ollama API operations
- Model downloads
- Transformers downloads
"""

from typing import TYPE_CHECKING, Optional
from textual.worker import Worker, WorkerState

from .base_handler import BaseWorkerHandler

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MiscWorkerHandler(BaseWorkerHandler):
    """Handles miscellaneous worker state changes."""
    
    # Worker groups this handler manages
    HANDLED_GROUPS = {
        "api_calls",
        "ollama_api", 
        "model_download",
        "transformers_download",
    }
    
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
        
        # Import here to avoid circular imports
        from tldw_chatbook.Event_Handlers import ingest_events
        from tldw_chatbook.Utils.log_widget_manager import LogWidgetManager
        
        if worker_info['group'] == "api_calls":
            await self._handle_api_calls(event, worker_info, ingest_events)
            
        elif worker_info['group'] == "ollama_api":
            # Ollama operations now use asyncio.to_thread instead of workers
            self.logger.info(f"Ollama API worker '{worker_info['name']}' finished with state {worker_info['state']}")
            
        elif worker_info['group'] == "model_download":
            await self._handle_model_download(event, worker_info)
            
        elif worker_info['group'] == "transformers_download":
            await self._handle_transformers_download(event, worker_info)
    
    async def _handle_api_calls(self, event: Worker.StateChanged, worker_info: dict,
                               ingest_events) -> None:
        """Handle TLDW API call workers."""
        self.logger.info(f"TLDW API worker '{worker_info['name']}' finished with state {worker_info['state']}")
        
        if worker_info['state'] == WorkerState.SUCCESS:
            await ingest_events.handle_tldw_api_worker_success(self.app, event)
            
        elif worker_info['state'] == WorkerState.ERROR:
            await ingest_events.handle_tldw_api_worker_failure(self.app, event)
    
    async def _handle_model_download(self, event: Worker.StateChanged, 
                                    worker_info: dict) -> None:
        """Handle model download workers."""
        if worker_info['state'] == WorkerState.PENDING:
            self.logger.info("Model download worker is PENDING")
            
        elif worker_info['state'] == WorkerState.RUNNING:
            self.logger.info("Model download worker is RUNNING")
            self.app.notify("Model download in progress...", title="Download Status")
            
        elif worker_info['state'] == WorkerState.SUCCESS:
            self.logger.info("Model download completed successfully")
            self.app.notify("Model downloaded successfully!", title="Download Complete", severity="information")
            
            # Re-enable download button if exists
            await self.update_button_state("download-model-button", disabled=False)
            
        elif worker_info['state'] == WorkerState.ERROR:
            error_msg = str(event.worker.error)[:100] if event.worker.error else "Unknown error"
            self.logger.error(f"Model download failed: {error_msg}")
            
            self.app.notify(
                f"Model download failed: {error_msg}",
                title="Download Error",
                severity="error"
            )
            
            # Re-enable download button if exists
            await self.update_button_state("download-model-button", disabled=False)
    
    async def _handle_transformers_download(self, event: Worker.StateChanged,
                                          worker_info: dict) -> None:
        """Handle transformers download workers."""
        self.logger.info(
            f"Transformers download worker (name='{worker_info['name']}') "
            f"state changed to {worker_info['state']}"
        )
        
        if worker_info['state'] == WorkerState.PENDING:
            self.logger.debug("Download worker is pending.")
            await self.update_button_state("start-download-button", disabled=True)
            
        elif worker_info['state'] == WorkerState.RUNNING:
            self.logger.info("Download worker is running.")
            
        elif worker_info['state'] == WorkerState.SUCCESS:
            self.logger.info("Download worker succeeded.")
            self.app.notify("Model download completed!", severity="information")
            await self.update_button_state("start-download-button", disabled=False)
            
        elif worker_info['state'] == WorkerState.ERROR:
            error_msg = str(event.worker.error) if event.worker.error else "Unknown error"
            self.logger.error(f"Download worker failed: {error_msg}")
            
            self.app.notify(
                f"Download failed: {error_msg[:100]}",
                severity="error"
            )
            await self.update_button_state("start-download-button", disabled=False)