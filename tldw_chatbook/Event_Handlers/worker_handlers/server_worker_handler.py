"""
Server Worker Handler - Handles server-related worker state changes.

This module manages state changes for various LLM server workers including:
- Llama.cpp server
- Llamafile server
- vLLM server
- MLX-LM server
- ONNX server
- Transformers server
"""

from typing import TYPE_CHECKING, Optional, Dict
from textual.worker import Worker, WorkerState
from textual.widgets import Button
from textual.css.query import QueryError

from .base_handler import BaseWorkerHandler

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ServerWorkerHandler(BaseWorkerHandler):
    """Handles server-related worker state changes."""
    
    # Mapping of server groups to their configuration
    SERVER_CONFIGS: Dict[str, Dict[str, str]] = {
        "llamacpp_server": {
            "name": "Llama.cpp",
            "start_button": "llamacpp-start-server-button",
            "stop_button": "llamacpp-stop-server-button",
            "process_attr": "llamacpp_server_process",
        },
        "llamafile_server": {
            "name": "Llamafile",
            "start_button": "llamafile-start-server-button",
            "stop_button": "llamafile-stop-server-button",
            "process_attr": "llamafile_server_process",
        },
        "vllm_server": {
            "name": "vLLM",
            "start_button": "vllm-start-server-button",
            "stop_button": "vllm-stop-server-button",
            "process_attr": "vllm_server_process",
        },
        "mlx_lm_server": {
            "name": "MLX-LM",
            "start_button": "mlx-start-server-button",
            "stop_button": "mlx-stop-server-button",
            "process_attr": "mlx_server_process",
        },
        "onnx_server": {
            "name": "ONNX",
            "start_button": "onnx-start-server-button",
            "stop_button": "onnx-stop-server-button",
            "process_attr": "onnx_server_process",
        },
    }
    
    def can_handle(self, worker_name: str, worker_group: Optional[str] = None) -> bool:
        """
        Check if this handler can process the given worker.
        
        Args:
            worker_name: The name attribute of the worker
            worker_group: The group attribute of the worker
            
        Returns:
            True if this is a server-related worker
        """
        return worker_group in self.SERVER_CONFIGS
    
    async def handle(self, event: Worker.StateChanged) -> None:
        """
        Handle the server worker state change event.
        
        Args:
            event: The worker state changed event
        """
        worker_info = self.get_worker_info(event)
        server_config = self.SERVER_CONFIGS.get(worker_info['group'])
        
        if not server_config:
            self.logger.error(f"No configuration found for server group: {worker_info['group']}")
            return
        
        server_name = server_config['name']
        self.log_state_change(worker_info, f"{server_name} Server: ")
        
        if worker_info['state'] == WorkerState.PENDING:
            await self._handle_pending_state(server_config)
            
        elif worker_info['state'] == WorkerState.RUNNING:
            await self._handle_running_state(server_config)
            
        elif worker_info['state'] == WorkerState.SUCCESS:
            await self._handle_success_state(event, server_config)
            
        elif worker_info['state'] == WorkerState.ERROR:
            await self._handle_error_state(event, server_config)
    
    async def _handle_pending_state(self, server_config: Dict[str, str]) -> None:
        """Handle the PENDING state for server workers."""
        self.logger.debug(f"{server_config['name']} server worker is PENDING")
        
        # Disable both buttons during startup
        await self.update_button_state(server_config['start_button'], disabled=True)
        await self.update_button_state(server_config['stop_button'], disabled=True)
    
    async def _handle_running_state(self, server_config: Dict[str, str]) -> None:
        """Handle the RUNNING state for server workers."""
        self.logger.info(f"{server_config['name']} server worker is RUNNING (subprocess launched)")
        
        # Keep start disabled, enable stop
        await self.update_button_state(server_config['start_button'], disabled=True)
        await self.update_button_state(server_config['stop_button'], disabled=False)
        
        self.app.notify(
            f"{server_config['name']} server process starting...",
            title="Server Status"
        )
    
    async def _handle_success_state(self, event: Worker.StateChanged, 
                                   server_config: Dict[str, str]) -> None:
        """Handle the SUCCESS state for server workers."""
        server_name = server_config['name']
        self.logger.info(f"{server_name} server worker finished successfully")
        
        # Analyze the result message
        result_message = str(event.worker.result).strip() if event.worker.result else \
                        "Worker completed with no specific result message."
        self.logger.info(f"{server_name} worker result message: '{result_message}'")
        
        # Check for server errors in the result
        is_actual_server_error = self._check_for_server_error(result_message)
        
        # Notify user based on result
        if is_actual_server_error:
            self.app.notify(
                f"{server_name} server process reported an error. Check logs.",
                title="Server Status",
                severity="error",
                timeout=10
            )
        elif "exited quickly with code: 0" in result_message.lower():
            self.app.notify(
                f"{server_name} server exited quickly (but successfully). "
                "Check logs if this was unexpected.",
                title="Server Status",
                severity="warning",
                timeout=10
            )
        else:
            self.app.notify(
                f"{server_name} server process finished.",
                title="Server Status"
            )
        
        # Clear the server process reference
        await self._clear_server_process(server_config)
        
        # Re-enable start button, disable stop
        await self.update_button_state(server_config['start_button'], disabled=False)
        await self.update_button_state(server_config['stop_button'], disabled=True)
    
    async def _handle_error_state(self, event: Worker.StateChanged, 
                                 server_config: Dict[str, str]) -> None:
        """Handle the ERROR state for server workers."""
        server_name = server_config['name']
        error_msg = str(event.worker.error)[:100] if event.worker.error else "Unknown error"
        
        self.logger.error(f"{server_name} server worker failed with exception: {event.worker.error}")
        
        self.app.notify(
            f"{server_name} worker error: {error_msg}",
            title="Server Worker Error",
            severity="error"
        )
        
        # Clear the server process reference
        await self._clear_server_process(server_config)
        
        # Re-enable start button, disable stop
        await self.update_button_state(server_config['start_button'], disabled=False)
        await self.update_button_state(server_config['stop_button'], disabled=True)
    
    def _check_for_server_error(self, result_message: str) -> bool:
        """
        Check if the result message indicates an actual server error.
        
        Args:
            result_message: The result message from the worker
            
        Returns:
            True if an error is detected
        """
        error_indicators = [
            "exited quickly with error code",
            "exited with non-zero code",
            "error:",
            "failed to start",
            "permission denied",
            "address already in use",
        ]
        
        result_lower = result_message.lower()
        return any(indicator in result_lower for indicator in error_indicators)
    
    async def _clear_server_process(self, server_config: Dict[str, str]) -> None:
        """
        Clear the server process reference.
        
        Args:
            server_config: Configuration for the server
        """
        process_attr = server_config['process_attr']
        server_name = server_config['name']
        
        if hasattr(self.app, process_attr):
            current_process = getattr(self.app, process_attr)
            if current_process is not None:
                self.logger.warning(
                    f"{server_name} worker finished, but app.{process_attr} was not None. "
                    "Clearing it now."
                )
                setattr(self.app, process_attr, None)