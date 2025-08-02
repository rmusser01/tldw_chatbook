"""
Log Widget Manager - Centralized management for log widget updates.

This module provides a unified interface for updating various log widgets
throughout the application, reducing code duplication in app.py.
"""

from typing import Optional, Dict, TYPE_CHECKING
from textual.widgets import RichLog
from textual.css.query import QueryError
from loguru import logger

if TYPE_CHECKING:
    from textual.app import App


class LogWidgetManager:
    """Manages updates to various log widgets in the application."""
    
    # Mapping of log types to their widget IDs
    LOG_WIDGET_IDS: Dict[str, str] = {
        'llamacpp': '#llamacpp-log-output',
        'transformers': '#transformers-log-output',
        'llamafile': '#llamafile-log-output',
        'vllm': '#vllm-log-output',
        'model_download': '#model-download-log-output',
        'mlx': '#mlx-log-output'
    }
    
    # User-friendly names for error messages
    LOG_NAMES: Dict[str, str] = {
        'llamacpp': 'Llama.cpp',
        'transformers': 'Transformers',
        'llamafile': 'Llamafile',
        'vllm': 'vLLM',
        'model_download': 'model download',
        'mlx': 'MLX-LM'
    }
    
    @staticmethod
    def update_log(app: 'App', log_type: str, message: str) -> None:
        """
        Update a specific log widget with a message.
        
        Args:
            app: The Textual app instance
            log_type: The type of log to update (e.g., 'llamacpp', 'vllm')
            message: The message to write to the log
        """
        widget_id = LogWidgetManager.LOG_WIDGET_IDS.get(log_type)
        if not widget_id:
            logger.error(f"Unknown log type: {log_type}")
            return
        
        log_name = LogWidgetManager.LOG_NAMES.get(log_type, log_type)
        
        try:
            log_widget = app.query_one(widget_id, RichLog)
            log_widget.write(message)
        except QueryError:
            logger.error(f"Failed to query {widget_id} to write message.")
        except Exception as e:
            logger.error(f"Error writing to {log_name} log: {e}", exc_info=True)
    
    @staticmethod
    def update_llamacpp_log(app: 'App', message: str) -> None:
        """Helper to write messages to the Llama.cpp log widget."""
        LogWidgetManager.update_log(app, 'llamacpp', message)
    
    @staticmethod
    def update_transformers_log(app: 'App', message: str) -> None:
        """Helper to write messages to the Transformers log widget."""
        LogWidgetManager.update_log(app, 'transformers', message)
    
    @staticmethod
    def update_llamafile_log(app: 'App', message: str) -> None:
        """Helper to write messages to the Llamafile log widget."""
        LogWidgetManager.update_log(app, 'llamafile', message)
    
    @staticmethod
    def update_vllm_log(app: 'App', message: str) -> None:
        """Helper to write messages to the vLLM log widget."""
        LogWidgetManager.update_log(app, 'vllm', message)
    
    @staticmethod
    def update_model_download_log(app: 'App', message: str) -> None:
        """Helper to write messages to the model download log widget."""
        LogWidgetManager.update_log(app, 'model_download', message)
    
    @staticmethod
    def update_mlx_log(app: 'App', message: str) -> None:
        """Helper to write messages to the MLX-LM log widget."""
        LogWidgetManager.update_log(app, 'mlx', message)