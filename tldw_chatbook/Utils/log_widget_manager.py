"""
Log Widget Manager - Centralized management for log widget updates.

This module provides a unified interface for updating various log widgets
throughout the application, reducing code duplication in app.py.
"""

from typing import Dict, TYPE_CHECKING
from textual.widgets import RichLog
from textual.css.query import QueryError
from loguru import logger

if TYPE_CHECKING:
    from textual.widget import Widget


class LogWidgetManager:
    """Manages updates to various log widgets in the application."""

    # Mapping of log types to their widget IDs
    LOG_WIDGET_IDS: Dict[str, str] = {
        "llamacpp": "#llamacpp-log-output",
        "transformers": "#transformers-log-output",
        "llamafile": "#llamafile-log-output",
        "vllm": "#vllm-log-output",
        "mlx": "#mlx-log-output",
        "onnx": "#onnx-log-output",
    }

    # User-friendly names for error messages
    LOG_NAMES: Dict[str, str] = {
        "llamacpp": "Llama.cpp",
        "transformers": "Transformers",
        "llamafile": "Llamafile",
        "vllm": "vLLM",
        "mlx": "MLX-LM",
        "onnx": "ONNX",
    }

    @staticmethod
    def update_log(query_root: "Widget", log_type: str, message: str) -> None:
        """
        Update a specific log widget with a message.

        Args:
            query_root: The mounted destination widget that owns the log.
            log_type: The type of log to update (e.g., 'llamacpp', 'vllm')
            message: The message to write to the log
        """
        widget_id = LogWidgetManager.LOG_WIDGET_IDS.get(log_type)
        if not widget_id:
            logger.error(f"Unknown log type: {log_type}")
            return

        log_name = LogWidgetManager.LOG_NAMES.get(log_type, log_type)

        try:
            log_widget = query_root.query_one(widget_id, RichLog)
            log_widget.write(message)
        except QueryError:
            logger.error(f"Failed to query {widget_id} to write message.")
        except Exception as exc:
            logger.error(
                "Destination log update failed (log={}, category={}).",
                log_name,
                type(exc).__name__,
            )

    @staticmethod
    def update_transformers_log(query_root: "Widget", message: str) -> None:
        """Helper to write messages to the Transformers log widget."""
        LogWidgetManager.update_log(query_root, "transformers", message)
