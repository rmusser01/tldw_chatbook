"""llm_management_events_vllm.py

A collection of helper callbacks, worker functions and event‑handler
coroutines specifically for the **vLLM** back‑end in the
**LLM Management** tab of *tldw‑cli*.

This module isolates vLLM-specific logic from the main llm_management_events.py.
"""

# Imports
from __future__ import annotations

#
import functools
from loguru import logger as _loguru_fallback_logger
import re
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List

#
# Third-party Libraries
from textual.widgets import Input, RichLog, TextArea, Button

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
#
# Local Imports
from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import (
    _make_path_update_callback,
)
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from .server_lifecycle import (
    ServerLaunchClaim,
    release_server_claim,
    reserve_server_launch,
    run_server_subprocess,
    stop_server_process,
)
#
#
########################################################################################################################
#
# Security functions for input validation


def validate_python_path(python_path: str) -> bool:
    """Validate python executable path to prevent command injection."""
    if not python_path:
        return False

    # Allow only simple python executable names or absolute paths
    # Reject paths with shell metacharacters
    safe_pattern = re.compile(r"^[a-zA-Z0-9_.\-/\\:]+$")
    if not safe_pattern.match(python_path):
        return False

    # Common python executable names
    allowed_names = {
        "python",
        "python3",
        "python3.8",
        "python3.9",
        "python3.10",
        "python3.11",
        "python3.12",
    }

    # If it's just a name (no path), check against whitelist
    if "/" not in python_path and "\\" not in python_path:
        return python_path in allowed_names

    # For paths, validate they don't contain dangerous patterns
    dangerous_patterns = ["&&", "||", ";", "|", ">", "<", "`", "$", "(", ")"]
    return not any(pattern in python_path for pattern in dangerous_patterns)


def validate_host(host: str) -> bool:
    """Validate host address to prevent command injection."""
    if not host:
        return False

    # IPv4 pattern
    ipv4_pattern = re.compile(
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    )
    # Hostname pattern (simplified)
    hostname_pattern = re.compile(
        r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$"
    )

    return (
        host == "localhost"
        or host == "127.0.0.1"
        or host == "0.0.0.0"
        or ipv4_pattern.match(host)
        or hostname_pattern.match(host)
    )


def validate_port(port: str) -> bool:
    """Validate port number to prevent command injection."""
    if not port:
        return False

    try:
        port_num = int(port)
        return 1 <= port_num <= 65535
    except ValueError:
        return False


def validate_model_path(model_path: str) -> bool:
    """Validate model path to prevent command injection."""
    if not model_path:
        return False

    # Allow alphanumeric, hyphens, underscores, dots, slashes for paths and HF repo IDs
    safe_pattern = re.compile(r"^[a-zA-Z0-9_.\-/\\:]+$")
    if not safe_pattern.match(model_path):
        return False

    # Reject paths with dangerous shell metacharacters
    dangerous_patterns = ["&&", "||", ";", "|", ">", "<", "`", "$", "(", ")"]
    return not any(pattern in model_path for pattern in dangerous_patterns)


def validate_additional_args(args_str: str) -> bool:
    """Validate additional arguments to prevent command injection."""
    if not args_str.strip():
        return True  # Empty is fine

    try:
        # Use shlex to parse - this will raise ValueError for malformed input
        parsed_args = shlex.split(args_str)

        # Check each argument for dangerous patterns
        dangerous_patterns = ["&&", "||", ";", "|", ">", "<", "`", "$"]
        for arg in parsed_args:
            if any(pattern in arg for pattern in dangerous_patterns):
                return False

        return True
    except ValueError:
        # shlex.split failed, indicating malformed shell syntax
        return False


########################################################################################################################
#
# Functions:


__all__ = [
    "handle_vllm_browse_python_button_pressed",
    "handle_vllm_browse_model_button_pressed",
    "run_vllm_server_worker",
    "handle_start_vllm_server_button_pressed",
    "handle_stop_vllm_server_button_pressed",
]

###############################################################################
# ─── vLLM UI helpers ────────────────────────────────────────────────────────
###############################################################################


async def handle_vllm_browse_python_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Let the user pick the Python interpreter used for vLLM (venv, etc.)."""

    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Python interpreter for vLLM",
            filters=Filters(
                ("Python executable", lambda p: p.name.startswith("python"))
            ),
            context="vllm_models",
        ),
        callback=_make_path_update_callback(window, app, "vllm-python-path"),
    )


async def handle_vllm_browse_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Model (checkpoint or GGUF) for vLLM",
            filters=Filters(("All files", lambda p: True)),
            context="vllm_models",
        ),
        callback=_make_path_update_callback(window, app, "vllm-model-path"),
    )


###############################################################################
# ─── Worker functions
###############################################################################


# Helper to set/clear the process on the app instance from the worker thread
def run_vllm_server_worker(
    app_instance: "TldwCli",
    command: List[str],
    claim: ServerLaunchClaim,
) -> str:
    return run_server_subprocess(
        app_instance,
        "vllm",
        command,
        claim,
        subprocess,
    )


async def handle_start_vllm_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("User requested to start vLLM server.")

    try:
        python_path_input = window.query_one("#vllm-python-path", Input)
        model_path_input = window.query_one("#vllm-model-path", Input)
        host_input = window.query_one("#vllm-host", Input)
        port_input = window.query_one("#vllm-port", Input)
        additional_args_input = window.query_one("#vllm-additional-args", TextArea)
        log_output_widget = window.query_one("#vllm-log-output", RichLog)

        python_path = python_path_input.value.strip() or "python"
        model_path = (
            model_path_input.value.strip()
        )  # Can be repo ID, so Path().exists() might not apply
        host = (
            host_input.value.strip() or "127.0.0.1"
        )  # Default from snippet was 127.0.0.1
        port = (
            port_input.value.strip() or "8000"
        )  # Default from snippet was 8000, not 8002
        additional_args_str = additional_args_input.text.strip()

        # Validate all inputs to prevent command injection
        if not validate_python_path(python_path):
            app.notify(
                "Invalid Python executable path.",
                severity="error",
            )
            python_path_input.focus()
            return

        if not validate_model_path(model_path):
            app.notify(
                "Invalid model path.",
                severity="error",
            )
            model_path_input.focus()
            return

        if not validate_host(host):
            app.notify(
                "Invalid vLLM host.",
                severity="error",
            )
            host_input.focus()
            return

        if not validate_port(port):
            app.notify(
                "Invalid vLLM port.",
                severity="error",
            )
            port_input.focus()
            return

        if not validate_additional_args(additional_args_str):
            app.notify(
                "Invalid additional arguments. Arguments contain unsafe shell metacharacters.",
                severity="error",
            )
            additional_args_input.focus()
            return

        command = [
            python_path,
            "-m",
            "vllm.entrypoints.api_server",  # Corrected entrypoint
            "--host",
            host,
            "--port",
            port,
        ]
        if model_path:  # model_path is required for vLLM server
            command.extend(["--model", model_path])
        else:
            app.notify(
                "Model path (or HuggingFace Repo ID) is required for vLLM.",
                severity="error",
            )
            model_path_input.focus()
            return

        if additional_args_str:
            command.extend(shlex.split(additional_args_str))

        claim = reserve_server_launch(app, "vllm")
        if claim is None:
            window._sync_process_controls("vllm")
            app.notify(
                "vLLM server is already starting or running.", severity="warning"
            )
            return
        log_output_widget.clear()
        log_output_widget.write("Starting vLLM server.\n")

        worker_callable = functools.partial(
            run_vllm_server_worker,
            app,
            command,
            claim,
        )
        app.run_worker(
            worker_callable,
            group="vllm_server",
            description="Running vLLM API server",
            exclusive=True,
            thread=True,
        )
        window._sync_process_controls("vllm")
        app.notify("vLLM server starting…")
    except Exception as err:  # pragma: no cover
        if "claim" in locals():
            release_server_claim(app, "vllm", claim)
        window._sync_process_controls("vllm")
        logger.error("vLLM start failed (category={}).", type(err).__name__)
        app.notify("Error setting up vLLM server start.", severity="error")


async def handle_stop_vllm_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Stops the vLLM server process if it's running."""
    await stop_server_process(app, "vllm", "vLLM server")
    return


# --- Button Handler Map ---
VLLM_BUTTON_HANDLERS = {
    "vllm-browse-python-button": handle_vllm_browse_python_button_pressed,
    "vllm-browse-model-button": handle_vllm_browse_model_button_pressed,
    "vllm-start-server-button": handle_start_vllm_server_button_pressed,
    "vllm-stop-server-button": handle_stop_vllm_server_button_pressed,
}

#
# End of llm_management_events_vllm.py
########################################################################################################################
