"""llm_management_events_vllm.py

A collection of helper callbacks, worker functions and event‑handler
coroutines specifically for the **vLLM** back‑end in the
**LLM Management** tab of *tldw‑cli*.

This module isolates vLLM-specific logic from the main llm_management_events.py.
"""

# Imports
from __future__ import annotations

#
import re
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

#
# Third-party Libraries
from textual.widgets import Button

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
#
# Local Imports
from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import (
    _make_path_update_callback,
)
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen as FileOpen,
    EnhancedSelectDirectory,
)
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from .server_lifecycle import (
    ServerLaunchClaim,
    run_server_subprocess,
    stop_server_process,
)
from tldw_chatbook.UI.LLM_Management.vllm_setup import VllmReadinessState
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
    "handle_vllm_setup_check_requested",
    "handle_vllm_setup_start_requested",
    "handle_vllm_setup_stop_requested",
    "handle_vllm_local_directory_browse_requested",
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
    """Reject the retired launcher; vLLM starts only from a checked draft."""

    app.notify("Use Check setup before starting vLLM.", severity="warning")


async def handle_stop_vllm_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Stops the vLLM server process if it's running."""
    await stop_server_process(app, "vllm", "vLLM server")
    return


def handle_vllm_setup_check_requested(
    window: "LLMManagementWindow", app: "TldwCli", event: Any
) -> None:
    """Compatibility delegate; LLMScreen owns the readiness lifecycle."""

    controller = getattr(
        getattr(window, "screen", None), "_on_vllm_check_requested", None
    )
    if callable(controller):
        controller(event)


def handle_vllm_setup_start_requested(
    window: "LLMManagementWindow", app: "TldwCli", event: Any
) -> None:
    """Compatibility delegate; command construction belongs to LLMScreen."""

    controller = getattr(
        getattr(window, "screen", None), "_on_vllm_start_requested", None
    )
    if callable(controller):
        controller(event)
        return
    view = window.query_one("#vllm-setup-view")
    view.apply_state(
        draft=event.draft,
        state=VllmReadinessState.NEEDS_ATTENTION,
        preflight=view.preflight,
    )


async def handle_vllm_setup_stop_requested(
    window: "LLMManagementWindow", app: "TldwCli", event: Any
) -> None:
    """Compatibility delegate for callers outside the mounted Lab screen."""

    controller = getattr(
        getattr(window, "screen", None), "_on_vllm_stop_requested", None
    )
    if callable(controller):
        result = controller(event)
        if hasattr(result, "__await__"):
            await result
        return

    view = window.query_one("#vllm-setup-view")
    view.apply_state(
        draft=view.draft,
        state=VllmReadinessState.STOPPING,
        preflight=view.preflight,
    )
    stopped = await stop_server_process(app, "vllm", "vLLM server")
    view.apply_state(
        draft=view.draft,
        state=(
            VllmReadinessState.NOT_CONFIGURED
            if stopped
            else VllmReadinessState.NEEDS_ATTENTION
        ),
        preflight=(None if stopped else view.preflight),
    )


async def handle_vllm_local_directory_browse_requested(
    window: "LLMManagementWindow", app: "TldwCli", event: Any
) -> None:
    """Open a local-only directory picker for the vLLM source field."""

    await app.push_screen(
        EnhancedSelectDirectory(
            location=str(Path.home()),
            title="Select local vLLM model directory",
            context="vllm_models",
        ),
        callback=_make_path_update_callback(window, app, "vllm-local-model-directory"),
    )


# --- Button Handler Map ---
VLLM_BUTTON_HANDLERS: dict[str, object] = {}

#
# End of llm_management_events_vllm.py
########################################################################################################################
