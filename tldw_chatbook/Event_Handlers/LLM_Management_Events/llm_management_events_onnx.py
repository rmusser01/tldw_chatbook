# /tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_onnx.py
#
from __future__ import annotations

#
# Imports
import functools
from loguru import logger as _loguru_fallback_logger
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

#
# 3rd-Party Imports
from textual.css.query import QueryError
from textual.widgets import Input, RichLog, TextArea, Button

#
# Local Imports
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
# Import shared helpers
from .llm_management_events import _make_path_update_callback
from .server_lifecycle import (
    ServerLaunchClaim,
    release_server_claim,
    reserve_server_launch,
    run_server_subprocess,
    stop_server_process,
)
#
########################################################################################################################
#
# --- Worker-specific functions ---


def run_onnx_server_worker(
    app_instance: "TldwCli",
    command: List[str],
    claim: ServerLaunchClaim,
) -> str | None:
    """Background worker to run a generic ONNX server script and stream its output."""
    return run_server_subprocess(
        app_instance,
        "onnx",
        command,
        claim,
        subprocess,
    )


async def handle_onnx_browse_python_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles browse for Python executable for ONNX server."""
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Python executable",
            context="onnx_models",
        ),
        callback=_make_path_update_callback(window, app, "onnx-python-path"),
    )


async def handle_onnx_browse_script_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles browse for ONNX server script."""
    filters = Filters(
        ("Python Scripts (*.py)", lambda p: p.suffix.lower() == ".py"),
        ("All files (*.*)", lambda p: True),
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select ONNX server script",
            filters=filters,
            context="onnx_models",
        ),
        callback=_make_path_update_callback(window, app, "onnx-script-path"),
    )


async def handle_onnx_browse_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles browse for ONNX model file or directory."""
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select ONNX Model Directory (select any file inside)",
            context="onnx_models",
        ),
        callback=_make_path_update_callback(
            window, app, "onnx-model-path", is_directory=True
        ),
    )


async def handle_start_onnx_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Start ONNX Server' button press."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("User requested to start ONNX server.")

    log_output_widget: Optional[RichLog] = None

    try:
        python_path = window.query_one("#onnx-python-path", Input).value.strip()
        script_path = window.query_one("#onnx-script-path", Input).value.strip()
        model_path = window.query_one("#onnx-model-path", Input).value.strip()
        host = window.query_one("#onnx-host", Input).value.strip() or "127.0.0.1"
        port = window.query_one("#onnx-port", Input).value.strip() or "8004"
        additional_args_str = window.query_one(
            "#onnx-additional-args", TextArea
        ).text.strip()
        log_output_widget = window.query_one("#onnx-log-output", RichLog)

        log_output_widget.clear()

        if not python_path:
            app.notify("Python path is required.", severity="error")
            return
        if not script_path:
            app.notify("Server script path is required.", severity="error")
            return
        if not Path(script_path).is_file():
            app.notify("ONNX server script was not found.", severity="error")
            return

        command = [python_path, script_path]
        if model_path:
            command.extend(["--model", model_path])
        if host:
            command.extend(["--host", host])
        if port:
            command.extend(["--port", port])
        if additional_args_str:
            command.extend(shlex.split(additional_args_str))

        claim = reserve_server_launch(app, "onnx")
        if claim is None:
            window._sync_process_controls("onnx")
            app.notify(
                "ONNX server is already starting or running.", severity="warning"
            )
            return
        log_output_widget.write("Starting ONNX server.\n")

        worker_callable = functools.partial(run_onnx_server_worker, app, command, claim)
        app.run_worker(
            worker_callable,
            group="onnx_server",
            description="Running ONNX server process",
            exclusive=True,
            thread=True,
        )
        window._sync_process_controls("onnx")
        app.notify("ONNX server starting…")

    except QueryError:
        if "claim" in locals():
            release_server_claim(app, "onnx", claim)
        window._sync_process_controls("onnx")
        logger.error("ONNX start failed (category=QueryError).")
        app.notify("Error accessing ONNX UI elements.", severity="error")
    except Exception as e:
        if "claim" in locals():
            release_server_claim(app, "onnx", claim)
        window._sync_process_controls("onnx")
        logger.error("ONNX start failed (category={}).", type(e).__name__)
        if log_output_widget:
            log_output_widget.write("ONNX start failed.")
        app.notify("An unexpected error occurred.", severity="error")


async def handle_stop_onnx_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Stop ONNX Server' button press."""
    await stop_server_process(app, "onnx", "ONNX server")
    return


# --- Button Handler Map ---
ONNX_BUTTON_HANDLERS = {
    "onnx-browse-python-button": handle_onnx_browse_python_button_pressed,
    "onnx-browse-script-button": handle_onnx_browse_script_button_pressed,
    "onnx-browse-model-button": handle_onnx_browse_model_button_pressed,
    "onnx-start-server-button": handle_start_onnx_server_button_pressed,
    "onnx-stop-server-button": handle_stop_onnx_server_button_pressed,
}

#
# End of llm_management_events_onnx.py
########################################################################################################################
