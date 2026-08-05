# /tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_mlx_lm.py
#
# Imports
from __future__ import annotations

#
import functools
from loguru import logger as _loguru_fallback_logger
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

#
# 3rd-party Imports
from textual.css.query import QueryError
from textual.widgets import Input, RichLog, TextArea, Button

#
# Local Imports
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import (
    _make_path_update_callback,
)
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


def run_mlx_lm_server_worker(
    app_instance: "TldwCli",
    command: List[str],
    claim: ServerLaunchClaim,
) -> str | None:
    """Background worker to run the MLX-LM server and stream its output."""
    return run_server_subprocess(
        app_instance,
        "mlx",
        command,
        claim,
        subprocess,
    )


async def handle_start_mlx_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Starts the MLX-LM server using a non-blocking worker."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("User requested to start MLX-LM server.")

    log_output_widget: Optional[RichLog] = None
    try:
        model_path_input = window.query_one("#mlx-model-path", Input)
        host_input = window.query_one("#mlx-host", Input)
        port_input = window.query_one("#mlx-port", Input)
        additional_args_area = window.query_one("#mlx-additional-args", TextArea)
        log_output_widget = window.query_one("#mlx-log-output", RichLog)

        model_path = model_path_input.value.strip()
        host = host_input.value.strip() or "127.0.0.1"
        port_str = port_input.value.strip() or "8080"
        additional_args = additional_args_area.text.strip()

        log_output_widget.clear()

        if not model_path:
            app.notify("MLX Model Path is required.", severity="error")
            return
        try:
            int(port_str)
        except ValueError:
            app.notify("Port must be a valid number.", severity="error")
            return

        command = [
            "python",
            "-m",
            "mlx_lm.server",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            port_str,
        ]
        if additional_args:
            command.extend(shlex.split(additional_args))

        claim = reserve_server_launch(app, "mlx")
        if claim is None:
            window._sync_process_controls("mlx")
            app.notify(
                "MLX-LM server is already starting or running.", severity="warning"
            )
            return
        log_output_widget.write("Starting MLX-LM server.\n")

        worker_callable = functools.partial(
            run_mlx_lm_server_worker, app, command, claim
        )
        app.run_worker(
            worker_callable,
            group="mlx_lm_server",
            description="Running MLX-LM server process",
            exclusive=True,
            thread=True,
        )
        window._sync_process_controls("mlx")
        app.notify("MLX-LM server starting…")
    except QueryError:
        if "claim" in locals():
            release_server_claim(app, "mlx", claim)
        window._sync_process_controls("mlx")
        logger.error("MLX start failed (category=QueryError).")
        app.notify("Error accessing MLX-LM UI elements.", severity="error")
    except Exception as e:
        if "claim" in locals():
            release_server_claim(app, "mlx", claim)
        window._sync_process_controls("mlx")
        logger.error("MLX start failed (category={}).", type(e).__name__)
        if log_output_widget:
            log_output_widget.write("MLX start failed.")
        app.notify("An unexpected error occurred.", severity="error")


async def handle_stop_mlx_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Stops the MLX-LM server process if it is running."""
    await stop_server_process(app, "mlx", "MLX-LM server")
    return


async def handle_mlx_browse_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Open the model picker for a local MLX model."""

    model_filters = Filters(("All files (*.*)", lambda p: True))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select MLX-LM Model or Directory",
            filters=model_filters,
            context="mlx_models",
        ),
        callback=_make_path_update_callback(window, app, "mlx-model-path"),
    )


# --- Button Handler Map ---
MLX_LM_BUTTON_HANDLERS = {
    "mlx-start-server-button": handle_start_mlx_server_button_pressed,
    "mlx-stop-server-button": handle_stop_mlx_server_button_pressed,
    "mlx-browse-model-button": handle_mlx_browse_model_button_pressed,
}

#
# End of llm_management_events_mlx-lm.py
########################################################################################################################
