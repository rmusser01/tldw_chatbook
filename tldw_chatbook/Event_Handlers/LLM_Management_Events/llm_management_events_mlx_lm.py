# /tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_mlx_lm.py
#
# Imports
from __future__ import annotations
#
import functools
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
#
# 3rd-party Imports
from textual.containers import Container
from textual.css.query import QueryError
from textual.widgets import Input, RichLog, TextArea, Button
#
# Local Imports
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import _make_path_update_callback
#
########################################################################################################################
#
# --- Worker-specific functions ---

def _set_mlx_lm_process_on_app(app_instance: "TldwCli", process: Optional[subprocess.Popen]):
    """Helper to set/clear the MLX-LM process on the app instance from the worker thread."""
    app_instance.mlx_server_process = process
    if process and hasattr(process, 'pid') and process.pid is not None:
        app_instance.loguru_logger.info(f"Stored MLX-LM process PID {process.pid} on app instance.")
    else:
        app_instance.loguru_logger.info("Cleared MLX-LM process from app instance (or process was None).")


def _update_mlx_log(app_instance: "TldwCli", message: str) -> None:
    """Helper to write messages to the MLX-LM log widget."""
    try:
        log_widget = app_instance.query_one("#mlx-log-output", RichLog)
        log_widget.write(message)
    except QueryError:
        app_instance.loguru_logger.error("Failed to query #mlx-log-output to write message.")
    except Exception as e:
        app_instance.loguru_logger.error(f"Error writing to MLX-LM log: {e}", exc_info=True)


def run_mlx_lm_server_worker(app_instance: "TldwCli", command: List[str]) -> str | None:
    """Background worker to run the MLX-LM server and stream its output."""
    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    quoted_command = ' '.join(shlex.quote(c) for c in command)
    logger.info(f"MLX-LM WORKER starting with command: {quoted_command}")

    process: Optional[subprocess.Popen] = None
    pid_str = "N/A"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            env=env
        )
        pid_str = str(process.pid) if process and process.pid else "UnknownPID"
        logger.info(f"MLX-LM WORKER: Subprocess launched, PID: {pid_str}")

        app_instance.call_from_thread(_set_mlx_lm_process_on_app, app_instance, process)
        app_instance.call_from_thread(_update_mlx_log, app_instance, f"[PID:{pid_str}] MLX-LM server starting...\n")

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                app_instance.call_from_thread(_update_mlx_log, app_instance, line)
            process.stdout.close()

        process.wait()
        exit_code = process.returncode if process.returncode is not None else -1
        final_status_message = f"MLX-LM server (PID:{pid_str}) exited with code: {exit_code}."
        logger.info(final_status_message)
        app_instance.call_from_thread(_update_mlx_log, app_instance, f"\n--- {final_status_message} ---\n")
        return final_status_message
    except FileNotFoundError:
        msg = f"ERROR: Python or mlx_lm.server not found. Command: {command[0]}"
        logger.error(msg)
        app_instance.call_from_thread(_update_mlx_log, app_instance, f"[bold red]{msg}[/]\n")
        raise
    except Exception as err:
        msg = f"CRITICAL ERROR in MLX-LM worker: {err} (Command: {quoted_command})"
        logger.error(msg, exc_info=True)
        app_instance.call_from_thread(_update_mlx_log, app_instance, f"[bold red]{msg}[/]\n")
        raise
    finally:
        logger.info(f"MLX-LM WORKER: Worker function for command '{quoted_command}' finishing.")
        app_instance.call_from_thread(_set_mlx_lm_process_on_app, app_instance, None)
        if process and process.poll() is None:
            from tldw_chatbook.Local_Inference.mlx_lm_inference_local import stop_mlx_lm_server
            logger.warning(f"MLX-LM WORKER (PID:{pid_str}): Process still running in finally. Terminating.")
            stop_mlx_lm_server(process)


# --- Event Handlers ---

async def handle_mlx_lm_nav_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Handle the MLX-LM navigation button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("MLX-LM nav button pressed.")

    try:
        content_pane = app.query_one("#llm-content-pane", Container)
        for view in content_pane.query(".llm-view-area"):
            if view.id:
                view.styles.display = "none"

        mlx_lm_view = app.query_one("#llm-view-mlx-lm", Container)
        mlx_lm_view.styles.display = "block"

        if not hasattr(app, 'mlx_server_process'):
            app.mlx_server_process = None

        start_button = mlx_lm_view.query_one("#mlx-start-server-button", Button)
        stop_button = mlx_lm_view.query_one("#mlx-stop-server-button", Button)

        is_running = app.mlx_server_process and app.mlx_server_process.poll() is None
        start_button.disabled = is_running
        stop_button.disabled = not is_running
    except QueryError as e:
        logger.error(f"QueryError in handle_mlx_lm_nav_button_pressed: {e}", exc_info=True)
        app.notify("Error switching to MLX-LM view: Could not find required UI elements.", severity="error")


async def handle_start_mlx_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Starts the MLX-LM server using a non-blocking worker."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start MLX-LM server.")

    log_output_widget: Optional[RichLog] = None
    try:
        llm_mlx_view_container = app.query_one("#llm-view-mlx-lm", Container)
        model_path_input = llm_mlx_view_container.query_one("#mlx-model-path", Input)
        host_input = llm_mlx_view_container.query_one("#mlx-host", Input)
        port_input = llm_mlx_view_container.query_one("#mlx-port", Input)
        additional_args_area = llm_mlx_view_container.query_one("#mlx-additional-args", TextArea)
        log_output_widget = llm_mlx_view_container.query_one("#mlx-log-output", RichLog)

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

        if app.mlx_server_process and app.mlx_server_process.poll() is None:
            app.notify("MLX-LM server is already running.", severity="warning")
            return

        command = ["python", "-m", "mlx_lm.server", "--model", model_path, "--host", host, "--port", port_str]
        if additional_args:
            command.extend(shlex.split(additional_args))

        log_output_widget.write(f"Executing: {' '.join(shlex.quote(c) for c in command)}\n")

        worker_callable = functools.partial(run_mlx_lm_server_worker, app, command)
        app.run_worker(
            worker_callable,
            group="mlx_lm_server",
            description="Running MLX-LM server process",
            exclusive=True,
            thread=True
        )
        app.notify("MLX-LM server starting…")
    except QueryError as e:
        logger.error(f"UI Error starting MLX server: {e}", exc_info=True)
        app.notify("Error accessing MLX-LM UI elements.", severity="error")
    except Exception as e:
        logger.error(f"Error starting MLX-LM server: {e}", exc_info=True)
        if log_output_widget:
            log_output_widget.write(f"An unexpected error occurred: {e}")
        app.notify(f"An unexpected error occurred: {e}", severity="error")


async def handle_stop_mlx_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Stops the MLX-LM server process if it is running."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to stop MLX-LM server.")

    from tldw_chatbook.Local_Inference.mlx_lm_inference_local import stop_mlx_lm_server

    log_output_widget: Optional[RichLog] = None
    try:
        log_output_widget = app.query_one("#mlx-log-output", RichLog)
        process_to_stop = app.mlx_server_process

        if process_to_stop and process_to_stop.poll() is None:
            pid = process_to_stop.pid
            log_output_widget.write(f"Stopping MLX-LM server (PID: {pid})...")
            stop_mlx_lm_server(process_to_stop)
            app.mlx_server_process = None
            log_output_widget.write(f"MLX-LM server (PID: {pid}) stop command issued.")
            app.notify("MLX-LM server stopped.")
        else:
            log_output_widget.write("MLX-LM server is not currently running.")
            app.notify("MLX-LM server is not running.", severity="warning")
            if hasattr(app, 'mlx_server_process'):
                app.mlx_server_process = None

    except QueryError as e:
        logger.error(f"UI Error stopping MLX server: {e}", exc_info=True)
        app.notify("Error accessing MLX-LM UI elements.", severity="error")
    except Exception as e:
        logger.error(f"Error stopping MLX-LM server: {e}", exc_info=True)
        if log_output_widget:
            log_output_widget.write(f"An unexpected error occurred while stopping the server: {e}")
        app.notify(f"An unexpected error occurred: {e}", severity="error")

async def handle_mlx_browse_model_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Handle the MLX-LM browse model button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("MLX-LM browse model button pressed.")

    # Define filters for model files or directories
    model_filters = Filters(
        ("All files (*.*)", lambda p: True))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select MLX-LM Model or Directory",
            filters=model_filters,
            context="mlx_models"
        ),
        callback=_make_path_update_callback(app, "mlx-model-path")
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
