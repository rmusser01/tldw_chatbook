# /tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_onnx.py
#
from __future__ import annotations
#
# Imports
import functools
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
#
# 3rd-Party Imports
from textual.containers import Container
from textual.css.query import QueryError
from textual.widgets import Input, RichLog, TextArea, Button
#
# Local Imports
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
# Import shared helpers
from .llm_management_events import _make_path_update_callback
#
########################################################################################################################
#
# --- Worker-specific functions ---

def _set_onnx_process_on_app(app_instance: "TldwCli", process: Optional[subprocess.Popen]):
    """Helper to set/clear the ONNX process on the app instance from the worker thread."""
    app_instance.onnx_server_process = process
    if process and process.pid:
        app_instance.loguru_logger.info(f"Stored ONNX process PID {process.pid} on app instance.")
    else:
        app_instance.loguru_logger.info("Cleared ONNX process from app instance.")


def _update_onnx_log(app_instance: "TldwCli", message: str) -> None:
    """Helper to write messages to the ONNX log widget."""
    try:
        log_widget = app_instance.query_one("#onnx-log-output", RichLog)
        log_widget.write(message)
    except QueryError:
        app_instance.loguru_logger.error("Failed to query #onnx-log-output to write message.")
    except Exception as e:
        app_instance.loguru_logger.error(f"Error writing to ONNX log: {e}", exc_info=True)


def run_onnx_server_worker(app_instance: "TldwCli", command: List[str]) -> str | None:
    """Background worker to run a generic ONNX server script and stream its output."""
    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    quoted_command = ' '.join(shlex.quote(c) for c in command)
    logger.info(f"ONNX WORKER starting with command: {quoted_command}")

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
        logger.info(f"ONNX WORKER: Subprocess launched, PID: {pid_str}")

        app_instance.call_from_thread(_set_onnx_process_on_app, app_instance, process)
        app_instance.call_from_thread(_update_onnx_log, app_instance, f"[PID:{pid_str}] ONNX server starting...\n")

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                app_instance.call_from_thread(_update_onnx_log, app_instance, line)
            process.stdout.close()

        process.wait()
        exit_code = process.returncode if process.returncode is not None else -1
        final_status_message = f"ONNX server (PID:{pid_str}) exited with code: {exit_code}."
        logger.info(final_status_message)
        app_instance.call_from_thread(_update_onnx_log, app_instance, f"\n--- {final_status_message} ---\n")
        return final_status_message
    except FileNotFoundError:
        msg = f"ERROR: Python interpreter or script not found. Command: {command[0]}"
        logger.error(msg)
        app_instance.call_from_thread(_update_onnx_log, app_instance, f"[bold red]{msg}[/]\n")
        raise
    except Exception as err:
        msg = f"CRITICAL ERROR in ONNX worker: {err} (Command: {quoted_command})"
        logger.error(msg, exc_info=True)
        app_instance.call_from_thread(_update_onnx_log, app_instance, f"[bold red]{msg}[/]\n")
        raise
    finally:
        logger.info(f"ONNX WORKER: Worker function for command '{quoted_command}' finishing.")
        app_instance.call_from_thread(_set_onnx_process_on_app, app_instance, None)
        if process and process.poll() is None:
            logger.warning(f"ONNX WORKER (PID:{pid_str}): Process still running in finally. Terminating.")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


# --- Event Handlers ---

async def handle_onnx_browse_python_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Handles browse for Python executable for ONNX server."""
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Python executable",
            context="onnx_models"
        ),
        callback=_make_path_update_callback(app, "onnx-python-path")
    )


async def handle_onnx_browse_script_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Handles browse for ONNX server script."""
    filters = Filters(("Python Scripts (*.py)", lambda p: p.suffix.lower() == ".py"),
                      ("All files (*.*)", lambda p: True))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select ONNX server script",
            filters=filters,
            context="onnx_models"
        ),
        callback=_make_path_update_callback(app, "onnx-script-path")
    )


async def handle_onnx_browse_model_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Handles browse for ONNX model file or directory."""
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select ONNX Model Directory (select any file inside)",
            context="onnx_models"
        ),
        callback=_make_path_update_callback(app, "onnx-model-path", is_directory=True)
    )


async def handle_start_onnx_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Handles the 'Start ONNX Server' button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start ONNX server.")

    log_output_widget: Optional[RichLog] = None

    try:
        python_path = app.query_one("#onnx-python-path", Input).value.strip()
        script_path = app.query_one("#onnx-script-path", Input).value.strip()
        model_path = app.query_one("#onnx-model-path", Input).value.strip()
        host = app.query_one("#onnx-host", Input).value.strip() or "127.0.0.1"
        port = app.query_one("#onnx-port", Input).value.strip() or "8004"
        additional_args_str = app.query_one("#onnx-additional-args", TextArea).text.strip()
        log_output_widget = app.query_one("#onnx-log-output", RichLog)

        log_output_widget.clear()

        if not python_path:
            app.notify("Python path is required.", severity="error")
            return
        if not script_path:
            app.notify("Server script path is required.", severity="error")
            return
        if not Path(script_path).is_file():
            app.notify(f"Script not found: {script_path}", severity="error")
            return

        command = [python_path, script_path]
        if model_path: command.extend(["--model", model_path])
        if host: command.extend(["--host", host])
        if port: command.extend(["--port", port])
        if additional_args_str: command.extend(shlex.split(additional_args_str))

        log_output_widget.write(f"Executing: {' '.join(shlex.quote(c) for c in command)}\n")

        worker_callable = functools.partial(run_onnx_server_worker, app, command)
        app.run_worker(
            worker_callable,
            group="onnx_server",
            description="Running ONNX server process",
            exclusive=True,
            thread=True
        )
        app.notify("ONNX server starting…")

    except QueryError as e:
        logger.error(f"UI Error starting ONNX server: {e}", exc_info=True)
        app.notify("Error accessing ONNX UI elements.", severity="error")
    except Exception as e:
        logger.error(f"Error starting ONNX server: {e}", exc_info=True)
        if log_output_widget:
            log_output_widget.write(f"An unexpected error occurred: {e}")
        app.notify(f"An unexpected error occurred: {e}", severity="error")


async def handle_stop_onnx_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Handles the 'Stop ONNX Server' button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to stop ONNX server.")

    log_output_widget: Optional[RichLog] = None
    try:
        log_output_widget = app.query_one("#onnx-log-output", RichLog)
        process_to_stop = app.onnx_server_process

        if process_to_stop and process_to_stop.poll() is None:
            pid = process_to_stop.pid
            log_output_widget.write(f"Stopping ONNX server (PID: {pid})...")
            process_to_stop.terminate()
            try:
                process_to_stop.wait(timeout=10)
                log_output_widget.write(f"ONNX server (PID: {pid}) stopped.")
                app.notify("ONNX server stopped.")
            except subprocess.TimeoutExpired:
                log_output_widget.write(f"ONNX server (PID: {pid}) did not stop gracefully. Killing...")
                process_to_stop.kill()
                process_to_stop.wait()
                app.notify("ONNX server killed.", severity="warning")
        else:
            log_output_widget.write("ONNX server is not running.")
            app.notify("ONNX server is not running.", severity="warning")

        if hasattr(app, 'onnx_server_process'):
            app.onnx_server_process = None

    except QueryError as e:
        logger.error(f"UI Error stopping ONNX server: {e}", exc_info=True)
        app.notify("Error accessing ONNX UI elements.", severity="error")
    except Exception as e:
        logger.error(f"Error stopping ONNX server: {e}", exc_info=True)
        if log_output_widget:
            log_output_widget.write(f"An unexpected error occurred: {e}")
        app.notify(f"An unexpected error occurred: {e}", severity="error")

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
