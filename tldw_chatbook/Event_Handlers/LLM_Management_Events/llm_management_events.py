# /tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py
#
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
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
#
# Third-party Imports
from textual.worker import Worker, WorkerState
from textual.widgets import Input, RichLog, TextArea, Button
from textual.css.query import QueryError
#
# Local Imports
from tldw_chatbook.Constants import LLAMA_CPP_SERVER_ARGS_HELP_TEXT, LLAMAFILE_SERVER_ARGS_HELP_TEXT, MLX_LM_SERVER_ARGS_HELP_TEXT
#
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters
#
########################################################################################################################
#
# Constants:

__all__ = [
    # Generic Helpers (Exported for other modules to use)
    "_make_path_update_callback",
    "_stream_process",
    "stream_worker_output_to_log",
    # Llamafile Handlers
    "handle_llamafile_browse_exec_button_pressed",
    "handle_llamafile_browse_model_button_pressed",
    "handle_start_llamafile_server_button_pressed",
    "handle_stop_llamafile_server_button_pressed",
    # Llama.cpp Handlers
    "handle_llamacpp_browse_exec_button_pressed",
    "handle_llamacpp_browse_model_button_pressed",
    "handle_start_llamacpp_server_button_pressed",
    "handle_stop_llamacpp_server_button_pressed",
    # General Setup
    "populate_llm_help_texts",
]

# --- Generic Helpers ---

def _make_path_update_callback(app: "TldwCli", input_widget_id: str, is_directory: bool = False):
    """
    Return a callback that sets an input widget's value to a picked path.
    If is_directory is True, it uses the parent directory of the selected file.
    """
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))

    async def _callback(selected_path: Optional[Path]) -> None:
        if selected_path:
            try:
                final_path = selected_path.parent if is_directory else selected_path
                input_widget = app.query_one(f"#{input_widget_id}", Input)
                input_widget.value = str(final_path)
                logger.info(f"Updated input {input_widget_id} with path: {final_path}")
            except Exception as err:
                logger.error(f"Error updating input #{input_widget_id}: {err}", exc_info=True)
                app.notify(f"Error setting path for {input_widget_id}.", severity="error")
        else:
            logger.info("File/Directory selection cancelled for #%s.", input_widget_id)
    return _callback


###############################################################################
# ─── Llamafile helpers ───────────────────────────────────────────────────────
###############################################################################


async def handle_llamafile_browse_exec_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Llamafile browse executable button pressed.")

    exec_filters = Filters(("Executables", lambda p: p.is_file()))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llamafile Executable",
            context="llm_models"
        ),
        callback=_make_path_update_callback(app, "llamafile-exec-path"))


async def handle_llamafile_browse_model_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Llamafile browse model button pressed.")

    gguf_filters = Filters(
        ("GGUF Models (*.gguf)", lambda p: p.suffix.lower() == ".gguf"),
        ("All files (*.*)", lambda p: True))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llamafile Model (.gguf)",
            filters=gguf_filters,
            context="llm_models"
        ),
        callback=_make_path_update_callback(app, "llamafile-model-path")
    )


###############################################################################
# ─── Worker functions
###############################################################################


# Each run_…_worker uses the same streaming pattern – consider refactoring, but
# explicit duplication keeps each implementation easy to tweak individually.

def _stream_process(app_instance: "TldwCli", log_fn_name: str, process: subprocess.Popen):
    """Utility: pump *process* stdout into *app_instance.log_fn_name*."""

    log_fn = getattr(app_instance, log_fn_name)
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            app_instance.call_from_thread(log_fn, line)
        process.stdout.close()


def _set_llamafile_process_on_app(app_instance: "TldwCli", process: Optional[subprocess.Popen]):
    app_instance.llamafile_server_process = process  # Assumes app_instance has this attribute
    if process and hasattr(process, 'pid') and process.pid is not None:
        app_instance.loguru_logger.info(f"Stored Llamafile process PID {process.pid} on app instance.")
    else:
        app_instance.loguru_logger.info("Cleared Llamafile process from app instance (or process was None).")


def run_llamafile_server_worker(app_instance: "TldwCli", command: List[str]) -> str:
    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    quoted_command = ' '.join(shlex.quote(c) for c in command)
    logger.info(f"Llamafile WORKER (diag v2) starting with command: {quoted_command}")

    process: Optional[subprocess.Popen] = None
    final_status_message = f"Llamafile WORKER (diag v2): Default status for {quoted_command}"
    pid_str = "N/A"

    # --- Determine the directory of the llamafile executable ---
    llamafile_executable_path = Path(command[0])
    llamafile_dir = llamafile_executable_path.parent
    logger.info(f"Llamafile WORKER: CWD will be: {llamafile_dir}")

    # Log a snippet of the current PATH for comparison
    current_env_path = os.environ.get("PATH", "PATH not found in os.environ")
    logger.debug(f"Llamafile WORKER: Python's current PATH (first 200 chars): {current_env_path[:200]}")

    try:
        logger.debug("Llamafile WORKER (diag v2): Attempting to start subprocess...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr separately
            text=True,
            universal_newlines=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            cwd=llamafile_dir,
            env=os.environ.copy()  # Pass a copy of the current environment
        )
        pid_str = str(process.pid) if process and process.pid else "UnknownPID"
        logger.info(f"Llamafile WORKER (diag v2): Subprocess launched, PID: {pid_str}, CWD: {llamafile_dir}")

        app_instance.call_from_thread(_set_llamafile_process_on_app, app_instance, process)
        app_instance.call_from_thread(app_instance._update_llamafile_log,
                                      f"[PID:{pid_str}] Llamafile server attempting to start (diag v2)...\n")

        # --- DIAGNOSTIC: Use communicate() with a short timeout for initial output ---
        initial_stdout = ""
        initial_stderr = ""
        quick_exit_code = None
        try:
            logger.debug(f"Llamafile WORKER (PID:{pid_str}): Attempting immediate communicate(timeout=5s)...")
            # This will block until process terminates OR timeout
            initial_stdout, initial_stderr = process.communicate(timeout=5)
            quick_exit_code = process.returncode
            logger.info(f"Llamafile WORKER (PID:{pid_str}): communicate() completed. Exit code: {quick_exit_code}")
            if initial_stdout:
                logger.info(f"Llamafile WORKER (PID:{pid_str}) Initial STDOUT:\n{initial_stdout.strip()}")
                app_instance.call_from_thread(app_instance._update_llamafile_log,
                                              f"--- Initial STDOUT (PID:{pid_str}) ---\n{initial_stdout.strip()}\n")
            if initial_stderr:
                logger.error(f"Llamafile WORKER (PID:{pid_str}) Initial STDERR:\n{initial_stderr.strip()}")
                app_instance.call_from_thread(app_instance._update_llamafile_log,
                                              f"--- Initial STDERR (PID:{pid_str}) ---\n[bold red]{initial_stderr.strip()}[/]\n")

            if quick_exit_code is not None:  # Process terminated within timeout
                if quick_exit_code != 0:
                    final_status_message = f"Llamafile server (PID:{pid_str}) EXITED QUICKLY with ERROR code: {quick_exit_code}."
                    if initial_stderr: final_status_message += f"\nInitial STDERR: {initial_stderr.strip()}"
                else:
                    final_status_message = f"Llamafile server (PID:{pid_str}) EXITED QUICKLY with code: {quick_exit_code} (SUCCESS)."
                    if initial_stdout: final_status_message += f"\nInitial STDOUT: {initial_stdout.strip()}"
                logger.info(final_status_message)
                app_instance.call_from_thread(app_instance._update_llamafile_log, f"{final_status_message}\n")
                return final_status_message  # Worker is done if process exited quickly

        except subprocess.TimeoutExpired:
            logger.info(
                f"Llamafile WORKER (PID:{pid_str}): communicate(timeout=5s) EXPIRED. Server is likely running (this is expected for a server).")
            app_instance.call_from_thread(app_instance._update_llamafile_log,
                                          f"[PID:{pid_str}] Server running after 5s. Switching to streaming...\n")
            # If it timed out, the server is running. Now proceed to the normal streaming loop.
            # The process object is still valid.
        # --- END DIAGNOSTIC ---

        # If communicate() timed out, the process is still running. Proceed with streaming.
        logger.info(f"Llamafile WORKER (PID:{pid_str}): Proceeding to continuous streaming as server is running...")
        stderr_lines_captured = []
        while True:  # Streaming loop
            output_received_in_iteration = False
            if process.poll() is not None:
                logger.info(f"Llamafile WORKER (PID:{pid_str}): Process terminated. Exit code: {process.returncode}")
                break

            if process.stdout:
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:
                            logger.info(f"Llamafile WORKER STDOUT (PID:{pid_str}): {line}")
                            app_instance.call_from_thread(app_instance._update_llamafile_log, f"{line}\n")
                            output_received_in_iteration = True
                    elif process.poll() is not None:
                        break
                except Exception as e_stdout:
                    logger.error(f"Llamafile WORKER (PID:{pid_str}): Exception reading stdout: {e_stdout}")
                    break

            if process.stderr:
                try:
                    line = process.stderr.readline()
                    if line:
                        line = line.strip()
                        if line:
                            logger.error(f"Llamafile WORKER STDERR (PID:{pid_str}): {line}")
                            stderr_lines_captured.append(line)  # Capture for final message
                            app_instance.call_from_thread(app_instance._update_llamafile_log,
                                                          f"[STDERR] [bold red]{line}[/]\n")
                            output_received_in_iteration = True
                    elif process.poll() is not None:
                        break
                except Exception as e_stderr:
                    logger.error(f"Llamafile WORKER (PID:{pid_str}): Exception reading stderr: {e_stderr}")
                    break

            if not output_received_in_iteration and process.poll() is None:
                time.sleep(0.1)

        if process.stdout:
            try:
                process.stdout.close()
            except Exception as e_close_stdout:
                logger.debug(f"Exception closing stdout (Llamafile): {e_close_stdout}")
        if process.stderr:
            try:
                process.stderr.close()
            except Exception as e_close_stderr:
                logger.debug(f"Exception closing stderr (Llamafile): {e_close_stderr}")

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Llamafile WORKER (PID:{pid_str}): Timeout on final wait post-streaming.")

        exit_code = process.returncode if process.returncode is not None else -1
        logger.info(
            f"Llamafile WORKER (PID:{pid_str}): Subprocess finally exited post-streaming with code: {exit_code}")

        if exit_code != 0:
            final_status_message = f"Llamafile server (PID:{pid_str}) exited post-streaming with non-zero code: {exit_code}."
            if stderr_lines_captured: final_status_message += "\nSTDERR:\n" + "\n".join(stderr_lines_captured)
        else:
            final_status_message = f"Llamafile server (PID:{pid_str}) exited post-streaming successfully (code: {exit_code})."

        app_instance.call_from_thread(app_instance._update_llamafile_log, f"{final_status_message}\n")
        return final_status_message

    except FileNotFoundError:
        msg = f"ERROR: Llamafile executable not found: {command[0]}"
        logger.error(msg)
        app_instance.call_from_thread(app_instance._update_llamafile_log, f"[bold red]{msg}[/]\n")
        raise
    except Exception as err:
        msg = f"CRITICAL ERROR in Llamafile worker: {err} (Command: {quoted_command})"
        logger.error(msg, exc_info=True)
        app_instance.call_from_thread(app_instance._update_llamafile_log, f"[bold red]{msg}[/]\n")
        raise
    finally:
        logger.info(f"Llamafile WORKER (persistent stream): Worker function for command '{quoted_command}' finishing.")
        # app_instance.call_from_thread(_set_llamafile_process_on_app, app_instance, None) # If managing process
        app_instance.call_from_thread(_set_llamafile_process_on_app, app_instance, None)
        if process and process.poll() is None:
            logger.warning(f"Llamafile WORKER (PID:{pid_str}): Process still running in finally. Terminating.")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.error(f"Llamafile WORKER (PID:{pid_str}): Process kill failed after terminate timeout.")
                process.kill()


# Helper to set/clear the process on the app instance from the worker thread
# This can be defined in this file or in app.py and imported.
def _set_llamacpp_process_on_app(app_instance: "TldwCli", process: Optional[subprocess.Popen]):
    """Helper to set/clear the Llama.cpp process on the app instance from the worker thread."""
    app_instance.llamacpp_server_process = process  # Assumes app_instance has this attribute
    if process and hasattr(process, 'pid') and process.pid is not None:
        app_instance.loguru_logger.info(f"Stored Llama.cpp process PID {process.pid} on app instance.")
    else:
        app_instance.loguru_logger.info("Cleared Llama.cpp process from app instance (or process was None).")


def run_llamacpp_server_worker(app_instance: "TldwCli", command: List[str]) -> str | None:
    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    quoted_command = ' '.join(shlex.quote(c) for c in command)
    logger.info(f"Llama.cpp WORKER (persistent stream) starting with command: {quoted_command}")

    process: Optional[subprocess.Popen] = None  # Ensure type hint
    final_status_message = f"Llama.cpp WORKER (persistent stream): Default status for {quoted_command}"
    pid_str = "N/A"

    try:
        logger.debug("Llama.cpp WORKER (persistent stream): Attempting to start subprocess...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        pid_str = str(process.pid) if process and process.pid else "UnknownPID"
        logger.info(f"Llama.cpp WORKER (persistent stream): Subprocess launched, PID: {pid_str}")

        # Store the process object on the app instance
        app_instance.call_from_thread(_set_llamacpp_process_on_app, app_instance, process)

        app_instance.call_from_thread(app_instance._update_llamacpp_log,
                                      f"[PID:{pid_str}] Llama.cpp server starting...\n")

        # Non-blocking read loop
        while True:
            output_received_in_iteration = False

            # Check process health first
            if process.poll() is not None:  # Process has terminated
                logger.info(
                    f"Llama.cpp WORKER (PID:{pid_str}): Process terminated during read loop. Exit code: {process.returncode}")
                break  # Exit the while True loop

            # Check stdout
            if process.stdout:
                try:
                    # For a truly non-blocking read on a pipe, os.set_blocking(fd, False) and select would be needed.
                    # readline() can block. If the server sends no newlines but keeps pipe open, it blocks.
                    # However, most servers line-buffer or send newlines.
                    line = process.stdout.readline()  # This can block
                    if line:  # If readline returns empty string, pipe is closed or EOF
                        line = line.strip()
                        if line:
                            logger.info(f"Llama.cpp WORKER STDOUT (PID:{pid_str}): {line}")
                            app_instance.call_from_thread(app_instance._update_llamacpp_log, f"{line}\n")
                            output_received_in_iteration = True
                    elif process.poll() is not None:  # Check again if process ended after readline returned empty
                        break
                except Exception as e_stdout:
                    logger.error(f"Llama.cpp WORKER (PID:{pid_str}): Exception reading stdout: {e_stdout}")
                    break

                    # Check stderr
            if process.stderr:
                try:
                    line = process.stderr.readline()  # This can block
                    if line:
                        line = line.strip()
                        if line:
                            logger.error(f"Llama.cpp WORKER STDERR (PID:{pid_str}): {line}")
                            app_instance.call_from_thread(app_instance._update_llamacpp_log,
                                                          f"[STDERR] [bold red]{line}[/]\n")
                            output_received_in_iteration = True
                    elif process.poll() is not None:  # Check again
                        break
                except Exception as e_stderr:
                    logger.error(f"Llama.cpp WORKER (PID:{pid_str}): Exception reading stderr: {e_stderr}")
                    break

            if not output_received_in_iteration and process.poll() is None:
                time.sleep(0.1)  # Small sleep if no output and process is alive, to prevent tight loop

        # Cleanup after loop (process has terminated or error occurred)
        if process.stdout:
            try:
                process.stdout.close()
            except Exception as e_close_stdout:
                logger.debug(f"Exception closing stdout: {e_close_stdout}")
        if process.stderr:
            try:
                process.stderr.close()
            except Exception as e_close_stderr:
                logger.debug(f"Exception closing stderr: {e_close_stderr}")

        # Final wait, though poll() should have indicated termination
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Llama.cpp WORKER (PID:{pid_str}): Timeout on final wait, process might be stuck.")

        exit_code = process.returncode if process.returncode is not None else -1  # Default if somehow None
        logger.info(f"Llama.cpp WORKER (PID:{pid_str}): Subprocess finally exited with code: {exit_code}")

        if exit_code != 0:
            final_status_message = f"Llama.cpp server (PID:{pid_str}) exited with non-zero code: {exit_code}."
        else:
            final_status_message = f"Llama.cpp server (PID:{pid_str}) exited successfully (code: {exit_code})."

        app_instance.call_from_thread(app_instance._update_llamacpp_log, f"{final_status_message}\n")
        return final_status_message

    except FileNotFoundError:
        msg = f"ERROR: Llama.cpp executable not found: {command[0]}"
        logger.error(msg)
        app_instance.call_from_thread(app_instance._update_llamacpp_log, f"[bold red]{msg}[/]\n")
        raise
    except Exception as err:
        msg = f"CRITICAL ERROR in Llama.cpp worker (persistent stream): {err} (Command: {quoted_command})"
        logger.error(msg, exc_info=True)
        app_instance.call_from_thread(app_instance._update_llamacpp_log, f"[bold red]{msg}[/]\n")
        raise
    finally:
        logger.info(f"Llama.cpp WORKER (persistent stream): Worker function for command '{quoted_command}' finishing.")
        # Clear the process from the app instance when the worker finishes
        app_instance.call_from_thread(_set_llamacpp_process_on_app, app_instance, None)
        if process and process.poll() is None:
            logger.warning(f"Llama.cpp WORKER (PID:{pid_str}): Process still running in finally. Terminating.")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.error(f"Llama.cpp WORKER (PID:{pid_str}): Process kill failed after terminate timeout.")
                process.kill()


# FIXME
def run_model_download_worker(app_instance: "TldwCli", command: List[str]):
    """Background worker that executes *command* to download a model.

    The implementation simply shells‑out to **huggingface‑cli** so that we
    do **not** add an unconditional *huggingface‑hub* dependency to
    *tldw‑cli*.  Users that prefer the Python API can adapt this easily.
    """

    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    logger.info("Model‑download worker begins: %s", " ".join(command))

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path.home()),  # run in a writable location
        )

        app_instance.call_from_thread(
            app_instance._update_model_download_log,
            f"Download started (PID: {process.pid})…\n")
        _stream_process(app_instance, "_update_model_download_log", process)
        process.wait()
        yield f"Download command exited with code: {process.returncode}\n"
    except FileNotFoundError:
        msg = "ERROR: huggingface‑cli (or specified executable) not found.\n"
        logger.error(msg.rstrip())
        app_instance.call_from_thread(app_instance._update_model_download_log, msg)
        yield msg
    except Exception as err:  # pragma: no cover
        msg = f"ERROR in model‑download worker: {err}\n"
        logger.error(msg.rstrip(), exc_info=True)
        app_instance.call_from_thread(app_instance._update_model_download_log, msg)
        yield msg


###############################################################################
# ─── shared output streaming coroutine ───────────────────────────────────────
###############################################################################


async def stream_worker_output_to_log(app: "TldwCli", worker: Worker, log_widget_id: str):
    """Forward *worker*'s yielded final messages to *log_widget_id*."""

    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))

    # FIXME
    try:
        log_widget = app.query_one(log_widget_id, RichLog)
        wrote_final = False
        if worker.state == WorkerState.SUCCESS:
            async for line in worker.output:
                log_widget.write(line.rstrip())
                wrote_final = True
            if not wrote_final:
                log_widget.write(
                    f"--- Worker {worker.name} finished successfully (no final message) ---"
                )
        elif worker.state == WorkerState.ERROR:
            async for line in worker.output:
                log_widget.write(line.rstrip())
                wrote_final = True
            if not wrote_final:
                log_widget.write(
                    f"--- Worker {worker.name} failed (no final message) ---"
                )
                if worker.result is not None:
                    log_widget.write(f"Error details: {worker.result}")
    except QueryError:
        logger.error("RichLog %s not found to attach worker %s", log_widget_id, worker.name)


###############################################################################
# ─── Llamafile – start handler ───────────────────────────────────────────────
###############################################################################


async def handle_start_llamafile_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start Llamafile server.")

    try:
        exec_path_input = app.query_one("#llamafile-exec-path", Input)
        model_path_input = app.query_one("#llamafile-model-path", Input)
        host_input = app.query_one("#llamafile-host", Input)
        port_input = app.query_one("#llamafile-port", Input)
        additional_args_input = app.query_one("#llamafile-additional-args", TextArea)
        log_output_widget = app.query_one("#llamafile-log-output", RichLog)

        exec_path = exec_path_input.value.strip()
        model_path = model_path_input.value.strip()
        host = host_input.value.strip() or "127.0.0.1"
        port = port_input.value.strip() or "8000"
        additional_args_str = additional_args_input.text.strip()  # .text for TextArea

        if not exec_path:
            app.notify("Llamafile executable path is required.", severity="error")
            exec_path_input.focus()
            return
        if not Path(exec_path).is_file():
            app.notify(f"Llamafile executable not found at: {exec_path}", severity="error")
            exec_path_input.focus()
            return

        if not model_path:
            app.notify("Model path is required.", severity="error")
            model_path_input.focus()
            return
        if not Path(model_path).is_file():
            app.notify(f"Model file not found at: {model_path}", severity="error")
            model_path_input.focus()
            return

        command = [
            exec_path,
            "-m",  # Llamafile typically uses -m for model
            model_path,
            "--host",
            host,
            "--port",
            port,
        ]
        if additional_args_str:
            command.extend(shlex.split(additional_args_str))

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(shlex.quote(c) for c in command)}\n")

        worker_callable = functools.partial(run_llamafile_server_worker, app, command)

        logger.debug(f"Preparing to call app.run_worker for Llamafile with partial: {worker_callable}")

        app.run_worker(
            worker_callable,
            group="llamafile_server",
            description="Running Llamafile server process",
            exclusive=True,  # Typically one server instance
            thread=True
            # NO 'args' or 'done' parameters
        )
        app.notify("Llamafile server starting…")
    except Exception as err:
        # Corrected the logger call to pass exc_info=True
        logger.error(f"Error preparing to start Llamafile server: {err}", exc_info=True)
        app.notify("Error setting up Llamafile server start.", severity="error")


async def handle_stop_llamafile_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to stop Llamafile server.")

    log_output_widget: Optional[RichLog] = None
    try:
        log_output_widget = app.query_one("#llamafile-log-output", RichLog)
    except QueryError:
        logger.error("Failed to find #llamafile-log-output widget for stop server messages.")
        # Attempt to notify even if log widget is missing
        app.notify("Log output widget for Llamafile not found.", severity="error")
        return  # Can't proceed meaningfully without log output for user feedback

    process_to_stop = app.llamafile_server_process  # Assumes app.llamafile_server_process exists

    if process_to_stop and process_to_stop.poll() is None:
        pid = process_to_stop.pid
        logger.info(f"Attempting to stop Llamafile server process (PID: {pid}).")
        if log_output_widget: log_output_widget.write(f"Stopping Llamafile server (PID: {pid})...\n")

        process_to_stop.terminate()
        try:
            process_to_stop.wait(timeout=10)
            logger.info(f"Llamafile server process (PID: {pid}) terminated gracefully.")
            if log_output_widget: log_output_widget.write(f"Llamafile server (PID: {pid}) stopped.\n")
            app.notify("Llamafile server stopped.")
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout waiting for Llamafile server (PID: {pid}) to terminate. Killing.")
            if log_output_widget: log_output_widget.write(
                f"Llamafile server (PID: {pid}) did not stop gracefully, killing...\n")
            process_to_stop.kill()
            try:
                process_to_stop.wait(timeout=5)
                logger.info(f"Llamafile server process (PID: {pid}) killed.")
                if log_output_widget: log_output_widget.write(f"Llamafile server (PID: {pid}) killed.\n")
            except Exception as e_kill_wait:
                logger.error(f"Error waiting for Llamafile server (PID: {pid}) to die after kill: {e_kill_wait}")
                if log_output_widget: log_output_widget.write(
                    f"Error ensuring Llamafile server (PID: {pid}) was killed: {e_kill_wait}\n")
            app.notify("Llamafile server killed after timeout.", severity="warning")
        except Exception as e_term:
            logger.error(f"Error during Llamafile server termination (PID: {pid}): {e_term}", exc_info=True)
            if log_output_widget: log_output_widget.write(f"Error stopping Llamafile server (PID: {pid}): {e_term}\n")
            app.notify(f"Error stopping Llamafile server: {e_term}", severity="error")
        finally:
            if app.llamafile_server_process is process_to_stop:
                app.llamafile_server_process = None
            logger.info(f"Cleared Llamafile server process reference (PID: {pid}) after stop attempt.")
            try:
                app.query_one("#llamafile-start-server-button", Button).disabled = False
                app.query_one("#llamafile-stop-server-button", Button).disabled = True
            except QueryError:
                logger.warning("Could not find Llamafile server buttons to update after stop action.")
    else:
        logger.info("Llamafile server is not running or process attribute is missing/stale.")
        if log_output_widget: log_output_widget.write("Llamafile server is not currently running.\n")
        app.notify("Llamafile server is not running.", severity="warning")
        if app.llamafile_server_process is not None:
            app.llamafile_server_process = None
        try:
            app.query_one("#llamafile-start-server-button", Button).disabled = False
            app.query_one("#llamafile-stop-server-button", Button).disabled = True
        except QueryError:
            logger.warning("Could not find Llamafile server buttons to update (server not running).")


###############################################################################
# ─── Llama.cpp UI helpers ────────────────────────────────────────────────────
###############################################################################


async def handle_llamacpp_browse_exec_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Llama.cpp browse executable button pressed.")

    exec_filters = Filters(("Executables", lambda p: p.is_file()))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llama.cpp executable (e.g. main, server.py)",
            filters=exec_filters,
            context="llm_models"
        ),
        callback=_make_path_update_callback(app, "llamacpp-exec-path")
    )


async def handle_llamacpp_browse_model_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    gguf_filters = Filters(
        ("GGUF Models (*.gguf)", lambda p: p.suffix.lower() == ".gguf"),
        ("All files (*.*)", lambda p: True))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llama.cpp Model (.gguf)",
            filters=gguf_filters,
            context="llm_models"
        ),
        callback=_make_path_update_callback(app, "llamacpp-model-path")
    )


async def handle_start_llamacpp_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start Llama.cpp server.")

    try:
        exec_path_input = app.query_one("#llamacpp-exec-path", Input)
        model_path_input = app.query_one("#llamacpp-model-path", Input)
        host_input = app.query_one("#llamacpp-host", Input)
        port_input = app.query_one("#llamacpp-port", Input)
        additional_args_input = app.query_one("#llamacpp-additional-args", Input)
        log_output_widget = app.query_one("#llamacpp-log-output", RichLog)

        exec_path = exec_path_input.value.strip()
        model_path = model_path_input.value.strip()
        host = host_input.value.strip() or "127.0.0.1"
        port = port_input.value.strip() or "8001"
        additional_args_str = additional_args_input.value.strip()

        if not exec_path:
            app.notify("Executable path is required.", severity="error")
            exec_path_input.focus()
            return
        if not Path(exec_path).is_file():
            app.notify(f"Executable not found at: {exec_path}", severity="error")
            exec_path_input.focus()
            return

        if not model_path:
            app.notify("Model path is required.", severity="error")
            model_path_input.focus()
            return
        if not Path(model_path).is_file():
            app.notify(f"Model file not found at: {model_path}", severity="error")
            model_path_input.focus()
            return

        command: List[str] = [
            exec_path,
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            port,
        ]
        if additional_args_str:
            command.extend(shlex.split(additional_args_str))

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(shlex.quote(c) for c in command)}\n")

        worker_callable = functools.partial(run_llamacpp_server_worker, app, command)

        logger.debug(f"Type of 'app' object: {type(app)}")
        logger.debug(f"Bound 'app.run_worker' method: {app.run_worker}")
        logger.debug(f"Preparing to call app.run_worker for Llama.cpp with partial: {worker_callable}")

        app.run_worker(
            worker_callable,
            group="llamacpp_server",
            description="Running Llama.cpp server process",
            exclusive=True,
            thread=True
        )

        app.notify("Llama.cpp server starting…")
    except Exception as err:
        logger.error(f"Error preparing to start Llama.cpp server: {err}", exc_info=True)
        app.notify("Error setting up Llama.cpp server start.", severity="error")


async def handle_stop_llamacpp_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Stops the Llama.cpp server process if it is running."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to stop Llama.cpp server.")

    log_output_widget = None
    try:
        log_output_widget = app.query_one("#llamacpp-log-output", RichLog)

        if hasattr(app, 'llamacpp_server_process') and \
                app.llamacpp_server_process and \
                app.llamacpp_server_process.poll() is None:
            pid = app.llamacpp_server_process.pid
            logger.info(f"Stopping Llama.cpp server (PID: {pid}).")
            log_output_widget.write(f"Stopping Llama.cpp server (PID: {pid})...\n")

            app.llamacpp_server_process.terminate()
            try:
                app.llamacpp_server_process.wait(timeout=5)  # Wait for graceful termination
                logger.info("Llama.cpp server terminated.")
                log_output_widget.write("Llama.cpp server stopped successfully.\n")
                app.notify("Llama.cpp server stopped.")
                # Update button states
                start_button = app.query_one("#llamacpp-start-server-button", Button)
                stop_button = app.query_one("#llamacpp-stop-server-button", Button)
                start_button.disabled = False
                stop_button.disabled = True
            except subprocess.TimeoutExpired:
                logger.warning(f"Llama.cpp server (PID: {pid}) did not terminate gracefully. Killing.")
                log_output_widget.write(f"Llama.cpp server (PID: {pid}) did not stop in time, killing...\n")
                app.llamacpp_server_process.kill()  # Force kill
                app.llamacpp_server_process.wait()  # Wait for kill to complete
                logger.info("Llama.cpp server killed.")
                log_output_widget.write("Llama.cpp server killed.\n")
                app.notify("Llama.cpp server killed.", severity="warning")
                # Update button states even after kill
                start_button = app.query_one("#llamacpp-start-server-button", Button)
                stop_button = app.query_one("#llamacpp-stop-server-button", Button)
                start_button.disabled = False
                stop_button.disabled = True
            except Exception as e:  # Catch other errors during wait/terminate like process already exited
                logger.error(f"Error during Llama.cpp server termination (PID: {pid}): {e}", exc_info=True)
                log_output_widget.write(f"Error stopping Llama.cpp server (PID: {pid}): {e}\n")
                app.notify(f"Error stopping Llama.cpp server: {e}", severity="error")
                # Attempt to reset buttons on error
                try:
                    start_button = app.query_one("#llamacpp-start-server-button", Button)
                    stop_button = app.query_one("#llamacpp-stop-server-button", Button)
                    start_button.disabled = False
                    stop_button.disabled = True
                except QueryError as q_err:  # pragma: no cover
                    logger.error(f"Failed to query buttons to reset state after termination error: {q_err}",
                                 exc_info=True)
        else:
            logger.info("Llama.cpp server is not running or process attribute is missing.")
            log_output_widget.write("Llama.cpp server is not running or was already stopped.\n")
            app.notify("Llama.cpp server is not running.", severity="warning")
            # Update button states
            start_button = app.query_one("#llamacpp-start-server-button", Button)
            stop_button = app.query_one("#llamacpp-stop-server-button", Button)
            start_button.disabled = False
            stop_button.disabled = True

    except QueryError as e:  # pragma: no cover
        logger.error(f"Could not find #llamacpp-log-output or a button: {e}", exc_info=True)
        app.notify("Error: UI widget not found during stop operation.", severity="error")
        # Attempt to reset buttons even if log widget is missing
        try:
            start_button = app.query_one("#llamacpp-start-server-button", Button)
            stop_button = app.query_one("#llamacpp-stop-server-button", Button)
            start_button.disabled = False
            stop_button.disabled = True
        except QueryError as q_err:  # pragma: no cover
            logger.error(f"Failed to query buttons to reset state after QueryError: {q_err}", exc_info=True)
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"Error stopping Llama.cpp server: {e}", exc_info=True)
        if log_output_widget:  # Check if log_widget was found before error
            log_output_widget.write(f"An unexpected error occurred while stopping the server: {e}\n")
        app.notify(f"An unexpected error occurred: {e}", severity="error")
        # Attempt to reset buttons on generic error
        try:
            start_button = app.query_one("#llamacpp-start-server-button", Button)
            stop_button = app.query_one("#llamacpp-stop-server-button", Button)
            start_button.disabled = False
            stop_button.disabled = True
        except QueryError as q_err:  # pragma: no cover
            logger.error(f"Failed to query buttons to reset state after generic error: {q_err}", exc_info=True)
    finally:
        if hasattr(app, 'llamacpp_server_process'):
            app.llamacpp_server_process = None
        logger.debug("Set app.llamacpp_server_process to None.")
        # Final attempt to ensure buttons are in a consistent state
        try:
            start_button = app.query_one("#llamacpp-start-server-button", Button)
            stop_button = app.query_one("#llamacpp-stop-server-button", Button)
            # If server process is None here, it means it's stopped or was never started.
            is_running = hasattr(app, 'llamacpp_server_process') and \
                         app.llamacpp_server_process and \
                         app.llamacpp_server_process.poll() is None

            start_button.disabled = is_running
            stop_button.disabled = not is_running
        except QueryError as q_err:  # pragma: no cover
            logger.error(f"Failed to query buttons in finally block: {q_err}", exc_info=True)


###############################################################################
# ─── Model download UI helpers ──────────────────────────────────────────────
###############################################################################


async def handle_browse_models_dir_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Open a directory picker so the user can choose the *models* directory."""
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Models Directory (select any file inside)",
            context="llm_models"
        ),
        callback=_make_path_update_callback(app, "models-dir-path", is_directory=True)
    )


async def handle_start_model_download_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Validate inputs and launch *run_model_download_worker*."""

    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to download a model from Hugging Face.")

    try:
        models_dir_input = app.query_one("#models-dir-path", Input)
        url_or_repo_input = app.query_one("#model-url-or-repo", Input)
        revision_input = app.query_one("#model-revision", Input)
        log_output_widget = app.query_one("#model-download-log-output", RichLog)

        models_dir = models_dir_input.value.strip()
        url_or_repo = url_or_repo_input.value.strip()
        revision = revision_input.value.strip()

        if not models_dir:
            app.notify("Models directory is required.", severity="error")
            models_dir_input.focus()
            return
        if not Path(models_dir).is_dir():
            app.notify(f"Models directory not found or is not a directory: {models_dir}", severity="error")
            models_dir_input.focus()
            return
        if not url_or_repo:
            app.notify("Model URL or repo name is required.", severity="error")
            url_or_repo_input.focus()
            return

        command = [
            "huggingface-cli",
            "download",
            url_or_repo,
            "--local-dir",
            str(Path(models_dir) / url_or_repo.split("/")[-1]),  # Download into a subfolder
            "--local-dir-use-symlinks",
            "False",  # Explicitly False for actual copy
        ]
        if revision:
            command.extend(["--revision", revision])

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(command)}\n")

        app.run_worker(
            run_model_download_worker,
            args=[app, command],
            group="model_download",
            description="Downloading model via huggingface-cli",
            exclusive=False,  # Can run multiple downloads
            thread=True,  # <--- ADDED THIS
            done=lambda w: app.call_from_thread(
                stream_worker_output_to_log, app, w, "#model-download-log-output"
            ))
        app.notify("Model download started…")
    except Exception as err:  # pragma: no cover
        logger.error("Error preparing model download: %s", err, exc_info=True)
        app.notify("Error setting up model download.", severity="error")

async def populate_llm_help_texts(app: 'TldwCli') -> None:
    """Populates the RichLog widgets with help text for LLM arguments."""
    app.loguru_logger.info("Populating LLM argument help texts...")
    
    # Check if LLM Management Window exists first
    try:
        llm_window = app.query_one("#llm_management-window")
    except QueryError:
        app.loguru_logger.debug("LLM Management Window not found, skipping help text population.")
        return
    
    # Wait a bit for widgets to be fully mounted
    import asyncio
    await asyncio.sleep(0.1)
    
    try:
        # Llama.cpp - only populate if the view is active
        if app.llm_active_view == "llm-view-llama-cpp":
            llamacpp_help_widget = llm_window.query_one("#llamacpp-args-help-display", RichLog)
            llamacpp_help_widget.clear()  # Clear any old content
            llamacpp_help_widget.write(LLAMA_CPP_SERVER_ARGS_HELP_TEXT)
            app.loguru_logger.debug("Populated Llama.cpp args help.")
    except QueryError:
        app.loguru_logger.debug("Failed to find #llamacpp-args-help-display widget.")
    except Exception as e:
        app.loguru_logger.error(f"Error populating Llama.cpp help: {e}", exc_info=True)
    try:
        # Llamafile - only populate if the view is active
        if app.llm_active_view == "llm-view-llamafile":
            llamafile_help_widget = llm_window.query_one("#llamafile-args-help-display", RichLog)
            llamafile_help_widget.clear()  # Clear existing content
            llamafile_help_widget.write(LLAMAFILE_SERVER_ARGS_HELP_TEXT)  # Write new content
            app.loguru_logger.debug("Populated Llamafile args help.")
    except QueryError:
        app.loguru_logger.debug("Failed to find #llamafile-args-help-display widget.")
    except Exception as e:
        app.loguru_logger.error(f"Error populating Llamafile help: {e}", exc_info=True)
    try:
        # MLX-LM - only populate if the view is active
        if app.llm_active_view == "llm-view-mlx-lm":
            mlx_help_widget = llm_window.query_one("#mlx-args-help-display", RichLog)
            mlx_help_widget.clear()  # Clear existing content
            mlx_help_widget.write(MLX_LM_SERVER_ARGS_HELP_TEXT)  # Write new content
        app.loguru_logger.debug("Populated MLX-LM args help.")
    except QueryError:
        app.loguru_logger.debug("Failed to find #mlx-args-help-display widget.")
    except Exception as e:
        app.loguru_logger.error(f"Error populating MLX-LM help: {e}", exc_info=True)

# --- Button Handler Map ---
LLM_MANAGEMENT_BUTTON_HANDLERS = {
    # Llamafile
    "llamafile-browse-exec-button": handle_llamafile_browse_exec_button_pressed,
    "llamafile-browse-model-button": handle_llamafile_browse_model_button_pressed,
    "llamafile-start-server-button": handle_start_llamafile_server_button_pressed,
    "llamafile-stop-server-button": handle_stop_llamafile_server_button_pressed,
    # Llama.cpp
    "llamacpp-browse-exec-button": handle_llamacpp_browse_exec_button_pressed,
    "llamacpp-browse-model-button": handle_llamacpp_browse_model_button_pressed,
    "llamacpp-start-server-button": handle_start_llamacpp_server_button_pressed,
    "llamacpp-stop-server-button": handle_stop_llamacpp_server_button_pressed,
    # Model Download (Legacy)
    "browse-models-dir-button": handle_browse_models_dir_button_pressed,
    "start-model-download-button": handle_start_model_download_button_pressed,
    # Ollama handlers will be added in app.py to avoid circular imports
}

#
# End of llm_management_events.py
########################################################################################################################
