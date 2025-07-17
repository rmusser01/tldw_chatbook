"""llm_management_events_vllm.py

A collection of helper callbacks, worker functions and event‑handler
coroutines specifically for the **vLLM** back‑end in the
**LLM Management** tab of *tldw‑cli*.

This module isolates vLLM-specific logic from the main llm_management_events.py.
"""
# Imports
from __future__ import annotations
#
import logging
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
#
# Third-party Libraries
from textual.widgets import Input, RichLog, TextArea, Button

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
#
# Local Imports
from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import _make_path_update_callback
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters
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
    safe_pattern = re.compile(r'^[a-zA-Z0-9_.\-/\\:]+$')
    if not safe_pattern.match(python_path):
        return False
    
    # Common python executable names
    allowed_names = {'python', 'python3', 'python3.8', 'python3.9', 'python3.10', 'python3.11', 'python3.12'}
    
    # If it's just a name (no path), check against whitelist
    if '/' not in python_path and '\\' not in python_path:
        return python_path in allowed_names
    
    # For paths, validate they don't contain dangerous patterns
    dangerous_patterns = ['&&', '||', ';', '|', '>', '<', '`', '$', '(', ')']
    return not any(pattern in python_path for pattern in dangerous_patterns)

def validate_host(host: str) -> bool:
    """Validate host address to prevent command injection."""
    if not host:
        return False
    
    # IPv4 pattern
    ipv4_pattern = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    # Hostname pattern (simplified)
    hostname_pattern = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$')
    
    return (host == 'localhost' or 
            host == '127.0.0.1' or 
            host == '0.0.0.0' or
            ipv4_pattern.match(host) or 
            hostname_pattern.match(host))

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
    safe_pattern = re.compile(r'^[a-zA-Z0-9_.\-/\\:]+$')
    if not safe_pattern.match(model_path):
        return False
    
    # Reject paths with dangerous shell metacharacters
    dangerous_patterns = ['&&', '||', ';', '|', '>', '<', '`', '$', '(', ')']
    return not any(pattern in model_path for pattern in dangerous_patterns)

def validate_additional_args(args_str: str) -> bool:
    """Validate additional arguments to prevent command injection."""
    if not args_str.strip():
        return True  # Empty is fine
    
    try:
        # Use shlex to parse - this will raise ValueError for malformed input
        parsed_args = shlex.split(args_str)
        
        # Check each argument for dangerous patterns
        dangerous_patterns = ['&&', '||', ';', '|', '>', '<', '`', '$']
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


async def handle_vllm_browse_python_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Let the user pick the Python interpreter used for vLLM (venv, etc.)."""

    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Python interpreter for vLLM",
            filters=Filters(("Python executable", lambda p: p.name.startswith("python"))),
            context="vllm_models"
        ),
        callback=_make_path_update_callback(app, "vllm-python-path")
    )


async def handle_vllm_browse_model_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Model (checkpoint or GGUF) for vLLM",
            filters=Filters(("All files", lambda p: True)),
            context="vllm_models"
        ),
        callback=_make_path_update_callback(app, "vllm-model-path")
    )

###############################################################################
# ─── Worker functions
###############################################################################

# Helper to set/clear the process on the app instance from the worker thread
def _set_vllm_process_on_app(app_instance: "TldwCli", process: Optional[subprocess.Popen]):
    app_instance.vllm_server_process = process
    if process and hasattr(process, 'pid') and process.pid is not None:
        app_instance.loguru_logger.info(f"Stored vLLM process PID {process.pid} on app instance.")
    else:
        app_instance.loguru_logger.info("Cleared vLLM process from app instance (or process was None).")


async def run_vllm_server_worker(app_instance: "TldwCli", command: List[str]) -> str:
    logger = getattr(app_instance, "loguru_logger", logging.getLogger(__name__))
    quoted_command = ' '.join(shlex.quote(c) for c in command)
    logger.info(f"vLLM WORKER (persistent stream) starting with command: {quoted_command}")

    process: Optional[subprocess.Popen] = None
    final_status_message = f"vLLM WORKER: Default status for {quoted_command}"
    pid_str = "N/A"

    try:
        logger.debug("vLLM WORKER: Attempting to start subprocess...")
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
        logger.info(f"vLLM WORKER: Subprocess launched, PID: {pid_str}")

        app_instance.call_from_thread(_set_vllm_process_on_app, app_instance, process)
        app_instance.call_from_thread(app_instance._update_vllm_log,
                                      f"[PID:{pid_str}] vLLM server starting...\n")  # Assumes _update_vllm_log exists

        while True:
            output_received_in_iteration = False
            if process.poll() is not None:
                logger.info(f"vLLM WORKER (PID:{pid_str}): Process terminated. Exit code: {process.returncode}")
                break

            if process.stdout:
                try:
                    line = process.stdout.readline()
                    if line:
                        line = line.strip()
                        if line:
                            logger.info(f"vLLM WORKER STDOUT (PID:{pid_str}): {line}")
                            app_instance.call_from_thread(app_instance._update_vllm_log, f"{line}\n")
                            output_received_in_iteration = True
                    elif process.poll() is not None:
                        break
                except Exception as e_stdout:
                    logger.error(f"vLLM WORKER (PID:{pid_str}): Exception reading stdout: {e_stdout}")
                    break

            if process.stderr:
                try:
                    line = process.stderr.readline()
                    if line:
                        line = line.strip()
                        if line:
                            logger.error(f"vLLM WORKER STDERR (PID:{pid_str}): {line}")
                            app_instance.call_from_thread(app_instance._update_vllm_log,
                                                          f"[STDERR] [bold red]{line}[/]\n")
                            output_received_in_iteration = True
                    elif process.poll() is not None:
                        break
                except Exception as e_stderr:
                    logger.error(f"vLLM WORKER (PID:{pid_str}): Exception reading stderr: {e_stderr}")
                    break

            if not output_received_in_iteration and process.poll() is None:
                time.sleep(0.1)

        if process.stdout: process.stdout.close()
        if process.stderr: process.stderr.close()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"vLLM WORKER (PID:{pid_str}): Timeout on final wait.")

        exit_code = process.returncode if process.returncode is not None else -1
        logger.info(f"vLLM WORKER (PID:{pid_str}): Subprocess finally exited with code: {exit_code}")

        if exit_code != 0:
            final_status_message = f"vLLM server (PID:{pid_str}) exited with non-zero code: {exit_code}."
        else:
            final_status_message = f"vLLM server (PID:{pid_str}) exited successfully (code: {exit_code})."

        app_instance.call_from_thread(app_instance._update_vllm_log, f"{final_status_message}\n")
        return final_status_message

    except FileNotFoundError:
        msg = f"ERROR: vLLM python or entrypoint not found (command[0]): {command[0]}"
        logger.error(msg)
        app_instance.call_from_thread(app_instance._update_vllm_log, f"[bold red]{msg}[/]\n")
        raise
    except Exception as err:
        msg = f"CRITICAL ERROR in vLLM worker: {err} (Command: {quoted_command})"
        logger.error(msg, exc_info=True)
        app_instance.call_from_thread(app_instance._update_vllm_log, f"[bold red]{msg}[/]\n")
        raise
    finally:
        logger.info(f"vLLM WORKER: Worker function for command '{quoted_command}' finishing.")
        app_instance.call_from_thread(_set_vllm_process_on_app, app_instance, None)
        if process and process.poll() is None:
            logger.warning(f"vLLM WORKER (PID:{pid_str}): Process still running in finally. Terminating.")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

###############################################################################
# ─── vLLM – start/stop handlers ──────────────────────────────────────────────
###############################################################################

async def handle_start_vllm_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to start vLLM server.")

    try:
        python_path_input = app.query_one("#vllm-python-path", Input)
        model_path_input = app.query_one("#vllm-model-path", Input)
        host_input = app.query_one("#vllm-host", Input)
        port_input = app.query_one("#vllm-port", Input)
        additional_args_input = app.query_one("#vllm-additional-args", TextArea)
        log_output_widget = app.query_one("#vllm-log-output", RichLog)

        python_path = python_path_input.value.strip() or "python"
        model_path = model_path_input.value.strip() # Can be repo ID, so Path().exists() might not apply
        host = host_input.value.strip() or "127.0.0.1" # Default from snippet was 127.0.0.1
        port = port_input.value.strip() or "8000" # Default from snippet was 8000, not 8002
        additional_args_str = additional_args_input.text.strip()

        # Validate all inputs to prevent command injection
        if not validate_python_path(python_path):
            app.notify(f"Invalid Python path: {python_path}. Only safe Python executable names/paths are allowed.", severity="error")
            python_path_input.focus()
            return

        if not validate_model_path(model_path):
            app.notify(f"Invalid model path: {model_path}. Path contains unsafe characters.", severity="error")
            model_path_input.focus()
            return

        if not validate_host(host):
            app.notify(f"Invalid host: {host}. Only valid IP addresses and hostnames are allowed.", severity="error")
            host_input.focus()
            return

        if not validate_port(port):
            app.notify(f"Invalid port: {port}. Port must be a number between 1 and 65535.", severity="error")
            port_input.focus()
            return

        if not validate_additional_args(additional_args_str):
            app.notify("Invalid additional arguments. Arguments contain unsafe shell metacharacters.", severity="error")
            additional_args_input.focus()
            return

        # vLLM model can be a HuggingFace repo ID, so Path(model_path).exists() is not always appropriate.
        # We'll let vLLM handle model path validation.
        # if model_path and not Path(model_path).exists() and not "/" in model_path: # Basic check for local path
        #     app.notify(f"Local model path not found: {model_path}", severity="error")
        #     model_path_input.focus()
        #     return

        command = [
            python_path,
            "-m",
            "vllm.entrypoints.api_server", # Corrected entrypoint
            "--host",
            host,
            "--port",
            port,
        ]
        if model_path: # model_path is required for vLLM server
            command.extend(["--model", model_path])
        else:
            app.notify("Model path (or HuggingFace Repo ID) is required for vLLM.", severity="error")
            model_path_input.focus()
            return

        if additional_args_str:
            command.extend(shlex.split(additional_args_str))

        log_output_widget.clear()
        log_output_widget.write(f"Executing: {' '.join(shlex.quote(c) for c in command)}\n") # Quote for safety

        app.run_worker(
            run_vllm_server_worker(app, command),
            group="vllm_server",
            description="Running vLLM API server",
            exclusive=True,
            thread=True
        )
        app.notify("vLLM server starting…")
    except Exception as err:  # pragma: no cover
        logger.error(f"Error preparing to start vLLM server: {err}", exc_info=True)
        app.notify("Error setting up vLLM server start.", severity="error")


async def handle_stop_vllm_server_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    """Stops the vLLM server process if it's running."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.info("User requested to stop vLLM server.")

    log_output_widget = app.query_one("#vllm-log-output", RichLog)

    if hasattr(app, "vllm_server_process") and app.vllm_server_process:
        process = app.vllm_server_process
        if process.poll() is None:  # Process is running
            logger.info(f"Stopping vLLM server process (PID: {process.pid}).")
            log_output_widget.write(f"Stopping vLLM server (PID: {process.pid})...\n")
            process.terminate()  # or process.kill()
            try:
                process.wait(timeout=10)  # Wait for up to 10 seconds
                logger.info("vLLM server process terminated.")
                log_output_widget.write("vLLM server stopped.\n")
                app.notify("vLLM server stopped.")
            except subprocess.TimeoutExpired:
                logger.warning("Timeout waiting for vLLM server to terminate. Killing.")
                log_output_widget.write("vLLM server did not stop gracefully, killing...\n")
                process.kill()
                process.wait() # Ensure it's killed
                log_output_widget.write("vLLM server killed.\n")
                app.notify("vLLM server killed after timeout.", severity="warning")
            except Exception as e: # pylint: disable=broad-except
                logger.error(f"Error during vLLM server termination: {e}", exc_info=True)
                log_output_widget.write(f"Error stopping vLLM server: {e}\n")
                app.notify(f"Error stopping vLLM server: {e}", severity="error")
            finally:
                app.vllm_server_process = None
        else:
            logger.info("vLLM server process was found but is not running.")
            log_output_widget.write("vLLM server is not currently running.\n")
            app.notify("vLLM server is not running.", severity="warning")
            app.vllm_server_process = None # Clear the stale process reference
    else:
        logger.info("No vLLM server process found to stop.")
        log_output_widget.write("vLLM server is not currently running.\n")
        app.notify("vLLM server is not running.", severity="warning")

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
