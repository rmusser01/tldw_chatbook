# /tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py
#
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
# Third-party Imports
from textual.widgets import Input, RichLog, TextArea, Button

#
# Local Imports
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from .server_lifecycle import (
    ServerLaunchClaim,
    current_llm_destination,
    release_server_claim,
    reserve_server_launch,
    run_server_subprocess,
    stop_server_process,
)
#
########################################################################################################################
#
# Constants:

__all__ = [
    # Generic Helpers (Exported for other modules to use)
    "_make_path_update_callback",
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
]

# --- Generic Helpers ---


def _make_path_update_callback(
    window: "LLMManagementWindow",
    app: "TldwCli",
    input_widget_id: str,
    is_directory: bool = False,
):
    """
    Return a callback that sets an input widget's value to a picked path.
    If is_directory is True, it uses the parent directory of the selected file.
    """
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)

    async def _callback(selected_path: Optional[Path]) -> None:
        if selected_path:
            if current_llm_destination(app) is not window:
                return
            try:
                final_path = selected_path.parent if is_directory else selected_path
                input_widget = window.query_one(f"#{input_widget_id}", Input)
                input_widget.value = str(final_path)
                logger.info("Updated local path input (field={}).", input_widget_id)
            except Exception as err:
                logger.error(
                    "Local path update failed (field={}, category={}).",
                    input_widget_id,
                    type(err).__name__,
                )
                app.notify(
                    f"Error setting path for {input_widget_id}.", severity="error"
                )
        else:
            logger.info("File/Directory selection cancelled for #%s.", input_widget_id)

    return _callback


###############################################################################
# ─── Llamafile helpers ───────────────────────────────────────────────────────
###############################################################################


async def handle_llamafile_browse_exec_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Llamafile browse executable button pressed.")

    Filters(("Executables", lambda p: p.is_file()))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llamafile Executable",
            context="llm_models",
        ),
        callback=_make_path_update_callback(window, app, "llamafile-exec-path"),
    )


async def handle_llamafile_browse_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Llamafile browse model button pressed.")

    gguf_filters = Filters(
        ("GGUF Models (*.gguf)", lambda p: p.suffix.lower() == ".gguf"),
        ("All files (*.*)", lambda p: True),
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llamafile Model (.gguf)",
            filters=gguf_filters,
            context="llm_models",
        ),
        callback=_make_path_update_callback(window, app, "llamafile-model-path"),
    )


###############################################################################
# ─── Worker functions
###############################################################################


# Each run_…_worker uses the same streaming pattern – consider refactoring, but
# explicit duplication keeps each implementation easy to tweak individually.


def run_llamafile_server_worker(
    app_instance: "TldwCli",
    command: List[str],
    claim: ServerLaunchClaim,
) -> str:
    return run_server_subprocess(
        app_instance,
        "llamafile",
        command,
        claim,
        subprocess,
        cwd=Path(command[0]).parent,
    )


def run_llamacpp_server_worker(
    app_instance: "TldwCli",
    command: List[str],
    claim: ServerLaunchClaim,
) -> str | None:
    return run_server_subprocess(
        app_instance,
        "llamacpp",
        command,
        claim,
        subprocess,
    )


async def handle_start_llamafile_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("User requested to start Llamafile server.")

    try:
        exec_path_input = window.query_one("#llamafile-exec-path", Input)
        model_path_input = window.query_one("#llamafile-model-path", Input)
        host_input = window.query_one("#llamafile-host", Input)
        port_input = window.query_one("#llamafile-port", Input)
        additional_args_input = window.query_one("#llamafile-additional-args", TextArea)
        log_output_widget = window.query_one("#llamafile-log-output", RichLog)

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
            app.notify("Llamafile executable was not found.", severity="error")
            exec_path_input.focus()
            return

        if not model_path:
            app.notify("Model path is required.", severity="error")
            model_path_input.focus()
            return
        if not Path(model_path).is_file():
            app.notify("Llamafile model file was not found.", severity="error")
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

        claim = reserve_server_launch(app, "llamafile")
        if claim is None:
            window._sync_process_controls("llamafile")
            app.notify(
                "Llamafile server is already starting or running.", severity="warning"
            )
            return
        log_output_widget.clear()
        log_output_widget.write("Starting Llamafile server.\n")

        worker_callable = functools.partial(
            run_llamafile_server_worker, app, command, claim
        )

        app.run_worker(
            worker_callable,
            group="llamafile_server",
            description="Running Llamafile server process",
            exclusive=True,  # Typically one server instance
            thread=True,
            # NO 'args' or 'done' parameters
        )
        window._sync_process_controls("llamafile")
        app.notify("Llamafile server starting…")
    except Exception as err:
        if "claim" in locals():
            release_server_claim(app, "llamafile", claim)
        window._sync_process_controls("llamafile")
        logger.error(
            "Llamafile start failed (category={}).",
            type(err).__name__,
        )
        app.notify("Error setting up Llamafile server start.", severity="error")


async def handle_stop_llamafile_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    await stop_server_process(app, "llamafile", "Llamafile server")


###############################################################################
# ─── Llama.cpp UI helpers ────────────────────────────────────────────────────
###############################################################################


async def handle_llamacpp_browse_exec_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Llama.cpp browse executable button pressed.")

    exec_filters = Filters(("Executables", lambda p: p.is_file()))
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llama.cpp executable (e.g. main, server.py)",
            filters=exec_filters,
            context="llm_models",
        ),
        callback=_make_path_update_callback(window, app, "llamacpp-exec-path"),
    )


async def handle_llamacpp_browse_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    gguf_filters = Filters(
        ("GGUF Models (*.gguf)", lambda p: p.suffix.lower() == ".gguf"),
        ("All files (*.*)", lambda p: True),
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Llama.cpp Model (.gguf)",
            filters=gguf_filters,
            context="llm_models",
        ),
        callback=_make_path_update_callback(window, app, "llamacpp-model-path"),
    )


async def handle_start_llamacpp_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("User requested to start Llama.cpp server.")

    try:
        exec_path_input = window.query_one("#llamacpp-exec-path", Input)
        model_path_input = window.query_one("#llamacpp-model-path", Input)
        host_input = window.query_one("#llamacpp-host", Input)
        port_input = window.query_one("#llamacpp-port", Input)
        additional_args_input = window.query_one("#llamacpp-additional-args", Input)
        log_output_widget = window.query_one("#llamacpp-log-output", RichLog)

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
            app.notify("Llama.cpp executable was not found.", severity="error")
            exec_path_input.focus()
            return

        if not model_path:
            app.notify("Model path is required.", severity="error")
            model_path_input.focus()
            return
        if not Path(model_path).is_file():
            app.notify("Llama.cpp model file was not found.", severity="error")
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

        claim = reserve_server_launch(app, "llamacpp")
        if claim is None:
            window._sync_process_controls("llamacpp")
            app.notify(
                "Llama.cpp server is already starting or running.", severity="warning"
            )
            return
        log_output_widget.clear()
        log_output_widget.write("Starting Llama.cpp server.\n")

        worker_callable = functools.partial(
            run_llamacpp_server_worker, app, command, claim
        )

        app.run_worker(
            worker_callable,
            group="llamacpp_server",
            description="Running Llama.cpp server process",
            exclusive=True,
            thread=True,
        )

        window._sync_process_controls("llamacpp")
        app.notify("Llama.cpp server starting…")
    except Exception as err:
        if "claim" in locals():
            release_server_claim(app, "llamacpp", claim)
        window._sync_process_controls("llamacpp")
        logger.error(
            "Llama.cpp start failed (category={}).",
            type(err).__name__,
        )
        app.notify("Error setting up Llama.cpp server start.", severity="error")


async def handle_stop_llamacpp_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Stops the Llama.cpp server process if it is running."""
    await stop_server_process(app, "llamacpp", "Llama.cpp server")
    return


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
}

#
# End of llm_management_events.py
########################################################################################################################
