"""llm_management_events_ollama.py

A collection of helper callbacks, worker functions and event‑handler
coroutines specifically for the **Ollama** back‑end in the
**LLM Management** tab of *tldw‑cli*.

This module isolates Ollama-specific logic from the main llm_management_events.py.
"""

from __future__ import annotations
import asyncio
import functools
import json

from loguru import logger as _loguru_fallback_logger
from pathlib import Path
from rich.text import Text
import subprocess
from typing import Any, TYPE_CHECKING

from textual.css.query import QueryError
from textual.widgets import Button, Input, RichLog

from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import (
    _make_path_update_callback,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
    release_server_claim,
    reserve_server_launch,
    run_server_subprocess,
    stop_server_process,
)
from tldw_chatbook.Local_Inference.ollama_model_mgmt import (
    ollama_list_local_models,
    ollama_model_info,
    ollama_delete_model,
    ollama_copy_model,
    ollama_create_model,
    ollama_push_model,
    ollama_pull_model,
    ollama_list_running_models,
    ollama_generate_embeddings,
)
from tldw_chatbook.Utils.log_sanitizer import sanitize_dict, sanitize_string
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

__all__ = [
    # ─── Ollama ───────────────────────────────────────────────────────────────
    "handle_ollama_browse_exec_button_pressed",
    "handle_ollama_start_service_button_pressed",
    "handle_ollama_stop_service_button_pressed",
    "handle_ollama_list_models_button_pressed",
    "handle_ollama_show_model_button_pressed",
    "handle_ollama_delete_model_button_pressed",
    "handle_ollama_copy_model_button_pressed",
    "handle_ollama_pull_model_button_pressed",
    "handle_ollama_create_model_button_pressed",
    "handle_ollama_browse_modelfile_button_pressed",
    "handle_ollama_push_model_button_pressed",
    "handle_ollama_embeddings_button_pressed",
    "handle_ollama_ps_button_pressed",
    "OLLAMA_BUTTON_HANDLERS",
]

MAX_OLLAMA_SUCCESS_OUTPUT_CHARS = 32_768

###############################################################################
# ─── Ollama UI helpers ──────────────────────────────────────────────────────
###############################################################################


def _format_ollama_success_payload(data: dict[str, Any]) -> str:
    """Render a bounded successful API result with credential fields redacted."""

    rendered = json.dumps(
        sanitize_dict(data),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if len(rendered) <= MAX_OLLAMA_SUCCESS_OUTPUT_CHARS:
        return rendered
    suffix = "\n… output truncated"
    return rendered[: MAX_OLLAMA_SUCCESS_OUTPUT_CHARS - len(suffix)] + suffix


def _sync_ollama_process_controls(
    window: "LLMManagementWindow", app: "TldwCli"
) -> None:
    """Derive controls from app-owned lifecycle state."""

    window._sync_process_controls("ollama")


def _safe_ollama_model_names(models: object) -> list[str] | None:
    """Return bounded sanitized model names from a successful API result."""

    if not isinstance(models, list):
        return None
    safe_names: list[str] = []
    for item in models[:500]:
        if not isinstance(item, dict):
            return None
        name = item.get("name") or item.get("model")
        if not isinstance(name, str):
            return None
        safe_name = sanitize_string(name).replace("\n", " ").strip()[:256]
        if not safe_name:
            return None
        safe_names.append(safe_name)
    return safe_names


def run_ollama_service_worker(
    app: "TldwCli",
    command: list[str],
    claim: ServerLaunchClaim,
) -> str:
    """Run Ollama without persisting raw subprocess output."""

    return run_server_subprocess(
        app,
        "ollama",
        command,
        claim,
        subprocess,
    )


async def handle_ollama_browse_exec_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Browse' button press for Ollama executable path."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama browse executable button pressed.")

    exec_filters = Filters(
        ("Executables", lambda p: p.is_file() and p.name.lower() == "ollama"),
        ("All files (*.*)", lambda p: True),
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Ollama Executable",
            filters=exec_filters,
            context="ollama_models",
        ),
        callback=_make_path_update_callback(window, app, "ollama-exec-path"),
    )


async def handle_ollama_start_service_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Start Ollama Service' button press."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Start Service' button pressed.")

    try:
        exec_path_input = window.query_one("#ollama-exec-path", Input)
        log_output_widget = window.query_one("#ollama-log-output", RichLog)

        exec_path = exec_path_input.value.strip()

        # Check if we have a valid executable path
        if not exec_path:
            # Try to find ollama in PATH
            import shutil

            exec_path = shutil.which("ollama")
            if exec_path:
                exec_path_input.value = exec_path
                logger.info("Found Ollama executable in PATH.")
            else:
                app.notify(
                    "Ollama executable path is required. Please browse for the ollama executable or ensure it's in PATH.",
                    severity="error",
                )
                exec_path_input.focus()
                return

        # Verify the executable exists
        if not Path(exec_path).is_file():
            app.notify("Ollama executable was not found.", severity="error")
            exec_path_input.focus()
            return

        claim = reserve_server_launch(app, "ollama")
        if claim is None:
            _sync_ollama_process_controls(window, app)
            app.notify(
                "Ollama service is already starting or running.",
                severity="warning",
            )
            return
        log_output_widget.clear()
        log_output_widget.write("Starting Ollama service.")

        # Start the Ollama service
        cmd = [exec_path, "serve"]

        worker_callable = functools.partial(
            run_ollama_service_worker,
            app,
            cmd,
            claim,
        )
        _sync_ollama_process_controls(window, app)
        app.run_worker(
            worker_callable,
            thread=True,
            name="ollama_serve_process",
            group="ollama_serve",
            exclusive=True,
            description="Running Ollama service",
        )

        app.notify("Ollama service starting...", severity="information")

    except QueryError:
        if "claim" in locals():
            release_server_claim(app, "ollama", claim)
        _sync_ollama_process_controls(window, app)
        logger.error("Ollama start failed (category=QueryError).")
        app.notify(
            "Error accessing UI elements for starting Ollama service.", severity="error"
        )
    except Exception as e:
        if "claim" in locals():
            release_server_claim(app, "ollama", claim)
        _sync_ollama_process_controls(window, app)
        logger.error(
            "Ollama start failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while starting Ollama service.",
            severity="error",
        )


async def handle_ollama_stop_service_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Stop Ollama Service' button press."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Stop Service' button pressed.")

    try:
        await stop_server_process(app, "ollama", "Ollama service")
    except QueryError:
        logger.error("Ollama stop failed (category=QueryError).")
        app.notify(
            "Error accessing UI elements for stopping Ollama service.", severity="error"
        )
    except Exception as e:
        logger.error(
            "Ollama stop failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while stopping Ollama service.",
            severity="error",
        )


async def handle_ollama_list_models_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'List Models' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'List Models' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        log_output_widget = window.query_one("#ollama-combined-output", RichLog)

        base_url = base_url_input.value.strip()
        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return

        log_output_widget.clear()
        log_output_widget.write("Listing Ollama models.")
        generation = window._begin_async_presentation("ollama-combined-output")
        del base_url_input, log_output_widget

        data, error = await asyncio.to_thread(
            ollama_list_local_models, base_url=base_url
        )
        if not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        log_output_widget = window.query_one("#ollama-combined-output", RichLog)

        if error:
            log_output_widget.write("Ollama model listing failed.")
            app.notify("Error listing Ollama models.", severity="error")
        elif data and data.get("models"):
            safe_names = _safe_ollama_model_names(data["models"])
            if safe_names is None:
                log_output_widget.write(
                    f"Found {len(data['models'])} models; names were withheld."
                )
            else:
                log_output_widget.write("\n".join(safe_names))
            app.notify(f"Successfully listed {len(data['models'])} Ollama models.")
        else:
            log_output_widget.write("No models found or unexpected response.")
            app.notify(
                "No Ollama models found or unexpected response.", severity="warning"
            )
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        logger.error("Ollama model listing failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for listing models.", severity="error"
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        logger.error(
            "Ollama model listing failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while listing Ollama models.",
            severity="error",
        )


async def handle_ollama_show_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Show Model Info' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Show Model Info' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        model_name_input = window.query_one("#ollama-show-model-name", Input)
        log_output_widget = window.query_one("#ollama-combined-output", RichLog)
        # general_log_widget = window.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required to show info.", severity="error")
            model_name_input.focus()
            return

        log_output_widget.clear()
        generation = window._begin_async_presentation("ollama-combined-output")
        del base_url_input, model_name_input, log_output_widget
        data, error = await asyncio.to_thread(
            ollama_model_info, base_url=base_url, model_name=model_name
        )
        if not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        log_output_widget = window.query_one("#ollama-combined-output", RichLog)
        if error:
            log_output_widget.write("Ollama model information request failed.")
            app.notify("Error fetching model information.", severity="error")
        elif data:
            log_output_widget.write(Text(_format_ollama_success_payload(data)))
            app.notify("Successfully fetched model information.")
        else:
            log_output_widget.write("No model information returned.")
            app.notify("No model information returned.", severity="warning")
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        logger.error("Ollama model information failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for showing model info.",
            severity="error",
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        logger.error(
            "Ollama model information failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while showing model info.", severity="error"
        )


async def handle_ollama_delete_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Delete Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Delete Model' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        model_name_input = window.query_one("#ollama-delete-model-name", Input)
        log_output_widget = window.query_one(
            "#ollama-log-output", RichLog
        )  # General log for delete

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required to delete.", severity="error")
            model_name_input.focus()
            return

        # Show confirmation dialog
        from ...Widgets.delete_confirmation_dialog import create_delete_confirmation

        dialog = create_delete_confirmation(
            item_type="Model",
            item_name=model_name,
            additional_warning="This will uninstall the model from your system.",
            permanent=True,
        )

        generation = window._begin_async_presentation("ollama-log-output")
        del base_url_input, model_name_input, log_output_widget
        confirmed = await app.push_screen_wait(dialog)
        if not window._owns_async_presentation("ollama-log-output", generation):
            return
        log_output_widget = window.query_one("#ollama-log-output", RichLog)
        if not confirmed:
            logger.info("Ollama model deletion cancelled.")
            log_output_widget.write("Model deletion cancelled.")
            return

        log_output_widget.write("Deleting selected model.")
        del log_output_widget

        def stream_to_log(message: str):
            del message

        data, error = await asyncio.to_thread(
            ollama_delete_model,
            base_url=base_url,
            model_name=model_name,
            stream_log_callback=stream_to_log,
        )
        if not window._owns_async_presentation("ollama-log-output", generation):
            return
        log_output_widget = window.query_one("#ollama-log-output", RichLog)
        if error:
            log_output_widget.write("[bold red]Model deletion failed.[/bold red]")
            app.notify("Error deleting model.", severity="error")
        else:
            # Stream callback should have provided detailed progress.
            # 'data' might contain final status if any, or might be None if stream handled all.
            if data and data.get("status") == "success":
                log_output_widget.write("Model deleted successfully.")
                app.notify("Model deleted.")
            elif not data and not error:  # Common for stream-focused ops
                log_output_widget.write("Model deletion process completed.")
                app.notify("Model deletion process completed.")
            else:  # Some other response
                log_output_widget.write("Model deletion process finished.")
                app.notify("Model deletion process finished.")
        # Optionally, refresh the model list:
        # app.call_after_refresh(lambda: app.run_action("ollama_list_models_button_pressed"))
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error("Ollama model deletion failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for deleting model.", severity="error"
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error(
            "Ollama model deletion failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while deleting model.", severity="error"
        )


async def handle_ollama_copy_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Copy Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Copy Model' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        source_model_input = window.query_one("#ollama-copy-source-model", Input)
        dest_model_input = window.query_one("#ollama-copy-destination-model", Input)
        log_output_widget = window.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        source_model = source_model_input.value.strip()
        dest_model = dest_model_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not source_model:
            app.notify("Source model name is required for copy.", severity="error")
            source_model_input.focus()
            return
        if not dest_model:
            app.notify("Destination model name is required for copy.", severity="error")
            dest_model_input.focus()
            return

        log_output_widget.write("Copying selected model.")
        generation = window._begin_async_presentation("ollama-log-output")
        del (
            base_url_input,
            source_model_input,
            dest_model_input,
            log_output_widget,
        )

        data, error = await asyncio.to_thread(
            ollama_copy_model,
            base_url=base_url,
            source=source_model,
            destination=dest_model,
        )
        if not window._owns_async_presentation("ollama-log-output", generation):
            return
        log_output_widget = window.query_one("#ollama-log-output", RichLog)
        if error:
            log_output_widget.write("[bold red]Model copy failed.[/bold red]")
            app.notify("Error copying model.", severity="error")
        else:
            # Ollama copy API returns 200 OK on success with no body.
            # The ollama_model_mgmt.py wrapper might return a success message in 'data'.
            if data and data.get("status") == "success":
                log_output_widget.write("Model copied successfully.")
                app.notify("Model copied.")
            else:  # Should be success if no error
                log_output_widget.write("Model copy initiated.")
                app.notify("Model copy initiated.")
        # Optionally, refresh model list
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error("Ollama model copy failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for copying model.", severity="error"
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error(
            "Ollama model copy failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while copying model.", severity="error"
        )


async def handle_ollama_pull_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Pull Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Pull Model' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        model_name_input = window.query_one("#ollama-pull-model-name", Input)
        log_output_widget = window.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required to pull.", severity="error")
            model_name_input.focus()
            return

        log_output_widget.write("Pulling selected model.")
        generation = window._begin_async_presentation("ollama-log-output")
        del base_url_input, model_name_input, log_output_widget

        def stream_to_log(message: str):
            del message

        # Consider adding 'insecure' parameter if UI supports it, default False
        data, error = await asyncio.to_thread(
            ollama_pull_model,
            base_url=base_url,
            model_name=model_name,
            stream_log_callback=stream_to_log,
        )
        if not window._owns_async_presentation("ollama-log-output", generation):
            return
        log_output_widget = window.query_one("#ollama-log-output", RichLog)
        if error:
            log_output_widget.write("[bold red]Model pull failed.[/bold red]")
            app.notify("Error pulling model.", severity="error")
        else:
            log_output_widget.write("Model pull process finished.")
            app.notify("Model pull completed.")
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error("Ollama model pull failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for pulling model.", severity="error"
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error(
            "Ollama model pull failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while pulling model.", severity="error"
        )


async def handle_ollama_browse_modelfile_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Browse for Modelfile' button press for Ollama create model."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Browse for Modelfile' button pressed.")

    # No specific filters for "Modelfile" by extension, so allow all files or common text files.
    # Users should know what a Modelfile looks like.
    modelfile_filters = Filters(
        (
            "Modelfiles (Modelfile, *.txt)",
            lambda p: p.name.lower() == "modelfile" or p.suffix.lower() == ".txt",
        ),
        ("All files (*.*)", lambda p: True),
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.cwd()),
            title="Select Modelfile",
            filters=modelfile_filters,
            context="ollama_models",
        ),
        callback=_make_path_update_callback(
            window, app, "ollama-create-modelfile-path"
        ),
    )


async def handle_ollama_create_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Create Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Create Model' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        model_name_input = window.query_one("#ollama-create-model-name", Input)
        modelfile_path_input = window.query_one("#ollama-create-modelfile-path", Input)
        log_output_widget = window.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()
        modelfile_path = modelfile_path_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("New model name is required for creation.", severity="error")
            model_name_input.focus()
            return
        if not modelfile_path:
            app.notify("Path to Modelfile is required for creation.", severity="error")
            # modelfile_path_input.focus() # This is read-only, so focus the browse button indirectly or notify.
            app.notify(
                "Use 'Browse for Modelfile' to select a file.", severity="information"
            )
            return
        if not Path(modelfile_path).is_file():
            app.notify("Selected Modelfile was not found.", severity="error")
            # modelfile_path_input.focus()
            return

        log_output_widget.write("Creating selected model.")
        generation = window._begin_async_presentation("ollama-log-output")
        del (
            base_url_input,
            model_name_input,
            modelfile_path_input,
            log_output_widget,
        )

        def stream_to_log(message: str):
            del message

        data, error = await asyncio.to_thread(
            ollama_create_model,
            base_url=base_url,
            model_name=model_name,
            path=modelfile_path,
            stream_log_callback=stream_to_log,
        )
        if not window._owns_async_presentation("ollama-log-output", generation):
            return
        log_output_widget = window.query_one("#ollama-log-output", RichLog)
        if error:
            log_output_widget.write("[bold red]Model creation failed.[/bold red]")
            app.notify("Error creating model.", severity="error")
        else:
            log_output_widget.write("Model creation process finished.")
            app.notify("Model creation completed.")
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error("Ollama model creation failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for creating model.", severity="error"
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error(
            "Ollama model creation failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while creating model.", severity="error"
        )


async def handle_ollama_push_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Push Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Push Model' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        model_name_input = window.query_one("#ollama-push-model-name", Input)
        log_output_widget = window.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required to push.", severity="error")
            model_name_input.focus()
            return

        log_output_widget.write("Pushing selected model.")
        generation = window._begin_async_presentation("ollama-log-output")
        del base_url_input, model_name_input, log_output_widget

        def stream_to_log(message: str):
            del message

        # Consider adding 'insecure' parameter if UI supports it, default False
        data, error = await asyncio.to_thread(
            ollama_push_model,
            base_url=base_url,
            model_name=model_name,
            stream_log_callback=stream_to_log,
        )
        if not window._owns_async_presentation("ollama-log-output", generation):
            return
        log_output_widget = window.query_one("#ollama-log-output", RichLog)
        if error:
            log_output_widget.write("[bold red]Model push failed.[/bold red]")
            app.notify("Error pushing model.", severity="error")
        else:
            log_output_widget.write("Model push process finished.")
            app.notify("Model push completed.")
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error("Ollama model push failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for pushing model.", severity="error"
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-log-output",
            generation,
        ):
            return
        logger.error(
            "Ollama model push failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while pushing model.", severity="error"
        )


async def handle_ollama_embeddings_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'Generate Embeddings' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'Generate Embeddings' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        model_name_input = window.query_one("#ollama-embeddings-model-name", Input)
        prompt_input = window.query_one("#ollama-embeddings-prompt", Input)
        embeddings_output_widget = window.query_one("#ollama-combined-output", RichLog)
        # general_log_widget = window.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        model_name = model_name_input.value.strip()
        prompt = prompt_input.value.strip()

        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return
        if not model_name:
            app.notify("Model name is required for embeddings.", severity="error")
            model_name_input.focus()
            return
        if not prompt:
            app.notify("Prompt is required for embeddings.", severity="error")
            prompt_input.focus()
            return

        embeddings_output_widget.clear()
        generation = window._begin_async_presentation("ollama-combined-output")
        del (
            base_url_input,
            model_name_input,
            prompt_input,
            embeddings_output_widget,
        )

        data, error = await asyncio.to_thread(
            ollama_generate_embeddings,
            base_url=base_url,
            model_name=model_name,
            prompt=prompt,
        )
        if not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        embeddings_output_widget = window.query_one(
            "#ollama-combined-output",
            RichLog,
        )
        if error:
            embeddings_output_widget.write("Embedding generation failed.")
            app.notify("Error generating embeddings.", severity="error")
        elif data and data.get("embedding"):
            embeddings_output_widget.write(Text(_format_ollama_success_payload(data)))
            app.notify("Embeddings generated successfully.")
        else:
            embeddings_output_widget.write("No embedding returned.")
            app.notify(
                "No embeddings returned or unexpected response.", severity="warning"
            )
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        logger.error("Ollama embeddings failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for generating embeddings.",
            severity="error",
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        logger.error(
            "Ollama embeddings failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while generating embeddings.",
            severity="error",
        )


async def handle_ollama_ps_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    """Handles the 'List Running Models (ps)' button press for Ollama."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Ollama 'List Running Models (ps)' button pressed.")
    try:
        base_url_input = window.query_one("#ollama-server-url", Input)
        ps_output_widget = window.query_one("#ollama-combined-output", RichLog)
        # general_log_widget = window.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return

        ps_output_widget.clear()
        generation = window._begin_async_presentation("ollama-combined-output")
        del base_url_input, ps_output_widget
        data, error = await asyncio.to_thread(
            ollama_list_running_models, base_url=base_url
        )
        if not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        ps_output_widget = window.query_one("#ollama-combined-output", RichLog)
        if error:
            ps_output_widget.write("Running-model listing failed.")
            app.notify("Error listing running Ollama models.", severity="error")
        elif data and data.get("models"):
            safe_names = _safe_ollama_model_names(data["models"])
            if safe_names is None:
                ps_output_widget.write(
                    f"Found {len(data['models'])} running models; names were withheld."
                )
            else:
                ps_output_widget.write("\n".join(safe_names))
            app.notify(
                f"Successfully listed {len(data['models'])} running Ollama models."
            )
        else:
            ps_output_widget.write("No running models found or unexpected response.")
            app.notify(
                "No running Ollama models found or response format issue.",
                severity="warning",
            )
    except QueryError:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        logger.error("Ollama running-model listing failed (category=QueryError).")
        app.notify(
            "Error accessing Ollama UI elements for listing running models.",
            severity="error",
        )
    except Exception as e:  # pragma: no cover
        if "generation" in locals() and not window._owns_async_presentation(
            "ollama-combined-output",
            generation,
        ):
            return
        logger.error(
            "Ollama running-model listing failed (category={}).",
            type(e).__name__,
        )
        app.notify(
            "An unexpected error occurred while listing running models.",
            severity="error",
        )


# --- Button Handler Map ---
OLLAMA_BUTTON_HANDLERS = {
    "ollama-browse-exec-button": handle_ollama_browse_exec_button_pressed,
    "ollama-start-service-button": handle_ollama_start_service_button_pressed,
    "ollama-stop-service-button": handle_ollama_stop_service_button_pressed,
    "ollama-list-models-button": handle_ollama_list_models_button_pressed,
    "ollama-show-model-button": handle_ollama_show_model_button_pressed,
    "ollama-delete-model-button": handle_ollama_delete_model_button_pressed,
    "ollama-copy-model-button": handle_ollama_copy_model_button_pressed,
    "ollama-pull-model-button": handle_ollama_pull_model_button_pressed,
    "ollama-create-model-button": handle_ollama_create_model_button_pressed,
    "ollama-browse-modelfile-button": handle_ollama_browse_modelfile_button_pressed,
    "ollama-push-model-button": handle_ollama_push_model_button_pressed,
    "ollama-embeddings-button": handle_ollama_embeddings_button_pressed,
    "ollama-ps-button": handle_ollama_ps_button_pressed,
}

#
# End of llm_management_events_ollama.py
########################################################################################################################
