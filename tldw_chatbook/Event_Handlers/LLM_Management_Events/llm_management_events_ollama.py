"""llm_management_events_ollama.py

A collection of helper callbacks, worker functions and event‑handler
coroutines specifically for the **Ollama** back‑end in the
**LLM Management** tab of *tldw‑cli*.

This module isolates Ollama-specific logic from the main llm_management_events.py.
"""
from __future__ import annotations
import asyncio

import json
import logging
import shlex
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from textual.containers import Container
from textual.css.query import QueryError
from textual.widgets import Input, TextArea, RichLog

from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import (
    _make_path_update_callback,
    _stream_process,
    stream_worker_output_to_log
)
from tldw_chatbook.Local_Inference.ollama_model_mgmt import ollama_list_local_models, ollama_model_info, ollama_delete_model, \
    ollama_copy_model, ollama_create_model, ollama_push_model, ollama_pull_model, ollama_list_running_models, \
    ollama_generate_embeddings
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen
from tldw_chatbook.Third_Party.textual_fspicker import Filters

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

__all__ = [
    # ─── Ollama ───────────────────────────────────────────────────────────────
    "handle_ollama_nav_button_pressed",
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

###############################################################################
# ─── Ollama UI helpers ──────────────────────────────────────────────────────
###############################################################################


async def handle_ollama_nav_button_pressed(app: "TldwCli") -> None:
    """Handle the Ollama navigation button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama nav button pressed.")

    try:
        content_pane = app.query_one("#llm-content-pane", Container)
        view_areas = content_pane.query(".llm-view-area")

        for view in view_areas:
            if view.id:  # Only hide if it has an ID
                logger.debug(f"Hiding view #{view.id}")
                view.styles.display = "none"
            else: # pragma: no cover
                logger.warning("Found a .llm-view-area without an ID, not hiding it.")

        ollama_view = app.query_one("#llm-view-ollama", Container)
        logger.debug(f"Showing view #{ollama_view.id}")
        ollama_view.styles.display = "block"

        # Clear/initialize Ollama specific fields
        try:
            # Input fields for actions
            app.query_one("#ollama-show-model-name", Input).value = ""
            app.query_one("#ollama-delete-model-name", Input).value = ""
            app.query_one("#ollama-copy-source-model", Input).value = ""
            app.query_one("#ollama-copy-destination-model", Input).value = ""
            app.query_one("#ollama-pull-model-name", Input).value = ""
            app.query_one("#ollama-create-model-name", Input).value = ""
            app.query_one("#ollama-create-modelfile-path", Input).value = ""
            app.query_one("#ollama-push-model-name", Input).value = ""
            app.query_one("#ollama-embeddings-model-name", Input).value = ""
            app.query_one("#ollama-embeddings-prompt", Input).value = ""

            # Output TextAreas
            app.query_one("#ollama-combined-output", RichLog).clear()
            # Main log output
            log_output = app.query_one("#ollama-log-output", RichLog)
            log_output.clear()
            log_output.write("Switched to Ollama view. Output log cleared and initialized.")

        except QueryError as qe: # pragma: no cover
            logger.warning(f"Ollama UI clear: Could not find one or more UI elements during view switch: {qe}")
            app.notify("Warning: Some Ollama UI elements might not have been reset properly.", severity="warning")

        logger.info("Switched to Ollama view and cleared/initialized fields.")
        # app.notify("Switched to Ollama view.") # Optional notification

    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_nav_button_pressed: {e}", exc_info=True)
        app.notify("Error switching to Ollama view: Could not find required UI elements.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_nav_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while switching to Ollama view.", severity="error")


async def handle_ollama_browse_exec_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Browse' button press for Ollama executable path."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama browse executable button pressed.")

    exec_filters = Filters(
        ("Executables", lambda p: p.is_file() and p.name.lower() == "ollama"),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.home()),
            title="Select Ollama Executable",
            filters=exec_filters,
            context="ollama_models"
        ),
        callback=_make_path_update_callback(app, "ollama-exec-path")
    )


async def handle_ollama_start_service_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Start Ollama Service' button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Start Service' button pressed.")
    
    try:
        exec_path_input = app.query_one("#ollama-exec-path", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)
        
        exec_path = exec_path_input.value.strip()
        
        # Check if we have a valid executable path
        if not exec_path:
            # Try to find ollama in PATH
            import shutil
            exec_path = shutil.which("ollama")
            if exec_path:
                exec_path_input.value = exec_path
                logger.info(f"Found ollama in PATH at: {exec_path}")
            else:
                app.notify("Ollama executable path is required. Please browse for the ollama executable or ensure it's in PATH.", severity="error")
                exec_path_input.focus()
                return
        
        # Verify the executable exists
        if not Path(exec_path).is_file():
            app.notify(f"Ollama executable not found at: {exec_path}", severity="error")
            exec_path_input.focus()
            return
        
        log_output_widget.clear()
        log_output_widget.write(f"Starting Ollama service using: {exec_path}")
        
        # Start the Ollama service
        cmd = [exec_path, "serve"]
        
        app.run_worker(
            _stream_process,
            cmd,
            stream_worker_output_to_log(app, "ollama-log-output"),
            thread=True,
            name="ollama_serve_process",
            group="ollama_serve",
            exclusive=True,
            description="Running Ollama service")
        
        app.notify("Ollama service starting...", severity="information")
        
    except QueryError as e:
        logger.error(f"QueryError in handle_ollama_start_service_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing UI elements for starting Ollama service.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error in handle_ollama_start_service_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while starting Ollama service.", severity="error")


async def handle_ollama_stop_service_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Stop Ollama Service' button press."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Stop Service' button pressed.")
    
    try:
        log_output_widget = app.query_one("#ollama-log-output", RichLog)
        
        # Cancel any running Ollama service workers
        cancelled = app.workers.cancel_group("ollama_serve")
        
        if cancelled:
            log_output_widget.write("\n[bold yellow]Stopping Ollama service...[/bold yellow]")
            app.notify("Ollama service stopped.", severity="information")
        else:
            log_output_widget.write("\n[bold yellow]No running Ollama service found.[/bold yellow]")
            app.notify("No running Ollama service to stop.", severity="warning")
            
    except QueryError as e:
        logger.error(f"QueryError in handle_ollama_stop_service_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing UI elements for stopping Ollama service.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error in handle_ollama_stop_service_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while stopping Ollama service.", severity="error")


async def handle_ollama_list_models_button_pressed(app: "TldwCli") -> None:
    """Handles the 'List Models' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'List Models' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        log_output_widget = app.query_one("#ollama-combined-output", RichLog)

        base_url = base_url_input.value.strip()
        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return

        log_output_widget.clear()
        log_output_widget.write(f"Attempting to list models from: {base_url}...")

        data, error = await asyncio.to_thread(
            ollama_list_local_models,
            base_url=base_url
        )
        
        if error:
            log_output_widget.write(f"Error listing models: {error}")
            app.notify("Error listing Ollama models.", severity="error")
        elif data and data.get('models'):
            try:
                # Format the models list
                formatted_models = json.dumps(data['models'], indent=2)
                log_output_widget.write(formatted_models)
                app.notify(f"Successfully listed {len(data['models'])} Ollama models.")
            except (TypeError, KeyError, json.JSONDecodeError) as e:
                log_output_widget.write(f"Error processing model list response: {e}\nRaw data: {data}")
                app.notify("Error processing model list from Ollama.", severity="error")
        else:
            log_output_widget.write("No models found or unexpected response.")
            app.notify("No Ollama models found or unexpected response.", severity="warning")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_list_models_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for listing models.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_list_models_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while listing Ollama models.", severity="error")


async def handle_ollama_show_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Show Model Info' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Show Model Info' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-show-model-name", Input)
        log_output_widget = app.query_one("#ollama-combined-output", RichLog)
        # general_log_widget = app.query_one("#ollama-log-output", RichLog)

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
        # general_log_widget.write(f"Attempting to show info for model: {model_name} from {base_url}")

        data, error = await asyncio.to_thread(
            ollama_model_info,
            base_url=base_url,
            model_name=model_name
        )
        if error:
            log_output_widget.write(f"Error showing model info for '{model_name}': {error}")
            app.notify(f"Error fetching info for {model_name}.", severity="error")
        elif data:
            try:
                formatted_info = json.dumps(data, indent=2)
                log_output_widget.write(formatted_info)
                app.notify(f"Successfully fetched info for {model_name}.")
            except (TypeError, json.JSONDecodeError) as e:
                log_output_widget.write(f"Error processing model info response: {e}\nRaw data: {data}")
                app.notify(f"Error processing info for {model_name}.", severity="error")
        else:
            log_output_widget.write(f"No information returned for model '{model_name}'.")
            app.notify(f"No info for {model_name} or unexpected response.", severity="warning")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_show_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for showing model info.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_show_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while showing model info.", severity="error")


async def handle_ollama_delete_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Delete Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Delete Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-delete-model-name", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog) # General log for delete

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
            permanent=True
        )
        
        confirmed = await app.push_screen_wait(dialog)
        if not confirmed:
            logger.info(f"Model deletion cancelled for: {model_name}")
            log_output_widget.write(f"Model deletion cancelled for: {model_name}")
            return

        log_output_widget.write(f"Attempting to delete model: {model_name} from {base_url}")

        def stream_to_log(message: str):
            app.call_from_thread(log_output_widget.write, message)

        data, error = await asyncio.to_thread(
            ollama_delete_model,
            base_url=base_url,
            model_name=model_name,
            stream_log_callback=stream_to_log
        )
        if error:
            log_output_widget.write(f"[bold red]Error deleting model '{model_name}': {error}[/bold red]")
            app.notify(f"Error deleting {model_name}.", severity="error")
        else:
            # Stream callback should have provided detailed progress.
            # 'data' might contain final status if any, or might be None if stream handled all.
            if data and data.get('status') == 'success':
                log_output_widget.write(f"Model '{model_name}' deleted successfully (final status).")
                app.notify(f"Model {model_name} deleted.")
            elif not data and not error: # Common for stream-focused ops
                 log_output_widget.write(f"Model '{model_name}' delete process finished. Check logs above for status.")
                 app.notify(f"Model {model_name} delete process completed.")
            else: # Some other response
                log_output_widget.write(f"Delete model response for '{model_name}': {data if data else 'No specific final status message.'}")
                app.notify(f"Model {model_name} deletion process finished.")
        # Optionally, refresh the model list:
        # app.call_after_refresh(lambda: app.run_action("ollama_list_models_button_pressed"))
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_delete_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for deleting model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_delete_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while deleting model.", severity="error")


async def handle_ollama_copy_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Copy Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Copy Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        source_model_input = app.query_one("#ollama-copy-source-model", Input)
        dest_model_input = app.query_one("#ollama-copy-destination-model", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)

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

        log_output_widget.write(f"Attempting to copy model: {source_model} to {dest_model} from {base_url}")

        data, error = await asyncio.to_thread(
            ollama_copy_model,
            base_url=base_url,
            source=source_model,
            destination=dest_model
        )
        if error:
            log_output_widget.write(f"[bold red]Error copying model '{source_model}' to '{dest_model}': {error}[/bold red]")
            app.notify(f"Error copying {source_model}.", severity="error")
        else:
            # Ollama copy API returns 200 OK on success with no body.
            # The ollama_model_mgmt.py wrapper might return a success message in 'data'.
            if data and data.get('status') == 'success':
                 log_output_widget.write(f"Model '{source_model}' copied to '{dest_model}' successfully.")
                 app.notify(f"Model {source_model} copied to {dest_model}.")
            else: # Should be success if no error
                 log_output_widget.write(f"Model '{source_model}' copy to '{dest_model}' initiated. Ollama provides no detailed progress for copy via API. Assuming success if no error reported.")
                 app.notify(f"Model {source_model} copy to {dest_model} initiated.")
        # Optionally, refresh model list
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_copy_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for copying model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_copy_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while copying model.", severity="error")


async def handle_ollama_pull_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Pull Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Pull Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-pull-model-name", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)

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

        log_output_widget.write(f"Attempting to pull model: {model_name} from {base_url}")

        def stream_to_log(message: str):
            app.call_from_thread(log_output_widget.write, message)

        # Consider adding 'insecure' parameter if UI supports it, default False
        data, error = await asyncio.to_thread(
            ollama_pull_model,
            base_url=base_url,
            model_name=model_name,
            stream_log_callback=stream_to_log
        )
        if error:
            log_output_widget.write(f"[bold red]Error pulling model '{model_name}': {error}[/bold red]")
            app.notify(f"Error pulling {model_name}.", severity="error")
        else:
            log_output_widget.write(f"Model '{model_name}' pull process finished. Check logs above for status.")
            app.notify(f"Model {model_name} pull completed.")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_pull_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for pulling model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_pull_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while pulling model.", severity="error")


async def handle_ollama_browse_modelfile_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Browse for Modelfile' button press for Ollama create model."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Browse for Modelfile' button pressed.")

    # No specific filters for "Modelfile" by extension, so allow all files or common text files.
    # Users should know what a Modelfile looks like.
    modelfile_filters = Filters(
        ("Modelfiles (Modelfile, *.txt)", lambda p: p.name.lower() == "modelfile" or p.suffix.lower() == ".txt"),
        ("All files (*.*)", lambda p: True)
    )
    await app.push_screen(
        FileOpen(
            location=str(Path.cwd()),
            title="Select Modelfile",
            filters=modelfile_filters,
            context="ollama_models"
        ),
        callback=_make_path_update_callback(app, "ollama-create-modelfile-path")
    )


async def handle_ollama_create_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Create Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Create Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-create-model-name", Input)
        modelfile_path_input = app.query_one("#ollama-create-modelfile-path", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)

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
            app.notify("Use 'Browse for Modelfile' to select a file.", severity="information")
            return
        if not Path(modelfile_path).is_file():
            app.notify(f"Modelfile not found at: {modelfile_path}", severity="error")
            # modelfile_path_input.focus()
            return


        log_output_widget.write(f"Attempting to create model: {model_name} using Modelfile: {modelfile_path} from {base_url}")

        def stream_to_log(message: str):
            app.call_from_thread(log_output_widget.write, message)

        data, error = await asyncio.to_thread(
            ollama_create_model,
            base_url=base_url,
            model_name=model_name,
            path=modelfile_path,
            stream_log_callback=stream_to_log
        )
        if error:
            log_output_widget.write(f"[bold red]Error creating model '{model_name}': {error}[/bold red]")
            app.notify(f"Error creating {model_name}.", severity="error")
        else:
            log_output_widget.write(f"Model '{model_name}' creation process finished. Check logs above for status.")
            app.notify(f"Model {model_name} creation completed.")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_create_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for creating model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_create_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while creating model.", severity="error")


async def handle_ollama_push_model_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Push Model' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Push Model' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-push-model-name", Input)
        log_output_widget = app.query_one("#ollama-log-output", RichLog)

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

        log_output_widget.write(f"Attempting to push model: {model_name} from {base_url}")

        def stream_to_log(message: str):
            app.call_from_thread(log_output_widget.write, message)

        # Consider adding 'insecure' parameter if UI supports it, default False
        data, error = await asyncio.to_thread(
            ollama_push_model,
            base_url=base_url,
            model_name=model_name,
            stream_log_callback=stream_to_log
        )
        if error:
            log_output_widget.write(f"[bold red]Error pushing model '{model_name}': {error}[/bold red]")
            app.notify(f"Error pushing {model_name}.", severity="error")
        else:
            log_output_widget.write(f"Model '{model_name}' push process finished. Check logs above for status.")
            app.notify(f"Model {model_name} push completed.")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_push_model_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for pushing model.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_push_model_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while pushing model.", severity="error")


async def handle_ollama_embeddings_button_pressed(app: "TldwCli") -> None:
    """Handles the 'Generate Embeddings' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'Generate Embeddings' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        model_name_input = app.query_one("#ollama-embeddings-model-name", Input)
        prompt_input = app.query_one("#ollama-embeddings-prompt", Input)
        embeddings_output_widget = app.query_one("#ollama-combined-output", RichLog)
        # general_log_widget = app.query_one("#ollama-log-output", RichLog)


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

        # general_log_widget = app.query_one("#ollama-log-output", RichLog) # If you want general logs too
        # general_log_widget.write(f"Attempting to generate embeddings for model: {model_name} with prompt: '{prompt[:30]}...' from {base_url}")

        data, error = await asyncio.to_thread(
            ollama_generate_embeddings,
            base_url=base_url,
            model_name=model_name,
            prompt=prompt
        )
        if error:
            embeddings_output_widget.write(f"Error generating embeddings: {error}")
            app.notify("Error generating embeddings.", severity="error")
        elif data and data.get('embedding'):
            try:
                # The embedding is usually a list of floats.
                formatted_embedding = json.dumps(data['embedding'], indent=2)
                embeddings_output_widget.write(formatted_embedding)
                app.notify("Embeddings generated successfully.")
            except (TypeError, KeyError, json.JSONDecodeError) as e:
                embeddings_output_widget.write(f"Error processing embeddings response: {e}\nRaw data: {data}")
                app.notify("Error processing embeddings response.", severity="error")
        else:
            embeddings_output_widget.write(f"No embeddings returned or unexpected response: {data}")
            app.notify("No embeddings returned or unexpected response.", severity="warning")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_embeddings_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for generating embeddings.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_embeddings_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while generating embeddings.", severity="error")


async def handle_ollama_ps_button_pressed(app: "TldwCli") -> None:
    """Handles the 'List Running Models (ps)' button press for Ollama."""
    logger = getattr(app, "loguru_logger", logging.getLogger(__name__))
    logger.debug("Ollama 'List Running Models (ps)' button pressed.")
    try:
        base_url_input = app.query_one("#ollama-server-url", Input)
        ps_output_widget = app.query_one("#ollama-combined-output", RichLog)
        # general_log_widget = app.query_one("#ollama-log-output", RichLog)

        base_url = base_url_input.value.strip()
        if not base_url:
            app.notify("Ollama Server URL is required.", severity="error")
            base_url_input.focus()
            return

        ps_output_widget.clear()
        # general_log_widget = app.query_one("#ollama-log-output", RichLog)
        # general_log_widget.write(f"Attempting to list running models (ps) from: {base_url}")

        data, error = await asyncio.to_thread(
            ollama_list_running_models,
            base_url=base_url
        )
        if error:
            ps_output_widget.write(f"Error listing running models: {error}")
            app.notify("Error listing running Ollama models.", severity="error")
        elif data and data.get('models'):
            try:
                formatted_ps_info = json.dumps(data['models'], indent=2)
                ps_output_widget.write(formatted_ps_info)
                app.notify(f"Successfully listed {len(data['models'])} running Ollama models.")
            except (TypeError, KeyError, json.JSONDecodeError) as e:
                ps_output_widget.write(f"Error processing running models response: {e}\nRaw data: {data}")
                app.notify("Error processing running models list.", severity="error")
        else:
            ps_output_widget.write("No running models found or unexpected response.")
            app.notify("No running Ollama models found or response format issue.", severity="warning")
    except QueryError as e: # pragma: no cover
        logger.error(f"QueryError in handle_ollama_ps_button_pressed: {e}", exc_info=True)
        app.notify("Error accessing Ollama UI elements for listing running models.", severity="error")
    except Exception as e: # pragma: no cover
        logger.error(f"Unexpected error in handle_ollama_ps_button_pressed: {e}", exc_info=True)
        app.notify("An unexpected error occurred while listing running models.", severity="error")

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
