# /tldw_chatbook/Event_Handlers/llm_management_events_transformers.py
from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as _loguru_fallback_logger
from rich.text import Text
from textual.widgets import Button, Input, RichLog

from tldw_chatbook.Utils.input_validation import sanitize_string as sanitize_input
from tldw_chatbook.Utils.optional_deps import get_safe_import
from .llm_management_events import _make_path_update_callback

_huggingface_hub = get_safe_import("huggingface_hub")
hf_constants = (
    importlib.import_module("huggingface_hub.constants")
    if _huggingface_hub is not None
    else None
)

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    # textual_fspicker is imported dynamically in the handler

MAX_LOCAL_MODEL_RESULTS = 500


def scan_transformers_local_models(models_path: Path) -> list[str]:
    """Return bounded display names for local model roots containing weights."""

    found_models: list[str] = []
    seen: set[str] = set()
    for config_path in models_path.rglob("config.json"):
        if not config_path.is_file():
            continue
        model_root = config_path.parent
        if not any(
            (model_root / filename).exists()
            for filename in (
                "pytorch_model.bin",
                "model.safetensors",
                "tf_model.h5",
            )
        ):
            continue
        try:
            relative_root = model_root.relative_to(models_path)
            parts = relative_root.parts
            if parts and parts[0].startswith("models--"):
                display_name = parts[0][len("models--") :].replace("--", "/", 1)
            else:
                display_name = str(relative_root)
        except ValueError:
            display_name = (
                models_path.name if model_root == models_path else model_root.name
            )
        display_name = " ".join(sanitize_input(display_name, max_length=256).split())
        if display_name and display_name not in seen:
            seen.add(display_name)
            found_models.append(display_name)
            if len(found_models) >= MAX_LOCAL_MODEL_RESULTS:
                break
    return found_models


async def handle_transformers_list_local_models_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("Transformers list local models button pressed.")

    models_dir_input: Input = window.query_one("#transformers-models-dir-path", Input)
    models_list_widget: RichLog = window.query_one(
        "#transformers-local-models-list", RichLog
    )
    log_output_widget: RichLog = window.query_one("#transformers-log-output", RichLog)

    models_dir_str = models_dir_input.value.strip()
    if not models_dir_str:
        app.notify("Please specify a local models directory first.", severity="warning")
        models_dir_input.focus()
        return

    models_path = Path(models_dir_str).resolve()  # Resolve to absolute path
    if not models_path.is_dir():
        app.notify("Local models directory was not found.", severity="error")
        models_dir_input.focus()
        return

    generation = window._begin_async_presentation("transformers-model-scan")
    models_list_widget.clear()
    log_output_widget.write("Scanning for local models.\n")
    app.notify("Scanning for local models...")
    del models_dir_input, models_list_widget, log_output_widget

    try:
        found_models_display = await asyncio.to_thread(
            scan_transformers_local_models,
            models_path,
        )
        if not window._owns_async_presentation(
            "transformers-model-scan",
            generation,
        ):
            return
        models_list_widget = window.query_one(
            "#transformers-local-models-list",
            RichLog,
        )
        log_output_widget = window.query_one("#transformers-log-output", RichLog)
        if found_models_display:
            model_listing = (
                f"Found {len(found_models_display)} potential local models.\n"
                + "\n".join(found_models_display)
            )
            models_list_widget.write(Text(model_listing))
            app.notify(
                f"Found {len(found_models_display)} potential local models "
                "(based on config.json and weights)."
            )
        else:
            models_list_widget.write(
                "No model directories found with config.json and model weights."
            )
            app.notify(
                "No local models found with this scan method.", severity="information"
            )
        log_output_widget.write("Local model scan complete.\n")

    except Exception as e:
        logger.error("Local model scan failed (category={}).", type(e).__name__)
        if not window._owns_async_presentation(
            "transformers-model-scan",
            generation,
        ):
            return
        log_output_widget = window.query_one("#transformers-log-output", RichLog)
        log_output_widget.write("[bold red]Local model scan failed.[/]\n")
        app.notify("Error during local model scan.", severity="error")


async def handle_transformers_browse_models_dir_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Transformers browse models directory button pressed.")

    try:
        from textual_fspicker import FileOpen
    except ImportError:
        app.notify(
            "File picker utility (textual-fspicker) not available.", severity="error"
        )
        logger.error("textual_fspicker not found for Transformers model dir browsing.")
        return

    default_loc_str = str(Path.home())
    if hf_constants is not None:
        try:
            # Use HF_HOME if set, otherwise default cache.
            # hf_constants.HF_HUB_CACHE points to the 'hub' subdir, e.g., ~/.cache/huggingface/hub
            # We might want to default to ~/.cache/huggingface or where user typically stores models
            hf_cache_dir = Path(hf_constants.HF_HUB_CACHE)
            if hf_cache_dir.is_dir():
                default_loc_str = str(hf_cache_dir)
            elif (
                hf_cache_dir.parent.is_dir()
            ):  # Try one level up, e.g. ~/.cache/huggingface
                default_loc_str = str(hf_cache_dir.parent)
        except Exception:  # pylint: disable=broad-except
            pass

    logger.debug("Opening Transformers models directory picker.")

    await app.push_screen(
        FileOpen(
            location=default_loc_str,
            select_dirs=True,  # We want to select a directory
            title="Select Local Hugging Face Models Directory",
            # No specific filters needed for directory selection
        ),
        # This callback will update the Input widget with id "transformers-models-dir-path"
        callback=_make_path_update_callback(
            window, app, "transformers-models-dir-path"
        ),
    )


# --- Button Handler Map ---
TRANSFORMERS_BUTTON_HANDLERS = {
    "transformers-list-local-models-button": handle_transformers_list_local_models_button_pressed,
    "transformers-browse-models-dir-button": handle_transformers_browse_models_dir_button_pressed,
}

#
# End of llm_management_events_transformers.py
########################################################################################################################
