# /tldw_chatbook/Event_Handlers/llm_management_events_transformers.py
from __future__ import annotations

import asyncio
import functools
from loguru import logger as _loguru_fallback_logger
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from rich.text import Text
from textual.widgets import Input, RichLog, Button
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    current_llm_destination,
    terminate_process_bounded,
)
from tldw_chatbook.Utils.log_sanitizer import sanitize_string
from tldw_chatbook.Utils.log_widget_manager import LogWidgetManager

# For listing local models, you might need to interact with huggingface_hub or scan directories
try:
    from huggingface_hub import HfApi, constants as hf_constants  # noqa: F401

    # from huggingface_hub import list_models, model_info as hf_model_info # For online search
    # from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False
    hf_constants = None  # type: ignore

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    # textual_fspicker is imported dynamically in the handler

# Import shared helpers if needed
from .llm_management_events import (
    _make_path_update_callback,
)

MAX_LOCAL_MODEL_RESULTS = 500


def _valid_huggingface_repo_id(repo_id: str) -> bool:
    """Accept ordinary one/two-segment repo IDs without path or option syntax."""

    if not repo_id or len(repo_id) > 256 or "\\" in repo_id:
        return False
    parts = repo_id.split("/")
    if len(parts) not in (1, 2):
        return False
    for part in parts:
        if (
            part in {"", ".", ".."}
            or part.startswith("-")
            or not all(character.isalnum() or character in "._-" for character in part)
        ):
            return False
    return True


def _valid_huggingface_revision(revision: str) -> bool:
    """Accept a branch, tag, or commit value without CLI option syntax."""

    if (
        not revision
        or len(revision) > 256
        or revision.startswith("-")
        or "\\" in revision
    ):
        return False
    parts = revision.split("/")
    return all(
        part not in {"", ".", ".."}
        and all(character.isalnum() or character in "._-" for character in part)
        for part in parts
    )


# --- Worker function for model download (can be similar to the existing one) ---
def run_transformers_model_download_worker(
    app_instance: "TldwCli",
    command: List[str],
    models_base_dir_for_cwd: str,
) -> str:
    """Download a model while suppressing command, output, and error payloads."""

    process: Optional[subprocess.Popen] = None
    try:
        cwd = (
            models_base_dir_for_cwd if Path(models_base_dir_for_cwd).is_dir() else None
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
        )
        pid = getattr(process, "pid", "unknown")
        _publish_transformers_status(
            app_instance,
            f"[PID:{pid}] Download started.\n",
        )
        try:
            process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            terminate_process_bounded(process, timeout=5)
            message = f"Model download (PID:{pid}) timed out."
            _publish_transformers_status(
                app_instance,
                f"{message}\n",
            )
            return message
        return_code = process.returncode
        if return_code == 0:
            message = f"Model download (PID:{pid}) completed successfully."
        else:
            message = f"Model download (PID:{pid}) failed with code: {return_code}."
        _publish_transformers_status(
            app_instance,
            f"{message}\n",
        )
        return message
    except Exception as exc:
        exception_category = type(exc).__name__
        message = f"Model download failed (category={exception_category})."
        _publish_transformers_status(
            app_instance,
            f"{message}\n",
        )
        return message
    finally:
        if process is not None and process.poll() is None:
            terminate_process_bounded(process, timeout=5)


def _publish_transformers_status(app: "TldwCli", message: str) -> None:
    """Marshal bounded worker status without letting presentation fail the worker."""

    try:
        app.call_from_thread(
            _update_current_transformers_log,
            app,
            message,
        )
    except Exception:
        pass


def _update_current_transformers_log(app: "TldwCli", message: str) -> None:
    """Write model-operation output only to the current mounted destination."""

    window = current_llm_destination(app)
    if window is not None:
        LogWidgetManager.update_transformers_log(window, message)


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
        display_name = sanitize_string(display_name).replace("\n", " ").strip()[:256]
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


async def handle_transformers_download_model_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("Transformers download model button pressed.")

    repo_id_input: Input = window.query_one("#transformers-download-repo-id", Input)
    revision_input: Input = window.query_one("#transformers-download-revision", Input)
    models_dir_input: Input = window.query_one("#transformers-models-dir-path", Input)
    log_output_widget: RichLog = window.query_one("#transformers-log-output", RichLog)

    repo_id = repo_id_input.value.strip()
    revision = revision_input.value.strip() or None
    models_dir_str = models_dir_input.value.strip()

    if not repo_id:
        app.notify("Model Repo ID is required to download.", severity="error")
        repo_id_input.focus()
        return
    if not _valid_huggingface_repo_id(repo_id):
        app.notify("Model Repo ID is invalid.", severity="error")
        repo_id_input.focus()
        return
    if revision is not None and not _valid_huggingface_revision(revision):
        app.notify("Model revision is invalid.", severity="error")
        revision_input.focus()
        return

    if not models_dir_str:
        # Default to HF cache if not specified, but warn user.
        if (
            HUGGINGFACE_HUB_AVAILABLE
            and hf_constants
            and Path(hf_constants.HF_HUB_CACHE).is_dir()
        ):
            models_dir_str = str(hf_constants.HF_HUB_CACHE)
            app.notify(
                "No local directory set; using the Hugging Face cache.",
                severity="warning",
                timeout=7,
            )
            models_dir_input.value = models_dir_str  # Update UI
        else:
            app.notify(
                "Local models directory must be set to specify download location.",
                severity="error",
            )
            models_dir_input.focus()
            return

    # huggingface-cli download --local-dir specifies the *target* directory for THIS model's files.
    # It will create subdirectories based on the repo structure under this path.
    # Example: if --local-dir is /my/models/bert, files go into /my/models/bert/snapshots/hash/...
    # We want the user-provided models_dir_str to be the root under which models are organized.
    # So, the --local-dir for huggingface-cli should be models_dir_str itself, or a subfolder we define.
    # Let's make it download into a subfolder named after the repo_id within models_dir_str for clarity.

    # Sanitize repo_id for use as a directory name part
    safe_repo_id_subdir = repo_id.replace("/", "--")
    target_model_specific_dir = Path(models_dir_str) / safe_repo_id_subdir

    log_output_widget.write("Starting model download.\n")
    target_model_specific_dir.mkdir(
        parents=True, exist_ok=True
    )  # Ensure target dir exists

    command = [
        "huggingface-cli",
        "download",
        repo_id,
        "--local-dir",
        str(target_model_specific_dir),
        "--local-dir-use-symlinks",
        "False",  # Usually want actual files for local management
    ]
    if revision:
        command.extend(["--revision", revision])

    # The worker CWD should be a neutral place, or the parent of target_model_specific_dir
    worker_cwd = models_dir_str

    worker_callable = functools.partial(
        run_transformers_model_download_worker,
        app,
        command,
        worker_cwd,
    )
    app.run_worker(
        worker_callable,
        group="transformers_download",
        description="Downloading Hugging Face model",
        exclusive=False,
        thread=True,
    )
    app.notify("Starting model download.")


async def handle_transformers_browse_models_dir_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Transformers browse models directory button pressed.")

    try:
        from textual_fspicker import (
            FileOpen,
            Filters,  # noqa: F401
        )  # Ensure it's imported for runtime
    except ImportError:
        app.notify(
            "File picker utility (textual-fspicker) not available.", severity="error"
        )
        logger.error("textual_fspicker not found for Transformers model dir browsing.")
        return

    default_loc_str = str(Path.home())
    if HUGGINGFACE_HUB_AVAILABLE and hf_constants:
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
    "transformers-download-model-button": handle_transformers_download_model_button_pressed,
    "transformers-browse-models-dir-button": handle_transformers_browse_models_dir_button_pressed,
}

#
# End of llm_management_events_transformers.py
########################################################################################################################
