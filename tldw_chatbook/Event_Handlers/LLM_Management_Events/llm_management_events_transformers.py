# /tldw_chatbook/Event_Handlers/llm_management_events_transformers.py
from __future__ import annotations

import logging
from loguru import logger as _loguru_fallback_logger
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional
import functools  # For download worker

from textual.widgets import Input, RichLog, Button
from textual.css.query import QueryError

# For listing local models, you might need to interact with huggingface_hub or scan directories
try:
    from huggingface_hub import HfApi, constants as hf_constants

    # from huggingface_hub import list_models, model_info as hf_model_info # For online search
    # from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False
    hf_constants = None  # type: ignore

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    # textual_fspicker is imported dynamically in the handler

# Import shared helpers if needed
from .llm_management_events import \
    _make_path_update_callback  # _stream_process, stream_worker_output_to_log (not used by download worker directly)


# --- Worker function for model download (can be similar to the existing one) ---
async def run_transformers_model_download_worker(app_instance: "TldwCli", command: List[str],
                                                 models_base_dir_for_cwd: str) -> str:
    logger = getattr(app_instance, "loguru_logger", _loguru_fallback_logger)
    quoted_command = ' '.join(shlex.quote(c) for c in command)
    # The actual target download path is part of the command (--local-dir)
    logger.info(f"Transformers Download WORKER starting: {quoted_command}")

    process: Optional[subprocess.Popen] = None
    final_status_message = f"Transformers Download WORKER: Default status for {quoted_command}"
    pid_str = "N/A"

    try:
        # The command already includes --local-dir pointing to the exact target.
        # We might want to run huggingface-cli from a neutral directory or models_base_dir_for_cwd
        # if --local-dir is relative, but since we make it absolute, cwd is less critical.
        # For consistency, let's use models_base_dir_for_cwd if provided and valid.
        cwd_to_use = models_base_dir_for_cwd if Path(models_base_dir_for_cwd).is_dir() else None

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
            bufsize=1,
            cwd=cwd_to_use
        )
        pid_str = str(process.pid) if process and process.pid else "UnknownPID"
        logger.info(f"Transformers Download WORKER: Subprocess launched, PID: {pid_str}")
        app_instance.call_from_thread(app_instance._update_transformers_log, f"[PID:{pid_str}] Download starting...\n")

        # communicate() waits for termination
        stdout_data, stderr_data = process.communicate(timeout=600)  # 10 min timeout for download

        logger.info(
            f"Transformers Download WORKER: communicate() completed. PID {pid_str}, Exit Code: {process.returncode}")

        if stdout_data:
            logger.info(f"Transformers Download WORKER STDOUT:\n{stdout_data.strip()}")
            app_instance.call_from_thread(app_instance._update_transformers_log,
                                          f"--- STDOUT (PID:{pid_str}) ---\n{stdout_data.strip()}\n")
        if stderr_data:
            logger.error(f"Transformers Download WORKER STDERR:\n{stderr_data.strip()}")
            app_instance.call_from_thread(app_instance._update_transformers_log,
                                          f"--- STDERR (PID:{pid_str}) ---\n[bold red]{stderr_data.strip()}[/]\n")

        if process.returncode != 0:
            final_status_message = f"Model download (PID:{pid_str}) failed with code: {process.returncode}."
            if stderr_data: final_status_message += f"\nSTDERR: {stderr_data.strip()}"
        else:
            final_status_message = f"Model download (PID:{pid_str}) completed successfully (code: {process.returncode}). Model should be in target --local-dir."

        app_instance.call_from_thread(app_instance._update_transformers_log, f"{final_status_message}\n")
        return final_status_message

    except FileNotFoundError:
        msg = f"ERROR: huggingface-cli not found. Please ensure it's installed and in PATH."
        logger.error(msg)
        app_instance.call_from_thread(app_instance._update_transformers_log, f"[bold red]{msg}[/]\n")
        raise
    except subprocess.TimeoutExpired:
        msg = f"ERROR: Model download (PID:{pid_str}) timed out after 600s."
        logger.error(msg)
        if process: process.kill()
        app_instance.call_from_thread(app_instance._update_transformers_log, f"[bold red]{msg}[/]\n")
        raise RuntimeError(msg)  # Make worker fail
    except Exception as err:
        msg = f"CRITICAL ERROR in Transformers Download worker: {err} (Command: {quoted_command})"
        logger.opt(exception=True).error(msg)
        app_instance.call_from_thread(app_instance._update_transformers_log, f"[bold red]{msg}[/]\n")
        raise
    finally:
        logger.info(f"Transformers Download WORKER: Worker for '{quoted_command}' finishing.")
        if process and process.poll() is None:
            logger.warning(f"Transformers Download WORKER (PID:{pid_str}): Process still running in finally. Killing.")
            process.kill()


async def handle_transformers_list_local_models_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("Transformers list local models button pressed.")

    models_dir_input: Input = app.query_one("#transformers-models-dir-path", Input)
    models_list_widget: RichLog = app.query_one("#transformers-local-models-list", RichLog)
    log_output_widget: RichLog = app.query_one("#transformers-log-output", RichLog)

    models_dir_str = models_dir_input.value.strip()
    if not models_dir_str:
        app.notify("Please specify a local models directory first.", severity="warning")
        models_dir_input.focus()
        return

    models_path = Path(models_dir_str).resolve()  # Resolve to absolute path
    if not models_path.is_dir():
        app.notify(f"Directory not found: {models_path}", severity="error")
        models_dir_input.focus()
        return

    models_list_widget.clear()
    log_output_widget.write(f"Scanning for models in: {models_path}...\n")
    app.notify("Scanning for local models...")

    found_models_display = []
    try:
        # This basic scan looks for directories that might be model repos.
        # A 'blobs' and 'refs' subdirectory alongside 'snapshots' is common for full cache structure.
        # Individual model downloads might just have 'snapshots' or be flat.

        # Heuristic 1: Look for 'snapshots' directory, then list its children
        # These children are usually named after commit hashes. Inside them are the actual files.
        # We need to find a way to map these back to a model name.
        # Often, a .gitattributes or similar file at a higher level might exist.

        # Heuristic 2: Look for directories containing config.json
        # This is simpler but might find nested utility models or non-root model dirs.

        count = 0
        for item_path in models_path.rglob("config.json"):
            if item_path.is_file():
                model_root_dir = item_path.parent
                # Try to infer a model name. This is tricky.
                # If models_path is like ".../hub/models--org--modelname", then model_root_dir might be a snapshot hash.
                # If models_path is a custom dir where user put "org/modelname" folders, it's easier.

                display_name = ""
                try:
                    # Attempt to make a "repo_id" like name from the path relative to models_path
                    relative_to_scan_root = model_root_dir.relative_to(models_path)
                    # If models_path is the HF cache, relative_to_scan_root might be "models--org--repo/snapshots/hash"
                    # We want to extract "org/repo"
                    parts = list(relative_to_scan_root.parts)
                    if parts and parts[0].startswith("models--"):
                        name_part = parts[0].replace("models--", "")
                        display_name = name_part.replace("--", "/", 1)  # Replace only first --
                    else:  # Assume a flatter structure or direct model name as folder
                        display_name = str(relative_to_scan_root)
                except ValueError:  # Not a subpath, models_path itself might be the model_root_dir
                    if model_root_dir == models_path:
                        display_name = models_path.name
                    else:  # Some other structure
                        display_name = model_root_dir.name  # Best guess

                # Check for actual model files
                has_weights = (model_root_dir / "pytorch_model.bin").exists() or \
                              (model_root_dir / "model.safetensors").exists() or \
                              (model_root_dir / "tf_model.h5").exists()

                if has_weights:
                    count += 1
                    found_models_display.append(f"[green]{display_name}[/] ([dim]at {model_root_dir}[/dim])")

        if found_models_display:
            models_list_widget.write("\n".join(found_models_display))
            app.notify(f"Found {count} potential local models (based on config.json and weights).")
        else:
            models_list_widget.write("No model directories found with config.json and model weights.")
            app.notify("No local models found with this scan method.", severity="information")
        log_output_widget.write("Local model scan complete.\n")

    except Exception as e:
        logger.opt(exception=True).error(f"Error scanning for local models: {e}")
        log_output_widget.write(f"[bold red]Error scanning models: {e}[/]\n")
        app.notify("Error during local model scan.", severity="error")


async def handle_transformers_download_model_button_pressed(app: "TldwCli") -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("Transformers download model button pressed.")

    repo_id_input: Input = app.query_one("#transformers-download-repo-id", Input)
    revision_input: Input = app.query_one("#transformers-download-revision", Input)
    models_dir_input: Input = app.query_one("#transformers-models-dir-path", Input)
    log_output_widget: RichLog = app.query_one("#transformers-log-output", RichLog)

    repo_id = repo_id_input.value.strip()
    revision = revision_input.value.strip() or None
    models_dir_str = models_dir_input.value.strip()

    if not repo_id:
        app.notify("Model Repo ID is required to download.", severity="error")
        repo_id_input.focus()
        return

    if not models_dir_str:
        # Default to HF cache if not specified, but warn user.
        if HUGGINGFACE_HUB_AVAILABLE and hf_constants and Path(hf_constants.HF_HUB_CACHE).is_dir():
            models_dir_str = str(hf_constants.HF_HUB_CACHE)
            app.notify(f"No local directory set, will download to Hugging Face cache: {models_dir_str}",
                       severity="warning", timeout=7)
            models_dir_input.value = models_dir_str  # Update UI
        else:
            app.notify("Local models directory must be set to specify download location.", severity="error")
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

    log_output_widget.write(
        f"Attempting to download '{repo_id}' (rev: {revision or 'latest'}) to '{target_model_specific_dir}'...\n")
    target_model_specific_dir.mkdir(parents=True, exist_ok=True)  # Ensure target dir exists

    command = [
        "huggingface-cli",
        "download",
        repo_id,
        "--local-dir", str(target_model_specific_dir),
        "--local-dir-use-symlinks", "False"  # Usually want actual files for local management
    ]
    if revision:
        command.extend(["--revision", revision])

    # The worker CWD should be a neutral place, or the parent of target_model_specific_dir
    worker_cwd = models_dir_str

    app.run_worker(
        run_transformers_model_download_worker(app, command, worker_cwd),
        group="transformers_download",
        description=f"Downloading HF Model {repo_id}",
        exclusive=False,
        thread=True,
    )
    app.notify(f"Starting download for {repo_id}...")


async def handle_transformers_browse_models_dir_button_pressed(app: "TldwCli", event: Button.Pressed) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.debug("Transformers browse models directory button pressed.")

    try:
        from textual_fspicker import FileOpen, Filters  # Ensure it's imported for runtime
    except ImportError:
        app.notify("File picker utility (textual-fspicker) not available.", severity="error")
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
            elif hf_cache_dir.parent.is_dir():  # Try one level up, e.g. ~/.cache/huggingface
                default_loc_str = str(hf_cache_dir.parent)
        except Exception:  # pylint: disable=broad-except
            pass

    logger.debug(f"Transformers browse models dir: starting location '{default_loc_str}'")

    await app.push_screen(
        FileOpen(
            location=default_loc_str,
            select_dirs=True,  # We want to select a directory
            title="Select Local Hugging Face Models Directory",
            # No specific filters needed for directory selection
        ),
        # This callback will update the Input widget with id "transformers-models-dir-path"
        callback=_make_path_update_callback(app, "transformers-models-dir-path"),
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
