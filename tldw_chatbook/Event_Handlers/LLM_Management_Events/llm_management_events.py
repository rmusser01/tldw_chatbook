# /tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py
#
#
# Imports
from __future__ import annotations

#
import functools
import os
import shlex
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from loguru import logger as _loguru_fallback_logger

#
# Third-party Imports
from textual.widgets import Button, Input, RichLog, TextArea

#
# Local Imports
if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.Model_Artifacts.gguf_admission import (
    GGUFPathError,
    GGUFSourceChangedError,
    inspect_gguf_structure,
    open_local_gguf,
)
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.Third_Party.textual_fspicker import Filters
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen

from .gguf_source_modes import (
    GGUFSourceMode,
    GGUFSourceSelection,
    acquire_managed_gguf,
    gguf_source_failure_message,
    initial_gguf_selection,
)
from .server_lifecycle import (
    ServerLaunchClaim,
    SnapshotLaunchContext,
    attach_server_claim_resource,
    current_llm_destination,
    release_server_claim,
    reserve_server_launch,
    run_server_subprocess,
    stop_server_process,
    sync_current_llm_destination,
)

#
########################################################################################################################
#
# Constants:

_GGUF_RUNTIME_LOAD_FAILURE = (
    "The runtime could not load this GGUF. Check that its architecture and "
    "quantization are supported."
)
_GGUF_PRIMARY_SOURCE_ARGUMENTS = frozenset(
    {
        "-m",
        "--model",
        "-mu",
        "--model-url",
        "-dr",
        "--docker-repo",
        "-hf",
        "-hfr",
        "--hf-repo",
        "-hff",
        "--hf-file",
        "--models-dir",
        "--models-preset",
        "--embd-gemma-default",
        "--fim-qwen-1.5b-default",
        "--fim-qwen-3b-default",
        "--fim-qwen-7b-default",
        "--fim-qwen-7b-spec",
        "--fim-qwen-14b-spec",
        "--fim-qwen-30b-default",
        "--gpt-oss-20b-default",
        "--gpt-oss-120b-default",
        "--vision-gemma-4b-default",
        "--vision-gemma-12b-default",
    }
)


def _validate_gguf_additional_args(arguments: tuple[str, ...]) -> None:
    """Reject primary source selectors while preserving accepted arguments exactly."""
    if any(
        argument.partition("=")[0] in _GGUF_PRIMARY_SOURCE_ARGUMENTS
        for argument in arguments
    ):
        raise ValueError("additional arguments cannot select a model source")


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


def _settle_source_preparation(
    app: "TldwCli",
    provider: str,
    claim: ServerLaunchClaim,
    status: str | None = None,
) -> bool:
    """Release one exact pre-spawn claim and update only its current destination."""

    def settle() -> bool:
        if not release_server_claim(app, provider, claim):
            return False
        sync_current_llm_destination(app, provider, status)
        return True

    try:
        return bool(app.call_from_thread(settle))
    except Exception:
        try:
            return release_server_claim(app, provider, claim)
        except Exception:
            return False


def _gguf_server_source_failure_message(error: BaseException) -> str:
    """Map ordered external failures before the shared managed taxonomy."""
    from tldw_chatbook.LLM_Management.snapshot_models import SnapshotError

    if isinstance(error, SnapshotError) and error.code == "snapshot_owned_options":
        return (
            "Snapshot management owns the slot options. Remove custom slot flags "
            "and slot environment settings, or disable snapshots for this launch."
        )
    if isinstance(error, GGUFSourceChangedError):
        return "The selected external GGUF changed during validation. Retry."
    if isinstance(error, GGUFPathError):
        return "The selected external GGUF is unavailable. Browse for another file."
    return gguf_source_failure_message(error)


def _close_worker_lease(app: "TldwCli", provider: str, leased: object) -> None:
    """Close a lease that was not transferred without exposing private details."""
    try:
        leased.close()  # type: ignore[attr-defined]
    except BaseException:
        logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
        logger.error(
            "GGUF launch lease close failed (provider={}, category=resource_close_failed).",
            provider,
        )


def _build_gguf_server_command(
    provider: str,
    executable: str,
    model_path: Path | None,
    host: str,
    port: str,
    additional_args: tuple[str, ...],
) -> list[str]:
    command = [executable]
    if model_path is not None:
        command.extend(["--model" if provider == "llamacpp" else "-m", str(model_path)])
    command.extend(("--host", host, "--port", port, *additional_args))
    return command


def _snapshot_listener_exists(base_url: str) -> bool:
    """Probe only the admission-validated numeric endpoint from the launch worker."""
    target = urlsplit(base_url)
    try:
        with socket.create_connection((target.hostname, target.port), timeout=5):
            return True
    except ConnectionRefusedError:
        return False
    # Ambiguous network failures are preflight failures, never proof of ownership.


def _prepare_snapshot_launch(app, command, claim):
    owner = getattr(app, "llamacpp_snapshot_service", None)
    if claim.provider != "llamacpp" or owner is None:
        return command, {}
    from tldw_chatbook.LLM_Management.snapshot_admission import (
        has_owned_slot_options,
        prepare_launch,
    )
    from tldw_chatbook.LLM_Management.snapshot_models import SnapshotError
    from tldw_chatbook.LLM_Management.snapshot_settings import load_snapshot_preferences

    if not load_snapshot_preferences().enabled:
        return command, {}
    environment = dict(os.environ)
    if has_owned_slot_options(tuple(command), environment):
        raise SnapshotError("snapshot_owned_options", submission_possible=False)
    descriptor = prepare_launch(tuple(command), environment, claim, uuid4().hex)
    if descriptor.disabled_reason:
        # Retain safe guidance, but leave the ordinary child and its transport alone.
        claim._snapshot_context = SnapshotLaunchContext(descriptor, None)
        return command, {}
    if owner.store is None:
        raise SnapshotError("snapshot_storage_preparing", submission_possible=False)
    if _snapshot_listener_exists(descriptor.base_url):
        raise SnapshotError("snapshot_endpoint_in_use", submission_possible=False)
    directory = owner.store.prepare_launch_directory(descriptor.launch_id)
    claim._snapshot_context = SnapshotLaunchContext(descriptor, directory)
    return [*command, "--slots", "--slot-save-path", str(directory) + os.sep], {
        "env": descriptor.child_env,
        "private_umask": 0o077,
    }


def _run_gguf_server_worker(
    app: "TldwCli",
    provider: str,
    executable: str,
    host: str,
    port: str,
    additional_args: tuple[str, ...],
    selection: GGUFSourceSelection,
    claim: ServerLaunchClaim,
) -> str:
    """Prepare exactly one GGUF authority, then delegate process ownership."""
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    leased: object | None = None
    try:
        selection.validate_for(provider)
        if claim.cancel_event.is_set():
            _settle_source_preparation(app, provider, claim)
            return f"{provider} launch cancelled"

        model_path: Path | None
        if selection.mode is GGUFSourceMode.EMBEDDED:
            model_path = None
        elif selection.mode is GGUFSourceMode.EXTERNAL:
            with open_local_gguf(selection.external_path) as opened:
                inspect_gguf_structure(
                    opened.handle,
                    file_size=opened.identity.size_bytes,
                )
                if claim.cancel_event.is_set():
                    _settle_source_preparation(app, provider, claim)
                    return f"{provider} launch cancelled"
                model_path = opened.path
        else:
            model_path, leased = acquire_managed_gguf(
                managed_service(),
                selection.managed_ref,  # type: ignore[arg-type]
            )
            if not attach_server_claim_resource(
                app,
                provider,
                claim,
                leased,
            ):
                _settle_source_preparation(app, provider, claim)
                return f"{provider} launch cancelled"
            leased = None
        command = _build_gguf_server_command(
            provider,
            executable,
            model_path,
            host,
            port,
            additional_args,
        )
        command, snapshot_options = _prepare_snapshot_launch(app, command, claim)
        if claim.cancel_event.is_set():
            _settle_source_preparation(app, provider, claim)
            return f"{provider} launch cancelled"
        return run_server_subprocess(
            app,
            provider,
            command,
            claim,
            subprocess,
            cwd=Path(executable).parent if provider == "llamafile" else None,
            nonzero_status=_GGUF_RUNTIME_LOAD_FAILURE,
            **snapshot_options,
        )
    except Exception as error:
        logger.error(
            "GGUF source preparation failed (provider={}, category=source_preparation_failed).",
            provider,
        )
        _settle_source_preparation(
            app,
            provider,
            claim,
            _gguf_server_source_failure_message(error),
        )
        return f"{provider} source preparation failed"
    finally:
        if leased is not None:
            _close_worker_lease(app, provider, leased)


def run_llamafile_server_worker(
    app_instance: "TldwCli",
    executable: str,
    host: str,
    port: str,
    additional_args: tuple[str, ...],
    selection: GGUFSourceSelection,
    claim: ServerLaunchClaim,
) -> str:
    return _run_gguf_server_worker(
        app_instance,
        "llamafile",
        executable,
        host,
        port,
        additional_args,
        selection,
        claim,
    )


def run_llamacpp_server_worker(
    app_instance: "TldwCli",
    executable: str,
    host: str,
    port: str,
    additional_args: tuple[str, ...],
    selection: GGUFSourceSelection,
    claim: ServerLaunchClaim,
) -> str | None:
    return _run_gguf_server_worker(
        app_instance,
        "llamacpp",
        executable,
        host,
        port,
        additional_args,
        selection,
        claim,
    )


def _source_snapshot(
    window: "LLMManagementWindow",
    provider: str,
    legacy_input_id: str,
) -> GGUFSourceSelection:
    snapshot = getattr(window, "gguf_source_snapshot", None)
    if callable(snapshot):
        return snapshot(provider).validate_for(provider)
    model_path = window.query_one(f"#{legacy_input_id}", Input).value
    return initial_gguf_selection(provider, model_path).validate_for(provider)


async def handle_start_llamafile_server_button_pressed(
    window: "LLMManagementWindow", app: "TldwCli", event: Button.Pressed
) -> None:
    logger = getattr(app, "loguru_logger", _loguru_fallback_logger)
    logger.info("User requested to start Llamafile server.")

    try:
        exec_path_input = window.query_one("#llamafile-exec-path", Input)
        host_input = window.query_one("#llamafile-host", Input)
        port_input = window.query_one("#llamafile-port", Input)
        additional_args_input = window.query_one("#llamafile-additional-args", TextArea)
        log_output_widget = window.query_one("#llamafile-log-output", RichLog)

        exec_path = exec_path_input.value.strip()
        host = host_input.value.strip() or "127.0.0.1"
        port = port_input.value.strip() or "8000"
        additional_args_str = additional_args_input.text.strip()  # .text for TextArea
        selection = _source_snapshot(window, "llamafile", "llamafile-model-path")

        if not exec_path:
            app.notify("Llamafile executable path is required.", severity="error")
            exec_path_input.focus()
            return
        if not Path(exec_path).is_file():
            app.notify("Llamafile executable was not found.", severity="error")
            exec_path_input.focus()
            return
        additional_args = tuple(shlex.split(additional_args_str))
        try:
            _validate_gguf_additional_args(additional_args)
        except ValueError:
            app.notify(
                "Additional arguments cannot select another model source. "
                "Remove the model source option and try again.",
                severity="error",
            )
            return
        claim = reserve_server_launch(
            app,
            "llamafile",
            authority=selection.authority,
        )
        if claim is None:
            window._sync_process_controls("llamafile")
            app.notify(
                "Llamafile server is already starting or running.", severity="warning"
            )
            return
        window._sync_process_controls("llamafile")
        log_output_widget.clear()
        log_output_widget.write("Starting Llamafile server.\n")

        worker_callable = functools.partial(
            run_llamafile_server_worker,
            app,
            exec_path,
            host,
            port,
            additional_args,
            selection,
            claim,
        )

        app.run_worker(
            worker_callable,
            group="llamafile_server",
            description="Running Llamafile server process",
            exclusive=True,  # Typically one server instance
            thread=True,
            # NO 'args' or 'done' parameters
        )
        app.notify(
            f"Llamafile server starting… — endpoint will be "
            f"http://{host}:{port} once the chip shows 'running'."
        )
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
        host_input = window.query_one("#llamacpp-host", Input)
        port_input = window.query_one("#llamacpp-port", Input)
        additional_args_input = window.query_one("#llamacpp-additional-args", Input)
        log_output_widget = window.query_one("#llamacpp-log-output", RichLog)

        exec_path = exec_path_input.value.strip()
        host = host_input.value.strip() or "127.0.0.1"
        port = port_input.value.strip() or "8001"
        additional_args_str = additional_args_input.value.strip()
        selection = _source_snapshot(window, "llamacpp", "llamacpp-model-path")

        if not exec_path:
            app.notify("Executable path is required.", severity="error")
            exec_path_input.focus()
            return
        if not Path(exec_path).is_file():
            app.notify("Llama.cpp executable was not found.", severity="error")
            exec_path_input.focus()
            return
        additional_args = tuple(shlex.split(additional_args_str))
        try:
            _validate_gguf_additional_args(additional_args)
        except ValueError:
            app.notify(
                "Additional arguments cannot select another model source. "
                "Remove the model source option and try again.",
                severity="error",
            )
            return
        claim = reserve_server_launch(
            app,
            "llamacpp",
            authority=selection.authority,
        )
        if claim is None:
            window._sync_process_controls("llamacpp")
            app.notify(
                "Llama.cpp server is already starting or running.", severity="warning"
            )
            return
        window._sync_process_controls("llamacpp")
        log_output_widget.clear()
        log_output_widget.write("Starting Llama.cpp server.\n")

        worker_callable = functools.partial(
            run_llamacpp_server_worker,
            app,
            exec_path,
            host,
            port,
            additional_args,
            selection,
            claim,
        )

        app.run_worker(
            worker_callable,
            group="llamacpp_server",
            description="Running Llama.cpp server process",
            exclusive=True,
            thread=True,
        )

        app.notify(
            f"Llama.cpp server starting… — endpoint will be "
            f"http://{host}:{port} once the chip shows 'running'."
        )
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
