"""Managed Parakeet completion, fenced to its original Import selection.

Artifact activation and persistent preference keep their consented lifecycle;
only automatic batch selection belongs to the form that requested installation.
"""

from pathlib import Path
from typing import Any

from loguru import logger
from textual.css.query import NoMatches, QueryError
from textual.widgets import Static

from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey
from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

logger = logger.bind(module="LibraryScreen")


def capture_selection(screen: Any) -> tuple[Any, ...]:
    """Capture form ownership, accepted edits and not-yet-delivered controls."""
    form = screen._ingest_state.form
    names = (
        "transcription_provider",
        "transcription_precision",
        "transcription_model_dir",
    )
    live = []
    for selector in (
        "library-ingest-path",
        *(f"opt-audio_video-{name}" for name in names),
    ):
        try:
            live.append(screen.query_one(f"#{selector}").value)
        except (NoMatches, QueryError):
            live.append(None)
    options = form.type_options.get("audio_video", {})
    return (
        form,
        getattr(screen, "_library_external_submit_generation", 0),
        form.path,
        tuple(options.get(name) for name in names),
        tuple(live),
    )


def apply_parakeet_install_result(
    self: Any, installed: Path | None, error: str | None
) -> None:
    """Publish installation and select only the unchanged requesting form."""
    pending = self._parakeet_v2_pending_selection
    current = capture_selection(self)
    select_current = (
        pending is not None and pending[0] is current[0] and pending[1:] == current[1:]
    )
    self._parakeet_v2_pending_selection = None
    self._parakeet_v2_install_worker = None
    self._parakeet_v2_pending_report = None
    self._parakeet_v2_install_progress = None
    self._library_model_install_progress_label = ""
    self._library_model_install_progress_owner = None
    try:
        self.query_one("#library-model-install-progress-label", Static).display = False
        self.query_one(
            "#library-model-install-progress", ModelInstallProgress
        ).display = False
    except NoMatches:
        pass
    if error is not None or installed is None:
        self.app_instance.notify(
            f"Parakeet v2 install failed: {error or 'unknown error'}",
            severity="error",
            markup=False,
        )
        return
    try:
        self.app_instance._ensure_parakeet_source_service().prefer_managed(
            ParakeetSourceKey.V2_INT8
        )
    except Exception as exc:
        logger.warning(
            "Could not prefer the installed Library Parakeet source "
            "(error_type={}).",
            type(exc).__name__,
        )
        self.app_instance.notify(
            "Parakeet v2 installed, but its source preference could not be updated.",
            severity="error",
        )
        return
    if not select_current:
        self.app_instance.notify(
            "Parakeet v2 INT8 installed; current batch choices retained.",
            severity="information",
        )
        return
    options = self._ingest_state.form.type_options.setdefault("audio_video", {})
    options["transcription_provider"] = "parakeet-onnx"
    options["transcription_precision"] = "int8"
    options.pop("transcription_model_dir", None)
    self.app_instance.notify(
        "Parakeet v2 INT8 installed and selected for this batch.",
        severity="information",
    )
    self.refresh(recompose=True)
