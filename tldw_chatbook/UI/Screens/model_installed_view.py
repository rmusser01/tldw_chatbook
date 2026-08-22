"""Lazy Installed view for managed and legacy local models."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import DescendantFocus
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Static

from tldw_chatbook.Model_Artifacts.gguf_admission import (
    GGUFBoundsError,
    GGUFParseError,
    GGUFPathError,
    GGUFVersionError,
)
from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseTimeoutError
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDiskUsage,
    ArtifactInUseError,
    ArtifactNotReadyError,
    ArtifactRemovalAvailability,
    ArtifactRef,
    ArtifactIntegrityError,
    ArtifactStateError,
    LocalGGUFImportProgress,
    LocalGGUFImportResult,
    ModelArtifactService,
    ReconcileReport,
)
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactRemovalPreview,
    AudioCppModelLibraryObservationSnapshot,
    build_audio_cpp_artifact_removal_preview,
    is_curated_audio_cpp_artifact_reference,
)
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.UI.Screens.model_browser_state import (
    InventoryRow,
    UnmanagedRow,
    format_mib,
    inventory_rows,
)
from tldw_chatbook.UI.Screens.model_curated_view import (
    AudioCppObservationProvider,
    AudioCppPackageProjection,
    ModelLibraryFocusLocator,
    audio_cpp_package_projection,
    clear_audio_cpp_observation,
    model_library_focus_locator,
    project_audio_cpp_observation,
    restore_model_library_focus,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
    ActivationRequested,
    DeletionRequested,
    ModelActivationControls,
)
from tldw_chatbook.Widgets.ModelArtifacts.install_progress import (
    ModelInstallProgress,
)
from tldw_chatbook.Widgets.ModelArtifacts.local_gguf_import import (
    LocalGGUFImportConsentModal,
    LocalGGUFImportControls,
    LocalGGUFImportRequested,
)
from tldw_chatbook.Widgets.delete_confirmation_dialog import (
    DeleteConfirmationDialog,
)
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress

MAX_UNMANAGED_MODELS = 500
_MODEL_EXTENSIONS = frozenset({".gguf", ".bin", ".safetensors", ".pt", ".pth", ".onnx"})
_MIN_LEGACY_MODEL_BYTES = 1024 * 1024
_DELETE_RECOVERY_TEXT = {
    "recycle-check": "Checking for an idle model to unload…",
    "recycle-retry": "Idle model unloaded; retrying deletion…",
}
_IMPORT_CANCELLED_TEXT = (
    "Import cancelled. The original file and prior models are unchanged."
)


def _is_gguf(path: Path) -> bool:
    """Return whether a picker candidate is a GGUF file."""
    return path.suffix.casefold() == ".gguf"


_GGUF_FILTERS = Filters(("GGUF models (*.gguf)", _is_gguf))


class _PrivateGGUFFileOpen(EnhancedFileOpen):
    """Use the real picker without persisting the selected user path."""

    def _get_last_directory(self) -> None:
        return None

    def dismiss(self, result: Path | list[Path] | None) -> None:
        Screen.dismiss(self, result)


def lifecycle_failure_message(exc: BaseException, *, operation: str) -> str:
    """Map lifecycle errors to stable text without leaking raw details.

    Args:
        exc: Error raised by activate, delete, or reconcile.
        operation: User action that failed.

    Returns:
        Sanitized user-visible failure text.
    """
    if isinstance(exc, ArtifactInUseError):
        return "This model is in use. Stop active work and retry deletion."
    if isinstance(exc, ArtifactNotReadyError):
        return "This model is not ready and cannot be activated."
    return f"Model {operation} failed. See the application log for details."


def reconcile_result_message(report: ReconcileReport) -> str:
    """Summarize every reconciliation outcome without exposing local paths.

    Args:
        report: Completed managed-store reconciliation report.

    Returns:
        Sanitized user-visible repair summary.
    """
    return (
        "Repair completed: "
        f"{report.readiness_created} readiness records restored · "
        f"{report.state_removed} stale state records removed · "
        f"{len(report.staging_entries)} staging entries observed · "
        f"{len(report.staging_removed)} staging entries removed · "
        f"{len(report.corrupt_artifacts)} corrupt models found."
    )


def local_import_failure_message(exc: BaseException) -> str:
    """Map a local-import failure to path-free recovery guidance.

    Args:
        exc: Failure raised by local admission or managed-store import.

    Returns:
        Stable user-facing guidance that excludes exception details and paths.
    """
    if isinstance(exc, GGUFPathError):
        return "The selected GGUF could not be read safely. Choose another file."
    if isinstance(exc, GGUFVersionError):
        return "This is not a supported GGUF file. Choose another file."
    if isinstance(exc, (GGUFBoundsError, GGUFParseError)):
        return "This is not a valid GGUF model. Choose another file."
    if isinstance(exc, ArtifactIntegrityError):
        return "The managed copy failed integrity verification. Retry or choose another file."
    if isinstance(exc, ArtifactLeaseTimeoutError):
        return "The managed model store is busy. Retry shortly."
    return "This GGUF could not be imported. Retry or choose another file."


class InstalledView(Widget):
    """List and manage the shared local model inventory."""

    BUNDLED_CSS = """
    InstalledView {
        height: 100%;
    }

    InstalledView .installed-header {
        height: 3;
    }

    InstalledView .installed-header Button {
        width: auto;
        margin-right: 1;
    }

    InstalledView .installed-list {
        height: 1fr;
    }

    InstalledView .installed-model-row {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $surface-lighten-1;
    }

    InstalledView .installed-model-row.-revealed {
        border: solid $accent;
        background: $accent 8%;
    }

    InstalledView .installed-model-title {
        text-style: bold;
    }

    InstalledView .installed-model-muted {
        color: $text-muted;
    }

    InstalledView .installed-import-actions {
        height: 3;
    }

    InstalledView .installed-import-actions Button {
        width: auto;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        service_factory: Callable[[], ModelArtifactService] = managed_service,
        legacy_dir: Path | None = None,
        on_root_activated: Callable[[ArtifactRef], None] | None = None,
        may_delete: Callable[[ArtifactRef], str | None] | None = None,
        recycle_idle: Callable[[ArtifactRef], bool] | None = None,
        can_start_import: Callable[[], bool] | None = None,
        on_import_lane_changed: Callable[[bool], None] | None = None,
        observation_provider: AudioCppObservationProvider | None = None,
        id: str | None = None,
    ) -> None:
        """Create an idle view; no filesystem work occurs here.

        Args:
            service_factory: Lazy managed-store service factory.
            legacy_dir: Legacy downloader directory to scan on activation.
            on_root_activated: Called after successful core activation.
            may_delete: Return a user-visible reason to block deletion.
            recycle_idle: Retire an idle worker that leases the exact artifact.
            can_start_import: Return whether the host store lane is free.
            on_import_lane_changed: Report ownership changes to the host.
            id: Optional Textual widget id.
        """
        self._service_factory = service_factory
        self._legacy_dir = legacy_dir or Path("~/Downloads/tldw_models").expanduser()
        self._on_root_activated = on_root_activated or (lambda _reference: None)
        self._may_delete = may_delete or (lambda _reference: None)
        self._recycle_idle = (
            recycle_idle if recycle_idle is not None else lambda _reference: False
        )
        self._can_start_import = can_start_import or (lambda: True)
        self._on_import_lane_changed = on_import_lane_changed or (lambda _active: None)
        self._observation_provider = observation_provider
        self._import_lane_owned = False
        self._service: ModelArtifactService | None = None
        self._rows: tuple[InventoryRow, ...] = ()
        self._usage: ArtifactDiskUsage | None = None
        self._loaded = False
        self._loading = False
        self._reload_after_load = False
        self._load_error: str | None = None
        self._audio_cpp_projections: dict[ArtifactRef, AudioCppPackageProjection] = {}
        self._lifecycle_status: str | None = None
        self._install_active = False
        self._install_progress: AcquisitionProgress | None = None
        self._operation_reference: ArtifactRef | None = None
        self._operation_name: str | None = None
        self._pending_delete_reference: ArtifactRef | None = None
        self._pending_removal_preview: AudioCppArtifactRemovalPreview | None = None
        self._import_generation = 0
        self._import_selecting = False
        self._import_active = False
        self._import_worker_generation: int | None = None
        self._import_thread_entered: threading.Event | None = None
        self._import_cancelable = False
        self._import_cancel_event: threading.Event | None = None
        self._import_progress: LocalGGUFImportProgress | None = None
        self._pending_import_path: Path | None = None
        self._import_status: str | None = None
        self._import_retry_available = False
        self._restore_header_focus_id: str | None = None
        self._revealed_reference: ArtifactRef | None = None
        self._reveal_status: str | None = None
        self._reveal_focus_attempts = 0
        self._observation_generation = 0
        self._observation_focus_locator: ModelLibraryFocusLocator | None = None
        # TASK-19563: monotonic inventory-read counter; see `_apply_inventory`.
        self._inventory_generation = 0
        super().__init__(id=id)

    def _non_import_lifecycle_pending(self) -> bool:
        """Return whether an incumbent non-import operation owns the store."""
        return (
            self._loading
            or self._install_active
            or self._operation_reference is not None
            or self._operation_name is not None
            or self._pending_delete_reference is not None
        )

    def _lifecycle_pending(self) -> bool:
        """Return whether any managed-store mutation currently owns the view."""
        return (
            self._non_import_lifecycle_pending()
            or self._import_selecting
            or self._import_active
        )

    def _set_import_lane_owned(self, active: bool) -> None:
        """Report one idempotent host-lane transition."""
        if active == self._import_lane_owned:
            return
        self._import_lane_owned = active
        self._on_import_lane_changed(active)

    def compose(self) -> ComposeResult:
        """Compose from retained in-memory state without performing I/O."""
        lifecycle_pending = self._lifecycle_pending()
        with Horizontal(classes="installed-header"):
            yield Button(
                "Refresh",
                id="installed-models-refresh",
                variant="primary",
                disabled=lifecycle_pending,
            )
            yield Button(
                "Repair",
                id="installed-models-repair",
                variant="default",
                disabled=lifecycle_pending,
            )
            yield Button(
                "Import GGUF…",
                id="installed-models-import-gguf",
                variant="default",
                disabled=lifecycle_pending,
            )
        progress = ModelInstallProgress(
            self._install_progress,
            id="installed-model-install-progress",
        )
        progress.display = self._install_active
        yield progress
        import_progress = ModelInstallProgress(
            self._import_progress,
            id="installed-gguf-import-progress",
        )
        import_progress.display = self._import_active
        yield import_progress
        if self._import_status is not None:
            yield Static(
                self._import_status,
                id="installed-gguf-import-status",
                markup=False,
            )
        if self._lifecycle_status is not None:
            yield Static(
                self._lifecycle_status,
                id="installed-lifecycle-status",
                classes="installed-recovery-status",
                markup=False,
            )
        if self._reveal_status is not None:
            yield Static(
                self._reveal_status,
                id="installed-reveal-status",
                classes="installed-recovery-status",
                markup=False,
            )
        if self._import_active:
            with Horizontal(classes="installed-import-actions"):
                yield Button(
                    "Cancel import",
                    id="installed-gguf-import-cancel",
                    disabled=not self._import_cancelable,
                )
        elif self._import_retry_available:
            with Horizontal(classes="installed-import-actions"):
                yield Button("Retry", id="installed-gguf-import-retry")
                yield Button(
                    "Choose another file",
                    id="installed-gguf-import-choose",
                )
        if self._loading:
            yield Static("Loading installed models…", markup=False)
        elif self._load_error:
            yield Static(self._load_error, markup=False)
        elif not self._loaded:
            yield Static(
                "Open Installed to load the local model inventory.", markup=False
            )
        else:
            yield self._summary()

        with VerticalScroll(classes="installed-list", can_focus=False):
            if self._loaded and not self._rows:
                yield Static(
                    "No managed or legacy models found. Use Import GGUF… for a "
                    "managed copy, or choose External GGUF under Llama.cpp or "
                    "Llamafile to use a file in place.",
                    markup=False,
                )
            for row in self._rows:
                yield self._row_widget(row)

    def _summary(self) -> Static:
        """Return the managed-store disk summary."""
        if self._usage is None:
            return Static("Disk usage unavailable.", markup=False)
        return Static(
            "Managed: "
            f"{format_mib(self._usage.installed_bytes)} installed, "
            f"{format_mib(self._usage.staging_bytes)} staging · "
            f"{format_mib(self._usage.free_bytes)} free",
            markup=False,
        )

    def set_install_state(
        self,
        progress: AcquisitionProgress | None,
        *,
        active: bool,
    ) -> None:
        """Mirror the current managed install into this persistent view.

        Args:
            progress: Latest worker progress, when one has been emitted.
            active: Whether provisioning is still running.
        """
        state_changed = active != self._install_active
        self._install_active = active
        if progress is not None:
            self._install_progress = progress
        if not active:
            self._install_progress = None
        if state_changed:
            self.refresh(recompose=True)
            return
        try:
            widget = self.query_one(
                "#installed-model-install-progress",
                ModelInstallProgress,
            )
        except NoMatches:
            self.refresh(recompose=True)
            return
        widget.display = active
        if progress is not None:
            widget.update_progress(progress)

    def _row_widget(self, row: InventoryRow) -> Vertical:
        """Build one inventory row from pure render state."""
        audio_cpp = (
            self._audio_cpp_projections.get(row.reference)
            if row.reference is not None
            else None
        )
        children: list[Widget] = [
            Static(row.model_label, classes="installed-model-title", markup=False),
        ]
        if audio_cpp is None:
            children.append(
                Static(row.provenance, classes="installed-model-muted", markup=False)
            )
        if row.reference is not None:
            children.append(
                Static(
                    f"Revision: {row.revision} · Precision: {row.precision}",
                    markup=False,
                )
            )
            if row.dependencies:
                children.append(
                    Static(
                        "Dependencies: "
                        + ", ".join(
                            f"{ref.artifact_id}@{ref.revision}/{ref.variant}"
                            for ref in row.dependencies
                        ),
                        markup=False,
                    )
                )
        if row.size_bytes is not None:
            children.append(Static(f"Size: {format_mib(row.size_bytes)}"))
        if audio_cpp is None:
            children.append(Static(row.action_hint, markup=False))
        else:
            children.extend(self._audio_cpp_facts(row, audio_cpp))
        if row.reference == self._operation_reference:
            recovery_text = _DELETE_RECOVERY_TEXT.get(self._operation_name or "")
            if recovery_text is not None:
                children.append(
                    Static(
                        recovery_text,
                        classes="installed-model-muted",
                        markup=False,
                    )
                )
        if row.reference is not None and not row.is_broken:
            controls = ModelActivationControls(
                row.reference,
                active=row.active,
                ready=row.ready,
                allow_activation=(
                    False if audio_cpp is not None else row.activation_allowed
                ),
                pending=self._lifecycle_pending(),
                disabled_reason=(
                    "Delete unavailable — another model package operation is in progress."
                    if audio_cpp is not None and self._lifecycle_pending()
                    else None
                ),
            )
            if audio_cpp is not None:
                controls.add_class("audio-cpp-actions")
            children.append(controls)
        elif row.is_unmanaged and row.path.suffix.casefold() == ".gguf":
            children.append(
                LocalGGUFImportControls(
                    row.path,
                    pending=self._lifecycle_pending(),
                )
            )
        classes = (
            "installed-model-row audio-cpp-model-row"
            if audio_cpp is not None
            else "installed-model-row"
        )
        if row.reference == self._revealed_reference:
            classes += " -revealed"
        widget = Vertical(*children, classes=classes)
        widget.reference = row.reference
        return widget

    @staticmethod
    def _audio_cpp_facts(
        row: InventoryRow,
        projection: AudioCppPackageProjection,
    ) -> tuple[Widget, ...]:
        """Render package truth without inferring configuration or runtime use."""

        companions = Collapsible(
            Static("\n".join(projection.companion_paths) or "None", markup=False),
            title=f"Companion files ({len(projection.companion_paths)})",
            classes="audio-cpp-companions",
            collapsed=True,
        )
        return (
            Static(f"Available: {projection.availability}", markup=False),
            Static("Installed package: Local record found", markup=False),
            Static(
                "Integrity: Unknown — package record needs Repair"
                if row.error is not None
                else "Integrity: Not checked this session",
                markup=False,
            ),
            Static(f"Recipe: {projection.recipe}", markup=False),
            Static(f"Compatibility: {projection.compatibility}", markup=False),
            Static(
                f"Configured: {projection.configured}",
                classes="audio-cpp-configured",
                markup=False,
            ),
            Static(
                f"Running: {projection.running}",
                classes="audio-cpp-running",
                markup=False,
            ),
            Static(f"Speech tasks: {projection.speech_tasks}", markup=False),
            Static(
                f"Required package files: {projection.required_files}", markup=False
            ),
            Static(f"Pinned source: {projection.pinned_source}", markup=False),
            Static(
                f"Manifest authority: {projection.manifest_authority}", markup=False
            ),
            Static(f"Package size: {projection.package_size}", markup=False),
            companions,
            Static(
                "Model package only — audiocpp_server is not included",
                classes="installed-model-muted audio-cpp-package-copy",
                markup=False,
            ),
        )

    @staticmethod
    def scan_unmanaged(
        root: Path,
        *,
        excluded_root: Path | None = None,
        limit: int = MAX_UNMANAGED_MODELS,
    ) -> tuple[UnmanagedRow, ...]:
        """Return a bounded scan of legacy model files.

        Args:
            root: Legacy downloader directory.
            excluded_root: Managed artifacts subtree to prune from the scan.
            limit: Maximum rows returned.

        Returns:
            Bounded legacy model rows in deterministic path order.
        """
        root = validate_path_simple(root, require_exists=False).resolve()
        excluded = (
            validate_path_simple(excluded_root, require_exists=False).resolve()
            if excluded_root is not None
            else None
        )
        if limit <= 0 or not root.is_dir():
            return ()
        rows: list[UnmanagedRow] = []
        for directory, directories, filenames in os.walk(root):
            current = Path(directory).resolve()
            if excluded is not None and (
                current == excluded or excluded in current.parents
            ):
                directories[:] = []
                continue
            if excluded is not None:
                directories[:] = [
                    name
                    for name in directories
                    if (current / name).resolve() != excluded
                    and excluded not in (current / name).resolve().parents
                ]
            directories.sort()
            filenames.sort()
            for filename in filenames:
                path = Path(directory) / filename
                if path.suffix.casefold() not in _MODEL_EXTENSIONS or path.is_symlink():
                    continue
                try:
                    size_bytes = path.stat().st_size
                except OSError:
                    continue
                if size_bytes <= _MIN_LEGACY_MODEL_BYTES:
                    continue
                rows.append(UnmanagedRow(path=path, size_bytes=size_bytes))
                if len(rows) >= limit:
                    return tuple(rows)
        return tuple(rows)

    def ensure_loaded(self, *, force: bool = False) -> None:
        """Start the inventory worker on first activation or explicit refresh.

        Args:
            force: Reload even when a prior inventory is retained.
        """
        if self._loading:
            if force:
                self._reload_after_load = True
            return
        if self._loaded and not force:
            return
        self._observation_generation += 1
        self._inventory_generation += 1
        self._loading = True
        self._load_error = None
        self.refresh(recompose=True)
        self._load_inventory(self._inventory_generation)

    def _service_for_worker(self) -> ModelArtifactService:
        """Create the managed service lazily on a worker thread."""
        if self._service is None:
            self._service = self._service_factory()
        return self._service

    @work(
        thread=True, group="installed_models_load", exclusive=True, exit_on_error=False
    )
    def _load_inventory(self, generation: int | None = None) -> None:
        """Read managed inventory, disk totals, and legacy files off-loop.

        TASK-19563: the generation captured at dispatch travels with the read.
        This is a *thread* worker, so `Worker.cancel()` cannot stop the body --
        it finishes in the executor and the `call_from_thread` callbacks below
        still land, potentially on a view the user has already left.
        """
        try:
            service = self._service_for_worker()
            installed = service.list_installed()
            usage = service.disk_usage()
            unmanaged = self.scan_unmanaged(
                self._legacy_dir,
                excluded_root=getattr(service, "artifacts_path", None),
            )
            rows = inventory_rows(installed, usage, unmanaged)
            audio_cpp = {
                item.descriptor.reference: projection
                for item in installed
                if item.descriptor is not None
                and (projection := audio_cpp_package_projection(item.descriptor))
                is not None
            }
        except Exception:
            logger.opt(exception=True).error(
                "Managed model inventory load failed; legacy_scan_configured={}",
                self._legacy_dir is not None,
            )
            self.app.call_from_thread(
                self._apply_inventory,
                (),
                None,
                "The local model inventory could not be loaded.",
                None,
                generation,
            )
            return
        self.app.call_from_thread(
            self._apply_inventory, rows, usage, None, audio_cpp, generation
        )

    def _apply_inventory(
        self,
        rows: tuple[InventoryRow, ...],
        usage: ArtifactDiskUsage | None,
        error: str | None,
        audio_cpp: dict[ArtifactRef, AudioCppPackageProjection] | None = None,
        generation: int | None = None,
    ) -> None:
        """Apply a completed inventory read on the Textual event loop.

        TASK-19563: this is the arrival end of a *thread* worker, which
        `Worker.cancel()` cannot stop -- the body finishes in the executor and
        this callback lands regardless. Two refusals therefore live here:

        * a read a newer `ensure_loaded()` has already superseded is dropped
          outright, so a slow first read can never overwrite a fast second one;
        * a read that arrives after the view has left the DOM records its state
          but drives no UI. Everything below the `is_attached` check recomposes,
          re-drives the observation pass, and restores focus -- work that lands
          on whatever screen happens to be current, not on this one.

        `is_attached` is the check that can actually be `False`; `is_mounted`
        is never reset once set (see `UI/Screens/library_screen.py`), which is
        why it is useless as a post-hop detach guard.
        """
        if generation is not None and generation != self._inventory_generation:
            return
        self._rows = rows
        self._usage = usage
        self._loading = False
        self._loaded = error is None
        self._load_error = error
        if audio_cpp is not None:
            self._audio_cpp_projections = audio_cpp
        reload_after_load = self._reload_after_load
        self._reload_after_load = False
        revealed = self._revealed_reference
        if error is None and revealed is not None and not reload_after_load:
            if any(row.reference == revealed for row in rows):
                self._reveal_status = None
            else:
                self._revealed_reference = None
                self._reveal_status = (
                    "That managed model is no longer available. Refresh Installed "
                    "models and try again."
                )
        if reload_after_load:
            self.ensure_loaded(force=True)
        elif self.is_attached:
            self.refresh(recompose=True)
            if error is None and self._observation_provider is not None:
                self.refresh_observations()
            if self._import_status is not None:
                self.call_after_refresh(self._focus_import_recovery)
            elif self._restore_header_focus_id is not None:
                focus_id = self._restore_header_focus_id
                self._restore_header_focus_id = None
                self.call_after_refresh(self.restore_focus, focus_id)
            elif self._revealed_reference is not None:
                self._schedule_revealed_focus()

    def reveal_reference(self, reference: ArtifactRef) -> None:
        """Reveal and focus one exact managed row without activating it.

        Args:
            reference: Verified managed identity selected by another Models view.
        """
        if type(reference) is not ArtifactRef:
            return
        had_recovery = self._reveal_status is not None
        self._reveal_status = None
        self._revealed_reference = reference
        if any(row.reference == reference for row in self._rows):
            if had_recovery and self.is_attached:
                self.refresh(recompose=True)
                self._schedule_revealed_focus()
                return
            for row in self.query(".installed-model-row"):
                row.set_class(
                    getattr(row, "reference", None) == reference,
                    "-revealed",
                )
            self._focus_revealed_reference()
            return
        self.ensure_loaded(force=True)

    def _schedule_revealed_focus(self) -> None:
        """Focus after recompose, retrying across Textual's child-mount gap."""
        self._reveal_focus_attempts = 3
        self.call_after_refresh(self._focus_revealed_after_recompose)

    def _focus_revealed_after_recompose(self) -> None:
        """Resolve one bounded post-recompose focus attempt."""
        if self._focus_revealed_reference():
            self._reveal_focus_attempts = 0
            return
        self._reveal_focus_attempts -= 1
        if self._reveal_focus_attempts > 0:
            self.set_timer(0.01, self._focus_revealed_after_recompose)

    def _focus_revealed_reference(self) -> bool:
        """Scroll the revealed row into view and focus its first safe action."""
        reference = self._revealed_reference
        if reference is None:
            return False
        row = next(
            (
                widget
                for widget in self.query(".installed-model-row")
                if getattr(widget, "reference", None) == reference
            ),
            None,
        )
        if row is None:
            return False
        row.scroll_visible(animate=False, immediate=True, force=True)
        action = next(
            (
                button
                for button in row.query(".model-activate, .model-delete").results(
                    Button
                )
                if not button.disabled
            ),
            None,
        )
        if action is not None:
            action.focus()
            return True
        return False

    def refresh_observations(self) -> None:
        """Refresh current exact refs without re-reading managed inventory."""

        if self._observation_provider is None:
            return
        references = tuple(self._audio_cpp_projections)
        if not references:
            return
        self._observation_generation += 1
        focused = self.app.focused
        locator = (
            self.focus_locator(focused)
            if focused is not None and self in focused.ancestors_with_self
            else None
        )
        if locator is not None:
            self._observation_focus_locator = locator
        else:
            locator = self._observation_focus_locator
        projections = {
            reference: clear_audio_cpp_observation(projection)
            for reference, projection in self._audio_cpp_projections.items()
        }
        if projections != self._audio_cpp_projections:
            self._audio_cpp_projections = projections
            self._update_audio_cpp_observation_facts()
        self.refresh()
        self.call_after_refresh(
            self._start_audio_cpp_observation,
            self._observation_generation,
            references,
            locator,
        )

    def _start_audio_cpp_observation(
        self,
        generation: int,
        references: tuple[ArtifactRef, ...],
        locator: ModelLibraryFocusLocator | None,
    ) -> None:
        """Start only the still-current deferred observation generation."""

        if (
            generation != self._observation_generation
            or tuple(self._audio_cpp_projections) != references
        ):
            return
        self._observation_focus_locator = None
        self._observe_audio_cpp_rows(generation, references, locator)

    @work(group="installed_audio_cpp_observation", exclusive=True, exit_on_error=False)
    async def _observe_audio_cpp_rows(
        self,
        generation: int,
        references: tuple[ArtifactRef, ...],
        locator: ModelLibraryFocusLocator | None,
    ) -> None:
        """Apply one generation-gated bulk Settings/runtime observation."""

        provider = self._observation_provider
        if provider is None:
            return
        try:
            snapshot = await provider(references)
        except Exception:
            return
        if (
            type(snapshot) is not AudioCppModelLibraryObservationSnapshot
            or generation != self._observation_generation
            or not self.is_attached
            or tuple(self._audio_cpp_projections) != references
        ):
            return
        evidence_by_ref = {item.reference: item for item in snapshot.observations}
        projections = {
            reference: project_audio_cpp_observation(
                projection,
                reference,
                evidence_by_ref[reference],
            )
            if reference in evidence_by_ref
            else projection
            for reference, projection in self._audio_cpp_projections.items()
        }
        if projections == self._audio_cpp_projections:
            return
        focused = self.app.focused
        if focused is not None and self in focused.ancestors_with_self:
            locator = self.focus_locator(focused) or locator
        self._audio_cpp_projections = projections
        self._update_audio_cpp_observation_facts()
        self.refresh()
        if locator is not None:
            self.restore_focus(locator)

    def _update_audio_cpp_observation_facts(self) -> None:
        """Update the two observed facts without replacing row actions."""

        for widget in self.query(".audio-cpp-model-row"):
            projection = self._audio_cpp_projections.get(
                getattr(widget, "reference", None)
            )
            if projection is None:
                continue
            try:
                widget.query_one(".audio-cpp-configured", Static).update(
                    f"Configured: {projection.configured}"
                )
                widget.query_one(".audio-cpp-running", Static).update(
                    f"Running: {projection.running}"
                )
            except NoMatches:
                continue

    def _focus_import_recovery(self) -> None:
        """Restore focus to one stable import control after recomposition."""
        for selector in (
            "#installed-gguf-import-retry",
            "#installed-models-import-gguf",
        ):
            try:
                control = self.query_one(selector, Button)
            except NoMatches:
                continue
            if not control.disabled:
                self.screen.set_focus(control)
                return

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Keep keyboard-selected disclosures and actions inside the viewport."""

        event.widget.scroll_visible(animate=False, immediate=True, force=True)

    def focus_locator(self, widget: Widget) -> ModelLibraryFocusLocator | None:
        """Return a stable exact-row locator for focus across pane changes."""

        return model_library_focus_locator(
            self,
            widget,
            row_selector=".installed-model-row",
            action_class="model-delete",
            action_role="delete",
        )

    def restore_focus(self, locator: ModelLibraryFocusLocator) -> None:
        """Restore an id-based header or exact-ref row control."""

        restore_model_library_focus(
            self,
            locator,
            row_selector=".installed-model-row",
            action_role="delete",
            action_selector=".model-delete",
        )

    @on(Button.Pressed, "#installed-models-import-gguf")
    def _header_import_pressed(self) -> None:
        """Open the local GGUF picker from the persistent header action."""
        self._open_import_picker()

    @on(LocalGGUFImportRequested)
    def _row_import_requested(self, event: LocalGGUFImportRequested) -> None:
        """Open the same picker from an outside-GGUF inventory row."""
        event.stop()
        self._open_import_picker(event.path)

    def _open_import_picker(self, suggested_path: Path | None = None) -> None:
        """Open one GGUF-only picker and give it ownership of a generation."""
        if self._lifecycle_pending():
            return
        if not self._can_start_import():
            self.notify(
                "Another model operation is already running.",
                severity="information",
            )
            return
        self._set_import_lane_owned(True)
        if self._import_cancel_event is not None:
            self._import_cancel_event.set()
        self._import_generation += 1
        generation = self._import_generation
        self._import_selecting = True
        self._pending_import_path = None
        self._import_retry_available = False
        self.refresh(recompose=True)
        location = (
            suggested_path.parent if suggested_path is not None else self._legacy_dir
        )
        self.app.push_screen(
            _PrivateGGUFFileOpen(
                location=location,
                title="Choose a GGUF model",
                filters=_GGUF_FILTERS,
                must_exist=True,
                context="managed_gguf_import_private",
                select_button="Choose",
            ),
            lambda result: self._import_path_picked(generation, result),
        )

    def _import_path_picked(
        self,
        generation: int,
        result: Path | list[Path] | None,
    ) -> None:
        """Ask for managed-copy consent after one current GGUF selection."""
        if (
            not self.is_attached
            or generation != self._import_generation
            or not self._import_selecting
        ):
            return
        if not isinstance(result, Path):
            self._release_import_selection()
            return
        if result.suffix.casefold() != ".gguf":
            self._release_import_selection()
            self.notify("Choose a GGUF model file.", severity="warning")
            return
        try:
            size_bytes = result.stat().st_size
        except OSError:
            self._release_import_selection()
            self.notify(
                "The selected GGUF could not be read safely.",
                severity="error",
            )
            return
        self._pending_import_path = result
        self.app.push_screen(
            LocalGGUFImportConsentModal(result, size_bytes),
            lambda confirmed: self._import_consent_decided(generation, confirmed),
        )

    def _import_consent_decided(self, generation: int, confirmed: bool) -> None:
        """Begin managed I/O only after explicit consent for the current path."""
        if (
            not self.is_attached
            or generation != self._import_generation
            or not self._import_selecting
        ):
            return
        source = self._pending_import_path
        if not confirmed or source is None:
            self._release_import_selection()
            return
        self._begin_import(source, generation=generation)

    def _release_import_selection(self) -> None:
        """Release transient picker/consent ownership without exposing its path."""
        self._import_selecting = False
        self._pending_import_path = None
        self._import_retry_available = False
        self._set_import_lane_owned(False)
        if self.is_attached:
            self.refresh(recompose=True)

    def _begin_import(self, source: Path, *, generation: int | None = None) -> None:
        """Start one cancellable local import from retained transient state."""
        app = self.app
        if self._non_import_lifecycle_pending():
            if generation == self._import_generation and self._import_selecting:
                self._release_import_selection()
            return
        if generation is not None:
            if generation != self._import_generation or not self._import_selecting:
                return
            self._import_selecting = False
        elif self._import_selecting:
            return
        if self._import_active:
            if self._import_cancel_event is not None:
                self._import_cancel_event.set()
            self._import_generation += 1
            generation = self._import_generation
        elif generation is None:
            self._import_generation += 1
            generation = self._import_generation
        elif generation != self._import_generation:
            return
        if not self._import_lane_owned:
            if not self._can_start_import():
                self.notify(
                    "Another model operation is already running.",
                    severity="information",
                )
                return
            self._set_import_lane_owned(True)
        self._pending_import_path = source
        self._import_cancel_event = threading.Event()
        self._import_thread_entered = threading.Event()
        self._import_active = True
        self._import_worker_generation = generation
        self._import_cancelable = True
        self._import_progress = None
        self._import_status = "Importing GGUF…"
        self._import_retry_available = False
        self.refresh(recompose=True)
        self._import_local_gguf(
            generation,
            source,
            self._import_cancel_event,
            self._import_thread_entered,
            app,
        )

    @work(
        thread=True,
        group="installed_gguf_import",
        exclusive=True,
        exit_on_error=False,
        description="Importing local GGUF model",
    )
    def _import_local_gguf(
        self,
        generation: int,
        source: Path,
        cancel_event: threading.Event,
        thread_entered: threading.Event,
        app: App,
    ) -> None:
        """Import and activate one exact managed reference off the UI loop."""
        thread_entered.set()
        try:
            if cancel_event.is_set():
                app.call_from_thread(
                    self._apply_import_failure,
                    generation,
                    _IMPORT_CANCELLED_TEXT,
                )
                return
            try:
                service = self._service_for_worker()
                result = service.import_local_gguf(
                    source,
                    cancelled=cancel_event.is_set,
                    progress=lambda progress: app.call_from_thread(
                        self._apply_import_progress,
                        generation,
                        progress,
                    ),
                )
            except Exception as exc:
                logger.error(
                    "Local GGUF import failed; phase=import; error_type={}",
                    type(exc).__name__,
                )
                cancelled = cancel_event.is_set() and isinstance(
                    exc, ArtifactStateError
                )
                app.call_from_thread(
                    self._apply_import_failure,
                    generation,
                    (
                        _IMPORT_CANCELLED_TEXT
                        if cancelled
                        else local_import_failure_message(exc)
                    ),
                )
                return
            if not app.call_from_thread(
                self._apply_import_finalizing,
                generation,
            ):
                return
            try:
                service.activate(result.reference)
            except Exception as exc:
                logger.error(
                    "Local GGUF import failed; phase=activation; error_type={}",
                    type(exc).__name__,
                )
                app.call_from_thread(
                    self._apply_import_activation_required,
                    generation,
                    result,
                )
                return
            app.call_from_thread(self._apply_import_success, generation, result)
        finally:
            app.call_from_thread(self._import_worker_stopped, generation)

    def _owns_import(self, generation: int) -> bool:
        """Return whether a callback still owns the mounted import lane."""
        return self.is_attached and generation == self._import_generation

    def _apply_import_progress(
        self,
        generation: int,
        progress: LocalGGUFImportProgress,
    ) -> None:
        """Paint one current progress event without replacing controls."""
        if not self._owns_import(generation) or not self._import_active:
            return
        self._import_progress = progress
        if progress.phase == "finalize":
            self._import_cancelable = False
            try:
                self.query_one("#installed-gguf-import-cancel", Button).disabled = True
            except NoMatches:
                pass
        try:
            self.query_one(
                "#installed-gguf-import-progress",
                ModelInstallProgress,
            ).update_progress(progress)
        except NoMatches:
            pass

    def _apply_import_finalizing(self, generation: int) -> bool:
        """Synchronously close cancellation before activation can begin."""
        if not self._owns_import(generation) or not self._import_active:
            return False
        self._import_cancelable = False
        self._import_status = "Finalizing managed model…"
        try:
            self.query_one("#installed-gguf-import-cancel", Button).disabled = True
        except NoMatches:
            pass
        try:
            self.query_one("#installed-gguf-import-status", Static).update(
                self._import_status
            )
        except NoMatches:
            pass
        return True

    def _import_worker_stopped(self, generation: int) -> None:
        """Release detached ownership only after its real worker has stopped."""
        if generation != self._import_worker_generation or self.is_attached:
            return
        self._import_worker_generation = None
        self._import_thread_entered = None
        self._import_active = False
        self._import_cancelable = False
        self._set_import_lane_owned(False)

    def _apply_import_success(
        self,
        generation: int,
        result: LocalGGUFImportResult,
    ) -> None:
        """Settle a successful import and reload the converged inventory."""
        if not self._owns_import(generation):
            return
        self._import_active = False
        self._import_worker_generation = None
        self._import_thread_entered = None
        self._import_cancelable = False
        self._import_progress = None
        self._pending_import_path = None
        self._import_retry_available = False
        self._import_status = (
            "Already imported and ready"
            if result.already_installed
            else "Imported and ready"
        )
        self._set_import_lane_owned(False)
        self.ensure_loaded(force=True)

    def _apply_import_activation_required(
        self,
        generation: int,
        _result: LocalGGUFImportResult,
    ) -> None:
        """Retain a promoted artifact whose automatic activation failed."""
        if not self._owns_import(generation):
            return
        self._import_active = False
        self._import_worker_generation = None
        self._import_thread_entered = None
        self._import_cancelable = False
        self._import_progress = None
        self._pending_import_path = None
        self._import_retry_available = False
        self._import_status = "Installed — activation required"
        self._set_import_lane_owned(False)
        self.ensure_loaded(force=True)

    def _apply_import_failure(self, generation: int, message: str) -> None:
        """Offer path-private recovery while retaining the source in memory."""
        if not self._owns_import(generation):
            return
        self._import_active = False
        self._import_worker_generation = None
        self._import_thread_entered = None
        self._import_cancelable = False
        self._import_progress = None
        self._import_status = message
        self._import_retry_available = self._pending_import_path is not None
        self._set_import_lane_owned(False)
        self.notify(message, severity="warning")
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_import_recovery)

    @on(Button.Pressed, "#installed-gguf-import-cancel")
    def _cancel_import_pressed(self) -> None:
        """Signal physical cancellation while the copy remains cancellable."""
        if (
            not self._import_active
            or not self._import_cancelable
            or self._import_cancel_event is None
        ):
            return
        self._import_cancel_event.set()
        self._import_cancelable = False
        self._import_status = "Cancelling import…"
        try:
            self.query_one("#installed-gguf-import-cancel", Button).disabled = True
        except NoMatches:
            pass
        try:
            self.query_one("#installed-gguf-import-status", Static).update(
                self._import_status
            )
        except NoMatches:
            pass

    @on(Button.Pressed, "#installed-gguf-import-retry")
    def _retry_import_pressed(self) -> None:
        """Retry the transient source without persisting or rendering its path."""
        source = self._pending_import_path
        if source is not None and not self._import_active:
            self._begin_import(source)

    @on(Button.Pressed, "#installed-gguf-import-choose")
    def _choose_import_pressed(self) -> None:
        """Forget the transient source before reopening the picker."""
        self._pending_import_path = None
        self._open_import_picker()

    def on_unmount(self) -> None:
        """Cancel and invalidate import callbacks before detaching."""
        if self._import_cancel_event is not None:
            self._import_cancel_event.set()
        queued_import = (
            self._import_active
            and self._import_thread_entered is not None
            and not self._import_thread_entered.is_set()
        )
        self._import_selecting = False
        self._pending_import_path = None
        self._import_generation += 1
        if queued_import:
            self._import_active = False
            self._import_worker_generation = None
            self._import_thread_entered = None
            self._import_cancelable = False
        if not self._import_active:
            self._set_import_lane_owned(False)

    @on(Button.Pressed, "#installed-models-refresh")
    def _refresh_pressed(self) -> None:
        if self._lifecycle_pending():
            return
        self._restore_header_focus_id = "installed-models-refresh"
        self.ensure_loaded(force=True)

    @on(Button.Pressed, "#installed-models-repair")
    def _repair_pressed(self) -> None:
        if self._lifecycle_pending():
            return
        self._restore_header_focus_id = "installed-models-repair"
        self._operation_name = "repair"
        self.refresh(recompose=True)
        self._repair_store()

    @on(ActivationRequested)
    def _activation_requested(self, event: ActivationRequested) -> None:
        event.stop()
        self._request_activation(event.reference)

    def _request_activation(self, reference: ArtifactRef) -> None:
        """Start activation unless another lifecycle operation is pending."""
        if self._lifecycle_pending():
            return
        self._operation_reference = reference
        self._operation_name = "activate"
        self.refresh(recompose=True)
        self._activate_model(reference)

    @on(DeletionRequested)
    def _deletion_requested(self, event: DeletionRequested) -> None:
        event.stop()
        if self._lifecycle_pending():
            return
        self._lifecycle_status = None
        blocked = self._may_delete(event.reference)
        if blocked is not None:
            self.notify(blocked, severity="warning")
            return
        self._pending_delete_reference = event.reference
        self.refresh(recompose=True)
        if is_curated_audio_cpp_artifact_reference(event.reference):
            self._operation_reference = event.reference
            self._operation_name = "review-removal"
            self._review_audio_cpp_deletion(event.reference)
            return
        self._show_delete_confirmation(event.reference)

    def _show_delete_confirmation(
        self,
        reference: ArtifactRef,
        preview: AudioCppArtifactRemovalPreview | None = None,
    ) -> None:
        warning = "The managed model files will be removed from this device."
        if preview is not None:
            impacts = (
                len(preview.settings_labels)
                + len(preview.profile_labels)
                + preview.assignment_count
                + preview.clone_reference_count
            )
            if impacts:
                warning = (
                    f"{len(preview.settings_labels)} Settings consumer(s), "
                    f"{len(preview.profile_labels)} profile(s), "
                    f"{preview.assignment_count} assignment(s), and "
                    f"{preview.clone_reference_count} private clone reference(s) "
                    "will remain unchanged and become unavailable."
                )
        dialog = DeleteConfirmationDialog(
            item_type="Model",
            item_name=f"{reference.artifact_id} ({reference.variant})",
            additional_warning=warning,
            permanent=True,
        )
        if preview is not None and (
            preview.settings_labels
            or preview.profile_labels
            or preview.assignment_count
            or preview.clone_reference_count
        ):
            dialog.confirm_label = "Remove package; keep consumers unavailable"
        self.app.push_screen(
            dialog,
            self._confirm_deletion,
        )

    async def _collect_audio_cpp_removal_preview(
        self,
        reference: ArtifactRef,
        *,
        include_probe: bool,
    ) -> AudioCppArtifactRemovalPreview:
        collector = getattr(self.app, "_audio_cpp_artifact_removal_evidence", None)
        if not callable(collector):
            raise ArtifactStateError("audio.cpp dependency review is unavailable")
        evidence = await collector(reference)
        generic_blocked = False
        if include_probe:
            coordinator = self.app._ensure_audio_cpp_artifact_lease_coordinator()
            availability = await coordinator.probe_removal_availability(reference)
            generic_blocked = availability is ArtifactRemovalAvailability.BUSY
        return build_audio_cpp_artifact_removal_preview(
            evidence,
            generic_lease_blocked=generic_blocked,
        )

    @work(group="installed_models_lifecycle", exclusive=True, exit_on_error=False)
    async def _review_audio_cpp_deletion(self, reference: ArtifactRef) -> None:
        """Build the pre-confirmation dependency review without blocking paint."""

        try:
            preview = await self._collect_audio_cpp_removal_preview(
                reference,
                include_probe=True,
            )
        except Exception:
            self._operation_reference = None
            self._operation_name = None
            self._pending_delete_reference = None
            self.notify(
                "Package dependencies could not be reviewed. Retry removal.",
                severity="error",
            )
            self._lifecycle_status = (
                "Dependency review failed — removal was not attempted. Retry removal."
            )
            self.refresh(recompose=True)
            self.call_after_refresh(self._focus_delete, reference)
            return
        if preview.staged_or_live or preview.generic_lease_blocked:
            self._operation_reference = None
            self._operation_name = None
            self._pending_delete_reference = None
            self.notify(
                (
                    "Another operation is using this package. "
                    "Stop or discard active work, then review removal again."
                ),
                severity="warning",
            )
            self._lifecycle_status = (
                "Package in use — removal blocked. Shut down or discard active work, "
                "then review removal again."
            )
            self.refresh(recompose=True)
            self.call_after_refresh(self._focus_delete, reference)
            return
        self._operation_reference = None
        self._operation_name = None
        self._pending_removal_preview = preview
        self._show_delete_confirmation(reference, preview)

    def _focus_delete(self, reference: ArtifactRef) -> None:
        """Restore focus to the exact package action after recovery paint."""

        for controls in self.query(ModelActivationControls):
            if controls.reference != reference:
                continue
            try:
                button = controls.query_one(".model-delete", Button)
            except NoMatches:
                return
            button.focus()
            button.scroll_visible(animate=False, immediate=True, force=True)
            return

    def _confirm_deletion(self, confirmed: bool) -> None:
        """Start deletion only after the confirmation dialog accepts it."""
        reference = self._pending_delete_reference
        self._pending_delete_reference = None
        if not confirmed or reference is None:
            self._pending_removal_preview = None
            self.refresh(recompose=True)
            return
        blocked = self._may_delete(reference)
        if blocked is not None:
            self.notify(blocked, severity="warning")
            self.refresh(recompose=True)
            return
        if (
            self._import_active
            or self._operation_reference is not None
            or self._operation_name is not None
        ):
            self.refresh(recompose=True)
            return
        self._operation_reference = reference
        self._operation_name = "delete"
        self.refresh(recompose=True)
        preview = self._pending_removal_preview
        self._pending_removal_preview = None
        if preview is None:
            self._delete_model(reference)
        else:
            self._delete_audio_cpp_model(reference, preview.fingerprint)

    @work(group="installed_models_lifecycle", exclusive=True, exit_on_error=False)
    async def _delete_audio_cpp_model(
        self,
        reference: ArtifactRef,
        fingerprint: str,
    ) -> None:
        """Own authority, drift revalidation, commit, and cleanup in one worker."""

        try:
            coordinator = self.app._ensure_audio_cpp_artifact_lease_coordinator()

            async def collect_fingerprint() -> str:
                current = await self._collect_audio_cpp_removal_preview(
                    reference,
                    include_probe=False,
                )
                return current.fingerprint

            outcome = await coordinator.remove_if_unchanged(
                reference,
                fingerprint,
                collect_fingerprint,
            )
            if outcome == "changed":
                self._apply_lifecycle_result(
                    "delete",
                    "Review changed dependencies. Open removal preview again.",
                )
                return
            self._apply_lifecycle_result("delete", None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._finish_delete_failure_on_loop(reference, exc)

    @work(
        thread=True,
        group="installed_models_lifecycle",
        exclusive=True,
        exit_on_error=False,
    )
    def _activate_model(self, reference: ArtifactRef) -> None:
        """Activate one exact verified model off the event loop."""
        try:
            activated = self._service_for_worker().activate(reference)
            self._on_root_activated(activated)
        except Exception as exc:
            logger.opt(exception=True).error(
                "Managed model activation failed for {}@{}/{}",
                reference.artifact_id,
                reference.revision,
                reference.variant,
            )
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                "activate",
                lifecycle_failure_message(exc, operation="activation"),
            )
            return
        self.app.call_from_thread(self._apply_lifecycle_result, "activate", None)

    @work(
        thread=True,
        group="installed_models_lifecycle",
        exclusive=True,
        exit_on_error=False,
    )
    def _delete_model(self, reference: ArtifactRef) -> None:
        """Delete one exact model, retrying once after proven idle recycle."""
        try:
            service = self._service_for_worker()
            service.delete(reference)
        except ArtifactInUseError as exc:
            if not self.app.call_from_thread(
                self._set_delete_phase,
                reference,
                "recycle-check",
            ):
                return
            try:
                recycled = self._recycle_idle(reference)
            except Exception as recycle_exc:
                logger.error(
                    "Managed model idle recycle failed; error_type={}",
                    type(recycle_exc).__name__,
                )
                self.app.call_from_thread(
                    self._apply_lifecycle_result,
                    "delete",
                    "Model deletion failed. See the application log for details.",
                )
                return
            if not recycled:
                self._finish_delete_failure(reference, exc)
                return
            if not self.app.call_from_thread(self._prepare_delete_retry, reference):
                return
            try:
                service.delete(reference)
            except Exception as retry_exc:
                self._finish_delete_failure(reference, retry_exc)
                return
        except Exception as exc:
            self._finish_delete_failure(reference, exc)
            return
        self.app.call_from_thread(self._apply_lifecycle_result, "delete", None)

    def _finish_delete_failure(
        self,
        reference: ArtifactRef,
        exc: BaseException,
    ) -> None:
        """Report one thread-worker delete failure through the UI-loop seam."""

        message = self._bounded_delete_failure(reference, exc)
        self.app.call_from_thread(
            self._apply_lifecycle_result,
            "delete",
            message,
        )

    def _finish_delete_failure_on_loop(
        self,
        reference: ArtifactRef,
        exc: BaseException,
    ) -> None:
        """Report one async-worker delete failure already on the app loop."""

        message = self._bounded_delete_failure(reference, exc)
        self._apply_lifecycle_result("delete", message)

    @staticmethod
    def _bounded_delete_failure(
        reference: ArtifactRef,
        exc: BaseException,
    ) -> str:
        """Log bounded fields only and return the bounded recovery copy."""

        if isinstance(exc, ArtifactInUseError):
            logger.warning("Managed model deletion blocked by a lease")
        else:
            logger.error(
                "Managed model deletion failed "
                "(error_type={}, code=operation_failed)",
                type(exc).__name__,
            )
        return lifecycle_failure_message(exc, operation="deletion")

    def _set_delete_phase(
        self,
        reference: ArtifactRef,
        phase: str,
    ) -> bool:
        """Paint one current delete-recovery phase on the event loop."""
        if self._operation_reference != reference or self._operation_name not in {
            "delete",
            "recycle-check",
            "recycle-retry",
        }:
            return False
        self._operation_name = phase
        self.refresh(recompose=True)
        return True

    def _prepare_delete_retry(self, reference: ArtifactRef) -> bool:
        """Recheck source policy and authorize one deletion retry."""
        if (
            self._operation_reference != reference
            or self._operation_name != "recycle-check"
        ):
            return False
        blocked = self._may_delete(reference)
        if blocked is not None:
            self._operation_reference = None
            self._operation_name = None
            self.notify(blocked, severity="warning")
            self.ensure_loaded(force=True)
            return False
        return self._set_delete_phase(reference, "recycle-retry")

    @work(
        thread=True,
        group="installed_models_lifecycle",
        exclusive=True,
        exit_on_error=False,
    )
    def _repair_store(self) -> None:
        """Run explicit reconciliation off the event loop."""
        try:
            report = self._service_for_worker().reconcile()
        except Exception as exc:
            logger.opt(exception=True).error(
                "Managed model repair failed; store={}",
                "shared",
            )
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                "repair",
                lifecycle_failure_message(exc, operation="repair"),
            )
            return
        self.app.call_from_thread(
            self._apply_lifecycle_result,
            "repair",
            None,
            reconcile_result_message(report),
        )

    def _apply_lifecycle_result(
        self,
        operation: str,
        error: str | None,
        success_message: str | None = None,
    ) -> None:
        """Complete a lifecycle operation and refresh inventory."""
        self._operation_reference = None
        self._operation_name = None
        self._lifecycle_status = None
        if error is not None:
            self.notify(error, severity="error")
        else:
            self.notify(
                success_message or f"Model {operation} completed.",
                severity="information",
            )
        self.ensure_loaded(force=True)
