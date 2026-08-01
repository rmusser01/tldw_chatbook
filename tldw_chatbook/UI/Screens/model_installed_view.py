"""Lazy Installed view for managed and legacy local models."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDiskUsage,
    ArtifactInUseError,
    ArtifactNotReadyError,
    ArtifactRef,
    ModelArtifactService,
    ReconcileReport,
)
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.UI.Screens.model_browser_state import (
    InventoryRow,
    UnmanagedRow,
    inventory_rows,
)
from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
    ActivationRequested,
    DeletionRequested,
    ModelActivationControls,
)
from tldw_chatbook.Widgets.ModelArtifacts.install_progress import (
    ModelInstallProgress,
)
from tldw_chatbook.Widgets.delete_confirmation_dialog import (
    DeleteConfirmationDialog,
)

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress

MAX_UNMANAGED_MODELS = 500
_MODEL_EXTENSIONS = frozenset({".gguf", ".bin", ".safetensors", ".pt", ".pth", ".onnx"})
_MIN_LEGACY_MODEL_BYTES = 1024 * 1024


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
        f"{len(report.staging_removed)} staging entries removed · "
        f"{len(report.corrupt_artifacts)} corrupt models found."
    )


class InstalledView(Widget):
    """List and manage the shared local model inventory."""

    DEFAULT_CSS = """
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

    InstalledView .installed-model-title {
        text-style: bold;
    }

    InstalledView .installed-model-muted {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        service_factory: Callable[[], ModelArtifactService] = managed_service,
        legacy_dir: Path | None = None,
        id: str | None = None,
    ) -> None:
        """Create an idle view; no filesystem work occurs here.

        Args:
            service_factory: Lazy managed-store service factory.
            legacy_dir: Legacy downloader directory to scan on activation.
            id: Optional Textual widget id.
        """
        self._service_factory = service_factory
        self._legacy_dir = legacy_dir or Path("~/Downloads/tldw_models").expanduser()
        self._service: ModelArtifactService | None = None
        self._rows: tuple[InventoryRow, ...] = ()
        self._usage: ArtifactDiskUsage | None = None
        self._loaded = False
        self._loading = False
        self._reload_after_load = False
        self._load_error: str | None = None
        self._install_active = False
        self._install_progress: AcquisitionProgress | None = None
        self._operation_reference: ArtifactRef | None = None
        self._operation_name: str | None = None
        self._pending_delete_reference: ArtifactRef | None = None
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        """Compose from retained in-memory state without performing I/O."""
        lifecycle_pending = (
            self._loading
            or self._operation_reference is not None
            or self._operation_name is not None
            or self._pending_delete_reference is not None
        )
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
        progress = ModelInstallProgress(
            self._install_progress,
            id="installed-model-install-progress",
        )
        progress.display = self._install_active
        yield progress
        if self._loading:
            yield Static("Loading installed models…", markup=False)
        elif self._load_error:
            yield Static(self._load_error, markup=False)
        elif not self._loaded:
            yield Static("Open Installed to load the local model inventory.", markup=False)
        else:
            yield self._summary()

        with VerticalScroll(classes="installed-list"):
            if self._loaded and not self._rows:
                yield Static("No managed or legacy models found.", markup=False)
            for row in self._rows:
                yield self._row_widget(row)

    def _summary(self) -> Static:
        """Return the managed-store disk summary."""
        if self._usage is None:
            return Static("Disk usage unavailable.", markup=False)
        return Static(
            "Managed: "
            f"{self._format_bytes(self._usage.installed_bytes)} installed, "
            f"{self._format_bytes(self._usage.staging_bytes)} staging · "
            f"{self._format_bytes(self._usage.free_bytes)} free",
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
        self._install_active = active
        if progress is not None:
            self._install_progress = progress
        if not active:
            self._install_progress = None
        self.refresh(recompose=True)

    def _row_widget(self, row: InventoryRow) -> Vertical:
        """Build one inventory row from pure render state."""
        children: list[Widget] = [
            Static(row.model_label, classes="installed-model-title", markup=False),
            Static(row.provenance, classes="installed-model-muted", markup=False),
        ]
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
            children.append(Static(f"Size: {self._format_bytes(row.size_bytes)}"))
        children.append(Static(row.action_hint, markup=False))
        if row.reference is not None and not row.is_broken:
            children.append(
                ModelActivationControls(
                    row.reference,
                    active=row.active,
                    ready=row.ready,
                    pending=(
                        self._loading
                        or self._operation_reference is not None
                        or self._operation_name is not None
                        or self._pending_delete_reference is not None
                    ),
                )
            )
        return Vertical(*children, classes="installed-model-row")

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        """Format bytes for compact inventory copy."""
        size = float(size_bytes)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if size < 1024 or unit == "TiB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TiB"

    @staticmethod
    def scan_unmanaged(
        root: Path,
        *,
        limit: int = MAX_UNMANAGED_MODELS,
    ) -> tuple[UnmanagedRow, ...]:
        """Return a bounded scan of legacy model files.

        Args:
            root: Legacy downloader directory.
            limit: Maximum rows returned.

        Returns:
            Bounded legacy model rows in deterministic path order.
        """
        if limit <= 0 or not root.is_dir():
            return ()
        rows: list[UnmanagedRow] = []
        for directory, directories, filenames in os.walk(root):
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
        self._loading = True
        self._load_error = None
        self.refresh(recompose=True)
        self._load_inventory()

    def _service_for_worker(self) -> ModelArtifactService:
        """Create the managed service lazily on a worker thread."""
        if self._service is None:
            self._service = self._service_factory()
        return self._service

    @work(thread=True, group="installed_models_load", exclusive=True, exit_on_error=False)
    def _load_inventory(self) -> None:
        """Read managed inventory, disk totals, and legacy files off-loop."""
        try:
            service = self._service_for_worker()
            installed = service.list_installed()
            usage = service.disk_usage()
            unmanaged = self.scan_unmanaged(self._legacy_dir)
            rows = inventory_rows(installed, usage, unmanaged)
        except Exception:
            logger.opt(exception=True).error("Managed model inventory load failed")
            self.app.call_from_thread(
                self._apply_inventory,
                (),
                None,
                "The local model inventory could not be loaded.",
            )
            return
        self.app.call_from_thread(self._apply_inventory, rows, usage, None)

    def _apply_inventory(
        self,
        rows: tuple[InventoryRow, ...],
        usage: ArtifactDiskUsage | None,
        error: str | None,
    ) -> None:
        """Apply a completed inventory read on the Textual event loop."""
        self._rows = rows
        self._usage = usage
        self._loading = False
        self._loaded = error is None
        self._load_error = error
        reload_after_load = self._reload_after_load
        self._reload_after_load = False
        if reload_after_load:
            self.ensure_loaded(force=True)
        else:
            self.refresh(recompose=True)

    @on(Button.Pressed, "#installed-models-refresh")
    def _refresh_pressed(self) -> None:
        if (
            self._loading
            or self._operation_reference is not None
            or self._operation_name is not None
            or self._pending_delete_reference is not None
        ):
            return
        self.ensure_loaded(force=True)

    @on(Button.Pressed, "#installed-models-repair")
    def _repair_pressed(self) -> None:
        if (
            self._loading
            or self._operation_reference is not None
            or self._operation_name is not None
            or self._pending_delete_reference is not None
        ):
            return
        self._operation_name = "repair"
        self.refresh(recompose=True)
        self._repair_store()

    @on(ActivationRequested)
    def _activation_requested(self, event: ActivationRequested) -> None:
        event.stop()
        self._request_activation(event.reference)

    def _request_activation(self, reference: ArtifactRef) -> None:
        """Start activation unless another lifecycle operation is pending."""
        if (
            self._loading
            or self._operation_reference is not None
            or self._operation_name is not None
            or self._pending_delete_reference is not None
        ):
            return
        self._operation_reference = reference
        self._operation_name = "activate"
        self.refresh(recompose=True)
        self._activate_model(reference)

    @on(DeletionRequested)
    def _deletion_requested(self, event: DeletionRequested) -> None:
        event.stop()
        if (
            self._loading
            or self._operation_reference is not None
            or self._operation_name is not None
            or self._pending_delete_reference is not None
        ):
            return
        self._pending_delete_reference = event.reference
        self.refresh(recompose=True)
        self.app.push_screen(
            DeleteConfirmationDialog(
                item_type="Model",
                item_name=(
                    f"{event.reference.artifact_id} "
                    f"({event.reference.variant})"
                ),
                additional_warning=(
                    "The managed model files will be removed from this device."
                ),
                permanent=True,
            ),
            self._confirm_deletion,
        )

    def _confirm_deletion(self, confirmed: bool) -> None:
        """Start deletion only after the confirmation dialog accepts it."""
        reference = self._pending_delete_reference
        self._pending_delete_reference = None
        if not confirmed or reference is None:
            self.refresh(recompose=True)
            return
        if self._operation_reference is not None or self._operation_name is not None:
            self.refresh(recompose=True)
            return
        self._operation_reference = reference
        self._operation_name = "delete"
        self.refresh(recompose=True)
        self._delete_model(reference)

    @work(thread=True, group="installed_models_lifecycle", exclusive=True, exit_on_error=False)
    def _activate_model(self, reference: ArtifactRef) -> None:
        """Activate one exact verified model off the event loop."""
        try:
            self._service_for_worker().activate(reference)
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

    @work(thread=True, group="installed_models_lifecycle", exclusive=True, exit_on_error=False)
    def _delete_model(self, reference: ArtifactRef) -> None:
        """Delete one exact model without bypassing service leases."""
        try:
            self._service_for_worker().delete(reference)
        except Exception as exc:
            logger.opt(exception=True).error(
                "Managed model deletion failed for {}@{}/{}",
                reference.artifact_id,
                reference.revision,
                reference.variant,
            )
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                "delete",
                lifecycle_failure_message(exc, operation="deletion"),
            )
            return
        self.app.call_from_thread(self._apply_lifecycle_result, "delete", None)

    @work(thread=True, group="installed_models_lifecycle", exclusive=True, exit_on_error=False)
    def _repair_store(self) -> None:
        """Run explicit reconciliation off the event loop."""
        try:
            report = self._service_for_worker().reconcile()
        except Exception as exc:
            logger.opt(exception=True).error("Managed model repair failed")
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
        if error is not None:
            self.notify(error, severity="error")
        else:
            self.notify(
                success_message or f"Model {operation} completed.",
                severity="information",
            )
        self.ensure_loaded(force=True)
