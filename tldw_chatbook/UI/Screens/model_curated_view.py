"""Lazy Curated view backed by the verified managed-model acquisition path."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Model_Artifacts.curated_registry import (
    CuratedRegistry,
    curated_registry,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactRef,
    ModelArtifactService,
)
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.UI.Screens.model_browser_state import (
    install_failure_message,
    provenance_label,
)
from tldw_chatbook.Widgets.ModelArtifacts import (
    InstallProgressed,
    ModelInstallModal,
    ModelInstallProgress,
    make_progress_callback,
)


@dataclass(frozen=True)
class CuratedRow:
    """One curated descriptor cross-referenced with installed inventory."""

    descriptor: ArtifactDescriptor
    installed: bool


class CuratedView(Widget):
    """Browse and explicitly install models curated by the application."""

    DEFAULT_CSS = """
    CuratedView {
        height: 100%;
    }

    CuratedView .curated-header {
        height: 3;
    }

    CuratedView .curated-header Button {
        width: auto;
    }

    CuratedView .curated-list {
        height: 1fr;
    }

    CuratedView .curated-model-row {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid $surface-lighten-1;
    }

    CuratedView .curated-model-title {
        text-style: bold;
    }

    CuratedView .curated-model-muted {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        service_factory: Callable[[], ModelArtifactService] = managed_service,
        registry_factory: Callable[[], CuratedRegistry] = curated_registry,
        id: str | None = None,
    ) -> None:
        """Create an idle curated view.

        Args:
            service_factory: Lazy managed-store service factory.
            registry_factory: Lazy curated-registry factory.
            id: Optional Textual widget id.
        """
        self._service_factory = service_factory
        self._registry_factory = registry_factory
        self._service: ModelArtifactService | None = None
        self._registry: CuratedRegistry | None = None
        self._rows: tuple[CuratedRow, ...] = ()
        self._loaded = False
        self._loading = False
        self._load_error: str | None = None
        self._operation_reference: ArtifactRef | None = None
        self._pending_report = None
        self._progress = None
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        """Compose from retained state without filesystem or network I/O."""
        with Horizontal(classes="curated-header"):
            yield Button("Refresh", id="curated-models-refresh", variant="primary")
        progress = ModelInstallProgress(
            self._progress,
            id="curated-model-install-progress",
        )
        progress.display = self._progress is not None
        yield progress
        if self._loading:
            yield Static("Loading curated models…", markup=False)
        elif self._load_error:
            yield Static(self._load_error, markup=False)
        elif not self._loaded:
            yield Static("Open Curated to load the offline model catalog.", markup=False)

        with VerticalScroll(classes="curated-list"):
            for row in self._rows:
                yield self._row_widget(row)

    def _row_widget(self, row: CuratedRow) -> Vertical:
        """Build one curated model row."""
        descriptor = row.descriptor
        install = Button(
            "Installed" if row.installed else "Review and install…",
            classes="curated-install",
            variant="primary",
            disabled=row.installed or self._operation_reference is not None,
        )
        install.reference = descriptor.reference
        return Vertical(
            Static(descriptor.model_id, classes="curated-model-title", markup=False),
            Static(
                f"{descriptor.model_family} · {descriptor.precision} · "
                f"{descriptor.format.value.upper()}",
                markup=False,
            ),
            Static(
                f"Revision: {descriptor.upstream_revision}",
                classes="curated-model-muted",
                markup=False,
            ),
            Static(
                f"License: {descriptor.license_id} · "
                f"{provenance_label(descriptor.provenance)}",
                classes="curated-model-muted",
                markup=False,
            ),
            install,
            classes="curated-model-row",
        )

    def _service_for_worker(self) -> ModelArtifactService:
        """Return the lazily created managed service."""
        if self._service is None:
            self._service = self._service_factory()
        return self._service

    def _registry_for_worker(self) -> CuratedRegistry:
        """Return the lazily created curated registry."""
        if self._registry is None:
            self._registry = self._registry_factory()
        return self._registry

    def ensure_loaded(self, *, force: bool = False) -> None:
        """Load the offline catalog when the Curated rail row is selected.

        Args:
            force: Reload installed-state cross-references.
        """
        if self._loading or (self._loaded and not force):
            return
        self._loading = True
        self._load_error = None
        self.refresh(recompose=True)
        self._load_curated()

    @work(thread=True, group="curated_models_load", exclusive=True, exit_on_error=False)
    def _load_curated(self) -> None:
        """Cross-reference curated descriptors with installed refs off-loop."""
        try:
            registry = self._registry_for_worker()
            installed = self._service_for_worker().list_installed()
            installed_refs = {
                item.descriptor.reference
                for item in installed
                if item.descriptor is not None
            }
            rows = tuple(
                CuratedRow(descriptor, descriptor.reference in installed_refs)
                for descriptor in registry.list()
            )
        except Exception:
            self.app.call_from_thread(
                self._apply_rows,
                (),
                "The curated model catalog could not be loaded.",
            )
            return
        self.app.call_from_thread(self._apply_rows, rows, None)

    def _apply_rows(
        self,
        rows: tuple[CuratedRow, ...],
        error: str | None,
    ) -> None:
        """Apply a completed curated/inventory cross-reference."""
        self._rows = rows
        self._loading = False
        self._loaded = error is None
        self._load_error = error
        self.refresh(recompose=True)

    @on(Button.Pressed, "#curated-models-refresh")
    def _refresh_pressed(self) -> None:
        self.ensure_loaded(force=True)

    @on(Button.Pressed, ".curated-install")
    def _install_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        reference = getattr(event.button, "reference", None)
        if not isinstance(reference, ArtifactRef) or self._operation_reference is not None:
            return
        self._operation_reference = reference
        self.refresh(recompose=True)
        self._preflight_model(reference)

    def _source_map(self) -> dict[ArtifactRef, dict[str, str]]:
        """Return sources for every descriptor in the curated registry."""
        registry = self._registry_for_worker()
        return {
            descriptor.reference: registry.sources(descriptor.reference)
            for descriptor in registry.list()
        }

    async def _preflight(self, reference: ArtifactRef):
        """Resolve a curated acquisition plan on the worker's event loop."""
        from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService

        acquisition = ArtifactAcquisitionService(self._service_for_worker())
        return await acquisition.preflight(
            reference,
            self._registry_for_worker(),
            sources=self._source_map(),
        )

    @work(thread=True, group="curated_model_install", exclusive=True, exit_on_error=False)
    def _preflight_model(self, reference: ArtifactRef) -> None:
        """Resolve the plan off the Textual event loop."""
        try:
            report = asyncio.run(self._preflight(reference))
        except Exception as exc:
            logger.opt(exception=True).error("Curated model preflight failed")
            self.app.call_from_thread(
                self._apply_preflight_result,
                None,
                install_failure_message(exc, model_label=reference.artifact_id),
            )
            return
        self.app.call_from_thread(self._apply_preflight_result, report, None)

    def _apply_preflight_result(self, report, error: str | None) -> None:
        """Show the shared plan modal or a sanitized failure."""
        if error is not None or report is None:
            self._operation_reference = None
            self.notify(error or "Model preflight failed.", severity="error")
            self.refresh(recompose=True)
            return
        self._pending_report = report
        descriptor = self._registry_for_worker().descriptor(report.root)
        self.app.push_screen(
            ModelInstallModal(report, model_label=descriptor.model_id),
            self._confirm_install,
        )

    def _confirm_install(self, confirmed: bool) -> None:
        """Start provisioning only after explicit consent."""
        if not confirmed:
            self._pending_report = None
            self._operation_reference = None
            self.refresh(recompose=True)
            return
        self._provision_model()

    async def _provision(self, report):
        """Provision the consented report on the worker's event loop."""
        from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService

        acquisition = ArtifactAcquisitionService(self._service_for_worker())
        return await acquisition.provision(
            report.root,
            report.grant(),
            self._registry_for_worker(),
            sources=self._source_map(),
            progress=make_progress_callback(self.post_message),
        )

    @work(thread=True, group="curated_model_install", exclusive=True, exit_on_error=False)
    def _provision_model(self) -> None:
        """Provision the consented plan off the Textual event loop."""
        report = self._pending_report
        if report is None:
            self.app.call_from_thread(
                self._apply_provision_result,
                "No install plan is available; review the model again.",
            )
            return
        try:
            asyncio.run(self._provision(report))
        except Exception as exc:
            logger.opt(exception=True).error("Curated model installation failed")
            self.app.call_from_thread(
                self._apply_provision_result,
                install_failure_message(
                    exc,
                    model_label=report.root.artifact_id,
                ),
            )
            return
        self.app.call_from_thread(self._apply_provision_result, None)

    @on(InstallProgressed)
    def _install_progressed(self, event: InstallProgressed) -> None:
        """Retain and render worker progress outside the modal."""
        event.stop()
        self._progress = event.progress
        progress = self.query_one(
            "#curated-model-install-progress",
            ModelInstallProgress,
        )
        progress.display = True
        progress.update_progress(event.progress)

    def _apply_provision_result(self, error: str | None) -> None:
        """Finish an installation and refresh curated installed-state."""
        self._pending_report = None
        self._operation_reference = None
        self._progress = None
        progress = self.query_one(
            "#curated-model-install-progress",
            ModelInstallProgress,
        )
        progress.display = False
        if error is not None:
            self.notify(error, severity="error")
        else:
            self.notify("Model installed and activated.", severity="information")
        self.ensure_loaded(force=True)
