"""Lazy Hugging Face GGUF discovery through the managed model store."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactAcquisitionService,
    EnvConfigCredentialResolver,
)
from tldw_chatbook.Model_Artifacts.remote_huggingface import (
    HuggingFaceRemoteAdapter,
    RemoteDiscoveryError,
    RemoteGGUFCandidate,
    RemoteModelSummary,
    ResolvedRemoteCatalog,
    ResolvedRemoteModel,
    build_remote_catalog,
    is_exact_repository,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactRef,
    ModelArtifactService,
)
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message
from tldw_chatbook.Widgets.ModelArtifacts import (
    InstallProgressed,
    InstallStatusChanged,
    ModelInstallModal,
    ModelInstallProgress,
    make_progress_callback,
)

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import CredentialResolver


class RemoteView(Widget):
    """Search and explicitly install pinned remote GGUF artifacts."""

    DEFAULT_CSS = """
    RemoteView {
        height: 100%;
    }

    RemoteView .remote-header {
        height: 3;
    }

    RemoteView #remote-model-query {
        width: 1fr;
    }

    RemoteView #remote-model-search {
        width: auto;
    }

    RemoteView .remote-list {
        height: 1fr;
    }

    RemoteView .remote-row {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid $surface-lighten-1;
    }

    RemoteView .remote-title {
        text-style: bold;
    }

    """

    def __init__(
        self,
        *,
        adapter_factory: Callable[[], HuggingFaceRemoteAdapter] = (
            HuggingFaceRemoteAdapter
        ),
        service_factory: Callable[[], ModelArtifactService] = managed_service,
        credential_resolver_factory: Callable[[], CredentialResolver] = (
            EnvConfigCredentialResolver
        ),
        id: str | None = None,
    ) -> None:
        """Create an idle Remote view without instantiating I/O dependencies.

        Args:
            adapter_factory: Lazy bounded metadata-adapter factory.
            service_factory: Lazy managed-store service factory.
            credential_resolver_factory: Lazy credential resolver factory.
            id: Optional Textual widget id.
        """
        self._adapter_factory = adapter_factory
        self._service_factory = service_factory
        self._credential_resolver_factory = credential_resolver_factory
        self._search_generation = 0
        self._resolve_generation = 0
        self._results: tuple[RemoteModelSummary, ...] = ()
        self._resolved: ResolvedRemoteModel | None = None
        self._selected_catalog: ResolvedRemoteCatalog | None = None
        self._pending_report = None
        self._operation_reference: ArtifactRef | None = None
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        """Compose retained display state without metadata or store access."""
        disabled = self._operation_reference is not None
        with Horizontal(classes="remote-header"):
            yield Input(
                placeholder="Search or enter owner/repository",
                id="remote-model-query",
                disabled=disabled,
            )
            yield Button(
                "Search",
                id="remote-model-search",
                variant="primary",
                disabled=disabled,
            )
        progress = ModelInstallProgress(None, id="remote-model-install-progress")
        progress.display = False
        yield progress
        yield Static(self._default_status(), id="remote-model-status", markup=False)

        yield VerticalScroll(
            *self._content_widgets(disabled=disabled),
            id="remote-model-content",
            classes="remote-list",
        )

    def _default_status(self) -> str:
        if self._resolved is not None:
            return (
                f"Pinned {self._resolved.repository} at "
                f"{self._resolved.commit}. Select one GGUF candidate."
            )
        if self._results:
            return "Select a repository to inspect its eligible GGUF files."
        return "Search runs only when you press Search."

    def _result_widget(self, summary: RemoteModelSummary, *, disabled: bool) -> Vertical:
        button = Button(
            "Inspect GGUF files",
            classes="remote-result",
            variant="primary",
            disabled=disabled,
        )
        button.repository = summary.repository
        access = "private" if summary.private else "public"
        return Vertical(
            Static(summary.repository, classes="remote-title", markup=False),
            Static(f"{access} · gated: {summary.gated}", markup=False),
            button,
            classes="remote-row",
        )

    def _content_widgets(self, *, disabled: bool):
        for summary in self._results:
            yield self._result_widget(summary, disabled=disabled)
        if self._resolved is not None:
            yield from self._resolved_widgets(self._resolved, disabled=disabled)

    def _resolved_widgets(
        self,
        resolved: ResolvedRemoteModel,
        *,
        disabled: bool,
    ):
        yield Static(
            "Runtime compatibility has not been verified.",
            classes="remote-title",
            markup=False,
        )
        yield Static(f"Repository: {resolved.repository}", markup=False)
        yield Static(f"Commit: {resolved.commit}", markup=False)
        license_label = (
            "Unknown / not declared"
            if resolved.license_id == "NOASSERTION"
            else resolved.license_id
        )
        yield Static(f"License: {license_label}", markup=False)
        yield Static(f"Source review page: {resolved.review_url}", markup=False)
        yield Static("Provenance: Local integrity recorded", markup=False)
        for warning in resolved.warnings:
            yield Static(f"Incomplete shard set: {warning}", markup=False)
        for candidate in resolved.candidates:
            button = Button(
                "Review and install…",
                classes="remote-candidate",
                variant="primary",
                disabled=disabled,
            )
            button.candidate = candidate
            yield Vertical(
                Static(candidate.label, classes="remote-title", markup=False),
                Static(
                    f"{len(candidate.files)} file(s) · {candidate.total_bytes} bytes",
                    markup=False,
                ),
                button,
                classes="remote-row",
            )

    def _set_status(self, message: str) -> None:
        self.query_one("#remote-model-status", Static).update(message)

    def _refresh_with_status(self, message: str) -> None:
        self._set_status(message)
        content = self.query_one("#remote-model-content", VerticalScroll)
        content.remove_children()
        content.mount(
            *self._content_widgets(disabled=self._operation_reference is not None)
        )

    def _set_metadata_controls_disabled(self, disabled: bool) -> None:
        for control in self.query("#remote-model-query, #remote-model-search"):
            control.disabled = disabled
        for button in self.query(".remote-result, .remote-candidate").results(Button):
            button.disabled = disabled

    @on(Button.Pressed, "#remote-model-search")
    @on(Input.Submitted, "#remote-model-query")
    def _search_submitted(self) -> None:
        if self._operation_reference is not None:
            return
        query = self.query_one("#remote-model-query", Input).value.strip()
        self._search_generation += 1
        self._resolve_generation += 1
        self._results = ()
        self._resolved = None
        self._selected_catalog = None
        self._set_metadata_controls_disabled(True)
        if is_exact_repository(query):
            self._set_status("Inspecting repository…")
            self._resolve_remote(query, self._resolve_generation, query)
            return
        self._set_status("Searching remote models…")
        self._search_remote(query, self._search_generation)

    @on(Button.Pressed, ".remote-result")
    def _result_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._operation_reference is not None:
            return
        repository = getattr(event.button, "repository", None)
        if not isinstance(repository, str):
            return
        self._resolve_generation += 1
        self._resolved = None
        self._selected_catalog = None
        self._set_metadata_controls_disabled(True)
        self._set_status("Inspecting repository…")
        relevant_input = self.query_one("#remote-model-query", Input).value.strip()
        self._resolve_remote(repository, self._resolve_generation, relevant_input)

    @on(Button.Pressed, ".remote-candidate")
    def _candidate_pressed(self, event: Button.Pressed) -> None:
        """Freeze one resolved candidate and begin managed preflight."""
        event.stop()
        candidate = getattr(event.button, "candidate", None)
        if (
            type(candidate) is not RemoteGGUFCandidate
            or self._resolved is None
            or self._operation_reference is not None
        ):
            return
        try:
            catalog = build_remote_catalog(self._resolved, candidate)
        except ValueError:
            self.notify(
                "This GGUF candidate cannot be safely prepared. Search again.",
                severity="error",
            )
            return
        self._selected_catalog = catalog
        self._operation_reference = catalog.artifact.reference
        self._set_metadata_controls_disabled(True)
        self._set_status("Preparing the managed install plan…")
        self._preflight_model(catalog)

    @work(thread=True, group="remote_model_search", exit_on_error=False)
    def _search_remote(self, query: str, generation: int) -> None:
        """Search explicit free text off the Textual event loop."""
        try:
            resolver = self._credential_resolver_factory()
            token = resolver.resolve(query)
            results = asyncio.run(self._adapter_factory().search(query, token=token))
        except Exception as exc:
            self.app.call_from_thread(
                self._apply_search_result,
                generation,
                query,
                (),
                exc,
            )
            return
        self.app.call_from_thread(
            self._apply_search_result,
            generation,
            query,
            results,
            None,
        )

    def _apply_search_result(
        self,
        generation: int,
        query: str,
        results: tuple[RemoteModelSummary, ...],
        error: BaseException | None,
    ) -> None:
        """Apply only the current query generation and current input value."""
        if generation != self._search_generation:
            return
        current_query = self.query_one("#remote-model-query", Input).value.strip()
        if current_query != query:
            self._set_metadata_controls_disabled(False)
            return
        self._set_metadata_controls_disabled(False)
        if error is not None:
            self._results = ()
            self._resolved = None
            self._refresh_with_status(_discovery_error_message(error))
            return
        self._results = results
        self._resolved = None
        message = (
            "Select a repository to inspect its eligible GGUF files."
            if results
            else "No matching repositories were found. Try another search."
        )
        self._refresh_with_status(message)

    @work(thread=True, group="remote_model_resolve", exit_on_error=False)
    def _resolve_remote(
        self,
        repository: str,
        generation: int,
        relevant_input: str,
    ) -> None:
        """Resolve one exact repository off the Textual event loop."""
        try:
            resolver = self._credential_resolver_factory()
            token = resolver.resolve(repository)
            resolved = asyncio.run(
                self._adapter_factory().resolve(repository, token=token)
            )
        except Exception as exc:
            self.app.call_from_thread(
                self._apply_resolve_result,
                generation,
                repository,
                relevant_input,
                None,
                exc,
            )
            return
        self.app.call_from_thread(
            self._apply_resolve_result,
            generation,
            repository,
            relevant_input,
            resolved,
            None,
        )

    def _apply_resolve_result(
        self,
        generation: int,
        requested_repository: str,
        relevant_input: str,
        resolved: ResolvedRemoteModel | None,
        error: BaseException | None,
    ) -> None:
        """Apply only the current repository-resolution identity."""
        if generation != self._resolve_generation:
            return
        current_input = self.query_one("#remote-model-query", Input).value.strip()
        repository_is_current = (
            requested_repository == relevant_input
            if is_exact_repository(relevant_input)
            else any(
                result.repository == requested_repository for result in self._results
            )
        )
        if (
            current_input != relevant_input
            or not repository_is_current
            or (
                resolved is not None
                and resolved.repository != requested_repository
            )
        ):
            self._set_metadata_controls_disabled(False)
            return
        self._set_metadata_controls_disabled(False)
        if error is not None or resolved is None:
            self._resolved = None
            self._refresh_with_status(
                _discovery_error_message(error or RuntimeError("resolve failed"))
            )
            return
        self._results = ()
        self._resolved = resolved
        self._refresh_with_status(
            f"Pinned {resolved.repository} at {resolved.commit}. "
            "Select one GGUF candidate."
        )

    async def _preflight(self, catalog: ResolvedRemoteCatalog):
        """Resolve the selected catalog's managed plan on a worker event loop."""
        resolver = self._credential_resolver_factory()
        acquisition = ArtifactAcquisitionService(
            self._service_factory(),
            credential_resolver=resolver,
        )
        return await acquisition.preflight(
            catalog.artifact.reference,
            catalog,
            sources=catalog.sources,
        )

    @work(thread=True, group="remote_model_install", exclusive=True, exit_on_error=False)
    def _preflight_model(self, catalog: ResolvedRemoteCatalog) -> None:
        """Resolve the frozen candidate plan off the Textual event loop."""
        try:
            report = asyncio.run(self._preflight(catalog))
        except Exception as exc:
            logger.error(
                "Remote model preflight failed for managed artifact {}",
                catalog.artifact.reference.artifact_id,
            )
            self.app.call_from_thread(
                self._apply_preflight_result,
                None,
                install_failure_message(
                    exc,
                    model_label=catalog.artifact.model_id,
                ),
            )
            return
        self.app.call_from_thread(self._apply_preflight_result, report, None)

    def _apply_preflight_result(self, report, error: str | None) -> None:
        """Show the shared consent modal or release the frozen selection."""
        catalog = self._selected_catalog
        if (
            error is not None
            or report is None
            or catalog is None
            or report.root != catalog.artifact.reference
        ):
            self._pending_report = None
            self._operation_reference = None
            self._selected_catalog = None
            self._set_metadata_controls_disabled(False)
            message = error or "The install plan changed. Search and review it again."
            self._set_status(message)
            self.notify(message, severity="error")
            return
        self._pending_report = report
        acknowledgment = (
            "No license was declared. I reviewed the source and want to continue."
            if catalog.artifact.license_id == "NOASSERTION"
            else None
        )
        self.app.push_screen(
            ModelInstallModal(
                report,
                model_label=catalog.artifact.model_id,
                required_acknowledgment=acknowledgment,
            ),
            self._confirm_install,
        )

    def _confirm_install(self, confirmed: bool) -> None:
        """Provision only after consent while preserving the frozen catalog."""
        if not confirmed:
            self._pending_report = None
            self._operation_reference = None
            self._selected_catalog = None
            self._set_metadata_controls_disabled(False)
            self._set_status(self._default_status())
            return
        if self._operation_reference is not None:
            self.post_message(
                InstallStatusChanged(self._operation_reference, active=True)
            )
        self._provision_model()

    async def _provision(self, report, catalog: ResolvedRemoteCatalog):
        """Install the reviewed catalog without activating it."""
        resolver = self._credential_resolver_factory()
        acquisition = ArtifactAcquisitionService(
            self._service_factory(),
            credential_resolver=resolver,
        )
        return await acquisition.provision(
            report.root,
            report.grant(),
            catalog,
            sources=catalog.sources,
            progress=make_progress_callback(self.post_message),
            activate=False,
        )

    @work(thread=True, group="remote_model_install", exclusive=True, exit_on_error=False)
    def _provision_model(self) -> None:
        """Provision the frozen plan and catalog off the Textual event loop."""
        report = self._pending_report
        catalog = self._selected_catalog
        if report is None or catalog is None:
            self.app.call_from_thread(
                self._apply_provision_result,
                "No install plan is available; search and review the model again.",
            )
            return
        try:
            asyncio.run(self._provision(report, catalog))
        except Exception as exc:
            logger.error(
                "Remote model installation failed for managed artifact {}",
                report.root.artifact_id,
            )
            self.app.call_from_thread(
                self._apply_provision_result,
                install_failure_message(
                    exc,
                    model_label=catalog.artifact.model_id,
                ),
            )
            return
        self.app.call_from_thread(self._apply_provision_result, None)

    @on(InstallProgressed)
    def _install_progressed(self, event: InstallProgressed) -> None:
        """Render shared acquisition progress in the Remote view."""
        try:
            progress = self.query_one(
                "#remote-model-install-progress",
                ModelInstallProgress,
            )
        except NoMatches:
            return
        progress.display = True
        progress.update_progress(event.progress)

    def _apply_provision_result(self, error: str | None) -> None:
        """Finish install-only provisioning and publish inventory refresh state."""
        reference = self._operation_reference
        self._pending_report = None
        self._operation_reference = None
        self._selected_catalog = None
        try:
            progress = self.query_one(
                "#remote-model-install-progress",
                ModelInstallProgress,
            )
        except NoMatches:
            progress = None
        if progress is not None:
            progress.display = False
        self._set_metadata_controls_disabled(False)
        if error is not None:
            self._set_status(error)
            self.notify(error, severity="error")
        else:
            message = (
                "Model downloaded and managed. Runtime compatibility has not "
                "been verified."
            )
            self._set_status(message)
            self.notify(message, severity="information")
        if reference is not None:
            self.post_message(
                InstallStatusChanged(
                    reference,
                    active=False,
                    succeeded=error is None,
                )
            )


def _discovery_error_message(error: BaseException) -> str:
    """Map metadata failures to fixed recovery copy without raw details."""
    if not isinstance(error, RemoteDiscoveryError):
        return "Remote model discovery failed. Retry."
    if error.code in {"authentication_required", "access_forbidden"}:
        return "Configure or verify Hugging Face access, then Retry."
    if error.code == "repository_not_found":
        return "Repository not found. Check the exact ID or search again."
    if error.code in {"invalid_response", "response_too_large"}:
        return "This repository cannot be safely inspected."
    if error.code == "no_eligible_gguf":
        return (
            "No eligible GGUF files were found. Files must be LFS-backed with "
            "size and SHA-256 metadata."
        )
    if error.retryable or error.code in {"network_error", "rate_limited"}:
        return "Remote request failed. Retry."
    if error.code in {"invalid_query", "invalid_repository"}:
        return "Enter a search term or exact owner/repository ID."
    return "Remote model discovery failed. Retry."
