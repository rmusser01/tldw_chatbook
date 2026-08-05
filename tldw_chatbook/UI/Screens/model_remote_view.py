"""Lazy Hugging Face GGUF discovery through the managed model store."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Static

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
from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionProgress,
        CredentialResolver,
    )


def _default_credential_resolver() -> "CredentialResolver":
    """Lazily construct the production env/config credential resolver.

    A plain function, not ``EnvConfigCredentialResolver`` itself, as the
    ``credential_resolver_factory`` constructor default: evaluating a
    default parameter value happens once, at class-definition (i.e.
    module-import) time, so binding the class itself there would import
    ``Model_Artifacts.acquisition`` at module scope. This function defers
    that import to its own body, run only when a ``RemoteView`` is actually
    constructed (or a test calls the factory directly) -- the same
    module-scope boundary this module's own docstring, and
    ``model_curated_view.py`` before it, both hold to.

    Returns:
        A fresh ``EnvConfigCredentialResolver``.
    """
    from tldw_chatbook.Model_Artifacts.acquisition import EnvConfigCredentialResolver

    return EnvConfigCredentialResolver()


class RemoteView(Widget):
    """Search remote GGUF repositories and request their installation.

    Metadata discovery -- ``Search``/repository inspection, driven by
    :meth:`_search_remote`/:meth:`_resolve_remote` -- stays owned by this
    view (TASK-1914): it is a read-only listing concern, exactly like
    ``CuratedView._load_curated`` and ``InstalledView``'s own inventory
    load, neither of which TASK-1803 moved either. Only acquisition --
    preflight, provision, and the consent modal in between -- crosses the
    browser's screen-owns-worker boundary: reviewing a candidate posts
    :class:`InstallRequested` and waits to be told what happened via
    :meth:`apply_progress`, :meth:`cancel_pending_install`, or
    :meth:`finish_install`. ``LLMScreen`` owns the actual preflight/
    provision workers (mirroring ``LibraryScreen``'s Parakeet v2 flow and
    ``CuratedView``'s own TASK-1803 fix) precisely so a several-hundred-
    megabyte download survives both the consent modal being dismissed and
    a screen-level recompose that tears down and rebuilds this exact view
    mid-install.

    Before this change, this view owned that worker directly (added by
    PR #1190, TASK-596.1) and had no compensating delivery logic at all: a
    screen-level recompose mid-install orphaned the worker, whose own
    ``post_message`` became a silent no-op once this instance was torn
    down, so progress stopped reaching the UI with nothing to catch it.
    Moving the worker to ``LLMScreen`` -- which already survives that same
    recompose for the equivalent curated install -- removes the need for
    any durable-delivery fallback here: ``LLMScreen`` is never the thing
    being torn down, so there is no orphaned poster left to compensate for.

    This module deliberately never imports ``ArtifactAcquisitionService``
    nor calls ``preflight()``/``provision()`` itself; ``Model_Artifacts.
    acquisition``/``fetch`` are only ever imported inside ``LLMScreen``'s
    own worker methods (and, lazily, inside :func:`_default_credential_
    resolver`, for the resolver this view's metadata search still performs
    itself).
    """

    class InstallRequested(Message):
        """Posted when a reviewed remote GGUF candidate is selected for install."""

        def __init__(
            self,
            catalog: ResolvedRemoteCatalog,
            candidate: RemoteGGUFCandidate,
            *,
            service: ModelArtifactService,
            credential_resolver: "CredentialResolver",
        ) -> None:
            """Carry everything the host screen needs to preflight/provision.

            Args:
                catalog: The one-item remote catalog this view already
                    froze via ``build_remote_catalog`` -- a pure, local
                    computation, not itself an acquisition call -- naming
                    the exact selected candidate as its root artifact.
                candidate: The exact GGUF candidate the catalog was built
                    from, carried alongside it so the host screen can
                    describe the selected files (size/digest/source URL)
                    on the shared consent modal without re-deriving them.
                service: The managed-store service resolved by this view's
                    own (possibly test-injected) ``service_factory`` --
                    captured here so ``LLMScreen``'s worker uses the exact
                    same instance a test constructed this view with.
                credential_resolver: The credential resolver resolved the
                    same way, via ``credential_resolver_factory`` -- the
                    same seam this view's own metadata search already uses,
                    so a gated repository's preflight/provision requests
                    carry the same credential its search did.
            """
            super().__init__()
            self.catalog = catalog
            self.candidate = candidate
            self.service = service
            self.credential_resolver = credential_resolver

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
        credential_resolver_factory: Callable[[], "CredentialResolver"] = (
            _default_credential_resolver
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
        self._operation_reference: ArtifactRef | None = None
        self._progress: "AcquisitionProgress | None" = None
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
        progress = ModelInstallProgress(
            self._progress,
            id="remote-model-install-progress",
        )
        progress.display = self._progress is not None
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

    def _result_widget(
        self, summary: RemoteModelSummary, *, disabled: bool
    ) -> Vertical:
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
        if resolved.total_candidate_count > len(resolved.candidates):
            yield Static(
                f"First {len(resolved.candidates)} of "
                f"{resolved.total_candidate_count}, sorted by upstream path",
                markup=False,
            )
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
        self._set_metadata_controls_disabled(True)
        self._set_status("Inspecting repository…")
        relevant_input = self.query_one("#remote-model-query", Input).value.strip()
        self._resolve_remote(repository, self._resolve_generation, relevant_input)

    @on(Button.Pressed, ".remote-candidate")
    def _candidate_pressed(self, event: Button.Pressed) -> None:
        """Post an install intent for one reviewed candidate; never preflight/provision here."""
        event.stop()
        candidate = getattr(event.button, "candidate", None)
        if (
            type(candidate) is not RemoteGGUFCandidate
            or self._resolved is None
            or candidate not in self._resolved.candidates
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
        self._operation_reference = catalog.artifact.reference
        self._set_metadata_controls_disabled(True)
        self._set_status("Preparing the managed install plan…")
        self.post_message(
            self.InstallRequested(
                catalog,
                candidate,
                service=self._service_factory(),
                credential_resolver=self._credential_resolver_factory(),
            )
        )

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
            or (resolved is not None and resolved.repository != requested_repository)
        ):
            self._results = ()
            self._resolved = None
            self._set_metadata_controls_disabled(False)
            self._refresh_with_status(
                "Repository selection changed. Press Search to inspect the current ID."
            )
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

    def apply_progress(self, progress: "AcquisitionProgress") -> None:
        """Render one acquisition progress event, retaining it for later.

        Called ONLY by the host screen (``LLMScreen``) -- via its own
        ``InstallProgressed`` handler for a live tick, and to re-apply the
        last known progress to a freshly (re)mounted instance after a
        screen-level recompose. This view has no ``@on(InstallProgressed)``
        handler of its own and never posts that message itself (TASK-1914
        moved the worker that used to post it to ``LLMScreen``); ``LLMScreen``
        is the sole caller, giving exactly one render per tick.

        ``self._progress`` is retained before either branch below runs, so
        a fallback ``refresh(recompose=True)`` still picks up the correct
        value on this view's next (complete) compose pass -- mirroring
        ``CuratedView.apply_progress``'s own tolerance for the same
        momentary mid-recompose gap.

        Args:
            progress: The acquisition progress event to render.
        """
        self._progress = progress
        try:
            widget = self.query_one(
                "#remote-model-install-progress",
                ModelInstallProgress,
            )
            widget.display = True
            widget.update_progress(progress)
        except NoMatches:
            self.refresh(recompose=True)

    def cancel_pending_install(self, message: str | None = None) -> None:
        """Clear the in-flight indicator without disturbing search/resolve state.

        Called by the host screen (``LLMScreen``) when a request this view
        posted did not lead to an install actually starting: a preflight
        failure, an explicit decline at the consent modal, or a request
        refused outright because a different install (curated or remote --
        both now share one screen-level lock, see ``LLMScreen``) is already
        running -- in which case this is the freshly (re)mounted view that
        just clicked install, not the instance whose install is still in
        flight.

        Args:
            message: Status copy to show in place of the in-flight
                indicator, e.g. a sanitized preflight failure. ``None``
                (an explicit decline) restores the default status derived
                from current search/resolve state.
        """
        self._operation_reference = None
        self._set_metadata_controls_disabled(False)
        self._set_status(message or self._default_status())

    def finish_install(self, message: str | None = None) -> None:
        """Clear the in-flight indicator and hide progress after a completed install.

        Called by the host screen (``LLMScreen``) once provisioning
        finishes, successfully or not.

        Args:
            message: The outcome copy to show (success or sanitized
                failure); ``None`` restores the default status.
        """
        self._operation_reference = None
        self._progress = None
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
        self._set_status(message or self._default_status())


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
        message = (
            "No eligible GGUF files were found. Files must be LFS-backed with "
            "size and SHA-256 metadata."
        )
        if error.details:
            details = "\n".join(
                f"Incomplete shard set: {detail}" for detail in error.details
            )
            message = f"{message}\n{details}"
        return message
    if error.retryable or error.code in {"network_error", "rate_limited"}:
        return "Remote request failed. Retry."
    if error.code in {"invalid_query", "invalid_repository"}:
        return "Enter a search term or exact owner/repository ID."
    return "Remote model discovery failed. Retry."
