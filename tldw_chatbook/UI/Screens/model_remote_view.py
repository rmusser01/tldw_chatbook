"""Lazy Hugging Face GGUF discovery through the managed model store."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger
from textual import events, on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Select, Static

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
from tldw_chatbook.Model_Artifacts.machine_memory import (
    MachineMemorySnapshot,
    project_gguf_memory,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactRef,
    ModelArtifactService,
)
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.UI.Screens.model_browser_state import (
    VariantGuidance,
    filter_variant_guidance,
    format_mib,
    sort_variant_guidance,
    variant_guidance,
)
from tldw_chatbook.UI.Screens.model_memory_presenter import (
    MachineMemoryPresentation,
    build_candidate_memory_presentation,
    build_machine_memory_presentation,
)
from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionProgress,
        CredentialResolver,
    )


_SINGLE_PANE_WIDTH = 72


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
    browser's screen-owns-worker boundary: confirming the selected candidate
    with the contextual install action posts :class:`InstallRequested` and
    waits to be told what happened via
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
        """Posted when a reviewed remote GGUF candidate is confirmed for install."""

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

    class OpenInstalledRequested(Message):
        """Request navigation to the exact completed managed model."""

        def __init__(self, reference: ArtifactRef) -> None:
            """Carry the verified managed root without deriving identity from UI copy.

            Args:
                reference: Exact verified managed root to reveal.
            """
            super().__init__()
            self.reference = reference

    class ConfigureRuntimeRequested(Message):
        """Request runtime selection for the exact completed managed model."""

        def __init__(self, reference: ArtifactRef) -> None:
            """Carry the verified managed root into the host-owned chooser.

            Args:
                reference: Exact verified managed root to configure.
            """
            super().__init__()
            self.reference = reference

    class DiscoveryStarted(Message):
        """Notify the host that prior durable completion is no longer current."""

        def __init__(self, query: str) -> None:
            """Carry the submitted provider query for lifecycle attribution.

            Args:
                query: Provider search text that supersedes prior completion.
            """
            super().__init__()
            self.query = query

    class MachineMemoryRequested(Message):
        """Request screen-owned local-memory observation or refresh."""

        def __init__(self, *, force: bool) -> None:
            """Create one presentation-only observation intent."""
            super().__init__()
            self.force = force

    BUNDLED_CSS = """
    RemoteView {
        height: 100%;
        background: $background;
    }

    RemoteView .remote-header {
        height: 5;
        padding: 0 1;
        background: $panel;
    }

    RemoteView .remote-source {
        height: 2;
        padding: 0;
        color: $text-muted;
    }

    RemoteView .remote-search-row {
        height: 3;
    }

    RemoteView #remote-model-query {
        width: 1fr;
    }

    RemoteView #remote-model-search {
        width: auto;
    }

    RemoteView .remote-workspace {
        height: 1fr;
    }

    RemoteView .remote-results-pane {
        width: 2fr;
        min-width: 24;
        border-right: solid $surface-lighten-1;
        background: $panel;
    }

    RemoteView .remote-detail-pane {
        width: 3fr;
        min-width: 32;
        background: $background;
    }

    RemoteView .remote-pane-title {
        height: 2;
        padding: 0 1;
        text-style: bold;
        background: $surface;
        color: $text;
    }

    RemoteView #remote-model-results,
    RemoteView #remote-model-details {
        height: 1fr;
    }

    RemoteView #remote-model-results {
        padding: 0 1;
    }

    RemoteView #remote-model-details {
        padding: 1 2;
    }

    RemoteView .remote-result-row {
        height: auto;
        padding: 1 0;
        border-bottom: solid $surface-lighten-1;
    }

    RemoteView .remote-title {
        text-style: bold;
    }

    RemoteView .remote-muted {
        color: $text-muted;
    }

    RemoteView .remote-result,
    RemoteView .remote-candidate {
        width: 100%;
        margin-top: 1;
    }

    RemoteView .remote-candidate-row {
        height: auto;
        padding: 1 0;
        border-bottom: solid $surface-lighten-1;
    }

    RemoteView .remote-compatibility {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
        background: $warning 12%;
        color: $text;
        text-style: bold;
    }

    RemoteView .remote-variant-guidance {
        height: auto;
        margin-top: 1;
        padding: 0 1;
        border-left: thick $accent;
        background: $accent 8%;
        color: $text;
    }

    RemoteView .remote-machine-panel {
        height: auto;
        margin-top: 1;
        padding: 1;
        border: solid $surface-lighten-1;
        background: $panel;
    }

    RemoteView .remote-machine-actions {
        height: auto;
        min-height: 3;
        margin-top: 1;
    }

    RemoteView .remote-machine-actions Button {
        width: auto;
        margin-right: 1;
    }

    RemoteView .remote-fit-outcome,
    RemoteView .remote-fit-details,
    RemoteView .remote-variant-filename {
        width: 100%;
        height: auto;
    }

    RemoteView .remote-fit-outcome {
        margin-top: 1;
        text-style: bold;
    }

    RemoteView .remote-fit-details {
        color: $text-muted;
    }

    RemoteView .remote-variant-controls {
        height: auto;
        min-height: 3;
        margin-top: 1;
    }

    RemoteView #remote-variant-filter {
        width: 1fr;
        min-width: 16;
    }

    RemoteView #remote-variant-sort {
        width: 24;
        min-width: 18;
        margin-left: 1;
        margin-bottom: 0;
    }

    RemoteView .remote-variant-list,
    RemoteView .remote-variant-facts,
    RemoteView .remote-variant-summary,
    RemoteView .remote-variant-empty {
        height: auto;
    }

    RemoteView .remote-variant-empty {
        padding: 1 0;
    }

    RemoteView .remote-source-review {
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }

    RemoteView .remote-detail-section {
        height: auto;
        margin-top: 1;
        text-style: bold;
        color: $accent;
    }

    RemoteView #remote-model-status {
        height: auto;
        min-height: 2;
        padding: 0 1;
        color: $text-muted;
    }

    RemoteView #remote-model-install-progress {
        height: auto;
        margin: 0 1;
        padding: 1;
        background: $surface;
    }

    RemoteView .remote-action-bar {
        height: auto;
        min-height: 4;
        padding: 1;
        border-top: solid $surface-lighten-1;
        background: $panel;
    }

    RemoteView #remote-model-selection {
        width: 1fr;
        height: auto;
        color: $text-muted;
    }

    RemoteView #remote-model-install {
        width: auto;
        min-width: 22;
    }

    RemoteView .remote-completion-actions {
        width: auto;
        height: 3;
    }

    RemoteView .remote-completion-actions Button {
        width: auto;
        margin-left: 1;
    }

    RemoteView.-narrow .remote-results-pane {
        width: 100%;
        min-width: 0;
        border-right: none;
    }

    RemoteView.-narrow .remote-detail-pane {
        width: 100%;
        min-width: 0;
    }

    RemoteView.-narrow .remote-header {
        height: 4;
        padding: 0;
    }

    RemoteView.-narrow .remote-source {
        height: 1;
    }

    RemoteView.-narrow #remote-model-details {
        padding: 0 1;
    }

    RemoteView.-narrow #remote-model-status {
        height: 1;
        min-height: 1;
        padding: 0;
    }

    RemoteView.-narrow .remote-variant-controls {
        layout: vertical;
        min-height: 6;
        margin-top: 0;
    }

    RemoteView.-narrow #remote-variant-filter,
    RemoteView.-narrow #remote-variant-sort {
        width: 100%;
        min-width: 0;
        margin-left: 0;
    }

    RemoteView.-narrow .remote-action-bar {
        layout: vertical;
        min-height: 4;
        padding: 0;
    }

    RemoteView.-narrow #remote-model-selection {
        height: 1;
    }

    RemoteView.-narrow #remote-model-selection,
    RemoteView.-narrow #remote-model-install,
    RemoteView.-narrow .remote-completion-actions {
        width: 100%;
        min-width: 0;
    }

    RemoteView.-narrow .remote-completion-actions Button {
        width: 1fr;
        min-width: 0;
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
        source_label: str = "Hugging Face",
        id: str | None = None,
    ) -> None:
        """Create an idle Remote view without instantiating I/O dependencies.

        Args:
            adapter_factory: Lazy bounded metadata-adapter factory.
            service_factory: Lazy managed-store service factory.
            credential_resolver_factory: Lazy credential resolver factory.
            source_label: Current provider authority shown without implying
                that Remote is permanently tied to one provider.
            id: Optional Textual widget id.
        """
        self._adapter_factory = adapter_factory
        self._service_factory = service_factory
        self._credential_resolver_factory = credential_resolver_factory
        self._source_label = source_label
        self._query_value = ""
        self._search_generation = 0
        self._resolve_generation = 0
        self._results: tuple[RemoteModelSummary, ...] = ()
        self._resolved: ResolvedRemoteModel | None = None
        self._selected_repository: str | None = None
        self._selected_candidate: RemoteGGUFCandidate | None = None
        self._variant_filter = ""
        self._variant_sort = "source"
        self._operation_reference: ArtifactRef | None = None
        self._completed_reference: ArtifactRef | None = None
        self._progress: "AcquisitionProgress | None" = None
        self._machine_snapshot: MachineMemorySnapshot | None = None
        self._machine_presentation = build_machine_memory_presentation(None)
        self._machine_details_expanded = True
        self._machine_details_touched = False
        self._single_pane_show_detail = False
        self._repository_focus_locator: str | None = None
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        """Compose retained display state without metadata or store access."""
        disabled = self._operation_reference is not None
        with Vertical(classes="remote-header"):
            yield Static(
                f"Source: {self._source_label}",
                classes="remote-source",
                markup=False,
            )
            with Horizontal(classes="remote-search-row"):
                yield Input(
                    value=self._query_value,
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
        with Horizontal(classes="remote-workspace"):
            with Vertical(classes="remote-results-pane"):
                yield Static("Repositories", classes="remote-pane-title")
                yield VerticalScroll(
                    *self._result_widgets(disabled=disabled),
                    id="remote-model-results",
                    can_focus=False,
                )
            with Vertical(classes="remote-detail-pane"):
                back = Button(
                    "Back to repositories",
                    id="remote-back-to-results",
                )
                back.display = self.has_class("-single-pane")
                yield back
                yield Static(
                    self._default_status(),
                    id="remote-model-status",
                    markup=False,
                )
                progress = ModelInstallProgress(
                    self._progress,
                    id="remote-model-install-progress",
                )
                progress.display = self._progress is not None
                yield progress
                yield VerticalScroll(
                    *self._detail_widgets(disabled=disabled),
                    id="remote-model-details",
                )
                with Horizontal(classes="remote-action-bar"):
                    yield Static(
                        self._selection_summary(),
                        id="remote-model-selection",
                        markup=False,
                    )
                    if self._completed_reference is None:
                        yield Button(
                            "Review and install…",
                            id="remote-model-install",
                            variant="primary",
                            disabled=disabled or self._selected_candidate is None,
                        )
                    else:
                        with Horizontal(classes="remote-completion-actions"):
                            yield Button(
                                "Open Installed",
                                id="remote-model-open-installed",
                                variant="default",
                            )
                            yield Button(
                                "Configure and use…",
                                id="remote-model-configure-runtime",
                                variant="primary",
                            )

    def on_mount(self) -> None:
        """Apply the measured compact layout without creating dependencies."""
        self._set_responsive(self.size.width or self.app.size.width)

    def on_resize(self, event: events.Resize) -> None:
        """Keep the internal browser usable inside the real Models body."""
        self._set_responsive(event.size.width)

    def _set_responsive(self, width: int) -> None:
        """Stack the action bar and release pane floors below the breakpoint."""
        single_pane = width < _SINGLE_PANE_WIDTH
        self.set_class(single_pane, "-narrow")
        self.set_class(single_pane, "-single-pane")
        if not self._machine_details_touched:
            self._machine_details_expanded = not single_pane
        self._apply_pane_visibility()
        self._apply_machine_details_visibility()

    def _apply_pane_visibility(self) -> None:
        """Show both panes or the selected narrow drill-down pane in place."""
        try:
            results = self.query_one(".remote-results-pane", Vertical)
            details = self.query_one(".remote-detail-pane", Vertical)
        except NoMatches:
            return
        single_pane = self.has_class("-single-pane")
        results.display = not single_pane or not self._single_pane_show_detail
        details.display = not single_pane or self._single_pane_show_detail
        for back in self.query("#remote-back-to-results").results(Button):
            back.display = single_pane

    def _show_repository_results(self, *, restore_focus: bool = True) -> None:
        """Show narrow results and restore the exact retained repository control."""
        self._single_pane_show_detail = False
        self._apply_pane_visibility()
        if not restore_focus:
            return
        repository = self._repository_focus_locator
        for button in self.query(".remote-result").results(Button):
            if getattr(button, "repository", None) == repository:
                self.call_later(self.screen.set_focus, button)
                return
        try:
            query = self.query_one("#remote-model-query", Input)
        except NoMatches:
            return
        self.call_later(self.screen.set_focus, query)

    def _show_repository_detail(self, repository: str | None = None) -> None:
        """Show narrow detail while retaining a bounded result focus locator."""
        if repository is not None:
            self._repository_focus_locator = repository
        self._single_pane_show_detail = True
        self._apply_pane_visibility()

    def _default_status(self) -> str:
        if self._completed_reference is not None:
            return "Downloaded · Verified · Managed · Not active"
        if self._selected_candidate is not None:
            return (
                "GGUF variant selected. Review the managed install before downloading."
            )
        if self._resolved is not None:
            return (
                f"Pinned {self._resolved.repository} at "
                f"{self._resolved.commit}. Select one GGUF candidate."
            )
        if self._results:
            return "Select a repository to inspect its eligible GGUF files."
        return "Search runs only when you press Search."

    def _selection_summary(self) -> str:
        """Describe the selected GGUF candidate beside the install action."""
        if self._completed_reference is not None:
            return "Choose where to configure the verified managed model."
        if self._selected_candidate is None:
            return "Select one GGUF file or complete shard set."
        return (
            f"Selected: {_candidate_primary_filename(self._selected_candidate)} · "
            f"{format_mib(self._selected_candidate.total_bytes)}"
        )

    def _result_widget(
        self, summary: RemoteModelSummary, *, disabled: bool
    ) -> Vertical:
        button = Button(
            "Inspect GGUF files",
            classes="remote-result",
            variant=(
                "primary"
                if summary.repository == self._selected_repository
                else "default"
            ),
            disabled=disabled,
        )
        button.repository = summary.repository
        access = "Private" if summary.private else "Public"
        metrics = (
            f"{_format_count(summary.downloads)} downloads · "
            f"{_format_count(summary.likes)} likes"
        )
        updated = _format_last_modified(summary.last_modified)
        return Vertical(
            Static(summary.repository, classes="remote-title", markup=False),
            Static(metrics, classes="remote-muted", markup=False),
            Static(updated, classes="remote-muted", markup=False),
            Static(f"{access} · Gated: {summary.gated}", markup=False),
            button,
            classes="remote-result-row",
        )

    def _result_widgets(self, *, disabled: bool):
        if not self._results:
            yield Static(
                "Search by model name, architecture, or publisher.",
                classes="remote-muted",
                markup=False,
            )
        for summary in self._results:
            yield self._result_widget(summary, disabled=disabled)

    def _detail_widgets(self, *, disabled: bool):
        if self._resolved is not None:
            yield from self._resolved_widgets(self._resolved, disabled=disabled)
            return
        if self._selected_repository is not None:
            yield Static(
                f"Inspecting {self._selected_repository}",
                classes="remote-title",
                markup=False,
            )
            yield Static(
                "Reading pinned repository metadata and eligible GGUF files…",
                classes="remote-muted",
                markup=False,
            )
            return
        yield Static("Select a model", classes="remote-title", markup=False)
        yield Static(
            (
                "Search on the left, then inspect a repository without losing "
                "your result list. You can also paste an exact owner/repository ID."
            ),
            classes="remote-muted",
            markup=False,
        )

    def _resolved_widgets(
        self,
        resolved: ResolvedRemoteModel,
        *,
        disabled: bool,
    ):
        yield Static(
            "Runtime compatibility has not been verified.",
            classes="remote-compatibility",
            markup=False,
        )
        yield Static(resolved.repository, classes="remote-title", markup=False)
        yield Static(f"Pinned commit: {resolved.commit}", markup=False)
        license_label = (
            "Unknown / not declared"
            if resolved.license_id == "NOASSERTION"
            else resolved.license_id
        )
        yield Static(f"License: {license_label}", markup=False)
        yield Static("Provenance: Local integrity recorded", markup=False)
        yield from self._machine_panel_widgets()
        yield Static("Available GGUF files", classes="remote-detail-section")
        if resolved.total_candidate_count > len(resolved.candidates):
            yield Static(
                f"First {len(resolved.candidates)} of "
                f"{resolved.total_candidate_count}, sorted by upstream path",
                markup=False,
            )
        for warning in resolved.warnings:
            yield Static(f"Incomplete shard set: {warning}", markup=False)
        yield Static(
            (
                "Filename-derived general guidance. Machine memory is estimated "
                "below; model-context support and runtime compatibility remain "
                "unverified."
            ),
            classes="remote-variant-guidance",
            markup=False,
        )
        yield Horizontal(
            Input(
                value=self._variant_filter,
                placeholder="Filter by filename or quantization",
                id="remote-variant-filter",
                disabled=disabled,
            ),
            Select(
                (
                    ("Source order", "source"),
                    ("Size: smallest first", "size-asc"),
                    ("Size: largest first", "size-desc"),
                    ("Quantization", "quantization"),
                ),
                value=self._variant_sort,
                allow_blank=False,
                compact=True,
                id="remote-variant-sort",
                disabled=disabled,
            ),
            classes="remote-variant-controls",
        )
        yield Vertical(
            *self._variant_widgets(resolved, disabled=disabled),
            classes="remote-variant-list",
        )
        yield Static(
            f"Source review page: {resolved.review_url}",
            classes="remote-source-review",
            markup=False,
        )

    def _variant_guidance_rows(
        self, resolved: ResolvedRemoteModel
    ) -> tuple[VariantGuidance, ...]:
        """Derive locally filtered and sorted guidance for resolved candidates."""
        rows: list[VariantGuidance] = []
        for index, candidate in enumerate(resolved.candidates):
            filenames = tuple(item.upstream_path for item in candidate.files)
            primary_filename = filenames[0] if filenames else candidate.label
            rows.append(
                variant_guidance(
                    primary_filename,
                    total_bytes=candidate.total_bytes,
                    file_count=len(candidate.files),
                    source_index=index,
                    filenames=filenames[1:],
                )
            )
        filtered = filter_variant_guidance(rows, self._variant_filter)
        return sort_variant_guidance(filtered, self._variant_sort)

    def _variant_widgets(
        self,
        resolved: ResolvedRemoteModel,
        *,
        disabled: bool,
    ):
        """Yield candidate rows from current local filter and sort state."""
        rows = self._variant_guidance_rows(resolved)
        if not rows:
            yield Static(
                "No GGUF variants match this filter.",
                classes="remote-variant-empty remote-muted",
                markup=False,
            )
            return
        for row in rows:
            candidate = resolved.candidates[row.source_index]
            selected = candidate == self._selected_candidate
            button = Button(
                "Selected variant" if selected else "Select variant",
                classes="remote-candidate",
                variant="primary" if selected else "default",
                disabled=disabled,
            )
            button.candidate = candidate
            file_set_label = (
                "1 file" if row.file_count == 1 else f"{row.file_count} shards"
            )
            quantization = row.quantization or "Not identified"
            yield Vertical(
                Static(
                    row.filename,
                    classes="remote-title remote-variant-filename",
                    markup=False,
                ),
                Static(
                    f"Quantization: {quantization} · {file_set_label} · "
                    f"{format_mib(row.total_bytes)}",
                    classes="remote-variant-facts",
                    markup=False,
                ),
                Static(
                    row.summary,
                    classes="remote-muted remote-variant-summary",
                    markup=False,
                ),
                *self._candidate_memory_widgets(row.source_index, candidate),
                button,
                classes="remote-candidate-row",
            )

    def _machine_panel_widgets(self):
        presentation = self._machine_presentation
        evidence = "\n".join(presentation.evidence_lines)
        details = "\n".join(
            (*presentation.limitation_lines, *presentation.accelerator_detail_lines)
        )
        details_static = Static(
            details,
            id="remote-machine-estimate-details",
            markup=False,
        )
        details_static.display = self._machine_details_expanded and bool(details)
        toggle = Button(
            "Hide estimate details"
            if self._machine_details_expanded
            else "Show estimate details",
            id="remote-machine-details-toggle",
        )
        toggle.display = bool(details)
        yield Vertical(
            Static(
                presentation.headline,
                id="remote-machine-headline",
                markup=False,
            ),
            Static(
                evidence,
                id="remote-machine-evidence",
                markup=False,
            ),
            Static(
                presentation.failure_line or "",
                id="remote-machine-failure",
                markup=False,
            ),
            details_static,
            Horizontal(
                Button(
                    presentation.action_label,
                    id="remote-machine-recheck",
                    disabled=presentation.action_disabled,
                ),
                toggle,
                classes="remote-machine-actions",
            ),
            classes="remote-machine-panel",
        )

    def _candidate_memory_presentation(self, candidate: RemoteGGUFCandidate):
        projection = (
            project_gguf_memory(candidate.total_bytes, self._machine_snapshot)
            if self._machine_snapshot is not None
            else None
        )
        return build_candidate_memory_presentation(
            projection,
            self._machine_snapshot,
            active=self._machine_presentation.action_disabled,
        )

    def _candidate_memory_widgets(
        self,
        source_index: int,
        candidate: RemoteGGUFCandidate,
    ) -> tuple[Static, Static]:
        presentation = self._candidate_memory_presentation(candidate)
        detail_lines = tuple(
            line
            for line in (
                presentation.details,
                presentation.pressure,
                self._machine_presentation.failure_line,
            )
            if line
        )
        details = Static(
            "\n".join(detail_lines),
            id=f"remote-fit-details-{source_index}",
            classes="remote-fit-details",
            markup=False,
        )
        details.display = self._machine_details_expanded
        return (
            Static(
                presentation.outcome,
                id=f"remote-fit-outcome-{source_index}",
                classes="remote-fit-outcome",
                markup=False,
            ),
            details,
        )

    def apply_machine_memory_state(
        self,
        presentation: MachineMemoryPresentation,
        snapshot: MachineMemorySnapshot | None,
    ) -> None:
        """Apply immutable screen-owned evidence without replacing controls."""
        if type(presentation) is not MachineMemoryPresentation:
            return
        if snapshot is not None and type(snapshot) is not MachineMemorySnapshot:
            return
        self._machine_presentation = presentation
        self._machine_snapshot = snapshot
        self._update_machine_panel_in_place()
        self._update_candidate_memory_statics_in_place()

    def _update_machine_panel_in_place(self) -> None:
        try:
            headline = self.query_one("#remote-machine-headline", Static)
            evidence = self.query_one("#remote-machine-evidence", Static)
            failure = self.query_one("#remote-machine-failure", Static)
            details = self.query_one("#remote-machine-estimate-details", Static)
            recheck = self.query_one("#remote-machine-recheck", Button)
            toggle = self.query_one("#remote-machine-details-toggle", Button)
        except NoMatches:
            return
        presentation = self._machine_presentation
        detail_text = "\n".join(
            (*presentation.limitation_lines, *presentation.accelerator_detail_lines)
        )
        headline.update(presentation.headline)
        evidence.update("\n".join(presentation.evidence_lines))
        failure.update(presentation.failure_line or "")
        details.update(detail_text)
        recheck.label = presentation.action_label
        recheck.disabled = presentation.action_disabled
        toggle.display = bool(detail_text)
        self._apply_machine_details_visibility()

    def _update_candidate_memory_statics_in_place(self) -> None:
        if self._resolved is None:
            return
        for source_index, candidate in enumerate(self._resolved.candidates):
            try:
                outcome = self.query_one(
                    f"#remote-fit-outcome-{source_index}", Static
                )
                details = self.query_one(
                    f"#remote-fit-details-{source_index}", Static
                )
            except NoMatches:
                continue
            presentation = self._candidate_memory_presentation(candidate)
            outcome.update(presentation.outcome)
            details.update(
                "\n".join(
                    line
                    for line in (
                        presentation.details,
                        presentation.pressure,
                        self._machine_presentation.failure_line,
                    )
                    if line
                )
            )
            details.display = self._machine_details_expanded

    def _apply_machine_details_visibility(self) -> None:
        try:
            details = self.query_one("#remote-machine-estimate-details", Static)
            toggle = self.query_one("#remote-machine-details-toggle", Button)
        except NoMatches:
            return
        detail_text = str(details.renderable)
        details.display = self._machine_details_expanded and bool(detail_text)
        toggle.display = bool(detail_text)
        toggle.label = (
            "Hide estimate details"
            if self._machine_details_expanded
            else "Show estimate details"
        )
        for candidate_details in self.query(".remote-fit-details").results(Static):
            candidate_details.display = self._machine_details_expanded

    @on(Button.Pressed, "#remote-machine-details-toggle")
    def _machine_details_toggle_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._machine_details_touched = True
        self._machine_details_expanded = not self._machine_details_expanded
        self._apply_machine_details_visibility()

    @on(Button.Pressed, "#remote-machine-recheck")
    def _machine_recheck_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._machine_presentation.action_disabled:
            return
        presentation_snapshot = (
            self._machine_snapshot
            if self._machine_snapshot is not None
            and self._machine_snapshot.total_bytes is not None
            else None
        )
        self.apply_machine_memory_state(
            build_machine_memory_presentation(presentation_snapshot, active=True),
            self._machine_snapshot,
        )
        self.post_message(self.MachineMemoryRequested(force=True))

    def _refresh_variant_list(self) -> None:
        """Refresh only candidate rows so local controls retain keyboard focus."""
        if self._resolved is None:
            return
        try:
            variant_list = self.query_one(".remote-variant-list", Vertical)
        except NoMatches:
            return
        disabled = self._operation_reference is not None
        variant_list.remove_children()
        variant_list.mount(
            *self._variant_widgets(self._resolved, disabled=disabled)
        )

    def _set_status(self, message: str) -> None:
        self.query_one("#remote-model-status", Static).update(message)

    def _refresh_with_status(self, message: str) -> None:
        disabled = self._operation_reference is not None
        results = self.query_one("#remote-model-results", VerticalScroll)
        results.remove_children()
        results.mount(*self._result_widgets(disabled=disabled))
        self._refresh_details_with_status(message)

    def _refresh_details_with_status(self, message: str) -> None:
        """Update only the right pane so browse context and focus remain stable."""
        self._set_status(message)
        disabled = self._operation_reference is not None
        details = self.query_one("#remote-model-details", VerticalScroll)
        details.remove_children()
        details.mount(*self._detail_widgets(disabled=disabled))
        self.query_one("#remote-model-selection", Static).update(
            self._selection_summary()
        )
        try:
            install = self.query_one("#remote-model-install", Button)
        except NoMatches:
            self.refresh(recompose=True)
            return
        install.disabled = disabled or self._selected_candidate is None

    def _set_selected_result_variant(self) -> None:
        """Paint repository selection without replacing the focused result row."""
        for button in self.query(".remote-result").results(Button):
            button.variant = (
                "primary"
                if getattr(button, "repository", None) == self._selected_repository
                else "default"
            )

    def _set_search_controls_disabled(self, disabled: bool) -> None:
        """Disable query submission without disturbing focused result rows."""
        for control in self.query("#remote-model-query, #remote-model-search"):
            control.disabled = disabled

    def _set_metadata_controls_disabled(self, disabled: bool) -> None:
        for control in self.query(
            "#remote-model-query, #remote-model-search, "
            "#remote-variant-filter, #remote-variant-sort"
        ):
            control.disabled = disabled
        for button in self.query(
            ".remote-result, .remote-candidate, #remote-model-install"
        ).results(Button):
            button.disabled = disabled

    @on(Button.Pressed, "#remote-model-search")
    @on(Input.Submitted, "#remote-model-query")
    def _search_submitted(self) -> None:
        if self._operation_reference is not None:
            return
        query = self.query_one("#remote-model-query", Input).value.strip()
        self._query_value = query
        self._show_repository_results(restore_focus=False)
        self.post_message(self.DiscoveryStarted(query))
        was_completed = self._completed_reference is not None
        self._search_generation += 1
        self._resolve_generation += 1
        self._results = ()
        self._resolved = None
        self._selected_repository = None
        self._selected_candidate = None
        self._variant_filter = ""
        self._completed_reference = None
        if was_completed:
            self.refresh(recompose=True)
            self.call_after_refresh(
                self._begin_discovery,
                query,
                self._search_generation,
                self._resolve_generation,
            )
            return
        self._begin_discovery(
            query,
            self._search_generation,
            self._resolve_generation,
        )

    def _begin_discovery(
        self,
        query: str,
        search_generation: int,
        resolve_generation: int,
    ) -> None:
        """Start a current discovery after any completion-state recompose."""
        if (
            search_generation != self._search_generation
            or resolve_generation != self._resolve_generation
            or query != self._query_value
        ):
            return
        self._set_metadata_controls_disabled(True)
        if is_exact_repository(query):
            self._selected_repository = query
            self._repository_focus_locator = None
            self._show_repository_detail()
            self._refresh_with_status("Inspecting repository…")
            self._resolve_remote(query, resolve_generation, query)
            return
        self._refresh_with_status("Searching remote models…")
        self._search_remote(query, search_generation)

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
        self._selected_repository = repository
        self._repository_focus_locator = repository
        self._selected_candidate = None
        self._variant_filter = ""
        self._set_search_controls_disabled(True)
        self._set_selected_result_variant()
        self._show_repository_detail(repository)
        self._refresh_details_with_status("Inspecting repository…")
        relevant_input = self.query_one("#remote-model-query", Input).value.strip()
        self._resolve_remote(repository, self._resolve_generation, relevant_input)

    @on(Button.Pressed, ".remote-candidate")
    def _candidate_pressed(self, event: Button.Pressed) -> None:
        """Select one current GGUF candidate without starting acquisition."""
        event.stop()
        candidate = getattr(event.button, "candidate", None)
        if (
            type(candidate) is not RemoteGGUFCandidate
            or self._resolved is None
            or candidate not in self._resolved.candidates
            or self._operation_reference is not None
        ):
            return
        self._selected_candidate = candidate
        self._set_status(
            "GGUF variant selected. Review the managed install before downloading."
        )
        for button in self.query(".remote-candidate").results(Button):
            selected = getattr(button, "candidate", None) == candidate
            button.variant = "primary" if selected else "default"
            button.label = "Selected variant" if selected else "Select variant"
        self.query_one("#remote-model-selection", Static).update(
            self._selection_summary()
        )
        self.query_one("#remote-model-install", Button).disabled = False

    @on(Button.Pressed, "#remote-back-to-results")
    def _back_to_results_pressed(self, event: Button.Pressed) -> None:
        """Return to narrow repository results with exact focus restoration."""
        event.stop()
        self._show_repository_results()

    @on(Button.Pressed, "#remote-model-install")
    def _install_pressed(self, event: Button.Pressed) -> None:
        """Post an install intent for the selected candidate; never acquire here."""
        event.stop()
        candidate = self._selected_candidate
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
        try:
            service = self._service_factory()
            credential_resolver = self._credential_resolver_factory()
        except Exception as exc:
            logger.error(
                "Remote install dependency construction failed; error_type={}",
                type(exc).__name__,
            )
            message = (
                "Could not prepare the managed install. Check model storage "
                "settings and try again."
            )
            self.notify(message, severity="error")
            self._set_status(message)
            return
        self._operation_reference = catalog.artifact.reference
        self._set_metadata_controls_disabled(True)
        self._set_status("Preparing the managed install plan…")
        self.post_message(
            self.InstallRequested(
                catalog,
                candidate,
                service=service,
                credential_resolver=credential_resolver,
            )
        )

    @on(Button.Pressed, "#remote-model-open-installed")
    def _open_installed_pressed(self, event: Button.Pressed) -> None:
        """Request the exact completed model in the persistent inventory."""
        event.stop()
        reference = self._completed_reference
        if type(reference) is ArtifactRef:
            self.post_message(self.OpenInstalledRequested(reference))

    @on(Button.Pressed, "#remote-model-configure-runtime")
    def _configure_runtime_pressed(self, event: Button.Pressed) -> None:
        """Request host-owned runtime choice for the completed model."""
        event.stop()
        reference = self._completed_reference
        if type(reference) is ArtifactRef:
            self.post_message(self.ConfigureRuntimeRequested(reference))

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
            self._selected_repository = None
            self._selected_candidate = None
            self._refresh_with_status(_discovery_error_message(error))
            return
        self._results = results
        self._resolved = None
        self._selected_repository = None
        self._selected_candidate = None
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
            self._selected_repository = None
            self._selected_candidate = None
            self._set_metadata_controls_disabled(False)
            self._refresh_with_status(
                "Repository selection changed. Press Search to inspect the current ID."
            )
            return
        self._set_metadata_controls_disabled(False)
        if error is not None or resolved is None:
            self._resolved = None
            self._selected_repository = None
            self._selected_candidate = None
            message = _discovery_error_message(error or RuntimeError("resolve failed"))
            if self._results:
                self._set_selected_result_variant()
                self._refresh_details_with_status(message)
            else:
                self._refresh_with_status(message)
            return
        self._resolved = resolved
        self._selected_repository = resolved.repository
        self._selected_candidate = None
        self._variant_filter = ""
        if self._machine_snapshot is None:
            self._machine_presentation = build_machine_memory_presentation(
                None, active=True
            )
        message = (
            f"Pinned {resolved.repository} at {resolved.commit}. "
            "Select one GGUF candidate."
        )
        if self._results:
            self._set_selected_result_variant()
            self._refresh_details_with_status(message)
        else:
            self._refresh_with_status(message)
        self.post_message(self.MachineMemoryRequested(force=False))

    @on(Input.Changed, "#remote-variant-filter")
    def _variant_filter_changed(self, event: Input.Changed) -> None:
        """Apply filename/quantization filtering without provider I/O."""
        self._variant_filter = event.value
        resolved = self._resolved
        if resolved is None:
            return
        visible_candidates = {
            resolved.candidates[row.source_index]
            for row in self._variant_guidance_rows(resolved)
        }
        if (
            self._selected_candidate is not None
            and self._selected_candidate not in visible_candidates
        ):
            self._selected_candidate = None
            self._set_status(
                "Variant filter changed. Choose a visible variant."
            )
            self.query_one("#remote-model-selection", Static).update(
                self._selection_summary()
            )
            self.query_one("#remote-model-install", Button).disabled = True
        self._refresh_variant_list()

    @on(Select.Changed, "#remote-variant-sort")
    def _variant_sort_changed(self, event: Select.Changed) -> None:
        """Apply deterministic local ordering while retaining selection."""
        if not isinstance(event.value, str):
            return
        self._variant_sort = event.value
        self._refresh_variant_list()

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

    def restore_install_context(
        self,
        catalog: ResolvedRemoteCatalog,
        candidate: RemoteGGUFCandidate,
        *,
        status_message: str = "Installing the selected GGUF variant…",
    ) -> bool:
        """Restore selected-model presentation after a screen recompose.

        The host screen retains the already-validated frozen catalog and exact
        candidate for the life of an install. Rebuilding the read-only detail
        state from those values keeps provenance, selection, and progress
        truthful without moving any acquisition responsibility into this view.

        Args:
            catalog: Frozen one-item catalog retained by ``LLMScreen``.
            candidate: Exact GGUF candidate retained alongside the catalog.
            status_message: Truthful host-owned lifecycle copy to restore.

        Returns:
            Whether the retained values reproduced the exact frozen catalog.
        """
        if (
            type(catalog) is not ResolvedRemoteCatalog
            or type(candidate) is not RemoteGGUFCandidate
        ):
            return False
        descriptor = catalog.artifact
        repository = descriptor.upstream_repository
        commit = descriptor.upstream_revision
        review_url = descriptor.license_url
        if not all(
            isinstance(value, str) and value
            for value in (repository, commit, review_url, descriptor.license_id)
        ):
            return False
        resolved = ResolvedRemoteModel(
            repository=repository,
            commit=commit,
            license_id=descriptor.license_id,
            review_url=review_url,
            candidates=(candidate,),
            total_candidate_count=1,
            warnings=(),
        )
        try:
            rebuilt = build_remote_catalog(resolved, candidate)
        except ValueError:
            return False
        if rebuilt != catalog:
            return False
        if (
            self._resolved == resolved
            and self._selected_candidate == candidate
            and self._operation_reference == descriptor.reference
        ):
            self._set_status(status_message)
            self._set_metadata_controls_disabled(True)
            return True
        self._resolved = resolved
        self._selected_repository = repository
        self._selected_candidate = candidate
        self._operation_reference = descriptor.reference
        self._refresh_with_status(status_message)
        self._set_metadata_controls_disabled(True)
        return True

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
        if message is None:
            self._selected_candidate = None
            self._refresh_details_with_status(self._default_status())
            return
        self._set_status(message)

    def finish_install(
        self,
        message: str | None = None,
        *,
        completed_reference: ArtifactRef | None = None,
    ) -> None:
        """Clear the in-flight indicator and hide progress after a completed install.

        Called by the host screen (``LLMScreen``) once provisioning
        finishes, successfully or not.

        Args:
            message: The outcome copy to show (success or sanitized
                failure); ``None`` restores the default status.
            completed_reference: Exact verified managed root on success.
                ``None`` retains the failure presentation without adoption
                actions.
        """
        self._operation_reference = None
        self._completed_reference = (
            completed_reference
            if type(completed_reference) is ArtifactRef
            else None
        )
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
        if self._completed_reference is not None:
            self.refresh(recompose=True)
            return
        self._set_status(message or self._default_status())


def _format_count(value: int | None) -> str:
    """Format bounded Hugging Face counters for compact result metadata."""
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _format_last_modified(value: str | None) -> str:
    """Normalize a printable ISO timestamp before exposing its calendar date."""
    if not isinstance(value, str) or not value or not value.isprintable():
        return "Updated —"
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return "Updated —"
    return f"Updated {parsed.date().isoformat()}"


def _candidate_primary_filename(candidate: RemoteGGUFCandidate) -> str:
    """Return the first exact file path in a selectable candidate set."""
    return candidate.files[0].upstream_path if candidate.files else candidate.label


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
