"""Lazy Curated view backed by the verified managed-model acquisition path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Model_Artifacts.curated_registry import (
    CuratedRegistry,
    curated_registry,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactRef,
    ArtifactRole,
    ModelArtifactService,
)
from tldw_chatbook.Model_Artifacts.store import managed_service
from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import parakeet_reference
from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey
from tldw_chatbook.UI.Screens.model_browser_state import provenance_label
from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress


@dataclass(frozen=True)
class CuratedRow:
    """One curated descriptor cross-referenced with installed inventory."""

    descriptor: ArtifactDescriptor
    installed: bool


class CuratedView(Widget):
    """Browse curated models and request their installation.

    Rendering only, past the catalog load: an Install click posts
    :class:`InstallRequested` and waits to be told what happened via
    :meth:`apply_progress`, :meth:`cancel_pending_install`, or
    :meth:`finish_install`. ``LLMScreen`` owns the actual preflight/
    provision workers (mirroring ``LibraryScreen``'s Parakeet v2 flow,
    ``library_screen.py``'s ``_run_parakeet_v2_preflight``/
    ``_run_parakeet_v2_install``) precisely so a several-hundred-megabyte
    download survives both the consent modal being dismissed and a
    screen-level recompose that tears down and rebuilds this exact view
    mid-install (TASK-1803).

    Before this change, this view owned that worker directly, and a
    screen-level recompose mid-install orphaned it: the torn-down
    instance's own ``post_message`` became a silent no-op, so progress
    stopped reaching the UI unless compensating durable-delivery logic
    (a fallback chain trying this instance, then the live ``CuratedView``,
    then ``LLMManagementWindow``, then the ``Screen``) papered over it.
    Moving the worker to ``LLMScreen`` -- which already survives that same
    recompose to hydrate this view's progress display, see
    ``LLMScreen._hydrate_curated_progress`` -- removes the need for any of
    that: ``LLMScreen`` is never the thing being torn down, so there is no
    orphaned poster left to compensate for.

    This module deliberately never imports ``ArtifactAcquisitionService``
    nor calls ``preflight()``/``provision()`` itself; ``Model_Artifacts.
    acquisition``/``fetch`` are only ever imported inside ``LLMScreen``'s
    own worker methods.
    """

    class InstallRequested(Message):
        """Posted when Install is pressed for a not-yet-installed model."""

        def __init__(
            self,
            reference: ArtifactRef,
            *,
            service: ModelArtifactService,
            registry: CuratedRegistry,
            sources: dict[ArtifactRef, dict[str, str]],
            already_installed: bool = False,
        ) -> None:
            """Carry everything the host screen needs to preflight/provision.

            Args:
                reference: The exact curated model reference to install.
                service: The managed-store service resolved by this view's
                    own (possibly test-injected) ``service_factory`` --
                    captured here so ``LLMScreen``'s worker uses the exact
                    same instance a test constructed this view with,
                    without ``LLMScreen`` needing its own factory-override
                    constructor knobs.
                registry: The curated registry resolved the same way, via
                    ``registry_factory``.
                sources: The file source map for every currently
                    registered curated descriptor (this view's own
                    ``_source_map()``); re-walked by ``preflight()``/
                    ``provision()`` themselves.
            """
            super().__init__()
            self.reference = reference
            self.already_installed = already_installed
            self.service = service
            self.registry = registry
            self.sources = sources

    class UseFromDiskRequested(Message):
        """Posted with the exact curated Parakeet root reference."""

        def __init__(self, reference: ArtifactRef) -> None:
            self.reference = reference
            super().__init__()

    BUNDLED_CSS = """
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

    CuratedView .curated-actions {
        height: 3;
        margin-top: 1;
    }

    CuratedView .curated-actions Button {
        width: auto;
        margin-right: 1;
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
        self._progress: "AcquisitionProgress | None" = None
        self._consumer_filter: str | None = None
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
        audio_cpp = descriptor.consumer == "audio_cpp"
        install = Button(
            (
                "Use installed package"
                if row.installed and audio_cpp
                else "Installed"
                if row.installed
                else "Review and install…"
            ),
            classes="curated-install",
            variant="primary",
            disabled=(row.installed and not audio_cpp)
            or self._operation_reference is not None,
        )
        install.reference = descriptor.reference
        install.already_installed = row.installed
        actions = [install]
        if self._external_key(descriptor.reference) is not None:
            use_from_disk = Button(
                "Use from disk…",
                classes="curated-use-from-disk",
                variant="default",
                disabled=self._operation_reference is not None,
            )
            use_from_disk.reference = descriptor.reference
            actions.append(use_from_disk)
        details: list[Widget] = [
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
        ]
        if audio_cpp:
            details.extend(self._audio_cpp_facts(descriptor))
        details.append(Horizontal(*actions, classes="curated-actions"))
        return Vertical(*details, classes="curated-model-row")

    @staticmethod
    def _audio_cpp_facts(descriptor: ArtifactDescriptor) -> tuple[Static, ...]:
        """Project joined recipe facts for one reviewed audio.cpp descriptor."""

        from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

        recipe = next(
            (
                item
                for item in AUDIO_CPP_RECIPE_REGISTRY.recipes
                if descriptor.reference.artifact_id in item.model_library_artifact_ids
            ),
            None,
        )
        if recipe is None:
            return (
                Static(
                    "Model package only — audiocpp_server is not included",
                    classes="curated-model-muted",
                    markup=False,
                ),
            )
        tasks = ", ".join(
            capability.upper()
            for capability in recipe.capabilities
            if capability in {"tts", "clone"}
        )
        compatibility = ", ".join(
            f"{evidence.system}/{evidence.architecture}/{evidence.backend.value}:"
            f" {evidence.state.value}"
            for evidence in recipe.backend_evidence
        )
        required = ", ".join(item.relative_path for item in recipe.required_files)
        model_path = recipe.projection.model_relative_path
        companions = (
            ", ".join(
                item.relative_path
                for item in (*recipe.required_files, *recipe.optional_files)
                if item.relative_path != model_path
            )
            or "None"
        )
        return (
            Static(f"Speech tasks: {tasks}", markup=False),
            Static(
                f"Recipe variant: {recipe.package_variant}",
                classes="curated-model-muted",
                markup=False,
            ),
            Static(
                f"Compatibility: {recipe.audio_cpp_release} · {compatibility}",
                classes="curated-model-muted",
                markup=False,
            ),
            Static(
                f"Required package files: {required}",
                classes="curated-model-muted",
                markup=False,
            ),
            Static(
                f"Companion files: {companions}",
                classes="curated-model-muted",
                markup=False,
            ),
            Static(
                "Model package only — audiocpp_server is not included",
                classes="curated-model-muted",
                markup=False,
            ),
        )

    def set_consumer_filter(self, consumer: str | None) -> None:
        """Select the optional audio.cpp presentation context."""

        if consumer not in (None, "audio_cpp"):
            raise ValueError("unsupported curated consumer filter")
        if consumer == self._consumer_filter:
            return
        self._consumer_filter = consumer
        if self._loaded:
            self.ensure_loaded(force=True)

    @staticmethod
    def _external_key(reference: ArtifactRef) -> ParakeetSourceKey | None:
        """Map only exact catalog-known Parakeet root references."""

        for key in ParakeetSourceKey:
            if reference == parakeet_reference(key.model_id, key.precision):
                return key
        return None

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
                if descriptor.role is ArtifactRole.ROOT
                and (
                    self._consumer_filter is None
                    or descriptor.consumer == self._consumer_filter
                )
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
        """Post an install intent; never call preflight/provision here."""
        event.stop()
        reference = getattr(event.button, "reference", None)
        if (
            not isinstance(reference, ArtifactRef)
            or self._operation_reference is not None
        ):
            return
        already_installed = getattr(event.button, "already_installed", False)
        if type(already_installed) is not bool:
            return
        self._operation_reference = reference
        self.refresh(recompose=True)
        self.post_message(
            self.InstallRequested(
                reference,
                already_installed=already_installed,
                service=self._service_for_worker(),
                registry=self._registry_for_worker(),
                sources=self._source_map(),
            )
        )

    @on(Button.Pressed, ".curated-use-from-disk")
    def _use_from_disk_pressed(self, event: Button.Pressed) -> None:
        """Post the exact catalog reference; the screen owns the picker."""

        event.stop()
        reference = getattr(event.button, "reference", None)
        if not isinstance(reference, ArtifactRef):
            return
        if self._external_key(reference) is not None:
            self.post_message(self.UseFromDiskRequested(reference))

    def _source_map(self) -> dict[ArtifactRef, dict[str, str]]:
        """Return sources for every descriptor in the curated registry."""
        registry = self._registry_for_worker()
        return {
            descriptor.reference: registry.sources(descriptor.reference)
            for descriptor in registry.list()
        }

    def apply_progress(self, progress: "AcquisitionProgress") -> None:
        """Render one acquisition progress event, retaining it for later.

        Called ONLY by the host screen (``LLMScreen``) -- via its own
        ``InstallProgressed`` handler for a live tick, and via
        ``_hydrate_curated_progress`` to re-apply the last known progress
        to a freshly (re)mounted instance after a screen-level recompose.
        This view has no ``@on(InstallProgressed)`` handler of its own and
        never posts that message itself (TASK-1803 moved the worker that
        used to post it to ``LLMScreen``); ``LLMScreen`` is the sole
        caller, giving exactly one render per tick.

        ``self._progress`` is retained before either branch below runs, so
        a fallback ``refresh(recompose=True)`` still picks up the correct
        value on this view's next (complete) compose pass: this method can
        run (via ``_hydrate_curated_progress``, scheduled by
        ``call_after_refresh``) at a point where this view itself has
        (re)mounted but its own ``ModelInstallProgress`` child has not yet
        finished composing ITS OWN children -- a widget that ``query_one``
        finds but whose own ``update_progress`` call then raises
        ``NoMatches`` reaching into it.

        Args:
            progress: The acquisition progress event to render.
        """
        self._progress = progress
        try:
            widget = self.query_one(
                "#curated-model-install-progress",
                ModelInstallProgress,
            )
            widget.display = True
            widget.update_progress(progress)
        except NoMatches:
            self.refresh(recompose=True)

    def cancel_pending_install(self) -> None:
        """Clear the in-flight indicator without reloading the catalog.

        Called by the host screen (``LLMScreen``) when a request this view
        posted did not lead to an install actually starting: a preflight
        failure, an explicit decline at the consent modal, or a request
        refused outright because a different install is already running
        (in which case this is the freshly (re)mounted view that just
        clicked Install, not the instance whose install is still in
        flight -- see ``LLMScreen._curated_install_requested``).
        """
        self._operation_reference = None
        self.refresh(recompose=True)

    def finish_install(self) -> None:
        """Clear the in-flight indicator and reload after a completed install.

        Called by the host screen (``LLMScreen``) once provisioning
        finishes, successfully or not -- reloading refreshes which rows
        show "Installed" and re-enables Install everywhere else.
        """
        self._operation_reference = None
        self._progress = None
        try:
            progress = self.query_one(
                "#curated-model-install-progress",
                ModelInstallProgress,
            )
        except NoMatches:
            progress = None
        if progress is not None:
            progress.display = False
        self.ensure_loaded(force=True)
