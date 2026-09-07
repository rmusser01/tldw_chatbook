"""Lazy Curated view backed by the verified managed-model acquisition path."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import DescendantFocus
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Static

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
from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
    AudioCppArtifactRemovalEvidence,
    AudioCppModelLibraryObservationSnapshot,
)
from tldw_chatbook.UI.Screens.model_browser_state import provenance_label
from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress


_AUDIO_CPP_CONFIGURED_UNKNOWN = "Unknown — Settings state was not checked"
_AUDIO_CPP_RUNNING_UNKNOWN = "Unknown — supervisor state was not checked"


@dataclass(frozen=True)
class AudioCppPackageProjection:
    """Immutable recipe facts safe to render without registry access."""

    recipe: str
    compatibility: str
    availability: str
    companion_paths: tuple[str, ...]
    speech_tasks: str
    required_files: str
    pinned_source: str
    manifest_authority: str
    package_size: str
    configured: str = _AUDIO_CPP_CONFIGURED_UNKNOWN
    running: str = _AUDIO_CPP_RUNNING_UNKNOWN


AudioCppObservationProvider = Callable[
    [tuple[ArtifactRef, ...]], Awaitable[AudioCppModelLibraryObservationSnapshot]
]
ModelLibraryFocusLocator = str | tuple[ArtifactRef, str]


def model_library_focus_locator(
    view: Widget,
    widget: Widget,
    *,
    row_selector: str,
    action_class: str,
    action_role: str,
) -> ModelLibraryFocusLocator | None:
    """Identify one header or exact-ref model-library control.

    Args:
        view: Model Library view containing the candidate control.
        widget: Candidate focused widget.
        row_selector: Selector for rows carrying exact artifact references.
        action_class: Class identifying the row action to remember.
        action_role: Semantic role assigned to that row action.

    Returns:
        A widget ID, an exact-reference role tuple, or ``None`` when the
        widget has no restorable Model Library role.
    """

    if widget.id:
        return widget.id
    for row in view.query(row_selector):
        reference = getattr(row, "reference", None)
        if (
            not isinstance(reference, ArtifactRef)
            or row not in widget.ancestors_with_self
        ):
            continue
        if isinstance(widget, Button) and widget.has_class(action_class):
            return (reference, action_role)
        if any(
            disclosure in widget.ancestors_with_self
            for disclosure in row.query(".audio-cpp-companions")
        ):
            return (reference, "disclosure")
    return None


def restore_model_library_focus(
    view: Widget,
    locator: ModelLibraryFocusLocator,
    *,
    row_selector: str,
    action_role: str,
    action_selector: str,
) -> None:
    """Restore one header or exact-ref control after a pane change.

    Args:
        view: Model Library view containing the replacement controls.
        locator: Previously captured widget ID or exact-reference role.
        row_selector: Selector for rows carrying exact artifact references.
        action_role: Semantic role identifying the row action.
        action_selector: Selector for that row action.

    Returns:
        None.
    """

    if isinstance(locator, str):
        try:
            target = view.query_one(f"#{locator}")
        except NoMatches:
            return
    else:
        reference, role = locator
        row = next(
            (
                item
                for item in view.query(row_selector)
                if getattr(item, "reference", None) == reference
            ),
            None,
        )
        if row is None:
            return
        try:
            item = row.query_one(
                action_selector if role == action_role else ".audio-cpp-companions"
            )
            target = (
                item.query_one("CollapsibleTitle") if role == "disclosure" else item
            )
        except NoMatches:
            return
    if getattr(target, "disabled", False):
        return
    view.screen.set_focus(target)
    target.scroll_visible(animate=False, immediate=True, force=True)


def project_audio_cpp_observation(
    projection: AudioCppPackageProjection,
    reference: ArtifactRef,
    evidence: AudioCppArtifactRemovalEvidence,
) -> AudioCppPackageProjection:
    """Apply only exact-reference Settings and live-supervisor evidence.

    Args:
        projection: Existing package projection to update.
        reference: Exact artifact reference expected by the projection.
        evidence: Settings and runtime observation to apply.

    Returns:
        The updated projection, or the original projection when the evidence
        belongs to another artifact reference.
    """

    if evidence.reference != reference:
        return projection
    scopes = {scope for scope, _label, _identity in evidence.settings_consumers}
    configured_parts = []
    if "saved" in scopes:
        configured_parts.append("Saved Settings")
    if "draft" in scopes:
        configured_parts.append("detached draft")
    configured = (
        " + ".join(configured_parts)
        if configured_parts
        else "Not configured — exact Settings state checked"
    )
    running = (
        "Applied supervisor generation is active"
        if evidence.live_runtime_ids
        else "Not running — applied supervisor state checked"
    )
    return replace(projection, configured=configured, running=running)


def clear_audio_cpp_observation(
    projection: AudioCppPackageProjection,
) -> AudioCppPackageProjection:
    """Discard prior-generation Settings/runtime claims while refreshing.

    Args:
        projection: Existing package projection.

    Returns:
        A projection with Configured and Running facts reset to unknown.
    """

    return replace(
        projection,
        configured=_AUDIO_CPP_CONFIGURED_UNKNOWN,
        running=_AUDIO_CPP_RUNNING_UNKNOWN,
    )


def audio_cpp_package_projection(
    descriptor: ArtifactDescriptor,
) -> AudioCppPackageProjection | None:
    """Join a reviewed audio.cpp descriptor to its frozen recipe facts.

    Args:
        descriptor: Curated or installed artifact descriptor to project.

    Returns:
        Truthful audio.cpp package facts, or ``None`` for another consumer.
    """

    if descriptor.consumer != "audio_cpp":
        return None
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        AUDIO_CPP_ARTIFACT_COMMIT,
        AUDIO_CPP_ARTIFACT_REPOSITORY,
        audio_cpp_artifact_identity_matches_recipe,
        audio_cpp_artifact_source_matches_descriptor,
    )
    from tldw_chatbook.TTS.audio_cpp_recipes import AUDIO_CPP_RECIPE_REGISTRY

    recipe = next(
        (
            item
            for item in AUDIO_CPP_RECIPE_REGISTRY.recipes
            if descriptor.reference.artifact_id in item.model_library_artifact_ids
        ),
        None,
    )
    exact = recipe is not None and audio_cpp_artifact_identity_matches_recipe(
        recipe_id=recipe.recipe_id,
        recipe_revision=recipe.recipe_revision,
        package_variant=recipe.package_variant,
        recipe_artifact_ids=recipe.model_library_artifact_ids,
        recipe_precision=recipe.precision,
        artifact_id=descriptor.reference.artifact_id,
        revision=descriptor.reference.revision,
        variant=descriptor.reference.variant,
    )
    if not exact or not audio_cpp_artifact_source_matches_descriptor(descriptor):
        return AudioCppPackageProjection(
            recipe="Unknown — exact catalog identity does not match a reviewed recipe",
            compatibility="Unknown — exact package identity needs review",
            availability=(
                "Unknown — exact catalog identity or pinned source does not match; "
                "review required"
            ),
            companion_paths=(),
            speech_tasks="Unknown — exact package identity needs review",
            required_files="Unknown — exact package identity needs review",
            pinned_source="Unknown — exact pinned source needs review",
            manifest_authority="Unknown — exact manifest authority needs review",
            package_size="Unknown — exact package identity needs review",
        )
    assert recipe is not None
    evidence = ", ".join(
        f"{item.system}/{item.architecture}/{item.backend.value}: {item.state.value}"
        for item in recipe.backend_evidence
    )
    model_path = recipe.projection.model_relative_path
    companions = tuple(
        item.relative_path
        for item in (*recipe.required_files, *recipe.optional_files)
        if item.relative_path != model_path
    )
    return AudioCppPackageProjection(
        recipe=(
            f"{recipe.recipe_id}@{recipe.recipe_revision} — exact manifest mapping "
            "recorded; installed scan not checked"
        ),
        compatibility=(
            "Catalog tuple evidence — exact installed package not checked · "
            f"{recipe.audio_cpp_release} · {evidence}"
        ),
        availability=("Complete pinned source recorded; live reachability not checked"),
        companion_paths=companions,
        speech_tasks=", ".join(item.upper() for item in recipe.capabilities),
        required_files=", ".join(item.path for item in descriptor.files),
        pinned_source=f"{AUDIO_CPP_ARTIFACT_REPOSITORY}@{AUDIO_CPP_ARTIFACT_COMMIT}",
        manifest_authority="Pinned sizes and SHA-256 digests recorded",
        package_size=f"{descriptor.expected_installed_bytes:,} bytes",
    )


@dataclass(frozen=True)
class CuratedRow:
    """One curated descriptor cross-referenced with installed inventory."""

    descriptor: ArtifactDescriptor
    installed: bool
    audio_cpp: AudioCppPackageProjection | None = None


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
        observation_provider: AudioCppObservationProvider | None = None,
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
        self._observation_provider = observation_provider
        self._service: ModelArtifactService | None = None
        self._registry: CuratedRegistry | None = None
        self._rows: tuple[CuratedRow, ...] = ()
        self._loaded = False
        self._loading = False
        self._load_error: str | None = None
        self._operation_reference: ArtifactRef | None = None
        self._progress: "AcquisitionProgress | None" = None
        self._consumer_filter: str | None = None
        self._allow_installed_return = False
        self._restore_focus_after_load = False
        self._recovery_message: str | None = None
        self._recovery_reference: ArtifactRef | None = None
        self._observation_generation = 0
        self._observation_focus_locator: ModelLibraryFocusLocator | None = None
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
            yield Static(
                "Open Curated to load the offline model catalog.", markup=False
            )

        with VerticalScroll(classes="curated-list", can_focus=False):
            for row in self._rows:
                yield self._row_widget(row)

    def _row_widget(self, row: CuratedRow) -> Vertical:
        """Build one curated model row."""
        descriptor = row.descriptor
        audio_cpp_return = (
            descriptor.consumer == "audio_cpp" and self._allow_installed_return
        )
        install = Button(
            (
                "Retry install…"
                if descriptor.reference == self._recovery_reference
                and not row.installed
                else "Use installed package"
                if row.installed and audio_cpp_return
                else "Installed"
                if row.installed
                else "Review and install…"
            ),
            classes="curated-install",
            variant="primary",
            disabled=(row.installed and not audio_cpp_return)
            or self._operation_reference is not None,
        )
        install.reference = descriptor.reference
        install.already_installed = row.installed
        if row.installed and not audio_cpp_return:
            install.tooltip = (
                "Already installed — open Guided Settings to review this package."
            )
        elif self._operation_reference is not None:
            install.tooltip = "Another model package operation is in progress."
        actions = [install]
        disabled_reasons: list[Widget] = []
        if row.installed and not audio_cpp_return:
            disabled_reasons.append(
                Static(
                    "Installed — open Guided Settings to review this package.",
                    classes="curated-disabled-reason",
                    markup=False,
                )
            )
        elif self._operation_reference is not None:
            disabled_reasons.append(
                Static(
                    "Install unavailable — another model package operation is in progress.",
                    classes="curated-disabled-reason",
                    markup=False,
                )
            )
        if self._external_key(descriptor.reference) is not None:
            use_from_disk = Button(
                "Use from disk…",
                classes="curated-use-from-disk",
                variant="default",
                disabled=self._operation_reference is not None,
            )
            use_from_disk.reference = descriptor.reference
            if self._operation_reference is not None:
                reason = (
                    "Use from disk unavailable — another model package operation "
                    "is in progress."
                )
                use_from_disk.tooltip = reason
                disabled_reasons.append(
                    Static(reason, classes="curated-disabled-reason", markup=False)
                )
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
                f"License: {descriptor.license_id}"
                + (
                    ""
                    if row.audio_cpp is not None
                    else f" · {provenance_label(descriptor.provenance)}"
                ),
                classes="curated-model-muted",
                markup=False,
            ),
        ]
        if descriptor.reference == self._recovery_reference:
            details.append(
                Static(
                    self._recovery_message or "Review and retry this install.",
                    classes="curated-recovery-status",
                    markup=False,
                )
            )
        if row.audio_cpp is not None:
            details.extend(self._audio_cpp_facts(row))
            details.append(
                Vertical(
                    *actions,
                    *disabled_reasons,
                    classes="curated-actions audio-cpp-actions",
                )
            )
            classes = "curated-model-row audio-cpp-model-row"
        else:
            details.append(Horizontal(*actions, classes="curated-actions"))
            details.extend(disabled_reasons)
            classes = "curated-model-row"
        widget = Vertical(*details, classes=classes)
        widget.reference = descriptor.reference
        return widget

    @staticmethod
    def _audio_cpp_facts(row: CuratedRow) -> tuple[Widget, ...]:
        """Render only the immutable audio.cpp package projection."""

        projection = row.audio_cpp
        assert projection is not None
        companions = Collapsible(
            Static("\n".join(projection.companion_paths) or "None", markup=False),
            title=f"Companion files ({len(projection.companion_paths)})",
            classes="audio-cpp-companions",
            collapsed=True,
        )
        return (
            Static(f"Available: {projection.availability}", markup=False),
            Static(
                "Integrity: Not checked this session"
                if row.installed
                else "Integrity: Not checked — package is not installed",
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
                classes="curated-model-muted audio-cpp-package-copy",
                markup=False,
            ),
        )

    def set_consumer_filter(
        self,
        consumer: str | None,
        *,
        allow_installed_return: bool = False,
    ) -> None:
        """Select the optional audio.cpp presentation context."""

        if consumer not in (None, "audio_cpp"):
            raise ValueError("unsupported curated consumer filter")
        if type(allow_installed_return) is not bool:
            raise TypeError("allow_installed_return must be a bool")
        if consumer is None and allow_installed_return:
            raise ValueError("installed return requires the audio.cpp filter")
        if (
            consumer == self._consumer_filter
            and allow_installed_return == self._allow_installed_return
        ):
            return
        self._consumer_filter = consumer
        self._allow_installed_return = allow_installed_return
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
        self._observation_generation += 1
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
            installed_by_ref = {
                item.descriptor.reference: item
                for item in installed
                if item.descriptor is not None
            }
            rows = tuple(
                CuratedRow(
                    descriptor,
                    descriptor.reference in installed_by_ref,
                    audio_cpp_package_projection(descriptor),
                )
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
        if error is None and self._observation_provider is not None:
            self.refresh_observations()
        if self._restore_focus_after_load:
            self._restore_focus_after_load = False
            self.call_after_refresh(self._restore_refresh_focus)
        elif self._recovery_reference is not None:
            self.call_after_refresh(
                self.restore_focus,
                (self._recovery_reference, "install"),
            )

    def refresh_observations(self) -> None:
        """Refresh current exact refs without re-reading catalog or inventory."""

        if self._observation_provider is None:
            return
        references = tuple(
            row.descriptor.reference for row in self._rows if row.audio_cpp is not None
        )
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
        rows = tuple(
            replace(row, audio_cpp=clear_audio_cpp_observation(row.audio_cpp))
            if row.audio_cpp is not None
            else row
            for row in self._rows
        )
        if rows != self._rows:
            self._rows = rows
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
            or tuple(
                row.descriptor.reference
                for row in self._rows
                if row.audio_cpp is not None
            )
            != references
        ):
            return
        self._observation_focus_locator = None
        self._observe_audio_cpp_rows(generation, references, locator)

    @work(group="curated_audio_cpp_observation", exclusive=True, exit_on_error=False)
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
            or tuple(
                row.descriptor.reference
                for row in self._rows
                if row.audio_cpp is not None
            )
            != references
        ):
            return
        evidence_by_ref = {item.reference: item for item in snapshot.observations}
        rows = tuple(
            replace(
                row,
                audio_cpp=project_audio_cpp_observation(
                    row.audio_cpp,
                    row.descriptor.reference,
                    evidence_by_ref[row.descriptor.reference],
                ),
            )
            if row.audio_cpp is not None and row.descriptor.reference in evidence_by_ref
            else row
            for row in self._rows
        )
        if rows == self._rows:
            return
        focused = self.app.focused
        if focused is not None and self in focused.ancestors_with_self:
            locator = self.focus_locator(focused) or locator
        self._rows = rows
        self._update_audio_cpp_observation_facts()
        self.refresh()
        if locator is not None:
            self.restore_focus(locator)

    def _update_audio_cpp_observation_facts(self) -> None:
        """Update the two observed facts without replacing row actions."""

        projections = {
            row.descriptor.reference: row.audio_cpp
            for row in self._rows
            if row.audio_cpp is not None
        }
        for widget in self.query(".audio-cpp-model-row"):
            projection = projections.get(getattr(widget, "reference", None))
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

    def _restore_refresh_focus(self) -> None:
        """Return keyboard focus to the remounted Refresh button."""

        try:
            refresh = self.query_one("#curated-models-refresh", Button)
        except NoMatches:
            return
        refresh.focus()
        refresh.scroll_visible(animate=False, immediate=True, force=True)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Keep keyboard-selected disclosures and actions inside the viewport."""

        event.widget.scroll_visible(animate=False, immediate=True, force=True)

    def focus_locator(self, widget: Widget) -> ModelLibraryFocusLocator | None:
        """Return a stable semantic locator for a focused row control."""

        return model_library_focus_locator(
            self,
            widget,
            row_selector=".curated-model-row",
            action_class="curated-install",
            action_role="install",
        )

    def restore_focus(self, locator: ModelLibraryFocusLocator) -> None:
        """Restore focus to an id or exact-ref row role after recomposition."""

        restore_model_library_focus(
            self,
            locator,
            row_selector=".curated-model-row",
            action_role="install",
            action_selector=".curated-install",
        )

    @on(Button.Pressed, "#curated-models-refresh")
    def _refresh_pressed(self) -> None:
        self._restore_focus_after_load = True
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
        if reference == self._recovery_reference:
            self._recovery_reference = None
            self._recovery_message = None
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

    def cancel_pending_install(self, message: str | None = None) -> None:
        """Clear the in-flight indicator without reloading the catalog.

        Called by the host screen (``LLMScreen``) when a request this view
        posted did not lead to an install actually starting: a preflight
        failure, an explicit decline at the consent modal, or a request
        refused outright because a different install is already running
        (in which case this is the freshly (re)mounted view that just
        clicked Install, not the instance whose install is still in
        flight -- see ``LLMScreen._curated_install_requested``).
        """
        reference = self._operation_reference
        self._operation_reference = None
        if message is not None and reference is not None:
            self._recovery_message = message
            self._recovery_reference = reference
        self.refresh(recompose=True)
        if self._recovery_reference is not None:
            self.call_after_refresh(
                self.call_later,
                self.restore_focus,
                (self._recovery_reference, "install"),
            )

    def finish_install(self, message: str | None = None) -> None:
        """Clear the in-flight indicator and reload after a completed install.

        Called by the host screen (``LLMScreen``) once provisioning
        finishes, successfully or not -- reloading refreshes which rows
        show "Installed" and re-enables Install everywhere else.
        """
        reference = self._operation_reference
        self._operation_reference = None
        if message is not None and reference is not None:
            self._recovery_message = message
            self._recovery_reference = reference
        elif message is None:
            self._recovery_message = None
            self._recovery_reference = None
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
