"""Models: the Lab destination's provider and model management screen."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Static
from textual.worker import Worker

from ...Model_Artifacts.service import ArtifactRef, ModelArtifactService
from ...Widgets.ModelArtifacts import (
    InstallProgressed,
    InstallStatusChanged,
    ModelInstallModal,
)
from ..Lab_Modules.lab_server_status import (
    LabServerRow,
    read_server_rows,
    server_row_id,
    server_row_text,
    servers_chip_text,
)
from ..Lab_Modules.lab_workbench import LAB_RAIL_ROW_CLASS
from ..LLM_Management_Window import LLMManagementWindow
from ..Workbench.workbench_state import WorkbenchHeaderState
from .lab_frame import LabInspectorRow, LabScreen, LabStatusChip
from .model_browser_state import install_failure_message
from .model_curated_view import CuratedView

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress, PreflightReport
    from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry

#: (section title, ((view key, label), ...)) in rail order. The view keys are
#: exactly LLMManagementWindow.view_mapping's keys.
MODELS_RAIL_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Local servers",
        (
            ("llama-cpp", "Llama.cpp"),
            ("llamafile", "Llamafile"),
            ("ollama", "Ollama"),
            ("vllm", "vLLM"),
            ("onnx", "ONNX"),
            ("transformers", "Transformers"),
            ("mlx-lm", "MLX-LM"),
        ),
    ),
    (
        "Models",
        (
            ("curated", "Curated"),
            ("installed", "Installed"),
            ("remote", "Remote"),
            ("download-models", "Download Models"),
        ),
    ),
)

#: How often to re-read server liveness. There is deliberately no
#: refresh-on-press: pressing Start does not synchronously create the
#: process -- the event handler assigns it from an async worker -- so a
#: press-triggered read would report "stopped".
LAB_SERVER_POLL_SECONDS = 2.0


class LLMScreen(LabScreen):
    """Models mode: provider rail, legacy management body, server status."""

    def __init__(self, app_instance: "TldwCli", **kwargs: Any) -> None:
        """Create the Models screen.

        Args:
            app_instance: The running application.
            kwargs: Forwarded to ``LabScreen``.
        """
        super().__init__(app_instance, "llm", **kwargs)
        self.llm_window: LLMManagementWindow | None = None
        #: Guards the ``set_interval`` poll below against a screen-level
        #: recompose: ``on_lab_body_ready()`` fires again on every recompose
        #: (see ``LabScreen.recompose()``) so it can rebind the
        #: ``active_view`` watch to the fresh ``LLMManagementWindow``, but
        #: the poll timer is owned by this screen -- not the recomposed
        #: body -- so it already survives the teardown; starting a second
        #: one alongside it would be a real leak.
        self._status_poll_started = False
        self._model_install_active = False
        self._model_install_phase: str | None = None
        self._model_install_succeeded: bool | None = None
        #: The last ``AcquisitionProgress`` this screen has seen for the
        #: active curated install, retained so a freshly (re)mounted
        #: ``CuratedView`` -- a screen-level ``LabScreen.recompose()``
        #: tears down and rebuilds the whole ``LLMManagementWindow``,
        #: ``CuratedView`` included, mid-download -- can be hydrated back
        #: to it immediately instead of starting blank (TASK-596 delta
        #: port). Mirrors ``LibraryScreen``'s own retained
        #: ``_parakeet_v2_install_progress``. Cleared whenever the install
        #: stops (see ``_model_install_status_changed``).
        self._model_install_last_progress: "AcquisitionProgress | None" = None
        #: The curated-install worker currently running (preflight OR
        #: provision -- reassigned when the second phase starts, mirroring
        #: ``LibraryScreen``'s single ``_parakeet_v2_install_worker``
        #: field across its own two phases). Guards
        #: ``_curated_install_requested`` against starting a second,
        #: concurrent install while this one is still in flight -- this
        #: screen owns exactly one ``WorkerManager``, unlike the
        #: view-owned workers TASK-1803 replaced, so this guard now holds
        #: across a screen-level recompose instead of only within one
        #: ``CuratedView`` instance's lifetime.
        self._model_install_worker: Worker | None = None
        #: The reference, service, registry, and source map the currently
        #: running (or about-to-run) curated install needs -- captured
        #: once from the posted ``CuratedView.InstallRequested`` (or, for
        #: ``_model_install_pending_report``, once preflight resolves) and
        #: read back by this screen's own worker methods for as long as
        #: the operation runs, so nothing here depends on the posting
        #: ``CuratedView`` instance still existing by the time provisioning
        #: finishes.
        self._model_install_reference: ArtifactRef | None = None
        self._model_install_service: ModelArtifactService | None = None
        self._model_install_registry: "CuratedRegistry | None" = None
        self._model_install_sources: dict[ArtifactRef, dict[str, str]] | None = None
        self._model_install_pending_report: "PreflightReport | None" = None
        #: Server rows snapshotted for the duration of one
        #: ``refresh_lab_status`` pass; None outside one. See
        #: :meth:`_current_server_rows`.
        self._server_rows_snapshot: tuple[LabServerRow, ...] | None = None

    def _current_server_rows(self) -> tuple[LabServerRow, ...]:
        """Return server liveness, shared across one refresh pass.

        ``refresh_lab_status`` calls three hooks that each need this --
        ``lab_header_state``, ``lab_status_chips`` and
        ``lab_inspector_rows``. Reading independently meant three separate
        ``poll()`` sweeps per tick, and a server exiting between them could
        render a header badge, a chip count and an inspector row that
        disagree with each other on the same frame.

        Returns:
            The snapshot taken by :meth:`refresh_lab_status`, or a fresh
            read when called outside a refresh (e.g. from ``compose``).
        """
        if self._server_rows_snapshot is not None:
            return self._server_rows_snapshot
        return read_server_rows(self.app_instance)

    def refresh_lab_status(self) -> None:
        """Snapshot server liveness once, then refresh through the frame."""
        self._server_rows_snapshot = read_server_rows(self.app_instance)
        try:
            super().refresh_lab_status()
        finally:
            self._server_rows_snapshot = None

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return the Models destination header copy and live readiness.

        The status is derived, not constant: it reads ``running`` while any
        local server tracked by :data:`LAB_SERVER_SOURCES` is alive and
        ``ready`` otherwise. It was hardcoded to ``"ready"``, which made the
        badge decoration wearing a status label -- it could never change, so
        it never told the user anything. ``refresh_lab_status`` re-syncs the
        header on the same poll as the chip, so the two never disagree.

        Returns:
            Header state whose ``status`` reflects current server liveness.
        """
        rows = self._current_server_rows()
        return WorkbenchHeaderState(
            title="Models",
            subtitle="Manage providers, models, and endpoints.",
            status="running" if any(row.running for row in rows) else "ready",
        )

    def lab_status_chips(self) -> tuple[LabStatusChip, ...]:
        """Return server and managed-install status chips.

        Returns:
            Chips summarising local servers and managed-model installation.
        """
        rows = self._current_server_rows()
        if self._model_install_active:
            phase = {
                "fetch": "downloading",
                "pre-verify": "checking",
                "verify-install": "installing",
                "activate": "activating",
            }.get(self._model_install_phase or "", "starting")
        elif self._model_install_succeeded is False:
            phase = "failed"
        else:
            phase = "idle"
        return (
            LabStatusChip(chip_id="servers", text=servers_chip_text(rows)),
            LabStatusChip(
                chip_id="model-install",
                text=f"Model install: {phase}",
            ),
        )

    @on(InstallProgressed)
    def _model_install_progressed(self, event: InstallProgressed) -> None:
        """Keep the Lab chip current, and re-render live progress (TASK-596).

        Forwards the event into whichever ``CuratedView`` is currently
        mounted -- not only the instance mounted when the install started.
        This screen itself is the sole poster of ``InstallProgressed`` for
        a curated install (TASK-1803: see ``_run_curated_provision`` and
        ``_deliver_curated``, which post at ``self.llm_window`` so the
        message bubbles through ``LLMManagementWindow``'s own mirroring
        handler on its way here) -- unlike before, when ``CuratedView``
        posted it and needed a durable fallback path to survive a
        screen-level recompose tearing it down mid-install. This screen
        is never what a recompose tears down, so no such fallback is
        needed any more.
        """
        self._model_install_active = True
        self._model_install_phase = event.progress.phase
        self._model_install_succeeded = None
        self._model_install_last_progress = event.progress
        self.refresh_lab_status()
        view = self._curated_view()
        if view is not None:
            view.apply_progress(event.progress)

    @on(InstallStatusChanged)
    def _model_install_status_changed(self, event: InstallStatusChanged) -> None:
        """Reflect managed-install start and completion in the Lab header."""
        self._model_install_active = event.active
        self._model_install_succeeded = event.succeeded
        if not event.active:
            self._model_install_phase = None
            self._model_install_last_progress = None
        self.refresh_lab_status()

    def _curated_view(self) -> "CuratedView | None":
        """Return the mounted ``CuratedView``, or None if it cannot be found.

        Returns:
            The view, or None when the window has not mounted yet, or a
            screen-level recompose has torn down the previous instance and
            the fresh one is not mounted yet either (see
            ``LabScreen.recompose()``).
        """
        if self.llm_window is None:
            return None
        try:
            return self.llm_window.query_one(CuratedView)
        except NoMatches:
            return None

    # -- Curated model install: this screen owns preflight/provision -----
    #
    # TASK-1803: CuratedView posts CuratedView.InstallRequested and renders
    # what it is told (apply_progress/cancel_pending_install/finish_install);
    # every method below -- resolving the plan, showing the consent modal,
    # and provisioning -- runs here, mirroring LibraryScreen's
    # handle_parakeet_v2_install_requested/_run_parakeet_v2_preflight/
    # _run_parakeet_v2_install. This screen survives the screen-level
    # recompose that used to orphan CuratedView's own worker (see
    # git history for CuratedView._deliver/_progress_screen, both removed
    # now that the thing they compensated for cannot happen), so no
    # durable-delivery fallback is needed: _deliver_curated below always
    # has a live target.

    @on(CuratedView.InstallRequested)
    def _curated_install_requested(self, event: CuratedView.InstallRequested) -> None:
        """Resolve an install plan for a curated model, off the Textual event loop.

        Refuses a second concurrent install outright (mirroring
        ``LibraryScreen.handle_parakeet_v2_install_requested``'s own
        worker-in-flight guard): the requesting ``CuratedView`` already
        disabled its own row before posting this, but only a screen-level
        recompose can hand a *different*, freshly (re)mounted instance a
        chance to post a second request while the first is still running
        -- ``cancel_pending_install()`` releases only that fresh
        instance's own indicator, leaving the still-running install's
        retained state (below) untouched.
        """
        event.stop()
        worker = self._model_install_worker
        if worker is not None and not worker.is_finished:
            self.notify(
                "A curated model install is already running.",
                severity="information",
            )
            view = self._curated_view()
            if view is not None:
                view.cancel_pending_install()
            return
        self._model_install_reference = event.reference
        self._model_install_service = event.service
        self._model_install_registry = event.registry
        self._model_install_sources = event.sources
        self._model_install_worker = self._run_curated_preflight()

    async def _preflight_curated(self, reference: ArtifactRef):
        """Resolve a curated acquisition plan on the worker's event loop.

        Args:
            reference: The exact curated model reference to preflight.

        Returns:
            The immutable ``PreflightReport`` describing the download.
        """
        from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService

        acquisition = ArtifactAcquisitionService(self._model_install_service)
        return await acquisition.preflight(
            reference,
            self._model_install_registry,
            sources=self._model_install_sources,
        )

    @work(thread=True, group="llm_curated_preflight", exit_on_error=False)
    def _run_curated_preflight(self) -> None:
        """Resolve the curated install plan off the Textual event loop."""
        reference = self._model_install_reference
        try:
            report = asyncio.run(  # policy-exception: worker-thread loop
                self._preflight_curated(reference)
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "Curated model preflight failed for {}@{}/{}",
                reference.artifact_id,
                reference.revision,
                reference.variant,
            )
            self.app.call_from_thread(
                self._apply_curated_preflight_result,
                None,
                install_failure_message(exc, model_label=reference.artifact_id),
            )
            return
        self.app.call_from_thread(self._apply_curated_preflight_result, report, None)

    def _apply_curated_preflight_result(
        self,
        report: "PreflightReport | None",
        error: str | None,
    ) -> None:
        """Show the shared consent modal, or a sanitized preflight failure."""
        self._model_install_worker = None
        if error is not None or report is None:
            self.notify(error or "Model preflight failed.", severity="error")
            self._clear_curated_install_state()
            return
        self._model_install_pending_report = report
        descriptor = self._model_install_registry.descriptor(report.root)
        self.app.push_screen(
            ModelInstallModal(report, model_label=descriptor.model_id),
            self._confirm_curated_install,
        )

    def _confirm_curated_install(self, confirmed: bool) -> None:
        """Start provisioning only after explicit consent."""
        if not confirmed:
            self._clear_curated_install_state()
            return
        reference = self._model_install_reference
        if reference is not None:
            self._deliver_curated(InstallStatusChanged(reference, active=True))
        self._model_install_worker = self._run_curated_provision()

    async def _provision_curated(self, report: "PreflightReport"):
        """Provision the consented report on the worker's event loop.

        Args:
            report: The plan ``_confirm_curated_install`` obtained consent
                for.

        Returns:
            The root ``ArtifactRef`` provisioned and (by default) activated.
        """
        from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService

        acquisition = ArtifactAcquisitionService(self._model_install_service)

        def deliver(progress: "AcquisitionProgress") -> None:
            self._deliver_curated(InstallProgressed(progress))

        return await acquisition.provision(
            report.root,
            report.grant(),
            self._model_install_registry,
            sources=self._model_install_sources,
            progress=deliver,
        )

    @work(thread=True, group="llm_curated_install", exit_on_error=False)
    def _run_curated_provision(self) -> None:
        """Provision the consented plan off the Textual event loop."""
        report = self._model_install_pending_report
        if report is None:
            self.app.call_from_thread(
                self._apply_curated_provision_result,
                "No install plan is available; review the model again.",
            )
            return
        try:
            asyncio.run(self._provision_curated(report))  # policy-exception: worker-thread loop
        except Exception as exc:
            logger.opt(exception=True).error(
                "Curated model installation failed for {}@{}/{}",
                report.root.artifact_id,
                report.root.revision,
                report.root.variant,
            )
            self.app.call_from_thread(
                self._apply_curated_provision_result,
                install_failure_message(exc, model_label=report.root.artifact_id),
            )
            return
        self.app.call_from_thread(self._apply_curated_provision_result, None)

    def _apply_curated_provision_result(self, error: str | None) -> None:
        """Finish an installation: notify, mirror lifecycle, and reset state."""
        reference = self._model_install_reference
        self._model_install_worker = None
        self._model_install_pending_report = None
        if error is not None:
            self.notify(error, severity="error")
        else:
            self.notify("Model installed and activated.", severity="information")
        if reference is not None:
            self._deliver_curated(
                InstallStatusChanged(reference, active=False, succeeded=error is None)
            )
        self._model_install_reference = None
        self._model_install_service = None
        self._model_install_registry = None
        self._model_install_sources = None
        view = self._curated_view()
        if view is not None:
            view.finish_install()

    def _clear_curated_install_state(self) -> None:
        """Reset this screen's own bookkeeping after a request that never
        started provisioning (a preflight failure or an explicit decline
        at the consent modal) -- neither ever posted
        ``InstallStatusChanged(active=True)``, so neither mirrors into
        ``InstalledView`` or reloads the catalog; the visible
        ``CuratedView`` (if still mounted) only needs its own in-flight
        indicator released.
        """
        self._model_install_reference = None
        self._model_install_service = None
        self._model_install_registry = None
        self._model_install_sources = None
        self._model_install_pending_report = None
        view = self._curated_view()
        if view is not None:
            view.cancel_pending_install()

    def _deliver_curated(self, message: InstallProgressed | InstallStatusChanged) -> None:
        """Post one curated-install message so it bubbles through ``LLMManagementWindow``.

        Tried first at ``self.llm_window`` -- read fresh on every call, so
        it already points at whichever ``LLMManagementWindow`` instance is
        currently mounted once a screen-level recompose has finished
        replacing it (``build_lab_body`` reassigns this attribute every
        time it runs). Posting there lets the message bubble CuratedView-
        less straight through ``LLMManagementWindow``'s own mirroring
        handlers (``_managed_install_progressed``/``_managed_install_
        status_changed``, which keep ``InstalledView`` in sync) and on up
        to this screen's own ``@on(InstallProgressed)``/
        ``@on(InstallStatusChanged)`` handlers -- the exact same single
        bubble path ``CuratedView`` used to originate (``CuratedView`` ->
        ``LLMManagementWindow`` -> ``LLMScreen``), except this screen is
        now the origin, not an orphanable view.

        ``self.llm_window`` is a plain attribute, not a live query:
        ``LabScreen.recompose()`` tears down and closes the old
        ``LLMManagementWindow`` SYNCHRONOUSLY, but only ``_mount_lab_body``
        -- deferred via ``call_after_refresh`` -- reassigns this attribute
        to the fresh instance. Between those two points, ``self.llm_window``
        still refers to the closed widget, and ``post_message`` on a closed
        target returns ``False`` without raising (this is the exact
        "closed widget, silent no-op" hazard the deleted ``CuratedView.
        _deliver``/``_progress_screen`` fallback chain existed to survive --
        moving the worker here did not eliminate it, only relocated it to
        this one narrower window). Checking that return value and falling
        back to posting directly on ``self`` restores the guarantee this
        method's own callers depend on: ``self`` is the screen running
        this method, so it is never itself the thing a recompose closes.
        This screen's own handlers -- the status chip and retained
        progress/lifecycle state -- always update, even when
        ``LLMManagementWindow``'s mirror is briefly unreachable; the very
        next tick, once the fresh window exists and this attribute is
        reassigned, resumes the full path.

        Args:
            message: The event to deliver; a fresh instance per call.
        """
        target: Widget = self.llm_window if self.llm_window is not None else self
        if target.post_message(message):
            return
        self.post_message(message)

    def compose_lab_rail(self) -> ComposeResult:
        """Yield the two rail sections and their nine provider rows."""
        for title, entries in MODELS_RAIL_SECTIONS:
            yield Static(title, classes="lab-rail-section")
            for view_key, label in entries:
                row = Button(
                    label,
                    id=f"lab-models-row-{view_key}",
                    classes=LAB_RAIL_ROW_CLASS,
                )
                # Carried as an attribute rather than parsed back out of the
                # id, mirroring library_collections_panel's collection_id.
                row.lab_view_key = view_key
                yield row

    def compose_lab_inspector(self) -> ComposeResult:
        """Yield the running-server list."""
        yield Static("Running servers", classes="lab-rail-section")
        for row in self._current_server_rows():
            yield Static(
                server_row_text(row),
                id=server_row_id(row.name),
                markup=False,
            )

    def lab_inspector_rows(self) -> tuple[LabInspectorRow, ...]:
        """Return the running-server rows to refresh in place.

        Read on the same 2-second poll as the status chip
        (``on_lab_body_ready``'s ``set_interval``), so the inspector never
        lags the chip the way it did when only the chip refreshed.
        """
        return tuple(
            LabInspectorRow(row_id=server_row_id(row.name), text=server_row_text(row))
            for row in self._current_server_rows()
        )

    def build_lab_body(self) -> Widget:
        """Build the legacy management window.

        Returns:
            The ``LLMManagementWindow``, mounted after first paint because
            composing its nine views costs 488-787 ms.
        """
        self.llm_window = LLMManagementWindow(self.app_instance, classes="window")
        self.llm_window.styles.height = "1fr"
        return self.llm_window

    def on_lab_body_ready(self) -> None:
        """Wire rail highlighting to the window's active_view, then poll.

        The watch is registered here because the window does not exist before
        this point. ``init=True`` fires the callback immediately, which seeds
        the rail highlight -- necessary because ``LLMManagementWindow.on_mount``
        sets ``active_view`` itself, so a press-only handler would leave the
        rail unhighlighted on arrival.

        This also fires again after a screen-level recompose (``LabScreen.
        recompose()`` reruns the deferred body mount, which calls this once
        the fresh body exists), so the watch is rebound to the NEW
        ``LLMManagementWindow`` instance each time -- the old one is gone.
        The poll timer, in contrast, is owned by this screen and already
        survives a recompose untouched, so ``_status_poll_started`` guards
        it against being started a second time.
        """
        if self.llm_window is None:
            return
        self.watch(self.llm_window, "active_view", self._sync_rail_active, init=True)
        self.refresh_lab_status()
        if not self._status_poll_started:
            self._status_poll_started = True
            self.set_interval(LAB_SERVER_POLL_SECONDS, self.refresh_lab_status)
        # This fires on every (re)mount of the body -- first mount AND every
        # later screen-level recompose (see this method's own docstring) --
        # which is exactly when a fresh CuratedView instance, with no
        # memory of an install this screen already knows is running, can
        # appear mid-download (TASK-596 delta port). call_after_refresh
        # (rather than calling directly) gives the freshly (re)mounted
        # LLMManagementWindow's own children -- CuratedView included -- a
        # chance to finish composing before _hydrate_curated_progress
        # queries for it.
        if self._model_install_active:
            self.call_after_refresh(self._hydrate_curated_progress)

    def _hydrate_curated_progress(self) -> None:
        """Re-apply the last known curated-install progress after a recompose.

        Without this, a freshly (re)mounted ``CuratedView``'s progress
        widget -- composed hidden every time a new instance is built, see
        ``CuratedView.compose`` -- would stay hidden for the rest of an
        install that outlived a screen-level ``LabScreen.recompose()``,
        even though ``CuratedView`` itself keeps rendering live updates
        that reach it (see ``_model_install_progressed``). Scheduled via
        ``call_after_refresh`` from ``on_lab_body_ready`` (which reruns on
        every recompose, not only first mount) -- mirroring
        ``LibraryScreen``'s own ``_hydrate_parakeet_v2_progress`` -- so the
        fresh ``CuratedView`` has actually finished mounting first.
        """
        if not self._model_install_active:
            return
        if self._model_install_last_progress is None:
            return
        view = self._curated_view()
        if view is not None:
            view.apply_progress(self._model_install_last_progress)

    def _sync_rail_active(self, active_view: str) -> None:
        """Move the rail highlight to the row matching the active view.

        Args:
            active_view: The window's current view key.
        """
        for row in self.query(f".{LAB_RAIL_ROW_CLASS}").results(Button):
            row.set_class(getattr(row, "lab_view_key", None) == active_view, "is-active")

    @on(Button.Pressed, f".{LAB_RAIL_ROW_CLASS}")
    def _handle_rail_press(self, event: Button.Pressed) -> None:
        """Point the window at the pressed provider's view.

        The window's own ``@on`` no longer fires: the buttons are the
        screen's children now, so their presses never reach it. Styling is
        not done here -- ``_sync_rail_active`` runs from the reactive watch,
        which also covers changes the window makes itself.
        """
        event.stop()
        view_key = getattr(event.button, "lab_view_key", None)
        if view_key is None or self.llm_window is None:
            return
        self.llm_window.active_view = view_key

    async def on_screen_resume(self) -> None:
        """Refresh server status when a modal pops back over this screen."""
        self.refresh_lab_status()
