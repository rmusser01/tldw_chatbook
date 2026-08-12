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

from ...Model_Artifacts.remote_huggingface import (
    RemoteGGUFCandidate,
    ResolvedRemoteCatalog,
)
from ...Model_Artifacts.service import ArtifactRef, ModelArtifactService
from ...Widgets.ModelArtifacts import (
    InstallProgressed,
    InstallStatusChanged,
    ModelInstallModal,
)
from ..Lab_Modules.lab_server_status import (
    LAB_SERVER_SOURCES,
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
from .model_installed_view import InstalledView
from .model_remote_view import RemoteView

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionProgress,
        CredentialResolver,
        PreflightReport,
    )
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

#: Back-compat alias for the (app attribute, display name) server-process
#: table; ``LAB_SERVER_SOURCES`` in ``lab_server_status`` is the canonical
#: copy and carries the same six providers.
_SERVER_PROCESS_ATTRS = LAB_SERVER_SOURCES


async def _probe_local_server(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Cheap TCP probe for an externally-started Ollama server.

    task-15473: this used to be a blocking `socket.create_connection(...,
    timeout=0.25)`, called directly from the Models screen's periodic
    status timer -- instant on ECONNREFUSED, but a genuinely blackholed
    port (firewalled/container setups) froze the WHOLE event loop for up
    to the full 250ms, once per tick, since a synchronous socket call on
    the loop thread blocks every other task in the process, not just this
    one. `asyncio.open_connection` under the same 0.25s `wait_for` cap
    keeps the exact semantics (up = connectable, down = refused or
    timeout, same interval) but the wait yields the loop instead of
    freezing it -- verified live (mutation-tested against the old
    blocking implementation): a concurrent heartbeat task ticks dozens
    of times during the wait against a real unresponsive address, versus
    zero for the old blocking call in the same window (see
    ``Tests/UI/test_llm_screen_ollama_probe_nonblocking.py``).
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=0.25
        )
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass
    return True


class LLMScreen(LabScreen):
    """Models mode: provider rail, legacy management body, server status.

    The ``DestinationHeader`` above the rail is composed by the ``LabScreen``
    frame from this mode's ``lab_header_state()`` (a ``WorkbenchHeaderState``)
    and re-synced on every ``refresh_lab_status()`` pass.
    """

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
        #: Which flow currently owns the fields below -- ``"curated"`` or
        #: ``"remote"``, or ``None`` when idle. TASK-1914: curated and
        #: remote installs share this screen's one set of retained state
        #: (see this field's own role below for why one shared lock is
        #: correct here, not two independent ones). Serves two purposes:
        #: (1) routing -- the only way to know which mounted view
        #: (``_curated_view()`` or ``_remote_view()``) the currently
        #: in-flight operation belongs to, since both views can be mounted
        #: at once (``LLMManagementWindow`` composes every rail view
        #: eagerly; only ``active_view`` picks which is visible), so a
        #: progress tick must be routed to the right one, not applied to
        #: both -- see ``_active_install_view``; and (2), since TASK-1914
        #: fix round 2, THE concurrency guard itself: set once, when
        #: ``_curated_install_requested``/``_remote_install_requested``
        #: accepts a request, and cleared ONLY in the terminal
        #: apply-provision-result/clear-state paths, so it stays non-
        #: ``None`` for an accepted request's entire lifecycle -- preflight
        #: running, pending consent, and provisioning -- unlike
        #: ``_model_install_worker`` below, which is briefly ``None``
        #: during the pending-consent window. See ``_install_in_progress``.
        self._model_install_kind: str | None = None
        #: The last ``AcquisitionProgress`` this screen has seen for the
        #: active install (curated or remote), retained so a freshly
        #: (re)mounted ``CuratedView``/``RemoteView`` -- a screen-level
        #: ``LabScreen.recompose()`` tears down and rebuilds the whole
        #: ``LLMManagementWindow`` mid-download -- can be hydrated back to
        #: it immediately instead of starting blank (TASK-596 delta port).
        #: Mirrors ``LibraryScreen``'s own retained
        #: ``_parakeet_v2_install_progress``. Cleared whenever the install
        #: stops (see ``_model_install_status_changed``).
        self._model_install_last_progress: "AcquisitionProgress | None" = None
        #: The install worker currently running (preflight OR provision --
        #: reassigned when the second phase starts, mirroring
        #: ``LibraryScreen``'s single ``_parakeet_v2_install_worker`` field
        #: across its own two phases), and briefly ``None`` BETWEEN those
        #: two phases while the shared consent modal awaits the user's
        #: decision. NOT the concurrency guard (TASK-1914 fix round 2 --
        #: it was, before this fix, and that was the bug: a worker-handle
        #: check is blind to the pending-consent window). ``_model_install_
        #: kind`` is the guard now (see ``_install_in_progress``); this
        #: field exists purely so the actual ``Worker`` handle is
        #: retained/inspectable across the two phases, mirroring
        #: ``LibraryScreen``'s equivalent field. Both flows still share
        #: this one slot (TASK-1914), not one each, because the managed
        #: store's own ``ArtifactAcquisitionService.provision`` already
        #: serializes concurrent installs behind one in-process lease
        #: regardless of which view started them (see that class's own
        #: docstring): a curated install and a remote install running "at
        #: the same time" would already queue against each other at that
        #: layer, so tracking two independently here would only mean
        #: paying for a wasted preflight before the second one blocked
        #: anyway, while doubling every field below for no operational
        #: benefit.
        self._model_install_worker: Worker | None = None
        #: The reference, service, and (curated-only) registry/source map
        #: the currently running (or about-to-run) curated install needs
        #: -- captured once from the posted ``CuratedView.InstallRequested``
        #: (or, for ``_model_install_pending_report``, once preflight
        #: resolves) and read back by this screen's own worker methods for
        #: as long as the operation runs, so nothing here depends on the
        #: posting ``CuratedView`` instance still existing by the time
        #: provisioning finishes. ``_model_install_reference`` and
        #: ``_model_install_service``/``_model_install_pending_report`` are
        #: shared with the remote flow below; ``_model_install_registry``/
        #: ``_model_install_sources`` are curated-only (left ``None``
        #: while a remote install runs).
        self._model_install_reference: ArtifactRef | None = None
        self._model_install_service: ModelArtifactService | None = None
        self._model_install_registry: "CuratedRegistry | None" = None
        self._model_install_sources: dict[ArtifactRef, dict[str, str]] | None = None
        self._model_install_pending_report: "PreflightReport | None" = None
        #: The frozen catalog, exact candidate, and credential resolver a
        #: running (or about-to-run) REMOTE install needs -- captured once
        #: from the posted ``RemoteView.InstallRequested``, mirroring the
        #: curated fields above. Remote-only; left ``None`` while a
        #: curated install runs.
        self._model_install_catalog: "ResolvedRemoteCatalog | None" = None
        self._model_install_candidate: "RemoteGGUFCandidate | None" = None
        self._model_install_credential_resolver: "CredentialResolver | None" = None
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

        Forwards the event into whichever view owns the currently in-
        flight install (``_active_install_view()``) -- not only the
        instance mounted when the install started. This screen itself is
        the sole poster of ``InstallProgressed`` for BOTH a curated and a
        remote install (TASK-1803/TASK-1914: see ``_run_curated_provision``/
        ``_run_remote_provision`` and ``_deliver_curated``, which post at
        ``self.llm_window`` so the message bubbles through
        ``LLMManagementWindow``'s own mirroring handler on its way here) --
        unlike before, when the view posted it directly and needed a
        durable fallback path to survive a screen-level recompose tearing
        it down mid-install. This screen is never what a recompose tears
        down, so no such fallback is needed any more.
        """
        self._model_install_active = True
        self._model_install_phase = event.progress.phase
        self._model_install_succeeded = None
        self._model_install_last_progress = event.progress
        self.refresh_lab_status()
        view = self._active_install_view()
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

    def _installed_view(self) -> "InstalledView | None":
        """Return the mounted ``InstalledView``, or None if it cannot be found.

        Returns:
            The view, or None when the window has not mounted yet, or a
            screen-level recompose has torn down the previous instance and
            the fresh one is not mounted yet either (see
            ``LabScreen.recompose()``).
        """
        if self.llm_window is None:
            return None
        try:
            return self.llm_window.query_one(InstalledView)
        except NoMatches:
            return None

    def _remote_view(self) -> "RemoteView | None":
        """Return the mounted ``RemoteView``, or None if it cannot be found.

        Returns:
            The view, or None when the window has not mounted yet, or a
            screen-level recompose has torn down the previous instance and
            the fresh one is not mounted yet either (see
            ``LabScreen.recompose()``).
        """
        if self.llm_window is None:
            return None
        try:
            return self.llm_window.query_one(RemoteView)
        except NoMatches:
            return None

    def _active_install_view(self) -> "CuratedView | RemoteView | None":
        """Return the view rendering the currently in-flight install, if any.

        ``LLMManagementWindow`` composes every rail view eagerly (only
        ``active_view`` picks which one is visible), so both
        ``CuratedView`` and ``RemoteView`` are mounted at once regardless
        of which install (if either) is running -- routing by
        ``_model_install_kind`` (set once, when ``_curated_install_
        requested``/``_remote_install_requested`` accepts a request) is
        what keeps a remote install's progress from also being rendered
        into the unrelated, currently-idle ``CuratedView``, and vice
        versa.

        Returns:
            ``_curated_view()``/``_remote_view()`` matching the in-flight
            install's kind, or ``None`` while idle.
        """
        if self._model_install_kind == "curated":
            return self._curated_view()
        if self._model_install_kind == "remote":
            return self._remote_view()
        return None

    def _install_in_progress(self) -> bool:
        """Return whether a curated or remote install is in ANY phase.

        TASK-1914 fix round 2: the concurrency guard in ``_curated_
        install_requested``/``_remote_install_requested`` used to check
        only ``_model_install_worker is not None and not worker.is_
        finished`` -- true while a preflight/provision thread is actually
        running, but ``_model_install_worker`` is deliberately set back to
        ``None`` by ``_apply_curated_preflight_result``/``_apply_remote_
        preflight_result`` the moment preflight succeeds, before the
        shared consent modal is even pushed (see those methods' own
        docstrings). A second ``InstallRequested`` landing during that
        pending-consent window -- no worker running, but an install very
        much still in progress, awaiting the user's decision -- passed the
        old guard and could overwrite ``_model_install_kind``/the pending
        report/reference for the install actually awaiting consent.

        ``_model_install_kind`` is the fix: set once, when a request is
        accepted (``_curated_install_requested``/``_remote_install_
        requested``), and cleared ONLY in the terminal paths -- the
        provision-result methods (success or failure) and the cancel/
        clear-state helpers (invalid payload, preflight failure, explicit
        decline) -- so it is non-``None`` for the ENTIRE lifecycle of an
        accepted request: preflight running, pending consent, and
        provisioning, not merely "a worker happens to be running right
        now".

        Returns:
            Whether an install (either kind) is currently in progress.
        """
        return self._model_install_kind is not None

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
        worker-in-flight guard, generalized -- see ``_install_in_progress``)
        -- including a concurrent REMOTE install (TASK-1914: both flows
        share this screen's one ``_model_install_kind`` lifecycle guard) --
        for as long as ANY phase of another install is in progress:
        preflight running, pending consent (the shared modal is up, no
        worker running), or provisioning (TASK-1914 fix round 2 -- see
        ``_install_in_progress``'s own docstring for why checking the
        worker handle alone left the pending-consent window unguarded).
        The requesting ``CuratedView`` already disabled its own row before
        posting this, but only a screen-level recompose can hand a
        *different*, freshly (re)mounted instance a chance to post a
        second request while the first is still in progress --
        ``cancel_pending_install()`` releases only that fresh instance's
        own indicator, leaving the still-in-progress install's retained
        state (below) untouched.

        Also validates the event's payload before storing any of it
        (TASK-1803 review round 2, Critical): ``CuratedView`` only ever
        posts a well-formed request today, but nothing enforces that at
        the type level, and ``_run_curated_preflight`` used to assume
        ``self._model_install_reference`` was always a valid
        ``ArtifactRef`` -- a malformed or missing reference reached
        ``reference.artifact_id`` in that worker's own exception handler,
        raising a SECOND exception that pre-empted
        ``_apply_curated_preflight_result`` entirely and stranded the
        retained install state with no path back to idle. Rejecting an
        invalid request here, before it is ever stored, is the primary
        defense; ``_run_curated_preflight``'s own guard and safe
        formatting are the defense-in-depth backstop.
        """
        event.stop()
        if self._install_in_progress():
            self.notify(
                "A curated model install is already running.",
                severity="information",
            )
            view = self._curated_view()
            if view is not None:
                view.cancel_pending_install()
            return
        if (
            not isinstance(event.reference, ArtifactRef)
            or event.service is None
            or event.registry is None
            or event.sources is None
        ):
            logger.error(
                "Curated install request carried an invalid reference, "
                "service, registry, or source map; refusing to start."
            )
            self.notify(
                "Could not start the model install: invalid request.",
                severity="error",
            )
            self._clear_curated_install_state()
            return
        self._model_install_kind = "curated"
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
        """Resolve the curated install plan off the Textual event loop.

        Defense-in-depth (TASK-1803 review round 2, Critical): ``_curated_
        install_requested`` already refuses to store an invalid reference,
        so ``reference`` below should always be a real ``ArtifactRef`` by
        the time this runs -- but this method no longer trusts that. A
        missing/malformed reference schedules
        ``_apply_curated_preflight_result`` directly instead of ever
        reaching the ``try`` block, and the ``except`` clause formats the
        reference defensively (``getattr(..., default)``) rather than
        assuming attribute access succeeds. Every path below -- the guard,
        the exception handler, and the success path -- ends in exactly
        one ``call_from_thread(self._apply_curated_preflight_result, ...)``,
        so the retained install state can never be left stranded by a
        second, unhandled exception inside the error handler itself.
        """
        reference = self._model_install_reference
        if reference is None:
            self.app.call_from_thread(
                self._apply_curated_preflight_result,
                None,
                "No install request is available; review the model again.",
            )
            return
        try:
            report = asyncio.run(  # policy-exception: worker-thread loop
                self._preflight_curated(reference)
            )
        except Exception as exc:
            artifact_id = getattr(reference, "artifact_id", "unknown")
            logger.opt(exception=True).error(
                "Curated model preflight failed for {}@{}/{}",
                artifact_id,
                getattr(reference, "revision", "unknown"),
                getattr(reference, "variant", "unknown"),
            )
            self.app.call_from_thread(
                self._apply_curated_preflight_result,
                None,
                install_failure_message(exc, model_label=artifact_id),
            )
            return
        self.app.call_from_thread(self._apply_curated_preflight_result, report, None)

    def _apply_curated_preflight_result(
        self,
        report: "PreflightReport | None",
        error: str | None,
    ) -> None:
        """Show the shared consent modal, or a sanitized preflight failure."""
        # _model_install_worker is None from this line until
        # _confirm_curated_install's own _run_curated_provision() call --
        # i.e. for as long as the shared consent modal is up -- but that no
        # longer matters for concurrency safety: _curated_install_
        # requested/_remote_install_requested guard on _install_in_
        # progress() (_model_install_kind is not None), and _model_install_
        # kind stays set through this entire pending-consent window (TASK-
        # 1914 fix round 2). A second InstallRequested landing here is
        # refused before it can touch any of this screen's retained state.
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
        """Provision the consented plan off the Textual event loop.

        Same defense-in-depth as ``_run_curated_preflight`` (TASK-1803
        review round 2): the ``except`` clause formats ``report.root``
        defensively rather than assuming attribute access succeeds, so a
        malformed report can never turn one failure into a second,
        unhandled exception that skips
        ``_apply_curated_provision_result`` and strands install state.
        """
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
            root = getattr(report, "root", None)
            artifact_id = getattr(root, "artifact_id", "unknown")
            logger.opt(exception=True).error(
                "Curated model installation failed for {}@{}/{}",
                artifact_id,
                getattr(root, "revision", "unknown"),
                getattr(root, "variant", "unknown"),
            )
            self.app.call_from_thread(
                self._apply_curated_provision_result,
                install_failure_message(exc, model_label=artifact_id),
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
        self._model_install_kind = None
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
        self._model_install_kind = None
        view = self._curated_view()
        if view is not None:
            view.cancel_pending_install()

    def _deliver_curated(self, message: InstallProgressed | InstallStatusChanged) -> None:
        """Post one install-lifecycle message so it bubbles through ``LLMManagementWindow``.

        Despite the name (kept for the existing call sites and tests that
        already depend on it), this is the single delivery path for BOTH
        the curated and the remote install flow (TASK-1914): each worker
        pair (``_run_curated_provision``/``_run_remote_provision``, and
        their ``_confirm_*_install`` counterparts for the leading
        ``InstallStatusChanged(active=True)``) calls this exact method --
        there is nothing curated-specific left in its own body, only in
        its history.

        Tried first at ``self.llm_window`` -- read fresh on every call, so
        it already points at whichever ``LLMManagementWindow`` instance is
        currently mounted once a screen-level recompose has finished
        replacing it (``build_lab_body`` reassigns this attribute every
        time it runs). Posting there lets the message bubble view-less
        straight through ``LLMManagementWindow``'s own mirroring handlers
        (``_managed_install_progressed``/``_managed_install_status_
        changed``, which keep ``InstalledView`` in sync) and on up to this
        screen's own ``@on(InstallProgressed)``/``@on(InstallStatusChanged)``
        handlers -- the exact same single bubble path ``CuratedView`` used
        to originate (``CuratedView`` -> ``LLMManagementWindow`` ->
        ``LLMScreen``) before TASK-1803, and ``RemoteView`` before
        TASK-1914, except this screen is now the origin, not an orphanable
        view.

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

    # -- Remote model install: this screen owns preflight/provision -------
    #
    # TASK-1914: mirrors the curated block above exactly. RemoteView posts
    # RemoteView.InstallRequested and renders what it is told
    # (apply_progress/cancel_pending_install/finish_install); resolving the
    # plan, showing the consent modal, and provisioning all run here. Both
    # flows share this screen's single _install_in_progress()/
    # _model_install_kind concurrency guard (see that method's own
    # docstring) and the InstallProgressed/InstallStatusChanged delivery
    # path (_deliver_curated, despite its name -- see its own docstring)
    # and hydration path (_hydrate_model_install_progress); only the
    # curated-vs-remote-specific plan-resolution/consent/provisioning
    # steps below are duplicated, exactly as the curated block duplicates
    # LibraryScreen's own Parakeet v2 shape.

    @on(RemoteView.InstallRequested)
    def _remote_install_requested(self, event: RemoteView.InstallRequested) -> None:
        """Resolve an install plan for a reviewed remote candidate, off the Textual event loop.

        Shares ``_curated_install_requested``'s guard (``_install_in_
        progress()``, checking ``_model_install_kind`` rather than the
        worker handle -- see that method's own docstring for why one
        screen-level lock, not two, is the correct shape now that both
        flows live on this screen, AND why the guard must span the whole
        install lifecycle, not just "a worker happens to be running")
        and the same validate-before-store discipline (TASK-1803 review
        round 2, Critical, applied here from the start rather than
        rediscovered): an invalid payload notifies, releases the clicking
        view's own indicator, and clears state via ``_clear_remote_
        install_state`` without ever starting a worker.
        """
        event.stop()
        if self._install_in_progress():
            self.notify(
                "A model install is already running.",
                severity="information",
            )
            view = self._remote_view()
            if view is not None:
                view.cancel_pending_install()
            return
        if (
            not isinstance(event.catalog, ResolvedRemoteCatalog)
            or not isinstance(event.candidate, RemoteGGUFCandidate)
            or event.service is None
            or event.credential_resolver is None
        ):
            logger.error(
                "Remote install request carried an invalid catalog, "
                "candidate, service, or credential resolver; refusing to "
                "start."
            )
            self.notify(
                "Could not start the model install: invalid request.",
                severity="error",
            )
            self._clear_remote_install_state()
            return
        self._model_install_kind = "remote"
        self._model_install_reference = event.catalog.artifact.reference
        self._model_install_service = event.service
        self._model_install_catalog = event.catalog
        self._model_install_candidate = event.candidate
        self._model_install_credential_resolver = event.credential_resolver
        self._model_install_worker = self._run_remote_preflight()

    async def _preflight_remote(self, catalog: "ResolvedRemoteCatalog"):
        """Resolve a remote acquisition plan on the worker's event loop.

        Args:
            catalog: The frozen one-item catalog to preflight.

        Returns:
            The immutable ``PreflightReport`` describing the download.
        """
        from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService

        acquisition = ArtifactAcquisitionService(
            self._model_install_service,
            credential_resolver=self._model_install_credential_resolver,
        )
        return await acquisition.preflight(
            catalog.artifact.reference,
            catalog,
            sources=catalog.sources,
        )

    @work(thread=True, group="llm_remote_preflight", exit_on_error=False)
    def _run_remote_preflight(self) -> None:
        """Resolve the remote install plan off the Textual event loop.

        Same defense-in-depth as ``_run_curated_preflight``: a missing
        ``_model_install_catalog`` schedules ``_apply_remote_preflight_
        result`` directly instead of ever reaching the ``try`` block, and
        the ``except`` clause walks ``catalog.artifact.reference``
        defensively (``getattr(..., "unknown")`` at each hop) rather than
        assuming attribute access succeeds, so every path ends in exactly
        one ``call_from_thread(self._apply_remote_preflight_result, ...)``.
        """
        catalog = self._model_install_catalog
        if catalog is None:
            self.app.call_from_thread(
                self._apply_remote_preflight_result,
                None,
                "No install request is available; search and review the model again.",
            )
            return
        try:
            report = asyncio.run(  # policy-exception: worker-thread loop
                self._preflight_remote(catalog)
            )
        except Exception as exc:
            from tldw_chatbook.Model_Artifacts.acquisition import TransferError

            artifact = getattr(catalog, "artifact", None)
            reference = getattr(artifact, "reference", None)
            artifact_id = getattr(reference, "artifact_id", "unknown")
            model_label = getattr(artifact, "model_id", "unknown")
            logger.error(
                "Remote model preflight failed for managed artifact {}; "
                "error_type={}, retryable={}",
                artifact_id,
                type(exc).__name__,
                isinstance(exc, TransferError) and getattr(exc, "retryable", False),
            )
            self.app.call_from_thread(
                self._apply_remote_preflight_result,
                None,
                install_failure_message(exc, model_label=model_label),
            )
            return
        self.app.call_from_thread(self._apply_remote_preflight_result, report, None)

    def _apply_remote_preflight_result(
        self,
        report: "PreflightReport | None",
        error: str | None,
    ) -> None:
        """Show the shared consent modal, or a sanitized preflight failure."""
        # See the identical note in _apply_curated_preflight_result: this
        # window is safe now because _install_in_progress() guards on
        # _model_install_kind, not on _model_install_worker (TASK-1914 fix
        # round 2).
        self._model_install_worker = None
        catalog = self._model_install_catalog
        candidate = self._model_install_candidate
        if (
            error is not None
            or report is None
            or catalog is None
            or report.root != catalog.artifact.reference
        ):
            message = error or "The install plan changed. Search and review it again."
            self.notify(message, severity="error")
            self._clear_remote_install_state(message)
            return
        self._model_install_pending_report = report
        acknowledgment = (
            "No license was declared. I reviewed the source and want to continue."
            if catalog.artifact.license_id == "NOASSERTION"
            else None
        )
        selected_file_details: tuple[tuple[str, int, str, str], ...] = ()
        if candidate is not None:
            remote_files = tuple(
                sorted(candidate.files, key=lambda item: item.upstream_path)
            )
            selected_file_details = tuple(
                (
                    remote_file.upstream_path,
                    remote_file.size_bytes,
                    remote_file.sha256,
                    catalog.sources[catalog.artifact.reference][artifact_file.path],
                )
                for remote_file, artifact_file in zip(
                    remote_files,
                    catalog.artifact.files,
                    strict=True,
                )
            )
        self.app.push_screen(
            ModelInstallModal(
                report,
                model_label=catalog.artifact.model_id,
                required_acknowledgment=acknowledgment,
                selected_file_details=selected_file_details,
            ),
            self._confirm_remote_install,
        )

    def _confirm_remote_install(self, confirmed: bool) -> None:
        """Start provisioning only after explicit consent."""
        if not confirmed:
            self._clear_remote_install_state()
            return
        reference = self._model_install_reference
        if reference is not None:
            self._deliver_curated(InstallStatusChanged(reference, active=True))
        self._model_install_worker = self._run_remote_provision()

    async def _provision_remote(
        self,
        report: "PreflightReport",
        catalog: "ResolvedRemoteCatalog",
    ):
        """Provision the consented report on the worker's event loop, without activating.

        Args:
            report: The plan ``_confirm_remote_install`` obtained consent
                for.
            catalog: The frozen catalog the report was resolved against.

        Returns:
            The root ``ArtifactRef`` provisioned but left inactive, exactly
            as the pre-TASK-1914 ``RemoteView._provision`` behaved --
            reviewing and downloading a remote GGUF file never implicitly
            switches the active provider/model.
        """
        from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService

        acquisition = ArtifactAcquisitionService(
            self._model_install_service,
            credential_resolver=self._model_install_credential_resolver,
        )

        def deliver(progress: "AcquisitionProgress") -> None:
            self._deliver_curated(InstallProgressed(progress))

        return await acquisition.provision(
            report.root,
            report.grant(),
            catalog,
            sources=catalog.sources,
            progress=deliver,
            activate=False,
        )

    @work(thread=True, group="llm_remote_install", exit_on_error=False)
    def _run_remote_provision(self) -> None:
        """Provision the consented plan off the Textual event loop.

        Same defense-in-depth as ``_run_remote_preflight``: the ``except``
        clause formats ``report.root``/``catalog.artifact`` defensively
        rather than assuming attribute access succeeds.
        """
        report = self._model_install_pending_report
        catalog = self._model_install_catalog
        if report is None or catalog is None:
            self.app.call_from_thread(
                self._apply_remote_provision_result,
                "No install plan is available; search and review the model again.",
            )
            return
        try:
            asyncio.run(self._provision_remote(report, catalog))  # policy-exception: worker-thread loop
        except Exception as exc:
            from tldw_chatbook.Model_Artifacts.acquisition import TransferError

            artifact = getattr(catalog, "artifact", None)
            root = getattr(report, "root", None)
            artifact_id = getattr(root, "artifact_id", "unknown")
            model_label = getattr(artifact, "model_id", "unknown")
            logger.error(
                "Remote model installation failed for managed artifact {}; "
                "error_type={}, retryable={}",
                artifact_id,
                type(exc).__name__,
                isinstance(exc, TransferError) and getattr(exc, "retryable", False),
            )
            self.app.call_from_thread(
                self._apply_remote_provision_result,
                install_failure_message(exc, model_label=model_label),
            )
            return
        self.app.call_from_thread(self._apply_remote_provision_result, None)

    def _apply_remote_provision_result(self, error: str | None) -> None:
        """Finish an installation: notify, mirror lifecycle, and reset state."""
        reference = self._model_install_reference
        self._model_install_worker = None
        self._model_install_pending_report = None
        if error is not None:
            message = error
            self.notify(message, severity="error")
        else:
            message = (
                "Model downloaded and managed. Runtime compatibility has "
                "not been verified."
            )
            self.notify(message, severity="information")
        if reference is not None:
            self._deliver_curated(
                InstallStatusChanged(reference, active=False, succeeded=error is None)
            )
        self._model_install_reference = None
        self._model_install_service = None
        self._model_install_catalog = None
        self._model_install_candidate = None
        self._model_install_credential_resolver = None
        self._model_install_kind = None
        view = self._remote_view()
        if view is not None:
            view.finish_install(message)

    def _clear_remote_install_state(self, message: str | None = None) -> None:
        """Reset this screen's own bookkeeping after a request that never
        started provisioning (a preflight failure or an explicit decline
        at the consent modal) -- neither ever posted
        ``InstallStatusChanged(active=True)``, so neither mirrors into
        ``InstalledView``; the visible ``RemoteView`` (if still mounted)
        only needs its own in-flight indicator released.

        Args:
            message: Status copy to hand to ``RemoteView.cancel_pending_
                install`` in place of the in-flight indicator (e.g. a
                sanitized preflight failure); ``None`` for an explicit
                decline, which restores the view's default status.
        """
        self._model_install_reference = None
        self._model_install_service = None
        self._model_install_catalog = None
        self._model_install_candidate = None
        self._model_install_credential_resolver = None
        self._model_install_pending_report = None
        self._model_install_kind = None
        view = self._remote_view()
        if view is not None:
            view.cancel_pending_install(message)

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
        # which is exactly when a fresh CuratedView/RemoteView instance,
        # with no memory of an install this screen already knows is
        # running, can appear mid-download (TASK-596 delta port).
        # call_after_refresh (rather than calling directly) gives the
        # freshly (re)mounted LLMManagementWindow's own children a chance
        # to finish composing before _hydrate_model_install_progress
        # queries for them.
        if self._model_install_active:
            self.call_after_refresh(self._hydrate_model_install_progress)

    @on(LLMManagementWindow.DeferredViewsMounted)
    def _on_deferred_views_mounted(self) -> None:
        """Re-run install-progress hydration once the deferred views exist.

        task-2900: `on_lab_body_ready`'s single `call_after_refresh` used to
        suffice because compose built every view synchronously; with the
        heavy views mounted after first paint, that hydration races the
        deferred mount and loses (its view lookups no-op). The window posts
        this message when the views are actually queryable — the correctly
        ordered second chance. `_hydrate_model_install_progress` is
        idempotent and internally guarded, so running both is safe.
        """
        if self._model_install_active:
            self._hydrate_model_install_progress()

    def _hydrate_model_install_progress(self) -> None:
        """Re-apply the last known install progress after a recompose.

        Covers both flows (TASK-1914): whichever view owns the in-flight
        install (``_active_install_view()``, keyed by
        ``_model_install_kind``) is hydrated the same way.

        Without this, a freshly (re)mounted ``CuratedView``/``RemoteView``'s
        progress widget -- composed hidden every time a new instance is
        built -- would stay hidden for the rest of an install that
        outlived a screen-level ``LabScreen.recompose()``, even though the
        view itself keeps rendering live updates that reach it (see
        ``_model_install_progressed``). Scheduled via ``call_after_refresh``
        from ``on_lab_body_ready`` (which reruns on every recompose, not
        only first mount) -- mirroring ``LibraryScreen``'s own
        ``_hydrate_parakeet_v2_progress`` -- so the fresh view has actually
        finished mounting first.

        Also mirrors the same retained state directly into the freshly
        (re)mounted ``LLMManagementWindow``'s own ``InstalledView`` (TASK-
        1803 review round 2, Important). ``_deliver_curated``'s fallback
        (see its own docstring) keeps THIS screen's own state current even
        when a tick lands in the narrow teardown -> remount gap, by
        posting directly on ``self`` when the stale, closed
        ``self.llm_window`` cannot receive it -- but a message posted at
        the Screen never reaches ``LLMManagementWindow``'s mirroring
        handlers (``_managed_install_progressed``/``_managed_install_
        status_changed``): Textual only ever bubbles a message UP, never
        back down into a sibling/descendant, and the Screen is already
        above that node. Left uncorrected, a fresh ``InstalledView`` would
        show a stale "not installing" state for however long it takes the
        NEXT tick to arrive naturally through the fresh window's own
        mirror -- this is the same mirroring gap PR #1185 fixed for
        ``CuratedView``, recurring one level deeper now that ``LLMScreen``
        owns the worker. Calling ``InstalledView.set_install_state``
        directly here closes it by construction, deterministically, on
        every (re)mount, rather than by trying to identify and replay
        whichever specific message the gap happened to swallow.
        """
        if not self._model_install_active:
            return
        if self._model_install_last_progress is None:
            return
        view = self._active_install_view()
        if view is not None:
            view.apply_progress(self._model_install_last_progress)
        installed = self._installed_view()
        if installed is not None:
            installed.set_install_state(self._model_install_last_progress, active=True)

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
