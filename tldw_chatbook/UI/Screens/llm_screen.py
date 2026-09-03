"""Models: the Lab destination's provider and model management screen."""

from __future__ import annotations

import asyncio
from datetime import datetime
from functools import partial
from pathlib import Path
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Static
from textual.worker import Worker, get_current_worker

from ...Local_Ingestion.parakeet_v2_artifact import parakeet_reference
from ...Model_Artifacts.remote_huggingface import (
    RemoteGGUFCandidate,
    ResolvedRemoteCatalog,
)
from ...Model_Artifacts.machine_memory import (
    MachineMemorySnapshot,
    ProbeReason,
    SystemMemoryState,
)
from ...Model_Artifacts.service import ArtifactRef, ModelArtifactService
from ...STT.parakeet_sources import (
    ManagedCopyConsent,
    ManagedCopyPlan,
    ParakeetSourceError,
    ParakeetSourceErrorCode,
    ParakeetSourceKey,
    PreparedExternalSelection,
)
from ...STT.parakeet_external import (
    ExternalParakeetVerificationError,
    format_external_parakeet_recovery,
)
from ...Third_Party.textual_fspicker import SelectDirectory
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.ModelArtifacts import (
    InstallProgressed,
    InstallStatusChanged,
    ManagedGGUFRuntimeChoiceModal,
    ModelInstallModal,
)
from ..Navigation.audio_cpp_model_handoff import (
    AudioCppModelInstallOperation,
    AudioCppModelInstallOwner,
    AudioCppModelLibraryRequest,
    AudioCppModelLibraryResult,
)
from ..Navigation.pending_handoff_store import (
    HandoffChannel,
    HandoffClaim,
    HandoffValueError,
    PendingHandoffStore,
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
from .model_external_view import ExternalModelView
from .model_installed_view import InstalledView
from .model_memory_presenter import build_machine_memory_presentation
from .model_remote_view import RemoteView

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Model_Artifacts.acquisition import (
        AcquisitionProgress,
        CredentialResolver,
        PreflightReport,
    )
    from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry


class _AudioCppConsentDeclined(Exception):
    """Internal terminal value for a reviewed install the user declined."""


def _insufficient_space_recovery(report: object) -> str | None:
    """Return byte-exact bounded recovery for one ungrantable real plan."""

    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport

    if type(report) is not PreflightReport or report.sufficient_space:
        return None
    return (
        f"Insufficient space — {report.required_bytes:,} bytes required; "
        f"{report.free_bytes:,} bytes free. Free space, then select Retry install."
    )


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
            ("external", "External"),
            ("remote", "Remote"),
        ),
    ),
)

#: How often to re-read server liveness. There is deliberately no
#: refresh-on-press: pressing Start does not synchronously create the
#: process -- the event handler assigns it from an async worker -- so a
#: press-triggered read would report "stopped".
LAB_SERVER_POLL_SECONDS = 2.0

_REMOTE_INSTALL_TERMINAL_FINISH = "finish"
_REMOTE_INSTALL_TERMINAL_CANCEL = "cancel"
_REMOTE_INSTALL_TERMINAL_ACTIONS = frozenset(
    {_REMOTE_INSTALL_TERMINAL_FINISH, _REMOTE_INSTALL_TERMINAL_CANCEL}
)

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

    def __init__(
        self,
        app_instance: "TldwCli",
        *,
        machine_memory_wall_clock: Callable[[], datetime] | None = None,
        machine_memory_monotonic_clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the Models screen.

        Args:
            app_instance: The running application.
            machine_memory_wall_clock: Injectable local wall clock for the fixed
                accepted-observation label.
            machine_memory_monotonic_clock: Injectable monotonic clock retained
                with accepted machine facts.
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
        self._local_gguf_import_active = False
        self._external_selection_generation = 0
        self._external_selection_token: tuple[int, int] | None = None
        self._external_scope_id: str | None = None
        self._external_scope_ids: dict[tuple[int, int], str] = {}
        self._external_commit_tokens: set[tuple[int, int]] = set()
        self._external_selection_worker: Worker | None = None
        self._external_operation_status = ""
        self._external_operation_error = False
        self._external_operation_active = False
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
        self._audio_cpp_model_install_operation: (
            AudioCppModelInstallOperation | None
        ) = None
        self._audio_cpp_operation_expects_return = False
        self._audio_cpp_consent_future: asyncio.Future[bool] | None = None
        self._audio_cpp_consent_modal: ModelInstallModal | None = None
        self._audio_cpp_reclaim_worker: Worker | None = None
        self._audio_cpp_presentation_worker: Worker | None = None
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
        #: Terminal Remote presentation retained only when the current view is
        #: inside LabScreen's teardown/remount gap. The acquisition fields
        #: above may then be cleared immediately without losing the selected
        #: repository or its outcome before a fresh RemoteView can consume it.
        self._remote_install_terminal_catalog: "ResolvedRemoteCatalog | None" = None
        self._remote_install_terminal_candidate: "RemoteGGUFCandidate | None" = None
        self._remote_install_terminal_action: str | None = None
        self._remote_install_terminal_message: str | None = None
        #: Last verified Remote root and its frozen discovery context. Unlike
        #: the narrow terminal-presentation bridge above, this remains after
        #: delivery so a later screen recompose preserves the adoption CTA.
        self._remote_install_completed_catalog: "ResolvedRemoteCatalog | None" = None
        self._remote_install_completed_candidate: "RemoteGGUFCandidate | None" = None
        self._remote_install_completed_reference: ArtifactRef | None = None
        self._remote_install_completed_message: str | None = None
        #: Exact runtime adoption intent retained by the screen while the
        #: current LLMManagementWindow validates a freshly-downloaded root.
        #: LabScreen recomposition replaces that window, so window-local
        #: ownership alone would strand the handoff with the detached worker.
        self._remote_runtime_handoff: tuple[str, ArtifactRef] | None = None
        #: Exact Installed-row navigation retained while the Installed pane
        #: performs its first lazy mount.
        self._pending_installed_reveal: ArtifactRef | None = None
        self._machine_memory_snapshot: MachineMemorySnapshot | None = None
        self._machine_memory_observed_label: str | None = None
        self._machine_memory_observed_monotonic: float | None = None
        self._machine_memory_generation = 0
        self._machine_memory_worker: Worker | None = None
        self._machine_memory_active = False
        self._machine_memory_failure: ProbeReason | None = None
        self._machine_memory_wall_clock = machine_memory_wall_clock or datetime.now
        self._machine_memory_monotonic_clock = (
            machine_memory_monotonic_clock or time.monotonic
        )
        self._machine_memory_probe_factory: (
            Callable[[], MachineMemorySnapshot] | None
        ) = None
        self._audio_cpp_model_request_claim: (
            HandoffClaim[AudioCppModelLibraryRequest] | None
        ) = None
        #: Server rows snapshotted for the duration of one
        #: ``refresh_lab_status`` pass; None outside one. See
        #: :meth:`_current_server_rows`.
        self._server_rows_snapshot: tuple[LabServerRow, ...] | None = None

    def on_mount(self) -> None:
        """Claim an optional Settings-owned audio.cpp Model Library request."""

        store = getattr(self.app_instance, "pending_handoffs", None)
        if type(store) is not PendingHandoffStore:
            return
        claim = cast(PendingHandoffStore, store).claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )
        if claim is not None and type(claim.value) is AudioCppModelLibraryRequest:
            self._audio_cpp_model_request_claim = claim
        elif self._audio_cpp_install_owner().active_count:
            self._audio_cpp_reclaim_worker = self._reclaim_audio_cpp_request()

    @work(group="audio_cpp_request_reclaim", exit_on_error=False)
    async def _reclaim_audio_cpp_request(self) -> None:
        """Retry one request claim after the prior app-owned operation settles."""

        await self._audio_cpp_install_owner().wait_until_idle()
        if not self.is_attached or self._audio_cpp_model_request_claim is not None:
            return
        store = getattr(self.app_instance, "pending_handoffs", None)
        if type(store) is not PendingHandoffStore:
            return
        claim = cast(PendingHandoffStore, store).claim(
            HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST
        )
        if claim is None or type(claim.value) is not AudioCppModelLibraryRequest:
            return
        self._audio_cpp_model_request_claim = claim
        await self._present_audio_cpp_request(claim)

    async def _present_audio_cpp_request(
        self,
        claim: HandoffClaim[AudioCppModelLibraryRequest],
    ) -> None:
        """Load and activate the exact audio.cpp return presentation."""

        for _ in range(200):
            if not self.is_attached or self._audio_cpp_model_request_claim is not claim:
                return
            view = self._curated_view()
            window = self.llm_window
            if view is None and window is not None:
                window.active_view = "curated"
            if view is not None and window is not None:
                view.set_consumer_filter("audio_cpp", allow_installed_return=True)
                view.ensure_loaded()
                window.active_view = "curated"
                return
            await asyncio.sleep(0.01)
        logger.warning("Audio.cpp Model Library presentation timed out")

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
        if event.active and getattr(self, "_model_install_kind", None) == "remote":
            self._sync_remote_install_context_status()

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

    def _external_view(self) -> "ExternalModelView | None":
        """Return the current deferred external-source edit view."""

        if self.llm_window is None:
            return None
        try:
            return self.llm_window.query_one(ExternalModelView)
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

    def _request_remote_machine_memory(self, *, force: bool) -> None:
        """Start or hydrate the one process-session machine observation."""
        if self._machine_memory_active and not force:
            self._hydrate_remote_machine_memory()
            return
        if not force and (
            self._machine_memory_snapshot is not None
            or self._machine_memory_generation > 0
        ):
            self._hydrate_remote_machine_memory()
            return
        self._machine_memory_generation += 1
        generation = self._machine_memory_generation
        self._machine_memory_active = True
        self._machine_memory_failure = None
        self._hydrate_remote_machine_memory()
        self._machine_memory_worker = self._run_machine_memory_probe(generation)

    @work(
        thread=True,
        group="remote_machine_memory",
        exclusive=True,
        exit_on_error=False,
        description="Observe local model memory capacity",
    )
    def _run_machine_memory_probe(self, generation: int) -> None:
        """Observe bounded local memory off-loop and return only safe facts."""
        factory = self._machine_memory_probe_factory
        if factory is None:
            from ...Model_Artifacts.machine_memory_probe import observe_machine_memory

            factory = observe_machine_memory
        try:
            result = factory()
        except Exception:
            result = None
        self.app.call_from_thread(
            self._apply_machine_memory_result,
            generation,
            result,
        )

    def _apply_machine_memory_result(
        self,
        generation: int,
        result: MachineMemorySnapshot | None,
    ) -> None:
        """Apply only the current probe, retaining valid RAM across failures."""
        if generation != self._machine_memory_generation:
            return
        self._machine_memory_active = False
        self._machine_memory_worker = None
        accepted = (
            type(result) is MachineMemorySnapshot
            and result.system_state
            in {SystemMemoryState.OBSERVED, SystemMemoryState.PARTIAL}
            and result.total_bytes is not None
        )
        current_is_valid = (
            type(self._machine_memory_snapshot) is MachineMemorySnapshot
            and self._machine_memory_snapshot.system_state
            in {SystemMemoryState.OBSERVED, SystemMemoryState.PARTIAL}
            and self._machine_memory_snapshot.total_bytes is not None
        )
        if accepted:
            self._machine_memory_snapshot = result
            self._machine_memory_observed_label = (
                self._machine_memory_wall_clock().strftime("%H:%M")
            )
            self._machine_memory_observed_monotonic = (
                self._machine_memory_monotonic_clock()
            )
            self._machine_memory_failure = None
        elif current_is_valid:
            self._machine_memory_failure = (
                result.system_reason
                if type(result) is MachineMemorySnapshot
                and result.system_reason is not None
                else ProbeReason.INVALID_MEMORY_VALUE
            )
        else:
            self._machine_memory_snapshot = (
                result if type(result) is MachineMemorySnapshot else None
            )
            self._machine_memory_failure = (
                result.system_reason
                if type(result) is MachineMemorySnapshot
                and result.system_reason is not None
                else ProbeReason.INVALID_MEMORY_VALUE
            )
        self._hydrate_remote_machine_memory()

    def _hydrate_remote_machine_memory(self) -> bool:
        """Publish retained machine facts into the currently mounted RemoteView."""
        view = self._remote_view()
        if view is None:
            return False
        presentation_snapshot = (
            self._machine_memory_snapshot
            if not self._machine_memory_active
            or (
                self._machine_memory_snapshot is not None
                and self._machine_memory_snapshot.total_bytes is not None
            )
            else None
        )
        presentation = build_machine_memory_presentation(
            presentation_snapshot,
            active=self._machine_memory_active,
            observed_at_label=self._machine_memory_observed_label,
            failure=self._machine_memory_failure,
        )
        view.apply_machine_memory_state(presentation, self._machine_memory_snapshot)
        return True

    def _active_install_view(self) -> "CuratedView | RemoteView | None":
        """Return the view rendering the currently in-flight install, if any.

        ``LLMManagementWindow`` now populates panes on first use, so either
        view may temporarily be absent during first mount or a screen-level
        recompose. Routing by
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
        """Return whether any managed-model acquisition owns the host.

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
        return self._model_install_kind is not None or getattr(
            self, "_local_gguf_import_active", False
        )

    def _can_start_local_gguf_import(self) -> bool:
        """Return whether Installed may reserve the shared host lane."""
        return not self._install_in_progress()

    def _set_local_gguf_import_active(self, active: bool) -> None:
        """Retain Installed ownership across picker, consent, and worker phases."""
        self._local_gguf_import_active = active

    # -- External Parakeet roots: screen-owned picker and workers -------

    @staticmethod
    def _external_key_for_reference(
        reference: ArtifactRef,
    ) -> ParakeetSourceKey | None:
        """Resolve only an exact catalog-known Parakeet root reference."""

        for key in ParakeetSourceKey:
            if reference == parakeet_reference(key.model_id, key.precision):
                return key
        return None

    def _next_external_token(self) -> tuple[int, int]:
        """Fence every picker and worker callback to this screen generation."""

        prior = self._external_selection_token
        if prior is not None and prior not in self._external_commit_tokens:
            self._release_external_scope(prior)
        worker = self._external_selection_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        self._external_selection_generation += 1
        token = (self._external_selection_generation, id(self))
        self._external_selection_token = token
        self._external_scope_id = f"llm-external-{token[1]}-{token[0]}"
        self._external_scope_ids[token] = self._external_scope_id
        self._external_selection_worker = None
        return token

    def _release_external_scope(self, token: tuple[int, int]) -> None:
        """Release one path-free verifier owner exactly once."""

        scope_id = self._external_scope_ids.pop(token, None)
        if scope_id is None:
            return
        if self._external_scope_id == scope_id:
            self._external_scope_id = None
        service = getattr(self.app, "_parakeet_source_service", None)
        if service is not None:
            service.release_scope(scope_id)

    def _owns_external_token(self, token: tuple[int, int]) -> bool:
        """Return whether a completion still belongs to this mounted screen."""

        return (
            token == self._external_selection_token
            and token[1] == id(self)
            and self.is_mounted
        )

    def _set_external_status(
        self,
        text: str,
        *,
        error: bool = False,
        active: bool | None = None,
    ) -> None:
        """Retain path-safe operation copy across deferred-view recomposition."""

        self._external_operation_status = text
        self._external_operation_error = error
        if active is not None:
            self._external_operation_active = active
        view = self._external_view()
        if view is not None:
            view.apply_operation_status(
                text,
                error=error,
                active=self._external_operation_active,
            )

    def _hydrate_external_status(self) -> None:
        """Apply screen-retained state to the current deferred view."""

        view = self._external_view()
        if view is None and self._external_operation_status:
            window = getattr(self, "llm_window", None)
            if window is not None:
                window.ensure_view_populated("external")
            return
        if view is not None and self._external_operation_status:
            view.apply_operation_status(
                self._external_operation_status,
                error=self._external_operation_error,
                active=self._external_operation_active,
            )

    def _reload_external_view(self) -> None:
        view = self._external_view()
        if view is not None:
            view.reload()

    @on(CuratedView.UseFromDiskRequested)
    def _use_from_disk_requested(
        self,
        event: CuratedView.UseFromDiskRequested,
    ) -> None:
        event.stop()
        self._begin_external_selection(event.reference)

    def _begin_external_selection(
        self,
        reference: ArtifactRef,
        *,
        start_directory: Path | None = None,
    ) -> None:
        """Open the real directory picker for one exact catalog root."""

        key = self._external_key_for_reference(reference)
        if key is None:
            self.notify(
                "This model does not support direct directory selection.",
                severity="error",
            )
            return
        token = self._next_external_token()
        picker = SelectDirectory(
            str(start_directory or Path.home()),
            title=f"Choose {key.model_id} {key.precision.upper()} directory",
        )
        self.app.push_screen(
            picker,
            lambda selected: self._external_directory_selected(
                token,
                key,
                selected,
            ),
        )

    def _external_directory_selected(
        self,
        token: tuple[int, int],
        key: ParakeetSourceKey,
        selected: Path | None,
    ) -> None:
        """Start verification only for the current non-cancelled picker."""

        if not self._owns_external_token(token):
            return
        if selected is None:
            self._release_external_scope(token)
            return
        window = self.llm_window
        if window is None or not window.is_mounted:
            self._release_external_scope(token)
            return
        window.active_view = "external"
        self._set_external_status("Verifying model files…", active=True)
        self._external_selection_worker = self._verify_external_source(
            token,
            key,
            Path(selected),
            "commit",
        )

    @on(ExternalModelView.ChangeRequested)
    def _change_external_source(
        self,
        event: ExternalModelView.ChangeRequested,
    ) -> None:
        event.stop()
        record = self.app._ensure_parakeet_source_service().records().get(event.key)
        self._begin_external_selection(
            parakeet_reference(event.key.model_id, event.key.precision),
            start_directory=record.directory if record is not None else None,
        )

    @on(ExternalModelView.StopRequested)
    def _stop_external_source(
        self,
        event: ExternalModelView.StopRequested,
    ) -> None:
        event.stop()
        if self._external_operation_active:
            self._cancel_external_operation()
            return
        token = self._next_external_token()
        self._set_external_status("Removing external source…", active=False)
        self._external_selection_worker = self._run_external_stop(token, event.key)

    @on(ExternalModelView.CancelRequested)
    def _cancel_first_external_operation(
        self,
        event: ExternalModelView.CancelRequested,
    ) -> None:
        event.stop()
        if self._external_operation_active:
            self._cancel_external_operation()

    def _cancel_external_operation(self) -> None:
        """Cancel the current worker and restore the prior configured state."""

        token = self._next_external_token()
        self._release_external_scope(token)
        message = "Operation cancelled. The prior source is unchanged."
        self._set_external_status(message, active=False)
        self.notify(message, severity="information")

    @work(
        thread=True,
        group="llm_external_stop",
        exclusive=True,
        exit_on_error=False,
        description="Stop using external Parakeet source",
    )
    def _run_external_stop(
        self,
        token: tuple[int, int],
        key: ParakeetSourceKey,
    ) -> None:
        """Persist external-source removal outside the Textual event loop."""

        worker = get_current_worker()
        if worker.is_cancelled or not self._owns_external_token(token):
            return
        try:
            self.app._ensure_parakeet_source_service().stop_using_external(
                key,
                cancelled=lambda: (
                    worker.is_cancelled or not self._owns_external_token(token)
                ),
            )
        except Exception as exc:
            logger.warning(
                "External Parakeet source removal failed; error_type={}",
                type(exc).__name__,
            )
            error = "The external source could not be removed. Try again."
        else:
            error = None
        self.app.call_from_thread(self._apply_external_stop_result, token, error)

    def _apply_external_stop_result(
        self,
        token: tuple[int, int],
        error: str | None,
    ) -> None:
        if not self._owns_external_token(token):
            return
        self._external_selection_worker = None
        self._release_external_scope(token)
        if error is not None:
            self._set_external_status(error, error=True, active=False)
            self.notify(error, severity="error")
            return
        self._set_external_status("External source removed.", active=False)
        self._reload_external_view()
        self.notify("External source removed.", severity="information")

    @on(ExternalModelView.CopyRequested)
    def _copy_external_source(
        self,
        event: ExternalModelView.CopyRequested,
    ) -> None:
        event.stop()
        service = self.app._ensure_parakeet_source_service()
        record = service.records().get(event.key)
        if record is None or record.directory is None:
            self._set_external_status(
                "No external directory is configured for this model.",
                error=True,
            )
            return
        token = self._next_external_token()
        self._set_external_status(
            "Verifying model files before copy…",
            active=True,
        )
        self._external_selection_worker = self._verify_external_source(
            token,
            event.key,
            record.directory,
            "copy",
        )

    @work(
        thread=True,
        group="llm_external_verify",
        exclusive=True,
        exit_on_error=False,
        description="Verify external Parakeet source",
    )
    def _verify_external_source(
        self,
        token: tuple[int, int],
        key: ParakeetSourceKey,
        directory: Path,
        action: str,
    ) -> None:
        """Hash the selected root outside the Textual event loop."""

        worker = get_current_worker()

        def progress(done: int, total: int) -> None:
            self.app.call_from_thread(
                self._apply_external_hash_progress,
                token,
                done,
                total,
            )

        try:
            prepared = self.app._ensure_parakeet_source_service().prepare_external(
                key,
                directory,
                owner=("scope", f"llm-external-{token[1]}-{token[0]}"),
                cancelled=lambda: worker.is_cancelled,
                progress=progress,
            )
        except ExternalParakeetVerificationError as exc:
            message, is_error = format_external_parakeet_recovery(exc.code)
            if is_error:
                logger.warning(
                    "External Parakeet verification rejected the selected source; "
                    "error_type={}",
                    type(exc).__name__,
                )
            self.app.call_from_thread(
                self._apply_external_verification_result,
                token,
                action,
                None,
                message,
                is_error,
            )
            return
        except Exception as exc:
            logger.warning(
                "External Parakeet verification failed unexpectedly; error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_external_verification_result,
                token,
                action,
                None,
                "The selected model could not be verified. Choose the directory again.",
                True,
            )
            return
        self.app.call_from_thread(
            self._apply_external_verification_result,
            token,
            action,
            prepared,
            None,
        )

    def _apply_external_hash_progress(
        self,
        token: tuple[int, int],
        done: int,
        total: int,
    ) -> None:
        if self._owns_external_token(token):
            self._set_external_status(
                f"Verifying model files · {done:,} / {total:,} bytes"
            )

    def _apply_external_verification_result(
        self,
        token: tuple[int, int],
        action: str,
        prepared: PreparedExternalSelection | None,
        error: str | None,
        error_is_failure: bool = True,
    ) -> None:
        """Commit, request VAD consent, or plan an optional managed copy."""

        if not self._owns_external_token(token):
            return
        self._external_selection_worker = None
        if error is not None or prepared is None:
            self._release_external_scope(token)
            message = error or "The selected model could not be verified."
            self._set_external_status(
                message,
                error=error_is_failure,
                active=False,
            )
            self.notify(
                message,
                severity="error" if error_is_failure else "information",
            )
            return
        if action == "copy":
            self._review_external_copy(token, prepared)
            return
        self._commit_external_or_request_vad(token, prepared)

    def _commit_external_or_request_vad(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        self._set_external_status("Saving external source…", active=False)
        self._external_commit_tokens.add(token)
        self._external_selection_worker = self._run_external_commit(token, prepared)

    @work(
        thread=True,
        group="llm_external_commit",
        exclusive=True,
        exit_on_error=False,
        description="Save external Parakeet source",
    )
    def _run_external_commit(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        """Persist one verified source and probe runtime readiness off-loop."""

        worker = get_current_worker()
        if worker.is_cancelled or not self._owns_external_token(token):
            self.app.call_from_thread(
                self._apply_external_commit_result,
                token,
                prepared,
                "cancelled",
                False,
            )
            return
        try:
            self.app._ensure_parakeet_source_service().commit_external(
                prepared,
                cancelled=lambda: (
                    worker.is_cancelled or not self._owns_external_token(token)
                ),
            )
        except ParakeetSourceError as exc:
            if exc.code is ParakeetSourceErrorCode.VAD_UNAVAILABLE:
                outcome = "vad"
            else:
                outcome = "error"
            runtime_ready = False
        except Exception as exc:
            logger.warning(
                "External Parakeet source save failed; error_type={}",
                type(exc).__name__,
            )
            outcome = "error"
            runtime_ready = False
        else:
            from tldw_chatbook.Utils.optional_deps import (
                parakeet_onnx_deps_installed,
            )

            outcome = "saved"
            runtime_ready = parakeet_onnx_deps_installed()
        self.app.call_from_thread(
            self._apply_external_commit_result,
            token,
            prepared,
            outcome,
            runtime_ready,
        )

    def _apply_external_commit_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        outcome: str,
        runtime_ready: bool,
    ) -> None:
        self._external_commit_tokens.discard(token)
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        if outcome == "vad":
            self._set_external_status(
                "Checking the managed VAD dependency…",
                active=False,
            )
            self._external_selection_worker = self._run_external_vad_preflight(
                token,
                prepared,
            )
            return
        self._release_external_scope(token)
        if outcome == "cancelled":
            return
        if outcome == "error":
            self._external_commit_failed()
            return
        self._finish_external_commit(runtime_ready=runtime_ready)

    def _external_commit_failed(self) -> None:
        message = (
            "The external source could not be saved. The prior source is unchanged."
        )
        self._set_external_status(message, error=True, active=False)
        self.notify(message, severity="error")

    def _finish_external_commit(self, *, runtime_ready: bool) -> None:
        message = "External source ready." if runtime_ready else "Runtime required"
        self._set_external_status(message, active=False)
        self._reload_external_view()
        self.notify(message, severity="information")

    @work(
        thread=True,
        group="llm_external_vad_preflight",
        exclusive=True,
        exit_on_error=False,
        description="Check managed VAD dependency",
    )
    def _run_external_vad_preflight(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        """Build the VAD-only acquisition plan outside the event loop."""

        from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
            run_parakeet_vad_preflight,
        )

        try:
            report = asyncio.run(run_parakeet_vad_preflight())
        except Exception as exc:
            logger.warning(
                "Managed VAD preflight failed; error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_external_vad_preflight_result,
                token,
                prepared,
                None,
                "The managed VAD dependency could not be prepared.",
            )
            return
        self.app.call_from_thread(
            self._apply_external_vad_preflight_result,
            token,
            prepared,
            report,
            None,
        )

    def _apply_external_vad_preflight_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: "PreflightReport | None",
        error: str | None,
    ) -> None:
        """Show consent only for an exact VAD-only report."""

        from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
            parakeet_vad_descriptor,
            parakeet_vad_reference,
        )

        if not self._owns_external_token(token):
            return
        self._external_selection_worker = None
        vad_reference = parakeet_vad_reference()
        vad_source_url = parakeet_vad_descriptor().source_url
        if (
            error is not None
            or report is None
            or report.root != vad_reference
            or not report.entries
            or any(
                entry.ref != vad_reference or entry.source_url != vad_source_url
                for entry in report.entries
            )
        ):
            self._release_external_scope(token)
            message = error or "The managed VAD plan changed. Choose the model again."
            self._set_external_status(message, error=True, active=False)
            self.notify(message, severity="error")
            return
        self.app.push_screen(
            ModelInstallModal(report, model_label="Silero VAD dependency"),
            lambda confirmed: self._confirm_external_vad(
                bool(confirmed),
                token,
                prepared,
                report,
            ),
        )

    def _confirm_external_vad(
        self,
        confirmed: bool,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: "PreflightReport",
    ) -> None:
        if not self._owns_external_token(token):
            return
        if not confirmed:
            self._release_external_scope(token)
            self._set_external_status(
                "VAD install cancelled. The prior source is unchanged.",
                active=False,
            )
            return
        self._set_external_status(
            "Installing the managed VAD dependency…",
            active=True,
        )
        self._external_selection_worker = self._run_external_vad_provision(
            token,
            prepared,
            report,
        )

    @work(
        group="llm_external_vad_install",
        exclusive=True,
        exit_on_error=False,
        description="Install managed VAD dependency",
    )
    async def _run_external_vad_provision(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: "PreflightReport",
    ) -> None:
        """Provision only the consented VAD dependency."""

        from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
            run_parakeet_vad_provision,
        )

        def progress(event: "AcquisitionProgress") -> None:
            self._apply_external_vad_progress(
                token,
                event.bytes_done,
                event.bytes_total,
            )

        try:
            await run_parakeet_vad_provision(report, progress=progress)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Managed VAD installation failed; error_type={}",
                type(exc).__name__,
            )
            self._apply_external_vad_provision_result(
                token,
                prepared,
                "The managed VAD dependency could not be installed.",
            )
            return
        self._apply_external_vad_provision_result(
            token,
            prepared,
            None,
        )

    def _apply_external_vad_progress(
        self,
        token: tuple[int, int],
        done: int,
        total: int,
    ) -> None:
        if self._owns_external_token(token):
            self._set_external_status(
                f"Installing managed VAD dependency · {done:,} / {total:,} bytes"
            )

    def _apply_external_vad_provision_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        error: str | None,
    ) -> None:
        if not self._owns_external_token(token):
            return
        self._external_selection_worker = None
        if error is not None:
            self._release_external_scope(token)
            self._set_external_status(error, error=True, active=False)
            self.notify(error, severity="error")
            return
        self._commit_external_or_request_vad(token, prepared)

    def _review_external_copy(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        self._set_external_status("Planning managed copy…", active=True)
        self._external_selection_worker = self._run_external_copy_plan(token, prepared)

    @work(
        thread=True,
        group="llm_external_copy_plan",
        exclusive=True,
        exit_on_error=False,
        description="Plan external Parakeet managed copy",
    )
    def _run_external_copy_plan(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        """Plan managed-store space outside the Textual event loop."""

        worker = get_current_worker()
        if worker.is_cancelled:
            return
        try:
            plan = self.app._ensure_parakeet_source_service().plan_managed_copy(
                prepared.verified
            )
        except Exception as exc:
            logger.warning(
                "External Parakeet copy planning failed; error_type={}",
                type(exc).__name__,
            )
            plan = None
            error = (
                "The managed copy could not be planned. "
                "The external source is unchanged."
            )
        else:
            error = None
        self.app.call_from_thread(
            self._apply_external_copy_plan,
            token,
            prepared,
            plan,
            error,
        )

    def _apply_external_copy_plan(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        plan: ManagedCopyPlan | None,
        error: str | None,
    ) -> None:
        """Apply only the current screen-owned copy plan."""

        if not self._owns_external_token(token):
            return
        self._external_selection_worker = None
        if error is not None or plan is None:
            self._set_external_status(
                error or "The managed copy could not be planned.",
                error=True,
                active=False,
            )
            self._release_external_scope(token)
            return
        if plan.already_installed:
            self._set_external_status(
                "This model is already in the managed store.",
                active=False,
            )
            self.notify(
                "This model is already in the managed store.",
                severity="information",
            )
            self._release_external_scope(token)
            return
        try:
            consent = plan.grant()
        except ParakeetSourceError:
            self._set_external_status(
                "Not enough managed-store space is available for this copy.",
                error=True,
                active=False,
            )
            self._release_external_scope(token)
            return
        self.app.push_screen(
            ConfirmationDialog(
                title="Copy into managed store?",
                message=(
                    f"Copy {plan.additional_bytes / 1024:.1f} KiB into Chatbook's "
                    "managed store? The external source remains active."
                ),
                confirm_label="Copy",
                cancel_label="Cancel",
            ),
            lambda confirmed: self._confirm_external_copy(
                bool(confirmed),
                token,
                prepared,
                consent,
            ),
        )
        self._external_operation_active = False

    def _confirm_external_copy(
        self,
        confirmed: bool,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        consent: ManagedCopyConsent,
    ) -> None:
        if not self._owns_external_token(token):
            return
        if not confirmed:
            self._release_external_scope(token)
            self._set_external_status(
                "Managed copy cancelled. External source unchanged.",
                active=False,
            )
            return
        self._set_external_status(
            "Copying model into the managed store…",
            active=True,
        )
        self._external_selection_worker = self._run_external_copy(
            token,
            prepared,
            consent,
        )

    @work(
        thread=True,
        group="llm_external_copy",
        exclusive=True,
        exit_on_error=False,
        description="Copy external Parakeet source",
    )
    def _run_external_copy(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        consent: ManagedCopyConsent,
    ) -> None:
        worker = get_current_worker()
        if worker.is_cancelled or not self._owns_external_token(token):
            return
        try:
            self.app._ensure_parakeet_source_service().copy_into_managed(
                prepared.verified,
                consent,
                cancelled=lambda: worker.is_cancelled,
            )
        except Exception as exc:
            logger.warning(
                "External Parakeet managed copy failed; error_type={}",
                type(exc).__name__,
            )
            error = "Managed copy failed. The external source is unchanged."
        else:
            error = None
        self.app.call_from_thread(
            self._apply_external_copy_result,
            token,
            error,
        )

    def _apply_external_copy_result(
        self,
        token: tuple[int, int],
        error: str | None,
    ) -> None:
        if not self._owns_external_token(token):
            return
        self._external_selection_worker = None
        self._release_external_scope(token)
        if error is not None:
            self._set_external_status(error, error=True, active=False)
            self.notify(error, severity="error")
            return
        message = "Model copied into the managed store. Activate it when ready."
        self._set_external_status(message, active=False)
        self.notify(message, severity="information")

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
            or type(event.already_installed) is not bool
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
        if event.already_installed:
            try:
                descriptor = event.registry.descriptor(event.reference)
            except (KeyError, TypeError, ValueError):
                descriptor = None
            if descriptor is None or descriptor.consumer != "audio_cpp":
                self.notify(
                    "Could not use the installed package: invalid request.",
                    severity="error",
                )
                self._clear_curated_install_state()
                return
            self._deliver_curated(InstallStatusChanged(event.reference, active=True))
            self._start_audio_cpp_installed_return()
            return
        try:
            descriptor = event.registry.descriptor(event.reference)
        except (KeyError, TypeError, ValueError):
            descriptor = None
        if descriptor is not None and descriptor.consumer == "audio_cpp":
            self._start_audio_cpp_preflight()
            return
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
            logger.error(
                "Curated model preflight failed; error_type={}",
                type(exc).__name__,
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
            self._clear_curated_install_state(error or "Model preflight failed.")
            return
        registry = self._model_install_registry
        if registry is None:
            self.notify("Model preflight state is unavailable.", severity="error")
            self._clear_curated_install_state()
            return
        self._model_install_pending_report = report
        descriptor = registry.descriptor(report.root)
        self.app.push_screen(
            ModelInstallModal(report, model_label=descriptor.model_id),
            self._confirm_curated_install,
        )

    def _confirm_curated_install(self, confirmed: bool) -> None:
        """Start provisioning only after explicit consent."""
        if not confirmed:
            report = self._model_install_pending_report
            self._clear_curated_install_state(_insufficient_space_recovery(report))
            return
        reference = self._model_install_reference
        if reference is not None:
            self._deliver_curated(InstallStatusChanged(reference, active=True))
        self._model_install_worker = self._run_curated_provision()

    async def _provision_curated(
        self,
        report: "PreflightReport",
        cancel_event: threading.Event | None = None,
    ):
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
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError
            message = InstallProgressed(progress)
            if cancel_event is None:
                self._deliver_curated(message)
            else:

                def deliver_current() -> None:
                    operation = self._audio_cpp_model_install_operation
                    if (
                        operation is not None
                        and operation.cancel_event is cancel_event
                        and not cancel_event.is_set()
                        and self.is_attached
                    ):
                        self._deliver_curated(message)

                self.app.call_from_thread(deliver_current)

        kwargs: dict[str, object] = {
            "sources": self._model_install_sources,
            "progress": deliver,
        }
        registry = self._model_install_registry
        if registry is None:
            raise RuntimeError("curated model registry is unavailable")
        descriptor = registry.descriptor(report.root)
        if descriptor.consumer == "audio_cpp":
            kwargs["activate"] = False
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        provisioned = await acquisition.provision(
            report.root,
            report.grant(),
            registry,
            **kwargs,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        return provisioned

    def _audio_cpp_installed_result(
        self,
        reference: ArtifactRef,
    ) -> AudioCppModelLibraryResult | None:
        """Lease and verify one exact installed root, then detach its return."""

        service = self._model_install_service
        if service is None:
            raise RuntimeError("model artifact service is unavailable")
        with service.acquire_installed_root(reference) as leased:
            paths = dict(leased.handle.paths)
            canonical_root = paths[reference]
            claim = self._audio_cpp_model_request_claim
            if claim is None:
                return None
            request = claim.value
            return AudioCppModelLibraryResult(
                token=request.token,
                draft_revision=request.draft_revision,
                artifact_id=reference.artifact_id,
                revision=reference.revision,
                variant=reference.variant,
                canonical_root=str(canonical_root),
            )

    def _audio_cpp_install_owner(self) -> AudioCppModelInstallOwner:
        """Return the app-owned durable audio.cpp operation owner."""

        owner = getattr(self.app_instance, "audio_cpp_model_install_owner", None)
        if type(owner) is not AudioCppModelInstallOwner:
            raise RuntimeError("audio.cpp install owner is unavailable")
        return owner

    def _start_audio_cpp_operation(
        self,
        *,
        installed: bool,
        include_preflight: bool = False,
    ) -> None:
        """Start one durable audio.cpp generation across requested phases."""

        owner = self._audio_cpp_install_owner()
        reference = self._model_install_reference
        report = self._model_install_pending_report
        operation: AudioCppModelInstallOperation | None = None

        async def runner(
            cancel_event: threading.Event,
        ) -> AudioCppModelLibraryResult | None:
            if installed:
                if reference is None:
                    raise RuntimeError("installed audio.cpp package is unavailable")
                if cancel_event.is_set():
                    raise asyncio.CancelledError
                result = await asyncio.to_thread(
                    self._audio_cpp_installed_result, reference
                )
                if cancel_event.is_set():
                    raise asyncio.CancelledError
                return result
            active_report = report
            if include_preflight:
                if reference is None:
                    raise RuntimeError("audio.cpp install request is unavailable")
                active_report = await asyncio.to_thread(
                    lambda: asyncio.run(self._preflight_curated(reference))
                )
                if cancel_event.is_set() or not self.is_attached:
                    raise asyncio.CancelledError
                assert operation is not None
                confirmed = await self._await_audio_cpp_consent(
                    operation,
                    active_report,
                    cancel_event,
                )
                if not confirmed:
                    raise _AudioCppConsentDeclined
                if cancel_event.is_set() or not self.is_attached:
                    raise asyncio.CancelledError
                self._deliver_curated(InstallStatusChanged(reference, active=True))
            if active_report is None:
                raise RuntimeError("audio.cpp install plan is unavailable")
            provisioned = await asyncio.to_thread(
                lambda: asyncio.run(
                    self._provision_curated(active_report, cancel_event)
                )
            )
            return await asyncio.to_thread(
                self._audio_cpp_installed_result, provisioned
            )

        def settled(
            result: AudioCppModelLibraryResult | None,
            error: BaseException | None,
            cancelled: bool,
        ) -> None:
            assert operation is not None
            self._audio_cpp_operation_settled(
                operation,
                result,
                error,
                cancelled,
            )

        operation = owner.start(runner, settled)
        self._audio_cpp_operation_expects_return = (
            self._audio_cpp_model_request_claim is not None
        )
        self._audio_cpp_model_install_operation = operation
        self._model_install_worker = self._wait_audio_cpp_operation(operation)

    def _start_audio_cpp_installed_return(self) -> None:
        """Start a durable exact installed-root verification and return."""

        self._start_audio_cpp_operation(installed=True)

    def _start_audio_cpp_preflight(self) -> None:
        """Start one generation spanning preflight, consent, and provision."""

        self._start_audio_cpp_operation(installed=False, include_preflight=True)

    async def _await_audio_cpp_consent(
        self,
        operation: AudioCppModelInstallOperation,
        report: "PreflightReport",
        cancel_event: threading.Event,
    ) -> bool:
        """Present and await consent only for the current mounted generation."""

        if (
            cancel_event.is_set()
            or not self.is_attached
            or self._audio_cpp_model_install_operation is not operation
        ):
            raise asyncio.CancelledError
        registry = self._model_install_registry
        if registry is None:
            raise RuntimeError("audio.cpp preflight state is unavailable")
        descriptor = registry.descriptor(report.root)
        self._model_install_pending_report = report
        future = asyncio.get_running_loop().create_future()
        self._audio_cpp_consent_future = future
        modal = ModelInstallModal(
            report,
            model_label=descriptor.model_id,
            selected_file_details=self._audio_cpp_selected_file_details(report),
        )
        self._audio_cpp_consent_modal = modal
        self.app.push_screen(
            modal,
            lambda confirmed: self._resolve_audio_cpp_consent(
                operation,
                confirmed,
            ),
        )
        while not future.done():
            if (
                cancel_event.is_set()
                or not self.is_attached
                or self._audio_cpp_model_install_operation is not operation
            ):
                raise asyncio.CancelledError
            await asyncio.sleep(0.01)
        if (
            cancel_event.is_set()
            or not self.is_attached
            or self._audio_cpp_model_install_operation is not operation
        ):
            raise asyncio.CancelledError
        return future.result()

    def _audio_cpp_selected_file_details(
        self,
        report: "PreflightReport",
    ) -> tuple[tuple[str, int, str, str], ...]:
        """Return exact immutable file facts for the reviewed closure."""

        registry = self._model_install_registry
        sources = self._model_install_sources
        if registry is None or sources is None:
            raise RuntimeError("audio.cpp source review state is unavailable")
        references = tuple(entry.ref for entry in report.entries) or (report.root,)
        return tuple(
            (artifact_file.path, artifact_file.size_bytes, artifact_file.sha256, source)
            for reference in references
            for artifact_file in registry.descriptor(reference).files
            for source in (sources[reference][artifact_file.path],)
        )

    def _resolve_audio_cpp_consent(
        self,
        operation: AudioCppModelInstallOperation,
        confirmed: bool,
    ) -> None:
        """Resolve consent only for the still-current audio.cpp generation."""

        future = self._audio_cpp_consent_future
        if (
            self._audio_cpp_model_install_operation is not operation
            or future is None
            or future.done()
        ):
            return
        self._audio_cpp_consent_modal = None
        future.set_result(bool(confirmed))

    @work(group="llm_curated_install", exit_on_error=False)
    async def _wait_audio_cpp_operation(
        self,
        operation: AudioCppModelInstallOperation,
    ) -> None:
        """Tie Textual cancellation to the durable owner, then join it."""

        owner = self._audio_cpp_install_owner()
        try:
            await owner.wait(operation)
        except asyncio.CancelledError:
            owner.request_cancel(operation)
            while not operation.task.done():
                try:
                    await owner.wait(operation)
                except asyncio.CancelledError:
                    continue
            raise

    def _audio_cpp_operation_settled(
        self,
        operation: AudioCppModelInstallOperation,
        result: AudioCppModelLibraryResult | None,
        error: BaseException | None,
        cancelled: bool,
    ) -> None:
        """Apply one owner-settled outcome without exposing private details."""

        if self._audio_cpp_model_install_operation is not operation:
            return
        self._audio_cpp_model_install_operation = None
        self._audio_cpp_consent_future = None
        self._audio_cpp_consent_modal = None
        expects_return = self._audio_cpp_operation_expects_return
        self._audio_cpp_operation_expects_return = False
        if cancelled or not self.is_attached:
            self._settle_detached_audio_cpp_operation()
            return
        if isinstance(error, _AudioCppConsentDeclined):
            self._model_install_worker = None
            self._clear_curated_install_state(
                _insufficient_space_recovery(self._model_install_pending_report)
            )
            return
        if error is not None:
            reference = self._model_install_reference
            artifact_id = (
                reference.artifact_id if reference is not None else "audio.cpp model"
            )
            logger.error(
                "Audio.cpp model installation failed; error_type={}",
                type(error).__name__,
            )
            self._apply_audio_cpp_provision_result(
                None,
                install_failure_message(error, model_label=artifact_id),
            )
            return
        if not expects_return:
            self._apply_audio_cpp_standalone_result()
            return
        self._apply_audio_cpp_provision_result(result, None)

    def _settle_detached_audio_cpp_operation(self) -> None:
        """Release request/state after actual work settles, without late UI."""

        claim = self._audio_cpp_model_request_claim
        store = getattr(self.app_instance, "pending_handoffs", None)
        if claim is not None and type(store) is PendingHandoffStore:
            cast(PendingHandoffStore, store).release(claim)
        self._audio_cpp_model_request_claim = None
        self._model_install_worker = None
        self._model_install_reference = None
        self._model_install_service = None
        self._model_install_registry = None
        self._model_install_sources = None
        self._model_install_pending_report = None
        self._model_install_kind = None

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
        app = self.app
        report = self._model_install_pending_report
        if report is None:
            app.call_from_thread(
                self._apply_curated_provision_result,
                "No install plan is available; review the model again.",
            )
            return
        try:
            reference = asyncio.run(
                self._provision_curated(report)
            )  # policy-exception: worker-thread loop
        except (Exception, asyncio.CancelledError) as exc:
            root = getattr(report, "root", None)
            artifact_id = getattr(root, "artifact_id", "unknown")
            logger.error(
                "Curated model installation failed; error_type={}",
                type(exc).__name__,
            )
            error = (
                "Model installation was cancelled."
                if isinstance(exc, asyncio.CancelledError)
                else install_failure_message(exc, model_label=artifact_id)
            )
            app.call_from_thread(self._apply_curated_provision_result, error)
            return
        key = self._external_key_for_reference(reference)
        if key is None:
            app.call_from_thread(self._apply_curated_provision_result, None)
            return
        try:
            app._ensure_parakeet_source_service().prefer_managed(key)
        except Exception as exc:
            logger.warning(
                "Activated Parakeet source preference update failed; error_type={}",
                type(exc).__name__,
            )
            error = (
                "Model installed, but the managed source preference could not be saved."
            )
        else:
            error = None
        app.call_from_thread(
            self._apply_curated_preference_result,
            reference,
            error,
        )

    def _apply_curated_provision_result(self, error: str | None) -> None:
        """Finish an installation: notify, mirror lifecycle, and reset state."""
        self._model_install_worker = None
        self._model_install_pending_report = None
        self._finish_curated_provision(error, succeeded=error is None)

    def _apply_audio_cpp_provision_result(
        self,
        result: AudioCppModelLibraryResult | None,
        error: str | None,
    ) -> None:
        """Stage one exact installed result and settle its request once."""

        self._model_install_worker = None
        self._model_install_pending_report = None
        claim = self._audio_cpp_model_request_claim
        store = getattr(self.app_instance, "pending_handoffs", None)
        returned_for_review = False
        installed_but_return_failed = False
        if error is None and result is None:
            error = "Model installed, but it could not be returned for review."
        if error is None and result is not None and claim is not None:
            request = claim.value
            reference = self._model_install_reference
            if (
                result.token != request.token
                or result.draft_revision != request.draft_revision
                or reference is None
                or result.artifact_id != reference.artifact_id
                or result.revision != reference.revision
                or result.variant != reference.variant
            ):
                installed_but_return_failed = True
                error = (
                    "Installed, but the Settings return expired. Reopen Guided "
                    "Settings and choose this package again."
                )
        if error is None and result is not None and claim is not None:
            staged = False
            try:
                if type(store) is not PendingHandoffStore:
                    raise HandoffValueError("handoff store is unavailable")
                handoffs = cast(PendingHandoffStore, store)
                handoffs.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT, result)
                staged = True
                if not handoffs.acknowledge(claim):
                    raise HandoffValueError("handoff request is no longer current")
            except (HandoffValueError, RuntimeError):
                if staged:
                    handoffs.clear_pending(
                        HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT
                    )
                installed_but_return_failed = True
                error = (
                    "Installed, but the Settings return expired. Reopen Guided "
                    "Settings and choose this package again."
                )
            else:
                self._audio_cpp_model_request_claim = None
                returned_for_review = True
                view = self._curated_view()
                if view is not None:
                    view.set_consumer_filter(
                        "audio_cpp",
                        allow_installed_return=False,
                    )
        if error is not None and not self.is_attached and claim is not None:
            if type(store) is PendingHandoffStore:
                cast(PendingHandoffStore, store).release(claim)
            self._audio_cpp_model_request_claim = None
        self._finish_curated_provision(
            error,
            succeeded=error is None or installed_but_return_failed,
            success_message=(
                "Installed — ready for review" if returned_for_review else "Installed"
            ),
        )

    def _apply_audio_cpp_standalone_result(self) -> None:
        """Finish a claim-less audio.cpp install without claiming a return."""

        self._model_install_worker = None
        self._model_install_pending_report = None
        self._finish_curated_provision(
            None,
            succeeded=True,
            success_message="Installed",
        )

    def _apply_curated_preference_result(
        self,
        reference: ArtifactRef,
        error: str | None,
    ) -> None:
        if (
            not self.is_attached
            or self._model_install_kind != "curated"
            or self._model_install_reference != reference
        ):
            return
        self._model_install_worker = None
        self._finish_curated_provision(error, succeeded=True)

    def _finish_curated_provision(
        self,
        error: str | None,
        *,
        succeeded: bool,
        success_message: str = "Model installed and activated.",
    ) -> None:
        """Deliver one terminal curated lifecycle result and clear state."""

        reference = self._model_install_reference
        if error is not None:
            self.notify(error, severity="error")
        else:
            self.notify(success_message, severity="information")
        if reference is not None:
            self._deliver_curated(
                InstallStatusChanged(reference, active=False, succeeded=succeeded)
            )
        self._model_install_reference = None
        self._model_install_service = None
        self._model_install_registry = None
        self._model_install_sources = None
        self._model_install_pending_report = None
        self._model_install_kind = None
        view = self._curated_view()
        if view is not None:
            view.finish_install(error)

    def on_unmount(self) -> None:
        """Cancel screen-owned work and release live verifier ownership."""

        operation = self._audio_cpp_model_install_operation
        if operation is not None:
            self._audio_cpp_install_owner().request_cancel(operation)
        modal = self._audio_cpp_consent_modal
        self._audio_cpp_consent_modal = None
        if modal is not None and modal.is_attached:
            modal.dismiss(False)
        reclaim_worker = self._audio_cpp_reclaim_worker
        if reclaim_worker is not None and not reclaim_worker.is_finished:
            reclaim_worker.cancel()
        presentation_worker = self._audio_cpp_presentation_worker
        if presentation_worker is not None and not presentation_worker.is_finished:
            presentation_worker.cancel()
        worker = self._external_selection_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        token = self._external_selection_token
        if token is not None and token not in self._external_commit_tokens:
            self._release_external_scope(token)
        self._external_selection_token = None
        claim = self._audio_cpp_model_request_claim
        store = getattr(self.app_instance, "pending_handoffs", None)
        if (
            claim is not None
            and not self._install_in_progress()
            and type(store) is PendingHandoffStore
        ):
            cast(PendingHandoffStore, store).release(claim)
            self._audio_cpp_model_request_claim = None

    def _clear_curated_install_state(self, message: str | None = None) -> None:
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
            if message is None:
                view.cancel_pending_install()
            else:
                view.cancel_pending_install(message)

    def _deliver_curated(
        self, message: InstallProgressed | InstallStatusChanged
    ) -> None:
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

    @on(RemoteView.MachineMemoryRequested)
    def _remote_machine_memory_requested(
        self,
        event: RemoteView.MachineMemoryRequested,
    ) -> None:
        """Delegate the presentation-only intent to screen-owned acquisition."""
        event.stop()
        self._request_remote_machine_memory(force=event.force)

    @on(RemoteView.OpenInstalledRequested)
    def _remote_open_installed_requested(
        self, event: RemoteView.OpenInstalledRequested
    ) -> None:
        """Open the exact verified row without activating or starting it."""
        event.stop()
        if self.llm_window is None or type(event.reference) is not ArtifactRef:
            return
        self._pending_installed_reveal = event.reference
        self.llm_window.active_view = "installed"
        self.llm_window.ensure_view_populated("installed")
        self._replay_pending_installed_reveal()

    def _replay_pending_installed_reveal(self) -> None:
        """Reveal a retained exact root once the Installed body is mounted."""

        reference = getattr(self, "_pending_installed_reveal", None)
        if reference is None:
            return
        installed = self._installed_view()
        if installed is None:
            return
        self._pending_installed_reveal = None
        self.call_after_refresh(installed.reveal_reference, reference)

    @on(RemoteView.ConfigureRuntimeRequested)
    def _remote_configure_runtime_requested(
        self, event: RemoteView.ConfigureRuntimeRequested
    ) -> None:
        """Present compatible runtime destinations for one verified root."""
        event.stop()
        if type(event.reference) is not ArtifactRef:
            return
        self.app.push_screen(
            ManagedGGUFRuntimeChoiceModal(),
            partial(self._remote_runtime_selected, event.reference),
        )

    def _remote_runtime_selected(
        self,
        reference: ArtifactRef,
        provider: str | None,
    ) -> None:
        """Apply one explicit runtime choice without activating or starting it."""
        if provider not in {"llamacpp", "llamafile"} or self.llm_window is None:
            return
        self._remote_runtime_handoff = (provider, reference)
        if not self.llm_window.configure_managed_gguf(provider, reference):
            self._remote_runtime_handoff = None
            self.notify(
                "Stop the active Llama.cpp or Llamafile server, then configure "
                "this managed model again.",
                severity="warning",
            )

    def _replay_remote_runtime_handoff(self) -> None:
        """Resume an exact runtime handoff in the current management window."""
        if self.llm_window is None or self._remote_runtime_handoff is None:
            return
        provider, reference = self._remote_runtime_handoff
        if not self.llm_window.configure_managed_gguf(provider, reference):
            self._remote_runtime_handoff = None
            self.notify(
                "Stop the active Llama.cpp or Llamafile server, then configure "
                "this managed model again.",
                severity="warning",
            )

    @on(LLMManagementWindow.ManagedGGUFHandoffResolved)
    def _managed_gguf_handoff_resolved(
        self,
        event: LLMManagementWindow.ManagedGGUFHandoffResolved,
    ) -> None:
        """Clear only the exact screen-owned handoff a window resolved."""
        event.stop()
        pending = getattr(self, "_remote_runtime_handoff", None)
        if pending != (event.provider, event.reference):
            return
        self._remote_runtime_handoff = None
        if event.succeeded:
            return
        if event.reason == "inventory-error":
            message = (
                "Managed models could not be loaded. Refresh Installed models, "
                "then try again."
            )
        elif event.reason == "server-active":
            message = (
                "Stop the active Llama.cpp or Llamafile server, then configure "
                "this managed model again."
            )
        else:
            message = (
                "That managed model is no longer available. Refresh Installed "
                "models, then try again."
            )
        self.notify(message, severity="warning")

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
        self._clear_remote_terminal_presentation()
        self._clear_remote_completed_presentation()
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
            model_label = getattr(artifact, "model_id", "unknown")
            logger.error(
                "Remote model preflight failed; error_type={}, retryable={}",
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
        catalog = getattr(self, "_model_install_catalog", None)
        candidate = getattr(self, "_model_install_candidate", None)
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
        self._sync_remote_install_context_status()
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
            asyncio.run(
                self._provision_remote(report, catalog)
            )  # policy-exception: worker-thread loop
        except Exception as exc:
            from tldw_chatbook.Model_Artifacts.acquisition import TransferError

            artifact = getattr(catalog, "artifact", None)
            model_label = getattr(artifact, "model_id", "unknown")
            logger.error(
                "Remote model installation failed; error_type={}, retryable={}",
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
            if (
                isinstance(reference, ArtifactRef)
                and isinstance(self._model_install_catalog, ResolvedRemoteCatalog)
                and isinstance(self._model_install_candidate, RemoteGGUFCandidate)
            ):
                self._remote_install_completed_catalog = self._model_install_catalog
                self._remote_install_completed_candidate = self._model_install_candidate
                self._remote_install_completed_reference = reference
                self._remote_install_completed_message = message
        if reference is not None:
            self._deliver_curated(
                InstallStatusChanged(reference, active=False, succeeded=error is None)
            )
        self._deliver_or_retain_remote_terminal_presentation(
            _REMOTE_INSTALL_TERMINAL_FINISH,
            message,
        )
        self._model_install_reference = None
        self._model_install_service = None
        self._model_install_catalog = None
        self._model_install_candidate = None
        self._model_install_credential_resolver = None
        self._model_install_kind = None

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
        self._deliver_or_retain_remote_terminal_presentation(
            _REMOTE_INSTALL_TERMINAL_CANCEL,
            message,
        )
        self._model_install_reference = None
        self._model_install_service = None
        self._model_install_catalog = None
        self._model_install_candidate = None
        self._model_install_credential_resolver = None
        self._model_install_pending_report = None
        self._model_install_kind = None

    def _clear_remote_terminal_presentation(self) -> None:
        """Discard a terminal Remote presentation after a mounted view consumes it."""
        self._remote_install_terminal_catalog = None
        self._remote_install_terminal_candidate = None
        self._remote_install_terminal_action = None
        self._remote_install_terminal_message = None

    def _clear_remote_completed_presentation(self) -> None:
        """Discard the last success when a new Remote journey supersedes it."""
        self._remote_install_completed_catalog = None
        self._remote_install_completed_candidate = None
        self._remote_install_completed_reference = None
        self._remote_install_completed_message = None

    @on(RemoteView.DiscoveryStarted)
    def _remote_discovery_started(self, event: RemoteView.DiscoveryStarted) -> None:
        """Make a submitted discovery the new Remote lifecycle authority."""
        event.stop()
        self._clear_remote_completed_presentation()

    def _deliver_or_retain_remote_terminal_presentation(
        self,
        action: str,
        message: str | None,
    ) -> None:
        """Show a Remote outcome now, or retain it across the remount gap.

        Args:
            action: ``"finish"`` for provisioning outcomes or ``"cancel"``
                for preflight/consent termination.
            message: Sanitized outcome copy, or ``None`` for a decline.
        """
        catalog = self._model_install_catalog
        candidate = self._model_install_candidate
        view = self._remote_view()
        if view is not None and view.is_mounted:
            if isinstance(catalog, ResolvedRemoteCatalog) and isinstance(
                candidate, RemoteGGUFCandidate
            ):
                view.restore_install_context(catalog, candidate)
            if action == _REMOTE_INSTALL_TERMINAL_FINISH:
                completed = getattr(self, "_remote_install_completed_reference", None)
                if isinstance(completed, ArtifactRef):
                    view.finish_install(message, completed_reference=completed)
                else:
                    view.finish_install(message)
            else:
                view.cancel_pending_install(message)
            self._clear_remote_terminal_presentation()
            return
        if isinstance(catalog, ResolvedRemoteCatalog) and isinstance(
            candidate, RemoteGGUFCandidate
        ):
            self._remote_install_terminal_catalog = catalog
            self._remote_install_terminal_candidate = candidate
            self._remote_install_terminal_action = action
            self._remote_install_terminal_message = message

    def _hydrate_remote_terminal_presentation(self) -> bool:
        """Deliver one retained Remote outcome to the fresh mounted view."""
        catalog = getattr(self, "_remote_install_terminal_catalog", None)
        candidate = getattr(self, "_remote_install_terminal_candidate", None)
        action = getattr(self, "_remote_install_terminal_action", None)
        if (
            not isinstance(catalog, ResolvedRemoteCatalog)
            or not isinstance(candidate, RemoteGGUFCandidate)
            or action not in _REMOTE_INSTALL_TERMINAL_ACTIONS
        ):
            return False
        view = self._remote_view()
        if view is None or not view.is_mounted:
            return False
        if not view.restore_install_context(catalog, candidate):
            return False
        message = getattr(self, "_remote_install_terminal_message", None)
        if action == _REMOTE_INSTALL_TERMINAL_FINISH:
            completed = getattr(self, "_remote_install_completed_reference", None)
            if isinstance(completed, ArtifactRef):
                view.finish_install(message, completed_reference=completed)
            else:
                view.finish_install(message)
        else:
            view.cancel_pending_install(message)
        self._clear_remote_terminal_presentation()
        return True

    def _hydrate_remote_completed_presentation(self) -> bool:
        """Restore the durable verified completion into a fresh Remote view."""
        catalog = getattr(self, "_remote_install_completed_catalog", None)
        candidate = getattr(self, "_remote_install_completed_candidate", None)
        reference = getattr(self, "_remote_install_completed_reference", None)
        if (
            not isinstance(catalog, ResolvedRemoteCatalog)
            or not isinstance(candidate, RemoteGGUFCandidate)
            or not isinstance(reference, ArtifactRef)
        ):
            return False
        view = self._remote_view()
        if view is None or not view.is_mounted:
            return False
        if not view.restore_install_context(catalog, candidate):
            return False
        view.finish_install(
            getattr(self, "_remote_install_completed_message", None),
            completed_reference=reference,
        )
        return True

    def _model_install_presentation_pending(self) -> bool:
        """Return whether a remounted install view needs host hydration."""
        return (
            self._model_install_active
            or (
                self._model_install_kind == "remote"
                and self._model_install_catalog is not None
                and self._model_install_candidate is not None
            )
            or getattr(self, "_remote_install_terminal_action", None) is not None
            or getattr(self, "_remote_install_completed_reference", None) is not None
        )

    def _remote_install_context_status(self) -> str:
        """Return truthful lifecycle copy for a reconstructed Remote detail."""
        if self._model_install_active or (
            self._model_install_pending_report is not None
            and self._model_install_worker is not None
        ):
            return "Installing the selected GGUF variant…"
        if self._model_install_pending_report is not None:
            return "Awaiting review; no download has started."
        return "Preparing the managed install plan…"

    def _sync_remote_install_context_status(self) -> bool:
        """Apply the current host lifecycle copy to the mounted Remote detail."""
        catalog = self._model_install_catalog
        candidate = self._model_install_candidate
        if not isinstance(catalog, ResolvedRemoteCatalog) or not isinstance(
            candidate, RemoteGGUFCandidate
        ):
            return False
        view = self._remote_view()
        if view is None or not view.is_mounted:
            return False
        return view.restore_install_context(
            catalog,
            candidate,
            status_message=self._remote_install_context_status(),
        )

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
                # id, mirroring other opaque Library row identities.
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
        self.llm_window = LLMManagementWindow(
            self.app_instance,
            can_start_import=self._can_start_local_gguf_import,
            on_import_lane_changed=self._set_local_gguf_import_active,
            classes="window",
        )
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
        if self._model_install_presentation_pending():
            self.call_after_refresh(self._hydrate_model_install_progress)

    @on(LLMManagementWindow.DeferredViewsMounted)
    def _on_deferred_views_mounted(self) -> None:
        """Re-run hydration whenever a lazy Models pane becomes ready.

        The window posts this after each first-used pane has composed its
        descendants. Hydration is idempotent, so the initial and later
        notifications safely share this handler.
        """
        claim = self._audio_cpp_model_request_claim
        if claim is not None:
            self._audio_cpp_presentation_worker = self.run_worker(
                self._present_audio_cpp_request(claim),
                group="audio_cpp_request_presentation",
                exclusive=True,
                exit_on_error=False,
            )
        if self._model_install_presentation_pending():
            self._hydrate_model_install_progress()
        self._hydrate_external_status()
        self._hydrate_remote_machine_memory()
        self._replay_remote_runtime_handoff()
        self._replay_pending_installed_reveal()

    def _hydrate_model_install_progress(self) -> None:
        """Re-apply selected-model context and progress after a recompose.

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
        window = getattr(self, "llm_window", None)
        install_kind = self._model_install_kind
        if window is not None:
            if install_kind in {"curated", "remote"}:
                window.ensure_view_populated(install_kind)
            if self._model_install_active:
                window.ensure_view_populated("installed")
        if not self._hydrate_remote_terminal_presentation():
            self._hydrate_remote_completed_presentation()
        view = self._active_install_view()
        if (
            isinstance(view, RemoteView)
            and self._model_install_catalog is not None
            and self._model_install_candidate is not None
        ):
            view.restore_install_context(
                self._model_install_catalog,
                self._model_install_candidate,
                status_message=self._remote_install_context_status(),
            )
        if not self._model_install_active:
            return
        if self._model_install_last_progress is None:
            return
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
            row.set_class(
                getattr(row, "lab_view_key", None) == active_view, "is-active"
            )

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
