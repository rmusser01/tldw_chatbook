"""Models: the Lab destination's provider and model management screen."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual import on
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Button, Static

from ...Widgets.ModelArtifacts import InstallProgressed, InstallStatusChanged
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

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

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
        """Keep the Lab chip current while a Curated install runs."""
        self._model_install_active = True
        self._model_install_phase = event.progress.phase
        self._model_install_succeeded = None
        self.refresh_lab_status()

    @on(InstallStatusChanged)
    def _model_install_status_changed(self, event: InstallStatusChanged) -> None:
        """Reflect managed-install start and completion in the Lab header."""
        self._model_install_active = event.active
        self._model_install_succeeded = event.succeeded
        if not event.active:
            self._model_install_phase = None
        self.refresh_lab_status()

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
