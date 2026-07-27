"""Models: the Lab destination's provider and model management screen."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual import on
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Button, Static

from ..Lab_Modules.lab_server_status import (
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
            ("local-models", "Local Models"),
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

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return the Models destination header copy."""
        return WorkbenchHeaderState(
            title="Models",
            subtitle="Manage providers, models, and endpoints.",
            status="ready",
        )

    def lab_status_chips(self) -> tuple[LabStatusChip, ...]:
        """Return the running-server chip.

        Returns:
            A single chip summarising how many local servers are alive.
        """
        rows = read_server_rows(self.app_instance)
        return (LabStatusChip(chip_id="servers", text=servers_chip_text(rows)),)

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
        for row in read_server_rows(self.app_instance):
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
            for row in read_server_rows(self.app_instance)
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
        """
        if self.llm_window is None:
            return
        self.watch(self.llm_window, "active_view", self._sync_rail_active, init=True)
        self.refresh_lab_status()
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
