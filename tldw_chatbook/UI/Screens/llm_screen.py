"""LLM Management screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..LLM_Management_Window import LLMManagementWindow
from ..Workbench.workbench_state import WorkbenchHeaderState, WorkbenchStatus
from ..Workbench.workbench_widgets import DestinationHeader
from .lab_mode_strip import LabModeStrip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


#: App-instance attributes tracking launched server processes, by backend.
_SERVER_PROCESS_ATTRS: tuple[tuple[str, str], ...] = (
    ("llamacpp_server_process", "Llama.cpp"),
    ("llamafile_server_process", "Llamafile"),
    ("ollama_server_process", "Ollama"),
    ("vllm_server_process", "vLLM"),
    ("onnx_server_process", "ONNX"),
    ("mlx_server_process", "MLX-LM"),
)

_STATUS_REFRESH_SECONDS = 3.0


class LLMScreen(BaseAppScreen):
    """
    LLM Management screen wrapper.
    """

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "llm", **kwargs)
        self.llm_window = None

    def compose_content(self) -> ComposeResult:
        """Compose the LLM management window content with its destination header."""
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Models",
                subtitle="Manage providers, models, and endpoints.",
                status="empty",
                status_label="No server running",
            ),
            id="llm-destination-header",
        )
        yield LabModeStrip(active_route="llm", id="lab-mode-strip")
        self.llm_window = LLMManagementWindow(self.app_instance, classes="window")
        # Leave room for the destination header above the window.
        self.llm_window.styles.height = "1fr"
        # Yield the window widget directly
        yield self.llm_window

    def on_mount(self) -> None:
        super().on_mount()
        self._refresh_status_chip()
        self.set_interval(_STATUS_REFRESH_SECONDS, self._refresh_status_chip)

    def _running_servers(self) -> list[str]:
        """Return labels of backends whose tracked server process is alive."""
        running: list[str] = []
        for attr, label in _SERVER_PROCESS_ATTRS:
            proc = getattr(self.app_instance, attr, None)
            if proc is not None and proc.poll() is None:
                running.append(label)
        return running

    def _refresh_status_chip(self) -> None:
        """Reflect real server state in the destination header chip."""
        running = self._running_servers()
        if running:
            status: WorkbenchStatus = "running"
            if len(running) == 1:
                label = f"{running[0]} running"
            else:
                label = f"{len(running)} servers running"
        else:
            status = "empty"
            label = "No server running"
        try:
            header = self.query_one("#llm-destination-header", DestinationHeader)
        except Exception:  # noqa: BLE001 - header not mounted yet
            return
        header.sync_state(
            WorkbenchHeaderState(
                title="Models",
                subtitle="Manage providers, models, and endpoints.",
                status=status,
                status_label=label,
            )
        )

    def save_state(self):
        """Save LLM window state."""
        state = super().save_state()
        # Add any LLM-specific state here
        return state

    def restore_state(self, state):
        """Restore LLM window state."""
        super().restore_state(state)
