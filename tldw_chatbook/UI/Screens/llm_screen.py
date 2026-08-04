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


def _probe_local_server(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """Cheap TCP probe for an externally-started Ollama server."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


class LLMScreen(BaseAppScreen):
    """
    LLM Management screen wrapper.
    """

    # Screen-level mirrors of LLMManagementWindow.BINDINGS so the advertised
    # keys work from the landed state (nav bar has initial focus; widget
    # bindings only fire with in-window focus).
    BINDINGS = LLMManagementWindow.BINDINGS

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "llm", **kwargs)
        self.llm_window = None

    def action_prev_llm_view(self) -> None:
        self.llm_window.action_prev_llm_view()

    def action_next_llm_view(self) -> None:
        self.llm_window.action_next_llm_view()

    def action_jump_view(self, index: int) -> None:
        self.llm_window.action_jump_view(index)

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
        self.register_footer_shortcuts(
            source="llm",
            shortcuts=(("1-9", "jump to view"), ("[ / ]", "cycle views")),
        )
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
            # No app-launched process: probe the default Ollama port so an
            # externally-started server doesn't read as "nothing running".
            status = "empty"
            label = "No server running"
            if _probe_local_server():
                status = "running"
                label = "Ollama detected on :11434 (external)"
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
