# tldw_chatbook/UI/LLM_Management_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.css.query import QueryError
from textual.widgets import Static, Button, Input, RichLog, Label, TextArea
# Local Imports
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:

class LLMManagementWindow(Container):
    """
    Container for the LLM Management Tab's UI.
    """
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        # Use a unique ID for this window if it's also set in app.py, e.g., "llm_management-window"
        # The id passed from app.py during instantiation will take precedence if set there.
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def on_mount(self) -> None:
        self.app_instance.loguru_logger.debug("LLMManagementWindow.on_mount called")
        # try:
        #     content_pane = self.query_one("#llm-content-pane", Container)
        #     view_areas = content_pane.query(".llm-view-area")
        #     if not view_areas:
        #         self.app_instance.loguru_logger.warning("LLMManagementWindow.on_mount: No .llm-view-area found in #llm-content-pane.")
        #         return
        #
        #     for view in view_areas:
        #         if view.id: # Only hide if it has an ID (sanity check)
        #             self.app_instance.loguru_logger.debug(f"LLMManagementWindow.on_mount: Hiding view #{view.id}")
        #             view.styles.display = "none"
        #         else:
        #             self.app_instance.loguru_logger.warning("LLMManagementWindow.on_mount: Found a .llm-view-area without an ID, not hiding it.")
        # except QueryError as e:
        #     self.app_instance.loguru_logger.error(f"LLMManagementWindow.on_mount: QueryError: {e}", exc_info=True)
        # except Exception as e:
        #     self.app_instance.loguru_logger.error(f"LLMManagementWindow.on_mount: Unexpected error: {e}", exc_info=True)
        pass

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="llm-nav-pane", classes="llm-nav-pane"):
            yield Static("LLM Options", classes="sidebar-title")
            yield Button("Llama.cpp", id="llm-nav-llama-cpp", classes="llm-nav-button")
            yield Button("Llamafile", id="llm-nav-llamafile", classes="llm-nav-button")
            yield Button("vLLM", id="llm-nav-vllm", classes="llm-nav-button")
            yield Button("Transformers", id="llm-nav-transformers", classes="llm-nav-button")
            yield Button("Local Models", id="llm-nav-local-models", classes="llm-nav-button")
            yield Button("Download Models", id="llm-nav-download-models", classes="llm-nav-button")

        with Container(id="llm-content-pane", classes="llm-content-pane"):
            with Container(id="llm-view-llama-cpp", classes="llm-view-area"):
                yield Label("Executable Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="llamacpp-exec-path", placeholder="/path/to/llama.cpp/server")
                    yield Button("Browse", id="llamacpp-browse-exec-button", classes="browse_button")
                yield Label("Model Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="llamacpp-model-path", placeholder="/path/to/model.gguf")
                    yield Button("Browse", id="llamacpp-browse-model-button", classes="browse_button")
                yield Label("Host:", classes="label")
                yield Input(id="llamacpp-host", value="127.0.0.1")
                yield Label("Port:", classes="label")
                yield Input(id="llamacpp-port", value="8001")
                yield Label("Additional Arguments:", classes="label")
                yield Input(id="llamacpp-additional-args", placeholder="e.g., --n-gpu-layers 1")
                with Container(classes="button_container"):
                    yield Button("Start Server", id="llamacpp-start-server-button", classes="action_button")
                    yield Button("Stop Server", id="llamacpp-stop-server-button", classes="action_button")
                yield RichLog(id="llamacpp-log-output", classes="log_output", wrap=True, highlight=True)
            yield Container(
                Static("Llamafile Management Area - Content Coming Soon!"),
                id="llm-view-llamafile",
                classes="llm-view-area",
            )
            with Container(id="llm-view-vllm", classes="llm-view-area"):
                yield Label("Python Interpreter Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="vllm-python-path", placeholder="Default: python")
                    yield Button("Browse", id="vllm-browse-python-button", classes="browse_button")
                yield Label("Model Path or Hugging Face ID:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="vllm-model-path", placeholder="e.g., /path/to/model or HuggingFaceName/ModelName")
                    yield Button("Browse", id="vllm-browse-model-button", classes="browse_button")
                yield Label("Host:", classes="label")
                yield Input(id="vllm-host", value="127.0.0.1")
                yield Label("Port:", classes="label")
                yield Input(id="vllm-port", value="8000")
                yield Label("Additional Arguments:", classes="label")
                yield TextArea(id="vllm-additional-args")
                with Container(classes="button_container"):
                    yield Button("Start Server", id="vllm-start-server-button", classes="action_button")
                    yield Button("Stop Server", id="vllm-stop-server-button", classes="action_button")
                yield RichLog(id="vllm-log-output", classes="log_output", wrap=True, highlight=True)
            yield Container(
                Static("Transformers Library Management Area - Content Coming Soon!"),
                id="llm-view-transformers",
                classes="llm-view-area",
            )
            yield Container(
                Static("Local Model Management Area - Content Coming Soon!"),
                id="llm-view-local-models",
                classes="llm-view-area",
            )
            yield Container(
                Static("Model Download Area - Content Coming Soon!"),
                id="llm-view-download-models",
                classes="llm-view-area",
            )

#
# End of LLM_Management_Window.py
#######################################################################################################################
