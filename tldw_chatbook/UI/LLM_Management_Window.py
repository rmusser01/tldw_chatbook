# tldw_chatbook/UI/LLM_Management_Window.py
#
#
# Imports
from typing import TYPE_CHECKING
#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.widgets import Static, Button, Input, RichLog, Label, TextArea, Collapsible

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
            yield Button("Ollama", id="llm-nav-ollama", classes="llm-nav-button")
            yield Button("vLLM", id="llm-nav-vllm", classes="llm-nav-button")
            yield Button("ONNX", id="llm-nav-onnx", classes="llm-nav-button")
            yield Button("Transformers", id="llm-nav-transformers", classes="llm-nav-button")
            yield Button("MLX-LM", id="llm-nav-mlx-lm", classes="llm-nav-button")
            yield Button("Local Models", id="llm-nav-local-models", classes="llm-nav-button")
            yield Button("Download Models", id="llm-nav-download-models", classes="llm-nav-button")

        with Container(id="llm-content-pane", classes="llm-content-pane"):
            with Container(id="llm-view-llama-cpp", classes="llm-view-area"):
                with VerticalScroll():
                    yield Label("🦙 Llama.cpp Configuration", classes="section-title")
                    yield Label("Launch a llama.cpp server instance with a GGUF model", classes="description")
                    yield Label("Llama.cpp Server Executable Path:", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="llamacpp-exec-path", placeholder="/path/to/llama.cpp/build/bin/server")
                        yield Button("Browse", id="llamacpp-browse-exec-button", classes="browse_button")
                    yield Label("GGUF Model File Path:", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="llamacpp-model-path", placeholder="/path/to/model.gguf")
                        yield Button("Browse", id="llamacpp-browse-model-button", classes="browse_button")
                    yield Label("Host:", classes="label")
                    yield Input(id="llamacpp-host", value="127.0.0.1")
                    yield Label("Port (default 8001):", classes="label")
                    yield Input(id="llamacpp-port", value="8001")
                    yield Label("Additional Arguments (single line):", classes="label")
                    yield Input(id="llamacpp-additional-args", placeholder="e.g., --n-gpu-layers 1 --threads 4")
                    with Collapsible(title="Common Llama.cpp Server Arguments", collapsed=True,
                                     id="llamacpp-args-help-collapsible"):
                        # RichLog for scrollable, formatted help text
                        yield RichLog(
                            id="llamacpp-args-help-display",
                            markup=True,
                            highlight=False,  # No syntax highlighting needed for this help text
                            classes="help-text-display"  # Add a class for styling
                        )
                    with Container(classes="button_container"):
                        yield Button("Start Server", id="llamacpp-start-server-button", classes="action_button")
                        yield Button("Stop Server", id="llamacpp-stop-server-button", classes="action_button")
                    yield RichLog(id="llamacpp-log-output", classes="log_output", wrap=True, highlight=True)
            with Container(id="llm-view-llamafile", classes="llm-view-area"):
                with VerticalScroll():
                    yield Label("📁 Llamafile Configuration", classes="section-title")
                    yield Label("Run a self-contained llamafile executable (model included)", classes="description")
                    yield Label("Llamafile Executable (.llamafile):", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="llamafile-exec-path", placeholder="/path/to/model.llamafile")
                        yield Button("Browse", id="llamafile-browse-exec-button", classes="browse_button")
                    yield Label("Optional External Model (GGUF):", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="llamafile-model-path", placeholder="/path/to/external-model.gguf (optional)")
                        yield Button("Browse", id="llamafile-browse-model-button", classes="browse_button")
                    yield Label("Host:", classes="label")
                    yield Input(id="llamafile-host", value="127.0.0.1")
                    yield Label("Port (default 8000):", classes="label")
                    yield Input(id="llamafile-port", value="8000")
                    yield Label("Additional Arguments (multi-line):", classes="label")
                    yield TextArea(id="llamafile-additional-args", classes="additional_args_textarea", language="bash", theme="vscode_dark") # Ensure TextArea is imported
                    with Collapsible(title="Common Llamafile Arguments", collapsed=True,
                                     id="llamafile-args-help-collapsible"):
                        yield RichLog(
                            id="llamafile-args-help-display",
                            markup=True,
                            highlight=False,
                            classes="help-text-display"
                        )
                    with Container(classes="button_container"):
                        yield Button("Start Server", id="llamafile-start-server-button", classes="action_button")
                        yield Button("Stop Server", id="llamafile-stop-server-button", classes="action_button")
                    yield RichLog(id="llamafile-log-output", classes="log_output", wrap=True, highlight=True)
            with Container(id="llm-view-vllm", classes="llm-view-area"):
                yield Label("Python Interpreter Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="vllm-python-path", value="python", placeholder="e.g., /path/to/venv/bin/python")
                    yield Button("Browse", id="vllm-browse-python-button", classes="browse_button")
                yield Label("Model Path (or HuggingFace Repo ID):", classes="label")
                with Container(classes="input_container"):
                    yield Input(id="vllm-model-path", placeholder="e.g., /path/to/model or HuggingFaceName/ModelName")
                    yield Button("Browse", id="vllm-browse-model-button", classes="browse_button")
                yield Label("Host:", classes="label")
                yield Input(id="vllm-host", value="127.0.0.1")
                yield Label("Port:", classes="label")
                yield Input(id="vllm-port", value="8000")
                yield Label("Additional Arguments:", classes="label")
                yield TextArea(id="vllm-additional-args", classes="additional_args_textarea", language="bash", theme="vscode_dark") # Ensure TextArea is imported
                # Add a similar Collapsible RichLog for vLLM args here
                # with Collapsible(title="Common vLLM Arguments", collapsed=True):
                #     yield RichLog(id="vllm-args-help-display", markup=True, classes="help-text-display")
                with Container(classes="button_container"):
                    yield Button("Start Server", id="vllm-start-server-button", classes="action_button")
                    yield Button("Stop Server", id="vllm-stop-server-button", classes="action_button")
                yield RichLog(id="vllm-log-output", classes="log_output", wrap=True, highlight=True)
            with Container(id="llm-view-onnx", classes="llm-view-area"):
                with VerticalScroll():
                    yield Label("Python Interpreter Path:", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="onnx-python-path", value="python", placeholder="e.g., /path/to/venv/bin/python")
                        yield Button("Browse", id="onnx-browse-python-button", classes="browse_button")
                    yield Label("Path to your ONNX Server Script (.py):", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="onnx-script-path", placeholder="/path/to/your/onnx_server_script.py")
                        yield Button("Browse Script", id="onnx-browse-script-button", classes="browse_button")
                    yield Label("Model to Load (Path for script):", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="onnx-model-path", placeholder="Path to your .onnx model file or directory")
                        yield Button("Browse Model", id="onnx-browse-model-button", classes="browse_button")
                    yield Label("Host:", classes="label")
                    yield Input(id="onnx-host", value="127.0.0.1", classes="input_field")
                    yield Label("Port:", classes="label")
                    yield Input(id="onnx-port", value="8004", classes="input_field")
                    yield Label("Additional Script Arguments:", classes="label")
                    yield TextArea(id="onnx-additional-args", classes="additional_args_textarea", language="bash", theme="vscode_dark")
                    with Container(classes="button_container"):
                        yield Button("Start ONNX Server", id="onnx-start-server-button", classes="action_button")
                        yield Button("Stop ONNX Server", id="onnx-stop-server-button", classes="action_button")
                    yield RichLog(id="onnx-log-output", classes="log_output", wrap=True, highlight=True)
            # --- Transformers View ---
            with Container(id="llm-view-transformers", classes="llm-view-area"):
                with VerticalScroll():
                    yield Label("Hugging Face Transformers Model Management",
                                classes="section_label")  # Use a consistent class like .section_label or .pane-title

                    yield Label("Local Models Root Directory (for listing/browsing):", classes="label")
                    with Container(classes="input_container"):  # Re-use styling for input  button
                        yield Input(id="transformers-models-dir-path",
                                    placeholder="/path/to/your/hf_models_cache_or_local_dir")
                        yield Button("Browse Dir", id="transformers-browse-models-dir-button",
                                     classes="browse_button")

                    yield Button("List Local Models", id="transformers-list-local-models-button",
                                 classes="action_button")
                    yield RichLog(id="transformers-local-models-list", classes="log_output", markup=True,
                                  highlight=False)  # markup=True for Rich tags
                    yield Static("---", classes="separator")  # Visual separator

                    yield Label("Download New Model:", classes="label section_label")  # Use consistent class
                    yield Label("Model Repo ID (e.g., 'google-bert/bert-base-uncased'):", classes="label")
                    yield Input(id="transformers-download-repo-id", placeholder="username/model_name")
                    yield Label("Revision/Branch (optional):", classes="label")
                    yield Input(id="transformers-download-revision", placeholder="main")
                    yield Button("Download Model", id="transformers-download-model-button", classes="action_button")
                    yield Static("---", classes="separator")
                    yield Label("Run Custom Transformers Server Script:", classes="label section_label")
                    yield Label("Python Interpreter:", classes="label")
                    yield Input(id="transformers-python-path", value="python", placeholder="e.g., /path/to/venv/bin/python")
                    yield Label("Path to your Server Script (.py):", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="transformers-script-path", placeholder="/path/to/your_transformers_server_script.py")
                        yield Button("Browse Script", id="transformers-browse-script-button", classes="browse_button")
                    yield Label("Model to Load (ID or Path for script):", classes="label")
                    yield Input(id="transformers-server-model-arg", placeholder="Script-dependent model identifier")
                    yield Label("Host:", classes="label")
                    yield Input(id="transformers-server-host", value="127.0.0.1")
                    yield Label("Port:", classes="label")
                    yield Input(id="transformers-server-port", value="8003") # Example port
                    yield Label("Additional Script Arguments:", classes="label")
                    yield TextArea(id="transformers-server-additional-args", classes="additional_args_textarea", language="bash", theme="vscode_dark")
                    yield Button("Start Transformers Server", id="transformers-start-server-button", classes="action_button")
                    yield Button("Stop Transformers Server", id="transformers-stop-server-button", classes="action_button")

                    yield Label("Operations Log:", classes="label section_label")  # Use consistent class
                    yield RichLog(id="transformers-log-output", classes="log_output", wrap=True, highlight=True)
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
            with Container(id="llm-view-mlx-lm", classes="llm-view-area"):
                with VerticalScroll():
                    yield Label("MLX Model Path (HuggingFace ID or local path):", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="mlx-model-path", placeholder="e.g., mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX")
                        yield Button("Browse", id="mlx-browse-model-button", classes="browse_button")
                    yield Label("Host:", classes="label")
                    yield Input(id="mlx-host", value="127.0.0.1", classes="input_field")
                    yield Label("Port:", classes="label")
                    yield Input(id="mlx-port", value="8080", classes="input_field")
                    with Collapsible(title="Common MLX-LM Server Arguments", collapsed=True,
                                    id="mlx-args-help-collapsible"):
                        yield RichLog(
                            id="mlx-args-help-display",
                            markup=True,
                            highlight=False,
                            classes="help-text-display"
                        )
                    yield Label("Additional Server Arguments:", classes="label")
                    yield TextArea(id="mlx-additional-args", classes="additional_args_textarea", language="bash", theme="vscode_dark")
                    with Container(classes="button_container"):
                        yield Button("Start MLX Server", id="mlx-start-server-button", classes="action_button")
                        yield Button("Stop MLX Server", id="mlx-stop-server-button", classes="action_button")
                    yield RichLog(id="mlx-log-output", classes="log_output", wrap=True, highlight=True)
            with Container(id="llm-view-ollama", classes="llm-view-area"):
                with VerticalScroll():
                    # Server URL - stays at top and takes full width
                    yield Label("Ollama Service Management", classes="label section_label")
                    yield Label("Ollama Executable Path:", classes="label")
                    with Container(classes="input_container"):
                        yield Input(id="ollama-exec-path",
                                    placeholder="Path to ollama executable (e.g., /usr/local/bin/ollama)")
                        yield Button("Browse", id="ollama-browse-exec-button", classes="browse_button")
                    with Horizontal(classes="ollama-button-bar"):
                        yield Button("Start Ollama Service", id="ollama-start-service-button")
                        yield Button("Stop Ollama Service", id="ollama-stop-service-button")

                    # API Management Section
                    yield Label("Ollama API Management (requires running service)", classes="label section_label")
                    yield Label("Ollama Server URL:", classes="label")
                    yield Input(id="ollama-server-url", value="http://localhost:11434", classes="input_field_long")

                    # General Actions Bar
                    with Horizontal(classes="ollama-button-bar"):
                        yield Button("List Local Models", id="ollama-list-models-button")
                        yield Button("List Running Models", id="ollama-ps-button")

                    # Grid for more complex operations
                    with Horizontal(classes="ollama-actions-grid"):
                        # --- Left Column ---
                        with Vertical(classes="ollama-actions-column"):
                            yield Static("Model Management", classes="column-title")

                            yield Label("Show Info:", classes="label")
                            with Container(classes="input_action_container"):
                                yield Input(id="ollama-show-model-name", placeholder="Model name", classes="input_field_short")
                                yield Button("Show", id="ollama-show-model-button", classes="action_button_short")

                            yield Label("Delete:", classes="label")
                            with Container(classes="input_action_container"):
                                yield Input(id="ollama-delete-model-name", placeholder="Model to delete", classes="input_field_short")
                                yield Button("Delete", id="ollama-delete-model-button", classes="action_button_short delete_button")

                            yield Label("Copy Model:", classes="label")
                            with Horizontal(classes="input_action_container"):
                                yield Input(id="ollama-copy-source-model", placeholder="Source", classes="input_field_short")
                                yield Input(id="ollama-copy-destination-model", placeholder="Destination", classes="input_field_short")
                            yield Button("Copy Model", id="ollama-copy-model-button", classes="full_width_button")

                        # --- Right Column ---
                        with Vertical(classes="ollama-actions-column"):
                            yield Static("Registry & Custom Models", classes="column-title")

                            yield Label("Pull Model from Registry:", classes="label")
                            with Container(classes="input_action_container"):
                                yield Input(id="ollama-pull-model-name", placeholder="e.g. llama3", classes="input_field_short")
                                yield Button("Pull", id="ollama-pull-model-button", classes="action_button_short")

                            yield Label("Push Model to Registry:", classes="label")
                            with Container(classes="input_action_container"):
                                yield Input(id="ollama-push-model-name", placeholder="e.g. my-registry/my-model", classes="input_field_short")
                                yield Button("Push", id="ollama-push-model-button", classes="action_button_short")

                            yield Label("Create Model from Modelfile:", classes="label")
                            yield Input(id="ollama-create-model-name", placeholder="New model name", classes="input_field_long")
                            with Horizontal(classes="input_action_container"):
                                yield Input(id="ollama-create-modelfile-path", placeholder="Path to Modelfile...", disabled=True, classes="input_field_short")
                                yield Button("Browse", id="ollama-browse-modelfile-button", classes="browse_button_short")
                            yield Button("Create Model", id="ollama-create-model-button", classes="full_width_button")

                    # --- Output Panes ---
                    yield Label("Result / Status:", classes="label section_label")
                    yield RichLog(id="ollama-combined-output", wrap=True, highlight=False, classes="output_textarea_medium")

                    yield Label("Streaming Log:", classes="label section_label")
                    yield RichLog(id="ollama-log-output", wrap=True, highlight=True, classes="log_output_large")

#
# End of LLM_Management_Window.py
#######################################################################################################################
