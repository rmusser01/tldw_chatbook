# tldw_chatbook/UI/LLM_Management_Window.py
#
#
# Imports
import inspect
from typing import TYPE_CHECKING, Callable

#
# 3rd-Party Imports
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.reactive import reactive
from textual.widgets import Static, Button, Input, RichLog, Label, TextArea, Collapsible
from loguru import logger

# Local Imports
from ..Event_Handlers.LLM_Management_Events.llm_management_events import (
    LLM_MANAGEMENT_BUTTON_HANDLERS,
)
from ..Event_Handlers.LLM_Management_Events.llm_management_events_mlx_lm import (
    MLX_LM_BUTTON_HANDLERS,
)
from ..Event_Handlers.LLM_Management_Events.llm_management_events_ollama import (
    OLLAMA_BUTTON_HANDLERS,
)
from ..Event_Handlers.LLM_Management_Events.llm_management_events_onnx import (
    ONNX_BUTTON_HANDLERS,
)
from ..Event_Handlers.LLM_Management_Events.llm_management_events_transformers import (
    TRANSFORMERS_BUTTON_HANDLERS,
)
from ..Event_Handlers.LLM_Management_Events.llm_management_events_vllm import (
    VLLM_BUTTON_HANDLERS,
)
from ..Event_Handlers.LLM_Management_Events.server_lifecycle import (
    current_llm_destination,
    server_is_active,
)
from ..Utils.log_widget_manager import LogWidgetManager

if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:


class LLMManagementWindow(Container):
    """
    Container for the LLM Management Tab's UI.
    Follows Textual best practices with proper navigation and view management.
    """

    DEFAULT_CSS = """
    LLMManagementWindow {
        layout: horizontal;
        height: 100%;
        width: 100%;
    }

    #llm-main-content {
        width: 1fr;
        height: 100%;
        background: $background;
        padding: 1 2;
    }

    .llm-view {
        display: none;
        height: 100%;
        width: 100%;
    }
    
    .llm-view.-active {
        display: block;
    }
    
    .section-title {
        text-style: bold;
        margin: 1 0;
        color: $primary;
    }
    
    .section_label {
        text-style: bold;
        margin: 1 0;
        color: $secondary;
    }
    
    .description {
        margin: 0 0 1 0;
        color: $text-muted;
    }
    
    .label {
        margin: 1 0 0 0;
    }
    
    .input_container {
        layout: horizontal;
        height: 3;
        margin: 0 0 1 0;
    }
    
    .input_container Input {
        width: 1fr;
    }
    
    .input_container Button {
        width: auto;
        margin: 0 0 0 1;
    }
    
    .button_container {
        layout: horizontal;
        margin: 1 0;
        height: 3;
    }
    
    .button_container Button {
        margin: 0 1 0 0;
    }
    
    .log_output {
        height: 15;
        border: solid $primary;
        margin: 1 0;
    }
    
    .help-text-display {
        height: 10;
        border: solid $secondary;
        padding: 1;
    }
    
    .additional_args_textarea {
        height: 5;
        margin: 0 0 1 0;
    }
    
    .separator {
        height: 1;
        margin: 1 0;
        color: $primary;
    }
    
    .ollama-button-bar {
        layout: horizontal;
        height: 3;
        margin: 1 0;
    }
    
    .ollama-button-bar Button {
        margin: 0 1 0 0;
    }
    
    .ollama-actions-grid {
        layout: horizontal;
        margin: 1 0;
    }
    
    .ollama-actions-column {
        width: 50%;
        padding: 0 1;
    }
    
    .column-title {
        text-style: bold;
        margin: 0 0 1 0;
        color: $secondary;
    }
    
    .input_field_short {
        width: 40%;
    }
    
    .input_field_long {
        width: 100%;
    }
    
    .action_button_short {
        width: auto;
    }
    
    .full_width_button {
        width: 100%;
        margin: 1 0;
    }
    
    .delete_button {
        background: $error;
    }
    
    .embeddings_container {
        layout: horizontal;
        margin: 1 0;
    }
    
    .embeddings_inputs {
        width: 70%;
    }
    
    .action_button_tall {
        width: 30%;
        margin: 0 0 0 1;
    }
    
    .output_textarea_medium {
        height: 10;
        margin: 1 0;
    }
    
    .log_output_large {
        height: 20;
        margin: 1 0;
    }
    """

    # Reactive property to track active view. Starts at "" with init=False
    # rather than "llama-cpp": Textual's reactive default-value watcher
    # otherwise fires once at mount, before the Lab frame's deferred body
    # mount means this window's own child views exist -- ten QueryErrors,
    # every arrival. Starting at "" (never a real view key, and init=False
    # skips even that empty-value fire) means the FIRST real assignment,
    # in _initialize_view, is the one and only trigger, made after the
    # children exist.
    active_view = reactive("", recompose=False, init=False)

    ACTION_HANDLERS: dict[str, Callable] = {
        **LLM_MANAGEMENT_BUTTON_HANDLERS,
        **MLX_LM_BUTTON_HANDLERS,
        **OLLAMA_BUTTON_HANDLERS,
        **ONNX_BUTTON_HANDLERS,
        **TRANSFORMERS_BUTTON_HANDLERS,
        **VLLM_BUTTON_HANDLERS,
    }
    SERVER_CONTROLS = {
        "llamacpp": ("llamacpp-start-server-button", "llamacpp-stop-server-button"),
        "llamafile": ("llamafile-start-server-button", "llamafile-stop-server-button"),
        "vllm": ("vllm-start-server-button", "vllm-stop-server-button"),
        "onnx": ("onnx-start-server-button", "onnx-stop-server-button"),
        "mlx": ("mlx-start-server-button", "mlx-stop-server-button"),
        "ollama": ("ollama-start-service-button", "ollama-stop-service-button"),
    }

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._async_presentation_generations: dict[str, int] = {}

        # Map navigation button IDs to view IDs
        self.view_mapping = {
            "llama-cpp": "llm-view-llama-cpp",
            "llamafile": "llm-view-llamafile",
            "ollama": "llm-view-ollama",
            "vllm": "llm-view-vllm",
            "onnx": "llm-view-onnx",
            "transformers": "llm-view-transformers",
            "mlx-lm": "llm-view-mlx-lm",
            "local-models": "llm-view-local-models",
            "download-models": "llm-view-download-models",
        }

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        logger.debug("LLMManagementWindow.on_mount called")
        # The child views don't exist until after this refresh (the window
        # itself may be mounted lazily by its parent screen), so defer the
        # initial activation until they do.
        self.call_after_refresh(self._initialize_view)

    def _initialize_view(self) -> None:
        """Activate the initial view now that the child views exist.

        Assigns ``"llama-cpp"`` rather than hand-invoking ``watch_active_view``:
        with the reactive's default now ``""`` (see ``active_view`` above),
        this is a genuine value change, so it fires the normal reactive
        path -- ``watch_active_view`` plus any external watchers registered
        via ``self.watch(...)`` (e.g. the Lab rail highlighter) -- with the
        child views already mounted.
        """
        self.active_view = "llama-cpp"
        self._sync_all_process_controls()

    def compose(self) -> ComposeResult:
        """Compose the LLM Management UI with sidebar navigation and content area."""
        # Main content area
        with Container(id="llm-main-content"):
            # Llama.cpp View
            with VerticalScroll(id="llm-view-llama-cpp", classes="llm-view"):
                yield Label("🦙 Llama.cpp Configuration", classes="section-title")
                yield Label(
                    "Launch a llama.cpp server instance with a GGUF model",
                    classes="description",
                )

                yield Label("Llama.cpp Server Executable Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="llamacpp-exec-path",
                        placeholder="/path/to/llama.cpp/build/bin/server",
                    )
                    yield Button(
                        "Browse",
                        id="llamacpp-browse-exec-button",
                        classes="browse_button",
                        tooltip="Choose the llama.cpp server executable.",
                    )

                yield Label("GGUF Model File Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="llamacpp-model-path", placeholder="/path/to/model.gguf"
                    )
                    yield Button(
                        "Browse",
                        id="llamacpp-browse-model-button",
                        classes="browse_button",
                        tooltip="Choose a GGUF model file for llama.cpp.",
                    )

                yield Label("Host:", classes="label")
                yield Input(id="llamacpp-host", value="127.0.0.1")

                yield Label("Port (default 8001):", classes="label")
                yield Input(id="llamacpp-port", value="8001")

                yield Label("Additional Arguments (single line):", classes="label")
                yield Input(
                    id="llamacpp-additional-args",
                    placeholder="e.g., --n-gpu-layers 1 --threads 4",
                )

                with Collapsible(
                    title="Common Llama.cpp Server Arguments",
                    collapsed=True,
                    id="llamacpp-args-help-collapsible",
                ):
                    yield RichLog(
                        id="llamacpp-args-help-display",
                        markup=True,
                        highlight=False,
                        classes="help-text-display",
                    )

                with Container(classes="button_container"):
                    yield Button(
                        "Start Server",
                        id="llamacpp-start-server-button",
                        classes="action_button",
                    )
                    yield Button(
                        "Stop Server",
                        id="llamacpp-stop-server-button",
                        classes="action_button",
                        disabled=True,
                    )

                yield RichLog(
                    id="llamacpp-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # Llamafile View
            with VerticalScroll(id="llm-view-llamafile", classes="llm-view"):
                yield Label("📁 Llamafile Configuration", classes="section-title")
                yield Label(
                    "Run a self-contained llamafile executable (model included)",
                    classes="description",
                )

                yield Label("Llamafile Executable (.llamafile):", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="llamafile-exec-path", placeholder="/path/to/model.llamafile"
                    )
                    yield Button(
                        "Browse",
                        id="llamafile-browse-exec-button",
                        classes="browse_button",
                        tooltip="Choose the llamafile executable.",
                    )

                yield Label("Optional External Model (GGUF):", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="llamafile-model-path",
                        placeholder="/path/to/external-model.gguf (optional)",
                    )
                    yield Button(
                        "Browse",
                        id="llamafile-browse-model-button",
                        classes="browse_button",
                        tooltip="Choose an optional external GGUF model for llamafile.",
                    )

                yield Label("Host:", classes="label")
                yield Input(id="llamafile-host", value="127.0.0.1")

                yield Label("Port (default 8000):", classes="label")
                yield Input(id="llamafile-port", value="8000")

                yield Label("Additional Arguments (multi-line):", classes="label")
                yield TextArea(
                    id="llamafile-additional-args",
                    classes="additional_args_textarea",
                    theme="vscode_dark",
                )

                with Collapsible(
                    title="Common Llamafile Arguments",
                    collapsed=True,
                    id="llamafile-args-help-collapsible",
                ):
                    yield RichLog(
                        id="llamafile-args-help-display",
                        markup=True,
                        highlight=False,
                        classes="help-text-display",
                    )

                with Container(classes="button_container"):
                    yield Button(
                        "Start Server",
                        id="llamafile-start-server-button",
                        classes="action_button",
                    )
                    yield Button(
                        "Stop Server",
                        id="llamafile-stop-server-button",
                        classes="action_button",
                        disabled=True,
                    )

                yield RichLog(
                    id="llamafile-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # vLLM View
            with VerticalScroll(id="llm-view-vllm", classes="llm-view"):
                yield Label("⚡ vLLM Configuration", classes="section-title")
                yield Label(
                    "High-performance LLM serving with vLLM", classes="description"
                )

                yield Label("Python Interpreter Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="vllm-python-path",
                        value="python",
                        placeholder="e.g., /path/to/venv/bin/python",
                    )
                    yield Button(
                        "Browse",
                        id="vllm-browse-python-button",
                        classes="browse_button",
                        tooltip="Choose the Python interpreter used to launch vLLM.",
                    )

                yield Label("Model Path (or HuggingFace Repo ID):", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="vllm-model-path",
                        placeholder="e.g., /path/to/model or HuggingFaceName/ModelName",
                    )
                    yield Button(
                        "Browse",
                        id="vllm-browse-model-button",
                        classes="browse_button",
                        tooltip="Choose a local model directory for vLLM, or type a Hugging Face repo ID.",
                    )

                yield Label("Host:", classes="label")
                yield Input(id="vllm-host", value="127.0.0.1")

                yield Label("Port:", classes="label")
                yield Input(id="vllm-port", value="8000")

                yield Label("Additional Arguments:", classes="label")
                yield TextArea(
                    id="vllm-additional-args",
                    classes="additional_args_textarea",
                    theme="vscode_dark",
                )

                with Container(classes="button_container"):
                    yield Button(
                        "Start Server",
                        id="vllm-start-server-button",
                        classes="action_button",
                    )
                    yield Button(
                        "Stop Server",
                        id="vllm-stop-server-button",
                        classes="action_button",
                        disabled=True,
                    )

                yield RichLog(
                    id="vllm-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # ONNX View
            with VerticalScroll(id="llm-view-onnx", classes="llm-view"):
                yield Label("🔧 ONNX Runtime Configuration", classes="section-title")
                yield Label(
                    "Run ONNX models with optimized inference", classes="description"
                )

                yield Label("Python Interpreter Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="onnx-python-path",
                        value="python",
                        placeholder="e.g., /path/to/venv/bin/python",
                    )
                    yield Button(
                        "Browse",
                        id="onnx-browse-python-button",
                        classes="browse_button",
                        tooltip="Choose the Python interpreter used to launch the ONNX server.",
                    )

                yield Label("Path to your ONNX Server Script (.py):", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="onnx-script-path",
                        placeholder="/path/to/your/onnx_server_script.py",
                    )
                    yield Button(
                        "Browse Script",
                        id="onnx-browse-script-button",
                        classes="browse_button",
                        tooltip="Choose the ONNX server script to run.",
                    )

                yield Label("Model to Load (Path for script):", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="onnx-model-path",
                        placeholder="Path to your .onnx model file or directory",
                    )
                    yield Button(
                        "Browse Model",
                        id="onnx-browse-model-button",
                        classes="browse_button",
                        tooltip="Choose the ONNX model file or directory to load.",
                    )

                yield Label("Host:", classes="label")
                yield Input(id="onnx-host", value="127.0.0.1", classes="input_field")

                yield Label("Port:", classes="label")
                yield Input(id="onnx-port", value="8004", classes="input_field")

                yield Label("Additional Script Arguments:", classes="label")
                yield TextArea(
                    id="onnx-additional-args",
                    classes="additional_args_textarea",
                    theme="vscode_dark",
                )

                with Container(classes="button_container"):
                    yield Button(
                        "Start ONNX Server",
                        id="onnx-start-server-button",
                        classes="action_button",
                    )
                    yield Button(
                        "Stop ONNX Server",
                        id="onnx-stop-server-button",
                        classes="action_button",
                        disabled=True,
                    )

                yield RichLog(
                    id="onnx-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # Transformers View
            with VerticalScroll(id="llm-view-transformers", classes="llm-view"):
                yield Label(
                    "🤗 Hugging Face Transformers Model Management",
                    classes="section-title",
                )

                yield Label(
                    "Local Models Root Directory (for listing/browsing):",
                    classes="label",
                )
                with Container(classes="input_container"):
                    yield Input(
                        id="transformers-models-dir-path",
                        placeholder="/path/to/your/hf_models_cache_or_local_dir",
                    )
                    yield Button(
                        "Browse Dir",
                        id="transformers-browse-models-dir-button",
                        classes="browse_button",
                        tooltip="Choose the local Transformers models root directory.",
                    )

                yield Button(
                    "List Local Models",
                    id="transformers-list-local-models-button",
                    classes="action_button",
                )
                yield RichLog(
                    id="transformers-local-models-list",
                    classes="log_output",
                    markup=True,
                    highlight=False,
                )

                yield Static("---", classes="separator")

                yield Label("Download New Model:", classes="section_label")
                yield Label(
                    "Model Repo ID (e.g., 'google-bert/bert-base-uncased'):",
                    classes="label",
                )
                yield Input(
                    id="transformers-download-repo-id",
                    placeholder="username/model_name",
                )
                yield Label("Revision/Branch (optional):", classes="label")
                yield Input(id="transformers-download-revision", placeholder="main")
                yield Button(
                    "Download Model",
                    id="transformers-download-model-button",
                    classes="action_button",
                )

                yield Label("Model Operations Output", classes="section_label")
                yield RichLog(
                    id="transformers-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # MLX-LM View
            with VerticalScroll(id="llm-view-mlx-lm", classes="llm-view"):
                yield Label("🍎 MLX-LM Configuration", classes="section-title")
                yield Label(
                    "Apple Silicon optimized LLM inference", classes="description"
                )

                yield Label(
                    "MLX Model Path (HuggingFace ID or local path):", classes="label"
                )
                with Container(classes="input_container"):
                    yield Input(
                        id="mlx-model-path",
                        placeholder="e.g., mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
                    )
                    yield Button(
                        "Browse",
                        id="mlx-browse-model-button",
                        classes="browse_button",
                        tooltip="Choose a local MLX model path, or type a Hugging Face repo ID.",
                    )

                yield Label("Host:", classes="label")
                yield Input(id="mlx-host", value="127.0.0.1", classes="input_field")

                yield Label("Port:", classes="label")
                yield Input(id="mlx-port", value="8080", classes="input_field")

                with Collapsible(
                    title="Common MLX-LM Server Arguments",
                    collapsed=True,
                    id="mlx-args-help-collapsible",
                ):
                    yield RichLog(
                        id="mlx-args-help-display",
                        markup=True,
                        highlight=False,
                        classes="help-text-display",
                    )

                yield Label("Additional Server Arguments:", classes="label")
                yield TextArea(
                    id="mlx-additional-args",
                    classes="additional_args_textarea",
                    theme="vscode_dark",
                )

                with Container(classes="button_container"):
                    yield Button(
                        "Start MLX Server",
                        id="mlx-start-server-button",
                        classes="action_button",
                    )
                    yield Button(
                        "Stop MLX Server",
                        id="mlx-stop-server-button",
                        classes="action_button",
                        disabled=True,
                    )

                yield RichLog(
                    id="mlx-log-output", classes="log_output", wrap=True, highlight=True
                )

            # Ollama View
            with VerticalScroll(id="llm-view-ollama", classes="llm-view"):
                yield Label("🦙 Ollama Service Management", classes="section-title")

                yield Label("Ollama Executable Path:", classes="label")
                with Container(classes="input_container"):
                    yield Input(
                        id="ollama-exec-path",
                        placeholder="Path to ollama executable (e.g., /usr/local/bin/ollama)",
                    )
                    yield Button(
                        "Browse",
                        id="ollama-browse-exec-button",
                        classes="browse_button",
                        tooltip="Choose the Ollama executable.",
                    )

                with Horizontal(classes="ollama-button-bar"):
                    yield Button(
                        "Start Ollama Service", id="ollama-start-service-button"
                    )
                    yield Button(
                        "Stop Ollama Service",
                        id="ollama-stop-service-button",
                        disabled=True,
                    )

                yield Label(
                    "Ollama API Management (requires running service)",
                    classes="section_label",
                )
                yield Label("Ollama Server URL:", classes="label")
                yield Input(
                    id="ollama-server-url",
                    value="http://localhost:11434",
                    classes="input_field_long",
                )

                with Horizontal(classes="ollama-button-bar"):
                    yield Button("List Local Models", id="ollama-list-models-button")
                    yield Button("List Running Models", id="ollama-ps-button")

                with Horizontal(classes="ollama-actions-grid"):
                    # Left Column
                    with Vertical(classes="ollama-actions-column"):
                        yield Static("Model Management", classes="column-title")

                        yield Label("Show Info:", classes="label")
                        with Container(classes="input_action_container"):
                            yield Input(
                                id="ollama-show-model-name",
                                placeholder="Model name",
                                classes="input_field_short",
                            )
                            yield Button(
                                "Show",
                                id="ollama-show-model-button",
                                classes="action_button_short",
                            )

                        yield Label("Delete:", classes="label")
                        with Container(classes="input_action_container"):
                            yield Input(
                                id="ollama-delete-model-name",
                                placeholder="Model to delete",
                                classes="input_field_short",
                            )
                            yield Button(
                                "Delete",
                                id="ollama-delete-model-button",
                                classes="action_button_short delete_button",
                            )

                        yield Label("Copy Model:", classes="label")
                        with Horizontal(classes="input_action_container"):
                            yield Input(
                                id="ollama-copy-source-model",
                                placeholder="Source",
                                classes="input_field_short",
                            )
                            yield Input(
                                id="ollama-copy-destination-model",
                                placeholder="Destination",
                                classes="input_field_short",
                            )
                        yield Button(
                            "Copy Model",
                            id="ollama-copy-model-button",
                            classes="full_width_button",
                        )

                    # Right Column
                    with Vertical(classes="ollama-actions-column"):
                        yield Static("Registry & Custom Models", classes="column-title")

                        yield Label("Pull Model from Registry:", classes="label")
                        with Container(classes="input_action_container"):
                            yield Input(
                                id="ollama-pull-model-name",
                                placeholder="e.g. llama3",
                                classes="input_field_short",
                            )
                            yield Button(
                                "Pull",
                                id="ollama-pull-model-button",
                                classes="action_button_short",
                            )

                        yield Label("Push Model to Registry:", classes="label")
                        with Container(classes="input_action_container"):
                            yield Input(
                                id="ollama-push-model-name",
                                placeholder="e.g. my-registry/my-model",
                                classes="input_field_short",
                            )
                            yield Button(
                                "Push",
                                id="ollama-push-model-button",
                                classes="action_button_short",
                            )

                        yield Label("Create Model from Modelfile:", classes="label")
                        yield Input(
                            id="ollama-create-model-name",
                            placeholder="New model name",
                            classes="input_field_long",
                        )
                        with Horizontal(classes="input_action_container"):
                            yield Input(
                                id="ollama-create-modelfile-path",
                                placeholder="Path to Modelfile...",
                                disabled=True,
                                classes="input_field_short",
                            )
                            yield Button(
                                "Browse",
                                id="ollama-browse-modelfile-button",
                                classes="browse_button_short",
                                tooltip="Choose the Modelfile used to create an Ollama model.",
                            )
                        yield Button(
                            "Create Model",
                            id="ollama-create-model-button",
                            classes="full_width_button",
                        )

                yield Label("Generate Embeddings:", classes="section_label")
                with Horizontal(classes="embeddings_container"):
                    with Vertical(classes="embeddings_inputs"):
                        yield Input(
                            id="ollama-embeddings-model-name",
                            placeholder="Model name for embeddings",
                            classes="input_field_long",
                        )
                        yield Input(
                            id="ollama-embeddings-prompt",
                            placeholder="Text to generate embeddings for",
                            classes="input_field_long",
                        )
                    yield Button(
                        "Generate Embeddings",
                        id="ollama-embeddings-button",
                        classes="action_button_tall",
                    )

                yield Label("Result / Status:", classes="section_label")
                yield RichLog(
                    id="ollama-combined-output",
                    wrap=True,
                    highlight=False,
                    classes="output_textarea_medium",
                )

                yield Label("Streaming Log:", classes="section_label")
                yield RichLog(
                    id="ollama-log-output",
                    wrap=True,
                    highlight=True,
                    classes="log_output_large",
                )

            # Local Models View (preserved unchanged)
            with Container(id="llm-view-local-models", classes="llm-view"):
                from ..Widgets.HuggingFace import LocalModelsWidget

                yield LocalModelsWidget(self.app_instance, id="local-models-widget")

            # Download Models View (preserved unchanged)
            with Container(id="llm-view-download-models", classes="llm-view"):
                from ..Widgets.HuggingFace import HuggingFaceModelBrowser

                yield HuggingFaceModelBrowser(
                    self.app_instance, id="huggingface-model-browser"
                )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route allowlisted actions inside this destination."""

        event.stop()
        button = event.button
        if not button.id:
            return

        callback = self.ACTION_HANDLERS.get(button.id)
        if callback is None:
            return

        try:
            result = callback(self, self.app_instance, event)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self._recover_failed_action(button.id, exc)

    def _recover_failed_action(self, action_id: str, exc: Exception) -> None:
        """Restore truthful controls and surface bounded, non-sensitive recovery."""

        exception_category = type(exc).__name__
        logger.error(
            "LLM destination action failed (action_id={}, exception_category={})",
            action_id,
            exception_category,
        )
        self._restore_process_controls(action_id)
        message = (
            f"Action {action_id} failed ({exception_category}). "
            "Try again or check application logs."
        )[:200]
        log_type = self._log_type_for_action(action_id)
        if log_type is not None:
            LogWidgetManager.update_log(self, log_type, message)
        self.app_instance.notify(message, severity="error")

    def _restore_process_controls(self, action_id: str) -> None:
        """Set start/stop controls from the actual process handle after a failure."""

        provider = next(
            (name for name in self.SERVER_CONTROLS if action_id.startswith(f"{name}-")),
            None,
        )
        if provider is None:
            return
        self._sync_process_controls(provider)

    def _sync_process_controls(self, provider: str) -> None:
        """Render one provider's controls from app-owned lifecycle state."""

        start_id, stop_id = self.SERVER_CONTROLS[provider]
        active = server_is_active(self.app_instance, provider)
        try:
            self.query_one(f"#{start_id}", Button).disabled = active
            self.query_one(f"#{stop_id}", Button).disabled = not active
        except QueryError:
            logger.warning(
                "Could not restore LLM process controls (provider={})",
                provider,
            )

    def _handle_server_process_state_change(
        self,
        provider: str,
        status: str | None = None,
    ) -> None:
        """Refresh one destination and surface only bounded worker status."""

        self._sync_process_controls(provider)
        if status is not None:
            self.app_instance.notify(status[:200], severity="error")

    def _sync_all_process_controls(self) -> None:
        """Render every process control pair from app-owned lifecycle state."""

        for provider in self.SERVER_CONTROLS:
            self._sync_process_controls(provider)

    def _begin_async_presentation(self, channel: str) -> int:
        """Reserve the next local completion generation for one output channel."""

        generation = self._async_presentation_generations.get(channel, 0) + 1
        self._async_presentation_generations[channel] = generation
        return generation

    def _owns_async_presentation(self, channel: str, generation: int) -> bool:
        """Return whether this mounted destination still owns one completion."""

        return (
            self._async_presentation_generations.get(channel) == generation
            and current_llm_destination(self.app_instance) is self
        )

    @staticmethod
    def _log_type_for_action(action_id: str) -> str | None:
        """Return the destination log category associated with an action."""

        if action_id.startswith("llamacpp-"):
            return "llamacpp"
        if action_id.startswith("llamafile-"):
            return "llamafile"
        if action_id.startswith("vllm-"):
            return "vllm"
        if action_id.startswith("mlx-"):
            return "mlx"
        if action_id.startswith("transformers-"):
            return "transformers"
        return None

    def watch_active_view(self, old_view: str, new_view: str) -> None:
        """React to active view changes."""
        logger.debug(f"LLM view changing from '{old_view}' to '{new_view}'")

        # Update view visibility
        for view_id in self.view_mapping.values():
            try:
                view = self.query_one(f"#{view_id}")
                view.remove_class("-active")
            except QueryError:
                logger.warning(f"View #{view_id} not found")

        # Show the new view
        if new_view in self.view_mapping:
            target_view_id = self.view_mapping[new_view]
            try:
                target_view = self.query_one(f"#{target_view_id}")
                target_view.add_class("-active")
                logger.info(f"Activated LLM view: {target_view_id}")

                # Populate help text for specific views
                self._populate_help_text(new_view, target_view)
            except QueryError:
                logger.error(f"Target view #{target_view_id} not found")

    def _populate_help_text(self, view_name: str, view_widget) -> None:
        """Populate help text for views that have it."""
        if view_name == "llama-cpp":
            try:
                help_widget = view_widget.query_one(
                    "#llamacpp-args-help-display", RichLog
                )
                if not help_widget.lines:
                    help_widget.clear()
                    # Import help text from Constants
                    from ..Constants import LLAMA_CPP_SERVER_ARGS_HELP_TEXT

                    help_widget.write(LLAMA_CPP_SERVER_ARGS_HELP_TEXT)
            except (QueryError, ImportError) as e:
                logger.debug(f"Could not populate Llama.cpp help text: {e}")

        elif view_name == "llamafile":
            try:
                help_widget = view_widget.query_one(
                    "#llamafile-args-help-display", RichLog
                )
                if not help_widget.lines:
                    help_widget.clear()
                    # Placeholder help text for Llamafile
                    help_text = """[bold cyan]Common Llamafile Arguments[/]

[bold]--port PORT[/] - Server port (default: 8080)
[bold]--host HOST[/] - Server host (default: 127.0.0.1)
[bold]--threads N[/] - Number of threads
[bold]--ctx-size N[/] - Context size
[bold]--batch-size N[/] - Batch size
[bold]--no-mmap[/] - Disable memory mapping
"""
                    help_widget.write(help_text)
            except QueryError as e:
                logger.debug(f"Could not populate Llamafile help text: {e}")

        elif view_name == "mlx-lm":
            try:
                help_widget = view_widget.query_one("#mlx-args-help-display", RichLog)
                if not help_widget.lines:
                    help_widget.clear()
                    # Placeholder help text for MLX-LM
                    help_text = """[bold cyan]Common MLX-LM Server Arguments[/]

[bold]--port PORT[/] - Server port (default: 8080)
[bold]--host HOST[/] - Server host (default: 0.0.0.0)
[bold]--model MODEL[/] - Model path or HuggingFace ID
[bold]--adapter-path PATH[/] - Path to LoRA adapters
[bold]--max-tokens N[/] - Maximum tokens to generate
[bold]--temp TEMP[/] - Temperature for sampling
"""
                    help_widget.write(help_text)
            except QueryError as e:
                logger.debug(f"Could not populate MLX-LM help text: {e}")


#
# End of LLM_Management_Window.py
#######################################################################################################################
