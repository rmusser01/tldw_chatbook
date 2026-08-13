# tldw_chatbook/UI/LLM_Management_Window.py
#
#
# Imports
import inspect
from typing import TYPE_CHECKING, Callable

#
# 3rd-Party Imports
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.message import Message
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
from ..Widgets.ModelArtifacts import InstallProgressed, InstallStatusChanged

if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:




class OllamaServiceView(VerticalScroll):
    """The Ollama view body, extracted verbatim from `compose` (task-2900).

    Deferred past first paint by `_mount_deferred_views`; the one dynamic
    piece of the original inline block — the prereq line — is computed by
    the window at build time and passed in.
    """

    def __init__(self, prereq_text: str, **kwargs) -> None:
        kwargs.setdefault("id", "llm-view-ollama")
        kwargs.setdefault("classes", "llm-view")
        super().__init__(**kwargs)
        self._prereq_text = prereq_text

    def compose(self) -> ComposeResult:
        yield Label("Ollama Service Management", classes="section-title")
        yield Static(self._prereq_text, classes="prereq-hint")

        with Container(classes="input_container"):
            yield Label("Ollama Executable Path:", classes="inline-label")
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

class LLMManagementWindow(Container):
    """
    Container for the LLM Management Tab's UI.
    Follows Textual best practices with proper navigation and view management.
    """

    class DeferredViewsMounted(Message):
        """The deferred model-management views now exist (task-2900).

        Posted at the end of `_finish_deferred_mount` so ancestors that
        hydrate state into those views on (re)mount — `LLMScreen`'s
        install-progress hydration fires from `on_lab_body_ready` via one
        `call_after_refresh`, which raced (and lost to) the deferred mount —
        get a second, correctly-ordered chance.
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

    .prereq-hint {
        color: $text-muted;
        margin: 0 0 1 0;
        height: auto;
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

    /* Side-by-side form rows (UX-054): the label lives inside the input row
     * instead of stacked above it, so server forms fit the fold. */
    .inline-label {
        width: 28;
        height: 3;
        content-align: right middle;
        padding-right: 1;
        color: $text;
    }
    
    .input_container Input {
        width: 1fr;
        min-width: 16;
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

    # htop-style view cycling (single printable keys; focused text inputs
    # consume them first, so forms are unaffected). See ADR-031.
    BINDINGS = [
        Binding("[", "prev_llm_view", "Previous view", show=False),
        Binding("]", "next_llm_view", "Next view", show=False),
        Binding("1", "jump_view(0)", "View 1", show=False),
        Binding("2", "jump_view(1)", "View 2", show=False),
        Binding("3", "jump_view(2)", "View 3", show=False),
        Binding("4", "jump_view(3)", "View 4", show=False),
        Binding("5", "jump_view(4)", "View 5", show=False),
        Binding("6", "jump_view(5)", "View 6", show=False),
        Binding("7", "jump_view(6)", "View 7", show=False),
        Binding("8", "jump_view(7)", "View 8", show=False),
        Binding("9", "jump_view(8)", "View 9", show=False),
    ]

    def __init__(
        self,
        app_instance: "TldwCli",
        *,
        can_start_import: Callable[[], bool] | None = None,
        on_import_lane_changed: Callable[[bool], None] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._can_start_import = can_start_import or (lambda: True)
        self._on_import_lane_changed = on_import_lane_changed or (lambda _active: None)
        self._async_presentation_generations: dict[str, int] = {}
        self._managed_install_active = False
        self._managed_install_progress = None

        # Map navigation button IDs to view IDs. Order matters: it drives the
        # [/] cycling and the position indicator, so it matches the sidebar's
        # visual order (Ollama first, library views last).
        self.view_mapping = {
            "ollama": "llm-view-ollama",
            "llama-cpp": "llm-view-llama-cpp",
            "llamafile": "llm-view-llamafile",
            "vllm": "llm-view-vllm",
            "onnx": "llm-view-onnx",
            "transformers": "llm-view-transformers",
            "mlx-lm": "llm-view-mlx-lm",
            "curated": "llm-view-curated",
            "installed": "llm-view-installed",
            "external": "llm-view-external",
            "remote": "llm-view-remote",
            "download-models": "llm-view-download-models",
        }

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        logger.debug("LLMManagementWindow.on_mount called")
        # task-2900: the five heavy hidden views mount here, after the first
        # refresh, then the initial activation and the one-shot Ollama
        # initializers run — in that order, because both touch views that no
        # longer exist at compose time (the autofill would otherwise
        # silently never fire, UX-078).
        self.call_after_refresh(self._finish_deferred_mount)
        self.set_interval(3.0, self._schedule_ollama_api_state)

    async def _finish_deferred_mount(self) -> None:
        """Mount deferred views, then run everything that assumes them.

        Each step is guarded individually: before task-2900 these were
        independent `call_after_refresh` callbacks, and one failing never
        stopped the others (a harness without a real app instance breaks
        `_initialize_view`'s process-control sync but must not kill the
        Ollama autofill, UX-078). Sequencing adds the ordering guarantee;
        it must not add failure coupling.
        """
        try:
            await self._mount_deferred_views()
        except Exception:
            logger.exception("Deferred LLM view mount failed")
        for step in (
            self._initialize_view,
            # Autofill the Ollama executable when it's discoverable (UX-078).
            self._autofill_ollama_path,
            # Keep the Ollama API controls gated on a live service (UX-091).
            # task-15473 made this step await the non-blocking Ollama
            # probe; task-15211 then moved the await into a widget-owned
            # worker (see _schedule_ollama_api_state) so an in-flight probe
            # cannot outlive the screen. The step stays in this loop for
            # its ordering slot, but it now only SCHEDULES.
            self._schedule_ollama_api_state,
        ):
            try:
                result = step()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(f"Post-mount step failed: {step.__name__}")
        self.post_message(self.DeferredViewsMounted())

    async def _mount_deferred_views(self) -> None:
        """Mount the deferred views that arrive CSS-hidden (task-2900).

        Screen survey: `#llm-view-download-models` (76 widgets),
        `#llm-view-ollama` (58) and the curated/installed/remote library
        views dominated this screen's 388-widget mount cost while arriving
        `display: none` behind the `-active` CSS mechanism. Mounting them
        here — off the click→paint critical path — leaves every activation
        path working: `watch_active_view` tolerates absent views, and view
        order inside `#llm-main-content` is irrelevant (exactly one view is
        ever shown). Idempotent for re-entered mounts.
        """
        try:
            content = self.query_one("#llm-main-content", Container)
        except QueryError:
            return
        if self.query("#llm-view-ollama"):
            return

        from .Screens.model_curated_view import CuratedView
        from .Screens.model_external_view import ExternalModelView
        from .Screens.model_installed_view import InstalledView
        from .Screens.model_remote_view import RemoteView
        from ..Widgets.HuggingFace import HuggingFaceModelBrowser

        curated = Container(id="llm-view-curated", classes="llm-view")
        installed = Container(id="llm-view-installed", classes="llm-view")
        external = Container(id="llm-view-external", classes="llm-view")
        remote = Container(id="llm-view-remote", classes="llm-view")
        download = Container(id="llm-view-download-models", classes="llm-view")
        await content.mount(
            OllamaServiceView(self._ollama_prereq_text()),
            curated,
            installed,
            external,
            remote,
            download,
        )

        legacy_dir = None
        app_config = getattr(self.app_instance, "app_config", {})
        if isinstance(app_config, dict):
            configured = app_config.get("llm_management", {}).get(
                "model_download_dir"
            )
            if configured:
                from pathlib import Path

                legacy_dir = Path(str(configured)).expanduser()

        await curated.mount(CuratedView(id="curated-models-view"))
        source_service = self.app_instance._ensure_parakeet_source_service()
        await installed.mount(
            InstalledView(
                legacy_dir=legacy_dir,
                on_root_activated=source_service.on_root_activated,
                may_delete=source_service.may_delete,
                recycle_idle=self.app_instance._recycle_idle_local_stt_reference,
                can_start_import=self._can_start_import,
                on_import_lane_changed=self._on_import_lane_changed,
                id="installed-models-view",
            )
        )
        await external.mount(
            ExternalModelView(source_service, id="external-models-view")
        )
        # Remote is explicitly idle until Search is submitted.
        await remote.mount(RemoteView(id="remote-models-view"))
        await download.mount(
            HuggingFaceModelBrowser(
                self.app_instance, id="huggingface-model-browser"
            )
        )

    async def _ollama_api_available(self) -> bool:
        """True when an Ollama service answers (app-launched or external)."""
        proc = getattr(self.app_instance, "ollama_server_process", None)
        if proc is not None and proc.poll() is None:
            return True
        from .Screens.llm_screen import _probe_local_server

        return await _probe_local_server()

    def _schedule_ollama_api_state(self) -> None:
        """Run the Ollama API-state refresh as a widget-owned worker.

        task-15211 (sweep catch): the refresh awaits a real TCP probe of
        127.0.0.1:11434, and a coroutine scheduled straight from the 3s
        interval (or the deferred-mount one-shot) is NOT tied to this
        widget's lifetime -- one already in flight when the screen unmounts
        kept awaiting, and its socket fired during test teardown. Seven
        Lab/LLM-screen tests hit the network guard exactly there. A worker
        owned by this widget is cancelled at unmount, so the probe dies
        with the screen; ``exclusive`` also collapses overlapping polls on
        a slow probe instead of stacking them.
        """
        self.run_worker(
            self._update_ollama_api_state(),
            exclusive=True,
            group="ollama-api-state",
            exit_on_error=False,
        )

    async def _update_ollama_api_state(self) -> None:
        """Disable API controls when no Ollama service is running.

        The banner already says "requires running service"; without gating,
        every dependent action fails at click-time (UX-091).
        """
        # Skip while the screen/tab is inactive so hidden tabs burn no CPU
        # (the probe hits the local Ollama HTTP endpoint on every tick).
        if not self.is_attached or not self.screen.is_active:
            return
        excluded = {
            "ollama-start-service-button",
            "ollama-stop-service-button",
            "ollama-browse-exec-button",
        }
        try:
            view = self.query_one("#llm-view-ollama")
        except Exception:  # noqa: BLE001 - view not mounted
            return
        available = await self._ollama_api_available()
        # task-15473 review: the probe above can take up to ~0.25s (it now
        # awaits instead of blocking), during which this widget can detach
        # or its screen can go inactive -- the old synchronous version was
        # atomic end-to-end, so the pre-await guard above was sufficient;
        # the async version needs the identical check repeated here before
        # touching any button, or a detach mid-probe mutates widgets that
        # are no longer the active screen's.
        if not self.is_attached or not self.screen.is_active:
            return
        for button in view.query(Button):
            if not button.id or button.id in excluded:
                continue
            if not hasattr(button, "_pre_gate_tooltip"):
                button._pre_gate_tooltip = button.tooltip  # type: ignore[attr-defined]
            if available:
                if button.disabled:
                    button.disabled = False
                    button.tooltip = button._pre_gate_tooltip  # type: ignore[attr-defined]
            else:
                button.disabled = True
                button.tooltip = "Requires a running Ollama service — start it above."

    def _autofill_ollama_path(self) -> None:
        """Prefill the Ollama executable path from PATH when empty."""
        import shutil

        found = shutil.which("ollama")
        if not found:
            return
        try:
            path_input = self.query_one("#ollama-exec-path", Input)
        except Exception:  # noqa: BLE001 - view not mounted
            return
        if not path_input.value.strip():
            path_input.value = found

    def on_resize(self) -> None:
        """Hide Detect buttons when the content area gets too narrow
        (the side-by-side row would clip the Browse button, UX-100)."""
        try:
            content = self.query_one("#llm-main-content")
        except Exception:  # noqa: BLE001 - not mounted yet
            return
        narrow = 0 < content.size.width < 70
        for button in self.query(".detect-button"):
            button.display = not narrow

    #: Detect button id -> (target input id, candidate binary names).
    _DETECT_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
        "llamacpp-detect-exec-button": (
            "llamacpp-exec-path",
            ("llama-server", "llama-cpp-server", "server"),
        ),
        "llamafile-detect-exec-button": (
            "llamafile-exec-path",
            ("llamafile", "llavafile"),
        ),
    }

    @staticmethod
    def _discover_binary(names: tuple[str, ...]) -> str | None:
        """Find a backend binary on PATH or in common install locations."""
        import shutil
        from pathlib import Path

        for name in names:
            found = shutil.which(name)
            if found:
                return found
        candidates = (
            Path.home() / ".local" / "bin",
            Path("/usr/local/bin"),
            Path("/opt/homebrew/bin"),
        )
        for directory in candidates:
            for name in names:
                candidate = directory / name
                if candidate.is_file():
                    return str(candidate)
        return None

    @on(Button.Pressed, ".detect-button")
    def handle_detect_button(self, event: Button.Pressed) -> None:
        """Fill the executable path from local discovery (UX-078)."""
        target = self._DETECT_TARGETS.get(event.button.id or "")
        if not target:
            return
        input_id, names = target
        found = self._discover_binary(names)
        path_input = self.query_one(f"#{input_id}", Input)
        if found:
            path_input.value = found
            self.app_instance.notify(f"Found: {found}", severity="information")
        else:
            self.app_instance.notify(
                f"Could not find {names[0]} on PATH or common install locations — "
                "use Browse to point at it manually.",
                severity="warning",
            )
            path_input.focus()

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

    def _ollama_prereq_text(self) -> str:
        """Prereq line for the Ollama view, with PATH detection (UX-070)."""
        import shutil

        found = shutil.which("ollama")
        if found:
            return f"Requires: Ollama installed (found: {found})"
        return (
            "Requires: Ollama installed — not found on PATH. "
            "Install from ollama.com, or use Browse to point at it."
        )

    def compose(self) -> ComposeResult:
        """Compose the LLM Management UI with sidebar navigation and content area."""
        # Main content area
        with Container(id="llm-main-content"):
            # Llama.cpp View
            with VerticalScroll(id="llm-view-llama-cpp", classes="llm-view"):
                yield Label("Llama.cpp Configuration", classes="section-title")
                yield Label(
                    "Launch a llama.cpp server instance with a GGUF model",
                    classes="description",
                )
                yield Static(
                    "Requires: a built llama.cpp server binary and a .gguf model file.",
                    classes="prereq-hint",
                )

                # Primary actions above the fold (UX-054): validation returns
                # focus to any missing field, so starting early is safe.
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

                with Container(classes="input_container"):
                    yield Label("Server Executable:", classes="inline-label")
                    yield Input(
                        id="llamacpp-exec-path",
                        placeholder="e.g. /opt/llama.cpp/build/bin/server",
                    )
                    yield Button(
                        "Browse",
                        id="llamacpp-browse-exec-button",
                        classes="browse_button",
                        tooltip="Choose the llama.cpp server executable.",
                    )
                    yield Button(
                        "Detect",
                        id="llamacpp-detect-exec-button",
                        classes="browse_button detect-button",
                        tooltip="Find the llama.cpp server binary on this machine.",
                    )

                with Container(classes="input_container"):
                    yield Label("GGUF Model File Path:", classes="inline-label")
                    yield Input(
                        id="llamacpp-model-path",
                        placeholder="e.g. /models/model.gguf",
                    )
                    yield Button(
                        "Browse",
                        id="llamacpp-browse-model-button",
                        classes="browse_button",
                        tooltip="Choose a GGUF model file for llama.cpp.",
                    )
                yield Static(
                    "The .gguf model file to serve.",
                    classes="description",
                )

                yield Label("Host:", classes="label")
                yield Input(id="llamacpp-host", value="127.0.0.1")

                yield Label("Port:", classes="label")
                yield Input(id="llamacpp-port", placeholder="8001")
                yield Static("Default 8001 — change it if another server already uses that port.", classes="prereq-hint")

                yield Label("Additional Arguments (single line):", classes="label")
                yield Input(
                    id="llamacpp-additional-args",
                    placeholder="e.g. --n-gpu-layers 1 --threads 4",
                )
                yield Static(
                    "Advanced — fine to leave empty.",
                    classes="description",
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

                yield RichLog(
                    id="llamacpp-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # Llamafile View
            with VerticalScroll(id="llm-view-llamafile", classes="llm-view"):
                yield Label("Llamafile Configuration", classes="section-title")
                yield Label(
                    "Run a self-contained llamafile executable (model included)",
                    classes="description",
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


                with Container(classes="input_container"):
                    yield Label("Llamafile Executable (.llamafile):", classes="inline-label")
                    yield Input(
                        id="llamafile-exec-path", placeholder="/path/to/model.llamafile"
                    )
                    yield Button(
                        "Browse",
                        id="llamafile-browse-exec-button",
                        classes="browse_button",
                        tooltip="Choose the llamafile executable.",
                    )
                    yield Button(
                        "Detect",
                        id="llamafile-detect-exec-button",
                        classes="browse_button detect-button",
                        tooltip="Find the llamafile executable on this machine.",
                    )

                with Container(classes="input_container"):
                    yield Label("Optional External Model (GGUF):", classes="inline-label")
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

                yield Label("Port:", classes="label")
                yield Input(id="llamafile-port", placeholder="8000")
                yield Static("Default 8000 — change it if another server already uses that port.", classes="prereq-hint")

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

                yield RichLog(
                    id="llamafile-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # vLLM View
            with VerticalScroll(id="llm-view-vllm", classes="llm-view"):
                yield Label("vLLM Configuration", classes="section-title")
                yield Label(
                    "High-performance LLM serving with vLLM", classes="description"
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


                with Container(classes="input_container"):
                    yield Label("Python Interpreter Path:", classes="inline-label")
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

                with Container(classes="input_container"):
                    yield Label("Model Path (or HuggingFace Repo ID):", classes="inline-label")
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
                yield Input(id="vllm-port", placeholder="8000")
                yield Static("Default 8000 — change it if another server already uses that port.", classes="prereq-hint")

                yield Label("Additional Arguments:", classes="label")
                yield TextArea(
                    id="vllm-additional-args",
                    classes="additional_args_textarea",
                    theme="vscode_dark",
                )

                yield RichLog(
                    id="vllm-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # ONNX View
            with VerticalScroll(id="llm-view-onnx", classes="llm-view"):
                yield Label("ONNX Runtime Configuration", classes="section-title")
                yield Label(
                    "Run ONNX models with optimized inference", classes="description"
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


                with Container(classes="input_container"):
                    yield Label("Python Interpreter Path:", classes="inline-label")
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

                with Container(classes="input_container"):
                    yield Label("Path to your ONNX Server Script (.py):", classes="inline-label")
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

                with Container(classes="input_container"):
                    yield Label("Model to Load (Path for script):", classes="inline-label")
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
                yield Input(id="onnx-port", placeholder="8004", classes="input_field")
                yield Static("Default 8004 — change it if another server already uses that port.", classes="prereq-hint")

                yield Label("Additional Script Arguments:", classes="label")
                yield TextArea(
                    id="onnx-additional-args",
                    classes="additional_args_textarea",
                    theme="vscode_dark",
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
                    "Hugging Face Transformers Model Management",
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
                yield Label("MLX-LM Configuration", classes="section-title")
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
                yield Input(id="mlx-port", placeholder="8080", classes="input_field")
                yield Static("Default 8080 — change it if another server already uses that port.", classes="prereq-hint")

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


    @on(InstallProgressed)
    def _managed_install_progressed(self, event: InstallProgressed) -> None:
        """Mirror Curated progress into the persistent Installed view."""
        from .Screens.model_installed_view import InstalledView

        self._managed_install_active = True
        self._managed_install_progress = event.progress
        try:
            installed = self.query_one("#installed-models-view", InstalledView)
        except QueryError:
            return
        installed.set_install_state(event.progress, active=True)

    @on(InstallStatusChanged)
    def _managed_install_status_changed(self, event: InstallStatusChanged) -> None:
        """Synchronize install lifecycle state and refresh completed inventory."""
        from .Screens.model_installed_view import InstalledView

        self._managed_install_active = event.active
        if not event.active:
            self._managed_install_progress = None
        try:
            installed = self.query_one("#installed-models-view", InstalledView)
        except QueryError:
            return
        installed.set_install_state(
            self._managed_install_progress,
            active=event.active,
        )
        if not event.active:
            installed.ensure_loaded(force=True)

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

    def action_prev_llm_view(self) -> None:
        """Cycle to the previous sidebar view ([ key)."""
        self._cycle_view(-1)

    def action_next_llm_view(self) -> None:
        """Cycle to the next sidebar view (] key)."""
        self._cycle_view(1)

    def action_jump_view(self, index: int) -> None:
        """Jump directly to sidebar view N (digit keys 1-9)."""
        views = list(self.view_mapping)
        if 0 <= index < len(views):
            self.active_view = views[index]

    def _cycle_view(self, step: int) -> None:
        """Move the active view through the sidebar order."""
        views = list(self.view_mapping)
        try:
            index = views.index(self.active_view)
        except ValueError:
            index = 0
        self.active_view = views[(index + step) % len(views)]

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
                self._start_view_work(new_view, target_view)
            except QueryError:
                logger.error(f"Target view #{target_view_id} not found")

    def _start_view_work(self, view_name: str, view_widget) -> None:
        """Kick off work a view should only do once it is actually shown.

        `compose()` builds all nine views eagerly, so anything a view does
        at mount time happens on every visit to this screen regardless of
        which view the user wanted. The HuggingFace browse was doing exactly
        that -- a live request to huggingface.co on arrival, for users who
        never open Download Models (task-887).
        """
        if view_name in {"curated", "installed"}:
            try:
                managed_view = view_widget.query_one(
                    "#curated-models-view"
                    if view_name == "curated"
                    else "#installed-models-view"
                )
            except QueryError:
                logger.debug(f"{view_name.title()} view is unavailable; skipped.")
                return
            managed_view.ensure_loaded()
            return
        if view_name != "download-models":
            return
        # Local import: this module is on the Models mount path, and the
        # point of the change is to keep that path cheap.
        from ..Widgets.HuggingFace.model_search_widget import ModelSearchWidget

        try:
            search = view_widget.query_one(ModelSearchWidget)
        except QueryError:
            logger.debug("Download Models view has no ModelSearchWidget; skipped.")
            return
        search.ensure_initial_browse()

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
