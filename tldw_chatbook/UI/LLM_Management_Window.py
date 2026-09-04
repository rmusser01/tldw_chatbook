# tldw_chatbook/UI/LLM_Management_Window.py
#
#
# Imports
import functools
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Callable

#
# 3rd-Party Imports
from textual import on
from textual.app import ComposeResult, compose as compose_widgets
from textual.binding import Binding
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.css.query import QueryError
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Static,
    Button,
    Input,
    RichLog,
    Label,
    TextArea,
    Collapsible,
    Select,
)
from loguru import logger

# Local Imports
from ..Event_Handlers.LLM_Management_Events.llm_management_events import (
    LLM_MANAGEMENT_BUTTON_HANDLERS,
)
from ..Event_Handlers.LLM_Management_Events.gguf_source_modes import (
    GGUFSourceMode,
    GGUFSourceSelection,
    ManagedGGUFChoice,
    initial_gguf_selection,
    managed_gguf_choices,
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
    handle_vllm_setup_check_requested,
    handle_vllm_local_directory_browse_requested,
    handle_vllm_setup_start_requested,
    handle_vllm_setup_stop_requested,
)
from .LLM_Management.vllm_setup_view import VllmSetupView
from ..Event_Handlers.LLM_Management_Events.server_lifecycle import (
    current_llm_destination,
    server_lifecycle_snapshot,
    server_is_active,
)
from ..Model_Artifacts.service import ArtifactRef
from ..Model_Artifacts.store import managed_service
from ..Utils.log_widget_manager import LogWidgetManager
from ..Widgets.ModelArtifacts import InstallProgressed, InstallStatusChanged

if TYPE_CHECKING:
    from ..app import TldwCli
#
#######################################################################################################################
#
# Functions:


class _LazyServerPane(VerticalScroll):
    """A stable provider pane whose pre-composed body can be mounted later."""

    def defer_body(self) -> tuple[Widget, ...]:
        """Detach pending compose children without mounting them."""

        body = tuple(self._pending_children)
        self._pending_children.clear()
        return body


class _LLMMainContent(Container):
    """Composition root exposing the provider panes collected below it."""

    @property
    def pending_views(self) -> tuple[Widget, ...]:
        """Return view roots captured by Textual's compose stack."""

        return tuple(self._pending_children)


class OllamaServiceView(VerticalScroll):
    """The Ollama view body, extracted verbatim from `compose` (task-2900).

    Mounted on first selection by `_mount_deferred_views`; the one dynamic
    piece of the original inline block — the prereq line — is computed by
    the window at build time and passed in.
    """

    def __init__(self, prereq_text: str, **kwargs) -> None:
        kwargs.setdefault("classes", "llm-view-body")
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
            yield Button("Start Ollama Service", id="ollama-start-service-button")
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
        """A lazy model-management view is ready for state hydration.

        The historical name is retained for message-handler compatibility.
        It is posted after initial llama.cpp setup and after each first-used
        pane has finished composing its descendants.
        """

    class ManagedGGUFHandoffResolved(Message):
        """Report exact managed-GGUF validation to the owning Models screen."""

        def __init__(
            self,
            provider: str,
            reference: ArtifactRef,
            *,
            succeeded: bool,
            reason: str | None = None,
        ) -> None:
            """Create a path-free handoff result.

            Args:
                provider: Internal GGUF runtime key.
                reference: Exact managed identity that was validated.
                succeeded: Whether the identity was committed to the runtime.
                reason: Allowlisted failure category, never a filesystem path.
            """
            super().__init__()
            self.provider = provider
            self.reference = reference
            self.succeeded = succeeded
            self.reason = reason

    BUNDLED_CSS = """
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
    GGUF_PROVIDERS = ("llamacpp", "llamafile")
    GGUF_SOURCE_CONTROLS = {
        provider: (
            f"{provider}-gguf-source-mode",
            f"{provider}-gguf-managed-select",
            f"{provider}-gguf-refresh-button",
            f"{provider}-model-path",
            f"{provider}-browse-model-button",
            f"{provider}-exec-path",
            f"{provider}-browse-exec-button",
            f"{provider}-detect-exec-button",
        )
        for provider in GGUF_PROVIDERS
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
        self._gguf_sources = {
            provider: initial_gguf_selection(provider, "")
            for provider in self.GGUF_PROVIDERS
        }
        self._managed_gguf_choices: tuple[ManagedGGUFChoice, ...] = ()
        self._managed_gguf_inventory_generation = 0
        self._managed_gguf_inventory_started = False
        self._managed_gguf_inventory_error = False
        self._pending_managed_gguf_handoff: tuple[str, ArtifactRef] | None = None
        self._server_active_states = {
            provider: False for provider in self.SERVER_CONTROLS
        }
        self._model_library_focus_ids = {
            "curated": "curated-models-refresh",
            "installed": "installed-models-refresh",
        }
        self._model_library_widget_ids = {
            "curated": "curated-models-view",
            "installed": "installed-models-view",
        }
        self._lazy_server_bodies: dict[str, tuple[Widget, ...]] = {}
        self._populated_views: set[str] = set()
        self._populating_views: set[str] = set()
        self._vllm_preflight_generation = 0

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
        }

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        logger.debug("LLMManagementWindow.on_mount called")
        self.watch(self.screen, "focused", self._record_model_library_focus, init=False)
        # The llama.cpp body is part of the first composition. All other
        # provider bodies stay unmounted until their first selection.
        self.call_after_refresh(self._finish_deferred_mount)
        self.set_interval(3.0, self._schedule_ollama_api_state)

    async def _finish_deferred_mount(self) -> None:
        """Activate llama.cpp after its first composed frame is available.

        Each step is guarded individually: before task-2900 these were
        independent `call_after_refresh` callbacks, and one failing never
        stopped the others (a harness without a real app instance breaks
        `_initialize_view`'s process-control sync but must not kill the
        Ollama autofill, UX-078). Sequencing adds the ordering guarantee;
        it must not add failure coupling.
        """
        for step in (
            self._initialize_view,
        ):
            try:
                result = step()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(f"Post-mount step failed: {step.__name__}")
        self.post_message(self.DeferredViewsMounted())

    async def _mount_deferred_views(self, view_name: str) -> None:
        """Populate one stable pane body on first selection.

        The method name is retained for compatibility with older task-level
        architecture checks. Unlike task-2900's batch deferral, this method
        mounts only the requested body and leaves it cached in its pane.
        """
        if view_name in self._populated_views:
            return
        view_id = self.view_mapping.get(view_name)
        if view_id is None:
            return
        try:
            pane = self.query_one(f"#{view_id}")
        except QueryError:
            return

        server_body = self._lazy_server_bodies.get(view_name)
        if server_body is not None:
            remaining = tuple(widget for widget in server_body if not widget.is_mounted)
            if remaining:
                await pane.mount_all(remaining)
            self._lazy_server_bodies.pop(view_name, None)
        elif view_name == "ollama":
            await pane.mount(OllamaServiceView(self._ollama_prereq_text()))
        elif view_name in {"curated", "installed", "external", "remote"}:
            from .Screens.model_curated_view import CuratedView
            from .Screens.model_external_view import ExternalModelView
            from .Screens.model_installed_view import InstalledView
            from .Screens.model_remote_view import RemoteView

            legacy_dir = None
            app_config = getattr(self.app_instance, "app_config", {})
            if isinstance(app_config, dict):
                configured = app_config.get("llm_management", {}).get(
                    "model_download_dir"
                )
                if configured:
                    legacy_dir = Path(str(configured)).expanduser()

            observation_provider = getattr(
                self.app_instance,
                "_audio_cpp_model_library_observation_snapshot",
                None,
            )
            if view_name == "curated":
                await pane.mount(
                    CuratedView(
                        observation_provider=observation_provider,
                        id="curated-models-view",
                    )
                )
            elif view_name == "remote":
                # Remote is explicitly idle until Search is submitted.
                await pane.mount(RemoteView(id="remote-models-view"))
            else:
                source_service = self.app_instance._ensure_parakeet_source_service()
                if view_name == "installed":
                    await pane.mount(
                        InstalledView(
                            legacy_dir=legacy_dir,
                            on_root_activated=source_service.on_root_activated,
                            may_delete=source_service.may_delete,
                            recycle_idle=(
                                self.app_instance._recycle_idle_local_stt_reference
                            ),
                            can_start_import=self._can_start_import,
                            on_import_lane_changed=self._on_import_lane_changed,
                            observation_provider=observation_provider,
                            id="installed-models-view",
                        )
                    )
                elif view_name == "external":
                    await pane.mount(
                        ExternalModelView(source_service, id="external-models-view")
                    )
        else:
            raise RuntimeError(f"Deferred body for {view_name!r} is unavailable")

        self._populated_views.add(view_name)
        self.call_after_refresh(self._view_population_ready, view_name)

    def _view_population_ready(self, view_name: str) -> None:
        """Hydrate and announce a first-selected body after child composition."""

        if not self.is_attached:
            return
        progress_ids = {
            "curated": "curated-model-install-progress",
            "installed": "installed-model-install-progress",
            "remote": "remote-model-install-progress",
        }
        progress_id = progress_ids.get(view_name)
        if progress_id is not None:
            try:
                progress = self.query_one(f"#{progress_id}")
            except QueryError:
                self.call_after_refresh(self._view_population_ready, view_name)
                return
            if not list(progress.query("#model-install-progress-phase")):
                self.call_after_refresh(self._view_population_ready, view_name)
                return
        if view_name == "installed" and self._managed_install_progress is not None:
            from .Screens.model_installed_view import InstalledView

            try:
                installed = self.query_one("#installed-models-view", InstalledView)
            except QueryError:
                pass
            else:
                installed.set_install_state(
                    self._managed_install_progress,
                    active=self._managed_install_active,
                )
        if view_name == "ollama":
            self._autofill_ollama_path()
            self._schedule_ollama_api_state()
        self._try_commit_pending_managed_gguf_handoff()
        self.post_message(self.DeferredViewsMounted())

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

        task-22220: the inactive-screen gate is hoisted here from the
        coroutine -- the 3 s tick on a hidden tab used to construct the
        coroutine and schedule a worker every fire just so the coroutine's
        own first line could drop it. An inactive screen now constructs
        nothing. The coroutine keeps its own pre-await guard (the
        scheduling->running race) and post-await re-check (mid-probe
        deactivation, task-15473).
        """
        if (
            "ollama" not in self._populated_views
            or not self.is_attached
            or not self.screen.is_active
        ):
            return
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
        self.set_class(narrow, "gguf-source-narrow")
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
        if not self.active_view:
            self.active_view = "llama-cpp"
        if "llama-cpp" in self._populated_views:
            self._render_gguf_source("llamacpp")
        if "llamafile" in self._populated_views:
            self._render_gguf_source("llamafile")
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

    def _gguf_mode_options(self, provider: str) -> tuple[tuple[str, str], ...]:
        """Return the exact path-free mode matrix for one GGUF provider."""

        if provider == "llamacpp":
            return (
                ("Managed GGUF", GGUFSourceMode.MANAGED.value),
                ("External GGUF", GGUFSourceMode.EXTERNAL.value),
            )
        return (
            ("Embedded", GGUFSourceMode.EMBEDDED.value),
            ("Managed GGUF", GGUFSourceMode.MANAGED.value),
            ("External GGUF", GGUFSourceMode.EXTERNAL.value),
        )

    def _server_active(self, provider: str) -> bool:
        """Read lifecycle truth, with the legacy app-less harness idle."""

        return self.app_instance is not None and server_is_active(
            self.app_instance, provider
        )

    def _gguf_managed_options(self) -> list[tuple[str, object]]:
        """Return path-free labels carrying exact artifact references."""

        if not self._managed_gguf_choices:
            return [("No managed GGUF models", Select.NULL)]
        return [
            (choice.label, choice.reference) for choice in self._managed_gguf_choices
        ]

    def _managed_gguf_ready(self, provider: str) -> bool:
        """Return whether the retained ref is present in the current inventory."""

        selection = self._gguf_sources[provider]
        return not self._managed_gguf_inventory_error and any(
            choice.reference == selection.managed_ref
            for choice in self._managed_gguf_choices
        )

    def _compose_gguf_source(self, provider: str, active: bool) -> ComposeResult:
        """Compose one compact, mutually-exclusive GGUF source control."""

        selection = self._gguf_sources[provider]
        with Horizontal(classes="gguf-source-mode-row"):
            yield Label("Model source:", classes="gguf-source-label")
            yield Select(
                self._gguf_mode_options(provider),
                value=selection.mode.value,
                allow_blank=False,
                compact=True,
                id=f"{provider}-gguf-source-mode",
                classes="gguf-source-mode",
                disabled=active,
            )

        with Vertical(
            id=f"{provider}-gguf-managed-region",
            classes=(
                "gguf-source-region gguf-source-managed"
                + (" -active" if selection.mode is GGUFSourceMode.MANAGED else "")
            ),
        ):
            with Horizontal(classes="gguf-source-choice-row"):
                yield Select(
                    self._gguf_managed_options(),
                    value=(
                        selection.managed_ref
                        if selection.managed_ref is not None
                        else Select.NULL
                    ),
                    prompt="Choose a managed GGUF model",
                    compact=True,
                    id=f"{provider}-gguf-managed-select",
                    classes="gguf-source-managed-select",
                    disabled=active or not self._managed_gguf_choices,
                )
                yield Button(
                    "Refresh managed models",
                    id=f"{provider}-gguf-refresh-button",
                    classes="gguf-source-refresh",
                    disabled=active,
                )

        with Vertical(
            id=f"{provider}-gguf-external-region",
            classes=(
                "gguf-source-region gguf-source-external"
                + (" -active" if selection.mode is GGUFSourceMode.EXTERNAL else "")
            ),
        ):
            with Horizontal(classes="gguf-source-choice-row"):
                yield Input(
                    id=f"{provider}-model-path",
                    placeholder="/path/to/external-model.gguf",
                    disabled=active,
                )
                yield Button(
                    "Browse",
                    id=f"{provider}-browse-model-button",
                    classes="browse_button gguf-source-browse",
                    disabled=active,
                    tooltip=(
                        "Choose a GGUF model file for llama.cpp."
                        if provider == "llamacpp"
                        else "Choose an optional external GGUF model for llamafile."
                    ),
                )
            yield Static(
                "Outside Chatbook · integrity unknown",
                classes="gguf-source-authority",
            )
            yield Static(
                "This file is used in place and is not imported, copied, "
                "deleted, or selected globally.",
                classes="gguf-source-copy",
            )

        with Vertical(
            id=f"{provider}-gguf-embedded-region",
            classes=(
                "gguf-source-region gguf-source-embedded"
                + (" -active" if selection.mode is GGUFSourceMode.EMBEDDED else "")
            ),
        ):
            yield Static(
                "Use the model embedded in this llamafile executable.",
                classes="gguf-source-copy",
            )

        yield Static(
            self._gguf_authority_text(provider),
            id=f"{provider}-gguf-source-status",
            classes="gguf-source-status",
        )

    def _compose_server_panes(self) -> ComposeResult:
        """Build the six server panes so inactive bodies can be detached."""
        initial_active = {
            provider: self._server_active(provider) for provider in self.GGUF_PROVIDERS
        }
        # Main content area
        with _LLMMainContent(id="llm-main-content"):
            # Llama.cpp View
            with _LazyServerPane(id="llm-view-llama-cpp", classes="llm-view"):
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
                        disabled=initial_active["llamacpp"],
                    )
                    llamacpp_stop = Button(
                        "Stop Server",
                        id="llamacpp-stop-server-button",
                        classes="action_button",
                        disabled=True,
                    )
                    llamacpp_stop.disabled = not initial_active["llamacpp"]
                    yield llamacpp_stop

                with Container(classes="input_container"):
                    yield Label("Server Executable:", classes="inline-label")
                    yield Input(
                        id="llamacpp-exec-path",
                        placeholder="e.g. /opt/llama.cpp/build/bin/server",
                        disabled=initial_active["llamacpp"],
                    )
                    yield Button(
                        "Browse",
                        id="llamacpp-browse-exec-button",
                        classes="browse_button",
                        disabled=initial_active["llamacpp"],
                        tooltip="Choose the llama.cpp server executable.",
                    )
                    yield Button(
                        "Detect",
                        id="llamacpp-detect-exec-button",
                        classes="browse_button detect-button",
                        disabled=initial_active["llamacpp"],
                        tooltip="Find the llama.cpp server binary on this machine.",
                    )

                yield from self._compose_gguf_source(
                    "llamacpp", initial_active["llamacpp"]
                )

                yield Label("Host:", classes="label")
                yield Input(id="llamacpp-host", value="127.0.0.1")

                yield Label("Port:", classes="label")
                yield Input(id="llamacpp-port", placeholder="8001")
                yield Static(
                    "Default 8001 — change it if another server already uses that port.",
                    classes="prereq-hint",
                )

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
            with _LazyServerPane(id="llm-view-llamafile", classes="llm-view"):
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
                        disabled=initial_active["llamafile"],
                    )
                    llamafile_stop = Button(
                        "Stop Server",
                        id="llamafile-stop-server-button",
                        classes="action_button",
                        disabled=True,
                    )
                    llamafile_stop.disabled = not initial_active["llamafile"]
                    yield llamafile_stop

                with Container(classes="input_container"):
                    yield Label(
                        "Llamafile Executable (.llamafile):", classes="inline-label"
                    )
                    yield Input(
                        id="llamafile-exec-path",
                        placeholder="/path/to/model.llamafile",
                        disabled=initial_active["llamafile"],
                    )
                    yield Button(
                        "Browse",
                        id="llamafile-browse-exec-button",
                        classes="browse_button",
                        disabled=initial_active["llamafile"],
                        tooltip="Choose the llamafile executable.",
                    )
                    yield Button(
                        "Detect",
                        id="llamafile-detect-exec-button",
                        classes="browse_button detect-button",
                        disabled=initial_active["llamafile"],
                        tooltip="Find the llamafile executable on this machine.",
                    )

                yield from self._compose_gguf_source(
                    "llamafile", initial_active["llamafile"]
                )

                yield Label("Host:", classes="label")
                yield Input(id="llamafile-host", value="127.0.0.1")

                yield Label("Port:", classes="label")
                yield Input(id="llamafile-port", placeholder="8000")
                yield Static(
                    "Default 8000 — change it if another server already uses that port.",
                    classes="prereq-hint",
                )

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
            with _LazyServerPane(id="llm-view-vllm", classes="llm-view"):
                yield VllmSetupView(id="vllm-setup-view")

            # ONNX View
            with _LazyServerPane(id="llm-view-onnx", classes="llm-view"):
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
                    yield Label(
                        "Path to your ONNX Server Script (.py):", classes="inline-label"
                    )
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
                    yield Label(
                        "Model to Load (Path for script):", classes="inline-label"
                    )
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
                yield Static(
                    "Default 8004 — change it if another server already uses that port.",
                    classes="prereq-hint",
                )

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
            with _LazyServerPane(id="llm-view-transformers", classes="llm-view"):
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

                yield Label("Model Operations Output", classes="section_label")
                yield RichLog(
                    id="transformers-log-output",
                    classes="log_output",
                    wrap=True,
                    highlight=True,
                )

            # MLX-LM View
            with _LazyServerPane(id="llm-view-mlx-lm", classes="llm-view"):
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
                yield Static(
                    "Default 8080 — change it if another server already uses that port.",
                    classes="prereq-hint",
                )

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

    def compose(self) -> ComposeResult:
        """Compose stable pane shells and only the initial llama.cpp body.

        Returns:
            A composition result yielding the stable Models content root.
        """

        roots = compose_widgets(self, self._compose_server_panes())
        if len(roots) != 1 or not isinstance(roots[0], _LLMMainContent):
            raise RuntimeError("LLM server pane composition produced an invalid root")
        content = roots[0]
        self._lazy_server_bodies.clear()
        self._populated_views = {"llama-cpp"}
        view_name_by_id = {
            view_id: view_name for view_name, view_id in self.view_mapping.items()
        }
        for pane in content.pending_views:
            if not isinstance(pane, _LazyServerPane):
                continue
            view_name = view_name_by_id.get(pane.id or "")
            if view_name is not None and view_name != "llama-cpp":
                self._lazy_server_bodies[view_name] = pane.defer_body()

        for view_name in ("ollama", "curated", "installed", "external", "remote"):
            content.compose_add_child(
                Container(id=self.view_mapping[view_name], classes="llm-view")
            )
        yield content

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

    def gguf_source_snapshot(self, provider: str) -> GGUFSourceSelection:
        """Return one immutable launch snapshot without store access."""

        try:
            selection = self._gguf_sources[provider]
        except KeyError:
            raise ValueError("unsupported GGUF source provider") from None
        if selection.mode is GGUFSourceMode.MANAGED:
            if self._managed_gguf_inventory_error:
                raise ValueError("managed GGUF inventory unavailable")
            if not self._managed_gguf_ready(provider):
                raise ValueError("managed GGUF selection unavailable")
        try:
            value = self.query_one(f"#{provider}-model-path", Input).value
        except QueryError:
            return selection.validate_for(provider)
        external_path = Path(value) if value.strip() else None
        if external_path != selection.external_path:
            selection = GGUFSourceSelection(
                mode=selection.mode,
                managed_ref=selection.managed_ref,
                external_path=external_path,
            )
            self._gguf_sources[provider] = selection
        return selection.validate_for(provider)

    def configure_managed_gguf(
        self,
        provider: str,
        reference: ArtifactRef,
    ) -> bool:
        """Open a GGUF runtime and preselect one exact managed model.

        The method changes configuration state only. It never activates a
        managed root, claims a server, or starts a process.

        Args:
            provider: Internal GGUF provider key (``llamacpp`` or ``llamafile``).
            reference: Exact verified managed root to select.

        Returns:
            ``True`` when the handoff was accepted, including while a fresh
            inventory read is resolving the exact reference.
        """
        if (
            provider not in self.GGUF_PROVIDERS
            or type(reference) is not ArtifactRef
            or any(self._server_active(item) for item in self.GGUF_PROVIDERS)
        ):
            return False
        self.active_view = "llama-cpp" if provider == "llamacpp" else "llamafile"
        self._pending_managed_gguf_handoff = (provider, reference)
        if reference in {choice.reference for choice in self._managed_gguf_choices}:
            self._try_commit_pending_managed_gguf_handoff()
            return True
        if not self._refresh_managed_gguf_inventory():
            self._pending_managed_gguf_handoff = None
            return False
        return True

    def _try_commit_pending_managed_gguf_handoff(self) -> None:
        """Commit a proven handoff once its lazy provider controls exist."""

        pending = self._pending_managed_gguf_handoff
        if pending is None:
            return
        provider, reference = pending
        if reference not in {choice.reference for choice in self._managed_gguf_choices}:
            return
        if any(self._server_active(item) for item in self.GGUF_PROVIDERS):
            self._pending_managed_gguf_handoff = None
            self.post_message(
                self.ManagedGGUFHandoffResolved(
                    provider,
                    reference,
                    succeeded=False,
                    reason="server-active",
                )
            )
            return
        try:
            self.query_one(f"#{provider}-gguf-source-mode", Select)
            self.query_one(f"#{provider}-gguf-managed-select", Select)
        except QueryError:
            view_name = "llama-cpp" if provider == "llamacpp" else "llamafile"
            self.ensure_view_populated(view_name)
            return
        self._commit_managed_gguf_handoff(provider, reference)

    def _commit_managed_gguf_handoff(
        self,
        provider: str,
        reference: ArtifactRef,
    ) -> None:
        """Commit a provider/ref pair already proven present in inventory."""
        selection = self._gguf_sources[provider]
        self._gguf_sources[provider] = GGUFSourceSelection(
            mode=GGUFSourceMode.MANAGED,
            managed_ref=reference,
            external_path=selection.external_path,
        )
        mode = self.query_one(f"#{provider}-gguf-source-mode", Select)
        managed = self.query_one(f"#{provider}-gguf-managed-select", Select)
        with mode.prevent(Select.Changed):
            mode.value = GGUFSourceMode.MANAGED.value
        with managed.prevent(Select.Changed):
            managed.set_options(self._gguf_managed_options())
            managed.value = reference
        self._pending_managed_gguf_handoff = None
        self._render_gguf_source(provider)
        self._sync_process_controls(provider)
        self.post_message(
            self.ManagedGGUFHandoffResolved(
                provider,
                reference,
                succeeded=True,
            )
        )

    def _render_gguf_source(self, provider: str) -> None:
        """Patch one source region and its path-free status in place."""

        selection = self._gguf_sources[provider]
        for mode in GGUFSourceMode:
            try:
                region = self.query_one(f"#{provider}-gguf-{mode.value}-region")
            except QueryError:
                continue
            region.set_class(mode is selection.mode, "-active")
        self._render_gguf_authority(provider)

    def _render_gguf_authority(self, provider: str) -> None:
        """Render selected or active authority from lifecycle truth."""

        try:
            status = self.query_one(f"#{provider}-gguf-source-status", Static)
        except QueryError:
            return
        status.update(self._gguf_authority_text(provider))

    def _gguf_authority_text(self, provider: str) -> str:
        """Return path-free authority text without querying mounted widgets."""

        if self.app_instance is None:
            claim, process = None, None
        else:
            claim, process = server_lifecycle_snapshot(self.app_instance, provider)
        if claim is not None:
            phase = "Running" if process is not None else "Pending"
            authority = claim.authority or "Local process"
            return f"{phase} authority: {authority}"
        if (
            self._managed_gguf_inventory_error
            and self._gguf_sources[provider].mode is GGUFSourceMode.MANAGED
        ):
            return (
                "Managed GGUF inventory unavailable. Refresh managed models to retry."
            )
        if (
            self._gguf_sources[provider].mode is GGUFSourceMode.MANAGED
            and self._gguf_sources[provider].managed_ref is None
        ):
            return (
                "Selected managed GGUF is unavailable. "
                "Choose another managed model or refresh."
            )
        if (
            self._gguf_sources[provider].mode is GGUFSourceMode.EXTERNAL
            and self._gguf_sources[provider].external_path is None
        ):
            return "Choose an external GGUF file to enable Start."
        return f"Selected authority: {self._gguf_sources[provider].authority}"

    def _select_source_mode(self, provider: str, mode: GGUFSourceMode) -> None:
        """Switch one provider without discarding inactive selections."""

        if self._server_active(provider):
            select = self.query_one(f"#{provider}-gguf-source-mode", Select)
            with select.prevent(Select.Changed):
                select.value = self._gguf_sources[provider].mode.value
            return
        self._gguf_sources[provider] = self._gguf_sources[provider].for_mode(mode)
        self._render_gguf_source(provider)
        self._sync_process_controls(provider)
        if mode is GGUFSourceMode.MANAGED:
            self._ensure_managed_gguf_inventory()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Apply source mode and exact managed-reference selections."""

        select_id = event.select.id or ""
        for provider in self.GGUF_PROVIDERS:
            if select_id == f"{provider}-gguf-source-mode":
                if event.value is not Select.NULL:
                    self._select_source_mode(provider, GGUFSourceMode(str(event.value)))
                return
            if select_id != f"{provider}-gguf-managed-select":
                continue
            if self._server_active(provider):
                with event.select.prevent(Select.Changed):
                    event.select.value = (
                        self._gguf_sources[provider].managed_ref or Select.NULL
                    )
                return
            if event.value is Select.NULL:
                return
            self._gguf_sources[provider] = GGUFSourceSelection(
                mode=self._gguf_sources[provider].mode,
                managed_ref=event.value,
                external_path=self._gguf_sources[provider].external_path,
            )
            self._sync_process_controls(provider)
            return

    def on_input_changed(self, event: Input.Changed) -> None:
        """Retain external GGUF paths while preserving lifecycle fencing."""

        input_id = event.input.id or ""
        for provider in self.GGUF_PROVIDERS:
            if input_id != f"{provider}-model-path":
                continue
            selection = self._gguf_sources[provider]
            if self._server_active(provider):
                expected = str(selection.external_path or "")
                if event.input.value != expected:
                    with event.input.prevent(Input.Changed):
                        event.input.value = expected
                return
            external_path = Path(event.value) if event.value.strip() else None
            mode = selection.mode
            if (
                provider == "llamafile"
                and mode is GGUFSourceMode.EMBEDDED
                and selection.external_path is None
                and external_path is not None
            ):
                mode = GGUFSourceMode.EXTERNAL
                mode_select = self.query_one(f"#{provider}-gguf-source-mode", Select)
                with mode_select.prevent(Select.Changed):
                    mode_select.value = mode.value
            self._gguf_sources[provider] = GGUFSourceSelection(
                mode=mode,
                managed_ref=selection.managed_ref,
                external_path=external_path,
            )
            self._render_gguf_source(provider)
            self._sync_process_controls(provider)
            return

    def _ensure_managed_gguf_inventory(self) -> None:
        """Load inventory once; explicit Refresh may always start a new generation."""

        if not self._managed_gguf_inventory_started:
            self._refresh_managed_gguf_inventory()

    def _refresh_managed_gguf_inventory(self) -> bool:
        """Start a path-free, generation-fenced inventory thread worker.

        Returns:
            ``True`` when a worker was scheduled, or ``False`` when current
            lifecycle authority prevents a refresh.
        """

        if self.app_instance is None:
            return False
        if any(self._server_active(p) for p in self.GGUF_PROVIDERS):
            return False
        self._managed_gguf_inventory_started = True
        self._managed_gguf_inventory_generation += 1
        generation = self._managed_gguf_inventory_generation
        self.app_instance.run_worker(
            functools.partial(self._load_managed_gguf_inventory, generation),
            thread=True,
            group="managed_gguf_inventory",
            description="Loading managed GGUF models",
            exclusive=True,
        )
        return True

    def _load_managed_gguf_inventory(self, generation: int) -> None:
        """Read store inventory off-loop and deliver only path-free choices."""

        try:
            choices = managed_gguf_choices(managed_service().list_installed())
            error = None
        except Exception:
            choices = ()
            error = True
            logger.error("Managed GGUF inventory load failed")
        self.app_instance.call_from_thread(
            self._apply_managed_gguf_inventory,
            generation,
            choices,
            error,
        )

    def _apply_managed_gguf_inventory(
        self,
        generation: int,
        choices: tuple[ManagedGGUFChoice, ...],
        error: bool | None,
    ) -> None:
        """Apply one current inventory result to the current destination only."""

        if (
            generation != self._managed_gguf_inventory_generation
            or current_llm_destination(self.app_instance) is not self
            or not self.is_attached
        ):
            return
        if any(self._server_active(p) for p in self.GGUF_PROVIDERS):
            self._managed_gguf_inventory_started = False
            pending = self._pending_managed_gguf_handoff
            if pending is not None:
                self._pending_managed_gguf_handoff = None
                provider, reference = pending
                self.post_message(
                    self.ManagedGGUFHandoffResolved(
                        provider,
                        reference,
                        succeeded=False,
                        reason="server-active",
                    )
                )
            return
        self._managed_gguf_choices = choices
        self._managed_gguf_inventory_error = bool(error)
        references = {choice.reference for choice in choices}
        for provider in self.GGUF_PROVIDERS:
            selection = self._gguf_sources[provider]
            if selection.managed_ref is None and choices:
                selection = GGUFSourceSelection(
                    mode=selection.mode,
                    managed_ref=choices[0].reference,
                    external_path=selection.external_path,
                )
                self._gguf_sources[provider] = selection
            elif (
                not error
                and selection.managed_ref is not None
                and selection.managed_ref not in references
            ):
                selection = GGUFSourceSelection(
                    mode=selection.mode,
                    managed_ref=None,
                    external_path=selection.external_path,
                )
                self._gguf_sources[provider] = selection
            try:
                select = self.query_one(f"#{provider}-gguf-managed-select", Select)
            except QueryError:
                # The sibling GGUF pane has not been selected/mounted yet.
                continue
            with select.prevent(Select.Changed):
                select.set_options(self._gguf_managed_options())
                select.value = (
                    selection.managed_ref
                    if selection.managed_ref in references
                    else Select.NULL
                )
            self._sync_process_controls(provider)
        pending = self._pending_managed_gguf_handoff
        if pending is not None:
            provider, reference = pending
            if not error and reference in references:
                self._try_commit_pending_managed_gguf_handoff()
            else:
                self._pending_managed_gguf_handoff = None
                self.post_message(
                    self.ManagedGGUFHandoffResolved(
                        provider,
                        reference,
                        succeeded=False,
                        reason="inventory-error" if error else "missing",
                    )
                )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route allowlisted actions inside this destination."""

        event.stop()
        button = event.button
        if not button.id:
            return

        if button.id in {
            f"{provider}-gguf-refresh-button" for provider in self.GGUF_PROVIDERS
        }:
            self._refresh_managed_gguf_inventory()
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

    @on(VllmSetupView.CheckRequested)
    def _on_vllm_setup_check_requested(self, event: VllmSetupView.CheckRequested) -> None:
        handle_vllm_setup_check_requested(self, self.app_instance, event)

    @on(VllmSetupView.StartRequested)
    def _on_vllm_setup_start_requested(self, event: VllmSetupView.StartRequested) -> None:
        handle_vllm_setup_start_requested(self, self.app_instance, event)

    @on(VllmSetupView.StopRequested)
    async def _on_vllm_setup_stop_requested(self, event: VllmSetupView.StopRequested) -> None:
        await handle_vllm_setup_stop_requested(self, self.app_instance, event)

    @on(VllmSetupView.LocalDirectoryBrowseRequested)
    async def _on_vllm_local_directory_browse_requested(
        self, event: VllmSetupView.LocalDirectoryBrowseRequested
    ) -> None:
        await handle_vllm_local_directory_browse_requested(self, self.app_instance, event)

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
        active = self._server_active(provider)
        was_active = self._server_active_states.get(provider, False)
        self._server_active_states[provider] = active
        try:
            start = self.query_one(f"#{start_id}", Button)
            stop = self.query_one(f"#{stop_id}", Button)
            source_ready = True
            if provider in self.GGUF_PROVIDERS:
                selection = self._gguf_sources[provider]
                source_ready = (
                    selection.mode is GGUFSourceMode.EMBEDDED
                    or (
                        selection.mode is GGUFSourceMode.MANAGED
                        and self._managed_gguf_ready(provider)
                    )
                    or (
                        selection.mode is GGUFSourceMode.EXTERNAL
                        and selection.external_path is not None
                    )
                )
            start.disabled = active or not source_ready
            stop.disabled = not active
            if provider in self.GGUF_PROVIDERS:
                for control_id in self.GGUF_SOURCE_CONTROLS[provider]:
                    control = self.query_one(f"#{control_id}")
                    if control_id == f"{provider}-gguf-managed-select":
                        control.disabled = (
                            active
                            or self._managed_gguf_inventory_error
                            or not self._managed_gguf_choices
                        )
                    else:
                        control.disabled = active
                self._render_gguf_authority(provider)
                if active and not was_active:
                    stop.focus()
                elif was_active and not active:
                    start.focus()
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
                if new_view not in self._populated_views:
                    self.ensure_view_populated(new_view)
                else:
                    self._finish_view_activation(new_view, target_view)
            except QueryError:
                logger.error(f"Target view #{target_view_id} not found")

    def ensure_view_populated(self, view_name: str) -> None:
        """Schedule first population for one pane without changing selection.

        Args:
            view_name: Stable provider or model-library view key to populate.
        """

        if (
            view_name in self._populated_views
            or view_name in self._populating_views
            or view_name not in self.view_mapping
        ):
            return
        self._populating_views.add(view_name)
        self.run_worker(
            self._activate_deferred_view(view_name),
            group=f"llm-view-mount-{view_name}",
            exclusive=True,
            exit_on_error=False,
        )

    async def _activate_deferred_view(self, view_name: str) -> None:
        """Mount a first-selected pane and finish activation if still visible."""

        try:
            await self._mount_deferred_views(view_name)
        except Exception:
            logger.exception("Lazy LLM view mount failed: {}", view_name)
            return
        finally:
            self._populating_views.discard(view_name)
        if self.active_view != view_name:
            return
        self.call_after_refresh(self._finish_deferred_view_activation, view_name)

    def _finish_deferred_view_activation(self, view_name: str) -> None:
        """Finish one lazy activation after its descendants have composed."""

        if self.active_view != view_name:
            return
        try:
            target = self.query_one(f"#{self.view_mapping[view_name]}")
        except QueryError:
            return
        self._finish_view_activation(view_name, target)

    def _finish_view_activation(self, view_name: str, target_view: Widget) -> None:
        """Run behavior that requires the selected pane body to exist."""

        gguf_provider = "llamacpp" if view_name == "llama-cpp" else view_name
        if gguf_provider in self.GGUF_PROVIDERS:
            self._render_gguf_source(gguf_provider)
            if self._managed_gguf_inventory_started:
                self._apply_managed_gguf_inventory(
                    self._managed_gguf_inventory_generation,
                    self._managed_gguf_choices,
                    self._managed_gguf_inventory_error,
                )
        provider = "mlx" if view_name == "mlx-lm" else view_name.replace("-cpp", "cpp")
        if provider in self.SERVER_CONTROLS:
            self._sync_process_controls(provider)
        self._populate_help_text(view_name, target_view)
        if view_name in self._model_library_focus_ids:
            self.call_after_refresh(self._restore_model_library_focus, view_name)
        self._start_view_work(view_name, target_view)

    def _record_model_library_focus(self, focused: Widget | None) -> None:
        """Retain stable row focus whenever the screen's reactive focus changes."""

        if focused is None:
            return
        for view_name in self._model_library_focus_ids:
            if view_name != self.active_view:
                continue
            view_id = self.view_mapping[view_name]
            try:
                pane = self.query_one(f"#{view_id}")
            except QueryError:
                continue
            if pane not in focused.ancestors_with_self:
                continue
            try:
                library = pane.query_one(
                    f"#{self._model_library_widget_ids[view_name]}"
                )
            except QueryError:
                return
            locator = library.focus_locator(focused)
            if locator is not None:
                self._model_library_focus_ids[view_name] = locator
            return

    def _restore_model_library_focus(self, view_name: str) -> None:
        """Move focus into the visible model-library pane after switching."""

        if self.active_view != view_name:
            return
        target_id = self.view_mapping[view_name]
        try:
            target = self.query_one(f"#{target_id}")
        except QueryError:
            return
        locator = self._model_library_focus_ids[view_name]
        try:
            library = target.query_one(f"#{self._model_library_widget_ids[view_name]}")
        except QueryError:
            library = None
        if library is not None and callable(getattr(library, "restore_focus", None)):
            library.restore_focus(locator)
            if (
                self.app.focused is not None
                and target in self.app.focused.ancestors_with_self
            ):
                return
        control = next(
            (button for button in target.query(Button) if not button.disabled),
            None,
        )
        if control is None or control.disabled:
            return
        control.focus()
        control.scroll_visible(animate=False, immediate=True, force=True)

    def _start_view_work(self, view_name: str, view_widget) -> None:
        """Kick off work a view should only do once it is actually shown."""
        if view_name in {"llama-cpp", "llamafile"}:
            self._ensure_managed_gguf_inventory()
            return
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
            already_loaded = bool(getattr(managed_view, "_loaded", False))
            managed_view.ensure_loaded()
            if already_loaded:
                # Restore the pane's semantic row focus first so the observer
                # can retain that exact locator across its evidence recompose.
                self.call_after_refresh(self.refresh_model_library_observations)
            return

    def refresh_model_library_observations(self) -> None:
        """Refresh evidence only for the currently visible loaded library pane."""

        view_name = self.active_view
        if view_name not in {"curated", "installed"}:
            return
        try:
            managed_view = self.query_one(
                "#curated-models-view"
                if view_name == "curated"
                else "#installed-models-view"
            )
        except QueryError:
            return
        if getattr(managed_view, "_loaded", False):
            managed_view.refresh_observations()

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
