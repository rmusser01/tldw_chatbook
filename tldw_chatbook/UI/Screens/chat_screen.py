"""Chat screen implementation with comprehensive state management."""

import inspect
import re
from typing import TYPE_CHECKING, Dict, Any, Optional
from datetime import datetime
import uuid
from loguru import logger
import toml
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static, TextArea, Select, Collapsible, Input
from textual.events import Key
from textual import on, work
from textual.reactive import reactive
from textual.css.query import NoMatches, QueryError

from ..Navigation.base_app_screen import BaseAppScreen
from .chat_screen_state import ChatScreenState, TabState, MessageData, TaskResumeState
from ...Chat.chat_conversation_service import derive_conversation_title
from ...Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON,
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
    ConsoleControlState,
    ConsoleInspectorState,
    ConsoleStagedContextState,
    coerce_non_negative_int,
)
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Chat.chat_models import ChatSessionData
from ...Chat.console_live_work import (
    ConsoleLiveWorkLaunch,
    ConsoleLiveWorkSourceReadinessState,
    ConsoleLiveWorkStatusCardState,
)
from ...Library.library_rag_service import (
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from ...Utils.chat_diagnostics import ChatDiagnostics
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...state.ui_state import UIState
from ...Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from ...Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from ...Widgets.Console import (
    ConsoleComposerBar,
    ConsoleControlBar,
    ConsoleRunInspector,
    ConsoleSessionSurface,
    ConsoleStagedContextTray,
)
from ...Widgets.compact_model_bar import CompactModelBar
from ..Views.RAGSearch.search_handoff import build_library_rag_console_live_work_payload

# Import the existing chat window to reuse its functionality
from ..Chat_Window_Enhanced import ChatWindowEnhanced
from ...Widgets.voice_input_widget import VoiceInputMessage

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="ChatScreen")
CONSOLE_LIBRARY_RAG_SOURCE_SCOPE = ("notes", "media", "conversations")
CONSOLE_LIBRARY_RAG_RECOVERY_COPY = "Review citations before sending."
CONSOLE_LIBRARY_RAG_QUERY_MAX_LENGTH = 2_000
CONSOLE_LIBRARY_RAG_QUERY_EMPTY_TOOLTIP = "Type a Library RAG query before running retrieval."
CONSOLE_LIBRARY_RAG_QUERY_INVALID_TOOLTIP = (
    "Enter a valid Library RAG query without scripts or unsafe markup."
)


def _is_empty_select_value(value: Any) -> bool:
    """Return True for Textual's blank/null select sentinels."""
    return value is None or value == Select.BLANK or str(value).startswith("Select.")


def _derive_tab_title(tab_state: TabState) -> str:
    assistant_name = None
    if tab_state.assistant_kind == "character":
        assistant_name = tab_state.character_name
    elif tab_state.assistant_kind == "persona" and tab_state.assistant_id:
        assistant_name = f"Persona {tab_state.assistant_id}"

    return derive_conversation_title(
        assistant_kind=tab_state.assistant_kind,
        assistant_name=assistant_name,
        fallback_title=tab_state.title,
        character_id=tab_state.character_id,
    )


def _source_mentions_rag(source: Any) -> bool:
    """Return whether a source label explicitly includes a RAG token.

    Args:
        source: Source label or source-like seam value.

    Returns:
        True when the normalized source tokens include `rag`.
    """
    tokens = re.split(r"[^a-z0-9]+", str(source or "").lower())
    return "rag" in tokens


def _sanitize_console_library_rag_query(value: Any) -> str:
    """Return a centralized-validation-safe Console Library RAG query."""
    sanitized = sanitize_string(
        str(value or ""),
        max_length=CONSOLE_LIBRARY_RAG_QUERY_MAX_LENGTH,
    )
    query = " ".join(sanitized.strip().split())
    if not query:
        return ""
    if not validate_text_input(
        query,
        max_length=CONSOLE_LIBRARY_RAG_QUERY_MAX_LENGTH,
        allow_html=False,
    ):
        return ""
    return query


def _has_selected_text(value: Any) -> bool:
    """Return whether a provider/model value is meaningfully selected.

    Args:
        value: Value from Textual select state or app/default configuration.

    Returns:
        True when the value is not an empty Textual select sentinel and has
        non-whitespace text.
    """
    return not _is_empty_select_value(value) and bool(str(value).strip())


class ChatScreen(BaseAppScreen):
    """
    Chat screen with comprehensive state management.
    
    This screen preserves all chat state including tabs, messages,
    input text, and UI preferences when navigating away and returning.
    """
    
    @on(Select.Changed, "#chat-api-provider")
    async def handle_provider_change(self, event: Select.Changed) -> None:
        """Handle API provider change and update model dropdown + compact bar."""
        logger.info(f"API provider changed to: {event.value}")

        try:
            from tldw_chatbook.config import get_cli_providers_and_models

            # Get the new provider's models
            providers_models = get_cli_providers_and_models()
            new_provider = str(event.value)
            available_models = providers_models.get(new_provider, [])
            logger.info(f"Found {len(available_models)} models for provider {new_provider}")

            # Find the model select widget within the chat window
            if self.chat_window:
                try:
                    model_select = self.chat_window.query_one("#chat-api-model", Select)

                    # Update options
                    new_model_options = [(model, model) for model in available_models]
                    model_select.set_options(new_model_options)

                    # Set to first model or blank if no models
                    if available_models:
                        model_select.value = available_models[0]
                        logger.info(f"Set model to: {available_models[0]}")
                    else:
                        model_select.value = Select.BLANK
                        logger.info("No models available, set to BLANK")

                    model_select.prompt = "Select Model..." if available_models else "No models available"
                    selected_model = None if _is_empty_select_value(model_select.value) else str(model_select.value)
                    self._sync_compact_shell_controls(
                        provider=new_provider,
                        model=selected_model,
                    )
                    logger.info(f"Successfully updated model dropdown with {len(available_models)} models")
                except Exception as e:
                    logger.error(f"Could not find model select widget: {e}")

                # Sync to compact model bar
                try:
                    from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar
                    compact_bar = self.chat_window.query_one("#compact-model-bar", CompactModelBar)
                    compact_bar.sync_from_sidebar(provider=new_provider)
                except Exception:
                    logger.debug("Compact bar not found for provider sync")
                self.chat_window.refresh_first_run_orientation(new_provider)
            else:
                logger.error("chat_window is None")

        except Exception as e:
            logger.error(f"Error updating model dropdown: {e}", exc_info=True)

    @on(Select.Changed, "#chat-api-model")
    def on_chat_api_model_changed(self, event: Select.Changed) -> None:
        """Mirror sidebar model changes into the compact shell controls."""
        model = None if _is_empty_select_value(event.value) else str(event.value)
        self._sync_compact_shell_controls(model=model)

    @on(Input.Changed, "#chat-temperature")
    def on_chat_temperature_changed(self, event: Input.Changed) -> None:
        """Mirror sidebar temperature changes into the compact shell controls."""
        self._sync_compact_shell_controls(temperature=event.value)

    @on(Select.Changed, "#compact-api-provider")
    def on_console_compact_provider_changed(self, event: Select.Changed) -> None:
        """Mirror native compact provider changes into Console-owned labels."""
        if not _is_empty_select_value(event.value):
            self._console_control_provider = str(event.value)
        self._sync_console_control_bar()

    @on(Select.Changed, "#compact-api-model")
    def on_console_compact_model_changed(self, event: Select.Changed) -> None:
        """Mirror native compact model changes into Console-owned labels."""
        if not _is_empty_select_value(event.value):
            self._console_control_model = str(event.value)
        self._sync_console_control_bar()
    
    
    # Reactive property for sidebar state persistence
    sidebar_state = reactive({}, layout=False)
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "chat", **kwargs)
        self.chat_window: Optional[ChatWindowEnhanced] = None
        self.console_session_surface: Optional[ConsoleSessionSurface] = None
        self.chat_state = ChatScreenState()
        self._state_dirty = False
        self._diagnostics_run = False
        self._handoff_consumption_in_progress = False
        self._pending_console_launch_context: Optional[ConsoleLiveWorkLaunch] = None
        self._console_control_provider: Optional[Any] = None
        self._console_control_model: Optional[Any] = None
        self._console_library_rag_query = ""
        self.ui_state = UIState()
        self._load_sidebar_state()

    def _ensure_chat_window(self) -> ChatWindowEnhanced:
        if self.chat_window is None:
            self.chat_window = ChatWindowEnhanced(
                self.app_instance,
                show_shell_compact_controls=False,
                id="chat-window",
                classes="window",
            )
        return self.chat_window

    def _ensure_console_session_surface(self) -> ConsoleSessionSurface:
        if self.console_session_surface is None:
            self.console_session_surface = ConsoleSessionSurface(
                self.app_instance,
                id="console-session-surface",
                classes="console-region",
            )
        return self.console_session_surface

    def _consume_pending_console_launch(self) -> Optional[ConsoleLiveWorkLaunch]:
        """Accept one-shot live-work launch context from another destination."""
        if self._pending_console_launch_context is not None:
            return self._pending_console_launch_context

        pending_launch = getattr(self.app_instance, "pending_console_launch", None)
        if (normalized_launch := ConsoleLiveWorkLaunch.from_pending(pending_launch)) is not None:
            self._pending_console_launch_context = normalized_launch
            self.app_instance.pending_console_launch = None
        return self._pending_console_launch_context

    def _chat_default_value(self, key: str) -> Any:
        config = getattr(self.app_instance, "app_config", {}) or {}
        defaults = config.get("chat_defaults", {})
        if isinstance(defaults, dict):
            return defaults.get(key)
        return None

    def _effective_console_provider_model(self) -> tuple[Any, Any]:
        """Return the canonical Console provider/model selection.

        Returns:
            A `(provider, model)` tuple using the same precedence for Console
            control labels and run-inspector readiness.
        """
        provider = (
            self._console_control_provider
            or getattr(self.app_instance, "chat_api_provider_value", None)
            or self._chat_default_value("provider")
        )
        model = (
            self._console_control_model
            or getattr(self.app_instance, "chat_api_model_value", None)
            or getattr(self.app_instance, "chat_model_value", None)
            or self._chat_default_value("model")
        )
        return provider, model

    def _build_console_control_state(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleControlState:
        """Build Console-owned control/readiness labels."""
        provider, model = self._effective_console_provider_model()
        source = pending_launch.source if pending_launch else None
        return ConsoleControlState.from_values(
            provider=provider,
            model=model,
            persona=None,
            rag_enabled=_source_mentions_rag(source),
            staged_source_count=1 if pending_launch else 0,
            tool_count=self._console_tool_count(),
            approval_count=self._console_pending_approval_count(),
        )

    def _build_console_staged_context_state(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleStagedContextState:
        if pending_launch is None:
            return ConsoleStagedContextState.empty()
        return ConsoleStagedContextState.from_live_work(pending_launch)

    @staticmethod
    def _launch_targets_chatbook_artifact(
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> bool:
        if pending_launch is None:
            return False
        source = str(pending_launch.source or "").strip().lower()
        target_id = str(pending_launch.payload.get("target_id") or "").strip()
        return source in {"artifacts", "chatbooks"} and ":chatbook:" in target_id

    @staticmethod
    def _launch_has_rag_source_payload(pending_launch: ConsoleLiveWorkLaunch) -> bool:
        source_keys = (
            "source_id",
            "source_count",
            "citation_count",
            "chunk_id",
            "query",
            "result_id",
        )
        return any(str(pending_launch.payload.get(key) or "").strip() for key in source_keys)

    def _console_pending_approval_count(self) -> int:
        explicit_count = getattr(self.app_instance, "console_pending_approval_count", None)
        if explicit_count is not None:
            return coerce_non_negative_int(explicit_count)

        pending_approval = getattr(self.app_instance, "pending_console_approval", None)
        if pending_approval:
            return 1

        task_state = self.chat_state.task_resume_state
        return 1 if task_state.has_pending_approval() else 0

    def _console_tool_count(self) -> int:
        return coerce_non_negative_int(getattr(self.app_instance, "console_tool_count", 0))

    def _console_rag_source_status(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> str:
        if pending_launch is None:
            return "not staged"
        if _source_mentions_rag(pending_launch.source):
            launch_status = str(pending_launch.status or "").strip().lower()
            if launch_status in {"blocked", "failed", "unavailable"}:
                return "unavailable"
            if launch_status == "empty":
                return "no results"
            if launch_status == "searching":
                return "retrieving from Library Search/RAG"
            if self._launch_has_rag_source_payload(pending_launch):
                return "staged from Library Search/RAG"
            return "missing source"
        return "not requested"

    def _console_artifact_status(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
        *,
        can_save_chatbook: bool,
    ) -> str:
        if can_save_chatbook:
            return "Chatbook artifact available"
        if pending_launch is not None:
            return "not available for this item"
        return "unavailable"

    def _build_console_inspector_state(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleInspectorState:
        provider, model = self._effective_console_provider_model()
        explicit_provider_ready = getattr(self.app_instance, "console_provider_ready", None)
        provider_ready = (
            bool(explicit_provider_ready)
            if explicit_provider_ready is not None
            else _has_selected_text(provider) and _has_selected_text(model)
        )
        can_save_chatbook = bool(
            getattr(self.app_instance, "console_chatbook_artifact_available", False)
            or self._launch_targets_chatbook_artifact(pending_launch)
        )
        return ConsoleInspectorState.from_values(
            live_work_title=pending_launch.title if pending_launch else None,
            provider_ready=provider_ready,
            provider_recovery=(
                "" if provider_ready else "Select a provider and model before sending."
            ),
            rag_status=self._console_rag_source_status(pending_launch),
            artifact_status=self._console_artifact_status(
                pending_launch,
                can_save_chatbook=can_save_chatbook,
            ),
            tool_count=self._console_tool_count(),
            approval_count=self._console_pending_approval_count(),
            can_save_chatbook=can_save_chatbook,
        )

    def _toggle_console_chat_sidebar(self) -> None:
        """Route Console-level compact control toggles to the embedded chat sidebar."""
        if self.chat_window and hasattr(self.chat_window, "handle_shell_sidebar_toggle_requested"):
            self.chat_window.handle_shell_sidebar_toggle_requested()
            return
        self.app_instance.notify(
            "Chat settings are still loading.",
            severity="warning",
        )

    def _render_console_live_work_status_card(self, launch: ConsoleLiveWorkLaunch) -> ComposeResult:
        """Render a reusable live-work status card for Console launch context."""
        card_state = ConsoleLiveWorkStatusCardState.from_launch(launch)
        with Container(id=card_state.container_id, classes=card_state.container_classes):
            yield Static(
                card_state.badge_text,
                id=card_state.badge_id,
                classes=card_state.badge_classes,
            )
            if card_state.primary_action is not None:
                yield Button(
                    card_state.primary_action.label,
                    id=card_state.primary_action.widget_id,
                    classes=card_state.primary_action.classes,
                    variant="primary",
                )
            for row in card_state.rows:
                yield Static(row.text, id=row.widget_id, classes=row.classes)

    def _console_library_rag_scope_label(self) -> str:
        return f"Library scope: {', '.join(CONSOLE_LIBRARY_RAG_SOURCE_SCOPE)}"

    def _render_console_live_work_source_readiness(self) -> ComposeResult:
        """Render Console source readiness when no live-work item is staged."""
        readiness = ConsoleLiveWorkSourceReadinessState.default()
        with Container(id=readiness.container_id, classes=readiness.container_classes):
            yield Static(
                readiness.title,
                id=readiness.title_id,
                classes=readiness.title_classes,
            )
            yield Static(
                self._console_library_rag_scope_label(),
                id="console-library-rag-scope",
                classes="destination-section console-library-rag-scope",
            )
            yield Input(
                value=self._console_library_rag_query,
                placeholder="Ask Library sources before sending",
                id="console-library-rag-query-input",
            )
            query_ready = bool(
                _sanitize_console_library_rag_query(self._console_library_rag_query)
            )
            yield Button(
                "Run Library RAG",
                id="console-run-library-rag",
                disabled=not query_ready,
                tooltip="" if query_ready else CONSOLE_LIBRARY_RAG_QUERY_EMPTY_TOOLTIP,
                classes="destination-action-button console-library-rag-run",
            )
            for row in readiness.rows:
                yield Static(row.text, id=row.widget_id, classes=row.classes)

    @on(Button.Pressed, "#console-live-work-primary-action")
    def handle_console_live_work_primary_action(self, event: Button.Pressed) -> None:
        """Route supported live-work card actions through the app-owned shell."""
        event.stop()
        launch = self._consume_pending_console_launch()
        handler = getattr(self.app_instance, "open_console_live_work_primary_action", None)
        if launch is not None and callable(handler):
            handled = bool(handler(launch))
            if handled:
                return
        self.app_instance.notify(
            "Console action is unavailable for this live-work item.",
            severity="warning",
        )

    @on(Input.Changed, "#console-library-rag-query-input")
    def update_console_library_rag_query(self, event: Input.Changed) -> None:
        """Track the Console-side Library RAG query and refresh the run action."""
        event.stop()
        raw_query = str(event.value or "")
        self._console_library_rag_query = _sanitize_console_library_rag_query(raw_query)
        try:
            run_button = self.query_one("#console-run-library-rag", Button)
        except QueryError:
            return
        query_ready = bool(self._console_library_rag_query)
        run_button.disabled = not query_ready
        invalid_query = bool(raw_query.strip()) and not query_ready
        disabled_tooltip = (
            CONSOLE_LIBRARY_RAG_QUERY_INVALID_TOOLTIP
            if invalid_query
            else CONSOLE_LIBRARY_RAG_QUERY_EMPTY_TOOLTIP
        )
        run_button.tooltip = "" if query_ready else disabled_tooltip

    @on(Button.Pressed, "#console-run-library-rag")
    def handle_console_run_library_rag(self, event: Button.Pressed) -> None:
        """Request Library retrieval from the Console source-readiness seam."""
        event.stop()
        query = _sanitize_console_library_rag_query(self._console_library_rag_query)
        if not query:
            self.app_instance.notify(
                CONSOLE_LIBRARY_RAG_QUERY_EMPTY_TOOLTIP,
                severity="warning",
            )
            return
        request = LibraryRagSearchRequest(
            query=query,
            source_types=CONSOLE_LIBRARY_RAG_SOURCE_SCOPE,
            mode="rag",
            top_k=5,
            include_citations=True,
        )
        self._stage_console_library_rag_launch(
            ConsoleLiveWorkLaunch.from_values(
                source="Library Search/RAG",
                title="Library Search/RAG retrieval",
                payload={
                    "query": request.query,
                    "source_scope": ", ".join(request.source_types),
                },
                status="searching",
                recovery="Retrieving Library Search/RAG evidence.",
                action_label="Review evidence in Console",
            )
        )
        self._execute_console_library_rag_search(request)

    def _stage_console_library_rag_launch(self, launch: ConsoleLiveWorkLaunch) -> None:
        self._pending_console_launch_context = launch
        self.refresh(recompose=True)

    @work(exclusive=True)
    async def _execute_console_library_rag_search(self, request: LibraryRagSearchRequest) -> None:
        outcome = await run_library_rag_search(self.app_instance, request)
        await self._apply_console_library_rag_search_outcome(request, outcome)

    async def _apply_console_library_rag_search_outcome(
        self,
        request: LibraryRagSearchRequest,
        outcome: Any,
    ) -> None:
        if not self.is_mounted:
            return
        if outcome.results:
            result = outcome.results[0]
            self._stage_console_library_rag_launch(
                ConsoleLiveWorkLaunch.from_values(
                    source="Library Search/RAG",
                    title=result.title,
                    payload=build_library_rag_console_live_work_payload(
                        result,
                        query=request.query,
                    ),
                    status="staged",
                    recovery=CONSOLE_LIBRARY_RAG_RECOVERY_COPY,
                    action_label="Review evidence in Console",
                )
            )
            return

        recovery_state = outcome.recovery_state
        recovery_copy = (
            recovery_state.visible_copy
            if recovery_state is not None
            else "Library Search/RAG did not return usable evidence."
        )
        self._stage_console_library_rag_launch(
            ConsoleLiveWorkLaunch.from_values(
                source="Library Search/RAG",
                title="Library Search/RAG retrieval",
                payload={
                    "query": request.query,
                    "source_scope": ", ".join(request.source_types),
                },
                status=outcome.status or "blocked",
                recovery=recovery_copy,
                action_label="Resolve Library RAG setup",
            )
        )
        
    def compose_content(self) -> ComposeResult:
        """Compose the chat content."""
        pending_launch = self._consume_pending_console_launch()
        control_state = self._build_console_control_state(pending_launch)
        staged_context_state = self._build_console_staged_context_state(pending_launch)
        inspector_state = self._build_console_inspector_state(pending_launch)
        with Vertical(id="console-shell"):
            yield Static("Console", id="console-title", classes="ds-destination-header")
            yield Static(
                "Agent workbench for chat, source handoffs, live runs, and control actions.",
                id="console-purpose",
                classes="destination-purpose",
            )
            yield Static(
                "Console | Agentic control surface | Chat-first | Local runtime",
                id="console-status-row",
                classes="destination-status-row",
            )
            yield Static(
                "Mode: Chat + agent control | Context: staged sources | Runs: live work",
                id="console-mode-bar",
                classes="ds-panel",
            )
            yield ConsoleControlBar(
                control_state,
                self.app_instance,
                on_sidebar_toggle_requested=self._toggle_console_chat_sidebar,
                id="console-control-bar",
                classes="ds-panel",
            )
            with Horizontal(id="console-workspace-grid", classes="ds-panel destination-workbench"):
                yield ConsoleStagedContextTray(
                    staged_context_state,
                    id="console-staged-context-tray",
                    classes="console-region destination-workbench-pane",
                )
                with Vertical(id="console-main-column"):
                    with Vertical(id="console-transcript-region", classes="console-region"):
                        yield self._ensure_console_session_surface()
                    yield ConsoleComposerBar(id="console-native-composer", classes="console-region ds-panel")
                with Vertical(
                    id="console-run-inspector",
                    classes="console-region destination-workbench-pane",
                ):
                    yield ConsoleRunInspector(
                        inspector_state,
                        id="console-run-inspector-state",
                    )
                    if pending_launch:
                        yield from self._render_console_live_work_status_card(pending_launch)
                    else:
                        yield from self._render_console_live_work_source_readiness()
    
    def on_mount(self) -> None:
        """Run diagnostics when first mounted (only once)."""
        # Call parent's on_mount
        super().on_mount()
        
        if not self._diagnostics_run and self.chat_window:
            self._diagnostics_run = True
            # Run diagnostic in the background for the legacy direct widget only.
            self.set_timer(0.5, self._run_diagnostic)
        
        # Restore collapsible states after mount
        self.set_timer(0.1, self._restore_collapsible_states)
        self.set_timer(0.05, self.sync_task_resume_state)
        self.set_timer(0.15, self._consume_pending_chat_handoff)
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save comprehensive chat state.
        
        Captures all tabs, messages, input text, and UI state
        to fully restore the chat experience on return.
        """
        logger.debug("Saving chat screen state")
        state = super().save_state()
        
        try:
            # Create fresh state object
            self.chat_state = ChatScreenState()
            self.chat_state.last_saved = datetime.now()

            # Save UI preferences even when Console no longer mounts the legacy chat window.
            self.chat_state.left_sidebar_collapsed = getattr(
                self.app_instance, 'chat_sidebar_collapsed', False
            )
            self.chat_state.right_sidebar_collapsed = getattr(
                self.app_instance, 'chat_right_sidebar_collapsed', False
            )

            # Try to detect and save from different chat interface types.
            tab_container = self._get_tab_container()
            
            if tab_container and hasattr(tab_container, 'sessions'):
                # Tabbed interface detected
                logger.debug(f"Detected tabbed interface with {len(tab_container.sessions)} tabs")

                # Save all tab sessions
                self._save_tab_sessions(tab_container)

                # Save active tab
                self.chat_state.active_tab_id = tab_container.active_session_id

                # Save tab order
                if hasattr(tab_container, 'tab_bar') and tab_container.tab_bar:
                    self.chat_state.tab_order = tab_container.tab_bar.get_tab_ids()

                # Also save messages for the active session
                if tab_container.active_session_id:
                    active_tab = self.chat_state.get_tab_by_id(tab_container.active_session_id)
                    if active_tab:
                        self._extract_and_save_messages(active_tab)
            elif self.chat_window:
                # Non-tabbed legacy direct widget - try to save single chat state
                logger.debug("Detected non-tabbed chat interface")
                self._save_non_tabbed_state()

            # Always try to save current input text directly
            self._save_direct_input_text()

            # Save sidebar settings (system prompt, temperature, etc.) when available.
            self._save_sidebar_settings()

            # Save scroll positions
            self._save_scroll_positions()

            # Save pending attachments
            self._save_attachments()
            
            # Convert to dict for storage
            state['chat_state'] = self.chat_state.to_dict()
            state['state_version'] = '1.0'
            state['interface_type'] = 'tabbed' if self.chat_state.tabs else 'single'
            
            logger.info(f"Saved chat state: {len(self.chat_state.tabs)} tabs, interface: {state.get('interface_type')}")
            
        except Exception as e:
            logger.error(f"Error saving chat state: {e}", exc_info=True)
        
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore comprehensive chat state.
        
        Recreates all tabs, messages, and UI state from saved data.
        """
        logger.debug("Restoring chat screen state")
        super().restore_state(state)
        
        try:
            if 'chat_state' in state:
                # Restore from saved state
                self.chat_state = ChatScreenState.from_dict(state['chat_state'])
                
                logger.debug(f"Restored state has {len(self.chat_state.tabs)} tabs")
                logger.debug(f"Active tab ID: {self.chat_state.active_tab_id}")
                logger.debug(f"Tab order: {self.chat_state.tab_order}")
                
                if self.chat_state.validate():
                    logger.info(f"Restoring {len(self.chat_state.tabs)} tabs")
                    
                    # Schedule restoration after mount
                    self.set_timer(0.1, self._perform_state_restoration)
                else:
                    logger.warning("Chat state validation failed, starting fresh")
                    self.chat_state = ChatScreenState()
            
        except Exception as e:
            logger.error(f"Error restoring chat state: {e}", exc_info=True)
            self.chat_state = ChatScreenState()
    
    async def _perform_state_restoration(self) -> None:
        """Perform actual state restoration after UI is ready."""
        if not self.chat_window and not self._get_tab_container():
            logger.warning("Console chat surface not ready for restoration")
            # Try again in a moment
            self.set_timer(0.2, self._perform_state_restoration)
            return
        
        try:
            logger.info("Starting state restoration...")
            
            # Restore UI preferences
            self.app_instance.chat_sidebar_collapsed = self.chat_state.left_sidebar_collapsed
            self.app_instance.chat_right_sidebar_collapsed = self.chat_state.right_sidebar_collapsed
            
            # Get tab container
            tab_container = self._get_tab_container()
            if tab_container:
                # Tabbed interface - restore tab sessions
                await self._restore_tab_sessions(tab_container)
                
                # Restore active tab
                if self.chat_state.active_tab_id:
                    await tab_container.switch_to_tab_async(self.chat_state.active_tab_id)
            else:
                # Non-tabbed interface - still need to restore state
                logger.debug("Non-tabbed interface detected, restoring state directly")
            
            # Always restore these regardless of tab container
            # Restore input text
            await self._restore_input_text()
            
            # Restore sidebar settings (system prompt, temperature, etc.)
            await self._restore_sidebar_settings()
            
            # Restore scroll positions
            await self._restore_scroll_positions()
            
            # Restore attachments
            await self._restore_attachments()
            
            # Restore conversation messages
            await self._restore_messages()
            self.sync_task_resume_state()
            await self._consume_pending_chat_handoff()
            
            logger.info("Chat state restoration complete")
            
        except Exception as e:
            logger.error(f"Error during state restoration: {e}", exc_info=True)
    
    def _get_tab_container(self):
        """Get the ChatTabContainer widget."""
        try:
            return self.query_one("#console-chat-tabs", ChatTabContainer)
        except NoMatches:
            pass

        try:
            if self.chat_window is not None:
                tab_container = getattr(self.chat_window, "_tab_container", None)
                if tab_container is not None:
                    return tab_container
            if isinstance(self.chat_window, ChatWindowEnhanced):
                return self.chat_window.query_one("ChatTabContainer")
        except NoMatches:
            return None
        return None

    def _get_active_chat_session(self):
        """Return the active native/legacy chat session widget when available."""
        tab_container = self._get_tab_container()
        if not tab_container:
            return None
        get_active_session = getattr(tab_container, "get_active_session", None)
        if callable(get_active_session):
            return get_active_session()
        active_session_id = getattr(tab_container, "active_session_id", None)
        sessions = getattr(tab_container, "sessions", {})
        if active_session_id and active_session_id in sessions:
            return sessions[active_session_id]
        return None

    def _chat_query_scope(self):
        """Prefer the legacy chat window when present, else the native Console tree."""
        return self.chat_window or self

    def _get_active_chat_input(self) -> Optional[TextArea]:
        """Return the active session input before falling back to legacy single-chat input."""
        session = self._get_active_chat_session()
        if session is not None:
            get_chat_input = getattr(session, "get_chat_input", None)
            if callable(get_chat_input):
                try:
                    chat_input = get_chat_input()
                    if chat_input is not None:
                        return chat_input
                except QueryError:
                    logger.debug("Active session chat input was not mounted")

        try:
            return self._chat_query_scope().query_one("#chat-input", TextArea)
        except QueryError:
            return None

    def _get_active_chat_log(self):
        """Return the active session chat log before falling back to legacy chat logs."""
        session = self._get_active_chat_session()
        if session is not None:
            get_chat_log = getattr(session, "get_chat_log", None)
            if callable(get_chat_log):
                try:
                    chat_log = get_chat_log()
                    if chat_log is not None:
                        return chat_log
                except QueryError:
                    logger.debug("Active session chat log was not mounted")

            session_data = getattr(session, "session_data", None)
            tab_id = getattr(session_data, "tab_id", None)
            if tab_id:
                try:
                    return session.query_one(f"#chat-log-{tab_id}", VerticalScroll)
                except QueryError:
                    logger.debug(f"Active session chat log for tab {tab_id} was not mounted")

        return None

    @staticmethod
    def _first_query_result(containers: Any) -> Any:
        if hasattr(containers, "first"):
            return containers.first()
        return containers[0]

    def _find_chat_log_container(self, selectors: list[str]):
        """Find the correct chat log, preferring the active tab over DOM order."""
        active_log = self._get_active_chat_log()
        if active_log is not None:
            return active_log

        try:
            return self.app_instance.query_one("#chat-log", VerticalScroll)
        except (LookupError, QueryError):
            pass

        for selector in selectors:
            try:
                containers = self._chat_query_scope().query(selector)
            except QueryError as e:
                logger.debug(f"Could not find chat log with {selector}: {e}")
                continue
            if containers:
                logger.debug(f"Found chat log container with selector: {selector}")
                return self._first_query_result(containers)

        return None

    def _session_data_for_handoff(self, payload: ChatHandoffPayload) -> ChatSessionData:
        title_item_type = payload.item_type.replace("-", " ").title()
        scope_type = payload.scope_type or "global"
        return ChatSessionData(
            tab_id=uuid.uuid4().hex[:8],
            title=f"{title_item_type}: {payload.title}",
            conversation_id=None,
            is_ephemeral=True,
            runtime_backend=payload.runtime_backend,
            discovery_owner=payload.discovery_owner,
            discovery_entity_id=payload.discovery_entity_id or payload.source_id,
            scope_type=scope_type,
            workspace_id=payload.workspace_id if scope_type == "workspace" else None,
            handoff_payload=payload,
        )

    async def _consume_pending_chat_handoff(self) -> None:
        payload = getattr(self.app_instance, "pending_chat_handoff", None)
        if payload is None:
            return

        if self._handoff_consumption_in_progress:
            return

        self._handoff_consumption_in_progress = True
        try:
            payload = ChatHandoffPayload.from_dict(payload)
            if payload is None:
                return

            tab_container = self._get_tab_container()
            if tab_container is None:
                self.app_instance.notify(
                    "Chat tabs are not available for Use in Chat.",
                    severity="warning",
                )
                return

            session_data = self._session_data_for_handoff(payload)
            tab_id = await tab_container.create_new_tab(session_data=session_data)
            if not tab_id:
                self.app_instance.notify(
                    "Could not create a chat session for this context.",
                    severity="error",
                )
                return

            await tab_container.switch_to_tab_async(tab_id)
            session = tab_container.sessions.get(tab_id)
            if session is not None:
                await self._apply_handoff_to_chat_session(session, payload)
            self.app_instance.pending_chat_handoff = None
        finally:
            self._handoff_consumption_in_progress = False

    async def _apply_handoff_to_chat_session(self, session: Any, payload: ChatHandoffPayload) -> None:
        mount_handoff_card = getattr(session, "mount_handoff_card", None)
        if callable(mount_handoff_card):
            result = mount_handoff_card(payload)
            if inspect.isawaitable(result):
                await result

        set_draft_text = getattr(session, "set_draft_text", None)
        if callable(set_draft_text):
            set_draft_text(payload.default_prompt())

    @on(Button.Pressed, "#console-send-message")
    async def handle_console_send_message(self, event: Button.Pressed) -> None:
        """Route the Console composer send action through the active chat session."""
        event.stop()
        session = self._get_active_chat_session()
        if session is None:
            self.app_instance.notify("No active Console chat session is available.", severity="error")
            return
        handler = getattr(session, "handle_send_stop_button", None)
        if not callable(handler):
            self.app_instance.notify("Console send is unavailable for this session.", severity="error")
            return
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    @on(Button.Pressed, "#console-stop-generation")
    async def handle_console_stop_generation(self, event: Button.Pressed) -> None:
        """Route the Console stop action through tab-aware chat stop handling."""
        event.stop()
        session = self._get_active_chat_session()
        if session is None:
            self.app_instance.notify("No active Console chat session is available.", severity="error")
            return
        from tldw_chatbook.Event_Handlers.Chat_Events import chat_events_tabs

        await chat_events_tabs.handle_stop_chat_generation_pressed_with_tabs(
            self.app_instance,
            event,
            session.session_data,
        )
        update_button = getattr(session, "_update_button_state", None)
        if callable(update_button):
            update_button()

    @on(Button.Pressed, "#console-attach-context")
    async def handle_console_attach_context(self, event: Button.Pressed) -> None:
        """Route the Console attach affordance through the active chat session adapter."""
        event.stop()
        session = self._get_active_chat_session()
        if session is None:
            self.app_instance.notify("No active Console chat session is available.", severity="error")
            return
        handler = getattr(session, "handle_attach_button", None)
        if not callable(handler):
            self.app_instance.notify("Console attachment is unavailable for this session.", severity="warning")
            return
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    @on(Button.Pressed, "#console-save-chatbook")
    def handle_console_save_chatbook(self, event: Button.Pressed) -> None:
        """Expose the current Chatbook save seam without inventing export behavior."""
        event.stop()
        self.app_instance.notify(
            "Save Chatbook is still owned by Artifacts/Chatbooks; this Console button is a compatibility adapter.",
            severity="information",
        )

    @on(Button.Pressed, f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}")
    def handle_console_inspector_review_approval(self, event: Button.Pressed) -> None:
        """Keep approval review reachable from the Console inspector seam."""
        event.stop()
        if self._console_pending_approval_count() <= 0:
            self.app_instance.notify(CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning")
            return
        self.app_instance.notify(
            "Approval review is available from the active Console task context.",
            severity="information",
        )

    @on(Button.Pressed, f"#{CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID}")
    def handle_console_inspector_review_tool_call(self, event: Button.Pressed) -> None:
        """Keep tool-call review reachable from the Console inspector seam."""
        event.stop()
        if self._console_tool_count() <= 0:
            self.app_instance.notify(CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON, severity="warning")
            return
        self.app_instance.notify(
            "Tool-call review is available from the active Console task context.",
            severity="information",
        )

    @on(Button.Pressed, f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}")
    def handle_console_inspector_save_chatbook(self, event: Button.Pressed) -> None:
        """Route inspector Chatbook action through the existing Console save seam."""
        self.handle_console_save_chatbook(event)

    def _get_shell_bar(self):
        """Get the mounted combined chat shell bar."""
        if not self.chat_window:
            return None

        if hasattr(self.chat_window, "get_shell_bar"):
            try:
                return self.chat_window.get_shell_bar()
            except Exception:
                logger.debug("Chat window shell bar seam was unavailable")

        try:
            return self.chat_window.query_one("#chat-shell-bar")
        except Exception:
            return None

    def _get_compact_model_bar(self) -> Optional[CompactModelBar]:
        """Get the embedded compact control bar from the mounted shell bar."""
        try:
            return self.query_one("#console-compact-model-bar", CompactModelBar)
        except QueryError:
            pass

        shell_bar = self._get_shell_bar()
        if not shell_bar:
            return None

        try:
            return shell_bar.query_one(CompactModelBar)
        except QueryError:
            return None
        except Exception as e:
            logger.debug(f"Legacy compact model bar unavailable: {e}")
            return None

    def _sync_console_control_bar(self) -> None:
        """Refresh Console-owned control labels from current selection state."""
        try:
            control_bar = self.query_one("#console-control-bar", ConsoleControlBar)
        except QueryError:
            control_bar = None
        if control_bar is not None:
            control_bar.sync_state(
                self._build_console_control_state(self._pending_console_launch_context)
            )
        try:
            inspector = self.query_one("#console-run-inspector-state", ConsoleRunInspector)
        except QueryError:
            return
        inspector.sync_state(
            self._build_console_inspector_state(self._pending_console_launch_context)
        )

    def _sync_compact_shell_controls(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[str] = None,
    ) -> None:
        """Push sidebar control values back into the compact shell bar."""
        updates: Dict[str, str] = {}
        if provider is not None:
            updates["provider"] = provider
            self._console_control_provider = provider
        if model is not None:
            updates["model"] = model
            self._console_control_model = model
        if temperature is not None:
            updates["temperature"] = temperature

        if not updates:
            return

        compact_bar = self._get_compact_model_bar()
        if compact_bar:
            compact_bar.sync_from_sidebar(**updates)
        else:
            logger.debug("No compact model bar available for reverse sync")
        self._sync_console_control_bar()

    def _sync_compact_shell_controls_from_sidebar(self) -> None:
        """Mirror the current sidebar widget values into the compact shell controls."""
        if not self.chat_window:
            return

        provider = None
        model = None
        temperature = None

        try:
            provider_select = self.chat_window.query_one("#chat-api-provider", Select)
            if not _is_empty_select_value(provider_select.value):
                provider = str(provider_select.value)
        except Exception:
            logger.debug("Sidebar provider select unavailable for compact sync")

        try:
            model_select = self.chat_window.query_one("#chat-api-model", Select)
            if not _is_empty_select_value(model_select.value):
                model = str(model_select.value)
        except Exception:
            logger.debug("Sidebar model select unavailable for compact sync")

        try:
            temperature_input = self.chat_window.query_one("#chat-temperature", Input)
            temperature = temperature_input.value
        except Exception:
            logger.debug("Sidebar temperature input unavailable for compact sync")

        self._sync_compact_shell_controls(
            provider=provider,
            model=model,
            temperature=temperature,
        )

    def sync_shell_bar_from_state(self) -> None:
        """Push the restored active tab state into the mounted shell bar."""
        shell_bar = self._get_shell_bar()
        if not shell_bar:
            logger.debug("No shell bar available for state sync")
            return

        active_tab = self.chat_state.get_active_tab()
        if active_tab is None and self.chat_state.tabs:
            active_tab = self.chat_state.tabs[0]

        if active_tab is None:
            logger.debug("No active tab available for shell bar sync")
            return

        try:
            shell_bar.sync_from_tab_state(active_tab)
            logger.debug(f"Synced shell bar from active tab {active_tab.tab_id}")
        except Exception as e:
            logger.error(f"Failed to sync shell bar from state: {e}", exc_info=True)

    def sync_shell_bar_from_session_data(self, session_data: Optional[ChatSessionData]) -> None:
        """Push the live active session contract into the mounted shell bar."""
        shell_bar = self._get_shell_bar()
        if not shell_bar:
            try:
                composer = self.query_one("#console-native-composer", ConsoleComposerBar)
                composer.sync_session_data(session_data)
            except QueryError:
                pass
            logger.debug("No shell bar available for live session sync")
            return

        try:
            shell_bar.sync_from_session_data(session_data)
            if session_data is None:
                logger.debug("Synced shell bar from cleared live session")
            else:
                logger.debug(
                    "Synced shell bar from live session {}",
                    getattr(session_data, "tab_id", None),
                )
        except Exception as e:
            logger.error(f"Failed to sync shell bar from live session: {e}", exc_info=True)

    def on_chat_tab_container_active_session_changed(
        self,
        message: ChatTabContainer.ActiveSessionChanged,
    ) -> None:
        """Update the shell bar when the live active tab changes."""
        self.sync_shell_bar_from_session_data(message.session_data)
    
    def _save_tab_sessions(self, tab_container) -> None:
        """Save all tab session states."""
        self.chat_state.tabs.clear()
        
        for session_id, session in tab_container.sessions.items():
            tab_state = TabState(
                tab_id=session_id,
                title=session.session_data.title,
                conversation_id=session.session_data.conversation_id,
                runtime_backend=session.session_data.runtime_backend,
                discovery_owner=session.session_data.discovery_owner,
                discovery_entity_id=session.session_data.discovery_entity_id,
                character_id=session.session_data.character_id,
                character_name=session.session_data.character_name,
                assistant_kind=session.session_data.assistant_kind,
                assistant_id=session.session_data.assistant_id,
                persona_memory_mode=session.session_data.persona_memory_mode,
                scope_type=session.session_data.scope_type,
                workspace_id=session.session_data.workspace_id,
                is_active=(session_id == tab_container.active_session_id),
                is_ephemeral=session.session_data.is_ephemeral,
                has_unsaved_changes=session.session_data.has_unsaved_changes,
                system_prompt_override=session.session_data.system_prompt_override,
                temperature_override=session.session_data.temperature_override,
                max_tokens_override=session.session_data.max_tokens_override,
            )
            
            # Save input text for this tab
            try:
                input_widget = session.query_one(f"#chat-input-{session_id}", TextArea)
                if input_widget:
                    tab_state.input_text = input_widget.text
                    # TextArea might not have cursor_position, use selection if available
                    if hasattr(input_widget, 'cursor_position'):
                        tab_state.cursor_position = input_widget.cursor_position
                    elif hasattr(input_widget, 'selection'):
                        tab_state.cursor_position = input_widget.selection.end if input_widget.selection else 0
                    else:
                        tab_state.cursor_position = len(input_widget.text)
            except:
                pass
            
            # Save scroll position
            try:
                scroll_widget = session.query_one(f"#chat-log-{session_id}")
                if scroll_widget:
                    tab_state.scroll_position = scroll_widget.scroll_y
            except:
                pass
            
            self.chat_state.tabs.append(tab_state)
    
    async def _restore_tab_sessions(self, tab_container) -> None:
        """Restore all tab sessions."""
        # Clear existing tabs except default
        for session_id in list(tab_container.sessions.keys()):
            if session_id != "default":
                await tab_container.close_tab(session_id)

        restored_reuse_keys = {}

        # Restore saved tabs
        for tab_state in self.chat_state.tabs:
            restored_title = _derive_tab_title(tab_state)
            reuse_key = None
            if tab_state.conversation_id:
                reuse_key = (tab_state.runtime_backend, tab_state.conversation_id)
                existing_live_tab_id = restored_reuse_keys.get(reuse_key)
                if existing_live_tab_id is not None:
                    if self.chat_state.active_tab_id == tab_state.tab_id:
                        self.chat_state.active_tab_id = existing_live_tab_id
                    continue

            if tab_state.tab_id == "default" and "default" in tab_container.sessions:
                # Update default tab
                session = tab_container.sessions["default"]
                session.session_data.title = restored_title
                session.session_data.conversation_id = tab_state.conversation_id
                session.session_data.runtime_backend = tab_state.runtime_backend
                session.session_data.discovery_owner = tab_state.discovery_owner
                session.session_data.discovery_entity_id = tab_state.discovery_entity_id
                session.session_data.character_id = tab_state.character_id
                session.session_data.character_name = tab_state.character_name
                session.session_data.assistant_kind = tab_state.assistant_kind
                session.session_data.assistant_id = tab_state.assistant_id
                session.session_data.persona_memory_mode = tab_state.persona_memory_mode
                session.session_data.scope_type = tab_state.scope_type
                session.session_data.workspace_id = tab_state.workspace_id
                session.session_data.is_ephemeral = tab_state.is_ephemeral
                session.session_data.has_unsaved_changes = tab_state.has_unsaved_changes
                if reuse_key is not None:
                    restored_reuse_keys[reuse_key] = "default"
            else:
                # Create new tab
                session_data = ChatSessionData(
                    tab_id=tab_state.tab_id,
                    title=restored_title,
                    conversation_id=tab_state.conversation_id,
                    is_ephemeral=tab_state.is_ephemeral,
                    runtime_backend=tab_state.runtime_backend,
                    discovery_owner=tab_state.discovery_owner,
                    discovery_entity_id=tab_state.discovery_entity_id,
                    character_id=tab_state.character_id,
                    character_name=tab_state.character_name,
                    assistant_kind=tab_state.assistant_kind,
                    assistant_id=tab_state.assistant_id,
                    persona_memory_mode=tab_state.persona_memory_mode,
                    scope_type=tab_state.scope_type,
                    workspace_id=tab_state.workspace_id,
                    has_unsaved_changes=tab_state.has_unsaved_changes,
                    system_prompt_override=tab_state.system_prompt_override,
                    temperature_override=tab_state.temperature_override,
                    max_tokens_override=tab_state.max_tokens_override,
                )
                tab_id = await tab_container.create_new_tab(session_data=session_data)
                if tab_id and tab_id in tab_container.sessions:
                    if reuse_key is not None and reuse_key in restored_reuse_keys:
                        if self.chat_state.active_tab_id == tab_state.tab_id:
                            self.chat_state.active_tab_id = restored_reuse_keys[reuse_key]
                        continue

                    session = tab_container.sessions[tab_id]
                    session.session_data.conversation_id = tab_state.conversation_id
                    session.session_data.runtime_backend = tab_state.runtime_backend
                    session.session_data.discovery_owner = tab_state.discovery_owner
                    session.session_data.discovery_entity_id = tab_state.discovery_entity_id
                    session.session_data.character_id = tab_state.character_id
                    session.session_data.character_name = tab_state.character_name
                    session.session_data.assistant_kind = tab_state.assistant_kind
                    session.session_data.assistant_id = tab_state.assistant_id
                    session.session_data.persona_memory_mode = tab_state.persona_memory_mode
                    session.session_data.scope_type = tab_state.scope_type
                    session.session_data.workspace_id = tab_state.workspace_id
                    session.session_data.is_ephemeral = tab_state.is_ephemeral
                    session.session_data.has_unsaved_changes = tab_state.has_unsaved_changes
                    if reuse_key is not None:
                        restored_reuse_keys[reuse_key] = tab_id
                    if self.chat_state.active_tab_id == tab_state.tab_id:
                        self.chat_state.active_tab_id = tab_id
    
    def _save_input_text(self) -> None:
        """Save input text for active tab."""
        try:
            tab_container = self._get_tab_container()
            if tab_container and tab_container.active_session_id:
                active_tab = self.chat_state.get_tab_by_id(tab_container.active_session_id)
                if active_tab:
                    input_widget = self._get_active_chat_input()
                    if input_widget:
                        active_tab.input_text = input_widget.text
                        logger.debug(f"Saved input text for tab {tab_container.active_session_id}: '{input_widget.text[:50]}...'")
                        # TextArea might not have cursor_position
                        if hasattr(input_widget, 'cursor_position'):
                            active_tab.cursor_position = input_widget.cursor_position
                        elif hasattr(input_widget, 'selection'):
                            active_tab.cursor_position = input_widget.selection.end if input_widget.selection else 0
                        else:
                            active_tab.cursor_position = len(input_widget.text)
        except Exception as e:
            logger.debug(f"Could not save input text: {e}")
    
    async def _restore_input_text(self) -> None:
        """Restore input text for active tab."""
        try:
            active_tab = self.chat_state.get_active_tab()
            if active_tab and active_tab.input_text:
                logger.info(f"Restoring input text: '{active_tab.input_text[:50]}...'")
                input_widget = self._get_active_chat_input()

                if input_widget and hasattr(input_widget, 'load_text'):
                    input_widget.load_text(active_tab.input_text)
                    logger.info(f"Successfully restored input text to widget")
                    
                    # Try to restore cursor position
                    if hasattr(input_widget, 'cursor_position'):
                        try:
                            input_widget.cursor_position = active_tab.cursor_position
                        except Exception:
                            pass
                elif input_widget and hasattr(input_widget, 'value'):
                    # Try setting value directly
                    input_widget.value = active_tab.input_text
                    logger.info(f"Restored input text via value property")
                else:
                    logger.warning(f"Could not find suitable method to restore text to widget: {type(input_widget)}")
            else:
                logger.debug("No input text to restore")
        except Exception as e:
            logger.error(f"Error restoring input text: {e}", exc_info=True)
    
    def _save_scroll_positions(self) -> None:
        """Save scroll positions for all tabs."""
        # Implementation depends on tab structure
        pass
    
    async def _restore_scroll_positions(self) -> None:
        """Restore scroll positions for visible tabs."""
        # Implementation depends on tab structure
        pass
    
    def _save_sidebar_settings(self) -> None:
        """Save sidebar settings including system prompt, temperature, etc."""
        try:
            if not self.chat_window:
                logger.debug("Legacy chat sidebar is not mounted; sidebar settings already live in Console controls")
                return

            active_tab = self.chat_state.get_active_tab()
            if not active_tab:
                # Create default tab if none exists
                active_tab = TabState(tab_id="default", title="Chat", is_active=True)
                self.chat_state.tabs = [active_tab]
                self.chat_state.active_tab_id = "default"
                self.chat_state.tab_order = ["default"]
            
            logger.debug("Attempting to save sidebar settings...")
            
            # Log widget IDs for debugging (only in debug mode)
            # Note: loguru doesn't have a simple .level property, skip debug logging for now
            # self._log_sidebar_widgets()
            
            # Save system prompt from sidebar
            system_prompt_saved = False
            try:
                system_prompt_widget = self.chat_window.query_one("#chat-system-prompt", TextArea)
                if system_prompt_widget and hasattr(system_prompt_widget, 'text'):
                    active_tab.system_prompt_override = system_prompt_widget.text
                    logger.info(f"✓ Saved system prompt: '{system_prompt_widget.text[:50]}...'")
                    system_prompt_saved = True
            except Exception as e:
                logger.debug(f"Could not find #chat-system-prompt: {e}")
            
            if not system_prompt_saved:
                # Try with all TextAreas and find the system prompt one
                try:
                    text_areas = self.chat_window.query("TextArea")
                    for ta in text_areas:
                        if ta.id and 'system-prompt' in str(ta.id):
                            active_tab.system_prompt_override = ta.text
                            logger.info(f"✓ Saved system prompt from {ta.id}: '{ta.text[:50]}...'")
                            system_prompt_saved = True
                            break
                except Exception as e:
                    logger.debug(f"Could not find system prompt TextArea: {e}")
            
            # Save temperature
            temp_saved = False
            try:
                temp_input = self.chat_window.query_one("#chat-temperature", Input)
                if temp_input and temp_input.value:
                    active_tab.temperature_override = float(temp_input.value)
                    logger.info(f"✓ Saved temperature: {temp_input.value}")
                    temp_saved = True
            except Exception as e:
                logger.debug(f"Could not find #chat-temperature: {e}")
            
            if not temp_saved:
                # Try to find temperature input by searching all inputs
                try:
                    inputs = self.chat_window.query("Input")
                    for inp in inputs:
                        if inp.id and 'temperature' in str(inp.id):
                            if inp.value:
                                active_tab.temperature_override = float(inp.value)
                                logger.info(f"✓ Saved temperature from {inp.id}: {inp.value}")
                                temp_saved = True
                                break
                except Exception as e:
                    logger.debug(f"Could not find temperature Input: {e}")
            
            # Save max tokens
            try:
                max_tokens_input = self.chat_window.query_one("#chat-llm-max-tokens", Input)
                if max_tokens_input and max_tokens_input.value:
                    active_tab.max_tokens_override = int(max_tokens_input.value)
                    logger.info(f"✓ Saved max tokens: {max_tokens_input.value}")
            except Exception:
                # Try alternative ID
                try:
                    max_tokens_input = self.chat_window.query_one("#chat-max-tokens", Input)
                    if max_tokens_input and max_tokens_input.value:
                        active_tab.max_tokens_override = int(max_tokens_input.value)
                        logger.info(f"✓ Saved max tokens: {max_tokens_input.value}")
                except Exception as e:
                    logger.debug(f"Could not find max tokens input: {e}")
            
            logger.debug(f"Sidebar settings saved - System prompt: {bool(active_tab.system_prompt_override)}, "
                        f"Temperature: {active_tab.temperature_override}, Max tokens: {active_tab.max_tokens_override}")
                
        except Exception as e:
            logger.error(f"Error saving sidebar settings: {e}", exc_info=True)
    
    def _save_attachments(self) -> None:
        """Save pending attachment states."""
        if self.chat_window and hasattr(self.chat_window, 'pending_image'):
            active_tab = self.chat_state.get_active_tab()
            if active_tab and self.chat_window.pending_image:
                active_tab.pending_attachments = [self.chat_window.pending_image]
    
    async def _restore_sidebar_settings(self) -> None:
        """Restore sidebar settings including system prompt, temperature, etc."""
        try:
            if not self.chat_window:
                logger.debug("Legacy chat sidebar is not mounted; skipping sidebar restore")
                return

            active_tab = self.chat_state.get_active_tab()
            if not active_tab:
                logger.debug("No active tab to restore sidebar settings from")
                return
            
            logger.debug(f"Attempting to restore sidebar settings - System prompt: {bool(active_tab.system_prompt_override)}, "
                        f"Temperature: {active_tab.temperature_override}, Max tokens: {active_tab.max_tokens_override}")
            
            # Restore system prompt to sidebar
            if active_tab.system_prompt_override is not None:
                system_restored = False
                try:
                    system_prompt_widget = self.chat_window.query_one("#chat-system-prompt", TextArea)
                    if system_prompt_widget:
                        if hasattr(system_prompt_widget, 'load_text'):
                            system_prompt_widget.load_text(active_tab.system_prompt_override)
                        elif hasattr(system_prompt_widget, 'text'):
                            system_prompt_widget.text = active_tab.system_prompt_override
                        else:
                            system_prompt_widget.value = active_tab.system_prompt_override
                        logger.info(f"✓ Restored system prompt to sidebar: '{active_tab.system_prompt_override[:50]}...'")
                        system_restored = True
                except Exception as e:
                    logger.debug(f"Could not restore to #chat-system-prompt: {e}")
                
                if not system_restored:
                    # Try finding any TextArea with system-prompt in ID
                    try:
                        text_areas = self.chat_window.query("TextArea")
                        for ta in text_areas:
                            if ta.id and 'system-prompt' in str(ta.id):
                                if hasattr(ta, 'load_text'):
                                    ta.load_text(active_tab.system_prompt_override)
                                elif hasattr(ta, 'text'):
                                    ta.text = active_tab.system_prompt_override
                                else:
                                    ta.value = active_tab.system_prompt_override
                                logger.info(f"✓ Restored system prompt to {ta.id}")
                                system_restored = True
                                break
                    except Exception as e:
                        logger.debug(f"Could not restore system prompt to any TextArea: {e}")
            
            # Restore temperature
            if active_tab.temperature_override is not None:
                temp_restored = False
                try:
                    temp_input = self.chat_window.query_one("#chat-temperature", Input)
                    if temp_input:
                        temp_input.value = str(active_tab.temperature_override)
                        logger.info(f"✓ Restored temperature: {active_tab.temperature_override}")
                        temp_restored = True
                except Exception as e:
                    logger.debug(f"Could not restore to #chat-temperature: {e}")
                
                if not temp_restored:
                    # Try finding any Input with temperature in ID
                    try:
                        inputs = self.chat_window.query("Input")
                        for inp in inputs:
                            if inp.id and 'temperature' in str(inp.id):
                                inp.value = str(active_tab.temperature_override)
                                logger.info(f"✓ Restored temperature to {inp.id}: {active_tab.temperature_override}")
                                temp_restored = True
                                break
                    except Exception as e:
                        logger.debug(f"Could not restore temperature to any Input: {e}")
            
            # Restore max tokens
            if active_tab.max_tokens_override is not None:
                try:
                    max_tokens_input = self.chat_window.query_one("#chat-llm-max-tokens", Input)
                    if max_tokens_input:
                        max_tokens_input.value = str(active_tab.max_tokens_override)
                        logger.info(f"✓ Restored max tokens: {active_tab.max_tokens_override}")
                except Exception:
                    # Try alternative ID
                    try:
                        max_tokens_input = self.chat_window.query_one("#chat-max-tokens", Input)
                        if max_tokens_input:
                            max_tokens_input.value = str(active_tab.max_tokens_override)
                            logger.info(f"✓ Restored max tokens: {active_tab.max_tokens_override}")
                    except Exception as e:
                        logger.debug(f"Could not restore max tokens: {e}")

            self._sync_compact_shell_controls_from_sidebar()
                    
        except Exception as e:
            logger.error(f"Error restoring sidebar settings: {e}", exc_info=True)
    
    async def _restore_attachments(self) -> None:
        """Restore pending attachments."""
        active_tab = self.chat_state.get_active_tab()
        if active_tab and active_tab.pending_attachments and self.chat_window:
            # Restore first attachment
            if active_tab.pending_attachments:
                self.chat_window.pending_image = active_tab.pending_attachments[0]
                # Update UI to show attachment indicator
                if hasattr(self.chat_window, 'attachment_handler'):
                    self.chat_window.attachment_handler._update_attachment_indicator()
    
    async def _restore_messages(self) -> None:
        """Restore conversation messages to the chat log."""
        try:
            active_tab = self.chat_state.get_active_tab()
            if not active_tab or not active_tab.messages:
                logger.debug("No messages to restore")
                return
                
            logger.info(f"Restoring {len(active_tab.messages)} messages to chat log")

            log_selectors = [
                "#chat-log",
                ".chat-log",
            ]
            chat_log = self._find_chat_log_container(log_selectors)
            
            if not chat_log:
                logger.warning("Could not find chat log container to restore messages")
                return
            
            # Import message widget class
            from ...Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
            
            # Clear existing messages (optional - you might want to keep them)
            # await chat_log.remove_children()
            
            # Restore each message
            for i, msg_data in enumerate(active_tab.messages):
                try:
                    # Create a new message widget
                    image_data = None
                    if msg_data.metadata and 'image_data' in msg_data.metadata:
                        image_data = msg_data.metadata['image_data']
                    
                    message_widget = ChatMessageEnhanced(
                        message=msg_data.content,
                        role=msg_data.role,
                        timestamp=msg_data.timestamp,
                        message_id=msg_data.message_id,
                        image_data=image_data,
                        generation_complete=True  # All restored messages are complete
                    )
                    
                    # Mount the message widget to the chat log
                    await chat_log.mount(message_widget)
                    
                    if i < 3:  # Log first few for debugging
                        logger.debug(f"Restored message {i+1}: {msg_data.role} - {msg_data.content[:50]}...")
                        
                except Exception as e:
                    logger.error(f"Error restoring message {i}: {e}")
            
            logger.info(f"Successfully restored {len(active_tab.messages)} messages")

            if self.chat_window and hasattr(self.chat_window, "hide_empty_state"):
                self.chat_window.hide_empty_state()
            else:
                try:
                    chat_log.display = True
                except Exception:
                    pass
            
            # Scroll to bottom to show latest messages
            chat_log.scroll_end(animate=False)
            
        except Exception as e:
            logger.error(f"Error in _restore_messages: {e}")
    
    def _save_non_tabbed_state(self) -> None:
        """Save state for non-tabbed chat interface."""
        try:
            # Create a single "default" tab to store the state
            default_tab = TabState(
                tab_id="default",
                title="Chat",
                is_active=True
            )
            
            # Try to find and save input text - be specific about chat input only
            input_selectors = [
                "#chat-input",  # Primary chat input ID
                "TextArea#chat-input",  # TextArea with chat-input ID
                ".chat-input",  # Class-based selector
                "#message-input"  # Alternative message input ID
            ]
            
            for selector in input_selectors:
                try:
                    input_widgets = self.chat_window.query(selector)
                    if input_widgets:
                        for widget in input_widgets:
                            # Make sure we're not saving system prompt or other TextAreas
                            if hasattr(widget, 'id') and widget.id:
                                widget_id = str(widget.id).lower()
                                # Skip if it's a system prompt or settings field
                                if any(x in widget_id for x in ['system', 'prompt', 'settings', 'config']):
                                    logger.debug(f"Skipping non-chat input: {widget.id}")
                                    continue
                            
                            if hasattr(widget, 'text'):
                                default_tab.input_text = widget.text
                                logger.info(f"Found chat input text in {selector}: '{widget.text[:50]}...'")
                                break
                        if default_tab.input_text:
                            break
                except Exception as e:
                    logger.debug(f"Could not query {selector}: {e}")
            
            # Save messages from chat log
            self._extract_and_save_messages(default_tab)
            
            self.chat_state.tabs = [default_tab]
            self.chat_state.active_tab_id = "default"
            self.chat_state.tab_order = ["default"]  # Fix validation issue
            
        except Exception as e:
            logger.error(f"Error saving non-tabbed state: {e}")
    
    def _save_direct_input_text(self) -> None:
        """Try to save input text directly from the chat input TextArea only."""
        try:
            # Be specific - only look for the chat input TextArea, not system prompt or other TextAreas
            chat_input = self._get_active_chat_input()
            if chat_input:
                logger.debug("Found chat input by #chat-input ID")
            
            if not chat_input:
                # Look for TextAreas but filter out system prompt and other non-chat inputs
                text_areas = self._chat_query_scope().query("TextArea")
                logger.debug(f"Found {len(text_areas)} TextArea widgets total")
                
                for text_area in text_areas:
                    # Skip system prompt inputs and other non-chat TextAreas
                    if text_area.id and any(x in str(text_area.id).lower() for x in ['system', 'prompt', 'settings', 'config']):
                        logger.debug(f"Skipping non-chat TextArea: {text_area.id}")
                        continue
                    
                    # Look for chat-related IDs
                    if text_area.id and any(x in str(text_area.id).lower() for x in ['chat-input', 'message', 'input']):
                        chat_input = text_area
                        logger.debug(f"Found likely chat input: {text_area.id}")
                        break
            
            # Save the chat input text if found
            if chat_input and hasattr(chat_input, 'text') and chat_input.text:
                logger.info(f"Saving chat input (id={chat_input.id}): '{chat_input.text[:50]}...'")
                
                # If we have a tab, save to it
                if self.chat_state.tabs:
                    # Save to first/active tab
                    active_tab = self.chat_state.get_active_tab() or self.chat_state.tabs[0]
                    if not active_tab.input_text:  # Don't overwrite if already saved
                        active_tab.input_text = chat_input.text
                        logger.info(f"Saved chat input to tab {active_tab.tab_id}")
                else:
                    # Create a default tab if none exist
                    default_tab = TabState(
                        tab_id="default",
                        title="Chat",
                        input_text=chat_input.text,
                        is_active=True
                    )
                    self.chat_state.tabs = [default_tab]
                    self.chat_state.active_tab_id = "default"
                    logger.info("Created default tab with chat input content")
            else:
                logger.debug("No chat input text to save")
                        
        except Exception as e:
            logger.debug(f"Error in _save_direct_input_text: {e}")
    
    def _extract_and_save_messages(self, tab_state: TabState) -> None:
        """Extract messages from the chat log and save them to the tab state.
        
        Args:
            tab_state: The tab state to save messages to
        """
        try:
            # Import message widget classes
            from ...Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced

            log_selectors = [
                "#chat-log",
                ".chat-log",
                "#chat-messages-container",
                ".chat-messages"
            ]
            chat_log = self._find_chat_log_container(log_selectors)
            
            if not chat_log:
                logger.warning("Could not find chat log container to save messages")
                return
            
            # Extract messages from the chat log
            messages_found = 0
            tab_state.messages = []  # Clear existing messages
            
            # Find all message widgets - try different selectors
            try:
                # Try to find ChatMessageEnhanced widgets
                enhanced_messages = list(chat_log.query(ChatMessageEnhanced))
                
                # If no enhanced messages, try generic approach
                if not enhanced_messages:
                    # Look for any widgets with message-like attributes
                    all_widgets = list(chat_log.children)
                    enhanced_messages = [w for w in all_widgets 
                                       if hasattr(w, 'role') and hasattr(w, 'message_text')]
                
                logger.info(f"Found {len(enhanced_messages)} message widgets in chat log")
                
                for msg_widget in enhanced_messages:
                    try:
                        # Extract message data from widget
                        message_data = MessageData(
                            message_id=getattr(msg_widget, 'message_id_internal', f"msg_{messages_found}"),
                            role=getattr(msg_widget, 'role', 'unknown'),
                            content=getattr(msg_widget, 'message_text', ''),
                            timestamp=getattr(msg_widget, 'timestamp', None)
                        )
                        
                        # Save image data if present
                        if hasattr(msg_widget, 'image_data') and msg_widget.image_data:
                            message_data.metadata = {'image_data': msg_widget.image_data}
                        
                        tab_state.messages.append(message_data)
                        messages_found += 1
                        
                        # Log first few messages for debugging
                        if messages_found <= 3:
                            logger.debug(f"Saved message {messages_found}: role={message_data.role}, content={message_data.content[:50]}...")
                            
                    except Exception as e:
                        logger.warning(f"Error extracting message data from widget: {e}")
                        
                logger.info(f"Successfully saved {messages_found} messages to tab state")
                
            except Exception as e:
                logger.error(f"Error querying for message widgets: {e}")
                
        except Exception as e:
            logger.error(f"Error in _extract_and_save_messages: {e}")
    
    def on_screen_suspend(self) -> None:
        """Called when navigating away from this screen."""
        logger.debug("Chat screen suspending - saving state")
        self.save_state()
        # Note: BaseAppScreen doesn't have on_screen_suspend, so no super() call
    
    def on_screen_resume(self) -> None:
        """Called when returning to this screen."""
        logger.debug("Chat screen resuming")
        self.sync_task_resume_state()
        # Note: BaseAppScreen doesn't have on_screen_resume, so no super() call

    def set_task_resume_state(self, task_state: TaskResumeState) -> None:
        """Update the persisted task resume state and sync it into the chat UI."""
        self.chat_state.task_resume_state = task_state
        self.sync_task_resume_state()

    def sync_task_resume_state(self) -> None:
        """Push the current task resume state into the chat window when available."""
        try:
            task_cards = self.query_one("#console-task-surface", ChatTaskCards)
            task_cards.sync_state(self.chat_state.task_resume_state)
            return
        except QueryError:
            pass

        if self.chat_window:
            self.chat_window.sync_task_resume_state(self.chat_state.task_resume_state)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button events at the screen level.
        This ensures buttons work properly with screen-based navigation.
        """
        button_id = event.button.id
        
        # Log for debugging
        logger.info(f"ChatScreen on_button_pressed called with button: {button_id}")
        
        # Sidebar toggle is handled in ChatWindowEnhanced via @on decorator
        
        # Buttons that are handled by @on decorators in ChatWindowEnhanced
        # These should NOT be delegated to avoid double handling
        handled_by_decorators = [
            "send-stop-chat",
            "attach-image",
            "chat-mic"
            # Removed sidebar toggles from here since they're handled above
        ]
        
        if button_id in handled_by_decorators:
            # These are already handled by @on decorators, just stop propagation
            event.stop()
            return
            
        # For remaining buttons that need legacy handling, delegate to ChatWindowEnhanced
        if self.chat_window:
            # The chat window knows how to handle its own buttons
            await self.chat_window.on_button_pressed(event)
            event.stop()  # Prevent bubbling to app level
    
    
    async def _run_diagnostic(self) -> None:
        """Run diagnostic tool on the chat widget structure."""
        try:
            if not self.chat_window:
                return
                
            logger.info("Running chat widget structure diagnostics...")
            diagnostics = ChatDiagnostics()
            report = diagnostics.inspect_widget_tree(self.chat_window, max_depth=5)
            
            # Log key findings
            logger.info(f"Diagnostic: {report['chat_structure']['type']} interface detected")
            logger.info(f"Found {report['text_areas']['count']} TextArea widgets")
            logger.info(f"Found {report['containers']['chat_containers']} chat containers")
            logger.info(f"Found {report['containers']['tab_containers']} tab containers")
            
            # Log any input widgets found
            if report['input_widgets']:
                for widget in report['input_widgets']:
                    logger.info(f"Input widget: {widget['id']} at {widget['path']}")
            
            # Store report for potential debugging
            self._diagnostic_report = report
            
            # Also log all sidebar-related widgets for debugging
            self._log_sidebar_widgets()
            
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}", exc_info=True)
    
    def _log_sidebar_widgets(self) -> None:
        """Log all sidebar widgets for debugging state preservation."""
        try:
            logger.info("=== Sidebar Widget IDs ===")
            
            # Find all TextAreas
            text_areas = self.chat_window.query("TextArea")
            for ta in text_areas:
                if ta.id:
                    logger.info(f"TextArea ID: {ta.id}, Has text: {bool(getattr(ta, 'text', None))}")
            
            # Find all Inputs
            inputs = self.chat_window.query("Input")
            for inp in inputs:
                if inp.id:
                    logger.info(f"Input ID: {inp.id}, Value: {getattr(inp, 'value', 'N/A')}")
            
            logger.info("=========================")
        except Exception as e:
            logger.debug(f"Error logging sidebar widgets: {e}")
    
    def watch_sidebar_state(self, new_state: dict) -> None:
        """Auto-save when sidebar state changes."""
        self._save_sidebar_state()
    
    def _load_sidebar_state(self) -> None:
        """Load sidebar state from config file."""
        config_path = Path.home() / ".config" / "tldw_cli" / "ui_state.toml"
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = toml.load(f)
                    sidebar_data = data.get("sidebar", {})
                    
                    # Load collapsible states into UIState
                    self.ui_state.collapsible_states = sidebar_data.get("collapsible_states", {})
                    self.ui_state.sidebar_search_query = sidebar_data.get("search_query", "")
                    self.ui_state.last_active_section = sidebar_data.get("last_active_section", None)
                    
                    # Update reactive property
                    self.sidebar_state = dict(self.ui_state.collapsible_states)
                    
                    logger.debug(f"Loaded sidebar state with {len(self.ui_state.collapsible_states)} collapsibles")
        except Exception as e:
            logger.error(f"Failed to load sidebar state: {e}")
            self.sidebar_state = {}
    
    def _save_sidebar_state(self) -> None:
        """Save sidebar state to config file."""
        config_path = Path.home() / ".config" / "tldw_cli" / "ui_state.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load existing config or create new
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = toml.load(f)
            else:
                data = {}
            
            # Update sidebar section
            data["sidebar"] = {
                "collapsible_states": dict(self.ui_state.collapsible_states),
                "search_query": self.ui_state.sidebar_search_query,
                "last_active_section": self.ui_state.last_active_section
            }
            
            # Save back to file
            with open(config_path, 'w') as f:
                toml.dump(data, f)
                
            logger.debug(f"Saved sidebar state with {len(self.ui_state.collapsible_states)} collapsibles")
        except Exception as e:
            logger.error(f"Failed to save sidebar state: {e}")
    
    def _restore_collapsible_states(self) -> None:
        """Restore collapsible states from saved state."""
        if not self.ui_state.collapsible_states:
            logger.debug("No collapsible states to restore")
            return
            
        try:
            # Find all collapsibles in the sidebar
            collapsibles = self.query(Collapsible)
            restored_count = 0
            
            for collapsible in collapsibles:
                if collapsible.id and collapsible.id in self.ui_state.collapsible_states:
                    collapsed_state = self.ui_state.collapsible_states[collapsible.id]
                    collapsible.collapsed = collapsed_state
                    restored_count += 1
                    logger.debug(f"Restored {collapsible.id}: collapsed={collapsed_state}")
            
            logger.info(f"Restored {restored_count} collapsible states")
        except Exception as e:
            logger.error(f"Error restoring collapsible states: {e}")
    
    @on(Collapsible.Toggled)
    def handle_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Save collapsible state when toggled."""
        try:
            collapsible_id = event.collapsible.id
            if collapsible_id:
                # Update UIState
                self.ui_state.set_collapsible_state(collapsible_id, event.collapsible.collapsed)
                
                # Update reactive property to trigger watcher
                new_state = dict(self.ui_state.collapsible_states)
                self.sidebar_state = new_state
                
                logger.debug(f"Toggled {collapsible_id}: collapsed={event.collapsible.collapsed}")
        except Exception as e:
            logger.error(f"Error handling collapsible toggle: {e}")
    
    @on(Button.Pressed, "#chat-expand-all")
    def handle_expand_all(self, event: Button.Pressed) -> None:
        """Expand all collapsible sections."""
        try:
            collapsibles = self.query(Collapsible)
            expanded_count = 0
            
            for collapsible in collapsibles:
                if collapsible.collapsed:
                    collapsible.collapsed = False
                    expanded_count += 1
                    if collapsible.id:
                        self.ui_state.set_collapsible_state(collapsible.id, False)
            
            # Update reactive property
            self.sidebar_state = dict(self.ui_state.collapsible_states)
            
            logger.info(f"Expanded {expanded_count} sections")
            self.notify(f"Expanded {expanded_count} sections", severity="information")
        except Exception as e:
            logger.error(f"Error expanding all sections: {e}")
    
    @on(Button.Pressed, "#chat-collapse-all")
    def handle_collapse_all(self, event: Button.Pressed) -> None:
        """Collapse all non-priority collapsible sections."""
        try:
            collapsibles = self.query(Collapsible)
            collapsed_count = 0
            
            for collapsible in collapsibles:
                # Keep priority sections open
                if "priority-high" not in collapsible.classes and not collapsible.collapsed:
                    collapsible.collapsed = True
                    collapsed_count += 1
                    if collapsible.id:
                        self.ui_state.set_collapsible_state(collapsible.id, True)
            
            # Update reactive property
            self.sidebar_state = dict(self.ui_state.collapsible_states)
            
            logger.info(f"Collapsed {collapsed_count} non-essential sections")
            self.notify(f"Collapsed {collapsed_count} sections", severity="information")
        except Exception as e:
            logger.error(f"Error collapsing sections: {e}")
    
    @on(Button.Pressed, "#chat-reset-settings")
    def handle_reset_settings(self, event: Button.Pressed) -> None:
        """Reset settings to defaults."""
        try:
            # Clear all saved collapsible states
            self.ui_state.collapsible_states.clear()
            self.sidebar_state = {}
            
            # Reset collapsibles to default states
            collapsibles = self.query(Collapsible)
            for collapsible in collapsibles:
                # Default state: priority sections open, others closed
                if "priority-high" in collapsible.classes:
                    collapsible.collapsed = False
                else:
                    collapsible.collapsed = True
            
            self._save_sidebar_state()
            logger.info("Reset sidebar to default state")
            self.notify("Settings reset to defaults", severity="success")
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
