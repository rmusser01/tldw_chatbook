"""Chat screen implementation with comprehensive state management."""

from dataclasses import replace
from datetime import datetime
import inspect
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, TYPE_CHECKING
import uuid

import toml
from loguru import logger
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.events import Click, Key, Paste
from textual.reactive import reactive
from textual.widgets import Button, Static, TextArea, Select, Collapsible, Input

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from .chat_screen_state import ChatScreenState, TabState, MessageData, TaskResumeState
from .provider_model_resolution import resolve_effective_provider_model
from ...Chat.chat_conversation_service import derive_conversation_title
from ...Chat.chat_persistence_service import ChatPersistenceService
from ...Chat.console_chat_controller import ConsoleChatController
from ...Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunStatus,
    ConsoleWorkspaceContext,
    ConsoleStagedSource,
)
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
    build_console_context_estimate,
    build_default_console_session_settings,
    build_console_settings_readiness,
    build_console_settings_summary_state,
)
from ...Chat.console_chat_store import ConsoleChatStore
from ...Chat.console_provider_gateway import (
    DEFAULT_LLAMACPP_BASE_URL,
    ConsoleProviderGateway,
    normalize_llamacpp_base_url,
)
from ...Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON,
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
    ConsoleControlState,
    ConsoleInspectorState,
    ConsoleStagedContextState,
    build_console_evidence_display_state,
    coerce_non_negative_int,
)
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Chat.chat_models import ChatSessionData
from ...Chat.provider_readiness import get_provider_readiness, provider_config_key
from ...Chat.console_message_actions import ConsoleActionResult, ConsoleMessageActionService
from ...Chat.console_live_work import (
    ConsoleLiveWorkLaunch,
    ConsoleLiveWorkSourceReadinessState,
    ConsoleLiveWorkStatusCardState,
)
from ...Chat.console_rail_state import (
    ConsoleRailPreferences,
    ConsoleRailState,
    build_console_rail_preference_key,
    build_console_rail_state,
    coerce_console_rail_preferences,
    serialize_console_rail_preferences,
)
from ...config import coerce_bool_setting, get_cli_providers_and_models, save_setting_to_cli_config
from ...Library.library_rag_service import (
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from ...Constants import TAB_SETTINGS
from ...Utils.chat_diagnostics import ChatDiagnostics
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...state.ui_state import UIState
from ...Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from ...Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from ...Widgets.Console import (
    ConsoleComposerBar,
    ConsoleControlBar,
    ConsoleRailHandle,
    ConsoleRunInspector,
    ConsoleSaveAsModal,
    ConsoleSessionSurface,
    ConsoleSettingsModal,
    ConsoleSettingsSummary,
    ConsoleStagedContextTray,
    ConsoleTranscript,
    ConsoleWorkspaceContextTray,
)
from ...Workspaces.display_state import (
    ConsoleWorkspaceContextState,
    build_console_workspace_state,
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
CONSOLE_LIBRARY_RAG_QUERY_EMPTY_MESSAGE = "Type a Library RAG query before running retrieval."
CONSOLE_FRAME_COLOR = "#6f7782"
CONSOLE_FRAME_BORDER = ("solid", CONSOLE_FRAME_COLOR)
CONSOLE_QUIET_FRAME_BORDER = ("none", CONSOLE_FRAME_COLOR)
CONSOLE_INLINE_GUIDANCE_COPY = "Ask in Composer. Attach as needed."
CONSOLE_START_HERE_COPY = ""
CONSOLE_ACTION_HINTS_COPY = ""


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

    @on(Button.Pressed, "#console-context-rail-collapse")
    def on_console_context_rail_collapse(self, event: Button.Pressed) -> None:
        """Collapse the Console context rail and persist the preference."""
        event.stop()
        self._set_console_rail_preference(left_open=False)

    @on(Button.Pressed, "#console-context-rail-open")
    def on_console_context_rail_open(self, event: Button.Pressed) -> None:
        """Open the Console context rail and persist the preference."""
        event.stop()
        self._set_console_rail_preference(left_open=True)

    @on(Button.Pressed, "#console-inspector-rail-collapse")
    def on_console_inspector_rail_collapse(self, event: Button.Pressed) -> None:
        """Collapse the Console inspector rail and persist the preference."""
        event.stop()
        self._set_console_rail_preference(right_open=False)

    @on(Button.Pressed, "#console-inspector-rail-open")
    def on_console_inspector_rail_open(self, event: Button.Pressed) -> None:
        """Open the Console inspector rail and persist the preference."""
        event.stop()
        self._set_console_rail_preference(right_open=True)

    async def on_console_settings_open(self, event: Button.Pressed) -> None:
        """Open Console session settings for the active native session."""
        event.stop()
        settings = self._ensure_active_console_session_settings()
        controller = self._ensure_console_chat_controller()
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=getattr(self.app_instance, "app_config", {}) or {},
            providers_models=self._providers_models(),
            context_estimate=self._active_console_settings_context_estimate(),
            can_save=controller.run_state.is_send_allowed,
        )

        def _apply_modal_result(result: ConsoleSessionSettings | None) -> None:
            if not isinstance(result, ConsoleSessionSettings):
                return
            self._replace_active_console_session_settings(result)
            self.run_worker(self._sync_native_console_chat_ui(), exclusive=True)

        self.app.push_screen(modal, callback=_apply_modal_result)
    
    
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
        self._console_chat_store: ConsoleChatStore | None = None
        self._console_provider_gateway: Any | None = None
        self._console_chat_controller: ConsoleChatController | None = None
        self._console_message_action_service = ConsoleMessageActionService()
        self._last_console_action: ConsoleActionResult | None = None
        self._console_transcript_sync_timer: Any | None = None
        self._console_sync_in_progress = False
        self._console_guidance_dismissed = False
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
        """Return a chat default value from app config for legacy call sites."""
        config = getattr(self.app_instance, "app_config", {}) or {}
        defaults = config.get("chat_defaults", {}) if isinstance(config, dict) else {}
        return defaults.get(key) if isinstance(defaults, dict) else None

    def _effective_console_provider_model(self) -> tuple[Any, Any]:
        """Return the canonical Console provider/model selection.

        Returns:
            A `(provider, model)` tuple using the same precedence for Console
            control labels and run-inspector readiness.
        """
        effective = resolve_effective_provider_model(
            self.app_instance,
            console_provider=self._console_control_provider,
            console_model=self._console_control_model,
        )
        return effective.provider, effective.model

    @staticmethod
    def _normalize_llamacpp_base_url(api_url: str | None) -> str:
        """Return the llama.cpp origin root used before appending OpenAI paths."""
        return normalize_llamacpp_base_url(api_url) or DEFAULT_LLAMACPP_BASE_URL

    @staticmethod
    def _config_section(config: dict[str, Any], key: str) -> dict[str, Any]:
        value = config.get(key, {})
        return value if isinstance(value, dict) else {}

    def _active_console_session_settings(self) -> ConsoleSessionSettings | None:
        """Return settings for the active native Console session, if one exists."""
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return None
        try:
            return store.session_settings(store.active_session_id)
        except KeyError:
            return None

    def _providers_models(self) -> dict[str, list[str]]:
        """Return configured provider/model options for Console settings."""
        providers_models = getattr(self.app_instance, "providers_models", None)
        if isinstance(providers_models, dict):
            return {
                str(provider): [str(model) for model in models]
                for provider, models in providers_models.items()
                if isinstance(models, (list, tuple))
            }
        try:
            return get_cli_providers_and_models()
        except Exception:
            logger.debug("Unable to load CLI provider/model registry for Console settings")
            return {}

    def _default_console_session_settings(self) -> ConsoleSessionSettings:
        """Build the default settings snapshot for a new native Console session."""
        provider, model = self._effective_console_provider_model()
        settings = build_default_console_session_settings(
            getattr(self.app_instance, "app_config", {}) or {},
            str(provider).strip() if _has_selected_text(provider) else None,
            str(model).strip() if _has_selected_text(model) else None,
        )
        provider_key = provider_config_key(settings.provider)
        return replace(
            settings,
            base_url=None if provider_key in {"llama_cpp", "local_llamacpp"} else settings.base_url,
        )

    def _ensure_active_console_session_settings(self) -> ConsoleSessionSettings:
        """Ensure the active native Console session owns a settings snapshot."""
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id,
            settings=self._default_console_session_settings(),
        )
        if session.settings is None:
            settings = self._default_console_session_settings()
            store.replace_session_settings(session.id, settings)
            return settings
        return session.settings

    def _replace_active_console_session_settings(
        self,
        settings: ConsoleSessionSettings,
    ) -> None:
        """Replace settings for only the active native Console session."""
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id,
            settings=self._default_console_session_settings(),
        )
        store.replace_session_settings(session.id, settings)
        self._sync_console_chat_core_state()
        self._sync_console_settings_summary()

    def _active_console_settings_context_estimate(self) -> ConsoleSettingsContextEstimate:
        """Return context usage for the active native Console settings snapshot."""
        settings = self._ensure_active_console_session_settings()
        workspace_context = self._current_console_workspace_context()
        staged_context_state = self._build_console_staged_context_state(
            self._pending_console_launch_context
        )
        messages: list[dict[str, str]] = []
        store = self._console_chat_store
        if store is not None and store.active_session_id is not None:
            try:
                messages = [
                    {
                        "role": str(message.role.value if hasattr(message.role, "value") else message.role),
                        "content": message.content,
                    }
                    for message in store.messages_for_session(store.active_session_id)
                ]
            except KeyError:
                messages = []
        return build_console_context_estimate(
            messages,
            settings.provider,
            settings.model,
            staged_source_count=len(workspace_context.staged_sources),
            staged_context_summary=staged_context_state.summary,
            max_tokens_response=settings.max_tokens,
        )

    def _build_console_settings_summary_state(self) -> ConsoleSettingsSummaryState:
        """Build compact summary state for the active Console session settings."""
        settings, readiness = self._active_console_settings_readiness()
        return build_console_settings_summary_state(
            settings,
            self._active_console_settings_context_estimate(),
            readiness,
        )

    def _sync_console_settings_summary(self) -> None:
        """Refresh the mounted Console settings summary if present."""
        try:
            summary = self.query_one("#console-settings-summary", ConsoleSettingsSummary)
        except (NoMatches, QueryError):
            return
        summary.sync_state(self._build_console_settings_summary_state())

    def _current_console_workspace_context(self) -> ConsoleWorkspaceContext:
        """Return explicit workspace policy context for native Console sends."""
        workspace_id = "global"
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        if registry_service is not None:
            try:
                active_workspace = registry_service.get_active_workspace()
                candidate = getattr(active_workspace, "workspace_id", None)
                if candidate:
                    workspace_id = str(candidate)
            except Exception:
                logger.debug("Console workspace registry was unavailable for send context")

        staged_sources: list[ConsoleStagedSource] = []
        pending_launch = self._pending_console_launch_context
        if pending_launch is not None:
            payload = pending_launch.payload
            source_id = (
                payload.get("source_id")
                or payload.get("target_id")
                or payload.get("run_id")
                or pending_launch.title
            )
            source_workspace = payload.get("workspace_id")
            staged_sources.append(
                ConsoleStagedSource(
                    source_id=str(source_id),
                    label=pending_launch.title,
                    source_type=str(pending_launch.source),
                    workspace_id=str(source_workspace) if source_workspace else None,
                )
            )

        return ConsoleWorkspaceContext(
            active_workspace_id=workspace_id,
            staged_sources=tuple(staged_sources),
        )

    def _build_console_provider_selection(self) -> ConsoleProviderSelection:
        """Return the effective native Console provider selection for sends."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        selection_settings = self._ensure_active_console_session_settings()
        _legacy_provider, legacy_model = self._effective_console_provider_model()
        provider = provider_config_key(selection_settings.provider) or "llama_cpp"
        explicit_model = (
            str(selection_settings.model).strip()
            if _has_selected_text(selection_settings.model)
            else None
        )
        api_settings = self._config_section(app_config, "api_settings")
        provider_config = self._config_section(api_settings, provider)
        console_config = self._config_section(app_config, "console")
        configured_model_value = (
            provider_config.get("model")
            or provider_config.get("api_model")
            or provider_config.get("default_model")
        )
        configured_model = (
            str(configured_model_value).strip()
            if _has_selected_text(configured_model_value)
            else None
        )
        if not _has_selected_text(legacy_model) and explicit_model == configured_model:
            explicit_model = None

        base_url: str | None = None
        if provider in {"llama_cpp", "local_llamacpp"}:
            fallback_url = (
                os.environ.get("TLDW_CONSOLE_LLAMA_CPP_BASE_URL")
                or console_config.get("llama_cpp_base_url_override")
                or provider_config.get("api_url")
                or provider_config.get("base_url")
                or provider_config.get("api_base")
            )
            override_url = (
                selection_settings.base_url
                if _has_selected_text(selection_settings.base_url)
                else fallback_url
            )
            base_url = self._normalize_llamacpp_base_url(
                str(override_url) if override_url is not None else None
            )
        elif _has_selected_text(selection_settings.base_url):
            base_url = str(selection_settings.base_url).strip()

        return ConsoleProviderSelection(
            provider=provider,
            base_url=base_url,
            explicit_model=explicit_model,
            configured_model=configured_model,
            temperature=selection_settings.temperature,
            top_p=selection_settings.top_p,
            min_p=selection_settings.min_p,
            top_k=selection_settings.top_k,
            max_tokens=selection_settings.max_tokens,
            streaming=selection_settings.streaming,
            workspace_context=self._current_console_workspace_context(),
        )

    def _active_console_provider_model_display(self) -> tuple[str, str | None, ConsoleSessionSettings]:
        """Return provider/model labels backed by active session settings."""
        settings = self._ensure_active_console_session_settings()
        selection = self._build_console_provider_selection()
        legacy_provider, _legacy_model = self._effective_console_provider_model()
        provider_display = selection.provider
        is_matching_provider = (
            provider_config_key(str(legacy_provider or "")) == selection.provider
        )
        if is_matching_provider and _has_selected_text(legacy_provider):
            provider_display = str(legacy_provider).strip()
        selected_model = selection.explicit_model or selection.configured_model
        return provider_display, selected_model, settings

    def _active_console_settings_readiness(self) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness]:
        """Return effective settings plus Console-native readiness for display/send surfaces."""
        settings = self._ensure_active_console_session_settings()
        selection = self._build_console_provider_selection()
        selected_model = selection.explicit_model or selection.configured_model
        effective_settings = replace(settings, model=selected_model)
        if not _has_selected_text(selected_model):
            return effective_settings, ConsoleSettingsReadiness(
                label="Missing model",
                detail="Select a model before sending.",
                native_send_supported=False,
            )
        readiness = build_console_settings_readiness(
            effective_settings,
            app_config=getattr(self.app_instance, "app_config", {}) or {},
        )
        return effective_settings, readiness

    def _ensure_console_chat_store(self) -> ConsoleChatStore:
        """Return the native Console chat store, creating it lazily."""
        if self._console_chat_store is None:
            persistence = None
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is not None:
                persistence = ChatPersistenceService(db)
            self._console_chat_store = ConsoleChatStore(
                persistence=persistence,
                workspace_context=self._current_console_workspace_context(),
            )
        return self._console_chat_store

    def _ensure_console_provider_gateway(self) -> Any:
        """Return the native Console provider gateway with a test injection seam."""
        if self._console_provider_gateway is None:
            factory = getattr(self.app_instance, "console_provider_gateway_factory", None)
            self._console_provider_gateway = (
                factory() if callable(factory) else ConsoleProviderGateway()
            )
        return self._console_provider_gateway

    def _ensure_console_chat_controller(self) -> ConsoleChatController:
        """Return the native Console chat controller with fresh selection state."""
        if self._console_chat_controller is None:
            selection = self._build_console_provider_selection()
            self._console_chat_controller = ConsoleChatController(
                store=self._ensure_console_chat_store(),
                provider_gateway=self._ensure_console_provider_gateway(),
                provider=selection.provider,
                model=selection.explicit_model,
                configured_model=selection.configured_model,
                base_url=selection.base_url,
                temperature=selection.temperature,
                top_p=selection.top_p,
                min_p=selection.min_p,
                top_k=selection.top_k,
                max_tokens=selection.max_tokens,
                streaming=selection.streaming,
            )
        self._sync_console_chat_core_state()
        return self._console_chat_controller

    def _sync_console_chat_core_state(self) -> ConsoleProviderSelection:
        """Push current workspace/provider selection into native Console services."""
        selection = self._build_console_provider_selection()
        self._ensure_console_chat_store().set_workspace_context(selection.workspace_context)
        if self._console_chat_controller is not None:
            update_selection = getattr(
                self._console_chat_controller,
                "update_provider_selection",
                None,
            )
            if callable(update_selection):
                update_selection(selection)
            else:
                self._console_chat_controller.provider = selection.provider
                self._console_chat_controller.model = selection.explicit_model
                self._console_chat_controller.configured_model = selection.configured_model
                self._console_chat_controller.base_url = selection.base_url
                self._console_chat_controller.temperature = selection.temperature
                self._console_chat_controller.top_p = selection.top_p
                self._console_chat_controller.min_p = selection.min_p
                self._console_chat_controller.top_k = selection.top_k
                self._console_chat_controller.max_tokens = selection.max_tokens
                self._console_chat_controller.streaming = selection.streaming
        return selection

    def _build_console_control_state(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleControlState:
        """Build Console-owned control/readiness labels."""
        provider, model, _settings = self._active_console_provider_model_display()
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

    def _current_console_conversation_id(
        self,
        session_data: Optional[ChatSessionData] = None,
    ) -> Optional[str]:
        """Return the active conversation id for Console context highlighting."""
        conversation_id = getattr(session_data, "conversation_id", None)
        if conversation_id:
            return str(conversation_id)

        active_tab = self.chat_state.get_active_tab()
        conversation_id = getattr(active_tab, "conversation_id", None)
        if conversation_id:
            return str(conversation_id)

        console_store = self._console_chat_store
        active_session_id = (
            console_store.active_session_id
            if console_store is not None
            else None
        )
        if console_store is not None and active_session_id is not None:
            for console_session in console_store.sessions():
                if console_session.id == active_session_id:
                    conversation_id = console_session.persisted_conversation_id
                    if conversation_id:
                        return str(conversation_id)
                    break

        session = self._get_active_chat_session()
        session_data = getattr(session, "session_data", None)
        conversation_id = getattr(session_data, "conversation_id", None)
        return str(conversation_id) if conversation_id else None

    def _active_native_console_session(self) -> Any | None:
        """Return the active native Console session without creating the store."""
        console_store = self._console_chat_store
        active_session_id = (
            console_store.active_session_id
            if console_store is not None
            else None
        )
        if console_store is None or active_session_id is None:
            return None
        for console_session in console_store.sessions():
            if console_session.id == active_session_id:
                return console_session
        return None

    def _current_console_rail_conversation_id(self) -> Optional[str]:
        """Return the conversation scope used only for Console rail persistence."""
        native_session = self._active_native_console_session()
        if native_session is not None:
            conversation_id = getattr(
                native_session,
                "persisted_conversation_id",
                None,
            )
            return str(conversation_id) if conversation_id else None
        return self._current_console_conversation_id()

    def _build_console_workspace_context_state(
        self,
        session_data: Optional[ChatSessionData] = None,
    ) -> ConsoleWorkspaceContextState:
        return build_console_workspace_state(
            registry_service=getattr(self.app_instance, "workspace_registry_service", None),
            current_conversation=self._current_console_conversation_id(session_data),
        )

    def _console_config(self) -> dict[str, Any]:
        """Return mutable Console app config, initializing the section if needed."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            app_config = {}
            setattr(self.app_instance, "app_config", app_config)
        console_config = app_config.get("console")
        if not isinstance(console_config, dict):
            console_config = {}
            app_config["console"] = console_config
        return console_config

    def _console_rail_state_config(self) -> dict[str, Any]:
        """Return mutable Console rail-state config, initializing it if needed."""
        console_config = self._console_config()
        rail_state_config = console_config.get("rail_state")
        if not isinstance(rail_state_config, dict):
            rail_state_config = {}
            console_config["rail_state"] = rail_state_config
        return rail_state_config

    def _stored_console_rail_preferences(
        self,
        key: str,
        fallback_key: str | None,
    ) -> Any:
        """Read stored Console rail preferences without writing persistence."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return None
        console_config = app_config.get("console")
        if not isinstance(console_config, dict):
            return None
        rail_state_config = console_config.get("rail_state")
        if not isinstance(rail_state_config, dict):
            return None
        if key in rail_state_config:
            return rail_state_config[key]
        if fallback_key and fallback_key in rail_state_config:
            return rail_state_config[fallback_key]
        return None

    def _persist_console_rail_preferences(
        self,
        key: str,
        preferences: ConsoleRailPreferences,
        *,
        notify_on_failure: bool = False,
    ) -> bool:
        """Queue best-effort persistence for an already-updated in-memory preference."""
        serialized = serialize_console_rail_preferences(preferences)
        self._save_console_rail_preferences(
            key,
            serialized,
            notify_on_failure=notify_on_failure,
        )
        return True

    @work(thread=True)
    def _save_console_rail_preferences(
        self,
        key: str,
        serialized: dict[str, bool],
        *,
        notify_on_failure: bool = False,
    ) -> None:
        """Persist Console rail preferences without blocking the UI thread."""
        try:
            saved = save_setting_to_cli_config(
                "console.rail_state",
                key,
                serialized,
            )
        except Exception as exc:
            logger.warning("Failed to persist Console rail preference: {}", exc)
            saved = False
        if not saved and notify_on_failure:
            self.app.call_from_thread(self._notify_console_rail_preference_save_failure)

    def _notify_console_rail_preference_save_failure(self) -> None:
        """Notify from the UI thread when background preference persistence fails."""
        self.app_instance.notify(
            "Console rail preference is saved for this session only.",
            severity="warning",
        )

    def _migrate_console_rail_fallback_preferences(
        self,
        key: str,
        fallback_key: str | None,
    ) -> None:
        """Copy temporary session rail preferences to a durable key when needed."""
        if not fallback_key:
            return
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        console_config = app_config.get("console")
        if not isinstance(console_config, dict):
            return
        rail_state_config = console_config.get("rail_state")
        if not isinstance(rail_state_config, dict):
            return
        if key in rail_state_config or fallback_key not in rail_state_config:
            return
        preferences = coerce_console_rail_preferences(rail_state_config[fallback_key])
        rail_state_config[key] = serialize_console_rail_preferences(preferences)
        self._persist_console_rail_preferences(
            key,
            preferences,
            notify_on_failure=False,
        )

    def _current_console_session_id(self) -> Optional[str]:
        """Return a durable external Console session scope when one is available."""
        session_id = getattr(self.app_instance, "console_rail_session_id", None)
        if session_id:
            return str(session_id)
        console_store = self._console_chat_store
        if console_store is not None and console_store.active_session_id is not None:
            return str(console_store.active_session_id)
        return None

    def _console_rail_available_columns(self) -> int | None:
        """Return available screen width for responsive rail state."""
        width = getattr(getattr(self, "size", None), "width", None)
        return int(width) if width else None

    def _current_console_run_status_value(self) -> str:
        """Return the current Console run status value for rail badging."""
        controller = self._console_chat_controller
        if controller is not None:
            run_state = getattr(controller, "run_state", None)
            status = getattr(run_state, "status", None)
            if status is not None:
                return str(getattr(status, "value", status))
        override = getattr(self.app_instance, "console_run_status_override", None)
        if override is not None:
            return str(getattr(override, "value", override))
        return "idle"

    def _build_console_rail_state(
        self,
        *,
        staged_context_state: ConsoleStagedContextState,
        inspector_state: ConsoleInspectorState,
        workspace_context_state: ConsoleWorkspaceContextState,
    ) -> ConsoleRailState:
        """Build the effective Console rail state for the current composition."""
        workspace_context = self._current_console_workspace_context()
        active_session_id = (
            self._console_chat_store.active_session_id
            if self._console_chat_store is not None
            else None
        )
        active_session = None
        if self._console_chat_store is not None and active_session_id is not None:
            for session in self._console_chat_store.sessions():
                if session.id == active_session_id:
                    active_session = session
                    break
        preference_key = build_console_rail_preference_key(
            workspace_id=workspace_context.active_workspace_id,
            conversation_id=self._current_console_rail_conversation_id(),
            session_id=self._current_console_session_id(),
        )
        self._migrate_console_rail_fallback_preferences(
            preference_key.value,
            preference_key.fallback_value,
        )
        return build_console_rail_state(
            preference_key=preference_key,
            stored_preferences=self._stored_console_rail_preferences(
                preference_key.value,
                preference_key.fallback_value,
            ),
            staged_source_count=len(workspace_context.staged_sources),
            staged_summary=staged_context_state.summary,
            workspace_label=workspace_context_state.workspace_label,
            session_label=getattr(active_session, "title", ""),
            run_status=self._current_console_run_status_value(),
            inspector_rows=self._console_badge_inspector_rows(inspector_state),
            tool_count=self._console_tool_count(),
            approval_count=self._console_pending_approval_count(),
            can_save_chatbook=inspector_state.can_save_chatbook,
            available_columns=self._console_rail_available_columns(),
        )

    @staticmethod
    def _console_badge_inspector_rows(
        inspector_state: ConsoleInspectorState,
    ) -> tuple[Any, ...]:
        """Return only rows whose blocked state should outrank review badges."""
        return tuple(
            row
            for row in inspector_state.rows
            if str(getattr(row, "label", "")).strip().lower()
            in {"provider", "rag/source", "evidence", "source"}
        )

    def _sync_console_rail_visibility(self, rail_state: ConsoleRailState) -> None:
        """Apply Console rail visibility without recomposing the screen."""
        for selector, label, badge in (
            (
                "#console-context-rail-handle",
                rail_state.left_label,
                rail_state.left_badge,
            ),
            (
                "#console-inspector-rail-handle",
                rail_state.right_label,
                rail_state.right_badge,
            ),
        ):
            try:
                handle = self.query_one(selector, ConsoleRailHandle)
            except QueryError:
                continue
            handle.sync_state(label, badge)

        targets = (
            ("#console-left-rail", rail_state.left_open),
            ("#console-context-rail-handle", not rail_state.left_open),
            ("#console-right-rail", rail_state.right_open),
            ("#console-inspector-rail-handle", not rail_state.right_open),
        )
        for selector, visible in targets:
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            widget.styles.display = "block" if visible else "none"
            widget.display = visible
            self._sync_console_rail_descendant_visibility(widget, visible)

        self.refresh(layout=True)

    @staticmethod
    def _sync_console_rail_descendant_visibility(widget: Any, visible: bool) -> None:
        """Cascade rail display state while preserving child display preferences."""
        for child in widget.query("*"):
            if visible:
                prior_display = getattr(child, "_console_rail_prior_display", None)
                if prior_display is None:
                    continue
                child.display = bool(prior_display)
                child.styles.display = "block" if prior_display else "none"
                delattr(child, "_console_rail_prior_display")
                continue

            if not hasattr(child, "_console_rail_prior_display"):
                setattr(child, "_console_rail_prior_display", bool(child.display))
            child.display = False
            child.styles.display = "none"

    def _current_console_rail_state(self) -> ConsoleRailState:
        """Build the current effective rail state from mounted Console context."""
        pending_launch = self._pending_console_launch_context
        staged_context_state = self._build_console_staged_context_state(pending_launch)
        inspector_state = self._build_console_inspector_state(pending_launch)
        workspace_context_state = self._build_console_workspace_context_state()
        return self._build_console_rail_state(
            staged_context_state=staged_context_state,
            inspector_state=inspector_state,
            workspace_context_state=workspace_context_state,
        )

    def _set_console_rail_preference(
        self,
        *,
        left_open: bool | None = None,
        right_open: bool | None = None,
        notify_on_failure: bool = True,
    ) -> ConsoleRailState:
        """Persist requested Console rail preference changes and return new state."""
        workspace_context = self._current_console_workspace_context()
        preference_key = build_console_rail_preference_key(
            workspace_id=workspace_context.active_workspace_id,
            conversation_id=self._current_console_rail_conversation_id(),
            session_id=self._current_console_session_id(),
        )
        self._migrate_console_rail_fallback_preferences(
            preference_key.value,
            preference_key.fallback_value,
        )
        rail_state_config = self._console_rail_state_config()
        current = coerce_console_rail_preferences(
            rail_state_config.get(preference_key.value)
        )
        next_preferences = ConsoleRailPreferences(
            left_open=current.left_open if left_open is None else bool(left_open),
            right_open=current.right_open if right_open is None else bool(right_open),
        )
        rail_state_config[preference_key.value] = serialize_console_rail_preferences(
            next_preferences
        )
        self._persist_console_rail_preferences(
            preference_key.value,
            next_preferences,
            notify_on_failure=notify_on_failure,
        )
        rail_state = self._current_console_rail_state()
        self._sync_console_rail_visibility(rail_state)
        return rail_state

    def _sync_console_workspace_context(
        self,
        session_data: Optional[ChatSessionData] = None,
    ) -> None:
        try:
            workspace_context = self.query_one(
                "#console-workspace-context",
                ConsoleWorkspaceContextTray,
            )
            workspace_context.sync_state(
                self._build_console_workspace_context_state(session_data)
            )
        except (NoMatches, QueryError):
            logger.debug("No Console workspace context tray available for sync")

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
        _provider_display, model, settings = self._active_console_provider_model_display()
        _effective_settings, settings_readiness = self._active_console_settings_readiness()
        explicit_provider_ready = getattr(self.app_instance, "console_provider_ready", None)
        provider_readiness = get_provider_readiness(
            settings.provider,
            getattr(self.app_instance, "app_config", {}) or {},
        )
        provider_runtime_ready = settings_readiness.native_send_supported and explicit_provider_ready is not False
        model_selected = _has_selected_text(model)
        provider_ready = (
            provider_runtime_ready
            and model_selected
        )
        provider_recovery = ""
        if not provider_ready:
            provider_recovery = (
                "Select a model before sending."
                if provider_runtime_ready and not model_selected
                else "Select a provider and model before sending."
                if explicit_provider_ready is False
                else provider_readiness.user_message
                if provider_readiness.reason == "Missing API key"
                else settings_readiness.detail
            )
        can_save_chatbook = bool(
            getattr(self.app_instance, "console_chatbook_artifact_available", False)
            or self._launch_targets_chatbook_artifact(pending_launch)
        )
        evidence_state = build_console_evidence_display_state(pending_launch)
        return ConsoleInspectorState.from_values(
            live_work_title=pending_launch.title if pending_launch else None,
            provider_ready=provider_ready,
            provider_recovery=provider_recovery,
            rag_status=self._console_rag_source_status(pending_launch),
            evidence_summary=evidence_state.summary if evidence_state else None,
            evidence_status=evidence_state.status if evidence_state else None,
            evidence_recovery=evidence_state.recovery if evidence_state else None,
            evidence_authority=evidence_state.authority if evidence_state else None,
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
        return f"Scope: {', '.join(CONSOLE_LIBRARY_RAG_SOURCE_SCOPE)}"

    @staticmethod
    def _hidden_static(text: str, *, id: str, classes: str = "") -> Static:
        widget = Static(text, id=id, classes=f"{classes} console-hidden-control".strip())
        widget.styles.display = "none"
        widget.styles.height = 0
        widget.styles.min_height = 0
        return widget

    @staticmethod
    def _collapse_console_hidden_control_bar(widget: ConsoleControlBar) -> ConsoleControlBar:
        """Keep the legacy Console control seam mounted without layout cost."""
        widget.styles.display = "none"
        widget.styles.height = 0
        widget.styles.min_height = 0
        widget.styles.max_height = 0
        return widget

    @staticmethod
    def _console_mode_summary(control_state: ConsoleControlState) -> str:
        def readiness_count(label: str) -> str:
            value = label.partition(":")[2].strip()
            return value.split(maxsplit=1)[0] if value else "0"

        persona = str(control_state.persona_label or "Assistant General")
        if persona.startswith("Assistant: "):
            persona = f"Assistant {persona.removeprefix('Assistant: ')}"
        return (
            "Chat | RAG | Run Follow"
            f" | {persona}"
            f" | Sources {readiness_count(control_state.sources_label)}"
            f" | Tools {readiness_count(control_state.tools_label)}"
            f" | Approvals {readiness_count(control_state.approvals_label)}"
        )

    def _console_provider_blocker_copy(self) -> str:
        """Return concise Console recovery copy for provider/model setup gaps."""
        provider, model, settings = self._active_console_provider_model_display()
        if not _has_selected_text(provider):
            return "Provider setup needed: choose a provider"
        if not _has_selected_text(model):
            return "Provider setup needed: choose a model"

        _effective_settings, settings_readiness = self._active_console_settings_readiness()
        if settings_readiness.native_send_supported:
            return ""
        provider_readiness = get_provider_readiness(
            settings.provider,
            getattr(self.app_instance, "app_config", {}) or {},
        )
        if provider_readiness.reason == "Missing API key":
            return f"Provider setup needed: {provider} missing API key"
        return f"Provider setup needed: {settings_readiness.detail}"

    def _console_provider_recovery_action(self) -> tuple[str, str, str]:
        """Return the label, target, and tooltip for Console provider recovery."""
        provider, model, settings = self._active_console_provider_model_display()
        if not _has_selected_text(provider):
            return (
                "Choose provider",
                "console",
                "Choose a provider for this Console session",
            )
        if not _has_selected_text(model):
            return (
                "Choose model",
                "console",
                "Choose a model for this Console session",
            )

        _effective_settings, settings_readiness = self._active_console_settings_readiness()
        if settings_readiness.native_send_supported:
            return ("Open Settings", "hidden", "Open provider settings")

        provider_readiness = get_provider_readiness(
            settings.provider,
            getattr(self.app_instance, "app_config", {}) or {},
        )
        if provider_readiness.reason == "Missing API key":
            return ("Open Settings", "settings", f"Add an API key for {provider}")
        return ("Review settings", "console", "Review this Console session's settings")

    def _console_provider_recovery_strip_visible(self, blocker_copy: str) -> bool:
        """Return whether provider recovery needs a persistent transcript row."""
        if not blocker_copy:
            return False
        action_label, _action_target, _action_tooltip = self._console_provider_recovery_action()
        return action_label not in {"Choose provider", "Choose model"}

    def _console_transcript_has_messages(self) -> bool:
        """Return whether the active Console transcript has user/session content."""
        if self._console_chat_store is not None:
            session_id = self._console_chat_store.active_session_id
            if session_id is not None and self._console_chat_store.messages_for_session(session_id):
                return True

        active_tab = self.chat_state.get_active_tab()
        if active_tab is not None and active_tab.messages:
            return True

        session = self._get_active_chat_session()
        session_data = getattr(session, "session_data", None)
        if coerce_non_negative_int(getattr(session_data, "message_count", 0)):
            return True

        chat_log = self._get_active_chat_log()
        if chat_log is not None:
            for selector in (
                ".message",
                ".console-transcript-system-event",
                "ChatMessageEnhanced",
            ):
                try:
                    if list(chat_log.query(selector)):
                        return True
                except Exception:
                    continue
        return False

    def _console_guidance_visible(self, blocker_copy: str | None = None) -> bool:
        """Return whether first-run Console guidance should still be visible."""
        if self._console_guidance_dismissed:
            return False
        if self._console_transcript_has_messages():
            return False
        if blocker_copy is None:
            blocker_copy = self._console_provider_blocker_copy()
        return not bool(blocker_copy)

    def _dismiss_console_guidance(self) -> None:
        """Hide first-run Console guidance after the user starts composing."""
        if self._console_guidance_dismissed:
            return
        self._console_guidance_dismissed = True
        self._sync_console_transcript_guidance()

    @staticmethod
    def _configure_console_copy_block(
        widget: Static,
        copy: str,
        *,
        visible: bool,
    ) -> None:
        """Update a compact Console status copy block without remounting it."""
        should_show = visible and bool(copy.strip())
        widget.update(copy if should_show else "")
        if should_show:
            row_count = copy.count("\n") + 1
            widget.styles.display = "block"
            widget.styles.height = row_count
            widget.styles.min_height = row_count
            widget.styles.max_height = row_count
            return
        widget.styles.display = "none"
        widget.styles.height = 0
        widget.styles.min_height = 0
        widget.styles.max_height = 0

    def _sync_console_transcript_guidance(self) -> None:
        """Refresh Console onboarding and provider recovery copy in place."""
        blocker_copy = self._console_provider_blocker_copy()
        guidance_visible = self._console_guidance_visible(blocker_copy)
        for selector, copy in (
            ("#console-start-here", CONSOLE_START_HERE_COPY),
            ("#console-action-hints", CONSOLE_ACTION_HINTS_COPY),
        ):
            try:
                widget = self.query_one(selector, Static)
            except QueryError:
                continue
            self._configure_console_copy_block(
                widget,
                copy,
                visible=guidance_visible,
            )

        try:
            surface = self.query_one("#console-session-surface", ConsoleSessionSurface)
        except QueryError:
            pass
        else:
            surface.sync_inline_guidance(
                visible=guidance_visible,
                copy=CONSOLE_INLINE_GUIDANCE_COPY,
            )

        try:
            provider_strip = self.query_one("#console-provider-recovery-strip", Horizontal)
            provider_blocker = self.query_one("#console-provider-blocker", Static)
        except QueryError:
            return
        recovery_visible = self._console_provider_recovery_strip_visible(blocker_copy)
        self._configure_console_provider_recovery_strip(
            provider_strip,
            provider_blocker,
            blocker_copy,
            visible=recovery_visible,
        )
        try:
            settings_button = self.query_one("#console-open-provider-settings", Button)
        except QueryError:
            return
        action_label, _action_target, action_tooltip = self._console_provider_recovery_action()
        self._configure_console_provider_settings_action(
            settings_button,
            visible=recovery_visible,
            label=action_label,
            tooltip=action_tooltip,
        )

    @staticmethod
    def _configure_console_provider_recovery_strip(
        strip: Horizontal,
        blocker: Static,
        copy: str,
        *,
        visible: bool,
    ) -> None:
        """Show provider recovery as one compact warning/action row."""
        strip.styles.height = "auto" if visible else 0
        strip.styles.min_height = 1 if visible else 0
        strip.styles.display = "block" if visible else "none"
        blocker.update(copy if visible else "")
        blocker.styles.display = "block" if visible else "none"
        blocker.styles.width = "1fr"
        blocker.styles.height = "auto" if visible else 0
        blocker.styles.min_height = 1 if visible else 0
        blocker.styles.margin = 0

    @staticmethod
    def _configure_console_provider_settings_action(
        button: Button,
        *,
        visible: bool,
        label: str = "Open Settings",
        tooltip: str = "Open provider settings",
    ) -> None:
        """Show or hide the provider recovery action with the blocker copy."""
        button.label = label or "Open Settings"
        button.tooltip = tooltip
        button.disabled = not visible
        if visible:
            button.styles.display = "block"
            button.styles.height = 1
            button.styles.min_height = 1
            return
        button.styles.display = "none"
        button.styles.height = 0
        button.styles.min_height = 0

    @staticmethod
    def _frame_console_region(
        widget: Any,
        *,
        top: bool = True,
        variant: str = "solid",
    ) -> Any:
        """Apply a visible Textual-native workbench frame."""
        if variant == "quiet":
            widget.add_class("console-frame-quiet")
            widget.styles.border = CONSOLE_QUIET_FRAME_BORDER
            return widget
        widget.add_class("console-frame-solid")
        widget.styles.border = CONSOLE_FRAME_BORDER
        if not top:
            widget.styles.border_top = ("none", CONSOLE_FRAME_COLOR)
        return widget

    @staticmethod
    def _staged_context_frame_variant(state: ConsoleStagedContextState) -> str:
        """Use quiet framing when the staged context tray is only an empty placeholder."""
        return "quiet" if not state.rows and state.summary == "No staged work." else "solid"

    @staticmethod
    def _workspace_context_frame_variant(state: ConsoleWorkspaceContextState) -> str:
        """Use quiet framing when workspace context has no actionable content."""
        has_actions = state.change_workspace_enabled or state.new_conversation_enabled
        return "solid" if state.conversation_rows or has_actions else "quiet"

    def _render_console_live_work_source_readiness(self) -> ComposeResult:
        """Render Console source readiness when no live-work item is staged."""
        acp_status = "not_configured"
        manager = getattr(self.app_instance, "acp_runtime_process_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if callable(snapshot):
            raw_snapshot = snapshot()
            if isinstance(raw_snapshot, dict):
                acp_status = str(raw_snapshot.get("status") or acp_status)
        readiness = ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status(acp_status)
        container = Container(id=readiness.container_id, classes=readiness.container_classes)
        container.styles.height = "auto"
        container.styles.min_height = 0
        with container:
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
        run_button.tooltip = ""

    @on(Input.Submitted)
    async def handle_console_tab_rename_submitted(self, event: Input.Submitted) -> None:
        """Commit inline Console tab rename submissions."""
        input_id = getattr(event.input, "id", None) or ""
        prefix = "console-session-rename-input-"
        if not input_id.startswith(prefix):
            return

        event.stop()
        session_id = input_id.removeprefix(prefix)
        try:
            self._ensure_console_chat_store().rename_session(session_id, event.value)
        except ValueError as exc:
            self.app_instance.notify(str(exc), severity="warning")
        except KeyError:
            self.app_instance.notify("Console tab is no longer available.", severity="error")

        try:
            self.query_one("#console-session-surface", ConsoleSessionSurface).cancel_renaming_session()
        except QueryError:
            pass
        await self._sync_native_console_chat_ui()

    @on(Button.Pressed, "#console-run-library-rag")
    def handle_console_run_library_rag(self, event: Button.Pressed) -> None:
        """Request Library retrieval from the Console source-readiness seam."""
        event.stop()
        query = _sanitize_console_library_rag_query(self._console_library_rag_query)
        if not query:
            self.app_instance.notify(
                CONSOLE_LIBRARY_RAG_QUERY_EMPTY_MESSAGE,
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
        workspace_context_state = self._build_console_workspace_context_state()
        rail_state = self._build_console_rail_state(
            staged_context_state=staged_context_state,
            inspector_state=inspector_state,
            workspace_context_state=workspace_context_state,
        )
        with Vertical(id="console-shell"):
            yield Static(
                "Console | Live agent control, chat, RAG, tools, approvals | Local",
                id="console-title",
                classes="destination-status-row",
            )
            yield self._hidden_static(
                "Agent workbench for chat, source handoffs, live runs, and control actions.",
                id="console-purpose",
                classes="destination-purpose",
            )
            yield self._hidden_static(
                "Console | Agentic control surface | Chat-first | Local runtime",
                id="console-status-row",
                classes="destination-status-row",
            )
            yield Static(
                self._console_mode_summary(control_state),
                id="console-mode-bar",
                classes="ds-panel",
            )
            yield self._collapse_console_hidden_control_bar(
                ConsoleControlBar(
                    control_state,
                    self.app_instance,
                    on_sidebar_toggle_requested=self._toggle_console_chat_sidebar,
                    id="console-control-bar",
                    classes="ds-panel console-hidden-control",
                )
            )
            workspace_grid = self._frame_console_region(
                Horizontal(id="console-workspace-grid", classes="ds-panel destination-workbench")
            )
            with workspace_grid:
                left_handle = ConsoleRailHandle(
                    label=rail_state.left_label,
                    badge=rail_state.left_badge,
                    button_id="console-context-rail-open",
                    badge_id="console-context-rail-badge",
                    side="left",
                    id="console-context-rail-handle",
                )
                left_handle.styles.width = 13
                left_handle.styles.min_width = 13
                left_handle.styles.max_width = 13
                if rail_state.left_open:
                    left_handle.styles.display = "none"
                yield self._frame_console_region(left_handle)

                left_rail = Vertical(
                    id="console-left-rail",
                    classes="console-region destination-workbench-pane",
                )
                left_rail.styles.width = "3fr"
                left_rail.styles.min_width = 30
                if not rail_state.left_open:
                    left_rail.styles.display = "none"
                with self._frame_console_region(left_rail):
                    with Horizontal(classes="console-rail-header"):
                        rail_label = Static(
                            "Context",
                            id="console-context-rail-title",
                            classes="console-rail-title",
                        )
                        rail_label.styles.width = "1fr"
                        yield rail_label
                        collapse_button = Button(
                            "<",
                            id="console-context-rail-collapse",
                            classes="console-rail-collapse-button",
                            compact=True,
                        )
                        collapse_button.tooltip = "Collapse Context rail"
                        collapse_button.styles.width = 3
                        collapse_button.styles.min_width = 3
                        collapse_button.styles.max_width = 3
                        yield collapse_button
                    staged_context_tray = ConsoleStagedContextTray(
                        staged_context_state,
                        id="console-staged-context-tray",
                        classes="console-left-rail-section",
                    )
                    staged_context_tray.styles.width = "100%"
                    staged_context_tray.styles.min_width = 0
                    staged_context_tray.styles.height = "1fr"
                    yield self._frame_console_region(
                        staged_context_tray,
                        variant=self._staged_context_frame_variant(staged_context_state),
                    )

                    settings_summary = ConsoleSettingsSummary(
                        self._build_console_settings_summary_state(),
                        id="console-settings-summary",
                        classes="console-left-rail-section console-settings-summary",
                    )
                    settings_summary.styles.width = "100%"
                    settings_summary.styles.min_width = 0
                    yield self._frame_console_region(settings_summary, variant="quiet")

                    workspace_context_tray = ConsoleWorkspaceContextTray(
                        workspace_context_state,
                        id="console-workspace-context",
                        classes="console-left-rail-section",
                    )
                    workspace_context_tray.styles.width = "100%"
                    workspace_context_tray.styles.min_width = 0
                    workspace_context_tray.styles.height = "2fr"
                    yield self._frame_console_region(
                        workspace_context_tray,
                        variant=self._workspace_context_frame_variant(workspace_context_state),
                    )

                main_column = Vertical(id="console-main-column")
                main_column.styles.width = "13fr"
                main_column.styles.min_width = 60
                with main_column:
                    transcript_region = self._frame_console_region(
                        Vertical(id="console-transcript-region", classes="console-region"),
                        top=False,
                    )
                    with transcript_region:
                        provider_blocker_copy = self._console_provider_blocker_copy()
                        guidance_visible = self._console_guidance_visible(provider_blocker_copy)
                        recovery_visible = self._console_provider_recovery_strip_visible(
                            provider_blocker_copy
                        )
                        provider_action_label, _provider_action_target, provider_action_tooltip = (
                            self._console_provider_recovery_action()
                        )
                        provider_recovery_strip = Horizontal(
                            id="console-provider-recovery-strip",
                            classes="console-provider-recovery-strip",
                        )
                        with provider_recovery_strip:
                            blocker = Static(
                                provider_blocker_copy,
                                id="console-provider-blocker",
                                classes="console-provider-blocker",
                            )
                            self._configure_console_provider_recovery_strip(
                                provider_recovery_strip,
                                blocker,
                                provider_blocker_copy,
                                visible=recovery_visible,
                            )
                            yield blocker
                            provider_settings_action = Button(
                                provider_action_label,
                                id="console-open-provider-settings",
                                classes="destination-action-button console-provider-settings-action",
                                disabled=not recovery_visible,
                                compact=True,
                                variant="primary",
                            )
                            self._configure_console_provider_settings_action(
                                provider_settings_action,
                                visible=recovery_visible,
                                label=provider_action_label,
                                tooltip=provider_action_tooltip,
                            )
                            yield provider_settings_action
                        start_here = Static(
                            CONSOLE_START_HERE_COPY,
                            id="console-start-here",
                            classes="console-start-here",
                        )
                        self._configure_console_copy_block(
                            start_here,
                            CONSOLE_START_HERE_COPY,
                            visible=guidance_visible,
                        )
                        yield start_here
                        action_hints = Static(
                            CONSOLE_ACTION_HINTS_COPY,
                            id="console-action-hints",
                            classes="console-action-hints",
                        )
                        self._configure_console_copy_block(
                            action_hints,
                            CONSOLE_ACTION_HINTS_COPY,
                            visible=guidance_visible,
                        )
                        yield action_hints
                        yield self._ensure_console_session_surface()

                right_rail = Vertical(
                    id="console-right-rail",
                    classes="console-region destination-workbench-pane",
                )
                right_rail.styles.width = "4fr"
                right_rail.styles.min_width = 34
                if not rail_state.right_open:
                    right_rail.styles.display = "none"
                with self._frame_console_region(right_rail):
                    with Horizontal(classes="console-rail-header"):
                        rail_label = Static(
                            "Inspector",
                            id="console-inspector-rail-title",
                            classes="console-rail-title",
                        )
                        rail_label.styles.width = "1fr"
                        yield rail_label
                        collapse_button = Button(
                            ">",
                            id="console-inspector-rail-collapse",
                            classes="console-rail-collapse-button",
                            compact=True,
                        )
                        collapse_button.tooltip = "Collapse Inspector rail"
                        collapse_button.styles.width = 3
                        collapse_button.styles.min_width = 3
                        collapse_button.styles.max_width = 3
                        yield collapse_button
                    with Vertical(id="console-run-inspector"):
                        yield ConsoleRunInspector(
                            inspector_state,
                            id="console-run-inspector-state",
                        )
                    if pending_launch:
                        yield from self._render_console_live_work_status_card(pending_launch)
                    else:
                        yield from self._render_console_live_work_source_readiness()

                right_handle = ConsoleRailHandle(
                    label=rail_state.right_label,
                    badge=rail_state.right_badge,
                    button_id="console-inspector-rail-open",
                    badge_id="console-inspector-rail-badge",
                    side="right",
                    id="console-inspector-rail-handle",
                )
                right_handle.styles.width = 9
                right_handle.styles.min_width = 9
                right_handle.styles.max_width = 9
                if rail_state.right_open:
                    right_handle.styles.display = "none"
                yield self._frame_console_region(right_handle, variant="quiet")
            yield self._frame_console_region(
                ConsoleComposerBar(
                    id="console-native-composer",
                    classes="ds-panel",
                    collapse_large_pastes=self._console_collapse_large_pastes_enabled(),
                )
            )

    def _console_collapse_large_pastes_enabled(self) -> bool:
        """Return the app-level Console paste-collapse preference."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        console_config = app_config.get("console", {})
        if not isinstance(console_config, dict):
            return True
        return coerce_bool_setting(console_config.get("collapse_large_pastes", True), True)
    
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
        self._focus_console_composer_if_needed(force=True)
        self.call_after_refresh(self._sync_native_console_chat_ui)
        self.call_after_refresh(lambda: self._focus_console_composer_if_needed(force=True))
        self.set_timer(0.2, lambda: self._focus_console_composer_if_needed(force=True))

    async def on_unmount(self) -> None:
        """Release Console-native resources owned by this screen."""
        self._stop_console_transcript_sync_timer()
        gateway = self._console_provider_gateway
        close = getattr(gateway, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        super().on_unmount()
    
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

    async def _append_console_system_event(self, message: str) -> None:
        """Append a Console-native status event without legacy chat message chrome."""
        try:
            session = self._get_active_chat_session()
            if session is None:
                raise QueryError("No active Console chat session")
            await session.get_chat_log().mount(
                Static(
                    Text(message),
                    classes="console-transcript-system-event",
                )
            )
            self._sync_console_transcript_guidance()
        except Exception:
            self.app_instance.notify(message, severity="warning")

    def _native_console_messages(self) -> list[Any]:
        """Return messages for the active native Console session."""
        store = self._ensure_console_chat_store()
        if store.active_session_id is None:
            return []
        return store.messages_for_session(store.active_session_id)

    async def _sync_native_console_transcript_to_legacy_surface(self) -> None:
        """Temporary bridge: render native Console messages in the existing surface."""
        try:
            transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        except QueryError:
            transcript = None

        if transcript is not None:
            transcript.set_messages(self._native_console_messages())
            await transcript.refresh_messages()
            self._sync_console_transcript_guidance()
            return

        chat_log = self._get_active_chat_log()
        if chat_log is None:
            return

        messages = self._native_console_messages()
        chat_log.remove_children()
        for message in messages:
            role_label = str(message.role.value if hasattr(message.role, "value") else message.role)
            content = message.content
            if message.status in {"streaming", "stopped", "failed"}:
                content = f"{content} [{message.status}]".strip()
            await chat_log.mount(
                Static(
                    Text(f"{role_label.title()}: {content}"),
                    classes=(
                        "console-transcript-system-event "
                        f"console-native-message console-native-message-{role_label}"
                    ),
                )
            )
        self._sync_console_transcript_guidance()

    def _native_run_status_copy(self) -> str:
        controller = self._console_chat_controller
        if controller is None:
            return ""
        run_state = controller.run_state
        if run_state.status is ConsoleRunStatus.IDLE:
            return ""
        return run_state.visible_copy or run_state.status.value

    def _sync_console_mode_bar(self) -> None:
        try:
            mode_bar = self.query_one("#console-mode-bar", Static)
        except QueryError:
            return
        control_state = self._build_console_control_state(self._pending_console_launch_context)
        mode_copy = self._console_mode_summary(control_state)
        if run_status := self._native_run_status_copy():
            mode_copy = f"{mode_copy} | Run: {run_status}"
        mode_bar.update(mode_copy)

    async def _sync_native_console_chat_ui(self) -> None:
        """Refresh visible Console-native state after send/stop transitions."""
        if self._console_sync_in_progress:
            return
        self._console_sync_in_progress = True
        try:
            self._sync_console_chat_core_state()
            self._sync_console_control_bar()
            self._sync_console_settings_summary()
            self._sync_console_mode_bar()
            await self._sync_console_native_session_tabs()
            await self._sync_native_console_transcript_to_legacy_surface()
            self._sync_console_rail_visibility(self._current_console_rail_state())
        finally:
            self._console_sync_in_progress = False

    async def _sync_console_native_session_tabs(self) -> None:
        """Refresh native Console session tabs from store state."""
        try:
            surface = self.query_one("#console-session-surface", ConsoleSessionSurface)
        except QueryError:
            return
        store = self._ensure_console_chat_store()
        store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id,
            settings=self._default_console_session_settings(),
        )
        self._ensure_active_console_session_settings()
        await surface.sync_sessions(
            sessions=store.sessions(),
            active_session_id=store.active_session_id,
        )

    async def _append_native_console_system_message(self, message: str) -> None:
        """Append a system message to native Console state and refresh the bridge."""
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.SYSTEM,
            content=message,
        )
        await self._sync_native_console_chat_ui()

    def _start_console_transcript_sync_timer(self) -> None:
        if self._console_transcript_sync_timer is not None:
            return

        async def _poll_transcript() -> None:
            await self._sync_native_console_chat_ui()
            controller = self._console_chat_controller
            active_statuses = {
                ConsoleRunStatus.VALIDATING,
                ConsoleRunStatus.RETRYING,
                ConsoleRunStatus.STREAMING,
            }
            if controller is None or controller.run_state.status not in active_statuses:
                self._stop_console_transcript_sync_timer()

        self._console_transcript_sync_timer = self.set_interval(0.2, _poll_transcript)

    def _stop_console_transcript_sync_timer(self) -> None:
        if self._console_transcript_sync_timer is None:
            return
        try:
            self._console_transcript_sync_timer.stop()
        finally:
            self._console_transcript_sync_timer = None

    async def _submit_console_native_draft(self, draft: str) -> None:
        controller = self._ensure_console_chat_controller()
        self._start_console_transcript_sync_timer()
        result = await controller.submit_draft(draft)
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            composer = None
        if result.should_clear_draft and composer is not None:
            composer.clear_draft()
        await self._sync_native_console_chat_ui()

    def _console_send_blocked_reason(self) -> str:
        """Return a user-facing reason if Console send cannot safely run."""
        pending_launch = self._consume_pending_console_launch()
        if pending_launch is not None and _source_mentions_rag(pending_launch.source):
            evidence_state = build_console_evidence_display_state(pending_launch)
            if evidence_state is None or evidence_state.available_count == 0:
                return (
                    "Console send blocked: Library Search/RAG has no available evidence. "
                    "Review source authority before sending."
                )
        _readiness_settings, readiness = self._active_console_settings_readiness()
        if not readiness.native_send_supported:
            return f"Console send blocked: {readiness.detail}"
        return ""

    async def handle_console_send_message(self, event: Button.Pressed) -> None:
        """Route the Console composer send action through the native controller."""
        event.stop()
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
            draft = composer.draft_text()
        except QueryError:
            draft = ""
        if not draft.strip():
            self._focus_console_composer_if_needed(force=True)
            return
        self._dismiss_console_guidance()
        if blocked_reason := self._console_send_blocked_reason():
            await self._append_native_console_system_message(blocked_reason)
            self._focus_console_composer_if_needed(force=True)
            return
        controller = self._ensure_console_chat_controller()
        if not controller.run_state.is_send_allowed:
            self.app_instance.notify("A Console run is already running.", severity="warning")
            return
        self.run_worker(self._submit_console_native_draft(draft), exclusive=True)

    async def handle_console_stop_generation(self, event: Button.Pressed) -> None:
        """Route the Console stop action through native run control."""
        event.stop()
        controller = self._ensure_console_chat_controller()
        if not controller.stop_active_run():
            self.app_instance.notify("No active Console run to stop.", severity="warning")
        await self._sync_native_console_chat_ui()

    @on(Button.Pressed, "#console-attach-context")
    async def handle_console_attach_context(self, event: Button.Pressed) -> None:
        """Route the Console attach affordance through the active chat session adapter."""
        await self._handle_console_attach_context(event)

    @on(Button.Pressed, "#console-staged-context-attach")
    async def handle_console_staged_context_attach(self, event: Button.Pressed) -> None:
        """Route the staged-context empty-state attach action through the same adapter."""
        await self._handle_console_attach_context(event)

    async def _handle_console_attach_context(self, event: Button.Pressed) -> None:
        """Route Console attach actions through the active chat session adapter."""
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
        """Route available Chatbook artifacts through the existing Artifacts handoff."""
        event.stop()
        launch = self._consume_pending_console_launch()
        if self._launch_targets_chatbook_artifact(launch):
            handler = getattr(self.app_instance, "open_console_live_work_primary_action", None)
            if callable(handler) and bool(handler(launch)):
                return
        self.app_instance.notify(
            "No Chatbook artifact is available to save yet.",
            severity="warning",
        )

    @on(Button.Pressed, "#console-open-provider-settings")
    async def handle_console_open_provider_settings(self, event: Button.Pressed) -> None:
        """Route provider setup recovery to the smallest relevant settings surface."""
        event.stop()
        _label, target, _tooltip = self._console_provider_recovery_action()
        if target == "console" and getattr(self, "is_mounted", False):
            await self.on_console_settings_open(event)
            return
        self.post_message(NavigateToScreen(TAB_SETTINGS))

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

    async def handle_console_message_action(self, event: Button.Pressed) -> bool:
        """Route selected transcript message actions through the native action service."""
        button_id = event.button.id or ""
        action_id, message_id = self._parse_console_message_action_button_id(button_id)
        if action_id is None or message_id is None:
            return False

        event.stop()
        store = self._ensure_console_chat_store()
        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify("Console message action target no longer exists.", severity="warning")
            return True

        if action_id == "save-as":
            await self.app_instance.push_screen(
                ConsoleSaveAsModal(
                    destinations=self._console_message_action_service.save_as_destinations(message)
                )
            )
            self._last_console_action = ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Opened Save as destinations.",
            )
            return True

        result = self._console_message_action_service.dispatch(action_id, message)
        self._last_console_action = result
        if result.clipboard_text is not None:
            copy_to_clipboard = getattr(self.app_instance, "copy_to_clipboard", None)
            if callable(copy_to_clipboard):
                copy_to_clipboard(result.clipboard_text)
        if action_id == "retry" and result.status == "completed":
            controller = self._ensure_console_chat_controller()
            self.run_worker(self._retry_console_message(controller, message_id), exclusive=True)
            return True
        if action_id == "regenerate" and result.status == "wip":
            controller = self._ensure_console_chat_controller()
            self.run_worker(self._regenerate_console_message(controller, message_id), exclusive=True)
            return True
        if action_id in {"variant-previous", "variant-next"} and result.status == "completed":
            self._select_console_message_variant(message_id, direction=action_id)
            await self._sync_native_console_chat_ui()
            return True
        if action_id == "continue" and result.status == "continue_requested":
            controller = self._ensure_console_chat_controller()
            self.run_worker(self._continue_console_message(controller, message_id), exclusive=True)
            return True
        severity = "information" if result.status in {"completed", "wip"} else "warning"
        self.app_instance.notify(result.visible_copy, severity=severity)
        return True

    @staticmethod
    def _parse_console_message_action_button_id(button_id: str) -> tuple[str | None, str | None]:
        prefixes = (
            ("console-message-action-feedback-up-", "feedback-up"),
            ("console-message-action-feedback-down-", "feedback-down"),
            ("console-message-action-variant-previous-", "variant-previous"),
            ("console-message-action-variant-next-", "variant-next"),
            ("console-message-action-save-as-", "save-as"),
            ("console-message-action-regenerate-", "regenerate"),
            ("console-message-action-continue-", "continue"),
            ("console-message-action-delete-", "delete"),
            ("console-message-action-retry-", "retry"),
            ("console-message-action-copy-", "copy"),
            ("console-message-action-edit-", "edit"),
        )
        for prefix, action_id in prefixes:
            if button_id.startswith(prefix):
                return action_id, button_id.removeprefix(prefix)
        return None, None

    async def _retry_console_message(
        self,
        controller: ConsoleChatController,
        message_id: str,
    ) -> None:
        result = await controller.retry_message(message_id)
        if result.visible_copy:
            severity = "warning" if not result.accepted else "information"
            self.app_instance.notify(result.visible_copy, severity=severity)
        await self._sync_native_console_chat_ui()

    async def _continue_console_message(
        self,
        controller: ConsoleChatController,
        message_id: str,
    ) -> None:
        result = await controller.continue_from_message(message_id)
        if result.visible_copy and not result.accepted:
            self.app_instance.notify(result.visible_copy, severity="warning")
        await self._sync_native_console_chat_ui()

    async def _regenerate_console_message(
        self,
        controller: ConsoleChatController,
        message_id: str,
    ) -> None:
        result = await controller.regenerate_message(message_id)
        if result.visible_copy and not result.accepted:
            self.app_instance.notify(result.visible_copy, severity="warning")
        await self._sync_native_console_chat_ui()

    def _select_console_message_variant(self, message_id: str, *, direction: str) -> None:
        store = self._ensure_console_chat_store()
        message = store.get_message(message_id)
        if message.variants is None:
            return
        selected_index = message.variants.selected_index
        if direction == "variant-previous":
            selected_index -= 1
        elif direction == "variant-next":
            selected_index += 1
        store.select_variant(message_id, selected_index)

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
        self._sync_console_transcript_guidance()
        try:
            inspector = self.query_one("#console-run-inspector-state", ConsoleRunInspector)
        except QueryError:
            inspector = None
        inspector_state = self._build_console_inspector_state(
            self._pending_console_launch_context
        )
        if inspector is not None:
            inspector.sync_state(inspector_state)
        self._sync_console_composer_action_state(
            can_save_chatbook=inspector_state.can_save_chatbook
            and self._console_chatbook_action_available()
        )
        self._sync_console_rail_visibility(self._current_console_rail_state())

    def _console_chatbook_action_available(self) -> bool:
        """Return True when the composer Chatbook action has a real target."""
        return (
            self._launch_targets_chatbook_artifact(self._pending_console_launch_context)
            and callable(
                getattr(self.app_instance, "open_console_live_work_primary_action", None)
            )
        )

    def _sync_console_composer_action_state(
        self,
        *,
        can_save_chatbook: bool,
    ) -> None:
        """Refresh Console composer action priority from draft, run, and artifact state."""
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return

        run_active = False
        send_blocked = False
        controller = self._console_chat_controller
        if controller is not None:
            run_state = getattr(controller, "run_state", None)
            run_active = bool(getattr(run_state, "is_stop_allowed", False))
            send_blocked = not bool(getattr(run_state, "is_send_allowed", True))

        composer.sync_action_state(
            has_draft=bool(composer.draft_text().strip()),
            run_active=run_active,
            can_save_chatbook=can_save_chatbook,
            send_blocked=send_blocked,
        )

    def _hide_console_legacy_chat_inputs(self) -> None:
        """Keep Console on a single native composer surface."""
        for widget in self.query(".chat-input-area"):
            widget.styles.display = "none"
            widget.styles.height = 0
            widget.styles.min_height = 0
            widget.disabled = True
        for widget in self.query(".chat-input"):
            widget.styles.display = "none"
            widget.styles.height = 0
            widget.styles.min_height = 0
            widget.disabled = True
            widget.can_focus = False

    def _focus_console_composer_if_needed(self, *, force: bool = False) -> None:
        """Route typing to the visible Console composer instead of hidden chat input."""
        self._hide_console_legacy_chat_inputs()
        focused = self.app.focused
        if not force and focused is not None and not (
            focused.id == "chat-input"
            or (focused.id or "").startswith("chat-input-")
            or focused.has_class("chat-input")
        ):
            return
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
            composer.focus()
        except QueryError:
            return

    @staticmethod
    def _is_legacy_chat_input_focus(focused: object | None) -> bool:
        """Return True when focus is on a hidden legacy chat input."""
        if focused is None:
            return False
        focused_id = getattr(focused, "id", None) or ""
        has_class = getattr(focused, "has_class", lambda _class_name: False)
        return (
            focused_id == "chat-input"
            or focused_id.startswith("chat-input-")
            or has_class("chat-input")
        )

    @staticmethod
    def _is_descendant_or_self(widget: object | None, ancestor: object) -> bool:
        """Return True when widget is ancestor or contained by ancestor."""
        current = widget
        while current is not None:
            if current is ancestor:
                return True
            current = getattr(current, "parent", None)
        return False

    def _should_capture_console_input(self, composer: ConsoleComposerBar) -> bool:
        """Return True when key/paste input should route to the Console composer."""
        focused = self.app.focused
        if focused is None:
            return True
        return self._is_descendant_or_self(
            focused,
            composer,
        ) or self._is_legacy_chat_input_focus(focused)

    def _cancel_console_tab_rename_if_focused(self) -> bool:
        """Cancel inline Console tab rename when Esc is pressed in the rename input."""
        focused_id = getattr(self.app.focused, "id", None) or ""
        if not focused_id.startswith("console-session-rename-input-"):
            return False
        try:
            surface = self.query_one("#console-session-surface", ConsoleSessionSurface)
        except QueryError:
            return False
        surface.cancel_renaming_session()
        self.run_worker(self._sync_native_console_chat_ui(), exclusive=True)
        return True

    def on_key(self, event: Key) -> None:
        """Treat the Console composer as the default printable text target."""
        if event.key == "escape" and self._cancel_console_tab_rename_if_focused():
            event.stop()
            event.prevent_default()
            return
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        if not self._should_capture_console_input(composer):
            return
        if event.key in {"backspace", "ctrl+h"}:
            composer.delete_left()
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            try:
                self.query_one("#console-send-message", Button).press()
            except QueryError:
                self.app_instance.notify("Console send is unavailable.", severity="error")
            return
        if event.key == "ctrl+u":
            composer.clear_draft()
            event.stop()
            event.prevent_default()
            return
        if event.is_printable and event.character is not None:
            composer.insert_text(event.character)
            self._dismiss_console_guidance()
            event.stop()
            event.prevent_default()

    def on_paste(self, event: Paste) -> None:
        """Treat pasted text as Console composer draft input by default."""
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        if not self._should_capture_console_input(composer):
            return
        composer.insert_pasted_text(event.text)
        self._dismiss_console_guidance()
        event.stop()

    def on_click(self, event: Click) -> None:
        """Reset pending paste unfurl confirmation when clicking outside the token."""
        target = getattr(event, "widget", None) or getattr(event, "control", None)
        if getattr(target, "id", None) == "console-command-visible-text":
            return
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        composer.reset_pending_unfurl()

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

        try:
            settings = self._ensure_active_console_session_settings()
            next_settings = settings
            if provider is not None:
                next_settings = replace(next_settings, provider=str(provider).strip())
            if model is not None:
                model_value = str(model).strip() if _has_selected_text(model) else None
                next_settings = replace(next_settings, model=model_value)
            if temperature is not None:
                try:
                    next_settings = replace(
                        next_settings,
                        temperature=float(str(temperature).strip()),
                    )
                except (TypeError, ValueError):
                    logger.debug("Ignoring invalid Console temperature sync value")
            if next_settings != settings:
                self._replace_active_console_session_settings(next_settings)
        except Exception as e:
            logger.debug(f"Unable to sync compact controls into Console session settings: {e}")

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
        self._sync_console_workspace_context(session_data)
        shell_bar = self._get_shell_bar()
        if not shell_bar:
            self._hide_console_legacy_chat_inputs()
            try:
                composer = self.query_one("#console-native-composer", ConsoleComposerBar)
                composer.sync_session_data(session_data)
                session = self._get_active_chat_session()
                draft = session.get_chat_input().text if session is not None else ""
                if not composer.draft_text():
                    composer.load_draft(draft)
                self._focus_console_composer_if_needed()
            except QueryError:
                pass
            self._sync_console_transcript_guidance()
            logger.debug("No shell bar available for live session sync")
            return

        try:
            shell_bar.sync_from_session_data(session_data)
            self._hide_console_legacy_chat_inputs()
            try:
                composer = self.query_one("#console-native-composer", ConsoleComposerBar)
                session = self._get_active_chat_session()
                draft = session.get_chat_input().text if session is not None else ""
                composer.sync_session_data(session_data)
                if not composer.draft_text():
                    composer.load_draft(draft)
                self._focus_console_composer_if_needed()
            except (QueryError, NoMatches):
                pass
            if session_data is None:
                logger.debug("Synced shell bar from cleared live session")
            else:
                logger.debug(
                    "Synced shell bar from live session {}",
                    getattr(session_data, "tab_id", None),
                )
            self._sync_console_transcript_guidance()
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

        if button_id == "console-send-message":
            await self.handle_console_send_message(event)
            return
        if button_id == "console-stop-generation":
            await self.handle_console_stop_generation(event)
            return
        if button_id == "console-settings-open":
            await self.on_console_settings_open(event)
            return
        if button_id == "console-new-chat-tab":
            event.stop()
            self._ensure_console_chat_controller().new_session(
                settings=self._default_console_session_settings(),
            )
            await self._sync_native_console_chat_ui()
            return
        if button_id and button_id.startswith("console-rename-session-tab-"):
            event.stop()
            session_id = button_id.removeprefix("console-rename-session-tab-")
            try:
                self._ensure_console_chat_controller().switch_session(session_id)
                surface = self.query_one("#console-session-surface", ConsoleSessionSurface)
            except (KeyError, QueryError):
                self.app_instance.notify("Console tab is no longer available.", severity="error")
                return
            surface.start_renaming_session(session_id)
            await self._sync_native_console_chat_ui()
            surface.focus_rename_input()
            return
        if button_id and button_id.startswith("console-close-session-tab-"):
            event.stop()
            session_id = button_id.removeprefix("console-close-session-tab-")
            store = self._ensure_console_chat_store()
            try:
                messages = store.messages_for_session(session_id)
            except KeyError:
                messages = []
            if messages:
                from ...Widgets.confirmation_dialog import ConfirmationDialog

                async def _do_close() -> None:
                    self._ensure_console_chat_controller().close_session(session_id)
                    await self._sync_native_console_chat_ui()

                dialog = ConfirmationDialog(
                    title="Close Tab",
                    message="This tab has messages that will be lost.\n\nClose it anyway?",
                    confirm_label="Close",
                    cancel_label="Keep",
                    confirm_callback=_do_close,
                )
                self.app.push_screen(dialog)
            else:
                self._ensure_console_chat_controller().close_session(session_id)
                await self._sync_native_console_chat_ui()
            return
        if button_id and button_id.startswith("console-session-tab-"):
            event.stop()
            session_id = button_id.removeprefix("console-session-tab-")
            self._ensure_console_chat_controller().switch_session(session_id)
            await self._sync_native_console_chat_ui()
            return
        if button_id and button_id.startswith("console-message-action-"):
            handled = await self.handle_console_message_action(event)
            if handled:
                return
        
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
