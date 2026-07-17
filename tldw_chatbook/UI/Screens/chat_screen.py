"""Chat screen implementation with comprehensive state management."""

from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import datetime
import asyncio
import inspect
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Literal, Optional, TYPE_CHECKING
import uuid

import toml
from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.events import Click, Key, MouseUp, Paste
from textual.message_pump import NoActiveAppError
from textual.reactive import reactive
from textual.widgets import Button, Static, TextArea, Select, Collapsible, Input

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from .chat_screen_state import ChatScreenState, TabState, MessageData, TaskResumeState
from .provider_model_resolution import (
    ResolvedProviderModelOption,
    resolve_effective_provider_model,
    resolve_provider_model_options,
)
from .settings_config_models import SettingsCategoryId
from ...Chat.chat_conversation_service import derive_conversation_title
from ...Chat.chat_persistence_service import ChatPersistenceService
from ...Chat.console_chat_controller import ConsoleChatController
from ...Event_Handlers.Chat_Events.chat_events_console_dictionaries import (
    console_attachable_dictionaries,
    console_attached_dictionaries,
    handle_console_dictionary_attach,
    handle_console_dictionary_detach,
)
from ...Chat.console_command_grammar import (
    KIND_COMMAND,
    KIND_FALLBACK,
    KIND_NOT_COMMAND,
    KIND_UNKNOWN,
    PROMPT_COMMAND_HANDLER_ID,
    PROMPT_COMMAND_NAME,
    SKILLS_COMMAND_HANDLER_ID,
    SKILLS_COMMAND_NAME,
    SYSTEM_COMMAND_HANDLER_ID,
    SYSTEM_COMMAND_NAME,
    CommandParse,
    ConsoleCommandRegistry,
    default_console_registry,
)
from ...Chat.console_skill_resolver import (
    SKILL_UNTRUSTED_REFUSE,
    SkillCommandCandidate,
    cap_skill_args,
    format_skills_list,
    make_skill_fallback_resolver,
    resolve_skill_command,
)
from ...Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    DEFAULT_CONSOLE_SESSION_TITLE,
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunStatus,
    ConsoleVariant,
    ConsoleVariantSet,
    ConsoleWorkspaceContext,
    ConsoleStagedSource,
    MessageAttachment,
)
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
    build_console_context_estimate,
    build_console_model_section_lines,
    build_console_rail_system_line,
    build_default_console_session_settings,
    build_console_settings_readiness,
    build_console_settings_summary_state,
)
from ...Chat.console_chat_store import (
    MAX_PENDING_ATTACHMENTS,
    ConsoleChatSession,
    ConsoleChatStore,
)
from ...Chat.console_provider_gateway import (
    DEFAULT_LLAMACPP_BASE_URL,
    ConsoleProviderGateway,
    normalize_llamacpp_base_url,
)
from ...Chat.console_provider_endpoints import first_configured_endpoint
from ...Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    CONSOLE_INSPECTOR_NO_TOOL_CALLS_REASON,
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
    ConsoleControlState,
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
    ConsoleStagedContextState,
    build_console_evidence_display_state,
    coerce_non_negative_int,
)
from ...Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    build_console_detected_server_action,
    build_console_setup_card_state,
    coerce_console_first_send_completed,
)
from ...Chat.local_server_discovery import (
    DiscoveredLocalServer,
    discover_local_servers,
)
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Chat.chat_models import ChatSessionData
from ...Chat.provider_readiness import get_provider_readiness, provider_config_key
from ...Chat.console_message_actions import ConsoleActionResult, ConsoleMessageActionService
from ...Chat.console_save_targets import (
    console_chatbook_artifact_payload,
    derive_console_save_title,
)
from ...Chat.console_live_work import (
    ConsoleLiveWorkLaunch,
    ConsoleLiveWorkSourceReadinessState,
    ConsoleLiveWorkStatusCardState,
)
from ...Chat.console_glyphs import GLYPH_COLLAPSE_LEFT, GLYPH_COLLAPSE_RIGHT
from ...Chat.console_image_view import (
    IMAGE_CACHE_MAX_ENTRIES,
    ConsoleImageRenderCache,
    ConsoleImageRowSpec,
    ConsoleImageViewState,
    next_view_mode,
    resolve_default_mode,
)
from ...Chat.console_paste_attach import (
    extract_dropped_path,
    grab_clipboard_image,
    looks_attachable,
)
from ...Chat.console_rail_state import (
    CONSOLE_RAIL_SECTION_IDS,
    ConsoleRailPreferences,
    ConsoleRailState,
    build_console_rail_preference_key,
    build_console_rail_state,
    coerce_console_rail_preferences,
    collect_prunable_console_rail_keys,
    serialize_console_rail_preferences,
)
from ...config import (
    DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    coerce_bool_setting,
    coerce_int_setting,
    delete_settings_from_cli_config,
    get_cli_providers_and_models,
    get_cli_setting,
    load_settings,
    save_setting_to_cli_config,
    save_settings_to_cli_config,
)
from ...Library.library_prompts_state import classify_prompt_save_error
from ...Library.library_rag_service import (
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from ...Notes.notes_scope_service import ScopeType
from ...Constants import TAB_SETTINGS
from ...Utils.chat_diagnostics import ChatDiagnostics
from ...Utils.console_background_effects import (
    ConsoleBackgroundEffectSettings,
    normalize_console_background_effects,
)
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...UI.Workbench import (
    CommandStrip,
    DestinationHeader,
    ModeStrip,
    RecoveryCallout,
    WorkbenchActionRequested,
    WorkbenchHelpPanel,
    WorkbenchHelpState,
)
from ...UI.Workbench.focus import WorkbenchFocusRegistry
from ...state.ui_state import UIState
from ...Widgets.AppFooterStatus import AppFooterStatus
from ...Widgets.Chat_Widgets.chat_tab_container import ChatTabContainer
from ...Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from ...Widgets.Console import (
    ConsoleComposerBar,
    ConsoleControlBar,
    ConsoleEditMessageModal,
    ConsoleRailHandle,
    ConsoleRenameSessionModal,
    ConsoleRunInspector,
    ConsoleSaveAsModal,
    ConsoleSessionSurface,
    ConsoleSettingsModal,
    ConsoleSettingsSummary,
    ConsoleSetupModal,
    ConsoleStagedContextTray,
    ConsoleTranscript,
    ConsoleWorkspaceContextTray,
    ConsoleWorkspaceSwitcherModal,
)
from ...Widgets.Console.console_model_popover import (
    CONSOLE_POPOVER_OPEN_FULL_SETTINGS,
    ConsoleModelPopover,
)
from ...Widgets.Console.console_prompt_picker_modal import (
    MODE_APPLY_SYSTEM as CONSOLE_PROMPT_PICKER_MODE_APPLY_SYSTEM,
    ConsolePromptPickerModal,
)
from ...Widgets.Console.console_skill_picker_modal import ConsoleSkillPickerModal
from ...Widgets.Console.console_system_prompt_modal import ConsoleSystemPromptModal
from ...Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION,
)
from ...Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)
from ...Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
)
from ...Widgets.Console.console_workspace_details import ConsoleWorkspaceDetailsTray
from ...Widgets.Console.console_workbench_state import build_console_workbench_state
from ...Workspaces.display_state import (
    CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
    ConsoleWorkspaceConversationRow,
    ConsoleWorkspaceConversationSectionState,
    ConsoleWorkspaceContextState,
    build_console_workspace_state,
    console_workspace_conversation_result_copy,
)
from ...Workspaces import (
    CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    ConsoleConversationBrowserInputRow,
    ConsoleConversationBrowserRow,
    DEFAULT_WORKSPACE_ID,
    WorkspaceRecord,
    build_console_conversation_browser_state,
)
from ...Widgets.compact_model_bar import CompactModelBar
from ...Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker
from ..Views.RAGSearch.search_handoff import build_library_rag_console_live_work_payload

# Import the existing chat window to reuse its functionality
from ..Chat_Window_Enhanced import ChatWindowEnhanced
from ...Widgets.voice_input_widget import VoiceInputMessage

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="ChatScreen")
CONSOLE_RUN_ALREADY_RUNNING_COPY = "A Console run is already running."
CONSOLE_LIBRARY_RAG_SOURCE_SCOPE = ("notes", "media", "conversations")
CONSOLE_LIBRARY_RAG_RECOVERY_COPY = "Review citations before sending."
CONSOLE_LIBRARY_RAG_QUERY_MAX_LENGTH = 2_000
CONSOLE_LIBRARY_RAG_QUERY_EMPTY_MESSAGE = "Type a Library RAG query before running retrieval."
CONSOLE_FRAME_COLOR = "#6f7782"
CONSOLE_FRAME_BORDER = ("solid", CONSOLE_FRAME_COLOR)
CONSOLE_QUIET_FRAME_BORDER = ("none", CONSOLE_FRAME_COLOR)
CONSOLE_START_HERE_COPY = ""
CONSOLE_ACTION_HINTS_COPY = ""
# Mirrors `library_screen.LIBRARY_PROMPT_SAVE_STATUS_COPY` verbatim: the
# Console `/system` editor's "Save to Library" action is always a brand-new
# prompt CREATE (never an update to an existing one, unlike the Library
# prompt editor's own save flow), but the outcome copy the user sees must
# read identically either way -- duplicated here rather than imported across
# screens to avoid a Screen-to-Screen import.
CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY = {
    "ok": "Saved.",
    "name-in-use": "Name already in use — pick another or open the existing prompt.",
    "soft-deleted-name": "A deleted prompt holds this name — restore it or choose another.",
    "error": "Couldn't save this prompt. Try again.",
}
CONSOLE_SYSTEM_PROMPT_NO_SYSTEM_PART_TEMPLATE = 'Prompt "{name}" has no system part.'
# Task 9 (Skills Console dispatch): a resolved-single skill run's raw command
# is submitted as the actual user turn (Task 10 renders the substitution at
# payload build); this TOOL-role marker is appended once that submit is
# ACCEPTED (the USER message has actually landed in the store -- see
# `_on_console_submission_accepted`, never right after `run_worker` merely
# *schedules* the submit, which raced the scheduled coroutine and could
# append this marker before the user turn it is meant to follow) so the
# transcript records which skill is "driving" the turn.
CONSOLE_SKILL_RUN_MARKER_TEMPLATE = "skill {name} → driving this turn"
# "Absent" bucket of the untrusted-refuse copy (Task 7's SKILL_UNTRUSTED_REFUSE):
# a typed skill name that matches nothing at all -- not a trusted candidate,
# not even a needs-review one.
CONSOLE_SKILL_RUN_REFUSE_REASON_ABSENT = "no matching skill"
# Review-mandated addition (Task 8 review, binding): a typed prefix that
# matches ONLY needs-review (trust-blocked) skills must not read like the
# generic "nothing found" empty state -- it must say so explicitly rather
# than silently opening a picker that can only ever show an empty list (the
# picker's `skill_search` closure is scoped to trusted skills only).
CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE = (
    "{count} matching skill(s) need review in Library ▸ Skills before running."
)
CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL = "Set up provider"
CONSOLE_PROVIDER_ACTION_ARROW = " ---------------------->"
NATIVE_CONSOLE_STATE_VERSION = "1.0"
# Roleplay P1h: bounds passed to `Chat_Dictionary_Lib.apply_active_chatdicts_to_text`
# for the native Console send-path applier (`_console_chat_dictionary_applier`).
_CHATDICT_MAX_TOKENS = 500
_CHATDICT_STRATEGY = "sorted_evenly"
# Statuses during which the 0.2s transcript poll is actively ticking
# (see `_start_console_transcript_sync_timer`) -- also used by the
# sub-agent badge-count cache (Finding A) to decide whether a live run
# justifies an eager re-count.
CONSOLE_ACTIVE_RUN_STATUSES = (
    ConsoleRunStatus.VALIDATING,
    ConsoleRunStatus.RETRYING,
    ConsoleRunStatus.STREAMING,
)
# Plan-B Task 7 Finding A: the conversation-browser `[N Sub-Agents]` badge
# count previously re-queried the DB once per visible row on every 0.2s
# poll tick. The batched replacement is still cheap to cache; this TTL is
# the fallback staleness bound when neither the row set changed nor a run
# is actively streaming (e.g. a sub-agent finished in a *different*
# Console session/tab).
CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS = 2.0
CONSOLE_FOCUS_REGISTRY = WorkbenchFocusRegistry(
    (
        "console-left-rail",
        "console-transcript-surface",
        "console-right-rail",
        "console-native-composer",
    )
)
CONSOLE_FOCUS_TARGETS_BY_PANE = {
    "console-left-rail": ("console-context-rail-collapse", "console-left-rail"),
    "console-transcript-surface": (
        "console-native-transcript",
        "console-transcript-surface",
    ),
    "console-right-rail": ("console-inspector-rail-collapse", "console-right-rail"),
    "console-native-composer": ("console-native-composer",),
}
CONSOLE_WORKBENCH_SHORTCUTS = (
    ("F6", "next pane"),
    ("Shift+F6", "previous pane"),
    ("F1", "help"),
    ("Enter", "send"),
    ("Ctrl+P", "palette"),
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


def _character_session_identity_from_handoff(
    payload: ChatHandoffPayload,
) -> tuple[int, str, str] | None:
    """Return character session identity for Personas Start Chat handoffs.

    Args:
        payload: Handoff payload staged by a source screen.

    Returns:
        A tuple of `(character_id, character_name, assistant_id)` when the
        payload represents a Personas character Start Chat handoff; otherwise
        `None`.
    """
    metadata = payload.metadata or {}
    if (
        str(metadata.get("intent") or "").strip() != "start_chat"
        or str(metadata.get("selected_kind") or "").strip() != "character"
    ):
        return None

    raw_record_id = metadata.get("selected_record_id")
    character_id_text = "" if raw_record_id is None else str(raw_record_id).strip()
    if not character_id_text:
        target_id = str(metadata.get("selected_target_id") or "").strip()
        match = re.search(r"(?:^|:)character:(\d+)$", target_id)
        if match:
            character_id_text = match.group(1)
    if not character_id_text.isdecimal():
        return None

    character_id = int(character_id_text)
    character_name = str(metadata.get("selected_name") or payload.title or "").strip()
    assistant_id = str(character_id)
    return character_id, character_name, assistant_id


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


def _apply_console_message_attachments(
    message: ConsoleChatMessage,
    attachments: "Iterable[MessageAttachment]",
) -> None:
    """Set a message's attachments tuple and mirror position 0 into scalars.

    Mirrors ``ConsoleChatStore._set_message_attachments``'s invariant --
    every attachments mutation sets the tuple AND the scalar image fields
    (``image_data``, ``image_mime_type``, ``attachment_label``) together --
    for call sites that build or rehydrate ``ConsoleChatMessage`` objects
    directly (screen-state restore, saved-conversation resume), outside the
    store, where that helper isn't reachable.
    """
    rebased = tuple(
        replace(attachment, position=index)
        for index, attachment in enumerate(attachments)
    )
    message.attachments = rebased
    first = rebased[0] if rebased else None
    message.image_data = first.data if first else None
    message.image_mime_type = first.mime_type if first else None
    message.attachment_label = (
        first.display_name if first and first.display_name else None
    )


def _has_selected_text(value: Any) -> bool:
    """Return whether a provider/model value is meaningfully selected.

    Args:
        value: Value from Textual select state or app/default configuration.

    Returns:
        True when the value is not an empty Textual select sentinel and has
        non-whitespace text.
    """
    return not _is_empty_select_value(value) and bool(str(value).strip())


def _run_dictionary_summary_off_thread(
    service: Any,
    conversation_id: Any,
    character_id: Any,
) -> Any:
    """Drive the async scope-service ``summarize_active_dictionaries`` call
    to completion on a *private* event loop.

    Called via ``asyncio.to_thread`` from
    ``ChatScreen.refresh_active_dictionaries_summary`` -- this function body
    runs in a worker thread, so ``asyncio.run`` here spins up a fresh loop on
    that thread rather than reentering the UI's event loop. That keeps the
    underlying (synchronous) DB read the local chat-dictionary service
    performs entirely off the UI thread.
    """
    return asyncio.run(
        service.summarize_active_dictionaries(conversation_id, character_id, mode="local")
    )


class ChatScreen(BaseAppScreen):
    """
    Chat screen with comprehensive state management.
    
    This screen preserves all chat state including tabs, messages,
    input text, and UI preferences when navigating away and returning.
    """

    BINDINGS = [
        # Textual's Screen base class binds tab/shift+tab to the "app."-namespaced
        # focus_next/focus_previous actions, which always dispatch to App.action_focus_next
        # (never to a Screen override of the same name). Re-declaring the keys here without
        # the "app." prefix replaces those merged bindings for this screen, so the actions
        # below run on ChatScreen and can trap focus inside the blocking Console setup modal
        # instead of tunnelling into the workbench beneath it. The inherited tab/shift+tab
        # entries are dropped from the ``BaseAppScreen.BINDINGS`` spread below (rather than
        # simply appended after them): Textual merges same-class BINDINGS entries that share
        # a key into one list checked in declaration order, so keeping both would let the
        # inherited "app.focus_next"/"app.focus_previous" entries win every time.
        *(
            binding
            for binding in BaseAppScreen.BINDINGS
            if binding.key not in ("tab", "shift+tab")
        ),
        Binding("tab", "focus_next", "Focus Next", show=False),
        Binding("shift+tab", "focus_previous", "Focus Previous", show=False),
        Binding("f1", "show_workbench_help", "Help", show=False),
        Binding("f6", "focus_next_workbench_pane", "Next pane", show=False, priority=True),
        Binding(
            "shift+f6",
            "focus_previous_workbench_pane",
            "Previous pane",
            show=False,
            priority=True,
        ),
        Binding("ctrl+k", "open_console_session_switcher", "Switch session", show=True),
        Binding("alt+m", "open_console_model_popover", "Model", show=True),
        Binding("alt+v", "paste_clipboard_image", "Paste image", show=True),
        # NOT priority: widget-level escapes (transcript clear-selection, modal
        # dismiss) must keep winning before this screen-level fallback runs.
        Binding("escape", "focus_console_composer_home", "Composer", show=False),
        Binding("ctrl+t", "new_console_tab", "New tab", show=True),
        Binding("alt+1", "jump_console_tab(1)", "Tab 1", show=False),
        Binding("alt+2", "jump_console_tab(2)", "Tab 2", show=False),
        Binding("alt+3", "jump_console_tab(3)", "Tab 3", show=False),
        Binding("alt+4", "jump_console_tab(4)", "Tab 4", show=False),
        Binding("alt+5", "jump_console_tab(5)", "Tab 5", show=False),
        Binding("alt+6", "jump_console_tab(6)", "Tab 6", show=False),
        Binding("alt+7", "jump_console_tab(7)", "Tab 7", show=False),
        Binding("alt+8", "jump_console_tab(8)", "Tab 8", show=False),
        Binding("alt+9", "jump_console_tab(9)", "Tab 9", show=False),
    ]

    def action_focus_next(self) -> None:
        """Move focus to the next widget, trapping Tab inside a blocking modal.

        While the Console setup modal is blocking the workbench, this keeps
        focus cycling within the modal's own focusables instead of letting
        Tab tunnel into rail/transcript/composer controls hidden beneath it.
        """
        if self._focus_console_setup_modal_if_blocking():
            return
        self.focus_next()

    def action_focus_previous(self) -> None:
        """Move focus to the previous widget, trapping Shift+Tab inside a blocking modal.

        Mirrors ``action_focus_next`` for the reverse direction.
        """
        if self._focus_console_setup_modal_if_blocking():
            return
        self.focus_previous()
    
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
            logger.opt(exception=True).error(f"Error updating model dropdown: {e}")

    @on(Select.Changed, "#chat-api-model")
    def on_chat_api_model_changed(self, event: Select.Changed) -> None:
        """Mirror sidebar model changes into the compact shell controls."""
        model = None if _is_empty_select_value(event.value) else str(event.value)
        self._sync_compact_shell_controls(model=model)

    @on(Input.Changed, "#chat-temperature")
    def on_chat_temperature_changed(self, event: Input.Changed) -> None:
        """Mirror sidebar temperature changes into the compact shell controls."""
        self._sync_compact_shell_controls(temperature=event.value)

    @on(Input.Changed, "#console-workspace-conversation-search")
    def on_console_workspace_conversation_search_changed(self, event: Input.Changed) -> None:
        """Debounce grouped conversation-browser search in the Console rail."""
        event.stop()
        event_input = getattr(event, "input", None)
        if getattr(event_input, "disabled", False):
            return
        next_query = str(event.value or "")
        if next_query == self._console_conversation_browser_query:
            return
        self._console_conversation_browser_query = next_query
        self._console_workspace_conversation_query = self._console_conversation_browser_query
        self._console_conversation_browser_search_token += 1
        self._console_workspace_conversation_search_token = (
            self._console_conversation_browser_search_token
        )
        token = self._console_conversation_browser_search_token
        query = self._console_conversation_browser_query
        if self._console_conversation_browser_search_timer is not None:
            self._console_conversation_browser_search_timer.stop()
            self._console_conversation_browser_search_timer = None
        if self._console_workspace_conversation_search_timer is not None:
            self._console_workspace_conversation_search_timer.stop()
            self._console_workspace_conversation_search_timer = None
        if not query.strip():
            self._console_conversation_browser_rows = ()
            self._console_conversation_browser_total = None
            self._console_conversation_browser_error = ""
            self._sync_console_workspace_context()
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
            return
        self._console_conversation_browser_rows = self._filter_console_browser_rows_for_query(
            self._merge_console_browser_rows(
                self._native_console_browser_rows(),
                self._membership_console_browser_rows(),
            ),
            query,
        )
        self._console_conversation_browser_total = None
        self._console_conversation_browser_error = ""
        self._sync_console_workspace_context()
        self._console_conversation_browser_search_timer = self.set_timer(
            0.2,
            lambda: self.run_worker(
                self._refresh_console_conversation_browser_search(
                    query,
                    token,
                ),
                group="console-workspace-conversation-search",
                exclusive=True,
            ),
        )
        self._console_workspace_conversation_search_timer = (
            self._console_conversation_browser_search_timer
        )

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

    @on(Button.Pressed, "#console-inspector-dictionaries-attach")
    def on_console_inspector_dictionaries_attach(self, event: Button.Pressed) -> None:
        """Open the attach-dictionary picker for the active Console conversation."""
        event.stop()
        if self._console_dictionary_dialog_active:
            return
        self._console_dictionary_dialog_active = True
        self.run_worker(self._console_dictionary_attach_worker(), group="console-io")

    @on(Button.Pressed, "#console-inspector-dictionaries-detach")
    def on_console_inspector_dictionaries_detach(self, event: Button.Pressed) -> None:
        """Open the detach-dictionary picker for the active Console conversation."""
        event.stop()
        if self._console_dictionary_dialog_active:
            return
        self._console_dictionary_dialog_active = True
        self.run_worker(self._console_dictionary_detach_worker(), group="console-io")

    async def _open_console_settings(self, *, focus_model: bool = False) -> None:
        """Open Console session settings for the active native session."""
        settings = self._ensure_active_console_session_settings()
        controller = self._ensure_console_chat_controller()
        modal = ConsoleSettingsModal(
            settings=settings,
            app_config=self._provider_readiness_app_config(),
            providers_models=await self._providers_models_for_console_settings(
                settings.provider,
                current_model=settings.model,
            ),
            context_estimate=self._active_console_settings_context_estimate(),
            can_save=controller.run_state.is_send_allowed,
            focus_model=focus_model,
        )

        def _apply_modal_result(result: ConsoleSessionSettings | None) -> None:
            if not isinstance(result, ConsoleSessionSettings):
                return
            # Modal results are explicit user selections; mark them so stale
            # default refresh never overrides them.
            self._replace_active_console_session_settings(replace(result, source="user"))
            self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")

        self.app.push_screen(modal, callback=_apply_modal_result)

    async def on_console_settings_open(self, event: Button.Pressed) -> None:
        """Open Console session settings for the active native session."""
        event.stop()
        summary_state = self._build_console_settings_summary_state()
        recovery_label, _recovery_target, _recovery_tooltip = self._console_provider_recovery_action()
        await self._open_console_settings(
            focus_model=(
                self._is_console_choose_model_action(summary_state.action_label)
                or self._is_console_choose_model_action(event.button.label)
                or self._is_console_choose_model_action(recovery_label)
            )
        )

    @on(WorkbenchActionRequested)
    async def on_console_workbench_action_requested(
        self,
        event: WorkbenchActionRequested,
    ) -> None:
        """Route visible Workbench actions through Console-owned helpers."""
        event.stop()
        action_id = event.action_id
        if action_id == "new-tab":
            await self._create_native_console_session_from_active_context()
        elif action_id == "settings":
            await self._open_console_settings(focus_model=False)
        elif action_id == "attach-context":
            self._set_console_rail_preference(left_open=True)
        elif action_id == "run-library-rag":
            self._run_console_library_rag_from_visible_action()
        elif action_id == "save-chatbook":
            self._save_console_chatbook_from_visible_action()
        elif action_id == "send":
            await self._send_console_message_from_visible_action()
        elif action_id == "stop":
            await self._stop_console_generation_from_visible_action()
        elif action_id == "help":
            await self.action_show_workbench_help()
        elif action_id == "provider-recovery":
            await self._open_console_provider_recovery()
        elif action_id == CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION:
            self._apply_detected_local_server()

    async def action_show_workbench_help(self) -> None:
        """Open contextual help for visible Console Workbench actions."""
        control_state = self._build_console_control_state(
            self._pending_console_launch_context
        )
        workbench_state = self._build_console_workbench_state(control_state)
        self.app.push_screen(
            WorkbenchHelpPanel(
                WorkbenchHelpState(
                    route_id=workbench_state.route_id,
                    title="Console",
                    actions=workbench_state.actions,
                    shortcuts=CONSOLE_WORKBENCH_SHORTCUTS,
                )
            )
        )

    async def action_focus_next_workbench_pane(self) -> None:
        """Move focus to the next visible Console Workbench pane."""
        if self._focus_console_setup_modal_if_blocking():
            return
        hidden = {
            pane_id
            for pane_id in CONSOLE_FOCUS_REGISTRY.pane_order
            if not self._is_console_widget_displayed(pane_id)
        }
        current = self._console_workbench_focus_id_for_widget(self.app.focused)
        target_id = CONSOLE_FOCUS_REGISTRY.next_after(current, hidden=hidden)
        if target_id is None:
            return
        self._focus_console_workbench_target(target_id)

    async def action_focus_previous_workbench_pane(self) -> None:
        """Move focus to the previous visible Console Workbench pane."""
        if self._focus_console_setup_modal_if_blocking():
            return
        hidden = {
            pane_id
            for pane_id in CONSOLE_FOCUS_REGISTRY.pane_order
            if not self._is_console_widget_displayed(pane_id)
        }
        current = self._console_workbench_focus_id_for_widget(self.app.focused)
        target_id = CONSOLE_FOCUS_REGISTRY.previous_before(current, hidden=hidden)
        if target_id is None:
            return
        self._focus_console_workbench_target(target_id)

    def _console_workbench_density(self) -> str:
        """Return the supported Console Workbench density from app config."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        appearance = app_config.get("appearance", {})
        if not isinstance(appearance, dict):
            return "normal"
        density = str(
            appearance.get("ui_density", appearance.get("density", "normal")) or ""
        ).strip().lower()
        return "compact" if density == "compact" else "normal"

    def _is_console_widget_displayed(self, widget_id: str) -> bool:
        """Return True when a Console focus target and its parents are visible."""
        try:
            current = self.query_one(f"#{widget_id}")
        except QueryError:
            return False
        while current is not None:
            if current.display is False or current.styles.display == "none":
                return False
            current = getattr(current, "parent", None)
        return True

    def _console_workbench_focus_id_for_widget(
        self,
        focused: object | None,
    ) -> str | None:
        """Return the owning Console Workbench pane id for a focused widget."""
        current = focused
        while current is not None:
            current_id = getattr(current, "id", None)
            if current_id in CONSOLE_FOCUS_REGISTRY.pane_order:
                return str(current_id)
            current = getattr(current, "parent", None)
        return None

    def _focus_console_workbench_target(self, widget_id: str) -> None:
        """Focus a visible Console Workbench target if it is available."""
        for target_id in CONSOLE_FOCUS_TARGETS_BY_PANE.get(widget_id, (widget_id,)):
            if not self._is_console_widget_displayed(target_id):
                continue
            try:
                widget = self.query_one(f"#{target_id}")
            except QueryError:
                continue
            widget.can_focus = True
            widget.focus()
            self._last_console_workbench_focus_id = widget_id
            return

    def _focus_console_setup_modal_if_blocking(self) -> bool:
        """Trap pane cycling on the setup modal while it blocks the workbench."""
        if not self._console_setup_modal_blocking():
            return False
        try:
            modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        except QueryError:
            return False
        modal.focus_primary_action()
        return True

    def _ensure_console_workbench_targets_focusable(self) -> None:
        """Make mounted visible Console Workbench focus targets focusable."""
        for pane_id in CONSOLE_FOCUS_REGISTRY.pane_order:
            for widget_id in CONSOLE_FOCUS_TARGETS_BY_PANE.get(pane_id, (pane_id,)):
                if not self._is_console_widget_displayed(widget_id):
                    continue
                try:
                    self.query_one(f"#{widget_id}").can_focus = True
                except QueryError:
                    continue

    def _restore_console_workbench_focus(self) -> None:
        """Restore focus to a visible Console Workbench pane after activation."""
        if self._focus_console_setup_modal_if_blocking():
            self._apply_console_setup_block(True)
            return
        self._ensure_console_workbench_targets_focusable()
        current = self._console_workbench_focus_id_for_widget(self.app.focused)
        if current is not None and self._is_console_widget_displayed(current):
            self._last_console_workbench_focus_id = current
            return
        for widget_id in (
            self._last_console_workbench_focus_id,
            "console-native-composer",
            "console-transcript-surface",
        ):
            if widget_id and self._is_console_widget_displayed(widget_id):
                self._focus_console_workbench_target(widget_id)
                return

    def _register_console_footer_shortcuts(self) -> None:
        """Register Console Workbench shortcuts with the app footer if mounted."""
        try:
            footer = self.app.query_one(AppFooterStatus)
        except QueryError:
            return
        set_shortcuts = getattr(footer, "set_workbench_shortcuts", None)
        if callable(set_shortcuts):
            set_shortcuts(source="console", shortcuts=CONSOLE_WORKBENCH_SHORTCUTS)

    def _clear_console_footer_shortcuts(self) -> None:
        """Clear Console Workbench shortcuts from the app footer if mounted."""
        try:
            footer = self.app.query_one(AppFooterStatus)
        except QueryError:
            return
        clear_shortcuts = getattr(footer, "clear_shortcut_context", None)
        if callable(clear_shortcuts):
            clear_shortcuts(source="console")

    def _open_console_session_rename_modal(self, session_id: str) -> None:
        """Open a modal for viewing and editing the active Console tab title."""
        store = self._ensure_console_chat_store()
        session = next(
            (candidate for candidate in store.sessions() if candidate.id == session_id),
            None,
        )
        if session is None:
            self.app_instance.notify("Console tab is no longer available.", severity="error")
            return

        def _apply_rename(result: str | None) -> None:
            if result is None:
                return
            try:
                store.rename_session(session_id, result)
            except ValueError as exc:
                self.app_instance.notify(str(exc), severity="warning")
                return
            except KeyError:
                self.app_instance.notify("Console tab is no longer available.", severity="error")
                return
            self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")

        self.app.push_screen(
            ConsoleRenameSessionModal(title=session.title),
            callback=_apply_rename,
        )

    def action_open_console_session_switcher(self) -> None:
        """Open the Ctrl+K fuzzy session switcher."""
        if self._console_setup_modal_blocking():
            return
        rows = [
            *self._native_console_browser_rows(),
            *self._membership_console_browser_rows(),
        ]
        persisted_rows, _total, _error = self._sync_persisted_console_browser_rows()
        rows.extend(persisted_rows)
        self.app.push_screen(
            ConsoleSessionSwitcherModal(rows=tuple(rows)),
            callback=self._apply_console_switcher_choice,
        )

    async def action_open_console_model_popover(self) -> None:
        """Open the Alt+M quick provider/model/temperature/streaming popover."""
        if self._console_setup_modal_blocking():
            return
        settings = self._ensure_active_console_session_settings()
        providers_models = await self._providers_models_for_console_settings(
            settings.provider,
            current_model=settings.model,
        )
        self.app.push_screen(
            ConsoleModelPopover(settings=settings, providers_models=providers_models),
            callback=self._apply_console_model_popover_result,
        )

    def _apply_console_model_popover_result(
        self, result: "ConsoleSessionSettings | str | None"
    ) -> None:
        """Apply the popover result: sentinel opens full settings, else replaces settings.

        Args:
            result: Popover result, full-settings sentinel, or ``None`` on cancel.
        """
        if result is None:
            return
        if result == CONSOLE_POPOVER_OPEN_FULL_SETTINGS:
            self.run_worker(self._open_console_settings(), exclusive=False)
            return
        # Popover results are explicit user selections; protect from refresh.
        self._replace_active_console_session_settings(replace(result, source="user"))

    def action_focus_console_composer_home(self) -> None:
        """Return keyboard focus to the Console composer (Escape, non-priority).

        Deliberately not ``priority=True`` so widget-level escapes — transcript
        selection-clear, and any pushed modal's own dismiss binding — are
        resolved first as the key event bubbles up; this screen-level action
        only fires once nothing closer to focus has claimed Escape.
        """
        if self._console_setup_modal_blocking():
            return
        self._focus_console_composer_if_needed(force=True)

    def action_new_console_tab(self) -> None:
        """Open a new native Console session tab from the active context (Ctrl+T)."""
        if self._console_setup_modal_blocking():
            return
        self.run_worker(self._create_native_console_session_from_active_context(), exclusive=False)

    def action_open_console_session_settings(self) -> None:
        """Open the full Console session settings modal, guarded by the setup modal.

        Routes the command-palette "Console: Session settings…" entry through
        the same blocking check every other Console action honors, instead of
        the palette calling ``_open_console_settings`` directly and bypassing
        the first-run setup modal.
        """
        if self._console_setup_modal_blocking():
            return
        self.run_worker(self._open_console_settings(), exclusive=False)

    def action_open_console_prompt_insert(self) -> None:
        """Open the `/prompt` insert picker from the command palette ("Insert prompt…").

        Mirrors bare `/prompt` (no args): opens the picker to browse rather
        than attempting a meaningless empty-name resolution.
        """
        if self._console_setup_modal_blocking():
            return
        self.run_worker(
            self._open_console_prompt_picker_for_insert(""),
            exclusive=False,
        )

    def action_open_console_system_prompt_editor(self) -> None:
        """Open the system prompt editor from the command palette ("Edit system prompt")."""
        if self._console_setup_modal_blocking():
            return
        self.run_worker(self._open_console_system_prompt_editor(), exclusive=False)

    async def action_jump_console_tab(self, number: int) -> None:
        """Jump directly to the Nth native Console session tab (Alt+1..9).

        Args:
            number: One-based session tab number to activate.
        """
        if self._console_setup_modal_blocking():
            return
        store = self._ensure_console_chat_store()
        sessions = store.sessions()
        if not (1 <= number <= len(sessions)):
            return
        await self._activate_native_console_session(sessions[number - 1].id)

    async def _activate_native_console_session(self, session_id: str) -> None:
        """Activate a native Console session through the shared activation sequence.

        Set the active workspace, switch the native session, await the UI
        sync, then force composer focus. Shared by the session-tab click
        handler, the Ctrl+K switcher callback, and Alt+1..9 tab-jump so all
        three entry points follow one activation path.

        Args:
            session_id: Native Console session id to activate.
        """
        controller = self._ensure_console_chat_controller()
        if controller.store.active_session_id != session_id:
            self._set_active_workspace_for_console_session(session_id)
            controller.switch_session(session_id)
            # Finding C: a sub-agent drill-in is scoped to the conversation
            # active when the user drilled in -- clear it immediately on
            # switch here (the shared activation path for tab clicks,
            # Ctrl+K, and Alt+1..9) rather than rely solely on the rail
            # render path's own defensive re-check on the next sync.
            self._console_agent_drilldown_run_id = None
            await self._sync_native_console_chat_ui()
        self._focus_console_composer_if_needed(force=True)

    async def _apply_console_switcher_choice(
        self, choice: ConsoleSwitcherChoice | None
    ) -> None:
        """Apply a switcher selection through the shared native-session activation helper.

        Mirrors the session-tab click handler and Alt+1..9 tab-jump: all three
        call ``_activate_native_console_session`` so there is one activation
        sequence (set workspace, switch, sync UI, focus composer) shared
        across Console session-selection entry points.

        Args:
            choice: Switcher result, or ``None`` if the switcher was cancelled.
        """
        if choice is None:
            return
        entry = choice.entry
        if choice.kind == "rename" and entry.native_session_id:
            self._open_console_session_rename_modal(entry.native_session_id)
            return
        if choice.kind != "activate":
            return
        if entry.native_session_id:
            await self._activate_native_console_session(entry.native_session_id)
            return
        if entry.conversation_id:
            await self._resume_console_workspace_conversation(
                entry.conversation_id,
                target_scope_type=entry.scope_type or None,
                target_workspace_id=entry.workspace_id,
            )

    async def _create_native_console_session_from_active_context(self) -> None:
        """Create and focus a native Console session in the active workspace context."""
        self._ensure_console_chat_controller().new_session(
            settings=(
                self._active_console_session_settings()
                or self._default_console_session_settings()
            ),
        )
        await self._sync_native_console_chat_ui()
        self._focus_console_composer_if_needed(force=True)

    @on(Button.Pressed, "#console-change-workspace")
    def on_console_change_workspace(self, event: Button.Pressed) -> None:
        """Open the active Console workspace switcher."""
        event.stop()
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        if registry_service is None:
            self.app_instance.notify("Workspace service is not ready.", severity="warning")
            return
        try:
            workspaces = tuple(registry_service.list_workspaces())
            active_workspace = registry_service.get_active_workspace()
        except Exception:
            logger.opt(exception=True).warning("Unable to open Console workspace switcher")
            self.app_instance.notify(
                "Workspace registry could not be read.",
                severity="error",
            )
            return
        if not workspaces:
            self.app_instance.notify(
                "Create a workspace in Library > Workspaces before switching.",
                severity="warning",
            )
            return

        active_workspace_id = (
            active_workspace.workspace_id if active_workspace is not None else None
        )

        def _apply_workspace_switch(workspace_id: str | None) -> None:
            if not workspace_id:
                return
            try:
                registry_service.set_active_workspace(workspace_id)
            except Exception:
                logger.opt(exception=True).warning(
                    "Unable to switch Console workspace",
                )
                self.app_instance.notify(
                    "Workspace could not be selected.",
                    severity="error",
                )
                return
            self._sync_console_chat_core_state()
            self._activate_console_session_for_workspace(workspace_id)
            self._sync_console_workspace_context()
            self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")

        self.app.push_screen(
            ConsoleWorkspaceSwitcherModal(
                workspaces=workspaces,
                active_workspace_id=active_workspace_id,
            ),
            callback=_apply_workspace_switch,
        )
    
    
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
        self._pending_console_launch_auto_open_inspector = False
        self._console_control_provider: Optional[Any] = None
        self._console_control_model: Optional[Any] = None
        self._console_library_rag_query = ""
        self._console_chat_store: ConsoleChatStore | None = None
        self._console_agent_bridge: Any | None = None
        self._console_agent_drilldown_run_id: str | None = None
        # Finding C: the conversation the drill-in was set for -- used to
        # detect a conversation/session switch and drop back to the
        # overview instead of showing a foreign conversation's sub-agent.
        self._console_agent_drilldown_conversation_id: str | None = None
        # Finding A: batched sub-agent badge-count cache + the staleness
        # markers used to decide whether it needs a fresh DB round trip.
        self._console_subagent_counts_cache: Dict[str, int] = {}
        self._console_subagent_counts_cache_row_ids: frozenset = frozenset()
        self._console_subagent_counts_cache_at: float = 0.0
        self._console_rail_prune_dispatched = False
        self._console_workspace_conversation_query = ""
        self._console_workspace_conversation_search_timer: Any | None = None
        self._console_workspace_conversation_search_token = 0
        self._console_workspace_conversation_search_rows: tuple[
            ConsoleWorkspaceConversationRow, ...
        ] = ()
        self._console_workspace_conversation_search_total: int | None = None
        self._console_workspace_conversation_search_error = ""
        self._console_workspace_conversation_workspace_id: str | None = None
        self._console_conversation_browser_query = ""
        self._console_conversation_browser_search_timer: Any | None = None
        self._console_conversation_browser_search_token = 0
        self._console_conversation_browser_rows: tuple[
            ConsoleConversationBrowserInputRow, ...
        ] = ()
        self._console_conversation_browser_total: int | None = None
        self._console_conversation_browser_error = ""
        self._console_visible_draft_session_id: str | None = None
        self._console_provider_gateway: Any | None = None
        self._console_chat_controller: ConsoleChatController | None = None
        self._console_command_registry: ConsoleCommandRegistry = default_console_registry()
        # Task 9: cached snapshot the bare `/skill-name` fallback resolver
        # closes over (refreshed on mount/resume by
        # `_refresh_console_skill_candidates`); dispatch itself always
        # re-resolves against a FRESH `get_context` for the authoritative
        # trust decision, since this snapshot may be stale.
        self._console_skill_candidates: tuple[SkillCommandCandidate, ...] = ()
        self._console_command_registry.register_fallback_resolver(
            make_skill_fallback_resolver(lambda: self._console_skill_candidates)
        )
        self._console_unknown_send_armed: str | None = None
        # Task 9 fix-wave (reviewer repro): the skill name to append as a
        # TOOL "driving this turn" marker once the submit it was staged for
        # is actually ACCEPTED (consumed by `_on_console_submission_accepted`,
        # fired synchronously right after the USER message lands and before
        # the ASSISTANT placeholder -- see `_run_resolved_console_skill`).
        # Cleared on consume and on any dispatch failure so a refused/
        # blocked run never leaks its marker onto a later, unrelated send.
        self._console_pending_skill_marker_name: str | None = None
        self._console_image_view_state: ConsoleImageViewState | None = None
        self._console_image_cache: ConsoleImageRenderCache | None = None
        self._console_image_default_mode: Literal["pixels", "graphics"] | None = None
        self._console_image_preparing: set[str] = set()
        self._console_message_action_service = ConsoleMessageActionService()
        self._console_model_option_warnings: dict[tuple[str, str], str] = {}
        self._last_console_action: ConsoleActionResult | None = None
        self._pending_console_delete_message_id: str | None = None
        self._console_transcript_sync_timer: Any | None = None
        self._console_sync_in_progress = False
        self._console_sync_requested = False
        self._last_native_transcript_refresh_key: tuple[int, tuple[Any, ...]] | None = None
        self._last_console_workbench_focus_id: str | None = None
        self._last_console_control_state: ConsoleControlState | None = None
        self._last_console_workbench_state: Any | None = None
        self._last_console_rail_state: ConsoleRailState | None = None
        self._console_guidance_dismissed = False
        self._console_first_send_completed_cached: bool | None = None
        self._console_detected_local_server: DiscoveredLocalServer | None = None
        self._console_local_discovery_started = False
        # P1g: cached "what's in play" chat-dictionary summary for the
        # active native Console session's conversation/character scope,
        # recomputed only by `refresh_active_dictionaries_summary()`.
        # Fix-wave (Critical, Task 4 review): the original recompute
        # trigger hooked the legacy `app.current_chat_conversation_id`/
        # `current_chat_active_character_data` reactives -- those are
        # written ONLY by the *legacy* sidebar chat flow
        # (`Event_Handlers/Chat_Events/chat_events.py`). The native
        # Console tracks its own session in `_console_chat_store` and
        # never touches them, so the summary was permanently stuck at
        # "No active chat" in the real app. `_sync_native_console_chat_ui()`
        # now recomputes whenever the active native session's
        # (conversation_id, character_id) scope changes -- see
        # `_active_console_dictionary_scope_ids()` and
        # `_refresh_active_dictionaries_summary_if_scope_changed()`.
        # `_build_console_inspector_state` and the inspector compose path
        # read ONLY this cache -- never a DB query on recompose.
        self._active_dictionaries_summary: dict | None = None
        # Last (conversation_id, character_id) scope a refresh was run
        # for. `_sync_native_console_chat_ui()` also runs on a 0.2s
        # transcript-poll timer while a run is streaming
        # (`_start_console_transcript_sync_timer`), so this guard is what
        # keeps the recompute from hitting the DB (via the scope service)
        # several times a second instead of only on a real scope change.
        self._last_console_dictionary_scope_ids: tuple[str | None, int | None] | None = None
        # P1g Task 5: guards the Console dictionary attach/detach picker
        # flow against a double-open (mirrors P1f's `_io_dialog_active`),
        # reset in a `finally` in both attach/detach workers.
        self._console_dictionary_dialog_active = False
        self.ui_state = UIState()
        self._load_sidebar_state()

    # Sections `load_settings()` always injects into a disk-loaded config but
    # which Console test fakes never carry. Used to tell a real boot snapshot
    # (safe to refresh from disk) apart from an injected test config (must be
    # honored verbatim; reading the developer's real config would break
    # hermetic tests). NOTE: verified against real `load_settings()` output on
    # a virgin template config - do not add keys (e.g. `splash_screen`) that
    # only `load_cli_config_and_ensure_existence()` emits, or the live app
    # never takes the fresh branch.
    _CONSOLE_LIVE_CONFIG_MARKER_SECTIONS = ("general", "logging")

    # Readiness labels that stale-default session refresh may recover from:
    # credential/endpoint gaps a Settings save can fix. Provider-identity
    # blockers (Unknown/Pending WIP providers) are deliberate choices and are
    # never auto-replaced.
    _CONSOLE_REFRESHABLE_BLOCKED_LABELS = frozenset(
        {"Missing key", "Not ready", "Invalid URL", "Endpoint not saved"}
    )

    def _provider_readiness_app_config(self) -> Any:
        """Return the freshest app config for provider-readiness checks.

        ``app.app_config`` is a boot-time snapshot: Settings saves invalidate
        the config module cache but never refresh the snapshot, so readiness
        built from it stays blocked until restart (core-loop UAT 2026-07,
        task-177). When the snapshot looks disk-loaded, re-source it from
        ``load_settings()`` - cheap (cached) except right after a save, which
        is exactly when the fresh read matters.
        """
        try:
            app_config = getattr(self.app, "app_config")
        except (AttributeError, NoActiveAppError):
            app_config = getattr(self.app_instance, "app_config", {}) or {}
        app_config = app_config or {}
        if not self._console_config_snapshot_is_disk_loaded(app_config):
            return app_config
        try:
            fresh = load_settings()
        except Exception:
            logger.debug("Console readiness refresh via load_settings() failed; using snapshot")
            return app_config
        if isinstance(fresh, Mapping) and fresh:
            return fresh
        return app_config

    @classmethod
    def _console_config_snapshot_is_disk_loaded(cls, app_config: Any) -> bool:
        """Return True when a config snapshot came from ``load_settings()``."""
        if not isinstance(app_config, Mapping):
            return False
        return all(
            section in app_config
            for section in cls._CONSOLE_LIVE_CONFIG_MARKER_SECTIONS
        )

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
        settings = self._console_background_effect_settings()
        if self.console_session_surface is None:
            self.console_session_surface = ConsoleSessionSurface(
                self.app_instance,
                background_effect_settings=settings,
                id="console-session-surface",
                classes="console-region",
            )
        else:
            self.console_session_surface.sync_background_effect_settings(settings)
        return self.console_session_surface

    def _ui_responsiveness_monitor(self) -> Any | None:
        """Return the app-level UI diagnostics monitor when available."""
        try:
            return getattr(self.app_instance, "ui_responsiveness_monitor", None)
        except Exception:
            return None

    def _record_ui_worker_started(self, name: str) -> None:
        """Best-effort worker diagnostic hook."""
        monitor = self._ui_responsiveness_monitor()
        try:
            if monitor is not None:
                monitor.record_worker_started(name)
        except Exception:
            return

    def _record_ui_worker_finished(self, name: str) -> None:
        """Best-effort worker diagnostic hook."""
        monitor = self._ui_responsiveness_monitor()
        try:
            if monitor is not None:
                monitor.record_worker_finished(name)
        except Exception:
            return

    def _record_ui_timer_created(self, name: str) -> None:
        """Best-effort timer diagnostic hook."""
        monitor = self._ui_responsiveness_monitor()
        try:
            if monitor is not None:
                monitor.record_timer_created(name)
        except Exception:
            return

    def _record_ui_timer_stopped(self, name: str) -> None:
        """Best-effort timer diagnostic hook."""
        monitor = self._ui_responsiveness_monitor()
        try:
            if monitor is not None:
                monitor.record_timer_stopped(name)
        except Exception:
            return

    def _consume_pending_console_launch(self) -> Optional[ConsoleLiveWorkLaunch]:
        """Accept one-shot live-work launch context from another destination."""
        if self._pending_console_launch_context is not None:
            return self._pending_console_launch_context

        pending_launch = getattr(self.app_instance, "pending_console_launch", None)
        if (normalized_launch := ConsoleLiveWorkLaunch.from_pending(pending_launch)) is not None:
            self._pending_console_launch_context = normalized_launch
            self._pending_console_launch_auto_open_inspector = True
            self.app_instance.pending_console_launch = None
        return self._pending_console_launch_context

    def _chat_default_value(self, key: str) -> Any:
        """Return a chat default value from app config for legacy call sites."""
        config = getattr(self.app_instance, "app_config", {}) or {}
        defaults = config.get("chat_defaults", {}) if isinstance(config, dict) else {}
        return defaults.get(key) if isinstance(defaults, dict) else None

    def _console_background_effect_settings(self) -> ConsoleBackgroundEffectSettings:
        """Return normalized Console transcript background effect settings."""
        config = getattr(self.app_instance, "app_config", {}) or {}
        console = config.get("console", {}) if isinstance(config, dict) else {}
        background = (
            console.get("background_effects", {})
            if isinstance(console, dict)
            else {}
        )
        return normalize_console_background_effects(background)

    @staticmethod
    def _is_console_choose_model_action(label: object) -> bool:
        """Return whether a button/action label is the Console model setup action."""
        return str(label).strip().lower() == "choose model"

    def _effective_console_provider_model(self) -> tuple[Any, Any]:
        """Return the canonical Console provider/model selection.

        Returns:
            A `(provider, model)` tuple using the same precedence for Console
            control labels and run-inspector readiness.
        """
        effective = resolve_effective_provider_model(
            self._console_resolution_view(),
            console_provider=self._console_control_provider,
            console_model=self._console_control_model,
        )
        return effective.provider, effective.model

    def _console_resolution_view(self) -> Any:
        """Return resolution inputs backed by the freshest config.

        ``resolve_effective_provider_model`` reads ``app_config`` chat defaults
        and the app-level provider/model reactives. Both are boot-time
        snapshots: after a Settings save the reactives still echo the template
        defaults (e.g. ``OpenAI``/``gpt-4o``) and would keep winning over the
        freshly saved ``chat_defaults`` (task-177 live regression). This view
        substitutes the fresh config and suppresses reactive values that are
        mere echoes of the boot defaults when the fresh defaults changed;
        genuinely user-chosen reactive values (which differ from the boot
        defaults) still win.
        """
        fresh_config = self._provider_readiness_app_config()
        boot_config = getattr(self.app_instance, "app_config", {}) or {}
        reactive_provider = getattr(self.app_instance, "chat_api_provider_value", None)
        reactive_model = (
            getattr(self.app_instance, "chat_api_model_value", None)
            or getattr(self.app_instance, "chat_model_value", None)
        )
        if fresh_config is not boot_config:
            boot_defaults = (
                boot_config.get("chat_defaults", {})
                if isinstance(boot_config, Mapping)
                else {}
            )
            fresh_defaults = (
                fresh_config.get("chat_defaults", {})
                if isinstance(fresh_config, Mapping)
                else {}
            )
            if not isinstance(boot_defaults, Mapping):
                boot_defaults = {}
            if not isinstance(fresh_defaults, Mapping):
                fresh_defaults = {}
            boot_provider = provider_config_key(str(boot_defaults.get("provider") or ""))
            fresh_provider = provider_config_key(str(fresh_defaults.get("provider") or ""))
            reactive_provider_key = provider_config_key(str(reactive_provider or ""))
            if (
                reactive_provider_key
                and reactive_provider_key == boot_provider
                and fresh_provider
                and fresh_provider != boot_provider
            ):
                reactive_provider = None
            boot_model = str(boot_defaults.get("model") or "").strip()
            fresh_model = str(fresh_defaults.get("model") or "").strip()
            reactive_model_text = str(reactive_model or "").strip()
            if (
                reactive_model_text
                and reactive_model_text == boot_model
                and fresh_model
                and fresh_model != boot_model
            ):
                reactive_model = None
        return SimpleNamespace(
            app_config=fresh_config,
            chat_api_provider_value=reactive_provider,
            chat_api_model_value=reactive_model,
            chat_model_value=None,
        )

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

    async def _providers_models_for_console_settings(
        self,
        provider: str,
        *,
        current_model: str | None = None,
    ) -> dict[str, list[str]]:
        """Return provider/model options including runtime-discovered models."""
        providers_models = self._providers_models()
        provider_key = provider_config_key(provider)
        if not provider_key:
            return providers_models
        try:
            model_options = await resolve_provider_model_options(
                self.app_instance,
                provider=provider_key,
                current_model=current_model,
            )
        except Exception:
            logger.exception(
                "Unable to resolve Console runtime-discovered models for provider=%s model=%s",
                provider_key,
                current_model,
            )
            return providers_models
        if not model_options:
            return providers_models

        merged = {
            provider_name: list(model_ids)
            for provider_name, model_ids in providers_models.items()
        }
        merged[provider_key] = [option.model_id for option in model_options]
        self._remember_console_model_options(provider_key, model_options)
        return merged

    def _remember_console_model_options(
        self,
        provider: str,
        options: list[ResolvedProviderModelOption],
    ) -> None:
        provider_key = provider_config_key(provider)
        self._console_model_option_warnings = {
            key: value
            for key, value in self._console_model_option_warnings.items()
            if key[0] != provider_key
        }
        for option in options:
            model_id = str(option.model_id or "").strip()
            if not model_id or not option.warning:
                continue
            self._console_model_option_warnings[(provider_key, model_id)] = option.warning

    def _console_model_capability_warning(
        self,
        provider: str,
        model: str | None,
    ) -> str:
        model_id = str(model or "").strip()
        if not model_id:
            return ""
        return self._console_model_option_warnings.get(
            (provider_config_key(provider), model_id),
            "",
        )

    def _default_console_session_settings(self) -> ConsoleSessionSettings:
        """Build the default settings snapshot for a new native Console session."""
        provider, model = self._effective_console_provider_model()
        settings = build_default_console_session_settings(
            self._provider_readiness_app_config(),
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
        workspace_id = store.workspace_context.active_workspace_id
        session = store.ensure_session(
            title=self._console_initial_session_title_for_workspace(workspace_id),
            workspace_id=workspace_id,
            settings=self._default_console_session_settings(),
        )
        if session.settings is None:
            settings = self._default_console_session_settings()
            store.replace_session_settings(session.id, settings)
            return settings
        return self._maybe_refresh_stale_default_console_settings(store, session)

    def _maybe_refresh_stale_default_console_settings(
        self,
        store: ConsoleChatStore,
        session: ConsoleChatSession,
    ) -> ConsoleSessionSettings:
        """Re-derive default-sourced settings for blocked, never-used sessions.

        First-run sessions snapshot template defaults (e.g. OpenAI without a
        key) and that snapshot survives navigation via screen-state restore.
        When the user then configures a working provider in Settings, an empty
        session the user never explicitly configured must converge on the new
        defaults instead of keeping the setup card blocked until restart
        (task-177 live regression). Explicit selections (``source == "user"``),
        sessions with any messages, and already-sendable settings are never
        touched; stale defaults are only replaced when the re-derived defaults
        are actually send-capable.
        """
        settings = session.settings
        if settings is None:
            settings = self._default_console_session_settings()
            store.replace_session_settings(session.id, settings)
            return settings
        if getattr(settings, "source", "derived") == "user":
            return settings
        try:
            if store.messages_for_session(session.id):
                return settings
        except KeyError:
            return settings
        app_config = self._provider_readiness_app_config()
        current_readiness = build_console_settings_readiness(settings, app_config=app_config)
        if current_readiness.native_send_supported:
            return settings
        if current_readiness.label not in self._CONSOLE_REFRESHABLE_BLOCKED_LABELS:
            # Unknown/WIP providers are a provider *choice* problem, not a
            # config-fixable credential/endpoint gap; never override choice.
            return settings
        fresh_defaults = self._default_console_session_settings()
        if fresh_defaults == settings:
            return settings
        fresh_readiness = build_console_settings_readiness(
            fresh_defaults,
            app_config=app_config,
        )
        if not fresh_readiness.native_send_supported:
            return settings
        if settings.system_prompt:
            # Carry forward an already-applied `/system` prompt across this
            # config-driven refresh -- it is explicit user intent, not part
            # of the provider/model "default" this refresh re-derives.
            # `fresh_defaults.system_prompt` is always ``None`` (defaults
            # never seed it), so without this guard a message-less session
            # where the user ran `/system` before fixing a blocked provider
            # in Settings would have its applied system prompt silently
            # discarded on the very next settings read.
            fresh_defaults = replace(fresh_defaults, system_prompt=settings.system_prompt)
        store.replace_session_settings(session.id, fresh_defaults)
        return fresh_defaults

    def _replace_active_console_session_settings(
        self,
        settings: ConsoleSessionSettings,
    ) -> None:
        """Replace settings for only the active native Console session."""
        store = self._ensure_console_chat_store()
        workspace_id = store.workspace_context.active_workspace_id
        session = store.ensure_session(
            title=self._console_initial_session_title_for_workspace(workspace_id),
            workspace_id=workspace_id,
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
            system_prompt=settings.system_prompt,
        )

    def _build_console_settings_summary_state(self) -> ConsoleSettingsSummaryState:
        """Build compact summary state for the active Console session settings."""
        settings, readiness = self._active_console_settings_readiness()
        return build_console_settings_summary_state(
            settings,
            self._active_console_settings_context_estimate(),
            readiness,
        )

    def _console_rail_system_line_state(self) -> tuple[str, bool]:
        """Return the Model rail's ``System: <preview>`` line text + dim flag.

        Args: none.

        Returns:
            Tuple of ``(line_text, is_dim)`` -- ``is_dim`` is ``True`` for
            the unset ``"System: none"`` sentinel state.
        """
        settings = self._ensure_active_console_session_settings()
        line_text = build_console_rail_system_line(settings.system_prompt)
        is_dim = not str(settings.system_prompt or "").strip()
        return line_text, is_dim

    def _sync_console_rail_system_line(self) -> None:
        """Targeted update of the mounted rail ``System:`` line, no recompose."""
        line_text, is_dim = self._console_rail_system_line_state()
        try:
            system_line = self.query_one("#console-rail-system-line", Static)
        except (NoMatches, QueryError):
            return
        system_line.update(line_text)
        system_line.set_class(is_dim, "console-rail-system-line-dim")

    def _sync_console_settings_summary(self) -> None:
        """Refresh the mounted Console settings summary surfaces if present."""
        summary_state = self._build_console_settings_summary_state()
        try:
            summary = self.query_one("#console-settings-summary", ConsoleSettingsSummary)
        except (NoMatches, QueryError):
            pass
        else:
            summary.sync_state(summary_state)
        model_line1, model_line2 = build_console_model_section_lines(summary_state)
        try:
            self.query_one("#console-model-section-line1", Static).update(model_line1)
            self.query_one("#console-model-section-line2", Static).update(model_line2)
        except (NoMatches, QueryError):
            pass
        self._sync_console_rail_system_line()
        self._sync_console_agent_section()

    def _console_agent_section_lines(self) -> tuple[str, str, str]:
        """Return the Agent rail's (status, steps, sub-agents) line text.

        Reads the live in-memory run snapshot (or, when drilled into one
        sub-agent, that run's durable record) via the Console agent bridge --
        the same bridge whose ``AgentRunsDB`` backs resume re-derivation, so
        this always reflects the latest known state without any extra event
        plumbing (the 0.2s Console poll re-calls this on every tick).

        Finding B: none of this text is escaped -- every string returned
        here is rendered into a ``markup=False`` Static (see the compose
        block below), so escaping would be a second guard stacked on top
        of ``markup=False`` and would render literal backslashes (e.g.
        ``fetch [docs]`` -> ``fetch \\[docs]``). Contrast with the
        conversation-browser badge label (``format_console_conversation_
        row_label``), which renders through ``Text.from_markup`` and must
        stay escaped.

        Finding C: a drill-in is scoped to the conversation active when
        the user drilled in. Every call here re-checks that scope --
        catching any switch path that doesn't itself clear the drill-down
        -- and falls back to the overview on a mismatch rather than show
        a foreign conversation's sub-agent detail.

        Gate Finding 2 (agent-runtime live gate): the top-level overview
        line used to read only ``bridge.live_snapshot`` -- an in-memory,
        per-process cache that starts empty every new bridge instance, so
        it showed "Agent: idle" for a resumed conversation right after an
        app restart even though the drill-in and the conversation-row
        badge both correctly re-derived from ``AgentRunsDB``. An idle live
        snapshot now falls back to ``bridge.historical_snapshot`` (cached
        by the bridge itself, so this does not add a DB hit per 0.2s poll
        tick) -- a live/in-process run always reports non-"idle" and keeps
        precedence over the fallback.
        """
        bridge = self._ensure_console_agent_bridge()
        conversation_id = self._current_console_rail_conversation_id() or ""
        if bridge is None:
            return ("Agent: unavailable", "", "")
        if conversation_id != self._console_agent_drilldown_conversation_id:
            # The active conversation/session changed since the drill-in
            # (tab switch, Ctrl+K switcher, saved-conversation resume,
            # workspace switch, ...) -- this self-heals even for a switch
            # path that doesn't explicitly clear the drill-down itself.
            self._console_agent_drilldown_run_id = None
            self._console_agent_drilldown_conversation_id = conversation_id
        drill = self._console_agent_drilldown_run_id
        if drill:
            record = bridge.subagent_run(drill)
            if record is not None and record.get("conversation_id") == conversation_id:
                steps = "\n".join(
                    f"{s.get('kind')}: "
                    f"{str(s.get('summary') or s.get('result') or '')[:80]}"
                    for s in record.get("steps", []))
                return (
                    f"Sub-agent · {record.get('status')} (Back)",
                    steps,
                    str(record.get("task") or ""),
                )
            # The drilled-into run vanished, or (defensive re-check) its
            # recorded conversation_id no longer matches the one now
            # active -- fall back to the live snapshot instead of showing
            # a stale/foreign drill-in view.
            self._console_agent_drilldown_run_id = None
        snapshot = bridge.live_snapshot(conversation_id)
        if snapshot.status == "idle":
            # Finding 2 (Plan-B agent-runtime gate): this bridge instance
            # has never run this conversation in-process -- most likely a
            # resumed conversation right after an app restart, since
            # ``live_snapshot`` is an in-memory-only, per-process cache
            # that starts empty every new instance. Fall back to
            # AgentRunsDB so the summary reflects history immediately
            # instead of showing "Agent: idle" until the next live run.
            # A live run already in progress/finished in this process
            # always reports a non-"idle" status above and keeps
            # precedence -- this fallback is only ever consulted when
            # there is nothing live to show. ``getattr`` tolerates a bare
            # test double that only implements ``live_snapshot``.
            historical = getattr(bridge, "historical_snapshot", None)
            if historical is not None:
                snapshot = historical(conversation_id)
        status = f"Agent: {snapshot.status}"
        if snapshot.status == "running":
            status = f"Agent: running · step {snapshot.step}"
        steps = "\n".join(f"· {s.text[:80]}" for s in snapshot.steps)
        glyphs = {"done": "✓", "running": "●", "stuck": "⚠", "error": "✗", "cancelled": "✗"}
        subagents = "\n".join(
            f"{glyphs.get(s.status, '●')} {s.text[:60]}"
            for s in snapshot.subagents)
        return (status, steps, subagents)

    def _sync_console_agent_section(self) -> None:
        """Refresh the mounted Agent rail Statics + Back-button visibility."""
        try:
            status_line, steps_text, subagents_text = self._console_agent_section_lines()
            self.query_one("#console-agent-section-status", Static).update(status_line)
            self.query_one("#console-agent-section-steps", Static).update(steps_text)
            self.query_one("#console-agent-section-subagents", Static).update(subagents_text)
            back_button = self.query_one("#console-agent-drilldown-back", Button)
            back_button.styles.display = (
                "block" if self._console_agent_drilldown_run_id else "none"
            )
        except (NoMatches, QueryError):
            pass

    def _toggle_console_agent_drilldown_from_subagents_click(self) -> None:
        """Step the drill-in through this conversation's sub-agent runs.

        Finding D: a conversation can have more than one sub-agent run,
        but the combined ``subagents`` rail line only ever opened
        ``runs[0]`` no matter how many times it was clicked, leaving every
        other sub-agent unreachable. Repeated clicks now cycle through
        ``runs[0], runs[1], ..., runs[n-1]`` (newest first, matching
        ``AgentRunsDB.list_runs``' order) and then back to the overview,
        rather than adding a new per-row widget for what is usually a
        small N. The dedicated Back button always returns to the overview
        directly, regardless of where the cycle currently is.
        """
        bridge = self._ensure_console_agent_bridge()
        conversation_id = self._current_console_rail_conversation_id() or ""
        runs = bridge.subagent_runs(conversation_id) if bridge is not None else []
        run_ids = [run.get("id") for run in runs]
        current = self._console_agent_drilldown_run_id
        if not run_ids:
            next_run_id = None
        elif current in run_ids:
            next_index = run_ids.index(current) + 1
            next_run_id = run_ids[next_index] if next_index < len(run_ids) else None
        else:
            next_run_id = run_ids[0]
        self._console_agent_drilldown_run_id = next_run_id
        self._console_agent_drilldown_conversation_id = conversation_id
        self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")

    def _current_console_workspace_context(self) -> ConsoleWorkspaceContext:
        """Return explicit workspace policy context for native Console sends."""
        workspace_id = CONSOLE_GLOBAL_WORKSPACE_ID
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        if registry_service is not None:
            try:
                ensure_default_workspace = getattr(
                    registry_service,
                    "ensure_default_workspace",
                    None,
                )
                active_workspace = (
                    ensure_default_workspace()
                    if callable(ensure_default_workspace)
                    else registry_service.get_active_workspace()
                )
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

    def _active_console_workspace_id_for_conversation_search(self) -> str:
        """Return the current active workspace id for Console conversation search."""
        try:
            workspace_id = str(
                self._current_console_workspace_context().active_workspace_id or ""
            ).strip()
        except Exception:
            logger.opt(exception=True).debug(
                "Unable to read current workspace context for conversation search",
            )
            workspace_id = ""
        if workspace_id:
            return workspace_id
        service = getattr(self.app_instance, "workspace_registry_service", None)
        get_active_workspace = getattr(service, "get_active_workspace", None)
        if callable(get_active_workspace):
            try:
                workspace = get_active_workspace()
            except Exception:
                logger.opt(exception=True).debug("Unable to read active workspace for conversation search")
                workspace = None
            workspace_id = str(getattr(workspace, "workspace_id", "") or "").strip()
            if workspace_id:
                return workspace_id
        store = self._console_chat_store
        if store is not None and store.workspace_context.active_workspace_id:
            return str(store.workspace_context.active_workspace_id)
        return ""

    def _focus_console_workspace_conversation_search(self) -> None:
        """Restore focus to the conversation search input when it is mounted."""
        try:
            search = self.query_one("#console-workspace-conversation-search", Input)
        except (NoMatches, QueryError):
            return
        search.focus()

    def _build_console_provider_selection(self) -> ConsoleProviderSelection:
        """Return the effective native Console provider selection for sends."""
        app_config = self._provider_readiness_app_config()
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
                or first_configured_endpoint(provider_config)
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
            seed=selection_settings.seed,
            presence_penalty=selection_settings.presence_penalty,
            frequency_penalty=selection_settings.frequency_penalty,
            reasoning_effort=selection_settings.reasoning_effort,
            reasoning_summary=selection_settings.reasoning_summary,
            verbosity=selection_settings.verbosity,
            thinking_effort=selection_settings.thinking_effort,
            thinking_budget_tokens=selection_settings.thinking_budget_tokens,
            streaming=selection_settings.streaming,
            system_prompt=selection_settings.system_prompt,
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
        effective_settings = replace(
            settings,
            model=selected_model,
            base_url=selection.base_url,
        )
        if not _has_selected_text(selected_model):
            return effective_settings, ConsoleSettingsReadiness(
                label="Missing model",
                detail="Select a model before sending.",
                native_send_supported=False,
            )
        readiness = build_console_settings_readiness(
            effective_settings,
            app_config=self._provider_readiness_app_config(),
        )
        model_warning = self._console_model_capability_warning(
            effective_settings.provider,
            selected_model,
        )
        if model_warning and readiness.native_send_supported:
            return effective_settings, replace(
                readiness,
                label="Capabilities unknown",
                detail=f"{readiness.detail}\n{model_warning}",
                native_send_supported=True,
            )
        return effective_settings, readiness

    def _ensure_console_chat_store(self) -> ConsoleChatStore:
        """Return the native Console chat store, creating it lazily."""
        if self._console_chat_store is None:
            persistence = None
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is not None:
                persistence = ChatPersistenceService(
                    db,
                    workspace_registry=getattr(
                        self.app_instance,
                        "workspace_registry_service",
                        None,
                    ),
                )
            self._console_chat_store = ConsoleChatStore(
                persistence=persistence,
                workspace_context=self._current_console_workspace_context(),
            )
        return self._console_chat_store

    def _ensure_console_agent_bridge(self) -> Any:
        """Return the native Console agent bridge, creating it lazily.

        Returns ``None`` (no agent runtime) when there is no durable
        ChaChaNotes DB to key the sibling ``AgentRunsDB`` file off of (e.g. an
        in-memory test harness) -- callers fall back to the legacy direct
        stream in that case regardless of the config gate.
        """
        if self._console_agent_bridge is not None:
            return self._console_agent_bridge
        db = getattr(self.app_instance, "chachanotes_db", None)
        db_path = getattr(db, "db_path", None) if db is not None else None
        if not db_path or str(db_path) == ":memory:":
            self._console_agent_bridge = None
            return None
        from pathlib import Path

        from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        runs_db = AgentRunsDB(Path(db_path).parent / "agent_runs.db")
        self._console_agent_bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db,
            store=self._ensure_console_chat_store(),
            provider_gateway=self._ensure_console_provider_gateway(),
            skills_service=getattr(self.app_instance, "skills_scope_service", None),
            native_tools_enabled=self._console_native_tool_calls_enabled,
        )
        return self._console_agent_bridge

    def _console_agent_runtime_enabled(self) -> bool:
        """Return whether ``[console] agent_runtime`` gates in the agent loop (default on)."""
        value = self._console_config().get("agent_runtime", True)
        return bool(value) if isinstance(value, (bool, int)) else True

    def _console_native_tool_calls_enabled(self) -> bool:
        """Return whether ``[console] native_tool_calls`` allows native provider tool-calls (default on)."""
        value = self._console_config().get("native_tool_calls", True)
        return bool(value) if isinstance(value, (bool, int)) else True

    def _ensure_console_image_view(self) -> tuple[ConsoleImageViewState, ConsoleImageRenderCache]:
        """Return (view state, render cache) for inline images, creating lazily."""
        if getattr(self, "_console_image_view_state", None) is None:
            self._console_image_view_state = ConsoleImageViewState()
            self._console_image_cache = ConsoleImageRenderCache()
            # `getattr(self, "app_instance", None)`, not `self.app_instance`:
            # test helpers build bare screens via `ChatScreen.__new__` to
            # exercise serialize/restore without a mounted app, which never
            # sets `app_instance` at all (not even to None).
            self._console_image_default_mode = resolve_default_mode(
                getattr(getattr(self, "app_instance", None), "app_config", {}) or {}
            )
        return self._console_image_view_state, self._console_image_cache

    def _recent_console_image_messages(self, messages) -> list[Any]:
        """Return the most recent image-bearing messages, bounded to cache capacity.

        Mirrors the provider payload's most-recent-N image policy
        (``_provider_message_payloads``'s ``image_ids[-image_budget:]``).
        """
        # Bound the working set to the cache capacity so prep can never evict
        # what the transcript still shows (churn guard).
        image_messages = [
            message for message in messages if getattr(message, "image_data", None) is not None
        ]
        return image_messages[-IMAGE_CACHE_MAX_ENTRIES:]

    def _build_console_image_specs(self, messages) -> dict[str, ConsoleImageRowSpec]:
        """Build image-row payloads for prepared, non-hidden image messages."""
        state, cache = self._ensure_console_image_view()
        default_mode = self._console_image_default_mode
        specs: dict[str, ConsoleImageRowSpec] = {}
        for message in self._recent_console_image_messages(messages):
            mode = state.mode_for(message.id, default=default_mode)
            if mode == "hidden":
                continue
            pil = cache.get_pil(message.id)
            if pil is None:
                continue
            specs[message.id] = ConsoleImageRowSpec(
                message_id=message.id,
                mode=mode,
                pixels=cache.get_pixels(message.id) if mode == "pixels" else None,
                pil=pil if mode == "graphics" else None,
            )
        return specs

    async def _prep_console_images(self, pending: list[tuple[str, bytes]]) -> None:
        """Prepare pending transcript images off-loop, then resync once."""
        _state, cache = self._ensure_console_image_view()

        def _prepare_all() -> None:
            for message_id, image_data in pending:
                cache.prepare(message_id, image_data)

        try:
            await asyncio.to_thread(_prepare_all)
            await self._sync_native_console_chat_ui()
        finally:
            # Covers cancellation too (the exclusive-worker re-kick below):
            # a cancelled batch's ids become eligible for re-kick, and the
            # cache's pending_ids recompute keeps the working set converged.
            self._console_image_preparing.difference_update(
                mid for mid, _ in pending
            )

    def _handle_console_toggle_image_view(self, message_id: str) -> None:
        """Cycle one message's inline-image view mode."""
        state, _cache = self._ensure_console_image_view()
        current = state.mode_for(message_id, default=self._console_image_default_mode)
        state.set_mode(
            message_id,
            next_view_mode(current),
            default=self._console_image_default_mode,
        )

    def _ensure_console_provider_gateway(self) -> Any:
        """Return the native Console provider gateway with a test injection seam."""
        if self._console_provider_gateway is None:
            factory = getattr(self.app_instance, "console_provider_gateway_factory", None)
            self._console_provider_gateway = (
                factory()
                if callable(factory)
                else ConsoleProviderGateway(
                    # Fresh-config source: the gateway re-resolves readiness at
                    # send time and must see Settings saves made after boot.
                    config_provider=self._provider_readiness_app_config,
                )
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
                seed=selection.seed,
                presence_penalty=selection.presence_penalty,
                frequency_penalty=selection.frequency_penalty,
                reasoning_effort=selection.reasoning_effort,
                reasoning_summary=selection.reasoning_summary,
                verbosity=selection.verbosity,
                thinking_effort=selection.thinking_effort,
                thinking_budget_tokens=selection.thinking_budget_tokens,
                streaming=selection.streaming,
                system_prompt=selection.system_prompt,
                agent_bridge=self._ensure_console_agent_bridge(),
                agent_runtime_enabled=self._console_agent_runtime_enabled(),
                skills_service=getattr(self.app_instance, "skills_scope_service", None),
                chat_dictionary_applier=self._console_chat_dictionary_applier,
            )
        self._console_chat_controller.on_submission_accepted = (
            self._on_console_submission_accepted
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
                self._console_chat_controller.seed = selection.seed
                self._console_chat_controller.presence_penalty = selection.presence_penalty
                self._console_chat_controller.frequency_penalty = selection.frequency_penalty
                self._console_chat_controller.reasoning_effort = selection.reasoning_effort
                self._console_chat_controller.reasoning_summary = selection.reasoning_summary
                self._console_chat_controller.verbosity = selection.verbosity
                self._console_chat_controller.thinking_effort = selection.thinking_effort
                self._console_chat_controller.thinking_budget_tokens = selection.thinking_budget_tokens
                self._console_chat_controller.streaming = selection.streaming
                self._console_chat_controller.system_prompt = selection.system_prompt
            # The `[console] agent_runtime` kill-switch and the agent
            # bridge were previously read only once, at controller
            # construction (Plan-B Task 6 Important 3) -- toggling the
            # config afterward had no effect until the whole screen (and
            # controller) was torn down and rebuilt. Refresh both here,
            # every time provider selection refreshes, so the gate takes
            # effect on the very next send.
            update_agent_runtime = getattr(
                self._console_chat_controller,
                "update_agent_runtime",
                None,
            )
            if callable(update_agent_runtime):
                update_agent_runtime(
                    enabled=self._console_agent_runtime_enabled(),
                    bridge=self._ensure_console_agent_bridge(),
                )
            else:
                self._console_chat_controller._agent_runtime_enabled = (
                    self._console_agent_runtime_enabled()
                )
                self._console_chat_controller._agent_bridge = (
                    self._ensure_console_agent_bridge()
                )
        return selection

    def _activate_console_session_for_workspace(self, workspace_id: str) -> None:
        """Activate or create the Console session for the selected workspace."""
        target_workspace_id = str(workspace_id).strip()
        if not target_workspace_id:
            return
        store = self._ensure_console_chat_store()
        inherited_settings = None
        if store.active_session_id is not None:
            try:
                inherited_settings = store.session_settings(store.active_session_id)
            except KeyError:
                inherited_settings = None
        if store.active_session_id is not None:
            for session in store.sessions():
                if (
                    session.id == store.active_session_id
                    and session.workspace_id == target_workspace_id
                ):
                    return
        for session in store.sessions():
            if session.workspace_id == target_workspace_id:
                store.switch_session(session.id)
                return
        store.create_session(
            title=self._console_workspace_session_title(target_workspace_id),
            workspace_id=target_workspace_id,
            settings=inherited_settings or self._default_console_session_settings(),
        )

    def _console_workspace_session_title(self, workspace_id: str) -> str:
        """Return a readable title for an auto-created workspace Console tab."""
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        workspace_name = str(workspace_id).strip()
        if registry_service is not None:
            try:
                workspace = registry_service.get_workspace(workspace_id)
                if workspace is not None:
                    workspace_name = workspace.name
            except Exception:
                logger.opt(exception=True).debug("Unable to read Console workspace title")
        if not workspace_name:
            workspace_name = "Workspace"
        return f"{workspace_name} Chat"

    def _console_initial_session_title_for_workspace(self, workspace_id: str | None) -> str:
        """Return the first Console tab title for the active workspace."""
        target_workspace_id = str(workspace_id or "").strip()
        if not target_workspace_id or target_workspace_id in {
            CONSOLE_GLOBAL_WORKSPACE_ID,
            DEFAULT_WORKSPACE_ID,
        }:
            return DEFAULT_CONSOLE_SESSION_TITLE
        return self._console_workspace_session_title(target_workspace_id)

    def _set_active_workspace_for_console_session(self, session_id: str) -> None:
        """Keep workspace context aligned when switching Console tabs."""
        store = self._ensure_console_chat_store()
        target_session = next(
            (session for session in store.sessions() if session.id == session_id),
            None,
        )
        if target_session is None:
            return
        workspace_id = str(target_session.workspace_id or "").strip()
        if not workspace_id or workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID:
            return
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        if registry_service is None:
            return
        try:
            active_workspace = registry_service.get_active_workspace()
            if (
                active_workspace is not None
                and active_workspace.workspace_id == workspace_id
            ):
                return
            registry_service.set_active_workspace(workspace_id)
        except Exception:
            logger.opt(exception=True).warning(
                "Unable to align Console workspace with selected tab",
            )

    def _console_composer_or_none(self) -> ConsoleComposerBar | None:
        """Return the native Console composer when it is mounted."""
        composers = list(self.query("#console-native-composer"))
        if composers and isinstance(composers[0], ConsoleComposerBar):
            return composers[0]
        return None

    def _sync_console_session_draft(self) -> None:
        """Reconcile the composer draft with the active runtime Console session.

        Saves the visible draft back to the session that owns it, then loads the
        active session's draft when the active session changed. Runs inside the
        native Console sync pass so session transitions cannot lose drafts.
        """
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            title=self._console_initial_session_title_for_workspace(
                store.workspace_context.active_workspace_id
            ),
            workspace_id=store.workspace_context.active_workspace_id,
            settings=self._default_console_session_settings(),
        )
        active_session_id = session.id
        composer = self._console_composer_or_none()
        if composer is None:
            return
        visible_session_id = self._console_visible_draft_session_id
        if visible_session_id is not None:
            try:
                store.set_session_draft(visible_session_id, composer.draft_text())
            except KeyError:
                pass
        if visible_session_id == active_session_id:
            return
        try:
            composer.load_draft(store.session_draft(active_session_id))
        except KeyError:
            composer.clear_draft()
        self._console_visible_draft_session_id = active_session_id

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

    def _console_session_id_for_workspace_conversation(
        self,
        conversation_id: str,
    ) -> str | None:
        """Return an open Console session id for a workspace conversation row."""
        target = str(conversation_id or "").strip()
        if not target:
            return None
        store = self._console_chat_store
        if store is None:
            return None
        if target.startswith("native:"):
            session_id = target.removeprefix("native:")
            if any(session.id == session_id for session in store.sessions()):
                return session_id
            return None
        for session in store.sessions():
            if str(session.persisted_conversation_id or "") == target:
                return session.id
        return None

    @staticmethod
    def _iter_console_tree_messages(nodes: Any) -> list[dict[str, Any]]:
        """Return persisted conversation tree messages in visible transcript order."""
        ordered: list[dict[str, Any]] = []

        def _visit(node: Any) -> None:
            if not isinstance(node, dict):
                return
            ordered.append(node)
            children = node.get("children")
            if isinstance(children, list) and children:
                _visit(children[-1])

        if isinstance(nodes, list):
            for root in nodes:
                _visit(root)
        return ordered

    @staticmethod
    def _console_message_role_from_persisted(message: dict[str, Any]) -> ConsoleMessageRole:
        """Return a native Console role for a persisted Chat message row."""
        raw_role = str(message.get("role") or "").strip().lower()
        if raw_role:
            try:
                return ConsoleMessageRole(raw_role)
            except ValueError:
                pass
        sender = str(message.get("sender") or "").strip().lower()
        if sender in {"user", "system", "tool"}:
            return ConsoleMessageRole(sender)
        return ConsoleMessageRole.ASSISTANT

    def _console_messages_from_conversation_tree(
        self,
        tree: dict[str, Any],
    ) -> list[ConsoleChatMessage]:
        """Build native Console messages from a persisted conversation tree."""
        messages: list[ConsoleChatMessage] = []
        for row in self._iter_console_tree_messages(tree.get("root_threads")):
            content = str(row.get("content") or "")
            raw_image = row.get("image_data")
            image_data = bytes(raw_image) if isinstance(raw_image, (bytes, bytearray)) else None
            raw_mime = row.get("image_mime_type")
            image_mime_type = str(raw_mime) if raw_mime else None
            if not content and image_data is None:
                continue
            persisted_message_id = row.get("id")
            # The tree only carries the legacy position-0 columns; positions
            # >= 1 (multi-attachment table rows) are batch-fetched below,
            # once for the whole resumed list.
            attachments: tuple[MessageAttachment, ...] = (
                (
                    MessageAttachment(
                        data=image_data,
                        mime_type=image_mime_type or "",
                        display_name="",
                        position=0,
                    ),
                )
                if image_data is not None
                else ()
            )
            messages.append(
                ConsoleChatMessage(
                    role=self._console_message_role_from_persisted(row),
                    content=content,
                    status="complete",
                    persisted_message_id=(
                        str(persisted_message_id)
                        if persisted_message_id is not None
                        else None
                    ),
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    attachments=attachments,
                )
            )
        self._batch_fetch_console_resume_attachments(messages)
        return messages

    def _inject_resume_agent_markers(
        self,
        messages: list[ConsoleChatMessage],
        conversation_id: str,
    ) -> list[ConsoleChatMessage]:
        """Re-derive and interleave TOOL markers from ``AgentRunsDB`` on resume.

        Plan-B final-review Medium-1: the rail already re-derives from
        ``AgentRunsDB`` on resume (``_console_agent_section_lines`` ->
        ``bridge.historical_snapshot``, and the ``[N Sub-Agents]`` badge);
        the inline transcript TOOL markers did not, since
        ``_console_messages_from_conversation_tree`` only ever reads
        persisted ChaChaNotes rows, where markers never land
        (``ConsoleAgentBridge._append_marker`` uses ``persist=False`` so
        agent activity survives a restart without being written into the
        conversation itself). See ``inject_resume_agent_markers`` for the
        placement/idempotency contract, and ``resume_marker_messages`` for
        how each run's marker block is derived.

        Returns ``messages`` unchanged when there is no durable agent
        bridge available (e.g. an in-memory test harness, matching
        ``_ensure_console_agent_bridge``'s own fallback).
        """
        bridge = self._ensure_console_agent_bridge()
        if bridge is None:
            return messages
        from tldw_chatbook.Chat.console_agent_bridge import inject_resume_agent_markers

        return inject_resume_agent_markers(
            messages, bridge.resume_marker_messages(conversation_id))

    def _batch_fetch_console_resume_attachments(
        self, messages: list[ConsoleChatMessage]
    ) -> None:
        """Fill positions >= 1 for resumed multi-attachment messages, once.

        ``get_conversation_tree`` only returns the legacy image columns
        (position 0); the ``message_attachments`` table (positions >= 1) is
        fetched here in a SINGLE batched call covering every message this
        resume produced, then folded into each message's attachments tuple
        via ``_apply_console_message_attachments`` (see that helper for the
        store mirror invariant it replicates by hand).
        """
        ids = [m.persisted_message_id for m in messages if m.persisted_message_id]
        if not ids:
            return
        db = getattr(self.app_instance, "chachanotes_db", None)
        getter = getattr(db, "get_attachments_for_messages", None)
        if not callable(getter):
            return
        try:
            rows_by_id = getter(ids)
        except Exception:
            logger.opt(exception=True).warning(
                "Console resume attachment batch fetch failed."
            )
            return
        if not isinstance(rows_by_id, dict):
            return
        for message in messages:
            extra_rows = (
                rows_by_id.get(message.persisted_message_id)
                if message.persisted_message_id
                else None
            )
            if not extra_rows:
                continue
            extras = [
                MessageAttachment(
                    data=row.get("data"),
                    mime_type=row.get("mime_type") or "",
                    display_name=row.get("display_name") or "",
                    position=int(row.get("position", 0)),
                )
                for row in extra_rows
            ]
            _apply_console_message_attachments(
                message, list(message.attachments) + extras
            )

    def _console_session_settings_for_resume(
        self,
        conversation: Mapping[str, Any],
    ) -> ConsoleSessionSettings:
        """Return settings for a resumed session, restoring its system prompt.

        Every other field is inherited from the currently active session's
        settings (or the config-derived defaults when there is none yet);
        only ``system_prompt`` is overridden from the persisted conversation
        row so a saved system prompt survives close/resume even though it is
        never seeded from ``[chat_defaults]``.
        """
        settings = self._active_console_session_settings() or self._default_console_session_settings()
        raw_system_prompt = conversation.get("system_prompt")
        # Only blank/whitespace-only text collapses to "no system prompt";
        # anything else is restored verbatim (leading/trailing whitespace
        # and internal formatting included) rather than stripped, so a
        # formatting-sensitive prompt survives close/resume unchanged.
        system_prompt = (
            raw_system_prompt
            if isinstance(raw_system_prompt, str) and raw_system_prompt.strip()
            else None
        )
        return replace(settings, system_prompt=system_prompt)

    async def _resume_console_workspace_conversation(
        self,
        conversation_id: str,
        *,
        target_scope_type: str | None = None,
        target_workspace_id: str | None = None,
    ) -> bool:
        """Load a persisted saved conversation into a native Console session."""
        target = str(conversation_id or "").strip()
        if not target:
            return False
        conversation_service = getattr(
            self.app_instance,
            "chat_conversation_scope_service",
            None,
        )
        get_conversation_tree = getattr(conversation_service, "get_conversation_tree", None)
        if not callable(get_conversation_tree):
            self.app_instance.notify(
                "Saved conversation resume is unavailable in this build.",
                severity="warning",
            )
            return False

        try:
            maybe_tree = get_conversation_tree(target, mode="local")
            tree = await maybe_tree if inspect.isawaitable(maybe_tree) else maybe_tree
        except Exception:
            logger.exception(
                f"Unable to resume Console saved conversation: conversation_id={target}"
            )
            self.app_instance.notify(
                "Unable to load this saved conversation.",
                severity="error",
            )
            return False

        if not isinstance(tree, dict) or not tree.get("conversation"):
            self.app_instance.notify(
                "Saved conversation was not found.",
                severity="warning",
            )
            return False

        conversation = tree.get("conversation")
        if not isinstance(conversation, dict):
            conversation = {}
        store = self._ensure_console_chat_store()
        active_workspace_id = str(store.workspace_context.active_workspace_id or "").strip()
        persisted_workspace_id = (
            str(conversation.get("workspace_id")).strip()
            if conversation.get("workspace_id") is not None
            else ""
        )
        target_scope = str(target_scope_type or "").strip()
        requested_workspace_id = (
            str(target_workspace_id).strip()
            if target_workspace_id is not None
            else ""
        )
        if target_scope == "global":
            workspace_id = CONSOLE_GLOBAL_WORKSPACE_ID
        else:
            workspace_id = (
                persisted_workspace_id
                or requested_workspace_id
                or active_workspace_id
                or None
            )
        title = str(conversation.get("title") or "Saved conversation").strip()
        if not title:
            title = "Saved conversation"
        messages = self._console_messages_from_conversation_tree(tree)
        messages = self._inject_resume_agent_markers(messages, target)
        session = store.restore_persisted_session(
            title=title,
            workspace_id=workspace_id,
            persisted_conversation_id=target,
            messages=messages,
            settings=self._console_session_settings_for_resume(conversation),
        )
        self._set_active_workspace_for_console_session(session.id)
        # Finding C: resuming a saved conversation switches the active
        # conversation just as much as a tab switch does -- clear any
        # sub-agent drill-in immediately rather than rely solely on the
        # rail render path's defensive re-check on the next sync.
        self._console_agent_drilldown_run_id = None
        self._sync_console_chat_core_state()
        await self._sync_native_console_chat_ui()
        self._focus_console_composer_if_needed(force=True)
        return True

    def _build_console_workspace_context_state(
        self,
        session_data: Optional[ChatSessionData] = None,
    ) -> ConsoleWorkspaceContextState:
        current_conversation = self._current_console_conversation_id(session_data)
        state = build_console_workspace_state(
            registry_service=getattr(self.app_instance, "workspace_registry_service", None),
            current_conversation=current_conversation,
            server_adapter_state=getattr(
                self.app_instance,
                "workspace_server_adapter_state",
                None,
            ),
            acp_handoff_state=getattr(
                self.app_instance,
                "workspace_acp_handoff_state",
                None,
            ),
        )
        state = self._with_native_console_session_rows(state)
        return self._with_console_conversation_browser_state(
            state,
            current_conversation_id=current_conversation,
        )

    @staticmethod
    def _console_workspace_row_key(row: ConsoleWorkspaceConversationRow) -> str:
        return str(row.conversation_id or "").strip()

    @staticmethod
    def _console_browser_row_key(row: ConsoleConversationBrowserInputRow) -> str:
        return str(row.row_key or row.conversation_id or "").strip()

    @staticmethod
    def _console_browser_row_scope_copy(row: ConsoleConversationBrowserInputRow) -> str:
        if row.scope_type == "global":
            return "global chats"
        if row.workspace_id == DEFAULT_WORKSPACE_ID:
            return "default workspace chats"
        if row.workspace_id:
            return f"workspace {row.workspace_label}"
        return "chats"

    @staticmethod
    def _console_browser_row_matches_query(
        row: ConsoleConversationBrowserInputRow,
        normalized_query: str,
    ) -> bool:
        haystack = " ".join(
            (
                str(row.title or ""),
                str(row.workspace_label or ""),
                str(row.status or ""),
                ChatScreen._console_browser_row_scope_copy(row),
            )
        ).lower()
        return normalized_query in haystack

    def _filter_console_browser_rows_for_query(
        self,
        rows: Iterable[ConsoleConversationBrowserInputRow],
        query: str,
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        normalized_query = str(query or "").strip().lower()
        row_tuple = tuple(rows)
        if not normalized_query:
            return row_tuple
        return tuple(
            row
            for row in row_tuple
            if self._console_browser_row_matches_query(row, normalized_query)
        )

    def _find_console_browser_row(
        self,
        row_key: str,
        *,
        conversation_id: str | None = None,
    ) -> ConsoleConversationBrowserRow | None:
        """Return the current grouped browser row for a rendered row key."""
        target_row_key = str(row_key or "").strip()
        target_conversation_id = str(conversation_id or "").strip()
        if not target_row_key and not target_conversation_id:
            return None
        state = self._build_console_workspace_context_state()
        browser = state.conversation_browser
        if browser is None:
            return None
        allow_conversation_fallback = not target_row_key
        fallback: ConsoleConversationBrowserRow | None = None
        for section in browser.sections:
            for row in section.rows:
                if target_row_key and row.row_key == target_row_key:
                    return row
                if (
                    allow_conversation_fallback
                    and fallback is None
                    and target_conversation_id
                    and row.conversation_id == target_conversation_id
                ):
                    fallback = row
            for group in section.groups:
                for row in group.rows:
                    if target_row_key and row.row_key == target_row_key:
                        return row
                    if (
                        allow_conversation_fallback
                        and fallback is None
                        and target_conversation_id
                        and row.conversation_id == target_conversation_id
                    ):
                        fallback = row
        return fallback

    def _activate_console_workspace_for_browser_row(
        self,
        row: ConsoleConversationBrowserRow,
    ) -> None:
        """Align active workspace context before opening a browser row."""
        scope_type = str(row.scope_type or "").strip()
        if scope_type == "global":
            return
        workspace_id = str(row.workspace_id or "").strip()
        if not workspace_id or workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID:
            return
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        if registry_service is None:
            return
        try:
            active_workspace = registry_service.get_active_workspace()
            if (
                active_workspace is None
                or active_workspace.workspace_id != workspace_id
            ):
                registry_service.set_active_workspace(workspace_id)
            self._ensure_console_chat_store().set_workspace_context(
                self._current_console_workspace_context()
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Unable to activate Console workspace for browser row",
            )

    def _console_session_id_for_browser_row(
        self,
        row: ConsoleConversationBrowserRow,
    ) -> str | None:
        """Return an open session matching a grouped browser row's identity."""
        store = self._console_chat_store
        if store is None:
            return None
        native_session_id = str(row.native_session_id or "").strip()
        if native_session_id:
            if any(session.id == native_session_id for session in store.sessions()):
                return native_session_id
            return None
        row_key = str(row.row_key or "").strip()
        if row_key.startswith("native:"):
            return self._console_session_id_for_workspace_conversation(row_key)
        conversation_id = str(row.conversation_id or "").strip()
        if not conversation_id:
            return None
        scope_type = str(row.scope_type or "").strip()
        expected_workspace_id = (
            CONSOLE_GLOBAL_WORKSPACE_ID
            if scope_type == "global"
            else str(row.workspace_id or "").strip()
        )
        fallback_session_id: str | None = None
        for session in store.sessions():
            if str(session.persisted_conversation_id or "").strip() != conversation_id:
                continue
            session_workspace_id = str(session.workspace_id or "").strip()
            if expected_workspace_id and session_workspace_id == expected_workspace_id:
                return session.id
            if fallback_session_id is None:
                fallback_session_id = session.id
        if str(row.source_kind or "").strip() == "membership" and expected_workspace_id:
            return None
        return fallback_session_id

    @staticmethod
    def _console_browser_display_identity(
        row: ConsoleConversationBrowserInputRow,
    ) -> tuple[str, str, str, str] | tuple[str, str]:
        """Return the display identity used to dedupe grouped browser rows."""
        conversation_id = str(row.conversation_id or "").strip()
        if conversation_id:
            scope_type = str(row.scope_type or "").strip() or "workspace"
            workspace_id = (
                ""
                if scope_type == "global"
                else str(row.workspace_id or "").strip()
            )
            return ("conversation", scope_type, workspace_id, conversation_id)
        return ("row", ChatScreen._console_browser_row_key(row))

    def _console_browser_workspace_records(self) -> tuple[WorkspaceRecord, ...]:
        """Return all local workspace records visible to the Console browser."""
        service = getattr(self.app_instance, "workspace_registry_service", None)
        if service is None:
            return ()
        ensure_default = getattr(service, "ensure_default_workspace", None)
        if callable(ensure_default):
            try:
                ensure_default()
            except Exception:
                logger.opt(exception=True).debug("Unable to ensure default workspace for Console browser")
        list_workspaces = getattr(service, "list_workspaces", None)
        if not callable(list_workspaces):
            return ()
        try:
            return tuple(list_workspaces())
        except Exception:
            logger.opt(exception=True).debug("Unable to list Console browser workspaces")
            return ()

    def _console_browser_workspace_labels(self) -> dict[str, str]:
        """Return workspace labels keyed by workspace id for browser rows."""
        labels: dict[str, str] = {}
        for record in self._console_browser_workspace_records():
            workspace_id = str(record.workspace_id or "").strip()
            if not workspace_id:
                continue
            labels[workspace_id] = (
                "Chats"
                if workspace_id == DEFAULT_WORKSPACE_ID
                else str(record.name or workspace_id)
            )
        labels.setdefault(DEFAULT_WORKSPACE_ID, "Chats")
        return labels

    def _console_browser_workspace_label(
        self,
        workspace_id: str | None,
        labels: dict[str, str] | None = None,
    ) -> str:
        """Return display label for a workspace/global browser row."""
        if not workspace_id or workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID:
            return "Chats"
        if workspace_id == DEFAULT_WORKSPACE_ID:
            return "Chats"
        workspace_labels = labels if labels is not None else self._console_browser_workspace_labels()
        return workspace_labels.get(workspace_id, workspace_id)

    def _starred_console_conversation_ids(self) -> set[str]:
        """Return locally starred durable conversation ids."""
        service = getattr(self.app_instance, "conversation_local_marks_service", None)
        list_marked = getattr(service, "list_marked_conversation_ids", None)
        if not callable(list_marked):
            return set()
        try:
            return {str(conversation_id) for conversation_id in list_marked()}
        except Exception:
            logger.opt(exception=True).debug("Unable to read local conversation stars")
            return set()

    def _apply_console_browser_star_state(
        self,
        row: ConsoleConversationBrowserInputRow,
        starred_ids: set[str] | None = None,
    ) -> ConsoleConversationBrowserInputRow:
        """Apply local star state and star eligibility to one browser row."""
        conversation_id = str(row.conversation_id or "").strip()
        ids = starred_ids if starred_ids is not None else self._starred_console_conversation_ids()
        star_enabled = bool(conversation_id) and not str(row.row_key or "").startswith("native:")
        return replace(
            row,
            conversation_id=conversation_id or None,
            starred=bool(conversation_id and conversation_id in ids),
            star_enabled=bool(star_enabled),
        )

    def _native_console_browser_rows(
        self,
        current_conversation_id: str | None = None,
    ) -> list[ConsoleConversationBrowserInputRow]:
        """Return open native Console sessions across all workspaces."""
        store = self._console_chat_store
        if store is None:
            return []
        labels = self._console_browser_workspace_labels()
        starred_ids = self._starred_console_conversation_ids()
        active_session_id = store.active_session_id
        rows: list[ConsoleConversationBrowserInputRow] = []
        for session in store.sessions():
            session_workspace_id = str(session.workspace_id or "").strip()
            scope_type = "global" if session_workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID else "workspace"
            workspace_id = None if scope_type == "global" else session_workspace_id
            persisted_id = (
                str(session.persisted_conversation_id).strip()
                if session.persisted_conversation_id
                else ""
            )
            row_key = persisted_id or f"native:{session.id}"
            selected = session.id == active_session_id
            row = ConsoleConversationBrowserInputRow(
                row_key=row_key,
                conversation_id=persisted_id or None,
                native_session_id=session.id,
                title=str(session.title or "Untitled conversation"),
                scope_type=scope_type,
                workspace_id=workspace_id,
                workspace_label=self._console_browser_workspace_label(workspace_id, labels),
                status="active session" if selected else "open session",
                selected=selected,
                source_kind="native",
                updated_sort=str(session.updated_at or ""),
            )
            rows.append(self._apply_console_browser_star_state(row, starred_ids))
        return rows

    def _membership_console_browser_rows(
        self,
        current_conversation_id: str | None = None,
    ) -> list[ConsoleConversationBrowserInputRow]:
        """Return conversation membership rows across every local workspace."""
        service = getattr(self.app_instance, "workspace_registry_service", None)
        list_conversations = getattr(service, "list_workspace_conversations", None)
        if not callable(list_conversations):
            return []
        labels = self._console_browser_workspace_labels()
        starred_ids = self._starred_console_conversation_ids()
        current_conversation = current_conversation_id or self._current_console_conversation_id()
        active_session = self._active_native_console_session()
        active_workspace_id = (
            str(active_session.workspace_id or "").strip()
            if active_session is not None
            else str(self._current_console_workspace_context().active_workspace_id or "").strip()
        )
        rows: list[ConsoleConversationBrowserInputRow] = []
        for record in self._console_browser_workspace_records():
            workspace_id = str(record.workspace_id or "").strip()
            if not workspace_id:
                continue
            try:
                memberships = list_conversations(workspace_id)
            except Exception:
                logger.opt(exception=True).debug(
                    "Unable to list Console browser workspace conversations "
                    "workspace_id={}",
                    workspace_id,
                )
                continue
            for membership in memberships:
                conversation_id = str(getattr(membership, "item_id", "") or "").strip()
                if not conversation_id:
                    continue
                title = str(getattr(membership, "title", "") or conversation_id)
                row = ConsoleConversationBrowserInputRow(
                    row_key=f"workspace:{workspace_id}:conversation:{conversation_id}",
                    conversation_id=conversation_id,
                    native_session_id=None,
                    title=title,
                    scope_type="workspace",
                    workspace_id=workspace_id,
                    workspace_label=self._console_browser_workspace_label(workspace_id, labels),
                    status=str(getattr(membership, "role", "") or "workspace-thread"),
                    selected=bool(
                        current_conversation
                        and current_conversation == conversation_id
                        and active_workspace_id == workspace_id
                    ),
                    source_kind="membership",
                    updated_sort=str(getattr(membership, "created_at", "") or ""),
                )
                rows.append(self._apply_console_browser_star_state(row, starred_ids))
        return rows

    async def _persisted_console_browser_rows(
        self,
        query: str = "",
    ) -> tuple[list[ConsoleConversationBrowserInputRow], int | None, str]:
        """Return persisted global/workspace rows for grouped browser search."""
        services: list[tuple[Any, bool]] = []
        scope_service = getattr(
            self.app_instance,
            "chat_conversation_scope_service",
            None,
        )
        local_service = getattr(self.app_instance, "local_chat_conversation_service", None)

        def add_service(candidate: Any, *, include_mode: bool) -> None:
            if candidate is None:
                return
            if any(candidate is existing for existing, _include_mode in services):
                return
            services.append((candidate, include_mode))

        add_service(scope_service, include_mode=True)
        add_service(getattr(scope_service, "local_service", None), include_mode=False)
        add_service(local_service, include_mode=False)
        if not services:
            return [], None, ""

        labels = self._console_browser_workspace_labels()
        scopes: list[tuple[str, str | None]] = [("global", None)]
        scopes.extend(
            ("workspace", str(record.workspace_id))
            for record in self._console_browser_workspace_records()
            if str(record.workspace_id or "").strip()
        )
        last_error = ""
        for service, include_mode in services:
            list_conversations = getattr(service, "list_conversations", None)
            if not callable(list_conversations):
                continue
            rows: list[ConsoleConversationBrowserInputRow] = []
            total_count = 0
            saw_total = False
            saw_result = False
            current_conversation = self._current_console_conversation_id()
            starred_ids = self._starred_console_conversation_ids()
            for scope_type, workspace_id in scopes:
                list_kwargs: dict[str, Any] = {
                    "query": query,
                    "scope_type": scope_type,
                    "workspace_id": workspace_id,
                    "limit": 25,
                    "offset": 0,
                }
                if include_mode:
                    list_kwargs["mode"] = "local"
                try:
                    result = list_conversations(**list_kwargs)
                    result = await result if inspect.isawaitable(result) else result
                except Exception as exc:
                    if (
                        isinstance(exc, ValueError)
                        and "service is unavailable" in str(exc).lower()
                    ):
                        logger.debug("Local persisted conversation service is unavailable")
                        last_error = ""
                        break
                    logger.exception(
                        "Unable to search Console conversation browser "
                        "query={!r} scope_type={} workspace_id={} include_mode={}",
                        query,
                        scope_type,
                        workspace_id,
                        include_mode,
                    )
                    return rows, None if not saw_total else total_count, (
                        "Workspace conversation search is unavailable."
                    )
                saw_result = True
                if not isinstance(result, dict):
                    continue
                items = result.get("items")
                if not isinstance(items, list):
                    items = []
                total = result.get("total")
                if total is None:
                    pagination = result.get("pagination")
                    if isinstance(pagination, dict):
                        total = pagination.get("total")
                try:
                    total_count += int(total)
                    saw_total = True
                except (TypeError, ValueError):
                    total_count += len(items)
                    saw_total = True
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    conversation_id = str(item.get("id") or "").strip()
                    if not conversation_id:
                        continue
                    item_scope_type = str(item.get("scope_type") or scope_type or "workspace")
                    item_workspace_id = item.get("workspace_id", workspace_id)
                    normalized_workspace_id = (
                        None
                        if item_scope_type == "global"
                        else str(item_workspace_id or workspace_id or "").strip()
                    )
                    row = ConsoleConversationBrowserInputRow(
                        row_key=conversation_id,
                        conversation_id=conversation_id,
                        native_session_id=None,
                        title=str(item.get("title") or "Untitled conversation"),
                        scope_type=item_scope_type,
                        workspace_id=normalized_workspace_id,
                        workspace_label=self._console_browser_workspace_label(
                            normalized_workspace_id,
                            labels,
                        ),
                        status=str(item.get("state") or "workspace-thread"),
                        selected=bool(
                            current_conversation and current_conversation == conversation_id
                        ),
                        source_kind="persisted",
                        updated_sort=str(
                            item.get("updated_at")
                            or item.get("created_at")
                            or item.get("last_updated")
                            or ""
                        ),
                    )
                    rows.append(self._apply_console_browser_star_state(row, starred_ids))
            if saw_result:
                return rows, total_count if saw_total else None, last_error
        return [], None, last_error

    def _sync_persisted_console_browser_rows(
        self,
        query: str = "",
        current_conversation_id: str | None = None,
    ) -> tuple[list[ConsoleConversationBrowserInputRow], int | None, str]:
        """Return persisted rows when the local listing seam is synchronous."""
        services: list[tuple[Any, bool]] = []
        local_service = getattr(self.app_instance, "local_chat_conversation_service", None)
        scope_service = getattr(
            self.app_instance,
            "chat_conversation_scope_service",
            None,
        )

        def add_service(candidate: Any, *, include_mode: bool) -> None:
            if candidate is None:
                return
            if any(candidate is existing for existing, _include_mode in services):
                return
            services.append((candidate, include_mode))

        add_service(local_service, include_mode=False)
        add_service(getattr(scope_service, "local_service", None), include_mode=False)
        add_service(scope_service, include_mode=True)
        if not services:
            return [], None, ""

        labels = self._console_browser_workspace_labels()
        scopes: list[tuple[str, str | None]] = [("global", None)]
        scopes.extend(
            ("workspace", str(record.workspace_id))
            for record in self._console_browser_workspace_records()
            if str(record.workspace_id or "").strip()
        )
        last_error = ""
        for service, include_mode in services:
            list_conversations = getattr(service, "list_conversations", None)
            if not callable(list_conversations):
                continue
            rows: list[ConsoleConversationBrowserInputRow] = []
            total_count = 0
            saw_total = False
            saw_sync_result = False
            current_conversation = (
                current_conversation_id or self._current_console_conversation_id()
            )
            starred_ids = self._starred_console_conversation_ids()
            for scope_type, workspace_id in scopes:
                list_kwargs: dict[str, Any] = {
                    "query": query,
                    "scope_type": scope_type,
                    "workspace_id": workspace_id,
                    "limit": 25,
                    "offset": 0,
                }
                if include_mode:
                    list_kwargs["mode"] = "local"
                try:
                    result = list_conversations(**list_kwargs)
                except Exception as exc:
                    if (
                        isinstance(exc, ValueError)
                        and "service is unavailable" in str(exc).lower()
                    ):
                        logger.debug(
                            "Local persisted conversation service is unavailable"
                        )
                        last_error = ""
                        break
                    logger.exception(
                        "Unable to list Console conversation browser "
                        "query={!r} scope_type={} workspace_id={} include_mode={}",
                        query,
                        scope_type,
                        workspace_id,
                        include_mode,
                    )
                    return rows, None if not saw_total else total_count, (
                        "Workspace conversation search is unavailable."
                    )
                if inspect.isawaitable(result):
                    try:
                        result.close()
                    except AttributeError:
                        pass
                    continue
                saw_sync_result = True
                if not isinstance(result, dict):
                    continue
                items = result.get("items")
                if not isinstance(items, list):
                    items = []
                total = result.get("total")
                if total is None:
                    pagination = result.get("pagination")
                    if isinstance(pagination, dict):
                        total = pagination.get("total")
                try:
                    total_count += int(total)
                    saw_total = True
                except (TypeError, ValueError):
                    total_count += len(items)
                    saw_total = True
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    conversation_id = str(item.get("id") or "").strip()
                    if not conversation_id:
                        continue
                    item_scope_type = str(
                        item.get("scope_type") or scope_type or "workspace"
                    )
                    item_workspace_id = item.get("workspace_id", workspace_id)
                    normalized_workspace_id = (
                        None
                        if item_scope_type == "global"
                        else str(item_workspace_id or workspace_id or "").strip()
                    )
                    row = ConsoleConversationBrowserInputRow(
                        row_key=conversation_id,
                        conversation_id=conversation_id,
                        native_session_id=None,
                        title=str(item.get("title") or "Untitled conversation"),
                        scope_type=item_scope_type,
                        workspace_id=normalized_workspace_id,
                        workspace_label=self._console_browser_workspace_label(
                            normalized_workspace_id,
                            labels,
                        ),
                        status=str(item.get("state") or "workspace-thread"),
                        selected=bool(
                            current_conversation
                            and current_conversation == conversation_id
                        ),
                        source_kind="persisted",
                        updated_sort=str(
                            item.get("updated_at")
                            or item.get("created_at")
                            or item.get("last_updated")
                            or ""
                        ),
                    )
                    rows.append(self._apply_console_browser_star_state(row, starred_ids))
            if saw_sync_result:
                return rows, total_count if saw_total else None, last_error
        return [], None, last_error

    def _merge_console_browser_rows(
        self,
        *row_groups: Iterable[ConsoleConversationBrowserInputRow],
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        """Merge browser rows with native, membership, then persisted precedence."""
        merged: list[ConsoleConversationBrowserInputRow] = []
        seen: set[tuple[str, ...]] = set()
        starred_ids = self._starred_console_conversation_ids()
        for group in row_groups:
            for raw_row in group:
                row = self._apply_console_browser_star_state(raw_row, starred_ids)
                identity = self._console_browser_display_identity(row)
                if not identity[-1] or identity in seen:
                    continue
                seen.add(identity)
                merged.append(row)
        return tuple(merged)

    def _current_console_browser_rows(
        self,
        query: str,
        current_conversation_id: str | None = None,
    ) -> tuple[tuple[ConsoleConversationBrowserInputRow, ...], int | None, str]:
        """Return current grouped browser rows plus optional search metadata."""
        local_rows = self._merge_console_browser_rows(
            self._native_console_browser_rows(current_conversation_id),
            self._membership_console_browser_rows(current_conversation_id),
        )
        persisted_rows, persisted_total, sync_error = self._sync_persisted_console_browser_rows(
            query,
            current_conversation_id=current_conversation_id,
        )
        cached_rows = self._console_conversation_browser_rows
        rows = self._merge_console_browser_rows(local_rows, persisted_rows, cached_rows)
        if str(query or "").strip():
            total = (
                self._console_conversation_browser_total
                if self._console_conversation_browser_total is not None
                else persisted_total
            )
        else:
            total = None
        return rows, total, self._console_conversation_browser_error or sync_error

    def _selected_console_workspace_conversation_summary(
        self,
        rows: list[ConsoleWorkspaceConversationRow],
    ) -> str:
        selected = next((row for row in rows if row.selected), None)
        if selected is None:
            return "No active conversation."
        title = ConsoleWorkspaceContextTray._conversation_title(selected.title)
        detail = ConsoleWorkspaceContextTray._conversation_detail_status(selected.status)
        return f"{title} - {detail or 'conversation'}"

    def _merge_console_workspace_rows(
        self,
        primary: list[ConsoleWorkspaceConversationRow],
        secondary: list[ConsoleWorkspaceConversationRow],
    ) -> list[ConsoleWorkspaceConversationRow]:
        merged: list[ConsoleWorkspaceConversationRow] = []
        seen: set[str] = set()
        for row in primary + secondary:
            key = self._console_workspace_row_key(row)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(row)
        return merged

    def _native_console_rows_for_workspace_search(
        self,
        workspace_id: str,
        query: str,
    ) -> list[ConsoleWorkspaceConversationRow]:
        """Return matching open native sessions for the active workspace search."""
        store = self._console_chat_store
        if store is None:
            return []
        needle = str(query or "").strip().lower()
        rows: list[ConsoleWorkspaceConversationRow] = []
        active_session_id = store.active_session_id
        for session in store.sessions():
            selected = session.id == active_session_id
            session_workspace_id = str(session.workspace_id or "").strip()
            if (
                workspace_id
                and workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID
                and session_workspace_id != workspace_id
                and not selected
            ):
                continue
            title = str(session.title or "Untitled conversation")
            if needle and needle not in title.lower():
                continue
            conversation_id = (
                str(session.persisted_conversation_id)
                if session.persisted_conversation_id
                else f"native:{session.id}"
            )
            rows.append(
                ConsoleWorkspaceConversationRow(
                    conversation_id=conversation_id,
                    title=title,
                    status="active" if selected else "open",
                    selected=selected,
                )
            )
        return rows

    def _membership_console_rows_for_workspace_search(
        self,
        workspace_id: str,
        query: str,
    ) -> list[ConsoleWorkspaceConversationRow]:
        """Return matching workspace conversation membership rows."""
        service = getattr(self.app_instance, "workspace_registry_service", None)
        list_conversations = getattr(service, "list_workspace_conversations", None)
        if not callable(list_conversations) or not workspace_id:
            return []
        needle = str(query or "").strip().lower()
        try:
            memberships = list_conversations(workspace_id)
        except Exception:
            logger.opt(exception=True).debug("Unable to search workspace conversation memberships")
            return []
        rows: list[ConsoleWorkspaceConversationRow] = []
        current_conversation = self._current_console_conversation_id()
        for membership in memberships:
            title = str(getattr(membership, "title", "") or getattr(membership, "item_id", ""))
            if needle and needle not in title.lower():
                continue
            conversation_id = str(getattr(membership, "item_id", "") or "")
            rows.append(
                ConsoleWorkspaceConversationRow(
                    conversation_id=conversation_id,
                    title=title,
                    status=str(getattr(membership, "role", "") or "workspace-thread"),
                    selected=bool(current_conversation and conversation_id == current_conversation),
                )
            )
        return rows

    async def _persisted_console_rows_for_workspace_search(
        self,
        workspace_id: str,
        query: str,
    ) -> tuple[list[ConsoleWorkspaceConversationRow], int | None, str]:
        """Return persisted workspace conversation search rows, total, and error copy."""
        scope_service = getattr(
            self.app_instance,
            "chat_conversation_scope_service",
            None,
        )
        list_conversations = getattr(scope_service, "list_conversations", None)
        if not callable(list_conversations) or not workspace_id:
            return [], None, ""
        if (
            hasattr(scope_service, "local_service")
            and getattr(scope_service, "local_service", None) is None
        ):
            return [], None, ""
        try:
            result = list_conversations(
                mode="local",
                query=query,
                scope_type="workspace",
                workspace_id=workspace_id,
                limit=CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
                offset=0,
            )
            result = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            if (
                isinstance(exc, ValueError)
                and "service is unavailable" in str(exc).lower()
            ):
                logger.debug("Local persisted conversation search service is unavailable")
                return [], None, ""
            logger.exception("Unable to search Console workspace conversations")
            return [], None, "Workspace conversation search is unavailable."
        if not isinstance(result, dict):
            return [], 0, ""
        items = result.get("items")
        if not isinstance(items, list):
            items = []
        total = result.get("total")
        if total is None:
            pagination = result.get("pagination")
            if isinstance(pagination, dict):
                total = pagination.get("total")
        try:
            total_count = int(total)
        except (TypeError, ValueError):
            total_count = len(items)
        current_conversation = self._current_console_conversation_id()
        rows: list[ConsoleWorkspaceConversationRow] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            conversation_id = str(item.get("id") or "").strip()
            if not conversation_id:
                continue
            rows.append(
                ConsoleWorkspaceConversationRow(
                    conversation_id=conversation_id,
                    title=str(item.get("title") or "Untitled conversation"),
                    status=str(item.get("state") or "workspace-thread"),
                    selected=bool(
                        current_conversation and current_conversation == conversation_id
                    ),
                )
            )
        return rows, total_count, ""

    def _refresh_console_workspace_conversation_search_if_current(
        self,
        workspace_id: str,
        query: str,
        token: int,
        *,
        restore_focus: bool = False,
    ) -> bool:
        """Refresh search results when workspace, query, and token still match."""
        if token != self._console_workspace_conversation_search_token:
            return False
        if workspace_id != self._active_console_workspace_id_for_conversation_search():
            return False
        if query != self._console_workspace_conversation_query:
            return False
        self._sync_console_workspace_context()
        if restore_focus:
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
        return True

    async def _refresh_console_workspace_conversation_search(
        self,
        workspace_id: str,
        query: str,
        token: int,
    ) -> None:
        """Refresh search results only if workspace and query are still current."""
        if token != self._console_workspace_conversation_search_token:
            return
        if workspace_id != self._active_console_workspace_id_for_conversation_search():
            return
        if query != self._console_workspace_conversation_query:
            return
        if not str(query or "").strip():
            self._console_workspace_conversation_search_rows = ()
            self._console_workspace_conversation_search_total = None
            self._console_workspace_conversation_search_error = ""
            self._sync_console_workspace_context()
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
            return
        native_rows = self._native_console_rows_for_workspace_search(
            workspace_id,
            query,
        )
        membership_rows = self._membership_console_rows_for_workspace_search(
            workspace_id,
            query,
        )
        local_rows = self._merge_console_workspace_rows(native_rows, membership_rows)
        self._console_workspace_conversation_search_rows = tuple(local_rows)
        self._console_workspace_conversation_search_total = len(local_rows)
        self._console_workspace_conversation_search_error = ""
        self._sync_console_workspace_context()
        self.call_after_refresh(self._focus_console_workspace_conversation_search)
        persisted_rows, persisted_total, error_copy = (
            await self._persisted_console_rows_for_workspace_search(
                workspace_id,
                query,
            )
        )
        if token != self._console_workspace_conversation_search_token:
            return
        if workspace_id != self._active_console_workspace_id_for_conversation_search():
            return
        if query != self._console_workspace_conversation_query:
            return
        merged = self._merge_console_workspace_rows(
            local_rows,
            persisted_rows,
        )
        result_total = persisted_total
        if result_total is None or result_total < len(merged):
            result_total = len(merged)
        self._console_workspace_conversation_search_rows = tuple(merged)
        self._console_workspace_conversation_search_total = result_total
        self._console_workspace_conversation_search_error = error_copy
        self._sync_console_workspace_context()
        self.call_after_refresh(self._focus_console_workspace_conversation_search)

    async def _refresh_console_workspace_conversation_search_after_selection(self) -> None:
        """Refresh active search rows after a conversation row changes selection."""
        query = self._console_workspace_conversation_query
        if not query.strip():
            return
        if self._console_workspace_conversation_search_timer is not None:
            self._console_workspace_conversation_search_timer.stop()
            self._console_workspace_conversation_search_timer = None
        self._console_workspace_conversation_search_token += 1
        token = self._console_workspace_conversation_search_token
        await self._refresh_console_workspace_conversation_search(
            self._active_console_workspace_id_for_conversation_search(),
            query,
            token,
        )

    async def _refresh_console_conversation_browser_search(
        self,
        query: str,
        token: int,
    ) -> None:
        """Refresh grouped browser search rows if query and token are current."""
        if token != self._console_conversation_browser_search_token:
            return
        if query != self._console_conversation_browser_query:
            return
        if not str(query or "").strip():
            self._console_conversation_browser_rows = ()
            self._console_conversation_browser_total = None
            self._console_conversation_browser_error = ""
            self._sync_console_workspace_context()
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
            return

        local_rows = self._filter_console_browser_rows_for_query(
            self._merge_console_browser_rows(
                self._native_console_browser_rows(),
                self._membership_console_browser_rows(),
            ),
            query,
        )
        self._console_conversation_browser_rows = local_rows
        self._console_conversation_browser_total = None
        self._console_conversation_browser_error = ""
        self._sync_console_workspace_context()
        self.call_after_refresh(self._focus_console_workspace_conversation_search)

        persisted_rows, persisted_total, error_copy = await self._persisted_console_browser_rows(
            query
        )
        if token != self._console_conversation_browser_search_token:
            return
        if query != self._console_conversation_browser_query:
            return
        merged = self._merge_console_browser_rows(local_rows, persisted_rows)
        result_total = persisted_total
        if result_total is None or result_total < len(merged):
            result_total = len(merged)
        self._console_conversation_browser_rows = merged
        self._console_conversation_browser_total = result_total
        self._console_conversation_browser_error = error_copy
        self._sync_console_workspace_context()
        self.call_after_refresh(self._focus_console_workspace_conversation_search)

    async def _refresh_console_conversation_browser_after_selection(self) -> None:
        """Refresh grouped browser rows after selection/star state changes."""
        query = self._console_conversation_browser_query
        if not query.strip():
            self._console_conversation_browser_rows = ()
            self._console_conversation_browser_total = None
            self._console_conversation_browser_error = ""
            self._sync_console_workspace_context()
            return
        if self._console_conversation_browser_search_timer is not None:
            self._console_conversation_browser_search_timer.stop()
            self._console_conversation_browser_search_timer = None
        self._console_conversation_browser_search_token += 1
        self._console_workspace_conversation_search_token = (
            self._console_conversation_browser_search_token
        )
        token = self._console_conversation_browser_search_token
        await self._refresh_console_conversation_browser_search(query, token)

    def _with_native_console_session_rows(
        self,
        state: ConsoleWorkspaceContextState,
    ) -> ConsoleWorkspaceContextState:
        """Include active native Console sessions in the workspace rail.

        The workspace registry only knows about conversations after durable
        persistence links them. Native Console sessions are still user-visible
        conversations and need to remain reachable from the rail while they are
        open, including before the first persisted message exists.
        """
        store = self._console_chat_store
        if store is None:
            return state

        active_workspace_id = str(store.workspace_context.active_workspace_id or "").strip()
        active_session_id = store.active_session_id
        rows = list(state.conversation_rows)
        existing_ids = {str(row.conversation_id) for row in rows}
        native_rows: list[ConsoleWorkspaceConversationRow] = []
        for session in store.sessions():
            session_workspace_id = str(session.workspace_id or "").strip()
            selected = session.id == active_session_id
            if (
                active_workspace_id
                and active_workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID
                and session_workspace_id != active_workspace_id
                and not selected
            ):
                continue

            conversation_id = (
                str(session.persisted_conversation_id)
                if session.persisted_conversation_id
                else f"native:{session.id}"
            )
            if conversation_id in existing_ids:
                continue

            native_rows.append(
                ConsoleWorkspaceConversationRow(
                    conversation_id=conversation_id,
                    title=session.title,
                    status="active" if selected else "open",
                    selected=selected,
                )
            )
            existing_ids.add(conversation_id)

        if not native_rows:
            return state
        native_rows.sort(key=lambda row: 0 if row.selected else 1)
        return replace(
            state,
            conversation_rows=tuple(
                self._merge_console_workspace_rows(native_rows, rows)
            ),
        )

    def _with_console_workspace_conversation_section(
        self,
        state: ConsoleWorkspaceContextState,
    ) -> ConsoleWorkspaceContextState:
        """Attach renderable Conversations subsection state to workspace context."""
        workspace_id = ""
        try:
            workspace_id = str(
                self._current_console_workspace_context().active_workspace_id or ""
            ).strip()
        except Exception:
            workspace_id = ""
        store = self._console_chat_store
        if not workspace_id:
            if store is not None and store.workspace_context.active_workspace_id:
                workspace_id = str(store.workspace_context.active_workspace_id)
            elif state.workspace_label.startswith("Workspace: "):
                workspace_id = state.workspace_label.removeprefix("Workspace: ").strip()

        if self._console_workspace_conversation_workspace_id != workspace_id:
            self._console_workspace_conversation_query = ""
            self._console_workspace_conversation_search_token += 1
            self._console_workspace_conversation_search_rows = ()
            self._console_workspace_conversation_search_total = None
            self._console_workspace_conversation_search_error = ""
            self._console_workspace_conversation_workspace_id = workspace_id

        rows = list(state.conversation_rows)
        if self._console_workspace_conversation_query.strip():
            rows = list(self._console_workspace_conversation_search_rows)
        selected_summary = self._selected_console_workspace_conversation_summary(rows)
        query = self._console_workspace_conversation_query
        result_total = (
            self._console_workspace_conversation_search_total
            if query.strip()
            else None
        )
        if (
            query.strip()
            and result_total is None
            and not self._console_workspace_conversation_search_error
        ):
            result_total = len(rows)
        status_copy = console_workspace_conversation_result_copy(
            query=query,
            result_total_count=result_total,
            result_limit=CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
        )
        section = ConsoleWorkspaceConversationSectionState(
            workspace_id=workspace_id,
            collapsed=self._console_workspace_conversations_collapsed(workspace_id),
            query=query,
            selected_summary=selected_summary,
            rows=tuple(rows),
            workspace_total_count=len(rows),
            result_total_count=result_total,
            status_copy=status_copy,
            empty_copy=(
                "No matches in this workspace."
                if query.strip()
                else state.conversation_empty_copy
            ),
            search_enabled=True,
            new_conversation_enabled=state.new_conversation_enabled,
            error_copy=self._console_workspace_conversation_search_error,
        )
        return replace(state, conversation_section=section)

    def _console_subagent_counts_refresh_needed(self, row_ids: frozenset) -> bool:
        """Decide whether the sub-agent badge-count cache needs a DB round trip.

        Finding A: refreshing on every 0.2s poll tick would re-issue the
        batched count query up to 5x/second even when nothing sub-agent
        related changed. This gates the refresh to three cheap-to-check
        conditions instead of refreshing unconditionally:

        1. The visible conversation row set changed (a rebuild) -- new
           rows may need counts we have never cached.
        2. A run is actively streaming/validating/retrying for this
           screen -- a just-spawned sub-agent's count should show up
           promptly rather than wait out the full TTL.
        3. The cache has aged past ``CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS``
           -- a fallback bound covering counts that changed from a
           different Console session/tab or a resumed run, where neither
           of the above two signals fires on this screen.

        Args:
            row_ids: The conversation ids of the currently visible browser
                rows (deduplicated, blanks excluded).

        Returns:
            ``True`` when the cache should be rebuilt from the DB.
        """
        if row_ids != self._console_subagent_counts_cache_row_ids:
            return True
        controller = self._console_chat_controller
        if controller is not None and controller.run_state.status in CONSOLE_ACTIVE_RUN_STATUSES:
            return True
        age = time.monotonic() - self._console_subagent_counts_cache_at
        return age >= CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS

    def _console_subagent_counts_for_rows(
        self,
        bridge: Any | None,
        rows: Iterable[Any],
    ) -> Dict[str, int]:
        """Return ``conversation_id -> sub-agent count`` for browser rows.

        Finding A: previously called ``bridge.subagent_count(cid)`` once
        per row (a fresh sqlite connection per call) on every poll tick --
        up to ~75 queries/tick. Replaced with one batched
        ``bridge.subagent_counts(...)`` call, gated by
        ``_console_subagent_counts_refresh_needed`` so it isn't reissued
        unconditionally every tick either.

        Args:
            bridge: The Console agent bridge, or ``None`` when the agent
                runtime is unavailable (e.g. in-memory test harness).
            rows: The conversation-browser input rows currently visible.

        Returns:
            Mapping of ``conversation_id -> count``; conversations with
            zero sub-agent runs are simply absent (see
            ``AgentRunsDB.count_subagents_by_conversation``).
        """
        if bridge is None:
            return {}
        row_ids = frozenset(
            cid for row in rows
            if (cid := getattr(row, "conversation_id", None))
        )
        if self._console_subagent_counts_refresh_needed(row_ids):
            self._console_subagent_counts_cache = (
                bridge.subagent_counts(list(row_ids)) if row_ids else {}
            )
            self._console_subagent_counts_cache_row_ids = row_ids
            self._console_subagent_counts_cache_at = time.monotonic()
        return self._console_subagent_counts_cache

    def _with_console_conversation_browser_state(
        self,
        state: ConsoleWorkspaceContextState,
        current_conversation_id: str | None = None,
    ) -> ConsoleWorkspaceContextState:
        """Attach grouped all-workspaces conversation browser state."""
        marks_service = getattr(
            self.app_instance,
            "conversation_local_marks_service",
            None,
        )
        query = self._console_conversation_browser_query
        rows, total, error_copy = self._current_console_browser_rows(
            query,
            current_conversation_id=current_conversation_id,
        )
        bridge = self._ensure_console_agent_bridge()
        subagent_counts = self._console_subagent_counts_for_rows(bridge, rows)
        browser = build_console_conversation_browser_state(
            rows=rows,
            active_workspace_id=self._current_console_workspace_context().active_workspace_id,
            group_collapse_preferences=(
                self._console_conversation_browser_collapse_preferences()
            ),
            query=query,
            marks_available=marks_service is not None,
            error_copy=error_copy or self._console_conversation_browser_error,
            result_total_count=total,
            result_limit=CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
            subagent_counts=subagent_counts,
        )
        legacy_state = self._with_console_workspace_conversation_section(state)
        return replace(
            state,
            conversation_browser=browser,
            conversation_section=legacy_state.conversation_section,
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

    def _console_conversation_section_config(self) -> dict[str, Any]:
        """Return mutable Console conversation-section UI preferences."""
        console_config = self._console_config()
        section_config = console_config.get("conversation_section")
        if not isinstance(section_config, dict):
            section_config = {}
            console_config["conversation_section"] = section_config
        return section_config

    def _console_conversation_browser_config(self) -> dict[str, Any]:
        """Return mutable grouped browser UI preferences."""
        console_config = self._console_config()
        browser_config = console_config.get("conversation_browser")
        if not isinstance(browser_config, dict):
            browser_config = {}
            console_config["conversation_browser"] = browser_config
        collapsed_groups = browser_config.get("collapsed_groups")
        if not isinstance(collapsed_groups, dict):
            browser_config["collapsed_groups"] = {}
        return browser_config

    def _console_conversation_browser_collapse_preferences(self) -> dict[str, bool]:
        """Return persisted grouped browser collapse preferences."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return {}
        console_config = app_config.get("console")
        if not isinstance(console_config, dict):
            return {}
        browser_config = console_config.get("conversation_browser")
        if not isinstance(browser_config, dict):
            return {}
        collapsed_groups = browser_config.get("collapsed_groups")
        if not isinstance(collapsed_groups, dict):
            return {}
        return {
            str(group_id): bool(collapsed)
            for group_id, collapsed in collapsed_groups.items()
        }

    def _set_console_conversation_browser_group_collapsed(
        self,
        group_id: str,
        collapsed: bool,
    ) -> None:
        """Store one grouped browser collapse preference."""
        normalized_group_id = str(group_id or "").strip()
        if not normalized_group_id:
            return
        browser_config = self._console_conversation_browser_config()
        collapsed_groups = browser_config.get("collapsed_groups")
        if not isinstance(collapsed_groups, dict):
            collapsed_groups = {}
            browser_config["collapsed_groups"] = collapsed_groups
        collapsed_groups[normalized_group_id] = bool(collapsed)

    def _console_workspace_conversations_collapsed(
        self,
        workspace_id: str | None,
    ) -> bool:
        """Return stored collapse preference for one workspace."""
        key = str(workspace_id or "global").strip() or "global"
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return False
        console_config = app_config.get("console")
        if not isinstance(console_config, dict):
            return False
        section_config = console_config.get("conversation_section")
        if not isinstance(section_config, dict):
            return False
        value = section_config.get(key)
        return bool(value.get("collapsed")) if isinstance(value, dict) else False

    def _set_console_workspace_conversations_collapsed(
        self,
        workspace_id: str | None,
        collapsed: bool,
    ) -> None:
        """Store collapse preference for one workspace in memory."""
        key = str(workspace_id or "global").strip() or "global"
        section_config = self._console_conversation_section_config()
        section_config[key] = {"collapsed": bool(collapsed)}

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

    @work(thread=True)
    def _delete_console_rail_preference_keys(self, keys: list[str]) -> None:
        """Remove superseded/orphaned rail preference keys off the UI thread."""
        try:
            delete_settings_from_cli_config("console.rail_state", keys)
        except Exception as exc:
            logger.warning("Failed to prune Console rail preference keys: {}", exc)

    def _dispatch_console_rail_preference_prune(self) -> None:
        """Queue the one-shot orphaned rail-preference cleanup after mount."""
        if self._console_rail_prune_dispatched:
            return
        store = self._console_chat_store
        if store is None:
            # Sessions not restored yet; retry on a later sync so open
            # unsaved sessions are never mistaken for orphans.
            return
        if getattr(self.app_instance, "chachanotes_db", None) is None:
            # Conversation liveness cannot be established yet; retry on a
            # later sync rather than latching and never pruning this session.
            return
        self._console_rail_prune_dispatched = True
        live_scope_ids: set[str] = set()
        for session in store.sessions():
            live_scope_ids.add(str(session.id))
            persisted_id = getattr(session, "persisted_conversation_id", None)
            if persisted_id:
                live_scope_ids.add(str(persisted_id))
        self._prune_console_rail_preferences(live_scope_ids)

    @work(thread=True)
    def _prune_console_rail_preferences(self, live_scope_ids: set[str]) -> None:
        """Drop rail preference sections whose conversation/session is gone.

        Rail preferences accumulate one config section per scope forever
        (deleted conversations included); this best-effort pass bounds the
        namespace to live scopes. It refuses to prune when conversation
        liveness cannot be established.
        """
        try:
            # Peek without _console_rail_state_config(): this is a read path
            # and must not materialize an empty rail_state table.
            app_config = getattr(self.app_instance, "app_config", None)
            if not isinstance(app_config, dict):
                return
            console_config = app_config.get("console")
            if not isinstance(console_config, dict):
                return
            rail_state_config = console_config.get("rail_state")
            if not isinstance(rail_state_config, dict) or not rail_state_config:
                return
            stored_keys = list(rail_state_config.keys())
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                return
            live = set(live_scope_ids)
            offset = 0
            page_size = 1000
            while True:
                rows = db.list_all_active_conversations(limit=page_size, offset=offset)
                live.update(str(row["id"]) for row in rows if row.get("id"))
                if len(rows) < page_size:
                    break
                offset += page_size
            prunable = collect_prunable_console_rail_keys(
                stored_keys, live_scope_ids=live
            )
            if not prunable:
                return
            if delete_settings_from_cli_config("console.rail_state", prunable):
                # The in-memory config dict is shared with UI-thread readers
                # and writers; mutate it back on the UI thread, not here.
                self.app.call_from_thread(
                    self._drop_console_rail_preference_keys_in_memory, prunable
                )
                logger.info(
                    "Pruned {} orphaned Console rail preference section(s)",
                    len(prunable),
                )
        except Exception as exc:
            logger.warning("Console rail preference prune skipped: {}", exc)

    def _drop_console_rail_preference_keys_in_memory(self, keys: list[str]) -> None:
        """Remove pruned keys from the live in-memory rail-state config (UI thread)."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        console_config = app_config.get("console")
        if not isinstance(console_config, dict):
            return
        rail_state_config = console_config.get("rail_state")
        if not isinstance(rail_state_config, dict):
            return
        for key in keys:
            rail_state_config.pop(key, None)

    def _notify_console_rail_preference_save_failure(self) -> None:
        """Notify from the UI thread when background preference persistence fails."""
        self.app_instance.notify(
            "Console rail preference is saved for this session only.",
            severity="warning",
        )

    def _console_first_send_completed(self) -> bool:
        """Return the persisted global first-send flag (cached per screen)."""
        if self._console_first_send_completed_cached is None:
            app_config = getattr(self.app_instance, "app_config", None)
            raw = None
            if isinstance(app_config, dict):
                onboarding = app_config.get("console", {})
                if isinstance(onboarding, dict):
                    onboarding = onboarding.get("onboarding", {})
                raw = (
                    onboarding.get("first_send_completed")
                    if isinstance(onboarding, dict)
                    else None
                )
            self._console_first_send_completed_cached = coerce_console_first_send_completed(raw)
        return self._console_first_send_completed_cached

    def _record_console_first_send(self) -> None:
        """Persist the one-time global first-send flag and refresh guidance."""
        if self._console_first_send_completed():
            return
        self._console_first_send_completed_cached = True
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            console_cfg = app_config.get("console")
            if not isinstance(console_cfg, dict):
                console_cfg = {}
                app_config["console"] = console_cfg
            onboarding_cfg = console_cfg.get("onboarding")
            if not isinstance(onboarding_cfg, dict):
                onboarding_cfg = {}
                console_cfg["onboarding"] = onboarding_cfg
            onboarding_cfg["first_send_completed"] = True
        self._save_console_onboarding_flag()
        self._sync_console_transcript_guidance()

    @work(thread=True)
    def _save_console_onboarding_flag(self) -> None:
        """Persist the first-send flag without blocking the UI thread."""
        try:
            save_setting_to_cli_config(
                "console.onboarding",
                "first_send_completed",
                True,
            )
        except Exception as exc:
            logger.warning("Failed to persist Console onboarding flag: {}", exc)

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
        # The session-scoped fallback is superseded by the durable key; drop
        # it so migrations stop leaving permanent orphan sections behind.
        rail_state_config.pop(fallback_key, None)
        self._delete_console_rail_preference_keys([fallback_key])

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
        stored_preferences = self._stored_console_rail_preferences(
            preference_key.value,
            preference_key.fallback_value,
        )
        rail_state = build_console_rail_state(
            preference_key=preference_key,
            stored_preferences=stored_preferences,
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
        if self._should_open_standard_width_inspector(
            rail_state=rail_state,
            stored_preferences=stored_preferences,
            inspector_state=inspector_state,
        ):
            return replace(rail_state, right_open=True, right_forced_collapsed=False)
        return rail_state

    def _should_open_standard_width_inspector(
        self,
        *,
        rail_state: ConsoleRailState,
        stored_preferences: Any,
        inspector_state: ConsoleInspectorState,
    ) -> bool:
        """Return whether the 120-column Console contract should show Inspector."""
        if rail_state.right_open:
            return False
        if isinstance(stored_preferences, dict) and "right_open" in stored_preferences:
            return False
        available_columns = self._console_rail_available_columns()
        if available_columns is None or not 118 <= available_columns <= 128:
            return False
        labels = {str(row.label).strip() for row in inspector_state.rows}
        return "Run recipe" in labels and bool(
            labels
            & {
                "Blocked impact",
                "Next action",
                "Sources",
                "Tools",
                "Approvals",
                "Artifacts",
            }
        )

    def _apply_pending_launch_inspector_auto_open(
        self,
        rail_state: ConsoleRailState,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleRailState:
        """Keep newly launched live work visible until the user chooses otherwise.

        Args:
            rail_state: Current Console rail state before launch visibility is applied.
            pending_launch: Live-work launch metadata, when a launch just occurred.

        Returns:
            The original rail state, or a copy with the Inspector rail opened.
        """
        if (
            pending_launch is not None
            and self._pending_console_launch_auto_open_inspector
            and not rail_state.right_forced_collapsed
        ):
            return replace(rail_state, right_open=True)
        return rail_state

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
        self._sync_console_rail_sections(rail_state)
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

    def _sync_console_rail_visibility_if_changed(
        self,
        rail_state: ConsoleRailState,
    ) -> None:
        """Apply rail visibility only when the visible rail state changes."""
        if rail_state == self._last_console_rail_state:
            return
        self._sync_console_rail_visibility(rail_state)
        self._last_console_rail_state = rail_state

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
        rail_state = self._build_console_rail_state(
            staged_context_state=staged_context_state,
            inspector_state=inspector_state,
            workspace_context_state=workspace_context_state,
        )
        return self._apply_pending_launch_inspector_auto_open(rail_state, pending_launch)

    def _set_console_rail_preference(
        self,
        *,
        left_open: bool | None = None,
        right_open: bool | None = None,
        section_updates: Mapping[str, bool] | None = None,
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
        changes: dict[str, bool] = {}
        if left_open is not None:
            changes["left_open"] = bool(left_open)
        if right_open is not None:
            changes["right_open"] = bool(right_open)
        for section_id, section_open in (section_updates or {}).items():
            if section_id in CONSOLE_RAIL_SECTION_IDS:
                changes[f"{section_id}_open"] = bool(section_open)
        next_preferences = replace(current, **changes)
        if next_preferences != current:
            rail_state_config[preference_key.value] = serialize_console_rail_preferences(
                next_preferences
            )
            self._persist_console_rail_preferences(
                preference_key.value,
                next_preferences,
                notify_on_failure=notify_on_failure,
            )
        if right_open is not None:
            self._pending_console_launch_auto_open_inspector = False
        rail_state = self._current_console_rail_state()
        self._sync_console_rail_visibility_if_changed(rail_state)
        return rail_state

    def _sync_console_rail_sections(self, rail_state: ConsoleRailState) -> None:
        """Apply left-rail section open flags to section bodies and headers.

        Stored section preferences are scoped per workspace/conversation, so a
        runtime scope switch (for example resuming a saved conversation after a
        relaunch) can change the effective flags without a recompose.
        """
        for section_id in CONSOLE_RAIL_SECTION_IDS:
            section_open = bool(getattr(rail_state, f"{section_id}_open", True))
            self._apply_console_rail_section_open(section_id, section_open)

    def _apply_console_rail_section_open(
        self,
        section_id: str,
        section_open: bool,
    ) -> None:
        """Sync one section's body display and header glyph to an open state."""
        try:
            body = self.query_one(f"#console-rail-section-body-{section_id}")
            header = self.query_one(
                f"#console-rail-section-header-{section_id}",
                ConsoleRailSectionHeader,
            )
        except (NoMatches, QueryError):
            return
        body.styles.display = "block" if section_open else "none"
        header.sync_open(section_open)

    def _toggle_console_rail_section(self, section_id: str) -> None:
        """Flip one left-rail section open state, then sync body and header."""
        if section_id not in CONSOLE_RAIL_SECTION_IDS:
            return
        rail_state = self._current_console_rail_state()
        next_open = not getattr(rail_state, f"{section_id}_open")
        self._set_console_rail_preference(
            section_updates={section_id: next_open},
            notify_on_failure=False,
        )
        self._apply_console_rail_section_open(section_id, next_open)

    def _sync_console_workspace_context(
        self,
        session_data: Optional[ChatSessionData] = None,
    ) -> None:
        try:
            workspace_context = self.query_one(
                "#console-workspace-context",
                ConsoleWorkspaceContextTray,
            )
            state = self._build_console_workspace_context_state(session_data)
            workspace_context.sync_state(state)
            try:
                details_tray = self.query_one(
                    "#console-workspace-details", ConsoleWorkspaceDetailsTray
                )
            except (NoMatches, QueryError):
                pass
            else:
                details_tray.sync_state(state)
            self.call_after_refresh(
                lambda: self.run_worker(
                    self._sync_console_legacy_workspace_context_aliases,
                    group="console-workspace-context-legacy-aliases",
                    exclusive=True,
                )
            )
        except (NoMatches, QueryError):
            logger.debug("No Console workspace context tray available for sync")

    async def _sync_console_legacy_workspace_context_aliases(self) -> None:
        """Expose transitional legacy new-conversation control while grouped browser is active."""
        try:
            workspace_context = self.query_one(
                "#console-workspace-context",
                ConsoleWorkspaceContextTray,
            )
        except (NoMatches, QueryError):
            return

        state = self._build_console_workspace_context_state()

        if not self.query("#console-new-workspace-conversation"):
            new_button = Button(
                "New conversation",
                id="console-new-workspace-conversation",
                classes="console-workspace-action",
                compact=True,
                disabled=not bool(state.new_conversation_enabled),
            )
            matches = list(self.query("#console-workspace-conversations"))
            before_status = matches[0] if matches else None
            if before_status is not None:
                await workspace_context.mount(new_button, before=before_status)
            else:
                await workspace_context.mount(new_button)

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
        provider_display, model, settings = self._active_console_provider_model_display()
        _effective_settings, settings_readiness = self._active_console_settings_readiness()
        explicit_provider_ready = getattr(self.app_instance, "console_provider_ready", None)
        provider_readiness = get_provider_readiness(
            (settings.provider if settings is not None else None) or provider_display,
            self._provider_readiness_app_config(),
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
        inspector_state = ConsoleInspectorState.from_values(
            live_work_title=pending_launch.title if pending_launch else None,
            provider_label=provider_display,
            model_label=model,
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
        setup_blocker_copy = self._console_provider_blocker_copy()
        if setup_blocker_copy:
            action_label, _action_target, _action_tooltip = self._console_provider_recovery_action()
            setup_rows = (
                ConsoleDisplayRow("Setup", "Provider configuration required", status="blocked"),
                ConsoleDisplayRow(
                    "Blocked impact",
                    "Send is blocked until setup is finished.",
                    status="blocked",
                    recovery=setup_blocker_copy,
                ),
                ConsoleDisplayRow(
                    "Next action",
                    action_label or "Open Settings",
                    status="blocked",
                ),
            )
            inspector_state = replace(
                inspector_state,
                rows=setup_rows + inspector_state.rows,
            )
        selected_rows = self._selected_console_message_inspector_rows()
        conversation_rows = self._selected_console_conversation_inspector_rows()
        if conversation_rows:
            inspector_state = replace(
                inspector_state,
                rows=conversation_rows + inspector_state.rows,
            )
        if selected_rows:
            inspector_state = replace(
                inspector_state,
                rows=inspector_state.rows + selected_rows,
            )
        # P1g: project the cached "what's in play" dictionary summary --
        # NO DB I/O here, only `self._active_dictionaries_summary` (kept
        # current by `refresh_active_dictionaries_summary()`).
        inspector_state = replace(
            inspector_state,
            dictionary_rows=self._console_dictionary_inspector_rows(),
            dictionary_actions=self._console_dictionary_inspector_actions(),
        )
        return inspector_state

    def _dictionary_scope_service(self) -> Any:
        """The app-level chat-dictionary scope service, or None when absent."""
        return getattr(self.app_instance, "chat_dictionary_scope_service", None)

    def _console_chat_dictionary_applier(self, conversation_id: str | None, text: str) -> str:
        """Bound applier handed to the native Console controller: apply the
        active CONVERSATION chat dictionaries to a send's text (never raises).

        Resolves the db lazily (at call time), so a controller built before the
        db is ready still works. Conversation-only: ``char_data`` is ``None``
        (native sessions carry no character card yet).
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None or not conversation_id or not isinstance(text, str):
            return text
        from ...Character_Chat import Chat_Dictionary_Lib as cdl
        return cdl.apply_active_chatdicts_to_text(
            db,
            conversation_id,
            None,
            text,
            max_tokens=_CHATDICT_MAX_TOKENS,
            strategy=_CHATDICT_STRATEGY,
        )

    def _active_console_dictionary_scope_ids(self) -> tuple[str | None, int | None]:
        """Return (conversation_id, character_id) for the active native
        Console session's chat-dictionary scope.

        `conversation_id` comes from `_current_console_rail_conversation_id()`
        -- the existing accessor that already resolves the active native
        session's `persisted_conversation_id` (falling back to the legacy
        tab-container conversation id when no native session exists, e.g. in
        legacy-only test harnesses). `character_id` is always `None`: native
        Console sessions do not yet track a numeric character id --
        `ConsoleSessionSettings.character_label` is only a free-text display
        string, never a DB id. Character-scoped dictionary attachment for the
        native Console is net-new work (tracked separately as Roleplay P1e
        Attachments); once it lands, this is the one place to source a real
        character id from.
        """
        return self._current_console_rail_conversation_id(), None

    async def refresh_active_dictionaries_summary(self) -> None:
        """Recompute and cache the "what's in play" chat-dictionary summary
        for the active native Console session's conversation/character, then
        refresh the Console inspector.

        This is the ONLY place that performs the (DB-backed) dictionary
        summarize call -- `_build_console_inspector_state` (and therefore
        every Console recompose/refresh) reads only the cache set here. The
        scope-service call is marshalled onto a worker thread via
        `asyncio.to_thread` so this never performs a synchronous DB read on
        the UI event loop.
        """
        conversation_id, character_id = self._active_console_dictionary_scope_ids()
        service = self._dictionary_scope_service()
        if service is None or (conversation_id is None and character_id is None):
            self._active_dictionaries_summary = {"dictionaries": []}
        else:
            try:
                summary = await asyncio.to_thread(
                    _run_dictionary_summary_off_thread,
                    service,
                    conversation_id,
                    character_id,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not summarize active chat dictionaries for the Console inspector."
                )
                summary = {"dictionaries": []}
            self._active_dictionaries_summary = (
                summary if isinstance(summary, dict) else {"dictionaries": []}
            )
        self._sync_console_control_bar()

    async def _refresh_active_dictionaries_summary_if_scope_changed(self) -> None:
        """Recompute the "what's in play" summary only when the active
        native Console session's dictionary scope actually changed.

        Called from `_sync_native_console_chat_ui()`, the central Console
        UI-sync entrypoint -- which also runs on a 0.2s transcript-poll timer
        while a run is streaming (`_start_console_transcript_sync_timer`).
        Without this guard, every one of those polls would re-run the
        DB-backed summarize call. Mirrors the change-guard the previous
        (legacy-reactive) wiring had on the app-level watchers, relocated to
        the actual native-session change signal.
        """
        scope_ids = self._active_console_dictionary_scope_ids()
        if scope_ids == self._last_console_dictionary_scope_ids:
            return
        self._last_console_dictionary_scope_ids = scope_ids
        await self.refresh_active_dictionaries_summary()

    def _console_dictionary_inspector_rows(self) -> tuple[ConsoleDisplayRow, ...]:
        """Project the cached dictionary summary into inspector rows.

        Reads ONLY `self._active_dictionaries_summary` (and the active native
        Console session's conversation/character ids, to distinguish "no
        active chat" from "no dictionaries attached yet") -- never touches
        the DB.
        """
        conversation_id, character_id = self._active_console_dictionary_scope_ids()
        if conversation_id is None and character_id is None:
            return (ConsoleDisplayRow("No active chat", ""),)

        summary = self._active_dictionaries_summary or {}
        dictionaries = summary.get("dictionaries") or []
        if not dictionaries:
            return (ConsoleDisplayRow("No dictionaries in play", ""),)

        rows = []
        for entry in dictionaries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "Unnamed")
            value = "from conversation" if entry.get("source") == "conversation" else "from character"
            if entry.get("shadowed"):
                value += " (shadowed)"
            if not entry.get("enabled", True):
                value += " (disabled)"
            rows.append(ConsoleDisplayRow(name, value))
        return tuple(rows)

    def _console_dictionary_inspector_actions(self) -> tuple[ConsoleInspectorAction, ...]:
        """Attach/Detach actions for the Console dictionary inspector block.

        Reads ONLY the cache + the active native Console session's
        conversation id -- never the DB.
        """
        conversation_id, _character_id = self._active_console_dictionary_scope_ids()
        summary = self._active_dictionaries_summary or {}
        dictionaries = summary.get("dictionaries") or []
        has_conversation_dictionary = any(
            isinstance(entry, dict) and entry.get("source") == "conversation"
            for entry in dictionaries
        )
        return (
            ConsoleInspectorAction(
                "console-inspector-dictionaries-attach",
                "Attach dictionary…",
                enabled=bool(conversation_id),
                disabled_reason="Start or load a conversation first",
            ),
            ConsoleInspectorAction(
                "console-inspector-dictionaries-detach",
                "Detach dictionary…",
                enabled=has_conversation_dictionary,
            ),
        )

    async def _console_dictionary_attach_worker(self) -> None:
        """Pick and attach a chat dictionary to the active Console conversation.

        Mirrors P1f's ``_character_dictionary_attach_worker``
        (``UI/Screens/personas_screen.py``) structurally: every await is
        individually guarded so no exception escapes the worker boundary --
        an uncaught worker exception kills the whole app under
        ``run_worker(exit_on_error=True)``.
        """
        try:
            conversation_id = self._current_console_rail_conversation_id()
            if not conversation_id:
                self.app_instance.notify("Start or load a conversation first.", severity="warning")
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            try:
                rows = await asyncio.to_thread(console_attachable_dictionaries, db, conversation_id)
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load dictionaries for the Console attach picker."
                )
                return
            if not rows:
                self.app_instance.notify("No more dictionaries to attach.", severity="information")
                return
            try:
                picked = await self.app_instance.push_screen_wait(DictionaryPicker(rows))
            except Exception:
                logger.opt(exception=True).warning("Could not show the Console dictionary picker.")
                return
            if not picked:
                return
            await handle_console_dictionary_attach(self.app_instance, conversation_id, picked)
            # Always resync after an attempted attach (spec AC5: ConflictError
            # -> notify + refresh): on success the summary gains the dict; on a
            # ConflictError the DB changed under us and the cache must re-read
            # the current truth rather than stay stale until the next switch.
            await self.refresh_active_dictionaries_summary()
        finally:
            self._console_dictionary_dialog_active = False

    async def _console_dictionary_detach_worker(self) -> None:
        """Pick and detach a chat dictionary from the active Console conversation.

        Analogous to :meth:`_console_dictionary_attach_worker`, over
        ``console_attached_dictionaries``/``handle_console_dictionary_detach``.
        """
        try:
            conversation_id = self._current_console_rail_conversation_id()
            if not conversation_id:
                self.app_instance.notify("Start or load a conversation first.", severity="warning")
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            try:
                rows = await asyncio.to_thread(console_attached_dictionaries, db, conversation_id)
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load dictionaries for the Console detach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No dictionaries attached to this conversation.", severity="information"
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(
                    DictionaryPicker(rows, title="Detach dictionary", confirm_label="Detach")
                )
            except Exception:
                logger.opt(exception=True).warning("Could not show the Console dictionary picker.")
                return
            if not picked:
                return
            await handle_console_dictionary_detach(self.app_instance, conversation_id, picked)
            # Always resync after an attempted detach (spec AC5: ConflictError
            # -> notify + refresh); see _console_dictionary_attach_worker.
            await self.refresh_active_dictionaries_summary()
        finally:
            self._console_dictionary_dialog_active = False

    def _selected_console_conversation_inspector_rows(self) -> tuple[ConsoleDisplayRow, ...]:
        """Return inspector rows for the active Console conversation/session."""
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return (
                ConsoleDisplayRow("Selected conversation", "No active conversation"),
                ConsoleDisplayRow("Conversation source", "none"),
            )

        try:
            active_session = store.switch_session(store.active_session_id)
        except KeyError:
            return (
                ConsoleDisplayRow("Selected conversation", "No active conversation"),
                ConsoleDisplayRow("Conversation source", "none"),
            )

        workspace_state = self._build_console_workspace_context_state()
        workspace_label = str(workspace_state.workspace_label or "").strip()
        if workspace_label.startswith("Workspace: "):
            workspace_label = workspace_label.removeprefix("Workspace: ").strip()
        workspace_label = workspace_label or str(active_session.workspace_id or "Default")
        persisted_id = str(active_session.persisted_conversation_id or "").strip()
        source = "saved conversation" if persisted_id else "native Console session"
        resume_state = (
            f"restored from {persisted_id}"
            if persisted_id
            else "local session, not persisted yet"
        )
        return (
            ConsoleDisplayRow("Selected conversation", active_session.title),
            ConsoleDisplayRow("Conversation source", source),
            ConsoleDisplayRow("Workspace", workspace_label),
            ConsoleDisplayRow("Resume state", resume_state),
        )

    def _selected_console_message_inspector_rows(self) -> tuple[ConsoleDisplayRow, ...]:
        """Return inspector guidance for the currently selected transcript message."""
        try:
            transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        except QueryError:
            return ()
        message_id = transcript.selected_message_id
        if message_id is None:
            return ()
        try:
            message = self._ensure_console_chat_store().get_message(message_id)
        except KeyError:
            return ()

        rows = [
            ConsoleDisplayRow(
                "Selected message",
                f"{self._console_message_role_label(message)} message",
            ),
            ConsoleDisplayRow(
                "Message actions",
                "Copy, Edit, Save as..., Regenerate, Continue, Feedback, Delete",
            ),
            ConsoleDisplayRow(
                "Keyboard",
                "Tab/Shift+Tab cycle actions; Enter activates; Esc clears selection",
            ),
        ]
        if message.variants is not None:
            rows.append(
                ConsoleDisplayRow(
                    "Variants",
                    (
                        f"{len(message.variants.variants)} variants, "
                        f"showing {message.variants.selected_index + 1}/"
                        f"{len(message.variants.variants)}"
                    ),
                )
            )
        excerpt = self._console_message_excerpt(message, max_length=90)
        if excerpt:
            rows.append(ConsoleDisplayRow("Excerpt", excerpt))
        if self._pending_console_delete_message_id == message.id:
            rows.append(
                ConsoleDisplayRow(
                    "Delete confirmation",
                    "Press Delete again to remove this message.",
                    status="blocked",
                )
            )
        return tuple(rows)

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
        widget.styles.max_height = 0
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
    def _compact_console_workbench_widget(widget: Any, height: int = 1) -> Any:
        """Keep Console Workbench primitives visible without shrinking the grid."""
        widget.styles.height = height
        widget.styles.min_height = height
        widget.styles.max_height = height
        return widget

    @staticmethod
    def _hidden_console_workbench_widget(widget: Any) -> Any:
        """Keep Console Workbench compatibility seams mounted without layout cost."""
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

        persona = str(control_state.persona_label or "General")
        if persona.startswith("Assistant: "):
            persona = persona.removeprefix("Assistant: ").strip() or "General"
        elif persona.startswith("Persona: "):
            persona = persona.removeprefix("Persona: ").strip() or "Persona"
        return (
            "Chat/RAG/Follow"
            f" | {persona}"
            f" | Sources {readiness_count(control_state.sources_label)}"
            f" | Tools {readiness_count(control_state.tools_label)}"
            f" | Approvals {readiness_count(control_state.approvals_label)}"
        )

    def _build_console_workbench_state(self, control_state: ConsoleControlState):
        blocker_copy = self._console_provider_blocker_copy()
        action_label, _action_target, _action_tooltip = self._console_provider_recovery_action()
        composer = self._console_composer_or_none()
        has_draft = bool(composer and composer.draft_text().strip())
        controller = self._console_chat_controller
        run_state = getattr(controller, "run_state", None) if controller is not None else None
        can_stop = bool(getattr(run_state, "is_stop_allowed", False))
        run_allows_send = bool(getattr(run_state, "is_send_allowed", True))
        can_send = (
            has_draft
            and not bool(self._console_setup_blocked_reason())
            and run_allows_send
        )
        return build_console_workbench_state(
            control_state=control_state,
            provider_blocker_copy=blocker_copy,
            provider_action_label=action_label,
            can_send=can_send,
            can_stop=can_stop,
            can_save_chatbook=self._console_chatbook_action_available(),
            density=self._console_workbench_density(),
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
            (settings.provider if settings is not None else None) or provider,
            self._provider_readiness_app_config(),
        )
        if provider_readiness.reason == "Missing API key":
            return f"Provider setup needed: {provider} missing API key"
        return f"Provider setup needed: {settings_readiness.detail}"

    @staticmethod
    def _console_empty_recovery_action_copy(
        blocker_copy: str,
        *,
        provider_action_label: str = "",
        provider_action_tooltip: str = "",
    ) -> tuple[str, str]:
        """Return empty-state provider recovery button label and tooltip."""
        blocker = blocker_copy.strip().lower()
        if provider_action_label:
            return provider_action_label, provider_action_tooltip.strip()
        if "choose a provider" in blocker:
            return "Choose provider", "Choose a provider for this Console session"
        if "choose a model" in blocker:
            return "Choose model", "Choose a model for this Console session"
        if "api key" in blocker:
            return (
                CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
                "Configure API and API key before sending",
            )
        if "endpoint" in blocker:
            return "Configure endpoint", "Configure the provider endpoint before sending"
        if blocker:
            return "Review settings", "Review Console provider settings before sending"
        return "Choose model", "Choose the provider and model for this Console session."

    def _console_setup_blocked_reason(self) -> str:
        """Return setup-specific send blocker copy for the native composer."""
        blocker = self._console_provider_blocker_copy().strip().lower()
        if not blocker:
            return ""
        if blocker == "provider setup needed: choose a model":
            return "Choose a model in Console Settings before sending."
        if "missing api key" in blocker:
            return "Add API key in Settings > Providers & Models before sending."
        if "save the endpoint in settings" in blocker:
            return "Save provider endpoint in Settings > Providers & Models before sending."
        return "Finish provider setup before sending."

    def _console_provider_recovery_field(self) -> str:
        """Return the Settings Providers & Models field targeted by recovery."""
        provider, model, settings = self._active_console_provider_model_display()
        if not _has_selected_text(provider) or not _has_selected_text(model):
            return ""

        _effective_settings, settings_readiness = self._active_console_settings_readiness()
        if settings_readiness.native_send_supported:
            return ""

        provider_readiness = get_provider_readiness(
            (settings.provider if settings is not None else None) or provider,
            self._provider_readiness_app_config(),
        )
        if provider_readiness.reason == "Missing API key":
            return "api_key"
        if settings_readiness.label == "Endpoint not saved":
            return "endpoint"
        return ""

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
            (settings.provider if settings is not None else None) or provider,
            self._provider_readiness_app_config(),
        )
        if provider_readiness.reason == "Missing API key":
            return (
                CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
                "settings",
                f"Configure {provider} API and API key in Settings",
            )
        if settings_readiness.label == "Endpoint not saved":
            return (
                "Configure endpoint",
                "settings",
                f"Save the {provider} endpoint in Settings",
            )
        return ("Review settings", "console", "Review this Console session's settings")

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

    def _active_console_transcript_has_messages(self) -> bool:
        """Return whether the active Console session's store transcript has messages."""
        store = self._console_chat_store
        if store is None:
            return False
        session_id = store.active_session_id
        if session_id is None:
            return False
        return bool(store.messages_for_session(session_id))

    def _build_console_setup_card_state(self) -> ConsoleSetupCardState:
        """Build the empty-transcript onboarding state from current readiness."""
        settings, readiness = self._active_console_settings_readiness()
        has_model = _has_selected_text(getattr(settings, "model", None))
        return build_console_setup_card_state(
            readiness=readiness,
            provider_label=str(getattr(settings, "provider", "") or "Provider"),
            has_model=has_model,
            first_send_completed=self._console_first_send_completed(),
            has_messages=self._active_console_transcript_has_messages(),
            guidance_dismissed=self._console_guidance_dismissed,
        )

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
        action_label, _action_target, action_tooltip = self._console_provider_recovery_action()
        empty_action_label, empty_action_tooltip = self._console_empty_recovery_action_copy(
            blocker_copy,
            provider_action_label=action_label if blocker_copy else "",
            provider_action_tooltip=action_tooltip if blocker_copy else "",
        )
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

        card_state = self._build_console_setup_card_state()
        try:
            surface = self.query_one("#console-session-surface", ConsoleSessionSurface)
        except QueryError:
            pass
        else:
            surface.sync_inline_guidance(
                card_state,
                provider_action_label=empty_action_label,
                provider_action_tooltip=empty_action_tooltip,
            )

        self._sync_console_setup_modal(
            card_state,
            action_label=empty_action_label,
            action_tooltip=empty_action_tooltip,
        )

    def _sync_console_setup_modal(
        self,
        card_state: ConsoleSetupCardState,
        *,
        action_label: str,
        action_tooltip: str,
    ) -> None:
        """Show/hide the blocking setup modal and keep the workbench inert."""
        try:
            modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        except QueryError:
            return
        modal.sync_card_state(
            card_state,
            action_label=action_label,
            action_tooltip=action_tooltip,
        )
        modal.sync_detected_server_action(
            build_console_detected_server_action(
                self._console_detected_local_server,
                card_mode=card_state.mode,
            )
        )
        blocking = modal.is_blocking
        self._apply_console_setup_block(blocking)
        if blocking:
            self._maybe_start_console_local_discovery()
            self.call_after_refresh(modal.focus_primary_action)

    def _maybe_start_console_local_discovery(self) -> None:
        """Start the one-shot local-server discovery worker while blocked.

        Discovery runs at most once per screen, in its own exclusive worker
        group so it can never cancel (or be duplicated alongside) the Console
        UI sync workers. Results only ever add a secondary card affordance;
        a quiet network stays quiet.
        """
        if self._console_local_discovery_started:
            return
        self._console_local_discovery_started = True
        self.run_worker(
            self._discover_local_servers_for_setup_card(),
            exclusive=True,
            group="console-local-server-discovery",
        )

    async def _discover_local_servers_for_setup_card(self) -> None:
        """Probe localhost servers and surface the first hit on the card.

        Uses the ``console_local_server_discovery`` app attribute as a test
        seam when present; otherwise probes via
        ``local_server_discovery.discover_local_servers`` (localhost-only,
        short timeout).
        """
        discover = getattr(self.app_instance, "console_local_server_discovery", None)
        if not callable(discover):
            discover = discover_local_servers
        try:
            servers = tuple(await discover(self._provider_readiness_app_config()) or ())
        except Exception:
            logger.debug("Console local-server discovery failed", exc_info=True)
            return
        if not servers:
            return
        self._console_detected_local_server = servers[0]
        self._sync_console_transcript_guidance()

    def _apply_detected_local_server(self) -> None:
        """Adopt the detected local server as the Console provider.

        Persists ``chat_defaults.provider``/``model`` and the provider's
        ``api_settings`` endpoint via ``save_settings_to_cli_config``, then
        applies the same selection to the active session as an explicit user
        choice (mirroring the settings-modal apply path) and re-evaluates the
        setup card from the fresh on-disk config (task-177 mechanics; no
        boot-time snapshots).
        """
        server = self._console_detected_local_server
        if server is None:
            return
        model_id = server.model_ids[0] if server.model_ids else None
        provider_values: dict[str, object] = {"api_url": server.base_url}
        chat_defaults: dict[str, object] = {"provider": server.provider_key}
        if model_id:
            provider_values["model"] = model_id
            chat_defaults["model"] = model_id
        try:
            saved = save_settings_to_cli_config(
                {
                    f"api_settings.{server.provider_key}": provider_values,
                    "chat_defaults": chat_defaults,
                }
            )
        except Exception:
            saved = False
        if not saved:
            logger.warning(
                "Could not persist detected local server defaults to config; "
                "applying to this session only"
            )
        settings = build_default_console_session_settings(
            self._provider_readiness_app_config(),
            server.provider_key,
            model_id,
        )
        if provider_config_key(settings.provider) in {"llama_cpp", "local_llamacpp"}:
            settings = replace(settings, base_url=None)
        self._replace_active_console_session_settings(replace(settings, source="user"))
        self._sync_console_transcript_guidance()
        self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")

    def _console_setup_modal_blocking(self) -> bool:
        """Return True when the first-run setup modal is covering the workbench."""
        try:
            modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        except QueryError:
            return False
        return bool(getattr(modal, "display", False)) and modal.is_blocking

    def _apply_console_setup_block(self, blocking: bool) -> None:
        """Disable composer focus/typing while the setup modal is up."""
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        composer.can_focus = not blocking
        if blocking and self._is_descendant_or_self(self.app.focused, composer):
            # Pull keyboard focus off the covered composer so typing can't tunnel.
            try:
                self.query_one("#console-setup-modal", ConsoleSetupModal).focus_primary_action()
            except QueryError:
                composer.blur()

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
    def _staged_context_frame_variant(_state: ConsoleStagedContextState) -> str:
        """Always use quiet framing; the rail frame is the single border source."""
        return "quiet"

    @staticmethod
    def _workspace_context_frame_variant(_state: ConsoleWorkspaceContextState) -> str:
        """Keep workspace context visually nested inside the framed left rail."""
        return "quiet"

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

    @on(Button.Pressed, "#console-run-library-rag")
    def handle_console_run_library_rag(self, event: Button.Pressed) -> None:
        """Request Library retrieval from the Console source-readiness seam."""
        event.stop()
        self._run_console_library_rag_from_visible_action()

    def _run_console_library_rag_from_visible_action(self) -> None:
        """Request Library retrieval from the visible Console action surface."""
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

    @work(exclusive=True, group="console-library-rag-search")
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
        self._pending_console_launch_auto_open_inspector = True
        
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
        rail_state = self._apply_pending_launch_inspector_auto_open(
            rail_state,
            pending_launch,
        )
        workbench_state = self._build_console_workbench_state(control_state)
        shell_classes = (
            "workbench-frame console-workbench-frame "
            f"density-{workbench_state.density}"
        )
        with Vertical(id="console-shell", classes=shell_classes):
            yield self._hidden_console_workbench_widget(
                DestinationHeader(
                    workbench_state.header,
                    id="console-workbench-header",
                    classes="workbench-header",
                )
            )
            yield self._hidden_console_workbench_widget(
                ModeStrip(
                    workbench_state.modes,
                    id="console-workbench-mode-strip",
                    classes="workbench-mode-strip",
                )
            )
            yield self._hidden_console_workbench_widget(
                CommandStrip(
                    workbench_state.actions,
                    id="console-workbench-command-strip",
                    classes="workbench-command-strip",
                )
            )
            yield self._compact_console_workbench_widget(
                RecoveryCallout(
                    workbench_state.recovery,
                    id="workbench-recovery-callout",
                    classes="workbench-recovery-callout",
                ),
                height=4,
            )
            # Compatibility selectors retained during Console Workbench parity:
            # #console-title and #console-mode-bar are legacy shell seams now
            # represented by DestinationHeader and ModeStrip. #console-control-bar
            # remains visible as the dense Console-owned control surface.
            yield self._hidden_static(
                "Console",
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
            yield self._hidden_static(
                self._console_mode_summary(control_state),
                id="console-mode-bar",
                classes="ds-panel",
            )
            yield self._compact_console_workbench_widget(
                ConsoleControlBar(
                    control_state,
                    self.app_instance,
                    actions=workbench_state.actions,
                    on_sidebar_toggle_requested=self._toggle_console_chat_sidebar,
                    id="console-control-bar",
                    classes="console-control-bar",
                ),
                height=2,
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
                left_rail.can_focus = True
                left_rail.styles.width = "3fr"
                # Compact contract: left rail + main column + the collapsed
                # inspector handle (11) must fit a 100-column terminal.
                left_rail.styles.min_width = 24
                if not rail_state.left_open:
                    left_rail.styles.display = "none"
                with self._frame_console_region(left_rail):
                    left_rail_header = Horizontal(classes="console-rail-header")
                    left_rail_header.styles.height = 1
                    left_rail_header.styles.min_height = 1
                    left_rail_header.styles.max_height = 1
                    with left_rail_header:
                        # Titled distinctly from the "Context" (staged sources)
                        # rail section below so no two rail titles collide.
                        rail_label = Static(
                            "Session & Context",
                            id="console-context-rail-title",
                            classes="console-rail-title",
                        )
                        rail_label.styles.width = "1fr"
                        yield rail_label
                        collapse_button = Button(
                            GLYPH_COLLAPSE_LEFT,
                            id="console-context-rail-collapse",
                            classes="console-rail-collapse-button",
                            compact=True,
                        )
                        collapse_button.tooltip = "Collapse Session & Context rail"
                        collapse_button.styles.width = 3
                        collapse_button.styles.min_width = 3
                        collapse_button.styles.max_width = 3
                        yield collapse_button
                    with VerticalScroll(
                        id="console-left-rail-body",
                        classes="console-left-rail-body",
                    ):
                        # Section 1: Session (workspace + conversations).
                        yield ConsoleRailSectionHeader(
                            "Session",
                            section_id="session",
                            open=rail_state.session_open,
                            id="console-rail-section-header-session",
                        )
                        session_body = Vertical(
                            id="console-rail-section-body-session",
                            classes="console-rail-section-body",
                        )
                        session_body.styles.height = "auto"
                        if not rail_state.session_open:
                            session_body.styles.display = "none"
                        with session_body:
                            workspace_context_tray = ConsoleWorkspaceContextTray(
                                workspace_context_state,
                                show_heading=False,
                                id="console-workspace-context",
                                classes="console-left-rail-section",
                            )
                            workspace_context_tray.styles.width = "100%"
                            workspace_context_tray.styles.min_width = 0
                            yield self._frame_console_region(
                                workspace_context_tray,
                                variant=self._workspace_context_frame_variant(
                                    workspace_context_state
                                ),
                            )

                        # Section 2: Context (staged sources).
                        yield ConsoleRailSectionHeader(
                            "Context",
                            section_id="context",
                            open=rail_state.context_open,
                            id="console-rail-section-header-context",
                        )
                        context_body = Vertical(
                            id="console-rail-section-body-context",
                            classes="console-rail-section-body",
                        )
                        context_body.styles.height = "auto"
                        if not rail_state.context_open:
                            context_body.styles.display = "none"
                        with context_body:
                            staged_context_tray = ConsoleStagedContextTray(
                                staged_context_state,
                                id="console-staged-context-tray",
                                classes="console-left-rail-section",
                            )
                            staged_context_tray.styles.width = "100%"
                            staged_context_tray.styles.min_width = 0
                            staged_context_tray.styles.height = "auto"
                            staged_context_tray.styles.min_height = (
                                3 if staged_context_state.is_empty else 4
                            )
                            staged_context_tray.styles.max_height = (
                                6 if staged_context_state.is_empty else 10
                            )
                            yield self._frame_console_region(
                                staged_context_tray,
                                variant=self._staged_context_frame_variant(
                                    staged_context_state
                                ),
                            )

                        # Section 3: Model (compact settings summary).
                        yield ConsoleRailSectionHeader(
                            "Model",
                            section_id="model",
                            open=rail_state.model_open,
                            id="console-rail-section-header-model",
                        )
                        model_body = Vertical(
                            id="console-rail-section-body-model",
                            classes="console-rail-section-body",
                        )
                        model_body.styles.height = "auto"
                        if not rail_state.model_open:
                            model_body.styles.display = "none"
                        with model_body:
                            model_line1, model_line2 = build_console_model_section_lines(
                                self._build_console_settings_summary_state()
                            )
                            line1 = Static(
                                model_line1,
                                id="console-model-section-line1",
                                classes="console-model-section-line",
                                markup=False,
                            )
                            # The rail line is clipped to one row; without
                            # nowrap a long model token word-wraps onto the
                            # hidden second row and vanishes ("llama_cpp / ").
                            line1.styles.text_wrap = "nowrap"
                            line1.styles.text_overflow = "ellipsis"
                            yield line1
                            line2 = Static(
                                model_line2,
                                id="console-model-section-line2",
                                classes="console-model-section-line",
                                markup=False,
                            )
                            line2.styles.text_wrap = "nowrap"
                            line2.styles.text_overflow = "ellipsis"
                            yield line2
                            system_line_text, system_line_dim = (
                                self._console_rail_system_line_state()
                            )
                            system_line = Static(
                                system_line_text,
                                id="console-rail-system-line",
                                markup=False,
                            )
                            # Same one-row clipping hazard as the model line
                            # above (task-186): nowrap + ellipsis so a long
                            # system prompt truncates visibly instead of
                            # word-wrapping onto a hidden second row.
                            system_line.styles.text_wrap = "nowrap"
                            system_line.styles.text_overflow = "ellipsis"
                            system_line.set_class(
                                system_line_dim, "console-rail-system-line-dim"
                            )
                            yield system_line
                            configure = Button(
                                "Configure",
                                id="console-model-section-configure",
                                classes="console-workspace-action",
                                compact=True,
                            )
                            configure.tooltip = "Configure Console session settings"
                            yield configure

                        # Section 4: Agent (run inspector -- the watch-and-drill
                        # surface for the live/most-recent agent run and its
                        # historical sub-agent runs).
                        yield ConsoleRailSectionHeader(
                            "Agent",
                            section_id="agent",
                            open=rail_state.agent_open,
                            id="console-rail-section-header-agent",
                        )
                        agent_body = Vertical(
                            id="console-rail-section-body-agent",
                            classes="console-rail-section-body console-agent-section",
                        )
                        agent_body.styles.height = "auto"
                        if not rail_state.agent_open:
                            agent_body.styles.display = "none"
                        with agent_body:
                            status_line, steps_text, subagents_text = (
                                self._console_agent_section_lines())
                            yield Static(status_line, id="console-agent-section-status",
                                         classes="console-agent-section-line", markup=False)
                            yield Static(steps_text, id="console-agent-section-steps",
                                         classes="console-agent-section-steps", markup=False)
                            yield Static(subagents_text, id="console-agent-section-subagents",
                                         classes="console-agent-section-subagents", markup=False)
                            back_button = Button(
                                "Back",
                                id="console-agent-drilldown-back",
                                classes="console-workspace-action console-agent-drilldown-back",
                                compact=True,
                            )
                            back_button.tooltip = "Return to the live agent run view"
                            if not self._console_agent_drilldown_run_id:
                                back_button.styles.display = "none"
                            yield back_button

                        # Section 5: Details (storage, sync, handoff plumbing).
                        yield ConsoleRailSectionHeader(
                            "Details",
                            section_id="details",
                            open=rail_state.details_open,
                            id="console-rail-section-header-details",
                        )
                        details_body = Vertical(
                            id="console-rail-section-body-details",
                            classes="console-rail-section-body",
                        )
                        details_body.styles.height = "auto"
                        if not rail_state.details_open:
                            details_body.styles.display = "none"
                        with details_body:
                            details_tray = ConsoleWorkspaceDetailsTray(
                                workspace_context_state,
                                id="console-workspace-details",
                                classes="console-left-rail-section",
                            )
                            details_tray.styles.width = "100%"
                            details_tray.styles.min_width = 0
                            yield details_tray

                main_column = Vertical(id="console-main-column")
                main_column.styles.width = "13fr"
                main_column.styles.min_width = 56
                with main_column:
                    transcript_region = self._frame_console_region(
                        Vertical(id="console-transcript-region", classes="console-region"),
                        top=False,
                    )
                    with transcript_region:
                        provider_blocker_copy = self._console_provider_blocker_copy()
                        guidance_visible = self._console_guidance_visible(provider_blocker_copy)
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
                right_rail.can_focus = True
                right_rail.styles.width = "4fr"
                right_rail.styles.min_width = 34
                if not rail_state.right_open:
                    right_rail.styles.display = "none"
                with self._frame_console_region(right_rail):
                    right_rail_header = Horizontal(classes="console-rail-header")
                    right_rail_header.styles.height = 1
                    right_rail_header.styles.min_height = 1
                    right_rail_header.styles.max_height = 1
                    with right_rail_header:
                        rail_label = Static(
                            "Inspector",
                            id="console-inspector-rail-title",
                            classes="console-rail-title",
                        )
                        rail_label.styles.width = "1fr"
                        yield rail_label
                        collapse_button = Button(
                            GLYPH_COLLAPSE_RIGHT,
                            id="console-inspector-rail-collapse",
                            classes="console-rail-collapse-button",
                            compact=True,
                        )
                        collapse_button.tooltip = "Collapse Inspector rail"
                        collapse_button.styles.width = 3
                        collapse_button.styles.min_width = 3
                        collapse_button.styles.max_width = 3
                        yield collapse_button
                    with VerticalScroll(
                        id="console-inspector-rail-body",
                        classes="console-inspector-rail-body",
                    ):
                        with Vertical(id="console-run-inspector"):
                            yield ConsoleRunInspector(
                                inspector_state,
                                id="console-run-inspector-state",
                            )
                            settings_summary = ConsoleSettingsSummary(
                                self._build_console_settings_summary_state(),
                                id="console-settings-summary",
                                classes="console-inspector-session-settings console-settings-summary",
                            )
                            settings_summary.styles.width = "100%"
                            settings_summary.styles.min_width = 0
                            yield settings_summary
                        if pending_launch:
                            yield from self._render_console_live_work_status_card(
                                pending_launch
                            )
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
                right_handle.styles.width = 11
                right_handle.styles.min_width = 11
                right_handle.styles.max_width = 11
                if rail_state.right_open:
                    right_handle.styles.display = "none"
                yield self._frame_console_region(right_handle, variant="quiet")
            yield self._frame_console_region(
                ConsoleComposerBar(
                    id="console-native-composer",
                    classes="ds-panel",
                    collapse_large_pastes=self._console_collapse_large_pastes_enabled(),
                    paste_collapse_threshold=self._console_paste_collapse_threshold(),
                )
            )
            # Console-scoped first-run blocker. Sits on a dedicated overlay
            # layer over the whole Console shell so the workbench (rail,
            # transcript, tabs, composer) is covered/inert while setup is
            # incomplete; the app tab bar lives outside the shell and stays
            # reachable. Hidden until a card-mode state is synced in.
            yield ConsoleSetupModal(id="console-setup-modal")

    def _console_collapse_large_pastes_enabled(self) -> bool:
        """Return the app-level Console paste-collapse preference."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        console_config = app_config.get("console", {})
        if not isinstance(console_config, dict):
            return True
        return coerce_bool_setting(console_config.get("collapse_large_pastes", True), True)

    def _console_paste_collapse_threshold(self) -> int:
        """Return the app-level Console paste-collapse character threshold."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        console_config = app_config.get("console", {})
        if not isinstance(console_config, dict):
            return DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD
        return coerce_int_setting(
            console_config.get(
                "paste_collapse_threshold",
                DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            ),
            DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            minimum=MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
            maximum=MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
        )
    
    def on_mount(self) -> None:
        """Run diagnostics when first mounted (only once)."""
        # Call parent's on_mount
        super().on_mount()
        self._register_console_footer_shortcuts()
        
        if not self._diagnostics_run and self.chat_window:
            self._diagnostics_run = True
            # Run diagnostic in the background for the legacy direct widget only.
            self.set_timer(0.5, self._run_diagnostic)
        
        # Restore collapsible states after mount
        self.set_timer(0.1, self._restore_collapsible_states)
        self.set_timer(0.05, self.sync_task_resume_state)
        self.set_timer(0.15, self._consume_pending_chat_handoff)
        # Mirrors the handoff timer above: the native composer is not
        # guaranteed to exist in the DOM yet at this exact point (it can
        # still be settling in immediately after mount, same reason every
        # composer-touching test here awaits `_wait_for_selector` first) --
        # firing this immediately risked a silent, unrecoverable miss (the
        # pending field is cleared on first read, so a `QueryError` here
        # would have thrown the staged text away with nothing left to retry).
        self.set_timer(0.15, self._consume_pending_console_prompt_insert)
        self.call_after_refresh(self._sync_native_console_chat_ui)
        self.call_after_refresh(self._restore_console_workbench_focus)
        self.set_timer(0.2, self._restore_console_workbench_focus)
        self.run_worker(self._refresh_console_skill_candidates(), exclusive=False)

    async def on_unmount(self) -> None:
        """Release Console-native resources owned by this screen."""
        self._clear_console_footer_shortcuts()
        self._stop_console_transcript_sync_timer()
        controller = self._console_chat_controller
        if controller is not None:
            await controller.shutdown()
        gateway = self._console_provider_gateway
        close = getattr(gateway, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        self._console_provider_gateway = None
        self._console_chat_controller = None
        super().on_unmount()

    @staticmethod
    def _serialize_console_settings(
        settings: ConsoleSessionSettings | None,
    ) -> dict[str, Any] | None:
        """Return a JSON-safe snapshot of per-session Console settings."""
        if settings is None:
            return None
        return asdict(settings)

    @staticmethod
    def _restore_console_settings(
        payload: Any,
    ) -> ConsoleSessionSettings | None:
        """Return per-session Console settings from a saved state payload."""
        if not isinstance(payload, dict):
            return None
        valid_fields = set(ConsoleSessionSettings.__dataclass_fields__)
        values = {key: value for key, value in payload.items() if key in valid_fields}
        provider = str(values.get("provider") or "").strip()
        if not provider:
            return None
        values["provider"] = provider
        try:
            return ConsoleSessionSettings(**values)
        except TypeError:
            logger.opt(exception=True).debug("Skipping invalid Console session settings payload")
            return None

    @staticmethod
    def _serialize_console_variants(
        variants: ConsoleVariantSet | None,
    ) -> dict[str, Any] | None:
        """Return a JSON-safe snapshot of regenerated message variants."""
        if variants is None:
            return None
        return {
            "turn_id": variants.turn_id,
            "selected_index": variants.selected_index,
            "variants": [
                {"id": variant.id, "content": variant.content}
                for variant in variants.variants
            ],
        }

    @staticmethod
    def _restore_console_variants(payload: Any) -> ConsoleVariantSet | None:
        """Return regenerated message variants from a saved state payload."""
        if not isinstance(payload, dict):
            return None
        raw_variants = payload.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            return None
        variants: list[ConsoleVariant] = []
        for raw_variant in raw_variants:
            if not isinstance(raw_variant, dict):
                continue
            content = str(raw_variant.get("content") or "")
            variant_id = str(raw_variant.get("id") or uuid.uuid4())
            variants.append(ConsoleVariant(content=content, id=variant_id))
        if not variants:
            return None
        selected_index = payload.get("selected_index", 0)
        if not isinstance(selected_index, int):
            selected_index = 0
        selected_index = min(max(selected_index, 0), len(variants) - 1)
        return ConsoleVariantSet(
            turn_id=str(payload.get("turn_id") or uuid.uuid4()),
            variants=variants,
            selected_index=selected_index,
        )

    @classmethod
    def _serialize_console_message(
        cls,
        message: ConsoleChatMessage,
    ) -> dict[str, Any]:
        """Return a JSON-safe snapshot of a native Console transcript message."""
        role = message.role.value if hasattr(message.role, "value") else message.role
        return {
            "id": message.id,
            "role": role,
            "content": message.content,
            "turn_id": message.turn_id,
            "status": message.status,
            "persisted_message_id": message.persisted_message_id,
            "feedback": message.feedback,
            "variants": cls._serialize_console_variants(message.variants),
            "image_mime_type": getattr(message, "image_mime_type", None),
            "attachment_label": getattr(message, "attachment_label", None),
            # Labels only -- bytes are dropped from screen-state snapshots
            # the same way the legacy `image_data` scalar always has been.
            # `getattr` (not `message.attachments`) tolerates plain-object
            # stand-ins (e.g. a bare SimpleNamespace) that predate the
            # `attachments` field, matching this method's existing
            # tolerance for `image_mime_type`/`attachment_label` above.
            "attachment_labels": [
                attachment.display_name
                for attachment in getattr(message, "attachments", ())
            ],
        }

    @classmethod
    def _restore_console_message(cls, payload: Any) -> ConsoleChatMessage | None:
        """Return a native Console transcript message from a saved state payload."""
        if not isinstance(payload, dict):
            return None
        try:
            role = ConsoleMessageRole(str(payload.get("role") or "system"))
        except ValueError:
            role = ConsoleMessageRole.SYSTEM
        status = str(payload.get("status") or "complete")
        if status not in {"complete", "pending", "streaming", "stopped", "failed"}:
            status = "complete"
        feedback = payload.get("feedback")
        if feedback not in {None, "up", "down"}:
            feedback = None
        image_mime_type = (
            str(payload["image_mime_type"]) if payload.get("image_mime_type") else None
        )
        attachment_label = (
            str(payload["attachment_label"]) if payload.get("attachment_label") else None
        )
        raw_labels = payload.get("attachment_labels")
        if isinstance(raw_labels, list):
            attachment_labels = [str(label) for label in raw_labels]
        else:
            # Legacy payloads (saved before `attachment_labels` existed)
            # carried at most one label -- the singular `attachment_label`.
            attachment_labels = [attachment_label] if attachment_label else []
        # Metadata-only: bytes were never serialized, so every reconstructed
        # attachment starts with `data=None` (refilled by
        # `_rehydrate_console_message_image`/`_rehydrate_console_message_attachments`
        # after restore). `image_mime_type` is the only mime carried across
        # a screen-state snapshot, so it stands in for every position until
        # per-attachment mime types come back from the DB.
        attachments = tuple(
            MessageAttachment(
                data=None,
                mime_type=image_mime_type or "",
                display_name=label,
                position=index,
            )
            for index, label in enumerate(attachment_labels)
        )
        return ConsoleChatMessage(
            role=role,
            content=str(payload.get("content") or ""),
            id=str(payload.get("id") or uuid.uuid4()),
            turn_id=(
                str(payload["turn_id"])
                if payload.get("turn_id") is not None
                else None
            ),
            status=status,  # type: ignore[arg-type]
            persisted_message_id=(
                str(payload["persisted_message_id"])
                if payload.get("persisted_message_id") is not None
                else None
            ),
            variants=cls._restore_console_variants(payload.get("variants")),
            feedback=feedback,  # type: ignore[arg-type]
            image_mime_type=image_mime_type,
            attachment_label=attachment_label,
            attachments=attachments,
        )

    # App-object attribute holding staged-but-unsent attachments across screen
    # recreation. Full PendingAttachment objects (bytes included, so clipboard
    # grabs survive too) live in process memory ONLY — the stash never enters
    # screen-state serialization (the no-bytes-in-screen-state spec constraint)
    # and dies with the app (restart drops pendings; accepted trade, TASK-218).
    _CONSOLE_PENDING_STASH_ATTR = "_console_pending_attachment_stash"

    def _stash_console_pending_attachments(self, store: ConsoleChatStore) -> None:
        """Snapshot every session's staged attachments onto the app object.

        Overwrites the whole stash each save, so cleared/sent attachments and
        closed sessions never leave stale entries behind. Bounded by the
        staging cap (5/session) times the live session count.
        """
        app = getattr(self, "app_instance", None)
        if app is None:
            return  # bare/unit harness: nowhere to stash — nothing to preserve
        stash: dict[str, tuple[Any, ...]] = {}
        for session in store.sessions():
            try:
                pendings = store.pending_attachments(session.id)
            except KeyError:
                continue
            if pendings:
                stash[session.id] = tuple(pendings)
        setattr(app, self._CONSOLE_PENDING_STASH_ATTR, stash)

    def _adopt_console_pending_attachments(self, store: ConsoleChatStore) -> None:
        """Re-stage stashed attachments into the restored store, then empty
        the stash. Entries for sessions that no longer exist are dropped.

        The stash attribute is reset the moment it is read — every adopt
        attempt releases the byte references, even when the stash turned out
        malformed or nothing could be adopted (self-healing; the bytes must
        never outlive their one restore opportunity).
        """
        app = getattr(self, "app_instance", None)
        if app is None:
            return
        stash = getattr(app, self._CONSOLE_PENDING_STASH_ATTR, None)
        setattr(app, self._CONSOLE_PENDING_STASH_ATTR, {})
        if not isinstance(stash, dict) or not stash:
            return
        live_ids = {session.id for session in store.sessions()}
        for session_id, pendings in stash.items():
            if session_id not in live_ids:
                continue
            if not isinstance(pendings, (list, tuple)):
                continue
            for pending in pendings:
                if not store.add_pending_attachment(session_id, pending):
                    break  # staging cap reached — matches live staging semantics

    def _serialize_native_console_state(self) -> dict[str, Any] | None:
        """Return the native Console in-session state for screen restoration."""
        store = self._console_chat_store
        if store is None or not store.sessions():
            return None

        self._stash_console_pending_attachments(store)
        visible_session_id = self._console_visible_draft_session_id
        composer = self._console_composer_or_none()
        if composer is not None and visible_session_id is not None:
            try:
                store.set_session_draft(visible_session_id, composer.draft_text())
            except KeyError:
                pass

        image_state, _cache = self._ensure_console_image_view()
        live_ids = {
            message.id
            for session in store.sessions()
            for message in store.messages_for_session(session.id)
        }
        image_state.prune(live_ids)

        return {
            "version": NATIVE_CONSOLE_STATE_VERSION,
            "active_session_id": store.active_session_id,
            "sessions": [
                {
                    "id": session.id,
                    "title": session.title,
                    "workspace_id": session.workspace_id,
                    "persisted_conversation_id": session.persisted_conversation_id,
                    "draft": session.draft,
                    "settings": self._serialize_console_settings(session.settings),
                    "updated_at": session.updated_at,
                }
                for session in store.sessions()
            ],
            "messages_by_session": {
                session.id: [
                    self._serialize_console_message(message)
                    for message in store.messages_for_session(session.id)
                ]
                for session in store.sessions()
            },
            "image_view_modes": image_state.serialize(),
        }

    def _restore_native_console_state(self, payload: Any) -> None:
        """Restore native Console in-session state saved by ``save_state``."""
        if not isinstance(payload, dict):
            return
        raw_sessions = payload.get("sessions")
        if not isinstance(raw_sessions, list) or not raw_sessions:
            return

        store = self._ensure_console_chat_store()
        raw_messages_by_session = payload.get("messages_by_session")
        messages_by_session = (
            raw_messages_by_session
            if isinstance(raw_messages_by_session, dict)
            else {}
        )
        restored_sessions: list[ConsoleChatSession] = []
        restored_messages_by_session: dict[str, list[ConsoleChatMessage]] = {}
        for raw_session in raw_sessions:
            if not isinstance(raw_session, dict):
                continue
            session_id = str(raw_session.get("id") or uuid.uuid4())
            session_kwargs: dict[str, Any] = dict(
                id=session_id,
                title=str(raw_session.get("title") or DEFAULT_CONSOLE_SESSION_TITLE),
                workspace_id=str(
                    raw_session.get("workspace_id")
                    or store.workspace_context.active_workspace_id
                    or CONSOLE_GLOBAL_WORKSPACE_ID
                ),
                persisted_conversation_id=(
                    str(raw_session["persisted_conversation_id"])
                    if raw_session.get("persisted_conversation_id") is not None
                    else None
                ),
                settings=self._restore_console_settings(raw_session.get("settings")),
                draft=str(raw_session.get("draft") or ""),
            )
            # Legacy payloads saved before `updated_at` was serialized omit the
            # key entirely; keep the ConsoleChatSession factory default (now)
            # for those instead of forcing an empty/invalid timestamp.
            raw_updated_at = raw_session.get("updated_at")
            if raw_updated_at:
                session_kwargs["updated_at"] = str(raw_updated_at)
            session = ConsoleChatSession(**session_kwargs)
            restored_sessions.append(session)
            restored_messages_by_session[session.id] = []
            raw_messages = messages_by_session.get(session.id, [])
            if not isinstance(raw_messages, list):
                continue
            for raw_message in raw_messages:
                message = self._restore_console_message(raw_message)
                if message is None:
                    continue
                self._rehydrate_console_message_image(message)
                restored_messages_by_session[session.id].append(message)

        # One batched `get_attachments_for_messages` call covers every
        # restored message across every session in this pass, instead of a
        # per-message round trip.
        self._rehydrate_console_message_attachments(
            [
                message
                for messages in restored_messages_by_session.values()
                for message in messages
            ]
        )

        active_session_id = payload.get("active_session_id")
        active_session_id = str(active_session_id) if active_session_id is not None else ""
        store.restore_state(
            sessions=restored_sessions,
            messages_by_session=restored_messages_by_session,
            active_session_id=active_session_id,
        )
        self._adopt_console_pending_attachments(store)
        self._console_visible_draft_session_id = None
        self._last_native_transcript_refresh_key = None

        image_state, cache = self._ensure_console_image_view()
        image_state.restore(payload.get("image_view_modes"))
        cache.clear()

    def _rehydrate_console_message_image(self, message: ConsoleChatMessage) -> None:
        """Refill image bytes dropped by screen-state restore (metadata-only).

        Screen-state restore only carries image metadata (mime type + label),
        never raw bytes, so a restored message that still points at an image
        has no bytes for the provider payload builder to attach even though
        its chip renders from metadata alone. Refetch the bytes from the
        ChaChaNotes DB using the message's persisted id; on any failure leave
        the message metadata-only so the chip still renders (graceful
        degradation) instead of raising.
        """
        if message.image_data is not None:
            return
        if not message.image_mime_type or not message.persisted_message_id:
            return
        db = getattr(self.app_instance, "chachanotes_db", None)
        try:
            row = (
                db.get_message_by_id(message.persisted_message_id)
                if db is not None
                else None
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Console restore image rehydration DB lookup failed."
            )
            return
        if not row:
            return
        image_data = row.get("image_data")
        if image_data is None:
            return
        message.image_data = image_data
        message.image_mime_type = row.get("image_mime_type") or message.image_mime_type

    def _rehydrate_console_message_attachments(
        self, messages: list[ConsoleChatMessage]
    ) -> None:
        """Batch-refill ``message_attachments`` table rows for restored messages.

        ``_rehydrate_console_message_image`` (still called per message, see
        its own docstring/tests) already refilled the legacy position-0
        bytes into each message's scalar mirror; this pass runs ONE batched
        ``get_attachments_for_messages`` call covering every message in this
        restore, then folds the now-current scalar mirror plus any table
        rows (positions >= 1) back into each message's attachments tuple.
        Any failure (missing DB, unreachable batch call) leaves messages
        metadata-only -- graceful degradation, matching
        ``_rehydrate_console_message_image``'s own contract.
        """
        ids = [m.persisted_message_id for m in messages if m.persisted_message_id]
        rows_by_id: Dict[str, list[dict[str, Any]]] = {}
        if ids:
            db = getattr(self.app_instance, "chachanotes_db", None)
            getter = getattr(db, "get_attachments_for_messages", None)
            if callable(getter):
                try:
                    fetched = getter(ids)
                except Exception:
                    logger.opt(exception=True).warning(
                        "Console restore attachment batch fetch failed."
                    )
                    fetched = None
                if isinstance(fetched, dict):
                    rows_by_id = fetched

        for message in messages:
            if not message.attachments:
                continue
            entries = list(message.attachments)
            # Position 0 mirrors whatever `_rehydrate_console_message_image`
            # just refilled into the scalar fields (bytes included, when it
            # found a row).
            entries[0] = replace(
                entries[0],
                data=message.image_data,
                mime_type=message.image_mime_type or entries[0].mime_type,
            )
            extra_rows = (
                rows_by_id.get(message.persisted_message_id, [])
                if message.persisted_message_id
                else []
            )
            rows_by_position = {
                int(row.get("position", 0)): row for row in extra_rows
            }
            for index in range(1, len(entries)):
                row = rows_by_position.get(index)
                if row is None:
                    continue
                entries[index] = replace(
                    entries[index],
                    data=row.get("data"),
                    mime_type=row.get("mime_type") or entries[index].mime_type,
                    display_name=row.get("display_name") or entries[index].display_name,
                )
            _apply_console_message_attachments(message, entries)

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

            native_console_state = self._serialize_native_console_state()
            if native_console_state is not None:
                state["native_console_state"] = native_console_state
            
            # Convert to dict for storage
            state['chat_state'] = self.chat_state.to_dict()
            state['state_version'] = '1.0'
            if native_console_state is not None:
                state["interface_type"] = "native_console"
            else:
                state['interface_type'] = 'tabbed' if self.chat_state.tabs else 'single'
            
            logger.info(f"Saved chat state: {len(self.chat_state.tabs)} tabs, interface: {state.get('interface_type')}")
            
        except Exception as e:
            logger.opt(exception=True).error(f"Error saving chat state: {e}")
        
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore comprehensive chat state.
        
        Recreates all tabs, messages, and UI state from saved data.
        """
        logger.debug("Restoring chat screen state")
        super().restore_state(state)
        
        try:
            if "native_console_state" in state:
                self._restore_native_console_state(state["native_console_state"])

            if 'chat_state' in state:
                # Restore from saved state
                self.chat_state = ChatScreenState.from_dict(state['chat_state'])
                
                logger.debug(f"Restored state has {len(self.chat_state.tabs)} tabs")
                logger.debug(f"Active tab ID: {self.chat_state.active_tab_id}")
                logger.debug(f"Tab order: {self.chat_state.tab_order}")
                
                if self.chat_state.validate():
                    logger.info(f"Restoring {len(self.chat_state.tabs)} tabs")

                    # Native Console does not mount legacy ChatTabContainer widgets.
                    if self.chat_state.tabs:
                        self.set_timer(0.1, self._perform_state_restoration)
                else:
                    logger.warning("Chat state validation failed, starting fresh")
                    self.chat_state = ChatScreenState()
            
        except Exception as e:
            logger.opt(exception=True).error(f"Error restoring chat state: {e}")
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
            logger.opt(exception=True).error(f"Error during state restoration: {e}")
    
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
        character_identity = _character_session_identity_from_handoff(payload)
        character_id = None
        character_name = None
        assistant_kind = None
        assistant_id = None
        discovery_owner = payload.discovery_owner
        if character_identity is not None:
            character_id, character_name, assistant_id = character_identity
            assistant_kind = "character"
            if discovery_owner == "general_chat" and payload.source == "personas":
                discovery_owner = "ccp_character"
        return ChatSessionData(
            tab_id=uuid.uuid4().hex[:8],
            title=f"{title_item_type}: {payload.title}",
            conversation_id=None,
            is_ephemeral=True,
            runtime_backend=payload.runtime_backend,
            discovery_owner=discovery_owner,
            discovery_entity_id=payload.discovery_entity_id or payload.source_id,
            character_id=character_id,
            character_name=character_name,
            assistant_kind=assistant_kind,
            assistant_id=assistant_id,
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
                # The native Console composes no legacy tab surface; stage the
                # handoff into the Console live-work lane so the context lands
                # in Staged Context instead of being dropped with a warning.
                self._stage_handoff_as_console_live_work(payload)
                self.app_instance.pending_chat_handoff = None
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

    def _stage_handoff_as_console_live_work(self, payload: ChatHandoffPayload) -> None:
        """Stage a Use-in-Console handoff into the native staged-context lane."""
        from pydantic import ValidationError

        from tldw_chatbook.Chat.citation_evidence_models import (
            EvidenceBundle,
            EvidenceReference,
        )
        from tldw_chatbook.Utils.input_validation import sanitize_string

        def _safe_text(value: Any, max_length: int = 500) -> str:
            return sanitize_string(str(value or ""), max_length=max_length).strip()

        # Handoff bodies can reach 80k characters; cap and sanitize at this
        # boundary before any of it lands in the staged payload.
        snippet = _safe_text(
            payload.display_summary or payload.body, max_length=4_000
        )
        title = _safe_text(payload.title) or "Untitled"
        launch_payload: dict[str, Any] = {
            "target_id": _safe_text(
                payload.content_ref or payload.source_id or title
            ),
            "item_type": _safe_text(payload.item_type),
            "source_id": _safe_text(payload.source_id),
            "snippet": snippet,
            "suggested_prompt": _safe_text(payload.suggested_prompt, max_length=4_000),
            "runtime_backend": _safe_text(payload.runtime_backend),
            "source_selector_state": _safe_text(payload.source_selector_state),
            "metadata": dict(payload.metadata or {}),
        }
        if "rag" in (payload.source or "").lower():
            # RAG-class sources gate Console sends on available evidence;
            # always carry a single-reference bundle (title stands in when
            # the snippet is empty) so the handoff cannot dead-end the send.
            try:
                launch_payload["evidence_bundle"] = EvidenceBundle(
                    bundle_id=_safe_text(payload.content_ref or payload.source_id)
                    or "handoff-evidence",
                    query=_safe_text(payload.suggested_prompt) or title,
                    source=_safe_text(payload.source) or "Search/RAG",
                    references=(
                        EvidenceReference(
                            evidence_id="S1",
                            source_id=_safe_text(payload.source_id) or "unknown",
                            source_type=_safe_text(payload.item_type) or "rag-result",
                            title=title,
                            snippet=snippet or title,
                            authority_label=_safe_text(payload.runtime_backend) or "local",
                            content_ref=payload.content_ref,
                        ),
                    ),
                ).to_payload()
            except (TypeError, ValueError, ValidationError):
                logger.opt(exception=True).warning("Could not build evidence bundle for handoff")

        self._pending_console_launch_context = ConsoleLiveWorkLaunch.from_values(
            source=payload.source,
            title=payload.title,
            payload=launch_payload,
            status=payload.status or "staged",
        )
        self._pending_console_launch_auto_open_inspector = True

        suggested_prompt = launch_payload["suggested_prompt"]
        if suggested_prompt:
            store = self._ensure_console_chat_store()
            session = store.ensure_session(
                title=self._console_initial_session_title_for_workspace(
                    store.workspace_context.active_workspace_id
                ),
                workspace_id=store.workspace_context.active_workspace_id,
                settings=self._default_console_session_settings(),
            )
            if not store.session_draft(session.id).strip():
                store.set_session_draft(session.id, suggested_prompt)
            try:
                composer = self.query_one("#console-native-composer", ConsoleComposerBar)
            except QueryError:
                pass
            else:
                if not composer.draft_text().strip():
                    composer.load_draft(suggested_prompt)

        self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")

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

    def _native_console_transcript_fingerprint(self, messages: list[Any]) -> tuple[Any, ...]:
        """Return a lightweight signature for native transcript refresh skipping."""
        store = self._ensure_console_chat_store()
        message_signatures = []
        for message in messages:
            variants = getattr(message, "variants", None)
            variant_signature = None
            if variants is not None:
                variant_signature = (
                    getattr(variants, "selected_index", None),
                    tuple(
                        (
                            getattr(variant, "id", None),
                            getattr(variant, "content", ""),
                        )
                        for variant in (getattr(variants, "variants", None) or ())
                    ),
                )
            message_signatures.append(
                (
                    getattr(message, "id", None),
                    getattr(getattr(message, "role", None), "value", getattr(message, "role", None)),
                    getattr(message, "content", ""),
                    getattr(message, "status", None),
                    getattr(message, "turn_id", None),
                    getattr(message, "persisted_message_id", None),
                    variant_signature,
                )
            )
        return (store.active_session_id, tuple(message_signatures))

    async def _sync_native_console_transcript_to_legacy_surface(self) -> None:
        """Temporary bridge: render native Console messages in the existing surface."""
        try:
            transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        except QueryError:
            transcript = None

        messages = self._native_console_messages()
        if transcript is not None:
            transcript.set_messages(messages)
            image_specs = self._build_console_image_specs(messages)
            transcript.set_image_specs(image_specs)
            _state, cache = self._ensure_console_image_view()
            # Same bounded subset as `_build_console_image_specs` — computing
            # pending work over the full transcript would prep messages the
            # LRU cache immediately evicts again (churn guard).
            # Exclude ids a prep worker is already chewing on: the 0.2s sync
            # tick would otherwise re-kick the exclusive `console-image-prep`
            # worker for the SAME pending ids on every tick, cancelling the
            # in-flight run and piling duplicate decodes into the executor.
            pending_images = [
                (mid, data)
                for mid, data in cache.pending_ids(self._recent_console_image_messages(messages))
                if mid not in self._console_image_preparing
            ]
            if pending_images:
                self._console_image_preparing.update(mid for mid, _ in pending_images)
                self.run_worker(
                    self._prep_console_images(pending_images),
                    exclusive=True,
                    group="console-image-prep",
                )
            # Image readiness resolves asynchronously (prep worker) after the
            # message-signature fingerprint below has already stabilized, so
            # fold the built specs (id + mode) into the gate too - otherwise
            # a sync that only differs by "the image finished decoding" (or
            # a view-mode toggle) would be skipped as a no-op refresh.
            image_signature = tuple(
                (message_id, image_specs[message_id].mode)
                for message_id in sorted(image_specs)
            )
            refresh_key = (
                id(transcript),
                self._native_console_transcript_fingerprint(messages),
                image_signature,
            )
            if refresh_key != self._last_native_transcript_refresh_key:
                await transcript.refresh_messages()
                self._last_native_transcript_refresh_key = refresh_key
            self._sync_console_transcript_guidance()
            return

        chat_log = self._get_active_chat_log()
        if chat_log is None:
            return

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

    def _clear_native_console_message_selection(self) -> None:
        """Dismiss contextual message actions when an action changes the transcript flow."""
        try:
            transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        except QueryError:
            return
        self._pending_console_delete_message_id = None
        transcript.selected_message_id = None
        self._last_native_transcript_refresh_key = None
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
            self._console_sync_requested = True
            return
        self._console_sync_in_progress = True
        self._record_ui_worker_started("console-sync")
        try:
            self._sync_console_chat_core_state()
            self._sync_console_session_draft()
            # Fix-wave (Critical, Task 4 review): this is the trigger for the
            # "what's in play" chat-dictionary summary now -- it replaces the
            # removed app-level `watch_current_chat_conversation_id`/
            # `watch_current_chat_active_character_data` watchers, which
            # hooked reactives the native Console never writes. This runs on
            # every native session switch/resume (`_activate_native_console_
            # session`, `_resume_console_workspace_conversation`) because
            # both call `_sync_native_console_chat_ui()`; placed before
            # `_sync_console_control_bar()` below so that call's inspector
            # build already sees the freshly recomputed cache instead of one
            # stale frame behind.
            await self._refresh_active_dictionaries_summary_if_scope_changed()
            self._sync_console_control_bar()
            self._sync_console_settings_summary()
            self._sync_console_mode_bar()
            await self._sync_console_native_session_tabs()
            self._sync_console_workspace_context()
            await self._sync_native_console_transcript_to_legacy_surface()
            self._sync_console_rail_visibility_if_changed(
                self._current_console_rail_state()
            )
            self._dispatch_console_rail_preference_prune()
        finally:
            self._record_ui_worker_finished("console-sync")
            self._console_sync_in_progress = False
            if self._console_sync_requested:
                self._console_sync_requested = False
                self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")

    async def _sync_console_native_session_tabs(self) -> None:
        """Refresh native Console session tabs from store state."""
        try:
            surface = self.query_one("#console-session-surface", ConsoleSessionSurface)
        except QueryError:
            return
        store = self._ensure_console_chat_store()
        store.ensure_session(
            title=self._console_initial_session_title_for_workspace(
                store.workspace_context.active_workspace_id
            ),
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
            title=self._console_initial_session_title_for_workspace(
                store.workspace_context.active_workspace_id
            ),
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
            if (controller is None
                    or controller.run_state.status not in CONSOLE_ACTIVE_RUN_STATUSES):
                self._stop_console_transcript_sync_timer()

        self._console_transcript_sync_timer = self.set_interval(0.2, _poll_transcript)
        self._record_ui_timer_created("console-transcript-sync")

    def _stop_console_transcript_sync_timer(self) -> None:
        if self._console_transcript_sync_timer is None:
            return
        try:
            self._console_transcript_sync_timer.stop()
        finally:
            self._record_ui_timer_stopped("console-transcript-sync")
            self._console_transcript_sync_timer = None

    async def _submit_console_native_draft(self, draft: str) -> None:
        controller = self._ensure_console_chat_controller()
        self._start_console_transcript_sync_timer()
        result = await controller.submit_draft(draft)
        if not result.accepted:
            # A resolved skill run (`_run_resolved_console_skill`) stages its
            # TOOL marker name BEFORE this worker even runs. `submit_draft`
            # can still refuse/block the submit for reasons only known once
            # it actually executes (provider not ready, policy block, an
            # active-run race) -- entirely separate from the composer-level
            # gate `_dispatch_console_draft_send` already passed to get here.
            # None of those paths ever reach the accepted-hook
            # (`_on_console_submission_accepted`) that would otherwise
            # consume and clear the staged name, so it must be cleared here
            # too -- otherwise it would silently leak onto whatever the
            # NEXT, unrelated accepted send turns out to be.
            self._console_pending_skill_marker_name = None
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            composer = None
        if result.should_clear_draft and composer is not None:
            composer.clear_draft()
        if result.accepted and controller.run_state.status is ConsoleRunStatus.COMPLETED:
            # Retry/continue/regenerate paths intentionally don't record the flag here —
            # they require an existing message, so ``has_messages`` already keeps the
            # card hidden and the flag was set by the originating submit.
            # Failed/stopped first sends must NOT set the one-time flag: the
            # setup card should return until a send completes with content.
            self._record_console_first_send()
        await self._sync_native_console_chat_ui()

    def _on_console_submission_accepted(self) -> None:
        """Clear the composer as soon as a submit is accepted, not at run end.

        Keeping the sent text in the composer for the whole run reads as
        "not sent" during long local-model generations; blocked submits never
        reach this hook, so their draft is preserved for correction.

        Also the sole consume point for a staged resolved-skill "driving this
        turn" TOOL marker (Task 9 fix-wave): ``ConsoleChatController.
        submit_draft`` invokes this hook synchronously after the USER
        message is appended and after its own skill-substitution/trust
        re-check settles, but before the ASSISTANT placeholder, so
        appending the marker here -- rather than back at the dispatch call
        site -- guarantees store order ``[USER, TOOL, ASSISTANT]`` instead of
        racing the ``run_worker``-scheduled submit.

        Qodo finding 3 (PR #636 bot review): this hook must NEVER fire for a
        submit that ``submit_draft`` ultimately blocks -- including a skill
        substitution refusal (an edited/untrusted skill re-checked at
        build time). ``submit_draft`` used to call this hook right after the
        USER append, before that trust re-check ran, so a refused skill
        submit still consumed the staged marker and appended it right before
        the refuse row -- a marker claiming the skill drove the turn even
        though it never ran. The controller now only calls this hook once
        the substitution check has confirmed the turn actually proceeds, so
        a refusal (like any other blocked submit) never reaches it.
        """
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            composer = None
        if composer is not None:
            composer.clear_draft()
        pending_skill_name = self._console_pending_skill_marker_name
        if pending_skill_name is not None:
            self._console_pending_skill_marker_name = None
            self._append_console_skill_run_marker(pending_skill_name)

    def _console_pending_image_attachment(self):
        """Return a staged image attachment, if any staged item qualifies.

        Scans the whole staged list (not just the first item) so a
        multi-attachment session still gates vision-capability/blocked-send
        checks correctly when the qualifying image isn't staged first.
        """
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return None
        try:
            pendings = store.pending_attachments(store.active_session_id)
        except KeyError:
            return None
        for pending in pendings:
            if (
                pending is not None
                and pending.insert_mode == "attachment"
                and pending.file_type == "image"
                and pending.data is not None
            ):
                return pending
        return None

    def _console_attachment_blocked_reason(self) -> str:
        """Return blocked-send copy when a staged image can't reach the model."""
        from tldw_chatbook.Chat.attachment_core import vision_block_reason

        if self._console_pending_image_attachment() is None:
            return ""
        effective_settings, _readiness = self._active_console_settings_readiness()
        return (
            vision_block_reason(effective_settings.provider, effective_settings.model)
            or ""
        )

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
        attachment_reason = self._console_attachment_blocked_reason()
        if attachment_reason:
            return attachment_reason
        return ""

    async def handle_console_send_message(self, event: Button.Pressed) -> None:
        """Route the Console composer send action through the native controller."""
        event.stop()
        await self._send_console_message_from_visible_action()

    async def _send_console_message_from_visible_action(self) -> None:
        """Route the visible Console send action through the native controller."""
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
            draft = composer.draft_text()
        except QueryError:
            composer = None
            draft = ""
        if not draft.strip() and self._console_pending_image_attachment() is None:
            self._focus_console_composer_if_needed(force=True)
            return
        self._dismiss_console_guidance()

        # Command parsing runs before any readiness/blocked gating: a
        # recognized command dispatch (or an unknown-command hint) never
        # sends, so it must work even while Send is blocked. Draft text
        # carrying any real paste-originated segment (regardless of its
        # current collapse/confirm/expanded display state) is never treated
        # as command input -- Task 9's grammar module deliberately leaves
        # that gating to the caller, since only the composer knows the real
        # segment state.
        if composer is not None and not composer.has_paste_segments():
            parse = self._console_command_registry.parse(draft)
        else:
            parse = CommandParse(kind=KIND_NOT_COMMAND)

        if parse.kind in (KIND_COMMAND, KIND_FALLBACK):
            self._console_unknown_send_armed = None
            await self._dispatch_console_command(parse)
            return

        if parse.kind == KIND_UNKNOWN:
            # Fold-in (Task 9 fix-wave review): the fallback resolver only
            # ever claims a word against the trusted-only cached
            # `_console_skill_candidates` snapshot -- a typed `/name` that
            # matches ONLY needs-review (trust-blocked) skills therefore
            # always reaches here as KIND_UNKNOWN, previously falling
            # through to the generic "Unknown command" hint just like any
            # other unrecognized word. Re-checking against a FRESH context
            # (never that cached snapshot) surfaces the same
            # `/skills <name>` needs-review response instead, before the
            # unknown-command arm/hint logic ever runs. This never arms the
            # unknown-command escape: a blocked match is a known-but-blocked
            # command, not an unrecognized one, so a repeated Enter shows
            # the same response again rather than silently falling through
            # to a literal send.
            context = await self._fetch_console_skill_context()
            blocked_summaries = self._console_skill_blocked_summaries(context)
            if await self._console_skill_blocked_match_response(parse.name, blocked_summaries):
                return
            if self._console_unknown_send_armed == draft:
                # Second consecutive Enter on the *same* unmodified draft:
                # disarm and fall through to a normal send below.
                self._console_unknown_send_armed = None
            else:
                self._console_unknown_send_armed = draft
                await self._append_native_console_system_message(
                    self._console_unknown_command_hint(parse.name)
                )
                return

        await self._dispatch_console_draft_send(draft)

    async def _dispatch_console_draft_send(self, draft: str) -> bool:
        """Run the send-blocked/readiness gate, then queue ``draft`` as the
        user turn (the normal-text-send tail, shared with Task 9's resolved
        `/skill-name` run path so both go through the exact same gating).

        Returns:
            ``True`` once the submit has actually been queued via
            ``run_worker`` (a caller-visible "this is really going out"
            signal -- Task 9's skill-run path only appends its "driving this
            turn" marker when this is ``True``); ``False`` when blocked or
            when a run is already in progress, in which case nothing is
            queued and an explanatory row/toast was already shown.
        """
        if blocked_reason := self._console_send_blocked_reason():
            setup_blocked_reason = self._console_setup_blocked_reason()
            if setup_blocked_reason and not blocked_reason.startswith(
                "Console send blocked: Library Search/RAG"
            ):
                await self._append_native_console_system_message(setup_blocked_reason)
                self.app_instance.notify(
                    setup_blocked_reason,
                    severity="warning",
                )
                self._focus_console_composer_if_needed(force=True)
                return False
            await self._append_native_console_system_message(blocked_reason)
            self._focus_console_composer_if_needed(force=True)
            return False
        controller = self._ensure_console_chat_controller()
        if not controller.run_state.is_send_allowed:
            self.app_instance.notify(CONSOLE_RUN_ALREADY_RUNNING_COPY, severity="warning")
            return False
        # group="console-run": a dedicated group so UI-sync kicks can never
        # cancel an in-flight run (TASK-228 — ungrouped exclusive workers all
        # share Textual's default group and cancel each other).
        self.run_worker(
            self._submit_console_native_draft(draft),
            exclusive=True,
            group="console-run",
        )
        return True

    _CONSOLE_COMMAND_NAME_TO_HANDLER_ID = {
        PROMPT_COMMAND_NAME: PROMPT_COMMAND_HANDLER_ID,
        SYSTEM_COMMAND_NAME: SYSTEM_COMMAND_HANDLER_ID,
        SKILLS_COMMAND_NAME: SKILLS_COMMAND_HANDLER_ID,
    }

    def _console_unknown_command_hint(self, name: str) -> str:
        """Return the Enter-again hint copy for an unrecognized `/name` draft.

        Derived from the registry's own ``available_names()`` (Task 9) rather
        than a hardcoded list, so a newly-registered command (e.g. `/skills`)
        is reflected here automatically.
        """
        available = ", ".join(f"/{name}" for name in self._console_command_registry.available_names())
        return f"Unknown command /{name} — available: {available}. Press Enter again to send as text."

    async def _dispatch_console_command(self, parse: CommandParse) -> None:
        """Dispatch a parsed Console slash command to its handler.

        A ``handler_id`` that resolves to nothing (an unrecognized command
        name) is consumed silently: nothing is sent and the draft is left
        untouched. ``KIND_FALLBACK`` (Task 9's bare `/skill-name` Console
        surface) always routes to the skill-run handler regardless of any
        handler-id lookup, since fallback-resolved parses never carry a
        registered command name.
        """
        if parse.kind == KIND_FALLBACK:
            await self._console_command_run_skill(parse.name, parse.args)
            return
        handler_id = self._CONSOLE_COMMAND_NAME_TO_HANDLER_ID.get(parse.name)
        dispatch_map = {
            "insert-prompt": self._console_command_insert_prompt,
            "apply-system": self._console_command_apply_system,
            SKILLS_COMMAND_HANDLER_ID: self._console_command_skills,
        }
        handler = dispatch_map.get(handler_id)
        if handler is None:
            return
        await handler(parse)

    # Bounded prompt-search page size for `/prompt` resolution and the
    # picker's own search callable -- mirrors Task 11's picker contract
    # (PromptScopeService.search_prompts, FTS-ranked, <= 25 rows).
    _CONSOLE_PROMPT_SEARCH_LIMIT = 25

    _LIBRARY_PROMPT_INSERT_BLOCKED_COPY = "Finish provider setup to insert prompts."

    async def _console_command_insert_prompt(self, parse: CommandParse) -> None:
        """Resolve and insert a saved prompt's ``user_prompt`` for `/prompt`.

        Resolution order (brief): exact case-insensitive name match over a
        bounded search page; else a unique case-insensitive name-prefix
        match over that same page; else (no args, 0 matches, or an
        ambiguous 2+ match at either stage) open the picker prefilled with
        the typed args. A resolved match REPLACES the composer draft
        wholesale (the draft IS the `/prompt ...` command being replaced by
        its result) via paste semantics, so an oversized body still
        collapses to a token exactly like a real paste would.
        """
        query = parse.args.strip()
        resolved = await self._resolve_console_prompt_by_name(query) if query else None
        if resolved is not None:
            self._insert_prompt_text_into_composer(
                str(resolved.get("user_prompt") or ""), replace=True
            )
            return
        await self._open_console_prompt_picker_for_insert(query)

    @staticmethod
    def _console_prompt_prefix_fts_query(text: str) -> str:
        """Build an FTS5 phrase-prefix MATCH expression for ``text``.

        Plain FTS5 MATCH requires a full token, which would defeat both the
        `/prompt` prefix-match resolution stage (a query like "Summ" would
        never match a stored name "Summarize") and a picker that is supposed
        to filter results as the user is still mid-word. Quoting the whole
        query as a phrase with a trailing ``*`` makes FTS5 match names whose
        tokens *start with* the typed text instead -- a prefix match trivially
        covers an exact match too, so one query shape serves both. Embedded
        quotes are doubled per FTS5 string-literal escaping (mirrors
        ``library_fts_query._quote_fts_term``), so user text can never break
        out of the quoted phrase to inject MATCH operators.
        """
        escaped = text.replace('"', '""')
        return f'"{escaped}"*'

    async def _console_prompt_search(self, query: str) -> list:
        """Bounded FTS prompt search bound to the active scope service.

        Shared by `/prompt` resolution and the picker's ``prompt_search``
        callable so both always read a fresh page rather than any cached
        boot-time snapshot.
        """
        service = getattr(self.app_instance, "prompt_scope_service", None)
        search_prompts = getattr(service, "search_prompts", None)
        if not callable(search_prompts):
            return []
        stripped_query = query.strip()
        fts_kwargs = (
            {"fts_match_query": self._console_prompt_prefix_fts_query(stripped_query)}
            if stripped_query
            else {}
        )
        try:
            return await search_prompts(
                mode="local",
                query=query,
                limit=self._CONSOLE_PROMPT_SEARCH_LIMIT,
                **fts_kwargs,
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Console prompt search failed for query {query!r}."
            )
            return []

    async def _resolve_console_prompt_by_name(self, query: str) -> Optional[Mapping[str, Any]]:
        """Resolve `/prompt <name>` to a single prompt record, or ``None``.

        ``None`` means the caller should fall back to the picker: no
        candidates, an ambiguous (2+) exact case-insensitive name match, or
        no/ambiguous unique prefix match either.
        """
        candidates = await self._console_prompt_search(query)
        normalized_query = query.strip().casefold()
        exact_matches = [
            record
            for record in candidates
            if str(record.get("name") or "").strip().casefold() == normalized_query
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return None
        prefix_matches = [
            record
            for record in candidates
            if str(record.get("name") or "").strip().casefold().startswith(normalized_query)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return None

    async def _open_console_prompt_picker_for_insert(self, initial_query: str) -> None:
        """Open the prompt picker for `/prompt`, inserting whatever is chosen."""

        def _apply_picker_choice(record: Optional[Mapping[str, Any]]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if record is None:
                return
            self._insert_prompt_text_into_composer(
                str(record.get("user_prompt") or ""), replace=True
            )

        self.app.push_screen(
            ConsolePromptPickerModal(
                mode="insert",
                initial_query=initial_query,
                prompt_search=self._console_prompt_search,
            ),
            callback=_apply_picker_choice,
        )

    def _insert_prompt_text_into_composer(self, text: str, *, replace: bool) -> bool:
        """Insert resolved prompt text into the Console composer via paste semantics.

        Args:
            text: The prompt's ``user_prompt`` body to insert.
            replace: ``True`` replaces the whole draft wholesale (the
                `/prompt` command's own draft IS the command being replaced
                by its result). ``False`` appends onto whatever draft
                already exists (Library's "Use in Console" handoff) -- an
                already-empty draft still gets a clean insert with no
                separator, but existing draft text is never clobbered.

        Returns:
            ``True`` when the composer widget was found and the insert
            applied, ``False`` when no native composer is mounted.
        """
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return False
        if replace:
            composer.clear_draft()
        elif composer.draft_text():
            # Appending onto an existing draft must never mash the two
            # payloads together with no boundary between them.
            composer.insert_text("\n")
        composer.insert_text_as_paste(text)
        return True

    async def _consume_pending_console_prompt_insert(self) -> None:
        """Consume a Library "Use in Console" staged prompt body, if any.

        Mirrors ``_consume_pending_chat_handoff``'s stage-then-consume
        shape, but the staged payload is a bare string appended into the
        composer -- never a ``ChatHandoffPayload``. Gated on the same
        first-run provider/model setup readiness the composer's own Send
        button uses: unlike the in-composer `/prompt` command (which Task
        10 deliberately lets run even while Send is blocked, since
        composing is not sending), this cross-screen hop is an unattended
        action the user did not consciously type into this composer, so a
        blocked first-run state gets an honest toast instead of a silent
        insert -- the draft is left untouched and nothing about the source
        Library prompt is touched either.
        """
        pending = getattr(self.app_instance, "pending_console_prompt_insert", None)
        if pending is None:
            return
        text = pending if isinstance(pending, str) else str(pending)
        if not text.strip():
            self.app_instance.pending_console_prompt_insert = None
            return
        if self._console_setup_blocked_reason():
            # A persistent state, not a mount-timing race -- always safe to
            # consume+notify here regardless of whether the composer widget
            # itself has finished mounting yet.
            self.app_instance.pending_console_prompt_insert = None
            self.app_instance.notify(
                self._LIBRARY_PROMPT_INSERT_BLOCKED_COPY,
                severity="warning",
            )
            return
        # Settle the active-session draft tracking BEFORE inserting so this
        # consumption is self-guarding no matter which lifecycle hook
        # (`on_mount`, `on_screen_resume`, or any other resume-adjacent path)
        # scheduled it. If a session switch races ahead of us,
        # `_console_visible_draft_session_id` can be stale relative to the
        # store's active session; a *later* `_sync_native_console_chat_ui`
        # pass would then unconditionally reload the composer from that
        # newly-active session's stored draft, silently discarding the
        # insert below (the pending field is already cleared once the
        # insert lands, so there is no retry). Calling this here -- and
        # nowhere between here and the insert, so the two run atomically
        # within this event-loop turn -- settles the tracker onto the
        # current active session first, so any subsequent sync pass takes
        # the no-op fast path instead of clobbering what we're about to
        # insert.
        self._sync_console_session_draft()
        # Only clear the staged field once the insert has actually landed --
        # if the native composer has not finished mounting yet (a transient
        # race the 0.15s mount-time delay above should normally avoid), leave
        # it pending for a later mount/resume to retry rather than silently
        # discarding it.
        if self._insert_prompt_text_into_composer(text, replace=False):
            self.app_instance.pending_console_prompt_insert = None
            self._focus_console_composer_if_needed(force=True)

    async def _console_command_apply_system(self, parse: CommandParse) -> None:
        """Resolve and apply a saved prompt's ``system_prompt`` for `/system`.

        Bare `/system` (no args) opens the system prompt editor modal seeded
        with the active session's current system prompt. With args,
        resolution mirrors `/prompt` (Task 12): exact case-insensitive name
        match over a bounded search page, else a unique case-insensitive
        name-prefix match; a resolved match with a blank ``system_prompt``
        shows an inline transcript error (the session is left unchanged,
        and the draft is deliberately left in place so the user can correct
        it) rather than silently clearing it, since that is very likely not
        what the user meant by naming that specific prompt. A resolved match
        WITH a system part applies it and clears the `/system <name>`
        command text from the composer -- mirrors `/prompt`'s successful
        insert always replacing its own draft (Task 12) -- so a handled
        command never leaves its own invocation text behind. 0 or 2+
        matches at either stage fall back to the apply-system picker mode
        (Task 11), prefilled with the typed args.
        """
        args = parse.args.strip()
        if not args:
            await self._open_console_system_prompt_editor()
            return
        resolved = await self._resolve_console_prompt_by_name(args)
        if resolved is not None:
            # Blank check only via strip(); the applied value below is the
            # raw prompt text so leading/trailing whitespace and internal
            # formatting survive verbatim.
            raw_system_prompt = resolved.get("system_prompt")
            system_prompt = raw_system_prompt if isinstance(raw_system_prompt, str) else ""
            if not system_prompt.strip():
                name = str(resolved.get("name") or args)
                await self._append_native_console_system_message(
                    CONSOLE_SYSTEM_PROMPT_NO_SYSTEM_PART_TEMPLATE.format(name=name)
                )
                return
            self._apply_console_session_system_prompt(system_prompt)
            self._clear_console_composer_draft()
            return
        await self._open_console_prompt_picker_for_apply_system(args)

    def _clear_console_composer_draft(self) -> None:
        """Clear the native Console composer's draft text, if mounted.

        Shared by any handled-command success path that applies a side
        effect (rather than inserting replacement text) but must still not
        leave its own invocation text sitting in the composer afterward --
        e.g. a successful named `/system <name>` apply. `/prompt`'s
        equivalent success path instead replaces the draft with the
        resolved prompt body via ``_insert_prompt_text_into_composer``,
        which already clears via the same ``clear_draft()`` seam.
        """
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.clear_draft()

    async def _open_console_prompt_picker_for_apply_system(self, initial_query: str) -> None:
        """Open the prompt picker in apply-system mode for `/system`.

        Rows without a ``system_prompt`` render dimmed and refuse selection
        (``ConsolePromptPickerModal``'s own ``MODE_APPLY_SYSTEM`` behavior,
        Task 11) -- this caller only needs to apply whatever record the
        picker actually dismisses with.
        """

        def _apply_picker_choice(record: Optional[Mapping[str, Any]]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if record is None:
                return
            # Blank check only via strip(); the applied value is the raw
            # prompt text so formatting survives verbatim.
            raw_system_prompt = record.get("system_prompt")
            system_prompt = raw_system_prompt if isinstance(raw_system_prompt, str) else ""
            if not system_prompt.strip():
                return
            self._apply_console_session_system_prompt(system_prompt)

        self.app.push_screen(
            ConsolePromptPickerModal(
                mode=CONSOLE_PROMPT_PICKER_MODE_APPLY_SYSTEM,
                initial_query=initial_query,
                prompt_search=self._console_prompt_search,
            ),
            callback=_apply_picker_choice,
        )

    def _apply_console_session_system_prompt(self, system_prompt: Optional[str]) -> None:
        """Apply (or, for a blank/``None`` value, clear) the active session's
        system prompt, persisting the change if the conversation is already
        saved (Task 13's ``ConsoleChatStore.set_session_system_prompt``), and
        refresh the rail preview + context-estimate surfaces in place.

        The in-memory session is always updated even when the durable write
        fails -- ``set_session_system_prompt`` never rolls that back (see
        its docstring) -- so a persistence failure only means the change may
        not survive a reload; it is surfaced here as an honest warning
        rather than silently swallowed or crashing this callback.
        """
        self._ensure_active_console_session_settings()
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            return
        _session, persisted = store.set_session_system_prompt(session_id, system_prompt)
        if not persisted:
            self.app_instance.notify(
                "System prompt applied for this session, but the change "
                "could not be saved -- it may not survive a reload.",
                severity="warning",
            )
        self._sync_console_chat_core_state()
        self._sync_console_settings_summary()

    async def _open_console_system_prompt_editor(self) -> None:
        """Open the system prompt editor modal for the active Console session."""
        settings = self._ensure_active_console_session_settings()

        def _apply_modal_result(result: Optional[str]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if result is None:
                return
            self._apply_console_session_system_prompt(result)

        self.app.push_screen(
            ConsoleSystemPromptModal(
                system_prompt=settings.system_prompt,
                save_to_library=self._save_console_system_prompt_to_library,
            ),
            callback=_apply_modal_result,
        )

    async def _save_console_system_prompt_to_library(self, name: str, text: str) -> str:
        """Save the system-prompt editor's text as a brand-new Library prompt.

        Always a CREATE (the Console `/system` editor never edits an
        existing Library prompt): pre-checks the name for a collision the
        same way ``library_screen._save_library_prompt``'s own create path
        does, so a genuine duplicate is classified via
        ``classify_prompt_save_error`` -- with ``exc=None`` and a manually
        built message -- rather than racing the DB's raw ``ConflictError``,
        and reports the SAME outcome copy that screen's own save flow shows.

        Args:
            name: Name for the new Library prompt.
            text: The prompt's ``system_prompt`` body (the modal's current
                editor text).

        Returns:
            User-facing outcome copy to display inline in the modal.
        """
        name = name.strip()
        if not name:
            return "Enter a name to save this system prompt to Library."
        text = text.strip()
        if not text:
            return "Enter a system prompt to save."
        service = getattr(self.app_instance, "prompt_scope_service", None)
        get_prompt = getattr(service, "get_prompt", None)
        save_prompt = getattr(service, "save_prompt", None)
        if not callable(get_prompt) or not callable(save_prompt):
            return CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY["error"]
        try:
            candidate = await get_prompt(
                mode="local", prompt_identifier=name, include_deleted=True
            )
        except Exception:
            candidate = None
        if isinstance(candidate, Mapping) and candidate:
            if candidate.get("deleted"):
                outcome = classify_prompt_save_error(
                    None, f"Prompt '{name}' exists but is soft-deleted.", None
                )
            else:
                outcome = classify_prompt_save_error(
                    None, f"Prompt '{name}' already exists.", None
                )
            return CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY.get(
                outcome, CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY["error"]
            )
        try:
            result = await save_prompt(mode="local", name=name, system_prompt=text, user_prompt="")
        except Exception as exc:
            logger.opt(exception=True).warning(
                f"Console system-prompt save-to-library failed for name {name!r}."
            )
            outcome = classify_prompt_save_error(None, str(exc), exc)
            return CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY.get(
                outcome, CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY["error"]
            )
        result_id = result.get("local_id") if isinstance(result, Mapping) else (1 if result else None)
        outcome = classify_prompt_save_error(result_id, "", None)
        return CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY.get(
            outcome, CONSOLE_SYSTEM_PROMPT_SAVE_STATUS_COPY["error"]
        )

    # ------------------------------------------------------------------
    # Task 9 (Skills spec, Phase 2): `/skills` registered command + bare
    # `/skill-name` fallback dispatch (list / run / refuse).
    # ------------------------------------------------------------------

    # Bounded skill-search page size for the picker's `skill_search`
    # callable -- mirrors `_CONSOLE_PROMPT_SEARCH_LIMIT`'s <= 25-row bound.
    _CONSOLE_SKILL_SEARCH_LIMIT = 25

    async def _fetch_console_skill_context(self) -> Mapping[str, Any]:
        """Fetch a FRESH ``skills_scope_service.get_context`` payload.

        Used by every run/list/search path that needs an authoritative (not
        cached) view -- the cached ``_console_skill_candidates`` snapshot is
        reserved for the fallback resolver's word-claiming decision alone.
        Returns ``{}`` (never raises) when the service is unavailable or the
        call fails, so callers can treat "no skills" and "service down" the
        same way: an empty candidate/blocked population.
        """
        service = getattr(self.app_instance, "skills_scope_service", None)
        get_context = getattr(service, "get_context", None)
        if not callable(get_context):
            return {}
        try:
            context = await get_context(mode="local")
        except Exception:
            logger.opt(exception=True).warning("Console skill context fetch failed.")
            return {}
        return context if isinstance(context, Mapping) else {}

    @staticmethod
    def _console_skill_trusted_candidates_from_context(
        context: Mapping[str, Any],
    ) -> tuple[SkillCommandCandidate, ...]:
        """Build the trusted, user-invocable candidate population.

        Scoped to ``user_invocable and not trust_blocked`` (blocked skills
        are excluded here even though ``get_context``'s own
        ``available_skills`` should already exclude them -- defensive, since
        a caller-supplied fake service isn't guaranteed to uphold that).
        Stably sorted by case-folded name (Task 7 review note) so result
        order never depends on the backend's own iteration order.
        """
        available = context.get("available_skills") if isinstance(context, Mapping) else None
        candidates = [
            SkillCommandCandidate(
                name=str(item.get("name")),
                description=str(item.get("description") or ""),
            )
            for item in (available or [])
            if isinstance(item, Mapping)
            and item.get("name")
            and item.get("user_invocable", True)
            and not item.get("trust_blocked", False)
        ]
        candidates.sort(key=lambda candidate: candidate.name.casefold())
        return tuple(candidates)

    @staticmethod
    def _console_skill_blocked_summaries(
        context: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """Return the needs-review (trust-blocked) skill summaries, named only."""
        blocked = context.get("blocked_skills") if isinstance(context, Mapping) else None
        return tuple(item for item in (blocked or []) if isinstance(item, Mapping) and item.get("name"))

    async def _refresh_console_skill_candidates(self) -> None:
        """Refresh the cached trusted-candidate snapshot for the fallback resolver.

        Called on Console mount/resume; the fallback resolver itself always
        reads through ``self._console_skill_candidates`` via a closure, so
        updating this attribute is all a refresh needs to do.
        """
        context = await self._fetch_console_skill_context()
        self._console_skill_candidates = self._console_skill_trusted_candidates_from_context(context)

    async def _console_skill_search(self, query: str) -> list[Mapping[str, object]]:
        """Bounded, fresh-fetched trusted-skill search for the skill picker."""
        context = await self._fetch_console_skill_context()
        candidates = self._console_skill_trusted_candidates_from_context(context)
        normalized_query = query.strip().casefold()
        if normalized_query:
            candidates = tuple(
                candidate
                for candidate in candidates
                if normalized_query in candidate.name.casefold()
                or normalized_query in candidate.description.casefold()
            )
        return [
            {"name": candidate.name, "description": candidate.description}
            for candidate in candidates[: self._CONSOLE_SKILL_SEARCH_LIMIT]
        ]

    @staticmethod
    def _split_console_skill_name_args(text: str) -> tuple[str, str]:
        """Split ``text`` (already stripped) into its leading word and the rest.

        Mirrors ``console_command_grammar._split_leading_token``'s single-
        whitespace-character split rule, used here to pull a skill name back
        out of `/skills <name> [args]`'s own ``args`` text.
        """
        for index, character in enumerate(text):
            if character.isspace():
                return text[:index], text[index + 1 :]
        return text, ""

    async def _console_command_skills(self, parse: CommandParse) -> None:
        """Handle the registered `/skills` command: bare list, or `<name> [args]` run."""
        args = parse.args.strip()
        if not args:
            context = await self._fetch_console_skill_context()
            candidates = self._console_skill_trusted_candidates_from_context(context)
            await self._append_native_console_system_message(format_skills_list(candidates))
            return
        name, rest = self._split_console_skill_name_args(args)
        await self._console_command_run_skill(name, rest)

    async def _console_command_run_skill(self, name: str, args: str) -> None:
        """Resolve and run a skill by name (spec Slash surface run semantics).

        Shared by the `/skills <name> [args]` registered command and the
        bare `/skill-name [args]` fallback -- both converge here. Always
        re-resolves against a FRESH ``get_context`` (never the cached
        fallback-resolver snapshot) for the authoritative trust decision.
        """
        args = cap_skill_args(args)
        context = await self._fetch_console_skill_context()
        trusted_candidates = self._console_skill_trusted_candidates_from_context(context)
        blocked_summaries = self._console_skill_blocked_summaries(context)
        resolution = resolve_skill_command(name, args, trusted_candidates)

        if resolution.kind == "resolved":
            await self._run_resolved_console_skill(resolution.name, args)
            return
        if resolution.kind == "ambiguous":
            await self._open_console_skill_picker(name, args)
            return

        # resolution.kind == "none": distinguish "genuinely absent" from
        # "exists but needs review" before falling back to the picker --
        # review-mandated addition (Task 8 review): a typed name/prefix that
        # matches ONLY needs-review skills must not read like the generic
        # empty state. Shared with the bare `/skill-name` KIND_UNKNOWN
        # fallback dispatch site (fix-wave fold-in below) via
        # `_console_skill_blocked_match_response`.
        if await self._console_skill_blocked_match_response(name, blocked_summaries):
            return
        if args:
            await self._open_console_skill_picker(name, args)
            return
        await self._append_skill_refuse_row(name, CONSOLE_SKILL_RUN_REFUSE_REASON_ABSENT)

    async def _console_skill_blocked_match_response(
        self, name: str, blocked_summaries: tuple[Mapping[str, Any], ...]
    ) -> bool:
        """Surface the needs-review response for ``name`` against blocked skill summaries.

        Shared by `/skills <name>`'s "none" branch above and the bare
        `/skill-name` `KIND_UNKNOWN` fallback dispatch site
        (``_send_console_message_from_visible_action``) -- fold-in from the
        Task 9 fix-wave review -- so a typed name/prefix that matches ONLY
        needs-review (trust-blocked) skills reads the same way from either
        surface, instead of the generic "genuinely absent"/unknown-command
        copy the fallback resolver's own trusted-only cached snapshot would
        otherwise fall through to.

        Args:
            name: The typed command word (no leading slash).
            blocked_summaries: Fresh ``get_context``-derived blocked-skill
                summaries (never the fallback resolver's cached trusted-only
                snapshot).

        Returns:
            ``True`` when a blocked match was found and a response was
            appended (an exact match appends the ``SKILL_UNTRUSTED_REFUSE``
            row; one or more prefix matches append the
            ``CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE`` hint) -- the caller
            should stop further handling. ``False`` when ``name`` matches no
            blocked skill at all, so the caller falls through to its own
            default (the skill picker for `/skills <name>`, the generic
            unknown-command hint for bare fallback).
        """
        name_lower = name.lower()
        exact_blocked = next(
            (item for item in blocked_summaries if str(item.get("name") or "").lower() == name_lower),
            None,
        )
        if exact_blocked is not None:
            reason = str(
                exact_blocked.get("trust_reason_code") or exact_blocked.get("trust_status") or "needs review"
            )
            await self._append_skill_refuse_row(str(exact_blocked.get("name") or name), reason)
            return True
        prefix_blocked = [
            item for item in blocked_summaries if str(item.get("name") or "").lower().startswith(name_lower)
        ]
        if prefix_blocked:
            await self._append_native_console_system_message(
                CONSOLE_SKILL_NEEDS_REVIEW_HINT_TEMPLATE.format(count=len(prefix_blocked))
            )
            return True
        return False

    async def _run_resolved_console_skill(self, name: str, args: str) -> None:
        """Submit a resolved skill's raw `/name [args]` command as the user turn.

        Task 10 (the substitution rule) renders this at payload build time --
        this method only sends the literal, unmodified command text through
        the exact same send-blocked/readiness gate a normal Enter-to-send
        goes through.

        Fix-wave note (reviewer repro): the "driving this turn" TOOL marker
        is NOT appended here anymore. ``_dispatch_console_draft_send``
        returning ``True`` only means ``run_worker`` *scheduled* the actual
        submit -- the USER message it appends has not necessarily landed yet
        by the time this coroutine resumes, so appending the marker
        immediately after could (and, per the repro, reliably did) land it
        BEFORE the user turn it is meant to follow. Instead, ``name`` is
        staged here and consumed by ``_on_console_submission_accepted``,
        which fires synchronously from inside the real submit -- right after
        the USER message is appended and before the ASSISTANT placeholder --
        guaranteeing store order ``[USER, TOOL, ASSISTANT]``. A dispatch that
        never even gets queued (blocked send, run already in progress) must
        not leave a stale marker staged for some later, unrelated send.
        """
        raw_command = f"/{name} {args}" if args else f"/{name}"
        self._console_pending_skill_marker_name = name
        queued = await self._dispatch_console_draft_send(raw_command)
        if not queued:
            self._console_pending_skill_marker_name = None

    def _append_console_skill_run_marker(self, name: str) -> None:
        """Append the TOOL-role "driving this turn" marker for a resolved skill run.

        Mirrors ``ConsoleAgentBridge._append_marker``: a raw (unescaped)
        content string, appended without persisting -- both transcript
        renderers (``console_transcript.py`` and the legacy fallback) render
        TOOL rows with markup off, so escaping here would just leave stray
        backslashes in what the user sees.
        """
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            return
        try:
            store.append_message(
                session_id,
                role=ConsoleMessageRole.TOOL,
                content=CONSOLE_SKILL_RUN_MARKER_TEMPLATE.format(name=name),
            )
        except KeyError:
            pass  # session vanished mid-dispatch; nothing left to annotate

    async def _append_skill_refuse_row(self, name: str, reason: str) -> None:
        """Append the `SKILL_UNTRUSTED_REFUSE` transcript row for a run attempt.

        Covers all three "nothing runs" buckets the spec groups together:
        untrusted (present but trust-blocked), edited (a fallback-cached
        resolution that no longer holds at fresh re-resolution), and absent
        (no matching skill at all). The draft is deliberately left
        untouched by this method -- the composer never gets cleared on a
        refusal, so the user can correct the name in place.
        """
        await self._append_native_console_system_message(
            SKILL_UNTRUSTED_REFUSE.format(name=name, reason=reason)
        )

    async def _open_console_skill_picker(self, initial_query: str, args: str) -> None:
        """Open the skill picker prefilled with ``initial_query`` (ambiguous or 0-match-with-args).

        ``args`` is the text that followed the unresolved word -- preserved
        across the picker detour so a picked skill still runs with whatever
        parameters the user already typed, exactly as a direct resolution
        would have.
        """

        def _apply_picker_choice(record: Optional[Mapping[str, Any]]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if record is None:
                return
            picked_name = str(record.get("name") or "").strip()
            if not picked_name:
                return
            self.run_worker(self._run_resolved_console_skill(picked_name, args), exclusive=False)

        self.app.push_screen(
            ConsoleSkillPickerModal(
                initial_query=initial_query,
                skill_search=self._console_skill_search,
            ),
            callback=_apply_picker_choice,
        )

    @on(Input.Changed, "#console-command-input")
    def _on_console_composer_draft_changed(self, event: Input.Changed) -> None:
        """Disarm the unknown-command Enter-again escape on any draft edit.

        ``ConsoleComposerBar`` keeps a hidden compatibility ``Input`` synced to
        the canonical draft text on every segment mutation (typing, pasting,
        backspace, clear, ``load_draft``); its reactive ``value`` posts this
        `Changed` message whenever that text actually changes. Any such edit
        must invalidate a pending unknown-command arm -- otherwise a user
        could edit away from an armed unknown draft and back to the exact
        same text and have a *second*, unrelated Enter silently send it.
        """
        self._console_unknown_send_armed = None

    async def handle_console_stop_generation(self, event: Button.Pressed) -> None:
        """Route the Console stop action through native run control."""
        event.stop()
        await self._stop_console_generation_from_visible_action()

    async def _stop_console_generation_from_visible_action(self) -> None:
        """Route the visible Console stop action through native run control."""
        controller = self._ensure_console_chat_controller()
        if not controller.stop_active_run():
            self.app_instance.notify("No active Console run to stop.", severity="warning")
        await self._sync_native_console_chat_ui()

    @on(Button.Pressed, "#console-attach-context")
    async def handle_console_attach_context(self, event: Button.Pressed) -> None:
        """Open the native Console file picker and stage the selected attachment."""
        await self._handle_console_attach_context(event)

    @on(Button.Pressed, "#console-staged-context-attach")
    async def handle_console_staged_context_attach(self, event: Button.Pressed) -> None:
        """Open the native Console file picker from the staged-context empty state."""
        await self._handle_console_attach_context(event)

    async def _handle_console_attach_context(self, event: Button.Pressed) -> None:
        """Open the native Console file picker and stage the selected attachment."""
        event.stop()
        from fnmatch import fnmatch

        from tldw_chatbook.Chat.attachment_core import attachment_filter_specs
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

        def create_filter(patterns: str):
            pattern_list = patterns.split(";")

            def filter_func(path: Path) -> bool:
                return any(fnmatch(path.name, pattern) for pattern in pattern_list)

            return filter_func

        file_filters = Filters(
            *[(label, create_filter(patterns)) for label, patterns in attachment_filter_specs()],
            ("All Files", lambda path: True),
        )

        def on_file_selected(file_path: Optional[Path]) -> None:
            if file_path:
                self.run_worker(
                    self._process_console_attachment(str(file_path)),
                    exclusive=True,
                    group="console-attachment",
                )

        await self.app.push_screen(
            EnhancedFileOpen(
                location=".",
                title="Select File to Attach",
                filters=file_filters,
                context="chat_images",
            ),
            callback=on_file_selected,
        )

    def action_paste_clipboard_image(self) -> None:
        """Grab an image from the OS clipboard into the pending attachment."""
        if self._console_setup_modal_blocking():
            return
        self.run_worker(
            self._paste_console_clipboard_image(),
            exclusive=True,
            group="console-clipboard-grab",
        )

    async def _paste_console_clipboard_image(self) -> None:
        """Read the clipboard off-loop and stage its image (or route paths)."""
        from datetime import datetime as _datetime

        grab = await asyncio.to_thread(grab_clipboard_image)
        if grab.kind == "unavailable":
            self.app_instance.notify(
                "Clipboard images aren't readable on this platform — "
                "use Attach or drop a file.",
                severity="warning",
            )
            return
        if grab.kind == "empty":
            self.app_instance.notify("No image on the clipboard.")
            return
        if grab.kind == "paths":
            total_dropped = len(grab.paths)
            attachable_paths = [p for p in grab.paths if looks_attachable(p)]
            if not attachable_paths:
                self.app_instance.notify("No image on the clipboard.")
                return
            store = self._ensure_console_chat_store()
            session = store.ensure_session(
                workspace_id=store.workspace_context.active_workspace_id
            )
            # Attach sequentially, stopping as soon as the cap is hit, so a
            # capacity-exhausted drop gets ONE truncation toast here instead
            # of one "limit reached" toast per remaining file.
            attached_count = 0
            for candidate in attachable_paths:
                if len(store.pending_attachments(session.id)) >= MAX_PENDING_ATTACHMENTS:
                    break
                await self._process_console_attachment(candidate)
                attached_count += 1
            if attached_count < total_dropped:
                self.app_instance.notify(
                    f"Attached first {attached_count} of {total_dropped} dropped files."
                )
            return
        from tldw_chatbook.Chat.attachment_core import process_attachment_bytes

        try:
            display_name = (
                f"clipboard-{_datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            )
            attachment = await asyncio.to_thread(
                lambda: asyncio.run(
                    process_attachment_bytes(
                        grab.png_bytes or b"", display_name=display_name
                    )
                )
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Clipboard image processing failed.")
            self.app_instance.notify(
                f"Could not attach clipboard image: {escape_markup(str(exc))}",
                severity="error",
            )
            return
        store = self._ensure_console_chat_store()
        session = store.ensure_session(
            workspace_id=store.workspace_context.active_workspace_id
        )
        if not store.add_pending_attachment(session.id, attachment):
            self.app_instance.notify(
                "Attachment limit reached (5 per message).", severity="warning"
            )
            self._sync_console_control_bar()
            return
        # Composer label reflects the whole staged list (1 vs N) and is
        # recomputed centrally by `_sync_console_composer_action_state`
        # (called via `_sync_console_control_bar` below) -- no direct
        # `set_pending_attachment_label` call needed here.
        self.app_instance.notify(
            f"{escape_markup(attachment.display_name)} attached"
        )
        self._sync_console_control_bar()

    async def _process_console_attachment(self, file_path: str) -> None:
        """Process a picked file and route it into the native Console composer."""
        from tldw_chatbook.Chat.attachment_core import process_attachment_path

        try:
            attachment = await asyncio.to_thread(
                lambda: asyncio.run(process_attachment_path(file_path))
            )
        except Exception as exc:
            logger.error(f"Console attachment processing failed for {file_path}: {exc}")
            self.app_instance.notify(
                str(exc) or "Failed to process attachment.", severity="error"
            )
            return
        composer = self._console_composer_or_none()
        if attachment.insert_mode == "inline":
            if composer is None or not attachment.text_content:
                self.app_instance.notify(
                    "Nothing to insert from this file.", severity="warning"
                )
                return
            composer.insert_file_segment(
                attachment.text_content, f"📄 {attachment.label}"
            )
            self.app_instance.notify(
                f"{escape_markup(attachment.display_name)} content inserted"
            )
        else:
            store = self._ensure_console_chat_store()
            session = store.ensure_session(
                workspace_id=store.workspace_context.active_workspace_id
            )
            if not store.add_pending_attachment(session.id, attachment):
                self.app_instance.notify(
                    "Attachment limit reached (5 per message).", severity="warning"
                )
                self._sync_console_control_bar()
                return
            # Composer label reflects the whole staged list (1 vs N) and is
            # recomputed centrally by `_sync_console_composer_action_state`
            # (called via `_sync_console_control_bar` below).
            self.app_instance.notify(f"{escape_markup(attachment.display_name)} attached")
        self._sync_console_control_bar()

    @on(Button.Pressed, "#console-clear-attachment")
    def handle_console_clear_attachment(self, event: Button.Pressed) -> None:
        """Remove the pending native Console attachment."""
        event.stop()
        store = self._ensure_console_chat_store()
        had_pending_attachment = False
        if store.active_session_id is not None:
            try:
                had_pending_attachment = (
                    store.pending_attachment(store.active_session_id) is not None
                )
                store.clear_pending_attachment(store.active_session_id)
            except KeyError:
                had_pending_attachment = False
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.set_pending_attachment_label(None)
        if had_pending_attachment:
            self.app_instance.notify("Attachment cleared")
        self._sync_console_control_bar()

    @on(Button.Pressed, "#console-save-chatbook")
    def handle_console_save_chatbook(self, event: Button.Pressed) -> None:
        """Route available Chatbook artifacts through the existing Artifacts handoff."""
        event.stop()
        self._save_console_chatbook_from_visible_action()

    def _save_console_chatbook_from_visible_action(self) -> None:
        """Route available Chatbook artifacts through the existing Artifacts handoff."""
        launch = self._consume_pending_console_launch()
        if self._launch_targets_chatbook_artifact(launch):
            handler = getattr(self.app_instance, "open_console_live_work_primary_action", None)
            if callable(handler) and bool(handler(launch)):
                return
        self.app_instance.notify(
            "No Chatbook artifact is available to save yet.",
            severity="warning",
        )

    async def _open_console_provider_recovery(self) -> None:
        """Route provider setup recovery to the smallest relevant settings surface."""
        _label, target, _tooltip = self._console_provider_recovery_action()
        if target in {"console", "hidden"} and getattr(self, "is_mounted", False):
            await self._open_console_settings(
                focus_model=(
                    target == "hidden"
                    or self._is_console_choose_model_action(_label)
                )
            )
            return
        provider, model, settings = self._active_console_provider_model_display()
        settings_provider = settings.provider if settings is not None else None
        provider_context = str(settings_provider or provider or "").strip()
        screen_context: dict[str, object] = {
            "category": SettingsCategoryId.PROVIDERS_MODELS.value,
        }
        if provider_context:
            screen_context["provider"] = provider_context
        settings_model = settings.model if settings is not None else None
        model_context = str(model or settings_model or "").strip()
        if model_context:
            screen_context["model"] = model_context
        field_context = self._console_provider_recovery_field()
        if field_context:
            screen_context["field"] = field_context
        self.post_message(
            NavigateToScreen(
                TAB_SETTINGS,
                screen_context=screen_context,
            )
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

        if action_id != "delete":
            self._pending_console_delete_message_id = None

        if action_id == "save-as":
            destinations = self._console_save_as_destinations(message)

            def _apply_save_as(destination: str | None) -> None:
                savers = {
                    "Note": self._save_console_message_as_note,
                    "Media": self._save_console_message_as_media,
                    "Prompt": self._save_console_message_as_prompt,
                    "Chatbook": self._save_console_message_as_chatbook,
                }
                saver = savers.get(destination or "")
                if saver is not None:
                    self.run_worker(
                        saver(message_id), exclusive=True, group="console-save-as"
                    )

            await self.app.push_screen(
                ConsoleSaveAsModal(
                    destinations=destinations,
                    message_role=self._console_message_role_label(message),
                    message_excerpt=self._console_message_excerpt(message),
                ),
                callback=_apply_save_as,
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
        if action_id == "edit" and result.status == "edit_requested":
            await self._open_console_message_edit_modal(
                message_id=message_id,
                content=result.target_content or "",
            )
            return True
        if action_id == "retry" and result.status == "completed":
            controller = self._ensure_console_chat_controller()
            # Gate BEFORE spawning: an exclusive console-run worker cancels the
            # in-flight run at creation time, before the controller's own
            # rejection can run — the screen must refuse, like the submit path.
            if not controller.run_state.is_send_allowed:
                self.app_instance.notify(CONSOLE_RUN_ALREADY_RUNNING_COPY, severity="warning")
                return True
            self.run_worker(
                self._retry_console_message(controller, message_id),
                exclusive=True,
                group="console-run",
            )
            return True
        if action_id == "regenerate" and result.status == "wip":
            controller = self._ensure_console_chat_controller()
            if not controller.run_state.is_send_allowed:
                self.app_instance.notify(CONSOLE_RUN_ALREADY_RUNNING_COPY, severity="warning")
                return True
            self.run_worker(
                self._regenerate_console_message(controller, message_id),
                exclusive=True,
                group="console-run",
            )
            return True
        if action_id in {"variant-previous", "variant-next"} and result.status == "completed":
            self._select_console_message_variant(message_id, direction=action_id)
            await self._sync_native_console_chat_ui()
            return True
        if action_id in {"feedback-up", "feedback-down"} and result.status == "completed":
            feedback = "up" if action_id == "feedback-up" else "down"
            store.set_message_feedback(message_id, feedback)
            await self._sync_native_console_chat_ui()
            self.app_instance.notify(result.visible_copy, severity="information")
            return True
        if action_id == "toggle-image-view" and result.status == "completed":
            self._handle_console_toggle_image_view(message_id)
            await self._sync_native_console_chat_ui()
            return True
        if action_id == "save-image" and result.status == "completed":
            self.run_worker(
                self._save_console_message_image(message_id),
                exclusive=True,
                group="console-save-image",
            )
            return True
        if action_id == "delete" and result.status == "completed":
            if self._pending_console_delete_message_id != message_id:
                self._pending_console_delete_message_id = message_id
                self._last_console_action = ConsoleActionResult(
                    action_id=action_id,
                    status="blocked",
                    visible_copy="Press Delete again to remove this message.",
                    target_message_id=message_id,
                )
                await self._sync_native_console_chat_ui()
                return True
            self._pending_console_delete_message_id = None
            store.delete_message(message_id)
            await self._sync_native_console_chat_ui()
            self.app_instance.notify(result.visible_copy, severity="information")
            return True
        if action_id == "continue" and result.status == "continue_requested":
            controller = self._ensure_console_chat_controller()
            if not controller.run_state.is_send_allowed:
                self.app_instance.notify(CONSOLE_RUN_ALREADY_RUNNING_COPY, severity="warning")
                return True
            self.run_worker(
                self._continue_console_message(controller, message_id),
                exclusive=True,
                group="console-run",
            )
            return True
        severity = "information" if result.status in {"completed", "wip"} else "warning"
        self.app_instance.notify(result.visible_copy, severity=severity)
        return True

    @staticmethod
    def _console_message_role_label(message: ConsoleChatMessage) -> str:
        """Return a user-facing role label for a Console transcript message."""
        role = message.role.value if hasattr(message.role, "value") else str(message.role)
        return role.title()

    @staticmethod
    def _console_message_content(message: ConsoleChatMessage) -> str:
        """Return the currently visible content for a Console transcript message."""
        if message.variants is not None:
            return message.variants.current.content
        return message.content

    @classmethod
    def _console_message_excerpt(
        cls,
        message: ConsoleChatMessage,
        *,
        max_length: int = 120,
    ) -> str:
        """Return a single-line excerpt for selected-message context surfaces."""
        normalized = " ".join(cls._console_message_content(message).split())
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[: max(0, max_length - 1)].rstrip()}…"

    def _console_save_as_destinations(self, message: Any) -> list[Any]:
        """Return Save-as destinations available in the current app runtime."""
        available_destinations: set[str] = set()
        unavailable_reasons: dict[str, str] = {}

        notes_scope_service = getattr(self.app_instance, "notes_scope_service", None)
        if callable(getattr(notes_scope_service, "save_note", None)):
            available_destinations.add("Note")
        else:
            unavailable_reasons["Note"] = "Notes service is not ready in this session."

        media_db = getattr(self.app_instance, "media_db", None)
        if callable(getattr(media_db, "add_media_with_keywords", None)):
            available_destinations.add("Media")
        else:
            unavailable_reasons["Media"] = "Media library is not ready in this session."

        prompts_db = getattr(self.app_instance, "prompts_db", None)
        if callable(getattr(prompts_db, "add_prompt", None)):
            available_destinations.add("Prompt")
        else:
            unavailable_reasons["Prompt"] = "Prompts service is not ready in this session."

        chatbook_service = getattr(self.app_instance, "local_chatbook_service", None)
        if not callable(getattr(chatbook_service, "create_chatbook", None)):
            unavailable_reasons["Chatbook"] = (
                "Chatbook artifacts service is not ready in this session."
            )
        elif not ConsoleMessageActionService._is_assistant_message(message):
            unavailable_reasons["Chatbook"] = (
                "Only assistant responses can be saved as Chatbook artifacts."
            )
        else:
            available_destinations.add("Chatbook")

        return ConsoleMessageActionService(
            available_save_destinations=available_destinations,
            unavailable_save_reasons=unavailable_reasons,
        ).save_as_destinations(message)

    def _console_save_source_title(self) -> str:
        """Return the active Console conversation title for save-as derivations."""
        session = self._active_native_console_session()
        return str(getattr(session, "title", "") or "").strip()

    async def _save_console_message_image(self, message_id: str) -> None:
        """Write ALL of a Console message's image attachments to disk.

        In-memory bytes are used first; any attachment still dataless (e.g.
        a metadata-only entry left by screen-state restore) falls back to
        one batched DB fetch -- the legacy `messages.image_data` column for
        position 0, `get_attachments_for_messages` for positions >= 1 --
        per the HARD interface contract split addressing.
        """
        import mimetypes as _mimetypes
        from datetime import datetime as _datetime

        store = self._ensure_console_chat_store()
        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message no longer exists.", severity="warning"
            )
            return

        attachments = list(message.attachments)
        if not attachments and (
            message.image_data is not None or message.persisted_message_id is not None
        ):
            # Legacy/raw-constructed messages may carry the scalar image
            # fields without a populated attachments tuple; synthesize a
            # position-0 entry so the fallback below still covers them.
            attachments = [
                MessageAttachment(
                    data=message.image_data,
                    mime_type=message.image_mime_type or "image/png",
                    display_name=message.attachment_label or "",
                    position=0,
                )
            ]

        missing_positions = any(a.data is None for a in attachments)
        if missing_positions and message.persisted_message_id is not None:
            db = getattr(self.app_instance, "chachanotes_db", None)
            persisted_message_id = message.persisted_message_id

            def _fetch_persisted_attachment_data() -> dict[int, tuple[Any, Optional[str]]]:
                fetched: dict[int, tuple[Any, Optional[str]]] = {}
                try:
                    row = (
                        db.get_message_by_id(persisted_message_id)
                        if db is not None
                        else None
                    )
                except Exception:
                    logger.opt(exception=True).warning(
                        "Console save-image DB fallback lookup failed."
                    )
                    row = None
                if row and row.get("image_data") is not None:
                    fetched[0] = (row.get("image_data"), row.get("image_mime_type"))
                getter = getattr(db, "get_attachments_for_messages", None)
                if callable(getter):
                    try:
                        batch = getter([persisted_message_id])
                    except Exception:
                        logger.opt(exception=True).warning(
                            "Console save-image attachment batch fetch failed."
                        )
                        batch = None
                    if isinstance(batch, dict):
                        for row_dict in batch.get(persisted_message_id, []) or []:
                            position = int(row_dict.get("position", 0))
                            fetched[position] = (
                                row_dict.get("data"),
                                row_dict.get("mime_type"),
                            )
                return fetched

            fetched = await asyncio.to_thread(_fetch_persisted_attachment_data)
            if fetched:
                attachments = [
                    replace(
                        attachment,
                        data=fetched[attachment.position][0],
                        mime_type=fetched[attachment.position][1] or attachment.mime_type,
                    )
                    if attachment.data is None and attachment.position in fetched
                    else attachment
                    for attachment in attachments
                ]

        saveable = [a for a in attachments if a.data]
        if not saveable:
            self.app_instance.notify(
                "No image data available for this message.", severity="warning"
            )
            return

        def _write_images_to_disk() -> tuple[list[Path], Path]:
            from tldw_chatbook.Utils.path_validation import validate_path_simple

            save_location = validate_path_simple(
                os.path.expanduser(
                    get_cli_setting("chat.images", "save_location", "~/Downloads")
                )
            )
            save_location.mkdir(parents=True, exist_ok=True)
            base_name = f"console_image_{_datetime.now().strftime('%Y%m%d_%H%M%S')}"
            written: list[Path] = []
            for attachment in saveable:
                extension = (
                    _mimetypes.guess_extension(attachment.mime_type or "image/png")
                    or ".png"
                )
                target = save_location / f"{base_name}{extension}"
                counter = 1
                while target.exists() or target in written:
                    target = save_location / f"{base_name}_{counter}{extension}"
                    counter += 1
                target.write_bytes(bytes(attachment.data))
                written.append(target)
            return written, save_location

        try:
            written, save_location = await asyncio.to_thread(_write_images_to_disk)
        except Exception as exc:
            logger.opt(exception=True).warning("Console save-image write failed.")
            self.app_instance.notify(
                f"Could not save image: {escape_markup(str(exc))}", severity="error"
            )
            return
        if len(written) == 1:
            self.app_instance.notify(f"Image saved to {escape_markup(str(written[0]))}")
        else:
            self.app_instance.notify(
                f"Saved {len(written)} images to {escape_markup(str(save_location))}"
            )

    async def _save_console_message_as_note(self, message_id: str) -> None:
        """Persist one selected Console message as a local Note."""
        notes_scope_service = getattr(self.app_instance, "notes_scope_service", None)
        save_note = getattr(notes_scope_service, "save_note", None)
        if not callable(save_note):
            self.app_instance.notify(
                "Save as Note is unavailable: Notes service is not ready.",
                severity="warning",
            )
            return

        try:
            message = self._ensure_console_chat_store().get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message action target no longer exists.",
                severity="warning",
            )
            return

        content = self._console_message_content(message)
        title = derive_console_save_title(self._console_save_source_title())
        try:
            result = save_note(
                scope=ScopeType.LOCAL_NOTE.value,
                title=title,
                content=content,
                note_id=None,
                version=None,
                user_id=getattr(self.app_instance, "current_user", None) or "default_user",
                workspace_id=None,
                keywords=["console"],
            )
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            logger.opt(exception=True).warning("Console save-as Note failed.")
            self.app_instance.notify(f"Save as Note failed: {exc}", severity="error")
            return
        if not result:
            self.app_instance.notify("Save as Note failed.", severity="error")
            return
        self._last_console_action = ConsoleActionResult(
            action_id="save-as-note",
            status="completed",
            visible_copy="Saved message as Note.",
            target_message_id=message_id,
            target_content=content,
        )
        self.app_instance.notify("Saved message as Note.", severity="information")

    async def _save_console_message_as_media(self, message_id: str) -> None:
        """Persist one selected Console message as a Library media item."""
        media_db = getattr(self.app_instance, "media_db", None)
        add_media = getattr(media_db, "add_media_with_keywords", None)
        if not callable(add_media):
            self.app_instance.notify(
                "Save as Media is unavailable: Media library is not ready.",
                severity="warning",
            )
            return

        try:
            message = self._ensure_console_chat_store().get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message action target no longer exists.",
                severity="warning",
            )
            return

        content = self._console_message_content(message)
        title = derive_console_save_title(
            self._console_save_source_title(),
            role_label=self._console_message_role_label(message),
        )
        try:
            media_id, _media_uuid, save_message = add_media(
                title=title,
                media_type="plaintext",
                content=content,
                keywords=["console"],
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Console save-as Media failed.")
            self.app_instance.notify(f"Save as Media failed: {exc}", severity="error")
            return
        if media_id is None:
            self.app_instance.notify(
                f"Save as Media failed: {save_message or 'no media record was created.'}",
                severity="error",
            )
            return
        self._last_console_action = ConsoleActionResult(
            action_id="save-as-media",
            status="completed",
            visible_copy="Saved message as Media.",
            target_message_id=message_id,
            target_content=content,
        )
        self.app_instance.notify(
            "Saved message as Media. It appears under Library ▸ Media.",
            severity="information",
        )

    async def _save_console_message_as_prompt(self, message_id: str) -> None:
        """Persist one selected Console message as a prompt in the Prompts library."""
        prompts_db = getattr(self.app_instance, "prompts_db", None)
        add_prompt = getattr(prompts_db, "add_prompt", None)
        if not callable(add_prompt):
            self.app_instance.notify(
                "Save as Prompt is unavailable: Prompts service is not ready.",
                severity="warning",
            )
            return

        try:
            message = self._ensure_console_chat_store().get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message action target no longer exists.",
                severity="warning",
            )
            return

        from tldw_chatbook.DB.Prompts_DB import ConflictError

        content = self._console_message_content(message)
        conversation_title = self._console_save_source_title()
        base_name = derive_console_save_title(conversation_title)
        details = (
            f"Saved from Console conversation: {conversation_title}."
            if conversation_title
            else "Saved from a Console conversation."
        )
        prompt_id = None
        saved_name = base_name
        try:
            for attempt in range(1, 10):
                saved_name = base_name if attempt == 1 else f"{base_name} ({attempt})"
                try:
                    prompt_id, _prompt_uuid, save_message = add_prompt(
                        name=saved_name,
                        author="Console",
                        details=details,
                        system_prompt=content,
                        keywords=["console"],
                        overwrite=False,
                    )
                except ConflictError:
                    continue
                if prompt_id is not None and "soft-deleted" in str(save_message or ""):
                    # Name collides with a soft-deleted prompt: nothing was
                    # saved, so keep probing suffixed names.
                    prompt_id = None
                    continue
                break
        except Exception as exc:
            logger.opt(exception=True).warning("Console save-as Prompt failed.")
            self.app_instance.notify(f"Save as Prompt failed: {exc}", severity="error")
            return
        if prompt_id is None:
            self.app_instance.notify(
                "Save as Prompt failed: a prompt with this name already exists.",
                severity="error",
            )
            return
        self._last_console_action = ConsoleActionResult(
            action_id="save-as-prompt",
            status="completed",
            visible_copy="Saved message as Prompt.",
            target_message_id=message_id,
            target_content=content,
        )
        self.app_instance.notify(
            f"Saved message as Prompt '{saved_name}' in the local Prompts library.",
            severity="information",
        )

    async def _save_console_message_as_chatbook(self, message_id: str) -> None:
        """Register one selected assistant message as a Chatbook artifact."""
        chatbook_service = getattr(self.app_instance, "local_chatbook_service", None)
        create_chatbook = getattr(chatbook_service, "create_chatbook", None)
        if not callable(create_chatbook):
            self.app_instance.notify(
                "Save as Chatbook is unavailable: Chatbook artifacts service is not ready.",
                severity="warning",
            )
            return

        try:
            message = self._ensure_console_chat_store().get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message action target no longer exists.",
                severity="warning",
            )
            return

        if not ConsoleMessageActionService._is_assistant_message(message):
            self.app_instance.notify(
                "Only assistant responses can be saved as Chatbook artifacts.",
                severity="warning",
            )
            return

        content = self._console_message_content(message)
        provider: str | None = None
        model: str | None = None
        try:
            provider, model, _settings = self._active_console_provider_model_display()
        except Exception:
            logger.opt(exception=True).debug(
                "Console save-as Chatbook could not resolve provider/model context."
            )
        payload = console_chatbook_artifact_payload(
            title=derive_console_save_title(self._console_save_source_title()),
            message_text=content,
            message_role=self._console_message_role_label(message),
            conversation_id=self._current_console_conversation_id(),
            message_id=message_id,
            provider=provider,
            model=model,
        )
        try:
            result = create_chatbook(**payload)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.opt(exception=True).warning("Console save-as Chatbook failed.")
            self.app_instance.notify(f"Save as Chatbook failed: {exc}", severity="error")
            return
        self._last_console_action = ConsoleActionResult(
            action_id="save-as-chatbook",
            status="completed",
            visible_copy="Saved message as Chatbook artifact.",
            target_message_id=message_id,
            target_content=content,
        )
        self.app_instance.notify(
            "Saved message as a Chatbook artifact. It appears under Artifacts.",
            severity="information",
        )

    async def _open_console_message_edit_modal(self, *, message_id: str, content: str) -> None:
        """Open the dedicated transcript edit modal for one Console message."""
        store = self._ensure_console_chat_store()

        def _apply_edit(result: str | None) -> None:
            if result is None:
                return
            try:
                store.update_message_content(message_id, result)
            except ValueError as exc:
                self.app_instance.notify(str(exc), severity="warning")
                return
            except KeyError:
                self.app_instance.notify(
                    "Console message action target no longer exists.",
                    severity="error",
                )
                return
            self._last_console_action = ConsoleActionResult(
                action_id="edit",
                status="completed",
                visible_copy="Edited message.",
                target_message_id=message_id,
                target_content=result,
            )
            self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")
            self.app_instance.notify("Edited message.", severity="information")

        await self.app.push_screen(
            ConsoleEditMessageModal(content=content),
            callback=_apply_edit,
        )

    @staticmethod
    def _parse_console_message_action_button_id(button_id: str) -> tuple[str | None, str | None]:
        prefixes = (
            ("console-message-action-feedback-up-", "feedback-up"),
            ("console-message-action-feedback-down-", "feedback-down"),
            ("console-message-action-variant-previous-", "variant-previous"),
            ("console-message-action-variant-next-", "variant-next"),
            ("console-message-action-save-as-", "save-as"),
            ("console-message-action-save-image-", "save-image"),
            ("console-message-action-toggle-image-view-", "toggle-image-view"),
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
        if result.accepted:
            self._clear_native_console_message_selection()
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
        self._sync_console_pending_delete_confirmation()
        control_state = self._build_console_control_state(
            self._pending_console_launch_context
        )
        workbench_state = self._build_console_workbench_state(control_state)
        control_state_changed = control_state != self._last_console_control_state
        workbench_state_changed = workbench_state != self._last_console_workbench_state
        if control_state_changed or workbench_state_changed:
            try:
                control_bar = self.query_one("#console-control-bar", ConsoleControlBar)
            except QueryError:
                control_bar = None
            if control_bar is not None:
                control_bar.sync_state(control_state, actions=workbench_state.actions)
            self._sync_console_workbench_state(control_state, workbench_state=workbench_state)
            self._last_console_control_state = control_state
            self._last_console_workbench_state = workbench_state
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
        self._sync_console_rail_visibility_if_changed(self._current_console_rail_state())

    def _sync_console_workbench_state(
        self,
        control_state: ConsoleControlState,
        *,
        workbench_state: Any | None = None,
    ) -> None:
        """Refresh visible Workbench primitives from current Console state."""
        if workbench_state is None:
            workbench_state = self._build_console_workbench_state(control_state)
        try:
            self.query_one("#console-workbench-header", DestinationHeader).sync_state(
                workbench_state.header
            )
        except QueryError:
            pass
        try:
            self.query_one("#console-workbench-mode-strip", ModeStrip).sync_modes(
                workbench_state.modes
            )
        except QueryError:
            pass
        try:
            self.query_one("#console-workbench-command-strip", CommandStrip).sync_actions(
                workbench_state.actions
            )
        except QueryError:
            pass
        try:
            self.query_one("#workbench-recovery-callout", RecoveryCallout).sync_state(
                workbench_state.recovery
            )
        except QueryError:
            pass

    def _sync_console_workbench_actions_from_draft(self) -> None:
        """Refresh Workbench command readiness after composer draft changes."""
        self._sync_console_workbench_state(
            self._build_console_control_state(self._pending_console_launch_context)
        )

    def _sync_console_pending_delete_confirmation(self) -> None:
        """Clear stale destructive-action confirmation when transcript selection changes."""
        if self._pending_console_delete_message_id is None:
            return
        try:
            transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        except QueryError:
            self._pending_console_delete_message_id = None
            return
        if transcript.selected_message_id != self._pending_console_delete_message_id:
            self._pending_console_delete_message_id = None

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
        setup_blocked_reason = self._console_setup_blocked_reason()
        attachment_blocked_reason = self._console_attachment_blocked_reason()
        send_blocked = (
            send_blocked
            or bool(setup_blocked_reason)
            or bool(attachment_blocked_reason)
        )

        pending = self._console_pending_image_attachment()

        composer.sync_action_state(
            has_draft=bool(composer.draft_text().strip()) or pending is not None,
            run_active=run_active,
            can_save_chatbook=can_save_chatbook,
            send_blocked=send_blocked,
            setup_blocked_reason=setup_blocked_reason or attachment_blocked_reason,
        )
        # sync_action_state resets the attach button's tooltip to generic copy
        # (console_composer_bar.py L303); apply the pending-attachment label
        # after, not before, so "Attached: ..." wins over the generic tooltip.
        # One staged item keeps its own descriptive label ("photo.png ·
        # 240 KB"); more than one collapses to an "N files" summary. The
        # composer prepends its own 📎 glyph to whatever label it's given
        # (console_composer_bar.py's `set_pending_attachment_label`, which
        # stays untouched), so the label passed here carries NO glyph and
        # the rendered indicator reads exactly "📎 {N} files". The full
        # per-file name list is surfaced via the "<name> attached" toast
        # each staged file already fires, not a composer tooltip.
        store = self._console_chat_store
        pendings: list[Any] = []
        if store is not None and store.active_session_id is not None:
            try:
                pendings = store.pending_attachments(store.active_session_id)
            except KeyError:
                pendings = []
        if not pendings:
            attachment_label = None
        elif len(pendings) == 1:
            attachment_label = pendings[0].label
        else:
            attachment_label = f"{len(pendings)} files"
        composer.set_pending_attachment_label(attachment_label)

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

    def on_key(self, event: Key) -> None:
        """Treat the Console composer as the default printable text target."""
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        if self._console_setup_modal_blocking():
            # Workbench is inert behind the first-run setup modal; never route
            # printable/edit keys into the covered composer.
            return
        if not self._should_capture_console_input(composer):
            return
        if event.key in {"ctrl+a", "super+a", "cmd+a", "meta+a"}:
            composer.select_all_draft()
            event.stop()
            event.prevent_default()
            return
        if (
            event.key in {"ctrl+c", "super+c", "cmd+c", "meta+c"}
            and composer.has_full_draft_selection()
        ):
            copy_to_clipboard = getattr(self.app_instance, "copy_to_clipboard", None)
            if callable(copy_to_clipboard):
                copy_to_clipboard(composer.draft_text())
            event.stop()
            event.prevent_default()
            return
        if event.key in {"backspace", "ctrl+h", "delete"}:
            composer.delete_left()
            self._sync_console_workbench_actions_from_draft()
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            if composer.activate_focused_paste_token():
                event.stop()
                event.prevent_default()
                return
            event.stop()
            event.prevent_default()
            try:
                self.query_one("#console-send-message", Button).press()
            except QueryError:
                self.app_instance.notify("Console send is unavailable.", severity="error")
            return
        if event.key == "ctrl+u":
            composer.clear_draft()
            self._sync_console_workbench_actions_from_draft()
            event.stop()
            event.prevent_default()
            return
        if event.is_printable and event.character is not None:
            composer.insert_text(event.character)
            self._sync_console_workbench_actions_from_draft()
            self._dismiss_console_guidance()
            event.stop()
            event.prevent_default()

    def on_paste(self, event: Paste) -> None:
        """Treat pasted text as Console composer draft input by default."""
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        if self._console_setup_modal_blocking():
            return
        if not self._should_capture_console_input(composer):
            return
        dropped = extract_dropped_path(event.text)
        if dropped is not None and looks_attachable(dropped.path):
            event.stop()
            self._dismiss_console_guidance()
            if dropped.total_dropped > 1:
                # `extract_dropped_path` only ever surfaces the first
                # decoded path (plus the total line count); terminal
                # drag-drop paste can attach at most that one file, so the
                # truncation toast's "n" is always 1 here.
                self.app_instance.notify(
                    f"Attached first 1 of {dropped.total_dropped} dropped files."
                )
            self.run_worker(
                self._process_console_attachment(dropped.path),
                exclusive=True,
                group="console-attachment",
            )
            return
        composer.insert_pasted_text(event.text)
        self._sync_console_workbench_actions_from_draft()
        self._dismiss_console_guidance()
        event.stop()

    def on_mouse_up(self, event: MouseUp) -> None:
        """Route terminal mouse-up events to paste tokens in textual-web."""
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        screen_x = getattr(event, "screen_x", None)
        screen_y = getattr(event, "screen_y", None)
        if screen_x is None or screen_y is None:
            return
        if not composer.activate_visible_draft_screen_position(screen_x, screen_y):
            return
        composer.suppress_next_draft_click()
        event.stop()
        event.prevent_default()

    def on_click(self, event: Click) -> None:
        """Reset pending paste unfurl confirmation when clicking outside the token."""
        target = getattr(event, "widget", None) or getattr(event, "control", None)
        if getattr(target, "id", None) == "console-command-visible-text":
            return
        if getattr(target, "id", None) == "console-rail-system-line":
            event.stop()
            self.run_worker(self._open_console_system_prompt_editor(), exclusive=False)
            return
        if getattr(target, "id", None) == "console-agent-section-subagents":
            event.stop()
            self._toggle_console_agent_drilldown_from_subagents_click()
            return
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        screen_x = getattr(event, "screen_x", None)
        screen_y = getattr(event, "screen_y", None)
        targets_visible_draft = (
            screen_x is not None
            and screen_y is not None
            and composer.is_visible_draft_screen_position(screen_x, screen_y)
        )
        if targets_visible_draft:
            if composer.consume_suppressed_draft_click():
                event.stop()
                event.prevent_default()
                return
            if composer.activate_visible_draft_screen_position(screen_x, screen_y):
                event.stop()
                event.prevent_default()
                return
        elif composer.has_suppressed_draft_click():
            composer.clear_suppressed_draft_click()
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
            if provider is not None or model is not None:
                app_config = getattr(self.app_instance, "app_config", {}) or {}
                current_defaults = build_default_console_session_settings(
                    app_config,
                    settings.provider,
                    settings.model,
                )
                override_fields = {
                    field: getattr(settings, field)
                    for field in (
                        "temperature",
                        "top_p",
                        "min_p",
                        "top_k",
                        "max_tokens",
                        "streaming",
                    )
                    if getattr(settings, field) != getattr(current_defaults, field)
                }
                target_provider = (
                    str(provider).strip()
                    if provider is not None and _has_selected_text(provider)
                    else settings.provider
                )
                target_model = (
                    str(model).strip()
                    if model is not None and _has_selected_text(model)
                    else settings.model
                )
                next_settings = build_default_console_session_settings(
                    app_config,
                    target_provider,
                    target_model,
                )
                if model is not None and not _has_selected_text(model):
                    next_settings = replace(next_settings, model=None)
                next_settings = replace(
                    next_settings,
                    **override_fields,
                    persona_label=settings.persona_label,
                    character_label=settings.character_label,
                )
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
            logger.opt(exception=True).error(f"Failed to sync shell bar from state: {e}")

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
            logger.opt(exception=True).error(f"Failed to sync shell bar from live session: {e}")

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
            logger.opt(exception=True).error(f"Error restoring input text: {e}")
    
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
            logger.opt(exception=True).error(f"Error saving sidebar settings: {e}")
    
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
            logger.opt(exception=True).error(f"Error restoring sidebar settings: {e}")
    
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

    # NOTE (task-247, perf): there used to be an on_screen_suspend() override
    # here that called self.save_state() again and discarded the result.
    # app.py already calls save_state() explicitly before switching screens
    # away from Console (see the pre-navigation save in switch_screen /
    # _screen_states bookkeeping) and stores that return value -- the second
    # call here was pure waste (a full O(sessions x messages) native-console
    # serialization) on every tab switch away from Console. Removed rather
    # than left as a no-op so it doesn't shadow a future base-class
    # implementation.

    def on_screen_resume(self) -> None:
        """Called when returning to this screen."""
        logger.debug("Chat screen resuming")
        # Re-evaluate setup-card/model readiness before touching focus. Some
        # recovery flows (e.g. certain providers' API-key recovery) navigate to
        # the full Settings screen and back rather than completing setup via
        # the in-Console settings modal callback, so the setup modal's blocking
        # state can be stale by the time this screen resumes. Without this,
        # `_restore_console_workbench_focus` below would just re-apply the
        # stale block and the modal could stick even after setup completed
        # elsewhere.
        self._sync_console_transcript_guidance()
        self.sync_task_resume_state()
        self._register_console_footer_shortcuts()
        # Delayed exactly like the `on_mount` consumption below, to give the
        # native composer a chance to finish mounting on first navigation to
        # this screen. Unlike `on_mount`, nothing here schedules an
        # equivalent `_sync_native_console_chat_ui` pass ahead of this timer,
        # so this call site cannot rely on timing to avoid the active-session
        # draft-load wipe race described on `_consume_pending_console_prompt_insert`
        # -- that method settles `_console_visible_draft_session_id` itself,
        # immediately before inserting, so the insert is self-guarding
        # regardless of which lifecycle hook scheduled it.
        self.set_timer(0.15, self._consume_pending_console_prompt_insert)
        self.call_after_refresh(self._restore_console_workbench_focus)
        self.run_worker(self._refresh_console_skill_candidates(), exclusive=False)
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
        if button_id == "console-model-section-configure":
            await self.on_console_settings_open(event)
            return
        if button_id == "console-agent-drilldown-back":
            event.stop()
            self._console_agent_drilldown_run_id = None
            self.run_worker(self._sync_native_console_chat_ui(), exclusive=True, group="console-sync")
            return
        if button_id and button_id.startswith(CONSOLE_RAIL_SECTION_TOGGLE_PREFIX):
            event.stop()
            self._toggle_console_rail_section(
                button_id.removeprefix(CONSOLE_RAIL_SECTION_TOGGLE_PREFIX)
            )
            return
        if button_id == "console-new-chat-tab":
            event.stop()
            await self._create_native_console_session_from_active_context()
            return
        if button_id and button_id.startswith("console-conversation-browser-section-toggle-"):
            event.stop()
            group_id = str(getattr(event.button, "group_id", "") or "").strip()
            state = self._build_console_workspace_context_state()
            section_id = group_id.removeprefix("section:")
            section = None
            browser = state.conversation_browser
            if browser is not None:
                section = next(
                    (
                        candidate
                        for candidate in browser.sections
                        if candidate.section_id == section_id
                    ),
                    None,
                )
            collapsed = not bool(section.collapsed if section is not None else False)
            self._set_console_conversation_browser_group_collapsed(group_id, collapsed)
            self._sync_console_workspace_context()
            return
        if button_id and button_id.startswith("console-conversation-browser-group-toggle-"):
            event.stop()
            group_id = str(getattr(event.button, "group_id", "") or "").strip()
            state = self._build_console_workspace_context_state()
            group = None
            browser = state.conversation_browser
            if browser is not None:
                for section in browser.sections:
                    group = next(
                        (
                            candidate
                            for candidate in section.groups
                            if candidate.group_id == group_id
                        ),
                        None,
                    )
                    if group is not None:
                        break
            collapsed = not bool(group.collapsed if group is not None else False)
            self._set_console_conversation_browser_group_collapsed(group_id, collapsed)
            self._sync_console_workspace_context()
            return
        if button_id and button_id.startswith("console-conversation-star-"):
            event.stop()
            conversation_id = str(
                getattr(event.button, "conversation_id", "") or ""
            ).strip()
            if not conversation_id:
                self.app_instance.notify(
                    "Save this conversation before starring it.",
                    severity="warning",
                )
                return
            marks_service = getattr(
                self.app_instance,
                "conversation_local_marks_service",
                None,
            )
            if marks_service is None:
                self.app_instance.notify(
                    "Local stars are unavailable.",
                    severity="warning",
                )
                return
            star_action = "resolve"
            try:
                is_starred = getattr(marks_service, "is_starred", None)
                currently_starred = (
                    bool(is_starred(conversation_id))
                    if callable(is_starred)
                    else bool(getattr(event.button, "starred", False))
                )
                star_action = "unstar" if currently_starred else "star"
                if currently_starred:
                    marks_service.unstar_conversation(conversation_id)
                else:
                    marks_service.star_conversation(conversation_id)
            except Exception:
                logger.exception(
                    "Unable to update local conversation star "
                    "conversation_id={} action={}",
                    conversation_id,
                    star_action,
                )
                self.app_instance.notify(
                    "Unable to update local star.",
                    severity="warning",
                )
                return
            self._sync_console_workspace_context()
            return
        if button_id == "console-workspace-conversations-toggle":
            event.stop()
            try:
                workspace_context = self.query_one(
                    "#console-workspace-context",
                    ConsoleWorkspaceContextTray,
                )
            except (NoMatches, QueryError):
                self._sync_console_workspace_context()
                return
            state = getattr(workspace_context, "state", None)
            if (
                state is not None
                and getattr(state, "conversation_browser", None) is None
            ):
                section = getattr(state, "conversation_section", None)
                if section is None:
                    return
                collapsed = not bool(section.collapsed)
                self._set_console_workspace_conversations_collapsed(
                    section.workspace_id,
                    collapsed,
                )
                workspace_context.sync_state(
                    replace(
                        state,
                        conversation_section=replace(
                            section,
                            collapsed=collapsed,
                        ),
                    )
                )
                return
            self._sync_console_workspace_context()
            return
        if button_id == "console-new-workspace-conversation":
            event.stop()
            await self._create_native_console_session_from_active_context()
            return
        if button_id == "console-workspace-conversation-search-clear":
            event.stop()
            if self._console_conversation_browser_search_timer is not None:
                self._console_conversation_browser_search_timer.stop()
                self._console_conversation_browser_search_timer = None
            if self._console_workspace_conversation_search_timer is not None:
                self._console_workspace_conversation_search_timer.stop()
                self._console_workspace_conversation_search_timer = None
            self._console_conversation_browser_query = ""
            self._console_conversation_browser_search_token += 1
            self._console_conversation_browser_rows = ()
            self._console_conversation_browser_total = None
            self._console_conversation_browser_error = ""
            self._console_workspace_conversation_query = ""
            self._console_workspace_conversation_search_token += 1
            self._console_workspace_conversation_search_rows = ()
            self._console_workspace_conversation_search_total = None
            self._console_workspace_conversation_search_error = ""
            self._sync_console_workspace_context()
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
            return
        if button_id and button_id.startswith("console-workspace-conversation-"):
            event.stop()
            conversation_id = str(
                getattr(event.button, "conversation_id", "") or ""
            ).strip()
            row_key = str(getattr(event.button, "row_key", "") or "").strip()
            browser_row = self._find_console_browser_row(
                row_key or conversation_id,
                conversation_id=conversation_id,
            )
            if browser_row is not None:
                self._activate_console_workspace_for_browser_row(browser_row)
                row_conversation_id = str(browser_row.conversation_id or "").strip()
                session_id = self._console_session_id_for_browser_row(browser_row)
            else:
                row_conversation_id = conversation_id
                session_id = self._console_session_id_for_workspace_conversation(
                    conversation_id
                )
            if session_id is None:
                if not row_conversation_id:
                    self.app_instance.notify(
                        "This conversation row is no longer available.",
                        severity="warning",
                    )
                    return
                resumed = await self._resume_console_workspace_conversation(
                    row_conversation_id,
                    target_scope_type=(
                        browser_row.scope_type if browser_row is not None else None
                    ),
                    target_workspace_id=(
                        browser_row.workspace_id if browser_row is not None else None
                    ),
                )
                if resumed:
                    await self._refresh_console_conversation_browser_after_selection()
                    return
                self.app_instance.notify(
                    "Open this saved conversation from Library before switching here.",
                    severity="warning",
                )
                return
            controller = self._ensure_console_chat_controller()
            if controller.store.active_session_id != session_id:
                if browser_row is None:
                    self._set_active_workspace_for_console_session(session_id)
                controller.switch_session(session_id)
                await self._sync_native_console_chat_ui()
            self._focus_console_composer_if_needed(force=True)
            await self._refresh_console_conversation_browser_after_selection()
            return
        if button_id and button_id.startswith("console-close-session-tab-"):
            event.stop()
            session_id = button_id.removeprefix("console-close-session-tab-")
            store = self._ensure_console_chat_store()
            try:
                messages = store.messages_for_session(session_id)
            except KeyError:
                messages = []
            closing_ids = [m.id for m in messages]

            def _evict_closing_session_images() -> None:
                _state, cache = self._ensure_console_image_view()
                cache.evict_session(closing_ids)

            if messages:
                from ...Widgets.confirmation_dialog import ConfirmationDialog

                async def _do_close() -> None:
                    self._ensure_console_chat_controller().close_session(session_id)
                    _evict_closing_session_images()
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
                _evict_closing_session_images()
                await self._sync_native_console_chat_ui()
            return
        if button_id and button_id.startswith("console-session-tab-"):
            event.stop()
            session_id = button_id.removeprefix("console-session-tab-")
            controller = self._ensure_console_chat_controller()
            if controller.store.active_session_id == session_id:
                self._open_console_session_rename_modal(session_id)
                return
            await self._activate_native_console_session(session_id)
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
            logger.opt(exception=True).error(f"Error running diagnostics: {e}")
    
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
