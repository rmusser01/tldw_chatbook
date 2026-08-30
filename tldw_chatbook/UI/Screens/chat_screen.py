"""Chat screen implementation with comprehensive state management."""

from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from dataclasses import dataclass, replace
import asyncio
from functools import partial
import inspect
import json
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING

import toml
from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.events import (
    Click,
    DescendantBlur,
    DescendantFocus,
    Key,
    MouseDown,
    MouseUp,
    Paste,
    Resize,
)
from textual.message_pump import NoActiveAppError
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Static, Select, Collapsible, Input

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Navigation.pending_handoff_store import (
    ConsoleProviderIntent,
    HandoffChannel,
)
from ..Navigation.screen_state_store import ConsolePromptTargetProjection
from .chat_screen_state import TaskResumeState

# `TrajectoryScreen` is deliberately NOT imported here (TASK-22213): its
# module drags `trajectory_import` -> `trajectory_export` plus the timeline
# widgets (~4,400 LOC) onto the Chat first-paint import leg, and its only
# use is the `y` action's push. It is imported locally in
# `action_open_trajectory_view`; the guard is
# `Tests/Packaging/test_rag_boot_import_closure.py`
# (`test_chat_screen_import_does_not_execute_the_deferred_packages`).
#
# The same constraint holds TRANSITIVELY, through files this comment cannot
# see from here (TASK-23020: #2126 breached it ~24 h after 22213 shipped, via
# `console_conversation_inspector` -> `console_exchange_export_dialog` /
# `Chat/console_exchange_export.py`, each importing ONE enum name from
# `Chat.trajectory_export`). Anything reachable from this module at import
# time must take `TraceExportProfile` from `Chat/trace_export_profiles.py`
# and the shared profile copy from
# `Widgets/Console/trace_export_profile_ui.py` -- never from
# `Chat.trajectory_export` or `Widgets/Console/trace_export_dialog.py`.
# Per-file guard that names the offender:
# `Tests/Packaging/test_exchange_export_trajectory_deferral.py`.
from .provider_model_resolution import (
    ResolvedProviderModelOption,
    resolve_effective_provider_model,
    resolve_provider_model_options,
)
from .settings_config_models import SettingsCategoryId
from ..Console_Modules.frame import (
    frame_console_region,
    sync_console_focus_paint,
)
from ..Console_Modules.status_row import (
    STATUS_CHIPS_POSITION_ABOVE,
    apply_status_chips_position,
    persist_status_chips_collapsed,
    poke_console_setting,
    resolve_status_chips_collapsed,
    resolve_status_chips_position,
)

# The Console controller classes themselves are deliberately NOT imported
# here: `..Console_Modules.wiring.build_console_controllers` constructs every
# one of them, so this module needs only the handful of helper symbols its own
# body actually uses. Tests that steer a controller patch it on the module
# that defines it (`..Console_Modules.dictation` and friends) rather than
# through this module's namespace -- see task-3023, which repointed them.
from ..Console_Modules.character_avatar_layout import (
    CharacterAvatarPrerender,
    build_character_avatar_prerender_job,
    fit_character_avatar_cell_box,
)
from ..Console_Modules.dictation import (
    ConsoleDictationEvent,
    ConsoleDictationLimitSignal,
    _VOICE_ACK_NOT_SENT,
    _VOICE_ACK_SESSION_CHANGED,
)
from ..Console_Modules.hands_free import (
    ConsoleHandsFreeSession,
)
from ..Console_Modules.agent import (
    CONSOLE_AGENT_CANCEL_ALL_ID,
    CONSOLE_AGENT_FLEET_SECTION_ID,
)
from ..Console_Modules.prompt_queue import (
    ConsolePromptDispatchStatus,
    ConsolePromptQueueRegion,
)
from ..Console_Modules.realtime import CONSOLE_REALTIME_CHIP_MESSAGES
from ..Console_Modules.dispatch_recovery import ConsoleDispatchRecoveryRegion
from ..Console_Modules.capture_policy_bindings import (
    build_inspector_capture_policy_wiring,
)
from ..Console_Modules.left_rail import (
    CONSOLE_DISCARD_DEFAULT_RETRY_ID,
    CONSOLE_DISMISS_DEFAULT_REFRESH_ID,
    CONSOLE_REFRESH_RUNNING_APP_ID,
    CONSOLE_RETRY_CONTEXT_SETTINGS_ID,
    CONSOLE_RETRY_DEFAULT_SAVE_ID,
    CONSOLE_RETRY_GENERATION_SETTINGS_ID,
    ConsoleLeftRail,
)
from ..Console_Modules.message import ConsoleMessageController
from ..Console_Modules.right_rail import ConsoleInspectorRail
from ..Console_Modules.provider_continuation_recovery import (
    ProviderContinuationTranscriptRegion as ConsoleTranscriptRegion,
)
from ..Console_Modules.retrieval import (
    sanitize_console_library_rag_query as _sanitize_console_library_rag_query,
    source_mentions_rag as _source_mentions_rag,
)
from ..Console_Modules.transcript import _ConsoleTranscriptReadingState
from ..Console_Modules.wiring import build_console_controllers
from ..Console_Modules import raw_cli as raw_cli_ui
from ..Console_Modules.session import (
    _has_selected_text,
    _is_empty_select_value,
)
from ...Chat.citation_trace_repository import ActiveCitationTraceState
from ...Chat.console_chat_controller import ConsoleChatController
from ...Chat.console_context_compaction import (
    EffectiveMemoryKind,
    complete_durable_units,
)
from ...Chat.console_runtime import ensure_console_runtime, leave_console_runtime
from ...Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
)
from ...Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsSurface,
    ConsoleSettingsSubmission,
    ConsoleSettingsTransfer,
)
from ...Chat.console_settings_durability import (
    ConsoleSettingsDurabilityOwner,
)
from ...Chat.console_settings_defaults import (
    ConsoleDefaultDurabilityState,
    ConsoleDefaultMutationIntent,
    ConsoleDefaultMutationOutcome,
    ConsoleDefaultRecoveryAction,
    ConsoleDefaultRecoveryRequest,
    ConsoleDefaultRuntimePublicationClaim,
    ConsoleDefaultSavePhase,
    abort_console_default_runtime_publication,
    apply_console_default_intent,
    build_console_default_intent,
    complete_console_default_runtime_publication,
    next_console_default_intent_generation,
    prepare_console_default_intent_reservation,
    prepare_console_default_runtime_publication,
    publish_console_default_runtime_if_current,
    refresh_console_runtime_after_saved_default,
    reserve_console_default_intent_generation,
)
from ...Chat.console_roleplay_identity import (
    ChatDisplayNameError,
    ConsoleMessagePresentation,
    ConsolePresentationContext,
    ConsoleTranscriptStyle,
    normalize_chat_display_name,
    normalize_console_transcript_style,
    resolve_console_message_presentation,
)
from ...Chat.prompt_history import PromptHistory
from ...Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostRow,
    ConsoleCostRowTotals,
    ConsoleCostState,
    TokenEstimateCache,
    build_cost_rows,
    build_cost_rows_totals,
    build_cost_snapshot,
    build_cost_state,
    console_cost_snapshot_messages,
    fingerprint_break_reason,
    token_estimate_signature,
)
from ...Chat.console_exchange_capture import ExchangeCapture, capture_from_storage
from ...LLM_Calls.pricing_catalog import get_pricing_catalog
from ...Event_Handlers.Chat_Events.chat_events_console_dictionaries import (
    console_attachable_dictionaries,
    console_attached_dictionaries,
    handle_console_dictionary_attach,
    handle_console_dictionary_detach,
)

# Reused rather than duplicated (task-6): the same conversation-scope
# resolution the native-Console chat entry point uses (task-5) -- session
# identity via `persisted_conversation_id`, `SessionScopeHolder` for
# unpersisted sessions, `EffectiveScope` state.
from ...Chat.rag_scope import RagScope
from ...Chat.console_command_grammar import (
    FEWER_PERMISSION_PROMPTS_COMMAND_HANDLER_ID,
    FEWER_PERMISSION_PROMPTS_COMMAND_NAME,
    GENERATE_IMAGE_COMMAND_HANDLER_ID,
    GENERATE_IMAGE_COMMAND_NAME,
    GENERATE_VIDEO_COMMAND_HANDLER_ID,
    GENERATE_VIDEO_COMMAND_NAME,
    KIND_COMMAND,
    KIND_NOT_COMMAND,
    KIND_UNKNOWN,
    PREFILL_COMMAND_HANDLER_ID,
    PREFILL_COMMAND_NAME,
    PROMPT_COMMAND_HANDLER_ID,
    PROMPT_COMMAND_NAME,
    RESEARCH_COMMAND_HANDLER_ID,
    RESEARCH_COMMAND_NAME,
    REWIND_COMMAND_HANDLER_ID,
    REWIND_COMMAND_NAME,
    SKILLS_COMMAND_HANDLER_ID,
    SKILLS_COMMAND_NAME,
    STREAM_VIDEO_COMMAND_HANDLER_ID,
    STREAM_VIDEO_COMMAND_NAME,
    SYSTEM_COMMAND_HANDLER_ID,
    SYSTEM_COMMAND_NAME,
    CommandParse,
    ConsoleCommandRegistry,
    default_console_registry,
)
from ...MCP.permission_prompt_reducer import format_permission_prompt_report
from ...Chat.console_prefill import (
    ACTION_CLEAR,
    ACTION_ERROR,
    ACTION_ONE_SHOT,
    ACTION_PIN,
    ACTION_STATUS,
    describe_prefill_preview,
    parse_prefill_args,
)
from ...Chat.console_generate_image import insert_style_token_into_draft
from ...Chat.console_side_chat import ConsoleSideChatService, render_prompt
from ...Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunStatus,
    FEEDBACK_ACTIVE_RUN_STATUSES,
    MessageAttachment,
    ConsoleWorkspaceContext,
    derive_console_session_title,
)
from ...UI.character_display_text import sanitize_character_display_label
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
    _estimate_tokens_locally,
    _summary_row_value,
    build_console_context_estimate,
    build_console_rail_system_line,
    build_default_console_session_settings,
    build_console_settings_readiness,
    build_console_settings_summary_state,
    build_target_default_console_session_settings,
    unsaved_console_endpoint_warning,
)
from ...Chat.console_chat_store import (
    MAX_PENDING_ATTACHMENTS,
    ConsoleChatSession,
    ConsoleChatStore,
    ConsoleRoleplayProjectionPersistencePlan,
    ConsoleRoleplayProjectionPersistenceResult,
    ConsoleSettingsComponent,
    ConsoleSettingsPolicyFailureLabel,
)
from ...Chat.console_provider_gateway import (
    DEFAULT_LLAMACPP_BASE_URL,
    normalize_llamacpp_base_url,
)
from ...Chat.console_provider_endpoints import (
    first_configured_endpoint,
    normalize_generic_endpoint_for_compare,
    safe_endpoint_display,
)
from ...Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
    ConsoleControlState,
    ConsoleDisplayRow,
    ConsoleInspectorState,
    ConsoleProjectInstructionState,
    ConsoleRetrievalScopeState,
    ConsoleStagedContextState,
    ConsoleStagedEvidenceStripState,
    build_console_evidence_display_state,
    build_console_staged_evidence_strip_state,
    coerce_non_negative_int,
    console_prompted_evidence_text,
    console_prompted_source_count,
    console_staged_source_count,
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
from ...Chat.provider_catalog import provider_display_name
from ...Chat.provider_readiness import get_provider_readiness, provider_config_key
from ...Chat.console_ephemeral import ACTION_SAVE_CHAT, blocked_reason
from ...Chat.console_live_work import (
    PENDING_LAUNCH_CARD_ID,
    SOURCE_READINESS_CARD_ID,
    ConsoleLiveWorkLaunch,
    ConsoleLiveWorkSourceReadinessState,
    ConsoleLiveWorkStatusCardState,
    console_setup_staged_receipt,
)
from ...Chat.console_command_suggestions import (
    completion_context_for_draft,
    suggestions_for_draft,
)
from ...Chat.console_image_view import (
    ConsoleImageRenderCache,
    ConsoleImageViewState,
    fit_image_cell_size,
    resolve_default_mode,
    resolve_show_character_avatar,
)
from ...Chat.console_paste_attach import (
    extract_dropped_path,
    grab_clipboard_image,
    looks_attachable,
)
from ...Chat.console_rail_state import (
    CONSOLE_INSPECTOR_AUTO_OPEN_MAX_COLUMNS,
    CONSOLE_INSPECTOR_AUTO_OPEN_MIN_COLUMNS,
    console_auto_open_would_evict_context,
    CONSOLE_INSPECTOR_MORE_DISCLOSURE_ID,
    CONSOLE_RAIL_LEFT_OPEN_EXPLICIT_KEY,
    CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS,
    CONSOLE_RAIL_SECTION_IDS,
    CONSOLE_RAIL_SHARED_LAYOUT_SCOPE,
    ConsoleRailPreferenceKey,
    ConsoleRailState,
    build_console_rail_preference_key,
    build_console_rail_state,
    coerce_console_rail_preferences,
    collect_prunable_console_rail_keys,
    console_context_reveal_preferences,
    console_rail_left_open_explicit,
    console_rail_width_band,
    normalize_console_rail_layout_scope,
    resolve_console_rail_priority,
    serialize_console_rail_preferences,
    serialize_console_rail_stored_preferences,
)
from ...config import (
    DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE,
    MAX_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    MIN_CONSOLE_PASTE_COLLAPSE_THRESHOLD,
    _get_effective_config_path,
    coerce_bool_setting,
    coerce_int_setting,
    delete_settings_from_cli_config,
    get_cli_providers_and_models,
    get_cli_setting,
    load_settings,
    save_setting_to_cli_config,
    save_settings_to_cli_config,
)
from ...Library.library_rag_service import LibraryRagSearchRequest
from ...Library.library_rag_state import (
    LIBRARY_RAG_FALLBACK_TOP_K,
    library_rag_profile_top_k,
)
from ...Constants import (
    CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID,
    LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE,
    TAB_SETTINGS,
    TAB_WATCHLISTS_COLLECTIONS,
    WATCHLISTS_NAV_CONTEXT_BRIEFING_ID,
)
from ...Utils.console_background_effects import (
    ConsoleBackgroundEffectSettings,
    normalize_console_background_effects,
)
from ...Utils.token_counter import estimate_tokens
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
from ...Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from ...Widgets.Chat_Widgets.skill_install_confirm_card import SkillInstallConfirmCard
from ...Widgets.Chat_Widgets.skill_script_confirm_card import SkillScriptConfirmCard
from ...Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from ...Widgets.Chat_Widgets.watchlists_operation_card import (
    WatchlistsOperationCard,
)
from ...Widgets.Console import (
    ConsoleBoundedSection,
    ConsoleCitationSourcesModal,
    ConsoleComposerBar,
    ConsoleComposerUndoHistory,
    ConsoleDraftStash,
    ConsoleControlBar,
    ConsoleSpeechControls,
    ConsoleRailHandle,
    ConsoleRetrievalScopeRow,
    ConsoleRunInspector,
    ConsoleSendAuthoritySummary,
    ConsoleSessionSurface,
    ConsoleSettingsModal,
    ConsoleSettingsSummary,
    ConsoleSetupModal,
    ConsoleStagedContextTray,
    ConsoleStagedEvidenceStrip,
    ConsoleTranscript,
    ConsoleWorkspaceContextTray,
    ConsoleWorkspaceTree,
    dismiss_message_more_menus,
    message_more_menus_on_screen,
    WorkspaceTreeConversationSelected,
    WorkspaceTreeExpansionChanged,
    WorkspaceTreeLoadMoreRequested,
    WorkspaceTreeRetryRequested,
    WorkspaceTreeStarRequested,
    WorkspaceTreeWorkspaceSelected,
)
from ...Widgets.Console.console_transcript import (
    derive_console_memory_banner_presentation,
)
from ...Widgets.Console.console_control_bar import (
    ConsoleAutoSpeakRetryRequested,
    ConsoleAutoSpeakResumeRequested,
)
from ...Widgets.Console.console_speech_controls import (
    ConsoleAutoSpeakChanged,
    ConsoleHandsFreeToggleRequested,
)
from ...Widgets.Console.console_settings_modal import ConsoleSettingsResult
from ...Widgets.Console.console_turn_file_card import ConsoleTurnFileCard
from ...Widgets.Console.console_context_controls import (
    ConsoleContextControlState,
    build_console_context_control_state,
)
from ...Widgets.Console.console_image_viewer_modal import (
    AvatarViewRequested,
    ConsoleImageViewerModal,
)
from ...Widgets.Console.console_agent_steering_bar import (
    ConsoleAgentSteeringBar,
    ConsoleAgentSteeringState,
)
from ...Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionState,
)
from ...Widgets.Console.console_command_popup import ConsoleCommandPopup
from ...Widgets.Console.console_review_notes_modal import ConsoleReviewNotesModal
from ...Widgets.Console.console_transcript import (
    ConsoleReviewNotesRequested,
    console_transcripts_on_screen,
)
from ...Widgets.Console.console_selection_menu import (
    ConsoleSelectionFeedbackRequested,
    ConsoleSelectionNoteRequested,
    ConsoleSelectionMenu,
    ConsoleSelectionQuoteRequested,
    ConsoleSideChatRequested,
    selection_menus_on_screen,
)
from ...Widgets.Console.console_side_chat_modal import ConsoleSideChatModal
from ...Widgets.Console import console_project_instructions as project_instruction_ui
from ...Widgets.Console.console_conversation_inspector import (
    TAB_COSTS,
    TAB_NEXT_SEND,
    ConsoleConversationInspector,
    InspectorTurn,
)
from ...Widgets.Console.console_citation_sources_modal import (
    selected_valid_evidence_ordinals,
)
from ...Widgets.Console.console_generation_card import generation_card_signature
from ...Widgets.Console.console_video_card import video_card_signature
from ...Widgets.Console.console_library_search_modal import (
    CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
    normalize_console_rag_source_types,
)
from ...Widgets.Console.console_status_chips import (
    ConsoleModelChip,
    ConsoleAssistantChip,
    ConsoleCostChip,
    ConsoleLibraryChip,
    ConsoleRunChip,
    ConsoleScopeChip,
    ConsoleSourcesChip,
    ConsoleStatusChips,
    ConsoleSystemPromptChip,
    ConsoleTemporaryChip,
    ConsoleToolsChip,
)
from ...Widgets.Console.console_retrieval_scope_row import (
    ROW_ID as CONSOLE_RETRIEVAL_SCOPE_ROW_ID,
)
from ...Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterChoice,
    ConsoleCharacterPickerModal,
)
from ...Widgets.Console.console_composer_menu_modal import (
    ACTION_GENERATE_CAPTION,
    ACTION_GENERATE_IMAGE,
    ACTION_ATTACH_CONTEXT,
    ACTION_IMPERSONATE,
    ACTION_IMPROVE_CURRENT_DRAFT,
    ACTION_PROMPTS,
    ACTION_SAVE_CHATBOOK,
    ACTION_UNDO_PROMPT_IMPROVEMENT,
    ConsoleComposerMenuModal,
)
from ...Widgets.Console.console_prompt_comparison_modal import (
    ConsolePromptComparisonModal,
    PromptComparisonResult,
)
from ...Widgets.Console.console_generate_image_modal import (
    ConsoleGenerateImageModal,
)
from ...Widgets.Console.console_scope_picker_modal import ConsoleScopePickerModal
from ...Widgets.Console.console_prompt_queue_modal import ConsolePromptQueueModal
from ...Widgets.Console.console_model_popover import (
    ConsoleModelPopover,
)
from ...Widgets.Console.console_style_picker_modal import ConsoleStylePickerModal
from ...Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_DETECTED_WORKBENCH_ACTION,
)
from ...Widgets.destination_rail import (
    DestinationRailSectionHeader,
)
from ...Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)
from ...Widgets.Console.console_rewind_modal import (
    ConsoleRewindChoice,
    ConsoleRewindModal,
    KIND_RESTORE,
    KIND_SUMMARIZE_FROM,
    KIND_SUMMARIZE_UP_TO,
    RewindPromptRow,
)
from ...Widgets.Console.console_workbench_state import build_console_workbench_state
from ...Workspaces.display_state import (
    ConsoleWorkspaceConversationRow,
    ConsoleWorkspaceContextState,
)
from ...Widgets.compact_model_bar import CompactModelBar
from ...Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker

FeedbackRequested = ConsoleSelectionFeedbackRequested
NoteRequested = ConsoleSelectionNoteRequested

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

logger = logger.bind(module="ChatScreen")
_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS = 8
Changed = Input.Changed
#: The Console's DEFAULT Library RAG source kinds, unchanged by RAG-44's
#: editable toggles: this same tuple is the settings modal's default
#: (`CONSOLE_RAG_DEFAULT_SOURCE_TYPES` -- one object, not a second copy),
#: so retrieval reads exactly what it always did until a user edits it.
#: This is a `source_types` selection (which KINDS of sources), NOT the
#: retrieval item scope (`EffectiveScope`, conversation ∩ workspace) that
#: `_resolve_console_library_rag_scope` resolves separately.
CONSOLE_LIBRARY_RAG_SOURCE_SCOPE = CONSOLE_RAG_DEFAULT_SOURCE_TYPES
CONSOLE_LIBRARY_RAG_QUERY_MAX_LENGTH = 2_000
CONSOLE_LIBRARY_RAG_FALLBACK_TOP_K = LIBRARY_RAG_FALLBACK_TOP_K
# TASK-346: below this terminal height the composer row was clipped out of
# existence at 97x30 (no input box, no warning). The visible header banner
# (title + purpose + Ready, ~5 rows) is pure chrome; dropping it below the
# threshold reclaims the rows the transcript+composer core loop needs.
# Measured live: composer clips at <=34 rows, fits at 35; the freed header
# lets it fit down to ~29-30 rows.
CONSOLE_COMPACT_HEIGHT_ROWS = 35
#: TASK-365: trailing affordance marking the clickable rail system-prompt line as
#: interactive (matches the ▸ the rail uses for its other actionable controls).
CONSOLE_RAIL_SYSTEM_EDIT_AFFORDANCE = "▸"
# Frame and focus-border constants live with their rendering helpers in
# `UI.Console_Modules.frame`; this screen imports only those helpers.
CONSOLE_START_HERE_COPY = ""
CONSOLE_ACTION_HINTS_COPY = ""
CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL = "Set up provider"
CONSOLE_PROVIDER_ACTION_ARROW = " ---------------------->"
NATIVE_CONSOLE_STATE_VERSION = "1.0"
# Roleplay P1h: bounds passed to `Chat_Dictionary_Lib.apply_active_chatdicts_to_text`
# for the native Console send-path applier (`_console_chat_dictionary_applier`).
_CHATDICT_MAX_TOKENS = 500
_CHATDICT_STRATEGY = "sorted_evenly"
_CONSOLE_RAIL_PREFERENCE_WRITE_LOCK = threading.Lock()
_RESUME_LOCAL_CONVERSATION_ID_MAX_LENGTH = 256
# Statuses during which the 0.2s transcript poll is actively ticking
# (see `_start_console_transcript_sync_timer`) -- also used by the
# sub-agent badge-count cache (Finding A) to decide whether a live run
# justifies an eager re-count, and by the selection feedback gating
# (Request changes / LGTM). Derived from the canonical
# ``FEEDBACK_ACTIVE_RUN_STATUSES`` (next to ``ConsoleRunStatus`` in
# ``Chat/console_chat_models.py``) so the feedback gating and this
# constant can never drift apart; other behaviors hang off this tuple,
# so its membership must stay exactly the canonical four active states.
# Sorted by value purely for a deterministic tuple; every consumer does
# membership checks only.
CONSOLE_ACTIVE_RUN_STATUSES: tuple[ConsoleRunStatus, ...] = tuple(
    sorted(FEEDBACK_ACTIVE_RUN_STATUSES, key=lambda status: status.value)
)
# Plan-B Task 7 Finding A: the conversation-browser `[N Sub-Agents]` badge
# count previously re-queried the DB once per visible row on every 0.2s
# poll tick. The batched replacement is still cheap to cache; this TTL is
# the fallback staleness bound when neither the row set changed nor a run
# is actively streaming (e.g. a sub-agent finished in a *different*
# Console session/tab).
CONSOLE_SUBAGENT_COUNTS_CACHE_TTL_SECONDS = 2.0
# TASK-251 (audit P1 B1): the persisted conversation-browser rows behind
# `_refresh_console_persisted_rows_cache` queries the DB per scope (global +
# every workspace) on every 0.2s poll tick -- measured 11-70ms/tick. Modeled
# directly on the sub-agent badge-count TTL cache above (same staleness
# bound, same "explicit invalidation is a nice-to-have, the TTL is the
# correctness backstop" philosophy).
CONSOLE_PERSISTED_ROWS_CACHE_TTL_SECONDS = 2.0
# Cost-ticker PR3 (task-5): the 0.2s transcript tick stops once a run leaves
# an active status (`_start_console_transcript_sync_timer`), so a WARM
# prompt cache that later goes EXPIRED on its own -- with no further sync
# call to notice -- needs its own slow repaint timer. 10s keeps the
# countdown's staleness bound well under the 300s cache TTL it is watching.
CONSOLE_COST_TTL_TICK_SECONDS = 10.0
# task-15470: every `Collapsible.Toggled` (plus expand-all/collapse-all/reset)
# used to reassign `sidebar_state` and have `watch_sidebar_state` open+parse+
# rewrite `ui_state.toml` synchronously on the event loop, per click. This
# debounce coalesces a burst of toggles into one write, dispatched off the
# loop (see `_flush_sidebar_state_after_debounce`); `on_unmount` force-flushes
# so a toggle immediately followed by quit is never lost.
SIDEBAR_STATE_SAVE_DEBOUNCE_SECONDS = 0.5
# P3c Task 3: the "Character" rail avatar box's fitted cell size (mirrors
# the transcript inline-image row's `fit_image_cell_size` usage, sized
# smaller for the rail's narrower column).
#: task-1682: pre-canned caption request. Pasted into the composer for
#: review rather than sent, matching how Generate Image hands back a
#: command the user can still edit.
CONSOLE_CAPTION_PROMPT = (
    "Describe the attached image in detail. Write one caption paragraph "
    "covering the subject, setting, action, and mood, then a single "
    "alt-text line under 125 characters."
)

CHARACTER_AVATAR_COLS = 16
CHARACTER_AVATAR_LINES = 8
#: task-1661: the rail avatar used to paint into the fixed box above no
#: matter how wide the rail was, leaving a ~50-column rail showing a
#: 16-column portrait pinned to the corner of an unsized (1fr) holder.
#: The box is now derived from the rail's live width, clamped so a very
#: tall portrait cannot swallow the whole rail.
CHARACTER_AVATAR_MAX_COLS = 44
CHARACTER_AVATAR_MAX_LINES = 22
# task-1537: remote inline images. Only the most recent assistant replies are
# scanned for image links (older history never triggers fetches), and one
# fetched body is capped well below the render cache's decode ceiling.
REMOTE_IMAGE_SCAN_WINDOW = 20
REMOTE_IMAGE_MAX_BYTES = 8 * 1024 * 1024
CONSOLE_FOCUS_REGISTRY = WorkbenchFocusRegistry(
    (
        "console-left-rail",
        "console-transcript-surface",
        "console-right-rail",
        "console-native-composer",
    )
)

#: TASK-24604: a pane whose COLLAPSED form is a different widget. F6 skips
#: panes that are not displayed, and the right rail ships collapsed, so the
#: Inspector pane silently dropped out of the cycle in its default state and
#: the only route in was a mouse click on the handle.
#:
#: The handle is deliberately NOT added to ``pane_order``: it is already
#: modelled as a between-panes widget in ``CONSOLE_FOCUS_PANE_FOR_WIDGET``
#: (TASK-2154.11), and ``_console_workbench_focus_id_for_widget`` checks
#: ``pane_order`` FIRST -- so making it a pane would make focus on the handle
#: report the handle rather than its logical pane, changing F6's behaviour
#: from inside the collapsed rail. Instead the pane stays one entry and
#: gains a collapsed stand-in for both the visibility test and the focus
#: target.
#: Scoped to the RIGHT rail deliberately. The left rail has the same
#: structural gap -- collapsed, it also drops out of the F6 cycle -- but it
#: defaults OPEN, so it is not the shipping-state defect TASK-24604 is about,
#: and `test_console_f6_cycles_visible_workbench_panes` encodes the current
#: left-rail behaviour as a contract. Widening this map to the left rail
#: changes which pane F6 lands on first and breaks that contract, which is a
#: separate decision with its own task, not a side effect of this one.
CONSOLE_PANE_COLLAPSED_STAND_IN = {
    "console-right-rail": "console-inspector-rail-handle",
}
CONSOLE_FOCUS_TARGETS_BY_PANE = {
    "console-left-rail": ("console-context-rail-collapse", "console-left-rail"),
    "console-transcript-surface": (
        "console-native-transcript",
        "console-transcript-surface",
    ),
    "console-right-rail": ("console-inspector-rail-collapse", "console-right-rail"),
    "console-native-composer": ("console-native-composer",),
}
#: TASK-2154.11 (AC-02): Tab/Shift+Tab cycle WITHIN the focused widget's
#: Console region instead of walking the app-level focus chain (which dragged
#: the tour through all 15 nav buttons between the composer and the control
#: bar). Each tuple is one region's root ids; the union of their subtrees is
#: the region's Tab cycle. F6/Shift+F6 remain the way to move BETWEEN panes.
#: The control bar pairs with the composer (the Console's two command
#: surfaces) so its buttons stay keyboard-reachable; the status chips pair
#: with the transcript region they annotate; each rail handle pairs with its
#: rail so a collapsed rail's open button still cycles sanely. Hidden panes
#: (display:none rails/handles) drop out of Textual's focus chain on their
#: own, so no explicit hidden-pane handling is needed here.
CONSOLE_TAB_REGIONS: tuple[tuple[str, ...], ...] = (
    (
        "console-workbench-header",
        "console-control-bar",
        "console-native-composer",
    ),
    ("console-context-rail-handle", "console-left-rail"),
    ("console-transcript-region", "console-status-chips"),
    ("console-inspector-rail-handle", "console-right-rail"),
)
#: TASK-2154.11: widgets that live BETWEEN the four F6 panes in the shell --
#: the control bar, the status chips, and each collapsed rail's handle -- map
#: to their logical pane (the same pairing as CONSOLE_TAB_REGIONS) so F6 from
#: one of them continues the pane cycle from that pane instead of restarting
#: at the first pane ("focus not in a pane" -> visible[0]).
CONSOLE_FOCUS_PANE_FOR_WIDGET = {
    "console-workbench-header": "console-native-composer",
    "console-control-bar": "console-native-composer",
    "console-status-chips": "console-transcript-surface",
    "console-context-rail-handle": "console-left-rail",
    "console-inspector-rail-handle": "console-right-rail",
}


#: `ConsoleDictationEvent`, `ConsoleDictationLimitSignal` and
#: `ConsoleStreamingDictationSession` moved to `UI/Console_Modules/
#: dictation.py` (wave-1 console decomposition, task 5), imported back
#: above for the `@on(...)` decorators below and for the handful of
#: tests that still construct them via `chat_screen_module.<name>`.
#: `CONSOLE_HANDS_FREE_DEGRADED_MESSAGE` and `ConsoleHandsFreeSession` moved
#: to `UI/Console_Modules/hands_free.py` (wave-2 console decomposition, task
#: 1) along with the rest of the V3 pipeline hands-free loop; `Console
#: HandsFreeController`/`ConsoleHandsFreeSession` are imported back above
#: for the same two reasons dictation's types are.


#: Maximum default wait for the screen-owned roleplay drain during unmount.
#: The immutable writer may outlive this deadline, but never the screen-bound
#: coordinator or its store/session presentation state.
CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS = 0.25


class _ConsoleTurnUndoUnavailableError(RuntimeError):
    """A card-level Undo All cannot proceed without touching disk."""


class _ConsoleTurnUndoAmbiguousError(_ConsoleTurnUndoUnavailableError):
    """One run has multiple snapshot windows for the same canonical root."""


class _ConsoleTurnUndoStalePreflightError(_ConsoleTurnUndoUnavailableError):
    """New edited-since files appeared after the user saw the confirmation."""


@dataclass(frozen=True, slots=True)
class _ConsoleTurnUndoPlan:
    """Exact rows, paths, and warnings behind one card confirmation."""

    rows_paths: tuple[tuple[dict, tuple[str, ...]], ...]
    root_labels: tuple[str, ...]
    edited_since: tuple[str, ...]
    edited_keys: frozenset[tuple[str, str]]
    disk_state: tuple[tuple[str, str, str], ...]
    total_files: int


def _console_turn_undo_run_active(provider: Any, roots: tuple[str, ...]) -> bool:
    """Read the provider's root-aware run probe with legacy fallback."""
    root_probe = getattr(provider, "run_active_for_root", None)
    if callable(root_probe):
        return any(bool(root_probe(root)) for root in roots)
    probe = getattr(provider, "run_active", None)
    return bool(probe()) if callable(probe) else False


def _prepare_console_turn_undo(provider: Any, run_id: str) -> _ConsoleTurnUndoPlan:
    """Build one safe whole-turn revert plan.

    This synchronous function performs provider/git reads and is therefore
    called only through ``asyncio.to_thread`` by the mounted screen.
    Duplicate-root windows are refused before any diff or preflight read: the
    compact card does not expose enough baseline ordering to consent safely.
    """
    turn = provider.turn_for_run(run_id)
    if turn is None or not turn.rows:
        raise _ConsoleTurnUndoUnavailableError(
            "Undo All is unavailable because this turn has no change snapshots."
        )

    seen_roots: set[str] = set()
    for row in turn.rows:
        root = str(row.get("root") or "")
        if not root:
            raise _ConsoleTurnUndoUnavailableError(
                "Undo All is unavailable because a change snapshot has no workspace."
            )
        canonical = os.path.normcase(os.path.normpath(root))
        if canonical in seen_roots:
            raise _ConsoleTurnUndoAmbiguousError(
                "This turn has multiple change windows for the same workspace."
            )
        seen_roots.add(canonical)
    roots = tuple(str(row["root"]) for row in turn.rows)
    if _console_turn_undo_run_active(provider, roots):
        raise _ConsoleTurnUndoUnavailableError(
            "A run is active on this workspace — finish or stop the run first."
        )

    rows_paths: list[tuple[dict, tuple[str, ...]]] = []
    for row in turn.rows:
        tracking_error = str(row.get("tracking_error") or "").strip()
        if tracking_error:
            raise _ConsoleTurnUndoUnavailableError(
                f"Undo All is unavailable: {tracking_error}"
            )
        changes = provider.changed_files(row)
        paths = tuple(dict.fromkeys(str(change.path) for change in changes))
        if paths:
            rows_paths.append((row, paths))
    if not rows_paths:
        raise _ConsoleTurnUndoUnavailableError(
            "Undo All is unavailable because no changed files can be restored."
        )

    multi_root = len(rows_paths) > 1
    leaf_labels = [
        Path(str(row["root"])).name or str(row["root"]) for row, _ in rows_paths
    ]
    root_labels = tuple(
        str(row["root"]) if leaf_labels.count(label) > 1 else label
        for (row, _paths), label in zip(rows_paths, leaf_labels)
    )
    edited_keys: set[tuple[str, str]] = set()
    edited_labels: list[str] = []
    disk_state: list[tuple[str, str, str]] = []
    for (row, paths), root_label in zip(rows_paths, root_labels):
        root = str(row["root"])
        preflight = provider.preflight_revert(row, list(paths))
        disk_state.extend(
            (root, str(path), str(fingerprint))
            for path, fingerprint in preflight.disk_state.items()
        )
        for path in preflight.edited_since:
            key = (root, str(path))
            if key in edited_keys:
                continue
            edited_keys.add(key)
            edited_labels.append(f"{root_label}/{path}" if multi_root else str(path))
    return _ConsoleTurnUndoPlan(
        rows_paths=tuple(rows_paths),
        root_labels=root_labels,
        edited_since=tuple(edited_labels),
        edited_keys=frozenset(edited_keys),
        disk_state=tuple(disk_state),
        total_files=sum(len(paths) for _row, paths in rows_paths),
    )


def _apply_console_turn_undo(provider: Any, plan: _ConsoleTurnUndoPlan) -> list:
    """Revalidate a confirmed plan, then invoke the existing revert engine."""
    from tldw_chatbook.Workspaces.change_revert import RevertOutcome

    roots = tuple(str(row["root"]) for row, _paths in plan.rows_paths)
    if _console_turn_undo_run_active(provider, roots):
        raise _ConsoleTurnUndoUnavailableError(
            "A run is active on this workspace — finish or stop the run first."
        )

    current_edited: set[tuple[str, str]] = set()
    current_disk_state: list[tuple[str, str, str]] = []
    for row, paths in plan.rows_paths:
        root = str(row["root"])
        preflight = provider.preflight_revert(row, list(paths))
        current_edited.update((root, str(path)) for path in preflight.edited_since)
        current_disk_state.extend(
            (root, str(path), str(fingerprint))
            for path, fingerprint in preflight.disk_state.items()
        )
    if (
        current_edited - plan.edited_keys
        or tuple(current_disk_state) != plan.disk_state
    ):
        raise _ConsoleTurnUndoStalePreflightError(
            "Files changed while the confirmation was open. Open Review and try again."
        )

    outcomes: list = []
    multi_root = len(plan.rows_paths) > 1
    for index, ((row, paths), root_label) in enumerate(
        zip(plan.rows_paths, plan.root_labels)
    ):
        try:
            row_outcomes = provider.revert(row, list(paths))
        except Exception as exc:  # noqa: BLE001 - retain earlier disk outcomes
            error = f"not processed: {str(exc)[:240]}"
            for (_pending_row, pending_paths), pending_label in zip(
                plan.rows_paths[index:], plan.root_labels[index:]
            ):
                outcomes.extend(
                    RevertOutcome(
                        path=(f"{pending_label}/{path}" if multi_root else str(path)),
                        ok=False,
                        error=error,
                    )
                    for path in pending_paths
                )
            break
        outcomes.extend(
            replace(outcome, path=f"{root_label}/{outcome.path}")
            if multi_root
            else outcome
            for outcome in row_outcomes
        )
    return outcomes


def _consume_console_roleplay_writer_completion(
    task: asyncio.Task[ConsoleRoleplayProjectionPersistenceResult | None],
    *,
    session_id: str,
    generation: int,
) -> None:
    """Consume an app-owned serializer result without retaining its screen."""
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        logger.exception(
            "App-owned Console roleplay projection writer failed "
            "(session_id={}, generation={}).",
            session_id,
            generation,
        )


def _consume_console_roleplay_repair_for_current_screen(
    app_instance: Any,
) -> None:
    """Ask the app's current Console owner to consume a repair marker."""
    try:
        current_screen = app_instance.screen
    except Exception:  # noqa: BLE001 - lifecycle repair is best-effort
        return
    consume = getattr(
        current_screen,
        "_consume_pending_console_roleplay_repair",
        None,
    )
    if callable(consume):
        consume()


CONSOLE_WORKBENCH_SHORTCUTS = (
    ("F6", "next pane"),
    ("Shift+F6", "previous pane"),
    ("F1", "help"),
    ("Enter", "send / queue"),
    ("Y", "trace"),
    ("Ctrl+K", "switch session"),
    ("Ctrl+T", "new tab"),
    # TASK-24604: the rail's own accelerator has to be advertised here or it
    # is only discoverable by reading the source. This is the footer/F1
    # vocabulary the Console teaches from.
    ("Alt+I", "inspect"),
    ("Ctrl+P", "palette"),
)

#: TASK-24703: `AppFooterStatus` degrades by keeping a PREFIX of the hint
#: list, so a hint near the end is the first thing dropped as width falls.
#: `("Alt+I", "inspect")` sat 9th of 11 and vanished at exactly 80 columns --
#: the width where the rail's edge handle is also hidden, making Alt+I the
#: ONLY route in. Below the single-pane threshold it is therefore promoted to
#: the front instead of merely being present, the same treatment the focus
#: toggle already gets for the same reason.
CONSOLE_WORKBENCH_SHORTCUTS_SINGLE_PANE = (
    ("Alt+I", "inspect"),
    *(pair for pair in CONSOLE_WORKBENCH_SHORTCUTS if pair[0] != "Alt+I"),
)

#: TASK-2154.8 (FR-06): while the first-run setup modal locks the composer,
#: advertising "Enter send" is a lie -- Enter activates the focused setup-card
#: action instead. The blocked variant hides the send hint and names the real
#: action. `_register_console_footer_shortcuts` swaps between the two.
CONSOLE_WORKBENCH_SHORTCUTS_SETUP_BLOCKED = tuple(
    ("Enter", "continue setup") if pair == ("Enter", "send / queue") else pair
    for pair in CONSOLE_WORKBENCH_SHORTCUTS
)

#: TASK-362: the full Console keyboard vocabulary for the F1 help panel, grouped
#: by surface. The flat CONSOLE_WORKBENCH_SHORTCUTS above stays the compact
#: footer set; the transcript j/k/c/e/r keys, F2, Shift+Enter and Alt+M were
#: previously undiscoverable anywhere in the app.
CONSOLE_WORKBENCH_SHORTCUT_GROUPS = (
    (
        "Panes",
        (
            ("F6", "next pane"),
            ("Shift+F6", "previous pane"),
            # TASK-2154.11 (AC-02): Tab is pane-local now; F6 is the way out.
            ("Tab / Shift+Tab", "cycle within the current pane"),
            ("Escape", "return to the composer"),
        ),
    ),
    (
        "Transcript",
        (
            ("j / k", "select next / previous message"),
            ("Enter", "show the selected message's actions"),
            ("c", "copy the selected message"),
            ("e", "edit the selected message"),
            ("f", "fork from the selected message"),
            ("r", "regenerate the selected message"),
            ("More…", "Save as, feedback, delete, and diagnostics"),
            ("Escape", "clear the selection"),
        ),
    ),
    (
        "Composer",
        (
            ("Enter", "send now or queue after an accepted turn"),
            ("Queue shelf", "manage, pause, resume, and recover queued prompts"),
            ("Ctrl+J", "insert a newline (works in any terminal)"),
            ("Shift+Enter", "insert a newline (where the terminal delivers it)"),
            ("Ctrl+Z", "undo the last draft edit"),
            (
                "Ctrl+Shift+Z / Ctrl+Y",
                "redo (Ctrl+Y also works where the terminal can't send Ctrl+Shift+Z)",
            ),
            ("Alt+V", "paste an image from the clipboard"),
            (
                "Attach file",
                f"attach files — up to {MAX_PENDING_ATTACHMENTS} per message",
            ),
            ("Paste / drop path", "paste or drop a file path to attach it"),
            ("Ctrl+K", "switch session"),
            ("Ctrl+T", "new tab"),
        ),
    ),
    (
        "Global & modals",
        (
            ("F1", "help"),
            ("Ctrl+P", "command palette"),
            ("Alt+M", "quick change model"),
            ("F2", "rename a session (in the Ctrl+K switcher)"),
        ),
    ),
    (
        # Fleet-UX expert review F2 (task-1232): Alt+W and Alt+1..9 are real
        # BINDINGS (see this screen's BINDINGS list) but the footer is a
        # single-line, non-wrapping Static already at ~120 chars for its 7
        # entries -- adding all of these there would just push more of an
        # already-overflowing line further off tested narrow-terminal widths
        # (this suite runs Console at 80 columns). Help is the reachable
        # surface for the full set; Ctrl+T/Ctrl+K are repeated here (they
        # also appear above) because this group is what a user scanning for
        # "how do multiple tabs work" will read top-to-bottom.
        "Agents & fleet",
        (
            ("Alt+W", "switch workspace"),
            ("Alt+1..9", "jump to tab 1-9"),
            ("Ctrl+T", "new tab (new agent)"),
            ("Ctrl+K", "switch session"),
        ),
    ),
    (
        # TASK-2154.20 (AC-03): on default macOS terminals Option is not Meta,
        # so Alt+M/W/V/1..9 type composed characters instead of firing these
        # bindings (Alt+H already moved to Ctrl+Shift+H for exactly this). The
        # reliable non-Alt paths: every Alt action has a Ctrl+P palette entry,
        # and Ctrl+K's session switcher covers the Alt+1..9 tab jumps.
        "macOS terminals",
        (
            ("Alt chords", "type composed characters when Option is not Meta"),
            ("Ctrl+P", "palette lists every Alt action — the reliable path"),
            ("Ctrl+K", "session switching (covers the Alt+1..9 tab jumps)"),
        ),
    ),
)

#: Fleet-UX expert review F4 (task-1233 also references this legend --
#: keep it a single, clearly-marked string so that task can find/reuse it
#: verbatim rather than re-deriving the copy).
#:
#: TWIN CONSTANT -- see `CONSOLE_RUN_MARKER_MEANINGS` in
#: `tldw_chatbook/Chat/console_chat_models.py` (task-1233's marker-aware
#: tab/sidebar tooltips). That dict deliberately uses its OWN fuller
#: in-context phrasing ("agent running"/"waiting for approval"/
#: "finished — unseen") rather than this line's shorter per-glyph words
#: ("running"/"needs approval"/"finished"/"failed") -- a deliberate
#: register split (task-1233 review round 1: a compact scannable legend
#: line vs. a specific in-context tooltip sentence), not drift. If you
#: change what a glyph MEANS, update both.
CONSOLE_FLEET_MARKER_LEGEND = (
    "Status markers: ● running · ◆ needs approval · ✓ finished · ✗ failed "
    "· ◈ sub-agent ended in background "
    "— clears once you visit that tab. Qn is the unsent prompt count."
)


def _console_workbench_agents_notes(max_parallel_runs: int) -> tuple[str, ...]:
    """Build the F1 Help "Agents" section lines (fleet-UX F2, task-1232).

    A function, not a module constant: the parallel-run cap is user-
    adjustable (``console.max_parallel_runs``, Settings > Console Behavior)
    and this must read the LIVE value, not the default baked in at import
    time.

    Args:
        max_parallel_runs: The live ``ConsoleChatController.max_parallel_runs``
            cap to quote in the second line.

    Returns:
        Ordered prose lines for the help panel's "Agents" notes block.
    """
    # task-1232 round 1 (Minor b): cap=1 is a supported floored value
    # (MIN_CONSOLE_MAX_PARALLEL_RUNS), so "1 runs" must not ship.
    run_noun = "run" if max_parallel_runs == 1 else "runs"
    return (
        "Each Console tab runs its own agent; a run keeps going in the "
        "background while you're on another tab.",
        f"Up to {max_parallel_runs} {run_noun} in parallel "
        "(change in Settings > Console Behavior).",
        "Built-in tools ask before running; a background session that "
        "needs approval parks with a ◆ badge and a toast.",
        "Accepted turns can hold up to 10 queued prompts per tab; use the "
        "queue shelf to manage or pause them.",
        CONSOLE_FLEET_MARKER_LEGEND,
        # PR3a-2 Task 4: post-3a-1 this line's old claim ("cancels any
        # runs still in progress") was half false -- background
        # sub-agents that outlived their turn keep running.
        "Leaving Console cancels replies still streaming -- you'll be "
        "asked first; background sub-agents keep running and notify you "
        "when they finish.",
    )


def character_avatar_box(available_cols: int) -> tuple[int, int]:
    """Avatar box (cols, lines) for a rail of ``available_cols`` width.

    task-1661: keeps the portrait proportional to the rail instead of a
    fixed 16x8 corner thumb, while clamping so a tall image cannot claim
    the entire rail. Lines are half the columns because terminal cells are
    roughly twice as tall as wide, which keeps the box near-square on
    screen.

    Args:
        available_cols: The holder's usable width in columns; values below
            the historical minimum fall back to it (layout not settled).

    Returns:
        ``(cols, lines)`` for `fit_image_cell_size`.
    """
    cols = max(CHARACTER_AVATAR_COLS, min(CHARACTER_AVATAR_MAX_COLS, available_cols))
    lines = max(
        CHARACTER_AVATAR_LINES, min(CHARACTER_AVATAR_MAX_LINES, round(cols / 2))
    )
    return cols, lines


def _character_avatar_fallback_renderable(
    pil: Any,
    *,
    box_cols: int = CHARACTER_AVATAR_COLS,
    box_lines: int = CHARACTER_AVATAR_LINES,
    monochrome: bool = False,
):
    """Bake the rail avatar's non-graphics renderable from a PIL image.

    Quadrant mosaic (2x2 subpixels per cell) at the rail's fitted box --
    double the horizontal detail of the previous half-block Pixels build
    with the same universal Block Elements font coverage.

    TASK-22221: the renderer itself now lives in ``character_avatar_layout``
    so the rail's off-loop prerender and this inline fallback are the same
    code, not two copies that could drift.

    Args:
        pil: The decoded portrait.
        box_cols: Target width in columns (task-1661: rail-derived).
        box_lines: Target height in lines.
        monochrome: Carry the image in shade GLYPHS rather than background
            colour. The coloured mosaic is spaces styled ``on rgb(...)``, so
            with colour unavailable it renders as a blank box -- the avatar
            does not degrade, it disappears. Textual switches the whole app
            to monochrome when ``NO_COLOR`` is set, which is one confirmed
            way a user sees no portrait at all.
    """
    from ..Console_Modules.character_avatar_layout import (
        render_character_avatar_mosaic,
    )

    return render_character_avatar_mosaic(
        pil, box_cols, box_lines, monochrome=monochrome
    )


def _is_personas_preview_handoff(payload: ChatHandoffPayload) -> bool:
    """Return whether a handoff is a Personas "Open in Console" preview transcript.

    These are staged by ``PersonasPreviewController.open_in_console`` with the
    workbench's fixed ``source="personas"`` identity and a
    ``"preview-conversation"`` item type. They carry no ``start_chat`` intent,
    so they miss the character-session path (task-427); task-428 routes them
    into a fresh, dedicated Console conversation rather than reusing (and
    polluting) whatever conversation happens to be active. The predicate is
    deliberately narrow: Personas "Start Chat" (``"{kind}-card"``) and "Attach"
    handoffs do not match.

    Args:
        payload: A handoff staged into the native Console.

    Returns:
        ``True`` only for a Personas "Open in Console" preview-conversation
        handoff; ``False`` for every other source/item type.
    """
    return (
        str(payload.source or "").strip() == "personas"
        and str(payload.item_type or "").strip() == "preview-conversation"
    )


def _console_library_rag_source_scope(screen: Any) -> tuple[str, ...]:
    """Return a Console screen's stored Library RAG source kinds (RAG-44).

    The ONE read seam for `_console_library_rag_source_types`, so every
    caller (the readiness-card label, the settings modal, the retrieval
    request) sees the same normalized tuple, and a screen that predates
    the attribute -- or never ran ``__init__`` -- still retrieves over the
    unchanged default instead of raising.

    Args:
        screen: The Console `ChatScreen` (or a stand-in) holding the state.

    Returns:
        The normalized, non-empty selection of Library source kinds.
    """
    return normalize_console_rag_source_types(
        getattr(screen, "_console_library_rag_source_types", None)
    )


#: RAG-43: above this length a composer draft reads as a paste/attachment,
#: not a retrieval question -- well under ``CONSOLE_LIBRARY_RAG_QUERY_MAX_
#: LENGTH`` (2000, a safety ceiling for an explicitly-typed query, not a
#: "does this look like a question" heuristic for an implicit prefill).
CONSOLE_LIBRARY_RAG_DRAFT_PREFILL_MAX_LENGTH = 200


def _console_library_rag_profile_top_k() -> int:
    """Return the ACTIVE RAG profile's result count (TASK-406/TASK-3170).

    Both Library RAG entry points on the Console -- the RAG chip's manual
    Run and the opt-in send-path auto-retrieve -- must honor the profile
    the user configured, the same way task-5 made the Library service
    honor its search mode -- a hardcoded count would silently ignore a
    profile tuned for more (or fewer) results. This is the one place both
    call sites read that count from, so they can never drift apart again.

    TASK-15020/B3 gave the Library window the same behavior, and this is now
    a DELEGATION to that seam (`library_rag_state.library_rag_profile_top_k`)
    rather than a second copy of the resolution: the Console chip and the
    Library window are two views of one retrieval stack, and a twin would be
    free to drift the moment either side changed. The delegation is kept as
    a named function because both Console call sites (and their tests) reach
    the profile through this name.

    Returns:
        The profile's ``search.default_top_k`` when it resolves to a
        positive integer, else the shared Library RAG fallback -- a
        broken/absent profile must degrade to retrieving, not to raising
        inside a send.
    """
    return library_rag_profile_top_k()


def _console_screen_is_torn_down(screen: Any) -> bool:
    """Whether ``screen``'s message pump has begun closing.

    task-15860 (cross-suite leak). Three deliberate choices:

    * **Not ``is_mounted``.** Measured on the live crash this exists to
      stop: the removed screen's ``ConsoleSessionSurface`` still reported
      ``is_mounted=True`` while its own pump reported ``is_running=False``
      and every child was already gone, so a mount check would have waved
      through the very tick that then raised ``NoMatches``.
    * **Not ``is_running``.** That is also False *before* a pump starts,
      which would silently no-op every harness that drives the sync tick
      on a hand-built, never-mounted ``ChatScreen``. ``_closing`` /
      ``_closed`` are the pair Textual sets when a pump is taken down
      (``MessagePump._close_messages`` sets ``_closing`` as its first
      statement, before any child comes down) and the pair Textual itself
      reads for ``is_parent_active``.
    * **A module function, not a method.** ``MagicMock(spec=ChatScreen)``
      is a common fixture here, and a spec'd mock auto-answers every
      *method* on the class -- truthily. As a method this predicate
      therefore reported "torn down" for every mocked screen and turned
      three live-screen `test_ui_responsiveness` tests red (measured: 15
      passed at the baseline, 3 failed with the method form). Read off
      the raw flags instead: they are set in ``MessagePump.__init__``, so
      they are absent from ``dir(ChatScreen)``, and a spec'd mock -- like
      a never-mounted screen -- correctly reads as LIVE.
    """
    return bool(getattr(screen, "_closing", False) or getattr(screen, "_closed", False))


def _console_inspector_turn_preview(content: Any) -> str:
    """Best-effort short text preview for one Conversation Inspector
    Costs-tab turn row (task-8 review finding 5).

    ``ConsoleChatMessage.content`` is declared ``str``, but a multimodal
    (structured, OpenAI-style content-block list) message is not
    guaranteed to have been coerced to text by the time it reaches here --
    several other modules in this codebase (``Chat/Chat_Functions.py``,
    ``console_provider_gateway.py``) carry their own ``isinstance(content,
    str)`` guards for exactly this reason. Slicing a list with ``[:60]``
    would silently yield up to 60 LIST ELEMENTS, not characters -- not a
    preview, and not obviously wrong-looking in a diff either. Falls back
    to the first text block's text (bounded to 60 chars, matching the str
    path), or ``""`` when nothing text-shaped is found -- never a
    fabricated summary.
    """
    if isinstance(content, str):
        return content[:60]
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    return text[:60]
    return ""


def _build_console_inspector_exchanges_loader(
    messages_by_native_id: Mapping[str, Any],
    db_accessor: Callable[[], Any],
    abandoned_run_tags_for: Callable[[str], AbstractSet[str]] | None = None,
) -> Callable[[str], Awaitable[list[tuple[ExchangeCapture, bool]]]]:
    """Build the Costs-tab ``exchanges_loader`` for
    ``ConsoleConversationInspector`` (task-8, extended task-9).

    A standalone function rather than a method-local closure specifically
    so it is unit-testable without mounting a ``ChatScreen`` (review
    finding 6) -- pure extraction, no behavior change from the closure
    this replaced in ``ChatScreen._build_console_inspector_cost_data``.

    Args:
        messages_by_native_id: ``ConsoleChatMessage.id`` -> the matching
            in-memory message, for the native-first check.
        db_accessor: Zero-arg callable returning the ChaChaNotes DB handle
            (or ``None``). Called lazily -- only on the DB-fallback path,
            never for a message resolved natively or with no persisted id
            -- so an ephemeral session never touches the DB at all.
        abandoned_run_tags_for: Optional ``native_message_id ->
            {run_tag, ...}`` lookup (task-9; ``ConsoleChatStore.
            abandoned_exchange_run_tags`` in production) used ONLY on the
            native-capture path to resolve each capture's real
            ``abandoned`` flag. Defaults to ``None``, which preserves the
            task-8 behavior of reporting ``abandoned=False`` for every
            native capture -- kept optional (rather than required) so the
            existing unit tests in
            ``Tests/UI/test_chat_screen_console_inspector_loader.py``,
            which construct this loader with just the first two
            positional args, are unaffected.

    Returns:
        An async ``native_message_id -> [(capture, abandoned), ...]``
        callable (see ``console_conversation_inspector``'s module
        docstring for the pair contract and the ordering caveat -- callers
        must NOT trust the returned order, only ``(created_at, seq)``).
        Prefers ``message.exchanges`` (native, in-memory captures resolve
        ``abandoned`` via ``abandoned_run_tags_for`` when supplied, else
        always ``False``) and only falls back to a threaded
        ``get_message_exchanges`` + ``capture_from_blob`` read when there
        is no native capture AND the message has a
        ``persisted_message_id`` (an ephemeral session has neither, so it
        returns ``[]`` without any DB call). A single corrupt/undecodable
        blob is logged and skipped -- not fatal to the rest of the turn's
        captures.
    """

    async def _exchanges_loader(
        native_message_id: str,
    ) -> list[tuple[ExchangeCapture, bool]]:
        message = messages_by_native_id.get(native_message_id)
        if message is not None and message.exchanges:
            # Native captures win when present -- they are fresher than
            # whatever was last flushed to the DB.
            abandoned_tags: AbstractSet[str] = (
                abandoned_run_tags_for(native_message_id)
                if abandoned_run_tags_for is not None
                else frozenset()
            )
            return [
                (capture, capture.run_tag in abandoned_tags)
                for capture in message.exchanges
            ]
        persisted_id = message.persisted_message_id if message is not None else None
        if not persisted_id:
            return []

        def _read() -> list[tuple[ExchangeCapture, bool]]:
            db = db_accessor()
            if db is None:
                return []
            out: list[tuple[ExchangeCapture, bool]] = []
            for db_row in db.get_message_exchanges(persisted_id):
                try:
                    out.append(
                        (
                            capture_from_storage(
                                db_row["capture_blob"],
                                db_row.get("capture_detail", "safe"),
                            ),
                            bool(db_row.get("abandoned", False)),
                        )
                    )
                except Exception as exc:
                    # No traceback (review finding M8): this frame holds
                    # `capture_blob` (raw compressed bytes) and, mid-loop,
                    # already-decoded ExchangeCapture payloads in `out` --
                    # loguru's diagnose formatter would annotate the
                    # failing source line's names with their values across
                    # the whole frame chain. The Exchange tab's own
                    # handlers (console_conversation_inspector.py's
                    # ``_load_turn_captures``) deliberately refuse
                    # tracebacks for the identical reason; this brings the
                    # loader's own decode failure to the same standard.
                    # type(exc).__name__ plus the message id is enough to
                    # diagnose and retry.
                    logger.warning(
                        f"exchange_blob_decode_failed: persisted_id="
                        f"{persisted_id!r}: {type(exc).__name__}"
                    )
            return out

        return await asyncio.to_thread(_read)

    return _exchanges_loader


class _ControllerState:
    """Read/write compatibility for state moved to a wired controller."""

    def __init__(self, owner_name: str, state_name: str) -> None:
        self._owner_name = owner_name
        self._state_name = state_name

    def _owner(self, instance: object) -> object:
        try:
            return object.__getattribute__(instance, self._owner_name)
        except AttributeError as exc:
            raise RuntimeError("controller not wired") from exc

    def __get__(self, instance: object, owner: type | None = None) -> object:
        if instance is None:
            return self
        return getattr(self._owner(instance), self._state_name)

    def __set__(self, instance: object, value: object) -> None:
        setattr(self._owner(instance), self._state_name, value)


class ChatScreen(BaseAppScreen):
    """
    Chat screen with comprehensive state management.

    This screen preserves all chat state including tabs, messages,
    input text, and UI preferences when navigating away and returning.
    """

    _imagegen_inflight_sessions = _ControllerState(
        "_image", "_imagegen_inflight_sessions"
    )
    _imagegen_inflight_message_ids = _ControllerState(
        "_image", "_imagegen_inflight_message_ids"
    )
    _console_h3_ui_generations = _ControllerState(
        "_image", "_console_h3_ui_generations"
    )
    _console_videogen_inflight = _ControllerState(
        "_video", "_console_videogen_inflight"
    )
    _console_videogen_cancels = _ControllerState("_video", "_console_videogen_cancels")
    _console_video_store = _ControllerState("_video", "_console_video_store")
    _pending_video_artifacts = _ControllerState("_video", "_pending_video_artifacts")
    _pending_video_artifacts_closed = _ControllerState(
        "_video", "_pending_video_artifacts_closed"
    )
    _pending_video_operation_cancels = _ControllerState(
        "_video", "_pending_video_operation_cancels"
    )
    _pending_video_active_operations = _ControllerState(
        "_video", "_pending_video_active_operations"
    )
    _pending_video_deferred_closes = _ControllerState(
        "_video", "_pending_video_deferred_closes"
    )
    _console_realtime = _ControllerState("_realtime", "session")
    _console_realtime_close_worker = _ControllerState("_realtime", "close_worker")
    _console_annotation_loaded_conversation = _ControllerState(
        "_review_selection", "annotation_loaded_conversation"
    )
    _console_annotation_previews = _ControllerState(
        "_review_selection", "annotation_previews"
    )
    _console_selection_feedback_inflight = _ControllerState(
        "_review_selection", "selection_feedback_inflight"
    )
    _console_persisted_rows_cache = _ControllerState(
        "_workspace", "_console_persisted_rows_cache"
    )
    _console_persisted_rows_cache_key = _ControllerState(
        "_workspace", "_console_persisted_rows_cache_key"
    )
    _console_persisted_rows_cache_at = _ControllerState(
        "_workspace", "_console_persisted_rows_cache_at"
    )
    _console_conversation_browser_query = _ControllerState(
        "_workspace", "_console_conversation_browser_query"
    )
    _console_conversation_browser_search_timer = _ControllerState(
        "_workspace", "_console_conversation_browser_search_timer"
    )
    _console_conversation_browser_search_token = _ControllerState(
        "_workspace", "_console_conversation_browser_search_token"
    )
    _console_conversation_browser_rows = _ControllerState(
        "_workspace", "_console_conversation_browser_rows"
    )
    _console_conversation_browser_total = _ControllerState(
        "_workspace", "_console_conversation_browser_total"
    )
    _console_conversation_browser_error = _ControllerState(
        "_workspace", "_console_conversation_browser_error"
    )
    _console_retrieval_scope_cache = _ControllerState(
        "_retrieval", "_console_retrieval_scope_cache"
    )
    _console_effective_scope_cache = _ControllerState(
        "_retrieval", "_console_effective_scope_cache"
    )
    _active_dictionaries_summary = _ControllerState(
        "_retrieval", "_active_dictionaries_summary"
    )
    _last_console_dictionary_scope_ids = _ControllerState(
        "_retrieval", "_last_console_dictionary_scope_ids"
    )
    _active_world_books_summary = _ControllerState(
        "_retrieval", "_active_world_books_summary"
    )
    _last_console_world_book_scope_ids = _ControllerState(
        "_retrieval", "_last_console_world_book_scope_ids"
    )
    _console_skill_candidates = _ControllerState("_skill", "_console_skill_candidates")
    _active_character_avatar = _ControllerState(
        "_character", "_active_character_avatar"
    )
    _active_character_avatar_name = _ControllerState(
        "_character", "_active_character_avatar_name"
    )
    _last_console_avatar_scope = _ControllerState(
        "_character", "_last_console_avatar_scope"
    )
    _console_expression_spec_cache = _ControllerState(
        "_character", "_console_expression_spec_cache"
    )

    # TASK-352: Textual docks notification toasts bottom-right by default —
    # directly over the Console composer's Send/Attach/Save cluster and the
    # staged-chip strip — and toasts intercept clicks, so a click aimed at those
    # controls during a ~5s toast dismisses the toast instead of pressing the
    # button. Dock the Console screen's toast rack to the TOP-right so feedback
    # never obscures, or swallows clicks aimed at, the composer's controls.
    # Kept in BUNDLED_CSS (not the CSS_PATH bundle) so it applies in both the
    # real app and ConsolidatedCSSApp-based test harnesses, which load the
    # generated widget-defaults sheet but not necessarily the full CSS_PATH
    # bundle.
    BUNDLED_CSS = """
    ChatScreen ToastRack {
        dock: top;
        align: right top;
        margin-top: 1;
        margin-bottom: 0;
    }
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
        Binding("f1", "show_workbench_help", "Help", show=True),
        Binding(
            "f6", "focus_next_workbench_pane", "Next pane", show=True, priority=True
        ),
        Binding(
            "shift+f6",
            "focus_previous_workbench_pane",
            "Previous pane",
            show=True,
            priority=True,
        ),
        Binding("ctrl+k", "open_console_session_switcher", "Switch session", show=True),
        # task-5 (console trajectory view): single-letter htop-style launch
        # key per ADR-031. 'y', NOT 'j': the focused transcript consumes
        # j/k for next/previous-message selection (console_transcript.py
        # on_key), which would make the advertised footer hint a lie in
        # exactly the surface a trajectory reader comes from. The footer
        # hint is registered via CONSOLE_WORKBENCH_SHORTCUTS like the rest
        # of the Console vocabulary.
        Binding("y", "open_trajectory_view", "Trace", show=True),
        Binding("alt+m", "open_console_model_popover", "Model", show=True),
        Binding("alt+w", "open_console_workspace_switcher", "Workspace", show=True),
        # TASK-24604: the Inspect rail ships CLOSED (unlike the left rail) and
        # F6 filters out non-displayed panes, so before this binding the only
        # way into the product's primary inspection surface was a mouse click
        # on a 9-character handle. TASK-24600: below 84 columns that handle is
        # hidden too, which made collapsing the rail there a one-way trip --
        # this binding is the way back, and it is why it must not itself be
        # gated on the rail being displayed.
        Binding("alt+i", "toggle_console_inspector_rail", "Inspect", show=True),
        Binding("alt+v", "paste_clipboard_image", "Paste image", show=True),
        # ctrl+shift+h, not alt+h: on macOS terminals "alt" is the Option
        # key, which types a composed character (˙) unless the profile
        # opts into Option-as-Meta -- the first live gate hit exactly that.
        # ctrl+shift+<letter> follows the existing ctrl+shift+p precedent.
        Binding(
            "ctrl+shift+h",
            "toggle_console_hands_free",
            "Hands-free",
            show=True,
        ),
        Binding("ctrl+shift+p", "view_chat_context", "View context", show=True),
        # Task-5 review I2: Esc exits the hands-free loop from ANY point --
        # a `priority=True` binding is what actually delivers "any point":
        # `on_key`'s own hands-free branch (hoisted above the focus gate,
        # see `on_key`) still only ever fires once a Key event bubbles up
        # from the focused widget, and a widget with its OWN escape
        # handling (the transcript's clear-selection, a modal's dismiss)
        # can consume/stop it before it ever reaches that far. Priority
        # bindings are resolved by the App BEFORE normal bubbling starts,
        # so this wins regardless of what currently holds focus.
        # `check_action` below gates it to hands-free-active only, so it
        # never shadows `expand_collapsed_console_composer` (also
        # priority=True, for the composer-collapsed case) outside the loop.
        Binding(
            "escape",
            "exit_console_hands_free",
            "Exit hands-free",
            show=False,
            priority=True,
        ),
        Binding(
            "escape",
            "expand_collapsed_console_composer",
            "Composer",
            show=False,
            priority=True,
        ),
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

    def check_action(
        self,
        action: str,
        parameters: tuple[object, ...],
    ) -> bool | None:
        """Return whether a named screen action is currently available.

        Args:
            action: Textual action name being checked.
            parameters: Parsed positional parameters for the action.

        Returns:
            The collapse-action availability, or the superclass result for
            every other action.
        """
        if action == "expand_collapsed_console_composer":
            return (
                self._console_composer_collapsed
                and not self._console_setup_modal_blocking()
            )
        if action == "exit_console_hands_free":
            # One-line delegation (wave-2 console decomposition, task 1).
            # See `ConsoleHandsFreeController.console_hands_free_exit_
            # available` for the real implementation.
            return self._hands_free.console_hands_free_exit_available()
        return super().check_action(action, parameters)

    def action_exit_console_hands_free(self) -> None:
        """Priority Esc: exit the hands-free loop from any point (task-5
        review I2) -- see `check_action`'s gate and the `BINDINGS` entry's
        docstring-comment for why this needs to be `priority=True` rather
        than relying on `on_key`'s own (bubbling-order) branch alone.

        TASK-24417: an open slash-command popup claims Escape FIRST -- the
        user asked the overlay to go away, not the loop to end (the popup
        also used to survive the exit, floating over a dead loop). This
        action is the priority binding's destination (it fires before key
        bubbling), so the claim lives here, not only in `on_key`.

        One-line delegation (wave-2 console decomposition, task 1); the
        `action_*` method has to stay on this class for Textual's action
        dispatch to find it. See `ConsoleHandsFreeController.action_exit_
        console_hands_free` for the real implementation, which covers BOTH
        engines (V4 task 5): "Esc from any point in the loop" is a promise
        the docs make about hands-free, not about one engine's
        implementation of it.
        """
        if self._dismiss_console_command_popup():
            return
        self._hands_free.action_exit_console_hands_free()

    def action_expand_collapsed_console_composer(self) -> None:
        """Expand the hidden Console composer and return keyboard focus to it.

        An open slash-command popup swallows Escape first and is dismissed
        instead, leaving the collapsed composer untouched.
        """
        if self._console_setup_modal_blocking():
            return
        if self._dismiss_console_command_popup():
            return
        self._set_console_composer_collapsed(False)

    def action_focus_next(self) -> None:
        """Move focus to the next widget within the focused Console region.

        An open slash-command popup claims Tab first: the highlighted
        suggestion is accepted into the draft instead of moving focus.
        While the Console setup modal is blocking the workbench, this keeps
        focus cycling within the modal's own focusables instead of letting
        Tab tunnel into rail/transcript/composer controls hidden beneath it.

        TASK-2154.11 (AC-02): past those two traps, Tab cycles within the
        focused widget's Console region (``CONSOLE_TAB_REGIONS``) rather than
        the whole app focus chain, so the tour no longer crosses all 15
        app-nav buttons mid-Console; F6/Shift+F6 move between panes. Focus
        sitting in app chrome (a clicked nav button) keeps the default
        app-wide chain so the nav bar stays keyboard-traversable once
        entered; with nothing focused at all, Tab lands on the composer --
        the Console's keyboard home base.
        """
        if self._accept_console_command_popup():
            return
        if self._focus_console_setup_modal_if_blocking():
            return
        focused = self.app.focused
        if focused is None:
            self._focus_console_workbench_target("console-native-composer")
            return
        selector = self._console_tab_region_selector(focused)
        if selector is None:
            self.focus_next()
            return
        self.focus_next(selector)

    def action_focus_previous(self) -> None:
        """Move focus to the previous widget within the focused Console region.

        Mirrors ``action_focus_next`` for the reverse direction.
        """
        if self._focus_console_setup_modal_if_blocking():
            return
        focused = self.app.focused
        if focused is None:
            self._focus_console_workbench_target("console-native-composer")
            return
        selector = self._console_tab_region_selector(focused)
        if selector is None:
            self.focus_previous()
            return
        self.focus_previous(selector)

    async def handle_model_catalog_refreshed(self, event) -> None:
        """Re-merge options when startup refresh updated the active provider.

        The legacy sidebar Selects (``#chat-api-provider``/``#chat-api-model``)
        this used to re-merge into lived only on the retired
        ``ChatWindowEnhanced`` (never mounted -- ``self.chat_window`` is
        permanently ``None``). Kept as a no-op call target: ``app.py``'s
        ``ModelCatalogRefreshed`` handler duck-types onto this method name via
        ``forward_model_catalog_refreshed`` for whichever screen is on the
        stack, so the method must keep existing and keep accepting the event.

        Args:
            event: The ``ModelCatalogRefreshed`` event forwarded by ``app.py``.
        """
        return

    @on(Input.Changed, "#console-workspace-conversation-search")
    def on_console_workspace_conversation_search_changed(self, event: Changed) -> None:
        event.stop()
        query = str(event.value or "")
        disabled = bool(getattr(getattr(event, "input", None), "disabled", False))
        self._workspace.transition_browser_search(query, disabled)

    @on(Input.Changed, "#console-workspace-search")
    def on_console_workspace_search_changed(self, event: Changed) -> None:
        """Keep Workspaces search independent from flat Conversations search."""

        event.stop()
        self._workspace.transition_workspace_tree_search(
            str(event.value or ""),
            bool(getattr(getattr(event, "input", None), "disabled", False)),
        )

    @on(WorkspaceTreeConversationSelected)
    @on(WorkspaceTreeRetryRequested)
    @on(WorkspaceTreeLoadMoreRequested)
    @on(WorkspaceTreeStarRequested)
    @on(WorkspaceTreeExpansionChanged)
    @on(WorkspaceTreeWorkspaceSelected)
    async def on_workspace_tree_action(
        self,
        event: (
            WorkspaceTreeConversationSelected
            | WorkspaceTreeRetryRequested
            | WorkspaceTreeLoadMoreRequested
            | WorkspaceTreeStarRequested
            | WorkspaceTreeExpansionChanged
            | WorkspaceTreeWorkspaceSelected
        ),
    ) -> None:
        event.stop()
        if isinstance(event, WorkspaceTreeWorkspaceSelected):
            self._workspace.activate_workspace_id(event.workspace_id)
        elif isinstance(event, WorkspaceTreeExpansionChanged):
            self._workspace.transition_workspace_tree_expansion(
                event.workspace_id,
                expanded=event.expanded,
            )
        elif isinstance(event, WorkspaceTreeStarRequested):
            self._workspace._toggle_console_conversation_star(
                event.conversation_id,
                starred=event.starred,
                conversation_title="",
            )
        elif isinstance(event, WorkspaceTreeLoadMoreRequested):
            self._workspace.request_next_workspace_tree_page(event.workspace_id)
        elif isinstance(event, WorkspaceTreeRetryRequested):
            self.run_worker(
                self._workspace.retry_workspace_tree_page(event.workspace_id),
                group=f"console-workspace-page-{event.workspace_id}",
                exclusive=False,
            )
        else:
            await self._workspace.open_console_workspace_conversation(
                event.conversation_id,
                target_workspace_id=event.workspace_id,
            )

    @on(Select.Changed, "#compact-api-provider")
    def on_console_compact_provider_changed(self, event: Select.Changed) -> None:
        """Mirror native compact provider changes into Console-owned labels.

        Coalesced through task-3010's seam rather than syncing directly:
        these two Select watchers ARE the mount-window burst that seam
        exists to absorb. Instrumenting one screen push showed seven
        control-bar syncs, four of them from here -- ``#compact-api-model``
        fires three times and ``#compact-api-provider`` once while the
        compact selects are populated, and the first three results are
        overwritten before anything paints. Coalescing them takes that push
        to three syncs.

        Both handlers must move together: leaving one direct while the
        other is coalesced fails ``test_requested_sync_still_executes``,
        because the direct call satisfies the pending request's work
        without clearing its scheduled flag.
        """
        if not _is_empty_select_value(event.value):
            self._console_control_provider = str(event.value)
        self._request_console_control_bar_sync()

    @on(Select.Changed, "#compact-api-model")
    def on_console_compact_model_changed(self, event: Select.Changed) -> None:
        """Mirror native compact model changes into Console-owned labels.

        Coalesced -- see ``on_console_compact_provider_changed`` above for
        why, and why the two cannot be split.
        """
        if not _is_empty_select_value(event.value):
            self._console_control_model = str(event.value)
        self._request_console_control_bar_sync()

    @on(ConsoleLeftRail.SectionToggled)
    def on_console_left_rail_section_toggled(
        self, message: ConsoleLeftRail.SectionToggled
    ) -> None:
        """Handle a left-rail section toggle button press.

        The rail catches its own toggle buttons and stops the underlying
        ``Button.Pressed`` (see ``ConsoleLeftRail.on_button_pressed``) --
        formerly this matched ``RAIL_SECTION_TOGGLE_PREFIX`` directly inside
        this screen's own ``on_button_pressed`` if-chain. The actual open/
        close decision, persistence, and Inspector-rail interaction stay
        here unchanged: they reach beyond the rail's own DOM.
        """
        self._toggle_console_rail_section(
            message.section_id,
            next_open=message.opened,
        )

    @on(ConsoleLeftRail.ReactionPickerRequested)
    async def _console_reaction_picker_requested(
        self, message: ConsoleLeftRail.ReactionPickerRequested
    ) -> None:
        message.stop()
        await self._session._open_console_reaction_picker()

    @on(ConsoleInspectorSection.RowActivated)
    def on_console_agent_fleet_row_activated(
        self, message: ConsoleInspectorSection.RowActivated
    ) -> None:
        """Drill into the sub-agent a fleet row was clicked for (TASK-4).

        Only the Agent rail's own fleet mini-section is a
        ``ConsoleInspectorSection`` today (``CONSOLE_AGENT_FLEET_SECTION_
        ID``) -- the ``section_id`` guard exists so a future sibling
        section built on the same component (Changes/Sources/Workspace,
        spec §7's stated direction) never gets routed here by accident.
        Replaces the old cycling click handler (id-string matching on
        ``#console-agent-section-subagents`` in ``on_click``) -- a row now
        posts its own typed message with its own stable ``row_id``,
        mirroring ``console_status_chips.py``'s chip-activation messages
        rather than matching by DOM id.
        """
        if message.section_id != CONSOLE_AGENT_FLEET_SECTION_ID:
            return
        message.stop()
        self._agent._drill_into_console_agent_subagent(message.row_id)

    @on(ConsoleInspectorSection.RowCancelRequested)
    def on_console_agent_fleet_row_cancel_requested(
        self, message: ConsoleInspectorSection.RowCancelRequested
    ) -> None:
        """Cooperatively cancel a fleet row's child (PR2b Task 5).

        Same ``section_id`` guard as the drill-in handler above, for the
        same reason. Routes through ``ConsoleAgentController._cancel_
        console_agent_fleet_row``, which itself routes through the
        EXISTING cancellation path (``AgentService.cancel_subagent`` ->
        ``_cancel_fleet_handles``) -- no second mechanism.

        A burst of cancels (the user cancelling more than one row in quick
        succession) each requests a coalesced fleet-section resync rather
        than a synchronous one, so N cancels in the same UI tick still
        produce exactly one ``_sync_console_agent_section`` run (task-5
        coalescing) instead of N redundant ones.
        """
        if message.section_id != CONSOLE_AGENT_FLEET_SECTION_ID:
            return
        message.stop()
        cancelled = self._agent._cancel_console_agent_fleet_row(message.row_id)
        if cancelled:
            self._request_console_agent_fleet_sync()

    @on(Button.Pressed, f"#{CONSOLE_AGENT_CANCEL_ALL_ID}")
    def on_console_agent_cancel_all(self, event: Button.Pressed) -> None:
        """Cancel every live child of the conversation (PR3b Task 5).

        Same delegation grammar as the per-row cancel handler above: the
        controller routes to ``ConsoleAgentBridge.cancel_all_subagents``,
        which walks the current service plus the retained survivor owners
        and cancels each live handle through the EXISTING per-handle
        cancel/approval-revoke path -- no second mechanism. A non-zero
        count requests one coalesced fleet resync so the rows and the
        affordance's own visibility reflect the stop on the next tick.

        Args:
            event: The press on the agent section's "Cancel all agents"
                button (``CONSOLE_AGENT_CANCEL_ALL_ID``); stopped here --
                the delegation is this handler's whole job.
        """
        event.stop()
        if self._agent._cancel_all_console_agents():
            self._request_console_agent_fleet_sync()

    @on(ConsoleAgentSteeringBar.SteeringSubmitted)
    def on_console_agent_steering_submitted(
        self, message: ConsoleAgentSteeringBar.SteeringSubmitted
    ) -> None:
        """Queue USER steering for the drilled-in child (PR3b Task 3).

        Same delegation grammar as the per-row cancel handler above: the
        controller routes to ``ConsoleAgentBridge.steer_subagent`` (which
        owns resolution + boundary validation), and a successful post
        requests one coalesced fleet-section resync so the queued-count
        line reflects the new entry on the next tick. The bar's draft is
        cleared only on a QUEUED submit (Qodo audit minor batch) -- a
        refusal keeps the user's text in the input for retry rather than
        destroying it with nothing delivered.
        """
        message.stop()
        queued = self._agent._steer_console_agent_drilldown_child(
            message.target_id, message.text
        )
        if queued:
            self._request_console_agent_fleet_sync()
            try:
                self.query_one(ConsoleAgentSteeringBar).clear_draft()
            except Exception:  # noqa: BLE001 -- a mid-recompose bar is fine
                pass

    @on(Button.Pressed, "#console-context-rail-collapse")
    def on_console_context_rail_collapse(self, event: Button.Pressed) -> None:
        """Collapse the Console context rail and persist the preference."""
        event.stop()
        self._set_console_rail_preference(left_open=False)

    @on(Button.Pressed, "#console-context-rail-open")
    def on_console_context_rail_open(self, event: Button.Pressed) -> None:
        """Open the Console context rail and persist the preference."""
        event.stop()
        available_columns = self._console_rail_available_columns()
        rail_state = self._current_console_rail_state(
            available_columns=available_columns
        )
        preference_changes = console_context_reveal_preferences(
            rail_state, available_columns
        )
        self._set_console_rail_preference(
            left_open=preference_changes["left_open"],
            right_open=preference_changes.get("right_open"),
        )

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

    async def action_toggle_console_inspector_rail(self) -> None:
        """Toggle the Inspect rail and put focus where the user just went.

        TASK-24604 / TASK-24600. Two properties matter more than the toggle:

        * It is **not gated on the rail being displayed.** Below
          ``CONSOLE_SINGLE_PANE_COLUMNS`` both handles hide, so a rail
          collapsed at 80 columns left nothing on screen referencing the
          Inspector and only a terminal resize brought it back. This action
          is the way back, which it can only be if it works while the rail
          and its handle are both invisible.
        * **Opening moves focus into the rail.** An accelerator that reveals
          a pane and leaves the caret in the composer makes the user reach
          for the mouse anyway, which is the problem being fixed.
        """

        if self._focus_console_setup_modal_if_blocking():
            return
        opening = not self._is_console_widget_displayed("console-right-rail")
        self._set_console_rail_preference(right_open=opening)
        if not opening:
            # Closing: do not strand focus on a widget that is about to be
            # hidden -- Textual would drop it to the screen and the next Tab
            # would restart from the top of the shell.
            self._focus_console_workbench_target("console-native-composer")
            return
        # The rail mounts asynchronously; focus once it is actually there.
        #
        # TASK-24703: target the pinned authority summary, NOT the pane's
        # default target. `CONSOLE_FOCUS_TARGETS_BY_PANE` resolves
        # `console-right-rail` to the collapse button first, so opening with
        # this shortcut used to land the caret on the control that closes the
        # pane the user just opened -- one stray Enter and they were back
        # where they started. The authority summary is focusable, is not
        # destructive, and is the thing they came to read. Falls back to the
        # pane's normal targets if it is not mounted.
        self.call_after_refresh(self._focus_opened_console_inspector_rail)

    def _focus_opened_console_inspector_rail(self) -> None:
        """Place the caret on the rail's most useful non-destructive target."""

        try:
            summary = self.query_one("#console-send-authority-summary")
        except QueryError:
            self._focus_console_workbench_target("console-right-rail")
            return
        if not self._is_console_widget_displayed("console-send-authority-summary"):
            self._focus_console_workbench_target("console-right-rail")
            return
        summary.can_focus = True
        summary.focus()
        self._last_console_workbench_focus_id = "console-right-rail"

    @on(ConsoleRunInspector.MoreToggled)
    def on_console_inspector_more_toggled(
        self, event: ConsoleRunInspector.MoreToggled
    ) -> None:
        """Persist a deliberate Inspector More disclosure change.

        Args:
            event: The disclosure event carrying the requested open state.
        """

        event.stop()
        self._set_console_rail_preference(
            section_updates={CONSOLE_INSPECTOR_MORE_DISCLOSURE_ID: event.open},
            notify_on_failure=False,
        )

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

    @on(Button.Pressed, "#console-inspector-worldbooks-attach")
    def on_console_inspector_worldbooks_attach(self, event: Button.Pressed) -> None:
        """Open the attach-world-book picker for the active Console conversation."""
        event.stop()
        if self._console_worldbook_dialog_active:
            return
        self._console_worldbook_dialog_active = True
        self.run_worker(self._console_worldbook_attach_worker(), group="console-io")

    @on(Button.Pressed, "#console-inspector-worldbooks-detach")
    def on_console_inspector_worldbooks_detach(self, event: Button.Pressed) -> None:
        """Open the detach-world-book picker for the active Console conversation."""
        event.stop()
        if self._console_worldbook_dialog_active:
            return
        self._console_worldbook_dialog_active = True
        self.run_worker(self._console_worldbook_detach_worker(), group="console-io")

    @staticmethod
    def _console_settings_initial_draft(
        settings: ConsoleSessionSettings,
        context_policy: ConsoleContextPolicyOverrides,
        *,
        exposed_fields: frozenset[str],
    ) -> ConsoleSettingsDraftState:
        """Build one process-local transaction from an exact live snapshot."""

        return ConsoleSettingsDraftState(
            settings=settings,
            context_policy_overrides=context_policy,
            field_drafts=tuple(
                ConsoleSettingsFieldDraft(
                    name=name,
                    effective_value=getattr(settings, name),
                    profile_override=getattr(settings, name),
                    provenance=ConsoleSettingsFieldProvenance.INHERITED,
                    dirty=False,
                )
                for name in sorted(exposed_fields)
            ),
            model_drafts=(),
            endpoint_draft=None,
        )

    def _console_default_durability_state(self) -> ConsoleDefaultDurabilityState:
        """Return the single app-lifetime default recovery holder."""

        state = getattr(
            self.app_instance,
            "console_default_durability_state",
            None,
        )
        if not isinstance(state, ConsoleDefaultDurabilityState):
            state = ConsoleDefaultDurabilityState()
            self.app_instance.console_default_durability_state = state
        if (
            type(
                getattr(self.app_instance, "console_new_chat_default_generation", None)
            )
            is not int
        ):
            self.app_instance.console_new_chat_default_generation = 0
        return state

    def _console_default_readiness(
        self,
        provider: str,
        model: str | None,
    ) -> ConsoleSettingsReadiness:
        """Resolve future-chat readiness through the target default chain."""

        app_config = self._provider_readiness_app_config()
        settings = build_target_default_console_session_settings(
            app_config,
            provider,
            model,
        )
        return build_console_settings_readiness(settings, app_config=app_config)

    def _commit_console_settings_submission_live(
        self,
        submission: ConsoleSettingsSubmission,
    ):
        """Revalidate/rebase and commit one exact-origin submission live."""

        owner = ChatScreen._console_settings_durability_owner(self)
        admission = owner.try_acquire()
        if admission is None:
            raise ValueError("Application is closing; nothing applied.")
        controller = self._ensure_console_chat_controller()
        try:
            exposed_fields = frozenset(
                field.name for field in submission.draft.field_drafts
            )
            rebased = controller.rebase_console_settings_draft(
                submission.draft,
                provider=submission.draft.settings.provider,
                model=submission.draft.settings.model,
                app_config=self._provider_readiness_app_config(),
                exposed_fields=exposed_fields,
            )
            if submission.surface is ConsoleSettingsSurface.QUICK_POPOVER:
                # Rebasing restores the config-owned endpoint draft. Quick
                # settings may use that endpoint live, but must never turn it
                # into a default-persistence intent.
                rebased = replace(
                    rebased,
                    model_drafts=tuple(
                        replace(model_draft, endpoint_draft=None)
                        for model_draft in rebased.model_drafts
                    ),
                    endpoint_draft=None,
                )
            live_commit = (
                self._ensure_console_chat_store().commit_console_settings_live(
                    replace(submission, draft=rebased)
                )
            )
        except BaseException:
            owner.release(admission)
            raise
        return replace(live_commit, durability_admission=admission)

    def _console_settings_durability_owner(self) -> ConsoleSettingsDurabilityOwner:
        """Return the app-owned settings admission and task registry."""

        app_instance = self.app_instance
        owner = getattr(app_instance, "console_settings_durability_owner", None)
        if not isinstance(owner, ConsoleSettingsDurabilityOwner):
            owner = ConsoleSettingsDurabilityOwner()
            app_instance.console_settings_durability_owner = owner
            app_instance.console_settings_durability_tasks = owner.tasks
        return owner

    def _reserve_console_default_intent(
        self,
        submission: ConsoleSettingsSubmission,
    ) -> ConsoleDefaultMutationIntent:
        """Synchronously reserve an intent for non-production callers/tests."""

        if submission.action is ConsoleSettingsAction.APPLY_TO_CHAT:
            raise ValueError("Apply to chat does not create a default intent")
        state = self._console_default_durability_state()
        generation = next_console_default_intent_generation(
            state.newest_intent_generation
        )
        for _attempt in range(_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS):
            intent = build_console_default_intent(
                generation=generation,
                action=submission.action,
                provider_config_key=provider_config_key(
                    submission.draft.settings.provider
                ),
                literal_model_id=str(submission.draft.settings.model or ""),
                field_drafts=submission.draft.field_drafts,
                field_mask=submission.default_field_mask,
                endpoint=submission.draft.endpoint_draft,
            )
            if reserve_console_default_intent_generation(
                intent,
                pending_runtime_publisher=(
                    self._accept_console_default_runtime_publication
                ),
            ):
                break
            generation = next_console_default_intent_generation(generation)
        else:
            raise RuntimeError("Console default reservation changed repeatedly")
        self.app_instance.console_default_durability_state = (
            ConsoleDefaultDurabilityState(newest_intent_generation=generation)
        )
        return intent

    async def _reserve_console_default_intent_off_event_loop(
        self,
        submission: ConsoleSettingsSubmission,
    ) -> ConsoleDefaultMutationIntent:
        """Serialize one reservation with every app-level claim publication."""

        if submission.action is ConsoleSettingsAction.APPLY_TO_CHAT:
            raise ValueError("Apply to chat does not create a default intent")
        async with ChatScreen._console_default_operation_lock(self):
            return await ChatScreen._reserve_console_default_intent_locked(
                self,
                submission,
            )

    def _console_default_operation_lock(self) -> asyncio.Lock:
        """Return the one app-lifetime serializer for claim UI operations."""

        app_instance = self.app_instance
        operation_lock = getattr(
            app_instance,
            "console_default_operation_lock",
            None,
        )
        if not isinstance(operation_lock, asyncio.Lock):
            operation_lock = asyncio.Lock()
            app_instance.console_default_operation_lock = operation_lock
        return operation_lock

    async def _reserve_console_default_intent_locked(
        self,
        submission: ConsoleSettingsSubmission,
    ) -> ConsoleDefaultMutationIntent:
        """Reserve while the caller owns the non-reentrant operation lock."""

        app_instance = self.app_instance
        state = self._console_default_durability_state()
        generation = await asyncio.to_thread(
            next_console_default_intent_generation,
            state.newest_intent_generation,
        )

        for _attempt in range(_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS):
            intent = build_console_default_intent(
                generation=generation,
                action=submission.action,
                provider_config_key=provider_config_key(
                    submission.draft.settings.provider
                ),
                literal_model_id=str(submission.draft.settings.model or ""),
                field_drafts=submission.draft.field_drafts,
                field_mask=submission.default_field_mask,
                endpoint=submission.draft.endpoint_draft,
            )
            (
                preparation,
                cancelled,
            ) = await ChatScreen._run_console_default_worker_settled(
                self,
                prepare_console_default_intent_reservation,
                intent,
            )
            if preparation.reserved:
                app_instance.console_default_durability_state = (
                    ConsoleDefaultDurabilityState(newest_intent_generation=generation)
                )
                if cancelled:
                    raise asyncio.CancelledError
                return intent
            claim = preparation.predecessor_claim
            if claim is None:
                if cancelled:
                    raise asyncio.CancelledError
                generation = await asyncio.to_thread(
                    next_console_default_intent_generation,
                    generation,
                )
                continue
            if cancelled:
                await ChatScreen._run_console_default_worker_settled(
                    self,
                    abort_console_default_runtime_publication,
                    claim,
                )
                raise asyncio.CancelledError
            try:
                published = self._accept_console_default_runtime_publication(
                    claim.intent_generation,
                    claim.action,
                    claim.settings_view,
                )
            except Exception:
                published = False
            if not published:
                await ChatScreen._run_console_default_worker_settled(
                    self,
                    abort_console_default_runtime_publication,
                    claim,
                )
                raise RuntimeError("Pending default publication was rejected")
            (
                completed,
                cancelled,
            ) = await ChatScreen._run_console_default_worker_settled(
                self,
                complete_console_default_runtime_publication,
                claim,
                successor_intent=intent,
            )
            if completed:
                app_instance.console_default_durability_state = (
                    ConsoleDefaultDurabilityState(newest_intent_generation=generation)
                )
                if cancelled:
                    raise asyncio.CancelledError
                return intent
            if cancelled:
                raise asyncio.CancelledError
            generation = await asyncio.to_thread(
                next_console_default_intent_generation,
                generation,
            )
        raise RuntimeError("Console default reservation changed repeatedly")

    async def _run_console_default_worker_settled(
        self,
        callback: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> tuple[object, bool]:
        """Await a mutating worker to completion before exposing cancellation."""

        worker = asyncio.create_task(
            asyncio.to_thread(partial(callback, *args, **kwargs))
        )
        cancelled = False
        while True:
            try:
                return await asyncio.shield(worker), cancelled
            except asyncio.CancelledError:
                cancelled = True

    async def _publish_console_default_outcome_off_event_loop(
        self,
        intent: ConsoleDefaultMutationIntent,
        outcome: ConsoleDefaultMutationOutcome,
    ) -> bool:
        """Serialize one publication with reservation and recovery claims."""

        async with ChatScreen._console_default_operation_lock(self):
            return await ChatScreen._publish_console_default_outcome_locked(
                self,
                intent,
                outcome,
            )

    async def _publish_console_default_outcome_locked(
        self,
        intent: ConsoleDefaultMutationIntent,
        outcome: ConsoleDefaultMutationOutcome,
    ) -> bool:
        """Publish while the caller owns the non-reentrant operation lock."""

        for _attempt in range(_CONSOLE_DEFAULT_RESERVATION_ATTEMPTS):
            claim, cancelled = await ChatScreen._run_console_default_worker_settled(
                self,
                prepare_console_default_runtime_publication,
                intent,
                outcome,
            )
            if claim is None:
                if cancelled:
                    raise asyncio.CancelledError
                return False
            if not isinstance(claim, ConsoleDefaultRuntimePublicationClaim):
                raise RuntimeError("Default runtime publication claim is invalid")
            if cancelled:
                await ChatScreen._run_console_default_worker_settled(
                    self,
                    abort_console_default_runtime_publication,
                    claim,
                )
                raise asyncio.CancelledError
            try:
                published = self._accept_console_default_runtime_publication(
                    claim.intent_generation,
                    claim.action,
                    claim.settings_view,
                )
            except Exception:
                published = False
            if not published:
                await ChatScreen._run_console_default_worker_settled(
                    self,
                    abort_console_default_runtime_publication,
                    claim,
                )
                return False
            completed, cancelled = await ChatScreen._run_console_default_worker_settled(
                self,
                complete_console_default_runtime_publication,
                claim,
            )
            if completed:
                if cancelled:
                    raise asyncio.CancelledError
                return True
            if cancelled:
                raise asyncio.CancelledError
        raise RuntimeError("Default runtime publication changed repeatedly")

    def _publish_console_default_outcome(
        self,
        intent: ConsoleDefaultMutationIntent,
        outcome: ConsoleDefaultMutationOutcome,
    ) -> bool:
        """Publish a fresh runtime mapping once for the newest intent."""

        return publish_console_default_runtime_if_current(
            intent,
            outcome,
            lambda settings_view: self._accept_console_default_runtime_publication(
                intent.generation,
                intent.action,
                settings_view,
            ),
        )

    def _accept_console_default_runtime_publication(
        self,
        intent_generation: int,
        action: ConsoleSettingsAction,
        settings_view: Mapping[str, object],
    ) -> bool:
        """Install one app view while the defaults service fences reservations."""

        state = self._console_default_durability_state()
        if intent_generation != state.newest_intent_generation:
            return False
        try:
            self.app_instance.app_config = settings_view
        except Exception:
            return False
        if state.runtime_published_intent_generation == intent_generation:
            return True
        next_state, accepted = state.accept_runtime_publication(intent_generation)
        if not accepted:
            return False
        self.app_instance.console_default_durability_state = next_state
        if action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT:
            self.app_instance.console_new_chat_default_generation += 1
        return True

    async def _open_console_settings(
        self,
        *,
        focus_model: bool = False,
        focus_context: bool = False,
        transfer: ConsoleSettingsTransfer | None = None,
    ) -> None:
        """Open Console session settings for the active native session."""
        controller = self._ensure_console_chat_controller()
        store = self._ensure_console_chat_store()
        if transfer is None:
            session_id = store.active_session_id
            if session_id is None:
                return
            origin = store.capture_console_settings_origin(session_id)
            settings = store.session_settings(session_id)
            if settings is None:
                return
            initial_draft = self._console_settings_initial_draft(
                settings,
                store.session_context_policy_overrides(session_id),
                exposed_fields=FULL_MODEL_DEFAULT_FIELDS,
            )
        else:
            origin = transfer.origin
            session_id = origin.session_id
            settings = transfer.draft.settings
            initial_draft = transfer.draft
        try:
            display_name = store.session_user_display_name_override(session_id)
        except KeyError:
            return
        effective_thinking_policy = (
            await controller.effective_thinking_history_policy_for_session(session_id)
        )
        context_estimate = self._console_settings_context_estimate_for_session(
            session_id,
            settings=settings,
        )
        context_state = self._console_context_control_state_for_session(
            session_id,
            estimate=context_estimate,
            settings=settings,
            thinking_history_effective_policy=effective_thinking_policy,
        )
        providers_models = await self._providers_models_for_console_settings(
            settings.provider,
            current_model=settings.model,
        )
        modal = ConsoleSettingsModal(
            settings=settings,
            origin=origin,
            initial_draft=initial_draft,
            transfer=transfer,
            user_display_name_override=display_name,
            global_user_display_name=self._global_chat_display_name(),
            app_config=self._provider_readiness_app_config(),
            providers_models=providers_models,
            context_estimate=context_estimate,
            context_state=context_state,
            can_save=controller.run_state_for(session_id).is_send_allowed,
            focus_model=focus_model,
            focus_context=focus_context,
            reset_current_memory=lambda: controller.reset_active_context_memory(
                session_id
            ),
            undo_current_memory_reset=controller.undo_context_memory_reset,
            reset_all_memories=lambda: controller.reset_all_context_memories(
                session_id
            ),
            compact_now=lambda: controller.compact_context_now(session_id),
            draft_rebaser=controller.rebase_console_settings_draft,
            live_committer=self._commit_console_settings_submission_live,
            default_readiness_resolver=self._console_default_readiness,
            default_durability_state=self._console_default_durability_state(),
            default_recovery_handler=self._handle_console_default_recovery,
        )

        def apply_origin_result(result) -> None:
            self._dispatch_console_settings_submission(result)

        self.app.push_screen(modal, callback=apply_origin_result)

    def _global_chat_display_name(self) -> str:
        """Return the live in-memory global chat label without touching disk."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        chat_defaults = (
            app_config.get("chat_defaults", {})
            if isinstance(app_config, Mapping)
            else {}
        )
        raw_value = (
            chat_defaults.get("user_display_name", "User")
            if isinstance(chat_defaults, Mapping)
            else "User"
        )
        try:
            return (
                normalize_chat_display_name(raw_value, blank_means_none=False) or "User"
            )
        except ChatDisplayNameError:
            return "User"

    def _console_transcript_style(self) -> ConsoleTranscriptStyle:
        """Return the live global Console transcript appearance preference."""

        app_config = getattr(self.app_instance, "app_config", {}) or {}
        appearance = (
            app_config.get("appearance", {}) if isinstance(app_config, Mapping) else {}
        )
        raw_value = (
            appearance.get("console_transcript_style", "role_accents")
            if isinstance(appearance, Mapping)
            else "role_accents"
        )
        return normalize_console_transcript_style(raw_value)

    def _show_model_thinking(self) -> bool:
        """Return the live device-local thinking presentation preference."""

        app_config = getattr(self.app_instance, "app_config", {}) or {}
        console = (
            app_config.get("console", {}) if isinstance(app_config, Mapping) else {}
        )
        raw = (
            console.get("show_model_thinking", True)
            if isinstance(console, Mapping)
            else True
        )
        return raw if type(raw) is bool else True

    def _console_message_presentation(
        self, message: ConsoleChatMessage
    ) -> ConsoleMessagePresentation:
        """Resolve one active-session message for every visible action surface."""
        return resolve_console_message_presentation(
            message, self._console_presentation_context()
        )

    def _console_presentation_context(self) -> ConsolePresentationContext:
        """Return the active Console session's live roleplay context."""
        session = self._session._active_native_console_session()
        if session is None:
            return ConsolePresentationContext(
                user_name=self._global_chat_display_name(),
                transcript_style=self._console_transcript_style(),
            )
        store = self._ensure_console_chat_store()
        return replace(
            store.presentation_context(session.id, self._global_chat_display_name()),
            transcript_style=self._console_transcript_style(),
        )

    def _sync_console_identity_surfaces(self) -> None:
        """Refresh mounted surfaces derived from active chat presentation."""
        self._sync_console_chat_core_state()
        self._sync_console_settings_summary()
        self._sync_console_rail_system_line()
        self._sync_console_control_bar()

    def _apply_console_settings_result(
        self,
        result: ConsoleSettingsResult | ConsoleSessionSettings | None,
        *,
        origin_session_id: str | None = None,
        origin_system_prompt: str | None = None,
    ) -> None:
        """Apply provider settings and the separately owned chat-name override."""
        if not isinstance(result, (ConsoleSettingsResult, ConsoleSessionSettings)):
            return
        settings = (
            result.settings if isinstance(result, ConsoleSettingsResult) else result
        )
        store = self._ensure_console_chat_store()
        session_id = origin_session_id or store.active_session_id
        if session_id is None:
            return
        try:
            current_settings = store.session_settings(session_id)
        except KeyError:
            return
        current_system_prompt = (
            origin_system_prompt
            if origin_session_id is not None
            else (
                current_settings.system_prompt if current_settings is not None else None
            )
        )
        store.replace_session_settings(
            session_id,
            replace(
                settings,
                source="user",
                system_prompt=current_system_prompt,
            ),
        )
        if isinstance(result, ConsoleSettingsResult):
            if result.context_policy_overrides is not None:
                _session, policy_persisted = store.set_session_context_policy_overrides(
                    session_id,
                    result.context_policy_overrides,
                )
                if not policy_persisted:
                    self.app_instance.notify(
                        "Context policy applied in memory but could not be saved.",
                        severity="warning",
                    )
            if result.thinking_history_policy is not None:
                _session, thinking_persisted = (
                    store.set_session_thinking_history_policy(
                        session_id,
                        result.thinking_history_policy,
                    )
                )
                if not thinking_persisted:
                    self.app_instance.notify(
                        "Thinking history policy applied in memory but could not be saved.",
                        severity="warning",
                    )
            _session, persisted = store.set_session_user_display_name_override(
                session_id,
                result.user_display_name_override,
                global_default=self._global_chat_display_name(),
            )
            self._last_console_roleplay_refresh_key = (
                session_id,
                self._global_chat_display_name(),
            )
            if not persisted:
                self.app_instance.notify(
                    "Name changed for this session, but it may not survive reopening.",
                    severity="warning",
                )
        if store.active_session_id == session_id:
            self._sync_console_identity_surfaces()
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
        self.app_instance.notify("Console settings saved.", severity="success")
        # task-16473: a session endpoint with no persisted backing works for
        # this run (llama.cpp readiness even reports "Ready") and then
        # silently evaporates on restart -- the exact trap behind the
        # "re-enter my llama.cpp IP:Port every boot" report.
        endpoint_warning = unsaved_console_endpoint_warning(
            settings,
            app_config=self._provider_readiness_app_config(),
        )
        if endpoint_warning:
            self.app_instance.notify(endpoint_warning, severity="warning")

    async def _refresh_console_roleplay_projections(
        self,
        plan: ConsoleRoleplayProjectionPersistencePlan,
    ) -> None:
        """Persist one current immutable plan without off-thread store access."""
        store = self._ensure_console_chat_store()
        if not store.is_roleplay_projection_plan_current(plan):
            if self._console_roleplay_repair_plan is plan:
                self._console_roleplay_repair_plan = None
                self._console_roleplay_repair_inflight_generation = 0
            return
        owner = ChatScreen._console_settings_durability_owner(self)
        admission = owner.try_acquire()
        if admission is None:
            return
        persistence_task = owner.launch(
            admission,
            store.persist_roleplay_projection_plan_serialized(plan),
            name=(f"console-roleplay-{plan.session_id}-{plan.generation}"),
        )
        persistence_task.add_done_callback(
            partial(
                _consume_console_roleplay_writer_completion,
                session_id=plan.session_id,
                generation=plan.generation,
            )
        )
        self._console_roleplay_writer_task = persistence_task
        try:
            result = await asyncio.shield(persistence_task)
        except asyncio.CancelledError:
            # Unmount abandons only the screen waiter; the app owner keeps and
            # drains the durable task. Mounted cancellation continues to wait
            # so the latest coalesced refresh is not lost.
            if self._console_roleplay_tearing_down:
                raise
            while not persistence_task.done():
                try:
                    await asyncio.shield(persistence_task)
                except asyncio.CancelledError:
                    continue
            result = persistence_task.result()
        finally:
            if self._console_roleplay_writer_task is persistence_task:
                self._console_roleplay_writer_task = None
        if result is None:
            if self._console_roleplay_repair_plan is plan:
                self._console_roleplay_repair_plan = None
                self._console_roleplay_repair_inflight_generation = 0
            return
        accepted = store.accept_roleplay_projection_persistence_result(result)
        if self._console_roleplay_repair_plan is plan:
            repair_generation = self._console_roleplay_repair_inflight_generation
            self._console_roleplay_repair_plan = None
            self._console_roleplay_repair_inflight_generation = 0
            if accepted and result.persisted and repair_generation > 0:
                self._console_roleplay_repair_generation = repair_generation
                self.app_instance._console_roleplay_repair_consumed_generation = max(
                    repair_generation,
                    int(
                        getattr(
                            self.app_instance,
                            "_console_roleplay_repair_consumed_generation",
                            0,
                        )
                        or 0
                    ),
                )
        if accepted and not result.persisted:
            self.app_instance.notify(
                "Your chat name is active, but updated character templates may not "
                "survive reopening.",
                severity="warning",
            )
        if accepted and store.active_session_id == plan.session_id:
            self._sync_console_identity_surfaces()

    async def _drain_console_roleplay_persistence(self) -> None:
        """Drain one active and one replaceable latest projection plan."""
        plan: ConsoleRoleplayProjectionPersistencePlan | None = None
        try:
            while self._console_roleplay_pending_plan is not None:
                plan = self._console_roleplay_pending_plan
                self._console_roleplay_pending_plan = None
                self._console_roleplay_active_plan = plan
                await self._refresh_console_roleplay_projections(plan)
                self._console_roleplay_active_plan = None
                plan = None
        except asyncio.CancelledError:
            if (
                self._console_roleplay_pending_plan is None
                and plan is not None
                and self._ensure_console_chat_store().is_roleplay_projection_plan_current(
                    plan
                )
            ):
                self._console_roleplay_pending_plan = plan
            raise
        finally:
            self._console_roleplay_active_plan = None
            self._console_roleplay_drain_scheduled = False

    async def _await_console_roleplay_persistence_task(
        self, task: asyncio.Task[None]
    ) -> None:
        """Let Textual cancel its waiter without cancelling the durable queue."""
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            return

    def _finish_console_roleplay_persistence_task(
        self, task: asyncio.Task[None]
    ) -> None:
        """Release a drained task and consume any unexpected exception."""
        if self._console_roleplay_persistence_task is task:
            self._console_roleplay_persistence_task = None
        if task.cancelled():
            error = None
        else:
            error = task.exception()
            if error is not None:
                failed_plan = (
                    self._console_roleplay_active_plan
                    or self._console_roleplay_pending_plan
                )
                logger.error(
                    "Console roleplay projection persistence task failed "
                    "(session_id={}, generation={}, task_name={}): {!r}",
                    failed_plan.session_id if failed_plan is not None else "unknown",
                    failed_plan.generation if failed_plan is not None else 0,
                    task.get_name(),
                    error,
                )
        if (
            self._console_roleplay_pending_plan is not None
            and self.is_mounted
            and not self._console_roleplay_tearing_down
        ):
            self._start_console_roleplay_persistence_drain()

    def _console_roleplay_unmount_timeout_seconds(self) -> float:
        """Return the bounded configurable drain deadline for this screen."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        console_config = app_config.get("console", {})
        raw_timeout = (
            console_config.get(
                "roleplay_refresh_teardown_timeout_seconds",
                CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS,
            )
            if isinstance(console_config, dict)
            else CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS
        )
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError):
            return CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS
        if not 0.01 <= timeout <= 5.0:
            return CONSOLE_ROLEPLAY_UNMOUNT_TIMEOUT_SECONDS
        return timeout

    def _publish_console_roleplay_repair_marker(self) -> None:
        """Publish the latest desired identity on the app, not this screen."""
        generation = (
            int(
                getattr(self.app_instance, "_console_roleplay_repair_generation", 0)
                or 0
            )
            + 1
        )
        self.app_instance._console_roleplay_repair_generation = generation
        self.app_instance._console_roleplay_repair_global_name = (
            self._global_chat_display_name()
        )
        # Textual resumes the uncovered screen before awaiting this screen's
        # unmount hook. A marker published only after the teardown deadline
        # therefore misses that screen's normal ``on_screen_resume`` probe.
        # The loop callback resolves the app's current screen only after pop
        # completes and captures neither this retiring screen nor its store.
        asyncio.get_running_loop().call_later(
            0.1,
            _consume_console_roleplay_repair_for_current_screen,
            self.app,
        )

    async def _teardown_console_roleplay_persistence(self) -> None:
        """Bound screen teardown while app-owned immutable durability continues."""
        self._console_roleplay_tearing_down = True
        task = self._console_roleplay_persistence_task
        if task is None or task.done():
            if (
                self._console_roleplay_pending_plan is not None
                or self._console_roleplay_active_plan is not None
                or (
                    self._console_roleplay_writer_task is not None
                    and not self._console_roleplay_writer_task.done()
                )
            ):
                self._publish_console_roleplay_repair_marker()
            self._console_roleplay_persistence_task = None
            self._console_roleplay_active_plan = None
            self._console_roleplay_pending_plan = None
            self._console_roleplay_writer_task = None
            self._console_roleplay_drain_scheduled = False
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._console_roleplay_unmount_timeout_seconds(),
            )
        except TimeoutError:
            self._publish_console_roleplay_repair_marker()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except Exception:
            self._publish_console_roleplay_repair_marker()
            failed_plan = (
                self._console_roleplay_active_plan
                or self._console_roleplay_pending_plan
            )
            logger.exception(
                "Console roleplay projection drain failed during teardown "
                "(session_id={}, generation={}, task_name={}).",
                failed_plan.session_id if failed_plan is not None else "unknown",
                failed_plan.generation if failed_plan is not None else 0,
                task.get_name(),
            )
        finally:
            self._console_roleplay_persistence_task = None
            self._console_roleplay_active_plan = None
            self._console_roleplay_pending_plan = None
            self._console_roleplay_writer_task = None
            self._console_roleplay_drain_scheduled = False

    def _start_console_roleplay_persistence_drain(self) -> None:
        """Start the sole retained persistence drain when work is pending."""
        if self._console_roleplay_pending_plan is None:
            return
        task = self._console_roleplay_persistence_task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._console_roleplay_drain_scheduled:
                return
            self._console_roleplay_drain_scheduled = True
            self.run_worker(
                self._drain_console_roleplay_persistence(),
                exclusive=False,
                group="console-roleplay-refresh",
            )
            return
        self._console_roleplay_drain_scheduled = True
        task = loop.create_task(self._drain_console_roleplay_persistence())
        self._console_roleplay_persistence_task = task
        task.add_done_callback(self._finish_console_roleplay_persistence_task)
        self.run_worker(
            partial(self._await_console_roleplay_persistence_task, task),
            exclusive=False,
            group="console-roleplay-refresh",
        )

    def _dispatch_active_console_roleplay_refresh(
        self,
        *,
        force_persistence: bool = False,
        repair_generation: int = 0,
    ) -> bool:
        """Coalesce refresh writes by active session and effective global name."""
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return False
        self._start_console_roleplay_persistence_drain()
        global_user_display_name = self._global_chat_display_name()
        refresh_key = (store.active_session_id, global_user_display_name)
        if (
            not force_persistence
            and refresh_key == self._last_console_roleplay_refresh_key
        ):
            return False
        self._last_console_roleplay_refresh_key = refresh_key
        try:
            plan = store.prepare_session_roleplay_projection_refresh(
                refresh_key[0],
                global_default=global_user_display_name,
                force_persistence=force_persistence,
            )
        except KeyError:
            return False
        self._sync_console_identity_surfaces()
        if plan is not None:
            if repair_generation > 0:
                self._console_roleplay_repair_plan = plan
                self._console_roleplay_repair_inflight_generation = repair_generation
            self._console_roleplay_pending_plan = plan
            self._start_console_roleplay_persistence_drain()
        return plan is not None if force_persistence else True

    def request_console_identity_refresh(self, generation: int | None = None) -> bool:
        """Consume a global display-name save without waiting for the sync tick."""
        observed = self._console_identity_refresh_generation
        if generation is None:
            generation = int(
                getattr(
                    self.app_instance,
                    "_console_identity_refresh_generation",
                    0,
                )
                or 0
            )
        if generation <= observed:
            return False
        self._console_identity_refresh_generation = generation
        self._last_console_roleplay_refresh_key = None
        return self._dispatch_active_console_roleplay_refresh()

    def request_console_appearance_refresh(self, generation: int | None = None) -> bool:
        """Refresh mounted transcript rows immediately after an Appearance save."""

        observed = self._console_appearance_refresh_generation
        if generation is None:
            generation = int(
                getattr(
                    self.app_instance,
                    "_console_appearance_refresh_generation",
                    0,
                )
                or 0
            )
        if generation <= observed:
            return False
        self._console_appearance_refresh_generation = generation
        self._last_native_transcript_refresh_key = None
        if self.is_mounted:
            try:
                transcript = self.query_one(
                    "#console-native-transcript", ConsoleTranscript
                )
            except QueryError:
                transcript = None
            if transcript is not None:
                # Appearance only changes message presentation. Updating the
                # transcript context avoids queueing behind the much broader
                # Console sync worker during a busy mount or active turn.
                transcript.set_presentation_context(
                    self._console_presentation_context(),
                    force=True,
                )
                transcript.set_model_thinking_visible(self._show_model_thinking())
        return True

    def _consume_pending_console_identity_refresh(self) -> bool:
        """Consume an identity generation missed while Console was inactive."""
        return self.request_console_identity_refresh()

    def _consume_pending_console_roleplay_repair(self) -> bool:
        """Force-persist the latest source projection after abandoned teardown."""
        generation = int(
            getattr(self.app_instance, "_console_roleplay_repair_generation", 0) or 0
        )
        app_consumed = int(
            getattr(
                self.app_instance,
                "_console_roleplay_repair_consumed_generation",
                0,
            )
            or 0
        )
        if generation <= max(self._console_roleplay_repair_generation, app_consumed):
            return False
        if self._console_roleplay_repair_inflight_generation >= generation:
            return False
        self._ensure_console_chat_store()
        self._last_console_roleplay_refresh_key = None
        dispatched = self._dispatch_active_console_roleplay_refresh(
            force_persistence=True,
            repair_generation=generation,
        )
        return dispatched

    async def on_console_settings_open(self, event: Button.Pressed) -> None:
        """Open Console session settings for the active native session."""
        event.stop()
        summary_state = self._build_console_settings_summary_state()
        recovery_label, _recovery_target, _recovery_tooltip = (
            self._console_provider_recovery_action()
        )
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
            await self._session._create_native_console_session_from_active_context()
        elif action_id == "settings":
            await self._open_console_settings(focus_model=False)
        elif action_id == "attach-context":
            available_columns = self._console_rail_available_columns()
            rail_state = self._current_console_rail_state(
                available_columns=available_columns
            )
            preference_changes = console_context_reveal_preferences(
                rail_state, available_columns
            )
            self._set_console_rail_preference(
                left_open=preference_changes["left_open"],
                right_open=preference_changes.get("right_open"),
            )
        elif action_id == "run-library-rag":
            self._open_console_library_search()
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
        # Fleet-UX expert review F2 (task-1232): read the LIVE parallel-run
        # cap so the help copy tracks a user override instead of quoting the
        # baked-in default.
        max_parallel_runs = self._ensure_console_chat_controller().max_parallel_runs
        shortcut_groups = CONSOLE_WORKBENCH_SHORTCUT_GROUPS
        if self._console_inspector_active():
            shortcut_groups = (
                *shortcut_groups,
                ("Inspector", (("n / p", "next / previous section"),)),
            )
        focused = self.app.focused
        if isinstance(focused, Widget):
            authority_summary = next(
                (
                    candidate
                    for candidate in focused.ancestors_with_self
                    if isinstance(candidate, ConsoleSendAuthoritySummary)
                ),
                None,
            )
            if authority_summary is not None:
                authority_rows = tuple(
                    (escape_markup(label), escape_markup(value))
                    for label, value in authority_summary.contextual_help_rows()
                )
                shortcut_groups = (
                    *shortcut_groups,
                    (
                        "What happens if I send now?",
                        authority_rows,
                    ),
                )
        if isinstance(self.app.focused, ConsoleWorkspaceTree):
            try:
                tray = self.query_one(
                    "#console-workspaces-context", ConsoleWorkspaceContextTray
                )
            except (NoMatches, QueryError):
                context_data = None
            else:
                context_data = getattr(tray, "_workspace_tree_context_data", None)
            label = escape_markup(
                str(getattr(context_data, "raw_label", "") or "Workspace tree")
            )
            shortcut_groups = (
                *shortcut_groups,
                (
                    "Workspaces",
                    (
                        ("Selected", label),
                        ("Single click", "select row; expand a collapsed workspace"),
                        ("Double-click", "open the selected workspace or conversation"),
                        ("Enter", "open the selected row"),
                        ("Space", "toggle workspace disclosure"),
                        ("Left", "collapse or move to the parent workspace"),
                        ("Right", "expand or move to the first child"),
                    ),
                ),
            )
        self.app.push_screen(
            WorkbenchHelpPanel(
                WorkbenchHelpState(
                    route_id=workbench_state.route_id,
                    title="Console",
                    actions=workbench_state.actions,
                    notes_heading="Agents",
                    notes=_console_workbench_agents_notes(max_parallel_runs),
                    shortcut_groups=shortcut_groups,
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
            if not self._console_pane_is_reachable(pane_id)
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
            if not self._console_pane_is_reachable(pane_id)
        }
        current = self._console_workbench_focus_id_for_widget(self.app.focused)
        target_id = CONSOLE_FOCUS_REGISTRY.previous_before(current, hidden=hidden)
        if target_id is None:
            return
        self._focus_console_workbench_target(target_id)

    def _console_footer_is_single_pane(self) -> bool:
        """Return whether the shell is below the single-pane threshold.

        TASK-24703: used only to decide footer hint ORDER, so it must never
        raise while the screen is mid-mount or being torn down -- a footer
        registration is not worth an exception. Any failure to resolve the
        rail state means "not single pane", which yields the normal ordering.
        """

        try:
            rail_state = self._current_console_rail_state(
                available_columns=self._console_rail_available_columns()
            )
        except Exception:  # noqa: BLE001 - ordering hint only, never fatal
            return False
        return bool(getattr(rail_state, "single_pane", False))

    def _console_workbench_density(self) -> str:
        """Return the supported Console Workbench density from app config."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        appearance = app_config.get("appearance", {})
        if not isinstance(appearance, dict):
            return "normal"
        density = (
            str(appearance.get("ui_density", appearance.get("density", "normal")) or "")
            .strip()
            .lower()
        )
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
            # TASK-2154.11: between-pane widgets (control bar, chips, rail
            # handles) count as their logical pane -- checked AFTER the pane
            # roots so a real pane always wins for its own subtree.
            mapped = CONSOLE_FOCUS_PANE_FOR_WIDGET.get(current_id or "")
            if mapped is not None:
                return mapped
            current = getattr(current, "parent", None)
        return None

    def _console_tab_region_selector(self, focused: object | None) -> str | None:
        """Return a CSS selector scoping Tab to the focused widget's region.

        TASK-2154.11 (AC-02): walks the focused widget's ancestor chain for a
        ``CONSOLE_TAB_REGIONS`` root; on a match, returns a selector union of
        that region's roots and their descendants, which Textual's
        ``Screen.focus_next``/``focus_previous`` use to filter the focus
        chain -- so Tab wraps within the region and never crosses into
        app-level chrome. Returns None when focus sits outside every Console
        region (nav bar, footer, header), where the default chain applies.
        """
        region_roots: tuple[str, ...] | None = None
        current = focused
        while current is not None:
            current_id = getattr(current, "id", None)
            if current_id is not None:
                for roots in CONSOLE_TAB_REGIONS:
                    if current_id in roots:
                        region_roots = roots
                        break
            if region_roots is not None:
                break
            current = getattr(current, "parent", None)
        if region_roots is None:
            return None
        return ", ".join(
            selector for root in region_roots for selector in (f"#{root}", f"#{root} *")
        )

    def _focus_console_workbench_target(self, widget_id: str) -> None:
        """Focus a visible Console Workbench target if it is available."""
        for target_id in self._console_workbench_focus_targets(widget_id):
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

    def _console_workbench_focus_targets(self, pane_id: str) -> tuple[str, ...]:
        """Return visible focus candidates for a Console Workbench pane."""
        if pane_id == "console-native-composer":
            if self._console_composer_collapsed:
                return ("console-composer-expand",)
            return ("console-native-composer",)
        targets = CONSOLE_FOCUS_TARGETS_BY_PANE.get(pane_id, (pane_id,))
        # TASK-24604: a collapsed rail's handle is the pane's last-resort
        # target. The caller already skips non-displayed candidates, so this
        # only takes effect when the rail itself is closed.
        stand_in = CONSOLE_PANE_COLLAPSED_STAND_IN.get(pane_id)
        if stand_in is not None and stand_in not in targets:
            targets = (*targets, stand_in)
        return targets

    def _console_pane_is_reachable(self, pane_id: str) -> bool:
        """Return whether F6 can land on a pane in its current form.

        TASK-24604: ``_is_console_widget_displayed`` alone answered "is the
        rail open", which is not the same question. A collapsed rail is still
        reachable -- through its handle -- and treating it as hidden dropped
        the Inspector out of the F6 cycle in its default state.
        """
        if self._is_console_widget_displayed(pane_id):
            return True
        stand_in = CONSOLE_PANE_COLLAPSED_STAND_IN.get(pane_id)
        return stand_in is not None and self._is_console_widget_displayed(stand_in)

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
            for widget_id in self._console_workbench_focus_targets(pane_id):
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
        if self._console_composer_collapsed:
            self._focus_console_workbench_target("console-native-composer")
            return
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
        """Register Console Workbench shortcuts with this screen's own footer.

        Routed through BaseAppScreen's persisting registration so the hints
        survive any screen-level recompose, which replaces the footer widget.
        (TASK-259: `_stage_console_library_rag_launch` no longer recomposes
        the screen, but the fallback path and future recompose sources keep
        this persisting registration load-bearing.)

        TASK-2154.8 (FR-06): while the setup modal locks the composer, the
        blocked variant is registered instead -- the "Enter send" hint is
        replaced by "Enter continue setup", which is what Enter actually does
        with focus on the setup card's primary action.
        """
        if self._console_setup_modal_blocking():
            shortcuts = CONSOLE_WORKBENCH_SHORTCUTS_SETUP_BLOCKED
        elif self._console_footer_is_single_pane():
            # TASK-24703: below the single-pane threshold the rail's edge
            # handle is hidden, so Alt+I is the only route in -- and it is
            # exactly the width where the footer's prefix-keeping degradation
            # was dropping that hint. Promote it rather than merely keep it.
            shortcuts = CONSOLE_WORKBENCH_SHORTCUTS_SINGLE_PANE
        else:
            shortcuts = CONSOLE_WORKBENCH_SHORTCUTS
        # task-18812 / ADR-031: advertise the focus toggle in the footer —
        # the only exit affordance visible in focus mode (no nav bar). The
        # label names the action the key will perform, per the truthfulness
        # rule. PREPENDED, not appended: AppFooterStatus's degradation drops
        # hints from the END of the context when width runs out, and the
        # Console context is already at that budget at common widths — an
        # appended focus hint never rendered (caught in live verification
        # at 160 cols: 153-cell context vs 152 available).
        focus_label = (
            "exit focus"
            if bool(getattr(self.app_instance, "focus_mode", False))
            else "focus"
        )
        shortcuts = (("Ctrl+Shift+F", focus_label), *shortcuts)
        if self._console_inspector_active():
            shortcuts = (("n/p", "Sections"), *shortcuts)
        self.register_footer_shortcuts(source="console", shortcuts=shortcuts)

    def _console_inspector_active(self) -> bool:
        """Return whether live focus is the Inspector rail or a descendant."""

        try:
            rail = self.query_one("#console-right-rail", ConsoleInspectorRail)
        except (NoMatches, QueryError):
            return False
        focused = self.app.focused
        return isinstance(focused, Widget) and rail.inspector_active(focused)

    def _apply_focus_chrome(self) -> None:
        """Mirror the app-level focus_mode flag onto this screen (task-18812).

        Idempotent: sets/removes the ``-focus`` class that suppresses the
        nav bar and workbench header (CSS: _agentic_terminal.tcss), and
        refreshes the footer hints so the focus toggle's label tracks the
        target state.
        """
        focused = bool(getattr(self.app_instance, "focus_mode", False))
        self.set_class(focused, "-focus")
        self._register_console_footer_shortcuts()

    def _clear_console_footer_shortcuts(self) -> None:
        """Clear Console Workbench shortcuts from this screen's own footer."""
        self.clear_footer_shortcuts(source="console")

    async def action_open_console_session_switcher(self) -> None:
        """Open the Ctrl+K fuzzy session switcher."""
        if self._console_setup_modal_blocking():
            return
        self.app.push_screen(
            ConsoleSessionSwitcherModal(
                rows=await self._workspace.console_session_switcher_rows()
            ),
            callback=self._session._apply_console_switcher_choice,
        )

    def action_open_trajectory_view(self) -> None:
        """Open Trace for the active Console conversation (``y``)."""
        self._review_selection.open_trajectory_view()

    async def action_open_console_model_popover(self) -> None:
        """Open the Alt+M quick provider/model/temperature/streaming popover."""
        if self._console_setup_modal_blocking():
            return
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            return
        origin = store.capture_console_settings_origin(session_id)
        settings = store.session_settings(session_id)
        if settings is None:
            return
        context_policy = store.session_context_policy_overrides(session_id)
        session = store.switch_session(session_id)
        initial_draft = self._console_settings_initial_draft(
            settings,
            context_policy,
            exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
        )
        providers_models = await self._providers_models_for_console_settings(
            settings.provider,
            current_model=settings.model,
        )
        effective_thinking_policy = await self._ensure_console_chat_controller().effective_thinking_history_policy_for_session(
            origin.session_id
        )
        context_state = self._console_context_control_state_for_session(
            origin.session_id,
            settings=settings,
            thinking_history_effective_policy=effective_thinking_policy,
        )
        self.app.push_screen(
            ConsoleModelPopover(
                origin=origin,
                app_config=self._provider_readiness_app_config(),
                initial_draft=initial_draft,
                providers_models=providers_models,
                context_state=context_state,
                scope_copy="Applies to this conversation",
                durability_copy=(
                    "Temporary until this chat is promoted"
                    if session.ephemeral
                    else "Saved with the conversation after its first message"
                    if session.persisted_conversation_id is None
                    else "Saved with this conversation"
                ),
                draft_rebaser=(
                    self._ensure_console_chat_controller().rebase_console_settings_draft
                ),
                live_committer=self._commit_console_settings_submission_live,
                default_readiness_resolver=self._console_default_readiness,
            ),
            callback=self._apply_console_model_popover_result,
        )

    def _apply_console_model_popover_result(
        self,
        result: object,
    ) -> None:
        """Route a typed quick-settings result through the shared coordinator."""
        if result is None:
            return
        if isinstance(result, ConsoleSettingsTransfer):
            self.run_worker(
                self._open_console_settings(focus_model=True, transfer=result),
                exclusive=False,
            )
            return
        self._dispatch_console_settings_submission(result)

    def _launch_console_settings_durability_task(
        self,
        committed: ConsoleSettingsCommittedSubmission,
        default_intent: ConsoleDefaultMutationIntent | None,
    ) -> asyncio.Task[None] | None:
        """Launch post-close durability under the application lifetime."""

        owner = ChatScreen._console_settings_durability_owner(self)
        admission = committed.live_commit.durability_admission
        if admission is None:
            admission = owner.try_acquire()
        if admission is None:
            logger.warning(
                "Console settings durability rejected after shutdown admission closed"
            )
            return None
        task = owner.launch(
            admission,
            self._coordinate_console_settings_submission(
                committed,
                default_intent,
            ),
            name=f"console-settings-{committed.submission.submission_id}",
        )

        def report_failure(completed: asyncio.Task[None]) -> None:
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.opt(exception=error).error(
                    "Console settings app-owned durability task failed"
                )

        task.add_done_callback(report_failure)
        return task

    def _dispatch_console_settings_submission(self, result: object) -> None:
        """Refresh live UI and launch durability exactly once per submission."""

        if not isinstance(result, ConsoleSettingsCommittedSubmission):
            return
        owner = ChatScreen._console_settings_durability_owner(self)
        admission = result.live_commit.durability_admission
        if admission is None:
            admission = owner.try_acquire()
            if admission is None:
                return
            result = replace(
                result,
                live_commit=replace(
                    result.live_commit,
                    durability_admission=admission,
                ),
            )
        submission_id = result.submission.submission_id
        coordinated = getattr(
            self,
            "_console_settings_coordinated_submission_ids",
            None,
        )
        if not isinstance(coordinated, deque):
            coordinated = deque(maxlen=64)
            self._console_settings_coordinated_submission_ids = coordinated
        if submission_id in coordinated:
            if admission is not None:
                owner.release(admission)
            return
        coordinated.append(submission_id)

        try:
            task = self._launch_console_settings_durability_task(result, None)
        except BaseException:
            owner.release(admission)
            raise
        if task is None:
            return
        store = self._ensure_console_chat_store()
        if store.active_session_id == result.live_commit.session_id:
            self._sync_console_identity_surfaces()
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
        self.app_instance.notify("This chat updated", severity="success")

    async def _coordinate_console_settings_submission(
        self,
        committed: ConsoleSettingsCommittedSubmission,
        default_intent: ConsoleDefaultMutationIntent | None,
    ) -> None:
        """Publish independent conversation and default durability outcomes."""

        store = self._ensure_console_chat_store()
        submission = committed.submission
        full_settings_submission = (
            submission.surface is ConsoleSettingsSurface.FULL_SETTINGS
        )
        policy_failure_label = (
            ConsoleSettingsPolicyFailureLabel.CONTEXT_SETTINGS
            if full_settings_submission
            else ConsoleSettingsPolicyFailureLabel.COMPACTION
        )
        display_name_plan: ConsoleRoleplayProjectionPersistencePlan | None = None
        display_name_prepare_failed = False
        if full_settings_submission:
            try:
                _session, display_name_plan = (
                    store.prepare_session_user_display_name_override_for_commit(
                        committed.live_commit,
                        submission.user_display_name_override,
                        global_default=self._global_chat_display_name(),
                    )
                )
            except Exception:
                logger.exception("Console settings display-name preparation failed")
                display_name_prepare_failed = True

        async def persist_display_name() -> None:
            if not full_settings_submission:
                return
            if display_name_prepare_failed:
                self.app_instance.notify(
                    "Name changed for this session, but it may not survive reopening.",
                    severity="warning",
                )
                return
            if display_name_plan is None:
                return
            try:
                result = await store.persist_roleplay_projection_plan_serialized(
                    display_name_plan,
                )
            except Exception:
                logger.exception("Console settings display-name persistence failed")
                self.app_instance.notify(
                    "Name changed for this session, but it may not survive reopening.",
                    severity="warning",
                )
                return
            if result is None:
                return
            accepted = store.accept_roleplay_projection_persistence_result(result)
            if not accepted:
                return
            if store.active_session_id == display_name_plan.session_id:
                self._sync_console_identity_surfaces()
            if not result.persisted:
                self.app_instance.notify(
                    "Name changed for this session, but it may not survive reopening.",
                    severity="warning",
                )

        async def persist_conversation() -> None:
            try:
                await store.persist_console_settings_commit_serialized(
                    committed.live_commit,
                    policy_failure_label=policy_failure_label,
                )
            except Exception:
                logger.exception("Console settings conversation persistence failed")
            finally:
                self._sync_console_settings_recovery_surfaces()

        async def persist_default() -> None:
            intent = default_intent
            if (
                intent is None
                and submission.action is ConsoleSettingsAction.APPLY_TO_CHAT
            ):
                return
            if intent is None:
                try:
                    intent = await self._reserve_console_default_intent_off_event_loop(
                        submission
                    )
                except Exception:
                    logger.exception("Console default reservation failed")
                    self._sync_console_settings_recovery_surfaces()
                    recovery = self._console_default_durability_state()
                    recovery_copy = (
                        "the previous default recovery remains available."
                        if recovery.recovery_intent is not None
                        else "try this default action again."
                    )
                    self.app_instance.notify(
                        "Default not saved for "
                        f"{provider_config_key(submission.draft.settings.provider)}/"
                        f"{submission.draft.settings.model}; {recovery_copy}",
                        severity="warning",
                    )
                    return
            try:
                outcome = await asyncio.to_thread(
                    apply_console_default_intent,
                    intent,
                )
            except Exception:
                logger.exception("Console default persistence failed")
                self._record_console_default_failure(
                    intent,
                    ConsoleDefaultSavePhase.BEFORE_REPLACE,
                )
                return
            try:
                published = await self._publish_console_default_outcome_off_event_loop(
                    intent,
                    outcome,
                )
            except Exception:
                logger.exception("Console default runtime publication failed")
                published = False
            if published:
                scope = (
                    "Eligible new-chat default saved"
                    if intent.action is ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT
                    else "Model profile default saved"
                )
                self.app_instance.notify(
                    f"{scope}: {intent.provider_config_key}/{intent.literal_model_id}",
                    severity="success",
                )
            elif outcome.failure_phase is not None:
                self._record_console_default_failure(
                    intent,
                    outcome.failure_phase,
                )
            elif outcome.runtime_published and outcome.settings_view is not None:
                self._record_console_default_failure(
                    intent,
                    ConsoleDefaultSavePhase.CACHE_PUBLICATION,
                )

        await asyncio.gather(
            persist_conversation(),
            persist_default(),
            persist_display_name(),
        )

    def _record_console_default_failure(
        self,
        intent: ConsoleDefaultMutationIntent,
        phase: ConsoleDefaultSavePhase,
    ) -> None:
        """Retain only a current app-global recovery record."""

        state = self._console_default_durability_state()
        if state.newest_intent_generation != intent.generation:
            return
        self.app_instance.console_default_durability_state = (
            ConsoleDefaultDurabilityState(
                newest_intent_generation=intent.generation,
                recovery_intent=intent,
                failure_phase=phase,
                runtime_published_intent_generation=(
                    state.runtime_published_intent_generation
                ),
            )
        )
        self._sync_console_settings_recovery_surfaces()

    async def _handle_console_default_recovery(
        self,
        request: ConsoleDefaultRecoveryRequest,
    ) -> ConsoleDefaultDurabilityState:
        """Admit and execute one generation-bound app-global recovery."""

        state = self._console_default_durability_state()
        if not isinstance(request, ConsoleDefaultRecoveryRequest):
            return state
        intent = state.recovery_intent
        if (
            intent is None
            or request.intent_generation != state.newest_intent_generation
        ):
            return state
        allowed_actions = {
            ConsoleDefaultSavePhase.BEFORE_REPLACE: {
                ConsoleDefaultRecoveryAction.RETRY_SAVE,
                ConsoleDefaultRecoveryAction.DISCARD_RETRY,
            },
            ConsoleDefaultSavePhase.CACHE_PUBLICATION: {
                ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP,
                ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
            },
        }
        if request.action not in allowed_actions.get(state.failure_phase, set()):
            return state
        owner = ChatScreen._console_settings_durability_owner(self)
        admission = owner.try_acquire()
        if admission is None:
            return state
        task = owner.launch(
            admission,
            ChatScreen._run_console_default_recovery(self, request),
            name=f"console-default-recovery-{request.intent_generation}",
        )
        return await asyncio.shield(task)

    async def _run_console_default_recovery(
        self,
        request: ConsoleDefaultRecoveryRequest,
    ) -> ConsoleDefaultDurabilityState:
        """Run one admitted recovery under generation/phase single-flight."""

        state = self._console_default_durability_state()
        if not isinstance(request, ConsoleDefaultRecoveryRequest):
            return state
        intent = state.recovery_intent
        if (
            intent is None
            or request.intent_generation != state.newest_intent_generation
        ):
            return state
        allowed_actions = {
            ConsoleDefaultSavePhase.BEFORE_REPLACE: {
                ConsoleDefaultRecoveryAction.RETRY_SAVE,
                ConsoleDefaultRecoveryAction.DISCARD_RETRY,
            },
            ConsoleDefaultSavePhase.CACHE_PUBLICATION: {
                ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP,
                ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
            },
        }
        failure_phase = state.failure_phase
        if request.action not in allowed_actions.get(failure_phase, set()):
            return state
        inflight = getattr(
            self.app_instance,
            "console_default_recovery_inflight",
            None,
        )
        if not isinstance(inflight, set):
            inflight = set()
            self.app_instance.console_default_recovery_inflight = inflight
        assert isinstance(failure_phase, ConsoleDefaultSavePhase)
        flight_key = (
            request.intent_generation,
            failure_phase.value,
        )
        if flight_key in inflight:
            return state
        inflight.add(flight_key)
        try:
            if request.action in {
                ConsoleDefaultRecoveryAction.DISCARD_RETRY,
                ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
            }:
                state = ConsoleDefaultDurabilityState(
                    newest_intent_generation=state.newest_intent_generation,
                    runtime_published_intent_generation=(
                        state.runtime_published_intent_generation
                    ),
                )
                self.app_instance.console_default_durability_state = state
                self._sync_console_settings_recovery_surfaces()
                return state
            if (
                request.action is ConsoleDefaultRecoveryAction.RETRY_SAVE
                and failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
            ):
                outcome = await asyncio.to_thread(
                    apply_console_default_intent,
                    intent,
                )
            elif (
                request.action is ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP
                and failure_phase is ConsoleDefaultSavePhase.CACHE_PUBLICATION
            ):
                refresh = await asyncio.to_thread(
                    refresh_console_runtime_after_saved_default
                )
                outcome = ConsoleDefaultMutationOutcome(
                    intent_generation=intent.generation,
                    file_replaced=True,
                    runtime_published=refresh.published,
                    settings_view=refresh.settings_view,
                    failure_phase=(
                        None
                        if refresh.published
                        else ConsoleDefaultSavePhase.CACHE_PUBLICATION
                    ),
                )
            else:
                return state
        except Exception:
            logger.exception("Console default recovery failed")
            current = self._console_default_durability_state()
            if (
                current.recovery_intent == intent
                and current.failure_phase is failure_phase
            ):
                self._record_console_default_failure(intent, failure_phase)
            return self._console_default_durability_state()
        finally:
            inflight.discard(flight_key)
        current = self._console_default_durability_state()
        if (
            current.recovery_intent != intent
            or current.failure_phase is not failure_phase
        ):
            return current
        try:
            published = await self._publish_console_default_outcome_off_event_loop(
                intent,
                outcome,
            )
        except Exception:
            logger.exception("Console default recovery publication failed")
            published = False
        if not published:
            phase = outcome.failure_phase or ConsoleDefaultSavePhase.CACHE_PUBLICATION
            self._record_console_default_failure(intent, phase)
        self._sync_console_settings_recovery_surfaces()
        return self._console_default_durability_state()

    def _sync_console_settings_recovery_surfaces(self) -> None:
        """Refresh the mounted rail's current session/app recovery rows."""

        try:
            rail = self.query_one("#console-left-rail", ConsoleLeftRail)
        except (NoMatches, QueryError):
            return
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        session = (
            getattr(store, "_sessions", {}).get(session_id)
            if session_id is not None
            else None
        )
        rail.sync_model_recovery(
            session_id=session_id,
            failures=(
                session.settings_persistence_failures if session is not None else {}
            ),
            default_state=self._console_default_durability_state(),
        )

    @on(Button.Pressed, f"#{CONSOLE_RETRY_GENERATION_SETTINGS_ID}")
    @on(Button.Pressed, f"#{CONSOLE_RETRY_CONTEXT_SETTINGS_ID}")
    async def on_console_settings_component_retry(
        self,
        event: Button.Pressed,
    ) -> None:
        """Retry one exact session/component revision from the Model rail."""

        event.stop()
        session_id = getattr(event.button, "console_settings_session_id", None)
        revision = getattr(event.button, "console_settings_revision", None)
        if type(session_id) is not str or type(revision) is not int:
            return
        component = (
            ConsoleSettingsComponent.GENERATION_SETTINGS
            if event.button.id == CONSOLE_RETRY_GENERATION_SETTINGS_ID
            else ConsoleSettingsComponent.CONTEXT_POLICY
        )
        owner = ChatScreen._console_settings_durability_owner(self)
        admission = owner.try_acquire()
        if admission is None:
            return
        task = owner.launch(
            admission,
            self._ensure_console_chat_store().retry_console_settings_persistence(
                session_id=session_id,
                component=component,
                revision=revision,
            ),
            name=f"console-settings-retry-{session_id}-{component.value}-{revision}",
        )
        await asyncio.shield(task)
        self._sync_console_settings_recovery_surfaces()

    @on(Button.Pressed, f"#{CONSOLE_RETRY_DEFAULT_SAVE_ID}")
    @on(Button.Pressed, f"#{CONSOLE_DISCARD_DEFAULT_RETRY_ID}")
    @on(Button.Pressed, f"#{CONSOLE_REFRESH_RUNNING_APP_ID}")
    @on(Button.Pressed, f"#{CONSOLE_DISMISS_DEFAULT_REFRESH_ID}")
    async def on_console_default_recovery(self, event: Button.Pressed) -> None:
        """Route an exact app-global recovery token through one handler."""

        event.stop()
        generation = getattr(
            event.button,
            "console_default_intent_generation",
            None,
        )
        actions = {
            CONSOLE_RETRY_DEFAULT_SAVE_ID: ConsoleDefaultRecoveryAction.RETRY_SAVE,
            CONSOLE_DISCARD_DEFAULT_RETRY_ID: (
                ConsoleDefaultRecoveryAction.DISCARD_RETRY
            ),
            CONSOLE_REFRESH_RUNNING_APP_ID: (
                ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP
            ),
            CONSOLE_DISMISS_DEFAULT_REFRESH_ID: (
                ConsoleDefaultRecoveryAction.DISMISS_REFRESH
            ),
        }
        action = actions.get(event.button.id or "")
        if type(generation) is not int or action is None:
            return
        await self._handle_console_default_recovery(
            ConsoleDefaultRecoveryRequest(
                action=action,
                intent_generation=generation,
            )
        )

    def action_focus_console_composer_home(self) -> None:
        """Return keyboard focus to the Console composer (Escape, non-priority).

        An open slash-command popup claims Escape first and is dismissed
        without moving focus. Deliberately not ``priority=True`` so
        widget-level escapes — transcript
        selection-clear, and any pushed modal's own dismiss binding — are
        resolved first as the key event bubbles up; this screen-level action
        only fires once nothing closer to focus has claimed Escape.
        """
        if self._console_setup_modal_blocking():
            return
        if self._dismiss_console_command_popup():
            return
        self._focus_console_composer_if_needed(force=True)

    def action_new_console_tab(self) -> None:
        """Open a new native Console session tab from the active context (Ctrl+T)."""
        if self._console_setup_modal_blocking():
            return
        self.run_worker(
            self._session._create_native_console_session_from_active_context(),
            exclusive=False,
        )

    def action_new_temporary_console_tab(self) -> None:
        """Open a temporary Console tab: never saved locally.

        Reached via the command palette ("Console: New temporary chat") or
        the tab-strip button -- not a keybinding. An ``alt+t`` chord was
        tried and removed: live verification found it never reached the
        screen when the composer had focus (Textual 8.2.7 treats it as a
        printable key, so the focused ``Input`` consumed it and inserted a
        literal "t" into the draft instead). See
        ``Docs/superpowers/specs/2026-07-31-temporary-conversations-design.md``.

        Born temporary rather than converted: a chat that persists its first
        exchange and is made temporary afterwards has already written rows.
        """
        if self._console_setup_modal_blocking():
            return
        self.run_worker(
            self._session._create_native_console_session_from_active_context(
                ephemeral=True
            ),
            exclusive=False,
        )

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
            self._prompts._open_console_prompt_picker_for_insert(""),
            exclusive=False,
        )

    def action_open_console_style_insert(self) -> None:
        """Open the image-style picker from the command palette ("Insert image style…").

        Mirrors `action_open_console_prompt_insert`'s guard + launch shape.
        The picker only inserts an `@<style-id>` token into the composer
        draft -- `/generate-image @<id> ...` is what later resolves and
        applies the style at generation time; this action never generates
        anything itself.

        F5 (task-9 review): the command it composes is refused at dispatch
        in a temporary chat, so offering this picker there would tease a
        command that can never run. The picker has no "disabled with a
        reason" affordance of its own (it is a modal launch, not a
        control), so this explains itself via a toast instead of opening.
        """
        if self._console_setup_modal_blocking():
            return
        image_blocked = blocked_reason(
            GENERATE_IMAGE_COMMAND_HANDLER_ID,
            ephemeral=self._console_active_session_is_ephemeral(),
        )
        if image_blocked is not None:
            self.app_instance.notify(image_blocked, severity="warning")
            return
        self.run_worker(
            self._open_console_style_picker_for_insert(),
            exclusive=False,
        )

    def action_open_console_system_prompt_editor(self) -> None:
        """Open the system prompt editor from the command palette ("Edit system prompt")."""
        if self._console_setup_modal_blocking():
            return
        self.run_worker(self._open_console_system_prompt_editor(), exclusive=False)

    def action_view_chat_context(self) -> None:
        """Open the Conversation Inspector's Next Send tab (Ctrl+Shift+P).

        task-8: this used to push a standalone modal (retired in task-10);
        it now pushes the shared ``ConsoleConversationInspector`` instead
        (same modal the cost chip opens, just starting on a different
        tab) -- the command-palette "view context" entry
        (``UI/console_command_provider.py``) follows automatically since
        it calls this same action. task-10 wired the Next Send tab to the
        factories built below, so opening from here renders real content.
        """
        if self._console_setup_modal_blocking():
            return
        controller = self._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        if not session_id:
            self.notify("No active conversation.", severity="warning")
            return

        factory, estimate_factory, token_estimate, in_progress = (
            self._console_inspector_next_send_factories(controller, session_id)
        )
        self._push_console_inspector(
            initial_tab=TAB_NEXT_SEND,
            snapshot_factory=factory,
            estimate_factory=estimate_factory,
            token_estimate=token_estimate,
            in_progress=in_progress,
            **project_instruction_ui.project_instruction_context_kwargs(
                self, controller, session_id
            ),
        )

    def _console_inspector_next_send_factories(
        self, controller: Any, session_id: str
    ) -> tuple[
        Callable[[], Awaitable[ConsoleContextSnapshot]],
        Callable[[], int | None],
        int | None,
        bool,
    ]:
        """Build the Next Send tab's snapshot/estimate factories (task-18300).

        Shared by BOTH ``ConsoleConversationInspector`` entry points (the
        cost chip and Ctrl+Shift+P) -- the two push the SAME modal
        instance, and the user can switch to the Next Send tab regardless
        of which tab it opened on, so a caller that skipped this would
        leave the tab showing nothing. Building the closures themselves is
        cheap (no I/O happens until one is actually CALLED); the Next Send
        pane calls ``snapshot_factory`` once on mount regardless of
        ``initial_tab`` (see ``ConsoleConversationInspector.on_mount``).

        ``session_id`` is threaded in rather than re-read from the store
        because the composer only reflects the ACTIVE session; see
        ``_captured_draft``.
        """

        def _captured_draft() -> str:
            if controller.store.active_session_id == session_id:
                try:
                    return self.query_one(
                        "#console-native-composer", ConsoleComposerBar
                    ).draft_text()
                except (NoMatches, QueryError):
                    pass
            session = next(
                (item for item in controller.store.sessions() if item.id == session_id),
                None,
            )
            return session.draft if session is not None else ""

        async def _factory() -> ConsoleContextSnapshot:
            current_draft = _captured_draft()
            pending = controller.store.pending_attachments(session_id)
            current_attachments = tuple(
                MessageAttachment(
                    data=pending_attachment.data,
                    mime_type=pending_attachment.mime_type or "image/png",
                    display_name=pending_attachment.display_name,
                    position=index,
                )
                for index, pending_attachment in enumerate(pending)
            )
            current_staged_sources = controller.store.workspace_context.allowed_sources

            return await controller.build_context_snapshot(
                draft=current_draft,
                attachments=current_attachments,
                staged_sources=current_staged_sources,
                session_id=session_id,
            )

        def _estimate_factory() -> int | None:
            return self._estimate_tokens({"draft": _captured_draft()})

        token_estimate = _estimate_factory()
        in_progress = controller.run_state.status in CONSOLE_ACTIVE_RUN_STATUSES
        return _factory, _estimate_factory, token_estimate, in_progress

    def _estimate_tokens(self, payload: dict[str, Any]) -> int | None:
        """Return a token estimate for the current draft text."""
        text = payload.get("draft", "")
        if not text:
            return None
        return estimate_tokens(text, "", "")

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
        await self._session._activate_native_console_session(sessions[number - 1].id)

    @on(Button.Pressed, "#console-change-workspace")
    def on_console_change_workspace(self, event: Button.Pressed) -> None:
        """Open the active Console workspace switcher."""
        event.stop()
        self._workspace._open_console_workspace_switcher()

    @on(Button.Pressed, "#console-new-temporary-tab")
    def on_console_new_temporary_tab(self, event: Button.Pressed) -> None:
        """Open a temporary Console tab from the tab strip."""
        event.stop()
        self.action_new_temporary_console_tab()

    @on(Button.Pressed, "#console-fleet-coachmark-dismiss")
    def on_console_fleet_coachmark_dismiss(self, event: Button.Pressed) -> None:
        """Dismiss the one-time fleet coach-mark and persist the seen flag."""
        event.stop()
        self._record_console_fleet_coachmark_dismissed()

    def action_open_console_workspace_switcher(self) -> None:
        """Open the workspace switcher (Alt+W / command palette, TASK-722)."""
        self._workspace._open_console_workspace_switcher()

    @on(Button.Pressed, "#console-new-workspace")
    def on_console_new_workspace(self, event: Button.Pressed) -> None:
        """Create a new local workspace from the Console rail and activate it."""
        event.stop()
        self._workspace._create_console_workspace()

    def action_new_console_workspace(self) -> None:
        """Create a local workspace from the command palette (TASK-722)."""
        self._workspace._create_console_workspace()

    # ---- Conversation action menu (TASK-23200) -------------------------

    def _open_console_conversation_action_menu(self, opener: Button) -> None:
        """Anchor the row action menu beneath the pressed asterisk.

        Args:
            opener: The `console-conversation-actions-*` button pressed.
        """
        from tldw_chatbook.Chat.console_conversation_actions import (
            ConversationMenuTarget,
        )
        from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
            ConsoleConversationActionMenu,
            dismiss_conversation_action_menus,
        )

        # Only one menu at a time; a second asterisk replaces the first
        # rather than stacking two overlays the user must dismiss twice.
        dismiss_conversation_action_menus(self)

        conversation_id = (
            str(getattr(opener, "conversation_id", "") or "").strip() or None
        )
        target = ConversationMenuTarget(
            conversation_id=conversation_id,
            title=str(getattr(opener, "conversation_title", "") or ""),
            state=self._console_conversation_state(conversation_id),
            starred=bool(getattr(opener, "starred", False)),
            favorites_available=bool(getattr(opener, "marks_available", True)),
        )
        # Clamp into the screen the same way the transcript's overflow menu
        # does: an asterisk near the bottom of a short terminal would
        # otherwise anchor a menu that runs off the end of the screen.
        region = opener.region
        menu_width = ConsoleConversationActionMenu.MENU_WIDTH
        # Root page is the tallest: five items plus the rounded border.
        menu_height = 7
        screen_region = self.region
        menu = ConsoleConversationActionMenu(
            target=target,
            opener_id=str(opener.id or ""),
            screen_x=max(
                screen_region.x, min(region.x, screen_region.right - menu_width)
            ),
            screen_y=max(
                screen_region.y,
                min(region.bottom, screen_region.bottom - menu_height),
            ),
        )
        self.screen.mount(menu)

    def _console_conversation_state(self, conversation_id: str | None) -> str:
        """Return one conversation's PERSISTED state, or the default.

        Qodo review, PR #2233. The row's ``status`` field was used here, and
        it is display copy, not a database state: rows reach the browser
        carrying ``active``, ``open``, ``workspace`` or ``workspace-thread``
        as well as real states, and first-wins deduplication keeps those
        ahead of the canonical persisted row. Every non-canonical value
        normalises to ``in-progress``, so a *resolved* conversation shown as
        an "active session" row offered **Archive** instead of Unarchive and
        marked the wrong current status.

        The menu asks the database instead. This is one primary-key lookup
        taken when the menu opens, not on every rail render.

        Args:
            conversation_id: Persisted conversation id, or None for a chat
                that has never been saved.

        Returns:
            A canonical state, or the default when it cannot be resolved --
            unsaved chats have no state, and the menu gates their
            state-changing entries anyway.
        """
        from tldw_chatbook.Chat.console_conversation_actions import (
            DEFAULT_CONVERSATION_STATE,
        )

        if not conversation_id:
            return DEFAULT_CONVERSATION_STATE
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return DEFAULT_CONVERSATION_STATE
        try:
            record = db.get_conversation_by_id(conversation_id)
        except Exception as exc:  # noqa: BLE001 - falls back, never blocks the menu
            logger.debug(
                "Console conversation state lookup failed: exception_type={}",
                type(exc).__name__,
            )
            return DEFAULT_CONVERSATION_STATE
        if not record:
            return DEFAULT_CONVERSATION_STATE
        return str(record.get("state") or DEFAULT_CONVERSATION_STATE)

    def on_conversation_action_menu_dismissed(self, event: Message) -> None:
        """Return focus to the asterisk that opened the menu.

        ADR-097: dispatched by Textual's handler-name convention rather than
        `@on(ConversationActionMenuDismissed)`. The decorator needs the class
        at import time, and that single module-level import was enough to pull
        the menu widget and its pure model onto the `_ui_ready` boot path,
        breaching the module-count ratchet. Every other reference to those two
        modules is already lazy, so this keeps the whole feature off the boot
        leg until a user actually opens the menu.
        """
        event.stop()
        if not event.opener_id:
            return
        try:
            self.query_one(f"#{event.opener_id}", Button).focus()
        except (NoMatches, QueryError):
            pass

    def on_conversation_action_chosen(self, event: Message) -> None:
        """Run the chosen row command against the captured conversation.

        Dispatched by handler-name convention -- see
        `on_conversation_action_menu_dismissed` for why.
        """
        from tldw_chatbook.Chat.console_conversation_actions import (
            ACTION_ARCHIVE,
            ACTION_DELETE,
            ACTION_FAVORITE,
            ACTION_RENAME,
            ACTION_UNARCHIVE,
            ACTION_UNFAVORITE,
            ARCHIVED_STATE,
            DEFAULT_CONVERSATION_STATE,
            state_from_action,
        )

        event.stop()
        target = event.target
        action_id = event.action_id
        conversation_id = (target.conversation_id or "").strip()
        if not conversation_id:
            self.app.notify(
                "Send or save this chat before managing it.", severity="warning"
            )
            return

        if action_id in (ACTION_FAVORITE, ACTION_UNFAVORITE):
            self._workspace._toggle_console_conversation_star(
                conversation_id,
                starred=action_id == ACTION_UNFAVORITE,
                conversation_title=target.title,
            )
            return

        if action_id == ACTION_RENAME:
            self._workspace.open_console_conversation_rename(
                conversation_id, target.title
            )
            return

        if action_id == ACTION_DELETE:
            self._workspace.confirm_console_conversation_delete(
                conversation_id, target.title
            )
            return

        new_state = state_from_action(action_id)
        if action_id == ACTION_ARCHIVE:
            new_state = ARCHIVED_STATE
        elif action_id == ACTION_UNARCHIVE:
            new_state = DEFAULT_CONVERSATION_STATE
        if new_state is None:
            return
        self._workspace.set_console_conversation_state(
            conversation_id, new_state, conversation_title=target.title
        )


    @on(Button.Pressed, "#console-workspace-rag-scope-open")
    def on_console_workspace_rag_scope_open(self, event: Button.Pressed) -> None:
        """Open the workspace-level RAG retrieval-scope picker (task-13).

        Args:
            event: The button-press event from the workspace row's
                "Scope" button (``#console-workspace-rag-scope-open``).
        """
        event.stop()
        self.run_worker(
            self._workspace._open_console_workspace_scope_picker(),
            exclusive=True,
            group="console-workspace-scope-open",
        )

    # Reactive property for sidebar state persistence
    sidebar_state = reactive(dict, layout=False)

    #: task-15475: one-shot "the mount already did this visit's refreshes"
    #: token. Textual posts ``ScreenResume`` when a screen is PUSHED, so
    #: ``on_mount`` and the mount's own ``on_screen_resume`` both fire on the
    #: first visit -- and both used to dispatch the (non-exclusive, so
    #: uncancelled) skill-candidate worker and both used to sync task-resume
    #: state. ``on_mount`` sets this; the very next resume consumes it and
    #: skips. Every LATER resume still refreshes, which is the point: a skill
    #: may have been installed, or an approval may have landed, while Console
    #: was suspended.
    #:
    #: Class-level default so it is readable on a screen built by ``__new__``
    #: (several Console tests construct one without running ``__init__``).
    _console_mount_visit_refreshed: bool = False
    _pending_resume_local_conversation_id: str | None = None
    _resume_navigation_startup_in_progress: bool = False

    def apply_navigation_context(self, context: Mapping[str, object]) -> None:
        """Capture one valid saved-conversation ID for post-mount resume.

        Args:
            context: Memory-only navigation values supplied for this screen visit.
        """
        value = context.get(CONSOLE_NAV_CONTEXT_RESUME_LOCAL_CONVERSATION_ID)
        if not isinstance(value, str):
            return
        conversation_id = value.strip()
        if (
            not conversation_id
            or len(conversation_id) > _RESUME_LOCAL_CONVERSATION_ID_MAX_LENGTH
        ):
            return
        self._pending_resume_local_conversation_id = conversation_id

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "chat", **kwargs)
        self.console_session_surface: Optional[ConsoleSessionSurface] = None
        self._task_resume_state = TaskResumeState()
        self._watchlists_operation_rows: dict[str, dict[str, Any]] = {}
        self._console_composer_collapsed = False
        self._console_composer_layout_revision = 0
        self._console_status_chips_collapsed = resolve_status_chips_collapsed(
            getattr(app_instance, "app_config", None)
        )
        self._console_status_chips_layout_revision = 0
        self._state_dirty = False
        self._console_settings_coordinated_submission_ids: deque[str] = deque(maxlen=64)
        self._handoff_consumption_in_progress = False
        self._pending_console_launch_context: Optional[ConsoleLiveWorkLaunch] = None
        self._pending_console_launch_auto_open_inspector = False
        # PR-4/task-1: source count of the evidence the LAST send consumed,
        # kept only until the next thing happens (a new send, new staging, or
        # an un-stage). It drives the strip's one-send "Evidence sent with
        # this message" line, which is the only round-trip confirmation an
        # unpersisted session gets -- the transcript's "Sources (N)" row
        # needs a persisted conversation. Deliberately NOT a timer: a timed
        # clear races the strip's own recompose.
        self._console_evidence_sent_notice: Optional[int] = None
        # TASK-259: dedupe guard for the scheduled inspector-rail card swap
        # (rapid searching->staged staging would otherwise remove+remount
        # the card once per stage; each swap re-reads the current context).
        self._console_live_work_card_swap_scheduled = False
        self._console_control_provider: Optional[Any] = None
        self._console_control_model: Optional[Any] = None
        self._console_library_rag_query = ""
        # RAG-44: which KINDS of Library sources "Run Library RAG" reads.
        # Console-local (the Library screen keeps its own screen-local
        # selection; neither is promoted to shared state), editable in the
        # RAG settings modal, and restored with the rest of the native
        # Console screen state. Defaults to today's three -- prompts OFF.
        self._console_library_rag_source_types: tuple[str, ...] = (
            CONSOLE_LIBRARY_RAG_SOURCE_SCOPE
        )
        # `_console_chat_store` is a PROPERTY over the app-owned runtime
        # (task-15860 lifetime landing) -- it deliberately has no `__init__`
        # slot any more. Assigning `None` here would clear the store a
        # surviving runtime is still using the instant a fresh ChatScreen is
        # constructed, which `_complete_screen_navigation` does BEFORE the
        # outgoing screen unmounts.
        self._last_console_roleplay_refresh_key: tuple[str, str] | None = None
        self._console_roleplay_persistence_task: asyncio.Task[None] | None = None
        self._console_roleplay_writer_task: (
            asyncio.Task[ConsoleRoleplayProjectionPersistenceResult | None] | None
        ) = None
        self._console_roleplay_active_plan: (
            ConsoleRoleplayProjectionPersistencePlan | None
        ) = None
        self._console_roleplay_pending_plan: (
            ConsoleRoleplayProjectionPersistencePlan | None
        ) = None
        self._console_roleplay_drain_scheduled = False
        self._console_roleplay_tearing_down = False
        self._console_roleplay_repair_generation = 0
        self._console_roleplay_repair_inflight_generation = 0
        self._console_roleplay_repair_plan: (
            ConsoleRoleplayProjectionPersistencePlan | None
        ) = None
        self._console_identity_refresh_generation = 0
        self._console_appearance_refresh_generation = 0
        # TASK-340: keyboard-send draft stashes — keypress->handler handoff,
        # then the queued submit's accept/refuse consumption slot.
        # `_console_pending_send_stash` stays a single slot: it is consumed
        # within the same keypress -> Button.press() handoff for whichever
        # composer currently has focus (bounded to one UI action, never
        # spans a provider round-trip), unlike the map below.
        self._console_pending_send_stash: ConsoleDraftStash | None = None
        # Task 3b: PER-SESSION -- a keyboard send's stash is written at
        # dispatch (keyed by the dispatching session) and read/cleared much
        # later, at that SAME session's own accept/refuse (`_notify_
        # submission_accepted` fires only after the provider-readiness
        # probe/skill-substitution awaits, which can run for seconds). A
        # single shared slot let a DIFFERENT session's concurrent dispatch
        # clobber this one's entry mid-flight (Task 3 made that genuinely
        # concurrent) -- e.g. session A's still-pending stash getting
        # silently replaced by session B's `None`, or a stale entry
        # restoring/clearing the WRONG session's composer. See
        # `_console_submit_session_by_task` for how the no-arg
        # `on_submission_accepted` hook still resolves its own session.
        self._console_inflight_send_stashes: dict[str, ConsoleDraftStash] = {}
        #: `asyncio.Task -> owning session id`, registered for the duration
        #: of `_submit_console_native_draft`'s own `await controller.
        #: submit_draft(...)` call so the no-arg `on_submission_accepted`
        #: callback (fired synchronously from deep inside that same await,
        #: on the SAME task) can resolve which session's stash entry above
        #: is its own, without changing that hook's public no-arg contract.
        self._console_submit_session_by_task: dict[asyncio.Task, str] = {}
        # TASK-1141: round/request ids (namespaced "mcp:<round_id>" /
        # "install:<request_id>" / "script:<request_id>") this screen has
        # already fired a park toast for -- see `_park_console_approval`'s
        # docstring for the re-invocation hazard this guards against. Never
        # pruned: entries are one-off UUIDs minted per approval-like round,
        # so this set's steady-state size is bounded by "how many rounds
        # this screen instance has EVER parked", not by anything unbounded
        # over a session's lifetime.
        self._console_toasted_park_round_ids: set[str] = set()
        #: TASK-1141 review round 1: the LAST non-empty snapshot of
        #: `_current_park_round_ids(controller, session_id)`, per session
        #: id -- the fallback identity `_park_console_approval` consults
        #: when a re-invocation arrives AFTER the round's own teardown has
        #: already popped it from every live `_parked_*_payloads` map (so
        #: `_current_park_round_ids` alone reads back empty and can no
        #: longer distinguish "a stray post-teardown re-announcement of a
        #: round already toasted" from "a session this screen has never
        #: parked anything for"). Overwritten (not merged) on every
        #: non-empty snapshot -- only the MOST RECENT live round/request
        #: id set for a session is a useful fallback key, since anything
        #: older has already been superseded or resolved. Never pruned,
        #: for the same "bounded by distinct sessions ever parked" reason
        #: as `_console_toasted_park_round_ids` above.
        self._console_last_parked_round_ids: dict[str, frozenset[str]] = {}
        # TASK-251: last-applied payload for the equality-guarded Agent
        # rail sub-sync (skip Static.update()/style work when the payload
        # `ConsoleAgentController._console_agent_section_payload` computes
        # hasn't changed since the last successful apply). Stays here, with
        # `_sync_console_agent_section` itself -- it is that DOM write's
        # memo and nothing else's (wave-4 console decomposition, task 3).
        self._console_agent_section_last: (
            tuple[
                str,
                str,
                ConsoleInspectorSectionState,
                str,
                bool,
                bool,
                bool,
                ConsoleAgentSteeringState,
            ]
            | None
        ) = None
        # PR2b Task 5: coalescing flag for `_request_console_agent_fleet_
        # sync` -- mirrors `_console_control_bar_sync_scheduled` exactly
        # (see that flag's own call site, `_request_console_control_bar_
        # sync`, for the precedent this follows).
        self._console_agent_fleet_sync_scheduled = False
        self._console_rail_system_line_last: tuple[str, bool] | None = None
        self._console_rail_prune_dispatched = False
        # The six Console controllers -- their construction and every
        # named dependency they take -- moved verbatim to
        # `Console_Modules/wiring.py` (wave-4 console decomposition,
        # task 1); not one keyword argument changed, only the call
        # site. This call sits at exactly the point the first
        # construction (`self._workspace`) occupied, because some of
        # the ~250 attribute assignments around it here are read by the
        # wiring's late-binding lambdas. See `build_console_
        # controllers`' docstring for the build order and why it is
        # documentation rather than a constraint.
        build_console_controllers(
            self,
            rag_source_types_accessor=(lambda: _console_library_rag_source_scope(self)),
            rag_top_k_accessor=lambda: _console_library_rag_profile_top_k(),
        )
        # `_console_provider_gateway`/`_console_chat_controller`: properties
        # over the app-owned runtime, no `__init__` slot -- see the note at
        # `_console_chat_store`'s old slot.
        self._console_command_registry: ConsoleCommandRegistry = (
            default_console_registry()
        )
        self._console_unknown_send_armed: str | None = None
        self._console_image_view_state: ConsoleImageViewState | None = None
        self._console_image_cache: ConsoleImageRenderCache | None = None
        self._console_image_default_mode: Literal["pixels", "graphics"] | None = None
        self._console_image_preparing: set[str] = set()
        self._console_model_option_warnings: dict[tuple[str, str], str] = {}
        # Message action, preview, speaking, and swipe state lives on
        # `self._message`; compatibility properties below proxy that owner.
        self._console_transcript_sync_timer: Any | None = None
        # Cost-ticker PR3 (task-5): the 10s WARM->EXPIRED repaint timer --
        # mirrors `_console_transcript_sync_timer` (started/stopped via the
        # `_record_ui_timer_created/_stopped("console-cost-ttl")` audit
        # pair, stopped in `on_unmount`) but on its own cadence, since the
        # 0.2s tick above stops as soon as a run leaves an active status.
        self._console_cost_ttl_timer: Any | None = None
        self._console_sync_in_progress = False
        self._console_sync_requested = False
        self._console_citation_counts: dict[str, int] = {}
        # Same precedent, same reason (task-18515 review-note management
        # task 3 fix round): a rapid double marker-click / double-`n` before
        # the first worker's off-thread DB read resolves must not stack two
        # ConsoleReviewNotesModals with independent DB-bound closures.
        self._console_review_notes_inflight = False
        self._console_citation_resolved_signatures: dict[
            str, tuple[str, str, str, str]
        ] = {}
        self._console_citation_repository_token: tuple[str, int, int, int] | None = None
        self._console_citation_input_signature: (
            tuple[str | None, tuple[tuple[str, str, str, str], ...]] | None
        ) = None
        self._console_citation_request_generation = 0
        self._last_native_transcript_refresh_key: tuple[int, tuple[Any, ...]] | None = (
            None
        )
        self._last_console_workbench_focus_id: str | None = None
        self._last_console_control_state: ConsoleControlState | None = None
        self._last_console_workbench_state: Any | None = None
        # Cost-ticker PR3 (task-5). `_last_console_cost_state` mirrors
        # `_last_console_control_state` above: the last pushed
        # `ConsoleCostState`, so `_sync_console_cost_chip` only re-renders
        # the chip on an actual change. `_console_cost_cache_state` holds
        # the raw `ConsoleCacheState` enum the last build computed --
        # `ConsoleCostState` itself only exposes `alert`/`cold` (a WARM
        # cache with no break reason renders identically to no cache
        # activity at all), so the TTL timer's start/stop decision needs
        # this separately. `_console_cost_fp_revisions`/
        # `_console_cost_break_reasons` memoize the last payload-fingerprint
        # comparison per session id (`!=`, not `>` -- `ConsoleChatStore.
        # restore_state` can reset a session's revision counter back down)
        # so the (relatively expensive) fingerprint recompute only runs
        # when the session's payload has actually changed since the last
        # check, and never while a run is actively streaming.
        self._last_console_cost_state: ConsoleCostState | None = None
        self._console_cost_cache_state: ConsoleCacheState = ConsoleCacheState.NONE
        self._console_cost_fp_revisions: dict[str, int] = {}
        self._console_cost_break_reasons: dict[str, str | None] = {}
        self._last_console_rail_state: ConsoleRailState | None = None
        # TASK-2154.1: last width band seen by the resize hook; a rebuild of
        # the (expensive) effective rail state only happens when the band
        # actually changes, not on every pixel of a resize drag.
        self._last_console_workspace_width_band: str | None = None
        self._console_guidance_dismissed = False
        self._console_first_send_completed_cached: bool | None = None
        # Fleet-UX expert review F2 (task-1232): one-time coach-mark shown
        # the first time the session count actually TRANSITIONS to 2 (not
        # merely "is 2" -- a restore that lands the store at 2+ sessions on
        # first sync must not misfire as a "creation" event). `None` means
        # "not seeded yet"; seeded from the count observed on this screen's
        # first sync tick.
        self._last_console_session_count: int | None = None
        self._console_fleet_coachmark_seen_cached: bool | None = None
        self._console_detected_local_server: DiscoveredLocalServer | None = None
        self._console_local_discovery_started = False
        # TASK-22305: one card-level destructive operation at a time. The
        # identity token prevents a late cancel/worker landing from releasing
        # a newer operation after a session switch or modal replacement.
        self._console_turn_undo_token: object | None = None
        # P1g Task 5: guards the Console dictionary attach/detach picker
        # flow against a double-open (mirrors P1f's `_io_dialog_active`),
        # reset in a `finally` in both attach/detach workers.
        self._console_dictionary_dialog_active = False
        # P2g-2 Task 4: same double-open guard, for the World Books
        # inspector block's Attach/Detach picker flow.
        self._console_worldbook_dialog_active = False
        self.ui_state = UIState()
        self._load_sidebar_state()
        # task-15470: debounce state for `watch_sidebar_state` -- see
        # `SIDEBAR_STATE_SAVE_DEBOUNCE_SECONDS`.
        self._sidebar_state_save_timer: Any | None = None
        self._sidebar_state_dirty = False
        self._sidebar_state_persist_worker: Any | None = None

    # Sections `load_settings()` always injects into a disk-loaded config but
    # which Console test fakes never carry. Used to tell a real boot snapshot
    # (safe to refresh from disk) apart from an injected test config (must be
    # honored verbatim; reading the developer's real config would break
    # hermetic tests). NOTE: verified against real `load_settings()` output on
    # a virgin template config - do not add keys (e.g. `splash_screen`) that
    # only `load_cli_config_and_ensure_existence()` emits, or the live app
    # never takes the fresh branch.
    _CONSOLE_LIVE_CONFIG_MARKER_SECTIONS = ("general", "logging")

    #: Memo for ONE synchronous Console state-derivation pass, or None when
    #: no pass is open. A CLASS attribute default because the hand-built
    #: `ChatScreen.__new__()` test fixtures never run `__init__`.
    _console_derivation_memo: dict[Any, Any] | None = None

    #: Cross-tick memo for the cost chip's per-row token estimates
    #: (task-15451), lazily created by `_console_cost_estimate_cache_or_new`.
    #: Unlike `_console_derivation_memo` this deliberately OUTLIVES a single
    #: pass -- the whole point is that the 0.2s tick stops re-tokenizing rows
    #: it already estimated on the previous tick. Also a CLASS attribute
    #: default, for the same `__new__()`-fixture reason.
    _console_cost_estimate_cache: TokenEstimateCache | None = None

    def _console_cost_estimate_cache_or_new(self) -> TokenEstimateCache:
        """Return this screen's token-estimate memo, creating it on first use.

        Held per screen rather than per session: entries are keyed by message
        id and every hit is re-verified against the row's own text, so the
        two sessions of a switched-between pair share the cache safely (see
        :class:`TokenEstimateCache`).
        """
        cache = self._console_cost_estimate_cache
        if cache is None:
            cache = TokenEstimateCache()
            self._console_cost_estimate_cache = cache
        return cache

    @contextmanager
    def _console_derivation_scope(self):
        """Memoize repeated provider lookups for one derivation pass.

        Building the Console control state and the Workbench state off it
        re-derives the same provider selection and readiness config once
        per leg: measured on dev, a single draft-edit sync ran
        `_build_console_provider_selection` 7x and
        `_provider_readiness_app_config` 63x (task-15452). None of those
        legs mutate anything, and the pass is synchronous, so one memo for
        its duration is exact rather than merely close.

        Deliberately opt-in and scoped: outside a `with` block every lookup
        is live, exactly as before. Re-entrant (an inner scope keeps the
        outer memo) and always torn down, so a raising leg cannot leave a
        stale selection cached for the next keystroke.
        """
        if self._console_derivation_memo is not None:
            yield
            return
        self._console_derivation_memo = {}
        try:
            yield
        finally:
            self._console_derivation_memo = None

    def _provider_readiness_app_config(self) -> Any:
        """Return the freshest app config for provider-readiness checks.

        ``app.app_config`` is a boot-time snapshot: Settings saves invalidate
        the config module cache but never refresh the snapshot, so readiness
        built from it stays blocked until restart (core-loop UAT 2026-07,
        task-177). When the snapshot looks disk-loaded, re-source it from
        ``load_settings()`` - cheap (cached) except right after a save, which
        is exactly when the fresh read matters.

        Served from the per-pass memo inside a `_console_derivation_scope`
        (task-15452): one draft-edit sync called this 63 times.
        """
        memo = self._console_derivation_memo
        if memo is not None and "app_config" in memo:
            return memo["app_config"]
        try:
            app_config = getattr(self.app, "app_config")
        except (AttributeError, NoActiveAppError):
            app_config = getattr(self.app_instance, "app_config", {}) or {}
        app_config = app_config or {}
        resolved = app_config
        if self._console_config_snapshot_is_disk_loaded(app_config):
            try:
                fresh = load_settings()
            except Exception:
                logger.debug(
                    "Console readiness refresh via load_settings() failed; "
                    "using snapshot"
                )
            else:
                if isinstance(fresh, Mapping) and fresh:
                    resolved = fresh
        if memo is not None:
            memo["app_config"] = resolved
        return resolved

    @classmethod
    def _console_config_snapshot_is_disk_loaded(cls, app_config: Any) -> bool:
        """Return True when a config snapshot came from ``load_settings()``."""
        if not isinstance(app_config, Mapping):
            return False
        return all(
            section in app_config
            for section in cls._CONSOLE_LIVE_CONFIG_MARKER_SECTIONS
        )

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
        """Accept one-shot live-work launch context from another destination.

        PR-T1/task-3 (D3): the resident-launch branch below is also what
        makes a launch restored by `_restore_native_console_state` (a
        tab-switch survivor, not a fresh handoff) safe to re-enter here.
        `restore_state` runs BEFORE this screen is ever composed/mounted
        (see `TldwCli._complete_screen_navigation`), so by the time
        `compose_content()` calls this method, a restored launch has already
        set `_pending_console_launch_context` to a non-`None` value.

        PR-T1 final review (C1): "resident wins, full stop" was WRONG once
        D3 made a launch survive navigation. Real flow: stage A from
        Library -> go back to Library (`save_state` persists A) -> stage B
        ("Use in Console") -> navigate to Console. `restore_state` runs
        BEFORE compose (`app.py` restore precedes the compose call), so the
        resident A was returned here and B stayed unclaimed in the store:
        B never displayed (the click looked dead), the next send consumed
        A, and the send AFTER that claimed B deep inside
        `_console_send_blocked_reason`/`_capture_console_staged_rag` and
        fed it to an unrelated message as evidence the user had never seen.
        A RAG-labelled B with zero available references was worse still: it
        BLOCKED an unrelated send citing evidence that was never on screen.

        The rule is now "a fresh explicit user action supersedes a stale
        survivor": when a resident launch coexists with an unclaimed store
        entry, the store entry is claimed and staged through
        `_stage_console_library_rag_launch`, which clears the previous
        send's "evidence sent" notice and syncs every mounted surface --
        so a claim can never be invisible, and in particular a claim can
        never first become live inside a send gate.
        """
        if self._pending_console_launch_context is not None:
            self._supersede_resident_console_launch_from_store()
            return self._pending_console_launch_context

        store = self.app_instance.pending_handoffs
        claim = store.claim(HandoffChannel.CONSOLE_LIVE_WORK)
        if claim is None:
            return self._pending_console_launch_context
        try:
            self._pending_console_launch_context = claim.value
            self._pending_console_launch_auto_open_inspector = True
        except Exception as exc:
            store.release(claim)
            logger.warning(
                "Console live-work handoff transfer failed "
                "(channel={}, revision={}, exception_category={})",
                claim.channel.value,
                claim.revision,
                type(exc).__name__,
            )
            return self._pending_console_launch_context
        store.acknowledge(claim)
        return self._pending_console_launch_context

    def _supersede_resident_console_launch_from_store(self) -> None:
        """Let a freshly staged handoff replace an already-resident launch.

        Only ever called from `_consume_pending_console_launch` with a
        non-`None` resident launch. `has_pending` is checked first so the
        overwhelmingly common case (a resident launch, an empty channel)
        costs one cheap slot read and touches nothing.

        Failure containment: the resident context is repointed at the new
        launch BEFORE anything fallible runs, and the claim is acknowledged
        (never released) once that assignment has happened. Releasing after
        the screen already owns the value would leave the same launch both
        resident AND pending -- the exact double-delivery this method
        exists to end. The surface refresh inside
        `_stage_console_library_rag_launch` is the only fallible step, and
        a failure there costs a stale chip, not lost or invisible evidence.
        """
        store = getattr(self.app_instance, "pending_handoffs", None)
        if store is None or not store.has_pending(HandoffChannel.CONSOLE_LIVE_WORK):
            return
        claim = store.claim(HandoffChannel.CONSOLE_LIVE_WORK)
        if claim is None:
            return
        launch = claim.value
        # Non-fallible ownership transfer first, then settle the claim.
        self._pending_console_launch_context = launch
        self._console_evidence_sent_notice = None
        # A superseding launch IS a fresh handoff, so it earns the
        # auto-open-once Inspector behavior; set BEFORE staging, because
        # staging syncs the rail state synchronously (same ordering as
        # every other `_stage_console_library_rag_launch` caller).
        self._pending_console_launch_auto_open_inspector = True
        store.acknowledge(claim)
        try:
            self._retrieval._stage_console_library_rag_launch(
                launch, allow_recompose=False
            )
        except Exception as exc:
            # Includes the never-composed screen shell, where the staging
            # seam's surface sync has no DOM to query.
            logger.warning(
                "Console live-work supersede surface refresh failed "
                "(channel={}, revision={}, exception_category={})",
                claim.channel.value,
                claim.revision,
                type(exc).__name__,
            )

    def _chat_default_value(self, key: str) -> Any:
        """Return a shared Console default value from app configuration."""
        config = getattr(self.app_instance, "app_config", {}) or {}
        defaults = config.get("chat_defaults", {}) if isinstance(config, dict) else {}
        return defaults.get(key) if isinstance(defaults, dict) else None

    def _console_background_effect_settings(self) -> ConsoleBackgroundEffectSettings:
        """Return normalized Console transcript background effect settings."""
        config = getattr(self.app_instance, "app_config", {}) or {}
        console = config.get("console", {}) if isinstance(config, dict) else {}
        background = (
            console.get("background_effects", {}) if isinstance(console, dict) else {}
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
            self._persisted_chat_defaults(),
            console_provider=self._console_control_provider,
            console_model=self._console_control_model,
        )
        return effective.provider, effective.model

    def _persisted_chat_defaults(self) -> Mapping[str, Any]:
        """Return the freshest persisted provider/model defaults."""
        config = self._provider_readiness_app_config()
        if not isinstance(config, Mapping):
            return {}
        defaults = config.get("chat_defaults", {})
        return defaults if isinstance(defaults, Mapping) else {}

    @staticmethod
    def _normalize_llamacpp_base_url(api_url: str | None) -> str:
        """Return the llama.cpp origin root used before appending OpenAI paths."""
        return normalize_llamacpp_base_url(api_url) or DEFAULT_LLAMACPP_BASE_URL

    @staticmethod
    def _config_section(config: dict[str, Any], key: str) -> dict[str, Any]:
        value = config.get(key, {})
        return value if isinstance(value, dict) else {}

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
            logger.debug(
                "Unable to load CLI provider/model registry for Console settings"
            )
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
                providers_models,
                getattr(
                    self.app_instance,
                    "llm_provider_catalog_scope_service",
                    None,
                ),
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
            self._console_model_option_warnings[(provider_key, model_id)] = (
                option.warning
            )

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

    def _configured_console_provider(
        self,
        provider: str,
    ) -> tuple[str, list[str]] | None:
        """Resolve a normalized intent against configured provider identities."""
        requested_key = provider_config_key(provider)
        for configured_provider, configured_models in self._providers_models().items():
            if provider_config_key(configured_provider) != requested_key:
                continue
            models = [
                str(model).strip()
                for model in configured_models
                if str(model or "").strip()
                and str(model).strip().lower() not in {"none", "null"}
            ]
            return requested_key, models
        return None

    def _configured_console_provider_default_model(
        self,
        provider: str,
        models: list[str],
    ) -> str | None:
        """Return a valid configured default model for one provider."""
        config = self._provider_readiness_app_config()
        api_settings = (
            config.get("api_settings", {}) if isinstance(config, Mapping) else {}
        )
        provider_settings: Mapping[str, Any] = {}
        if isinstance(api_settings, Mapping):
            for configured_provider, configured_settings in api_settings.items():
                if provider_config_key(str(configured_provider)) != provider:
                    continue
                if isinstance(configured_settings, Mapping):
                    provider_settings = configured_settings
                break
        candidates = (
            provider_settings.get("model"),
            provider_settings.get("api_model"),
            provider_settings.get("default_model"),
        )
        for candidate in candidates:
            model = str(candidate or "").strip()
            if model and model in models:
                return model

        defaults = self._persisted_chat_defaults()
        if provider_config_key(str(defaults.get("provider") or "")) == provider:
            default_model = str(defaults.get("model") or "").strip()
            if default_model and default_model in models:
                return default_model
        return models[0] if models else None

    def _apply_console_provider_intent(
        self,
        intent: ConsoleProviderIntent,
        *,
        store: ConsoleChatStore,
        session_id: str,
        settings: ConsoleSessionSettings,
    ) -> bool:
        """Apply one validated intent to the session captured by its consumer."""
        configured = self._configured_console_provider(intent.provider)
        if configured is None:
            self.app_instance.notify(
                "That provider is unavailable. Choose a configured provider in Settings.",
                severity="warning",
            )
            return False

        provider, models = configured
        model = self._configured_console_provider_default_model(provider, models)
        derived = build_default_console_session_settings(
            self._provider_readiness_app_config(),
            provider,
            model,
        )
        next_settings = replace(
            settings,
            provider=provider,
            model=model,
            base_url=derived.base_url,
            source="user",
        )
        store.replace_session_settings(session_id, next_settings)
        if store.active_session_id == session_id:
            self._console_control_provider = next_settings.provider
            self._console_control_model = next_settings.model
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            self._sync_console_control_bar()
        self.app_instance.notify(
            f"Console provider set to {provider} for this session.",
            severity="information",
        )
        return True

    def consume_pending_console_provider_intent(self) -> bool:
        """Consume one typed provider intent after the Console session is ready."""
        try:
            store = self._ensure_console_chat_store()
            settings = self._session._ensure_active_console_session_settings()
            session_id = store.active_session_id
            if session_id is None:
                return False
        except Exception as exc:
            logger.warning(
                "Console provider handoff is not ready (exception_category={})",
                type(exc).__name__,
            )
            return False

        claim = self.app_instance.pending_handoffs.claim(
            HandoffChannel.CONSOLE_PROVIDER
        )
        if claim is None:
            return False
        try:
            if not isinstance(claim.value, ConsoleProviderIntent):
                raise TypeError("Console provider handoff was not typed")
            self._apply_console_provider_intent(
                claim.value,
                store=store,
                session_id=session_id,
                settings=settings,
            )
        except Exception as exc:
            self.app_instance.pending_handoffs.release(claim)
            logger.warning(
                "Console provider handoff will retry "
                "(channel={}, revision={}, exception_category={})",
                claim.channel.value,
                claim.revision,
                type(exc).__name__,
            )
            self.app_instance.notify(
                "Console provider selection could not be applied yet; it will retry.",
                severity="warning",
            )
            return False
        self.app_instance.pending_handoffs.acknowledge(claim)
        return True

    def current_console_provider_for_command(self) -> str | None:
        """Return the active session provider without creating a session."""
        settings = self._session._active_console_session_settings()
        if settings is None:
            return None
        return str(settings.provider or "").strip() or None

    def _active_console_settings_context_estimate(
        self,
    ) -> ConsoleSettingsContextEstimate:
        """Return context usage for the active native Console settings snapshot."""
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            settings = self._session._ensure_active_console_session_settings()
            return build_console_context_estimate(
                [],
                settings.provider,
                settings.model,
                max_tokens_response=settings.max_tokens,
                system_prompt=settings.system_prompt,
            )
        return self._console_settings_context_estimate_for_session(session_id)

    def _console_settings_context_estimate_for_session(
        self,
        session_id: str,
        *,
        settings: ConsoleSessionSettings | None = None,
    ) -> ConsoleSettingsContextEstimate:
        """Return settings context derived from one captured session only."""
        store = self._ensure_console_chat_store()
        settings = settings or store.session_settings(session_id)
        if settings is None:
            raise KeyError(session_id)
        include_active_staging = store.active_session_id == session_id
        workspace_context = (
            self._workspace._current_console_workspace_context()
            if include_active_staging
            else None
        )
        pending_launch = (
            self._pending_console_launch_context if include_active_staging else None
        )
        staged_context_state = self._build_console_staged_context_state(pending_launch)
        messages: list[dict[str, str]] = []
        try:
            messages = [
                {
                    "role": str(
                        message.role.value
                        if hasattr(message.role, "value")
                        else message.role
                    ),
                    "content": message.content,
                }
                for message in store.messages_for_session(session_id)
            ]
        except KeyError:
            messages = []
        return build_console_context_estimate(
            messages,
            settings.provider,
            settings.model,
            staged_source_count=(
                len(workspace_context.staged_sources)
                if workspace_context is not None
                else 0
            ),
            staged_context_summary=staged_context_state.summary,
            max_tokens_response=settings.max_tokens,
            system_prompt=settings.system_prompt,
            # task-6: staged evidence used to move only the label's "; N
            # sources staged" suffix (`staged_source_count` above) while
            # `used_tokens` silently reported zero for content the send
            # is likely to carry. `console_prompted_evidence_text` reads
            # the same in-memory, zero-I/O staged bundle and produces the
            # formatted pre-authority estimate
            # `_current_console_workspace_context` already parses above --
            # no extra DB round trip. The actual send may shrink this after
            # its authority check.
            staged_text=console_prompted_evidence_text(pending_launch),
        )

    def _active_console_context_control_state(
        self,
        *,
        estimate: ConsoleSettingsContextEstimate | None = None,
        thinking_history_effective_policy: str | None = None,
    ) -> ConsoleContextControlState:
        """Build the shared quick/full context snapshot for the active session."""
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is None:
            settings = self._session._ensure_active_console_session_settings()
            estimate = estimate or self._active_console_settings_context_estimate()
            return build_console_context_control_state(
                settings=settings,
                estimate=estimate,
                overrides=ConsoleContextPolicyOverrides(),
                global_overrides=None,
                active_memory=None,
            )
        return self._console_context_control_state_for_session(
            session_id,
            estimate=estimate,
            thinking_history_effective_policy=thinking_history_effective_policy,
        )

    def _console_context_control_state_for_session(
        self,
        session_id: str,
        *,
        estimate: ConsoleSettingsContextEstimate | None = None,
        settings: ConsoleSessionSettings | None = None,
        thinking_history_effective_policy: str | None = None,
    ) -> ConsoleContextControlState:
        """Build context controls from one captured session binding."""
        store = self._ensure_console_chat_store()
        settings = settings or store.session_settings(session_id)
        if settings is None:
            raise KeyError(session_id)
        estimate = estimate or self._console_settings_context_estimate_for_session(
            session_id,
            settings=settings,
        )
        overrides = ConsoleContextPolicyOverrides()
        global_overrides = None
        memory = None
        controller = self._ensure_console_chat_controller()
        try:
            overrides, global_overrides, memory = controller.context_control_inputs(
                session_id
            )
        except (KeyError, ValueError):
            pass
        return build_console_context_control_state(
            settings=settings,
            estimate=estimate,
            overrides=overrides,
            global_overrides=global_overrides,
            active_memory=memory,
            thinking_history_policy=(
                store.session_thinking_history_policy(session_id)
                if session_id is not None
                else "auto"
            ),
            thinking_history_effective_policy=thinking_history_effective_policy,
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
        settings = self._session._ensure_active_console_session_settings()
        line_text = build_console_rail_system_line(settings.system_prompt)
        # TASK-365: this rail line is clickable (opens the system-prompt editor)
        # but otherwise reads as inert label text like the Provider/Model lines
        # above it. A trailing affordance (the same ▸ the rail uses elsewhere for
        # interactive controls) marks it as the one actionable row in the section.
        line_text = f"{line_text} {CONSOLE_RAIL_SYSTEM_EDIT_AFFORDANCE}"
        is_dim = not str(settings.system_prompt or "").strip()
        return line_text, is_dim

    def _sync_console_rail_system_line(self) -> None:
        """Targeted update of the mounted rail ``System:`` line, no recompose.

        TASK-251: equality-guarded -- the 0.2s tick called this
        unconditionally, forcing a ``Static.update()`` even when the system
        prompt hadn't changed since the last apply.
        """
        payload = self._console_rail_system_line_state()
        if payload == self._console_rail_system_line_last:
            return
        try:
            system_line = self.query_one("#console-rail-system-line", Static)
        except (NoMatches, QueryError):
            return
        line_text, is_dim = payload
        system_line.update(line_text)
        system_line.set_class(is_dim, "console-rail-system-line-dim")
        self._console_rail_system_line_last = payload
        self._request_console_context_allocation_reconcile()

    def _request_console_context_allocation_reconcile(self) -> None:
        """Safely invalidate the mounted Context allocator after a DOM mutation."""

        try:
            left_rail = self.query_one("#console-left-rail", ConsoleLeftRail)
        except (NoMatches, QueryError):
            return
        left_rail.request_allocation_reconcile()

    def _request_console_live_work_reconcile(self) -> None:
        """Settle the swapped Live Work body before its Inspector owner."""

        try:
            section = self.query_one(
                "#console-bounded-section-live-work", ConsoleBoundedSection
            )
            rail = self.query_one("#console-right-rail", ConsoleInspectorRail)
        except (NoMatches, QueryError):
            return
        section.request_reconcile()
        rail.request_outer_reconcile()

    def _sync_console_settings_summary(self) -> None:
        """Refresh the mounted Console settings summary surfaces if present."""
        summary_state = self._build_console_settings_summary_state()
        try:
            summary = self.query_one(
                "#console-settings-summary", ConsoleSettingsSummary
            )
        except (NoMatches, QueryError):
            pass
        else:
            # The child owns its bounded-body and rail invalidation.
            summary.sync_state(summary_state)
        provider_value = _summary_row_value(summary_state.provider_row) or "—"
        model_value = _summary_row_value(summary_state.model_row) or "—"
        temperature_match = re.search(r"T ([\d.]+)", summary_state.sampling_row or "")
        temperature_value = temperature_match.group(1) if temperature_match else "—"
        max_tokens_match = re.search(
            r"max_tokens (\d+)", summary_state.sampling_row or ""
        )
        max_tokens_value = max_tokens_match.group(1) if max_tokens_match else "—"
        readiness = (summary_state.readiness_label or "").strip()

        try:
            self.query_one(
                "#console-model-section-provider .console-model-section-value",
                Static,
            ).update(provider_value)
            self.query_one(
                "#console-model-section-model .console-model-section-value", Static
            ).update(model_value)
            self.query_one(
                "#console-model-section-temperature .console-model-section-value",
                Static,
            ).update(temperature_value)
            self.query_one(
                "#console-model-section-max-tokens .console-model-section-value",
                Static,
            ).update(max_tokens_value)
        except (NoMatches, QueryError):
            pass

        try:
            recovery = self.query_one("#console-model-section-recovery", Static)
        except (NoMatches, QueryError):
            pass
        else:
            if readiness and readiness != "Ready":
                recovery.update(readiness)
                recovery.styles.display = "block"
            else:
                recovery.styles.display = "none"

        self._sync_console_rail_system_line()
        self._sync_console_agent_section()
        self._request_console_context_allocation_reconcile()

    def _request_console_agent_fleet_sync(self) -> None:
        """Coalesce Agent-fleet-section syncs into one trailing run (task-5).

        Mirrors ``_request_console_control_bar_sync`` exactly -- a
        scheduled-flag + ``call_after_refresh`` trailing run, not a timer.
        That precedent's own docstring records the measurement that
        motivated the shape (one screen push ran a ~47ms sync 14 times --
        0.65s of a ~1.2s push, every caller individually justified, nothing
        deduplicating them); this is the same shape applied to the Agent
        fleet mini-section's own sync (``_sync_console_agent_section``)
        instead, for its own bursty callers -- e.g. the per-row Cancel
        handler below, which can fire several times in a row (a user
        cancelling more than one child in quick succession) while only one
        resync is ever needed to reflect all of them.

        Requests landing before the trailing run fires fold into it; the
        run always calls ``_sync_console_agent_section`` fresh (itself
        still equality-guarded against the last applied payload), so the
        last-writer semantics every caller relies on are preserved.
        """
        if self._console_agent_fleet_sync_scheduled:
            return
        self._console_agent_fleet_sync_scheduled = True
        self.call_after_refresh(self._run_coalesced_console_agent_fleet_sync)

    def _run_coalesced_console_agent_fleet_sync(self) -> None:
        """Execute one coalesced Agent-fleet-section sync (task-5)."""
        self._console_agent_fleet_sync_scheduled = False
        self._sync_console_agent_section()

    def _sync_console_agent_section(self) -> None:
        """Apply the Agent rail's derived payload to the mounted widgets.

        The equality guard plus the nine ``query_one`` writes; the
        derivation itself is `ConsoleAgentController._console_agent_section_
        payload` (wave-4 console decomposition, task 3) -- controllers own
        no DOM, so the cluster's boundary runs between the two halves.
        Every "why" behind the values applied below (TASK-251's equality
        guard, the fleet force-open the periodic tick must re-apply,
        TASK-870's "View full log" visibility) is documented on that method.

        The memo is written only after a successful apply, so a tick that
        raised part-way through the writes below is re-attempted next tick
        rather than recorded as painted.

        PR2b Task 4: the third payload element is now a
        ``ConsoleInspectorSectionState`` (rows + header summary) for the
        ``ConsoleInspectorSection`` mounted at ``#console-agent-section-
        subagents`` -- replacing the plain ``Static.update()`` this used to
        do, ``sync_state`` patches the section's rows/summary in place when
        possible (structural key unchanged) and only recomposes when the
        row set itself changed shape, per that component's own discipline
        (Task 3). The section is hidden entirely whenever there are no rows
        to show -- no fleet at all, or drilled into one child whose own
        detail the status/steps Statics above already carry (state 3).
        """
        payload = self._agent._console_agent_section_payload()
        if payload == self._console_agent_section_last:
            return
        (
            status_line,
            steps_text,
            fleet_section_state,
            fleet_line,
            back_visible,
            section_open,
            full_log_visible,
            steering_state,
            cancel_all_visible,
        ) = payload
        try:
            self.query_one("#console-agent-section-status", Static).update(status_line)
            self.query_one("#console-agent-section-steps", Static).update(steps_text)
            fleet_section = self.query_one(
                "#console-agent-section-subagents", ConsoleInspectorSection
            )
            fleet_section.sync_state(fleet_section_state)
            fleet_section.styles.display = (
                "block" if fleet_section_state.rows else "none"
            )
            fleet_summary = self.query_one("#console-agent-fleet-summary", Static)
            fleet_summary.update(fleet_line)
            fleet_summary.styles.display = "block" if fleet_line else "none"
            back_button = self.query_one("#console-agent-drilldown-back", Button)
            back_button.styles.display = "block" if back_visible else "none"
            full_log_button = self.query_one("#console-agent-view-full-log", Button)
            full_log_button.styles.display = "block" if full_log_visible else "none"
            # PR3b Task 3: the drill-in steering bar applies its own
            # visibility/queued-line writes from the derived state.
            self.query_one(
                "#console-agent-steering-bar", ConsoleAgentSteeringBar
            ).sync_state(steering_state)
            # PR3b Task 5: the whole-fleet kill switch paints only while
            # a live child exists (derived beside the steering state so
            # the two surfaces move together).
            cancel_all_button = self.query_one(
                f"#{CONSOLE_AGENT_CANCEL_ALL_ID}", Button
            )
            cancel_all_button.styles.display = "block" if cancel_all_visible else "none"
            agent_body = self.query_one("#console-rail-section-body-agent")
            agent_body.styles.display = "block" if section_open else "none"
            agent_header = self.query_one(
                "#console-rail-section-header-agent", DestinationRailSectionHeader
            )
            agent_header.sync_open(section_open)
        except (NoMatches, QueryError):
            return
        self._console_agent_section_last = payload
        self._request_console_context_allocation_reconcile()

    def _focus_console_workspace_conversation_search(self) -> None:
        """Restore focus to the conversation search input when it is mounted."""
        try:
            search = self.query_one("#console-workspace-conversation-search", Input)
        except (NoMatches, QueryError):
            return
        search.focus()

    def _build_console_provider_selection(
        self, session_id: str | None = None
    ) -> ConsoleProviderSelection:
        """Return an owning-session provider selection without switching tabs.

        Served from the per-pass memo inside a `_console_derivation_scope`
        (task-15452): one draft-edit sync built this 7 times for the same
        session.
        """
        memo = self._console_derivation_memo
        memo_key = ("provider_selection", session_id)
        if memo is not None and memo_key in memo:
            return memo[memo_key]
        selection = self._build_console_provider_selection_uncached(session_id)
        if memo is not None:
            memo[memo_key] = selection
        return selection

    def _build_console_provider_selection_uncached(
        self, session_id: str | None = None
    ) -> ConsoleProviderSelection:
        """Derive the provider selection with no memo in front of it."""
        app_config = self._provider_readiness_app_config()
        store = self._ensure_console_chat_store()
        if session_id is None:
            selection_settings = self._session._ensure_active_console_session_settings()
            target_session_id = store.active_session_id
        else:
            selection_settings = self._session._console_session_settings(session_id)
            if selection_settings is None:
                raise KeyError(f"Unknown Console session: {session_id}")
            target_session_id = session_id
        legacy_model = None
        if session_id is None:
            _legacy_provider, legacy_model = self._effective_console_provider_model()
        elif getattr(selection_settings, "source", "derived") == "user":
            legacy_model = selection_settings.model
        else:
            chat_defaults = self._config_section(app_config, "chat_defaults")
            legacy_model = chat_defaults.get("model")
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

        current_workspace_context = self._workspace._current_console_workspace_context()
        if target_session_id is None:
            workspace_context = current_workspace_context
        else:
            workspace_id = store.session_workspace_id(target_session_id)
            workspace_context = (
                current_workspace_context
                if current_workspace_context.active_workspace_id == workspace_id
                else ConsoleWorkspaceContext(active_workspace_id=workspace_id)
            )

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
            workspace_context=workspace_context,
        )

    def _active_console_provider_model_display(
        self,
    ) -> tuple[str, str | None, ConsoleSessionSettings]:
        """Return provider/model labels backed by active session settings.

        Served from the per-pass memo inside a `_console_derivation_scope`
        (task-15452): the control state and the Workbench state built off it
        each re-derive this leg.
        """
        memo = self._console_derivation_memo
        if memo is not None and "provider_model_display" in memo:
            return memo["provider_model_display"]
        display = self._active_console_provider_model_display_uncached()
        if memo is not None:
            memo["provider_model_display"] = display
        return display

    def _active_console_provider_model_display_uncached(
        self,
    ) -> tuple[str, str | None, ConsoleSessionSettings]:
        """Derive provider/model labels with no memo in front of them."""
        settings = self._session._ensure_active_console_session_settings()
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

    def _active_console_settings_readiness(
        self,
    ) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness]:
        """Return effective settings plus Console-native readiness for display/send surfaces.

        Served from the per-pass memo inside a `_console_derivation_scope`
        (task-15452): the blocker copy, the recovery action and the setup
        blocker each re-derive it, and `build_console_settings_readiness`
        is the single most expensive leg of a draft-edit sync.
        """
        memo = self._console_derivation_memo
        if memo is not None and "settings_readiness" in memo:
            return memo["settings_readiness"]
        readiness = self._active_console_settings_readiness_uncached()
        if memo is not None:
            memo["settings_readiness"] = readiness
        return readiness

    def _active_console_settings_readiness_uncached(
        self,
    ) -> tuple[ConsoleSessionSettings, ConsoleSettingsReadiness]:
        """Derive effective settings + readiness with no memo in front."""
        settings = self._session._ensure_active_console_session_settings()
        selection = self._build_console_provider_selection()
        selected_model = selection.explicit_model or selection.configured_model
        effective_settings = replace(
            settings,
            model=selected_model,
            base_url=selection.base_url,
        )
        readiness = build_console_settings_readiness(
            effective_settings,
            app_config=self._provider_readiness_app_config(),
        )
        if not _has_selected_text(selected_model):
            if not readiness.native_send_supported:
                # Provider is the real first blocker (FR-05): surface its
                # readiness instead of the model sentinel so the setup card
                # steps and the recovery action resolve the same first
                # incomplete step. The "Missing model" sentinel below now
                # strictly means provider-ready + model-missing.
                return effective_settings, readiness
            return effective_settings, ConsoleSettingsReadiness(
                label="Missing model",
                detail="Select a model before sending.",
                native_send_supported=False,
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

    def _console_runtime(self) -> Any:
        """Return the `ConsoleRuntime` this screen is the view of, memoised.

        task-15860 lifetime landing: the runtime OUTLIVES this screen, so
        the four handles below are properties over it rather than instance
        attributes. A fresh `ChatScreen`'s own `None` would otherwise
        SHADOW a live store/gateway/controller until `_ensure_*` happened
        to run -- and `_complete_screen_navigation` constructs (and
        `restore_state`s) the incoming screen BEFORE the outgoing one
        unmounts, so both screens are briefly alive and both legitimately
        reach these handles.

        **Memoised on purpose, and resolved exactly once.** Reading a
        handle must never re-claim: a screen that is already superseded
        (the overlapping window above) reads its OWN runtime here and can
        never reach through to the successor's.
        """
        runtime = getattr(self, "_console_runtime_ref", None)
        if runtime is None:
            runtime = ensure_console_runtime(
                getattr(self, "app_instance", None), view=self
            )
            self._console_runtime_ref = runtime
        return runtime

    @property
    def _console_chat_store(self) -> ConsoleChatStore | None:
        """The runtime's Console store, or `None` if none is built yet."""
        return self._console_runtime().chat_store

    @_console_chat_store.setter
    def _console_chat_store(self, value: ConsoleChatStore | None) -> None:
        self._console_runtime().set_chat_store(value)

    @property
    def _console_provider_gateway(self) -> Any | None:
        """The runtime's provider gateway, or `None`."""
        return self._console_runtime().provider_gateway

    @_console_provider_gateway.setter
    def _console_provider_gateway(self, value: Any | None) -> None:
        self._console_runtime().set_provider_gateway(value)

    @property
    def _console_chat_controller(self) -> "ConsoleChatController | None":
        """The runtime's Console chat controller, or `None`."""
        return self._console_runtime().chat_controller

    @_console_chat_controller.setter
    def _console_chat_controller(self, value: "ConsoleChatController | None") -> None:
        self._console_runtime().set_chat_controller(value)

    def _console_ordered_resume_pending(self) -> bool:
        """Return whether an explicit saved-chat startup still owns scope."""
        return (
            self._pending_resume_local_conversation_id is not None
            or self._resume_navigation_startup_in_progress
        )

    def _ensure_console_chat_store(self) -> ConsoleChatStore:
        """Return the native Console chat store, creating it lazily.

        task-15860: CONSTRUCTED and OWNED by the app-owned `ConsoleRuntime`
        (`Chat/console_runtime.py`). Name, laziness, return type and
        patchability are unchanged; the store now survives this screen's
        unmount, so a second Console visit re-uses it.

        TASK-2033: an app-owned store may already exist without a session
        before the Console screen mounts. Align only that empty store with
        the registry-active workspace before any compose-time caller creates
        the first tab. Established sessions remain authoritative, and the
        viewless runtime keeps its global default until an ordinary Console
        view asks for the store. An explicit saved-conversation resume keeps
        that safe global context until hydration resolves the conversation's
        own persisted scope.
        """
        store = self._console_chat_store
        resume_pending = self._console_ordered_resume_pending()
        if store is None:
            store = self._console_runtime().ensure_chat_store(
                workspace_context=(
                    None
                    if resume_pending
                    else self._workspace._current_console_workspace_context()
                ),
                on_scope_flushed=self._on_console_scope_flushed,
            )
        elif store.active_session_id is None and not resume_pending:
            store.set_workspace_context(
                self._workspace._current_console_workspace_context()
            )
        return store

    def _ensure_console_agent_bridge(self) -> Any:
        """Return the native Console agent bridge, creating it lazily.

        One-line delegation (wave-4 console decomposition, task 3). Reached
        from five screen-level call sites outside the agent cluster (the
        chat controller's construction and its core-state sync, the
        conversation browser's badge counts, the Change Review opener) and
        replaced by name on the screen instance by three tests
        (`Tests/Chat/test_change_turn_tracking.py`,
        `Tests/Chat/test_console_agent_swap.py`), which is why it keeps its
        original name here rather than moving with no residue. See
        `ConsoleAgentController._ensure_console_agent_bridge` for the real
        implementation, and `agent.py`'s module docstring for what such a
        patch does and does not steer.
        """
        return self._agent._ensure_console_agent_bridge()

    def _console_agent_runtime_enabled(self) -> bool:
        """Return whether ``[console] agent_runtime`` gates in the agent loop (default on)."""
        value = self._console_config().get("agent_runtime", True)
        return bool(value) if isinstance(value, (bool, int)) else True

    def _console_native_tool_calls_enabled(self) -> bool:
        """Return whether ``[console] native_tool_calls`` allows native provider tool-calls (default on)."""
        value = self._console_config().get("native_tool_calls", True)
        return bool(value) if isinstance(value, (bool, int)) else True

    def _ensure_console_image_view(
        self,
    ) -> tuple[ConsoleImageViewState, ConsoleImageRenderCache]:
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
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        for the image-view cluster's own staying callers and the
        pre-existing test suite's direct-call/monkeypatch convention."""
        return self._message._recent_console_image_messages(messages)

    def _console_generation_browse(self) -> dict[str, int]:
        """Return the lazily-created browsed-variant-index map for generation cards.

        Ephemeral (never persisted) and not initialized in ``__init__`` --
        mirrors the getattr/setdefault pattern
        ``_console_imagegen_inflight_sessions`` uses for other screen-owned,
        purely in-memory bookkeeping. Absent entries default a generation
        message to its canonical variant (index 0) until a later task's
        browse controls change it.
        """
        browse = getattr(self, "_generation_browse", None)
        if browse is None:
            browse = {}
            self._generation_browse = browse
        return browse

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
            self._console_image_preparing.difference_update(mid for mid, _ in pending)

    def _ensure_console_provider_gateway(self) -> Any:
        """Return the native Console provider gateway with a test injection seam.

        task-15860: constructed and OWNED by the app-owned `ConsoleRuntime`,
        which also reads the app's `console_provider_gateway_factory`
        injection seam. Name, laziness and behaviour are unchanged; the
        gateway is now closed at app exit (`ConsoleRuntime.dispose`) rather
        than on every navigation away.
        """
        if self._console_provider_gateway is None:
            self._console_runtime().ensure_provider_gateway(
                # Fresh-config source: the gateway re-resolves readiness at
                # send time and must see Settings saves made after boot.
                config_provider=self._provider_readiness_app_config,
            )
        return self._console_provider_gateway

    def _ensure_console_prompt_history(self) -> PromptHistory:
        """Delegate to `ConsolePromptsController` (wave-3 console decomposition, task 3)."""
        return self._prompts._ensure_console_prompt_history()

    def _ensure_console_chat_controller(self) -> ConsoleChatController:
        """Return the native Console chat controller with fresh selection state.

        task-15860 Task 1: CONSTRUCTED by the app-owned `ConsoleRuntime`,
        same keyword arguments in the same order. Everything below the
        construction block -- the UI hook wiring, the wake coordinator's
        `wire(app=...)`, the core-state sync -- still runs here on every
        call; rebinding those hooks for a viewless turn is Task 4.
        """
        if self._console_chat_controller is None:
            selection = self._build_console_provider_selection()
            self._console_runtime().ensure_chat_controller(
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
                world_info_applier=self._console_world_info_applier,
                rag_capture_provider=self._retrieval._capture_console_staged_rag,
                default_session_settings=self._session._blank_console_session_settings,
                library_provider_factory=self._library_activity.build_provider,
                global_user_display_name=self._global_chat_display_name,
                turn_context_provider=(
                    self._session._build_console_turn_execution_context
                ),
                provider_config=self._provider_readiness_app_config,
            )
        # task-15860: every screen-owned slot on the controller, the store
        # and the wake coordinator is (re)bound HERE, through the single
        # enumerated `CONSOLE_VIEW_HOOK_SLOTS` list, so that the same list
        # can clear all of them at detach. This block used to assign each
        # one by hand and had no counterpart anywhere.
        self._console_runtime().attach_view(self)
        self._console_chat_controller.remount_watchlists_operation_receipts()
        self._console_chat_controller._confirm_project_instruction_dispatch = (
            self._session._confirm_project_instruction_dispatch
        )
        self._console_chat_controller._select_project_instruction_binding = (
            self._session._select_project_instruction_binding
        )
        # MCP batch-approval bridge (task-5): `request_mcp_approvals` runs
        # on the agent bridge's worker thread and needs a
        # `call_from_thread`-capable App handle. Deliberately NOT a
        # view-hook slot: this is the APP, which outlives every view, and
        # clearing it at detach would break the bridge a surviving turn
        # still needs.
        self._console_chat_controller.app = self.app_instance
        # PR3a-2 Task 5 (auto-wake): the app object (durable-mark clear
        # seam + marks reads). getattr-guarded because several UI tests
        # swap in hand-built controller doubles before re-running this
        # wiring block. `delivery_ui_hook` is a view-hook slot and is
        # bound by `attach_view` above.
        wake = getattr(self._console_chat_controller, "fleet_wake", None)
        if wake is not None:
            wake.wire(app=self.app_instance)
        self._sync_console_chat_core_state()
        return self._console_chat_controller

    def console_view_hooks(self) -> dict[str, Any]:
        """Return this view's value for every `CONSOLE_VIEW_HOOK_SLOTS` slot.

        task-15860. The runtime outlives this screen, so every callable it
        holds that closes over `self` has to be re-bindable and, more
        importantly, CLEARABLE -- Task 0's P3 found five such slots still
        pointing at a dead `ChatScreen` after a real unmount, none of them
        raising, and a silent wrong answer from `wake_conversation_in_view`
        decides whether the unseen `◈` mark survives.

        Keys must match `CONSOLE_VIEW_HOOK_SLOTS` exactly; a test asserts
        the two sets are equal, which is what stops a slot being bound
        here and never cleared (or cleared and never bound).

        Returns:
            dict[str, Any]: slot name -> this view's value.
        """
        session = getattr(self, "_session", None)
        prompts = getattr(self, "_prompts", None)
        retrieval = getattr(self, "_retrieval", None)
        skill = getattr(self, "_skill", None)
        return {
            # constructor-supplied callables
            "_chat_dictionary_applier": self._console_chat_dictionary_applier,
            "_world_info_applier": self._console_world_info_applier,
            "_rag_capture_provider": getattr(
                retrieval, "_capture_console_staged_rag", None
            ),
            "_default_session_settings": getattr(
                session, "_blank_console_session_settings", None
            ),
            "_library_provider_factory": self._library_activity.build_provider,
            "_global_user_display_name": self._global_chat_display_name,
            "_turn_context_provider": getattr(
                session, "_build_console_turn_execution_context", None
            ),
            # post-construction UI bridges
            "on_submission_accepted": self._on_console_submission_accepted,
            "follow_watchlists_operations": (
                self._follow_console_watchlists_operations
            ),
            # TASK-1364: accepted sends are recorded to the shared prompt
            # history (inside `submit_draft`, past every block/refusal gate).
            "prompt_history": (
                self._ensure_console_prompt_history() if prompts is not None else None
            ),
            "set_pending_approval": self._set_console_pending_approval,
            # Task 9 (parked background approvals): UI-thread bridge target
            # for a NON-active session's approval round -- badge + one
            # toast, never the mounted-card path above.
            "park_pending_approval": self._park_console_approval,
            # Task 10 (background completion toasts): UI-thread bridge
            # target for a NON-active session's run finishing/failing -- the
            # one-per-run toast, invoked directly (never via
            # `call_from_thread`) from `_set_run_state`'s once-guarded
            # non-active terminal branch.
            "notify_run_outcome": self._notify_console_run_outcome,
            # task-2154.16 (FB-05): the ACTIVE session's own run failing --
            # one error toast carrying the run's visible copy.
            "notify_run_failure": self._notify_console_run_failure,
            "set_pending_skill_install": getattr(
                skill, "_set_console_pending_skill_install", None
            ),
            "set_pending_skill_script": getattr(
                skill, "_set_console_pending_skill_script", None
            ),
            # PR3a-2 Task 5, user-wins-ties.
            "wake_user_priority_probe": self._fleet._console_wake_user_priority,
            # task-15971: the delivery COMMIT's visibility probe -- a wake
            # completing while this conversation is not displayed-and-active
            # leaves the FLEET_UNSEEN mark set (the ◈ badge is how the user
            # learns an off-view delivery landed).
            "wake_conversation_in_view": (
                self._fleet._console_wake_conversation_in_view
            ),
            # the store's one screen-owned callback
            "on_scope_flushed": self._on_console_scope_flushed,
            # task-15862: a wake turn enters through the coordinator, never
            # the user-send worker that arms the 0.2s transcript poll --
            # without this hook nothing repaints the wake turn's streamed
            # reply, its terminal tab glyph, or the composer state (the live
            # 4+ minute mid-delivery freeze, PR3a-2 Task 7 finding 1).
            "delivery_ui_hook": self._fleet._on_console_wake_delivery_started,
        }

    def _release_consumed_console_launch(
        self,
        launch: ConsoleLiveWorkLaunch,
        result: Any,
    ) -> None:
        """Clear the launch context a send just consumed and refresh surfaces.

        Ordering matters. The clear happens FIRST and cannot raise; both
        fallible steps (counting what was prompted, refreshing surfaces) are
        contained here. This method sits on the provider's capture path, and
        ``ConsoleChatController._capture_rag_context`` converts ANY exception
        from that provider into ``context=None`` -- so an escaping failure
        here would send the message WITHOUT the evidence it had just
        consumed. A stale chip is recoverable; a silently unsent bundle is
        not.

        Args:
            launch: The launch handed to the capture adapter for this send.
                A staging that landed while the capture was awaited wins:
                the identity check below leaves the newer context alone
                rather than dropping evidence the user just staged.
            result: The capture's ``LocalRagContextResult``, read only for
                the exact prompted-entry count.
        """
        if self._pending_console_launch_context is not launch:
            return
        self._pending_console_launch_context = None
        self._pending_console_launch_auto_open_inspector = False
        try:
            self._console_evidence_sent_notice = self._console_prompted_source_count(
                launch, result
            )
            self._sync_console_pending_launch_surfaces()
        except Exception as exc:
            logger.warning(
                "Console staged-evidence surfaces did not refresh after a "
                "consuming send (exception_category={})",
                type(exc).__name__,
            )

    @staticmethod
    def _console_prompted_source_count(
        launch: ConsoleLiveWorkLaunch,
        result: Any,
    ) -> int:
        """Return how many sources this send actually put in front of the model.

        The "Evidence sent with this message" line is a claim about what
        reached the provider, so it must never report the staged total: the
        capture prompts only available, locally-owned references.

        Preference order:

        1. ``citation_repair_contract.allowed_ordinals`` -- one ordinal per
           FORMATTED prompt entry, so this is the exact count, and the
           capture attaches the contract to every context-bearing return.
        2. :func:`console_prompted_source_count` -- a best-effort formatted
           pre-authority estimate used only when the authoritative repair
           contract could not be built.

        Args:
            launch: The launch this send consumed.
            result: The capture's ``LocalRagContextResult``.

        Returns:
            The number of evidence entries carried into the prompt.
        """
        ordinals = getattr(
            getattr(result, "citation_repair_contract", None),
            "allowed_ordinals",
            None,
        )
        if isinstance(ordinals, tuple) and ordinals:
            return len(ordinals)
        return console_prompted_source_count(launch)

    def _clear_console_evidence_sent_notice(self) -> None:
        """Drop the one-send "evidence sent" line and refresh only the strip."""
        if self._console_evidence_sent_notice is None:
            return
        self._console_evidence_sent_notice = None
        self._sync_console_staged_evidence_strip()

    def _sync_console_chat_core_state(self) -> ConsoleProviderSelection:
        """Push current workspace/provider selection into native Console services."""
        selection = self._build_console_provider_selection()
        store = self._ensure_console_chat_store()
        if store.active_session_id is None and self._console_ordered_resume_pending():
            selection = replace(selection, workspace_context=store.workspace_context)
        store.set_workspace_context(selection.workspace_context)
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
                self._console_chat_controller.configured_model = (
                    selection.configured_model
                )
                self._console_chat_controller.base_url = selection.base_url
                self._console_chat_controller.temperature = selection.temperature
                self._console_chat_controller.top_p = selection.top_p
                self._console_chat_controller.min_p = selection.min_p
                self._console_chat_controller.top_k = selection.top_k
                self._console_chat_controller.max_tokens = selection.max_tokens
                self._console_chat_controller.seed = selection.seed
                self._console_chat_controller.presence_penalty = (
                    selection.presence_penalty
                )
                self._console_chat_controller.frequency_penalty = (
                    selection.frequency_penalty
                )
                self._console_chat_controller.reasoning_effort = (
                    selection.reasoning_effort
                )
                self._console_chat_controller.reasoning_summary = (
                    selection.reasoning_summary
                )
                self._console_chat_controller.verbosity = selection.verbosity
                self._console_chat_controller.thinking_effort = (
                    selection.thinking_effort
                )
                self._console_chat_controller.thinking_budget_tokens = (
                    selection.thinking_budget_tokens
                )
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

    #: Memoized ``#console-native-composer`` node, or None when nothing has
    #: been resolved (or the last resolved node went away). A CLASS attribute
    #: default because the hand-built ``ChatScreen.__new__()`` test fixtures
    #: never run ``__init__``. Never read directly -- always through
    #: ``_console_composer_or_none``, which revalidates before returning it.
    _console_composer_ref: ConsoleComposerBar | None = None

    def _console_composer_or_none(self) -> ConsoleComposerBar | None:
        """Return the native Console composer when it is mounted.

        TASK-15454 (folded in from task-15452's review): this used to run an
        uncached ``self.query()`` -- a full walk of the largest widget tree
        in the app -- and the draft-edit keystroke path alone calls it twice,
        which measured as the majority of that path's residual cost. The
        resolved node is memoized instead.

        A stale reference must be impossible, so the memo is revalidated on
        every hit rather than invalidated from teardown hooks: the cached
        widget is returned only while it is still mounted AND still reachable
        from this screen. A recompose that replaces the composer detaches the
        old node (``_parent`` cleared by the prune) and clears its mounted
        flag, so both halves fail closed and the next call re-queries. The
        memo therefore cannot outlive the widget it names even if some future
        teardown path forgets about it entirely.
        """
        cached = self._console_composer_ref
        if cached is not None:
            try:
                still_live = cached.is_mounted and self in cached.ancestors_with_self
            except Exception:  # pragma: no cover - defensive, see above
                still_live = False
            if still_live:
                return cached
            self._console_composer_ref = None
        composers = list(self.query("#console-native-composer"))
        if composers and isinstance(composers[0], ConsoleComposerBar):
            self._console_composer_ref = composers[0]
            return composers[0]
        return None

    # Dictation state moved to `ConsoleDictationController` (wave-1 console
    # decomposition, task 5). These eight properties keep `self._console_
    # dictation_*` / `self._console_pending_voice_action` readable AND
    # writable exactly as before, for the handful of screen methods that
    # are not part of the dictation cluster but still read this state
    # (the hands-free wiring, `on_button_pressed`, `_sync_console_composer_
    # action_state`) and for tests that poke it directly -- each proxies
    # straight through to `self._dictation`, so none of those call sites
    # needed to change.
    @property
    def _console_dictation_state(self) -> str:
        return self._dictation._console_dictation_state

    @_console_dictation_state.setter
    def _console_dictation_state(self, value: str) -> None:
        self._dictation._console_dictation_state = value

    @property
    def _console_dictation_session(self) -> Any:
        return self._dictation._console_dictation_session

    @_console_dictation_session.setter
    def _console_dictation_session(self, value: Any) -> None:
        self._dictation._console_dictation_session = value

    @property
    def _console_dictation_partial(self) -> str:
        return self._dictation._console_dictation_partial

    @_console_dictation_partial.setter
    def _console_dictation_partial(self, value: str) -> None:
        self._dictation._console_dictation_partial = value

    @property
    def _console_dictation_timer(self) -> Any:
        return self._dictation._console_dictation_timer

    @_console_dictation_timer.setter
    def _console_dictation_timer(self, value: Any) -> None:
        self._dictation._console_dictation_timer = value

    @property
    def _console_dictation_elapsed_timer(self) -> Any:
        return self._dictation._console_dictation_elapsed_timer

    @_console_dictation_elapsed_timer.setter
    def _console_dictation_elapsed_timer(self, value: Any) -> None:
        self._dictation._console_dictation_elapsed_timer = value

    @property
    def _console_dictation_origin_session_id(self) -> str | None:
        return self._dictation._console_dictation_origin_session_id

    @_console_dictation_origin_session_id.setter
    def _console_dictation_origin_session_id(self, value: str | None) -> None:
        self._dictation._console_dictation_origin_session_id = value

    @property
    def _console_pending_voice_action(self) -> str | None:
        return self._dictation._console_pending_voice_action

    @_console_pending_voice_action.setter
    def _console_pending_voice_action(self, value: str | None) -> None:
        self._dictation._console_pending_voice_action = value

    @property
    def _console_dictation_late_discard_ack(self) -> bool:
        return self._dictation._console_dictation_late_discard_ack

    @_console_dictation_late_discard_ack.setter
    def _console_dictation_late_discard_ack(self, value: bool) -> None:
        self._dictation._console_dictation_late_discard_ack = value

    # Message-cluster state moved to `ConsoleMessageController` (wave-3
    # console decomposition, task 1). These properties keep `self._console_
    # message_action_service`/`_last_console_action`/`_pending_console_
    # delete_message_id`/`_console_original_attempt_previews`/`_console_
    # speaking_message_id`/`_pending_console_swipe_selection` readable (and,
    # where the pre-move source ever wrote them from outside the cluster,
    # writable) exactly as before -- for the DOM-touching siblings that
    # stayed on `ChatScreen` (`_selected_console_message_inspector_rows`,
    # `_clear_native_console_message_selection`, `_sync_console_pending_
    # delete_confirmation`, `_sync_native_console_transcript`, `on_unmount`),
    # for `console_transcript.py`'s bare-name `getattr(self.screen,
    # "_console_speaking_message_id", None)` reach, and for tests that poke
    # this state directly -- each proxies straight through to
    # `self._message`, so none of those call sites needed to change.
    # `_console_message_action_service` has no proxy: nothing outside the
    # moved cluster ever read or wrote it (a pre-existing bare-screen test
    # fixture assigns it defensively; see the task-1 extraction report).
    # Every proxy below is read-WRITE. At baseline each of these was a plain
    # assignable instance attribute set in `ChatScreen.__init__`, so a
    # getter-only property would turn a legal write into `AttributeError` --
    # a behaviour change, which this decomposition's contract forbids
    # regardless of whether an in-repo writer exists today. (`_last_console_
    # action` has no writer outside the moved cluster right now; its setter
    # is here for exactly that reason -- see the task-1 fix-round report.)
    # Note this is the OPPOSITE case to the binding rule's write-only proxy,
    # where the *getter* deliberately raises `RuntimeError`: that forbids a
    # direction the pre-move source never had, this restores one it did.
    @property
    def _last_console_action(self) -> Any:
        return self._message._last_console_action

    @_last_console_action.setter
    def _last_console_action(self, value: Any) -> None:
        self._message._last_console_action = value

    @property
    def _pending_console_delete_message_id(self) -> str | None:
        return self._message._pending_console_delete_message_id

    @_pending_console_delete_message_id.setter
    def _pending_console_delete_message_id(self, value: str | None) -> None:
        self._message._pending_console_delete_message_id = value

    @property
    def _console_original_attempt_previews(self) -> dict[str, str]:
        return self._message._console_original_attempt_previews

    @_console_original_attempt_previews.setter
    def _console_original_attempt_previews(self, value: dict[str, str]) -> None:
        self._message._console_original_attempt_previews = value

    @property
    def _console_speaking_message_id(self) -> str | None:
        return self._message._console_speaking_message_id

    @_console_speaking_message_id.setter
    def _console_speaking_message_id(self, value: str | None) -> None:
        self._message._console_speaking_message_id = value

    @property
    def _console_speech_states(self) -> dict[str, str]:
        return self._message._console_speech_states

    @_console_speech_states.setter
    def _console_speech_states(self, value: dict[str, str]) -> None:
        self._message._console_speech_states = value

    @property
    def _pending_console_swipe_selection(self) -> str | None:
        return self._message._pending_console_swipe_selection

    @_pending_console_swipe_selection.setter
    def _pending_console_swipe_selection(self, value: str | None) -> None:
        self._message._pending_console_swipe_selection = value

    def _sync_console_dictation_availability(self) -> None:
        """Refresh the mic button's dictation-availability tooltip.

        One-line delegation (wave-1 console decomposition, task 5). Called
        from `on_mount` (post-mount probe, twice: `call_after_refresh` and
        a 0.15s retry) and from `ConsoleDictationController._request_
        console_dictation_start` (re-probe on every activation attempt).
        See `ConsoleDictationController._sync_console_dictation_
        availability` for the real implementation.
        """
        self._dictation._sync_console_dictation_availability()

    # V3 pipeline hands-free loop state moved to `ConsoleHandsFreeController`
    # (wave-2 console decomposition, task 1). These two properties keep
    # `self._console_hands_free`/`_console_hands_free_vad_degraded`/
    # `_console_hands_free_store_tap_installed` readable AND writable
    # exactly as before, for the several screen methods that are not part
    # of the hands-free cluster but still touch this state (`on_key`,
    # `on_button_pressed`, `on_unmount`, the realtime engine's own loud
    # fallback) and for tests that poke it directly -- each proxies
    # straight through to `self._hands_free`, so none of those call sites
    # needed to change.
    @property
    def _console_hands_free(self) -> ConsoleHandsFreeSession | None:
        return self._hands_free._console_hands_free

    @_console_hands_free.setter
    def _console_hands_free(self, value: ConsoleHandsFreeSession | None) -> None:
        self._hands_free._console_hands_free = value

    @property
    def _console_hands_free_vad_degraded(self) -> bool:
        return self._hands_free._console_hands_free_vad_degraded

    @_console_hands_free_vad_degraded.setter
    def _console_hands_free_vad_degraded(self, value: bool) -> None:
        self._hands_free._console_hands_free_vad_degraded = value

    @property
    def _console_hands_free_store_tap_installed(self) -> bool:
        return self._hands_free._console_hands_free_store_tap_installed

    @_console_hands_free_store_tap_installed.setter
    def _console_hands_free_store_tap_installed(self, value: bool) -> None:
        self._hands_free._console_hands_free_store_tap_installed = value

    # Agent-cluster state moved to `ConsoleAgentController` (wave-4 console
    # decomposition, task 3). These three properties keep
    # `self._console_agent_bridge`/`_console_agent_drilldown_run_id`/
    # `_agent_section_user_dismissed_while_busy` readable AND writable
    # exactly as before, for the
    # screen methods outside the agent cluster that still touch this state
    # (`compose_content`, `_build_console_inspector_state`,
    # `_toggle_console_rail_section`, `on_button_pressed`'s drill-down Back
    # branch), for `ConsoleSessionController`/`ConsoleWorkspaceController`'s
    # own drill-down clears (both reach `self._screen._console_agent_
    # drilldown_run_id` through their own proxies of the same name), and for
    # the tests that poke this state directly -- each proxies straight
    # through to `self._agent`, so none of those call sites needed to
    # change. Every one is read-WRITE: at baseline each was a plain
    # assignable instance attribute set in `ChatScreen.__init__`, and
    # `Tests/UI/test_console_agent_rail.py` alone assigns two of them 20+
    # times, so a getter-only property would turn a legal write into
    # `AttributeError`.
    #
    # The cluster's other six attributes (`_console_agent_drilldown_
    # conversation_id`, `_console_agent_full_log_cache_run_id`/`..._
    # available`, `_console_subagent_counts_cache`/`..._row_ids`/`..._at`)
    # get no proxy: no production code outside the moved methods ever read
    # or wrote them, and the tests that set the first one were repointed at
    # `screen._agent` alongside this move.
    @property
    def _console_agent_bridge(self) -> Any:
        return self._agent._console_agent_bridge

    @_console_agent_bridge.setter
    def _console_agent_bridge(self, value: Any) -> None:
        self._agent._console_agent_bridge = value

    @property
    def _console_agent_drilldown_run_id(self) -> str | None:
        return self._agent._console_agent_drilldown_run_id

    @_console_agent_drilldown_run_id.setter
    def _console_agent_drilldown_run_id(self, value: str | None) -> None:
        self._agent._console_agent_drilldown_run_id = value

    @property
    def _agent_section_user_dismissed_while_busy(self) -> bool:
        return self._agent._agent_section_user_dismissed_while_busy

    @_agent_section_user_dismissed_while_busy.setter
    def _agent_section_user_dismissed_while_busy(self, value: bool) -> None:
        self._agent._agent_section_user_dismissed_while_busy = value

    # Legacy Workspace-search names remain assignable compatibility aliases.
    # Scalar reads/writes share Workspace's canonical browser state; row reads
    # project rich rows to the legacy shape and row writes convert back to rich
    # rows. The bounded Input handler passes only query/disabled values, and the
    # Clear button delegates the complete transition to Workspace, so the screen
    # owns no duplicate browser backing state or refresh writer.
    @property
    def _console_workspace_conversation_query(self) -> str:
        return self._workspace._console_workspace_conversation_query

    @_console_workspace_conversation_query.setter
    def _console_workspace_conversation_query(self, value: str) -> None:
        self._workspace._console_workspace_conversation_query = value

    @property
    def _console_workspace_conversation_search_timer(self) -> Any:
        return self._workspace._console_workspace_conversation_search_timer

    @_console_workspace_conversation_search_timer.setter
    def _console_workspace_conversation_search_timer(self, value: Any) -> None:
        self._workspace._console_workspace_conversation_search_timer = value

    @property
    def _console_workspace_conversation_search_token(self) -> int:
        return self._workspace._console_workspace_conversation_search_token

    @_console_workspace_conversation_search_token.setter
    def _console_workspace_conversation_search_token(self, value: int) -> None:
        self._workspace._console_workspace_conversation_search_token = value

    @property
    def _console_workspace_conversation_search_rows(
        self,
    ) -> tuple[ConsoleWorkspaceConversationRow, ...]:
        return self._workspace._console_workspace_conversation_search_rows

    @_console_workspace_conversation_search_rows.setter
    def _console_workspace_conversation_search_rows(
        self, value: tuple[ConsoleWorkspaceConversationRow, ...]
    ) -> None:
        self._workspace._console_workspace_conversation_search_rows = value

    @property
    def _console_workspace_conversation_search_total(self) -> int | None:
        return self._workspace._console_workspace_conversation_search_total

    @_console_workspace_conversation_search_total.setter
    def _console_workspace_conversation_search_total(self, value: int | None) -> None:
        self._workspace._console_workspace_conversation_search_total = value

    @property
    def _console_workspace_conversation_search_error(self) -> str:
        return self._workspace._console_workspace_conversation_search_error

    @_console_workspace_conversation_search_error.setter
    def _console_workspace_conversation_search_error(self, value: str) -> None:
        self._workspace._console_workspace_conversation_search_error = value

    #: Per-session draft bookkeeping moved to `ConsoleSessionController`
    #: (wave-2 console decomposition, task 3). `self._console_visible_draft_
    #: session_id`/`_console_undo_histories` stay readable/writable via
    #: these two proxy properties under the ORIGINAL attribute names, so
    #: `_serialize_native_console_state`/`_restore_native_console_state`,
    #: `_submit_console_native_draft`/`_on_console_submission_accepted`, and
    #: `on_button_pressed`'s tab-close branch -- none of them this
    #: cluster's own -- needed no changes.
    @property
    def _console_visible_draft_session_id(self) -> str | None:
        return self._session._console_visible_draft_session_id

    @_console_visible_draft_session_id.setter
    def _console_visible_draft_session_id(self, value: str | None) -> None:
        self._session._console_visible_draft_session_id = value

    @property
    def _console_undo_histories(self) -> dict[str, ConsoleComposerUndoHistory]:
        return self._session._console_undo_histories

    def _speak_status(self, text: str) -> None:
        """Speak a status/ack/error via TTS when spoken feedback is on and idle.

        Posts `TTSRequestEvent(text)` iff `dictation.spoken_feedback` is
        truthy (default `false`) AND the microphone is not open right now
        (`_console_dictation_state == "idle"`). The state check is the hard
        microphone/speaker mutual-exclusion rule (spec): speech overlapping a
        live capture would be picked up by the recognizer and transcribed
        straight into the user's own draft, so this must never fire
        mid-capture regardless of the toggle.

        "Capture started" is deliberately never routed through this method --
        see `_request_console_dictation_start`'s unconditional
        `TTSPlaybackEvent(stop)`, which handles the opposite direction
        (silencing speech *before* a capture opens) instead.

        Args:
            text: The plain-text status, command acknowledgement, or error
                reason to speak -- the same copy the corresponding toast uses.
        """
        if self._console_dictation_state != "idle":
            return
        if not coerce_bool_setting(
            get_cli_setting("dictation.spoken_feedback", False), False
        ):
            return
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSRequestEvent,
        )

        self.app_instance.post_message(TTSRequestEvent(text=text))

    @on(ConsoleDictationEvent)
    def _handle_console_dictation_event(self, message: ConsoleDictationEvent) -> None:
        """Apply a streaming dictation event posted from any thread.

        One-line delegation (wave-1 console decomposition, task 5); the
        `@on` decorator has to stay on this class for Textual's message
        dispatch to find it. See `ConsoleDictationController._handle_
        console_dictation_event` for the real implementation.

        Args:
            message: The posted controller event.
        """
        self._dictation._handle_console_dictation_event(message)

    @on(ConsoleDictationLimitSignal)
    def _handle_console_dictation_buffer_limit(
        self, message: ConsoleDictationLimitSignal
    ) -> None:
        """Stop the capture whose recorder ran out of its PCM budget.

        One-line delegation (wave-1 console decomposition, task 5); the
        `@on` decorator has to stay on this class for Textual's message
        dispatch to find it. See `ConsoleDictationController._handle_
        console_dictation_buffer_limit` for the real implementation.

        Args:
            message: The posted buffer-limit signal.
        """
        self._dictation._handle_console_dictation_buffer_limit(message)

    #: `_deliver_console_hands_free_capture_ended` (and the wait bound it
    #: used to be attached to as a sibling class constant) moved to
    #: `ConsoleHandsFreeController` (wave-2 console decomposition, task 1).
    #: `ConsoleDictationController._handle_console_dictation_limit` reaches
    #: it through the injected `deliver_hands_free_capture_ended` callable;
    #: see `hands_free.py`'s module docstring.

    #: task-1683: the exact Impersonate text last inserted into the draft,
    #: so a second click REPLACES it instead of stacking drafts. Keyed by
    #: session id -- two tabs can each hold their own pending suggestion.
    _console_impersonate_last: dict[str, str]

    async def _open_console_composer_menu(self) -> None:
        """Open the composer overflow menu (task-1680)."""
        composer = self._console_composer_or_none()
        self.app.push_screen(
            ConsoleComposerMenuModal(
                attachment_kind=self._console_pending_attachment_kind(),
                ephemeral=self._console_active_session_is_ephemeral(),
                # Same input the action-row button read before it moved here,
                # so Save Chatbook's available/unavailable copy is unchanged.
                can_save_chatbook=self._console_chatbook_action_available(),
                draft_available=bool(
                    composer is not None and composer.draft_text().strip()
                ),
                improvement_undo_available=bool(
                    composer is not None and composer.improvement_undo_available
                ),
            ),
            callback=self._handle_console_composer_menu_choice,
        )

    def _console_pending_attachment_kind(self) -> str:
        """Classify the active session's staged attachment.

        task-1682 follow-up: Generate Caption used to enable for ANY
        attachment, so a PDF got an image-caption prompt. Reads the real
        staged records rather than guessing from the composer's label.

        Returns:
            ``"image"`` when at least one staged attachment is an image,
            ``"other"`` when something is staged but none are images, or
            ``"none"`` when nothing is staged.
        """
        store = self._ensure_console_chat_store()
        session_id = getattr(store, "active_session_id", None)
        if not session_id:
            return "none"
        try:
            pendings = store.pending_attachments(session_id)
        except KeyError:
            return "none"
        if not pendings:
            return "none"
        for attachment in pendings:
            mime = str(getattr(attachment, "mime_type", "") or "").lower()
            file_type = str(getattr(attachment, "file_type", "") or "").lower()
            if mime.startswith("image/") or file_type == "image":
                return "image"
        return "other"

    def _handle_console_composer_menu_choice(self, action_id: str | None) -> None:
        """Route the chosen menu action (task-1680)."""
        if not action_id:
            return
        if action_id == ACTION_SAVE_CHAT:
            self._session._dispatch_promote_console_temporary_session()
            return
        if action_id == ACTION_PROMPTS:
            self._open_console_prompts_modal()
            return
        if action_id == ACTION_IMPROVE_CURRENT_DRAFT:
            self._open_console_prompts_modal(initial_mode="improve")
            return
        if action_id == ACTION_UNDO_PROMPT_IMPROVEMENT:
            self._undo_console_prompt_improvement()
            return
        # Attach and Save Chatbook moved out of the width-bounded action row
        # into this menu. Both route to the SAME handlers their buttons used,
        # so the menu is a second entry point rather than a second
        # implementation -- including Save Chatbook's temporary-chat block,
        # which `build_composer_menu_entries` already applied by disabling
        # the row before it could be chosen.
        if action_id == ACTION_ATTACH_CONTEXT:
            self.run_worker(self._handle_console_attach_context(), exclusive=False)
            return
        if action_id == ACTION_SAVE_CHATBOOK:
            self._save_console_chatbook_from_visible_action()
            return
        if action_id == ACTION_GENERATE_IMAGE:
            self.run_worker(self._open_console_generate_image_modal(), exclusive=False)
        elif action_id == ACTION_GENERATE_CAPTION:
            self._insert_console_caption_prompt()
        elif action_id == ACTION_IMPERSONATE:
            self.run_worker(
                self._run_console_impersonate(),
                exclusive=True,
                group="console-impersonate",
            )

    def _undo_console_prompt_improvement(self) -> bool:
        """Restore the exact pre-improvement draft and persist it for this tab."""
        composer = self._console_composer_or_none()
        if composer is None or not composer.undo_improvement():
            return False
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if session_id is not None:
            store.set_session_draft(session_id, composer.draft_text())
        self._focus_console_composer_if_needed(force=True)
        return True

    def _open_console_prompt_comparison(self) -> None:
        """Open the safe before/after view for the current improvement Undo."""
        composer = self._console_composer_or_none()
        comparison = composer.improvement_comparison() if composer is not None else None
        if comparison is None:
            self._focus_console_composer_if_needed(force=True)
            return
        before, after = comparison
        self.app.push_screen(
            ConsolePromptComparisonModal(before=before, after=after),
            callback=self._handle_console_prompt_comparison_result,
        )

    def _handle_console_prompt_comparison_result(
        self, result: PromptComparisonResult | None
    ) -> None:
        """Keep the improved draft or consume Undo to restore the original."""
        if result == "restore":
            self._undo_console_prompt_improvement()
            return
        self._focus_console_composer_if_needed(force=True)

    def _open_console_prompts_modal(
        self, *, initial_mode: Literal["browse", "improve"] = "browse"
    ) -> None:
        """Delegate to `ConsolePromptsController` (wave-3 console decomposition, task 3)."""
        self._prompts._open_console_prompts_modal(initial_mode=initial_mode)

    @on(ConsoleTemporaryChip.SaveRequested)
    def on_console_temporary_chip_save(
        self, event: ConsoleTemporaryChip.SaveRequested
    ) -> None:
        """Save the temporary chat from its status chip."""
        event.stop()
        self._session._dispatch_promote_console_temporary_session()

    async def _open_console_generate_image_modal(self) -> None:
        """Collect image options, then paste the command into the draft."""
        backends: tuple[str, ...] = ()
        styles: dict[str, str] = {}
        try:
            from ...UI.Screens.settings_image_gen_defaults import BACKEND_IDS

            backends = tuple(BACKEND_IDS)
        except Exception:
            logger.opt(exception=True).debug("Image modal: backend list failed.")
        try:
            from ...Media_Creation.generation_templates import get_all_templates

            styles = {
                sid: getattr(tpl, "name", sid)
                for sid, tpl in (await asyncio.to_thread(get_all_templates)).items()
            }
        except Exception:
            logger.opt(exception=True).debug("Image modal: style list failed.")
        self.app.push_screen(
            ConsoleGenerateImageModal(backends=backends, styles=styles),
            callback=self._paste_console_generate_image_command,
        )

    def _paste_console_generate_image_command(self, command: str | None) -> None:
        """Paste the composed /generate-image command into the draft."""
        if not command:
            return
        self._append_to_console_draft(command)

    def _insert_console_caption_prompt(self) -> None:
        """Insert the pre-canned caption prompt for the attached image."""
        self._append_to_console_draft(CONSOLE_CAPTION_PROMPT)

    @staticmethod
    def _draft_addition(current: str, text: str) -> str:
        """Return ``text`` prefixed by a newline only when one is needed.

        Qodo PR #1160: a draft that already ends with a newline gained a
        SECOND one, so inserted text landed after a blank line.

        Args:
            current: The existing draft text.
            text: The text being appended.

        Returns:
            The exact string to concatenate onto ``current``.
        """
        if not current.strip() or current.endswith("\n"):
            return text
        return f"\n{text}"

    def _append_to_console_draft(self, text: str) -> str:
        """Append ``text`` to the active draft on its own line.

        Existing draft text is never replaced (task-1683).

        Args:
            text: The text to append.

        Returns:
            The exact string appended (including any leading newline), so
            callers that need to replace it later can match on it.
        """
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        composer = self._console_composer_or_none()
        if composer is not None and getattr(
            self, "_console_visible_draft_session_id", None
        ) not in (None, session_id):
            composer = None
        if composer is not None:
            current = composer.draft_text()
            addition = self._draft_addition(current, text)
            composer.load_draft(current + addition)
            if session_id:
                # Qodo PR #1160: a stale/closed active_session_id raises
                # KeyError from the store during tab transitions; the
                # composer already holds the text, so this is best-effort.
                try:
                    store.set_session_draft(session_id, composer.draft_text())
                except KeyError:
                    logger.debug("Composer action: session gone before draft save.")
            return addition
        if session_id:
            try:
                current = store.session_draft(session_id)
                addition = self._draft_addition(current, text)
                store.set_session_draft(session_id, current + addition)
                return addition
            except KeyError:
                logger.debug("Composer action: session gone before draft write.")
                return ""
        return ""

    async def _run_console_impersonate(self) -> None:
        """Draft the USER's next message with the current model (task-1683)."""
        controller = self._ensure_console_chat_controller()
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if not session_id:
            return
        self.app.notify("Impersonate: drafting a reply…", severity="information")
        try:
            result = await controller.impersonate_user_reply(session_id)
        except Exception:
            logger.opt(exception=True).warning("Impersonate failed.")
            self.app.notify("Impersonate failed.", severity="error")
            return
        suggestion = (result.text or "").strip()
        if not suggestion:
            # Qodo PR #1160: name the actual cause instead of reporting
            # every empty result as "the model returned nothing".
            message = {
                "provider-not-ready": result.detail
                or "Impersonate needs a ready provider and model.",
                "empty-transcript": (
                    "Impersonate needs at least one message to work from."
                ),
                "provider-error": "Impersonate: the provider call failed.",
                "empty-completion": "Impersonate: the model returned nothing.",
            }.get(result.reason, "Impersonate produced nothing.")
            self.app.notify(message, severity="warning")
            return
        self._replace_console_impersonate_text(session_id, suggestion)

    def _replace_console_impersonate_text(
        self, session_id: str, suggestion: str
    ) -> None:
        """Insert ``suggestion``, replacing a previous one when still present.

        Clicking Impersonate twice must not stack two drafts; the prior
        insertion is swapped out when the draft still ends with it, and
        otherwise the new text is simply appended (the user edited or moved
        it, so silently rewriting their text would be wrong).
        """
        previous = getattr(self, "_console_impersonate_last", {}).get(session_id)
        store = self._ensure_console_chat_store()
        # cubic PR #1160: the user can switch tabs while the draft is
        # generating. Only touch the MOUNTED composer when it still shows
        # this session -- otherwise write the stored draft and let the
        # normal tab sync render it. Same guard _insert_console_dictation
        # already uses for the identical race.
        composer = self._console_composer_or_none()
        if composer is not None and not (
            store.active_session_id == session_id
            and getattr(self, "_console_visible_draft_session_id", session_id)
            == session_id
        ):
            composer = None
        try:
            current = (
                composer.draft_text()
                if composer is not None
                else store.session_draft(session_id)
            )
        except KeyError:
            logger.debug("Impersonate: session gone before insertion.")
            return
        if previous and current.endswith(previous):
            trimmed = current[: len(current) - len(previous)]
            replacement = self._draft_addition(trimmed, suggestion)
            new_draft = trimmed + replacement
            if composer is not None:
                composer.load_draft(new_draft)
            try:
                store.set_session_draft(session_id, new_draft)
            except KeyError:
                logger.debug("Impersonate: session gone before draft save.")
            self._remember_console_impersonate(session_id, replacement)
            return
        appended = self._append_to_console_draft(suggestion)
        self._remember_console_impersonate(session_id, appended)

    def _remember_console_impersonate(self, session_id: str, text: str) -> None:
        if not hasattr(self, "_console_impersonate_last"):
            self._console_impersonate_last = {}
        self._console_impersonate_last[session_id] = text

    async def _run_pending_console_voice_action(
        self, origin_session_id: str | None
    ) -> None:
        """Fire the action a capture-ending `VoiceCommand` queued, if any.

        Only ever reached from `_stop_console_dictation`'s success tail --
        after the transcript (if any) has already been inserted and dictation
        is back at `idle` -- never from the exception branch above, which
        routes through `_notify_console_dictation_error` and drops the
        pending action instead of acting on it. This is what keeps `send`
        from ever shipping a message for a capture that failed to transcribe.

        Args:
            origin_session_id: The session the capture began in, and therefore
                the only session whose draft `send` may ship. The transcript
                was inserted there (`_insert_console_dictation`), while Send
                acts on whatever session is ACTIVE -- so if the user switched
                tabs during the transcribe window, pressing Send would ship a
                different session's half-written draft.
        """
        pending_action, self._console_pending_voice_action = (
            self._console_pending_voice_action,
            None,
        )
        if pending_action == "send":
            store = self._ensure_console_chat_store()
            if origin_session_id and store.active_session_id != origin_session_id:
                # Refuse rather than send the wrong draft. Toasted
                # unconditionally: this is a refusal the user did not ask for
                # and cannot otherwise see, and the dictated text is sitting
                # safely in the origin session's draft either way.
                self.app_instance.notify(_VOICE_ACK_SESSION_CHANGED, severity="warning")
                self._speak_status(_VOICE_ACK_SESSION_CHANGED)
                return
            try:
                send_button = self.query_one("#console-send-message", Button)
            except QueryError:
                logger.debug(
                    "Console voice send skipped; the send button is not mounted"
                )
                return
            # Awaited through the real handler rather than `Button.press()`:
            # a press only posts a message, so its outcome is unknowable here,
            # and the dispatch refuses on several reachable paths (empty
            # draft, send-blocked, a run already in progress, a `/`-command
            # dispatch) -- each with its own toast. Speaking "Sent." over any
            # of those is a straight lie about whether the message went out.
            sent = await self.handle_console_send_message(Button.Pressed(send_button))
            self._speak_status("Sent." if sent else _VOICE_ACK_NOT_SENT)
        elif pending_action == "new-session":
            self.action_new_console_tab()
            self._speak_status("New session.")
        elif pending_action == "read-that-back":
            await self._console_read_last_response_back()

    async def _console_read_last_response_back(self) -> None:
        """Speak the last completed assistant reply for "Console, read that back."

        Mirrors `handle_console_message_action`'s "speak" branch (task-559)
        exactly rather than inventing a second TTS path: post
        `TTSRequestEvent`, track the message as the one currently driving
        speech, and resync so the transcript's action row reflects it. Only
        the completed target selection and the two ack cases are new here.
        """
        # Own-guard for the microphone/speaker mutual-exclusion invariant.
        # The one caller already reaches here at `idle`, so this is defensive
        # rather than load-bearing -- but this method is the only dictation
        # path that speaks UNCONDITIONALLY (an explicit request, not ambient
        # feedback, so it deliberately bypasses `_speak_status`'s toggle), and
        # therefore the only one whose own idle check is not inherited.
        if self._console_dictation_state != "idle":
            return
        if self._console_run_active():
            self.app_instance.notify("Still responding.", severity="warning")
            self._speak_status("Still responding.")
            return
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        message = None
        if session_id:
            for candidate in reversed(store.messages_for_session(session_id)):
                if candidate.role == "assistant" and candidate.status == "complete":
                    message = candidate
                    break
        if message is None:
            self.app_instance.notify("Nothing to read yet.", severity="warning")
            self._speak_status("Nothing to read yet.")
            return
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSRequestEvent,
        )

        self.app_instance.post_message(
            TTSRequestEvent(text=message.content, message_id=message.id)
        )
        self._console_speaking_message_id = message.id
        await self._sync_native_console_chat_ui()

    # ------------------------------------------------------------------
    # V3 pipeline hands-free conversation loop: moved to
    # `ConsoleHandsFreeController` (wave-2 console decomposition, task 1),
    # along with the engine fork and both cross-engine action entry points
    # it shares with the V4 realtime loop below (`action_toggle_console_
    # hands_free` here is the one-line delegation Textual's action dispatch
    # needs to find on this class; `action_exit_console_hands_free` and
    # `check_action`'s `exit_console_hands_free` branch are the same shape,
    # above). `_enter_console_hands_free_loop` is kept as a one-line
    # delegation too, under its ORIGINAL private name -- it is reached from
    # outside this cluster (a spoken "Console, hands free." mid-capture,
    # via `ConsoleDictationController`'s own injected `enter_hands_free_
    # loop` callable, which points straight at `self._hands_free` and does
    # NOT go through this delegation; this one exists for the handful of
    # OTHER call sites, chiefly tests, that reach the fork directly on the
    # screen instance). See `hands_free.py`'s module docstring for the full
    # map of what moved and the two-engine boundary it draws.
    # ------------------------------------------------------------------

    @on(ConsoleHandsFreeToggleRequested)
    def on_console_hands_free_toggle_requested(
        self, event: "ConsoleHandsFreeToggleRequested"
    ) -> None:
        """The visible Switch flipped (task-18911 fix 2): same path as the
        keybinding. The switch is repainted by the session lifecycle sync,
        not here -- entering can fail (e.g. mic unavailable), and the
        control must reflect the session, not the wish."""
        event.stop()
        self._hands_free.action_toggle_console_hands_free()

    def action_toggle_console_hands_free(self) -> None:
        """`ctrl+shift+h`: enter the hands-free loop, or exit it if already
        running.

        One-line delegation (wave-2 console decomposition, task 1). See
        `ConsoleHandsFreeController.action_toggle_console_hands_free` for
        the real implementation.
        """
        self._hands_free.action_toggle_console_hands_free()

    def _enter_console_hands_free_loop(self, *, capture_live: bool) -> None:
        """Pick the hands-free engine, then start that engine's loop.

        One-line delegation (wave-2 console decomposition, task 1). See
        `ConsoleHandsFreeController._enter_console_hands_free_loop` for the
        real implementation (the engine fork).
        """
        self._hands_free._enter_console_hands_free_loop(capture_live=capture_live)

    # ------------------------------------------------------------------
    # Realtime (V4) hands-free loop: one live provider session for the
    # whole conversation. `Chat/console_realtime_loop.py`
    # (`RealtimeLoopController`, the headless FSM), `Audio/realtime_mic_
    # tap.py` (the raw 24 kHz mic tap), `LLM_Calls/realtime/` (the
    # provider-neutral session protocol + the OpenAI transport) and
    # `Audio/streaming_sink.py` (reply audio playback) are all pure/
    # headless; this section is their thin Console-screen wiring, mirroring
    # the V3 section above one for one. See `.superpowers/sdd/
    # 2026-08-04-realtime-voice-engine/`.
    # ------------------------------------------------------------------

    def _repaint_console_realtime_chip(self) -> None:
        """Paint the realtime loop's mode into the composer's voice chip.

        Driven through `set_voice_status` for every state, unlike V3 --
        which restores the ordinary dictation chip while `listening`
        because the one-shot pipeline is painting it. Nothing else paints
        during a realtime loop: the microphone here belongs to the tap, not
        to `_console_dictation_state`, which stays `idle` throughout.
        """
        session = self._console_realtime
        if session is None:
            return
        composer = self._console_composer_or_none()
        if composer is None:
            return
        message = CONSOLE_REALTIME_CHIP_MESSAGES.get(session.controller.state)
        if message is None:
            return
        composer.set_voice_status(session.controller.state, message=message)

    def _restore_console_voice_chip(self) -> None:
        """Repaint the chip from the REAL one-shot dictation state.

        Same idiom (and same reason) as `_teardown_console_hands_free_
        loop`'s closing lines: the realtime states are not lifecycle states
        `sync_dictation_state` knows, so only a fresh call with the actual
        current state clears the borrowed text.
        """
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.sync_dictation_state(self._console_dictation_state)

    def _request_console_dictation_stop(self) -> None:
        """Stop the live capture and insert its transcript.

        One-line delegation (wave-1 console decomposition, task 5). Called
        from the mic button (`on_button_pressed`, `recording` state) and
        from the hands-free wiring's own close-capture / force-send /
        limit-hit paths. See `ConsoleDictationController._request_console_
        dictation_stop` for the real implementation.
        """
        self._dictation._request_console_dictation_stop()

    def _request_console_dictation_cancel(self) -> None:
        """Abandon a `starting`/`recording` capture without inserting.

        One-line delegation (wave-1 console decomposition, task 5). Called
        from the mic button (`on_button_pressed`, `starting` state). See
        `ConsoleDictationController._request_console_dictation_cancel` for
        the real implementation.
        """
        self._dictation._request_console_dictation_cancel()

    def _request_console_dictation_start(self) -> None:
        """Open the microphone for a fresh one-shot dictation capture.

        One-line delegation (wave-1 console decomposition, task 5). Called
        from the mic button (`on_button_pressed`, `idle` state) and from
        the hands-free wiring's own open-capture path. See `ConsoleDictation
        Controller._request_console_dictation_start` for the real
        implementation.
        """
        self._dictation._request_console_dictation_start()

    def _console_transcript_region_or_none(self) -> ConsoleTranscriptRegion | None:
        """Return the mounted transcript region, or ``None`` before compose.

        A region widget only exists once the screen has composed, and a
        recompose replaces the instance -- so it is reached by id, never
        stored on the screen (the same way ``_sync_console_rail_visibility``
        reaches ``ConsoleLeftRail``).

        Note that ``NoMatches`` is a ``QueryError``, and so is ``WrongType``:
        an id collision that put some other widget class at
        ``#console-main-column`` degrades to ``None`` here rather than
        raising. That is deliberate — every caller already handles the
        not-yet-composed case — but it means this must never become the
        place a shape error is expected to surface.

        Returns:
            The ``#console-main-column`` region widget, or ``None`` when the
            shell is not (yet) mounted.
        """
        try:
            return self.query_one("#console-main-column", ConsoleTranscriptRegion)
        except QueryError:
            return None

    def _capture_console_transcript_reading_state(
        self,
    ) -> _ConsoleTranscriptReadingState | None:
        """Capture the semantic reading position before composer layout changes.

        Delegates to ``ConsoleTranscriptRegion.capture_reading_state``.

        Returns:
            The transcript's reading state, or ``None`` when unmounted.
        """
        region = self._console_transcript_region_or_none()
        return None if region is None else region.capture_reading_state()

    def _restore_console_transcript_reading_state(
        self,
        state: _ConsoleTranscriptReadingState | None,
    ) -> None:
        """Restore the transcript anchor, offset, and selected message.

        Delegates to ``ConsoleTranscriptRegion.restore_reading_state``.

        Args:
            state: The reading state captured before the layout change.
        """
        region = self._console_transcript_region_or_none()
        if region is not None:
            region.restore_reading_state(state)

    def _set_console_composer_collapsed(self, collapsed: bool) -> None:
        """Synchronize screen-owned collapse state with the mounted composer."""
        if self._console_setup_modal_blocking():
            return
        collapsed = bool(collapsed)
        composer = self._console_composer_or_none()
        if composer is None or self._console_composer_collapsed == collapsed:
            return
        reading_state = self._capture_console_transcript_reading_state()
        self._console_composer_collapsed = collapsed
        self._console_composer_layout_revision += 1
        revision = self._console_composer_layout_revision
        if collapsed:
            self._console_unknown_send_armed = None
            composer.reset_pending_unfurl()
            # Unconditional hide (not _sync_console_command_popup): the draft
            # may still match a completion context and would re-show it.
            popup = self._console_command_popup_or_none()
            if popup is not None:
                popup.hide()
        composer.set_collapsed(collapsed)
        composer.refresh(layout=True)
        self.call_after_refresh(
            self._finish_console_composer_layout_change,
            revision,
            collapsed,
            reading_state,
        )

    def _set_console_status_chips_collapsed(self, collapsed: bool) -> None:
        """Synchronize screen-owned collapse state with the mounted status row."""
        if self._console_setup_modal_blocking():
            return
        collapsed = bool(collapsed)
        if self._console_status_chips_collapsed == collapsed:
            return
        try:
            status_chips = self.query_one("#console-status-chips", ConsoleStatusChips)
        except QueryError:
            return
        self._console_status_chips_collapsed = collapsed
        # task-17652: the collapse choice survives Console re-entry and app
        # restart. Poke the live config synchronously (compose and __init__
        # read it back) and move only the disk write off the event loop —
        # never-raising, since a worker exception is fatal by default.
        poke_console_setting(
            getattr(self.app_instance, "app_config", None),
            "status_chips_collapsed",
            collapsed,
        )
        self.run_worker(
            partial(persist_status_chips_collapsed, collapsed),
            thread=True,
            group="console-status-row-pref",
            exclusive=True,
        )
        self._console_status_chips_layout_revision += 1
        revision = self._console_status_chips_layout_revision
        status_chips.set_collapsed(collapsed)
        status_chips.refresh(layout=True)
        self.call_after_refresh(
            self._finish_console_status_chips_layout_change,
            revision,
            collapsed,
        )

    def _finish_console_status_chips_layout_change(
        self,
        revision: int,
        expected_collapsed: bool,
    ) -> None:
        """Focus the inverse control after the latest status-row transition."""
        if (
            revision != self._console_status_chips_layout_revision
            or expected_collapsed != self._console_status_chips_collapsed
        ):
            return
        target_id = (
            "console-status-expand" if expected_collapsed else "console-status-collapse"
        )
        try:
            self.query_one(f"#{target_id}", Button).focus()
        except QueryError:
            return

    def _finish_console_composer_layout_change(
        self,
        revision: int,
        expected_collapsed: bool,
        reading_state: _ConsoleTranscriptReadingState | None,
    ) -> None:
        """Finish only the latest requested composer layout transition."""
        if (
            revision != self._console_composer_layout_revision
            or expected_collapsed != self._console_composer_collapsed
        ):
            return
        self._restore_console_transcript_reading_state(reading_state)
        if expected_collapsed:
            self._focus_console_workbench_target("console-transcript-surface")
        else:
            self._focus_console_workbench_target("console-native-composer")

    def _build_console_control_state(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleControlState:
        """Build Console-owned control/readiness labels."""
        provider, model, settings = self._active_console_provider_model_display()
        active_session = self._session._active_native_console_session()
        return ConsoleControlState.from_values(
            provider=provider,
            model=model,
            # The AI side of the conversation: whatever character this session is
            # actually roleplaying, so the chip stops being a constant. The
            # session's `character_label` is set by the character handoff; the
            # rail name covers resumed conversations.
            character=(
                getattr(settings, "character_label", None)
                or self._character._current_console_rail_character_name()
            ),
            assistant_kind=getattr(active_session, "assistant_kind", None),
            assistant_name=getattr(active_session, "assistant_name", None),
            assistant_id=getattr(active_session, "assistant_id", None),
            library_policy=(
                active_session.library_policy_holder.snapshot
                if active_session is not None
                else None
            ),
            # RAG UX v2 PR-4: was hardcoded to 1 while the staged bundle
            # routinely carries several references -- a four-result Library
            # RAG run advertised "Sources: 1 staged".
            staged_source_count=console_staged_source_count(pending_launch),
            tool_count=self._console_tool_count(),
            mcp_tool_count=self._console_mcp_tool_count(),
            approval_count=self._console_pending_approval_count(),
            system_prompt_set=bool(getattr(settings, "system_prompt", None)),
        )

    def _build_console_cost_state(self) -> ConsoleCostState | None:
        """Build the cost chip's display state for the active session (task-5).

        Returns ``None`` when there is no active NATIVE Console session (the
        chip renders hidden) -- this is a normal condition, not a failure,
        so it is not subject to the best-effort fallback below.

        Everything past that point is best-effort: an unexpected failure is
        logged and this returns the last computed state (``None`` if there
        never was one) rather than raising into the sync path -- a stale or
        missing chip is fine, a broken send is not.
        """
        try:
            session = self._session._active_native_console_session()
            store = self._console_chat_store
            if session is None or store is None:
                self._console_cost_cache_state = ConsoleCacheState.NONE
                return None
            session_id = session.id
            try:
                messages = store.messages_for_session(session_id)
            except KeyError:
                self._console_cost_cache_state = ConsoleCacheState.NONE
                return None

            # Keep the chip stable until provider rows finish; direct raw
            # commands are local actions and never provider-billed rows.
            snapshot_messages = console_cost_snapshot_messages(messages)
            # task-6: staged (not-yet-sent) evidence used to be invisible to
            # the cost chip entirely -- `ConsoleStagedSource` carries no
            # text, so a session with zero messages but several staged
            # sources (even a 942 KB one) showed "0 tok". Feed the same
            # formatted pre-authority staged-text estimate the context
            # estimate now counts (`console_prompted_evidence_text`,
            # pure/zero-I/O) in as one
            # more transcript row satisfying `build_cost_snapshot`'s
            # duck-typed contract (`.role`/`.content`/`.usage`) with
            # `usage=None` -- it prices through the ESTIMATED-row branch,
            # at the input rate, and flips `has_estimated_entries` so the
            # chip's `~` prefix (its existing "this includes an unsent
            # estimate" marker) shows rather than claiming a real total.
            staged_text = console_prompted_evidence_text(
                self._pending_console_launch_context
            )
            if staged_text:
                snapshot_messages = snapshot_messages + [
                    SimpleNamespace(role="user", content=staged_text, usage=None)
                ]
            provider, model, _settings = self._active_console_provider_model_display()
            # PR2b Task 5 (cost rollup): the active conversation's LIVE
            # sub-agent fleet spend, folded into the snapshot's token total
            # (never priced -- see `ConsoleCostSnapshot.fleet_tokens`'s
            # docstring for why). Read straight off the SAME live source
            # the fleet rail rows themselves read
            # (`_console_agent_fleet_token_total` sums `bridge.fleet_
            # snapshot(...)`'s `FleetHandle.total_tokens`), so the chip and
            # the rail can never disagree about a conversation's fleet
            # spend.
            fleet_tokens = self._agent._console_agent_fleet_token_total()
            # PR3a-1 Task 6b (audit F3): plus whatever a SURVIVING child
            # billed after its turn's usage was attached to the assistant
            # message. PR3a-2 Task 3 (tasks 15660/15667): that spend is now
            # INTERIM, not lost -- when the conversation's last fleet child
            # settles, the controller's "usage-reattach" drain consumer
            # folds the whole turn (survivors included) back onto the
            # message's own usage row and this line falls to zero. Until
            # that drain, the money is named here on the chip's unpriced
            # sub-agent line. No double count: `unattributed_fleet_tokens`
            # counts ONLY payloads closed out after the latest attach (the
            # fold resets its watermark), and a live handle's
            # `FleetHandle.total_tokens` -- what `_console_agent_fleet_token_
            # total` sums -- is 0 until it finishes, by which point it has
            # left `fleet_snapshot`.
            cost_controller = self._console_chat_controller
            if cost_controller is not None:
                unattributed = getattr(
                    cost_controller, "unattributed_fleet_tokens", None
                )
                if callable(unattributed):
                    fleet_tokens += int(unattributed(session_id) or 0)
            # task-15451: this method runs on the 0.2s tick for the whole
            # duration of a run (plus every control-bar sync pass and the
            # 10s TTL timer), and the equality guard in
            # `_sync_console_cost_chip` gates only the REPAINT -- the build
            # itself always ran. Without the memo below every usage-less row
            # (all user/system rows, legacy assistant rows, the staged
            # evidence pseudo-row) was re-tokenized by a per-character
            # Python loop 5x/s: ~28ms/tick on a 99KB transcript, measured.
            # The memo re-verifies each row's own text before serving a hit,
            # so it can change how long this takes but not what it returns.
            #
            # Gating the whole snapshot on `store.payload_revision` instead
            # was considered and rejected: usage is not payload-affecting, so
            # `ConsoleChatStore.set_message_usage` never bumps that counter --
            # a real priced usage landing on an ALREADY-terminal row (the
            # documented Stop-path ordering) would leave the chip showing the
            # estimated total until some unrelated edit moved the revision.
            estimate_cache = self._console_cost_estimate_cache_or_new()
            snapshot = build_cost_snapshot(
                snapshot_messages,
                provider=provider,
                model=model,
                fleet_tokens=fleet_tokens,
                estimate_cache=estimate_cache,
            )

            controller = self._console_chat_controller
            run_status = (
                controller.run_state_for(session_id).status
                if controller is not None
                else ConsoleRunStatus.IDLE
            )
            # Fingerprint compare is the expensive step (rebuilds the
            # pre-compaction provider payload) -- only pay it when the
            # session isn't actively streaming AND its payload has actually
            # changed since the last check (revision `!=`, not `>`: a
            # restore can reset the store's counter back down).
            break_reason = self._console_cost_break_reasons.get(session_id)
            if controller is not None and run_status not in CONSOLE_ACTIVE_RUN_STATUSES:
                current_revision = store.payload_revision(session_id)
                if current_revision != self._console_cost_fp_revisions.get(session_id):
                    baseline = controller.payload_fingerprint_baseline(session_id)
                    break_reason = None
                    if baseline is not None:
                        current_fp = controller.compute_current_fingerprint(session_id)
                        break_reason = fingerprint_break_reason(baseline, current_fp)
                    self._console_cost_fp_revisions[session_id] = current_revision
                    self._console_cost_break_reasons[session_id] = break_reason

            cache_state = ConsoleCacheState.NONE
            ttl_remaining_s: float | None = None
            if controller is not None:
                warm_until, had_activity = controller.cache_ttl_snapshot(session_id)
                if had_activity and warm_until is not None:
                    now = time.monotonic()
                    if now < warm_until:
                        cache_state = ConsoleCacheState.WARM
                        ttl_remaining_s = warm_until - now
                    else:
                        cache_state = ConsoleCacheState.EXPIRED
            self._console_cost_cache_state = cache_state

            projected_delta_usd: float | None = None
            pricing_as_of: str | None = None
            catalog = get_pricing_catalog()
            provider_key = provider_config_key(provider)
            pricing = catalog.get_pricing(provider_key, model or "")
            if pricing is not None:
                pricing_as_of = pricing.as_of
                if (
                    cache_state == ConsoleCacheState.WARM
                    and break_reason
                    and pricing.cache_write_per_mtok is not None
                    and pricing.cache_read_per_mtok is not None
                ):
                    # `break_reason` is required here (Qodo round, finding
                    # 3): `build_cost_state`/`_cache_state_line` only ever
                    # read `projected_delta_usd` inside their own
                    # `break_reason`-gated branches (the alert suffix on the
                    # label, and the "~+$" clause in the tooltip's cache
                    # line) -- with no break reason the value is built and
                    # then silently discarded every call. Skipping the
                    # (expensive: `_estimate_tokens_locally` over the WHOLE
                    # transcript) computation here avoids that on every
                    # 0.2s/10s sync tick for a long-running WARM session
                    # that never alerts.
                    #
                    # `snapshot_messages` (not `messages`): same mid-stream-
                    # animation guard as the snapshot above -- an in-flight
                    # row's growing content must not grow the projected
                    # break-delta either, e.g. when a NEW turn starts
                    # streaming while a PRIOR turn's alert is still showing
                    # (fingerprint recompute -- and so `break_reason` /
                    # `alert` -- is frozen during the run, but this
                    # projection is computed fresh every call).
                    #
                    # task-15451: gated, but not cheap -- an alerting
                    # session pays a WHOLE-transcript estimate on every tick
                    # for as long as the alert stands. Same memo, same
                    # guarantee: the hit is verified against every row's
                    # (role, content) before it is served.
                    projection_rows = tuple(
                        (
                            str(getattr(message.role, "value", message.role)),
                            message.content,
                        )
                        for message in snapshot_messages
                    )

                    def _estimate_projection() -> int:
                        return _estimate_tokens_locally(
                            [
                                {"role": role, "content": content}
                                for role, content in projection_rows
                            ],
                            model or "",
                            provider_key,
                        )

                    projection_cache = self._console_cost_estimate_cache_or_new()
                    estimated_tokens = projection_cache.estimate(
                        ("#cost-projection", session_id),
                        token_estimate_signature(
                            projection_rows, model or "", provider_key
                        ),
                        _estimate_projection,
                    )
                    rate_delta = (
                        pricing.cache_write_per_mtok - pricing.cache_read_per_mtok
                    ) / 1_000_000
                    projected_delta_usd = round(estimated_tokens * rate_delta, 6)

            return build_cost_state(
                snapshot,
                cache_state=cache_state,
                break_reason=break_reason,
                projected_delta_usd=projected_delta_usd,
                ttl_remaining_s=ttl_remaining_s,
                pricing_as_of=pricing_as_of,
            )
        except Exception:
            logger.opt(exception=True).warning("cost_chip_state_failed")
            return self._last_console_cost_state

    def _sync_console_cost_chip(self) -> None:
        """Refresh the cost chip from freshly built state (task-5).

        Called at the END of ``_sync_console_control_bar`` -- deliberately
        OUTSIDE that method's ``control_state_changed``/
        ``workbench_state_changed`` guard (mirroring the unconditional
        inspector build right above it there), since cost/cache state can
        change independently of whether the control labels changed -- and
        from the 10s TTL repaint timer below so a WARM cache that goes
        EXPIRED with no other sync call in between still repaints.
        """
        cost_state = self._build_console_cost_state()
        if cost_state != self._last_console_cost_state:
            self._last_console_cost_state = cost_state
            try:
                status_chips = self.query_one(
                    "#console-status-chips", ConsoleStatusChips
                )
            except QueryError:
                status_chips = None
            if status_chips is not None:
                status_chips.sync_cost_state(cost_state)
        if self._console_cost_cache_state == ConsoleCacheState.WARM:
            self._start_console_cost_ttl_timer()
        else:
            self._stop_console_cost_ttl_timer()

    def _start_console_cost_ttl_timer(self) -> None:
        """Start the 10s WARM->EXPIRED cost-chip repaint timer (task-5).

        No-ops when already running. Mirrors
        ``_start_console_transcript_sync_timer``'s audit pairing, but on its
        own cadence and stop condition -- see that method's docstring for
        why the 0.2s tick alone cannot cover this repaint.
        """
        if self._console_cost_ttl_timer is not None:
            return
        self._console_cost_ttl_timer = self.set_interval(
            CONSOLE_COST_TTL_TICK_SECONDS, self._sync_console_cost_chip
        )
        self._record_ui_timer_created("console-cost-ttl")

    def _stop_console_cost_ttl_timer(self) -> None:
        """Stop the cost-chip TTL repaint timer, if running."""
        if self._console_cost_ttl_timer is None:
            return
        try:
            self._console_cost_ttl_timer.stop()
        finally:
            self._record_ui_timer_stopped("console-cost-ttl")
            self._console_cost_ttl_timer = None

    def _build_console_staged_context_state(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleStagedContextState:
        if pending_launch is None:
            return ConsoleStagedContextState.empty()
        return ConsoleStagedContextState.from_live_work(pending_launch)

    def _sync_console_retrieval_scope_row(self) -> None:
        """Refresh the mounted retrieval-scope row AND status-strip chips.

        Task-10: the status-pills strip's ``#console-scope-chip`` (above the
        composer) renders from the
        exact same ``ConsoleRetrievalScopeState`` snapshot as the
        Inspector row -- computed once here and pushed into both, never a
        second state source or a second cache. This keeps the chip's
        refresh triggers identical to the row's (this method's own two
        call sites: after a scope-picker save, and the first-send
        persist-flush hook).

        Task-7 (temporary chip): this is also the one place ``#console-
        temporary-chip`` is refreshed for every session-switch trigger this
        method already covers (resume, tab activation, scope-picker save,
        first-persist flush) -- see ``ConsoleTemporaryChip``/
        ``sync_temporary_chip`` for why it is kept off the general
        control-bar sync tick, same as the scope chip.
        """
        state = self._retrieval._build_console_retrieval_scope_state()
        try:
            row = self.query_one(
                f"#{CONSOLE_RETRIEVAL_SCOPE_ROW_ID}", ConsoleRetrievalScopeRow
            )
        except QueryError:
            pass
        else:
            row.sync_state(state)
        try:
            status_chips = self.query_one("#console-status-chips", ConsoleStatusChips)
        except QueryError:
            return
        status_chips.sync_scope_chip(state)
        status_chips.sync_temporary_chip(self._console_active_session_is_ephemeral())

    async def _open_console_retrieval_scope_picker(self) -> None:
        """Open the RAG retrieval-scope picker for the active Console session.

        Reads the session's current scope off-loop before constructing the
        modal (so the modal's ``initial`` selection is accurate) -- a
        persisted session's stored scope via ``read_conversation_scope``,
        an unpersisted session's held ``SessionScopeHolder`` value (already
        in memory, no I/O).

        Also reads the linked workspace's scope (task-13, spec decision D3:
        "conversation narrows within workspace"): when the workspace has an
        active scope, the modal's ``universe`` is restricted to exactly
        that scope's items, so the conversation picker only ever offers
        (and lets "Select all matching" select) content already in the
        workspace's scope. No workspace scope set -> ``universe=None``
        (today's full-library behavior, byte-identical).
        """
        session = self._session._active_native_console_session()
        if session is None:
            return
        conversation_id = session.persisted_conversation_id
        if conversation_id is not None:
            db = getattr(self.app_instance, "chachanotes_db", None)
            initial = (
                await self._retrieval._read_console_retrieval_scope(db, conversation_id)
                if db is not None
                else None
            )
            self._console_retrieval_scope_cache[conversation_id] = initial
        else:
            initial = session.rag_scope_holder.scope

        universe: Optional[frozenset[tuple[str, str]]] = None
        workspace_id = getattr(session, "workspace_id", None)
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if workspace_id and registry_service is not None:
            try:
                ws_scope = await self._workspace._read_console_workspace_scope(
                    registry_service, workspace_id
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Unable to read workspace scope for {} while opening the "
                    "conversation scope picker",
                    workspace_id,
                )
                ws_scope = None
            if ws_scope is not None:
                universe = frozenset(
                    (item.source_type, item.source_id) for item in ws_scope.items
                )

        title = sanitize_character_display_label(
            session.title,
            max_characters=500,
        )
        target_label = title or "this conversation"
        media_lister, notes_lister, tag_lister = (
            self._retrieval._console_scope_picker_listers()
        )

        def _on_save(scope: Optional[RagScope]) -> None:
            self.run_worker(
                self._retrieval._apply_console_retrieval_scope_save(session, scope),
                exclusive=True,
                group="console-retrieval-scope-save",
            )

        self.app.push_screen(
            ConsoleScopePickerModal(
                target_label,
                universe,
                initial,
                _on_save,
                media_lister=media_lister,
                notes_lister=notes_lister,
                tag_lister=tag_lister,
            )
        )

    async def _clear_console_retrieval_scope(self) -> None:
        """Clear the active Console session's RAG retrieval scope."""
        session = self._session._active_native_console_session()
        if session is None:
            return
        await self._retrieval._apply_console_retrieval_scope_save(session, None)

    @on(Button.Pressed, ".console-retrieval-scope-open-btn")
    async def _console_retrieval_scope_open_pressed(
        self, event: Button.Pressed
    ) -> None:
        event.stop()
        await self._open_console_retrieval_scope_picker()

    @on(Button.Pressed, ".console-retrieval-scope-clear-btn")
    async def _console_retrieval_scope_clear_pressed(
        self, event: Button.Pressed
    ) -> None:
        event.stop()
        await self._clear_console_retrieval_scope()

    @on(ConsoleModelChip.OpenRequested)
    async def _console_model_chip_activated(
        self, event: ConsoleModelChip.OpenRequested
    ) -> None:
        """Open the quick model popover from the Provider/Model chips.

        task-1670: a second entry point into the same opener Alt+M uses,
        following the scope-chip precedent below.
        """
        event.stop()
        await self.action_open_console_model_popover()

    @on(ConsoleAssistantChip.OpenRequested)
    async def _console_assistant_chip_activated(
        self, event: ConsoleAssistantChip.OpenRequested
    ) -> None:
        """Open the character picker from the Character/Assistant chip."""
        event.stop()
        await self._open_console_character_picker()

    @on(ConsoleSystemPromptChip.OpenRequested)
    def _console_system_prompt_chip_activated(
        self, event: ConsoleSystemPromptChip.OpenRequested
    ) -> None:
        """Open the system prompt editor from the System Prompt chip.

        A third entry point into the same opener ``/system`` and the
        command palette use, following the model/assistant-chip precedent.
        """
        event.stop()
        self.action_open_console_system_prompt_editor()

    @on(ConsoleLibraryChip.OpenRequested)
    def _console_library_chip_activated(
        self, event: ConsoleLibraryChip.OpenRequested
    ) -> None:
        """Open the active conversation's two-axis Library policy editor."""
        event.stop()
        self._library_policy.open_access()

    @on(ConsoleSourcesChip.OpenRequested)
    def _console_sources_chip_activated(
        self, event: ConsoleSourcesChip.OpenRequested
    ) -> None:
        """Open the Inspector rail at the staged-sources tray (DS-06/LY-11).

        Below 150 cols the Inspector is the ONLY surface for staged
        sources; the compact-collapse override (TASK-2154.2) makes this
        work at every width, exactly like the rail handle.
        """
        event.stop()
        self._reveal_console_inspector_rail()

    @on(ConsoleToolsChip.OpenRequested)
    def _console_tools_chip_activated(
        self, event: ConsoleToolsChip.OpenRequested
    ) -> None:
        """Open the Inspector rail at the run inspector's tool rows (DS-06)."""
        event.stop()
        self._reveal_console_inspector_rail()

    @on(ConsoleRunChip.OpenRequested)
    def _console_run_chip_activated(self, event: ConsoleRunChip.OpenRequested) -> None:
        """Open the Inspector rail at the live run rows (FB-08)."""
        event.stop()
        self._reveal_console_inspector_rail()

    def _reveal_console_inspector_rail(self) -> None:
        """Open the Inspector rail at any width and focus it.

        TASK-2154.2 (DS-06): the chips' shared reveal path. Opening goes
        through the same preference seam as the rail handle, so the
        compact-collapse override honors it below 150 cols too. Focusing
        the rail afterwards means activation produces visible feedback
        even when the rail was already open (the focus frame repaints).
        """
        self._set_console_rail_preference(right_open=True)
        try:
            rail = self.query_one("#console-right-rail")
        except QueryError:
            return
        rail.focus()

    @on(ConsoleCostChip.ConsoleCostChipPressed)
    def _console_cost_chip_activated(
        self, event: ConsoleCostChip.ConsoleCostChipPressed
    ) -> None:
        """Open the Conversation Inspector's Costs tab from the cost chip (task-5/8)."""
        event.stop()
        self._open_console_cost_breakdown()

    def _open_console_cost_breakdown(self) -> None:
        """Push the Conversation Inspector on the Costs tab for the active
        native session (task-5/8).

        task-8: this used to push a standalone modal (retired in task-10);
        it now pushes the shared ``ConsoleConversationInspector`` (same
        modal Ctrl+Shift+P opens, just starting on a different tab). The
        Next Send factories are built too (via ``_console_inspector_next_
        send_factories``) so switching to that tab after opening from the
        chip renders real content, not stale/empty data.
        """
        controller = self._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        if not session_id:
            self.notify("No active conversation.", severity="warning")
            return
        factory, estimate_factory, token_estimate, in_progress = (
            self._console_inspector_next_send_factories(controller, session_id)
        )
        self._push_console_inspector(
            initial_tab=TAB_COSTS,
            snapshot_factory=factory,
            estimate_factory=estimate_factory,
            token_estimate=token_estimate,
            in_progress=in_progress,
        )

    def _push_console_inspector(
        self,
        *,
        initial_tab: str,
        snapshot_factory: Callable[[], Awaitable[ConsoleContextSnapshot]],
        estimate_factory: Callable[[], int | None] | None = None,
        token_estimate: int | None = None,
        in_progress: bool = False,
        project_instruction_state: ConsoleProjectInstructionState | None = None,
        project_instruction_state_factory: Callable[
            [], Awaitable[ConsoleProjectInstructionState]
        ]
        | None = None,
        project_instruction_session_id: str | None = None,
        project_instruction_recovery: Callable[
            [str | None, str], Awaitable[ConsoleProjectInstructionState | None]
        ]
        | None = None,
    ) -> None:
        """Build the Costs-tab inputs and push the shared inspector (task-8).

        Rows/totals are computed once here (``build_cost_rows``/
        ``build_cost_rows_totals`` are already best-effort and never raise
        on their own) and handed to the modal at construction -- the modal
        itself never queries the store directly, the same "already
        computed, just render it" shape the standalone modal it replaced
        (pre-task-8) used.

        The ``project_instruction_*`` kwargs (task-18300) all default to
        ``None`` -- ``_open_console_cost_breakdown`` (the cost-chip entry
        point) never passes them, so its Next Send tab simply mounts no
        project-instructions panel; only ``action_view_chat_context``
        supplies them, via ``project_instruction_ui.
        project_instruction_context_kwargs``.
        """
        rows, totals, turns, exchanges_loader = (
            self._build_console_inspector_cost_data()
        )
        controller = self._ensure_console_chat_controller()
        capture_policy_wiring = build_inspector_capture_policy_wiring(controller)
        if capture_policy_wiring is None:
            return
        inspector = ConsoleConversationInspector(
            rows=rows,
            totals=totals,
            turns=turns,
            exchanges_loader=exchanges_loader,
            snapshot_factory=snapshot_factory,
            token_estimate=token_estimate,
            estimate_factory=estimate_factory,
            in_progress=in_progress,
            ephemeral=self._console_active_session_is_ephemeral(),
            initial_tab=initial_tab,
            project_instruction_state=project_instruction_state,
            project_instruction_state_factory=project_instruction_state_factory,
            project_instruction_session_id=project_instruction_session_id,
            project_instruction_recovery=project_instruction_recovery,
            target_session_id=capture_policy_wiring.target_session_id,
            target_conversation_id=capture_policy_wiring.target_conversation_id,
            capture_revision_provider=capture_policy_wiring.capture_revision,
            capture_policy_bindings=capture_policy_wiring.bindings,
        )
        capture_policy_wiring.bind_inspector(inspector)
        self.app.push_screen(inspector)

    def _build_console_inspector_cost_data(
        self,
    ) -> tuple[
        list[ConsoleCostRow],
        ConsoleCostRowTotals,
        list[InspectorTurn],
        Callable[[str], Awaitable[list[tuple[ExchangeCapture, bool]]]],
    ]:
        """Shared Costs-tab inputs for ``ConsoleConversationInspector``
        (task-8, extended task-9 for the Exchange tab).

        Returns:
            ``(rows, totals, turns, exchanges_loader)`` -- ``rows``/
            ``totals`` are ``build_cost_rows``/``build_cost_rows_totals``'s
            output; ``turns`` is one :class:`InspectorTurn` per transcript
            message (NOT filtered to contributing ones -- the
            contributing-only property is enforced downstream, in
            ``ConsoleConversationInspector``); ``exchanges_loader`` is
            called by the modal with one turn's ``native_message_id`` and
            returns ``(capture, abandoned)`` pairs (see
            ``console_conversation_inspector``'s module docstring for the
            pair contract and the ordering caveat -- callers must NOT trust
            the returned order, only ``(created_at, seq)``).

            The loader checks the NATIVE (in-memory) store first --
            ``message.exchanges`` on the matching ``ConsoleChatMessage`` --
            and only falls back to a DB read when there is none, so an
            EPHEMERAL session (no ``persisted_message_id``, no DB row at
            all) still resolves its captures; a native capture wins over a
            DB one when both exist (the native copy is fresher -- see
            ``ConsoleChatStore.attach_message_exchanges``). task-9 closed
            the former "known gap" here: a native capture's ``abandoned``
            flag is now resolved through ``store.abandoned_exchange_run_
            tags`` (the store's new public accessor over its private
            ``_abandoned_exchange_run_tags`` bookkeeping) rather than
            always reporting ``False``.
        """
        store = self._console_chat_store
        messages: list[Any] = []
        if store is not None and store.active_session_id is not None:
            try:
                messages = store.messages_for_session(store.active_session_id)
            except KeyError:
                messages = []
        provider, model, _settings = self._active_console_provider_model_display()
        try:
            rows = build_cost_rows(messages, provider=provider, model=model)
        except Exception:
            logger.opt(exception=True).warning("cost_breakdown_rows_failed")
            rows = []
        totals: ConsoleCostRowTotals = build_cost_rows_totals(rows)

        turns = [
            InspectorTurn(
                message_id=message.persisted_message_id or "",
                native_message_id=message.id,
                index=index,
                role=(
                    message.role.value
                    if isinstance(message.role, ConsoleMessageRole)
                    else str(message.role)
                ),
                preview=_console_inspector_turn_preview(message.content),
            )
            for index, message in enumerate(messages)
        ]
        messages_by_native_id = {message.id: message for message in messages}
        exchanges_loader = _build_console_inspector_exchanges_loader(
            messages_by_native_id,
            lambda: getattr(self.app_instance, "chachanotes_db", None),
            store.abandoned_exchange_run_tags if store is not None else None,
        )

        return rows, totals, turns, exchanges_loader

    def _open_console_library_search(self) -> None:
        """Open one-shot Library search from the retrieval controller."""
        self._retrieval.open_library_search()

    def _set_console_library_rag_query(self, query: str) -> None:
        """Store the Library query and mirror it into mounted surfaces."""
        self._console_library_rag_query = query
        try:
            rail_input = self.query_one("#console-library-rag-query-input", Input)
        except QueryError:
            pass
        else:
            rail_input.value = query
        try:
            run_button = self.query_one("#console-run-library-rag", Button)
        except QueryError:
            pass
        else:
            run_button.disabled = False

    def _set_console_library_rag_source_scope(self, source_types: Any) -> None:
        """Store which Library source kinds retrieval reads, and re-label.

        The single writer, mirroring `_set_console_library_rag_query`: the
        readiness card's source line is updated in place so it cannot
        disagree with what the next retrieval will actually read.

        Args:
            source_types: The chosen selection (normalized here, so an
                empty or unknown value falls back to the default rather
                than retrieving over nothing).
        """
        self._console_library_rag_source_types = normalize_console_rag_source_types(
            source_types
        )
        try:
            scope_label = self.query_one("#console-library-rag-scope", Static)
        except QueryError:
            return
        scope_label.update(self._retrieval._console_library_rag_scope_label())

    async def _open_console_character_picker(self) -> None:
        """Load characters off-thread and open the picker modal (task-1672)."""
        options = await asyncio.to_thread(
            self._character._console_character_picker_options
        )
        if not options:
            self.app.notify(
                "No characters saved yet — import a card in Roleplay first.",
                severity="information",
            )
            return
        self.app.push_screen(
            ConsoleCharacterPickerModal(
                options=options,
                current_character_id=(
                    self._character._current_console_rail_character_id()
                ),
            ),
            callback=self._apply_console_character_choice,
        )

    def _apply_console_character_choice(
        self, choice: "ConsoleCharacterChoice | None"
    ) -> None:
        """Route the picker result to a swap or a new character session."""
        if choice is None:
            return
        self.run_worker(
            self._character._apply_console_character_choice_async(choice),
            exclusive=True,
            group="console-character-pick",
        )

    @on(ConsoleScopeChip.OpenRequested)
    async def _console_scope_chip_activated(
        self, event: ConsoleScopeChip.OpenRequested
    ) -> None:
        """Open the scope picker from the status-pills strip chip (task-10).

        Same handler seam as the Inspector row's Edit/Narrow… button
        (``_console_retrieval_scope_open_pressed`` above) -- just a second
        entry point into the same async opener.
        """
        event.stop()
        await self._open_console_retrieval_scope_picker()

    def _current_console_conversation_id(self) -> Optional[str]:
        """Return the active native Console session's persisted conversation id.

        One-line delegation to the session controller (task-16815): the
        browser consolidation (520b1ec12) and the ``/research`` delivery
        (e1f3a4424) both call this name on the screen, but the method only
        existed on ``ConsoleSessionController`` -- every Ctrl+K switcher
        open and ``/research <question>`` dispatch raised ``AttributeError``
        until this seam existed.
        """
        return self._session._current_console_conversation_id()

    async def _render_character_avatar_into_section(
        self,
        *,
        spec: dict | None,
        name: str | None,
        manual_label: str | None,
        is_current: Callable[[], bool],
    ) -> None:
        """Re-mount the avatar widget + name into the (already-composed) section.

        Async because Textual `Widget.mount()` returns an `AwaitMount` that
        must be awaited so the widget is present before the caller returns.
        `test_refresh_populates_avatar_cache_and_mounts` asserts the mounted
        DOM state (not just the cached spec dict) right after the refresh
        awaits this.
        """
        if not is_current():
            return
        try:
            left_rail = self.query_one("#console-left-rail", ConsoleLeftRail)
        except (NoMatches, QueryError):
            return  # section not composed (config off / not mounted)
        try:
            if not is_current():
                return
            left_rail.invalidate_character_avatar_geometry()
            fitted_box = left_rail.character_avatar_box
            replaced = await left_rail.replace_character_avatar_widget(
                lambda: self._build_character_avatar_widget(spec, box=fitted_box),
                is_current=is_current,
            )
            if not replaced or not is_current():
                return
            try:
                name_widget = self.query_one("#console-character-name", Static)
                if not is_current():
                    return
                name_widget.update(
                    Text(
                        sanitize_character_display_label(
                            name,
                            max_characters=180,
                        )
                        or "No character in this chat"
                    )
                )
            except QueryError:
                pass
            try:
                reaction_widget = self.query_one(
                    "#console-character-reaction-state", Static
                )
                if not is_current():
                    return
                reaction_widget.update(
                    f"Reaction: {manual_label} (manual)"
                    if manual_label
                    else "Reaction: Automatic"
                )
            except QueryError:
                pass
            self._request_console_context_allocation_reconcile()
        except Exception:
            # Must never raise: called from `_refresh_active_character_avatar_
            # if_scope_changed` at two sites outside that method's own
            # try/except, which is itself invoked unconditionally on every
            # 0.2s Console sync tick (`_sync_native_console_chat_ui`) -- some
            # worker dispatch sites run with `exit_on_error=True`, so an
            # escaping mount failure (e.g. a session-switch/resume tick
            # racing a transient layout state) could crash the app.
            logger.opt(exception=True).debug("avatar: render into section failed")

    @on(AvatarViewRequested)
    def _handle_avatar_view_requested(self, message: AvatarViewRequested) -> None:
        """Open the full-size portrait viewer for the rail avatar (task-1534)."""
        message.stop()
        spec = self._active_character_avatar or {}
        pil = spec.get("pil")
        if pil is None:
            return
        self.app.push_screen(
            ConsoleImageViewerModal(
                pil,
                title=sanitize_character_display_label(
                    self._active_character_avatar_name,
                    max_characters=180,
                )
                or "Character portrait",
            )
        )

    def _character_avatar_available_cols(self) -> int:
        """Usable width, in columns, available to the rail avatar.

        Measures the rail SECTION BODY, never the avatar holder: the
        holder is ``width: auto`` (task-1661, so it hugs the portrait
        instead of claiming the whole section), which makes its width a
        function of the child we are about to size -- measuring it fed the
        old child's width back in and pinned the box at the minimum
        forever. ``content_size`` already excludes padding and border.

        Returns:
            The section body's content width in columns, or 0 before
            layout settles so `character_avatar_box` uses its fallback.
        """
        try:
            body = self.query_one("#console-rail-section-body-character")
        except Exception:
            return 0
        try:
            return int(body.content_size.width)
        except Exception:
            return 0

    def _build_character_avatar_widget(
        self,
        spec: dict | None,
        *,
        box: tuple[int, int] | None = None,
        prerendered: "CharacterAvatarPrerender | None" = None,
    ) -> Widget:
        """Build a fresh avatar widget from the cached spec (data, not a widget).

        `spec` is `{character_id, name, mode, pil, pixels}` (T3 fills it via
        `_refresh_active_character_avatar_if_scope_changed`). With no spec,
        or a spec whose image decode failed/is pending, render a compact
        text placeholder. The cache holds this spec (data), not a live
        widget -- every (re)mount builds a fresh widget from it.

        This method must NEVER raise: it is reached from
        `_render_character_avatar_into_section`, which runs outside
        `_refresh_active_character_avatar_if_scope_changed`'s try/except (and
        that refresh itself must never raise into the 0.2s Console sync
        poll). Any image-build failure -- graphics mount OR the rich_pixels
        fallback -- degrades to the same text placeholder used for the
        no-image case.

        Args:
            spec: The cached avatar spec, or None.
            box: The rail's target cell box, if it has one.
            prerendered: A renderable already built off the event loop
                (TASK-22221). Used ONLY when its recorded image, resolved
                box, and colour mode still match what this build resolves --
                otherwise the mosaic is rebuilt inline, which is exactly what
                a viewport that moved mid-render must fall back to.
        """
        if not spec or (spec.get("pil") is None and spec.get("pixels") is None):
            if not (spec and spec.get("character_id") is not None):
                # TASK-23194: with no character set, `#console-character-name`
                # below already renders "No character in this chat". This
                # placeholder used to render the SAME sentence, so the rail
                # spent two of its scarcest rows saying one thing twice
                # (2026-08-29 UX audit). Stay mounted so the id keeps
                # resolving, but paint nothing.
                blank = Static("", id="console-character-avatar-empty")
                blank.styles.width = 0
                blank.styles.height = 0
                blank.styles.display = "none"
                return blank
            # width auto, not the Static default 100%: the holder is
            # width/height auto (task-1661), and a percentage-width child of
            # an auto container resolves to 0x0 under Textual 8.x -- the
            # placeholder would mount but paint nothing (task-3793).
            placeholder = Static("no avatar", id="console-character-avatar-empty")
            placeholder.styles.width = "auto"
            return placeholder
        if box == (0, 0):
            hidden = Static("", id="console-character-avatar-image")
            hidden.styles.width = 0
            hidden.styles.height = 0
            hidden.styles.display = "none"
            return hidden
        resolved_box = box or character_avatar_box(
            self._character_avatar_available_cols()
        )
        image = spec.get("pil")
        if image is not None:
            resolved_box = fit_character_avatar_cell_box(image, *resolved_box)
        if spec.get("mode") == "graphics" and spec.get("pil") is not None:
            try:
                from textual_image.widget import Image as _GraphicsImage

                widget = _GraphicsImage(
                    spec["pil"], id="console-character-avatar-image"
                )
                # Explicit fitted cell size, not just max-width/max-height --
                # see `console_transcript._image_row_widget`'s identical
                # guard: textual_image's "auto" sizing resolves its render
                # region from the parent's settled layout, and mounting a
                # tick before that settles can ask the renderer to scale
                # into a transient 0-width/height region, which PIL's
                # resize() raises on.
                box_cols, box_lines = resolved_box
                w, h = fit_image_cell_size(
                    spec["pil"].width,
                    spec["pil"].height,
                    box_cols,
                    box_lines,
                )
                widget.styles.width = w
                widget.styles.height = h
                return widget
            except Exception:
                logger.opt(exception=True).debug("avatar: graphics mount failed")
        try:
            box_cols, box_lines = resolved_box
            from ...Utils.mosaic_render import explicit_cell_size

            pixels = spec.get("pixels")
            if pixels is None and spec.get("pil") is not None:
                monochrome = bool(getattr(self.app, "no_color", False))
                if prerendered is not None and prerendered.matches(
                    spec["pil"], (box_cols, box_lines), monochrome
                ):
                    pixels = prerendered.renderable
                else:
                    pixels = _character_avatar_fallback_renderable(
                        spec["pil"],
                        box_cols=box_cols,
                        box_lines=box_lines,
                        monochrome=monochrome,
                    )
            widget = Static(
                pixels if pixels is not None else "",
                id="console-character-avatar-image",
            )
            # Explicit cell size derived from the baked renderable's own
            # grid, not just max-width/max-height: the holder is
            # width/height auto (task-1661), and this Static's default
            # width: 100% resolves to 0x0 inside it under Textual 8.x --
            # the avatar mounted but painted nothing at all (task-3793).
            # Same pattern as ConsoleImageViewerModal._build_full_size_widget;
            # explicit_cell_size returns None for a rich_pixels renderable,
            # which is sized for the box anyway.
            grid_size = explicit_cell_size(pixels)
            if grid_size is not None:
                widget.styles.width, widget.styles.height = grid_size
            else:
                widget.styles.width = box_cols
                widget.styles.height = box_lines
            widget.styles.max_width = box_cols
            widget.styles.max_height = box_lines
            return widget
        except Exception:
            logger.opt(exception=True).debug("avatar: pixels build failed")
            placeholder = Static("no avatar", id="console-character-avatar-empty")
            placeholder.styles.width = "auto"
            return placeholder

    def _console_messages_from_conversation_tree(
        self, tree: dict[str, Any]
    ) -> list[ConsoleChatMessage]:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        for the pre-existing test suite's direct-call convention (7 sites
        across 6 files); real production wiring (`ConsoleWorkspace
        Controller`'s `messages_from_conversation_tree_accessor`) now
        points at `self._message` directly, bypassing this delegation."""
        return self._message._console_messages_from_conversation_tree(tree)

    def _set_console_conversation_row_loading(
        self, conversation_id: str, loading: bool
    ) -> None:
        """Toggle a loading indicator on the matching workspace rail row.

        task-457(b): clicking a not-yet-open persisted conversation awaits
        ``_resume_console_workspace_conversation`` inline, so a slow or failing
        open otherwise reads as a dead click. Flagging the pressed row
        ``loading`` gives immediate feedback until the resume completes (the
        post-resume rail recompose rebuilds the row without the flag) or errors
        (the caller clears it in a ``finally``). Matches on the row's
        ``conversation_id`` attribute and no-ops when the row is no longer
        mounted, e.g. a recompose already replaced it.

        Args:
            conversation_id: The row's ``conversation_id`` attribute to match;
                a blank value is a no-op.
            loading: ``True`` to show the row's loading spinner, ``False`` to
                clear it.
        """
        target = str(conversation_id or "").strip()
        if not target:
            return
        for row in self.query(".console-workspace-conversation-row"):
            if str(getattr(row, "conversation_id", "") or "").strip() == target:
                row.loading = loading
                return

    def _mark_console_conversation_row_broken(self, conversation_id: str) -> None:
        """Record a conversation whose record is missing and refresh the rail.

        TASK-717: openability cannot be known at render time without probing
        the DB per row, so rows are marked lazily after the first informative
        failure and stay visibly broken for the rest of the session.
        """
        target = str(conversation_id or "").strip()
        if not target:
            return
        broken = getattr(self, "_console_broken_conversation_ids", None)
        if broken is None:
            broken = set()
            self._console_broken_conversation_ids = broken
        if target in broken:
            return
        broken.add(target)
        self._sync_console_workspace_context()

    def _on_console_scope_flushed(
        self, conversation_id: str, scope: Optional[RagScope]
    ) -> None:
        """Keep the retrieval-scope cache in sync with a first-persist flush.

        Wired into ``ConsoleChatStore`` as ``on_scope_flushed`` (task-9
        review finding 1). ``persist_session_if_needed`` calls this the
        moment it flushes a not-yet-persisted session's held
        ``SessionScopeHolder`` scope through to durable storage -- the real
        message-send path (``append_message(..., persist=True)`` ->
        ``_persist_new_message_or_defer``) reaches that flush with none of
        the Inspector row's other three read triggers (resume, modal-open,
        after-save) in between. Without this hook the row would keep
        rendering "everything" for the newly persisted conversation id
        until the user reopened Edit or saved a change, even though the
        scope was written correctly.

        This callback is itself synchronous (``ConsoleChatStore``'s
        ``on_scope_flushed`` contract), so it cannot ``await`` the
        off-loop workspace-intersection resolve (task-13) directly. It
        instead caches the immediate conversation-only approximation
        (byte-identical to pre-task-13 behavior, so the row/chip update in
        the same tick as the flush) and schedules a worker to resolve the
        full effective (conversation ∩ workspace) state and refresh again
        once that lands -- a workspace scope only ever NARROWS an already-
        correct conversation-only display, so this is never a regression,
        only an eventually-more-precise follow-up.
        """
        self._console_retrieval_scope_cache[conversation_id] = scope
        self._console_effective_scope_cache[conversation_id] = (
            ConsoleRetrievalScopeState.from_scope(scope)
        )
        if not self.is_mounted:
            return
        self._sync_console_retrieval_scope_row()
        self._sync_console_control_bar()
        session = self._session._active_native_console_session()
        if session is not None and session.persisted_conversation_id == conversation_id:
            self.run_worker(
                self._retrieval._refresh_console_effective_scope_and_sync(session),
                exclusive=True,
                group="console-effective-scope-refresh",
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

    def _console_rail_state_config(self) -> dict[str, Any]:
        """Return mutable Console rail-state config, initializing it if needed."""
        console_config = self._console_config()
        rail_state_config = console_config.get("rail_state")
        if not isinstance(rail_state_config, dict):
            rail_state_config = {}
            console_config["rail_state"] = rail_state_config
        return rail_state_config

    @work(thread=True)
    def _save_console_rail_preferences(
        self,
        key: str,
        serialized: dict[str, bool],
        *,
        notify_on_failure: bool = False,
    ) -> None:
        """Persist Console rail preferences without blocking the UI thread."""
        with _CONSOLE_RAIL_PREFERENCE_WRITE_LOCK:
            latest: Any = serialized
            app_config = getattr(self.app_instance, "app_config", None)
            if isinstance(app_config, Mapping):
                console_config = app_config.get("console")
                if isinstance(console_config, Mapping):
                    rail_state_config = console_config.get("rail_state")
                    if (
                        isinstance(rail_state_config, Mapping)
                        and key in rail_state_config
                    ):
                        latest = rail_state_config[key]
            latest_serialized = serialize_console_rail_stored_preferences(latest)
            try:
                saved = save_setting_to_cli_config(
                    "console.rail_state",
                    key,
                    latest_serialized,
                )
            except Exception as exc:
                logger.warning("Failed to persist Console rail preference: {}", exc)
                saved = False
        if not saved and notify_on_failure:
            self.app.call_from_thread(self._notify_console_rail_preference_save_failure)

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
            self._console_first_send_completed_cached = (
                coerce_console_first_send_completed(raw)
            )
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

    def _console_fleet_coachmark_seen(self) -> bool:
        """Return the persisted one-time fleet coach-mark flag (fleet-UX F2, task-1232).

        Mirrors ``_console_first_send_completed``'s manual nested-dict read
        rather than ``get_cli_setting`` -- ``get_cli_setting`` takes a flat
        ``(section, key)`` pair and does not resolve a dotted
        ``"console.onboarding"`` section (a prior program's documented trap).
        """
        if self._console_fleet_coachmark_seen_cached is None:
            app_config = getattr(self.app_instance, "app_config", None)
            raw = None
            if isinstance(app_config, dict):
                onboarding = app_config.get("console", {})
                if isinstance(onboarding, dict):
                    onboarding = onboarding.get("onboarding", {})
                raw = (
                    onboarding.get("fleet_coachmark_seen")
                    if isinstance(onboarding, dict)
                    else None
                )
            self._console_fleet_coachmark_seen_cached = coerce_bool_setting(raw, False)
        return self._console_fleet_coachmark_seen_cached

    def _maybe_show_fleet_coachmark(
        self,
        sessions: list[ConsoleChatSession],
        surface: ConsoleSessionSurface,
    ) -> None:
        """Show the one-time "each tab runs its own agent" coach-mark.

        Fleet-UX expert review F2 / Upgrade proposal 1 (task-1232): fires the
        first time the Console session count actually TRANSITIONS to
        exactly 2 (Ctrl+T, the tab strip's "New tab" button, a workspace
        auto-tab, a Personas "Start Chat" handoff -- every creation path
        already lands here via ``_sync_native_console_chat_ui`` ->
        ``_sync_console_native_session_tabs``). Seeded from whatever count
        this screen instance first observes, so a restore that starts the
        store already at 2+ sessions is never mistaken for a "creation".

        Args:
            sessions: The current Console session list (already fetched by
                the caller for the tab-strip sync).
            surface: The mounted Console session surface to render the
                banner on (already resolved by the caller).
        """
        current_count = len(sessions)
        previous_count = self._last_console_session_count
        self._last_console_session_count = current_count
        if previous_count is None:
            # First sync tick for this screen instance: seed only. Whatever
            # the store already holds is not a "creation" event.
            return
        if current_count != 2 or previous_count >= 2:
            return
        if self._console_fleet_coachmark_seen():
            return
        max_parallel_runs = self._ensure_console_chat_controller().max_parallel_runs
        surface.show_fleet_coachmark(
            f"Each tab runs its own agent — up to {max_parallel_runs} in "
            "parallel (change in Settings > Console Behavior)."
        )

    def _record_console_fleet_coachmark_dismissed(self) -> None:
        """Hide the fleet coach-mark and persist the one-time seen flag.

        The flag is written on DISMISS (not on show): an undismissed banner
        is allowed to reappear next time the session count transitions to 2
        (e.g. the user never noticed it, closed the tab, and reopened a
        second one) -- only an explicit acknowledgement makes it gone for
        good, including across restarts.
        """
        surface = self.console_session_surface
        if surface is not None:
            surface.hide_fleet_coachmark()
        if self._console_fleet_coachmark_seen_cached is True:
            return
        self._console_fleet_coachmark_seen_cached = True
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
            onboarding_cfg["fleet_coachmark_seen"] = True
        self._save_console_fleet_coachmark_flag()

    @work(thread=True)
    def _save_console_fleet_coachmark_flag(self) -> None:
        """Persist the fleet coach-mark seen flag without blocking the UI thread."""
        try:
            save_setting_to_cli_config(
                "console.onboarding",
                "fleet_coachmark_seen",
                True,
            )
        except Exception as exc:
            logger.warning("Failed to persist Console fleet coach-mark flag: {}", exc)

    def _ensure_console_rail_scope_seed(
        self,
        selected_key: ConsoleRailPreferenceKey,
        workspace_key: ConsoleRailPreferenceKey,
        *,
        persist: bool = True,
    ) -> Any:
        """Seed one absent layout scope without overwriting or deleting sources."""
        rail_state_config = self._console_rail_state_config()
        if selected_key.value in rail_state_config:
            return rail_state_config[selected_key.value]

        shared_key = build_console_rail_preference_key(layout_scope="global")
        candidates = (
            (workspace_key.value, workspace_key.fallback_value)
            if selected_key.scope_id == CONSOLE_RAIL_SHARED_LAYOUT_SCOPE
            else (workspace_key.fallback_value, shared_key.value)
        )
        source = next(
            (
                rail_state_config[key]
                for key in candidates
                if key and key in rail_state_config
            ),
            None,
        )
        serialized = serialize_console_rail_stored_preferences(source)
        rail_state_config[selected_key.value] = serialized
        if persist:
            self._save_console_rail_preferences(
                selected_key.value,
                serialized,
                notify_on_failure=False,
            )
        return serialized

    def _console_active_session_is_ephemeral(self) -> bool:
        """Return whether the active Console session is temporary.

        One-line delegation (wave-2 console decomposition, task 3): kept on
        `ChatScreen` under the original name because `Widgets/Console/
        console_transcript.py`'s `_console_ephemeral_active` reaches it by
        BARE NAME off `self.screen` (`getattr(screen,
        "_console_active_session_is_ephemeral", None)`), not through this
        controller. See `ConsoleSessionController._console_active_session_is_
        ephemeral` for the real implementation.
        """
        return self._session._console_active_session_is_ephemeral()

    def _sync_console_temporary_chip(self) -> None:
        """Push the active session's temporary flag to the status-strip chip.

        Deliberately scoped to ONLY the temporary chip -- unlike
        ``_sync_console_retrieval_scope_row`` this never builds or pushes
        scope state, so every place a Console session is created or
        switched can call this without also needing a
        ``ConsoleRetrievalScopeState`` on hand. Call at every such point:
        a stale value here is not cosmetic the way a stale scope chip is --
        it tells the user the opposite of the truth about whether their
        conversation is on disk (task-7 review finding).

        Called directly from ``_create_native_console_session_from_active_
        context`` (every "new session" entry point),
        ``_activate_console_session_for_workspace`` (workspace switch/
        create), the character-picker "new chat" flow, and the
        conversation-browser "already-open-tab" branch (task-7 review), and
        ``_promote_console_temporary_session`` (task-8) after a successful
        save -- ``_sync_native_console_chat_ui``'s regular tick never
        touches this chip, so a save that skipped this call would leave
        "Temporary" showing on an already-saved conversation.
        ``_sync_console_retrieval_scope_row`` (resume, tab activation,
        scope-picker save, first-persist flush) pushes the same thing
        inline instead of calling this helper -- it already has the
        ``ConsoleStatusChips`` instance in hand for the scope-chip push
        right above, so routing through here would only add a redundant
        second query.
        """
        try:
            status_chips = self.query_one("#console-status-chips", ConsoleStatusChips)
        except QueryError:
            return
        status_chips.sync_temporary_chip(self._console_active_session_is_ephemeral())

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
        available_columns: int | None = None,
    ) -> ConsoleRailState:
        """Build the effective Console rail state for the current composition.

        Args:
            available_columns: Optional width override for the responsive
                rules; the live-resize hook passes the event's width so the
                build never depends on widget-size update ordering.
        """
        workspace_context = self._workspace._current_console_workspace_context()
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
        workspace_key = build_console_rail_preference_key(
            workspace_id=workspace_context.active_workspace_id,
            conversation_id=(self._character._current_console_rail_conversation_id()),
            session_id=self._session._current_console_session_id(),
            layout_scope="workspace",
        )
        preference_key = build_console_rail_preference_key(
            workspace_id=workspace_context.active_workspace_id,
            layout_scope=normalize_console_rail_layout_scope(
                self._console_config().get("rail_layout_scope")
            ),
        )
        stored_preferences = self._ensure_console_rail_scope_seed(
            preference_key,
            workspace_key,
        )
        resolved_available_columns = (
            available_columns
            if available_columns is not None
            else self._console_rail_available_columns()
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
            available_columns=resolved_available_columns,
        )
        if self._should_open_standard_width_inspector(
            rail_state=rail_state,
            stored_preferences=stored_preferences,
            inspector_state=inspector_state,
            available_columns=resolved_available_columns,
        ):
            return replace(rail_state, right_open=True, right_forced_collapsed=False)
        return rail_state

    def _should_open_standard_width_inspector(
        self,
        *,
        rail_state: ConsoleRailState,
        stored_preferences: Any,
        inspector_state: ConsoleInspectorState,
        available_columns: int | None,
    ) -> bool:
        """Return whether the 120-column Console contract should show Inspector."""
        if rail_state.right_open:
            return False
        if console_auto_open_would_evict_context(rail_state, available_columns):
            # TASK-23197: opening here would trip
            # `resolve_console_rail_priority` and collapse the Context rail
            # the user can currently see -- the app taking away a panel they
            # were using to show one they never asked for. Between 118 and
            # 128 columns that swap happened on a single column of resize.
            return False
        if isinstance(stored_preferences, dict) and "right_open" in stored_preferences:
            return False
        if (
            available_columns is None
            or not CONSOLE_INSPECTOR_AUTO_OPEN_MIN_COLUMNS
            <= available_columns
            <= CONSOLE_INSPECTOR_AUTO_OPEN_MAX_COLUMNS
        ):
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
        try:
            left_rail = self.query_one("#console-left-rail", ConsoleLeftRail)
        except (NoMatches, QueryError):
            pass
        else:
            left_rail.sync_sections(rail_state)
        try:
            inspector = self.query_one(
                "#console-run-inspector-state", ConsoleRunInspector
            )
        except (NoMatches, QueryError):
            pass
        else:
            inspector.set_more_open(rail_state.inspector_more_open)
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

        # TASK-2154.1: mirrors the compose-time rules -- below 84 both handles
        # hide and the main minimum is waived. The default layout gives the
        # transcript the full grid; budget-eligible explicit rails may remain
        # visible and share it.
        targets = (
            ("#console-left-rail", rail_state.left_open),
            (
                "#console-context-rail-handle",
                not rail_state.left_open and not rail_state.single_pane,
            ),
            ("#console-right-rail", rail_state.right_open),
            (
                "#console-inspector-rail-handle",
                not rail_state.right_open and not rail_state.single_pane,
            ),
        )
        for selector, visible in targets:
            try:
                widget = self.query_one(selector)
            except QueryError:
                continue
            widget.styles.display = "block" if visible else "none"
            widget.display = visible
            self._sync_console_rail_descendant_visibility(widget, visible)

        try:
            main_column = self.query_one("#console-main-column")
        except QueryError:
            pass
        else:
            # Mirror compose-time geometry: these state flags waive the main
            # minimum during a live rail-visibility sync.
            main_column.styles.min_width = (
                0 if rail_state.single_pane or rail_state.compact_override else 56
            )

        self.refresh(layout=True)
        self._request_console_context_allocation_reconcile()

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

    def _current_console_rail_state(
        self,
        *,
        available_columns: int | None = None,
        inspector_state: ConsoleInspectorState | None = None,
    ) -> ConsoleRailState:
        """Build the current effective rail state from mounted Console context.

        Inside a run tick's `tick_workspace_build_scope` (TASK-22201) the
        workspace-context build below is served from the tick's shared,
        fingerprint-validated build; every other caller builds live.
        """
        resolved_available_columns = (
            available_columns
            if available_columns is not None
            else self._console_rail_available_columns()
        )
        pending_launch = self._pending_console_launch_context
        staged_context_state = self._build_console_staged_context_state(pending_launch)
        if inspector_state is None:
            inspector_state = self._build_console_inspector_state(pending_launch)
        workspace_context_state = (
            self._workspace._build_console_workspace_context_state()
        )
        rail_state = self._build_console_rail_state(
            staged_context_state=staged_context_state,
            inspector_state=inspector_state,
            workspace_context_state=workspace_context_state,
            available_columns=resolved_available_columns,
        )
        rail_state = resolve_console_rail_priority(
            rail_state, resolved_available_columns
        )
        rail_state = self._apply_pending_launch_inspector_auto_open(
            rail_state, pending_launch
        )
        rail_state = resolve_console_rail_priority(
            rail_state, resolved_available_columns
        )
        rail_state = self._agent._apply_fleet_agent_section_auto_open(rail_state)
        return resolve_console_rail_priority(rail_state, resolved_available_columns)

    def _set_console_rail_preference(
        self,
        *,
        left_open: bool | None = None,
        right_open: bool | None = None,
        section_updates: Mapping[str, bool] | None = None,
        notify_on_failure: bool = True,
    ) -> ConsoleRailState:
        """Persist requested Console rail preference changes and return new state."""
        workspace_context = self._workspace._current_console_workspace_context()
        workspace_key = build_console_rail_preference_key(
            workspace_id=workspace_context.active_workspace_id,
            conversation_id=(self._character._current_console_rail_conversation_id()),
            session_id=self._session._current_console_session_id(),
            layout_scope="workspace",
        )
        preference_key = build_console_rail_preference_key(
            workspace_id=workspace_context.active_workspace_id,
            layout_scope=normalize_console_rail_layout_scope(
                self._console_config().get("rail_layout_scope")
            ),
        )
        rail_state_config = self._console_rail_state_config()
        target_missing = preference_key.value not in rail_state_config
        self._ensure_console_rail_scope_seed(
            preference_key,
            workspace_key,
            persist=False,
        )
        prior_stored = rail_state_config.get(preference_key.value)
        current = coerce_console_rail_preferences(prior_stored)
        changes: dict[str, bool] = {}
        if left_open is not None:
            changes["left_open"] = bool(left_open)
        if right_open is not None:
            changes["right_open"] = bool(right_open)
        for section_id, section_open in (section_updates or {}).items():
            if section_id in CONSOLE_RAIL_PREFERENCE_DISCLOSURE_IDS:
                changes[f"{section_id}_open"] = bool(section_open)
        next_preferences = replace(current, **changes)
        # TASK-2154.2 (LY-11, ADR-043): an explicit rail toggle writes
        # through even when the coerced value is unchanged. Otherwise "open
        # the left rail" below 100 cols persisted nothing -- the default is
        # already left_open=True -- and the force-collapse rule (see
        # build_console_rail_state) kept the rail hidden: the exact silent
        # no-op this task removes. Ordinary writes serialize the preference
        # payload, so the left rail's explicitness cannot be read back from
        # key presence; a dedicated marker records the gesture. The marker
        # is also preserved across later writes that did not touch the left
        # rail (e.g. a section toggle). The right-open key is omitted when a
        # seed source omitted it so the 118-128-column auto-open band remains
        # distinguishable. The augmented dict goes to both the in-memory
        # config and the persisted file so the two never disagree.
        explicit_rail_toggle = left_open is not None or right_open is not None
        if next_preferences != current or explicit_rail_toggle or target_missing:
            serialized = serialize_console_rail_preferences(next_preferences)
            if left_open is not None or console_rail_left_open_explicit(prior_stored):
                serialized[CONSOLE_RAIL_LEFT_OPEN_EXPLICIT_KEY] = True
            if (
                right_open is None
                and isinstance(prior_stored, Mapping)
                and "right_open" not in prior_stored
            ):
                serialized.pop("right_open")
            rail_state_config[preference_key.value] = serialized
            self._save_console_rail_preferences(
                preference_key.value,
                serialized,
                notify_on_failure=notify_on_failure,
            )
        if right_open is not None:
            self._pending_console_launch_auto_open_inspector = False
        rail_state = self._current_console_rail_state()
        self._sync_console_rail_visibility_if_changed(rail_state)
        return rail_state

    def _toggle_console_rail_section(
        self,
        section_id: str,
        *,
        next_open: bool | None = None,
    ) -> None:
        """Flip one left-rail section open state, then sync body and header."""
        if section_id not in CONSOLE_RAIL_SECTION_IDS:
            return
        rail_state = self._current_console_rail_state()
        if next_open is None:
            next_open = not getattr(rail_state, f"{section_id}_open")
        if section_id == "agent":
            # TASK-915: track manual collapse/reopen of the Agent section
            # relative to the fleet's own busy signal -- never the
            # persisted preference below, which already records the user's
            # explicit choice either way.
            if next_open:
                self._agent_section_user_dismissed_while_busy = False
            elif self._agent._console_agent_fleet_summary_line():
                self._agent_section_user_dismissed_while_busy = True
        self._set_console_rail_preference(
            section_updates={section_id: next_open},
            notify_on_failure=False,
        )
        try:
            left_rail = self.query_one("#console-left-rail", ConsoleLeftRail)
        except (NoMatches, QueryError):
            pass
        else:
            left_rail.apply_section_open(section_id, next_open)
            self._request_console_context_allocation_reconcile()
        if section_id == "character" and next_open:
            # A collapsed body has `display: none`, so
            # `_character_avatar_available_cols()` measures 0 and
            # `character_avatar_box(0)` clamps to the 16-column MINIMUM. An
            # avatar first rendered while collapsed would then stay pinned
            # at that size forever, because
            # `_refresh_active_character_avatar_if_scope_changed` early-
            # returns while (character_id, state) is unchanged -- the
            # "~50-column rail showing a 16-column portrait" defect
            # task-1661 fixed for a different trigger. Clearing the scope
            # guard makes the next sync tick re-measure the now-visible body
            # and repaint at the rail's real width.
            self._character.invalidate_refresh_scope()

    def _sync_console_workspace_context(self) -> None:
        try:
            workspace_context = self.query_one(
                "#console-workspace-context",
                ConsoleWorkspaceContextTray,
            )
            state = self._workspace._build_console_workspace_context_state()
            # TASK-251: read the widget's OWN current state before it's
            # overwritten -- this is intentionally not a screen-level cache
            # (which would go stale across a full-screen recompose); the
            # freshly-(re)composed widget's ``.state`` is always correct at
            # construction time, so comparing against it here stays safe
            # across recomposes too.
            state_changed = state != workspace_context.state
            # TASK-344/349: the tray used to recompose unconditionally in its
            # own sync_state (a plain widget-level equality guard is still
            # forbidden -- it breaks grouped-browser click targeting). That
            # unconditional recompose ALSO self-healed a real DOM/state
            # desync: a full-screen recompose constructs a fresh tray whose
            # `.state` is set but whose rows can be superseded before they
            # settle, so `.state` says X while the DOM shows nothing -- the
            # next tick's recompose repaints it. So an equality guard is
            # unsafe in general.
            #
            # TASK-15454 replaced the unconditional recompose with an
            # evidence-based one (`ConsoleWorkspaceContextTray.
            # _can_skip_recompose`): the tray now skips only when the rows
            # actually mounted still match the ones its last completed
            # `compose` built, on an instance the rail has already pushed to.
            # That subsumes the self-heal above rather than removing it -- a
            # desynced or fresh tray fails the check and recomposes. The
            # screen-side skip here is kept as-is: it is a cheaper, earlier
            # exit on exactly the tick it was written for, and the two agree.
            # It IS safe on the ~5x/second run tick: the
            # workspace state is unchanged and recomposing the browser that
            # often tore it visibly down mid-run (the list vanished / showed
            # a half-composed frame and displaced clicks).
            #
            # Two guards keep the self-heal intact (PR #745 review):
            #  - gate on the SEMANTIC run-active status, not the transcript
            #    timer (which is still non-None on the final post-run poll,
            #    Qodo #2);
            #  - force at least ONE push per tray instance via a per-widget
            #    marker, so a fresh tray from ANY mid-run full-screen
            #    recompose still gets its healing recompose even under the
            #    skip; only unchanged ticks on an already-synced instance are
            #    skipped, where the DOM is known-consistent (Qodo #3). A
            #    concurrent in-run search changes the state, so it recomposes
            #    via `state_changed` regardless.
            controller = self._console_chat_controller
            run_active = (
                controller is not None
                and controller.run_state.status in CONSOLE_ACTIVE_RUN_STATUSES
            )
            already_synced = getattr(
                workspace_context, "_console_workspace_context_synced", False
            )
            if state_changed or not run_active or not already_synced:
                self.query_one(
                    "#console-left-rail", ConsoleLeftRail
                ).sync_workspace_context(state)
                self._request_console_context_allocation_reconcile()
            # PR #660 review: a full-screen recompose constructs a FRESH tray
            # already carrying the current state, so `state_changed` alone
            # would never re-kick the legacy-alias worker after a recompose —
            # leaving the transitional "New conversation" alias unmounted.
            # The kicked-marker lives on the tray instance (dies with it), so
            # a fresh tray always gets one kick regardless of state equality.
            alias_kick_needed = state_changed or not getattr(
                workspace_context, "_console_alias_kick_done", False
            )
            if alias_kick_needed:
                workspace_context._console_alias_kick_done = True
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

        state = self._workspace._build_console_workspace_context_state()

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
            self._request_console_context_allocation_reconcile()

    @on(ConsoleWorkspaceContextTray.Relabeled)
    def _on_console_workspace_context_relabeled(self) -> None:
        """Re-mount out-of-band tray controls after a width-driven relabel."""
        self.call_after_refresh(
            lambda: self.run_worker(
                self._sync_console_legacy_workspace_context_aliases,
                group="console-workspace-context-legacy-aliases",
                exclusive=True,
            )
        )

    @staticmethod
    def _launch_targets_chatbook_artifact(
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> bool:
        if pending_launch is None:
            return False
        source = str(pending_launch.source or "").strip().lower()
        target_id = str(pending_launch.payload.get("target_id") or "").strip()
        return source in {"artifacts", "chatbooks"} and ":chatbook:" in target_id

    def _console_pending_approval_count(self) -> int:
        explicit_count = getattr(
            self.app_instance, "console_pending_approval_count", None
        )
        if explicit_count is not None:
            return coerce_non_negative_int(explicit_count)

        pending_approval = getattr(self.app_instance, "pending_console_approval", None)
        if pending_approval and not (
            isinstance(pending_approval, dict)
            and pending_approval.get("phase") == "finishing"
        ):
            return 1

        task_approval = self._task_resume_state.pending_approval
        if not task_approval:
            return 0
        if (
            isinstance(task_approval, dict)
            and task_approval.get("phase") == "finishing"
        ):
            return 0
        return 1

    def _console_tool_count(self) -> int:
        return coerce_non_negative_int(
            getattr(self.app_instance, "console_tool_count", 0)
        )

    def _console_mcp_tool_count(self) -> Optional[int]:
        """MCP catalog size for the run inspector's "MCP" row (P5-T6).

        ``None`` (the default -- no seam wired) means "nothing to report":
        no ``unified_mcp_service``, the kill switch is on, or this app
        instance has not populated the hook yet -- ``ConsoleInspectorState.
        from_values`` omits the "MCP" row entirely in that case, mirroring
        ``_console_tool_count``'s own getattr-hook pattern above.
        """
        value = getattr(self.app_instance, "console_mcp_tool_count", None)
        return None if value is None else coerce_non_negative_int(value)

    def _console_mcp_not_connected_count(self) -> int:
        return coerce_non_negative_int(
            getattr(self.app_instance, "console_mcp_not_connected_count", 0)
        )

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

    def _console_can_save_chatbook_flag(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> bool:
        """Return whether a Chatbook artifact is available to save right now.

        TASK-251: factored out of ``_build_console_inspector_state`` so the
        composer's priority-action state can stay fresh from
        ``_sync_console_control_bar`` even while the right rail (and
        therefore the full inspector-state build) is hidden and skipped.
        """
        return bool(
            getattr(self.app_instance, "console_chatbook_artifact_available", False)
            or self._launch_targets_chatbook_artifact(pending_launch)
        )

    def _build_console_inspector_state(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleInspectorState:
        provider_display, model, settings = (
            self._active_console_provider_model_display()
        )
        _effective_settings, settings_readiness = (
            self._active_console_settings_readiness()
        )
        explicit_provider_ready = getattr(
            self.app_instance, "console_provider_ready", None
        )
        provider_readiness = get_provider_readiness(
            (settings.provider if settings is not None else None) or provider_display,
            self._provider_readiness_app_config(),
        )
        provider_runtime_ready = (
            settings_readiness.native_send_supported
            and explicit_provider_ready is not False
        )
        model_selected = _has_selected_text(model)
        provider_ready = provider_runtime_ready and model_selected
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
        can_save_chatbook = self._console_can_save_chatbook_flag(pending_launch)
        evidence_state = build_console_evidence_display_state(pending_launch)
        # TASK-24602: the controller already records a terminal FAILED run
        # state with user-visible copy (`_set_run_state(ConsoleRunState(
        # ConsoleRunStatus.FAILED, visible_copy))` on the agent-run failure
        # path). Nothing read it here, so the pinned authority line had no
        # way to know the last send had failed and answered "Ready".
        run_controller = self._console_chat_controller
        run_state = getattr(run_controller, "run_state", None)
        run_failed = (
            getattr(run_state, "status", None) is ConsoleRunStatus.FAILED
        )
        inspector_state = ConsoleInspectorState.from_values(
            live_work_title=pending_launch.title if pending_launch else None,
            run_active=self._console_run_active(),
            run_failed=run_failed,
            run_failure_reason=(
                str(getattr(run_state, "visible_copy", "") or "")
                if run_failed
                else ""
            ),
            provider_label=provider_display,
            model_label=model,
            provider_ready=provider_ready,
            provider_recovery=provider_recovery,
            rag_status=self._retrieval._console_rag_source_status(
                pending_launch,
                sent_source_count=self._console_evidence_sent_notice,
            ),
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
            mcp_tool_count=self._console_mcp_tool_count(),
            mcp_not_connected_count=self._console_mcp_not_connected_count(),
            can_save_chatbook=can_save_chatbook,
            scope_item_count=self._retrieval._console_retrieval_scope_run_recipe_count(),
            change_review_available=(
                getattr(self._console_agent_bridge, "change_tracking_enabled", False)
                if self._console_agent_bridge is not None
                else False
            ),
            ephemeral=self._console_active_session_is_ephemeral(),
            staged_source_count=console_staged_source_count(pending_launch),
        )
        setup_blocker_copy = self._console_provider_blocker_copy()
        if setup_blocker_copy:
            action_label, _action_target, _action_tooltip = (
                self._console_provider_recovery_action()
            )
            setup_rows = (
                ConsoleDisplayRow(
                    "Setup", "Provider configuration required", status="blocked"
                ),
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
            dictionary_rows=self._retrieval._console_dictionary_inspector_rows(),
            dictionary_actions=self._retrieval._console_dictionary_inspector_actions(),
            world_book_rows=self._retrieval._console_world_book_inspector_rows(),
            world_book_actions=self._retrieval._console_world_book_inspector_actions(),
        )
        return inspector_state

    def _dictionary_scope_service(self) -> Any:
        """The app-level chat-dictionary scope service, or None when absent."""
        return getattr(self.app_instance, "chat_dictionary_scope_service", None)

    def _console_chat_dictionary_applier(
        self, conversation_id: str | None, text: str
    ) -> str:
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

    def _console_world_info_applier(
        self, conversation_id: str | None, message_text: str, history: list
    ) -> str:
        """Bound applier handed to the native Console controller: inject the
        active CONVERSATION world-info into a send's text (never raises).

        Resolves the db lazily. Conversation-only: ``char_data`` is ``None``
        (native sessions carry no character card). Honors the same
        ``[character_chat] enable_world_info`` gate as the legacy send path
        (`Event_Handlers/Chat_Events/chat_events.py`).
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if (
            db is None
            or not conversation_id
            or not isinstance(message_text, str)
            or not get_cli_setting("character_chat", "enable_world_info", True)
        ):
            return message_text
        from ...Character_Chat.world_info_resolver import apply_world_info_to_message

        return apply_world_info_to_message(
            db, conversation_id, None, message_text, history or []
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
            conversation_id = self._character._current_console_rail_conversation_id()
            if not conversation_id:
                self.app_instance.notify(
                    "Start or load a conversation first.", severity="warning"
                )
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            try:
                rows = await asyncio.to_thread(
                    console_attachable_dictionaries, db, conversation_id
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load dictionaries for the Console attach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No more dictionaries to attach.", severity="information"
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(
                    DictionaryPicker(rows)
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the Console dictionary picker."
                )
                return
            if not picked:
                return
            await handle_console_dictionary_attach(
                self.app_instance, conversation_id, picked
            )
            # Always resync after an attempted attach (spec AC5: ConflictError
            # -> notify + refresh): on success the summary gains the dict; on a
            # ConflictError the DB changed under us and the cache must re-read
            # the current truth rather than stay stale until the next switch.
            await self._retrieval.refresh_active_dictionaries_summary()
        finally:
            self._console_dictionary_dialog_active = False

    async def _console_worldbook_attach_worker(self) -> None:
        """Pick and attach a world book to the active Console conversation.

        Mirrors :meth:`_console_dictionary_attach_worker`: every await is
        individually guarded so no exception escapes the worker boundary --
        an uncaught worker exception kills the whole app under
        ``run_worker(exit_on_error=True)``.
        """
        try:
            conversation_id = self._character._current_console_rail_conversation_id()
            if not conversation_id:
                self.app_instance.notify(
                    "Start or load a conversation first.", severity="warning"
                )
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                return
            from ...Character_Chat.world_book_manager import WorldBookManager
            from ...Widgets.Persona_Widgets.world_book_picker import WorldBookPicker

            def _attachable() -> list[dict]:
                mgr = WorldBookManager(db)
                attached_ids = {
                    b.get("id")
                    for b in mgr.get_world_books_for_conversation(
                        str(conversation_id), enabled_only=False
                    )
                }
                return [
                    {"world_book_id": int(b.get("id")), "name": str(b.get("name"))}
                    for b in (mgr.list_world_books(include_disabled=False) or [])
                    if b.get("id") not in attached_ids
                ]

            try:
                rows = await asyncio.to_thread(_attachable)
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load world books for the Console attach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No more world books to attach.", severity="information"
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(WorldBookPicker(rows))
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the Console world-book picker."
                )
                return
            if not picked:
                return
            try:
                await asyncio.to_thread(
                    WorldBookManager(db).associate_world_book_with_conversation,
                    str(conversation_id),
                    int(picked),
                )
            except Exception as exc:
                logger.opt(exception=True).warning("Could not attach the world book.")
                self.app_instance.notify(f"Attach failed: {exc}", severity="error")
                return
            await self._retrieval.refresh_active_world_books_summary()
        finally:
            self._console_worldbook_dialog_active = False

    async def _console_worldbook_detach_worker(self) -> None:
        """Pick and detach a world book from the active Console conversation.

        Analogous to :meth:`_console_worldbook_attach_worker`.
        """
        try:
            conversation_id = self._character._current_console_rail_conversation_id()
            if not conversation_id:
                self.app_instance.notify(
                    "Start or load a conversation first.", severity="warning"
                )
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            if db is None:
                return
            from ...Character_Chat.world_book_manager import WorldBookManager
            from ...Widgets.Persona_Widgets.world_book_picker import WorldBookPicker

            def _attached() -> list[dict]:
                mgr = WorldBookManager(db)
                return [
                    {"world_book_id": int(b.get("id")), "name": str(b.get("name"))}
                    for b in mgr.get_world_books_for_conversation(
                        str(conversation_id), enabled_only=False
                    )
                ]

            try:
                rows = await asyncio.to_thread(_attached)
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load world books for the Console detach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No world books attached to this conversation.",
                    severity="information",
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(
                    WorldBookPicker(
                        rows, title="Detach world book", confirm_label="Detach"
                    )
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the Console world-book picker."
                )
                return
            if not picked:
                return
            try:
                await asyncio.to_thread(
                    WorldBookManager(db).disassociate_world_book_from_conversation,
                    str(conversation_id),
                    int(picked),
                )
            except Exception as exc:
                logger.opt(exception=True).warning("Could not detach the world book.")
                self.app_instance.notify(f"Detach failed: {exc}", severity="error")
                return
            await self._retrieval.refresh_active_world_books_summary()
        finally:
            self._console_worldbook_dialog_active = False

    async def _console_dictionary_detach_worker(self) -> None:
        """Pick and detach a chat dictionary from the active Console conversation.

        Analogous to :meth:`_console_dictionary_attach_worker`, over
        ``console_attached_dictionaries``/``handle_console_dictionary_detach``.
        """
        try:
            conversation_id = self._character._current_console_rail_conversation_id()
            if not conversation_id:
                self.app_instance.notify(
                    "Start or load a conversation first.", severity="warning"
                )
                return
            db = getattr(self.app_instance, "chachanotes_db", None)
            try:
                rows = await asyncio.to_thread(
                    console_attached_dictionaries, db, conversation_id
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not load dictionaries for the Console detach picker."
                )
                return
            if not rows:
                self.app_instance.notify(
                    "No dictionaries attached to this conversation.",
                    severity="information",
                )
                return
            try:
                picked = await self.app_instance.push_screen_wait(
                    DictionaryPicker(
                        rows, title="Detach dictionary", confirm_label="Detach"
                    )
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Could not show the Console dictionary picker."
                )
                return
            if not picked:
                return
            await handle_console_dictionary_detach(
                self.app_instance, conversation_id, picked
            )
            # Always resync after an attempted detach (spec AC5: ConflictError
            # -> notify + refresh); see _console_dictionary_attach_worker.
            await self._retrieval.refresh_active_dictionaries_summary()
        finally:
            self._console_dictionary_dialog_active = False

    def _selected_console_conversation_inspector_rows(
        self,
    ) -> tuple[ConsoleDisplayRow, ...]:
        """Return inspector rows for the active Console conversation/session."""
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return (
                ConsoleDisplayRow("Selected conversation", "No active conversation"),
                ConsoleDisplayRow("Conversation source", "none"),
            )

        active_session = next(
            (
                session
                for session in store.sessions()
                if session.id == store.active_session_id
            ),
            None,
        )
        if active_session is None:
            return (
                ConsoleDisplayRow("Selected conversation", "No active conversation"),
                ConsoleDisplayRow("Conversation source", "none"),
            )

        workspace_state = self._workspace._build_console_workspace_context_state()
        workspace_value = workspace_state.workspace_label
        if isinstance(workspace_value, str) and workspace_value.startswith(
            "Workspace: "
        ):
            workspace_value = workspace_value.removeprefix("Workspace: ")
        workspace_label = (
            sanitize_character_display_label(
                workspace_value,
                max_characters=500,
            )
            or sanitize_character_display_label(
                active_session.workspace_id,
                max_characters=500,
            )
            or "Default"
        )
        persisted_id = str(active_session.persisted_conversation_id or "").strip()
        source = "saved conversation" if persisted_id else "native Console session"
        resume_state = (
            f"restored from {persisted_id}"
            if persisted_id
            else "local session, not persisted yet"
        )
        prefill_rows: list[ConsoleDisplayRow] = []
        one_shot = active_session.one_shot_prefill
        if one_shot:
            prefill_rows.append(
                ConsoleDisplayRow(
                    "Prefill (next send only)", describe_prefill_preview(one_shot)
                )
            )
        session_settings = active_session.settings
        pinned = (
            session_settings.pinned_prefill if session_settings is not None else None
        )
        if pinned:
            prefill_rows.append(
                ConsoleDisplayRow("Prefill (pinned)", describe_prefill_preview(pinned))
            )
        return (
            ConsoleDisplayRow(
                "Selected conversation",
                sanitize_character_display_label(
                    active_session.title,
                    max_characters=500,
                ),
            ),
            ConsoleDisplayRow("Conversation source", source),
            ConsoleDisplayRow("Workspace", workspace_label),
            ConsoleDisplayRow("Resume state", resume_state),
            *prefill_rows,
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
        store = self._ensure_console_chat_store()
        thinking_owner_id = transcript.thinking_owner_message_id(message_id)
        try:
            owner_session_id = store.session_id_for_message(
                thinking_owner_id or message_id
            )
        except KeyError:
            return ()
        if owner_session_id != store.active_session_id:
            return ()
        try:
            message = store.get_message(message_id)
        except KeyError:
            # Display-only TOOL/Thinking activity markers deliberately never
            # enter the store tree. Resolve only from this transcript's
            # current session projection; the store-owned session check above
            # prevents the pre-transcript phase of a switch reviving stale UI.
            message = transcript.display_message(message_id)
            if message is None:
                return ()

        rows = [
            ConsoleDisplayRow(
                "Selected message",
                f"{self._message._console_message_role_label(message)} message",
            ),
            ConsoleDisplayRow(
                "Message actions",
                "Copy, Edit, Fork, Regenerate/Retry, Continue, More…",
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
        # TASK-251 (audit P1 B1): while a message is streaming, its content
        # (and therefore this excerpt) changes every tick -- rendering the
        # live text here forced a full inspector-panel recompose 5x/second
        # for the whole duration of the stream. The transcript already shows
        # the live text, so the inspector shows a stable placeholder instead
        # and reveals the real excerpt once the message settles. Deliberate
        # UX change: flagged for the user gate per the task-251 report.
        excerpt = (
            "Streaming…"
            if message.status == "streaming"
            else (
                transcript.thinking_detail_text(message_id)
                if thinking_owner_id is not None
                else self._message._console_message_excerpt(message, max_length=90)
            )
        )
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
        self.app_instance.notify(
            "Chat settings are still loading.",
            severity="warning",
        )

    def _build_console_live_work_status_card(
        self, launch: ConsoleLiveWorkLaunch
    ) -> Container:
        """Build the mounted live-work status card for Console launch context.

        Shared by compose-time rendering and the TASK-259 targeted card swap
        in ``_apply_console_live_work_card_swap`` (which mounts the returned
        container without recomposing the screen).

        Args:
            launch: Live-work launch metadata to display.

        Returns:
            The card container (id ``console-pending-launch-card``) with its
            badge, optional primary action, and payload rows as children.
        """
        card_state = ConsoleLiveWorkStatusCardState.from_launch(launch)
        children: list[Any] = []
        if card_state.primary_action is not None:
            children.append(
                Button(
                    card_state.primary_action.label,
                    id=card_state.primary_action.widget_id,
                    classes=card_state.primary_action.classes,
                    variant="primary",
                )
            )
        children.extend(
            Static(row.text, id=row.widget_id, classes=row.classes)
            for row in card_state.rows
        )
        container = Container(
            *children,
            id=card_state.container_id,
            classes=card_state.container_classes,
        )
        container.styles.height = "auto"
        container.styles.min_height = 0
        return container

    @staticmethod
    def _hidden_static(text: str, *, id: str, classes: str = "") -> Static:
        widget = Static(
            text,
            id=id,
            classes=f"{classes} console-hidden-control".strip(),
            markup=False,
        )
        widget.styles.display = "none"
        widget.styles.height = 0
        widget.styles.min_height = 0
        widget.styles.max_height = 0
        return widget

    @staticmethod
    def _collapse_console_hidden_control_bar(
        widget: ConsoleControlBar,
    ) -> ConsoleControlBar:
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
            if not value:
                return "0"
            first_token = value.split(maxsplit=1)[0]
            # Fleet-UX expert review F7 (task-1234): `tools_label` can read
            # "Tools: —" (ConsoleControlState.from_values' neutral
            # placeholder at a zero count) instead of always
            # "Tools: N ready" -- naively taking the first word rendered
            # this compact summary as the nonsensical "Tools not". Any
            # non-numeric first token falls back to the same neutral dash
            # rather than a truncated word fragment.
            return first_token if first_token.isdigit() else "—"

        assistant = str(control_state.assistant_label or "Assistant: General")
        return (
            "Chat/RAG/Follow"
            f" | {assistant}"
            f" | Sources {readiness_count(control_state.sources_label)}"
            f" | Tools {readiness_count(control_state.tools_label)}"
            f" | Approvals {readiness_count(control_state.approvals_label)}"
        )

    def _console_run_active(self) -> bool:
        """Return whether a native Console generation is actively running.

        TASK-347: the header chip and Inspector status/live-work surfaces
        read this so they stop claiming "Ready"/"No active work" mid-run.
        """
        store = self._console_chat_store
        session_id = store.active_session_id if store is not None else None
        image_edit_active = (
            session_id is not None
            and self._image._h3_image_edit_registry().active(session_id) is not None
        )
        controller = self._console_chat_controller
        return image_edit_active or (
            controller is not None
            and controller.run_state.status in CONSOLE_ACTIVE_RUN_STATUSES
        )

    def _build_console_workbench_state(self, control_state: ConsoleControlState):
        blocker_copy = self._console_provider_blocker_copy()
        action_label, _action_target, _action_tooltip = (
            self._console_provider_recovery_action()
        )
        composer = self._console_composer_or_none()
        has_draft = bool(composer and composer.draft_text().strip())
        controller = self._console_chat_controller
        run_state = (
            getattr(controller, "run_state", None) if controller is not None else None
        )
        store = self._console_chat_store
        active_session_id = store.active_session_id if store is not None else None
        image_edit_active = (
            active_session_id is not None
            and self._image._h3_image_edit_registry().active(active_session_id)
            is not None
        )
        can_stop = image_edit_active or bool(
            getattr(run_state, "is_stop_allowed", False)
        )
        run_allows_send = (
            bool(getattr(run_state, "is_send_allowed", True)) and not image_edit_active
        )
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
            density=self._console_workbench_density(),
            run_active=self._console_run_active(),
            ephemeral=self._console_active_session_is_ephemeral(),
        )

    def _console_provider_blocker_copy(self) -> str:
        """Return concise Console recovery copy for provider/model setup gaps."""
        provider, _model, settings = self._active_console_provider_model_display()
        session_provider = str(getattr(settings, "provider", "") or "").strip()
        if not session_provider and settings is None:
            # Tolerate missing session settings: fall back to the display
            # provider rather than reporting no provider at all.
            session_provider = provider_config_key(provider)
        if not session_provider:
            return "Provider setup needed: choose a provider"

        _effective_settings, settings_readiness = (
            self._active_console_settings_readiness()
        )
        if settings_readiness.native_send_supported:
            return ""
        if settings_readiness.label == "Missing model":
            # Provider is send-ready; the model is the only remaining gap.
            return "Provider setup needed: choose a model"
        provider_readiness = get_provider_readiness(
            session_provider or provider,
            self._provider_readiness_app_config(),
        )
        if provider_readiness.reason == "Missing API key":
            display_name = provider_display_name(provider_config_key(provider))
            return f"Provider setup needed: {display_name} missing API key"
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
            return (
                "Configure endpoint",
                "Configure the provider endpoint before sending",
            )
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
        provider, _model, settings = self._active_console_provider_model_display()
        session_provider = str(getattr(settings, "provider", "") or "").strip()
        if not session_provider and settings is None:
            # Tolerate missing session settings: fall back to the display
            # provider rather than reporting no provider at all.
            session_provider = provider_config_key(provider)
        if not session_provider:
            return ""

        _effective_settings, settings_readiness = (
            self._active_console_settings_readiness()
        )
        if settings_readiness.native_send_supported:
            return ""
        if settings_readiness.label == "Missing model":
            # Choosing a model is a Console Settings action, not a Providers
            # & Models field fix.
            return ""

        provider_readiness = get_provider_readiness(
            session_provider or provider,
            self._provider_readiness_app_config(),
        )
        if provider_readiness.reason == "Missing API key":
            return "api_key"
        if settings_readiness.label in {"Endpoint not saved", "Invalid URL"}:
            return "endpoint"
        return ""

    def _console_provider_recovery_action(self) -> tuple[str, str, str]:
        """Return the label, target, and tooltip for Console provider recovery."""
        provider, _model, settings = self._active_console_provider_model_display()
        session_provider = str(getattr(settings, "provider", "") or "").strip()
        if not session_provider and settings is None:
            # Tolerate missing session settings: fall back to the display
            # provider rather than reporting no provider at all.
            session_provider = provider_config_key(provider)
        if not session_provider:
            return (
                "Choose provider",
                "console",
                "Choose a provider for this Console session",
            )

        _effective_settings, settings_readiness = (
            self._active_console_settings_readiness()
        )
        if settings_readiness.native_send_supported:
            return ("Open Settings", "hidden", "Open provider settings")
        if settings_readiness.label == "Missing model":
            return (
                "Choose model",
                "console",
                "Choose a model for this Console session",
            )

        provider_readiness = get_provider_readiness(
            session_provider or provider,
            self._provider_readiness_app_config(),
        )
        display_name = provider_display_name(provider_config_key(provider))
        if provider_readiness.reason == "Missing API key":
            return (
                CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
                "settings",
                f"Configure {display_name} API and API key in Settings",
            )
        if settings_readiness.label in {"Endpoint not saved", "Invalid URL"}:
            return (
                "Configure endpoint",
                "settings",
                f"Save the {display_name} endpoint in Settings",
            )
        if settings_readiness.label == "Unknown":
            return (
                "Choose provider",
                "console",
                "Choose a supported provider for this Console session",
            )
        return ("Review settings", "console", "Review this Console session's settings")

    def _build_console_setup_card_state(self) -> ConsoleSetupCardState:
        """Build the empty-transcript onboarding state from current readiness."""
        settings, _display_readiness = self._active_console_settings_readiness()
        # The card steps must reflect raw provider readiness (FR-05): the
        # screen-level readiness collapses provider-ready + model-missing to a
        # "Missing model" sentinel, which would wrongly re-activate step 1.
        readiness = build_console_settings_readiness(
            settings,
            app_config=self._provider_readiness_app_config(),
        )
        has_model = _has_selected_text(getattr(settings, "model", None))
        return build_console_setup_card_state(
            readiness=readiness,
            provider_label=str(getattr(settings, "provider", "") or "Provider"),
            has_model=has_model,
            first_send_completed=self._console_first_send_completed(),
            has_messages=self._message._active_console_transcript_has_messages(),
            guidance_dismissed=self._console_guidance_dismissed,
        )

    def _dismiss_console_guidance(self) -> None:
        """Hide first-run Console guidance after the user starts composing."""
        if self._console_guidance_dismissed:
            return
        self._console_guidance_dismissed = True
        self._sync_console_transcript_guidance()

    def _configure_console_copy_block(
        self,
        widget: Static,
        copy: str,
        *,
        visible: bool,
    ) -> None:
        """Update a compact Console status copy block without remounting it.

        task-280: skips the ``.update()``/style writes when both the copy
        and the show/hide state already match what was last applied to this
        exact widget instance. The applied tuple is stored ON the widget
        (PR #660 review): it dies with the widget, so a recomposed fresh
        instance always gets its first apply, nothing accumulates across
        recomposes, and a recycled ``id()`` can never alias a new widget to
        a dead one's cache entry.

        Args:
            widget: The Static copy block being configured.
            copy: The status copy to display (may be empty).
            visible: Whether the block should be shown at all.

        Returns:
            None.
        """
        cache_value = (copy, visible)
        if getattr(widget, "_console_copy_block_applied", None) == cache_value:
            return
        should_show = visible and bool(copy.strip())
        widget.update(copy if should_show else "")
        if should_show:
            row_count = copy.count("\n") + 1
            widget.styles.display = "block"
            widget.styles.height = row_count
            widget.styles.min_height = row_count
            widget.styles.max_height = row_count
        else:
            widget.styles.display = "none"
            widget.styles.height = 0
            widget.styles.min_height = 0
            widget.styles.max_height = 0
        widget._console_copy_block_applied = cache_value

    def _sync_console_transcript_guidance(self) -> None:
        """Refresh Console onboarding and provider recovery copy in place."""
        blocker_copy = self._console_provider_blocker_copy()
        action_label, _action_target, action_tooltip = (
            self._console_provider_recovery_action()
        )
        if blocker_copy:
            empty_action_label, empty_action_tooltip = (
                self._console_empty_recovery_action_copy(
                    blocker_copy,
                    provider_action_label=action_label,
                    provider_action_tooltip=action_tooltip,
                )
            )
        else:
            # TASK-2154.8 (FR-03): no blocker -> the empty transcript offers no
            # recovery action (it would dead-end as a misleading "Choose model"
            # button when nothing is broken). The setup modal keeps its own
            # default-label fallback, so card mode is unaffected.
            empty_action_label, empty_action_tooltip = "", ""

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
        # TASK-2154.10 (AC-04): vestibular-accessible static backdrop when the
        # user opts into reduced motion; refreshed with every guidance sync.
        modal.reduced_motion = bool(
            get_cli_setting("appearance", "reduce_motion", False)
        )
        modal.sync_card_state(
            card_state,
            action_label=action_label,
            action_tooltip=action_tooltip,
            staged_evidence_notice=console_setup_staged_receipt(
                self._pending_console_launch_context
            ),
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
            if not self._resume_navigation_startup_in_progress:
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

        task-16476: a provider that already has a DIFFERENT user-configured
        endpoint keeps it -- the endpoint write fills only when absent, and
        the detected endpoint is applied to the session instead, so "Use
        detected ..." stays effective without clobbering persisted config
        (discovery is loopback-only and can never see the LAN server the
        configured endpoint may point at).
        """
        server = self._console_detected_local_server
        if server is None:
            return
        model_id = server.model_ids[0] if server.model_ids else None
        app_config = self._provider_readiness_app_config()
        provider_key = provider_config_key(server.provider_key)
        provider_settings = self._config_section(
            self._config_section(app_config, "api_settings"),
            provider_key,
        )
        configured_endpoint = first_configured_endpoint(provider_settings)
        provider_values: dict[str, object] = {}
        # Qodo review (PR #1720): compare connection identities, not raw
        # strings -- a configured endpoint differing only by a trailing
        # slash (or a llama.cpp endpoint-path suffix) is the SAME server and
        # must not warn or skip the canonicalizing write. Same vocabulary
        # ``_endpoint_differs_for_provider`` uses.
        if configured_endpoint and self._adoption_endpoints_differ(
            provider_key, configured_endpoint, server.base_url
        ):
            self.app_instance.notify(
                "Keeping the saved endpoint "
                f"{safe_endpoint_display(configured_endpoint) or configured_endpoint} "
                f"for {provider_key}; using the detected "
                "server for this session only.",
                severity="warning",
            )
        else:
            provider_values["api_url"] = server.base_url
        chat_defaults: dict[str, object] = {"provider": server.provider_key}
        if model_id:
            provider_values["model"] = model_id
            chat_defaults["model"] = model_id
        if provider_values:
            try:
                saved = save_settings_to_cli_config(
                    {
                        f"api_settings.{server.provider_key}": provider_values,
                        "chat_defaults": chat_defaults,
                    }
                )
            except Exception:
                saved = False
        else:
            try:
                saved = save_settings_to_cli_config({"chat_defaults": chat_defaults})
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
        settings = replace(settings, base_url=server.base_url)
        self._session._replace_active_console_session_settings(
            replace(settings, source="user")
        )
        self._sync_console_transcript_guidance()
        self.run_worker(
            self._sync_native_console_chat_ui(), exclusive=True, group="console-sync"
        )

    @staticmethod
    def _adoption_endpoints_differ(
        provider_key: str,
        configured_endpoint: str,
        detected_endpoint: str,
    ) -> bool:
        """Return whether adoption endpoints differ by connection identity.

        Qodo review (PR #1720): raw string inequality treats a trailing
        slash (or a llama.cpp endpoint-path suffix) as a different server,
        warning and skipping the write for what is the same connection. Uses
        the same normalization vocabulary as
        ``_endpoint_differs_for_provider``.

        Args:
            provider_key: Normalized provider readiness key.
            configured_endpoint: Persisted endpoint for the provider.
            detected_endpoint: Discovered server base URL.

        Returns:
            ``True`` only when the two endpoints normalize to different
            connection identities.
        """
        if provider_key in {"llama_cpp", "local_llamacpp"}:
            configured = normalize_generic_endpoint_for_compare(
                normalize_llamacpp_base_url(configured_endpoint)
            )
            detected = normalize_generic_endpoint_for_compare(
                normalize_llamacpp_base_url(detected_endpoint)
            )
            return configured != detected
        return normalize_generic_endpoint_for_compare(
            configured_endpoint
        ) != normalize_generic_endpoint_for_compare(detected_endpoint)

    def _console_setup_modal_blocking(self) -> bool:
        """Return True when the first-run setup modal is covering the workbench."""
        try:
            modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        except QueryError:
            return False
        return bool(getattr(modal, "display", False)) and modal.is_blocking

    def _apply_console_setup_block(self, blocking: bool) -> None:
        """Disable composer focus/typing while the setup modal is up."""
        # FR-06 (TASK-2154.8): the footer hints must track the block state on
        # every transition, including before the composer query below can
        # early-return (e.g. mid-recompose).
        self._register_console_footer_shortcuts()
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        composer.can_focus = not blocking and not self._console_composer_collapsed
        if blocking and self._is_descendant_or_self(self.app.focused, composer):
            if self._resume_navigation_startup_in_progress:
                self.set_focus(None)
                return
            # Pull keyboard focus off the covered composer so typing can't tunnel.
            try:
                self.query_one(
                    "#console-setup-modal", ConsoleSetupModal
                ).focus_primary_action()
            except QueryError:
                composer.blur()

    @staticmethod
    def _frame_console_region(
        widget: Any,
        *,
        edges: tuple[str, ...],
        variant: str = "solid",
    ) -> Any:
        """Apply a visible Textual-native workbench frame.

        Delegates to `UI.Console_Modules.frame.frame_console_region` (wave-1
        console decomposition, task 2). Kept as a thin shim so the
        not-yet-extracted call sites in `compose_content` are untouched.
        Task 6 (wave 1's close-out) checked this; five call sites remain
        inside `compose_content` (the main-column block, wave 2's job), so
        the shim stays rather than being force-removed. It is safe to
        delete once every remaining call site imports
        `frame_console_region` from `UI.Console_Modules.frame` directly —
        expected during wave 2's main-column extraction, not before.

        Args:
            widget: The Console shell region widget to frame in place.
            edges: The exact application or interior edges owned by the region.
            variant: ``"solid"`` or ``"quiet"`` framing.

        Returns:
            The same `widget`, mutated in place with frame styling applied.
        """
        return frame_console_region(widget, edges=edges, variant=variant)

    def _build_console_live_work_source_readiness_card(self) -> Container:
        """Build the mounted source-readiness card shown without a launch.

        Shared by compose-time rendering and the TASK-259 targeted card swap
        (which mounts the returned container without recomposing the screen).

        Returns:
            The readiness container (id ``console-live-work-source-readiness``)
            with title, Library RAG query controls, and per-source rows.
        """
        acp_status = "not_configured"
        manager = getattr(self.app_instance, "acp_runtime_process_manager", None)
        snapshot = getattr(manager, "snapshot", None)
        if callable(snapshot):
            raw_snapshot = snapshot()
            if isinstance(raw_snapshot, dict):
                acp_status = str(raw_snapshot.get("status") or acp_status)
        # TASK-24601: MCP and RAG are the two rows on this card that can be
        # measured, so they are measured here rather than asserted in the
        # builder. Anything this call site cannot look up stays `None` and
        # renders as "Not checked" instead of borrowing a readiness word.
        # Imported locally, not at module scope: this screen is on the boot
        # path and ADR-097's import-weight budget is a ratchet that may only
        # ever fall.
        from ...Utils.optional_deps import DEPENDENCIES_AVAILABLE

        readiness = ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status(
            acp_status,
            mcp_tool_count=self._console_mcp_tool_count(),
            rag_available=bool(DEPENDENCIES_AVAILABLE.get("embeddings_rag")),
        )
        children: list[Any] = [
            Static(
                self._retrieval._console_library_rag_scope_label(),
                id="console-library-rag-scope",
                classes="destination-section",
            ),
            Input(
                value=self._console_library_rag_query,
                placeholder="Ask Library sources before sending",
                id="console-library-rag-query-input",
            ),
            Button(
                "Search Library",
                id="console-run-library-rag",
                classes="destination-action-button",
            ),
        ]
        children.extend(
            Static(row.text, id=row.widget_id, classes=row.classes)
            for row in readiness.rows
        )
        container = Container(
            *children,
            id=readiness.container_id,
            classes=readiness.container_classes,
        )
        container.styles.height = "auto"
        container.styles.min_height = 0
        return container

    @on(Button.Pressed, "#console-live-work-primary-action")
    def handle_console_live_work_primary_action(self, event: Button.Pressed) -> None:
        """Route supported live-work card actions through the app-owned shell."""
        event.stop()
        launch = self._consume_pending_console_launch()
        handler = getattr(
            self.app_instance, "open_console_live_work_primary_action", None
        )
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
        run_button.disabled = False
        run_button.tooltip = ""

    @on(Button.Pressed, "#console-run-library-rag")
    def handle_console_run_library_rag(self, event: Button.Pressed) -> None:
        """Open the one-shot Library search surface from the Inspector."""
        event.stop()
        self._open_console_library_search()

    def _run_console_library_rag_from_visible_action(self) -> None:
        """Run the one-shot Library search accepted by the modal."""
        query = _sanitize_console_library_rag_query(self._console_library_rag_query)
        if not query:
            composer = self._console_composer_or_none()
            draft_text = composer.draft_text() if composer is not None else ""
            draft_query = _sanitize_console_library_rag_query(draft_text)
            if draft_query:
                self._set_console_library_rag_query(draft_query)
                query = draft_query
        if not query:
            self._open_console_library_search()
            return
        request = LibraryRagSearchRequest(
            query=query,
            source_types=_console_library_rag_source_scope(self),
            mode="rag",
            top_k=_console_library_rag_profile_top_k(),
            include_citations=True,
        )
        self._retrieval._stage_console_library_rag_launch(
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

    def _sync_console_pending_launch_surfaces(self) -> bool:
        """Refresh every mounted reader of the pending launch context in place.

        Reader audit (TASK-259) -- outputs of builders that read
        ``_pending_console_launch_context`` and how each stays fresh here:

        * ``_build_console_control_state`` -> control bar, Workbench header/
          mode strip/command strip/recovery callout, hidden mode bar:
          ``_sync_console_control_bar`` + ``_sync_console_mode_bar``.
        * ``_build_console_inspector_state`` -> ``ConsoleRunInspector`` rows
          and composer Chatbook action: pushed inside
          ``_sync_console_control_bar``.
        * ``build_console_staged_evidence_strip_state`` -> the composer-level
          staged-evidence strip (``_sync_console_staged_evidence_strip``),
          the main-surface reader added by PR-4/task-1.
        * ``_build_console_staged_context_state`` -> staged-context tray
          (``_sync_console_staged_context_tray``), rail badges/summary and
          the pending-launch inspector auto-open (both applied through the
          rail-state build inside ``_sync_console_control_bar``), and the
          settings context estimate (``_sync_console_settings_summary``).
        * ``_current_console_workspace_context`` (staged sources include the
          launch) -> workspace context + details trays:
          ``_sync_console_workspace_context``.
        * The pending-launch status card / source-readiness card in the
          inspector rail: swapped via ``_apply_console_live_work_card_swap``.
        * Remaining readers (``action_show_workbench_help``,
          ``_sync_console_workbench_actions_from_draft``,
          ``_console_send_blocked_reason``, the live-work/Chatbook button
          handlers) build their state on demand at event time and cannot go
          stale.

        Returns:
            True when the Console shell was mounted and synced; False when
            the caller must fall back to a full recompose.
        """
        try:
            self.query_one("#console-inspector-rail-body", VerticalScroll)
        except QueryError:
            return False
        if not self._console_live_work_card_swap_scheduled:
            self._console_live_work_card_swap_scheduled = True
            self.call_later(self._apply_console_live_work_card_swap)
        self._sync_console_staged_evidence_strip()
        self._sync_console_staged_context_tray()
        self._request_console_control_bar_sync()
        self._sync_console_workspace_context()
        self._sync_console_settings_summary()
        self._sync_console_mode_bar()
        return True

    def _build_console_staged_evidence_strip_state(
        self,
        pending_launch: Optional[ConsoleLiveWorkLaunch],
    ) -> ConsoleStagedEvidenceStripState:
        """Build the composer-level staged-evidence strip state."""
        return build_console_staged_evidence_strip_state(
            pending_launch,
            sent_source_count=self._console_evidence_sent_notice,
        )

    def _sync_console_staged_evidence_strip(self) -> None:
        """Refresh the mounted staged-evidence strip from the launch context."""
        try:
            strip = self.query_one(
                "#console-staged-evidence-strip", ConsoleStagedEvidenceStrip
            )
        except QueryError:
            return
        strip.sync_state(
            self._build_console_staged_evidence_strip_state(
                self._pending_console_launch_context
            )
        )

    @on(Button.Pressed, "#console-unstage-evidence")
    def handle_console_unstage_evidence(self, event: Button.Pressed) -> None:
        """Drop the whole staged live-work context from the main surface.

        One button clears the entire launch (not per-reference): the bundle
        is staged, prompted, and captured as ONE unit, so a partial un-stage
        would advertise a granularity the send path does not have.
        """
        event.stop()
        # M4 (final review): resync BEFORE the early return so a strip that
        # is still showing stale staged rows -- because the context field
        # was already cleared from under it elsewhere (e.g. a send's
        # consume-on-send clear) without a matching surface sync -- heals
        # on click instead of dead-ending as a silent no-op.
        self._sync_console_staged_evidence_strip()
        if self._pending_console_launch_context is None:
            return
        self._pending_console_launch_context = None
        self._pending_console_launch_auto_open_inspector = False
        self._console_evidence_sent_notice = None
        if not self._sync_console_pending_launch_surfaces():
            self.refresh(recompose=True)
        self.notify("Staged evidence cleared")

    async def _apply_console_live_work_card_swap(self) -> None:
        """Swap the inspector-rail live-work card to match the launch context.

        Removes whichever of the pending-launch / source-readiness cards is
        mounted, then mounts the card for the CURRENT context (re-read after
        the awaits -- staging can happen again while a swap is in flight).
        The scheduled flag stays set for the WHOLE swap, so a mid-swap
        staging can never start a second, overlapping swap regardless of how
        the caller reached the scheduler (PR #691 review); the tail re-check
        below converges on the latest context instead.
        """
        swapped_context = None
        swap_completed = False
        try:
            try:
                local_section = self.query_one(
                    "#console-bounded-section-live-work", ConsoleBoundedSection
                )
                pending_header = self.query_one(
                    "#console-live-work-status-badge", Static
                )
                readiness_header = self.query_one(
                    "#console-live-work-source-readiness-title", Static
                )
            except QueryError:
                return
            for selector in (
                f"#{PENDING_LAUNCH_CARD_ID}",
                f"#{SOURCE_READINESS_CARD_ID}",
            ):
                try:
                    stale_card = local_section.query_one(selector)
                except QueryError:
                    continue
                await stale_card.remove()
            launch = self._pending_console_launch_context
            card = (
                self._build_console_live_work_status_card(launch)
                if launch is not None
                else self._build_console_live_work_source_readiness_card()
            )
            await local_section.viewport.mount(card)
            pending_header.display = launch is not None
            readiness_header.display = launch is None
            self.call_after_refresh(self._request_console_live_work_reconcile)
            swapped_context = launch
            swap_completed = True
        finally:
            self._console_live_work_card_swap_scheduled = False
        # Gate on completion: the rail-unmounted early return must NOT
        # re-schedule, or a lingering context would loop this forever.
        if (
            swap_completed
            and self._pending_console_launch_context is not swapped_context
        ):
            # Staging changed the context after this swap's re-read; run one
            # more swap so the mounted card converges on the latest context.
            self._console_live_work_card_swap_scheduled = True
            self.call_later(self._apply_console_live_work_card_swap)

    def _sync_console_staged_context_tray(self) -> None:
        """Refresh the mounted staged-context tray from the launch context."""
        try:
            tray = self.query_one(
                "#console-staged-context-tray", ConsoleStagedContextTray
            )
        except QueryError:
            return
        state = self._build_console_staged_context_state(
            self._pending_console_launch_context
        )
        # The child owns its bounded-body and rail invalidation.
        tray.sync_state(state)

    @work(exclusive=True, group="console-library-rag-search")
    async def _execute_console_library_rag_search(
        self, request: LibraryRagSearchRequest
    ) -> None:
        await self._retrieval._execute_console_library_rag_search(request)

    def _has_staged_console_evidence(self) -> bool:
        """Whether this send already has evidence the user staged themselves.

        Two places count as "staged", because
        ``_consume_pending_console_launch`` reads both on the very next line
        of the send: the resident launch context, and an UNCLAIMED
        ``CONSOLE_LIVE_WORK`` handoff (a Library "Use in Console" the user
        just clicked). Ignoring the second would spend a retrieval whose
        result the claim immediately supersedes -- the exact "no double
        spend" the auto-retrieve gate exists to prevent.

        Returns:
            True when manual staging is present and must win.
        """
        if self._pending_console_launch_context is not None:
            return True
        try:
            store = self.app_instance.pending_handoffs
            return bool(store.has_pending(HandoffChannel.CONSOLE_LIVE_WORK))
        except Exception:
            # An unreadable store is not evidence; retrieving is the safe
            # failure here (worst case a claim supersedes the result).
            return False

    def compose_content(self) -> ComposeResult:
        """Compose the chat content."""
        pending_launch = self._consume_pending_console_launch()
        control_state = self._build_console_control_state(pending_launch)
        staged_context_state = self._build_console_staged_context_state(pending_launch)
        inspector_state = self._build_console_inspector_state(pending_launch)
        workspace_context_state = (
            self._workspace._build_console_workspace_context_state()
        )
        # task-10: built once, shared verbatim by the header's "Scope" chip
        # and the Inspector's retrieval-scope row below -- one zero-DB
        # state, two renderers.
        retrieval_scope_state = self._retrieval._build_console_retrieval_scope_state()
        available_columns = self._console_rail_available_columns()
        rail_state = self._build_console_rail_state(
            staged_context_state=staged_context_state,
            inspector_state=inspector_state,
            workspace_context_state=workspace_context_state,
            available_columns=available_columns,
        )
        rail_state = resolve_console_rail_priority(rail_state, available_columns)
        rail_state = self._apply_pending_launch_inspector_auto_open(
            rail_state,
            pending_launch,
        )
        rail_state = resolve_console_rail_priority(rail_state, available_columns)
        rail_state = self._agent._apply_fleet_agent_section_auto_open(rail_state)
        rail_state = resolve_console_rail_priority(rail_state, available_columns)
        workbench_state = self._build_console_workbench_state(control_state)
        shell_classes = (
            f"workbench-frame console-workbench-frame density-{workbench_state.density}"
        )
        with Vertical(id="console-shell", classes=shell_classes):
            # The destination identity header is the visible Console header;
            # it stays live via _sync_console_workbench_state. The legacy
            # #console-title/#console-purpose/#console-status-row compat
            # statics below remain mounted but hidden for contract tests.
            yield DestinationHeader(
                workbench_state.header,
                before_status=ConsoleSpeechControls(id="console-speech-controls"),
                id="console-workbench-header",
                classes="workbench-header console-header-inline",
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
            yield ConsoleControlBar(
                control_state,
                self.app_instance,
                actions=workbench_state.actions,
                on_sidebar_toggle_requested=self._open_console_settings,
                id="console-control-bar",
                classes="console-control-bar",
            )
            workspace_grid = self._frame_console_region(
                Horizontal(
                    id="console-workspace-grid",
                    classes="ds-panel destination-workbench",
                ),
                edges=("top", "bottom"),
            )
            workspace_grid.styles.min_height = 0
            stack_rail_labels = self._stack_collapsed_rail_labels()
            with workspace_grid:
                left_handle = ConsoleRailHandle(
                    label=rail_state.left_label,
                    badge=rail_state.left_badge,
                    button_id="console-context-rail-open",
                    badge_id="console-context-rail-badge",
                    side="left",
                    vertical=stack_rail_labels,
                    id="console-context-rail-handle",
                )
                left_handle_width = (
                    ConsoleRailHandle.VERTICAL_WIDTH if stack_rail_labels else 13
                )
                left_handle.styles.width = left_handle_width
                left_handle.styles.min_width = left_handle_width
                left_handle.styles.max_width = left_handle_width
                if rail_state.left_open or rail_state.single_pane:
                    # TASK-2154.1: single-pane mode hides both handles -- the
                    # transcript is the only pane left to point at.
                    left_handle.styles.display = "none"
                yield self._frame_console_region(left_handle, edges=("right",))

                # The section-level values below are computed here, on the
                # screen, exactly as they were computed inline before this
                # extraction (wave-1 console decomposition, task 3) --
                # session-settings resolution, the agent bridge, and
                # character avatar rendering all stay screen-owned; only the
                # already-computed results are handed to `ConsoleLeftRail`.
                fleet_line = self._agent._console_agent_fleet_summary_line()
                settings_summary_state = self._build_console_settings_summary_state()
                settings_store = self._ensure_console_chat_store()
                settings_session_id = settings_store.active_session_id
                settings_session = (
                    getattr(settings_store, "_sessions", {}).get(settings_session_id)
                    if settings_session_id is not None
                    else None
                )
                system_line_text, system_line_dim = (
                    self._console_rail_system_line_state()
                )
                # PR2b Task 4: the third element (the old joined sub-agents
                # string) is no longer painted -- the fleet mini-section's
                # rows/summary are derived independently, straight from the
                # bridge, by `_console_agent_fleet_section_state`.
                agent_status_line, agent_steps_text, _agent_subagents_text = (
                    self._agent._console_agent_section_lines()
                )
                agent_fleet_section_state = (
                    self._agent._console_agent_fleet_section_state()
                )
                show_character_section = resolve_show_character_avatar(
                    getattr(getattr(self, "app_instance", None), "app_config", {}) or {}
                )
                # `character_avatar_widget_builder` hands `ConsoleLeftRail` a
                # box-aware callable, not a pre-built widget: the rail's own
                # `compose()` calls it, so a future recompose always mounts a
                # fresh, currently fitted avatar widget built from the CURRENT
                # `self._active_character_avatar` rather than re-yielding a
                # stale instance `_render_character_avatar_into_section` may
                # already have removed from the DOM (final review finding 1).
                # The callable closes over `self` and reads
                # `self._active_character_avatar` at CALL time, matching
                # `ConsoleDictationController`'s late-binding constructor rule
                # (see `dictation.py`'s module docstring) -- not a bound
                # method or a snapshotted spec, which would freeze today's
                # value instead.
                character_avatar_widget_builder = None
                character_avatar_fit_box = None
                character_avatar_prerender_job = None
                character_avatar_name = ""
                if show_character_section:

                    def character_avatar_widget_builder(box=None, prerendered=None):
                        return self._build_character_avatar_widget(
                            self._active_character_avatar,
                            box=box,
                            prerendered=prerendered,
                        )

                    def character_avatar_prerender_job(box):
                        """Snapshot the live spec + colour mode for an off-loop
                        render of ``box`` (TASK-22221); the eligibility policy
                        and the job itself are the pure module's."""
                        return build_character_avatar_prerender_job(
                            self._active_character_avatar,
                            box,
                            monochrome=bool(getattr(self.app, "no_color", False)),
                        )

                    def character_avatar_fit_box(
                        available_cols: int,
                        available_lines: int,
                    ) -> tuple[int, int] | None:
                        spec = self._active_character_avatar or {}
                        image = spec.get("pil")
                        if image is None:
                            return None
                        return fit_character_avatar_cell_box(
                            image,
                            available_cols,
                            available_lines,
                        )

                    character_avatar_name = (
                        sanitize_character_display_label(
                            self._active_character_avatar_name,
                            max_characters=180,
                        )
                        or "No character in this chat"
                    )

                left_rail = ConsoleLeftRail(
                    rail_state=rail_state,
                    workspace_context_state=workspace_context_state,
                    settings_summary_state=settings_summary_state,
                    system_line_text=system_line_text,
                    system_line_dim=system_line_dim,
                    fleet_line=fleet_line,
                    agent_status_line=agent_status_line,
                    agent_steps_text=agent_steps_text,
                    agent_fleet_section_state=agent_fleet_section_state,
                    agent_drilldown_active=bool(self._console_agent_drilldown_run_id),
                    agent_full_log_available=(
                        self._agent._console_agent_full_log_available()
                    ),
                    agent_steering_state=(self._agent._console_agent_steering_state()),
                    agent_cancel_all_visible=(
                        self._agent._console_agent_cancel_all_visible()
                    ),
                    show_character_section=show_character_section,
                    character_avatar_widget_builder=character_avatar_widget_builder,
                    character_avatar_name=character_avatar_name,
                    character_avatar_fit_box=character_avatar_fit_box,
                    character_avatar_prerender_job=character_avatar_prerender_job,
                    workspace_tree_expanded_ids=(
                        self._workspace.workspace_tree_expansion_preferences()
                    ),
                    workspace_tree_expansion_preferences_changed=(
                        self._workspace.set_workspace_tree_expansion_preferences
                    ),
                    manual_reaction_label=(
                        self._session._manual_reaction_label_for_current_actor()
                    ),
                    settings_session_id=settings_session_id,
                    settings_persistence_failures=(
                        settings_session.settings_persistence_failures
                        if settings_session is not None
                        else {}
                    ),
                    default_durability_state=(self._console_default_durability_state()),
                )
                left_rail.can_focus = True
                left_rail.styles.width = "3fr"
                # TASK-19639 (formerly TASK-18913) compact contract: at exactly 100 columns the
                # workspace grid has all 100 application columns. Default
                # horizontal-label geometry resolves as Context 30 + main
                # outer 59 + collapsed Inspector handle 11 = 100. The main
                # min-width waiver keeps
                # the row solvable; `rail_state.left_open` determines Context
                # visibility. Below 100, default Context force-collapses
                # without rewriting preference; eligible explicit opens
                # instead receive the same layout-only waiver.
                # TASK-2154.3 (LY-01/LY-07): 30 is the Context outer minimum
                # (13-cell label with gutter + 10-cell value + 7-cell chrome).
                # A 3-column handle applies only with stacked rail labels.
                left_rail.styles.min_width = 30
                if not rail_state.left_open:
                    left_rail.styles.display = "none"
                yield self._frame_console_region(left_rail, edges=("right",))

                # A zero-arg builder, not a pre-built widget, for the same
                # reason `character_avatar_widget_builder` above is one --
                # Sizing stays here because it describes this pane among its siblings
                # (3fr / 13fr / 4fr), exactly as both rails are wired.
                main_column = ConsoleTranscriptRegion(
                    session_surface_builder=(
                        lambda: self._ensure_console_session_surface()
                    ),
                    recovery_message_builder=(
                        lambda: (
                            self._ensure_console_chat_controller().provider_continuation_recovery_message()
                        )
                    ),
                    recovery_replay_available_builder=(
                        lambda: (
                            self._ensure_console_chat_controller().provider_continuation_replay_available()
                        )
                    ),
                    on_recovery_action=(
                        lambda action, message_id, version: (
                            self._ensure_console_chat_controller().recover_provider_continuation(
                                action, message_id, version
                            )
                        )
                    ),
                )
                main_column.styles.width = "13fr"
                # TASK-2154.1 (LY-09): below 84 the handles hide and the main
                # minimum is waived. The default layout is transcript-only;
                # budget-eligible explicit rails may still render from their
                # 70/74 floors through 83 via compact override. At 84, default
                # horizontal handles (13 + 11) leave 60 application columns
                # for the main, above its 56-column floor; stacked handles are
                # 3 columns each.
                # TASK-2154.2/TASK-19639 (formerly TASK-18913): ``compact_override`` is only
                # layout-minimum-waiver authority. It covers eligible explicit
                # opens below the thresholds, default Context at exactly 100,
                # and Inspector priority; it is not preference or user intent.
                # Open rails retain their 30/34-column minimums.
                main_column.styles.min_width = (
                    0 if rail_state.single_pane or rail_state.compact_override else 56
                )
                main_column.styles.min_height = 0
                yield main_column

                # The live-work card is the one piece of this rail's content
                # that reaches beyond rail-local state (self.app_instance,
                # self._console_library_rag_query via
                # _build_console_live_work_source_readiness_card) -- built
                # here, on the screen, exactly as it was built inline before
                # this extraction (wave-1 console decomposition, task 4).
                # `live_work_card_builder` hands `ConsoleInspectorRail` a
                # zero-arg callable, not a finished widget: the rail's own
                # `compose()` calls it, so a future recompose always mounts a
                # fresh card instead of re-yielding a stale instance
                # `_apply_console_live_work_card_swap` may already have
                # removed from the DOM (final review finding 1) -- the same
                # builder shape `character_avatar_widget_builder` uses for
                # `ConsoleLeftRail` above. The lambda closes over `self` and
                # reads `self._pending_console_launch_context` at CALL time
                # (matching `ConsoleDictationController`'s late-binding
                # constructor rule -- see `dictation.py`'s module docstring),
                # which is the same value `pending_launch` above was just
                # resolved from and nothing mutates in between.
                right_rail = ConsoleInspectorRail(
                    staged_context_state=staged_context_state,
                    retrieval_scope_state=retrieval_scope_state,
                    inspector_state=inspector_state,
                    project_instruction_state=project_instruction_ui.project_instruction_ui_state_for_screen(
                        self
                    ),
                    settings_summary_state=self._build_console_settings_summary_state(),
                    live_work_card_builder=(
                        lambda: (
                            self._build_console_live_work_status_card(
                                self._pending_console_launch_context
                            )
                            if self._pending_console_launch_context
                            else self._build_console_live_work_source_readiness_card()
                        )
                    ),
                    library_activity_view=self._library_activity.view,
                    library_activity_citation_count=(
                        self._library_activity.selected_citation_count()
                    ),
                    library_activity_flush_result=self._library_activity.flush_result,
                    library_activity_retry=self._library_activity.retry,
                    inspector_more_open=rail_state.inspector_more_open,
                )
                right_rail.can_focus = True
                right_rail.styles.width = "4fr"
                right_rail.styles.min_width = 34
                if not rail_state.right_open:
                    right_rail.styles.display = "none"
                yield self._frame_console_region(right_rail, edges=("left",))

                right_handle = ConsoleRailHandle(
                    label=rail_state.right_label,
                    badge=rail_state.right_badge,
                    button_id="console-inspector-rail-open",
                    badge_id="console-inspector-rail-badge",
                    side="right",
                    vertical=stack_rail_labels,
                    id="console-inspector-rail-handle",
                )
                right_handle_width = (
                    ConsoleRailHandle.VERTICAL_WIDTH if stack_rail_labels else 11
                )
                right_handle.styles.width = right_handle_width
                right_handle.styles.min_width = right_handle_width
                right_handle.styles.max_width = right_handle_width
                if rail_state.right_open or rail_state.single_pane:
                    right_handle.styles.display = "none"
                yield self._frame_console_region(right_handle, edges=("left",))
            # task-17652: the status row's side of the composer cluster is
            # user-configurable ([console] status_chips_position). "above"
            # (the default; owner ruling 2026-08-17) tops the cluster
            # directly under the workspace grid; "below" restores the
            # TASK-15704 bottom row. Built once here so both branches yield
            # the same widget; the command popup's clearance loop handles
            # both placements (see ConsoleCommandPopup.reposition).
            # task-5 (PR3 cost ticker): same F1 precedent as the ephemeral
            # flag -- compose the cost chip correctly on the very first
            # frame rather than waiting for a post-mount sync call.
            # Best-effort: `_build_console_cost_state` already never raises
            # on its own, but this call site still tolerates an unexpected
            # failure rather than ever taking down the whole compose.
            try:
                initial_cost_state = self._build_console_cost_state()
            except Exception:
                logger.opt(exception=True).warning("cost_chip_state_failed")
                initial_cost_state = None
            status_chips_position = resolve_status_chips_position(
                getattr(self.app_instance, "app_config", {}) or {}
            )
            status_chips = ConsoleStatusChips(
                control_state,
                scope_state=retrieval_scope_state,
                collapsed=self._console_status_chips_collapsed,
                # F1 (final review): compose the chip correctly on the very
                # first render instead of relying on a post-mount sync call
                # that some code paths (screen recreation via
                # restore_state) never make.
                ephemeral=self._console_active_session_is_ephemeral(),
                cost_state=initial_cost_state,
                # FB-08 (TASK-2154.18): same first-frame precedent for the
                # run chip -- returning to Console while a background run
                # is still active must show it before the next sync tick.
                run_copy=self._console_active_run_copy(),
                id="console-status-chips",
                classes="ds-panel",
            )
            # RAG-40: staged evidence belongs on the MAIN surface -- not only
            # in an Inspector rail the staging path never opens. task-17661:
            # ALL transient strips (staged evidence, prompt queue) sit at the
            # TOP of the control deck, above the status line, so the area
            # around the composer stays visually quiet.
            yield ConsoleStagedEvidenceStrip(
                self._build_console_staged_evidence_strip_state(pending_launch),
                id="console-staged-evidence-strip",
                classes="ds-panel",
            )
            store = self._console_chat_store
            recovery = None
            if store is not None and store.active_session_id is not None:
                recovery = store.dispatch_recovery_for_presentation(
                    store.active_session_id
                )
            yield ConsoleDispatchRecoveryRegion(
                recovery,
                session_id=(store.active_session_id if store is not None else ""),
                id="console-dispatch-recovery",
                classes="ds-panel",
                on_action=self._dispatch_console_recovery_action,
            )
            yield ConsolePromptQueueRegion(
                id="console-prompt-queue",
                on_manage_requested=(
                    lambda session_id, revision: self.app.push_screen(
                        ConsolePromptQueueModal(
                            session_id=session_id,
                            revision=revision,
                            queue_controller=self._prompt_queue,
                        )
                    )
                ),
                on_primary_requested=(
                    lambda session_id, revision, action: self.run_worker(
                        self._prompt_queue.handle_primary_intent(
                            session_id,
                            action=action,
                            expected_revision=revision,
                        ),
                        exclusive=True,
                        group="console-prompt-queue-shelf",
                    )
                ),
            )
            if status_chips_position == STATUS_CHIPS_POSITION_ABOVE:
                yield status_chips
            composer = ConsoleComposerBar(
                id="console-native-composer",
                classes="ds-panel",
                collapsed=self._console_composer_collapsed,
                collapse_large_pastes=self._console_collapse_large_pastes_enabled(),
                paste_collapse_threshold=self._console_paste_collapse_threshold(),
                send_price_tooltip_provider=(
                    lambda draft: self._send_price.tooltip_for_draft(draft)
                ),
                # TASK-23018: the cheap half. `sync_action_state` runs on
                # every printable keystroke and only needs to know WHETHER a
                # price exists (the Send label's "| $" suffix); the tooltip
                # itself is derived on hover.
                send_price_available_provider=(
                    lambda draft: self._send_price.availability_for_draft(draft)
                ),
            )
            # TASK-1364: the composer shares the screen's prompt-history
            # store with the controller (which records accepted sends) so
            # ghost text and Up/Down recall see this app's own past prompts.
            composer.set_prompt_history(self._ensure_console_prompt_history())
            store = self._console_chat_store
            if store is not None and store.active_session_id is not None:
                try:
                    composer.load_draft(store.session_draft(store.active_session_id))
                except KeyError:
                    pass
            # TASK-17651: the composer is a dense-form field, not a framed
            # region — CSS owns its left-edge marker and focus treatment.
            yield composer
            # In "below" mode the chips close the shell as a bottom status
            # row: the composer cluster (staged evidence, prompt queue,
            # composer) stays contiguous with the transcript, and the chips
            # annotate the whole surface from underneath.
            if status_chips_position != STATUS_CHIPS_POSITION_ABOVE:
                yield status_chips
            yield ConsoleCommandPopup()
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
        return coerce_bool_setting(
            console_config.get("collapse_large_pastes", True), True
        )

    def _stack_collapsed_rail_labels(self) -> bool:
        """Return whether fresh collapsed Console handles use stacked labels."""
        app_config = getattr(self.app_instance, "app_config", {}) or {}
        console_config = app_config.get("console", {})
        if not isinstance(console_config, dict):
            return False
        return coerce_bool_setting(
            console_config.get("stack_collapsed_rail_labels", False),
            False,
        )

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
        """Initialize the native Console screen."""
        # No super().on_mount(): the dispatcher already invokes
        # BaseAppScreen.on_mount separately for this Mount event.

        self.app_instance._console_h3_image_edit_screen = self
        self._apply_focus_chrome()
        if not hasattr(self, "_console_h3_terminal_generations"):
            self._console_h3_terminal_generations: set[str] = set()
        ordered_resume_pending = self._pending_resume_local_conversation_id is not None
        self._resume_navigation_startup_in_progress = ordered_resume_pending
        # This handoff is session/config only and does not need mounted DOM.
        # Consume it before ordinary UI restoration can create a competing
        # default session with a different identity. An explicit Resume owns
        # presentation first and settles this as its first ordered intent.
        if not ordered_resume_pending:
            self._session.consume_pending_console_first_chat_intent()
        self._notify_console_fleet_teardown_if_any()
        # PR3a-2 Task 5: claim staged auto-wakes SYNCHRONOUSLY, before any
        # timer or worker below can run the first tab sync -- whose
        # view-clear consumes the ACTIVE conversation's FLEET_UNSEEN mark
        # (Task 4's stated ordering hazard: read marks BEFORE activation).
        self._fleet._claim_console_fleet_wake_marks()
        self._console_auto_speak.mount()

        # Restore collapsible states after mount
        self.set_timer(0.1, self._restore_collapsible_states)
        self.set_timer(0.05, self.sync_task_resume_state)
        if self._task_resume_state.followed_watchlists_operations:
            self.set_timer(
                0.06,
                partial(
                    self._follow_console_watchlists_operations,
                    self._task_resume_state.followed_watchlists_operations,
                ),
            )
        if ordered_resume_pending:
            self.set_timer(0.15, self._start_resume_navigation_startup)
        else:
            self.set_timer(0.15, self._consume_pending_chat_handoff)
            self.set_timer(0.15, self._consume_pending_console_roleplay_repair)
            # Mirrors the handoff timer above: the native composer is not
            # guaranteed to exist in the DOM yet at this exact point (it can
            # still be settling in immediately after mount, same reason every
            # composer-touching test here awaits `_wait_for_selector` first) --
            # a failed early attempt releases its claim for this screen's
            # existing resume/user-triggered retry paths.
            self.set_timer(0.15, self._consume_pending_console_prompt_insert)
            self.set_timer(0.15, self.consume_pending_console_provider_intent)
            # PR3a-2 Task 4: claim a background sub-agent completion's deep
            # link (staged while Console was not mounted) and switch to the
            # settled conversation's session. Same 0.15s settle hedge as the
            # surrounding handoff timers.
            self.set_timer(
                0.15,
                self._fleet.consume_pending_console_fleet_completion,
            )
        # PR3a-2 Task 4 (task-15664): mount hedge for the survivor tick --
        # the primary arming point is the transcript poll's self-stop
        # edge, but a controller wired at mount with survivors already
        # live (e.g. a future above-screen bridge) must not stay frozen.
        # A no-op when nothing is live.
        self.set_timer(0.3, self._fleet._maybe_start_console_fleet_survivor_tick)
        # Same hedge as the handoff timers above: the native composer is not
        # guaranteed to exist in the DOM yet at `call_after_refresh` time
        # either, and `_sync_console_dictation_availability` silently no-ops
        # when it doesn't -- without this retry, a mount that loses that race
        # would leave the mic showing its unmounted-default tooltip until the
        # user's first activation attempt re-probes it.
        self.call_after_refresh(self._sync_console_dictation_availability)
        self.set_timer(0.15, self._sync_console_dictation_availability)
        self.call_after_refresh(self._image._reconcile_h3_image_edit_completions)
        if not ordered_resume_pending:
            self.call_after_refresh(self._sync_native_console_chat_ui)
            self.call_after_refresh(self._restore_console_workbench_focus)
            self.set_timer(0.2, self._restore_console_workbench_focus)
        self.run_worker(
            self._skill._refresh_console_skill_candidates(), exclusive=False
        )
        # task-15475: claim this visit's refreshes; the ScreenResume Textual
        # posts for this very mount consumes the token and skips its own copy.
        self._console_mount_visit_refreshed = True

    def _start_resume_navigation_startup(self) -> None:
        """Start the one ordered worker for an explicit saved-chat resume."""
        self.run_worker(
            self._consume_resume_navigation_startup(),
            exclusive=True,
            group="console-resume-navigation-startup",
        )

    async def _consume_resume_navigation_startup(self) -> None:
        """Consume older Console intents before the explicit resume target."""
        target = self._pending_resume_local_conversation_id
        self._pending_resume_local_conversation_id = None
        opened: bool | None = None
        try:
            if target is None:
                return
            self._session.consume_pending_console_first_chat_intent(
                defer_presentation=True,
            )
            await self._consume_pending_chat_handoff(
                suppress_released_failure=True,
            )
            self._consume_pending_console_roleplay_repair()
            await self._consume_pending_console_prompt_insert()
            self.consume_pending_console_provider_intent()
            fleet_result = self._fleet.consume_pending_console_fleet_completion()
            if inspect.isawaitable(fleet_result):
                await fleet_result
            opened = await self._workspace.open_console_workspace_conversation(target)
        finally:
            self._resume_navigation_startup_in_progress = False
        if opened is True:
            return
        store = self._ensure_console_chat_store()
        if store.active_session_id is not None:
            return
        try:
            self._workspace._reconcile_console_session_with_registry()
        except Exception:
            logger.opt(exception=True).debug(
                "Unable to reconcile Console after failed saved-chat startup resume"
            )
        await self._sync_native_console_chat_ui()

    def _notify_console_fleet_teardown_if_any(self) -> None:
        """One-shot toasts reporting the LAST Console instance's teardown.

        TASK-1143 (F5): navigating away from Console unmounts the screen,
        and ``on_unmount`` (via ``_record_console_fleet_teardown``)
        records the teardown's two truthful counts on the app object --
        ``_console_fleet_teardown_notice`` for sessions ``shutdown()``
        genuinely killed (active turn / pending approval, whose in-flight
        children die with the turn) and, since PR3a-2 Task 4,
        ``_console_fleet_survivor_notice`` for sessions whose only work
        was cross-turn survivors, which KEEP RUNNING through teardown
        (Task 1 A4 executed the old copy's lie: it called those
        "cancelled" while they finished ``done``). The app outlives this
        screen instance, and screens are never cached
        (``TldwCli._create_navigation_screen`` always builds a fresh
        instance). Each non-zero slot is shown exactly once and cleared
        so an ordinary mount (nothing to report) stays silent.
        """
        killed = getattr(self.app_instance, "_console_fleet_teardown_notice", 0)
        surviving = getattr(self.app_instance, "_console_fleet_survivor_notice", 0)
        if killed:
            self.app_instance._console_fleet_teardown_notice = 0
            noun = "run" if killed == 1 else "runs"
            verb = "was" if killed == 1 else "were"
            self.app_instance.notify(
                f"{killed} agent {noun} {verb} cancelled when you left Console.",
                severity="warning",
            )
        # PR3a-2 Task 4 (Task 1 A4): the pre-3a-2 notice counted these
        # sessions in the "cancelled" toast above -- while their
        # sub-agents in fact kept running through shutdown() and finished
        # (executed, not inferred). Report what actually happens: the work
        # continues in the background, its results land in the run log,
        # its spend folds onto the message row (Task 3), and its settle
        # raises the app-wide completion toast + unseen badge (this task).
        if surviving:
            self.app_instance._console_fleet_survivor_notice = 0
            if surviving == 1:
                copy = (
                    "1 conversation's sub-agents kept running in the "
                    "background when you left Console — you'll be notified "
                    "as they finish."
                )
            else:
                copy = (
                    f"{surviving} conversations' sub-agents kept running in "
                    "the background when you left Console — you'll be "
                    "notified as they finish."
                )
            self.app_instance.notify(copy, severity="information")

    async def confirm_navigation(self) -> bool:
        """Delegate revision-pinned Console loss confirmation."""

        controller = self._console_chat_controller
        if controller is None:
            return True
        return await self._session.confirm_navigation(controller)

    async def confirm_quit(self) -> bool:
        """Delegate revision-pinned Console loss confirmation for app quit."""

        controller = self._console_chat_controller
        if controller is None:
            return True
        return await self._session.confirm_quit(controller)

    def prepare_for_quit(self) -> None:
        """Tombstone Console future work before application cleanup."""

        controller = self._console_chat_controller
        if controller is not None:
            controller.begin_shutdown()

    async def on_unmount(self) -> None:
        """Release Console-native resources owned by this screen."""
        # task-15470: flush a pending debounced sidebar-state write FIRST,
        # ahead of every other teardown step below -- several of those can
        # raise, and a raised exception must not strand an unpersisted
        # toggle-then-quit.
        await self._flush_sidebar_state_now()
        self._message.invalidate_console_speech_context()
        self._console_auto_speak.unmount()
        registry = self._image._h3_image_edit_registry()
        store = self._console_chat_store
        if store is not None:
            terminal = getattr(self, "_console_h3_terminal_generations", None)
            if terminal is None:
                terminal = set()
                self._console_h3_terminal_generations = terminal
            for session in store.sessions():
                operation = registry.request_cancel(session.id)
                if operation is not None:
                    terminal.add(operation.generation)
        if getattr(self.app_instance, "_console_h3_image_edit_screen", None) is self:
            self.app_instance._console_h3_image_edit_screen = None
        self._video._drain_pending_console_videos()
        self._stop_console_transcript_sync_timer()
        self._fleet._stop_console_fleet_survivor_tick()
        self._stop_console_cost_ttl_timer()
        await self._teardown_console_roleplay_persistence()
        # The pipeline hands-free loop's own two-statement abandon teardown
        # now lives in the decomposed controller (wave-2 console
        # decomposition, task 1); calling it here keeps this method one line
        # per subsystem, matching `_dictation.teardown()` below.
        self._hands_free.teardown()
        await self._realtime.teardown()
        # Dictation's own seven-statement teardown now lives in the
        # decomposed controller (dev's wave-1 extraction); calling it here
        # keeps this method one line per subsystem.
        await self._dictation.teardown()
        self._console_original_attempt_previews.clear()
        self._hands_free.uninstall_console_hands_free_store_tap()
        controller = self._console_chat_controller
        if controller is not None:
            await self._fleet._record_console_fleet_teardown()
        else:
            # No controller was ever built, but the view still has to let
            # go: `detach_view` clears the store's `on_scope_flushed` and
            # drops the claim.
            await leave_console_runtime(self.app_instance, view=self)
        super().on_unmount()

    @classmethod
    def _serialize_console_message(cls, message: ConsoleChatMessage) -> dict[str, Any]:
        """Delegate to `ConsoleMessageController` (wave-3 task 1).

        **No production caller since task-15860 Task 3**: message state
        stopped travelling in the screen-state snapshot, so nothing in the
        app serializes a Console message any more. Kept (unbound,
        `ChatScreen.X(...)`) only for the pre-existing test suite's
        direct-call convention; retiring it and its counterpart is a
        separate, mechanical cleanup.
        """
        return ConsoleMessageController._serialize_console_message(message)

    @classmethod
    def _restore_console_message(cls, payload: Any) -> ConsoleChatMessage | None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1).

        **No production caller since task-15860 Task 3** -- see
        `_serialize_console_message` above.
        """
        return ConsoleMessageController._restore_console_message(payload)

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
        for completion in self._image._h3_image_edit_registry().completions():
            self._image._filter_h3_attachment_from_app_stash(
                completion.session_id, completion.attachment_id
            )

    def _adopt_console_pending_attachments(self, store: ConsoleChatStore) -> None:
        """Re-stage stashed attachments into the restored store, then empty
        the stash. Entries for sessions that no longer exist are dropped.

        The stash attribute is reset the moment it is read — every adopt
        attempt releases the byte references, even when the stash turned out
        malformed or nothing could be adopted (self-healing; the bytes must
        never outlive their one restore opportunity).

        task-15860 (Task 3): the store now SURVIVES the navigation, so the
        session this re-stages into still holds the very pendings the stash
        was copied from. Each adopted session is therefore cleared first —
        without it every navigation would DOUBLE the staged attachments
        (up to the cap) instead of restoring them. The stash stays
        authoritative for exactly one thing the store cannot know: an H3
        image edit that completed while Console was away has already been
        filtered out of it.
        """
        app = getattr(self, "app_instance", None)
        if app is None:
            return
        stash = getattr(app, self._CONSOLE_PENDING_STASH_ATTR, None)
        setattr(app, self._CONSOLE_PENDING_STASH_ATTR, {})
        if not isinstance(stash, dict) or not stash:
            self._image._reconcile_h3_image_edit_completions(store)
            return
        live_ids = {session.id for session in store.sessions()}
        completed_attachment_ids = {
            completion.session_id: completion.attachment_id
            for completion in self._image._h3_image_edit_registry().completions()
        }
        for session_id, pendings in stash.items():
            if session_id not in live_ids:
                continue
            if not isinstance(pendings, (list, tuple)):
                continue
            store.clear_pending_attachments(session_id)
            for pending in pendings:
                if getattr(
                    pending, "attachment_id", None
                ) == completed_attachment_ids.get(session_id):
                    continue
                if not store.add_pending_attachment(session_id, pending):
                    break  # staging cap reached — matches live staging semantics
        self._image._reconcile_h3_image_edit_completions(store)

    def _serialize_native_console_state(self) -> dict[str, Any] | None:
        """Return the native Console VIEW state for screen restoration.

        task-15860 (Task 3): message state no longer travels here. The
        app-owned `ConsoleRuntime`'s store outlives every `ChatScreen`, so
        `sessions`, `messages_by_session` and `active_session_id` are read
        straight off the surviving store by the next visit -- carrying
        copies in a `ScreenStateStore` snapshot made the snapshot a SECOND
        source of truth that silently won at the next mount. Task 0's P3b
        executed the cost: a wake turn that ran, spent money and stamped
        the ledger while Console was unmounted persisted four rows, and
        the user returning saw the two that predated the snapshot.

        What stays is genuinely SCREEN-instance state, which dies with the
        screen and has nowhere else to live: the image view-mode overrides,
        the task-resume projection, the Library RAG source scope, the
        staged live-work launch and the "evidence sent" memory.

        The composer flush below stays too, and is now load-bearing rather
        than incidental: it is the one place the VIEW's uncommitted draft
        is written back into the store that will outlive it.

        The pending-attachment stash also stays. It never travelled in this
        payload (bytes are forbidden here; it lives on the APP object), so
        it is not a second source of truth for message state -- and it is
        what `_adopt_console_pending_attachments` re-stages the H3-filtered
        set from.
        """
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

        pending_launch = getattr(self, "_pending_console_launch_context", None)
        sent_notice = getattr(self, "_console_evidence_sent_notice", None)

        return {
            "version": NATIVE_CONSOLE_STATE_VERSION,
            "task_resume_state": self._task_resume_state.to_dict(),
            "image_view_modes": image_state.serialize(),
            # RAG-44: an edited Library RAG source selection is Console-local,
            # but it must survive a tab switch like the sessions around it --
            # otherwise "Prompts on" silently reverts the next time the user
            # comes back and retrieval quietly reads something else.
            "library_rag_source_types": list(_console_library_rag_source_scope(self)),
            # PR-T1/task-3 (D3): `_pending_console_launch_context` and
            # `_console_evidence_sent_notice` are screen-INSTANCE state set in
            # `ChatScreen.__init__`, not app-owned -- and screens are never
            # cached/reused (`TldwCli._create_navigation_screen` builds a
            # fresh instance on every navigation). Without carrying them here,
            # ANY navigation away from Console silently dropped staged
            # evidence, with no error and no user-visible warning. Both are
            # read via `getattr` (not a bare attribute access) so this method
            # keeps working against the bare screen shells several existing
            # tests build with `ChatScreen.__new__` to exercise serialize/
            # restore without a mounted app.
            # `to_pending_payload()` is the exact shape `PendingHandoffStore`
            # already stores a Console live-work launch in, so restoring it
            # via `ConsoleLiveWorkLaunch.from_pending` is the same
            # reconstruction a handoff claim goes through.
            "pending_console_launch": (
                pending_launch.to_pending_payload()
                if pending_launch is not None
                else None
            ),
            "console_evidence_sent_notice": sent_notice,
        }

    def _restore_native_console_state(self, payload: Any) -> None:
        """Restore native Console VIEW state saved by ``save_state``.

        task-15860 (Task 3): this method no longer rebuilds the store. The
        sessions, their transcripts and the active session are already
        there -- the app-owned `ConsoleRuntime` holds the same
        `ConsoleChatStore` across every navigation, and `store.restore_state`
        would REPLACE its contents with a snapshot taken before the last
        turn (Task 0's P3b: four persisted rows, two shown). Dropping the
        replacement also stops five losses the snapshot round trip caused
        by construction, because none of them had a slot in the payload:
        the message TREE (branch/variant history was flattened to a linear
        chain), the local active-leaf cursor, the `/rewind` context summary,
        per-session speech preferences and the one-shot prefill.

        Reaching the runtime here is still load-bearing:
        `_complete_screen_navigation` restores the INCOMING screen before
        `switch_screen` unmounts the outgoing one, so this is where the
        incoming view CLAIMS the runtime (`ensure_console_runtime(app,
        view=self)` -> `attach_view`), in time for the outgoing screen's
        later `detach_view` to find a different claimant and do nothing.

        Measured, not assumed: a mutation that removed no-runtime-touch
        from this method went red on the headless-wake continuity test.
        A weaker mutation -- swapping `_ensure_console_chat_store()` for a
        bare `self._console_chat_store` read -- stayed GREEN, because that
        attribute is itself a runtime-backed property and claims just the
        same. So it is the runtime CONTACT that matters here, not this
        particular spelling of it.
        """
        if not isinstance(payload, dict):
            return

        store = self._ensure_console_chat_store()
        self._message.invalidate_console_speech_context()
        self._adopt_console_pending_attachments(store)
        self._console_visible_draft_session_id = None
        self._last_native_transcript_refresh_key = None

        image_state, cache = self._ensure_console_image_view()
        image_state.restore(payload.get("image_view_modes"))
        cache.clear()
        self._task_resume_state = TaskResumeState.from_dict(
            payload.get("task_resume_state")
        )
        # Plain assignment, not `_set_console_library_rag_source_scope`: the
        # readiness card is (re)built after a restore, and this path also
        # runs against screen shells that never mounted a DOM to query.
        # A legacy payload has no key at all -> the unchanged default.
        self._console_library_rag_source_types = normalize_console_rag_source_types(
            payload.get("library_rag_source_types")
        )
        # PR-T1/task-3 (D3): restore the staged live-work launch and the
        # "evidence sent" memory `_serialize_native_console_state` saved
        # above. `ConsoleLiveWorkLaunch.from_pending` returns `None` for a
        # legacy payload (key absent entirely) exactly as it does for an
        # explicit `None`, so an old save restores cleanly to "nothing
        # staged" -- no `NATIVE_CONSOLE_STATE_VERSION` gating needed here,
        # matching every other field in this method (each tolerates an
        # absent key with `payload.get(...)` rather than branching on
        # "version").
        #
        # A non-`None` result is a fully reconstructed `ConsoleLiveWorkLaunch`,
        # not a `PendingHandoffStore` claim -- `_consume_pending_console_
        # launch`'s early return (`self._pending_console_launch_context is
        # not None`) treats it as already-claimed and never reaches back into
        # the store for it, so restoring a launch here can never re-trigger
        # `store.claim()`/`store.acknowledge()`. `_pending_console_launch_
        # auto_open_inspector` is reset to its `__init__` default (`False`):
        # the auto-open-once behavior is for a launch that JUST arrived via a
        # live handoff, not one merely surviving a tab switch.
        self._pending_console_launch_context = ConsoleLiveWorkLaunch.from_pending(
            payload.get("pending_console_launch")
        )
        self._pending_console_launch_auto_open_inspector = False
        raw_sent_notice = payload.get("console_evidence_sent_notice")
        self._console_evidence_sent_notice = (
            raw_sent_notice
            if isinstance(raw_sent_notice, int)
            and not isinstance(raw_sent_notice, bool)
            else None
        )

    def _rehydrate_console_message_image(self, message: ConsoleChatMessage) -> None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1).

        **No production caller since task-15860 Task 3**: with the store
        surviving the navigation there is no snapshot to rehydrate from --
        the live message objects never lost their bytes. Kept for the
        pre-existing test suite's direct-call convention.
        """
        self._message._rehydrate_console_message_image(message)

    def _rehydrate_console_message_attachments(
        self, messages: list[ConsoleChatMessage]
    ) -> None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1).

        **No production caller since task-15860 Task 3** -- see
        `_rehydrate_console_message_image` above.
        """
        self._message._rehydrate_console_message_attachments(messages)

    def _rehydrate_console_message_generation_metadata(
        self,
        store: "ConsoleChatStore",
        restored_messages_by_session: Dict[str, list[ConsoleChatMessage]],
    ) -> None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1).

        **No production caller since task-15860 Task 3** -- see
        `_rehydrate_console_message_image` above.
        """
        self._message._rehydrate_console_message_generation_metadata(
            store, restored_messages_by_session
        )

    def save_state(self) -> Dict[str, Any]:
        """Save only state owned by the native Console."""
        state = super().save_state()
        native_console_state = self._serialize_native_console_state()
        if native_console_state is not None:
            state["native_console_state"] = native_console_state
            state["interface_type"] = "native_console"
        return state

    def console_prompt_target_projection(
        self,
    ) -> ConsolePromptTargetProjection | None:
        """Return the sanitized live Prompt target owned by the controller.

        Returns:
            The active Console target projection, or ``None`` when unavailable.
        """
        return self._prompts.console_prompt_target_projection()

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore only state owned by the native Console."""
        super().restore_state(state)
        native_console_state = state.get("native_console_state")
        if native_console_state is not None:
            self._restore_native_console_state(native_console_state)
        # task-15860 Task 5: the snapshot's `task_resume_state` is a VIEW
        # projection taken when the last Console visit ended, so restoring
        # it plainly (`_restore_native_console_state`) ERASES an approval
        # round armed since -- the headless case, where a risk-tagged tool
        # in a wake turn arms one with nothing mounted. The app-owned
        # controller is the only source of truth for what is armed, so
        # re-derive from it AFTER the snapshot lands. Measured: without
        # this the attach-time remount ran, set the card, and was
        # overwritten microseconds later by the snapshot.
        #
        # Scoped deliberately: this MOUNTS an armed round, it does not
        # CLEAR a stale one. A snapshot carrying a `pending_approval` for
        # a round that has since resolved still restores a dead card
        # (clicking it resolves nothing -- `resolve_pending_approval`
        # fails closed on the missing round id). That is a pre-existing
        # defect on this path, no red was reproduced for it here, and
        # fixing it belongs with whoever does.
        self._console_runtime().remount_pending_approval()
        self.sync_task_resume_state()

    async def _consume_pending_chat_handoff(
        self,
        *,
        suppress_released_failure: bool = False,
    ) -> None:
        """Claim one Chat handoff and stage it directly in native Console.

        Args:
            suppress_released_failure: Return after a transfer failure only
                when this invocation released its exact claim for retry.
        """
        if self._handoff_consumption_in_progress:
            return

        try:
            store = self.app_instance.pending_handoffs
            claim = store.claim(HandoffChannel.CHAT)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Chat handoff acquisition failed (channel={}, exception_category={})",
                HandoffChannel.CHAT.value,
                type(exc).__name__,
            )
            raise
        if claim is None:
            return

        self._handoff_consumption_in_progress = True
        try:
            payload = claim.value

            # The native Console composes no legacy tab surface. A
            # Personas Start-Chat character handoff gets a dedicated
            # character-bound session with its greeting seeded
            # (task-427); anything else -- or a character session that
            # failed to build -- stages into the Console live-work lane
            # so the context lands in Staged Context instead of being
            # dropped with a warning.
            if await self._session._start_character_console_session(payload):
                store.acknowledge(claim)
                return
            self._stage_handoff_as_console_live_work(payload)
            store.acknowledge(claim)
        except asyncio.CancelledError:
            try:
                store.release(claim)
            except Exception as exc:
                logger.warning(
                    "Chat handoff cancellation release failed "
                    "(channel={}, revision={}, exception_category={})",
                    claim.channel.value,
                    claim.revision,
                    type(exc).__name__,
                )
            raise
        except Exception as exc:
            try:
                released = store.release(claim)
            except Exception as release_exc:
                released = False
                logger.warning(
                    "Chat handoff transfer release failed "
                    "(channel={}, revision={}, exception_category={})",
                    claim.channel.value,
                    claim.revision,
                    type(release_exc).__name__,
                )
            logger.warning(
                "Chat handoff transfer failed "
                "(channel={}, revision={}, exception_category={})",
                claim.channel.value,
                claim.revision,
                type(exc).__name__,
            )
            if suppress_released_failure and released:
                return
            raise
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
        snippet = _safe_text(payload.display_summary or payload.body, max_length=4_000)
        title = _safe_text(payload.title) or "Untitled"
        launch_payload: dict[str, Any] = {
            "target_id": _safe_text(payload.content_ref or payload.source_id or title),
            "item_type": _safe_text(payload.item_type),
            "source_id": _safe_text(payload.source_id),
            "snippet": snippet,
            "suggested_prompt": _safe_text(payload.suggested_prompt, max_length=4_000),
            "runtime_backend": _safe_text(payload.runtime_backend),
            "source_selector_state": _safe_text(payload.source_selector_state),
            "metadata": dict(payload.metadata or {}),
        }
        # Task-2 review bonus find (Task 9): this used to run only when
        # `"rag" in (payload.source or "").lower()`. RAG-class sources gate
        # Console sends on available evidence, and that gate is exactly why
        # this branch existed -- but restricting bundle-building to a
        # source-name substring meant every OTHER handoff (Library media,
        # Library conversations, Library notes, and any future source that
        # doesn't happen to spell "rag") staged visibly in the strip/tray
        # while `capture_console_staged_evidence_for_chat` silently returned
        # `LocalRagContextResult(None, None)` on send, because
        # `payload.get("evidence_bundle")` was never a mapping for them: a
        # live content-loss bug, not merely a missing gate. The gate itself
        # is dropped rather than widened to an allowlist of known source
        # names, since an allowlist only recreates the same class of bug for
        # the next new source. Every handoff staged here now always carries
        # a single-reference bundle (title stands in when the snippet is
        # empty) so it can never dead-end the send, and the non-RAG sources
        # this restores content for are never subject to the RAG evidence
        # send-gate in the first place (`_console_send_blocked_reason` only
        # checks it for a source whose label mentions "rag").
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
        except (TypeError, ValueError, ValidationError) as exc:
            logger.warning(
                "Could not build evidence bundle for handoff (exception_category={})",
                type(exc).__name__,
            )

        # PR-4/task-1: route through the staging SEAM, never a bare
        # assignment. This method finishes via `_sync_native_console_chat_ui`,
        # which refreshes the chip but not the staged-evidence strip or the
        # Inspector tray -- so a "Use in Console" handoff landing on a
        # composed screen used to read "Sources: 1 staged" with nothing
        # listed and no reachable un-stage control, and then had the strip
        # announce "Evidence sent" for evidence the user was never shown.
        # The auto-open flag is set BEFORE staging for the same reason the
        # Library-RAG failure path does: staging syncs the rail state
        # synchronously, so a flag set afterwards misses that pass.
        self._pending_console_launch_auto_open_inspector = True
        self._retrieval._stage_console_library_rag_launch(
            ConsoleLiveWorkLaunch.from_values(
                source=payload.source,
                title=payload.title,
                payload=launch_payload,
                status=payload.status or "staged",
            )
        )

        suggested_prompt = launch_payload["suggested_prompt"]
        if suggested_prompt:
            store = self._ensure_console_chat_store()
            if _is_personas_preview_handoff(payload):
                # task-428: a Roleplay "Open in Console" handoff must land in
                # its own fresh, focused conversation -- never bleed into (or
                # reuse) whatever Console conversation is already active.
                # ``create_session`` pre-activates the new session, which IS an
                # active-session switch: snapshot the composer first (TASK-339)
                # so any keystrokes typed in the settle window before the
                # deferred ``_sync_console_session_draft`` carry forward into
                # the new session instead of being saved to the old one and
                # wiped. We deliberately do NOT poke the composer here -- the
                # sync pass below owns it and loads the new session's draft;
                # poking it would save this prompt back into the old session.
                # ``title`` (a sanitized ``payload.title``) is a real,
                # non-default title, so the send-time auto-titler
                # (``_maybe_auto_title_session``) leaves it as-is rather than
                # renaming it after the prefilled instruction.
                self._session._capture_console_draft_switch_snapshot()
                session = store.create_session(
                    title=title,
                    workspace_id=store.workspace_context.active_workspace_id,
                    settings=self._session._default_console_session_settings(),
                )
                store.set_session_draft(session.id, suggested_prompt)
            else:
                session = store.ensure_session(
                    title=self._workspace._console_initial_session_title_for_workspace(
                        store.workspace_context.active_workspace_id
                    ),
                    workspace_id=store.workspace_context.active_workspace_id,
                    settings=self._session._default_console_session_settings(),
                )
                if not store.session_draft(session.id).strip():
                    store.set_session_draft(session.id, suggested_prompt)
                try:
                    composer = self.query_one(
                        "#console-native-composer", ConsoleComposerBar
                    )
                except QueryError:
                    pass
                else:
                    if not composer.draft_text().strip():
                        composer.load_draft(suggested_prompt)
                        self._sync_console_command_popup()

        self.run_worker(
            self._sync_native_console_chat_ui,
            exclusive=True,
            group="console-sync",
        )

    def _console_citation_modal_request_is_current(
        self,
        *,
        native_message_id: str,
        persisted_message_id: str,
        current_body: str,
        repository: Any,
        repository_token: tuple[str, int, int, int],
    ) -> bool:
        """Return whether one open modal still targets the active message."""

        current_token, current_repository = (
            self._console_citation_repository_readiness()
        )
        if current_repository is not repository or current_token != repository_token:
            return False
        matching_messages = [
            message
            for message in self._message._native_console_messages()
            if getattr(message, "id", None) == native_message_id
        ]
        if len(matching_messages) != 1:
            return False
        message = matching_messages[0]
        return (
            getattr(message, "role", None) is ConsoleMessageRole.ASSISTANT
            and getattr(message, "status", None) == "complete"
            and getattr(message, "persisted_message_id", None) == persisted_message_id
            and self._console_citation_message_body(message) == current_body
        )

    @on(Button.Pressed, ".console-transcript-citation-sources")
    def handle_console_citation_sources(self, event: Button.Pressed) -> None:
        """Open one lazy Sources modal for a current persisted assistant."""

        event.stop()
        native_message_id = getattr(event.button, "native_message_id", None)
        if type(native_message_id) is not str or not native_message_id:
            return
        matching_messages = [
            message
            for message in self._message._native_console_messages()
            if getattr(message, "id", None) == native_message_id
        ]
        if len(matching_messages) != 1:
            return
        message = matching_messages[0]
        persisted_message_id = getattr(message, "persisted_message_id", None)
        if (
            getattr(message, "role", None) is not ConsoleMessageRole.ASSISTANT
            or getattr(message, "status", None) != "complete"
            or type(persisted_message_id) is not str
            or not persisted_message_id
        ):
            return
        current_body = self._console_citation_message_body(message)
        repository_token, repository = self._console_citation_repository_readiness()
        if repository is None:
            return
        modal = ConsoleCitationSourcesModal(
            native_message_id=native_message_id,
            persisted_message_id=persisted_message_id,
            current_body=current_body,
            repository=repository,
            request_is_current=lambda: self._console_citation_modal_request_is_current(
                native_message_id=native_message_id,
                persisted_message_id=persisted_message_id,
                current_body=current_body,
                repository=repository,
                repository_token=repository_token,
            ),
        )

        def _open_source_in_library(result: dict[str, str] | None) -> None:
            if not isinstance(result, dict):
                return
            source_type = result.get(LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE)
            source_id = result.get(LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID)
            if type(source_type) is not str or type(source_id) is not str:
                return
            self.app.post_message(
                NavigateToScreen(
                    "library",
                    {
                        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE: source_type,
                        LIBRARY_NAV_CONTEXT_OPEN_SOURCE_ID: source_id,
                    },
                )
            )

        self.app.push_screen(modal, callback=_open_source_in_library)

    def _console_citation_message_body(self, message: Any) -> str:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        for the citation cluster's own staying callers."""
        return self._message._console_citation_message_body(message)

    def _console_citation_signature(
        self,
        messages: list[Any],
    ) -> tuple[str | None, tuple[tuple[str, str, str, str], ...]]:
        """Return the active-session signature for eligible citation lookups."""
        store = self._ensure_console_chat_store()
        eligible: list[tuple[str, str, str, str]] = []
        for message in messages:
            if (
                getattr(message, "role", None) is not ConsoleMessageRole.ASSISTANT
                or getattr(message, "status", None) != "complete"
            ):
                continue
            native_message_id = getattr(message, "id", None)
            persisted_message_id = getattr(message, "persisted_message_id", None)
            if (
                not isinstance(native_message_id, str)
                or not native_message_id
                or not isinstance(persisted_message_id, str)
                or not persisted_message_id
            ):
                continue
            eligible.append(
                (
                    native_message_id,
                    persisted_message_id,
                    self._console_citation_message_body(message),
                    "complete",
                )
            )
        return (store.active_session_id, tuple(eligible))

    def _console_citation_repository_readiness(
        self,
    ) -> tuple[tuple[str, int, int, int], Any | None]:
        """Return a bounded repository identity token and a valid repository."""
        repository = getattr(
            self.app_instance,
            "citation_trace_repository",
            None,
        )
        if repository is None:
            return (("missing", 0, 0, 0), None)
        app_db = getattr(self.app_instance, "chachanotes_db", None)
        repository_db = getattr(repository, "db", None)
        if repository_db is not app_db:
            return (
                (
                    "mismatch",
                    id(repository),
                    id(repository_db),
                    id(app_db),
                ),
                None,
            )
        return (
            (
                "valid",
                id(repository),
                id(repository_db),
                id(app_db),
            ),
            repository,
        )

    def _sync_console_citation_count_discovery(self, messages: list[Any]) -> None:
        """Dispatch one count lookup worker when eligible inputs change."""
        signature = self._console_citation_signature(messages)
        repository_token, repository = self._console_citation_repository_readiness()
        repository_changed = repository_token != self._console_citation_repository_token
        if repository_changed:
            self._console_citation_repository_token = repository_token
            self._console_citation_input_signature = signature
            self._console_citation_request_generation += 1
            self._console_citation_counts = {}
            self._console_citation_resolved_signatures = {}
            if repository is None or not signature[1]:
                return
            unresolved = signature[1]
            generation = self._console_citation_request_generation
            self.run_worker(
                self._discover_console_citation_counts(
                    repository,
                    signature,
                    generation,
                    unresolved,
                    repository_token,
                ),
                exclusive=True,
                group="console-citation-counts",
            )
            return
        if repository is None:
            if signature != self._console_citation_input_signature:
                self._console_citation_input_signature = signature
                self._console_citation_request_generation += 1
            self._console_citation_counts = {}
            self._console_citation_resolved_signatures = {}
            return
        if signature == self._console_citation_input_signature:
            return

        previous_signature = self._console_citation_input_signature
        same_session = (
            previous_signature is not None and previous_signature[0] == signature[0]
        )
        current_entries = {item[0]: item for item in signature[1]}
        if not same_session:
            self._console_citation_counts = {}
            self._console_citation_resolved_signatures = {}
        else:
            cached_ids = set(self._console_citation_counts) | set(
                self._console_citation_resolved_signatures
            )
            for native_message_id in cached_ids:
                if self._console_citation_resolved_signatures.get(
                    native_message_id
                ) != current_entries.get(native_message_id):
                    self._console_citation_counts.pop(native_message_id, None)
                    self._console_citation_resolved_signatures.pop(
                        native_message_id,
                        None,
                    )

        self._console_citation_input_signature = signature
        self._console_citation_request_generation += 1
        generation = self._console_citation_request_generation
        unresolved = tuple(
            item
            for item in signature[1]
            if self._console_citation_resolved_signatures.get(item[0]) != item
            or item[0] not in self._console_citation_counts
        )
        if not unresolved:
            return
        self.run_worker(
            self._discover_console_citation_counts(
                repository,
                signature,
                generation,
                unresolved,
                repository_token,
            ),
            exclusive=True,
            group="console-citation-counts",
        )

    @staticmethod
    def _read_console_citation_counts(
        repository: Any,
        eligible: tuple[tuple[str, str, str, str], ...],
    ) -> dict[str, int]:
        """Read verified non-governed trace metadata into integer counts."""
        counts: dict[str, int] = {}
        for native_message_id, persisted_message_id, current_body, _status in eligible:
            counts[native_message_id] = 0
            try:
                result = repository.get_active_trace_for_current_message(
                    persisted_message_id,
                    current_body,
                )
                summary = getattr(result, "summary", None)
                if (
                    getattr(result, "state", None)
                    is not ActiveCitationTraceState.ACTIVE
                    or summary is None
                    or getattr(result, "availability_warning", None) is not None
                    or not repository.verify_active_trace_result(result)
                ):
                    continue
                evidence_ordinals = selected_valid_evidence_ordinals(summary.trace)
            except Exception:
                logger.exception(
                    "Unable to read Console citation count: "
                    "native_message_id={} persisted_message_id={}",
                    native_message_id,
                    persisted_message_id,
                )
                continue
            if evidence_ordinals:
                counts[native_message_id] = len(evidence_ordinals)
        return counts

    def _apply_console_citation_counts(
        self,
        signature: tuple[str | None, tuple[tuple[str, str, str, str], ...]],
        generation: int,
        counts: Mapping[str, int],
        eligible: tuple[tuple[str, str, str, str], ...] | None = None,
        repository_token: tuple[str, int, int, int] | None = None,
    ) -> bool:
        """Apply count-only results when their full captured input is current."""
        current_repository_token, current_repository = (
            self._console_citation_repository_readiness()
        )
        if (
            generation != self._console_citation_request_generation
            or signature != self._console_citation_input_signature
            or current_repository is None
            or repository_token != current_repository_token
            or signature
            != self._console_citation_signature(
                self._message._native_console_messages()
            )
        ):
            return False
        current_entries = {item[0]: item for item in signature[1]}
        for item in signature[1] if eligible is None else eligible:
            native_message_id = item[0]
            if current_entries.get(native_message_id) != item:
                continue
            count = counts.get(native_message_id, 0)
            self._console_citation_counts[native_message_id] = (
                count if type(count) is int and count >= 0 else 0
            )
            self._console_citation_resolved_signatures[native_message_id] = item
        return True

    async def _discover_console_citation_counts(
        self,
        repository: Any,
        signature: tuple[str | None, tuple[tuple[str, str, str, str], ...]],
        generation: int,
        eligible: tuple[tuple[str, str, str, str], ...] | None = None,
        repository_token: tuple[str, int, int, int] | None = None,
    ) -> None:
        """Discover citation footer counts off-loop and refresh current rows."""
        if repository_token is None:
            repository_token, current_repository = (
                self._console_citation_repository_readiness()
            )
            if current_repository is not repository:
                return
        queried = signature[1] if eligible is None else eligible
        counts = await asyncio.to_thread(
            self._read_console_citation_counts,
            repository,
            queried,
        )
        if not self._apply_console_citation_counts(
            signature,
            generation,
            counts,
            queried,
            repository_token,
        ):
            return
        await self._sync_native_console_chat_ui()

    def _native_console_transcript_fingerprint(
        self, messages: list[Any]
    ) -> tuple[Any, ...]:
        """Return a lightweight signature for native transcript refresh skipping."""
        store = self._ensure_console_chat_store()
        presentation_context = self._console_presentation_context()
        presentation_signature = (
            presentation_context.user_name,
            presentation_context.assistant_kind,
            presentation_context.character_name,
            presentation_context.transcript_style.value,
            presentation_context.revision,
        )
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
                    getattr(
                        getattr(message, "role", None),
                        "value",
                        getattr(message, "role", None),
                    ),
                    getattr(message, "content", ""),
                    getattr(message, "status", None),
                    getattr(message, "turn_id", None),
                    getattr(message, "persisted_message_id", None),
                    variant_signature,
                    getattr(message, "citation_presentation", None),
                )
            )
        return (
            store.active_session_id,
            tuple(message_signatures),
            presentation_signature,
        )

    async def _sync_native_console_transcript(self) -> None:
        """Render native Console messages in the native transcript."""
        try:
            transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        except QueryError:
            transcript = None

        active_messages = self._message._native_console_messages()
        messages = self._change_review_projection.project(active_messages)
        if region := self._console_transcript_region_or_none():
            region.sync_recovery()
        if transcript is not None:
            selected_id, selected_fork_eligibility = (
                self._message.sync_selected_fork_eligibility(transcript)
            )
            transcript.set_presentation_context(self._console_presentation_context())
            # Turn file card spec: keeps the mounted transcript's provider
            # factory current every tick -- late-bound so a session switch
            # or a bridge becoming available never needs a fresh instance.
            transcript.set_change_review_provider_factory(
                self._review_selection._console_change_review_provider
            )
            self._sync_console_citation_count_discovery(messages)
            message_ids = {message.id for message in messages}
            controller = self._console_chat_controller
            for message_id in tuple(self._console_original_attempt_previews):
                original_attempt = (
                    controller.original_attempt_for_message(message_id)
                    if controller is not None and message_id in message_ids
                    else None
                )
                if original_attempt is None:
                    self._console_original_attempt_previews.pop(message_id, None)
                else:
                    self._console_original_attempt_previews[message_id] = (
                        original_attempt
                    )
            # task-501: transfer a sibling-swipe selection handoff onto the
            # CURRENT transcript instance right before the push — the widget
            # applies it at ingest time once the landed sibling's id is in
            # the pushed set (see ConsoleTranscript.pending_selection_id).
            if self._pending_console_swipe_selection is not None:
                transcript.pending_selection_id = self._pending_console_swipe_selection
                self._pending_console_swipe_selection = None
            transcript.set_model_thinking_visible(self._show_model_thinking())
            transcript.set_messages(
                messages,
                session_id=self._ensure_console_chat_store().active_session_id,
            )
            # Live turn-activity line. `apply_turn_activity` hands back the
            # EFFECTIVE value ("" unless a row is actually in flight), which
            # is what joins the refresh key below -- so an idle transcript
            # can never repaint once a second off a stale run snapshot.
            turn_activity = transcript.apply_turn_activity(
                self._agent.console_turn_activity()
            )
            visible_citation_counts = {
                message_id: count
                for message_id, count in self._console_citation_counts.items()
                if type(count) is int and count > 0
            }
            transcript.set_citation_counts(visible_citation_counts)
            visible_library_activity_counts = self._library_activity.sync_transcript(
                transcript
            )
            transcript.set_original_attempt_previews(
                self._console_original_attempt_previews.copy()
            )
            # TASK-575: presentation consumes the same typed, branch-validated
            # selector result as provider dispatch. Transcript rows remain the
            # source of render identities; no stored summary field is inferred.
            store = self._ensure_console_chat_store()
            self._review_selection._sync_console_annotation_discovery(store)
            transcript.set_annotation_previews(self._console_annotation_previews)
            memory_banner = None
            if store.active_session_id is not None:
                try:
                    memory_controller = (
                        controller or self._ensure_console_chat_controller()
                    )
                    _global, _local, effective_memory = (
                        memory_controller.context_control_inputs(
                            store.active_session_id
                        )
                    )
                except Exception:
                    logger.warning(
                        "Console effective-memory presentation selection failed"
                    )
                else:
                    memory_banner = derive_console_memory_banner_presentation(
                        effective_memory,
                        active_messages,
                    )
            transcript.set_memory_banner_presentation(memory_banner)
            # TASK-371: reflect run state in the jump-to-latest pill when the
            # reader is scrolled up during / just after a streaming reply.
            transcript.sync_jump_indicator(self._current_console_run_status_value())
            image_specs = self._image._build_console_image_specs(messages)
            transcript.set_image_specs(image_specs)
            card_specs = self._image._build_generation_card_specs(messages)
            transcript.set_generation_card_specs(card_specs)
            video_specs = self._video._build_video_card_specs(messages)
            transcript.set_video_card_specs(video_specs)
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
                for mid, data in cache.pending_ids(
                    self._recent_console_image_messages(messages)
                )
                if mid not in self._console_image_preparing
            ]
            # Browsed generation-card variants share the same off-thread
            # prep worker (its pending list is already opaque
            # (cache-key, bytes) pairs) and the same in-flight guard set.
            pending_images.extend(
                (cache_key, data)
                for cache_key, data in self._image._pending_console_generation_card_images(
                    messages, card_specs
                )
                if cache_key not in self._console_image_preparing
            )
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
            # a view-mode toggle) would be skipped as a no-op refresh. The
            # generation-card signature covers the same case for cards
            # (browse/keep changes, mode toggles, and variant decode
            # completion all alter it -- see `generation_card_signature`).
            image_signature = tuple(
                (message_id, image_specs[message_id].mode)
                for message_id in sorted(image_specs)
            )
            card_signature = tuple(
                generation_card_signature(card_specs[message_id])
                for message_id in sorted(card_specs)
            )
            # Same load-bearing role as card_signature: a video's file
            # appearing or expiring flips the spec's status, and that alone
            # must force a refresh (the message set is otherwise unchanged).
            video_signature = tuple(
                video_card_signature(video_specs[message_id])
                for message_id in sorted(video_specs)
            )
            refresh_key = (
                id(transcript),
                self._native_console_transcript_fingerprint(messages),
                (selected_id, selected_fork_eligibility),
                image_signature,
                card_signature,
                video_signature,
                # A typed presentation change alone must repaint even when the
                # transcript message set itself did not change.
                memory_banner,
                turn_activity,
                tuple(sorted(self._console_original_attempt_previews.items())),
                tuple(sorted(visible_citation_counts.items())),
                tuple(sorted(visible_library_activity_counts.items())),
                # task-18515: an annotation added, edited, or deleted must
                # force a refresh on its own. Without this the marker row
                # only changed when something ELSE in the key did -- phase 4
                # looked correct because writing a note also dispatches a
                # message, while edit/delete leave the app idle and left a
                # deleted note's marker on screen (caught live).
                tuple(sorted(self._console_annotation_previews.items())),
                tuple(sorted(self._console_speech_states.items())),
            )
            if refresh_key != self._last_native_transcript_refresh_key:
                await transcript.refresh_messages()
                self._last_native_transcript_refresh_key = refresh_key
            self._sync_console_transcript_guidance()
            return

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
        store = self._console_chat_store
        session_id = store.active_session_id if store is not None else None
        image_edit = (
            self._image._h3_image_edit_registry().active(session_id)
            if session_id is not None
            else None
        )
        if image_edit is not None:
            return (
                "Stopping image edit…"
                if image_edit.cancel_event.is_set()
                else "Editing image…"
            )
        controller = self._console_chat_controller
        if controller is None:
            return ""
        run_state = controller.run_state
        if run_state.status is ConsoleRunStatus.IDLE:
            return ""
        return run_state.visible_copy or run_state.status.value

    def _console_active_run_copy(self) -> str:
        """Return the viewed session's active-run copy, or "" when not active.

        TASK-2154.18 (FB-08): unlike ``_native_run_status_copy`` -- which
        reports ANY non-IDLE status, including lingering terminal copy --
        this is gated on ``CONSOLE_ACTIVE_RUN_STATUSES``, the run chip's
        visibility contract. Falls back to the status value when a
        transition set no visible copy.
        """
        store = self._console_chat_store
        session_id = store.active_session_id if store is not None else None
        image_edit = (
            self._image._h3_image_edit_registry().active(session_id)
            if session_id is not None
            else None
        )
        if image_edit is not None:
            return (
                "Stopping image edit…"
                if image_edit.cancel_event.is_set()
                else "Editing image…"
            )
        controller = self._console_chat_controller
        run_state = controller.run_state if controller is not None else None
        if run_state is None or run_state.status not in CONSOLE_ACTIVE_RUN_STATUSES:
            return ""
        return run_state.visible_copy or run_state.status.value

    def _sync_console_mode_bar(self) -> None:
        try:
            mode_bar = self.query_one("#console-mode-bar", Static)
        except QueryError:
            return
        control_state = self._build_console_control_state(
            self._pending_console_launch_context
        )
        mode_copy = self._console_mode_summary(control_state)
        if run_status := self._native_run_status_copy():
            mode_copy = f"{mode_copy} | Run: {run_status}"
        mode_bar.update(mode_copy)
        # TASK-2154.18 (FB-08): the mode bar itself is a hidden compat
        # static -- the run chip in the status strip is run copy's
        # persistent VISIBLE home. This sync runs on send/stop
        # transitions and on every 0.2s transcript-poll tick while a run
        # is active, so the chip tracks VALIDATING -> STREAMING ->
        # terminal within a tick; the widget's own equality guard keeps
        # unchanged ticks free. Active statuses only: terminal outcomes
        # already toast (task-2154.16/.17) and mark the tab.
        try:
            status_chips = self.query_one("#console-status-chips", ConsoleStatusChips)
        except QueryError:
            return
        active_run_copy = self._console_active_run_copy()
        status_chips.sync_run_chip(bool(active_run_copy), active_run_copy)

    async def _sync_native_console_chat_ui(self) -> None:
        """Refresh visible Console-native state after send/stop transitions.

        **A torn-down screen renders nothing** (task-15860, cross-suite
        leak). This tick is a screen-owned `console-sync` worker, and
        Textual workers default to ``exit_on_error=True``: anything it
        raises reaches ``App._handle_exception`` and takes the whole TUI
        down. It also touches the DOM (``_sync_console_native_session_
        tabs`` -> ``ConsoleSessionSurface.sync_sessions`` ->
        ``query_one("#console-native-tab-strip")``), which is exactly what
        a navigation away from Console removes. Three guards here close
        every half of that: the entry check (a tick that STARTS after
        teardown), the teardown-scoped ``except`` (a tick that is
        mid-flight when teardown arrives), and the ``finally``'s re-arm
        check (a tick that would CREATE one of the first kind).

        Measured, not assumed: with a wake turn in flight, navigating away
        from Console had this tick's own ``finally`` re-arm (below)
        schedule a FRESH worker on the screen Textual had already closed;
        that worker ran a full sync against the removed surface, raised
        ``NoMatches``, and killed the app -- after which every later
        ``post_message`` was silently dropped and navigation was dead.
        """
        if _console_screen_is_torn_down(self):
            # The re-arm below is skipped for the same reason; clearing the
            # flag keeps a resurrected screen from inheriting this one's
            # coalesced request.
            self._console_sync_requested = False
            return
        if self._console_sync_in_progress:
            self._console_sync_requested = True
            return
        self._console_sync_in_progress = True
        self._record_ui_worker_started("console-sync")
        try:
            store = self._console_chat_store
            self._message.reconcile_console_speech_context()
            if store is not None:
                live_session_ids = {session.id for session in store.sessions()}
                registry = self._image._h3_image_edit_registry()
                prior_session_ids = getattr(self, "_console_h3_known_session_ids", None)
                if prior_session_ids is not None:
                    for missing_session_id in prior_session_ids - live_session_ids:
                        registry.drop_session(missing_session_id)
                self._console_h3_known_session_ids = live_session_ids
                self._image._reconcile_h3_image_edit_completions(store)
            self._sync_console_chat_core_state()
            self._session._sync_console_session_draft()
            # PR#757 review (comment 4): warm the effective-scope cache for
            # an already-active persisted session before anything below
            # reads it -- see `_warm_console_effective_scope_cache_if_stale`
            # docstring for why the picker/resume/flush warmers alone leave
            # a restore_state-reactivated session uncovered.
            await self._retrieval._warm_console_effective_scope_cache_if_stale()
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
            await (
                self._retrieval._refresh_active_dictionaries_summary_if_scope_changed()
            )
            await self._retrieval._refresh_active_world_books_summary_if_scope_changed()
            # P3c Task 4: mirrors the dictionary/world-book scope-guarded
            # refresh pattern immediately above -- safe to call unconditionally
            # on every sync tick because the refresh is itself scope-guarded
            # (no-op when the active character hasn't changed) and never
            # raises (see `_refresh_active_character_avatar_if_scope_changed`
            # docstring, T3).
            await self._character._refresh_active_character_avatar_if_scope_changed()
            # task-280: hand the control bar a pre-await snapshot (its own
            # pre-existing timing). The rail-VISIBILITY call below must NOT
            # reuse this snapshot: `_sync_console_native_session_tabs` can
            # create/activate a session, changing what the rail derivation
            # sees, and pre-task-280 the visibility check always computed
            # fresh post-await state (PR #660 review caught the reuse as a
            # staleness regression — the one-tuple-per-tick dedupe is
            # withdrawn for the visibility half).
            #
            # TASK-22201: the workspace-context builds of one tick (the rail
            # states here, the workspace-context push, the control bar's and
            # agent section's inspector legs — six per tick, measured) share
            # ONE fingerprint-validated build through this scope. The PR
            # #660 ruling is kept by MECHANISM rather than by always
            # rebuilding: a session created/activated across the awaits
            # changes the store fingerprint and the later reads rebuild,
            # while a settled tick pays for one build. Task-scoped: only
            # THIS coroutine's task reads the cache — workers and handlers
            # interleaving during the awaits keep building live.
            with self._workspace.tick_workspace_build_scope():
                rail_state = self._current_console_rail_state()
                self._sync_console_control_bar(rail_state)
                self._sync_console_settings_summary()
                # Settings failures may arrive after Apply has returned:
                # ordinary first persistence and temporary-chat promotion
                # both update the session ledger on their own later path.
                # Project that current-session truth during the existing
                # general sync so switches and delayed writes cannot leave
                # the mounted recovery rows stale. This is deliberately a
                # direct DOM sync: it starts no worker and emits no toast.
                self._sync_console_settings_recovery_surfaces()
                self._sync_console_mode_bar()
                await self._sync_console_native_session_tabs()
                self._dispatch_active_console_roleplay_refresh()
                self._sync_console_workspace_context()
                project_instruction_ui.sync_project_instruction_status_for_screen(self)
                await self._sync_native_console_transcript()
                self._sync_console_rail_visibility_if_changed(
                    self._current_console_rail_state()
                )
            self._dispatch_console_rail_preference_prune()
        except Exception:
            # Teardown-scoped ONLY. A tick that was mid-flight when the
            # screen was closed is querying widgets Textual has already
            # removed; that is not a defect to surface, and surfacing it
            # here means killing the app (this runs in a worker whose
            # `exit_on_error` is Textual's default True). Anything raised
            # by a LIVE screen still propagates untouched -- narrowing to
            # `NoMatches` would not be narrower in the way that matters,
            # since a torn-down DOM raises several different types
            # depending on how far the tick had got.
            if not _console_screen_is_torn_down(self):
                raise
            # fmt: off
            logger.debug(
                "Console sync tick raced this screen's teardown; nothing to "
                "render.",
            )
            # fmt: on
        finally:
            self._record_ui_worker_finished("console-sync")
            self._console_sync_in_progress = False
            if self._console_sync_requested:
                self._console_sync_requested = False
                # A dead screen must not re-arm itself: `run_worker` here
                # runs AFTER Textual's unmount sweep
                # (`Widget._on_unmount` -> `workers.cancel_node`), so the
                # worker it creates is never in the cancelled set and
                # outlives the screen that owns it.
                if not _console_screen_is_torn_down(self):
                    self.run_worker(
                        self._sync_native_console_chat_ui(),
                        exclusive=True,
                        group="console-sync",
                    )

    async def _sync_console_native_session_tabs(self) -> None:
        """Refresh native Console session tabs from store state."""
        try:
            surface = self.query_one("#console-session-surface", ConsoleSessionSurface)
        except QueryError:
            return
        store = self._ensure_console_chat_store()
        self._session._ensure_active_console_session_settings()
        controller = getattr(self, "_console_chat_controller", None)
        streaming_session_id = (
            controller.streaming_session_id() if controller is not None else None
        )
        sessions = store.sessions()
        # PR3a-2 Task 4: viewing IS the clear -- the active session's
        # conversation carrying the durable unseen-completion mark means
        # the user is now looking at the conversation the badge points to,
        # so the mark (and with it every surface it drives) is cleared
        # through the named seam. Guarded by the cached set first, so the
        # common no-mark tick costs no DB access.
        # Parallel-agents spec PA-T8: per-session fleet marker (RUNNING /
        # NEEDS_APPROVAL / FINISHED_OK / FINISHED_FAILED), superseding the
        # legacy single-session `streaming_session_id` cursor above for tabs
        # that have a controller -- `run_marker_for` already derives RUNNING
        # from the same live-busy definition `streaming_session_id` used, so
        # this is a strict superset, not a second notion of "in-flight".
        # PR3a-2 Task 4 threads the durable unseen-completion mark in as
        # the lowest-precedence marker (`_console_run_marker_with_unseen`).
        run_markers = self._fleet.prepare_session_run_markers(
            tuple(sessions),
            store.active_session_id,
        )
        queue_counts = (
            {
                session.id: controller.activity_for(session.id).queued_count
                for session in sessions
            }
            if controller is not None
            else None
        )
        self._maybe_show_fleet_coachmark(sessions, surface)
        await surface.sync_sessions(
            sessions=sessions,
            active_session_id=store.active_session_id,
            streaming_session_id=streaming_session_id,
            run_markers=run_markers,
            queue_counts=queue_counts,
        )

    async def _append_native_console_system_message(
        self, message: str, *, session_id: str | None = None
    ) -> None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        under the original name for its ~10 staying callers across other
        clusters and for tests that monkeypatch/call it directly."""
        await self._message._append_native_console_system_message(
            message, session_id=session_id
        )

    def _start_console_transcript_sync_timer(self) -> None:
        if self._console_transcript_sync_timer is not None:
            return

        async def _poll_transcript() -> None:
            await self._sync_native_console_chat_ui()
            controller = self._console_chat_controller
            if controller is None:
                self._workspace._invalidate_console_persisted_rows_cache()
                self._stop_console_transcript_sync_timer()
                return
            # Fix round 1 / Critical 1 (parallel-agents spec PA-T8 review):
            # `controller.run_state` is a read-only facade for the VIEWED
            # session ONLY (parallel-agents spec §2 -- see its docstring).
            # The old check looked at nothing else, so it self-stopped the
            # instant the viewed tab went idle even while a DIFFERENT
            # session was still streaming (or parked mid-`submit_draft`
            # awaiting an approval decision -- that session's own
            # end-of-run resync never separately fires, since it is still
            # inside the same await). Once stopped, `_sync_native_console_
            # chat_ui()` above never fires again, so tab glyphs and the
            # Agent-rail fleet line (both driven by that call) froze stale
            # until some unrelated event forced a manual resync.
            # `in_flight_run_count()` is the exact same live-busy
            # definition `run_marker_for`/`fleet_summary_counts` already
            # use for those glyphs/line, so gating the stop on it too is
            # not a new notion of "in-flight" -- it is the one this timer
            # exists to keep current. The persisted-rows-cache invalidate
            # stays coupled to the SAME combined condition as the stop
            # (not a bare "viewed session idle") so a long-running
            # background session cannot reintroduce the per-tick DB query
            # TASK-251's TTL cache exists to prevent; the resulting bound
            # on staleness is `CONSOLE_PERSISTED_ROWS_CACHE_TTL_SECONDS`
            # (2s), the documented backstop for exactly this gap.
            # task-15862: a wake delivery scheduled but not yet busy (the
            # coordinator's `_delivering` is set synchronously BEFORE its
            # asyncio task first runs) must not let a poll beat in that gap
            # self-stop -- the wake turn would then stream with no poll and
            # freeze exactly as before the delivery hook existed.
            wake = getattr(controller, "fleet_wake", None)
            delivering_read = getattr(wake, "delivering_conversation_id", None)
            wake_delivering = (
                callable(delivering_read) and delivering_read() is not None
            )
            review_coordinator = self._console_runtime().change_review_coordinator
            review_pending = (
                review_coordinator.publication_signal.snapshot().pending > 0
                if review_coordinator is not None
                else False
            )
            if (
                controller.run_state.status not in CONSOLE_ACTIVE_RUN_STATUSES
                and controller.in_flight_run_count() == 0
                and not wake_delivering
                and not review_pending
            ):
                # TASK-251: the run just left an active status -- invalidate
                # so the finalized conversation's title/timestamps appear in
                # the browser promptly instead of waiting out the TTL.
                self._workspace._invalidate_console_persisted_rows_cache()
                self._stop_console_transcript_sync_timer()
                # PR3a-2 Task 4 (task-15664): this stop edge is EXACTLY
                # where the UI used to go blind on a surviving sub-agent
                # -- no run occupies a slot, so this poll dies while a
                # child is still working and nothing repaints its elapsed
                # segment, the tab glyphs, or (on settle) the unseen
                # badge. Hand off to the 1s survivor tick, which runs only
                # while a drain is still owed and stops itself after one
                # final paint.
                self._fleet._maybe_start_console_fleet_survivor_tick()

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

    # -- PR3a-2 Task 4 (task-15664): the survivor tick ---------------------

    async def _submit_console_native_draft(
        self, draft: str, session_id: str | None = None
    ) -> None:
        controller = self._ensure_console_chat_controller()
        self._start_console_transcript_sync_timer()
        # Task 3b: `session_id` is the session THIS worker was dispatched
        # for (`_dispatch_console_draft_send` already resolved it via the
        # `console-run-{session_id}` group). Defaulted to the currently
        # active session only for direct-call test idioms that predate the
        # per-session stash map -- equivalent to the old singular-slot
        # behavior for the (overwhelmingly common) single-session case.
        if session_id is None:
            session_id = controller.store.active_session_id or ""
        dispatch_composer = self._console_composer_or_none()
        dispatch_draft_revision = (
            (
                dispatch_composer.edit_serial,
                dispatch_composer.capture_draft_snapshot().generation,
            )
            if dispatch_composer is not None
            and self._console_visible_draft_session_id == session_id
            else None
        )
        task = asyncio.current_task()
        if task is not None:
            # See `_on_console_submission_accepted`: it fires synchronously
            # from deep inside the `submit_draft` await below, on this SAME
            # task, and has no session id of its own to key by.
            self._console_submit_session_by_task[task] = session_id
        # TASK-340: a keyboard send already cleared the composer at the Enter
        # keypress. The accepted-hook consumes this slot; a refusal below
        # restores it instead. Snapshot before submit_draft so the hook's
        # consumption is observable here.
        inflight_stash = self._console_inflight_send_stashes.get(session_id)
        try:
            # F4 fix (Qodo wave): thread the session THIS worker was
            # dispatched for all the way into the controller -- previously
            # `submit_draft` re-resolved "the session to submit into" via
            # `store.active_session_id` at execution time, so a tab switch
            # racing the scheduling gap between `run_worker(...)` and this
            # coroutine body actually running could submit into whichever
            # session the user switched TO instead of the dispatching one.
            result = await controller.run_prompt_chain(draft, session_id=session_id)
        except Exception:
            # An unexpected submit crash must not eat the keypress-cleared
            # draft — and must not escape the worker (exit_on_error would
            # take the whole app down with it).
            leaked_stash = self._console_inflight_send_stashes.pop(session_id, None)
            if leaked_stash is not None:
                self._restore_console_send_stash(leaked_stash)
            logger.exception("Console submit failed unexpectedly")
            self.app_instance.notify(
                "Console send failed unexpectedly — your draft was restored.",
                severity="error",
            )
            return
        finally:
            if task is not None:
                self._console_submit_session_by_task.pop(task, None)
        # TASK-251: a submit may have created/updated a persisted
        # conversation (title, updated_at) -- invalidate so the browser
        # reflects it on the very next sync instead of the TTL window.
        self._workspace._invalidate_console_persisted_rows_cache()
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            composer = None
        # Task 3b: only the composer that STILL SHOWS this session gets
        # mutated on its behalf. A background session's dispatch can
        # complete long after the user switched away -- restoring an
        # abandoned draft (or clearing should_clear_draft below) into
        # whatever composer happens to be visible would leak this
        # session's text into a DIFFERENT session's tab.
        composer_reflects_session = (
            composer is not None and controller.store.active_session_id == session_id
        )
        # TASK-1281 review NEW-5: `clear_draft`/`clear_history` below must
        # only ever touch the composer when it PROVABLY shows this exact
        # session's draft right now, not merely when the store's active
        # session id happens to match -- `composer_reflects_session` above
        # is Task 3b's pre-existing (looser) check, kept as-is for
        # `restore_stashed_draft` below, but during the TASK-339
        # session-switch settle window `active_session_id` can already
        # equal `session_id` while the composer still visibly shows a
        # DIFFERENT session (see F1) -- clearing on that weaker guard would
        # wipe the wrong session's on-screen draft. Unified with
        # `_on_console_submission_accepted`'s own guard shape.
        composer_visible_for_session = (
            composer is not None
            and self._console_visible_draft_session_id == session_id
        )
        stash = self._console_inflight_send_stashes.pop(session_id, None)
        if not result.accepted and stash is not None and composer_reflects_session:
            # Controller-level refusal of a keyboard send: the composer was
            # cleared at the keypress, so hand the draft back (ahead of any
            # keystrokes typed since).
            composer.restore_stashed_draft(stash)
        if result.session_closed:
            # Task 4 (D2 fix wave): `_session_closed_result` is `accepted`
            # (see its own docstring) so the restore above never fires, and
            # its owning session no longer exists to hold a SYSTEM row --
            # there is nothing left to write into and nowhere to restore a
            # keypress-cleared draft TO. A toast is the one surface still
            # available: without it this outcome was completely silent
            # (composer already cleared, no row, no notification).
            # Fix-round-2 (I2/M2): `session_closed` is now set ONLY at the
            # dispatch-gap call site (the OTHER ~19 `_session_closed_result`
            # sites -- mid-run closes the user already confirmed -- leave it
            # `False`), and that ONE site's `visible_copy` is always the
            # informative "...before your message could send." string, not
            # the generic "Session closed." every other site uses -- so
            # `result.visible_copy` is used directly, with no dead fallback.
            self.app_instance.notify(result.visible_copy, severity="warning")
        if (
            result.should_clear_draft
            and composer_visible_for_session
            and inflight_stash is None
            and (
                dispatch_draft_revision is None
                or (
                    composer.edit_serial,
                    composer.capture_draft_snapshot().generation,
                )
                == dispatch_draft_revision
            )
        ):
            # Stashed sends were cleared at the keypress — clearing again
            # here would eat keystrokes typed after Enter (the next draft).
            composer.clear_draft()
            # TASK-1281 review F2: send is a history barrier -- see
            # `_on_console_submission_accepted`'s identical comment. This
            # site covers the same "content is genuinely gone" moment for
            # sends that reach here without an inflight keypress stash
            # (e.g. the mouse-click Send path).
            composer.clear_history()
            self._sync_console_command_popup()
        if result.accepted:
            # TASK-1281 review NEW-5: only an ACCEPTED send makes this
            # session's pre-send history genuinely stale -- a refusal
            # (blocked/failed/canceled) sent nothing, so a background
            # session's banked undo/redo history must survive it exactly
            # as it would have survived never attempting the send at all.
            self._console_undo_histories.pop(session_id, None)
        if (
            result.accepted
            and controller.run_state.status is ConsoleRunStatus.COMPLETED
        ):
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
        ``ConsoleChatController.submit_draft`` invokes this hook only once
        its own skill-substitution/trust re-check has confirmed the turn
        actually proceeds (Qodo finding 3, PR #636 bot review) -- a
        substitution refusal, like any other blocked submit, never reaches
        it, so a refused draft stays in the composer too.

        Task 3b: this fires synchronously from deep inside ``submit_draft``,
        on the SAME task as the ``_submit_console_native_draft`` worker that
        awaited it -- ``_console_submit_session_by_task`` resolves which
        session's stash entry (if any) is this call's own, without changing
        this hook's public no-arg ``Callable[[], None]`` contract (still
        assignable via ``controller.on_submission_accepted = ...`` exactly
        as before). A lookup miss (direct-call test idioms, or no wrapping
        task) falls back to the active session -- the pre-Task-3b behavior.
        """
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            composer = None
        task = asyncio.current_task()
        session_id = (
            self._console_submit_session_by_task.get(task) if task is not None else None
        )
        active_session_id = self._ensure_console_chat_store().active_session_id or ""
        if session_id is None:
            session_id = active_session_id
        if session_id in self._console_inflight_send_stashes:
            # TASK-340: this submit's draft was captured and cleared at the
            # Enter keypress — clearing now would eat keystrokes typed since
            # (they are the NEXT draft). Consume the stash instead.
            self._console_inflight_send_stashes.pop(session_id, None)
        elif composer is not None and active_session_id == session_id:
            composer.clear_draft()
            self._sync_console_command_popup()
        # TASK-1281 review F2: this hook fires ONLY once submit_draft has
        # confirmed the turn actually proceeds (never for a blocked/refused
        # send -- see the docstring above), so every call here represents a
        # draft that is genuinely, irrevocably gone. Clearing just the
        # draft text (above) is not enough: the mutations that PRODUCED it
        # stay reachable on the undo stack either way (a `clear_draft()`
        # with no `record_history=True` records nothing, so it doesn't
        # cover them), and Ctrl+Z would resurrect already-sent content back
        # into the composer -- and, via the undo/redo re-persist, right
        # back into the store as the "live" draft for a message that has
        # already shipped. Drops the banked history unconditionally (a sent
        # session can never be usefully switched back into with anything
        # from before the send), and the composer's own live stacks too
        # when it still shows this exact session.
        self._console_undo_histories.pop(session_id, None)
        if (
            composer is not None
            and self._console_visible_draft_session_id == session_id
        ):
            composer.clear_history()
        # A send can finish while navigation is tearing this screen down. Do
        # not create a coroutine that Textual will reject after unmounting;
        # the next mounted view rebuilds from the durable chat store.
        if not self.is_mounted:
            return
        # task-351(a): echo the just-appended USER message immediately rather
        # than waiting up to a full 0.2s transcript-poll cycle (and a heavy
        # first poll after it). The composer clears here at acceptance, so
        # without this the transcript still read "No messages yet" for ~600ms
        # after the text vanished — reading as "not sent". This hook only fires
        # once submit_draft has confirmed the turn actually proceeds (never for
        # a blocked/refused send), so the USER row is already in the store.
        # `_sync_native_console_chat_ui` coalesces against a running sync via
        # its own `_console_sync_in_progress`/`_console_sync_requested` guard
        # (a concurrent call sets "requested" and the in-progress run re-fires
        # from its `finally`), so the echo still lands. NOT `exclusive=True`:
        # that would CANCEL a console-sync worker mid-flight, and a sync
        # cancelled after it advanced a scope sentinel but before its awaited
        # refresh completed would leave inspector/summary caches stale until the
        # scope next changes (Qodo #2). Coalescing gives the echo without that
        # cancellation. `exit_on_error=False`: best-effort acknowledgment — if
        # the screen is tearing down (or a send races a navigation away) the
        # sync can hit a removed widget and raise `NoMatches`; the poll runs the
        # same coroutine from a timer whose exceptions Textual already absorbs,
        # so a transient failure here must likewise never crash the app (default
        # `exit_on_error=True` would) — the next poll re-renders regardless.
        self.run_worker(
            self._sync_native_console_chat_ui(),
            group="console-sync",
            exit_on_error=False,
        )

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
                    "Console send blocked: Library search has no available evidence. "
                    "Review source authority before sending."
                )
        _readiness_settings, readiness = self._active_console_settings_readiness()
        if not readiness.native_send_supported:
            return f"Console send blocked: {readiness.detail}"
        attachment_reason = self._console_attachment_blocked_reason()
        if attachment_reason:
            return attachment_reason
        return ""

    async def handle_console_send_message(self, event: Button.Pressed) -> bool:
        """Route the Console composer send action through the native controller.

        Args:
            event: The Send button press. Stopped here, so a synthesized
                `Button.Pressed` from a programmatic caller behaves the same
                as a real one.

        Returns:
            Whether the draft was actually queued as a user turn. The button
            path discards this; the spoken-command path (`Console, send.`)
            needs it, because every refusal below returns without sending and
            an ack that says otherwise is simply wrong.
        """
        event.stop()
        return await self._send_console_message_from_visible_action()

    async def _send_console_message_from_visible_action(self) -> bool:
        """Route the visible Console send action through the native controller.

        Returns:
            True once the draft has been queued as a user turn; False on every
            refusal -- an empty draft with no attachment, a `/`-command or
            unknown-command dispatch (which never sends by design), and every
            gate inside `_dispatch_console_draft_send`. Each refusal has
            already shown its own toast or system row.
        """
        # TASK-340: a keyboard send captured its payload at the Enter
        # keypress; the mouse path still reads the live draft here.
        stash = self._console_pending_send_stash
        self._console_pending_send_stash = None
        stash, composer, draft, raw_cli_handled = raw_cli_ui.prepare_visible_send(
            stash, self._console_composer_or_none, self._raw_cli.start_user_command
        )
        if raw_cli_handled:
            return False
        if not draft.strip() and self._console_pending_image_attachment() is None:
            if composer is not None:
                composer.restore_stashed_draft(stash)
            self._focus_console_composer_if_needed(force=True)
            return False
        self._dismiss_console_guidance()

        # Command parsing runs before any readiness/blocked gating: a
        # recognized command dispatch (or an unknown-command hint) never
        # sends, so it must work even while Send is blocked. Draft text
        # carrying any real paste-originated segment (regardless of its
        # current collapse/confirm/expanded display state) is never treated
        # as command input -- Task 9's grammar module deliberately leaves
        # that gating to the caller, since only the composer knows the real
        # segment state.
        has_paste = (
            stash.has_paste
            if stash is not None
            else (composer is not None and composer.has_paste_segments())
        )
        if composer is not None and not has_paste:
            parse = self._console_command_registry.parse(draft)
        else:
            parse = CommandParse(kind=KIND_NOT_COMMAND)

        argument_free_rewind = (
            parse.kind == KIND_COMMAND
            and parse.name == REWIND_COMMAND_NAME
            and parse.args == ""
        )
        if argument_free_rewind:
            self._console_unknown_send_armed = None
            opening_composer = composer if stash is None else None
            opening_revision = None
            if opening_composer is not None:
                opening_revision = (
                    opening_composer.edit_serial,
                    opening_composer.capture_draft_snapshot().generation,
                    draft,
                )
            opened = False
            try:
                opened = await self._console_command_rewind(parse)
            finally:
                if not opened and composer is not None:
                    composer.restore_stashed_draft(stash)
            if opened and opening_composer is not None and opening_revision is not None:
                current = self._console_composer_or_none()
                current_snapshot = (
                    current.capture_draft_snapshot()
                    if current is opening_composer
                    else None
                )
                if (
                    current is opening_composer
                    and current.edit_serial == opening_revision[0]
                    and current_snapshot is not None
                    and current_snapshot.generation == opening_revision[1]
                    and current.draft_text() == opening_revision[2]
                ):
                    self._clear_console_composer_draft()
            return False

        if parse.kind == KIND_COMMAND:
            # Commands operate on the live composer draft (`/prompt` replaces
            # it wholesale, unrecognized handlers leave it untouched) — put
            # the stash back first so their semantics stay identical.
            if composer is not None:
                composer.restore_stashed_draft(stash)
            self._console_unknown_send_armed = None
            await self._dispatch_console_command(parse)
            return False

        if parse.kind == KIND_UNKNOWN:
            # Fold-in (Task 9 fix-wave review; hard removal Task 4 -- there
            # is no fallback resolver at all anymore, so EVERY unmatched
            # `/word` reaches here as KIND_UNKNOWN): a typed `/name` that
            # matches ONLY needs-review (trust-blocked) skills would
            # otherwise fall through to the generic "Unknown command" hint
            # just like any other unrecognized word. Checking against a
            # FRESH context surfaces the same needs-review response instead,
            # before the unknown-command arm/hint logic ever runs. This
            # never arms the unknown-command escape: a blocked match is a
            # known-but-blocked command, not an unrecognized one, so a
            # repeated Enter shows the same response again rather than
            # silently falling through to a literal send.
            context = await self._skill._fetch_console_skill_context()
            blocked_summaries = self._skill._console_skill_blocked_summaries(context)
            if await self._skill._console_skill_blocked_match_response(
                parse.name, blocked_summaries
            ):
                if composer is not None:
                    composer.restore_stashed_draft(stash)
                return False
            if self._console_unknown_send_armed == draft:
                # Second consecutive Enter on the *same* unmodified draft:
                # disarm and fall through to a normal send below.
                self._console_unknown_send_armed = None
            else:
                self._console_unknown_send_armed = draft
                if composer is not None:
                    composer.restore_stashed_draft(stash)
                await self._append_native_console_system_message(
                    self._console_unknown_command_hint(parse.name)
                )
                return False

        return await self._dispatch_console_draft_send(draft, stash=stash)

    async def _dispatch_console_draft_send(
        self, draft: str, stash: "ConsoleDraftStash | None" = None
    ) -> bool:
        """Compatibility delegate for the one typed queue-aware dispatcher."""

        result = await self._prompt_queue.dispatch(draft, stash=stash)
        return result.status is not ConsolePromptDispatchStatus.REFUSED

    def _note_console_follow_intent(self) -> None:
        """Stamp a programmatic jump-to-tail intent on the transcript (TASK-336).

        Delegates to ``ConsoleTranscriptRegion.note_follow_intent`` (wave-3
        task 2), which carries the task-3b audit note about why this stays
        singular/view-only. Kept as a screen method because the session and
        workspace controllers are both wired to it by this name
        (``note_follow_intent=lambda: self._note_console_follow_intent()``).
        """
        region = self._console_transcript_region_or_none()
        if region is not None:
            region.note_follow_intent()

    def _restore_console_send_stash(self, stash: "ConsoleDraftStash | None") -> None:
        """Hand a keypress-captured draft back to the composer (TASK-340)."""
        if stash is None:
            return
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        composer.restore_stashed_draft(stash)

    _CONSOLE_COMMAND_NAME_TO_HANDLER_ID = {
        PROMPT_COMMAND_NAME: PROMPT_COMMAND_HANDLER_ID,
        SYSTEM_COMMAND_NAME: SYSTEM_COMMAND_HANDLER_ID,
        SKILLS_COMMAND_NAME: SKILLS_COMMAND_HANDLER_ID,
        FEWER_PERMISSION_PROMPTS_COMMAND_NAME: (
            FEWER_PERMISSION_PROMPTS_COMMAND_HANDLER_ID
        ),
        PREFILL_COMMAND_NAME: PREFILL_COMMAND_HANDLER_ID,
        GENERATE_IMAGE_COMMAND_NAME: GENERATE_IMAGE_COMMAND_HANDLER_ID,
        GENERATE_VIDEO_COMMAND_NAME: GENERATE_VIDEO_COMMAND_HANDLER_ID,
        STREAM_VIDEO_COMMAND_NAME: STREAM_VIDEO_COMMAND_HANDLER_ID,
        REWIND_COMMAND_NAME: REWIND_COMMAND_HANDLER_ID,
        RESEARCH_COMMAND_NAME: RESEARCH_COMMAND_HANDLER_ID,
    }

    def _console_unknown_command_hint(self, name: str) -> str:
        """Return the Enter-again hint copy for an unrecognized `/name` draft.

        Derived from the registry's own ``available_names()`` (Task 9) rather
        than a hardcoded list, so a newly-registered command (e.g. `/skills`)
        is reflected here automatically.
        """
        available = ", ".join(
            f"/{name}" for name in self._console_command_registry.available_names()
        )
        return f"Unknown command /{name} — available: {available}. Press Enter again to send as text."

    async def _dispatch_console_command(self, parse: CommandParse) -> None:
        """Dispatch a parsed Console slash command to its handler.

        A ``handler_id`` that resolves to nothing (an unrecognized command
        name) is consumed silently: nothing is sent and the draft is left
        untouched.
        """
        handler_id = self._CONSOLE_COMMAND_NAME_TO_HANDLER_ID.get(parse.name)
        # F5 (task-9 review): the composer-menu entry that BUILDS a
        # /generate-image draft was gated, but typing (or pasting) the
        # command itself was not -- this is the actual choke point every
        # path to running it passes through, so gating here closes all of
        # them at once rather than chasing each way a draft gets composed.
        if handler_id == GENERATE_IMAGE_COMMAND_HANDLER_ID:
            image_blocked = blocked_reason(
                GENERATE_IMAGE_COMMAND_HANDLER_ID,
                ephemeral=self._console_active_session_is_ephemeral(),
            )
            if image_blocked is not None:
                await self._append_native_console_system_message(image_blocked)
                return
        # task-3401.5: the video twin of the image gate above -- typing or
        # pasting /generate-video hits the same choke point, so a temporary
        # chat cannot reach the disk-writing sink through the command either.
        if handler_id == GENERATE_VIDEO_COMMAND_HANDLER_ID:
            video_blocked = blocked_reason(
                GENERATE_VIDEO_COMMAND_HANDLER_ID,
                ephemeral=self._console_active_session_is_ephemeral(),
            )
            if video_blocked is not None:
                await self._append_native_console_system_message(video_blocked)
                return
        dispatch_map = {
            "insert-prompt": self._console_command_insert_prompt,
            "apply-system": self._console_command_apply_system,
            SKILLS_COMMAND_HANDLER_ID: self._console_command_skills,
            FEWER_PERMISSION_PROMPTS_COMMAND_HANDLER_ID: (
                self._console_command_fewer_permission_prompts
            ),
            PREFILL_COMMAND_HANDLER_ID: self._console_command_prefill,
            GENERATE_IMAGE_COMMAND_HANDLER_ID: self._console_command_generate_image,
            GENERATE_VIDEO_COMMAND_HANDLER_ID: self._console_command_generate_video,
            STREAM_VIDEO_COMMAND_HANDLER_ID: self._console_command_stream_video,
            REWIND_COMMAND_HANDLER_ID: self._console_command_rewind,
            RESEARCH_COMMAND_HANDLER_ID: self._console_command_research,
        }
        handler = dispatch_map.get(handler_id)
        if handler is None:
            return
        await handler(parse)

    async def _console_command_insert_prompt(self, parse: CommandParse) -> None:
        """Delegate to `ConsolePromptsController` (wave-3 console decomposition, task 3)."""
        await self._prompts._console_command_insert_prompt(parse)

    async def _open_console_style_picker_for_insert(self) -> None:
        """Open the image-style picker, inserting whatever style is chosen."""

        def _apply_picker_choice(record: Optional[Mapping[str, Any]]) -> None:
            self._focus_console_composer_if_needed(force=True)
            if record is None:
                return
            style_id = str(record.get("id") or "").strip()
            if not style_id:
                return
            self._insert_console_style_token_into_composer(style_id)

        self.app.push_screen(ConsoleStylePickerModal(), callback=_apply_picker_choice)

    def _insert_console_style_token_into_composer(self, style_id: str) -> bool:
        """Compose an ``@<style_id>`` token into a valid `/generate-image` draft.

        Delegates the actual composition to `insert_style_token_into_draft`
        (the pure grammar-aware helper `/generate-image`'s own parser is
        built from) rather than blindly prepending the token: a bare
        prepend would land the token BEFORE the command word on an
        unedited draft like ``/generate-image a dragon``, producing
        ``@style_anime /generate-image a dragon`` -- text `parse_generate_
        image_args` never sees as a command at all (`ConsoleCommandRegistry
        .parse` only recognizes drafts that START with `/`), so the whole
        thing would ship to the LLM as plain chat text instead of
        generating anything.

        The whole draft is replaced wholesale with the composed result --
        same clear-then-paste idiom `_insert_prompt_text_into_composer`
        uses for `replace=True` -- since the composed draft may reorder or
        drop text relative to the original (e.g. replacing an existing
        leading `@style` token), so a plain in-place insert cannot express
        it.

        Args:
            style_id: The resolved style template's `id` (e.g. "style_anime").

        Returns:
            ``True`` when the composer widget was found and the insert
            applied, ``False`` when no native composer is mounted.
        """
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return False
        new_draft = insert_style_token_into_draft(composer.draft_text(), style_id)
        composer.clear_draft()
        composer.insert_text_as_paste(new_draft)
        return True

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
            composer.insert_text_as_paste(text)
        elif composer.draft_text():
            # Appending onto an existing draft must never mash the two
            # payloads together with no boundary between them. The composer
            # caret is editable now, so seek the end first to keep this an
            # append rather than a mid-draft splice. TASK-1281 review N1:
            # the separator and the body are inserted as ONE
            # `insert_text_as_paste` call (rather than a separate
            # `insert_text("\n")` followed by the paste) so they also
            # record as a single undo entry -- previously one Ctrl+Z only
            # removed the pasted body and left a stray blank line behind.
            #
            # Review NEW-4 (known, deliberately unfixed): when `text` is
            # long enough to collapse, the leading "\n" lands INSIDE that
            # one collapsed segment, so the collapsed token itself carries
            # no visible boundary marker between "existing draft" and the
            # pasted body (canonical text is still correct either way).
            # Splitting this back into two calls to restore a literal,
            # always-visible separator would reintroduce the exact N1 bug
            # this comment describes (two undo entries, a stray blank line
            # surviving one Ctrl+Z) -- not a trivial fix, so left as a
            # documented display-only limitation rather than reverted.
            composer.move_cursor_end()
            composer.insert_text_as_paste(f"\n{text}")
        else:
            composer.insert_text_as_paste(text)
        self._sync_console_command_popup()
        return True

    async def _consume_pending_console_prompt_insert(self) -> None:
        """Delegate to `ConsolePromptsController` (wave-3 console decomposition, task 3)."""
        await self._prompts._consume_pending_console_prompt_insert()

    async def _console_command_apply_system(self, parse: CommandParse) -> None:
        """Delegate to `ConsolePromptsController` (wave-3 console decomposition, task 3)."""
        await self._prompts._console_command_apply_system(parse)

    async def _console_command_prefill(self, parse: CommandParse) -> None:
        """Set, pin, clear, or report the Console response prefill (`/prefill`).

        One-shot (`/prefill <text>`) applies to the next normal send only
        and wins over pinned; `/prefill pin <text>` applies to every
        submit/retry/regenerate until cleared and write-throughs to
        conversation metadata when the session is persisted. Errors leave
        the draft in place for correction (mirrors `/system`'s
        no-system-part behavior); handled outcomes clear it.
        """
        action = parse_prefill_args(parse.args)
        store = self._ensure_console_chat_store()
        self._session._ensure_active_console_session_settings()
        session = store.ensure_session()
        if session.settings is None:
            # `ensure_session` only applies `settings=` when it CREATES the
            # session; one created earlier without settings (e.g. by a bare
            # system-message append) would make the pinned-prefill update a
            # silent no-op in `set_session_pinned_prefill` (PR #729 Qodo
            # finding 3), so seed defaults before any pin/clear below.
            session = store.replace_session_settings(
                session.id, self._session._default_console_session_settings()
            )
        if action.kind == ACTION_ERROR:
            await self._append_native_console_system_message(action.error)
            return
        if action.kind == ACTION_STATUS:
            one_shot = store.session_one_shot_prefill(session.id)
            settings = store.session_settings(session.id)
            pinned = getattr(settings, "pinned_prefill", None) if settings else None
            lines = []
            if one_shot:
                lines.append(
                    f"Prefill (next send only): '{describe_prefill_preview(one_shot)}'"
                )
            if pinned:
                lines.append(f"Prefill (pinned): '{describe_prefill_preview(pinned)}'")
            if not lines:
                lines.append("No prefill armed.")
            # Clear before the message-append await: `_append_native_console_
            # system_message` triggers nested syncs (transcript render among
            # them) that make the confirmation visible to a polling caller
            # before this coroutine resumes past the `await` -- clearing
            # first closes that window instead of racing it.
            self._clear_console_composer_draft()
            await self._append_native_console_system_message("\n".join(lines))
            return
        if action.kind == ACTION_CLEAR:
            store.set_session_one_shot_prefill(session.id, None)
            _session, persisted = store.set_session_pinned_prefill(session.id, None)
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            copy = "Prefill cleared."
            if not persisted:
                copy += " (Warning: saved conversation not updated.)"
            self._clear_console_composer_draft()
            await self._append_native_console_system_message(copy)
            return
        # Deliberately NOT markup-escaped: native Console transcript rows
        # render as literal Content (never parsed as Rich markup), so an
        # escape here would surface as a visible backslash in previews
        # containing '[' — matching every neighboring system-row handler.
        preview = describe_prefill_preview(action.text)
        if action.kind == ACTION_PIN:
            _session, persisted = store.set_session_pinned_prefill(
                session.id, action.text
            )
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            copy = (
                f"Prefill pinned: '{preview}'. Applies to every send, retry, and "
                "regenerate until /prefill clear. The reply continues directly "
                "from the last character; tool calling is skipped on prefilled sends."
            )
            if not persisted:
                copy += " (Warning: saved conversation not updated.)"
            self._clear_console_composer_draft()
            await self._append_native_console_system_message(copy)
            return
        if action.kind == ACTION_ONE_SHOT:
            store.set_session_one_shot_prefill(session.id, action.text)
            self._sync_console_chat_core_state()
            self._sync_console_settings_summary()
            self._clear_console_composer_draft()
            await self._append_native_console_system_message(
                f"Prefill armed for next send: '{preview}'. The reply continues "
                "directly from the last character; tool calling is skipped on "
                "prefilled sends."
            )
        return

    async def _console_command_research(self, parse: CommandParse) -> None:
        """``/research <question>``: launch a local deep-research run whose
        completed report is delivered back into THIS conversation (task-16481).

        The run executes in a worker; the handoff inserts an assistant
        message on completion, and the existing terminal-run notification
        remains the fallback when insertion is impossible.
        """
        from tldw_chatbook.UI.Console_Modules.research_command import (
            parse_research_command,
        )

        try:
            intent = parse_research_command(parse.args or "")
        except ValueError as usage_error:
            await self._append_native_console_system_message(
                f"/research: {usage_error}"
            )
            return
        question = intent.question
        source_policy = intent.source_policy
        provider_overrides = intent.provider_overrides()
        conversation_id = self._current_console_conversation_id()
        if not conversation_id:
            await self._append_native_console_system_message(
                "Deep research needs an active conversation to deliver its report into."
            )
            return
        app = self.app
        local_service = getattr(app, "local_research_service", None)
        if local_service is None:
            await self._append_native_console_system_message(
                "Local research service is unavailable; cannot start a run."
            )
            return

        from tldw_chatbook.Research_Interop.chat_handoff import (
            insert_research_completion_message,
        )
        from tldw_chatbook.Research_Interop.local_research_engine import (
            LocalResearchEngine,
        )

        search_params: dict = {}
        paper_search_fn = None
        try:
            from tldw_chatbook.Tools.web_tool_impls import deep_search_pipeline_params

            search_params = deep_search_pipeline_params()
            if getattr(app, "research_window_academic_enabled", False):
                from tldw_chatbook.Research_Interop.academic_providers import (
                    search_papers,
                )

                paper_search_fn = search_papers
        except Exception:
            pass

        db = getattr(app, "chachanotes_db", None)

        async def _run_research() -> None:
            launch_kwargs: dict = {
                "query": question,
                "chat_handoff": {
                    "conversation_id": conversation_id,
                    "origin": "console",
                },
                "source_policy": source_policy,
            }
            if provider_overrides:
                launch_kwargs["provider_overrides"] = provider_overrides
            engine = LocalResearchEngine(
                local_service,
                search_params=search_params,
                paper_search_fn=paper_search_fn,
                completion_handoff=(
                    (lambda payload: insert_research_completion_message(db, payload))
                    if db is not None
                    else None
                ),
            )
            try:
                # TASK-21105 review fix: launch_run is the store's FIRST USE
                # now that the research DB opens lazily, so a corrupt or
                # unreadable database raises HERE rather than at app boot
                # (where construction failure used to null the service and
                # trip the unavailable-guard above). It must sit inside the
                # worker's guard: this worker runs with Textual's default
                # exit_on_error=True, so an unhandled raise tears down the
                # whole app.
                run = local_service.launch_run(**launch_kwargs)
                await engine.execute_run(run["id"])
            except Exception as exc:  # noqa: BLE001 - worker must not crash the screen
                logger.warning(f"Console research run failed: {exc}")

        self.run_worker(
            _run_research(),
            group="console-research",
            exclusive=False,
            description=f"Console research: {question[:60]}",
        )
        policy_note = (
            f" [policy: {source_policy}]" if source_policy != "balanced" else ""
        )
        await self._append_native_console_system_message(
            f"Deep research started: {question}{policy_note}\n"
            "The report will be added to this conversation when the run "
            "completes."
        )

    async def _console_command_generate_image(self, parse: CommandParse) -> None:
        """Delegate the registry-bound image command to its controller."""
        await self._image._console_command_generate_image(parse)

    # -- /generate-video (task-3401.5) --------------------------------------

    async def _console_command_generate_video(self, parse: CommandParse) -> None:
        """Delegate the registry-bound video command to its controller."""
        await self._video._console_command_generate_video(parse)

    async def _wait_for_console_screen_result(self, screen) -> Any:
        """Wait for a Console modal through a non-exclusive Textual worker."""
        worker = self.run_worker(
            self.app_instance.push_screen_wait(screen),
            exclusive=False,
            exit_on_error=False,
        )
        return await worker.wait()

    def _open_video_with_os(path: Path) -> None:
        """Launch a video path with the platform default player."""
        import subprocess
        import sys

        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])  # nosec B603
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]  # nosec B606
        else:
            subprocess.Popen(["xdg-open", str(path)])  # nosec B603

    async def _console_command_stream_video(self, parse: CommandParse) -> None:
        """Delegate the registry-bound stream command to its controller."""
        await self._video._console_command_stream_video(parse)

    # Preview length used to build `/rewind` menu rows -- collapses the
    # prompt to one line and truncates it (with a trailing ellipsis) via the
    # same helper session titles use.
    _CONSOLE_REWIND_PREVIEW_MAX_LENGTH = 60

    def _console_rewind_prompt_rows(
        self, session_id: str
    ) -> tuple[RewindPromptRow, ...]:
        """Build newest-first `/rewind` menu rows for a session's USER turns.

        Args:
            session_id: Native Console session id whose active-path USER
                messages become rows.

        Returns:
            One `RewindPromptRow` per USER message in `messages_for_session`
            (which walks the active path; a message not on the active path
            never appears here), ordered newest first. `index_label` is the
            USER turn's 1-based chronological position ("#1", "#2", ...);
            `preview` is a collapsed, truncated single-line preview of its
            content.
        """
        store = self._ensure_console_chat_store()
        user_messages = [
            message
            for message in store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.USER
        ]
        rows = [
            RewindPromptRow(
                message_id=message.id,
                index_label=f"#{position}",
                preview=derive_console_session_title(
                    message.content,
                    max_length=self._CONSOLE_REWIND_PREVIEW_MAX_LENGTH,
                )
                or "(empty prompt)",
            )
            for position, message in enumerate(user_messages, start=1)
        ]
        rows.reverse()
        return tuple(rows)

    async def _console_command_rewind(self, parse: CommandParse) -> bool:
        """Open the `/rewind` menu over the active session's prior USER prompts.

        Collects the active path's USER-turn rows (newest first) and pushes
        `ConsoleRewindModal`; a session with no USER turns yet (or no active
        session at all) is a no-op notify rather than an empty modal.

        Returns:
            True when the modal opened; False when no USER prompt rows exist.
        """
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        rows = self._console_rewind_prompt_rows(session_id) if session_id else ()
        if not rows:
            self.app_instance.notify("Nothing to rewind.", severity="warning")
            return False

        controller = self._ensure_console_chat_controller()
        summary_disabled_reason = self._console_rewind_summary_disabled_reason(
            controller, session_id
        )
        try:
            active_path_identity = tuple(store.active_path_message_ids(session_id))
        except KeyError:
            self.app_instance.notify("Nothing to rewind.", severity="warning")
            return False
        has_effective_memory = False
        try:
            _local, _global, effective_memory = controller.context_control_inputs(
                session_id
            )
            has_effective_memory = effective_memory.kind is not EffectiveMemoryKind.RAW
        except Exception:
            # Keep the modal available, but fail closed for disclosure: the
            # lookup may have failed while replacement memory exists.
            has_effective_memory = True
            logger.warning(
                "Console rewind effective-memory lookup failed; "
                "showing conservative replacement warning."
            )

        async def _apply_choice(choice: "ConsoleRewindChoice | None") -> None:
            await self._apply_console_rewind_choice(
                session_id,
                choice,
                active_path_identity=active_path_identity,
            )

        self.app.push_screen(
            ConsoleRewindModal(
                prompts=rows,
                has_effective_memory=has_effective_memory,
                summary_disabled_reason=summary_disabled_reason,
            ),
            callback=_apply_choice,
        )
        return True

    @staticmethod
    def _console_rewind_summary_disabled_reason(
        controller: ConsoleChatController,
        session_id: str,
    ) -> str:
        """Return only synchronously known run/tip refusal guidance."""
        if not controller.run_state_for(session_id).is_send_allowed:
            return "A run is already running in this tab."
        try:
            path = controller.store.active_path_message_ids(session_id)
            messages = {
                message.id: message
                for message in controller.store.messages_for_session(session_id)
            }
            tip = messages.get(path[-1]) if path else None
        except KeyError:
            tip = None
        if tip is not None and (
            tip.role is ConsoleMessageRole.USER
            or (tip.role is ConsoleMessageRole.ASSISTANT and tip.status != "complete")
        ):
            return "Finish the current exchange before summarizing."
        snapshots = controller._durable_context_snapshots(session_id)
        if snapshots:
            units = complete_durable_units(snapshots)
            if not units or units[-1].boundary_message_id != snapshots[-1].message_id:
                return "Finish the current exchange before summarizing."
        return ""

    async def _apply_console_rewind_choice(
        self,
        session_id: str,
        choice: "ConsoleRewindChoice | None",
        *,
        active_path_identity: tuple[str, ...] | None = None,
    ) -> None:
        """Apply a `/rewind` modal result.

        `None` (Escape / "Never mind") just returns focus to the composer.
        `"restore"` is pure tree navigation: gated on
        `controller.send_refusal_copy(...)` (mirrors regenerate/resend --
        never mutates while a run is streaming; returns non-empty refusal
        copy exactly when a send would currently be blocked, parallel-
        agents spec §4), the new active leaf is the selected prompt's
        PARENT found by an id lookup in `active_path_message_ids` (never
        positional -- display-only TOOL rows can pad
        `messages_for_session`'s view without being tree nodes), with
        `None` (empty transcript) when the selected prompt was the root.
        The selected prompt's own text is written back into the composer
        via the same paste-semantics seam `/prompt` uses.
        Both summary kinds run their controller flow on the same exclusive
        `console-run-{session_id}` worker group, gated on `send_refusal_copy`
        the same way restore is (never mutates while a run is active).

        A `ModalScreen` blocks session switching while the rewind modal is up,
        so today this is theoretical -- but the callback still re-checks the
        store's active session against the one captured when the modal opened
        (`session_id`) before doing anything, and no-ops with a notify on a
        mismatch. This keeps the flow robust against future modal/timing
        changes (e.g. a background auto-switch or a non-modal rewind surface)
        that could let the active session change out from under a pending
        choice.

        Args:
            session_id: Native Console session id the modal was opened for.
            choice: The modal's result, or `None`.
            active_path_identity: Exact ordered path captured for the modal.
        """
        store = self._ensure_console_chat_store()
        try:
            if store.active_session_id != session_id:
                self.app_instance.notify(
                    "Console session changed — rewind cancelled.",
                    severity="warning",
                )
                return
            if choice is None:
                return
            if choice.kind in {KIND_SUMMARIZE_UP_TO, KIND_SUMMARIZE_FROM}:
                try:
                    current_path_identity = tuple(
                        store.active_path_message_ids(session_id)
                    )
                except KeyError:
                    current_path_identity = ()
                if (
                    active_path_identity is not None
                    and current_path_identity != active_path_identity
                ):
                    self.app_instance.notify(
                        "Conversation changed before summarization could start.",
                        severity="warning",
                    )
                    return
                captured_path_identity = (
                    active_path_identity
                    if active_path_identity is not None
                    else current_path_identity
                )
                controller = self._ensure_console_chat_controller()
                # Gate BEFORE spawning: an exclusive console-run worker cancels
                # any in-flight run at creation time, before the controller's
                # own rejection can run -- refuse first, like regenerate.
                target_session_id = controller.store.active_session_id or ""
                refusal = controller.send_refusal_copy(target_session_id)
                if refusal:
                    self.app_instance.notify(refusal, severity="warning")
                    return
                target_is_current = choice.message_id in captured_path_identity
                if not target_is_current:
                    self.app_instance.notify(
                        "Console message action target no longer exists.",
                        severity="error",
                    )
                    return
                worker = (
                    self._summarize_console_from
                    if choice.kind == KIND_SUMMARIZE_FROM
                    else self._summarize_console_up_to
                )
                self.run_worker(
                    worker(
                        controller,
                        session_id,
                        choice.message_id,
                        captured_path_identity,
                    ),
                    exclusive=True,
                    group=f"console-run-{target_session_id}",
                )
                return
            if choice.kind != KIND_RESTORE:
                return
            controller = self._ensure_console_chat_controller()
            refusal = controller.send_refusal_copy(controller.store.active_session_id)
            if refusal:
                self.app_instance.notify(refusal, severity="warning")
                return
            try:
                path = store.active_path_message_ids(session_id)
                index = path.index(choice.message_id)
            except (KeyError, ValueError):
                self.app_instance.notify(
                    "Console message action target no longer exists.",
                    severity="error",
                )
                return
            self._message.apply_rewind_position(
                session_id, choice.message_id, path, index
            )
            # The lookup above proves `choice.message_id` is a live message, so
            # this can't raise -- fetch the FULL text rather than reusing
            # `choice.prompt_text`, which is only the modal row's truncated
            # display preview (see `ConsoleRewindChoice`/`RewindPromptRow`).
            full_text = store.get_message(choice.message_id).content
            self._insert_prompt_text_into_composer(full_text, replace=True)
            await self._sync_native_console_chat_ui()
        finally:
            self._focus_console_composer_if_needed(force=True)

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
            self._sync_console_command_popup()

    async def _open_console_system_prompt_editor(self) -> None:
        """Delegate to `ConsolePromptsController` (wave-3 console decomposition, task 3)."""
        await self._prompts._open_console_system_prompt_editor()

    async def _console_command_skills(self, parse: CommandParse) -> None:
        await self._skill._console_command_skills(parse)

    async def _console_command_fewer_permission_prompts(
        self, parse: CommandParse
    ) -> None:
        """Render local MCP prompt-reduction recommendations."""
        del parse
        self._clear_console_composer_draft()
        service = getattr(self.app_instance, "unified_mcp_service", None)
        loader = getattr(service, "permission_prompt_recommendations", None)
        if not callable(loader):
            await self._append_native_console_system_message(
                "MCP prompt recommendations unavailable - MCP service is not ready."
            )
            return
        try:
            report = await loader()
        except Exception as exc:  # noqa: BLE001 -- render recovery, never send command
            logger.warning(
                "MCP prompt recommendations command failed (exception_type={})",
                type(exc).__name__,
            )
            await self._append_native_console_system_message(
                "MCP prompt recommendations unavailable - local analysis failed."
            )
            return
        await self._append_native_console_system_message(
            format_permission_prompt_report(report)
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

        PR3a-2 Task 5: this is also the auto-wake retry poke for the
        user-wins-ties deferral. This handler (unlike its
        ``DraftChanged`` sibling) fires on EVERY draft mutation from any
        source -- typing, backspace-to-empty, clear, ``load_draft`` -- so
        the moment the composer empties, the user's sending claim is gone
        and a deferred wake may try again. A no-op when nothing is
        pending (``retry_soon`` -> gated ``_attempt``).
        """
        self._console_unknown_send_armed = None
        if not str(event.value or "").strip():
            wake = getattr(self._console_chat_controller, "fleet_wake", None)
            if wake is not None:
                wake.retry_soon()

    @on(Button.Pressed, "#console-composer-collapse")
    def handle_console_composer_collapse(self, event: Button.Pressed) -> None:
        """Collapse the Console composer into reading mode."""
        event.stop()
        self._set_console_composer_collapsed(True)

    @on(Button.Pressed, "#console-composer-expand")
    def handle_console_composer_expand(self, event: Button.Pressed) -> None:
        """Expand the Console composer and restore draft focus."""
        event.stop()
        self._set_console_composer_collapsed(False)

    @on(Button.Pressed, "#console-status-collapse")
    def handle_console_status_collapse(self, event: Button.Pressed) -> None:
        """Collapse the Console status row."""
        event.stop()
        self._set_console_status_chips_collapsed(True)

    @on(Button.Pressed, "#console-status-expand")
    def handle_console_status_expand(self, event: Button.Pressed) -> None:
        """Expand the Console status row."""
        event.stop()
        self._set_console_status_chips_collapsed(False)

    async def handle_console_stop_generation(self, event: Button.Pressed) -> None:
        """Route the Console stop action through native run control."""
        event.stop()
        if self._console_setup_modal_blocking():
            return
        await self._stop_console_generation_from_visible_action()

    async def _stop_console_generation_from_visible_action(self) -> None:
        """Route the visible Console stop action through native run control."""
        # task-3401.5: a video generation in flight for the active session
        # takes the stop first -- it is not a controller "run", so
        # stop_active_run() would only toast "No active Console run to stop."
        # The cancel event wakes the adapter's poll loop immediately.
        store = self._console_chat_store
        active_session_id = store.active_session_id if store is not None else None
        image_edit = (
            self._image._h3_image_edit_registry().request_cancel(active_session_id)
            if active_session_id is not None
            else None
        )
        if image_edit is not None:
            self.app_instance.notify("Stopping image edit…", severity="information")
            self._request_console_control_bar_sync()
            return
        cancel_event = (
            self._video._console_videogen_cancel_events().get(active_session_id)
            if active_session_id is not None
            else None
        )
        if cancel_event is not None:
            cancel_event.set()
            self.app_instance.notify(
                "Stopping video generation…", severity="information"
            )
            return
        # TASK-337 AC1: acknowledge at the click, synchronously — the
        # stopped state itself renders via the (possibly coalesced) sync.
        stop_button: Button | None = None
        try:
            stop_button = self.query_one("#console-stop-generation", Button)
        except QueryError:
            stop_button = None
        if stop_button is not None:
            stop_button.label = "Stopping…"
            stop_button.disabled = True
        try:
            controller = self._ensure_console_chat_controller()
            if not controller.stop_active_run():
                self.app_instance.notify(
                    "No active Console run to stop.", severity="warning"
                )
            await self._sync_native_console_chat_ui()
        finally:
            if stop_button is not None:
                # The bar's sync governs visibility/variant but not the
                # label — restore it so a later run's Stop button never
                # reads Stopping…. Scheduled after the next refresh so the
                # acknowledgment is guaranteed at least one painted frame
                # even when the sync above coalesced away; the finally
                # covers stop/sync exceptions leaving the button stuck.
                def _restore_stop_button(button=stop_button) -> None:
                    button.label = "Stop"
                    button.disabled = False

                self.call_after_refresh(_restore_stop_button)

    @on(Button.Pressed, "#console-attach-context")
    async def handle_console_attach_context(self, event: Button.Pressed) -> None:
        """Open the native Console file picker and stage the selected attachment."""
        await self._handle_console_attach_context(event)

    @on(Button.Pressed, "#console-staged-context-attach")
    async def handle_console_staged_context_attach(self, event: Button.Pressed) -> None:
        """Open the native Console file picker from the staged-context empty state."""
        await self._handle_console_attach_context(event)

    async def _handle_console_attach_context(
        self, event: Button.Pressed | None = None
    ) -> None:
        """Open the native Console file picker and stage the selected attachment.

        Args:
            event: The originating button press, or ``None`` when reached
                from the ☰ composer menu (which has no button event to stop
                because the row lives in a modal, not the action row).
        """
        if event is not None:
            event.stop()
        # TASK-377: block at the cap BEFORE opening the picker. Otherwise the user
        # navigates the picker and selects a sixth file only to have it silently
        # rejected by a transient toast after a full picker round-trip (dead work).
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        if (
            session_id is not None
            and len(store.pending_attachments(session_id)) >= MAX_PENDING_ATTACHMENTS
        ):
            self.app_instance.notify(
                f"Attachment limit reached ({MAX_PENDING_ATTACHMENTS} per message). "
                "Remove one to attach another.",
                severity="warning",
            )
            return
        from fnmatch import fnmatch

        from tldw_chatbook.Chat.attachment_core import attachment_filter_specs
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

        def create_filter(patterns: str):
            pattern_list = patterns.split(";")

            def filter_func(path: Path) -> bool:
                return any(fnmatch(path.name, pattern) for pattern in pattern_list)

            return filter_func

        file_filters = Filters(
            *[
                (label, create_filter(patterns))
                for label, patterns in attachment_filter_specs()
            ],
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
            self._session._ensure_active_console_session_settings()
            session = store.ensure_session()
            # Attach sequentially, stopping as soon as the cap is hit, so a
            # capacity-exhausted drop gets ONE truncation toast here instead
            # of one "limit reached" toast per remaining file.
            attached_count = 0
            for candidate in attachable_paths:
                if (
                    len(store.pending_attachments(session.id))
                    >= MAX_PENDING_ATTACHMENTS
                ):
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
            display_name = f"clipboard-{_datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
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
        self._session._ensure_active_console_session_settings()
        session = store.ensure_session()
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
        self.app_instance.notify(f"{escape_markup(attachment.display_name)} attached")
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
            # Attach appends to the draft: seek the editable caret to the end
            # so file content never splices into the middle of a draft.
            composer.move_cursor_end()
            composer.insert_file_segment(
                attachment.text_content, f"📄 {attachment.label}"
            )
            # TASK-376: name the action distinctly -- a text file is inserted as
            # draft text, not attached like an image, so the user isn't misled
            # into thinking they attached a file.
            self.app_instance.notify(
                f"{escape_markup(attachment.display_name)} inserted as text "
                "(not attached)"
            )
        else:
            store = self._ensure_console_chat_store()
            self._session._ensure_active_console_session_settings()
            session = store.ensure_session()
            if not store.add_pending_attachment(session.id, attachment):
                self.app_instance.notify(
                    "Attachment limit reached (5 per message).", severity="warning"
                )
                self._sync_console_control_bar()
                return
            # Composer label reflects the whole staged list (1 vs N) and is
            # recomputed centrally by `_sync_console_composer_action_state`
            # (called via `_sync_console_control_bar` below).
            self.app_instance.notify(
                f"{escape_markup(attachment.display_name)} attached"
            )
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
            handler = getattr(
                self.app_instance, "open_console_live_work_primary_action", None
            )
            if callable(handler) and bool(handler(launch)):
                # FB-07 (TASK-2154.17): the handoff used to succeed silently.
                # The artifact already exists (the button is gated on a
                # completed live-work launch), so confirm before navigation.
                self.app_instance.notify(
                    "Saved — opening the artifact in Artifacts.",
                    severity="success",
                )
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
                    target == "hidden" or self._is_console_choose_model_action(_label)
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

    def _console_change_review_run_id(
        self, store: ConsoleChatStore, message_id: str
    ) -> str | None:
        """Resolve the run id a review-changes action should open (TASK-2030).

        The transcript's display model is checked FIRST: the ✎ summary row
        is a display-only TOOL marker that the store's tree lookup can never
        resolve. The store remains the fallback for tree-node rows.

        Args:
            store: The native Console chat store.
            message_id: The action button's message id.

        Returns:
            The run id to review, or ``None`` when no rendered or stored
            row with that id carries one.
        """
        try:
            transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        except QueryError:
            transcript = None
        if transcript is not None:
            row = transcript.display_message(message_id)
            run_id = (
                getattr(row, "change_review_run_id", None) if row is not None else None
            )
            if run_id:
                return str(run_id)
        try:
            run_id = getattr(
                store.get_message(message_id), "change_review_run_id", None
            )
        except KeyError:
            return None
        return str(run_id) if run_id else None

    def _open_change_review(
        self,
        run_id: str | None = None,
        *,
        initial_path: str | None = None,
        initial_snapshot_id: int | None = None,
    ) -> None:
        """Push the Change Review screen for the active conversation.

        TASK-1972. Honest empty states are the SCREEN's job: opening with no
        recorded turns shows "No file changes recorded", so this opener only
        needs a provider -- absent (no tracker / no git / no persisted
        conversation) it explains instead of silently no-oping.

        Args:
            run_id: Turn to select on open; ``None`` opens the latest.
            initial_path: TASK-18060 Task 5 (review-rail spec §2's
                click-through recipe): the rail's own opener passes the
                pressed row's file so the screen opens focused on it
                instead of the turn's first leaf. ``None`` for every other
                (pre-existing) caller -- byte-compatible default.
            initial_snapshot_id: Paired with ``initial_path`` -- disambiguates
                two windows of the SAME run covering the same path (spec
                §2). ``None`` when the caller has no snapshot row to pin to.
        """
        provider = self._review_selection._console_change_review_provider()
        if provider is None:
            self.app_instance.notify(
                "Change review needs git and a saved conversation.",
                severity="warning",
            )
            return
        from tldw_chatbook.UI.Screens.change_review_screen import (
            ChangeReviewScreen,
        )

        # TASK-16801 arc B (Task 9): the conversation's LIVE workspace
        # roots -- see `_console_change_review_workspace_roots`'s docstring
        # for why this matters (it is what makes `current` mode reachable
        # from a fresh conversation with no recorded turns).
        workspace_roots = (
            self._review_selection._console_change_review_workspace_roots()
        )

        # initial_run_id/initial_path/initial_snapshot_id all ride the
        # constructor: a post-push select_turn/select_file call raced the
        # screen's own compose (NoMatches) -- caught by the opener wiring
        # test (TASK-1972 / TASK-18060 Task 3).
        self.app.push_screen(
            ChangeReviewScreen(
                provider,
                initial_run_id=run_id,
                initial_path=initial_path,
                initial_snapshot_id=initial_snapshot_id,
                workspace_roots=workspace_roots,
            )
        )

    @on(Button.Pressed, "#console-inspector-review-changes")
    def handle_console_inspector_review_changes(self, event: Button.Pressed) -> None:
        """Open the Change Review screen from the run inspector (TASK-1972)."""
        event.stop()
        self._open_change_review()

    @on(ConsoleTurnFileCard.ReviewRequested)
    def handle_console_turn_file_card_review_requested(
        self, event: ConsoleTurnFileCard.ReviewRequested
    ) -> None:
        """Open the Change Review screen from a card's own `Review` button.

        TASK-16800 (V1.5): the same opener recipe as `v` and the run
        inspector's own button above, except the run id needs no lookup at
        all -- the card that posted this message already knows exactly
        which run it renders.
        """
        event.stop()
        self._open_change_review(event.run_id)

    @on(ConsoleTurnFileCard.UndoAllRequested)
    async def handle_console_turn_file_card_undo_all_requested(
        self, event: ConsoleTurnFileCard.UndoAllRequested
    ) -> None:
        """Prepare and confirm a direct whole-turn revert off the UI thread."""
        event.stop()
        card = event.card
        if self._console_turn_undo_token is not None:
            card.finish_undo_all(success=False)
            self.app_instance.notify(
                "Another Undo All is already in progress.", severity="warning"
            )
            return
        provider = self._review_selection._console_change_review_provider()
        if provider is None:
            card.finish_undo_all(success=False)
            self.app_instance.notify(
                "Undo All needs git and a saved conversation.", severity="warning"
            )
            return

        token = object()
        self._console_turn_undo_token = token
        try:
            plan = await asyncio.to_thread(
                _prepare_console_turn_undo, provider, event.run_id
            )
        except _ConsoleTurnUndoAmbiguousError as exc:
            self._finish_console_turn_undo(token, card, success=False)
            self.app_instance.notify(
                f"{exc} Opening Review to undo safely.", severity="warning"
            )
            self._open_change_review(event.run_id)
            return
        except _ConsoleTurnUndoUnavailableError as exc:
            self._finish_console_turn_undo(token, card, success=False)
            self.app_instance.notify(str(exc), severity="warning")
            return
        except Exception as exc:  # noqa: BLE001 -- provider failure must degrade
            logger.opt(exception=True).warning(
                "Console turn-card Undo All preparation failed."
            )
            self._finish_console_turn_undo(token, card, success=False)
            self.app_instance.notify(
                f"Undo All is unavailable: {str(exc)[:240]}", severity="warning"
            )
            return

        from tldw_chatbook.UI.Screens.change_review_screen import (
            ChangeRevertConfirmModal,
        )

        def _confirmed(confirmed: bool | None) -> None:
            if not confirmed:
                self._finish_console_turn_undo(token, card, success=False)
                return
            if not self.is_mounted:
                self._finish_console_turn_undo(token, card, success=False)
                return
            self.run_worker(
                self._run_console_turn_undo(token, card, provider, plan),
                group="console-turn-undo-all",
                exit_on_error=False,
            )

        self.app.push_screen(
            ChangeRevertConfirmModal(
                f"Undo all {plan.total_files} files from this turn?",
                list(plan.edited_since),
            ),
            callback=_confirmed,
        )

    async def _run_console_turn_undo(
        self,
        token: object,
        card: ConsoleTurnFileCard,
        provider: Any,
        plan: _ConsoleTurnUndoPlan,
    ) -> None:
        """Apply a confirmed plan off-thread and publish honest outcomes."""
        try:
            outcomes = await asyncio.to_thread(_apply_console_turn_undo, provider, plan)
        except _ConsoleTurnUndoUnavailableError as exc:
            self.app_instance.notify(str(exc), severity="warning")
            self._finish_console_turn_undo(token, card, success=False)
            return
        except Exception as exc:  # noqa: BLE001 -- provider failure must degrade
            from tldw_chatbook.Workspaces.change_revert import RevertRefusedError

            if isinstance(exc, RevertRefusedError):
                self.app_instance.notify(str(exc), severity="warning")
            else:
                logger.opt(exception=True).warning(
                    "Console turn-card Undo All apply failed."
                )
                self.app_instance.notify(
                    f"Undo All failed before completion: {str(exc)[:240]}",
                    severity="error",
                )
            self._finish_console_turn_undo(token, card, success=False)
            return

        failed = [outcome for outcome in outcomes if not outcome.ok]
        succeeded = [outcome for outcome in outcomes if outcome.ok]
        success = bool(outcomes) and not failed and len(outcomes) == plan.total_files
        if failed:
            detail = ", ".join(
                f"{outcome.path} ({outcome.error})" for outcome in failed[:5]
            )
            self.app_instance.notify(
                f"{len(succeeded)} file(s) reverted; "
                f"{len(failed)} file(s) not reverted: {detail}",
                severity="error",
            )
        elif success:
            self.app_instance.notify(f"Reverted {len(outcomes)} file(s).")
        else:
            self.app_instance.notify(
                "Undo All did not return a complete result. Open Review and try again.",
                severity="warning",
            )
        self._finish_console_turn_undo(token, card, success=success)

    def _finish_console_turn_undo(
        self, token: object, card: ConsoleTurnFileCard, *, success: bool
    ) -> None:
        """Release one exact operation and update its card when still mounted."""
        if self._console_turn_undo_token is not token:
            return
        self._console_turn_undo_token = None
        if card.is_mounted:
            card.finish_undo_all(success=success)

    @on(Button.Pressed, f"#{CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID}")
    def handle_console_inspector_review_approval(self, event: Button.Pressed) -> None:
        """Focus the pending approval card from the Console inspector seam."""
        event.stop()
        if self._console_pending_approval_count() <= 0:
            self.app_instance.notify(
                CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning"
            )
            return
        card = next(
            (
                candidate
                for candidate in self.query("#chat-approval-card")
                if candidate.display
            ),
            None,
        )
        if card is None:
            self.app_instance.notify(
                CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning"
            )
            return
        try:
            card.scroll_visible(animate=False)
        except Exception:
            pass
        # `set_batch` (the card's sole production entry point, task-914) is
        # the only body it ever renders, so a displayed card's action is
        # always its "Submit" button.
        try:
            card.focus_first_decision()
        except Exception:
            pass

    @on(Button.Pressed, f"#{CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID}")
    def handle_console_inspector_save_chatbook(self, event: Button.Pressed) -> None:
        """Route inspector Chatbook action through the existing Console save seam."""
        self.handle_console_save_chatbook(event)

    async def handle_console_message_action(self, event: Button.Pressed) -> bool:
        """Delegate to `ConsoleMessageController` (wave-3 task 1). Kept
        under the original name for `on_button_pressed` and the
        pre-existing test suite's direct-call convention -- see
        `message.py`'s module docstring."""
        if raw_cli_ui.handle_raw_cli_message_action(
            self._raw_cli, event, self._ensure_console_chat_store()
        ):
            return True
        return await self._message.handle_console_message_action(event)

    def _console_save_as_destinations(self, message: Any) -> list[Any]:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        for the pre-existing test suite's direct-call convention."""
        return self._message._console_save_as_destinations(message)

    async def _save_console_message_image(self, message_id: str) -> None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        for the pre-existing test suite's direct-call convention."""
        await self._message._save_console_message_image(message_id)

    async def _save_console_message_as_note(self, message_id: str) -> None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        for the pre-existing test suite's direct-call convention."""
        await self._message._save_console_message_as_note(message_id)

    async def _open_console_message_edit_modal(
        self, *, message_id: str, content: str
    ) -> None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        for the pre-existing test suite's direct-call convention."""
        await self._message._open_console_message_edit_modal(
            message_id=message_id, content=content
        )

    @staticmethod
    def _parse_console_message_action_button_id(
        button_id: str,
    ) -> tuple[str | None, str | None]:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        (unbound, `ChatScreen.X(...)`) for the pre-existing test suite."""
        return ConsoleMessageController._parse_console_message_action_button_id(
            button_id
        )

    async def _summarize_console_up_to(
        self,
        controller: ConsoleChatController,
        session_id: str,
        message_id: str,
        active_path_identity: tuple[str, ...] | None = None,
    ) -> None:
        """Run the prefix summary through the shared guarded worker flow."""
        await self._summarize_console_range(
            controller,
            session_id=session_id,
            message_id=message_id,
            from_here=False,
            active_path_identity=active_path_identity,
        )

    async def _summarize_console_from(
        self,
        controller: ConsoleChatController,
        session_id: str,
        message_id: str,
        active_path_identity: tuple[str, ...] | None = None,
    ) -> None:
        """Run the inclusive range summary through the shared guarded flow."""
        await self._summarize_console_range(
            controller,
            session_id=session_id,
            message_id=message_id,
            from_here=True,
            active_path_identity=active_path_identity,
        )

    async def _summarize_console_range(
        self,
        controller: ConsoleChatController,
        *,
        session_id: str,
        message_id: str,
        from_here: bool,
        active_path_identity: tuple[str, ...] | None = None,
    ) -> None:
        """Apply one captured rewind summary without touching transcript/draft."""
        should_sync = False
        try:
            try:
                current_path_identity = tuple(
                    controller.store.active_path_message_ids(session_id)
                )
                expected_path_identity = (
                    active_path_identity
                    if active_path_identity is not None
                    else current_path_identity
                )
                captured_selection_is_current = (
                    controller.store.active_session_id == session_id
                    and current_path_identity == expected_path_identity
                    and message_id in expected_path_identity
                )
            except KeyError:
                captured_selection_is_current = False
            if not captured_selection_is_current:
                self.app_instance.notify(
                    "Conversation changed before summarization could start.",
                    severity="warning",
                )
                return
            should_sync = True
            self.app_instance.notify(
                "Summarizing selected range...", severity="information"
            )
            self._start_console_transcript_sync_timer()
            try:
                result = await (
                    controller.summarize_from(message_id)
                    if from_here
                    else controller.summarize_up_to(message_id)
                )
            except asyncio.CancelledError:
                self.app_instance.notify(
                    "Summarization was cancelled.", severity="warning"
                )
                return
            except Exception:
                self.app_instance.notify(
                    "Couldn't summarize the conversation. Try again.",
                    severity="warning",
                )
                return
            visible_copy = (
                "Conversation memory updated."
                if result.accepted
                else result.visible_copy
                or "Couldn't summarize the conversation. Try again."
            )
            self.app_instance.notify(
                visible_copy,
                severity="information" if result.accepted else "warning",
            )
        finally:
            try:
                if should_sync:
                    await self._sync_native_console_chat_ui()
            finally:
                self._focus_console_composer_if_needed(force=True)

    def _select_console_message_variant(
        self, message_id: str, *, direction: str
    ) -> str | None:
        """Delegate to `ConsoleMessageController` (wave-3 task 1) -- kept
        for the pre-existing test suite's direct-call convention (12
        sites across 2 files)."""
        return self._message._select_console_message_variant(
            message_id, direction=direction
        )

    def _get_shell_bar(self):
        """Get the mounted combined chat shell bar.

        ``ChatWindowEnhanced`` is retired (``self.chat_window`` is
        permanently ``None``), so this always returns ``None`` now; kept as
        a stable seam for its remaining live callers' fallback branches.
        """
        return None

    def _get_compact_model_bar(self) -> Optional[CompactModelBar]:
        """Get the native Console compact control bar."""
        try:
            return self.query_one("#console-compact-model-bar", CompactModelBar)
        except QueryError:
            return None

    def _request_console_control_bar_sync(self) -> None:
        """Coalesce control-bar syncs into one trailing run (task-3010).

        Every direct caller of `_sync_console_control_bar` was individually
        justified (mount hooks, session activation, restore, watchers), but
        nothing deduplicated them: one screen push executed the ~47ms sync
        14 times — 0.65s of a ~1.2s settled push (cProfile in the task).
        Requests landing before the trailing run fires fold into it; the
        run always computes fresh state, so the last-writer semantics every
        caller relied on are preserved.
        """
        if getattr(self, "_console_control_bar_sync_scheduled", False):
            return
        self._console_control_bar_sync_scheduled = True
        self.call_after_refresh(self._run_coalesced_control_bar_sync)

    @on(ConsoleAutoSpeakChanged)
    def on_console_auto_speak_changed(self, event: ConsoleAutoSpeakChanged) -> None:
        """Delegate a Console auto-speak state change.

        Args:
            event: Change event carrying the requested enabled state.
        """
        event.stop()
        self._hands_free.on_console_auto_speak_changed(event)

    @on(ConsoleAutoSpeakResumeRequested)
    def on_console_auto_speak_resume_requested(
        self, event: ConsoleAutoSpeakResumeRequested
    ) -> None:
        """Delegate a Console auto-speak resume request.

        Args:
            event: Resume request from the Console speech controls.
        """
        event.stop()
        self._hands_free.on_console_auto_speak_resume_requested(event)

    @on(ConsoleAutoSpeakRetryRequested)
    def on_console_auto_speak_retry_requested(
        self, event: ConsoleAutoSpeakRetryRequested
    ) -> None:
        """Delegate a Console auto-speak retry request.

        Args:
            event: Retry request from the Console speech controls.
        """
        event.stop()
        self._hands_free.on_console_auto_speak_retry_requested(event)

    def _run_coalesced_control_bar_sync(self) -> None:
        """Execute one coalesced control-bar sync (task-3010)."""
        self._console_control_bar_sync_scheduled = False
        self._sync_console_control_bar()

    def _sync_console_control_bar(
        self,
        rail_state: Optional[ConsoleRailState] = None,
    ) -> None:
        """Refresh Console-owned control labels from current selection state.

        Args:
            rail_state: Pre-computed rail state (TASK-251: the 0.2s tick
                computes this once in ``_sync_native_console_chat_ui`` and
                passes it in here, instead of this method redundantly
                recomputing it -- which itself rebuilds workspace-context
                and inspector state). Other callers may omit it; it is
                computed on demand when not given.
        """
        self._sync_console_pending_delete_confirmation()
        self._library_activity.sync_projection()
        control_state = self._build_console_control_state(
            self._pending_console_launch_context
        )
        workbench_state = self._build_console_workbench_state(control_state)
        self._push_console_control_state_if_changed(control_state, workbench_state)
        self._sync_console_transcript_guidance()
        # TASK-251 (audit P1 B1) -- DEVIATION FROM THE BRIEF, documented in
        # the task-251 report: the brief's Change 3 asked to skip this
        # build+push entirely while the right rail is hidden. Measured
        # against the actual test suite, that broke real behavior --
        # Console keeps the inspector's mounted content fresh in the
        # background regardless of paint visibility (selecting a message,
        # a setup blocker appearing, resuming a conversation, etc. all
        # still need `#console-run-inspector-state`'s children to reflect
        # the latest state even while collapsed, and several existing
        # tests assert exactly that). The audit's actual measured
        # complaint -- "streaming-excerpt selection = 5 teardowns/s" -- is
        # already fixed below by `_selected_console_message_inspector_rows`
        # rendering a stable "Streaming…" placeholder: the built state stops
        # changing tick-to-tick while streaming, so `ConsoleRunInspector.
        # sync_state`'s own equality guard (`if state == self.state: return`)
        # already skips the recompose regardless of visibility. So this
        # keeps building and pushing unconditionally, as before task-251.
        try:
            inspector = self.query_one(
                "#console-run-inspector-state", ConsoleRunInspector
            )
        except QueryError:
            inspector = None
        try:
            authority_summary = self.query_one(
                "#console-send-authority-summary", ConsoleSendAuthoritySummary
            )
        except QueryError:
            authority_summary = None
        inspector_state = self._build_console_inspector_state(
            self._pending_console_launch_context
        )
        if inspector is not None:
            # Strict ownership validates the complete snapshot before the
            # resilient pinned projection publishes any part of it.
            inspector.sync_state(inspector_state)
        if authority_summary is not None:
            authority_summary.sync_state(inspector_state)
        self._sync_console_composer_action_state(
            can_save_chatbook=inspector_state.can_save_chatbook
            and self._console_chatbook_action_available()
        )
        if rail_state is None:
            rail_state = self._current_console_rail_state(
                inspector_state=inspector_state
            )
        self._sync_console_rail_visibility_if_changed(rail_state)
        # Cost-ticker PR3 (task-5): deliberately OUTSIDE the
        # `control_state_changed or workbench_state_changed` guard above
        # (same reasoning as the unconditional inspector build) -- cost/
        # cache state changes independently of whether the control labels
        # did, e.g. every streamed token grows the running total, and the
        # cache TTL counts down with no control-state change at all.
        # `_sync_console_cost_chip` owns its own equality guard.
        self._sync_console_cost_chip()
        self._console_auto_speak.sync_controls()

    def _push_console_control_state_if_changed(
        self,
        control_state: ConsoleControlState,
        workbench_state: Any,
    ) -> bool:
        """Push control/Workbench state into the widgets only when it moved.

        The one place `_last_console_control_state` /
        `_last_console_workbench_state` are read and written (task-15452).
        Every caller that pushes must go through here: the control bar
        consumes `workbench_state.actions` as well as `control_state`, so a
        caller that pushed only the Workbench widgets and then recorded the
        new `_last_*` values would make the NEXT `_sync_console_control_bar`
        see "unchanged" and skip a control-bar refresh it still owed --
        which is exactly the trap the draft-edit path used to avoid only by
        never recording anything.

        Args:
            control_state: Freshly built Console control/readiness state.
            workbench_state: Workbench state built from that control state.

        Returns:
            True when the widgets were pushed, False when nothing moved.
        """
        if (
            control_state == self._last_console_control_state
            and workbench_state == self._last_console_workbench_state
        ):
            return False
        try:
            control_bar = self.query_one("#console-control-bar", ConsoleControlBar)
        except QueryError:
            control_bar = None
        if control_bar is not None:
            control_bar.sync_state(control_state, actions=workbench_state.actions)
        try:
            status_chips = self.query_one("#console-status-chips", ConsoleStatusChips)
        except QueryError:
            status_chips = None
        if status_chips is not None:
            status_chips.sync_state(control_state)
        self._sync_console_workbench_state(
            control_state, workbench_state=workbench_state
        )
        self._last_console_control_state = control_state
        self._last_console_workbench_state = workbench_state
        return True

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
            self.query_one(
                "#console-workbench-command-strip", CommandStrip
            ).sync_actions(workbench_state.actions)
        except QueryError:
            pass
        try:
            self.query_one("#workbench-recovery-callout", RecoveryCallout).sync_state(
                workbench_state.recovery
            )
        except QueryError:
            pass

    def _sync_console_workbench_actions_from_draft(self) -> None:
        """Refresh Workbench command readiness after composer draft changes.

        Runs on every printable keystroke, so it goes through the same
        equality gate as `_sync_console_control_bar` (task-15452): before
        that it rebuilt and re-pushed Workbench state unconditionally --
        ~12 layout-invalidating `Static.update` calls plus two
        `sort_children` (which bump the DOM version up the whole ancestor
        chain and so evict the screen-wide `query_one` cache) for a state
        that is identical between any two characters of a word.

        `_sync_console_command_popup` stays OUTSIDE the gate: the popup
        filters on the draft text itself, which moves on every keystroke
        while the derived Workbench state does not.
        """
        with self._console_derivation_scope():
            control_state = self._build_console_control_state(
                self._pending_console_launch_context
            )
            workbench_state = self._build_console_workbench_state(control_state)
        self._push_console_control_state_if_changed(control_state, workbench_state)
        self._sync_console_command_popup()

    def _console_command_popup_or_none(self) -> ConsoleCommandPopup | None:
        try:
            return self.query_one("#console-command-popup", ConsoleCommandPopup)
        except QueryError:
            return None

    #: Draft text `_sync_console_command_popup` last ran against, or
    #: None before the first sync. A CLASS attribute, deliberately: the
    #: hand-built `ChatScreen.__new__()` test fixtures never run `__init__`,
    #: and this programme has shipped one fixture AttributeError per wave.
    _console_popup_synced_draft: str | None = None

    #: Completion context the popup was Escape-dismissed in, or None when
    #: the trigger is armed (TASK-24416). A CLASS attribute for the same
    #: fixture reason as `_console_popup_synced_draft` above. Sticky within
    #: one context: further edits in the SAME context keep the popup hidden;
    #: the latch clears the moment the draft leaves completion context (a
    #: space, a clear, a send), so a fresh `/` re-arms the trigger.
    _console_popup_dismissed_context: str | None = None

    def _sync_console_command_popup(self) -> None:
        """Show/hide the slash-command popup from the current composer draft.

        Records the draft text it ran against (NOT the composer's
        `_draft_generation` -- that is an undo-checkpoint counter that
        `insert_text` never advances; a gate on it no-oped on every
        keystroke, measured), so
        `_ensure_console_command_popup_current` can tell a popup that is
        merely CLOSED (Escape) from one that is STALE (an edit happened but
        its `DraftChanged` has not been delivered yet). Recorded on every
        path, including the hide paths -- "synced" means "reflects this
        draft", not "open".

        TASK-24416: an Escape dismissal is STICKY -- later edits in the same
        completion context keep the popup hidden (there was previously no
        way to compose a slash-prefixed draft with it dismissed); leaving
        the context re-arms the trigger.
        """
        popup = self._console_command_popup_or_none()
        if popup is None:
            return
        composer = self._console_composer_or_none()
        if composer is None:
            return
        self._console_popup_synced_draft = composer.draft_text()
        if composer.has_paste_segments():
            popup.hide()
            return
        context_info = completion_context_for_draft(composer.draft_text())
        if context_info is None:
            # Left the completion context entirely: re-arm the trigger.
            self._console_popup_dismissed_context = None
        suggestions = suggestions_for_draft(
            composer.draft_text(),
            self._console_command_registry,
            self._console_skill_candidates,
        )
        if not suggestions:
            popup.hide()
            return
        if (
            context_info is not None
            and self._console_popup_dismissed_context == context_info[0]
        ):
            popup.hide()
            return
        popup.show_suggestions(suggestions)

    def _ensure_console_command_popup_current(self) -> None:
        """Re-sync the popup only if the draft moved since the last sync.

        Closes the same-driver-read window (task-3790): when `/`+Down or
        `/`+Enter arrive in one read, the second key's routing used to
        consult a popup whose `DraftChanged` was still queued. Re-deriving
        here is safe ONLY because it is gated on the synced draft text -- an ungated
        re-sync would re-open a popup the user just dismissed with Escape
        (dismissal edits nothing, so the text does not move, so this
        is a no-op for it). The queued `DraftChanged` still delivers and
        re-runs the full sync; by then the generation matches and
        `show_suggestions` is idempotent for identical rows, so the
        highlight a routed Down moved is not yanked back.
        """
        composer = self._console_composer_or_none()
        if composer is None:
            return
        if composer.draft_text() != self._console_popup_synced_draft:
            self._sync_console_command_popup()

    def _dismiss_console_command_popup(self) -> bool:
        """Hide the popup if open. Returns True when it was open.

        TASK-24416: a dismissal latches the completion context it happened
        in, so subsequent edits in that context do not re-open the popup;
        the latch clears in `_sync_console_command_popup` when the draft
        leaves completion context.
        """
        popup = self._console_command_popup_or_none()
        if popup is None or not popup.is_open:
            return False
        popup.hide()
        composer = self._console_composer_or_none()
        draft = composer.draft_text() if composer is not None else ""
        context_info = completion_context_for_draft(draft)
        self._console_popup_dismissed_context = (
            context_info[0] if context_info is not None else None
        )
        return True

    def _accept_console_command_popup(self) -> bool:
        """Insert the highlighted suggestion into the draft. True when accepted.

        TASK-24416: the replacement routes through
        `replace_draft_via_completion` so the pre-accept draft stays
        Ctrl+Z-able -- `load_draft` here used to wipe the undo history.
        """
        popup = self._console_command_popup_or_none()
        if popup is None or not popup.is_open:
            return False
        suggestion = popup.accept_selected()
        if suggestion is None:
            return False
        composer = self._console_composer_or_none()
        if composer is None:
            return False
        composer.replace_draft_via_completion(suggestion.insert_text)
        self._sync_console_workbench_actions_from_draft()
        return True

    def on_resize(self) -> None:
        """Keep an open command popup anchored above the composer."""
        popup = self._console_command_popup_or_none()
        if popup is not None and popup.is_open:
            popup.reposition()

    def _persist_console_composer_draft_after_history_navigation(
        self, composer: ConsoleComposerBar
    ) -> None:
        """Re-sync store + Workbench state after an undo/redo mutates the draft.

        Mirrors `_insert_console_dictation`'s own re-persist (TASK-1281):
        undo/redo mutate the composer directly, bypassing every other
        draft-mutation call site's own `store.set_session_draft` follow-up,
        so without this the store and the visible composer would split-brain
        the instant a session switch (or app restore) next reads the store's
        copy instead of the live widget.

        Callers must have already confirmed
        `_console_composer_history_session_synced()` before mutating the
        composer at all (F1) -- this method persists to whatever session is
        CURRENTLY active, trusting that check rather than re-deriving it,
        so it must never be called from a stale window.
        """
        store = self._ensure_console_chat_store()
        if store.active_session_id is not None:
            try:
                store.set_session_draft(store.active_session_id, composer.draft_text())
            except KeyError:
                pass
        self._sync_console_workbench_actions_from_draft()

    def _console_composer_undo(self) -> None:
        """Undo the most recent Console composer draft mutation (TASK-1281).

        A no-op composer-side and here when there is nothing to undo -- no
        store write, no Workbench resync, matching the "silent no-op" AC.
        Also a no-op, with the composer left entirely untouched, while the
        session-switch settle window is open (F1) -- applying an undo in
        that window at all (not just skipping the persist) would leave the
        composer showing content that belongs to neither the session it
        still visibly reflects nor the one the store now considers active.
        """
        if not self._session._console_composer_history_session_synced():
            return
        composer = self._console_composer_or_none()
        if composer is None or not composer.undo():
            return
        self._persist_console_composer_draft_after_history_navigation(composer)

    def _console_composer_redo(self) -> None:
        """Redo a Console composer draft mutation that was just undone (TASK-1281).

        See `_console_composer_undo` -- the same settle-window guard (F1)
        applies here.
        """
        if not self._session._console_composer_history_session_synced():
            return
        composer = self._console_composer_or_none()
        if composer is None or not composer.redo():
            return
        self._persist_console_composer_draft_after_history_navigation(composer)

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
        return self._launch_targets_chatbook_artifact(
            self._pending_console_launch_context
        ) and callable(
            getattr(self.app_instance, "open_console_live_work_primary_action", None)
        )

    def _dispatch_console_recovery_action(
        self,
        session_id: str,
        assistant_message_id: str,
        action: str,
    ) -> None:
        """Run one mounted recovery intent against the currently pinned owner."""

        controller = self._console_chat_controller
        if controller is None or not session_id or not assistant_message_id:
            return
        recovery = controller.store.dispatch_recovery_for_session(session_id)
        if (
            recovery is None
            or recovery.assistant_message_id != assistant_message_id
            or not recovery.recovery_needed
        ):
            return
        revision = controller.prompt_queue_registry.snapshot(session_id).revision
        self.run_worker(
            self._prompt_queue.handle_primary_intent(
                session_id,
                action=action,
                expected_revision=revision,
            ),
            exclusive=True,
            group="console-dispatch-recovery-action",
        )

    def _sync_console_composer_action_state(
        self,
        *,
        can_save_chatbook: bool,
    ) -> None:
        """Refresh Console composer action priority from draft, run, and artifact state.

        F1 (task-9 review): the composer bar's own Save Chatbook button is a
        second door onto the same write the workbench action already gates
        -- reads ``_console_active_session_is_ephemeral()`` directly here so
        both doors consult the same accessor without a caller having to
        remember to thread it through.
        """
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return

        run_active = False
        send_blocked = False
        controller = self._console_chat_controller
        queue_presentation = None
        if controller is not None:
            run_state = getattr(controller, "run_state", None)
            run_active = bool(getattr(run_state, "is_stop_allowed", False))
            send_blocked = not bool(getattr(run_state, "is_send_allowed", True))
            active_id = controller.store.active_session_id or ""
            if active_id:
                queue_presentation = self._prompt_queue.presentation_for(
                    active_id,
                    composer_collapsed=composer.collapsed,
                )
                # TASK-22000 (owner decision, 2026-08-24): for a session with
                # a live queue projection the PRESENTATION is the authority on
                # whether Send accepts a draft -- not the raw run state. That
                # was ADR-098's original shape (an assignment here, not an
                # `or`); `2c7fcd200` folded `send_blocked` back in with `or`
                # alongside the new recovery predicate, and since
                # `not is_send_allowed` is exactly the VALIDATING/STREAMING/
                # CHECKING_CITATIONS/RETRYING set that `derive_prompt_queue_
                # presentation` already reads as `occupies_slot`, the only
                # thing that `or` could still change was the one state ADR-098
                # exists for: an ACCEPTED live turn, which must read "Queue"
                # and admit a FIFO follow-up. It rendered as a greyed-out
                # button labelled "Queue" for the whole duration of every run.
                #
                # Nothing is lost by deferring: before acceptance the same
                # projection returns "Preparing..." with `send_enabled=False`,
                # so an unaccepted live run still refuses. The recovery
                # predicate stays in the `or` and still refuses for a genuinely
                # unresolved owner (see `dispatch_recovery_blocks_submission`).
                send_blocked = (
                    not queue_presentation.send_enabled
                    or controller.store.dispatch_recovery_blocks_submission(active_id)
                )
                try:
                    queue_region = self.query_one(
                        "#console-prompt-queue", ConsolePromptQueueRegion
                    )
                except QueryError:
                    pass
                else:
                    queue_region.sync_presentation(active_id, queue_presentation)
                composer.sync_prompt_queue_state(
                    count=queue_presentation.count,
                    paused=queue_presentation.paused,
                )
                try:
                    recovery_region = self.query_one(
                        "#console-dispatch-recovery",
                        ConsoleDispatchRecoveryRegion,
                    )
                except QueryError:
                    pass
                else:
                    recovery_region.sync_recovery(
                        active_id,
                        controller.store.dispatch_recovery_for_presentation(active_id),
                    )
        # task-3401.5: an in-flight video generation shows the same Stop
        # affordance (it sets the adapter's cooperative cancel event).
        store_for_videogen = self._console_chat_store
        active_session_id = (
            store_for_videogen.active_session_id
            if store_for_videogen is not None
            else None
        )
        videogen_active = (
            active_session_id is not None
            and active_session_id in self._video._console_videogen_inflight_sessions()
        )
        image_edit_active = (
            active_session_id is not None
            and self._image._h3_image_edit_registry().active(active_session_id)
            is not None
        )
        run_active = run_active or videogen_active or image_edit_active
        send_blocked = send_blocked or image_edit_active
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
            setup_blocked_reason=(
                setup_blocked_reason
                or attachment_blocked_reason
                or (
                    queue_presentation.send_tooltip
                    if queue_presentation is not None
                    and not queue_presentation.send_enabled
                    else ""
                )
            ),
            ephemeral=self._console_active_session_is_ephemeral(),
            send_label=(
                queue_presentation.send_label
                if queue_presentation is not None
                else "Send"
            ),
            # task-15862 AC#3: mid-wake, the queue tooltip above rode the
            # setup slot and painted as "finish provider setup"; the flag
            # makes the composer name the wake instead.
            wake_turn_active=self._fleet._console_wake_turn_active(active_session_id),
        )
        composer.sync_dictation_state(self._console_dictation_state)
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
        composer.set_pending_attachment_label(
            attachment_label,
            count=len(pendings),
            total=MAX_PENDING_ATTACHMENTS,
        )

    def _focus_console_composer_if_needed(self, *, force: bool = False) -> None:
        """Focus the native Console composer when no other control owns focus."""
        if self._console_composer_collapsed:
            self._focus_console_workbench_target("console-native-composer")
            return
        try:
            composer = self.query_one("#console-native-composer", ConsoleComposerBar)
        except QueryError:
            return
        focused = self.app.focused
        if (
            not force
            and focused is not None
            and not self._is_descendant_or_self(focused, composer)
        ):
            return
        composer.focus()

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
        """Return True when key or paste input belongs to the Console composer."""
        if composer.collapsed:
            return False
        focused = self.app.focused
        if getattr(focused, "id", None) in {
            "console-composer-collapse",
            "console-composer-expand",
            "console-collapsed-stop-generation",
        }:
            return False
        return focused is None or self._is_descendant_or_self(focused, composer)

    @on(Resize)
    def _adapt_console_shell_to_height(self, event: Resize) -> None:
        """Drop the header banner at small heights to preserve the composer.

        TASK-346: at <=34 rows the composer was pushed off the bottom with
        no warning (a silently broken core loop at 97x30, larger than the
        80x24 default terminal). Toggling `-console-compact` hides the ~5-row
        header banner (title/purpose/Ready) so transcript+composer stay
        visible; the setup-card/onboarding overlay is unaffected.

        TASK-361: a live resize also dismisses any hover tooltip. The review saw
        a nav-tab tooltip ("Open the live agent Console.") stick over the header
        across reflows — a mounted overlay that survived the repaint. Clearing it
        on resize removes that stale-overlay class of artifact. (The pane reflow
        itself converges to the cold-start layout on a native resize; the
        divergence in the review was specific to textual-serve's browser-viewport
        resize path at the pre-TASK-346 state and is regression-locked below.)
        """
        clear_tooltip = getattr(self, "_clear_tooltip", None)
        if callable(clear_tooltip):
            clear_tooltip()
        try:
            shell = self.query_one("#console-shell")
        except QueryError:
            return
        compact = event.size.height < CONSOLE_COMPACT_HEIGHT_ROWS
        shell.set_class(compact, "-console-compact")
        self._request_console_context_allocation_reconcile()

    @on(Resize)
    def _adapt_console_workspace_to_width(self, event: Resize) -> None:
        """Re-evaluate the width-driven rail rules on live terminal resizes.

        TASK-2154.1 (LY-08/LY-09): available_columns feeds
        build_console_rail_state only when rail state is (re)built -- at
        compose or on a console sync tick -- and a pure terminal resize
        triggers neither, so without this hook crossing the 100/84-column
        thresholds mid-session left the grid in its stale (possibly broken)
        layout until some unrelated state change happened to rebuild it.
        Guarded on the width BAND so a resize drag does not rebuild the
        (store-reading) effective rail state on every event.
        """
        band = console_rail_width_band(event.size.width)
        if band == self._last_console_workspace_width_band:
            return
        self._last_console_workspace_width_band = band
        try:
            self.query_one("#console-workspace-grid")
        except QueryError:
            return
        focused = self.app.focused
        left_rail = self.query_one("#console-left-rail")
        right_rail = self.query_one("#console-right-rail")
        left_handle = self.query_one("#console-context-rail-handle")
        right_handle = self.query_one("#console-inspector-rail-handle")
        focused_in_left_rail = self._is_descendant_or_self(focused, left_rail)
        focused_in_right_rail = self._is_descendant_or_self(focused, right_rail)
        focused_in_left_handle = self._is_descendant_or_self(focused, left_handle)
        focused_in_right_handle = self._is_descendant_or_self(focused, right_handle)
        rail_state = self._current_console_rail_state(
            available_columns=event.size.width
        )
        self._sync_console_rail_visibility_if_changed(rail_state)
        for focused_in_rail, focused_in_handle, rail_open, rail, handle, buttons in (
            (
                focused_in_left_rail,
                focused_in_left_handle,
                rail_state.left_open,
                left_rail,
                left_handle,
                ("#console-context-rail-open", "#console-context-rail-collapse"),
            ),
            (
                focused_in_right_rail,
                focused_in_right_handle,
                rail_state.right_open,
                right_rail,
                right_handle,
                (
                    "#console-inspector-rail-open",
                    "#console-inspector-rail-collapse",
                ),
            ),
        ):
            if focused_in_rail and not rail_open:
                target, button_selector = handle, buttons[0]
            elif focused_in_handle and rail_open:
                target, button_selector = rail, buttons[1]
            else:
                continue
            button = self.query_one(button_selector, Button)
            if target.display and button.display:
                button.focus()

    @on(DescendantBlur)
    @on(DescendantFocus)
    def _paint_console_rail_focus_frame(
        self, event: DescendantFocus | DescendantBlur
    ) -> None:
        """Paint dimension-stable workbench focus cues.

        TASK-20937.3: expanded rails and collapsed handles keep the exact
        same border cells focused or unfocused. The focus class reinforces
        the owning control with bold/underline/background, so color is not
        the only cue. The transcript owns neither divider; its class marks
        only the stable title row.
        """
        focused = (
            event.widget if isinstance(event, DescendantFocus) else self.app.focused
        )
        sync_console_focus_paint(self, focused)

    #: Task 4 fix-round-2 (I3): how long `_recover_stuck_console_send_stash`
    #: waits before treating `_console_pending_send_stash` as abandoned.
    #: `Button.press()` only POSTS `Button.Pressed`; the message pump
    #: normally delivers and consumes it within a pump cycle or two (well
    #: under this), so this is a generous margin against a false-positive
    #: recovery racing the normal path, not a tight deadline.
    _CONSOLE_SEND_PENDING_STASH_WATCHDOG_SECONDS: float = 0.75

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
        # Task-5 review I2: this branch is deliberately ABOVE the
        # `_should_capture_console_input` focus gate below -- keyboard
        # barge-in and Esc are the loop's PRIMARY interruption/exit
        # mechanism ("press any key"/"Esc from any point in the loop", per
        # the docs), and must keep working even when focus has moved to,
        # say, the transcript (clicking a message, scrolling with the
        # mouse) rather than staying pinned to the composer. Byte-identical
        # `on_key` outside the loop is unaffected -- this whole branch is
        # gated on `hands_free is not None`, and it never touches the
        # composer itself (typed input keeps its normal, focus-gated
        # semantics via the unmoved checks below).
        hands_free = self._console_hands_free
        if hands_free is not None:
            if event.key == "escape":
                # TASK-24417: an open slash-command popup claims Escape
                # before the loop exit ("make the overlay go away" must not
                # cost the loop). The priority binding's action makes the
                # same claim; this bubbling-order branch is the fallback.
                if self._dismiss_console_command_popup():
                    event.stop()
                    event.prevent_default()
                    return
                # Task 5: Esc/mic press/spoken "stop" all exit the loop from
                # any state -- scoped to hands-free-active ONLY, ahead of
                # the screen's own `escape -> focus_console_composer_home`
                # binding (below, :1627 pre-Task-5) so that binding's normal
                # semantics are restored the instant the loop is not
                # running.
                hands_free.controller.on_exit_request()
                event.stop()
                event.prevent_default()
                return
            # Every other key barges in per the controller's own state
            # guards (a no-op in `listening`/`idle` -- see `on_composer_
            # key`'s docstring) and is NOT stopped here: it falls through to
            # the ordinary handling below (still focus-gated), so a
            # countdown-cancelling Enter still sends the TYPED draft via the
            # normal path afterward (this call runs first in the SAME
            # keypress, cancelling any armed countdown/suppressing an
            # awaiting reply BEFORE the Enter branch below presses Send)
            # rather than double-firing hands-free's own voice-triggered
            # send.
            hands_free.controller.on_composer_key()
        # TASK-24417: while a realtime loop is active, an open slash popup
        # claims Escape before the loop's key policy consumes it -- the same
        # claim the hands-free branches make above.
        if (
            event.key == "escape"
            and self._realtime.session is not None
            and self._dismiss_console_command_popup()
        ):
            event.stop()
            event.prevent_default()
            return
        if self._realtime.handle_key(event.key):
            event.stop()
            event.prevent_default()
            return
        if not self._should_capture_console_input(composer):
            return
        self._ensure_console_command_popup_current()
        popup = self._console_command_popup_or_none()
        if popup is not None and popup.is_open:
            if event.key == "up":
                popup.move_highlight(-1)
                event.stop()
                event.prevent_default()
                return
            if event.key == "down":
                popup.move_highlight(1)
                event.stop()
                event.prevent_default()
                return
            if event.key == "enter":
                # TASK-24416: an empty-prefix list (bare `/`, or a bare
                # `/skills `) is the user probing the trigger -- Enter means
                # send there, not "silently stage the first listed command".
                # A non-empty filtered prefix keeps Enter-accept as before.
                context_info = completion_context_for_draft(composer.draft_text())
                if context_info is not None and context_info[1]:
                    self._accept_console_command_popup()
                    event.stop()
                    event.prevent_default()
                    return
                # Empty prefix: close the popup (latched -- a restored
                # unknown-command draft must not re-open it over the
                # transcript guidance) and let the ordinary Enter path below
                # own the keystroke.
                self._dismiss_console_command_popup()
        # Decomposition wave 5: the keys whose whole handling is a composer
        # operation (select-all and caret movement, including Up/Down's
        # history-recall-first shape, which still falls through UNCONSUMED
        # on a boundary row where nothing moved) now live on the composer
        # itself. TASK-3749 added the draft-EDITING keys to that set -- they
        # post `ConsoleComposerBar.DraftChanged`, which
        # `_handle_console_composer_draft_edit` below turns back into the
        # Workbench/guidance resync this method used to do inline. That
        # includes the printable fallthrough, so this delegation is now also
        # where ordinary typing lands. Everything below stays because it
        # reaches past the composer -- the clipboard, undo/redo's store
        # persistence, send, transcript paging.
        if composer.handle_console_key(event):
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
        if event.key == "enter":
            if composer.activate_focused_paste_token():
                event.stop()
                event.prevent_default()
                return
            event.stop()
            event.prevent_default()
            if self._console_pending_send_stash is not None:
                # A send keypress is already on its way to the Pressed
                # handler; a second Enter in that window would stash the
                # now-empty composer (None) over the pending payload and
                # eat the message. Swallow the duplicate.
                return
            # TASK-340: capture the payload NOW — Button.press() only posts a
            # message, and printable keys handled before that message runs
            # used to fold into the sent text.
            stash = composer.stash_draft_for_send()
            self._console_pending_send_stash = stash
            try:
                send_button = self.query_one("#console-send-message", Button)
            except QueryError:
                self._console_pending_send_stash = None
                composer.restore_stashed_draft(stash)
                self.app_instance.notify(
                    "Console send is unavailable.", severity="error"
                )
                return
            if send_button.disabled and send_button.display:
                # TASK-2154.6 (FR-04): Send is now genuinely disabled
                # while blocked/empty, and `Button.press()` is a no-op
                # on a disabled control — a plain press here would
                # silently kill the Enter hotkey's blocked-attempt
                # feedback (toast + transcript system row) and strand
                # the pending stash (the next Enter would then be
                # swallowed as a duplicate above). Dispatch the same
                # handler a press reaches, exactly as the voice-send
                # path already does for its synthesized press.
                self.run_worker(
                    self.handle_console_send_message(Button.Pressed(send_button))
                )
                return
            if not send_button.display:
                # Task 4 (D2 fix wave): Textual 8.2.7's `Button.press()`
                # returns immediately -- without posting `Button.Pressed` --
                # when the button is not `display`ed (which is also `False`
                # while the button is being pruned, e.g. any
                # `refresh(recompose=True)` mid-keypress). Without this
                # check, `_console_pending_send_stash` above is set and
                # never consumed (the Pressed handler that would clear it
                # never runs), so the draft is stuck stashed with an empty
                # composer AND the duplicate-guard just above permanently
                # swallows every subsequent Enter, since the stash slot
                # never goes back to `None` on its own.
                # Fix-round-2 (M1): this branch was itself silent -- log the
                # button state so a recurrence is diagnosable (the reviewer's
                # own note: the pure no-op-press hypothesis alone can't
                # explain "a second keyboard send worked", so a log here is
                # what would confirm or rule this mechanism out if D2
                # resurfaces).
                logger.warning(
                    "Console send Enter: no-op press guard tripped "
                    "(disabled={}, display={}) -- restoring the draft "
                    "instead of losing it.",
                    send_button.disabled,
                    send_button.display,
                )
                self._console_pending_send_stash = None
                composer.restore_stashed_draft(stash)
                return
            send_button.press()
            # Fix-round-2 (I3): `.press()` only POSTS `Button.Pressed` for
            # the message pump to deliver later -- the check just above
            # closes the case where `press()` itself no-ops, but NOT the
            # narrower race where display/disabled were still fine at check
            # time and go bad in the gap before the pump actually delivers
            # the message (a prune beginning mid-flight). That drops the
            # posted message with nothing to consume `_console_pending_
            # send_stash`, latching the duplicate guard above shut forever.
            # This watchdog is the backstop: if the stash is STILL this
            # exact object once the window passes, nothing consumed it, so
            # recover it instead of leaving it stuck.
            self.set_timer(
                self._CONSOLE_SEND_PENDING_STASH_WATCHDOG_SECONDS,
                partial(self._recover_stuck_console_send_stash, stash),
            )
            return
        if event.key in {"pageup", "pagedown"}:
            # TASK-348: scrollback must be keyboard-reachable. The composer
            # never uses paging keys (Home/End move its caret), so route
            # them to the transcript — the standard chat-app idiom.
            try:
                transcript = self.query_one(
                    "#console-native-transcript", ConsoleTranscript
                )
            except QueryError:
                return
            if event.key == "pageup":
                transcript.scroll_page_up(animate=False)
            else:
                transcript.scroll_page_down(animate=False)
            event.stop()
            event.prevent_default()
            return
        if event.key == "ctrl+z":
            self._console_composer_undo()
            event.stop()
            event.prevent_default()
            return
        if event.key in {"ctrl+shift+z", "ctrl+shift+Z"}:
            # TASK-1281: both tokens are real Textual key strings for this
            # chord depending on how the terminal (or its keyboard protocol)
            # reports the shifted keycap's codepoint -- verified against
            # `textual._xterm_parser.XTermParser._parse_extended_key` with
            # synthetic Kitty CSI-u sequences: codepoint 122 ('z') + shift+
            # ctrl modifiers yields "ctrl+shift+z", while codepoint 90 ('Z')
            # with the same modifiers yields "ctrl+shift+Z". `Pilot.press()`
            # (and this project's existing `ctrl+shift+p/c/a` bindings) use
            # the lowercase form, which is the primary/expected token; the
            # uppercase alias is defensive.
            self._console_composer_redo()
            event.stop()
            event.prevent_default()
            return
        if event.key == "ctrl+y":
            # TASK-1733: terminals without the Kitty keyboard protocol
            # (Terminal.app, stock iTerm2) collapse ctrl+shift+z to plain
            # ctrl+z at the wire, making redo unreachable there. ctrl+y is
            # the C0 control EM (0x19) -- textual's `_ansi_sequences` maps
            # it to "ctrl+y" unconditionally, Kitty or not, so it survives
            # every terminal. Same composer-owns-keystroke conditions and
            # same always-consume shape as ctrl+shift+z above (including on
            # an empty redo stack, where `_console_composer_redo` is a
            # silent no-op) -- an addition alongside it, not a replacement.
            self._console_composer_redo()
            event.stop()
            event.prevent_default()
            return

    @on(ConsoleComposerBar.DraftChanged)
    def _handle_console_composer_draft_edit(
        self, event: ConsoleComposerBar.DraftChanged
    ) -> None:
        """React to a draft edit the composer made handling a Console key.

        TASK-3749, the inverse of what `on_key` used to do inline: instead
        of the screen editing the draft through the composer and then
        calling itself back, the composer announces the edit and the screen
        does the two screen-owned follow-ups here -- re-derive Workbench
        command readiness (which is also what opens/closes the
        slash-command popup), and, for a text-ADDING edit only, retire the
        first-run guidance. Deletions deliberately do not dismiss it: "the
        user has started composing" is a claim only an insertion makes, and
        that is precisely the split those keys had before the message
        existed.

        NOT to be confused with the sibling `_on_console_composer_draft_
        changed` (`Input.Changed` on the composer's hidden compatibility
        input), which fires on EVERY draft mutation from any source --
        `load_draft`, paste, dictation, a session restore -- and disarms
        the unknown-command escape. That signal is deliberately not reused
        here: syncing the Workbench and dismissing guidance off it would
        fire those on mutation paths that do neither today. (The two also
        must not share a method NAME: the second definition would silently
        replace the first in the class body, which is exactly how the
        first draft of this handler killed the disarm subscription.)

        `event.stop()` because this is a Console-composer-internal
        notification: nothing above the screen subscribes, so letting it
        bubble on would only cost a dispatch.

        Args:
            event: The composer's draft-change notification.
        """
        event.stop()
        self._sync_console_workbench_actions_from_draft()
        if event.is_insertion:
            self._dismiss_console_guidance()

    @on(ConsoleSelectionQuoteRequested)
    def _console_selection_quote_requested(
        self, event: ConsoleSelectionQuoteRequested
    ) -> None:
        """Insert a transcript selection into the composer as a block quote.

        Console selection phase 1: the transcript's floating menu posted
        this after its "Add to chat" action; the quote lands at the
        composer's caret (end of draft when unfocused). ``event.stop()``
        because nothing above this screen subscribes -- the transcript
        already consumed the originating ``AddToChat``.
        """
        event.stop()
        composer = self._console_composer_or_none()
        if composer is None:
            return
        composer.insert_quote(event.quote)
        if not event.quote.strip():
            # The row range was cleared while the menu was open (streaming
            # replace, reconciliation): ``insert_quote`` is a no-op on
            # blank input, and notifying "Added selection to composer"
            # for an insert that never happened would be a lie (final
            # review).
            return
        self.notify("Added selection to composer")

    @on(ConsoleSideChatRequested)
    def _console_side_chat_requested(self, event: ConsoleSideChatRequested) -> None:
        """Open the ephemeral side-chat modal about a transcript selection.

        Console selection phase 2: the transcript's floating menu posted
        this after its "More Details" / "Ask in Side Chat" action. More
        Details renders the configured prompt template with the capped
        quote and auto-sends on mount; Ask opens the modal freeform. The
        model resolves from ``[console] sidechat_model`` when set, else
        falls back to the active session's provider selection (the modal's
        identity line shows the request; the service applies the
        precedence). Nothing is persisted: the side-chat service streams
        through the provider gateway only, and the reply never leaves the
        modal. ``event.stop()`` because nothing above this screen
        subscribes -- the transcript already consumed the originating
        menu action.
        """
        event.stop()
        if not event.quote.strip():
            # Same blank-selection window as ``_console_selection_quote_
            # requested`` above: the row range was cleared while the menu
            # was open (streaming replace, reconciliation), so there is
            # nothing to ask about -- pushing the modal (or auto-sending
            # a contentless More Details prompt) would be a lie (T5 final
            # review).
            return
        sidechat_model = str(get_cli_setting("console", "sidechat_model", "") or "")
        template = str(
            get_cli_setting(
                "console",
                "sidechat_prompt_template",
                DEFAULT_CONSOLE_SIDECHAT_PROMPT_TEMPLATE,
            )
            or ""
        )
        auto_send_prompt = (
            render_prompt(template, event.quote)
            if event.mode == ConsoleSideChatRequested.MODE_MORE_DETAILS
            else None
        )
        self.app.push_screen(
            ConsoleSideChatModal(
                service=ConsoleSideChatService(self._ensure_console_provider_gateway()),
                provider_selection=self._build_console_provider_selection(),
                sidechat_model=sidechat_model,
                quote=event.quote,
                auto_send_prompt=auto_send_prompt,
            )
        )

    @on(ConsoleSelectionNoteRequested)
    def on_console_selection_note_requested(self, event: NoteRequested) -> None:
        event.stop()
        self._review_selection.request_selection_note(event.quote)

    @on(ConsoleSelectionFeedbackRequested)
    def on_console_selection_feedback_requested(self, event: FeedbackRequested) -> None:
        event.stop()
        self._review_selection.request_selection_feedback(
            event.action, event.quote, event.anchor_message_id
        )

    @on(ConsoleReviewNotesRequested)
    def on_console_review_notes_requested(
        self, event: ConsoleReviewNotesRequested
    ) -> None:
        """Open the review-notes modal for one message's annotations.

        task-18515 review-note management, task 3: posted by
        ``ConsoleAnnotationMarker.on_click`` and by
        ``ConsoleTranscript.action_open_review_notes`` (the ``n`` binding).
        Dispatches a worker because ``push_screen_wait`` requires an active
        worker context (see ``_console_selection_feedback_flow``'s identical
        note); ``event.stop()`` because nothing above this screen subscribes.

        ``_console_review_notes_inflight`` guards a rapid double-trigger the
        same way ``_console_selection_feedback_inflight`` guards the
        selection-feedback flow: the worker is deliberately NOT exclusive
        (see the flow's own docstring), so a re-trigger while a flow is
        still in flight is ignored here rather than stacking a second
        modal with its own independent DB-bound closures.
        """
        event.stop()
        if self._console_review_notes_inflight:
            # Rapid double-trigger (double marker-click or double-`n` before
            # the first worker's off-thread read resolves): one flow, one
            # modal, one set of DB-bound closures.
            return
        self._console_review_notes_inflight = True
        self.run_worker(
            self._console_review_notes_flow(event.anchor_message_id),
            group="console-review-notes",
            exit_on_error=False,
        )

    async def _console_review_notes_flow(self, anchor_message_id: str) -> None:
        """Resolve, browse, and (maybe) mutate one message's review notes.

        Runs as a worker (see the handler above); NOT ``exclusive=True`` for
        the same reason ``_console_selection_feedback_flow`` isn't --
        ``push_screen_wait``'s internal ``asyncio.shield`` protects the wait,
        not the already-mounted modal, from cancellation, so a superseding
        exclusive cancel would strand a live modal with no owner for its
        result. The handler's inflight flag is this flow's actual mutual
        exclusion; the ``finally`` below releases it on every exit path.

        The NATIVE anchor id is resolved to its persisted message id via the
        store's own messages (the inverse of ``_load_console_annotation_
        previews``'s ``native_by_persisted`` map), rows are read off-thread,
        and -- when at least one row matches -- the modal is pushed with
        ``on_edit``/``on_delete`` wrappers that write straight to SQLite
        (single-row indexed writes, the same synchronous-on-the-UI-thread
        precedent as ``_record_console_feedback_event``'s annotation write)
        and never raise. A change forces the existing discovery machinery to
        reload the preview map on its next sync tick.
        """
        try:
            controller = self._ensure_console_chat_controller()
            store = controller.store
            database = (
                getattr(store.persistence, "db", None) if store.persistence else None
            )
            if database is None:
                self.notify(
                    "Review notes are unavailable (no notes database).",
                    severity="warning",
                )
                return
            session = getattr(store, "_sessions", {}).get(store.active_session_id)
            conversation_id = getattr(session, "persisted_conversation_id", None)
            if not conversation_id:
                self.notify("No review notes for this message.", severity="warning")
                return
            conversation_id = str(conversation_id)
            persisted_by_native = {
                message.id: message.persisted_message_id
                for message in self._message._native_console_messages()
            }
            persisted_message_id = persisted_by_native.get(anchor_message_id)
            if persisted_message_id is None:
                self.notify("No review notes for this message.", severity="warning")
                return
            try:
                rows = await asyncio.to_thread(
                    database.get_transcript_annotations,
                    conversation_id,
                    str(persisted_message_id),
                )
            except Exception:
                logger.warning(
                    f"Console review notes: load failed for {conversation_id!r}",
                    exc_info=True,
                )
                self.notify("Could not load review notes.", severity="warning")
                return
            # The query already filtered by anchor; this keeps the flow
            # correct if a caller ever passes unfiltered rows.
            matching = [
                row
                for row in rows
                if str(row.get("message_id")) == str(persisted_message_id)
            ]
            if not matching:
                self.notify("No review notes for this message.", severity="warning")
                return
            rows_by_id = {str(row["annotation_id"]): row for row in matching}

            def _conversation_still_current() -> bool:
                """Guard the write against a conversation switch mid-modal.

                The flow captured ``conversation_id`` when it opened; a
                background switch while the modal is up would otherwise
                write into a conversation the user has left.
                """
                live = getattr(store, "_sessions", {}).get(store.active_session_id)
                return str(getattr(live, "persisted_conversation_id", "") or "") == (
                    conversation_id
                )

            def _edit_blocking(annotation_id: str, new_comment: str) -> bool:
                """Runs on a worker thread (see ``_on_edit``)."""
                row = rows_by_id.get(annotation_id)
                if row is None:
                    return False
                # Lost-update guard without a schema change: the row's
                # updated_at is the version we loaded. If someone else wrote
                # in the meantime, refuse rather than clobber silently.
                current = {
                    str(fresh["annotation_id"]): fresh
                    for fresh in database.get_transcript_annotations(
                        conversation_id, str(row["message_id"])
                    )
                }.get(annotation_id)
                if current is None or str(current.get("updated_at")) != str(
                    row.get("updated_at")
                ):
                    return False
                database.upsert_transcript_annotation(
                    conversation_id=conversation_id,
                    row_key=row["row_key"],
                    message_id=row["message_id"],
                    quote_text=row["quote_text"],
                    comment=new_comment,
                    annotation_id=annotation_id,
                )
                # Keep the snapshot's version current so a second edit in the
                # same modal session is not mistaken for a conflict.
                refreshed = {
                    str(fresh["annotation_id"]): fresh
                    for fresh in database.get_transcript_annotations(
                        conversation_id, str(row["message_id"])
                    )
                }.get(annotation_id)
                if refreshed is not None:
                    rows_by_id[annotation_id] = refreshed
                return True

            async def _on_edit(annotation_id: str, new_comment: str) -> bool:
                from tldw_chatbook.Utils.input_validation import validate_text_input
                from tldw_chatbook.Widgets.Console.console_selection import (
                    SELECTION_QUOTE_CAP,
                )

                # Boundary check through the shared module, matching the
                # create-note path. allow_html because a review note about
                # code legitimately contains markup-looking text and notes
                # render as plain text; size is the enforceable bound.
                if not validate_text_input(
                    new_comment, max_length=SELECTION_QUOTE_CAP, allow_html=True
                ):
                    self.notify("That note is too long to save.", severity="warning")
                    return False
                if not _conversation_still_current():
                    self.notify(
                        "The conversation changed; that note was not edited.",
                        severity="warning",
                    )
                    return False
                try:
                    # OFF-THREAD: a SQLite write on the UI event loop waits out
                    # the connection's 15s busy timeout under contention, with
                    # the interface frozen.
                    ok = await asyncio.to_thread(
                        _edit_blocking, annotation_id, new_comment
                    )
                except Exception:
                    logger.warning(
                        f"Console review notes: edit failed for {annotation_id!r}",
                        exc_info=True,
                    )
                    self.notify("Could not save the note.", severity="warning")
                    return False
                if not ok:
                    self.notify(
                        "That note changed elsewhere; reopen it to edit.",
                        severity="warning",
                    )
                return ok

            async def _on_delete(annotation_id: str) -> bool:
                if not _conversation_still_current():
                    self.notify(
                        "The conversation changed; that note was not deleted.",
                        severity="warning",
                    )
                    return False
                try:
                    return bool(
                        await asyncio.to_thread(
                            database.soft_delete_transcript_annotation, annotation_id
                        )
                    )
                except Exception:
                    logger.warning(
                        f"Console review notes: delete failed for {annotation_id!r}",
                        exc_info=True,
                    )
                    self.notify("Could not delete the note.", severity="warning")
                    return False

            changed = await self.app.push_screen_wait(
                ConsoleReviewNotesModal(
                    matching, on_edit=_on_edit, on_delete=_on_delete
                )
            )
            if changed:
                # Reload INLINE and push to the transcript, rather than
                # marking the conversation stale and waiting for the next
                # sync tick. Live verification caught the difference: the
                # transcript sync timer only runs while a run is active, so
                # the annotation-WRITE precedent appeared to refresh live
                # only because it dispatches a message (starting a run).
                # Edit/delete leave the app idle, so a tick-dependent
                # reload left a deleted note's marker on screen until the
                # user's next send. We are already in a worker here, so the
                # loader can simply be awaited.
                # Only if the user is still IN this conversation: a switch
                # while the modal was open would otherwise re-latch the old
                # id and paint its previews onto the new transcript for a
                # frame. The discovery tick reloads the live conversation.
                if _conversation_still_current():
                    self._review_selection.annotation_loaded_conversation = (
                        conversation_id
                    )
                    await self._review_selection._load_console_annotation_previews(
                        database, store, conversation_id
                    )
                    await self._sync_native_console_transcript()
        finally:
            # Every exit path -- empty-notes toast, modal cancel/dismiss, or
            # an error above -- releases the in-flight guard; a latched flag
            # would silently kill the feature after its first use (same
            # precedent as `_console_selection_feedback_flow`'s finally).
            self._console_review_notes_inflight = False

    def _recover_stuck_console_send_stash(
        self, stash: "ConsoleDraftStash | None"
    ) -> None:
        """Recover a keypress-captured draft `Button.Pressed` never consumed.

        Task 4 fix-round-2 (I3): the Enter handler's own no-op-press check
        (``send_button.disabled or not send_button.display`` right before
        ``.press()``) only catches the case where the button was ALREADY
        disabled/hidden at that instant. ``.press()`` itself just POSTS
        ``Button.Pressed`` for the message pump to deliver later -- if the
        button (or its composer) is pruned in the gap between that post and
        the pump actually delivering it, the message is dropped and
        ``handle_console_send_message``/``_send_console_message_from_
        visible_action`` -- the ONLY code that consumes ``_console_pending_
        send_stash`` -- never runs. Without this recovery, that leaves the
        stash slot permanently non-``None``, and the duplicate-send guard at
        the top of the ``"enter"`` branch swallows every subsequent Enter
        forever (D2's exact shape, via a narrower door than the no-op-press
        check alone closes).

        Scheduled once per send via ``set_timer`` right after ``.press()``;
        a no-op in the overwhelmingly common case where the Pressed handler
        already consumed the slot (or a later send's own stash superseded
        this one -- blocked from happening while this slot is still set by
        the duplicate guard itself, but checked by identity anyway as a
        cheap belt-and-suspenders).

        Args:
            stash: The exact stash object this watchdog was scheduled for.
        """
        if self._console_pending_send_stash is not stash:
            return
        logger.warning(
            "Console send Enter: pending stash was never consumed by the "
            "Pressed handler after {:.2f}s -- recovering the draft instead "
            "of leaving the duplicate-send guard latched shut.",
            self._CONSOLE_SEND_PENDING_STASH_WATCHDOG_SECONDS,
        )
        self._console_pending_send_stash = None
        self._restore_console_send_stash(stash)

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
        if composer.collapsed:
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

    def _dismiss_console_selection_menus_outside_transcript(
        self, target: object
    ) -> None:
        """Fold selection menus when a click lands outside every transcript.

        Console selection phase 1 (click-outside dismissal, screen half).
        Clicks that stay INSIDE a transcript are handled there (rows stop
        their own clicks; the transcript's ``on_click`` owns the in-area
        dismissal), so this only fires for clicks that bubbled up from
        elsewhere -- the composer, the control bar, the rail: the user
        moved on, and a menu left floating over the transcript folds with
        no side effects (ADR-068's dismiss contract). The ancestor walk is
        the guard: a transcript-area click that somehow reached this
        screen handler finds its ``ConsoleTranscript`` ancestor and is
        left alone. Menus whose removal is already scheduled (Textual
        marks them ``_pruning`` synchronously) are skipped so a repeated
        dismissal stays single-shot per menu.

        TASK-21119 (idle cost): this runs on BOTH ``on_mouse_down`` and
        ``on_click`` of the same physical press, on every press anywhere on
        the Console -- and it used to open with two full-screen ``query``
        walks (plus a third inside ``_remove_selection_menu``) on the
        largest DOM in the app: 6 walks per rail press, 3 per composer
        press, virtually always to find nothing. Both collections now come
        from the widgets' own registries (constructor-registered, so they
        can never miss a mounted node) with attachment re-derived from the
        live DOM, and the pass returns before any of it when there is
        nothing to dismiss. "Nothing to dismiss" deliberately covers more
        than a mounted menu: keyboard-selection mode arms a highlight with
        no menu, and that highlight must still fold on a click elsewhere.

        Args:
            target: The clicked widget (``event.widget``/``event.control``).
        """
        # The ancestor walk stays FIRST (review round 1, MINOR-2): it is the
        # cheapest exit and it owns the drag hot path -- every press inside a
        # transcript returns here having touched neither registry.
        node: object = target
        while node is not None:
            if isinstance(node, (ConsoleTranscript, ConsoleSelectionMenu)):
                return  # the transcript/menu own their in-area interaction
            if getattr(node, "id", None) == "console-message-more-menu":
                return  # the transcript/menu own their in-area interaction
            node = getattr(node, "parent", None)
        menus = selection_menus_on_screen(self)
        more_menus = message_more_menus_on_screen(self)
        # Only transcripts the cleanup would actually change; on the rest,
        # all three steps below are provable no-ops (see
        # ``ConsoleTranscript.has_pending_selection_ui``).
        transcripts = [
            transcript
            for transcript in console_transcripts_on_screen(self)
            if transcript.has_pending_selection_ui
        ]
        if not menus and not more_menus and not transcripts:
            return  # the common case: no selection UI anywhere on the screen
        # Menus mount on the screen now; route the dismissal through every
        # transcript's centralized selection-UI cleanup (clears highlight +
        # manager state), then remove any stragglers (e.g. menus mounted by
        # harnesses without a transcript ancestor).
        for transcript in transcripts:
            transcript._remove_selection_menu()
            transcript.selection_manager.cancel()
            transcript._selection_origin_row = None
        for menu in menus:
            if not getattr(menu, "_pruning", False):
                menu.remove()
        dismiss_message_more_menus(more_menus)

    def on_mouse_down(self, event: MouseDown) -> None:
        """Dismiss selection UI before descendants may consume the click."""
        target = getattr(event, "widget", None) or getattr(event, "control", None)
        if target is None:
            try:
                target, _offset = self.get_widget_at(event.screen_x, event.screen_y)
            except Exception:
                target = None
        self._dismiss_console_selection_menus_outside_transcript(target)

    def on_click(self, event: Click) -> None:
        """Reset pending paste unfurl confirmation when clicking outside the token."""
        target = getattr(event, "widget", None) or getattr(event, "control", None)
        self._dismiss_console_selection_menus_outside_transcript(target)
        if getattr(target, "id", None) == "console-command-visible-text":
            return
        if getattr(target, "id", None) == "console-rail-system-line":
            event.stop()
            self.run_worker(self._open_console_system_prompt_editor(), exclusive=False)
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
        """Push sidebar control values back into the compact shell bar.

        task-16474: this programmatic sync no longer writes the
        ``_console_control_provider``/``_console_control_model`` mirrors --
        those track genuine user selections only (the mirrors outrank
        ``chat_defaults`` when fresh session defaults are derived, so
        ambient writes decided the provider of the next session). The bar's
        displayed values and the session settings replacement below are
        unchanged.
        """
        updates: Dict[str, str] = {}
        if provider is not None:
            updates["provider"] = provider
        if model is not None:
            updates["model"] = model
        if temperature is not None:
            updates["temperature"] = temperature

        if not updates:
            return

        try:
            settings = self._session._ensure_active_console_session_settings()
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
                self._session._replace_active_console_session_settings(next_settings)
        except Exception as e:
            logger.debug(
                f"Unable to sync compact controls into Console session settings: {e}"
            )

        compact_bar = self._get_compact_model_bar()
        if compact_bar:
            compact_bar.sync_from_sidebar(**updates)
        else:
            logger.debug("No compact model bar available for reverse sync")
        self._sync_console_control_bar()

    # NOTE (task-247, perf): there used to be an on_screen_suspend() override
    # here that called self.save_state() again and discarded the result.
    # app.py already calls save_state() explicitly before switching screens
    # away from Console and offers that return value to ScreenStateStore; the
    # second call here was pure waste (a full O(sessions x messages)
    # native-console serialization) on every tab switch away from Console.
    # Removed rather than left as a no-op so it doesn't shadow a future
    # base-class implementation.

    def on_screen_resume(self) -> None:
        """Called when returning to this screen."""
        logger.debug("Chat screen resuming")
        # task-17652: a Settings change to the status-row position must land
        # on this cached screen without a recompose.
        apply_status_chips_position(self)
        # task-15475: consume the mount's one-shot token. On the FIRST visit
        # this resume is the mount's own, and on_mount already dispatched the
        # skill-candidate worker and scheduled the task-resume sync; running
        # them again here just doubled the work. Consumed (not merely read),
        # so every subsequent resume refreshes normally.
        mount_already_refreshed = self._console_mount_visit_refreshed
        self._console_mount_visit_refreshed = False
        ordered_resume_active = self._resume_navigation_startup_in_progress
        # task-18310: reconcile the Console session against the registry's
        # active workspace on every ordinary resume, including the mount's
        # own. An explicit saved-chat resume is the sole exception: its
        # canonical opener activates the target workspace last, while this
        # reconciliation can switch sessions and schedule an intermediate
        # native sync. Every in-Console activation path (Alt+W
        # switcher, the shared create modal, conversation-browser row-open)
        # already keeps the registry and the store's active session in
        # lockstep, so the common case is an O(1) early exit; the mount
        # path itself never reconciles against the registry, and the store
        # is app-level (it can carry a session that predates this screen's
        # first mount), so skipping it here on the mount's own resume would
        # leave that gap uncaught. Cross-screen activation (Settings'
        # create-modal/"Set active" button, Library's create-workspace
        # flow) only updates the registry -- this is the seam that repairs
        # the resulting drift. See
        # `ConsoleWorkspaceController._reconcile_console_session_with_registry`.
        if not ordered_resume_active:
            try:
                self._workspace._reconcile_console_session_with_registry()
            except Exception:
                logger.opt(exception=True).debug(
                    "Unable to reconcile Console session with registry-active workspace"
                )
        if not ordered_resume_active:
            self._session.consume_pending_console_first_chat_intent()
        # Re-evaluate setup-card/model readiness before touching focus. Some
        # recovery flows (e.g. certain providers' API-key recovery) navigate to
        # the full Settings screen and back rather than completing setup via
        # the in-Console settings modal callback, so the setup modal's blocking
        # state can be stale by the time this screen resumes. Without this,
        # `_restore_console_workbench_focus` below would just re-apply the
        # stale block and the modal could stick even after setup completed
        # elsewhere.
        self._sync_console_transcript_guidance()
        if not mount_already_refreshed:
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
        if not ordered_resume_active:
            self.set_timer(0.15, self._consume_pending_console_prompt_insert)
            self.set_timer(0.15, self.consume_pending_console_provider_intent)
            # PR3a-2 Task 4: mirrors the on_mount claim -- a completion staged
            # while the user was on another screen is claimed on resume too.
            self.set_timer(
                0.15,
                self._fleet.consume_pending_console_fleet_completion,
            )
            self.call_after_refresh(self._restore_console_workbench_focus)
        if not ordered_resume_active:
            repair_dispatched = self._consume_pending_console_roleplay_repair()
            if (
                not repair_dispatched
                and not self._consume_pending_console_identity_refresh()
            ):
                self._dispatch_active_console_roleplay_refresh()
        if not mount_already_refreshed:
            self.run_worker(
                self._skill._refresh_console_skill_candidates(), exclusive=False
            )
        # Textual's MRO dispatch also invokes BaseAppScreen's shared reconciliation;
        # this handler extends that resume event with Console-owned replay work.

    def set_task_resume_state(self, task_state: TaskResumeState) -> None:
        """Update native Console task-resume state and refresh its cards."""
        self._task_resume_state = task_state
        self.sync_task_resume_state()

    def sync_task_resume_state(self) -> None:
        """Push native Console task-resume state into its task cards."""
        try:
            task_cards = self.query_one("#console-task-surface", ChatTaskCards)
            task_cards.sync_state(
                self._task_resume_state,
                operation_rows=self._watchlists_operation_rows,
            )
        except QueryError:
            pass

    def _follow_console_watchlists_operations(
        self, operation_ids: tuple[str, ...]
    ) -> None:
        """Follow canonical durable receipts without retaining tool payloads."""
        followed = tuple(
            dict.fromkeys(
                (*self._task_resume_state.followed_watchlists_operations, *operation_ids)
            )
        )
        self._task_resume_state = replace(
            self._task_resume_state,
            followed_watchlists_operations=followed,
        )
        for operation_id in followed:
            self._watchlists_operation_rows.setdefault(
                operation_id,
                {"id": operation_id, "status_detail": "queued"},
            )
        self.sync_task_resume_state()
        if followed and getattr(self, "is_mounted", False):
            self.run_worker(
                self._poll_console_watchlists_operations(),
                group="console-watchlists-operation-follow",
                exclusive=True,
            )

    async def _poll_console_watchlists_operations(self) -> None:
        """Refresh safe receipt projections while this Console is mounted."""
        active = {"queued", "running", "generating"}
        while getattr(self, "is_mounted", False):
            operation_ids = self._task_resume_state.followed_watchlists_operations
            if not operation_ids:
                return
            try:
                rows = await asyncio.to_thread(
                    self._read_console_watchlists_operation_rows,
                    operation_ids,
                )
            except Exception:  # noqa: BLE001 - retain last durable projection
                rows = {}
            self._watchlists_operation_rows.update(rows)
            self.sync_task_resume_state()
            if not any(
                str(
                    self._watchlists_operation_rows.get(operation_id, {}).get(
                        "status_detail"
                    )
                    or ""
                ).casefold()
                in active
                for operation_id in operation_ids
            ):
                return
            await asyncio.sleep(2.0)

    def _read_console_watchlists_operation_rows(
        self, operation_ids: tuple[str, ...]
    ) -> dict[str, dict[str, Any]]:
        """Read only bounded, tool-shaped operation metadata from local SQLite."""
        from tldw_chatbook.Tools.watchlists_tool_service import (
            WatchlistsToolService,
        )

        service = WatchlistsToolService(
            db_resolver=lambda: getattr(self.app_instance, "subscriptions_db", None),
            runtime_source_loader=lambda: "local",
        )
        rows: dict[str, dict[str, Any]] = {}
        for operation_id in operation_ids:
            payload = json.loads(
                service.get_operation_status({"operation_id": operation_id})
            )
            operation = payload.get("operation")
            if payload.get("status") == "ok" and isinstance(operation, dict):
                rows[operation_id] = operation
        return rows

    @on(WatchlistsOperationCard.StopFollowingRequested)
    def on_watchlists_operation_stop_following(
        self, event: WatchlistsOperationCard.StopFollowingRequested
    ) -> None:
        """Remove one receipt card without cancelling its durable operation."""
        controller = self._console_chat_controller
        if controller is not None:
            controller.unfollow_watchlists_operation(event.operation_id)
        followed = tuple(
            operation_id
            for operation_id in self._task_resume_state.followed_watchlists_operations
            if operation_id != event.operation_id
        )
        self._watchlists_operation_rows.pop(event.operation_id, None)
        self.set_task_resume_state(
            replace(
                self._task_resume_state,
                followed_watchlists_operations=followed,
            )
        )

    @on(WatchlistsOperationCard.InspectRequested)
    def on_watchlists_operation_inspect(
        self, event: WatchlistsOperationCard.InspectRequested
    ) -> None:
        """Open the matching Watchlists history destination."""
        context: dict[str, object] = {
            "section": event.destination,
            "backend": "local",
        }
        if event.destination == "runs":
            context["run_id"] = event.operation_id
        elif event.destination == "artifacts":
            context[WATCHLISTS_NAV_CONTEXT_BRIEFING_ID] = event.operation_id
        self.post_message(
            NavigateToScreen(
                TAB_WATCHLISTS_COLLECTIONS,
                screen_context=context,
            )
        )

    @on(WatchlistsOperationCard.CancelRequested)
    def on_watchlists_operation_cancel(
        self, event: WatchlistsOperationCard.CancelRequested
    ) -> None:
        """Cancel a source check only when its local domain supports it."""
        if not event.operation_id.startswith("local:watchlist_run:"):
            return
        service = getattr(self.app_instance, "local_watchlists_service", None)
        if service is None:
            return
        run_id = int(event.operation_id.rsplit(":", 1)[-1])
        self.run_worker(service.cancel_run(run_id), exclusive=False)

    @on(WatchlistsOperationCard.RetryRequested)
    def on_watchlists_operation_retry(
        self, event: WatchlistsOperationCard.RetryRequested
    ) -> None:
        """Retry a failed receipt through the same app-owned coordinator."""
        self.run_worker(
            self._retry_console_watchlists_operation(event.operation_id),
            exclusive=False,
        )

    async def _retry_console_watchlists_operation(self, operation_id: str) -> None:
        coordinator = getattr(
            self.app_instance, "watchlists_operation_coordinator", None
        )
        operation = self._watchlists_operation_rows.get(operation_id, {})
        if coordinator is None:
            return
        if operation_id.startswith("local:watchlist_run:"):
            source = operation.get("source")
            source_id = source.get("id") if isinstance(source, Mapping) else None
            if (
                not isinstance(source_id, str)
                or re.fullmatch(r"local:subscription:[1-9][0-9]*", source_id)
                is None
            ):
                return
            receipts = await coordinator.accept_checks(
                [int(source_id.rsplit(":", 1)[-1])]
            )
            followed = tuple(
                f"local:watchlist_run:{int(receipt['run_id'])}"
                for receipt in receipts
            )
        else:
            collection = operation.get("collection")
            collection_id = (
                collection.get("id") if isinstance(collection, Mapping) else None
            )
            if (
                not isinstance(collection_id, str)
                or re.fullmatch(r"local:watchlist:[1-9][0-9]*", collection_id)
                is None
            ):
                return
            receipt = await coordinator.accept_briefing(
                int(collection_id.rsplit(":", 1)[-1])
            )
            followed = (f"local:briefing:{int(receipt['id'])}",)
        self._follow_console_watchlists_operations(followed)

    def _set_console_pending_approval(self, approval: Dict[str, Any] | None) -> None:
        """Set or clear the native Console's pending MCP approval batch."""
        self.set_task_resume_state(
            replace(self._task_resume_state, pending_approval=approval)
        )

    def _park_console_approval(self, session_id: str) -> None:
        """PA-T9 (parked background approvals): badge a NON-viewed session's
        pending approval round without mounting the (singleton) approval
        card, and fire the one-per-round toast.

        UI-thread bridge target for ``ConsoleChatController.
        request_mcp_approvals``' park branch (invoked via ``app_instance.
        call_from_thread`` exactly once per parked round -- the round's
        session differs from the store's active session at round-start).

        TASK-910: also the shared UI-thread bridge target for
        ``request_skill_install_confirm``'s and ``request_skill_script_
        confirm``'s OWN park branches -- one badge/toast seam for all three
        approval-like bridges, per the train's "same marker/toast
        machinery" convention, rather than a bespoke copy per bridge.
        Deliberately does NOT touch ``task_resume_state``/``set_task_
        resume_state`` -- that slot is reserved for whichever session is
        actually being viewed (``_set_console_pending_approval`` and the
        `ConsoleSkillController` pending-state setters); parking must never
        steal the mounted card
        out from under the session the user is currently looking at. The
        controller's own ``_parked_approval_payloads``/``_parked_skill_
        install_payloads``/``_parked_skill_script_payloads`` maps
        (populated by each bridge before this fires) are what
        ``ConsoleChatController.switch_session``/``new_session``/
        ``close_session`` later read to mount the SAME payload once the
        user actually visits ``session_id``.

        Also usable directly as a test seam to drive the park path without
        a live worker thread/round -- setting the badge itself here (via
        the deprecated ``set_run_pending_approval`` shim, ONLY when no real
        round is registered yet) is what makes that safe: this method is
        fully self-contained.

        TASK-1050 (Defect A): the owning bridge (``request_mcp_approvals``/
        ``request_skill_install_confirm``/``request_skill_script_confirm``)
        already registers THIS round's own real round/request id via
        ``add_pending_round`` moments before invoking this park callback --
        by the time this runs, ``has_pending_approval_round(session_id)``
        is normally already ``True``. This method has no round id of its
        own to register (its public contract, wired as ``ConsoleChatController
        .park_pending_approval``, is a single-arg ``Callable[[str], None]``
        -- several tests wire it directly to a plain single-arg collector),
        so it must NOT unconditionally stamp the deprecated boolean shim:
        doing so would register the shim's synthetic sentinel round id
        ALONGSIDE the real one, and the real round's own teardown
        (``discard_pending_round``) would then leave that sentinel behind,
        leaking a stale NEEDS_APPROVAL badge past the round's actual
        resolution. Only falls back to the shim when no round is
        registered yet -- i.e. when this method is used standalone (the
        test-seam usage the docstring above describes).

        TASK-1141 (UAT F2): this callback's "once per round" guarantee
        previously relied ENTIRELY on the structural assumption that each
        owning bridge invokes it exactly once per round -- true for a
        single, race-free `request_mcp_approvals`/`request_skill_install_
        confirm`/`request_skill_script_confirm` call, but the callback
        itself carried no memory of which round it had already announced.
        Live UAT observed a duplicate toast for an unchanged, still-parked
        round: with a background session already parked (toast shown), a
        DIFFERENT viewed session's own run completing re-fired the exact
        same toast text for the backgrounded round, even though nothing
        about that round had changed. Exhaustively tracing the suspects
        (`_set_run_state`'s COMPLETED branch, `_finalize_agent_*`,
        `switch_session`/`_remount_parked_*`'s re-derive step, the
        unvisited-marker stamp) found none of them invoke this callback a
        second time for the SAME round under single-threaded/synchronous
        conditions -- but nothing in this method itself prevented a
        second, differently-triggered invocation (e.g. a re-marshal racing
        `call_from_thread`, or any future caller of the shared park seam)
        from re-announcing a round whose identity hasn't changed. Rather
        than depend on every CALLER staying single-invocation-per-round
        forever, this method now keys its own idempotency directly off the
        round/request id(s) the owning controller is CURRENTLY retaining
        for `session_id` (`_parked_approval_payloads`/`_parked_skill_
        install_payloads`/`_parked_skill_script_payloads` -- the exact
        maps `switch_session` re-derives the mounted card from), via
        `_current_park_round_ids`. A round/request id already recorded in
        `_console_toasted_park_round_ids` is a re-announcement of a round
        this screen already toasted for -- silently absorbed. A round/
        request id NOT yet recorded (a genuinely new round, even for a
        session that already has an outstanding one) still toasts, per
        spec (parking must never go silent just because a SIBLING round is
        also live). When none of the three maps carry any id for
        `session_id` yet (the standalone test-seam usage described above,
        or a caller that races ahead of the owning bridge's own
        `_parked_*_payloads` write), there is no identity to key on, so
        this falls back to the pre-TASK-1141 unconditional toast --
        preserving every existing direct-call test's behavior.

        TASK-1141 review round 1: the live-map lookup above alone is
        blind to a re-invocation that arrives AFTER the round's own
        teardown -- every owning bridge's `finally` pops its round out of
        `_parked_*_payloads` once resolved (`request_mcp_approvals`'
        docstring on that pop explains why), so a STRAY re-invocation
        landing post-teardown finds all three maps empty and, pre-review,
        fell straight into the "no identity to key on" unconditional-toast
        branch above -- exactly the live-reproduced gap this review round
        closes. `_console_last_parked_round_ids` remembers the most recent
        NON-empty snapshot `_current_park_round_ids` ever returned for
        `session_id`; when the live lookup now comes back empty, that
        remembered snapshot is consulted instead: if every id in it is
        already in `_console_toasted_park_round_ids`, this is a
        post-teardown re-announcement of an already-surfaced round --
        absorbed, same as the still-live case. Only when NO snapshot was
        ever recorded for `session_id` (this screen has truly never seen a
        live round for it -- the standalone test-seam case) does the
        unconditional-toast fallback still apply.

        Args:
            session_id: The parked round's OWNING session.
        """
        controller = self._console_chat_controller
        if controller is None:
            return
        if not controller.has_pending_approval_round(session_id):
            controller.set_run_pending_approval(session_id, True)
        current_round_ids = self._current_park_round_ids(controller, session_id)
        if current_round_ids:
            # Remember this as the most recent LIVE snapshot for
            # `session_id`, unconditionally -- consulted below by a later
            # invocation that arrives after this round's own teardown has
            # already emptied every live map (review round 1).
            self._console_last_parked_round_ids[session_id] = current_round_ids
            new_round_ids = current_round_ids - self._console_toasted_park_round_ids
            if not new_round_ids:
                # Every round/request id currently parked for this session
                # has already been toasted -- this invocation is a
                # re-announcement (re-marshal/re-derive/re-park) of round(s)
                # already surfaced, not a genuinely new one.
                return
            self._console_toasted_park_round_ids.update(current_round_ids)
        else:
            # Nothing is currently live for `session_id` in any of the
            # three bridges' payload maps. Fall back to the last snapshot
            # this screen ever saw live for it (review round 1): if every
            # id in that snapshot was already toasted, this is a stray
            # post-teardown re-invocation for an already-surfaced round --
            # absorbed. No snapshot at all means this screen has never
            # parked a round for `session_id` (the standalone test-seam
            # usage this method's own docstring describes), so it falls
            # through and toasts, preserving that pre-TASK-1141 behavior.
            last_round_ids = self._console_last_parked_round_ids.get(session_id)
            if (
                last_round_ids
                and last_round_ids <= self._console_toasted_park_round_ids
            ):
                return
        session_title, workspace_name = (
            self._workspace._console_session_title_and_workspace_name(
                controller, session_id
            )
        )
        session_title = escape_markup(
            sanitize_character_display_label(session_title, max_characters=500)
        )
        workspace_name = escape_markup(
            sanitize_character_display_label(workspace_name, max_characters=500)
        )
        self.app_instance.notify(
            f"Agent in {session_title} ({workspace_name}) needs approval."
        )

    @staticmethod
    def _current_park_round_ids(
        controller: ConsoleChatController, session_id: str
    ) -> frozenset[str]:
        """Return every round/request id CURRENTLY retained for ``session_id``.

        TASK-1141: namespaced per bridge (``"mcp:"``/``"install:"``/
        ``"script:"``) since the three bridges mint their ids
        independently -- two different bridges could theoretically mint
        the same raw UUID by construction, however astronomically
        unlikely, and namespacing costs nothing. Reads the SAME three
        retained-payload maps ``switch_session``/``_remount_parked_
        skill_install``/``_remount_parked_skill_script`` already treat as
        the single source of truth for "what round is this session's card
        showing right now" -- deliberately not a separate/parallel piece
        of state that could itself drift from theirs.

        Args:
            controller: The owning Console chat controller.
            session_id: The session to look up.

        Returns:
            A ``frozenset`` of namespaced round/request ids, empty when
            none of the three bridges currently retain a payload for
            ``session_id``.
        """
        ids: set[str] = set()
        for prefix, store, id_key in (
            ("mcp", controller._parked_approval_payloads, "round_id"),
            ("install", controller._parked_skill_install_payloads, "request_id"),
            ("script", controller._parked_skill_script_payloads, "request_id"),
        ):
            # PR0 (task-15661): `_parked_approval_payloads` is keyed by
            # ROUND now, so a session can retain SEVERAL payloads at once
            # and a `.get(session_id)` would find none of them. Scanning by
            # the payload's own `session_id` reads either key shape, which
            # also means the two skill maps needed no change here when they
            # were re-keyed. Returning EVERY live id (not just the
            # session's mounted head) is what this method already promises
            # and what the caller's dedupe needs: a genuinely new sibling
            # round must still toast.
            for payload in controller._session_round_payloads(store, session_id):
                round_id = payload.get(id_key)
                if round_id:
                    ids.add(f"{prefix}:{round_id}")
        return frozenset(ids)

    def _notify_console_run_outcome(
        self, session_id: str, status: ConsoleRunStatus
    ) -> None:
        """Task 10 (background completion toasts): one toast for a
        NON-viewed session's run finishing (COMPLETED) or failing (FAILED).

        UI-thread bridge target for ``ConsoleChatController.
        notify_run_outcome``, invoked DIRECTLY (never via ``app_instance.
        call_from_thread``, unlike ``_park_console_approval`` above) from
        ``_set_run_state``'s once-guarded non-active terminal branch --
        every terminal ``_set_run_state`` call already runs on the main
        event-loop thread (worker-thread agent runs resume here only after
        ``await asyncio.to_thread(...)`` returns in ``_run_agent_reply``),
        so no thread marshaling is needed. Shares ``_console_session_
        title_and_workspace_name`` (which itself uses ``_console_
        workspace_display_name``) with ``_park_console_approval`` above --
        one resolver, not a byte-duplicated copy (fix wave, rider 6).
        The viewed session's own terminal transition is visible live in its
        transcript and never reaches this method (``_set_run_state`` only
        calls it from the non-active branch).

        Args:
            session_id: The run's OWNING session (non-active at call time).
            status: ``ConsoleRunStatus.COMPLETED`` or ``.FAILED`` -- the
                only two statuses ``_set_run_state`` ever calls this with.
        """
        controller = self._console_chat_controller
        if controller is None:
            return
        session_title, workspace_name = (
            self._workspace._console_session_title_and_workspace_name(
                controller, session_id
            )
        )
        session_title = escape_markup(
            sanitize_character_display_label(session_title, max_characters=500)
        )
        workspace_name = escape_markup(
            sanitize_character_display_label(workspace_name, max_characters=500)
        )
        verb = "finished" if status is ConsoleRunStatus.COMPLETED else "failed"
        self.app_instance.notify(f"Agent in {session_title} ({workspace_name}) {verb}.")

    def _notify_console_run_failure(self, visible_copy: str) -> None:
        """task-2154.16 (FB-05): one error toast for the VIEWED session's run
        failing.

        UI-thread bridge target for ``ConsoleChatController.
        notify_run_failure``, invoked directly from ``_set_run_state``'s
        once-guarded active-session FAILED branch (same main-loop guarantee
        as ``_notify_console_run_outcome`` above). The copy is the run's
        ``visible_copy`` -- the same text as the transcript system row -- so
        the toast and the row never disagree.

        Args:
            visible_copy: The failed run's user-facing copy (e.g.
                "Provider stream failed: unexpected provider error (...)").
        """
        self.app_instance.notify(visible_copy, severity="error")

    @on(ChatApprovalCard.ApprovalDecided)
    def handle_console_approval_decided(
        self, event: ChatApprovalCard.ApprovalDecided
    ) -> None:
        """Forward the user's batch decisions to the controller's waiting worker thread.

        Task 9 fix round 1: forwards ``event.round_id`` too -- this message
        is delivered asynchronously (it can arrive after a `switch_session`
        already moved the active session elsewhere), so
        `resolve_pending_approval` must resolve the round the card was
        actually showing, never "whichever session is active right now".
        """
        event.stop()
        controller = self._console_chat_controller
        if controller is not None:
            controller.resolve_pending_approval(
                event.decisions, round_id=event.round_id
            )

    @on(SkillInstallConfirmCard.InstallDecided)
    def handle_console_skill_install_decided(self, event: Any) -> None:
        event.stop()
        self._skill.handle_console_skill_install_decided(
            event.allow, request_id=event.request_id
        )

    @on(SkillScriptConfirmCard.ScriptDecided)
    def handle_console_skill_script_decided(self, event: Any) -> None:
        event.stop()
        self._skill.handle_console_skill_script_decided(
            event.allow, event.remember, request_id=event.request_id
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button events at the screen level.
        This ensures buttons work properly with screen-based navigation.
        """
        button_id = event.button.id

        if event.button.has_class("console-transcript-library-activity"):
            event.stop()
            self._library_activity.open_selected(event.button)
            return

        # Log for debugging
        logger.info(f"ChatScreen on_button_pressed called with button: {button_id}")

        if button_id == "console-composer-menu":
            event.stop()
            await self._open_console_composer_menu()
            return
        if button_id == "console-prompt-improvement-undo":
            event.stop()
            self._undo_console_prompt_improvement()
            return
        if button_id == "console-prompt-improvement-review":
            event.stop()
            self._open_console_prompt_comparison()
            return
        if button_id == "console-send-message":
            await self.handle_console_send_message(event)
            return
        if button_id == "console-dictation":
            event.stop()
            self._dictation._handle_console_dictation_button()
            return
        if button_id in {
            "console-stop-generation",
            "console-collapsed-stop-generation",
        }:
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
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            return
        if button_id == "console-agent-view-full-log":
            event.stop()
            self._agent._open_console_agent_run_log_viewer()
            return
        if button_id == "console-new-chat-tab":
            event.stop()
            await self._session._create_native_console_session_from_active_context()
            return
        if button_id and button_id.startswith(
            "console-conversation-browser-section-toggle-"
        ):
            event.stop()
            self._workspace._toggle_console_conversation_browser_section(
                str(getattr(event.button, "group_id", "") or "").strip()
            )
            return
        if button_id and button_id.startswith(
            "console-conversation-browser-group-toggle-"
        ):
            event.stop()
            self._workspace._toggle_console_conversation_browser_group(
                str(getattr(event.button, "group_id", "") or "").strip()
            )
            return
        if button_id and button_id.startswith("console-conversation-actions-"):
            # TASK-23200: the asterisk opens the row's action menu rather
            # than silently toggling a favourite, so favouriting, status,
            # archive, rename and delete all reach the user from one place.
            event.stop()
            self._open_console_conversation_action_menu(event.button)
            return
        # NOTE: the `console-workspace-conversations-toggle` branch that stood
        # here was deleted in wave 4. Commit 3b0374479 removed the only button
        # carrying that id; the string survives only as a CSS class on toggles
        # whose ids are `console-conversation-browser-{section,group}-toggle-*`,
        # and those take their own branches below. The body was dead twice over:
        # it also required `state.conversation_browser is None`, a state that
        # same commit retired. `Tests/UI/test_console_button_routing.py` pins the
        # id's absence so it cannot quietly come back as a branch nobody reaches.
        if button_id == "console-new-workspace-conversation":
            event.stop()
            await self._session._create_native_console_session_from_active_context()
            return
        if button_id == "console-workspace-conversation-search-clear":
            event.stop()
            self._workspace.clear_console_conversation_browser_search()
            return
        if button_id == "console-workspace-search-retry":
            event.stop()
            self.run_worker(
                self._workspace.retry_workspace_tree_search(),
                group="console-workspace-tree-search",
                exclusive=True,
            )
            return
        if button_id and button_id.startswith("console-workspace-conversation-"):
            event.stop()
            await self._workspace.open_console_workspace_conversation(
                str(getattr(event.button, "conversation_id", "") or ""),
                row_key=str(getattr(event.button, "row_key", "") or ""),
            )
            return
        if button_id and button_id.startswith("console-close-session-tab-"):
            event.stop()
            self._session.start_close_console_session_tab(
                button_id.removeprefix("console-close-session-tab-")
            )
            return
        if button_id and button_id.startswith("console-session-tab-"):
            event.stop()
            await self._session._handle_console_session_tab_press(
                button_id.removeprefix("console-session-tab-")
            )
            return
        if button_id and button_id.startswith("console-message-action-"):
            handled = await self.handle_console_message_action(event)
            if handled:
                return

    def watch_sidebar_state(self, new_state: dict) -> None:
        """Debounce persistence when sidebar state changes.

        task-15470: this used to call `_save_sidebar_state()` directly --
        synchronous open+parse+rewrite of `ui_state.toml` on the event loop,
        once per `Collapsible.Toggled`. Now it only marks the state dirty and
        (re)arms one debounce timer; a burst of toggles collapses into a
        single write, dispatched off the loop by
        `_flush_sidebar_state_after_debounce`. `on_unmount` force-flushes any
        pending write so a toggle immediately followed by quit is not lost.
        """
        self._schedule_sidebar_state_save()

    def _schedule_sidebar_state_save(self) -> None:
        """Mark the sidebar state dirty and (re)arm the debounce timer.

        The single scheduling point -- `watch_sidebar_state` and any direct
        caller that mutates `ui_state.collapsible_states` without going
        through the reactive (e.g. a bulk reset that may reassign an
        already-`{}` `sidebar_state`, which the reactive would then treat as
        a no-op and never call the watcher for) both route through here so
        a pending write is unconditionally scheduled.
        """
        self._sidebar_state_dirty = True
        if self._sidebar_state_save_timer is not None:
            self._sidebar_state_save_timer.stop()
        self._sidebar_state_save_timer = self.set_timer(
            SIDEBAR_STATE_SAVE_DEBOUNCE_SECONDS,
            self._flush_sidebar_state_after_debounce,
        )

    def _flush_sidebar_state_after_debounce(self) -> None:
        """Debounce timer callback: hand the actual write to a worker."""
        self._sidebar_state_save_timer = None
        self._sidebar_state_persist_worker = self.run_worker(
            self._persist_sidebar_state_off_loop(),
            exclusive=True,
            group="sidebar-state-persist",
        )

    async def _persist_sidebar_state_off_loop(self) -> None:
        """Write `ui_state.toml` on a worker thread, off the event loop.

        Snapshots `self.ui_state` here, on the main thread, before handing
        the write to `to_thread` -- a further toggle can still arrive and
        mutate `collapsible_states` while this write is in flight, and it
        must not race the worker thread's read of that same dict.

        Clears `_sidebar_state_dirty` immediately after taking the
        snapshot, NOT after the write completes (review round,
        task-15470): the awaited `to_thread` call below yields to the
        event loop, and a further toggle can land while this write is
        still in flight. Clearing dirty only after the write finished
        would blindly stamp it False again on completion -- clobbering
        the True a mid-flight toggle had just set -- so a quit landing
        before that toggle's own new debounce timer fires would see
        `dirty=False` and lose it. Clearing right here instead means the
        dirty flag always answers "is there a toggle newer than the
        snapshot this worker is holding", which a mid-flight toggle
        correctly flips back to True.
        """
        snapshot = self._sidebar_state_snapshot()
        self._sidebar_state_dirty = False
        await asyncio.to_thread(self._write_sidebar_state_snapshot, snapshot)

    async def _flush_sidebar_state_now(self) -> None:
        """Force-flush a pending sidebar-state write (unmount/quit path).

        Cancels any pending debounce timer and writes off the loop via
        `to_thread` so the screen never unmounts with an unpersisted toggle
        -- the AC #2 flush-on-quit guarantee. If a debounced write is
        already in flight (the timer fired moments before quit), this waits
        for it rather than dispatching a second writer against the same
        file -- `_write_sidebar_state_snapshot` does an unlocked
        read-modify-write of `ui_state.toml`, so two concurrent writers
        could interleave.
        """
        if self._sidebar_state_save_timer is not None:
            self._sidebar_state_save_timer.stop()
            self._sidebar_state_save_timer = None
        worker = self._sidebar_state_persist_worker
        if worker is not None and not worker.is_finished:
            try:
                await worker.wait()
            except Exception as error:
                logger.error(
                    "Pending sidebar-state write failed: {}", type(error).__name__
                )
            # Falls through to the dirty re-check below (review round,
            # task-15470) rather than returning here: a toggle can land
            # while THIS await was in flight, re-dirtying the state after
            # the awaited worker already took its own snapshot. Returning
            # unconditionally after the wait would silently drop it.
        if self._sidebar_state_dirty:
            snapshot = self._sidebar_state_snapshot()
            await asyncio.to_thread(self._write_sidebar_state_snapshot, snapshot)
            self._sidebar_state_dirty = False

    def _load_sidebar_state(self) -> None:
        """Load sidebar state from config file."""
        config_path = _get_effective_config_path().parent / "ui_state.toml"

        try:
            if config_path.exists():
                with open(config_path, "r") as f:
                    data = toml.load(f)
                    sidebar_data = data.get("sidebar", {})

                    # Load collapsible states into UIState
                    self.ui_state.collapsible_states = sidebar_data.get(
                        "collapsible_states", {}
                    )
                    self.ui_state.sidebar_search_query = sidebar_data.get(
                        "search_query", ""
                    )
                    self.ui_state.last_active_section = sidebar_data.get(
                        "last_active_section", None
                    )

                    # Update reactive property
                    self.sidebar_state = dict(self.ui_state.collapsible_states)

                    logger.debug(
                        f"Loaded sidebar state with {len(self.ui_state.collapsible_states)} collapsibles"
                    )
        except Exception as e:
            logger.error(f"Failed to load sidebar state: {e}")
            self.sidebar_state = {}

    def _sidebar_state_snapshot(self) -> Dict[str, Any]:
        """Copy the sidebar-persisted fields off `self.ui_state`.

        `collapsible_states` is a plain mutable dict; taking this copy on
        the caller's thread (always the main/event-loop thread -- see
        `_persist_sidebar_state_off_loop`) before handing the write to a
        worker thread means the worker never reads `self.ui_state` directly,
        so a toggle arriving while that write is in flight cannot race it.
        """
        return {
            "collapsible_states": dict(self.ui_state.collapsible_states),
            "search_query": self.ui_state.sidebar_search_query,
            "last_active_section": self.ui_state.last_active_section,
        }

    def _write_sidebar_state_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Write a pre-captured sidebar-state snapshot to `ui_state.toml`.

        Safe to call from a worker thread: touches only the passed-in
        `snapshot`, never `self.ui_state`.
        """
        config_path = _get_effective_config_path().parent / "ui_state.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load existing config or create new
            if config_path.exists():
                with open(config_path, "r") as f:
                    data = toml.load(f)
            else:
                data = {}

            # Update sidebar section
            data["sidebar"] = snapshot

            # Save back to file
            with open(config_path, "w") as f:
                toml.dump(data, f)

            logger.debug(
                f"Saved sidebar state with {len(snapshot['collapsible_states'])} collapsibles"
            )
        except Exception as e:
            logger.error(f"Failed to save sidebar state: {e}")

    def _save_sidebar_state(self) -> None:
        """Save sidebar state to config file, synchronously, on this thread.

        Convenience wrapper around `_sidebar_state_snapshot` +
        `_write_sidebar_state_snapshot` for a caller that is already off the
        event loop (a worker thread via `to_thread`) or does not care (a
        direct test call). Callers on the event loop that must NOT block it
        should go through `watch_sidebar_state`'s debounce instead.
        """
        self._write_sidebar_state_snapshot(self._sidebar_state_snapshot())

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
                if (
                    collapsible.id
                    and collapsible.id in self.ui_state.collapsible_states
                ):
                    collapsed_state = self.ui_state.collapsible_states[collapsible.id]
                    collapsible.collapsed = collapsed_state
                    restored_count += 1
                    logger.debug(
                        f"Restored {collapsible.id}: collapsed={collapsed_state}"
                    )

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
                self.ui_state.set_collapsible_state(
                    collapsible_id, event.collapsible.collapsed
                )

                # Update reactive property to trigger watcher
                new_state = dict(self.ui_state.collapsible_states)
                self.sidebar_state = new_state

                logger.debug(
                    f"Toggled {collapsible_id}: collapsed={event.collapsible.collapsed}"
                )
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
                if (
                    "priority-high" not in collapsible.classes
                    and not collapsible.collapsed
                ):
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

            self._schedule_sidebar_state_save()
            logger.info("Reset sidebar to default state")
            self.notify("Settings reset to defaults", severity="success")
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
