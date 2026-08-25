"""Contracts for the shared safe-modal dismissal boundary."""

from __future__ import annotations

import asyncio
import ast
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
from types import MethodType
from typing import Any

import pytest
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Select, Static

from Tests.UI.background_signals import (
    await_background_task,
    wait_for_background_signal,
)
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRowTotals
from tldw_chatbook.Chat.console_prompt_queue import ConsolePromptQueueRegistry
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Prompt_Management.prompt_variables import PromptVariableApplication
from tldw_chatbook.UI.Screens.change_review_screen import (
    ChangeGitCommitModal,
    ChangeGitPushModal,
    ChangeReviewScreen,
    ChangeRevertConfirmModal,
)
from tldw_chatbook.UI.Screens.trajectory_screen import TrajectoryScreen
from tldw_chatbook.UI.Screens.video_player_screen import VideoPlayerScreen
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel, WorkbenchHelpState
from tldw_chatbook.Widgets.cancel_confirmation_dialog import (
    CancelConfirmationDialog,
)
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal
from tldw_chatbook.Widgets.delete_confirmation_dialog import DeleteConfirmationDialog
from tldw_chatbook.Widgets.Console.console_auto_speak_consent import (
    AutoSpeakConsentModal,
)
from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
    ConsoleComposerMenuModal,
)
from tldw_chatbook.Widgets.Console.console_edit_message_modal import (
    ConsoleEditMessageModal,
    ConsoleEditResult,
)
from tldw_chatbook.Widgets.Console.console_feedback_comment_modal import (
    ConsoleFeedbackCommentModal,
)
from tldw_chatbook.Widgets.Console.console_generate_image_modal import (
    ConsoleGenerateImageModal,
)
from tldw_chatbook.Widgets.Console.console_rag_settings_modal import (
    ConsoleRagSettingsModal,
    ConsoleRagSettingsResult,
)
from tldw_chatbook.Widgets.Console.console_rename_session_modal import (
    ConsoleRenameSessionModal,
)
from tldw_chatbook.Widgets.Console.console_rewind_modal import (
    ConsoleRewindChoice,
    ConsoleRewindModal,
)
from tldw_chatbook.Widgets.Console.console_save_as_modal import ConsoleSaveAsModal
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
)
from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
    ConsoleSystemPromptModal,
)
from tldw_chatbook.Widgets.Console.console_workspace_switcher_modal import (
    ConsoleWorkspaceRenameModal,
    ConsoleWorkspaceSwitcherModal,
)
from tldw_chatbook.Widgets.Console.console_character_picker_modal import (
    ConsoleCharacterOption,
    ConsoleCharacterPickerModal,
)
from tldw_chatbook.Widgets.Console.console_citation_sources_modal import (
    ConsoleCitationSourcesModal,
)
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
)
from tldw_chatbook.Widgets.Console.console_image_viewer_modal import (
    ConsoleImageViewerModal,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_prompt_picker_modal import (
    MODE_INSERT,
    ConsolePromptPickerModal,
)
from tldw_chatbook.Widgets.Console.console_prompt_queue_modal import (
    ConsolePromptQueueModal,
)
from tldw_chatbook.Widgets.Console.console_project_instructions import (
    ProjectInstructionNoticeModal,
    ProjectInstructionSetupModal,
)
from tldw_chatbook.Widgets.Console.console_prompt_comparison_modal import (
    ConsolePromptComparisonModal,
)
from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsModal
from tldw_chatbook.Widgets.Console.console_reaction_picker_modal import (
    ConsoleReactionPickerModal,
)
from tldw_chatbook.Widgets.Console.console_review_notes_modal import (
    ConsoleReviewNotesModal,
)
from tldw_chatbook.Widgets.Console.console_run_log_modal import ConsoleRunLogModal
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsInput,
    ConsoleSettingsModal,
    _settings_screen_region,
)
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal
from tldw_chatbook.Widgets.Console.console_side_chat_modal import (
    ConsoleSideChatModal,
)
from tldw_chatbook.Widgets.Console.console_scope_picker_modal import (
    ScopeListPage,
    TagCount,
    ConsoleScopePickerModal,
)
from tldw_chatbook.Widgets.Console.console_style_picker_modal import (
    ConsoleStylePickerModal,
)
from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
    ConsoleVideoCapacityModal,
)
from tldw_chatbook.Widgets.Console.trace_export_dialog import TraceExportDialog
from tldw_chatbook.Widgets.Console.prompt_variables_dialog import (
    PromptVariablesDialog,
    PromptVariablesDialogRequest,
)
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen,
    # task-18810: declared launches -- see CONSOLE_MODAL_LAUNCH_EDGES.
    EnhancedFileSave,
)
from tldw_chatbook.Widgets.modal_dismissal import (
    SafeModalDismissMixin,
    is_modal_backdrop_click,
)
from tldw_chatbook.Widgets.Persona_Widgets.dictionary_picker import DictionaryPicker
from tldw_chatbook.Widgets.Persona_Widgets.world_book_picker import WorldBookPicker


@dataclass(frozen=True)
class _Task2ModalContract:
    modal_type: type[ModalScreen[Any]]
    factory: Callable[[], ModalScreen[Any]]
    content_selector: str
    cancel_result: object
    opener: str
    pre_cancel_hook: str | None
    guard: str
    focus_postcondition: str


@dataclass(frozen=True)
class _Task3ModalContract:
    modal_type: type[ModalScreen[Any]]
    factory: Callable[[], ModalScreen[Any]]
    content_selector: str
    cancel_result: object
    success_result_types: tuple[type[object], ...]
    opener: str
    pre_cancel_hook: str | None
    guard: str
    focus_postcondition: str


@dataclass(frozen=True)
class _Task4ModalContract:
    modal_type: type[ModalScreen[Any]]
    content_selector: str | None
    cancel_result: object
    escape_action: str
    opener: str
    pre_cancel_hook: str | None
    guard: str
    focus_postcondition: str


@dataclass(frozen=True)
class _Task5TransitionContract:
    state: str
    gesture: str
    visible_state: str
    result: object
    callback_count: int
    focus_postcondition: str


@dataclass(frozen=True)
class _ExceptionalConsoleModalContract:
    modal_type: type[ModalScreen[Any]]
    content_selector: str
    cancel_result: object
    opener: str
    pre_cancel_hook: str | None
    guard: str
    focus_postcondition: str


@dataclass(frozen=True)
class _ModalLaunchEdge:
    owner: str | type[Screen[Any]]
    launched: tuple[type[Screen[Any]], ...]
    source_paths: tuple[str, ...]
    source_functions: tuple[str, ...] | None = None


_RESTORE_OPENER = "restore opener or Console composer fallback"


async def _empty_context_snapshot() -> ConsoleContextSnapshot:
    return ConsoleContextSnapshot(current_messages=[], next_send_payload={})


async def _empty_exchanges_loader(_native_message_id: str) -> list[tuple[Any, bool]]:
    return []


def _inspector_factory() -> ConsoleConversationInspector:
    """task-8: the Conversation Inspector replaced the two standalone
    modals it superseded (retired in task-10) as the Console root's actual
    launch target -- both entry points now push this instead (see
    ``chat_screen.py``'s ``_push_console_inspector``)."""
    return ConsoleConversationInspector(
        rows=[],
        totals=ConsoleCostRowTotals(0, 0.0, False, 0),
        turns=[],
        exchanges_loader=_empty_exchanges_loader,
        snapshot_factory=_empty_context_snapshot,
    )


async def _empty_records(_query: str) -> list[dict[str, object]]:
    return []


class _EmptySourceLister:
    async def list_page(self, **_kwargs: object) -> ScopeListPage:
        return ScopeListPage(items=(), total_matching=0)

    async def list_ids(self, **_kwargs: object) -> tuple[str, ...]:
        return ()


async def _empty_tags(_query: str) -> tuple[TagCount, ...]:
    return ()


def _citation_factory() -> ConsoleCitationSourcesModal:
    modal = ConsoleCitationSourcesModal(
        native_message_id="native-1",
        persisted_message_id="persisted-1",
        current_body="body",
        repository=object(),
        request_is_current=lambda: True,
    )
    modal._worker_started = True
    return modal


def _image_factory() -> ConsoleImageViewerModal:
    modal = ConsoleImageViewerModal(object())  # type: ignore[arg-type]
    modal._build_full_size_widget = MethodType(  # type: ignore[method-assign]
        lambda _self: Static("image", id="console-image-viewer-image"), modal
    )
    return modal


def _queue_factory() -> ConsolePromptQueueModal:
    registry = ConsolePromptQueueRegistry()
    snapshot = registry.snapshot("contract-session")
    return ConsolePromptQueueModal(
        session_id="contract-session",
        revision=snapshot.revision,
        queue_controller=registry,
    )


def _scope_factory() -> ConsoleScopePickerModal:
    source_lister = _EmptySourceLister()
    return ConsoleScopePickerModal(
        "contract target",
        None,
        None,
        lambda _scope: None,
        media_lister=source_lister,
        notes_lister=source_lister,
        tag_lister=_empty_tags,
    )


async def _save_system_prompt(_name: str, _text: str) -> str:
    return "saved"


class _IdleSideChatService:
    """Contract-fixture side-chat service: an async generator that yields
    nothing, so the dismissal factory never starts a stream."""

    async def run(self, **_kwargs: object) -> Any:
        return
        yield  # pragma: no cover -- makes run() an async generator


def _side_chat_factory() -> ConsoleSideChatModal:
    return ConsoleSideChatModal(
        service=_IdleSideChatService(),  # type: ignore[arg-type]
        provider_selection=None,
        sidechat_model="",
        quote="selected transcript text",
        auto_send_prompt=None,
    )


def _feedback_comment_factory() -> ConsoleFeedbackCommentModal:
    return ConsoleFeedbackCommentModal(
        action="comment",
        quote="selected transcript text",
    )


def _review_notes_factory() -> ConsoleReviewNotesModal:
    return ConsoleReviewNotesModal(
        notes=[
            {
                "annotation_id": "contract-anno-1",
                "conversation_id": "contract-conv-1",
                "row_key": "message:contract-msg-1",
                "message_id": "contract-msg-1",
                "quote_text": "quoted transcript text",
                "comment": "contract review note",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ],
        on_edit=lambda _annotation_id, _new_comment: True,
        on_delete=lambda _annotation_id: True,
    )


def _prompt_variables_factory() -> PromptVariablesDialog:
    return PromptVariablesDialog(
        PromptVariablesDialogRequest(
            system_text=None,
            user_text="Hello {name}",
            destination="replace_snapshot",
            target_session_id="contract-session",
            composer_fingerprint="a" * 64,
            system_fingerprint=None,
        )
    )


TASK2_MODAL_CONTRACTS = (
    _Task2ModalContract(
        AutoSpeakConsentModal,
        lambda: AutoSpeakConsentModal("TTS provider", "https://tts.example", False),
        "#console-auto-speak-consent-modal",
        False,
        "Console auto-speak toggle",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleCharacterPickerModal,
        lambda: ConsoleCharacterPickerModal(options=[]),
        "#console-character-picker",
        None,
        "Console character chip",
        "_cancel_query_debounce",
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleReactionPickerModal,
        lambda: ConsoleReactionPickerModal(options=[]),
        "#console-reaction-picker-modal",
        None,
        "Console character reaction action",
        "_cancel_pending_updates",
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleCitationSourcesModal,
        _citation_factory,
        "#console-citation-sources-modal",
        None,
        "Console citation marker",
        "increment _request_generation",
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleConversationInspector,
        _inspector_factory,
        "#console-inspector-modal",
        None,
        "Console cost chip / Console context action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleImageViewerModal,
        _image_factory,
        "#console-image-viewer",
        None,
        "Console avatar",
        None,
        "intentional click-anywhere cancel",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleModelPopover,
        lambda: ConsoleModelPopover(
            settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
            providers_models={"openai": ["gpt-test"]},
        ),
        "#console-model-popover",
        None,
        "Console model chip",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsolePromptPickerModal,
        lambda: ConsolePromptPickerModal(
            mode=MODE_INSERT, prompt_search=_empty_records
        ),
        "#console-prompt-picker-modal",
        None,
        "Console composer prompt command",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsolePromptQueueModal,
        _queue_factory,
        "#console-prompt-queue-dialog",
        None,
        "Console prompt queue",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleRunLogModal,
        lambda: ConsoleRunLogModal(run_id="run-1", log_text="log"),
        "#console-run-log-modal",
        None,
        "Console run log action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleScopePickerModal,
        _scope_factory,
        "#console-scope-picker-modal",
        None,
        "Console RAG scope action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleStylePickerModal,
        lambda: ConsoleStylePickerModal(),
        "#console-style-picker-modal",
        None,
        "Console image style action",
        "_cancel_search_debounce",
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleSideChatModal,
        _side_chat_factory,
        "#console-side-chat-modal",
        None,
        "Console selection More Details / Ask in Side Chat actions",
        "cancel side-chat worker",
        "none",
        _RESTORE_OPENER,
    ),
    _Task2ModalContract(
        ConsoleReviewNotesModal,
        _review_notes_factory,
        "#console-review-notes-modal",
        False,
        "Console annotation marker click / `n` review-notes action",
        None,
        "mid-edit escape closes the open editor before the second cancel dismisses",
        _RESTORE_OPENER,
    ),
)


TASK3_MODAL_CONTRACTS = (
    _Task3ModalContract(
        ConsoleComposerMenuModal,
        ConsoleComposerMenuModal,
        "#console-composer-menu",
        None,
        (str,),
        "Console composer Menu button",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleEditMessageModal,
        lambda: ConsoleEditMessageModal(content="original"),
        "#console-edit-message-modal",
        None,
        (ConsoleEditResult,),
        "Console transcript message action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleFeedbackCommentModal,
        _feedback_comment_factory,
        "#console-feedback-comment-modal",
        None,
        (str,),
        "Console selection Request changes / LGTM / Comment actions",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleGenerateImageModal,
        ConsoleGenerateImageModal,
        "#console-generate-image-modal",
        None,
        (str,),
        "Composer Generate Image menu action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleRagSettingsModal,
        ConsoleRagSettingsModal,
        "#console-rag-settings",
        None,
        (ConsoleRagSettingsResult,),
        "Console Library search chip",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleRenameSessionModal,
        lambda: ConsoleRenameSessionModal(title="Contract session"),
        "#console-rename-session-modal",
        None,
        (str,),
        "Console session tab action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleRewindModal,
        lambda: ConsoleRewindModal(prompts=()),
        "#console-rewind-modal",
        None,
        (ConsoleRewindChoice,),
        "Console /rewind command",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleSaveAsModal,
        lambda: ConsoleSaveAsModal(destinations=[]),
        "#console-save-as-modal",
        None,
        (str,),
        "Console message Save as action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleSessionSwitcherModal,
        lambda: ConsoleSessionSwitcherModal(rows=()),
        "#console-switcher-modal",
        None,
        (ConsoleSwitcherChoice,),
        "Console session switcher command",
        "_cancel_query_debounce",
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleSystemPromptModal,
        lambda: ConsoleSystemPromptModal(
            system_prompt="Be concise", save_to_library=_save_system_prompt
        ),
        "#console-system-prompt-modal",
        None,
        (str,),
        "Console system prompt chip",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleWorkspaceSwitcherModal,
        lambda: ConsoleWorkspaceSwitcherModal(workspaces=(), active_workspace_id=None),
        "#console-workspace-switcher-modal",
        None,
        (tuple,),
        "Console workspace context action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        ConsoleWorkspaceRenameModal,
        lambda: ConsoleWorkspaceRenameModal(current_name="Research"),
        "#console-workspace-rename-modal",
        None,
        (str,),
        "Workspace Switcher Rename action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task3ModalContract(
        PromptVariablesDialog,
        _prompt_variables_factory,
        "#prompt-variables-dialog",
        None,
        (PromptVariableApplication,),
        "Console prompt application flow",
        None,
        "none",
        _RESTORE_OPENER,
    ),
)


TASK4_MODAL_CONTRACTS = (
    _Task4ModalContract(
        WorkbenchHelpPanel,
        "#workbench-help-panel",
        None,
        "request_safe_cancel",
        "Console contextual help action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        DictionaryPicker,
        "#dictionary-picker-dialog",
        None,
        "request_safe_cancel",
        "Console character dictionary action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        WorldBookPicker,
        "#world-book-picker-dialog",
        None,
        "request_safe_cancel",
        "Console character world-book action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        ConfirmationDialog,
        "#confirmation-dialog",
        False,
        "request_safe_cancel",
        "Console confirmation launch sites",
        "cancel_callback via run_cancel_effect_once",
        "none",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        CancelConfirmationDialog,
        "#cancel-confirmation-dialog",
        False,
        "request_safe_cancel",
        "Console prompt queue cancellation",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        WorkspaceCreateModal,
        "#workspace-create-modal",
        None,
        "request_safe_cancel",
        "Console workspace browser create action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        SelectDirectory,
        "#file-system-picker-dialog",
        None,
        "request_safe_cancel",
        "Workspace create modal folder bind",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        EnhancedFileOpen,
        "#enhanced-file-dialog",
        None,
        "smart_dismiss",
        "Console attachment action",
        None,
        "path/search/recent/bookmarks peel before terminal Escape",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        EnhancedFileSave,
        "#enhanced-file-dialog",
        None,
        "smart_dismiss",
        "Console generated-video export",
        None,
        "path/search/recent/bookmarks peel before terminal Escape",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        VideoPlayerScreen,
        None,
        None,
        "request_safe_cancel",
        "Console video Play action",
        "player cleanup on unmount",
        "whole screen is content; no synthetic backdrop",
        _RESTORE_OPENER,
    ),
    _Task4ModalContract(
        ChangeRevertConfirmModal,
        "#change-revert-confirm",
        False,
        "request_safe_cancel",
        "ChangeReviewScreen revert actions",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    # TASK-16801 arc B (T7): the git commit confirm modal -- the surface
    # that writes to the user's REAL repository. Cancels to None (the
    # mixin's default cancel result), so an abandoned dialog can never be
    # mistaken for an empty commit request.
    _Task4ModalContract(
        ChangeGitCommitModal,
        "#change-git-commit",
        None,
        "request_safe_cancel",
        "ChangeReviewScreen commit action",
        None,
        "none",
        _RESTORE_OPENER,
    ),
    # TASK-16801 arc B (T8): the push / PR target-confirmation modal -- the
    # surface that confirms a write to a REMOTE. Cancels to None (the
    # mixin's default cancel result), so an abandoned dialog can never be
    # mistaken for a confirmed push.
    _Task4ModalContract(
        ChangeGitPushModal,
        "#change-git-push",
        None,
        "request_safe_cancel",
        "ChangeReviewScreen push / open-PR actions",
        None,
        "none",
        _RESTORE_OPENER,
    ),
)


TASK5_PROMPTS_TRANSITIONS = (
    tuple(
        _Task5TransitionContract(
            "clean-root", gesture, "closed", None, 1, "Console composer"
        )
        for gesture in ("escape", "backdrop")
    )
    + tuple(
        _Task5TransitionContract(
            "clean-nested", gesture, "closed", None, 1, "Console composer"
        )
        for gesture in ("escape", "backdrop")
    )
    + tuple(
        _Task5TransitionContract(
            state,
            gesture,
            "dirty guard",
            None,
            0,
            "Keep editing",
        )
        for state in ("dirty-edit", "dirty-recipe")
        for gesture in ("escape", "backdrop")
    )
    + (
        _Task5TransitionContract(
            "guard-visible",
            "escape",
            "editor",
            None,
            0,
            "remembered editor control",
        ),
        _Task5TransitionContract(
            "guard-visible", "backdrop", "dirty guard", None, 0, "Keep editing"
        ),
    )
    + tuple(
        _Task5TransitionContract(
            "active-improvement", gesture, "cancelling", None, 0, "active control"
        )
        for gesture in ("escape", "backdrop")
    )
    + tuple(
        _Task5TransitionContract(
            "cancelling-improvement",
            gesture,
            "cancelling",
            None,
            0,
            "active control",
        )
        for gesture in ("escape", "backdrop")
    )
    + (
        _Task5TransitionContract(
            "expanded-descendant",
            "primary click",
            "select overlay",
            None,
            0,
            "Select overlay",
        ),
    )
)


TASK567_MODAL_CONTRACTS = (
    _ExceptionalConsoleModalContract(
        ConsolePromptsModal,
        "#console-prompts-modal",
        None,
        "Console prompt workbench action",
        None,
        "dirty-state and active-improvement guards",
        _RESTORE_OPENER,
    ),
    _ExceptionalConsoleModalContract(
        ConsoleSettingsModal,
        "#console-settings-modal",
        None,
        "Console Settings action",
        None,
        "memory-reset and active-compaction close guards",
        _RESTORE_OPENER,
    ),
    _ExceptionalConsoleModalContract(
        ConsoleVideoCapacityModal,
        "#video-capacity-dialog",
        "discard after explicit confirmation only",
        "Console generated-video capacity resolver",
        None,
        "staged-artifact discard confirmation",
        _RESTORE_OPENER,
    ),
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONSOLE_ROOT = "Console"
_CONSOLE_ROOT_SOURCE_PATHS = (
    "tldw_chatbook/UI/Screens/chat_screen.py",
    "tldw_chatbook/Widgets/Console/console_auto_speak_consent.py",
    *tuple(
        str(path.relative_to(_REPO_ROOT))
        for path in sorted(
            (_REPO_ROOT / "tldw_chatbook/UI/Console_Modules").glob("*.py")
        )
    ),
)
# The side-chat modal launches from the Console root (phase 2 task 5:
# ChatScreen's ConsoleSideChatRequested handler pushes it after the
# selection menu's More Details / Ask in Side Chat actions), so it rides
# the declared root launches like every other root-owned modal. The
# feedback comment modal rides the same way (phase 3 task 5:
# ChatScreen's ConsoleSelectionFeedbackRequested flow pushes it before
# routing the composed feedback through the prompt queue).
_CONSOLE_DIRECT_MODAL_TYPES = tuple(
    contract.modal_type
    for contract in (*TASK2_MODAL_CONTRACTS, *TASK3_MODAL_CONTRACTS)
    if contract.modal_type is not ConsoleWorkspaceRenameModal
) + tuple(contract.modal_type for contract in TASK567_MODAL_CONTRACTS)
_DIRECT_SHARED_MODAL_TYPES = tuple(
    contract.modal_type
    for contract in TASK4_MODAL_CONTRACTS
    # Shared modals the Console root does NOT construct itself: each is
    # declared on the edge of the owner that actually opens it
    # (ChangeReviewScreen; the workspace create dialog -- task-18810).
    if contract.modal_type
    not in {
        ChangeRevertConfirmModal,
        ChangeGitCommitModal,
        ChangeGitPushModal,
        SelectDirectory,
    }
)
CONSOLE_MODAL_LAUNCH_EDGES = (
    _ModalLaunchEdge(
        _CONSOLE_ROOT,
        (
            *_CONSOLE_DIRECT_MODAL_TYPES,
            *_DIRECT_SHARED_MODAL_TYPES,
            ChangeReviewScreen,
            TrajectoryScreen,
            WorkspaceCreateModal,
        ),
        _CONSOLE_ROOT_SOURCE_PATHS,
    ),
    _ModalLaunchEdge(
        ConsoleWorkspaceSwitcherModal,
        (ConsoleWorkspaceRenameModal,),
        ("tldw_chatbook/UI/Console_Modules/workspace.py",),
        ("_open_console_workspace_rename",),
    ),
    # task-18810: the Console workspace browser opens the shared create
    # dialog (`_create_console_workspace`), which itself opens the vendored
    # directory picker. Both were reachable but undeclared, which is what
    # made the inventory walk abort before its own later assertions.
    _ModalLaunchEdge(
        WorkspaceCreateModal,
        (SelectDirectory,),
        ("tldw_chatbook/Widgets/workspace_create_modal.py",),
    ),
    # task-18810: the review-notes modal confirms each delete. This launch
    # shipped in task-18515 WITHOUT being caught by this test -- the walk
    # was already aborting on WorkspaceCreateModal above, which is the
    # silent-skip this task exists to close.
    _ModalLaunchEdge(
        ConsoleReviewNotesModal,
        (ConfirmationDialog,),
        ("tldw_chatbook/Widgets/Console/console_review_notes_modal.py",),
    ),
    _ModalLaunchEdge(
        ConsolePromptQueueModal,
        (CancelConfirmationDialog,),
        ("tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py",),
    ),
    _ModalLaunchEdge(
        ConsoleVideoCapacityModal,
        (CancelConfirmationDialog,),
        ("tldw_chatbook/Widgets/Console/console_video_capacity_modal.py",),
    ),
    _ModalLaunchEdge(
        TrajectoryScreen,
        (TrajectoryScreen, EnhancedFileOpen),
        ("tldw_chatbook/UI/Screens/trajectory_screen.py",),
    ),
    # TASK-16801 arc B (T7): the review screen also opens the git commit
    # confirm modal (`_land_commit_preflight`), which is where a commit into
    # the user's REAL repository is confirmed -- declared here so this walk
    # keeps covering every modal reachable from the Console.
    _ModalLaunchEdge(
        ChangeReviewScreen,
        (ChangeRevertConfirmModal, ChangeGitCommitModal, ChangeGitPushModal),
        ("tldw_chatbook/UI/Screens/change_review_screen.py",),
    ),
)


def _discover_console_modal_types() -> set[type[ModalScreen[Any]]]:
    """Discover Console modal classes by AST class inventory and runtime MRO."""
    discovered: set[type[ModalScreen[Any]]] = set()
    console_root = _REPO_ROOT / "tldw_chatbook/Widgets/Console"
    for source_path in sorted(console_root.glob("*.py")):
        module_name = ".".join(
            source_path.relative_to(_REPO_ROOT).with_suffix("").parts
        )
        module = importlib.import_module(module_name)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            runtime_type = getattr(module, node.name, None)
            if (
                inspect.isclass(runtime_type)
                and runtime_type.__module__ == module_name
                and issubclass(runtime_type, ModalScreen)
            ):
                discovered.add(runtime_type)
    return discovered


def _resolve_ast_reference(
    node: ast.expr, bindings: dict[str, object]
) -> object | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolve_ast_reference(node.value, bindings)
        return getattr(owner, node.attr, None)
    return None


def _constructed_modal_types(
    source_paths: tuple[str, ...],
    *,
    source_overrides: dict[str, str] | None = None,
    included_functions: set[str] | None = None,
    included_classes: set[str] | None = None,
    excluded_functions: set[str] | None = None,
) -> set[type[ModalScreen[Any]]]:
    """Resolve actual modal constructors in source, including imported aliases."""
    constructed: set[type[ModalScreen[Any]]] = set()
    overrides = source_overrides or {}
    for relative_path in source_paths:
        source_path = _REPO_ROOT / relative_path
        source = overrides.get(relative_path)
        if source is None:
            source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=source_path)
        bindings: dict[str, object] = {}
        module_name = ".".join(Path(relative_path).with_suffix("").parts)
        if relative_path not in overrides:
            runtime_module = importlib.import_module(module_name)
            bindings.update(vars(runtime_module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = importlib.import_module(alias.name)
                    binding_name = alias.asname or alias.name.split(".")[0]
                    bindings[binding_name] = (
                        imported
                        if alias.asname
                        else importlib.import_module(binding_name)
                    )
            elif isinstance(node, ast.ImportFrom):
                imported_name = node.module or ""
                if node.level:
                    package = module_name.rpartition(".")[0]
                    imported_name = importlib.util.resolve_name(
                        f"{'.' * node.level}{imported_name}", package
                    )
                imported = importlib.import_module(imported_name)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bindings[alias.asname or alias.name] = getattr(imported, alias.name)

        class _ConstructorVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_stack: list[str] = []
                self.class_stack: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node: ast.Call) -> None:
                functions = set(self.function_stack)
                classes = set(self.class_stack)
                if included_functions is not None and not (
                    functions & included_functions
                ):
                    self.generic_visit(node)
                    return
                if included_classes is not None and not (classes & included_classes):
                    self.generic_visit(node)
                    return
                if excluded_functions and functions & excluded_functions:
                    self.generic_visit(node)
                    return
                runtime_type = _resolve_ast_reference(node.func, bindings)
                if inspect.isclass(runtime_type) and issubclass(
                    runtime_type, ModalScreen
                ):
                    constructed.add(runtime_type)
                self.generic_visit(node)

        _ConstructorVisitor().visit(tree)
    return constructed


def _binding_key_action(binding: object) -> tuple[str, str]:
    if isinstance(binding, Binding):
        return binding.key, binding.action
    return binding[0], binding[1]  # type: ignore[index,return-value]


@dataclass(frozen=True)
class _LaunchWalkResult:
    """Outcome of one launch-graph walk (task-18810)."""

    reachable: set[str | type[Screen[Any]]]
    mismatches: tuple[str, ...]


def _walk_modal_launch_graph(
    root: str | type[Screen[Any]],
    edges: tuple[_ModalLaunchEdge, ...],
    *,
    source_overrides: dict[str, str] | None = None,
    owner_source_paths: dict[type[Screen[Any]], tuple[str, ...]] | None = None,
) -> "_LaunchWalkResult":
    """Walk the launch graph, collecting every declaration mismatch.

    Args:
        root: The graph root (the Console owner token or a Screen class).
        edges: Declared launch edges.
        source_overrides: Synthetic sources, for the tests that exercise
            this walker itself.
        owner_source_paths: Source paths for owners the walker cannot
            resolve by inspection.

    Returns:
        The reachable DECLARED nodes and every mismatch found. Nothing is
        raised here (task-18810): the caller decides, so a mismatch cannot
        stop the caller's own inventory assertions from running.
    """
    edges_by_owner = {edge.owner: edge for edge in edges}
    source_paths_by_owner = owner_source_paths or {}
    reachable: set[str | type[Screen[Any]]] = {root}
    frontier: list[str | type[Screen[Any]]] = [root]
    mismatches: list[str] = []
    scanned_strays: set[type[Screen[Any]]] = set()
    while frontier:
        owner = frontier.pop()
        edge = edges_by_owner.get(owner)
        actual: set[type[ModalScreen[Any]]] = set()
        if inspect.isclass(owner) and issubclass(owner, Screen):
            defining_paths = source_paths_by_owner.get(owner)
            if defining_paths is None:
                source_file = inspect.getsourcefile(owner)
                assert source_file is not None
                defining_paths = (str(Path(source_file).relative_to(_REPO_ROOT)),)
            actual.update(
                _constructed_modal_types(
                    defining_paths,
                    source_overrides=source_overrides,
                    included_classes={owner.__name__},
                )
            )

        if edge is None:
            assert inspect.isclass(owner) and issubclass(owner, Screen)
            declared: tuple[type[Screen[Any]], ...] = ()
        else:
            declared = edge.launched
            nested_functions = {
                function
                for other_edge in edges
                if other_edge is not edge
                and set(other_edge.source_paths) & set(edge.source_paths)
                for function in (other_edge.source_functions or ())
            }
            actual.update(
                _constructed_modal_types(
                    edge.source_paths,
                    source_overrides=source_overrides,
                    included_functions=(
                        set(edge.source_functions) if edge.source_functions else None
                    ),
                    excluded_functions=nested_functions,
                )
            )
        expected = {
            launched
            for launched in declared
            if inspect.isclass(launched) and issubclass(launched, ModalScreen)
        }
        unexpected = actual - expected
        missing = expected - actual
        if unexpected or missing:
            # COLLECTED, never raised (task-18810): asserting per-owner
            # aborted the walk at the first mismatch, so every later owner
            # -- and every assertion in the calling test after the walk --
            # silently stopped being checked. Two real launches shipped
            # undeclared behind exactly that: WorkspaceCreateModal and
            # ConsoleReviewNotesModal's delete confirmation.
            mismatches.append(
                f"{getattr(owner, '__name__', owner)} modal launch mismatch; "
                f"unexpected={sorted(item.__name__ for item in unexpected)}, "
                f"missing={sorted(item.__name__ for item in missing)}"
            )
        for launched in declared:
            if launched not in reachable:
                reachable.add(launched)
                frontier.append(launched)
        # Traverse UNDECLARED launches too, without adding them to the
        # returned reachable set (task-18810 review): a stale parent
        # declaration would otherwise hide every mismatch beneath the modal
        # it failed to declare. `reachable` stays the declared-reachable
        # set the contract table is compared against.
        for stray in unexpected:
            if stray not in reachable and stray not in scanned_strays:
                scanned_strays.add(stray)
                frontier.append(stray)
    return _LaunchWalkResult(reachable=reachable, mismatches=tuple(mismatches))


def test_console_modal_launch_declarations_match_runtime_construction() -> None:
    """Every declared launch edge matches what the code actually constructs.

    Split from the inventory test (task-18810): when a declaration goes
    stale, THIS test fails while the inventory test still runs and reports
    its own count/set drift, instead of the first mismatch hiding both.
    """
    result = _walk_modal_launch_graph(_CONSOLE_ROOT, CONSOLE_MODAL_LAUNCH_EDGES)
    assert not result.mismatches, "\n".join(result.mismatches)


def test_console_modal_inventory_matches_runtime_ast_and_transitive_launches() -> None:
    console_contract_types = {
        contract.modal_type
        for contract in (
            *TASK2_MODAL_CONTRACTS,
            *TASK3_MODAL_CONTRACTS,
            *TASK567_MODAL_CONTRACTS,
        )
    }
    discovered_console_types = _discover_console_modal_types()
    # task-8 replaced both of the Console root's original standalone
    # launch targets with ConsoleConversationInspector (chat_screen.py no
    # longer constructs either); task-10 deleted the two now-orphaned
    # module files outright.
    #
    # These modals live under the scanned Console package but are launched
    # outside the legacy task-2/3/5/6/7 dismissal-contract graph. Keep them
    # explicit so a newly added modal still fails this inventory gate.
    inventory_only_types: set[type[ModalScreen[Any]]] = {
        ConsolePromptComparisonModal,
        ProjectInstructionNoticeModal,
        ProjectInstructionSetupModal,
        TraceExportDialog,
    }

    assert discovered_console_types - console_contract_types == inventory_only_types
    assert discovered_console_types == console_contract_types | inventory_only_types

    # The mismatch check is its own test below: a bad declaration must not
    # stop the inventory assertions in THIS test from running (task-18810
    # review) -- that masking is the whole defect being fixed.
    reachable = _walk_modal_launch_graph(
        _CONSOLE_ROOT, CONSOLE_MODAL_LAUNCH_EDGES
    ).reachable

    reachable_modal_types = {
        node
        for node in reachable
        if inspect.isclass(node) and issubclass(node, ModalScreen)
    }
    # dev baseline 42 (43 minus the two Console modals another task
    # unwires -- ConsoleCostModal/ConsoleContextModal -- plus the
    # inspector that replaced them); 43 since TASK-16801 arc B added the
    # review screen's git commit modal, and 44 since its push /
    # open-PR confirmation modal (T8).
    assert len(reachable_modal_types) == 44
    all_contract_types = console_contract_types | {
        contract.modal_type for contract in TASK4_MODAL_CONTRACTS
    } | {TrajectoryScreen}
    assert reachable_modal_types == all_contract_types
    assert {EnhancedFileOpen, EnhancedFileSave} <= reachable_modal_types
    assert CancelConfirmationDialog in reachable_modal_types
    assert ChangeRevertConfirmModal in reachable_modal_types

    assert issubclass(ConsoleSetupModal, Vertical)
    assert not issubclass(ConsoleSetupModal, ModalScreen)
    assert ConsoleSetupModal not in reachable


def test_launch_inventory_rejects_an_uncontracted_constructed_modal() -> None:
    synthetic_path = "synthetic_console_launch.py"
    # ConsoleSideChatModal stands in here as "some other real, importable
    # modal" -- this test is only exercising ``_constructed_modal_types``'s
    # own AST resolution (never actually instantiates either class; the
    # source below is parsed, not executed), so which concrete modal it
    # names is arbitrary.
    source = """
def launch():
    from tldw_chatbook.Widgets.Console.console_side_chat_modal import ConsoleSideChatModal as Extra
    import tldw_chatbook.Widgets.Console.console_run_log_modal as run_log

    Extra()
    run_log.ConsoleRunLogModal(run_id='extra', log_text='extra')
"""

    actual = _constructed_modal_types(
        (synthetic_path,), source_overrides={synthetic_path: source}
    )

    assert actual == {ConsoleSideChatModal, ConsoleRunLogModal}
    with pytest.raises(AssertionError):
        assert actual == {ConsoleSideChatModal}


def test_modal_dismissal_uses_a_public_monotonic_clock() -> None:
    source = (_REPO_ROOT / "tldw_chatbook/Widgets/modal_dismissal.py").read_text(
        encoding="utf-8"
    )

    assert "textual._time" not in source
    assert "time.monotonic()" in source


class _SyntheticRowlessOwner(Screen[None]):
    pass


class _SyntheticDeclaredOwner(Screen[None]):
    pass


class _SyntheticStrayOwner(ModalScreen[None]):
    """An undeclared modal that itself launches another (task-18810)."""

    pass


def test_launch_inventory_scans_reachable_owners_without_declared_rows() -> None:
    root_path = "synthetic_root.py"
    rowless_path = "synthetic_rowless_owner.py"
    edges = (
        _ModalLaunchEdge(
            "SyntheticRoot",
            (_SyntheticRowlessOwner,),
            (root_path,),
        ),
    )
    sources = {
        root_path: "",
        rowless_path: """
class _SyntheticRowlessOwner:
    def launch_nested(self):
        from tldw_chatbook.Widgets.Console.console_run_log_modal import ConsoleRunLogModal as Extra
        Extra(run_id='extra', log_text='extra')
""",
    }

    result = _walk_modal_launch_graph(
        "SyntheticRoot",
        edges,
        source_overrides=sources,
        owner_source_paths={_SyntheticRowlessOwner: (rowless_path,)},
    )
    assert any("ConsoleRunLogModal" in entry for entry in result.mismatches), (
        result.mismatches
    )


def test_launch_inventory_unions_declared_helpers_with_owner_class_body() -> None:
    root_path = "synthetic_declared_root.py"
    helper_path = "synthetic_declared_helper.py"
    owner_path = "synthetic_declared_owner.py"
    edges = (
        _ModalLaunchEdge(
            "SyntheticRoot",
            (_SyntheticDeclaredOwner,),
            (root_path,),
        ),
        _ModalLaunchEdge(
            _SyntheticDeclaredOwner,
            (ConsoleSideChatModal,),
            (helper_path,),
        ),
    )
    sources = {
        root_path: "",
        # ConsoleSideChatModal again stands in as "some other real,
        # importable modal" -- see the sibling test above.
        helper_path: """
from tldw_chatbook.Widgets.Console.console_side_chat_modal import ConsoleSideChatModal as Expected
Expected()
""",
        owner_path: """
class _SyntheticDeclaredOwner:
    def launch_nested(self):
        from tldw_chatbook.Widgets.Console.console_run_log_modal import ConsoleRunLogModal as Extra
        Extra(run_id='extra', log_text='extra')
""",
    }

    result = _walk_modal_launch_graph(
        "SyntheticRoot",
        edges,
        source_overrides=sources,
        owner_source_paths={_SyntheticDeclaredOwner: (owner_path,)},
    )
    assert any("ConsoleRunLogModal" in entry for entry in result.mismatches), (
        result.mismatches
    )


def test_task2_modal_contract_table_is_complete_and_adopted() -> None:
    assert len(TASK2_MODAL_CONTRACTS) == 14
    assert {contract.modal_type.__name__ for contract in TASK2_MODAL_CONTRACTS} == {
        "AutoSpeakConsentModal",
        "ConsoleCharacterPickerModal",
        "ConsoleReactionPickerModal",
        "ConsoleCitationSourcesModal",
        "ConsoleConversationInspector",
        "ConsoleImageViewerModal",
        "ConsoleModelPopover",
        "ConsolePromptPickerModal",
        "ConsolePromptQueueModal",
        "ConsoleReviewNotesModal",
        "ConsoleRunLogModal",
        "ConsoleScopePickerModal",
        "ConsoleSideChatModal",
        "ConsoleStylePickerModal",
    }
    expected_hooks = {
        "ConsoleCharacterPickerModal": "_cancel_query_debounce",
        "ConsoleReactionPickerModal": "_cancel_pending_updates",
        "ConsoleCitationSourcesModal": "increment _request_generation",
        "ConsoleSideChatModal": "cancel side-chat worker",
        "ConsoleStylePickerModal": "_cancel_search_debounce",
    }
    expected_guards = {
        "ConsoleImageViewerModal": "intentional click-anywhere cancel",
        "ConsoleReviewNotesModal": (
            "mid-edit escape closes the open editor before the second "
            "cancel dismisses"
        ),
    }
    for contract in TASK2_MODAL_CONTRACTS:
        assert issubclass(contract.modal_type, SafeModalDismissMixin)
        assert contract.modal_type.SAFE_MODAL_CONTENT == contract.content_selector
        escape_actions = [
            action
            for binding in contract.modal_type.BINDINGS
            for key, action in [_binding_key_action(binding)]
            if key == "escape"
        ]
        assert escape_actions == ["request_safe_cancel"]
        assert contract.cancel_result is None or contract.cancel_result is False
        assert contract.opener
        assert contract.pre_cancel_hook == expected_hooks.get(
            contract.modal_type.__name__
        )
        assert contract.guard == expected_guards.get(
            contract.modal_type.__name__, "none"
        )
        assert contract.focus_postcondition == _RESTORE_OPENER


def test_task3_modal_contract_table_is_complete_and_adopted() -> None:
    assert len(TASK3_MODAL_CONTRACTS) == 13
    assert {contract.modal_type.__name__ for contract in TASK3_MODAL_CONTRACTS} == {
        "ConsoleComposerMenuModal",
        "ConsoleEditMessageModal",
        "ConsoleFeedbackCommentModal",
        "ConsoleGenerateImageModal",
        "ConsoleRagSettingsModal",
        "ConsoleRenameSessionModal",
        "ConsoleRewindModal",
        "ConsoleSaveAsModal",
        "ConsoleSessionSwitcherModal",
        "ConsoleSystemPromptModal",
        "ConsoleWorkspaceRenameModal",
        "ConsoleWorkspaceSwitcherModal",
        "PromptVariablesDialog",
    }
    for contract in TASK3_MODAL_CONTRACTS:
        assert issubclass(contract.modal_type, SafeModalDismissMixin)
        assert contract.modal_type.SAFE_MODAL_CONTENT == contract.content_selector
        escape_actions = [
            action
            for binding in contract.modal_type.BINDINGS
            for key, action in [_binding_key_action(binding)]
            if key == "escape"
        ]
        assert escape_actions == ["request_safe_cancel"]
        assert contract.cancel_result is None
        assert contract.success_result_types
        assert type(None) not in contract.success_result_types
        assert contract.opener
        assert contract.pre_cancel_hook == (
            "_cancel_query_debounce"
            if contract.modal_type is ConsoleSessionSwitcherModal
            else None
        )
        assert contract.guard == "none"
        assert contract.focus_postcondition == _RESTORE_OPENER


def test_task4_transitive_modal_contract_table_is_complete_and_adopted() -> None:
    assert len(TASK4_MODAL_CONTRACTS) == 13
    assert {contract.modal_type.__name__ for contract in TASK4_MODAL_CONTRACTS} == {
        "WorkbenchHelpPanel",
        "DictionaryPicker",
        "WorldBookPicker",
        "ConfirmationDialog",
        "CancelConfirmationDialog",
        "EnhancedFileOpen",
        "EnhancedFileSave",
        "VideoPlayerScreen",
        "ChangeRevertConfirmModal",
        "ChangeGitCommitModal",
        "ChangeGitPushModal",
        "WorkspaceCreateModal",
        "SelectDirectory",
    }
    for contract in TASK4_MODAL_CONTRACTS:
        assert issubclass(contract.modal_type, SafeModalDismissMixin)
        assert contract.modal_type.SAFE_MODAL_CONTENT == contract.content_selector
        escape_actions = [
            action
            for binding in contract.modal_type.BINDINGS
            for key, action in [_binding_key_action(binding)]
            if key == "escape"
        ]
        assert escape_actions == [contract.escape_action]
        assert contract.cancel_result is None or contract.cancel_result is False
        assert contract.opener
        assert contract.guard
        assert contract.focus_postcondition == _RESTORE_OPENER

    launch_source = inspect.getsource(ChangeReviewScreen._confirm_and_revert)
    launch_source += inspect.getsource(ChangeReviewScreen.action_undo_all)
    assert "ChangeRevertConfirmModal(" in launch_source

    # TASK-16801 arc B: the two git modals are launched from ONE method
    # each -- pinning the launch SITE (not just the class's existence) is
    # what makes an accidental move to an undeclared opener show up here.
    assert "ChangeGitCommitModal(" in inspect.getsource(
        ChangeReviewScreen._land_commit_preflight
    )
    assert "ChangeGitPushModal(" in inspect.getsource(
        ChangeReviewScreen._land_git_target_preflight
    )


def test_capacity_modal_uses_guarded_safe_dismissal_contract() -> None:
    assert issubclass(ConsoleVideoCapacityModal, SafeModalDismissMixin)
    assert ConsoleVideoCapacityModal.SAFE_MODAL_CONTENT == "#video-capacity-dialog"
    assert [
        action
        for binding in ConsoleVideoCapacityModal.BINDINGS
        for key, action in [_binding_key_action(binding)]
        if key == "escape"
    ] == ["request_safe_cancel"]


@pytest.mark.asyncio
async def test_capacity_modal_mount_resets_one_generation_and_guard_identity() -> None:
    app = _Task2Harness()
    modal = ConsoleVideoCapacityModal(
        reason="over_capacity",
        size_bytes=2,
        max_bytes=1,
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(modal)
        await pilot.pause()
        assert modal._safe_mount_generation == 1

        await modal.request_safe_cancel(source="escape")
        await pilot.pause()
        first_guard = app.screen
        assert isinstance(first_guard, CancelConfirmationDialog)

        app.pop_screen()
        app.pop_screen()
        await pilot.pause()
        app.push_screen(modal)
        await pilot.pause()

        assert app.screen is modal
        assert modal._safe_mount_generation == 2
        assert not modal._discard_confirmation_open
        assert modal._discard_confirmation_guard is None
        assert modal._discard_confirmation_generation is None

        await modal.request_safe_cancel(source="escape")
        await pilot.pause()
        assert isinstance(app.screen, CancelConfirmationDialog)
        assert app.screen is not first_guard


@pytest.mark.parametrize("terminal", ["keep", "save", "navigation"])
@pytest.mark.asyncio
async def test_capacity_modal_stale_request_cannot_push_orphan_guard(
    terminal: str,
) -> None:
    app = _Task2Harness()
    modal = ConsoleVideoCapacityModal(
        reason="over_capacity",
        size_bytes=2,
        max_bytes=1,
    )

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        stale_request = modal.request_safe_cancel(source="escape")

        if terminal == "keep":
            await pilot.click("#video-capacity-keep")
        elif terminal == "save":
            await pilot.click("#video-capacity-save")
        else:
            app.pop_screen()
        await pilot.pause()

        await stale_request
        await pilot.pause()

        assert not isinstance(app.screen, CancelConfirmationDialog)
        assert modal not in app.screen_stack
        assert not modal._discard_confirmation_open


@pytest.mark.parametrize("repushed_state", ["plain", "guarded"])
@pytest.mark.asyncio
async def test_capacity_modal_stale_guard_callback_cannot_mutate_repush(
    repushed_state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _Task2Harness()
    modal = ConsoleVideoCapacityModal(
        reason="store_failure",
        size_bytes=2,
        max_bytes=1,
    )
    first_results: list[object] = []
    second_results: list[object] = []
    queued_callbacks: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(modal, callback=first_results.append)
        await pilot.pause()
        await modal.request_safe_cancel(source="escape")
        await pilot.pause()
        first_guard = app.screen
        assert isinstance(first_guard, CancelConfirmationDialog)

        callback_requester = first_guard._result_callbacks[-1].requester
        original_call_next = callback_requester.call_next

        def capture_callback(callback, *args):  # type: ignore[no-untyped-def]
            if args == (True,):
                queued_callbacks.append((callback, args))
                return
            original_call_next(callback, *args)

        monkeypatch.setattr(callback_requester, "call_next", capture_callback)
        first_guard.dismiss(True)
        monkeypatch.setattr(callback_requester, "call_next", original_call_next)
        modal.dismiss("keep")
        await pilot.pause()
        app.push_screen(modal, callback=second_results.append)
        await pilot.pause()

        assert first_results == ["keep"]
        assert len(queued_callbacks) == 1
        assert app.screen is modal
        repushed_generation = modal._safe_mount_generation

        second_guard = None
        if repushed_state == "guarded":
            await modal.request_safe_cancel(source="escape")
            await pilot.pause()
            second_guard = app.screen
            assert isinstance(second_guard, CancelConfirmationDialog)
            assert modal._discard_confirmation_guard is second_guard

        callback, args = queued_callbacks[0]
        callback(*args)
        await pilot.pause()

        if repushed_state == "plain":
            assert app.screen is modal
            assert second_results == []
            assert not modal._discard_confirmation_open
        else:
            assert app.screen is second_guard
            assert modal._safe_mount_generation == repushed_generation
            assert modal._discard_confirmation_open
            assert modal._discard_confirmation_guard is second_guard
            assert modal._discard_confirmation_generation == repushed_generation


def test_task5_prompt_workbench_transition_table_is_complete_and_adopted() -> None:
    assert len(TASK5_PROMPTS_TRANSITIONS) == 15
    assert {
        (contract.state, contract.gesture) for contract in TASK5_PROMPTS_TRANSITIONS
    } == {
        ("clean-root", "escape"),
        ("clean-root", "backdrop"),
        ("clean-nested", "escape"),
        ("clean-nested", "backdrop"),
        ("dirty-edit", "escape"),
        ("dirty-edit", "backdrop"),
        ("dirty-recipe", "escape"),
        ("dirty-recipe", "backdrop"),
        ("guard-visible", "escape"),
        ("guard-visible", "backdrop"),
        ("active-improvement", "escape"),
        ("active-improvement", "backdrop"),
        ("cancelling-improvement", "escape"),
        ("cancelling-improvement", "backdrop"),
        ("expanded-descendant", "primary click"),
    }
    assert issubclass(ConsolePromptsModal, SafeModalDismissMixin)
    assert ConsolePromptsModal.SAFE_MODAL_CONTENT == "#console-prompts-modal"
    assert [
        action
        for binding in ConsolePromptsModal.BINDINGS
        for key, action in [_binding_key_action(binding)]
        if key == "escape"
    ] == ["request_safe_cancel"]


def test_task6_settings_close_contract_is_adopted() -> None:
    assert issubclass(ConsoleSettingsModal, SafeModalDismissMixin)
    assert ConsoleSettingsModal.SAFE_MODAL_CONTENT == "#console-settings-modal"
    assert [
        action
        for binding in ConsoleSettingsModal.BINDINGS
        for key, action in [_binding_key_action(binding)]
        if key == "escape"
    ] == ["request_safe_cancel"]


class _Task2Harness(ConsolidatedCSSApp):
    CSS = """
    Screen { align: center middle; }
    #console-citation-sources-modal,
    #console-style-picker-modal,
    #change-revert-confirm { width: 60; height: 20; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []


class _SettingsMROHarness(App[None]):
    CSS = """
    ConsoleSettingsModal { align: center middle; }
    ConsoleSettingsModal #console-settings-modal { width: 100; height: 90%; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []

    def compose(self) -> ComposeResult:
        yield Input(id="settings-dismiss-opener")


def _settings_mro_modal() -> ConsoleSettingsModal:
    return ConsoleSettingsModal(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        app_config={"api_settings": {"llama_cpp": {}}},
        providers_models={
            "llama_cpp": ["model-a"],
            "local_llamacpp": ["local-model"],
        },
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
    )


@pytest.mark.parametrize("source", ["visible-cancel", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_settings_clean_close_sources_restore_opener_focus(source: str) -> None:
    app = _SettingsMROHarness()
    modal = _settings_mro_modal()

    async with app.run_test(size=(120, 42)) as pilot:
        host_screen = app.screen
        opener = app.query_one("#settings-dismiss-opener", Input)
        opener.focus()
        await pilot.pause()
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        if source == "visible-cancel":
            await pilot.click("#console-settings-cancel")
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()
        await pilot.pause()

        assert app.results == [None]
        assert app.screen is host_screen
        assert app.focused is opener


@pytest.mark.asyncio
async def test_settings_redirected_select_click_uses_real_mro_dispatch() -> None:
    app = _SettingsMROHarness()
    modal = _settings_mro_modal()

    async with app.run_test(size=(140, 60)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        focused_input = modal.query_one(
            "#console-settings-temperature", ConsoleSettingsInput
        )
        provider_select = modal.query_one("#console-settings-provider", Select)
        focused_input.focus()
        await pilot.pause()
        provider_region = _settings_screen_region(provider_select)
        content_region = modal.query_one("#console-settings-modal", Vertical).region
        assert content_region.contains(provider_region.right - 1, provider_region.y), (
            content_region,
            provider_region,
        )
        click = events.Click(
            modal,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=provider_region.right - 1,
            screen_y=provider_region.y,
        )

        await modal._dispatch_message(click)
        await pilot.pause()

        assert provider_select.expanded
        assert app.screen is modal
        assert app.results == []


@dataclass(frozen=True)
class _MountedTask4Contract:
    name: str
    factory: Callable[[], ModalScreen[Any]]
    visible_cancel: str
    cancel_result: object


MOUNTED_TASK4_CONTRACTS = (
    _MountedTask4Contract(
        "workbench-help",
        lambda: WorkbenchHelpPanel(
            WorkbenchHelpState(route_id="console", title="Console help")
        ),
        "#workbench-help-close",
        None,
    ),
    _MountedTask4Contract(
        "dictionary-picker",
        lambda: DictionaryPicker([]),
        "#dict-pick-cancel",
        None,
    ),
    _MountedTask4Contract(
        "world-book-picker",
        lambda: WorldBookPicker([]),
        "#worldbook-pick-cancel",
        None,
    ),
    _MountedTask4Contract(
        "confirmation",
        ConfirmationDialog,
        "#cancel-button",
        False,
    ),
    _MountedTask4Contract(
        "cancel-confirmation",
        CancelConfirmationDialog,
        "#continue-btn",
        False,
    ),
    _MountedTask4Contract(
        "change-revert-confirm",
        lambda: ChangeRevertConfirmModal("Revert a.txt?", []),
        "#change-revert-no",
        False,
    ),
)


@pytest.mark.parametrize("contract", MOUNTED_TASK4_CONTRACTS, ids=lambda row: row.name)
@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_task4_shared_modal_cancel_sources_return_exact_result(
    contract: _MountedTask4Contract,
    source: str,
) -> None:
    app = _Task2Harness()
    modal = contract.factory()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        if source == "visible":
            await pilot.click(contract.visible_cancel)
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert len(app.results) == 1
    assert app.results[0] is contract.cancel_result


@pytest.mark.asyncio
async def test_delete_confirmation_inherits_safe_backdrop_cancel_contract() -> None:
    app = _Task2Harness()
    modal = DeleteConfirmationDialog(item_type="Conversation", item_name="Example")

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert len(app.results) == 1
    assert app.results[0] is False


@pytest.mark.asyncio
async def test_confirmation_cancel_callback_is_once_across_repeated_and_nested_input():
    entered = asyncio.Event()
    release = asyncio.Event()
    callback_calls = 0
    nested = _NestedModal()
    app = _Task2Harness()

    async def cancel_callback() -> None:
        nonlocal callback_calls
        callback_calls += 1
        entered.set()
        await release.wait()
        app.push_screen(nested)

    modal = ConfirmationDialog(cancel_callback=cancel_callback)

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        first_request = asyncio.create_task(modal.action_request_safe_cancel())
        await wait_for_background_signal(
            entered,
            first_request,
            what="the confirmation cancel callback to start",
        )
        repeated_escape = asyncio.create_task(modal.action_request_safe_cancel())
        await pilot.click(offset=(0, 0))
        assert callback_calls == 1
        assert app.screen is modal

        release.set()
        await await_background_task(
            first_request,
            what="the confirmation cancel callback to finish",
        )
        await await_background_task(
            repeated_escape,
            what="the repeated confirmation cancel request to finish",
        )
        await pilot.pause()
        assert app.screen is nested
        assert app.results == []

        nested.dismiss(None)
        await pilot.pause()
        assert app.screen is modal

        await modal.action_request_safe_cancel()
        await pilot.pause()

    assert callback_calls == 1
    assert app.results == [False]


@pytest.mark.parametrize(
    "contract", TASK2_MODAL_CONTRACTS, ids=lambda row: row.modal_type.__name__
)
@pytest.mark.asyncio
async def test_task2_contract_selector_exists_and_escape_returns_cancel_result(
    contract: _Task2ModalContract,
) -> None:
    app = _Task2Harness()
    modal = contract.factory()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        assert modal.query_one(contract.content_selector)
        await pilot.press("escape")
        await pilot.pause()

    assert app.results == [contract.cancel_result]


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_auto_speak_consent_cancel_sources_return_false(source: str) -> None:
    app = _Task2Harness()
    modal = AutoSpeakConsentModal("TTS provider", "https://tts.example", False)

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        if source == "visible":
            await pilot.click("#console-auto-speak-consent-cancel")
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert app.results == [False]


@pytest.mark.parametrize(
    "contract", TASK3_MODAL_CONTRACTS, ids=lambda row: row.modal_type.__name__
)
@pytest.mark.asyncio
async def test_task3_contract_selector_exists_and_escape_returns_cancel_result(
    contract: _Task3ModalContract,
) -> None:
    app = _Task2Harness()
    modal = contract.factory()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        assert modal.query_one(contract.content_selector)
        await pilot.press("escape")
        await pilot.pause()

    assert app.results == [contract.cancel_result]


class _CountingComposerMenuModal(ConsoleComposerMenuModal):
    def __init__(self) -> None:
        super().__init__()
        self.dismiss_calls: list[object] = []

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.dismiss_calls.append(result)
        return super().dismiss(result)


class _CountingRagSettingsModal(ConsoleRagSettingsModal):
    def __init__(self) -> None:
        super().__init__()
        self.dismiss_calls: list[object] = []

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.dismiss_calls.append(result)
        return super().dismiss(result)


@pytest.mark.parametrize(
    "factory",
    [_CountingComposerMenuModal, _CountingRagSettingsModal],
    ids=["composer-menu", "rag-settings"],
)
@pytest.mark.asyncio
async def test_task3_real_backdrop_dispatch_dismisses_once_through_full_mro(
    factory: Callable[[], _CountingComposerMenuModal | _CountingRagSettingsModal],
) -> None:
    app = _Task2Harness()
    modal = factory()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click(offset=(0, 0))
        await pilot.pause()
        await pilot.pause()

    assert modal.dismiss_calls == [None]
    assert app.results == [None]


class _SettingsAdjacentClickModal(SafeModalDismissMixin, ModalScreen[None]):
    SAFE_MODAL_CONTENT = "#settings-adjacent-content"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel")]
    CSS = """
    _SettingsAdjacentClickModal { align: center middle; }
    #settings-adjacent-content { width: 30; height: 7; background: $surface; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.unrelated_clicks = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-adjacent-content"):
            yield Static("Unrelated setting", id="settings-adjacent-action")

    def on_click(self, _event: events.Click) -> None:
        self.unrelated_clicks += 1


@pytest.mark.asyncio
async def test_task3_mixin_keeps_settings_adjacent_unrelated_click_handler() -> None:
    app = _Task2Harness()
    modal = _SettingsAdjacentClickModal()

    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()

        await pilot.click("#settings-adjacent-action")
        await pilot.pause()

        assert app.screen is modal
        assert modal.unrelated_clicks == 1


class _TrackedSessionSwitcherModal(ConsoleSessionSwitcherModal):
    def __init__(self) -> None:
        super().__init__(rows=())
        self.cleanup_calls = 0
        self.order: list[str] = []

    def _cancel_query_debounce(self) -> None:
        self.cleanup_calls += 1
        self.order.append("cleanup")
        super()._cancel_query_debounce()

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.order.append("dismiss")
        return super().dismiss(result)


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_task3_session_switcher_cancel_sources_stop_real_debounce_once(
    source: str,
) -> None:
    app = _Task2Harness()
    modal = _TrackedSessionSwitcherModal()

    async with app.run_test(size=(120, 48)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        modal._query_debounce_timer = modal.set_timer(60, lambda: None)
        modal.cleanup_calls = 0
        modal.order.clear()

        if source == "visible":
            await pilot.click("#console-switcher-cancel")
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert modal._query_debounce_timer is None
    assert modal.cleanup_calls == 1
    assert modal.order[:2] == ["cleanup", "dismiss"]
    assert app.results == [None]


class _LifecycleCharacterModal(ConsoleCharacterPickerModal):
    def __init__(self) -> None:
        super().__init__(options=[])
        self.initialization_calls = 0

    async def _refresh_results(self, query: str) -> None:
        self.initialization_calls += 1
        await super()._refresh_results(query)


@pytest.mark.asyncio
async def test_textual_mro_runs_mixin_and_modal_mount_once(monkeypatch) -> None:
    mixin_mount_calls = 0
    original_mixin_mount = SafeModalDismissMixin.on_mount

    def count_mixin_mount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mixin_mount_calls
        mixin_mount_calls += 1
        original_mixin_mount(self)

    monkeypatch.setattr(SafeModalDismissMixin, "on_mount", count_mixin_mount)
    app = _Task2Harness()
    modal = _LifecycleCharacterModal()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert mixin_mount_calls == 1
        assert modal.initialization_calls == 1


@pytest.mark.asyncio
async def test_textual_mro_runs_citation_mixin_unmount_once(monkeypatch) -> None:
    mixin_unmount_calls = 0
    original_mixin_unmount = SafeModalDismissMixin.on_unmount

    def count_mixin_unmount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mixin_unmount_calls
        mixin_unmount_calls += 1
        original_mixin_unmount(self)

    monkeypatch.setattr(SafeModalDismissMixin, "on_unmount", count_mixin_unmount)
    app = _Task2Harness()
    modal = _citation_factory()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        generation = modal._request_generation

        modal.dismiss(None)
        await pilot.pause()

        assert mixin_unmount_calls == 1
        assert modal._request_generation == generation + 1


@pytest.mark.asyncio
async def test_prompt_workbench_lifecycle_dispatches_mixin_once_per_mount(
    monkeypatch,
) -> None:
    mixin_mount_calls = 0
    mixin_unmount_calls = 0
    original_mixin_mount = SafeModalDismissMixin.on_mount
    original_mixin_unmount = SafeModalDismissMixin.on_unmount

    def count_mixin_mount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mixin_mount_calls
        mixin_mount_calls += 1
        original_mixin_mount(self)

    def count_mixin_unmount(self) -> None:  # type: ignore[no-untyped-def]
        nonlocal mixin_unmount_calls
        mixin_unmount_calls += 1
        original_mixin_unmount(self)

    monkeypatch.setattr(SafeModalDismissMixin, "on_mount", count_mixin_mount)
    monkeypatch.setattr(SafeModalDismissMixin, "on_unmount", count_mixin_unmount)
    app = _Task2Harness()
    modal = ConsolePromptsModal(
        capabilities=lambda _source: object(),
        list_page=lambda _source, _page: [],
        search=lambda _source, _query: [],
        detail=lambda _source, _identifier: {},
        save=lambda **_payload: {},
    )

    async with app.run_test(size=(100, 40)) as pilot:
        for expected_mounts in (1, 2):
            await app.push_screen(modal)
            await pilot.pause()
            assert mixin_mount_calls == expected_mounts

            modal.dismiss(None)
            await pilot.pause()
            assert mixin_unmount_calls == expected_mounts


class _TrackedCharacterModal(ConsoleCharacterPickerModal):
    def __init__(self) -> None:
        super().__init__(options=[ConsoleCharacterOption(1, "Ada")])
        self.cleanup_calls = 0
        self.order: list[str] = []

    def _cancel_query_debounce(self) -> None:
        self.cleanup_calls += 1
        self.order.append("cleanup")
        super()._cancel_query_debounce()

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.order.append("dismiss")
        return super().dismiss(result)


class _TrackedCitationModal(ConsoleCitationSourcesModal):
    def __init__(self) -> None:
        super().__init__(
            native_message_id="native-1",
            persisted_message_id="persisted-1",
            current_body="body",
            repository=object(),
            request_is_current=lambda: True,
        )
        self._worker_started = True
        self.generation_at_dismiss: list[int] = []

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.generation_at_dismiss.append(self._request_generation)
        return super().dismiss(result)


class _TrackedStyleModal(ConsoleStylePickerModal):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_calls = 0
        self.order: list[str] = []

    def _cancel_search_debounce(self) -> None:
        self.cleanup_calls += 1
        self.order.append("cleanup")
        super()._cancel_search_debounce()

    def dismiss(self, result=None):  # type: ignore[no-untyped-def]
        self.order.append("dismiss")
        return super().dismiss(result)


async def _request_task2_cancel(modal, pilot, source: str) -> None:  # type: ignore[no-untyped-def]
    if source == "visible":
        if isinstance(modal, _TrackedCitationModal):
            await pilot.click("#console-citation-sources-close")
        else:
            result = modal.action_dismiss_picker()
            if inspect.isawaitable(result):
                await result
    elif source == "escape":
        await pilot.press("escape")
    else:
        await pilot.click(offset=(0, 0))
    await pilot.pause()


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_character_cancel_sources_run_debounce_cleanup_once(source: str) -> None:
    app = _Task2Harness()
    modal = _TrackedCharacterModal()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        modal._query_debounce_timer = modal.set_timer(60, lambda: None)
        modal.cleanup_calls = 0
        modal.order.clear()

        await _request_task2_cancel(modal, pilot, source)

    assert app.results == [None]
    assert modal.cleanup_calls == 1
    assert modal.order[:2] == ["cleanup", "dismiss"]


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_citation_cancel_sources_invalidate_generation_once_before_dismiss(
    source: str,
) -> None:
    app = _Task2Harness()
    modal = _TrackedCitationModal()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        modal._request_generation = 10

        await _request_task2_cancel(modal, pilot, source)

    assert app.results == [None]
    assert modal.generation_at_dismiss == [11]


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_style_cancel_sources_run_debounce_cleanup_once(source: str) -> None:
    app = _Task2Harness()
    modal = _TrackedStyleModal()
    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal, callback=app.results.append)
        await pilot.pause()
        modal._search_debounce_timer = modal.set_timer(60, lambda: None)
        modal.cleanup_calls = 0
        modal.order.clear()

        await _request_task2_cancel(modal, pilot, source)

    assert app.results == [None]
    assert modal.cleanup_calls == 1
    assert modal.order[:2] == ["cleanup", "dismiss"]


@dataclass(frozen=True)
class _FakeRegion:
    contains_point: bool

    def contains(self, _x: int, _y: int) -> bool:
        return self.contains_point


@dataclass(frozen=True)
class _FakeContent:
    region: _FakeRegion


@pytest.mark.parametrize(
    ("button", "known", "descendant", "contains", "expected"),
    [
        pytest.param(1, True, False, False, True, id="primary-outside"),
        pytest.param(1, True, True, False, False, id="primary-descendant"),
        pytest.param(1, True, False, True, False, id="primary-in-region"),
        pytest.param(2, True, False, False, False, id="non-primary-outside"),
        pytest.param(1, False, False, False, False, id="unknown-provenance"),
    ],
)
def test_classifier_identifies_only_known_primary_backdrop_clicks(
    button: int,
    known: bool,
    descendant: bool,
    contains: bool,
    expected: bool,
) -> None:
    content = _FakeContent(_FakeRegion(contains))

    assert (
        is_modal_backdrop_click(
            button=button,
            provenance_known=known,
            target_is_content_or_descendant=descendant,
            point_is_in_content_region=content.region.contains(7, 9),
        )
        is expected
    )


class _HostScreen(Screen[None]):
    """Revealed screen with the same optional focus seam as Console."""

    def __init__(self) -> None:
        super().__init__()
        self.composer_fallback_calls: list[bool] = []
        self.underlying_button_presses = 0
        self.unrelated_button_presses = 0
        self.screen_mouse_ups = 0
        self.screen_clicks = 0

    def compose(self) -> ComposeResult:
        yield Button("Underlying action", id="modal-test-underlying-action")
        yield Input(id="modal-test-opener")
        yield Input(id="modal-test-other-focus")
        yield Button("Unrelated action", id="modal-test-unrelated-action")
        yield Static("host", id="modal-test-host-label")

    @on(Button.Pressed, "#modal-test-underlying-action")
    def _underlying_action(self) -> None:
        self.underlying_button_presses += 1

    @on(Button.Pressed, "#modal-test-unrelated-action")
    def _unrelated_action(self) -> None:
        self.unrelated_button_presses += 1

    def on_mouse_up(self, _event: events.MouseUp) -> None:
        self.screen_mouse_ups += 1

    def on_click(self, _event: events.Click) -> None:
        self.screen_clicks += 1

    def _focus_console_composer_if_needed(self, *, force: bool) -> None:
        self.composer_fallback_calls.append(force)


class _ModalHarness(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.host = _HostScreen()

    async def on_mount(self) -> None:
        await self.push_screen(self.host)


class _NestedModal(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Static("nested", id="modal-test-nested")


CancelEffect = Callable[[], Awaitable[None]]


class _SafeTestModal(SafeModalDismissMixin, ModalScreen[bool | None]):
    SAFE_MODAL_CONTENT = "#modal-test-content"
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel")]
    CSS = """
    _SafeTestModal {
        align: center middle;
    }

    #modal-test-content {
        width: 30;
        height: 7;
        background: $surface;
    }
    """

    def __init__(
        self,
        *,
        result: bool | None = False,
        cancel_effect: CancelEffect | None = None,
    ) -> None:
        super().__init__()
        self._cancel_result = result
        self._cancel_effect = cancel_effect

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-test-content"):
            yield Static("safe modal", id="modal-test-descendant")
            yield Button("Cancel", id="modal-test-cancel")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._cancel_effect is not None:
            await self.run_cancel_effect_once(self._cancel_effect)
        self.dismiss_safe_once(self._cancel_result)

    @on(Button.Pressed, "#modal-test-cancel")
    async def _cancel_from_button(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")


async def _mount_modal(
    app: _ModalHarness,
    pilot,
    modal: _SafeTestModal,
    results: list[bool | None],
    *,
    opener_selector: str = "#modal-test-opener",
) -> Input:
    opener = app.host.query_one(opener_selector, Input)
    opener.focus()
    await pilot.pause()
    assert app.host.focused is opener
    app.push_screen(modal, results.append)
    await pilot.pause()
    assert app.screen is modal
    return opener


def _outside_click(
    modal: _SafeTestModal, screen_x: int = 0, screen_y: int = 0
) -> events.Click:
    return events.Click(
        modal,
        screen_x,
        screen_y,
        0,
        0,
        1,
        False,
        False,
        False,
        screen_x=screen_x,
        screen_y=screen_y,
    )


@pytest.mark.asyncio
async def test_single_shot_consumes_repeated_escape_and_backdrop_while_pending():
    entered = asyncio.Event()
    release = asyncio.Event()
    effect_calls = 0

    async def delayed_effect() -> None:
        nonlocal effect_calls
        effect_calls += 1
        entered.set()
        await release.wait()

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=delayed_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        first_escape = asyncio.create_task(modal.action_request_safe_cancel())
        await wait_for_background_signal(
            entered,
            first_escape,
            what="the single-shot cancel effect to start",
        )
        second_escape = asyncio.create_task(modal.action_request_safe_cancel())
        backdrop = _outside_click(modal)
        backdrop_request = asyncio.create_task(modal.on_click(backdrop))
        try:
            await asyncio.sleep(0)
            assert second_escape.done()
            assert backdrop_request.done()
            assert effect_calls == 1
            assert app.screen is modal
            assert backdrop._stop_propagation
            assert backdrop._no_default_action
        finally:
            release.set()
            await await_background_task(
                first_escape,
                what="the single-shot cancel effect to finish",
            )
            await await_background_task(
                second_escape,
                what="the repeated single-shot cancel request to finish",
            )
            await await_background_task(
                backdrop_request,
                what="the pending backdrop request to finish",
            )
        await pilot.pause()

        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.asyncio
async def test_top_screen_check_preserves_nested_modal_and_retry_skips_effect():
    effect_calls = 0
    nested = _NestedModal()

    async def push_nested() -> None:
        nonlocal effect_calls
        effect_calls += 1
        app.push_screen(nested)

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=push_nested)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert app.screen is nested
        assert modal.is_mounted
        assert results == []
        assert effect_calls == 1

        nested.dismiss(None)
        await pilot.pause()
        assert app.screen is modal

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert app.screen is app.host
        assert results == [False]
        assert effect_calls == 1


@pytest.mark.asyncio
async def test_single_shot_cancel_effect_commitment_survives_exception():
    effect_calls = 0

    async def failing_effect() -> None:
        nonlocal effect_calls
        effect_calls += 1
        raise RuntimeError("cancel failed")

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=failing_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        with pytest.raises(RuntimeError, match="cancel failed"):
            await modal.action_request_safe_cancel()
        assert app.screen is modal

        await modal.action_request_safe_cancel()
        await pilot.pause()

        assert effect_calls == 1
        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.parametrize("source", ["button", "backdrop"])
@pytest.mark.asyncio
async def test_visible_cancel_and_backdrop_return_exact_typed_value(source: str):
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(result=False)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        if source == "button":
            await pilot.click("#modal-test-cancel")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

        assert len(results) == 1
        assert results[0] is False
        assert app.screen is app.host


@pytest.mark.asyncio
async def test_opener_focus_restores_the_exact_eligible_console_opener():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        opener = await _mount_modal(app, pilot, modal, results)
        app.host.set_focus(app.host.query_one("#modal-test-other-focus", Input))
        assert app.host.focused is not opener

        await modal.action_request_safe_cancel()
        await pilot.pause()
        await pilot.pause()

        assert app.host.focused is opener
        assert app.host.composer_fallback_calls == []


@pytest.mark.asyncio
async def test_console_composer_fallback_runs_when_opener_was_removed():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        opener = await _mount_modal(app, pilot, modal, results)
        await opener.remove()

        await modal.action_request_safe_cancel()
        await pilot.pause()
        await pilot.pause()

        assert app.host.composer_fallback_calls == [True]


@pytest.mark.asyncio
async def test_backdrop_shield_is_inert_to_revealed_screen_and_focus():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (
            underlying.region.x + 1,
            underlying.region.y + 1,
        )
        opener = await _mount_modal(app, pilot, modal, results)
        assert not modal.query_one("#modal-test-content", Vertical).region.contains(
            *click_point
        )

        await pilot.click(offset=click_point)
        await pilot.click(offset=click_point)
        await pilot.pause()

        assert results == [False]
        assert app.host.underlying_button_presses == 0
        assert app.host.screen_mouse_ups == 0
        assert app.host.screen_clicks == 0
        assert app.host.focused is opener
        assert app.mouse_captured is None

        await pilot.pause(app.CLICK_CHAIN_TIME_THRESHOLD + 0.05)

        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 1


@pytest.mark.asyncio
async def test_backdrop_shield_allows_an_unrelated_coordinate_action():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        unrelated = app.host.query_one("#modal-test-unrelated-action", Button)
        origin = (underlying.region.x + 1, underlying.region.y + 1)
        unrelated_point = (unrelated.region.x + 1, unrelated.region.y + 1)
        await _mount_modal(app, pilot, modal, results)

        await pilot.click(offset=origin)
        await pilot.click(offset=unrelated_point)
        await pilot.pause()

        assert app.host.underlying_button_presses == 0
        assert app.host.unrelated_button_presses == 1
        assert app.mouse_captured is None


@pytest.mark.asyncio
async def test_safe_modal_state_resets_when_same_instance_is_repushed():
    effect_calls = 0

    async def effect() -> None:
        nonlocal effect_calls
        effect_calls += 1

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=effect)

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)
        await modal.action_request_safe_cancel()
        await pilot.pause()
        assert app.screen is app.host

        second_opener = await _mount_modal(
            app,
            pilot,
            modal,
            results,
            opener_selector="#modal-test-other-focus",
        )
        app.host.set_focus(app.host.query_one("#modal-test-opener", Input))
        await modal.action_request_safe_cancel()
        await pilot.pause()
        await pilot.pause()

        assert app.screen is app.host
        assert results == [False, False]
        assert effect_calls == 2
        assert app.host.focused is second_opener


@pytest.mark.asyncio
async def test_real_click_dispatch_keeps_descendant_and_inside_geometry_open():
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)

        await pilot.click("#modal-test-descendant")
        await pilot.pause()
        assert app.screen is modal

        content = modal.query_one("#modal-test-content", Vertical)
        inside_blank_point = (content.region.right - 2, content.region.bottom - 1)
        await pilot.click(offset=inside_blank_point)
        await pilot.pause()

        assert app.screen is modal
        assert results == []


@pytest.mark.parametrize("button", [2, 3], ids=["middle", "secondary"])
@pytest.mark.asyncio
async def test_real_non_primary_backdrop_dispatch_keeps_modal_open(button: int):
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        await _mount_modal(app, pilot, modal, results)
        event = events.Click(
            modal,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=button,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=0,
            screen_y=0,
        )

        await modal._dispatch_message(event)
        await pilot.pause()

        assert app.screen is modal
        assert results == []


@pytest.mark.parametrize("source", ["escape", "button"])
@pytest.mark.asyncio
async def test_non_backdrop_cancel_does_not_shield_revealed_screen(source: str):
    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal()

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, results)
        if source == "escape":
            await modal.action_request_safe_cancel()
        else:
            await pilot.click("#modal-test-cancel")
        await pilot.pause()

        assert app.screen is app.host
        assert app.mouse_captured is None
        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 1


@pytest.mark.asyncio
async def test_pending_escape_records_backdrop_before_terminal_dismissal():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_effect() -> None:
        entered.set()
        await release.wait()

    app = _ModalHarness()
    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=delayed_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, results)

        pending_escape = asyncio.create_task(modal.action_request_safe_cancel())
        await wait_for_background_signal(
            entered,
            pending_escape,
            what="the pending escape effect to start",
        )
        await pilot.click(offset=click_point)
        assert app.screen is modal

        release.set()
        await await_background_task(
            pending_escape,
            what="the pending escape effect to finish",
        )
        await pilot.pause()
        assert app.screen is app.host

        await pilot.click(offset=click_point)
        await pilot.pause()

        assert app.host.underlying_button_presses == 0
        assert app.host.screen_mouse_ups == 0
        assert app.host.screen_clicks == 0
        assert app.mouse_captured is None

        await pilot.pause(app.CLICK_CHAIN_TIME_THRESHOLD + 0.05)
        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 1


@pytest.mark.parametrize("retry_source", ["escape", "button"])
@pytest.mark.asyncio
async def test_stale_backdrop_attempt_does_not_shield_later_retry(
    retry_source: str,
):
    app = _ModalHarness()
    nested = _NestedModal()

    async def push_nested() -> None:
        app.push_screen(nested)

    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=push_nested)

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, results)

        await pilot.click(offset=(0, 0))
        await pilot.pause()
        assert app.screen is nested

        nested.dismiss(None)
        await pilot.pause(app.CLICK_CHAIN_TIME_THRESHOLD + 0.05)
        assert app.screen is modal

        if retry_source == "escape":
            await modal.action_request_safe_cancel()
        else:
            await pilot.click("#modal-test-cancel")
        await pilot.pause()
        await pilot.pause()

        assert app.screen is app.host
        assert app.mouse_captured is None

        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 1


@pytest.mark.asyncio
async def test_expired_backdrop_chain_adds_no_shield():
    app = _ModalHarness()

    async def outlive_click_chain() -> None:
        await asyncio.sleep(app.CLICK_CHAIN_TIME_THRESHOLD + 0.05)

    results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=outlive_click_chain)

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, results)

        await pilot.click(offset=click_point)
        assert app.screen is app.host

        await pilot.click(offset=click_point)
        await pilot.pause()

        assert app.host.underlying_button_presses == 1
        assert app.mouse_captured is None


@pytest.mark.asyncio
async def test_old_request_generation_cannot_dismiss_repushed_presentation():
    first_entered = asyncio.Event()
    first_release = asyncio.Event()
    second_entered = asyncio.Event()
    second_release = asyncio.Event()
    effect_calls = 0

    async def generation_effect() -> None:
        nonlocal effect_calls
        effect_calls += 1
        if effect_calls == 1:
            first_entered.set()
            await first_release.wait()
        else:
            second_entered.set()
            await second_release.wait()

    app = _ModalHarness()
    first_results: list[bool | None] = []
    second_results: list[bool | None] = []
    modal = _SafeTestModal(cancel_effect=generation_effect)

    async with app.run_test(size=(80, 24)) as pilot:
        underlying = app.host.query_one("#modal-test-underlying-action", Button)
        click_point = (underlying.region.x + 1, underlying.region.y + 1)
        await _mount_modal(app, pilot, modal, first_results)
        old_request = asyncio.create_task(modal.action_request_safe_cancel())
        await wait_for_background_signal(
            first_entered,
            old_request,
            what="the first modal generation effect to start",
        )

        modal.dismiss(None)
        await pilot.pause()
        assert first_results == [None]
        assert app.screen is app.host

        await _mount_modal(app, pilot, modal, second_results)
        new_request = asyncio.create_task(
            modal.on_click(_outside_click(modal, *click_point))
        )
        await wait_for_background_signal(
            second_entered,
            new_request,
            what="the replacement modal generation effect to start",
        )

        first_release.set()
        await await_background_task(
            old_request,
            what="the retired modal generation effect to finish",
        )
        await pilot.pause()

        try:
            assert app.screen is modal
            assert second_results == []
            await modal.action_request_safe_cancel()
            assert app.screen is modal
            assert effect_calls == 2
        finally:
            second_release.set()
            await await_background_task(
                new_request,
                what="the replacement modal generation effect to finish",
            )
        await pilot.pause()

        assert app.screen is app.host
        assert second_results == [False]
        await pilot.click(offset=click_point)
        await pilot.pause()
        assert app.host.underlying_button_presses == 0


def test_launch_walk_reports_every_mismatch_not_just_the_first() -> None:
    """Every mismatch is reported, not just the first one found.

    task-18810: the walk used to assert per owner, so the FIRST bad owner
    aborted it and every later owner -- plus the calling test's own
    assertions -- silently stopped being checked. Two real undeclared
    launches shipped behind that. Mismatches are now collected and returned
    together, so one stale declaration cannot hide the rest.
    """
    first_path = "synthetic_multi_first.py"
    second_path = "synthetic_multi_second.py"
    edges = (
        _ModalLaunchEdge(
            "SyntheticRoot",
            (_SyntheticDeclaredOwner,),
            (first_path,),
        ),
        _ModalLaunchEdge(
            _SyntheticDeclaredOwner,
            (),
            (second_path,),
        ),
    )
    sources = {
        first_path: """
from tldw_chatbook.Widgets.Console.console_image_viewer_modal import ConsoleImageViewerModal as First
First(image_bytes=b"", mime_type="image/png")
""",
        second_path: """
from tldw_chatbook.Widgets.Console.console_run_log_modal import ConsoleRunLogModal as Second
Second(run_id='extra', log_text='extra')
""",
    }

    result = _walk_modal_launch_graph(
        "SyntheticRoot",
        edges,
        source_overrides=sources,
        owner_source_paths={_SyntheticDeclaredOwner: (second_path,)},
    )

    message = "\n".join(result.mismatches)
    assert "ConsoleImageViewerModal" in message, message
    assert "ConsoleRunLogModal" in message, message


def test_launch_walk_scans_beneath_an_undeclared_modal() -> None:
    """Mismatches beneath an undeclared modal are reported too.

    task-18810 review: an UNDECLARED modal used to be reported but never
    traversed, so a stale parent declaration still hid every mismatch below
    it. The walk now scans strays as well, without counting them as
    reachable -- that stays the declared set the contract table is compared
    against.
    """
    root_path = "synthetic_stray_root.py"
    stray_path = "synthetic_stray_owner.py"
    edges = (
        _ModalLaunchEdge("SyntheticRoot", (), (root_path,)),
        _ModalLaunchEdge(_SyntheticStrayOwner, (), (stray_path,)),
    )
    sources = {
        # The root constructs the stray owner's modal without declaring it.
        root_path: """
from Tests.UI.test_console_modal_dismissal import _SyntheticStrayOwner as Stray
Stray()
""",
        # ...and that stray itself constructs a second undeclared modal.
        stray_path: """
from tldw_chatbook.Widgets.Console.console_run_log_modal import ConsoleRunLogModal as Hidden
Hidden(run_id='hidden', log_text='hidden')
""",
    }

    result = _walk_modal_launch_graph(
        "SyntheticRoot",
        edges,
        source_overrides=sources,
        owner_source_paths={_SyntheticStrayOwner: (stray_path,)},
    )

    message = "\n".join(result.mismatches)
    assert "_SyntheticStrayOwner" in message, message
    # The mismatch BELOW the undeclared modal is the point of this test.
    assert "ConsoleRunLogModal" in message, message
    # Strays are scanned, not promoted into the declared-reachable set.
    assert _SyntheticStrayOwner not in result.reachable
