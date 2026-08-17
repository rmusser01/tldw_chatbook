"""Workspace and conversation-browser ownership for the Console.

`ConsoleWorkspaceController` owns 56 non-DOM methods covering Workspace
policy and lifecycle, scope selection, persisted-conversation resume, and the
grouped conversation browser. Browser query, timer, token, rich rows, totals,
errors, and persisted-row cache have one canonical home here. Legacy
Workspace row and scalar names are compatibility aliases: reads project from
the rich rows and writes convert back into canonical rich state.

The Textual screen retains only framework and DOM edges. Its bounded search
handler extracts plain query/disabled values and delegates the transition;
the Clear button delegates the complete clear transition. Browser services
and sibling-owned data enter through explicit late-bound constructor
callables so replacements remain observable and moved methods never query the
DOM or reach through sibling controllers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import partial
from typing import Any, Optional, TYPE_CHECKING
import asyncio
import inspect
import time

from loguru import logger
from rich.markup import escape as escape_markup

from ...Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    CONSOLE_RUN_MARKER_GLYPHS,
    DEFAULT_CONSOLE_SESSION_TITLE,
    ConsoleRunMarker,
    ConsoleStagedSource,
    ConsoleWorkspaceContext,
)
from ...Chat.console_display_state import evidence_bundle_from_launch
from ...Chat.console_live_work import ConsoleLiveWorkLaunch
from ...Chat.console_conversation_hydration import (
    ConversationLoadFailed,
    ConversationServiceUnavailable,
    hydrate_console_session,
    load_console_conversation_tree,
)
from ...Chat.rag_scope import RagScope
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.glyph_fallback import resolve_glyph
from ...Widgets.Console import (
    ConsoleWorkspaceContextTray,
    ConsoleWorkspaceRenameModal,
    ConsoleWorkspaceSwitcherModal,
)
from ...Widgets.Console.console_scope_picker_modal import ConsoleScopePickerModal
from ...Workspaces import (
    CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    ConsoleConversationBrowserInputRow,
    ConsoleConversationBrowserRow,
    DEFAULT_WORKSPACE_ID,
    WorkspaceRecord,
    build_console_conversation_browser_state,
    console_persisted_row_updated_sort,
)
from ...Workspaces.display_state import (
    CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationRow,
    ConsoleWorkspaceConversationSectionState,
    build_console_workspace_state,
    console_workspace_conversation_result_copy,
)
from ...Workspaces.registry_service import (
    WorkspaceNotFound,
    WorkspaceRegistryServiceError,
    next_local_workspace_identity,
)
from ..character_display_text import sanitize_character_display_label

if TYPE_CHECKING:
    from ...Chat.console_chat_controller import ConsoleChatController
    from ..Screens.chat_screen import ChatScreen

logger = logger.bind(module="ChatScreen")

CONSOLE_PERSISTED_ROWS_CACHE_TTL_SECONDS = 2.0


class ConsoleWorkspaceController:
    """Own Workspace lifecycle, resume, scope, and conversation browsing.

    The controller holds the canonical rich browser state and exposes the
    legacy Workspace search shape only as a compatibility projection. Its 56
    owned methods use explicit late-bound dependencies for non-Workspace
    services; Textual event binding, focus, and rendering remain on the screen.
    """

    def __init__(
        self,
        screen: "ChatScreen",
        *,
        app_instance: Any,
        chat_store_accessor: Callable[[], Any],
        current_chat_store_accessor: Callable[[], Any],
        current_conversation_id_accessor: Callable[[], str | None],
        native_session_rows_accessor: Callable[
            [ConsoleWorkspaceContextState], ConsoleWorkspaceContextState
        ],
        capture_draft_switch_snapshot: Callable[[], None],
        sync_chat_core_state: Callable[[], None],
        sync_native_console_chat_ui: Callable[[], Any],
        sync_temporary_chip: Callable[[], None],
        default_session_settings_accessor: Callable[[], Any],
        scope_picker_listers_accessor: Callable[[], tuple[Any, Any, Any]],
        active_native_session_accessor: Callable[[], Any],
        refresh_effective_scope_and_sync: Callable[[Any], Any],
        messages_from_conversation_tree_accessor: Callable[[dict], list],
        session_settings_for_resume_accessor: Callable[[Any], Any],
        resolve_resumed_character_name: Callable[[int], Any],
        inject_resume_agent_markers_accessor: Callable[[list, str], list],
        resolve_effective_scope_state: Callable[[Any], Any],
        sync_retrieval_scope_row: Callable[[], None],
        note_follow_intent: Callable[[], None],
        focus_composer_if_needed: Callable[..., None],
        conversation_section_config_accessor: Callable[[], dict],
        conversation_browser_config: Callable[[], dict[str, Any]],
        focus_conversation_search: Callable[[], None],
        sync_workspace_context: Callable[[], None],
        schedule_timer: Callable[[float, Callable[[], None]], Any],
        screen_running_accessor: Callable[[], bool],
        current_chat_controller_accessor: Callable[[], Any],
        fleet_unseen_ids_accessor: Callable[[], frozenset[str]],
        run_marker_with_unseen: Callable[[Any, Any, frozenset[str]], Any],
        broken_conversation_ids_accessor: Callable[[], set[str]],
        ensure_agent_bridge: Callable[[], Any],
        subagent_counts_for_rows: Callable[[Any, Iterable[Any]], dict[str, int]],
        conversation_browser_collapse_preferences: Callable[[], dict[str, bool]],
        wake_retry_poke: Callable[[], None] | None = None,
    ) -> None:
        """Bind canonical Workspace state and its late-bound dependencies.

        Framework services are read from ``screen`` when used. Browser data,
        configuration, scheduling, markers, sub-agent counts, focus, and final
        Workspace synchronization are supplied as named callables, keeping
        controller ownership explicit without snapshotting replaceable screen
        collaborators.

        Args:
            screen: Owning Console screen for retained framework services.
            app_instance: Application service container.
            chat_store_accessor: Resolve or create the Console chat store.
            current_chat_store_accessor: Return the current Console chat store.
            current_conversation_id_accessor: Return the active conversation id.
            native_session_rows_accessor: Add native session rows to Workspace state.
            capture_draft_switch_snapshot: Preserve the draft before switching.
            sync_chat_core_state: Refresh Console chat-core state.
            sync_native_console_chat_ui: Refresh the native Console UI.
            sync_temporary_chip: Refresh the temporary-chat chip.
            default_session_settings_accessor: Return default session settings.
            scope_picker_listers_accessor: Return scope-picker data providers.
            active_native_session_accessor: Return the active native session.
            refresh_effective_scope_and_sync: Refresh effective retrieval scope.
            messages_from_conversation_tree_accessor: Convert a saved message tree.
            session_settings_for_resume_accessor: Resolve resumed session settings.
            resolve_resumed_character_name: Resolve a resumed character label.
            inject_resume_agent_markers_accessor: Add agent markers on resume.
            resolve_effective_scope_state: Resolve effective scope state.
            sync_retrieval_scope_row: Refresh the retrieval-scope row.
            note_follow_intent: Record that Console should follow the selection.
            focus_composer_if_needed: Restore composer focus when appropriate.
            conversation_section_config_accessor: Return legacy section config.
            conversation_browser_config: Return grouped-browser config.
            focus_conversation_search: Focus the browser search input.
            sync_workspace_context: Render current Workspace state.
            schedule_timer: Schedule a delayed browser callback.
            screen_running_accessor: Return whether the owning screen is running.
            current_chat_controller_accessor: Return the current chat controller.
            fleet_unseen_ids_accessor: Return unseen conversation ids.
            run_marker_with_unseen: Project unseen state into a run marker.
            broken_conversation_ids_accessor: Return non-openable conversation ids.
            ensure_agent_bridge: Return the Console agent bridge.
            subagent_counts_for_rows: Compute sub-agent counts for browser rows.
            conversation_browser_collapse_preferences: Return collapse preferences.
            wake_retry_poke: Optionally request a staged-wake retry after resume.
        """
        self._screen = screen
        self.app_instance = app_instance
        self._chat_store_accessor = chat_store_accessor
        self._current_chat_store_accessor = current_chat_store_accessor
        self._current_conversation_id_accessor = current_conversation_id_accessor
        self._native_session_rows_accessor = native_session_rows_accessor
        self._capture_draft_switch_snapshot_fn = capture_draft_switch_snapshot
        self._sync_chat_core_state_fn = sync_chat_core_state
        self._sync_native_console_chat_ui_fn = sync_native_console_chat_ui
        self._sync_temporary_chip_fn = sync_temporary_chip
        self._default_session_settings_accessor = default_session_settings_accessor
        self._scope_picker_listers_accessor = scope_picker_listers_accessor
        self._active_native_session_accessor = active_native_session_accessor
        self._refresh_effective_scope_and_sync_fn = refresh_effective_scope_and_sync
        self._messages_from_conversation_tree_accessor = (
            messages_from_conversation_tree_accessor
        )
        self._session_settings_for_resume_accessor = (
            session_settings_for_resume_accessor
        )
        self._resolve_resumed_character_name_fn = resolve_resumed_character_name
        self._inject_resume_agent_markers_accessor = (
            inject_resume_agent_markers_accessor
        )
        self._resolve_effective_scope_state_fn = resolve_effective_scope_state
        self._sync_retrieval_scope_row_fn = sync_retrieval_scope_row
        self._note_follow_intent_fn = note_follow_intent
        self._focus_composer_if_needed_fn = focus_composer_if_needed
        self._conversation_section_config_accessor = (
            conversation_section_config_accessor
        )
        self._conversation_browser_config_fn = conversation_browser_config
        self._focus_conversation_search_fn = focus_conversation_search
        self._sync_workspace_context_fn = sync_workspace_context
        self._schedule_timer_fn = schedule_timer
        self._screen_running_accessor = screen_running_accessor
        self._current_chat_controller_accessor = current_chat_controller_accessor
        self._fleet_unseen_ids_accessor = fleet_unseen_ids_accessor
        self._run_marker_with_unseen_fn = run_marker_with_unseen
        self._broken_conversation_ids_accessor = broken_conversation_ids_accessor
        self._ensure_agent_bridge_fn = ensure_agent_bridge
        self._subagent_counts_for_rows_fn = subagent_counts_for_rows
        self._conversation_browser_collapse_preferences_fn = (
            conversation_browser_collapse_preferences
        )
        #: task-15864 AC#2: `ChatScreen._poke_console_wake_retry` -- resume
        #: is the one loader of persisted conversations into sessions, so
        #: session-open becomes a wake retry trigger here. Optional so the
        #: pre-existing direct-construction tests need no new kwarg.
        self._wake_retry_poke_fn = wake_retry_poke

        # Canonical grouped-browser state. Legacy workspace-search names below
        # are aliases/projections, never a second writer or backing row store.
        self._console_persisted_rows_cache = None
        self._console_persisted_rows_cache_key = None
        self._console_persisted_rows_cache_at = 0.0
        self._console_persisted_rows_cache_token = 0
        self._console_persisted_rows_refresh_key = None
        self._console_conversation_browser_query = ""
        self._console_conversation_browser_search_timer = None
        self._console_conversation_browser_search_token = 0
        self._console_conversation_browser_rows: tuple[
            ConsoleConversationBrowserInputRow, ...
        ] = ()
        self._console_conversation_browser_total = None
        self._console_conversation_browser_error = ""
        self._console_workspace_conversation_workspace_id: str | None = None
        # task-15471: serializes star toggles now that the durable write
        # runs off the loop -- two rapid presses must queue (each resolving
        # current truth before toggling), never race two pool threads into
        # a stale double-star.
        self._console_star_toggle_lock = asyncio.Lock()

    @property
    def _console_workspace_conversation_query(self) -> str:
        return self._console_conversation_browser_query

    @_console_workspace_conversation_query.setter
    def _console_workspace_conversation_query(self, value: str) -> None:
        self._console_conversation_browser_query = value

    @property
    def _console_workspace_conversation_search_timer(self) -> Any:
        return self._console_conversation_browser_search_timer

    @_console_workspace_conversation_search_timer.setter
    def _console_workspace_conversation_search_timer(self, value: Any) -> None:
        self._console_conversation_browser_search_timer = value

    @property
    def _console_workspace_conversation_search_token(self) -> int:
        return self._console_conversation_browser_search_token

    @_console_workspace_conversation_search_token.setter
    def _console_workspace_conversation_search_token(self, value: int) -> None:
        self._console_conversation_browser_search_token = value

    @property
    def _console_workspace_conversation_search_rows(
        self,
    ) -> tuple[ConsoleWorkspaceConversationRow, ...]:
        return tuple(
            ConsoleWorkspaceConversationRow(
                conversation_id=str(row.conversation_id or ""),
                title=row.title,
                status=row.status,
                selected=row.selected,
            )
            for row in self._console_conversation_browser_rows
        )

    @_console_workspace_conversation_search_rows.setter
    def _console_workspace_conversation_search_rows(
        self, value: Iterable[ConsoleWorkspaceConversationRow]
    ) -> None:
        self._console_conversation_browser_rows = tuple(
            ConsoleConversationBrowserInputRow(
                row_key=row.conversation_id,
                conversation_id=row.conversation_id,
                native_session_id=None,
                title=row.title,
                scope_type="workspace",
                workspace_id=DEFAULT_WORKSPACE_ID,
                workspace_label="Chats",
                status=row.status,
                selected=row.selected,
                source_kind="persisted",
            )
            for row in value
        )

    @property
    def _console_workspace_conversation_search_total(self) -> int | None:
        return self._console_conversation_browser_total

    @_console_workspace_conversation_search_total.setter
    def _console_workspace_conversation_search_total(self, value: int | None) -> None:
        self._console_conversation_browser_total = value

    @property
    def _console_workspace_conversation_search_error(self) -> str:
        return self._console_conversation_browser_error

    @_console_workspace_conversation_search_error.setter
    def _console_workspace_conversation_search_error(self, value: str) -> None:
        self._console_conversation_browser_error = value

    # -- Framework services (kind 1: live-read via `@property`) ------------

    @property
    def run_worker(self) -> Any:
        """`Screen.run_worker`, bound. See `__init__`'s docstring for why
        this is a property rather than a value snapshotted once."""
        return self._screen.run_worker

    @property
    def push_screen(self) -> Any:
        """`Screen.app.push_screen`, bound. See `__init__`'s docstring."""
        return self._screen.app.push_screen

    @property
    def call_after_refresh(self) -> Any:
        """`Screen.call_after_refresh`, bound. See `__init__`'s docstring."""
        return self._screen.call_after_refresh

    # -- Sibling clusters' reach-backs (kind 2) -----------------------------

    @property
    def _pending_console_launch_context(self) -> Optional[ConsoleLiveWorkLaunch]:
        """The staged-evidence cluster's own attribute (not extracted this
        wave). Live `@property` through `screen`, never snapshotted."""
        return self._screen._pending_console_launch_context

    @property
    def _console_agent_drilldown_run_id(self) -> str | None:
        """The sub-agent drill-in cluster's own attribute. Read+write:
        the moved resume body clears it unconditionally on every resume."""
        return self._screen._console_agent_drilldown_run_id

    @_console_agent_drilldown_run_id.setter
    def _console_agent_drilldown_run_id(self, value: str | None) -> None:
        self._screen._console_agent_drilldown_run_id = value

    # -- Named constructor dependencies (kind 3) ----------------------------
    #
    # Each property below is a thin wrapper around a stored callable, kept
    # under the SAME name the original `ChatScreen` method/attribute used --
    # see `__init__`'s docstring. Two shapes:
    #
    # - The original was a METHOD CALL (`self._name(...)`): the property
    #   returns the stored callable itself, unbound-called by the moved
    #   body exactly as before (`self._name(...)` still resolves, and a
    #   bare reference like `self.call_after_refresh(self._name)` also
    #   still works, since the property hands back the same callable
    #   object either way).
    # - The original was a PLAIN ATTRIBUTE READ (`self._name`, no `()`):
    #   the property CALLS the stored accessor immediately and returns the
    #   value, matching the bare-read call shape (`_console_chat_store`
    #   only).

    @property
    def _ensure_console_chat_store(self) -> Any:
        return self._chat_store_accessor

    @property
    def _console_chat_store(self) -> Any:
        return self._current_chat_store_accessor()

    @property
    def _current_console_conversation_id(self) -> Any:
        return self._current_conversation_id_accessor

    @property
    def _with_native_console_session_rows(self) -> Any:
        return self._native_session_rows_accessor

    @property
    def _capture_console_draft_switch_snapshot(self) -> Any:
        return self._capture_draft_switch_snapshot_fn

    @property
    def _sync_console_chat_core_state(self) -> Any:
        return self._sync_chat_core_state_fn

    @property
    def _sync_native_console_chat_ui(self) -> Any:
        return self._sync_native_console_chat_ui_fn

    @property
    def _sync_console_temporary_chip(self) -> Any:
        return self._sync_temporary_chip_fn

    @property
    def _default_console_session_settings(self) -> Any:
        return self._default_session_settings_accessor

    @property
    def _console_scope_picker_listers(self) -> Any:
        return self._scope_picker_listers_accessor

    @property
    def _active_native_console_session(self) -> Any:
        return self._active_native_session_accessor

    @property
    def _refresh_console_effective_scope_and_sync(self) -> Any:
        return self._refresh_effective_scope_and_sync_fn

    @property
    def _console_messages_from_conversation_tree(self) -> Any:
        return self._messages_from_conversation_tree_accessor

    @property
    def _console_session_settings_for_resume(self) -> Any:
        return self._session_settings_for_resume_accessor

    @property
    def _resolve_resumed_character_name(self) -> Any:
        return self._resolve_resumed_character_name_fn

    @property
    def _inject_resume_agent_markers(self) -> Any:
        return self._inject_resume_agent_markers_accessor

    @property
    def _resolve_console_effective_scope_state(self) -> Any:
        return self._resolve_effective_scope_state_fn

    @property
    def _sync_console_retrieval_scope_row(self) -> Any:
        return self._sync_retrieval_scope_row_fn

    @property
    def _note_console_follow_intent(self) -> Any:
        return self._note_follow_intent_fn

    @property
    def _focus_console_composer_if_needed(self) -> Any:
        return self._focus_composer_if_needed_fn

    @property
    def _console_conversation_section_config(self) -> Any:
        return self._conversation_section_config_accessor

    @property
    def _console_conversation_browser_config(self) -> Any:
        """The injected `conversation_browser_config`. The config TREE stays
        on `ChatScreen` (it holds rail-state and search preferences this
        cluster does not own); only the grouped-browser collapse write below
        moved here, since nothing on the screen calls it. See `__init__`'s
        docstring."""
        return self._conversation_browser_config_fn

    def _set_console_conversation_browser_group_collapsed(
        self,
        group_id: str,
        collapsed: bool,
    ) -> None:
        """Store one grouped browser collapse preference.

        Args:
            group_id: The browser group whose state is being recorded; blank
                or whitespace-only ids are ignored.
            collapsed: True to record the group as collapsed.
        """
        normalized_group_id = str(group_id or "").strip()
        if not normalized_group_id:
            return
        browser_config = self._console_conversation_browser_config()
        collapsed_groups = browser_config.get("collapsed_groups")
        if not isinstance(collapsed_groups, dict):
            collapsed_groups = {}
            browser_config["collapsed_groups"] = collapsed_groups
        collapsed_groups[normalized_group_id] = bool(collapsed)

    @property
    def _focus_console_workspace_conversation_search(self) -> Any:
        """Stays on `ChatScreen` (DOM: `query_one`). See module docstring."""
        return self._focus_conversation_search_fn

    @property
    def _sync_console_workspace_context(self) -> Any:
        """Stays on `ChatScreen` (DOM: `query_one`/`query`). See module
        docstring."""
        return self._sync_workspace_context_fn

    @property
    def _schedule_console_browser_timer(self) -> Any:
        return self._schedule_timer_fn

    @property
    def _console_chat_controller(self) -> Any:
        return self._current_chat_controller_accessor()

    @property
    def _console_fleet_unseen_ids(self) -> Any:
        return self._fleet_unseen_ids_accessor

    @property
    def _console_run_marker_with_unseen(self) -> Any:
        return self._run_marker_with_unseen_fn

    @property
    def _console_broken_conversation_ids(self) -> set[str]:
        return self._broken_conversation_ids_accessor()

    @property
    def _ensure_console_agent_bridge(self) -> Any:
        return self._ensure_agent_bridge_fn

    @property
    def _console_subagent_counts_for_rows(self) -> Any:
        return self._subagent_counts_for_rows_fn

    @property
    def _console_conversation_browser_collapse_preferences(self) -> Any:
        return self._conversation_browser_collapse_preferences_fn

    def transition_browser_search(self, query: str, disabled: bool) -> None:
        """Apply one input change and schedule its canonical browser refresh."""
        if disabled or query == self._console_conversation_browser_query:
            return
        self._console_conversation_browser_query = query
        self._console_conversation_browser_search_token += 1
        token = self._console_conversation_browser_search_token
        if self._console_conversation_browser_search_timer is not None:
            self._console_conversation_browser_search_timer.stop()
            self._console_conversation_browser_search_timer = None
        self._console_conversation_browser_rows = (
            self._filter_console_browser_rows_for_query(
                self._console_conversation_browser_rows,
                query,
            )
        )
        self._console_conversation_browser_search_timer = (
            self._schedule_console_browser_timer(
                0.2,
                partial(
                    self._start_console_conversation_browser_search,
                    query,
                    token,
                ),
            )
        )

    def clear_console_conversation_browser_search(self) -> None:
        """Clear canonical browser search state and restore search focus."""
        if self._console_conversation_browser_search_timer is not None:
            self._console_conversation_browser_search_timer.stop()
            self._console_conversation_browser_search_timer = None
        self._console_conversation_browser_search_token += 1
        self._console_conversation_browser_query = ""
        self._console_conversation_browser_rows = ()
        self._console_conversation_browser_total = None
        self._console_conversation_browser_error = ""
        self._sync_console_workspace_context()
        self.call_after_refresh(self._focus_console_workspace_conversation_search)

    def _start_console_conversation_browser_search(
        self,
        query: str,
        token: int,
    ) -> None:
        """Run the debounced half of a rail conversation-search keystroke."""
        if token != self._console_conversation_browser_search_token:
            return
        if query != self._console_conversation_browser_query:
            return
        self._invalidate_console_persisted_rows_cache()
        if not query.strip():
            self._console_conversation_browser_rows = ()
            self._console_conversation_browser_total = None
            self._console_conversation_browser_error = ""
            self._sync_console_workspace_context()
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
            return
        self._console_conversation_browser_rows = (
            self._filter_console_browser_rows_for_query(
                self._merge_console_browser_rows(
                    self._native_console_browser_rows(),
                    self._membership_console_browser_rows(),
                ),
                query,
            )
        )
        self._console_conversation_browser_total = None
        self._console_conversation_browser_error = ""
        self._sync_console_workspace_context()
        self.run_worker(
            self._refresh_console_conversation_browser_search(query, token),
            group="console-workspace-conversation-search",
            exclusive=True,
        )

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
                ConsoleWorkspaceController._console_browser_row_scope_copy(row),
            )
        ).lower()
        return normalized_query in haystack

    def _filter_console_browser_rows_for_query(
        self,
        rows: Iterable[ConsoleConversationBrowserInputRow],
        query: str,
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        """Return rows matching ``query`` without service or database access."""
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

    @staticmethod
    def _console_browser_display_identity(
        row: ConsoleConversationBrowserInputRow,
    ) -> tuple[str, str, str, str] | tuple[str, str]:
        """Return the display identity used to dedupe grouped browser rows."""
        conversation_id = str(row.conversation_id or "").strip()
        if conversation_id:
            scope_type = str(row.scope_type or "").strip() or "workspace"
            workspace_id = (
                "" if scope_type == "global" else str(row.workspace_id or "").strip()
            )
            return ("conversation", scope_type, workspace_id, conversation_id)
        return (
            "row",
            ConsoleWorkspaceController._console_browser_row_key(row),
        )

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
        ids = (
            starred_ids
            if starred_ids is not None
            else self._starred_console_conversation_ids()
        )
        star_enabled = bool(conversation_id) and not str(row.row_key or "").startswith(
            "native:"
        )
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
        controller = self._console_chat_controller
        rows: list[ConsoleConversationBrowserInputRow] = []
        for session in store.sessions():
            session_workspace_id = str(session.workspace_id or "").strip()
            scope_type = (
                "global"
                if session_workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
                else "workspace"
            )
            workspace_id = None if scope_type == "global" else session_workspace_id
            persisted_id = (
                str(session.persisted_conversation_id).strip()
                if session.persisted_conversation_id
                else ""
            )
            row_key = persisted_id or f"native:{session.id}"
            selected = session.id == active_session_id
            run_marker = (
                resolve_glyph(
                    CONSOLE_RUN_MARKER_GLYPHS.get(
                        self._console_run_marker_with_unseen(
                            controller, session, self._console_fleet_unseen_ids()
                        ),
                        "",
                    )
                )
                if controller is not None
                else ""
            )
            queued_count = (
                controller.activity_for(session.id).queued_count
                if controller is not None
                else 0
            )
            row = ConsoleConversationBrowserInputRow(
                row_key=row_key,
                conversation_id=persisted_id or None,
                native_session_id=session.id,
                title=str(session.title or "Untitled conversation"),
                scope_type=scope_type,
                workspace_id=workspace_id,
                workspace_label=self._console_browser_workspace_label(
                    workspace_id, labels
                ),
                status="active session" if selected else "open session",
                selected=selected,
                source_kind="native",
                updated_sort=str(session.updated_at or ""),
                run_marker=run_marker,
                queued_count=queued_count,
            )
            rows.append(self._apply_console_browser_star_state(row, starred_ids))
        return rows

    def _console_browser_unseen_marker(self, conversation_id: str | None) -> str:
        """Return the unseen glyph for a marked sessionless conversation."""
        conversation_key = str(conversation_id or "").strip()
        if not conversation_key:
            return ""
        if conversation_key not in self._console_fleet_unseen_ids():
            return ""
        return resolve_glyph(
            CONSOLE_RUN_MARKER_GLYPHS.get(ConsoleRunMarker.SUBAGENT_UNSEEN, "")
        )

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
        current_conversation = (
            current_conversation_id or self._current_console_conversation_id()
        )
        active_session = self._active_native_console_session()
        active_workspace_id = (
            str(active_session.workspace_id or "").strip()
            if active_session is not None
            else str(
                self._current_console_workspace_context().active_workspace_id or ""
            ).strip()
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
                    workspace_label=self._console_browser_workspace_label(
                        workspace_id, labels
                    ),
                    status=str(getattr(membership, "role", "") or "workspace-thread"),
                    selected=bool(
                        current_conversation
                        and current_conversation == conversation_id
                        and active_workspace_id == workspace_id
                    ),
                    source_kind="membership",
                    updated_sort=str(getattr(membership, "created_at", "") or ""),
                    run_marker=self._console_browser_unseen_marker(conversation_id),
                    openable=(
                        conversation_id not in self._console_broken_conversation_ids
                    ),
                )
                rows.append(self._apply_console_browser_star_state(row, starred_ids))
        return rows

    async def _persisted_console_browser_rows(
        self,
        query: str = "",
        current_conversation_id: str | None = None,
    ) -> tuple[list[ConsoleConversationBrowserInputRow], int | None, str]:
        """Return persisted global/workspace rows for grouped browser search."""
        services: list[tuple[Any, bool]] = []
        scope_service = getattr(
            self.app_instance,
            "chat_conversation_scope_service",
            None,
        )
        local_service = getattr(
            self.app_instance, "local_chat_conversation_service", None
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
            saw_result = False
            current_conversation = (
                current_conversation_id or self._current_console_conversation_id()
            )
            starred_ids = self._starred_console_conversation_ids()
            for scope_type, workspace_id in scopes:
                list_kwargs: dict[str, Any] = {
                    "query": query,
                    "scope_type": scope_type,
                    "workspace_id": workspace_id,
                    "limit": CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
                    "offset": 0,
                }
                if include_mode:
                    list_kwargs["mode"] = "local"
                try:
                    if include_mode and not inspect.iscoroutinefunction(
                        list_conversations
                    ):
                        result = await asyncio.to_thread(
                            list_conversations, **list_kwargs
                        )
                    elif include_mode:
                        result = list_conversations(**list_kwargs)
                    else:
                        db = getattr(service, "db", None)
                        if bool(getattr(db, "is_memory_db", False)):
                            result = list_conversations(**list_kwargs)
                        else:
                            result = await asyncio.to_thread(
                                list_conversations, **list_kwargs
                            )
                    if inspect.isawaitable(result):
                        result = await result
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
                        "Unable to search Console conversation browser "
                        "query={!r} scope_type={} workspace_id={} include_mode={}",
                        query,
                        scope_type,
                        workspace_id,
                        include_mode,
                    )
                    return (
                        rows,
                        None if not saw_total else total_count,
                        "Workspace conversation search is unavailable.",
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
                        updated_sort=console_persisted_row_updated_sort(item),
                        run_marker=self._console_browser_unseen_marker(conversation_id),
                    )
                    rows.append(
                        self._apply_console_browser_star_state(row, starred_ids)
                    )
            if saw_result:
                return rows, total_count if saw_total else None, last_error
        return [], None, last_error

    def _invalidate_console_persisted_rows_cache(self) -> None:
        """Clear the TTL-cached persisted conversation-browser rows."""
        self._console_persisted_rows_cache = None
        self._console_persisted_rows_cache_key = None
        self._console_persisted_rows_cache_at = 0.0
        self._console_persisted_rows_cache_token += 1
        self._console_persisted_rows_refresh_key = None

    def _sync_persisted_console_browser_rows(
        self,
        query: str = "",
        current_conversation_id: str | None = None,
    ) -> tuple[list[ConsoleConversationBrowserInputRow], int | None, str]:
        """Return cached persisted rows without service or database access."""
        result = self._compute_persisted_console_browser_rows(
            query,
            current_conversation_id,
        )
        if result is not None:
            return result
        refresh_key = (
            query,
            current_conversation_id,
            self._console_persisted_rows_cache_token,
        )
        if self._console_persisted_rows_refresh_key != refresh_key:
            # task-15791: this sync derivation is called from state BUILDERS,
            # which the suite's bare-screen convention runs on an unmounted
            # ChatScreen (no active app) -- and `run_worker` on such a node
            # raises NoActiveAppError (520b1ec12 introduced the spawn here).
            # The refresh is best-effort by design ("without service or
            # database access"), so skip scheduling when the pump is not
            # running; the mounted app always is, and the guard also avoids
            # minting a coroutine that can never be awaited.
            if self._screen_running_accessor():
                self._console_persisted_rows_refresh_key = refresh_key
                self.run_worker(
                    self._refresh_console_persisted_rows_cache(
                        query,
                        current_conversation_id=current_conversation_id,
                        refresh_key=refresh_key,
                    ),
                    group="console-persisted-browser-cache",
                    exclusive=True,
                )
        return [], None, ""

    async def _refresh_console_persisted_rows_cache(
        self,
        query: str = "",
        current_conversation_id: str | None = None,
        *,
        refresh_key: tuple[str, str | None, int] | None = None,
    ) -> tuple[list[ConsoleConversationBrowserInputRow], int | None, str]:
        """Refresh persisted rows asynchronously when the TTL cache is stale."""
        cached = self._compute_persisted_console_browser_rows(
            query,
            current_conversation_id,
        )
        if cached is not None:
            if (
                refresh_key is not None
                and self._console_persisted_rows_refresh_key == refresh_key
            ):
                self._console_persisted_rows_refresh_key = None
                self._sync_console_workspace_context()
            return cached
        try:
            result = await self._persisted_console_browser_rows(
                query,
                current_conversation_id=current_conversation_id,
            )
        except asyncio.CancelledError:
            if self._console_persisted_rows_refresh_key == refresh_key:
                self._console_persisted_rows_refresh_key = None
            raise
        if (
            refresh_key is not None
            and self._console_persisted_rows_refresh_key != refresh_key
        ):
            return result
        self._console_persisted_rows_cache = result
        self._console_persisted_rows_cache_key = (query, current_conversation_id)
        self._console_persisted_rows_cache_at = time.monotonic()
        if refresh_key is not None:
            self._console_persisted_rows_refresh_key = None
            self._sync_console_workspace_context()
        return result

    def _compute_persisted_console_browser_rows(
        self,
        query: str = "",
        current_conversation_id: str | None = None,
    ) -> tuple[list[ConsoleConversationBrowserInputRow], int | None, str] | None:
        """Return a fresh matching cache entry without performing I/O."""
        cache_key = (query, current_conversation_id)
        if (
            self._console_persisted_rows_cache is not None
            and self._console_persisted_rows_cache_key == cache_key
            and (time.monotonic() - self._console_persisted_rows_cache_at)
            < CONSOLE_PERSISTED_ROWS_CACHE_TTL_SECONDS
        ):
            return self._console_persisted_rows_cache
        return None

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
        persisted_rows, persisted_total, sync_error = (
            self._sync_persisted_console_browser_rows(
                query,
                current_conversation_id=current_conversation_id,
            )
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

        (
            persisted_rows,
            persisted_total,
            error_copy,
        ) = await self._persisted_console_browser_rows(query)
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
        """Refresh grouped browser rows after selection or star changes."""
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
        token = self._console_conversation_browser_search_token
        await self._refresh_console_conversation_browser_search(query, token)

    def _with_console_conversation_browser_state(
        self,
        state: ConsoleWorkspaceContextState,
        current_conversation_id: str | None = None,
    ) -> ConsoleWorkspaceContextState:
        """Attach grouped all-workspaces conversation browser state."""
        legacy_state = self._with_console_workspace_conversation_section(state)
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
            active_workspace_id=(
                self._current_console_workspace_context().active_workspace_id
            ),
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
        return replace(
            state,
            conversation_browser=browser,
            conversation_section=legacy_state.conversation_section,
        )

    # -- Workspace policy context -------------------------------------------

    def _current_console_workspace_context(self) -> ConsoleWorkspaceContext:
        """Return explicit workspace policy context for native Console sends."""
        workspace_id = CONSOLE_GLOBAL_WORKSPACE_ID
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
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
                logger.debug(
                    "Console workspace registry was unavailable for send context"
                )

        staged_sources: list[ConsoleStagedSource] = []
        pending_launch = self._pending_console_launch_context
        if pending_launch is not None:
            payload = pending_launch.payload
            source_workspace = payload.get("workspace_id")
            launch_workspace_id = str(source_workspace) if source_workspace else None
            # RAG UX v2 PR-4: this builder is the SINGLE seam feeding both
            # the Inspector rail badge ("{n} staged") and the settings
            # context estimate, and it used to append exactly ONE staged
            # source per launch -- so the status chip could read "Sources: 4
            # staged" while its two siblings read 1. One row per bundle
            # reference puts all three in the same vocabulary (and gives the
            # workspace policy check a real per-source id to gate on).
            bundle = evidence_bundle_from_launch(pending_launch)
            references = bundle.references if bundle is not None else ()
            for reference in references:
                chunk_id = reference.metadata.get("chunk_id")
                # Two chunks of one document share a source_id; qualify with
                # the chunk id (exactly as the capture adapter does) so
                # `allowed_sources`' source_id dedupe cannot collapse them.
                staged_sources.append(
                    ConsoleStagedSource(
                        source_id=str(
                            chunk_id
                            if isinstance(chunk_id, str) and chunk_id
                            else reference.source_id
                        ),
                        label=reference.title,
                        source_type=reference.source_type,
                        workspace_id=reference.workspace_id or launch_workspace_id,
                    )
                )
            if not staged_sources:
                # A launch with no (or an empty) evidence bundle is still one
                # staged item -- the same fallback `console_staged_source_count`
                # and the strip use, so all three surfaces still agree.
                source_id = (
                    payload.get("source_id")
                    or payload.get("target_id")
                    or payload.get("run_id")
                    or pending_launch.title
                )
                staged_sources.append(
                    ConsoleStagedSource(
                        source_id=str(source_id),
                        label=pending_launch.title,
                        source_type=str(pending_launch.source),
                        workspace_id=launch_workspace_id,
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
                logger.opt(exception=True).debug(
                    "Unable to read active workspace for conversation search"
                )
                workspace = None
            workspace_id = str(getattr(workspace, "workspace_id", "") or "").strip()
            if workspace_id:
                return workspace_id
        store = self._console_chat_store
        if store is not None and store.workspace_context.active_workspace_id:
            return str(store.workspace_context.active_workspace_id)
        return ""

    # -- Workspace switcher / rename / archive / create ---------------------

    def _open_console_workspace_switcher(self) -> None:
        """Open the active Console workspace switcher."""
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            self.app_instance.notify(
                "Workspace service is not ready.", severity="warning"
            )
            return
        try:
            workspaces = tuple(registry_service.list_workspaces())
            active_workspace = registry_service.get_active_workspace()
        except Exception:
            logger.opt(exception=True).warning(
                "Unable to open Console workspace switcher"
            )
            self.app_instance.notify(
                "Workspace registry could not be read.",
                severity="error",
            )
            return
        if not workspaces:
            self.app_instance.notify(
                "Create one with the rail's New button or in Settings > Workspaces.",
                severity="warning",
            )
            return

        active_workspace_id = (
            active_workspace.workspace_id if active_workspace is not None else None
        )

        def _switch_to(workspace_id: str) -> None:
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
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )

        def _apply_workspace_switch(
            result: tuple[str, str] | None,
        ) -> None:
            if not result:
                return
            action, workspace_id = result
            if action == "switch":
                _switch_to(workspace_id)
            elif action == "rename":
                self._open_console_workspace_rename(workspace_id)
            elif action == "archive":
                self._confirm_console_workspace_archive(workspace_id)

        self.push_screen(
            ConsoleWorkspaceSwitcherModal(
                workspaces=workspaces,
                active_workspace_id=active_workspace_id,
            ),
            callback=_apply_workspace_switch,
        )

    def _open_console_workspace_rename(self, workspace_id: str) -> None:
        """Prompt for and apply a new workspace name (TASK-714)."""
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            self.app_instance.notify(
                "Workspace service is not ready.", severity="warning"
            )
            return
        record = registry_service.get_workspace(workspace_id)
        if record is None:
            self.app_instance.notify(
                "Workspace is no longer available.", severity="warning"
            )
            return

        def _apply_rename(new_name: str | None) -> None:
            if not new_name:
                return
            try:
                renamed = registry_service.rename_workspace(workspace_id, new_name)
            except WorkspaceRegistryServiceError as exc:
                self.app_instance.notify(str(exc), severity="warning")
                return
            except Exception:
                logger.opt(exception=True).warning("Unable to rename Console workspace")
                self.app_instance.notify(
                    "Workspace could not be renamed.", severity="error"
                )
                return
            self._sync_console_chat_core_state()
            self._sync_console_workspace_context()
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            self.app_instance.notify(
                f"Renamed workspace to {renamed.name}.", severity="information"
            )

        self.push_screen(
            ConsoleWorkspaceRenameModal(current_name=record.name),
            callback=_apply_rename,
        )

    def _confirm_console_workspace_archive(self, workspace_id: str) -> None:
        """Confirm and archive a workspace (TASK-714)."""
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            self.app_instance.notify(
                "Workspace service is not ready.", severity="warning"
            )
            return
        record = registry_service.get_workspace(workspace_id)
        if record is None:
            self.app_instance.notify(
                "Workspace is no longer available.", severity="warning"
            )
            return
        was_active = bool(record.active)

        # ConfirmationDialog awaits its confirm callback, so this must be a
        # coroutine function.
        async def _archive() -> None:
            try:
                registry_service.archive_workspace(workspace_id)
            except WorkspaceRegistryServiceError as exc:
                self.app_instance.notify(str(exc), severity="warning")
                return
            except Exception:
                logger.opt(exception=True).warning(
                    "Unable to archive Console workspace"
                )
                self.app_instance.notify(
                    "Workspace could not be archived.", severity="error"
                )
                return
            self._sync_console_chat_core_state()
            if was_active:
                self._activate_console_session_for_workspace(DEFAULT_WORKSPACE_ID)
            self._sync_console_workspace_context()
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
            suffix = " Console switched to the Default workspace." if was_active else ""
            self.app_instance.notify(
                f"Archived {record.name}. Its conversations stay saved in "
                f"Library.{suffix}",
                severity="information",
            )

        self.push_screen(
            ConfirmationDialog(
                title="Archive workspace?",
                message=(
                    f"Archive {record.name}? Its conversations stay saved and "
                    "remain visible in Library; the workspace disappears from "
                    "the switcher and the Console browser."
                ),
                confirm_label="Archive",
                confirm_callback=_archive,
            )
        )

    def _create_console_workspace(self) -> None:
        """Create a new local workspace and activate it."""
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            self.app_instance.notify(
                "Workspace service is not ready.", severity="warning"
            )
            return
        try:
            workspace_id, workspace_name = next_local_workspace_identity(
                registry_service
            )
            registry_service.create_workspace(
                workspace_id=workspace_id,
                name=workspace_name,
                description="Local workspace created from Console.",
            )
            registry_service.set_active_workspace(workspace_id)
        except WorkspaceRegistryServiceError:
            logger.opt(exception=True).warning("Unable to create Console workspace")
            self.app_instance.notify(
                "Workspace could not be created.", severity="error"
            )
            return
        except Exception:
            logger.opt(exception=True).warning(
                "Unexpected error creating Console workspace"
            )
            self.app_instance.notify(
                "Workspace could not be created.", severity="error"
            )
            return
        self._sync_console_chat_core_state()
        self._activate_console_session_for_workspace(workspace_id)
        self._sync_console_workspace_context()
        self.run_worker(
            self._sync_native_console_chat_ui(), exclusive=True, group="console-sync"
        )
        # TASK-713: creation also activates the workspace and opens a tab;
        # without a notification the whole sequence is invisible when the
        # Workspace status row is scrolled out of view.
        self.app_instance.notify(
            f"Created {workspace_name} and switched Console to it.",
            severity="information",
        )

    # -- Workspace RAG-scope picker ------------------------------------------

    async def _open_console_workspace_scope_picker(self) -> None:
        """Open the RAG retrieval-scope picker for the ACTIVE workspace.

        Task-13 workspace entry point (design spec section 4, "Workspace
        entry: Scope button beside the workspace row in the Session area").
        ``universe=None`` -- the workspace picker offers the full library,
        unlike the conversation-target picker, which restricts to the
        workspace's own items once one is set (D3, see
        ``_open_console_retrieval_scope_picker``).

        Only a REAL registry workspace can be scoped -- gated the same way
        the mounted button itself is gated
        (``ConsoleWorkspaceContextState.rag_scope_enabled``): the built-in
        Default workspace has a real ``workspace_id`` row
        (``DEFAULT_WORKSPACE_ID``) once ``ensure_default_workspace`` has
        run, so it IS scopable; only the "Local Default"/error/no-registry
        sentinel states (no real workspace row at all) are refused here.
        """
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            self.app_instance.notify(
                "Workspace service is not ready.", severity="warning"
            )
            return
        try:
            active_workspace = registry_service.get_active_workspace()
        except Exception:
            logger.opt(exception=True).warning(
                "Unable to read active workspace for the scope picker"
            )
            self.app_instance.notify(
                "Workspace registry could not be read.", severity="error"
            )
            return
        if active_workspace is None:
            self.app_instance.notify(
                "Create or select a workspace before setting a RAG scope.",
                severity="warning",
            )
            return
        workspace_id = active_workspace.workspace_id

        try:
            initial = await self._read_console_workspace_scope(
                registry_service, workspace_id
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Unable to read workspace scope for {}", workspace_id
            )
            initial = None

        target_label = f"workspace '{active_workspace.name}'"
        media_lister, notes_lister, tag_lister = self._console_scope_picker_listers()

        def _on_save(scope: Optional[RagScope]) -> None:
            self.run_worker(
                self._apply_console_workspace_scope_save(workspace_id, scope),
                exclusive=True,
                group="console-workspace-scope-save",
            )

        self.push_screen(
            ConsoleScopePickerModal(
                target_label,
                None,
                initial,
                _on_save,
                media_lister=media_lister,
                notes_lister=notes_lister,
                tag_lister=tag_lister,
            )
        )

    async def _apply_console_workspace_scope_save(
        self,
        workspace_id: str,
        scope: Optional[RagScope],
    ) -> None:
        """Persist (or clear) a workspace's RAG retrieval scope (task-13).

        ``WorkspaceNotFound`` is caught deliberately -- the workspace may
        have been archived/deleted (e.g. from Library > Workspaces) between
        opening the picker and saving. Refreshes the ACTIVE Console
        session's effective-scope display afterward: a workspace-scope
        change can widen or narrow retrieval for every conversation linked
        to it, but only the currently active session's row/chip are
        mounted to refresh.

        Args:
            workspace_id: The workspace whose scope was just chosen.
            scope: The new scope, or ``None`` to clear it.
        """
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            self.app_instance.notify(
                "Couldn't save scope: workspace service is not ready.",
                severity="error",
            )
            return
        try:
            await self._write_console_workspace_scope(
                registry_service, workspace_id, scope
            )
        except WorkspaceNotFound:
            logger.opt(exception=True).warning(
                "Workspace scope save target missing: {}", workspace_id
            )
            self.app_instance.notify(
                "Couldn't save scope: this workspace no longer exists.",
                severity="error",
            )
            return
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to write workspace scope for {}", workspace_id
            )
            self.app_instance.notify("Couldn't save workspace scope.", severity="error")
            return
        session = self._active_native_console_session()
        if session is not None:
            await self._refresh_console_effective_scope_and_sync(session)

    # -- Workspace-scoped Console session activation -------------------------

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
                self._capture_console_draft_switch_snapshot()
                store.switch_session(session.id)
                # task-7 review: this switches the active session with no
                # other chip-refresh call anywhere in the caller chain (all
                # three callers only run `_sync_native_console_chat_ui()`
                # afterward, which never touches the temporary chip -- see
                # `_sync_console_temporary_chip`). Without this the chip
                # could keep reading "Temporary" on a workspace's saved
                # session, or stay hidden on one that is actually temporary.
                self._sync_console_temporary_chip()
                return
        self._capture_console_draft_switch_snapshot()
        store.create_session(
            title=self._console_workspace_session_title(target_workspace_id),
            workspace_id=target_workspace_id,
            settings=inherited_settings or self._default_console_session_settings(),
        )
        # task-7 review: `create_session` activates the new (never
        # ephemeral -- no `ephemeral=` passed) session inline; same
        # staleness risk as the switch branch above if the workspace's
        # previous session was temporary.
        self._sync_console_temporary_chip()

    def _console_workspace_session_title(self, workspace_id: str) -> str:
        """Return a readable title for an auto-created workspace Console tab."""
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        workspace_name = str(workspace_id).strip()
        if registry_service is not None:
            try:
                workspace = registry_service.get_workspace(workspace_id)
                if workspace is not None:
                    workspace_name = workspace.name
            except Exception:
                logger.opt(exception=True).debug(
                    "Unable to read Console workspace title"
                )
        if not workspace_name:
            workspace_name = "Workspace"
        return f"{workspace_name} Chat"

    def _console_initial_session_title_for_workspace(
        self, workspace_id: str | None
    ) -> str:
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
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            return
        try:
            active_workspace = registry_service.get_active_workspace()
            if not workspace_id or workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID:
                # task-15120 (owner ruling): the workspace context follows the
                # conversation -- a global conversation's context IS the
                # global scope, on both layers. This used to early-return,
                # leaving the registry on the previous workspace while the
                # store's context flipped to "global": two sources of truth
                # disagreeing, and the previous workspace's capabilities
                # bleeding into a global conversation. The registry's stable
                # representation of "no explicit workspace" is the built-in
                # Default (`ensure_default_workspace` floors every context
                # read to it, deliberately -- capability-less, safe), so a
                # global conversation lands there, not on bare None.
                if (
                    active_workspace is not None
                    and active_workspace.workspace_id != DEFAULT_WORKSPACE_ID
                ):
                    registry_service.clear_active_workspace()
                    ensure_default = getattr(
                        registry_service, "ensure_default_workspace", None
                    )
                    if callable(ensure_default):
                        ensure_default()
                return
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

    # -- Workspace-scope persistence primitives ------------------------------

    @staticmethod
    async def _read_console_workspace_scope(
        registry_service: Any, workspace_id: str
    ) -> Optional[RagScope]:
        """Read a workspace's stored scope, off-loop for a file-backed registry.

        Mirrors ``_read_console_retrieval_scope``'s in-memory-DB guard
        (task-13): ``LocalWorkspaceRegistryService.db`` is never actually
        ``:memory:``-backed in production (``WorkspaceDB`` (task-3011) holds
        one connection per THREAD rather than sharing a single connection
        for ``:memory:``, so each new thread touching it would open its own
        empty, table-less ``:memory:`` database -- broken beyond a single
        thread), but the guard is applied anyway for the same defensive
        discipline the conversation-scope read uses.
        """
        db = getattr(registry_service, "db", None)
        if getattr(db, "is_memory_db", False):
            return registry_service.get_workspace_scope(workspace_id)
        return await asyncio.to_thread(
            registry_service.get_workspace_scope, workspace_id
        )

    @staticmethod
    async def _write_console_workspace_scope(
        registry_service: Any, workspace_id: str, scope: Optional[RagScope]
    ) -> None:
        """Write (or clear) a workspace's stored scope; see the read twin's
        in-memory-registry-DB guard docstring for why this isn't
        unconditionally ``asyncio.to_thread``."""
        db = getattr(registry_service, "db", None)
        if getattr(db, "is_memory_db", False):
            registry_service.set_workspace_scope(workspace_id, scope)
        else:
            await asyncio.to_thread(
                registry_service.set_workspace_scope, workspace_id, scope
            )

    # -- Resuming a persisted conversation ------------------------------------

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

    async def _resume_console_workspace_conversation(
        self,
        conversation_id: str,
        *,
        target_scope_type: str | None = None,
        target_workspace_id: str | None = None,
    ) -> bool | None:
        """Load a persisted saved conversation into a native Console session.

        Returns:
            True on success; None on a transient failure this method already
            notified about (service unavailable / load error); False when the
            conversation record is missing - the caller owns that failure's
            feedback (TASK-717).
        """
        target = str(conversation_id or "").strip()
        if not target:
            return None
        # TASK-339: keystrokes typed while the conversation tree loads
        # belong to the resumed session — snapshot the composer now.
        self._capture_console_draft_switch_snapshot()
        # task-15860 Task 6: the tree load and the session build moved to
        # `Chat/console_conversation_hydration.py` -- the launch wake has to
        # hydrate a conversation with no screen in existence, and one policy
        # beats two. Everything BELOW the hydration call is this screen's own
        # work (marker overlay, character identity, scope warm, repaint,
        # focus) and stays here; so do both failure toasts, because the UX
        # for each failure is a view concern.
        try:
            tree = await load_console_conversation_tree(self.app_instance, target)
        except ConversationServiceUnavailable:
            self.app_instance.notify(
                "Saved conversation resume is unavailable in this build.",
                severity="warning",
            )
            return None
        except ConversationLoadFailed:
            logger.exception(
                f"Unable to resume Console saved conversation: conversation_id={target}"
            )
            self.app_instance.notify(
                "Unable to load this saved conversation.",
                severity="error",
            )
            return None

        if tree is None:
            # TASK-717: missing record - the caller owns this failure's UX
            # (honest toast + marking the row visibly broken), so do not
            # stack a second notification here.
            return False

        conversation = tree.get("conversation")
        if not isinstance(conversation, dict):
            conversation = {}
        store = self._ensure_console_chat_store()
        session = hydrate_console_session(
            app=self.app_instance,
            store=store,
            conversation_id=target,
            tree=tree,
            settings=self._console_session_settings_for_resume(conversation),
            target_scope_type=target_scope_type,
            target_workspace_id=target_workspace_id,
        )
        # Re-derive display-only agent TOOL markers from AgentRunsDB and overlay
        # them onto the restored active-path VIEW (markers are never tree nodes;
        # the next tree mutation's recompute rebuilds the view from live nodes
        # and drops them, matching how live markers are ephemeral in Phase A).
        store.apply_resume_marker_overlay(
            session.id,
            self._inject_resume_agent_markers(
                store.messages_for_session(session.id), target
            ),
        )
        # Local presentation remains keyed only by the numeric local
        # projection. Opaque server identity never enters local card/avatar/
        # dictionary lookup paths.
        if (
            session.runtime_backend == "local"
            and session.assistant_kind == "character"
            and session.character_id is not None
        ):
            character_name = await self._resolve_resumed_character_name(
                session.character_id
            )
            if character_name:
                session.character_name = character_name
            # Always (re)set the label on a local character resume -- to the
            # resolved name, or clear it when unresolved. ``settings`` are
            # otherwise inherited from the currently active session, so
            # leaving an inherited ``character_label`` in place would make
            # a card-less resume show a *different* character's name.
            if session.settings is not None:
                session.settings = replace(
                    session.settings, character_label=character_name
                )
        elif session.settings is not None:
            session.settings = replace(session.settings, character_label="")
        self._set_active_workspace_for_console_session(session.id)
        # task-9/task-13: warm the EFFECTIVE (conversation ∩ workspace)
        # scope cache for this session now (off-loop) so the Inspector row
        # reflects reality immediately on resume, rather than defaulting to
        # "everything" until the user opens Edit or saves a change (the
        # picker's other two read triggers).
        try:
            await self._resolve_console_effective_scope_state(session)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to resolve retrieval scope for conversation {}", target
            )
        # task-10 review finding 2: warming the cache above is not enough
        # by itself -- neither `_sync_native_console_chat_ui()` below nor
        # its own `_sync_console_control_bar()` call ever touches the
        # retrieval-scope row or `ConsoleStatusChips.sync_scope_chip`
        # (`sync_scope_chip` is deliberately its own method, kept off the
        # general control-bar sync tick -- see its docstring). Without this
        # explicit call the MOUNTED row/chip stayed on whatever state they
        # last rendered until the user opened Edit/Narrow or saved a
        # change, even though the cache above already had the right
        # answer. This is the same helper (and the same one-state,
        # two-renderers push) the scope-picker save path already uses.
        self._sync_console_retrieval_scope_row()
        # Finding C: resuming a saved conversation switches the active
        # conversation just as much as a tab switch does -- clear any
        # sub-agent drill-in immediately rather than rely solely on the
        # rail render path's defensive re-check on the next sync.
        self._console_agent_drilldown_run_id = None
        self._note_console_follow_intent()
        self._sync_console_chat_core_state()
        await self._sync_native_console_chat_ui()
        # task-15864 AC#2: opening a conversation creates the session a
        # mount-claimed (or otherwise staged) wake has been waiting for --
        # session-open IS a retry trigger. Before this, a restart-staged
        # wake sat pending until an unrelated composer keystroke (live
        # scenario 5). The poke only schedules `_attempt_all`; every
        # delivery gate (kill switch, send gate, user-wins-ties) still
        # applies unchanged.
        if callable(self._wake_retry_poke_fn):
            self._wake_retry_poke_fn()
        self._focus_console_composer_if_needed(force=True)
        return True

    # -- Workspace context state / grouped conversation rows -----------------

    def _build_console_workspace_context_state(self) -> ConsoleWorkspaceContextState:
        current_conversation = self._current_console_conversation_id()
        state = build_console_workspace_state(
            registry_service=getattr(
                self.app_instance, "workspace_registry_service", None
            ),
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
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            return
        try:
            active_workspace = registry_service.get_active_workspace()
            if (
                active_workspace is None
                or active_workspace.workspace_id != workspace_id
            ):
                registry_service.set_active_workspace(workspace_id)
                # TASK-713: opening a row from another workspace's group
                # retargets the whole Console context; the Workspace status
                # row is usually scrolled out of view at that moment, so the
                # side effect needs an explicit announcement.
                switched = registry_service.get_active_workspace()
                switched_name = switched.name if switched is not None else workspace_id
                self.app_instance.notify(
                    f"Switched Console to {switched_name}.",
                    severity="information",
                )
            self._ensure_console_chat_store().set_workspace_context(
                self._current_console_workspace_context()
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Unable to activate Console workspace for browser row",
            )

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
                logger.opt(exception=True).debug(
                    "Unable to ensure default workspace for Console browser"
                )
        list_workspaces = getattr(service, "list_workspaces", None)
        if not callable(list_workspaces):
            return ()
        try:
            return tuple(list_workspaces())
        except Exception:
            logger.opt(exception=True).debug(
                "Unable to list Console browser workspaces"
            )
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
        workspace_labels = (
            labels if labels is not None else self._console_browser_workspace_labels()
        )
        return workspace_labels.get(workspace_id, workspace_id)

    def _selected_console_workspace_conversation_summary(
        self,
        rows: list[ConsoleWorkspaceConversationRow],
    ) -> str:
        selected = next((row for row in rows if row.selected), None)
        if selected is None:
            return "No active conversation."
        title = ConsoleWorkspaceContextTray._conversation_title(selected.title)
        detail = ConsoleWorkspaceContextTray._conversation_detail_status(
            selected.status
        )
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
            logger.opt(exception=True).debug(
                "Unable to search workspace conversation memberships"
            )
            return []
        rows: list[ConsoleWorkspaceConversationRow] = []
        current_conversation = self._current_console_conversation_id()
        for membership in memberships:
            title = str(
                getattr(membership, "title", "") or getattr(membership, "item_id", "")
            )
            if needle and needle not in title.lower():
                continue
            conversation_id = str(getattr(membership, "item_id", "") or "")
            rows.append(
                ConsoleWorkspaceConversationRow(
                    conversation_id=conversation_id,
                    title=title,
                    status=str(getattr(membership, "role", "") or "workspace-thread"),
                    selected=bool(
                        current_conversation and conversation_id == current_conversation
                    ),
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
                logger.debug(
                    "Local persisted conversation search service is unavailable"
                )
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
            if self._console_workspace_conversation_search_timer is not None:
                self._console_workspace_conversation_search_timer.stop()
                self._console_workspace_conversation_search_timer = None
            self._invalidate_console_persisted_rows_cache()
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
            self._console_workspace_conversation_search_total if query.strip() else None
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

    # -- Conversation-browser press handling (wave-4 task 2) ----------------
    #
    # Three of `ChatScreen.on_button_pressed`'s 19 branches mutated nothing
    # but this cluster's state, so their bodies moved here whole and the
    # screen's branches became calls. Each takes the values the pressed
    # button carried rather than the `Button.Pressed` event: Textual's
    # event object stays on the screen (it is the screen that must
    # `event.stop()`), and a controller that never sees a widget cannot
    # start reading one.

    def _toggle_console_conversation_browser_section(self, group_id: str) -> None:
        """Flip one grouped-browser SECTION's collapse preference.

        Args:
            group_id: The pressed toggle's `group_id`, always
                ``"section:<section_id>"`` (see
                `ConsoleWorkspaceContextTray._compose_conversation_browser_
                section_header`). A section whose id no longer appears in
                current state collapses, matching the pre-move behaviour of
                treating a missing section as expanded.
        """
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

    def _toggle_console_conversation_browser_group(self, group_id: str) -> None:
        """Flip one grouped-browser GROUP's collapse preference.

        Deliberately not folded together with
        `_toggle_console_conversation_browser_section` above: the two
        search different levels of the same tree (sections by
        `section_id`, groups by `group_id` across every section's
        `groups`), and the pre-move branches were two separate bodies. A
        shared helper here would hide exactly the difference a reader
        comes to this pair to find.

        Args:
            group_id: The pressed toggle's `group_id` -- a workspace group
                key, never the ``"section:"``-prefixed form.
        """
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

    def _toggle_console_conversation_star(
        self,
        conversation_id: str,
        *,
        starred: bool,
        conversation_title: str,
    ) -> None:
        """Star or unstar one conversation and confirm the change.

        Moved verbatim out of `ChatScreen.on_button_pressed`'s
        `console-conversation-star-` branch (wave-4 task 2), the
        second-largest of its 19. The durable write goes through the app's
        `conversation_local_marks_service`; everything else here is the
        failure and confirmation copy that write needs to not be silent.

        Args:
            conversation_id: The pressed star's `conversation_id`. Blank
                for a native session that has never been persisted -- the
                tray disables those stars, so this guard only catches a
                stale row, and it explains rather than no-ops.
            starred: The pressed star's own `starred` attribute, used ONLY
                as a fallback when the marks service cannot answer
                `is_starred` -- current truth comes from the service, not
                from whatever the button was painted with.
            conversation_title: The pressed star's `conversation_title`,
                for the confirmation toast.
        """
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
        # task-15471: the resolve+toggle pair used to run right here, on the
        # event loop -- a read transaction plus a durable write transaction
        # per click, with the click frozen for the DB's whole busy_timeout
        # if any other writer held the file. It now runs on a worker; the
        # wrapper owns the failure/confirmation copy that followed it inline.
        self.run_worker(
            self._toggle_console_conversation_star_off_loop(
                marks_service,
                conversation_id,
                starred=starred,
                conversation_title=conversation_title,
            ),
            group="console-conversation-star",
            exit_on_error=False,
        )

    async def _toggle_console_conversation_star_off_loop(
        self,
        marks_service: Any,
        conversation_id: str,
        *,
        starred: bool,
        conversation_title: str,
    ) -> None:
        """Resolve current star truth, toggle it off the loop, and confirm.

        The deferred body of `_toggle_console_conversation_star`
        (task-15471). `_console_star_toggle_lock` serializes concurrent
        presses so each one resolves current truth from the service before
        toggling -- a rapid double-press still nets toggle-twice, exactly
        the pre-worker semantics, never two pool threads racing the same
        pre-state into a stale double-star.

        Args:
            marks_service: The app's `conversation_local_marks_service`,
                captured non-None by the dispatching guard.
            conversation_id: See `_toggle_console_conversation_star`.
            starred: Fallback only, as before -- used when the service
                cannot answer `is_starred`.
            conversation_title: For the confirmation toast.
        """
        progress = {"action": "resolve"}
        async with self._console_star_toggle_lock:
            try:

                def _resolve_and_toggle() -> str:
                    is_starred = getattr(marks_service, "is_starred", None)
                    currently_starred = (
                        bool(is_starred(conversation_id))
                        if callable(is_starred)
                        else bool(starred)
                    )
                    action = "unstar" if currently_starred else "star"
                    progress["action"] = action
                    if currently_starred:
                        marks_service.unstar_conversation(conversation_id)
                    else:
                        marks_service.star_conversation(conversation_id)
                    return action

                db = getattr(marks_service, "db", None)
                if bool(getattr(db, "is_memory_db", False)):
                    # A per-connection :memory: DB is only visible to the
                    # thread that migrated it -- same guard as the browser
                    # search threading (`chat_screen.py`, task-15455).
                    star_action = _resolve_and_toggle()
                else:
                    star_action = await asyncio.to_thread(_resolve_and_toggle)
            except asyncio.CancelledError:
                # Fix round (review minor 3): cancellation cannot stop the
                # pool thread, so the durable write may still land -- and
                # `CancelledError` is a `BaseException` that would sail past
                # the `except Exception` below, recreating exactly the
                # TASK-357 silent-toggle shape (write landed, no repaint).
                # Best-effort re-sync so the rail repaints from truth, then
                # let the cancellation propagate.
                try:
                    self._sync_console_workspace_context()
                except Exception:
                    logger.debug("Star-toggle cancellation re-sync failed")
                raise
            except Exception:
                logger.exception(
                    "Unable to update local conversation star "
                    "conversation_id={} action={}",
                    conversation_id,
                    progress["action"],
                )
                self.app_instance.notify(
                    "Unable to update local star.",
                    severity="warning",
                )
                return
        # TASK-357: confirm the toggle so a star/unstar is not a silent state
        # change (the review saw an accidental star go unnoticed).
        # `"".splitlines()` is `[]`, so indexing [0] raised IndexError on an
        # untitled conversation -- after the durable star write, which meant
        # the toggle landed but the user got no confirmation and the context
        # rail never re-synced (task-3024). The empty case was always
        # intended: `title_suffix` below already drops the quoted name when
        # `title` is falsy.
        title = next(iter(str(conversation_title or "").splitlines()), "").strip()
        # notify() interprets Rich markup, so escape the stored title before
        # interpolating it (a title like "[red]x[/red]" would otherwise inject
        # styling into the toast) — matches the escape_markup convention used
        # for the attachment toasts on the screen.
        title_suffix = f' "{escape_markup(title)}"' if title else ""
        if star_action == "star":
            self.app_instance.notify(f"Starred{title_suffix}.")
        elif star_action == "unstar":
            self.app_instance.notify(f"Unstarred{title_suffix}.")
        self._sync_console_workspace_context()

    # -- Misc toast helpers ----------------------------------------------------

    def _console_session_title_and_workspace_name(
        self, controller: "ConsoleChatController", session_id: str
    ) -> tuple[str, str]:
        """Return ``(session_title, workspace_name)`` for a fleet toast.

        Fix wave (rider 6, final review): shared by ``_park_console_
        approval`` and ``_notify_console_run_outcome``, which previously
        duplicated this exact lookup byte-for-byte. Falls back to the raw
        ``session_id``/workspace id when the session has already closed or
        the workspace can't be resolved -- a toast about a session that
        vanished microseconds ago must still say SOMETHING coherent rather
        than raise.
        """
        session_title = session_id
        workspace_id = CONSOLE_GLOBAL_WORKSPACE_ID
        for session in controller.store.sessions():
            if session.id == session_id:
                session_title = session.title
                workspace_id = session.workspace_id
                break
        return (
            sanitize_character_display_label(session_title, max_characters=500),
            sanitize_character_display_label(
                self._console_workspace_display_name(workspace_id),
                max_characters=500,
            ),
        )

    def _console_workspace_display_name(self, workspace_id: str) -> str:
        """Return ``workspace_id``'s display name via the registry, falling
        back to the raw id when the service is unavailable or the
        workspace can't be resolved (PA-T9 toast copy)."""
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is not None:
            try:
                workspace = registry_service.get_workspace(workspace_id)
                if workspace is not None and workspace.name:
                    return workspace.name
            except Exception:
                logger.opt(exception=True).debug(
                    "Unable to resolve Console workspace name for approval toast"
                )
        return str(workspace_id)
