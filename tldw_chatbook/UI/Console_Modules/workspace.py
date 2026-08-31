"""Workspace and conversation-browser ownership for the Console.

`ConsoleWorkspaceController` owns 57 non-DOM methods covering Workspace
policy and lifecycle, scope selection, persisted-conversation resume, the
named-workspace Tree projection, and the flat Default/unassigned browser.
Workspaces and Conversations own independent search attempts; page generations
are scoped per workspace. Legacy Workspace row and scalar names are
compatibility aliases over the flat lane.

The Textual screen retains only framework and DOM edges. Its bounded search
handler extracts plain query/disabled values and delegates the transition;
the Clear button delegates the complete clear transition. Browser services
and sibling-owned data enter through explicit late-bound constructor
callables so replacements remain observable and moved methods never query the
DOM or reach through sibling controllers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
    ConsoleGenerationSettingsHydration,
    hydrate_console_session,
    load_console_conversation_tree,
)
from ...Chat.console_session_settings import blank_console_session_settings
from ...Chat.rag_scope import RagScope
from ...config import save_setting_to_cli_config
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.glyph_fallback import resolve_glyph
from ...Widgets.Console import (
    ConsoleWorkspaceContextTray,
    ConsoleWorkspaceRenameModal,
    ConsoleWorkspaceSwitcherModal,
)
from ...Widgets.Console.console_scope_picker_modal import ConsoleScopePickerModal
from ...Widgets.Console.console_workspace_files_modal import (
    ConsoleWorkspaceFilesModal,
    WorkspaceFilesAttention,
    WorkspaceFilesBinding,
    WorkspaceFilesService,
)
from ...Workspaces.file_inspector import ScopeCaptureError, WorkspaceFileInspector
from ...Workspaces.models import RuntimeBindingKind, RuntimeBindingStatus
from ...Widgets.project_skills_import_modal import maybe_offer_project_skills_import
from ...Workspaces import (
    CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    ConsoleConversationBrowserInputRow,
    ConsoleConversationBrowserRow,
    DEFAULT_WORKSPACE_ID,
    WorkspaceRecord,
    WorkspaceTreeWorkspace,
    build_console_conversation_browser_state,
    build_workspace_tree_state,
    console_conversation_browser_group_row_limit,
    console_persisted_row_updated_sort,
    overlay_console_conversation_markers,
)
from ...Workspaces.display_state import (
    CONSOLE_WORKSPACE_CONVERSATION_RESULT_LIMIT,
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationRow,
    ConsoleWorkspaceConversationSectionState,
    build_console_workspace_state,
    console_workspace_conversation_result_copy,
)
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Workspaces.registry_service import (
    WorkspaceNotFound,
    WorkspaceRegistryServiceError,
)
from ..character_display_text import sanitize_character_display_label

if TYPE_CHECKING:
    from ...Chat.console_chat_controller import ConsoleChatController
    from ...Widgets.workspace_create_modal import WorkspaceCreateResult
    from ..Screens.chat_screen import ChatScreen

logger = logger.bind(module="ChatScreen")

CONSOLE_PERSISTED_ROWS_CACHE_TTL_SECONDS = 2.0
CONSOLE_SAVED_CONVERSATION_RESUME_FAILURE_COPY = (
    "Couldn't resume this saved conversation: it was deleted or couldn't be read.\n"
    "Your previous Console chat is still active."
)


def persist_console_workspace_tree_expansion_preferences(
    workspace_ids: list[str],
) -> None:
    """Write native Workspace Tree disclosure preferences off the UI loop."""

    try:
        save_setting_to_cli_config(
            "console.conversation_browser",
            "expanded_workspace_ids",
            list(workspace_ids),
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist Workspace Tree disclosure (exception_type={})",
            type(exc).__name__,
        )


#: Ceiling for a user-entered conversation title. Titles render in the
#: Context rail, the tab strip and the transcript header, so the limit is
#: about legibility in a narrow column, not storage.
_CONVERSATION_TITLE_MAX = 200


class UnknownMembership:
    __slots__ = ()


_MEMBERSHIP_UNKNOWN = UnknownMembership()

# TASK-25827: two unrelated failures used to share one sentence, so the rail
# could not tell "we could not check your access" from "the query failed" --
# and at the rail's width the shared sentence clipped to "Workspace
# conversations a...", which named neither. Keep these short enough to read
# in the rail and distinct enough to act on.
WORKSPACE_CONVERSATIONS_ACCESS_UNKNOWN = "Workspace access unknown."
WORKSPACE_CONVERSATIONS_LOAD_FAILED = "Couldn't load conversations."



@dataclass(slots=True)
class SearchAttemptState:
    """One projection's explicit debounce, request, result, and Retry state."""

    query: str = ""
    debounce: Any = None
    generation: int = 0
    request_key: tuple[str, str, int, object, object, int] | None = None
    rows: tuple[ConsoleConversationBrowserInputRow, ...] = ()
    total: int | None = None
    settled_rows: tuple[ConsoleConversationBrowserInputRow, ...] = ()
    settled_total: int | None = None
    settled_query: str = ""
    error: str = ""
    cache: dict[
        str, tuple[tuple[ConsoleConversationBrowserInputRow, ...], int | None]
    ] = field(default_factory=dict)
    retry_query: str | None = None
    worker: Any = None


@dataclass(slots=True)
class PageAttemptState:
    """One workspace's bounded page request and accumulated child state."""

    generation: int = 0
    loading: bool = False
    request_key: tuple[str, tuple[str, ...], int, int] | None = None
    membership_token: tuple[str, ...] | None = None
    owner_token: object = None
    lifecycle_token: object = None
    rows: tuple[ConsoleConversationBrowserInputRow, ...] = ()
    next_cursor: int | None = None
    error: str = ""
    retry_cursor: int | None = None
    membership_unknown: bool = False
    worker: Any = None


WORKSPACE_FILES_NO_FOLDERS_COPY = "No local folders are attached. Add one in Settings."
WORKSPACE_FILES_OTHER_VISIT_COPY = (
    "Close Workspace Files before inspecting another workspace."
)
WORKSPACE_FILES_MINIMUM_COPY = "Workspace Files needs at least 80 × 24 terminal cells."


@dataclass(frozen=True)
class _WorkspaceFilesResolution:
    """Off-loop registry snapshot used to construct one pinned visit."""

    workspace_id: str
    workspace_name: str
    active_workspace_id: str | None
    active_workspace_name: str
    bindings: tuple[WorkspaceFilesBinding, ...]
    had_bindings: bool


def _normalized_console_workspace_id(workspace_id: str | None) -> str:
    """Fold the "no explicit workspace" sentinels onto one identity.

    A Console session's default ``workspace_id`` (unset, or the explicit
    ``CONSOLE_GLOBAL_WORKSPACE_ID`` sentinel) and the registry's built-in
    Default workspace row (``DEFAULT_WORKSPACE_ID`` -- the resting state
    ``ensure_default_workspace`` establishes at boot and switch seams, and
    the id the read-only context resolution floors a missing active
    workspace to, TASK-21118) are THE SAME state on two layers (task-15120
    owner ruling, see ``_set_active_workspace_for_console_session``), not
    two different workspaces that happen to share a session. Any comparison between a
    session's ``workspace_id`` and a registry workspace id must normalize
    through this before comparing, or an aligned "global"/Default session
    reads as diverged purely because of which layer's spelling it carries.
    """
    normalized = str(workspace_id or "").strip()
    if not normalized or normalized == CONSOLE_GLOBAL_WORKSPACE_ID:
        return DEFAULT_WORKSPACE_ID
    return normalized


class _ConsoleRegistryDisplayReads:
    """Generation-keyed read-through view over the workspace registry.

    TASK-22201 (extending the TASK-21118 memo pattern): the Console run tick
    rebuilt its workspace context state up to six times per 0.2 s (see
    :class:`ConsoleTickWorkspaceBuilds`), and each build performed the
    registry's whole display read set as synchronous SQLite on the event
    loop -- measured at ~80 WorkspaceDB round-trips per settled tick. Every one of those reads is a pure function of registry
    tables that ``mutation_generation`` versions, so this view serves them
    from a cache revalidated against (service identity, generation) and every
    registry mutation anywhere in the app -- create, rename, archive,
    set-active, binding and membership writes -- invalidates it on the very
    next read.

    Contract details:

    * Only the four display reads are intercepted (``get_active_workspace``,
      ``list_workspaces``, ``list_runtime_bindings``,
      ``list_workspace_memberships``); every other attribute delegates to the
      wrapped service, so this object can stand in for it inside
      ``build_console_workspace_state``.
    * A raised read is never cached -- callers keep their existing degraded
      paths, and the next read retries live.
    * Doubles without a real ``int`` generation are never cached (a
      MagicMock's auto-attribute compares equal to itself forever and would
      freeze the view -- the TASK-21118 lesson), so reduced test doubles stay
      on live reads.
    * ADR-028 is preserved by construction: only the SQL is cached.
      ``display_state._safe_runtime_bindings`` still recomputes filesystem
      binding status straight from disk on every build.
    """

    __slots__ = ("_service", "_generation", "_cache")

    def __init__(self, service: Any) -> None:
        self._service = service
        self._generation: int | None = None
        self._cache: dict[tuple, Any] = {}

    @property
    def service(self) -> Any:
        """The wrapped registry service (identity checks by the owner)."""
        return self._service

    def _cacheable(self) -> bool:
        """Revalidate the cache against the service's mutation generation."""
        generation = getattr(self._service, "mutation_generation", None)
        if isinstance(generation, bool) or not isinstance(generation, int):
            self._cache.clear()
            self._generation = None
            return False
        if generation != self._generation:
            self._cache.clear()
            self._generation = generation
        return True

    def _read(self, key: tuple, method_name: str, *args: Any) -> Any:
        cacheable = self._cacheable()
        if cacheable and key in self._cache:
            return self._cache[key]
        value = getattr(self._service, method_name)(*args)
        if cacheable:
            self._cache[key] = value
        return value

    def get_active_workspace(self) -> Any:
        return self._read(("active",), "get_active_workspace")

    def list_workspaces(self, *, include_archived: bool = False) -> Any:
        if include_archived:
            # Rare (Settings archive lists); not worth a cache slot.
            return self._service.list_workspaces(include_archived=True)
        return self._read(("workspaces",), "list_workspaces")

    def list_runtime_bindings(self, workspace_id: str) -> Any:
        return self._read(
            ("bindings", str(workspace_id)), "list_runtime_bindings", workspace_id
        )

    def list_workspace_memberships(self, workspace_id: str) -> Any:
        return self._read(
            ("memberships", str(workspace_id)),
            "list_workspace_memberships",
            workspace_id,
        )

    def __getattr__(self, name: str) -> Any:
        # Private names never delegate: with ``__slots__`` a not-yet-bound
        # ``_service`` would otherwise recurse straight back through here.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._service, name)


class ConsoleTickWorkspaceBuilds:
    """One run tick's shared workspace-context build (TASK-22201).

    ``_sync_native_console_chat_ui`` used to build
    ``_build_console_workspace_context_state()`` SIX times per 0.2 s tick
    (measured with a stack probe): the two rail-state legs, the
    workspace-context push, the control bar's and the agent section's
    inspector legs, and the settings summary's rail read. This object is
    created fresh for ONE tick by ``tick_workspace_build_scope`` and serves
    every build the tick's own task performs; each read revalidates the
    controller's volatile-input fingerprint, so:

    * a settled tick pays for exactly one build;
    * the PR #660 freshness ruling holds -- a session created/activated by
      ``_sync_console_native_session_tabs`` mid-tick changes the store
      fingerprint, and the workspace-context push and the visibility check
      rebuild rather than reuse the pre-await snapshot;
    * a ``None`` fingerprint (reduced doubles, any component failure) means
      every read builds live, exactly the pre-cache behavior.

    Task-scoped and never shared across ticks: ``accepts_current_task``
    admits only the coroutine task that opened the scope, so workers,
    message handlers, and search settles interleaving during the tick's
    awaits keep building live -- a mutation followed by a push can never be
    masked. Inputs outside the fingerprint changing DURING one tick are
    repainted by the next tick, at most 0.2 s later -- the same cadence
    every other tick-driven surface repaints at.
    """

    __slots__ = ("_controller", "_task", "_fingerprint", "_state", "_building")

    def __init__(self, controller: "ConsoleWorkspaceController") -> None:
        self._controller = controller
        try:
            self._task = asyncio.current_task()
        except RuntimeError:
            self._task = None
        self._fingerprint: tuple | None = None
        self._state: Any = None
        self._building = False

    def accepts_current_task(self) -> bool:
        """Whether the calling context is the tick task that owns this cache.

        Also False while the shared build itself is running (re-entrancy
        guard) and when the scope was opened outside any asyncio task.
        """
        if self._building or self._task is None:
            return False
        try:
            return asyncio.current_task() is self._task
        except RuntimeError:
            return False

    def state(self) -> Any:
        """Return the current context state, rebuilding when inputs changed."""
        controller = self._controller
        fingerprint = controller._console_workspace_build_fingerprint()
        if (
            self._state is not None
            and fingerprint is not None
            and fingerprint == self._fingerprint
        ):
            return self._state
        self._building = True
        try:
            state = controller._build_console_workspace_context_state()
        finally:
            self._building = False
        # Recomputed AFTER the build: building advances canonical-owner
        # bookkeeping some fingerprint components include, and the stored
        # token must describe the state actually cached.
        self._fingerprint = controller._console_workspace_build_fingerprint()
        self._state = state
        return state


class ConsoleWorkspaceController:
    """Own Workspace lifecycle, resume, scope, and conversation browsing.

    The controller holds the canonical rich browser state and exposes the
    legacy Workspace search shape only as a compatibility projection. Its 57
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
        workspace_tree_owner_accessor: Callable[[], object] | None = None,
        flat_conversation_owner_accessor: Callable[[], object] | None = None,
        screen_lifecycle_token_accessor: Callable[[], object] | None = None,
        persist_workspace_tree_expansion_preferences: (
            Callable[[list[str]], None] | None
        ) = None,
        session_id_for_browser_row: (
            Callable[[ConsoleConversationBrowserInputRow], str | None] | None
        ) = None,
        ensure_chat_controller: Callable[[], Any] | None = None,
        set_conversation_row_loading: Callable[[str, bool], None] | None = None,
        mark_conversation_row_broken: Callable[[str], None] | None = None,
        rail_body_height_accessor: Callable[[], int | None] | None = None,
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
            workspace_tree_owner_accessor: Return the mounted Tree owner identity.
            flat_conversation_owner_accessor: Return the mounted flat owner identity.
            screen_lifecycle_token_accessor: Return the current screen mount identity.
            persist_workspace_tree_expansion_preferences: Persist the exact Tree
                disclosure set to durable Console configuration.
            session_id_for_browser_row: Resolve an already-open session for a row.
            ensure_chat_controller: Resolve or create the native chat controller.
            set_conversation_row_loading: Paint persisted-row loading state.
            mark_conversation_row_broken: Mark a missing persisted record.
            rail_body_height_accessor: Return the Console rail body height in
                terminal lines (or ``None`` before layout) so the browser's
                per-section/group visible-row cap can adapt to fill the
                available space.
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
        #: task-15864 AC#2: `ConsoleFleetLifecycleController._poke_console_wake_retry` -- resume
        #: is the one loader of persisted conversations into sessions, so
        #: session-open becomes a wake retry trigger here. Optional so the
        #: pre-existing direct-construction tests need no new kwarg.
        self._wake_retry_poke_fn = wake_retry_poke
        self._workspace_tree_owner_accessor = workspace_tree_owner_accessor
        self._flat_conversation_owner_accessor = flat_conversation_owner_accessor
        self._screen_lifecycle_token_accessor = screen_lifecycle_token_accessor
        self._persist_workspace_tree_expansion_preferences = (
            persist_workspace_tree_expansion_preferences
        )
        self._session_id_for_browser_row_fn = session_id_for_browser_row or (
            lambda _row: None
        )
        self._ensure_chat_controller_fn = (
            ensure_chat_controller or current_chat_controller_accessor
        )
        self._set_conversation_row_loading_fn = set_conversation_row_loading or (
            lambda _conversation_id, _loading: None
        )
        self._mark_conversation_row_broken_fn = mark_conversation_row_broken or (
            lambda _conversation_id: None
        )
        self._rail_body_height_accessor = rail_body_height_accessor
        self._workspace_files_visit_workspace_id: str | None = None
        self._workspace_files_modal: ConsoleWorkspaceFilesModal | None = None
        self._workspace_files_attention_generation = 0
        self._workspace_files_admission_lock = asyncio.Lock()
        self._workspace_files_admission_claim: str | None = None

        self._workspace_tree_search = SearchAttemptState()
        self._flat_conversation_search = SearchAttemptState()
        self._workspace_page_attempts: dict[str, PageAttemptState] = {}
        self._collapsed_workspace_ids: set[str] = set()
        self._workspace_membership_rows: dict[
            str, tuple[ConsoleConversationBrowserInputRow, ...]
        ] = {}
        self._canonical_membership_revision = 0
        self._canonical_owner_observations: dict[str, str] = {}

        # Canonical flat-browser state. Legacy workspace-search names below are
        # aliases/projections, never a second writer or backing row store.
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
    def _console_conversation_browser_query(self) -> str:
        return self._flat_conversation_search.query

    @_console_conversation_browser_query.setter
    def _console_conversation_browser_query(self, value: str) -> None:
        self._flat_conversation_search.query = str(value or "")

    @property
    def _console_conversation_browser_search_timer(self) -> Any:
        return self._flat_conversation_search.debounce

    @_console_conversation_browser_search_timer.setter
    def _console_conversation_browser_search_timer(self, value: Any) -> None:
        self._flat_conversation_search.debounce = value

    @property
    def _console_conversation_browser_search_token(self) -> int:
        return self._flat_conversation_search.generation

    @_console_conversation_browser_search_token.setter
    def _console_conversation_browser_search_token(self, value: int) -> None:
        self._flat_conversation_search.generation = int(value)

    @property
    def _console_conversation_browser_rows(
        self,
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        return self._flat_conversation_search.rows

    @_console_conversation_browser_rows.setter
    def _console_conversation_browser_rows(
        self, value: Iterable[ConsoleConversationBrowserInputRow]
    ) -> None:
        self._flat_conversation_search.rows = tuple(value)

    @property
    def _console_conversation_browser_total(self) -> int | None:
        return self._flat_conversation_search.total

    @_console_conversation_browser_total.setter
    def _console_conversation_browser_total(self, value: int | None) -> None:
        self._flat_conversation_search.total = value

    @property
    def _console_conversation_browser_error(self) -> str:
        return self._flat_conversation_search.error

    @_console_conversation_browser_error.setter
    def _console_conversation_browser_error(self, value: str) -> None:
        self._flat_conversation_search.error = str(value or "")

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

    def open_workspace_files_modal(
        self,
        *,
        inspector: WorkspaceFilesService,
        inspected_workspace_id: str,
        inspected_workspace_name: str,
        active_workspace_id: str | None,
        active_workspace_name: str,
        bindings: Sequence[WorkspaceFilesBinding],
        attention: WorkspaceFilesAttention | None = None,
        on_back_to_console: Callable[[], None] | None = None,
        on_visit_closed: Callable[[], None] | None = None,
    ) -> Any:
        """Push one already-resolved, read-only Workspace Files visit.

        This deliberately accepts only presentation-safe identities and the
        narrow read-only inspector supplied by its future entry owner.  It
        neither reads the registry nor updates Console workspace/session/
        context state; Task 3 owns admission and resolution from its controls.
        """
        safe_bindings = tuple(
            binding
            if (
                not binding.available
                or binding.scope is None
                or binding.scope.workspace_id == inspected_workspace_id
            )
            else replace(
                binding,
                scope=None,
                available=False,
                availability_copy="Unavailable: binding belongs to a different workspace.",
            )
            for binding in bindings
        )
        modal = ConsoleWorkspaceFilesModal(
            inspector=inspector,
            inspected_workspace_id=inspected_workspace_id,
            inspected_workspace_name=inspected_workspace_name,
            active_workspace_id=active_workspace_id,
            active_workspace_name=active_workspace_name,
            bindings=safe_bindings,
            attention=attention,
            on_back_to_console=on_back_to_console,
            on_visit_closed=on_visit_closed,
        )
        self.push_screen(modal)
        return modal

    async def request_workspace_files(
        self, workspace_id: str, *, expected_available: bool = False
    ) -> None:
        """Admit one non-activating Workspace Files visit for ``workspace_id``.

        Registry and binding inspection happens off the event loop.  The only
        main-loop effects are the modal push/focus and a generic notification;
        neither changes the Console workspace, session, or staged context.
        """
        requested_id = str(workspace_id or DEFAULT_WORKSPACE_ID).strip()
        size = self._screen.size
        if size.width < 80 or size.height < 24:
            self.app_instance.notify(WORKSPACE_FILES_MINIMUM_COPY, severity="warning")
            return

        async with self._workspace_files_admission_lock:
            modal = self._workspace_files_modal
            if modal is not None:
                if self._workspace_files_visit_workspace_id == requested_id:
                    if modal.is_mounted:
                        modal.query_one("#console-workspace-files-back").focus()
                else:
                    self.app_instance.notify(WORKSPACE_FILES_OTHER_VISIT_COPY, severity="warning")
                return
            if self._workspace_files_admission_claim is not None:
                if self._workspace_files_admission_claim != requested_id:
                    self.app_instance.notify(WORKSPACE_FILES_OTHER_VISIT_COPY, severity="warning")
                return
            self._workspace_files_admission_claim = requested_id
        try:
            resolution = await asyncio.to_thread(
                self._resolve_workspace_files_visit, requested_id
            )
            if resolution is None or (
                not resolution.had_bindings and not expected_available
            ):
                self.app_instance.notify(
                    WORKSPACE_FILES_NO_FOLDERS_COPY, severity="warning"
                )
                return

            attention = self._workspace_files_attention_snapshot()

            def _closed() -> None:
                if self._workspace_files_visit_workspace_id == resolution.workspace_id:
                    self._workspace_files_visit_workspace_id = None
                    self._workspace_files_modal = None

            async with self._workspace_files_admission_lock:
                if self._workspace_files_modal is not None:
                    # A closing visit owns the ledger until its awaited
                    # unmount callback clears it; never retarget it.
                    return
                self._workspace_files_visit_workspace_id = resolution.workspace_id
                self._workspace_files_modal = self.open_workspace_files_modal(
                    inspector=WorkspaceFileInspector(
                        getattr(self.app_instance, "workspace_registry_service", None)
                    ),
                    inspected_workspace_id=resolution.workspace_id,
                    inspected_workspace_name=resolution.workspace_name,
                    active_workspace_id=resolution.active_workspace_id,
                    active_workspace_name=resolution.active_workspace_name,
                    bindings=resolution.bindings,
                    attention=attention,
                    on_visit_closed=_closed,
                )
        finally:
            # Cancellation is a normal outcome for a Textual exclusive
            # worker. The claim must never survive it and block later visits.
            async with self._workspace_files_admission_lock:
                if self._workspace_files_admission_claim == requested_id:
                    self._workspace_files_admission_claim = None

    def _resolve_workspace_files_visit(
        self, workspace_id: str
    ) -> _WorkspaceFilesResolution | None:
        """Read current workspace/bindings and capture safe scopes off-loop."""
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        if registry is None or workspace_id == DEFAULT_WORKSPACE_ID:
            return None
        try:
            workspace = registry.get_workspace(workspace_id)
            if workspace is None or workspace.archived:
                return None
            raw_bindings = tuple(registry.list_folder_bindings(workspace_id))
            active = registry.get_active_workspace()
        except Exception:
            return None
        inspector = WorkspaceFileInspector(registry)
        bindings: list[WorkspaceFilesBinding] = []
        for binding in raw_bindings:
            try:
                scope = inspector.capture_binding(workspace_id, binding.binding_id)
            except ScopeCaptureError:
                bindings.append(
                    WorkspaceFilesBinding(
                        binding_id=binding.binding_id,
                        label=binding.label,
                        scope=None,
                        available=False,
                        availability_copy="Unavailable: folder access changed.",
                    )
                )
            else:
                bindings.append(
                    WorkspaceFilesBinding(
                        binding_id=binding.binding_id,
                        label=binding.label,
                        scope=scope,
                    )
                )
        return _WorkspaceFilesResolution(
            workspace_id=workspace.workspace_id,
            workspace_name=workspace.name,
            active_workspace_id=(active.workspace_id if active is not None else None),
            active_workspace_name=(active.name if active is not None else "Local Default"),
            bindings=tuple(bindings),
            had_bindings=bool(raw_bindings),
        )

    def _workspace_files_attention_snapshot(self) -> WorkspaceFilesAttention:
        """Return only generic Console-attention copy; never payload details."""
        count_getter = getattr(self._screen, "_console_pending_approval_count", None)
        count = int(count_getter()) if callable(count_getter) else 0
        controller = self._current_chat_controller_accessor()
        run_state = getattr(controller, "run_state", None)
        status = str(getattr(run_state, "status", "") or "").casefold()
        # These all derive from existing Console state: run status and the
        # durable fleet-unseen marker. No raw tool/approval payload crosses
        # this boundary.
        blocked = "block" in status or "approval" in status
        failed = "fail" in status or "error" in status
        new_activity = bool(self._fleet_unseen_ids_accessor())
        if count:
            noun = "approval" if count == 1 else "approvals"
            return WorkspaceFilesAttention(
                f"Console needs attention · {count} {noun} waiting",
                pending_approval_count=count,
                has_blocked_activity=blocked,
                has_failed_activity=failed,
                has_new_activity=new_activity,
            )
        if blocked or failed or new_activity:
            return WorkspaceFilesAttention(
                "Console has new activity",
                has_blocked_activity=blocked,
                has_failed_activity=failed,
                has_new_activity=new_activity,
            )
        return WorkspaceFilesAttention()

    def update_workspace_files_attention(self) -> None:
        """Publish one monotonically ordered generic attention snapshot."""
        modal = self._workspace_files_modal
        if modal is None:
            return
        self._workspace_files_attention_generation += 1
        modal.update_attention(
            self._workspace_files_attention_snapshot(),
            self._workspace_files_attention_generation,
        )

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

    def _blank_console_session_settings(self) -> Any:
        """Build config-owned defaults for an eligible workspace blank chat."""
        app_config = getattr(self.app_instance, "app_config", {})
        if not isinstance(app_config, Mapping):
            app_config = {}
        return blank_console_session_settings(app_config)

    def _console_new_chat_default_generation(self) -> int:
        """Return the current app-owned explicit-default generation."""
        generation = getattr(
            self.app_instance,
            "console_new_chat_default_generation",
            0,
        )
        return generation if type(generation) is int and generation >= 0 else 0

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

    def workspace_tree_expansion_preferences(self) -> frozenset[str] | None:
        """Return the persisted disclosure set, preserving explicit emptiness."""

        raw = self._console_conversation_browser_config().get("expanded_workspace_ids")
        if not isinstance(raw, (list, tuple, set, frozenset)):
            return None
        return frozenset(
            workspace_id for value in raw if (workspace_id := str(value or "").strip())
        )

    def set_workspace_tree_expansion_preferences(
        self, workspace_ids: frozenset[str]
    ) -> None:
        """Persist the exact non-search disclosure preference set."""

        serialized = sorted(
            str(workspace_id).strip()
            for workspace_id in workspace_ids
            if str(workspace_id).strip()
        )
        self._console_conversation_browser_config()["expanded_workspace_ids"] = (
            serialized
        )
        callback = self._persist_workspace_tree_expansion_preferences
        if callback is not None:
            callback(serialized)

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

    def _console_rail_body_height(self) -> int | None:
        accessor = self._rail_body_height_accessor
        return accessor() if accessor is not None else None

    def _workspace_tree_owner_token(self) -> object | None:
        accessor = self._workspace_tree_owner_accessor
        return accessor() if accessor is not None else None

    def _flat_conversation_owner_token(self) -> object:
        accessor = self._flat_conversation_owner_accessor
        return accessor() if accessor is not None else self._screen

    def _screen_lifecycle_token(self) -> object:
        accessor = self._screen_lifecycle_token_accessor
        return accessor() if accessor is not None else self._screen

    async def refresh_workspace_tree_search(self, query: str) -> None:
        """Run one full-scope named-workspace search attempt."""
        lane = self._workspace_tree_search
        lane.query = str(query or "")
        lane.generation += 1
        generation = lane.generation
        owner_token = self._workspace_tree_owner_token()
        lifecycle_token = self._screen_lifecycle_token()
        request_key = (
            "workspaces",
            lane.query,
            generation,
            owner_token,
            lifecycle_token,
            self._canonical_membership_revision,
        )
        lane.request_key = request_key
        lane.error = ""
        lane.retry_query = None
        if self._screen_running_accessor():
            self._sync_console_workspace_context()
        try:
            rows, total = await self._load_workspace_tree_search_rows(lane.query)
        except Exception:
            if self._workspace_search_attempt_is_current(request_key):
                lane.error = "Workspace search is unavailable."
                lane.retry_query = lane.query
            if lane.request_key == request_key:
                lane.request_key = None
                if self._screen_running_accessor():
                    self._sync_console_workspace_context()
            return
        if not self._workspace_search_attempt_is_current(request_key):
            if lane.request_key == request_key:
                lane.request_key = None
                if self._screen_running_accessor():
                    self._sync_console_workspace_context()
            return
        lane.rows = self._merge_console_browser_rows(rows)
        self._record_canonical_owner_rows(lane.rows)
        lane.total = max(len(lane.rows), int(total or 0))
        lane.cache = {lane.query: (lane.rows, lane.total)}
        lane.settled_rows = lane.rows
        lane.settled_total = lane.total
        lane.settled_query = lane.query
        lane.error = ""
        lane.retry_query = None
        lane.request_key = None
        self._sync_console_workspace_context()

    async def refresh_flat_conversation_search(self, query: str) -> None:
        """Run one Default/unassigned search attempt independently."""
        lane = self._flat_conversation_search
        lane.query = str(query or "")
        lane.generation += 1
        generation = lane.generation
        owner_token = self._flat_conversation_owner_token()
        lifecycle_token = self._screen_lifecycle_token()
        request_key = (
            "conversations",
            lane.query,
            generation,
            owner_token,
            lifecycle_token,
            self._canonical_membership_revision,
        )
        lane.request_key = request_key
        lane.error = ""
        lane.retry_query = None
        try:
            rows, total = await self._load_flat_conversation_search_rows(lane.query)
        except Exception:
            if self._flat_search_attempt_is_current(request_key):
                lane.error = "Conversation search is unavailable."
                lane.retry_query = lane.query
            if lane.request_key == request_key:
                lane.request_key = None
                if self._screen_running_accessor():
                    self._sync_console_workspace_context()
            return
        if not self._flat_search_attempt_is_current(request_key):
            if lane.request_key == request_key:
                lane.request_key = None
                if self._screen_running_accessor():
                    self._sync_console_workspace_context()
            return
        lane.rows = self._merge_console_browser_rows(
            row for row in rows if self._row_belongs_to_flat_projection(row)
        )
        self._record_canonical_owner_rows(lane.rows)
        lane.total = max(len(lane.rows), int(total or 0))
        lane.cache = {lane.query: (lane.rows, lane.total)}
        lane.settled_rows = lane.rows
        lane.settled_total = lane.total
        lane.settled_query = lane.query
        lane.error = ""
        lane.retry_query = None
        lane.request_key = None
        self._sync_console_workspace_context()

    async def retry_workspace_tree_search(self) -> None:
        """Replace the failed workspace-search generation, if any."""
        query = self._workspace_tree_search.retry_query
        if query is not None:
            await self.refresh_workspace_tree_search(query)

    async def retry_flat_conversation_search(self) -> None:
        """Replace the failed flat-search generation, if any."""
        query = self._flat_conversation_search.retry_query
        if query is not None:
            await self.refresh_flat_conversation_search(query)

    def _workspace_search_attempt_is_current(
        self, request_key: tuple[str, str, int, object, object, int]
    ) -> bool:
        lane = self._workspace_tree_search
        return bool(
            self._screen_running_accessor()
            and lane.request_key == request_key
            and lane.generation == request_key[2]
            and lane.query == request_key[1]
            and self._workspace_tree_owner_token() is request_key[3]
            and self._screen_lifecycle_token() is request_key[4]
            and self._canonical_membership_revision == request_key[5]
        )

    def _flat_search_attempt_is_current(
        self, request_key: tuple[str, str, int, object, object, int]
    ) -> bool:
        lane = self._flat_conversation_search
        return bool(
            self._screen_running_accessor()
            and lane.request_key == request_key
            and lane.generation == request_key[2]
            and lane.query == request_key[1]
            and self._flat_conversation_owner_token() is request_key[3]
            and self._screen_lifecycle_token() is request_key[4]
            and self._canonical_membership_revision == request_key[5]
        )

    async def _load_workspace_tree_search_rows(
        self, query: str
    ) -> tuple[tuple[ConsoleConversationBrowserInputRow, ...], int | None]:
        request_key = self._workspace_tree_search.request_key
        offset = 0
        named_rows: tuple[ConsoleConversationBrowserInputRow, ...] = ()
        while True:
            rows, total, error = await self._persisted_console_browser_rows(
                query,
                scopes=(("all", None),),
                offset=offset,
            )
            if error:
                raise RuntimeError(error)
            named_rows = self._merge_console_browser_rows(
                named_rows,
                (
                    row
                    for row in rows
                    if row.scope_type == "workspace"
                    and row.workspace_id not in (None, DEFAULT_WORKSPACE_ID)
                ),
            )
            if (
                request_key is not None
                and not self._workspace_search_attempt_is_current(request_key)
            ):
                return (), None
            offset += CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT
            if total is None or offset >= total:
                break
        native_rows = self._filter_console_browser_rows_for_query(
            self._native_console_browser_rows(),
            query,
        )
        named_rows = self._merge_console_browser_rows(
            (
                row
                for row in native_rows
                if row.scope_type == "workspace"
                and row.workspace_id not in (None, DEFAULT_WORKSPACE_ID)
            ),
            named_rows,
        )
        if request_key is not None and not self._workspace_search_attempt_is_current(
            request_key
        ):
            return (), None
        return named_rows, len(named_rows)

    async def _load_flat_conversation_search_rows(
        self, query: str
    ) -> tuple[tuple[ConsoleConversationBrowserInputRow, ...], int | None]:
        rows, total, error = await self._persisted_console_browser_rows(query)
        if error:
            raise RuntimeError(error)
        return tuple(rows), total

    @staticmethod
    def _row_belongs_to_flat_projection(
        row: ConsoleConversationBrowserInputRow,
    ) -> bool:
        return row.scope_type == "global" or row.workspace_id in (
            None,
            DEFAULT_WORKSPACE_ID,
        )

    @staticmethod
    def _new_workspace_page_state(
        *,
        rows: Iterable[ConsoleConversationBrowserInputRow] = (),
        next_cursor: int | None = None,
    ) -> PageAttemptState:
        return PageAttemptState(rows=tuple(rows), next_cursor=next_cursor)

    def _workspace_membership_token(
        self, workspace_id: str
    ) -> tuple[str, ...] | UnknownMembership:
        service = getattr(self.app_instance, "workspace_registry_service", None)
        list_conversations = getattr(service, "list_workspace_conversations", None)
        if not callable(list_conversations):
            return _MEMBERSHIP_UNKNOWN
        try:
            return tuple(
                sorted(
                    str(getattr(row, "item_id", "") or "")
                    for row in list_conversations(workspace_id)
                    if str(getattr(row, "item_id", "") or "")
                )
            )
        except Exception:
            return _MEMBERSHIP_UNKNOWN

    async def load_workspace_tree_page(self, workspace_id: str, cursor: int) -> None:
        """Load one bounded workspace page if its full request key stays current."""
        workspace_id = str(workspace_id or "").strip()
        cursor = max(0, int(cursor))
        attempt = self._workspace_page_attempts.setdefault(
            workspace_id, PageAttemptState()
        )
        if attempt.loading and attempt.request_key is not None:
            return
        attempt.generation += 1
        generation = attempt.generation
        membership_token = self._workspace_membership_token(workspace_id)
        if membership_token is _MEMBERSHIP_UNKNOWN:
            attempt.loading = False
            attempt.error = WORKSPACE_CONVERSATIONS_ACCESS_UNKNOWN
            attempt.retry_cursor = cursor
            attempt.membership_unknown = True
            self._sync_console_workspace_context()
            return
        attempt.membership_unknown = False
        request_key = (
            workspace_id,
            membership_token,
            self._workspace_tree_search.generation,
            cursor,
        )
        attempt.request_key = request_key
        attempt.owner_token = self._workspace_tree_owner_token()
        attempt.lifecycle_token = self._screen_lifecycle_token()
        attempt.loading = True
        attempt.error = ""
        attempt.retry_cursor = None
        attempt.membership_unknown = False
        self._sync_console_workspace_context()
        try:
            try:
                rows, next_cursor = await self._fetch_workspace_tree_page(
                    workspace_id, cursor
                )
            except Exception:
                self._commit_workspace_page_failure(
                    workspace_id, generation, request_key
                )
                return
            if not self._workspace_page_attempt_is_current(
                workspace_id, generation, request_key
            ):
                return
            attempt.rows = self._merge_page_rows(attempt.rows, rows)
            self._record_canonical_owner_rows(rows)
            attempt.membership_token = request_key[1]
            attempt.next_cursor = next_cursor
            attempt.error = ""
            attempt.retry_cursor = None
        finally:
            self._settle_workspace_page_attempt(workspace_id, generation, request_key)

    async def retry_workspace_tree_page(self, workspace_id: str) -> None:
        """Replace the failed page generation for one workspace."""
        attempt = self._workspace_page_attempts.get(workspace_id)
        if attempt is not None and attempt.retry_cursor is not None:
            await self.load_workspace_tree_page(workspace_id, attempt.retry_cursor)

    def request_workspace_tree_page(self, workspace_id: str, cursor: int) -> None:
        """Schedule one page worker without canceling another workspace lane."""
        target = str(workspace_id or "").strip()
        attempt = self._workspace_page_attempts.setdefault(target, PageAttemptState())
        if attempt.loading:
            return
        attempt.loading = True
        attempt.error = ""
        attempt.retry_cursor = None
        self._sync_console_workspace_context()
        attempt.worker = self.run_worker(
            self.load_workspace_tree_page(target, cursor),
            group=f"console-workspace-page-{target}",
            exclusive=False,
        )

    def _workspace_page_attempt_is_current(
        self,
        workspace_id: str,
        generation: int,
        request_key: tuple[str, tuple[str, ...], int, int] | None,
    ) -> bool:
        attempt = self._workspace_page_attempts.get(workspace_id)
        request_is_current = bool(
            request_key is not None
            and attempt is not None
            and self._screen_running_accessor()
            and attempt.generation == generation
            and attempt.request_key == request_key
            and request_key[2] == self._workspace_tree_search.generation
            and attempt.owner_token is self._workspace_tree_owner_token()
            and workspace_id not in self._collapsed_workspace_ids
        )
        if not request_is_current:
            return False
        if attempt.lifecycle_token is not self._screen_lifecycle_token():
            return False
        current_membership = self._workspace_membership_token(workspace_id)
        if current_membership is _MEMBERSHIP_UNKNOWN:
            self._mark_workspace_membership_unknown(workspace_id)
            return False
        if request_key[1] != current_membership:
            return False
        return True

    def transition_workspace_tree_expansion(
        self, workspace_id: str, *, expanded: bool
    ) -> None:
        """Fence a collapsed workspace page lane while retaining loaded rows."""

        target = str(workspace_id or "").strip()
        if not target:
            return
        if expanded:
            self._collapsed_workspace_ids.discard(target)
            if target not in self._workspace_page_attempts:
                self.request_workspace_tree_page(target, 0)
            return
        self._collapsed_workspace_ids.add(target)
        attempt = self._workspace_page_attempts.get(target)
        if attempt is None:
            return
        attempt.generation += 1
        attempt.request_key = None
        attempt.loading = False
        worker = attempt.worker
        attempt.worker = None
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            cancel()

    def request_next_workspace_tree_page(self, workspace_id: str) -> None:
        """Request the retained lane's next cursor, when one exists."""

        attempt = self._workspace_page_attempts.get(str(workspace_id or "").strip())
        if attempt is not None and attempt.next_cursor is not None:
            self.request_workspace_tree_page(workspace_id, attempt.next_cursor)

    def activate_workspace_id(self, workspace_id: str) -> None:
        """Activate a workspace selected from the native Tree."""

        self._switch_console_workspace(workspace_id)

    def _switch_console_workspace(self, workspace_id: str) -> bool:
        """Switch registry, session, controller, rail, and native UI together."""

        target = str(workspace_id or "").strip()
        service = getattr(self.app_instance, "workspace_registry_service", None)
        if not target or service is None:
            return False
        try:
            service.set_active_workspace(target)
        except Exception:
            logger.opt(exception=True).warning("Unable to switch Console workspace")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Workspace could not be selected.", severity="error")
            return False
        self._sync_console_chat_core_state()
        self._activate_console_session_for_workspace(target)
        self._sync_console_workspace_context()
        native_sync = self._sync_native_console_chat_ui()
        if inspect.isawaitable(native_sync):
            self.run_worker(
                native_sync,
                exclusive=True,
                group="console-sync",
            )
        return True

    def _settle_workspace_page_attempt(
        self,
        workspace_id: str,
        generation: int,
        request_key: tuple[str, tuple[str, ...], int, int] | None,
    ) -> None:
        """Settle only the captured page attempt and publish to its current owner."""
        attempt = self._workspace_page_attempts.get(workspace_id)
        if (
            attempt is None
            or attempt.generation != generation
            or attempt.request_key != request_key
        ):
            return
        attempt.loading = False
        if (
            self._screen_running_accessor()
            and attempt.owner_token is self._workspace_tree_owner_token()
        ):
            self._sync_console_workspace_context()

    def _commit_workspace_page_failure(
        self,
        workspace_id: str,
        generation: int,
        request_key: tuple[str, tuple[str, ...], int, int] | None,
    ) -> None:
        if not self._workspace_page_attempt_is_current(
            workspace_id, generation, request_key
        ):
            return
        attempt = self._workspace_page_attempts[workspace_id]
        attempt.error = WORKSPACE_CONVERSATIONS_LOAD_FAILED
        attempt.retry_cursor = request_key[3] if request_key is not None else None
        attempt.membership_unknown = False

    @staticmethod
    def _merge_page_rows(
        current: Iterable[ConsoleConversationBrowserInputRow],
        incoming: Iterable[ConsoleConversationBrowserInputRow],
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        merged: dict[str, ConsoleConversationBrowserInputRow] = {}
        for row in (*tuple(current), *tuple(incoming)):
            conversation_id = str(row.conversation_id or "").strip()
            if conversation_id:
                merged.setdefault(conversation_id, row)
        return tuple(merged.values())

    @staticmethod
    def _canonical_owner_id(row: ConsoleConversationBrowserInputRow) -> str:
        workspace_id = str(row.workspace_id or "").strip()
        if row.scope_type == "global" or workspace_id in ("", DEFAULT_WORKSPACE_ID):
            return DEFAULT_WORKSPACE_ID
        return workspace_id

    def _record_canonical_owner_rows(
        self, rows: Iterable[ConsoleConversationBrowserInputRow]
    ) -> None:
        for row in rows:
            conversation_id = str(row.conversation_id or "").strip()
            if conversation_id:
                self._canonical_owner_observations[conversation_id] = (
                    self._canonical_owner_id(row)
                )

    def _rows_with_latest_canonical_owner(
        self, rows: Iterable[ConsoleConversationBrowserInputRow]
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        return tuple(
            row
            for row in rows
            if not row.conversation_id
            or (
                observed := self._canonical_owner_observations.get(
                    str(row.conversation_id)
                )
            )
            is None
            or self._canonical_owner_id(row) == observed
        )

    async def _fetch_workspace_tree_page(
        self, workspace_id: str, cursor: int
    ) -> tuple[tuple[ConsoleConversationBrowserInputRow, ...], int | None]:
        rows, total, error = await self._fetch_workspace_rows(
            workspace_id, query="", cursor=cursor
        )
        if error:
            raise RuntimeError(error)
        next_cursor = cursor + CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT
        if total is None or next_cursor >= total:
            next_cursor = None
        return rows, next_cursor

    async def _fetch_workspace_rows(
        self, workspace_id: str, *, query: str, cursor: int
    ) -> tuple[tuple[ConsoleConversationBrowserInputRow, ...], int | None, str]:
        service = getattr(self.app_instance, "chat_conversation_scope_service", None)
        list_conversations = getattr(service, "list_conversations", None)
        if not callable(list_conversations):
            return (), None, ""
        try:
            list_kwargs = {
                "mode": "local",
                "query": query,
                "scope_type": "workspace",
                "workspace_id": workspace_id,
                "limit": CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
                "offset": cursor,
            }
            result = (
                list_conversations(**list_kwargs)
                if inspect.iscoroutinefunction(list_conversations)
                else await asyncio.to_thread(list_conversations, **list_kwargs)
            )
            result = await result if inspect.isawaitable(result) else result
        except Exception:
            # TASK-25827: this was the ONLY record of why the rail failed and
            # it sat at debug, so "check the app log" led nowhere.
            logger.opt(exception=True).warning(
                "Unable to load Console workspace page workspace_id={}", workspace_id
            )
            return (), None, WORKSPACE_CONVERSATIONS_LOAD_FAILED
        if not isinstance(result, dict):
            return (), 0, ""
        items = result.get("items") if isinstance(result.get("items"), list) else []
        labels = self._console_browser_workspace_labels()
        starred_ids = self._starred_console_conversation_ids()
        rows: list[ConsoleConversationBrowserInputRow] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            conversation_id = str(item.get("id") or "").strip()
            if not conversation_id:
                continue
            row = ConsoleConversationBrowserInputRow(
                row_key=conversation_id,
                conversation_id=conversation_id,
                native_session_id=None,
                title=str(item.get("title") or "Untitled conversation"),
                scope_type="workspace",
                workspace_id=workspace_id,
                workspace_label=labels.get(workspace_id, workspace_id),
                status=str(item.get("state") or "workspace-thread"),
                selected=conversation_id == self._current_console_conversation_id(),
                source_kind="persisted",
                updated_sort=console_persisted_row_updated_sort(item),
                run_marker=self._console_browser_unseen_marker(conversation_id),
            )
            rows.append(self._apply_console_browser_star_state(row, starred_ids))
        total = result.get("total")
        if total is None and isinstance(result.get("pagination"), dict):
            total = result["pagination"].get("total")
        try:
            total_count = int(total)
        except (TypeError, ValueError):
            total_count = len(rows)
        return tuple(rows), total_count, ""

    def apply_workspace_membership_snapshot(
        self,
        memberships: Mapping[str, tuple[str, ...]],
        *,
        complete: bool,
        workspace_labels: Mapping[str, str] | None = None,
        canonical_rows: Iterable[ConsoleConversationBrowserInputRow] = (),
    ) -> None:
        """Apply already-read canonical membership without projection-time I/O."""
        current_tokens = {
            str(workspace_id): tuple(dict.fromkeys(conversation_ids))
            for workspace_id, conversation_ids in memberships.items()
        }
        labels = dict(workspace_labels or {})
        owner_by_conversation: dict[str, str] = {}
        for workspace_id in sorted(current_tokens):
            for conversation_id in current_tokens[workspace_id]:
                owner_by_conversation.setdefault(conversation_id, workspace_id)
        canonical_by_conversation: dict[str, ConsoleConversationBrowserInputRow] = {}
        canonical_by_owner: dict[
            tuple[str, str], ConsoleConversationBrowserInputRow
        ] = {}
        for row in canonical_rows:
            conversation_id = str(row.conversation_id or "").strip()
            if not conversation_id:
                continue
            owner_id = self._canonical_owner_id(row)
            owner_by_conversation[conversation_id] = owner_id
            canonical_by_conversation[conversation_id] = row
            canonical_by_owner[conversation_id, owner_id] = row
            labels.setdefault(owner_id, str(row.workspace_label or owner_id))

        if complete:
            self._canonical_membership_revision += 1
            complete_owners: dict[str, str] = {}
            for workspace_id in sorted(current_tokens):
                for conversation_id in current_tokens[workspace_id]:
                    complete_owners.setdefault(conversation_id, workspace_id)
            complete_workspace_ids = set(current_tokens)
            for conversation_id, owner_id in tuple(
                self._canonical_owner_observations.items()
            ):
                if (
                    owner_id in complete_workspace_ids
                    and conversation_id not in complete_owners
                ):
                    self._canonical_owner_observations.pop(conversation_id, None)
            self._canonical_owner_observations.update(complete_owners)
            for conversation_id, owner_id in tuple(owner_by_conversation.items()):
                if (
                    owner_id in complete_workspace_ids
                    and conversation_id not in complete_owners
                ):
                    owner_by_conversation.pop(conversation_id, None)
            owner_by_conversation.update(complete_owners)
        for conversation_id, owner_id in self._canonical_owner_observations.items():
            owner_by_conversation[conversation_id] = owner_id
            canonical_row = canonical_by_owner.get((conversation_id, owner_id))
            if canonical_row is not None:
                canonical_by_conversation[conversation_id] = canonical_row
            labels.setdefault(
                owner_id,
                "Default" if owner_id == DEFAULT_WORKSPACE_ID else owner_id,
            )

        changed_workspaces: set[str] = set()
        for workspace_id, attempt in self._workspace_page_attempts.items():
            token = current_tokens.get(workspace_id)
            owner_conflict = any(
                owner_by_conversation.get(str(row.conversation_id or "").strip())
                not in (None, workspace_id)
                for row in attempt.rows
            )
            if token is None:
                if owner_conflict:
                    changed_workspaces.add(workspace_id)
                continue
            if complete:
                if attempt.membership_unknown:
                    attempt.error = ""
                    attempt.retry_cursor = None
                    attempt.membership_unknown = False
                if attempt.membership_token is None:
                    attempt.membership_token = token
                elif attempt.membership_token != token:
                    changed_workspaces.add(workspace_id)
            if owner_conflict:
                changed_workspaces.add(workspace_id)

        moved_rows: dict[str, list[ConsoleConversationBrowserInputRow]] = {}
        for workspace_id, cached_rows in tuple(self._workspace_membership_rows.items()):
            if workspace_id not in current_tokens:
                continue
            retained: list[ConsoleConversationBrowserInputRow] = []
            for row in cached_rows:
                conversation_id = str(row.conversation_id or "").strip()
                owner_id = owner_by_conversation.get(conversation_id)
                if owner_id == workspace_id or (owner_id is None and not complete):
                    retained.append(row)
                elif owner_id is not None:
                    moved_rows.setdefault(owner_id, []).append(
                        replace(
                            row,
                            workspace_id=owner_id,
                            workspace_label=labels.get(owner_id, owner_id),
                        )
                    )
            retained_rows = tuple(retained)
            if retained_rows != cached_rows:
                self._workspace_membership_rows[workspace_id] = retained_rows

        for workspace_id in changed_workspaces:
            attempt = self._workspace_page_attempts[workspace_id]
            retained: list[ConsoleConversationBrowserInputRow] = []
            for row in attempt.rows:
                conversation_id = str(row.conversation_id or "").strip()
                owner_id = owner_by_conversation.get(conversation_id)
                if owner_id == workspace_id:
                    retained.append(row)
                elif owner_id == DEFAULT_WORKSPACE_ID:
                    canonical_row = canonical_by_conversation.get(conversation_id)
                    if canonical_row is not None and (
                        self._canonical_owner_id(canonical_row) != owner_id
                    ):
                        canonical_row = None
                    moved_rows.setdefault(owner_id, []).append(
                        canonical_row
                        or replace(
                            row,
                            workspace_id=owner_id,
                            workspace_label="Default",
                        )
                    )
                elif owner_id in labels and owner_id != DEFAULT_WORKSPACE_ID:
                    canonical_row = canonical_by_conversation.get(conversation_id)
                    if canonical_row is not None and (
                        self._canonical_owner_id(canonical_row) != owner_id
                    ):
                        canonical_row = None
                    moved_rows.setdefault(owner_id, []).append(
                        canonical_row
                        or replace(
                            row,
                            workspace_id=owner_id,
                            workspace_label=labels[owner_id],
                        )
                    )
                elif owner_id is None and not complete:
                    retained.append(row)
            attempt.rows = tuple(retained)
            attempt.membership_token = current_tokens.get(
                workspace_id,
                tuple(
                    str(row.conversation_id) for row in retained if row.conversation_id
                ),
            )
            attempt.generation += 1
            attempt.request_key = None
            attempt.loading = False
            attempt.error = ""
            attempt.retry_cursor = None
            attempt.next_cursor = 0
            attempt.membership_unknown = False

        for workspace_id, rows in moved_rows.items():
            if workspace_id == DEFAULT_WORKSPACE_ID:
                current = self._workspace_membership_rows.get(workspace_id, ())
                self._workspace_membership_rows[workspace_id] = self._merge_page_rows(
                    current, rows
                )
                continue
            attempt = self._workspace_page_attempts.setdefault(
                workspace_id, PageAttemptState()
            )
            attempt.rows = self._merge_page_rows(attempt.rows, rows)
            attempt.membership_token = current_tokens[workspace_id]

    def _mark_workspace_membership_unknown(self, workspace_id: str) -> None:
        attempt = self._workspace_page_attempts.get(workspace_id)
        if attempt is None:
            return
        attempt.error = WORKSPACE_CONVERSATIONS_ACCESS_UNKNOWN
        attempt.loading = False
        attempt.retry_cursor = (
            attempt.next_cursor if attempt.next_cursor is not None else 0
        )
        attempt.membership_unknown = True

    def _prune_stale_workspace_page_attempts(self) -> tuple[WorkspaceRecord, ...]:
        """Drop page attempts for workspaces gone from the registry.

        Returns:
            Live named-workspace records (non-archived, excluding the
            built-in Default), the set the workspace tree renders.
        """
        all_records = tuple(
            record
            for record in self._console_browser_workspace_records()
            if str(record.workspace_id or "").strip()
            and not bool(getattr(record, "archived", False))
        )
        records = tuple(
            record
            for record in all_records
            if str(record.workspace_id or "").strip() != DEFAULT_WORKSPACE_ID
        )
        workspace_ids = {str(record.workspace_id) for record in records}
        for workspace_id in tuple(self._workspace_page_attempts):
            if workspace_id not in workspace_ids:
                self._workspace_page_attempts.pop(workspace_id, None)
        return records

    def workspace_tree_projection(
        self,
        rows: Iterable[ConsoleConversationBrowserInputRow] = (),
        *,
        prepared_rows: tuple[ConsoleConversationBrowserInputRow, ...] | None = None,
    ) -> tuple[WorkspaceTreeWorkspace, ...]:
        """Return the current immutable named-workspace projection.

        Args:
            rows: Browser input rows to merge with the page-attempt rows on
                the no-query path.
            prepared_rows: The state build's already merged + canonical-owner
                + overlay processed union of browser and page-attempt rows
                (TASK-22201). When provided and no tree query is active, the
                merge/canonical/overlay pipeline is NOT re-run here -- the
                build already ran it once over the identical input set.
                Standalone callers omit it and keep the self-contained path.
        """
        records = self._prune_stale_workspace_page_attempts()
        workspace_lane = self._workspace_tree_search
        projection_query = (
            workspace_lane.settled_query
            if workspace_lane.error
            else workspace_lane.query
        )
        if projection_query.strip():
            source_rows = (
                workspace_lane.settled_rows
                if workspace_lane.error
                else workspace_lane.rows
            )
            source_rows = self._rows_with_latest_canonical_owner(source_rows)
            source_rows = self._overlay_current_console_browser_markers(source_rows)
        elif prepared_rows is not None:
            source_rows = prepared_rows
        else:
            source_rows = self._merge_console_browser_rows(
                rows,
                *(attempt.rows for attempt in self._workspace_page_attempts.values()),
            )
            source_rows = self._rows_with_latest_canonical_owner(source_rows)
            source_rows = self._overlay_current_console_browser_markers(source_rows)
        return build_workspace_tree_state(
            workspaces=(
                (str(record.workspace_id), str(record.name or record.workspace_id))
                for record in records
            ),
            rows=source_rows,
            next_cursors={
                workspace_id: attempt.next_cursor
                for workspace_id, attempt in self._workspace_page_attempts.items()
            },
            loading={
                workspace_id: attempt.loading
                for workspace_id, attempt in self._workspace_page_attempts.items()
            },
            errors={
                workspace_id: attempt.error
                for workspace_id, attempt in self._workspace_page_attempts.items()
            },
            retry_cursors={
                workspace_id: attempt.retry_cursor
                for workspace_id, attempt in self._workspace_page_attempts.items()
            },
            membership_unknown={
                workspace_id: attempt.membership_unknown
                for workspace_id, attempt in self._workspace_page_attempts.items()
            },
            query=projection_query,
            active_workspace_id=(
                self._current_console_workspace_context().active_workspace_id
            ),
        )

    def transition_browser_search(self, query: str, disabled: bool) -> None:
        """Apply one input change and schedule its canonical browser refresh."""
        if disabled or query == self._console_conversation_browser_query:
            return
        if not query.strip():
            self.clear_console_conversation_browser_search()
            return
        self._console_conversation_browser_query = query
        self._console_conversation_browser_search_token += 1
        token = self._console_conversation_browser_search_token
        self._flat_conversation_search.request_key = (
            "conversations",
            query,
            token,
            self._flat_conversation_owner_token(),
            self._screen_lifecycle_token(),
            self._canonical_membership_revision,
        )
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

    def transition_workspace_tree_search(self, query: str, disabled: bool) -> None:
        """Debounce Workspaces search without touching Conversations state."""
        lane = self._workspace_tree_search
        if disabled or query == lane.query:
            return
        if lane.debounce is not None:
            lane.debounce.stop()
            lane.debounce = None
        lane.query = str(query or "")
        lane.generation += 1
        generation = lane.generation
        if not lane.query.strip():
            worker = lane.worker
            cancel = getattr(worker, "cancel", None)
            if callable(cancel):
                cancel()
            lane.worker = None
            lane.request_key = None
            lane.rows = ()
            lane.total = None
            lane.settled_rows = ()
            lane.settled_total = None
            lane.settled_query = ""
            lane.error = ""
            lane.retry_query = None
            self._sync_console_workspace_context()
            return
        lane.debounce = self._schedule_console_browser_timer(
            0.2,
            partial(self._start_workspace_tree_search, lane.query, generation),
        )

    def _start_workspace_tree_search(self, query: str, generation: int) -> None:
        lane = self._workspace_tree_search
        if generation != lane.generation or query != lane.query:
            return
        lane.worker = self.run_worker(
            self.refresh_workspace_tree_search(query),
            group="console-workspace-tree-search",
            exclusive=True,
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
        self._flat_conversation_search.request_key = None
        self._flat_conversation_search.retry_query = None
        self._flat_conversation_search.settled_rows = ()
        self._flat_conversation_search.settled_total = None
        self._flat_conversation_search.settled_query = ""
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
        self._flat_conversation_search.request_key = (
            "conversations",
            query,
            token,
            self._flat_conversation_owner_token(),
            self._screen_lifecycle_token(),
            self._canonical_membership_revision,
        )
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
                    *self._workspace_membership_rows.values(),
                ),
                query,
            )
        )
        self._console_conversation_browser_total = None
        self._console_conversation_browser_error = ""
        self._sync_console_workspace_context()
        self._flat_conversation_search.worker = self.run_worker(
            self._refresh_console_conversation_browser_search(query, token),
            group="console-flat-conversation-search",
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

    def _overlay_current_console_browser_markers(
        self,
        rows: Iterable[ConsoleConversationBrowserInputRow],
        current_conversation_id: str | None = None,
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        row_tuple = tuple(rows)
        unseen_ids = self._console_fleet_unseen_ids()
        unseen_marker = resolve_glyph(
            CONSOLE_RUN_MARKER_GLYPHS.get(ConsoleRunMarker.SUBAGENT_UNSEEN, "")
        )
        run_markers = {
            str(row.conversation_id): "" for row in row_tuple if row.conversation_id
        }
        live_conversation_ids: set[str] = set()
        store = self._console_chat_store
        controller = self._console_chat_controller
        if store is not None and controller is not None:
            for session in store.sessions():
                conversation_id = str(session.persisted_conversation_id or "").strip()
                if not conversation_id:
                    continue
                live_conversation_ids.add(conversation_id)
                marker = self._console_run_marker_with_unseen(
                    controller, session, unseen_ids
                )
                run_markers[conversation_id] = resolve_glyph(
                    CONSOLE_RUN_MARKER_GLYPHS.get(marker, "")
                )
        for conversation_id in unseen_ids:
            conversation_id = str(conversation_id)
            if conversation_id not in live_conversation_ids:
                run_markers[conversation_id] = unseen_marker
        return overlay_console_conversation_markers(
            row_tuple,
            starred_ids=self._starred_console_conversation_ids(),
            selected_conversation_id=(
                current_conversation_id or self._current_console_conversation_id()
            ),
            run_markers=run_markers,
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

    def _membership_console_browser_rows(
        self,
        current_conversation_id: str | None = None,
    ) -> list[ConsoleConversationBrowserInputRow]:
        """Return conversation membership rows across every local workspace.

        This complete registry scan is reserved for the explicit Ctrl+K
        switcher action. Ordinary Workspaces rendering remains page-backed and
        never calls it while composing or reconciling the rail.
        """

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
                    "Unable to list Console switcher workspace conversations "
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
                    row_key=(
                        f"workspace:{workspace_id}:conversation:{conversation_id}"
                    ),
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

    async def _load_flat_conversation_history_rows(
        self,
        current_conversation_id: str | None = None,
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        """Load complete Default/unassigned history for one Ctrl+K open."""

        rows: tuple[ConsoleConversationBrowserInputRow, ...] = ()
        for scope in (
            ("global", None),
            ("workspace", DEFAULT_WORKSPACE_ID),
        ):
            offset = 0
            while True:
                page, total, error = await self._persisted_console_browser_rows(
                    current_conversation_id=current_conversation_id,
                    scopes=(scope,),
                    offset=offset,
                )
                if error:
                    break
                rows = self._merge_console_browser_rows(rows, page)
                offset += CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT
                if not page or total is None or offset >= total:
                    break
        return rows

    async def console_session_switcher_rows(
        self,
    ) -> tuple[ConsoleConversationBrowserInputRow, ...]:
        """Return complete history plus live rows for the Ctrl+K switcher.

        The complete membership scan happens only on this explicit action;
        ordinary rail projection stays bounded to expanded/search/page state.
        """

        current_conversation_id = self._current_console_conversation_id()
        native_rows = self._native_console_browser_rows(current_conversation_id)
        membership_rows = self._membership_console_browser_rows(current_conversation_id)
        persisted_rows = await self._load_flat_conversation_history_rows(
            current_conversation_id
        )
        # Resolve ownership from weakest to strongest evidence. Persisted flat
        # history supersedes stale session observations after a move out of a
        # named workspace; complete registry membership remains authoritative
        # when both services still expose the same conversation temporarily.
        self._record_canonical_owner_rows(native_rows)
        self._record_canonical_owner_rows(persisted_rows)
        self._record_canonical_owner_rows(membership_rows)
        named_rows = self._merge_console_browser_rows(
            self._workspace_tree_search.rows,
            self._workspace_tree_search.settled_rows,
            *(attempt.rows for attempt in self._workspace_page_attempts.values()),
            *self._workspace_membership_rows.values(),
        )
        rows = self._merge_console_browser_rows(
            native_rows,
            membership_rows,
            named_rows,
            persisted_rows,
        )
        rows = self._rows_with_latest_canonical_owner(rows)
        return self._overlay_current_console_browser_markers(
            rows,
            current_conversation_id=current_conversation_id,
        )

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

    async def _persisted_console_browser_rows(
        self,
        query: str = "",
        current_conversation_id: str | None = None,
        *,
        scopes: tuple[tuple[str, str | None], ...] | None = None,
        offset: int = 0,
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
        # The flat Conversations lane owns only unassigned/global and Default
        # records. Named-workspace search and pages have separate service calls.
        query_scopes = scopes or (
            ("global", None),
            ("workspace", DEFAULT_WORKSPACE_ID),
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
            for scope_type, workspace_id in query_scopes:
                list_kwargs: dict[str, Any] = {
                    "query": query,
                    "scope_type": scope_type,
                    "workspace_id": workspace_id,
                    "limit": CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
                    "offset": max(0, int(offset)),
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
                        "Conversation search is unavailable.",
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
                    partial(
                        self._refresh_console_persisted_rows_cache,
                        query,
                        current_conversation_id=current_conversation_id,
                        refresh_key=refresh_key,
                    ),
                    # task-24460: `run_worker` derives the worker name from
                    # `getattr(work, "__name__", "")`, and a `functools.partial`
                    # has no `__name__` -- so wrapping this call in a partial
                    # silently renamed the worker to "". That broke its
                    # boot-census allowlist row (which still reads
                    # `_refresh_console_persisted_rows_cache`) and made the
                    # worker anonymous in every worker diagnostic. Name it
                    # explicitly so the partial cannot erase it again.
                    name="_refresh_console_persisted_rows_cache",
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
        membership_revision = self._canonical_membership_revision
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
        if membership_revision != self._canonical_membership_revision:
            if (
                refresh_key is not None
                and self._console_persisted_rows_refresh_key == refresh_key
            ):
                self._console_persisted_rows_refresh_key = None
                self._sync_console_workspace_context()
            return result
        rows, _total, error = result
        if not error:
            self._record_canonical_owner_rows(rows)
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
        native_rows = self._native_console_browser_rows(current_conversation_id)
        self._record_canonical_owner_rows(native_rows)
        local_rows = self._merge_console_browser_rows(
            native_rows,
            *self._workspace_membership_rows.values(),
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
        request_key = (
            "conversations",
            query,
            token,
            self._flat_conversation_owner_token(),
            self._screen_lifecycle_token(),
            self._canonical_membership_revision,
        )
        scheduled_attempt = self._flat_conversation_search.request_key == request_key
        settled_rows = (
            self._flat_conversation_search.settled_rows
            if scheduled_attempt
            else self._console_conversation_browser_rows
        )
        settled_total = (
            self._flat_conversation_search.settled_total
            if scheduled_attempt
            else self._console_conversation_browser_total
        )
        self._flat_conversation_search.request_key = request_key
        if not str(query or "").strip():
            self._console_conversation_browser_rows = ()
            self._console_conversation_browser_total = None
            self._console_conversation_browser_error = ""
            self._sync_console_workspace_context()
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
            return

        local_rows = tuple(
            row
            for row in self._filter_console_browser_rows_for_query(
                self._merge_console_browser_rows(
                    self._native_console_browser_rows(),
                    *self._workspace_membership_rows.values(),
                ),
                query,
            )
            if self._row_belongs_to_flat_projection(row)
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
        if not self._flat_search_attempt_is_current(request_key):
            return
        if error_copy:
            restored_rows = settled_rows or local_rows
            self._console_conversation_browser_rows = restored_rows
            self._console_conversation_browser_total = (
                settled_total if settled_total is not None else len(restored_rows)
            )
            self._console_conversation_browser_error = error_copy
            self._flat_conversation_search.retry_query = query
            self._sync_console_workspace_context()
            self.call_after_refresh(self._focus_console_workspace_conversation_search)
            return
        merged = self._merge_console_browser_rows(
            local_rows,
            (
                row
                for row in persisted_rows
                if self._row_belongs_to_flat_projection(row)
            ),
        )
        result_total = persisted_total
        if result_total is None or result_total < len(merged):
            result_total = len(merged)
        self._console_conversation_browser_rows = merged
        self._record_canonical_owner_rows(merged)
        self._console_conversation_browser_total = result_total
        self._console_conversation_browser_error = error_copy
        self._flat_conversation_search.cache = {query: (merged, result_total)}
        self._flat_conversation_search.settled_rows = merged
        self._flat_conversation_search.settled_total = result_total
        self._flat_conversation_search.settled_query = query
        self._flat_conversation_search.retry_query = query if error_copy else None
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
        projection_query = (
            self._flat_conversation_search.settled_query
            if self._flat_conversation_search.error
            else query
        )
        rows, total, error_copy = self._current_console_browser_rows(
            projection_query,
            current_conversation_id=current_conversation_id,
        )
        active_workspace_id = str(
            self._current_console_workspace_context().active_workspace_id or ""
        ).strip()
        active_workspace_label = str(
            legacy_state.workspace_label.removeprefix("Workspace: ") or "Chats"
        )
        legacy_membership_rows = tuple(
            ConsoleConversationBrowserInputRow(
                row_key=row.conversation_id,
                conversation_id=row.conversation_id,
                native_session_id=None,
                title=row.title,
                scope_type="workspace",
                workspace_id=active_workspace_id or DEFAULT_WORKSPACE_ID,
                workspace_label=active_workspace_label,
                status=row.status,
                selected=row.selected,
                source_kind="persisted",
            )
            for row in legacy_state.conversation_rows
        )
        rows = self._merge_console_browser_rows(
            rows,
            (
                row
                for row in legacy_membership_rows
                if self._row_belongs_to_flat_projection(row)
            ),
        )
        workspace_rows = (
            self._workspace_tree_search.settled_rows
            if self._workspace_tree_search.error
            else self._workspace_tree_search.rows
        )
        materialized_ids = {
            str(row.conversation_id)
            for group in (
                rows,
                legacy_membership_rows,
                workspace_rows,
                *(attempt.rows for attempt in self._workspace_page_attempts.values()),
                *self._workspace_membership_rows.values(),
            )
            for row in group
            if row.conversation_id
        }
        for conversation_id in tuple(self._canonical_owner_observations):
            if conversation_id not in materialized_ids:
                self._canonical_owner_observations.pop(conversation_id, None)
        canonical_rows = (*rows, *legacy_membership_rows, *workspace_rows)
        canonical_memberships: dict[str, list[str]] = {}
        canonical_labels: dict[str, str] = {}
        for row in canonical_rows:
            conversation_id = str(row.conversation_id or "").strip()
            if not conversation_id:
                continue
            workspace_id = str(row.workspace_id or "").strip()
            owner_id = (
                DEFAULT_WORKSPACE_ID
                if row.scope_type == "global"
                or workspace_id in ("", DEFAULT_WORKSPACE_ID)
                else workspace_id
            )
            canonical_memberships.setdefault(owner_id, []).append(conversation_id)
            canonical_labels.setdefault(owner_id, str(row.workspace_label or owner_id))
        for conversation_id, owner_id in self._canonical_owner_observations.items():
            canonical_memberships.setdefault(owner_id, []).append(conversation_id)
            canonical_labels.setdefault(
                owner_id,
                "Default" if owner_id == DEFAULT_WORKSPACE_ID else owner_id,
            )
        if canonical_memberships:
            self.apply_workspace_membership_snapshot(
                {
                    workspace_id: tuple(conversation_ids)
                    for workspace_id, conversation_ids in canonical_memberships.items()
                },
                complete=False,
                workspace_labels=canonical_labels,
                canonical_rows=canonical_rows,
            )
        # TASK-22201: ONE canonical-owner + overlay pass per build. The
        # browser rows and the workspace tree's no-query source (browser
        # rows + surviving page-attempt rows) used to run this pipeline
        # separately -- twice per build, up to six times per run tick. Both
        # passes are per-row (the canonical filter and every overlay marker
        # depend only on the row itself plus controller/store state), so one
        # pass over the merged union, partitioned back out by display
        # identity, is exactly equivalent. Stale page attempts are pruned
        # FIRST -- the projection used to do that before its own merge, and
        # a just-deleted workspace's rows must not ride in via the union.
        self._prune_stale_workspace_page_attempts()
        browser_identities = {
            self._console_browser_display_identity(row) for row in rows
        }
        union_rows = self._merge_console_browser_rows(
            rows,
            *(attempt.rows for attempt in self._workspace_page_attempts.values()),
        )
        union_rows = self._rows_with_latest_canonical_owner(union_rows)
        union_rows = self._overlay_current_console_browser_markers(
            union_rows, current_conversation_id
        )
        rows = tuple(
            row
            for row in union_rows
            if self._console_browser_display_identity(row) in browser_identities
        )
        if not projection_query.strip() and not self._flat_conversation_search.error:
            ordinary_rows = tuple(
                row for row in rows if self._row_belongs_to_flat_projection(row)
            )
            self._flat_conversation_search.settled_rows = ordinary_rows
            self._flat_conversation_search.settled_total = total
            self._flat_conversation_search.settled_query = ""
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
            query=projection_query,
            marks_available=marks_service is not None,
            error_copy=error_copy or self._console_conversation_browser_error,
            result_total_count=total,
            result_limit=CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
            subagent_counts=subagent_counts,
            # The visible-row cap grows with the measured rail body height so
            # the Chats section expands to fill its even share of the rail
            # alongside the Workspaces tree; the historical 12-row default
            # still floors it on short/unmeasured rails.
            group_row_limit=console_conversation_browser_group_row_limit(
                self._console_rail_body_height()
            ),
        )
        return replace(
            state,
            conversation_browser=browser,
            conversation_section=legacy_state.conversation_section,
            workspace_tree=self.workspace_tree_projection(
                rows, prepared_rows=union_rows
            ),
            workspace_query=self._workspace_tree_search.query,
            workspace_loading=self._workspace_tree_search.request_key is not None,
            workspace_error=str(self._workspace_tree_search.error or ""),
            workspace_retry_available=(
                self._workspace_tree_search.retry_query is not None
            ),
            workspace_marks_available=marks_service is not None,
        )

    # -- Workspace policy context -------------------------------------------

    #: Memoized active-workspace resolution for the per-keystroke Console
    #: context read: (registry service instance, its ``mutation_generation``
    #: at read time, resolved workspace id), or ``None`` before the first
    #: cacheable read. A CLASS attribute default, matching the screen's
    #: memo conventions, so hand-built fixtures that skip ``__init__``
    #: still read a defined value.
    _console_workspace_id_memo: "tuple[Any, int, str] | None" = None

    def _resolve_console_active_workspace_id(self) -> str:
        """Resolve the active workspace id read-only, memoized per screen.

        TASK-21118: this sits on the per-keystroke path (DraftChanged ->
        control-state build -> provider selection ->
        ``_current_console_workspace_context``), which used to call
        ``ensure_default_workspace`` ~1.25x per key -- a synchronous SQLite
        read on the UI thread, plus that method's repair side-effects (a
        probing SELECT and, with stale Default bindings, a DELETE write
        transaction). Two changes here:

        * READ-ONLY: the keystroke path now calls ``get_active_workspace``
          only. The ensure/repair behavior lives at session-start and
          workspace-switch seams instead (app wiring's
          ``ensure_default_workspace``, ``set_active_workspace``'s
          switch-to-Default repair, ``_set_active_workspace_for_console_
          session``'s global branch, ``_console_browser_workspace_records``,
          and ``archive_workspace``). A missing active workspace is floored
          to ``DEFAULT_WORKSPACE_ID`` in memory -- the same id ``ensure_
          default_workspace`` returned for that state -- without writing.
        * MEMOIZED: the resolution is served from a memo revalidated
          against the registry's in-memory ``mutation_generation`` (bumped
          by every workspace-record mutator, from any screen), so a warm
          keystroke performs zero DB round-trips while a workspace change
          anywhere -- Console switcher, browser row, session switch,
          Settings "Set active", Library create, archive -- invalidates it
          on the very next read. The generation must be a real ``int``
          before anything is cached: a MagicMock double's auto-attribute
          compares equal to itself forever and would freeze the memo, so
          doubles without an integer generation stay on live reads.

        This memo COMPOSES with the task-15452 per-pass derivation memo
        rather than replacing it: that memo dedupes the provider-selection
        legs within one synchronous pass, and each pass's single remaining
        context read is what this cross-pass memo serves.
        """
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            return CONSOLE_GLOBAL_WORKSPACE_ID
        generation = getattr(registry_service, "mutation_generation", None)
        if isinstance(generation, bool) or not isinstance(generation, int):
            generation = None
        # getattr, not a bare read: `_current_console_workspace_context` is
        # deliberately callable unbound on duck-typed screen stand-ins (see
        # test_console_staged_evidence_strip's bundleless-launch pin), which
        # lack this class-attribute default.
        memo = getattr(self, "_console_workspace_id_memo", None)
        if (
            memo is not None
            and generation is not None
            and memo[0] is registry_service
            and memo[1] == generation
        ):
            return memo[2]
        workspace_id = CONSOLE_GLOBAL_WORKSPACE_ID
        try:
            get_active_workspace = getattr(
                registry_service, "get_active_workspace", None
            )
            if callable(get_active_workspace):
                active_workspace = get_active_workspace()
                if active_workspace is None:
                    # Read-only floor: `ensure_default_workspace` used to
                    # create/activate the built-in Default here and return
                    # it. Boot wiring and every switch seam keep the
                    # registry resting on an active workspace, so this
                    # only covers the moments in between -- same id,
                    # minus the write.
                    if generation is not None:
                        self._console_workspace_id_memo = (
                            registry_service,
                            generation,
                            DEFAULT_WORKSPACE_ID,
                        )
                    return DEFAULT_WORKSPACE_ID
            else:
                # Reduced test doubles supply only `ensure_default_workspace`;
                # production always has the read-only accessor above.
                ensure_default_workspace = getattr(
                    registry_service, "ensure_default_workspace", None
                )
                if not callable(ensure_default_workspace):
                    return workspace_id
                active_workspace = ensure_default_workspace()
            candidate = getattr(active_workspace, "workspace_id", None)
            if candidate:
                workspace_id = str(candidate)
        except Exception:
            logger.debug("Console workspace registry was unavailable for send context")
            return workspace_id
        if generation is not None:
            self._console_workspace_id_memo = (
                registry_service,
                generation,
                workspace_id,
            )
        return workspace_id

    def _current_console_workspace_context(self) -> ConsoleWorkspaceContext:
        """Return explicit workspace policy context for native Console sends.

        Runs on every printable keystroke; the active-workspace resolution
        is memoized and read-only (see
        ``_resolve_console_active_workspace_id``), and the staged-launch
        evidence bundle below is parsed at most once per launch
        (``evidence_bundle_from_launch`` caches on the launch object).

        Resolution goes through the class, not ``self``: this method is
        callable unbound on duck-typed screen stand-ins exposing only
        ``app_instance`` and ``_pending_console_launch_context``, and both
        legs preserve that contract.
        """
        workspace_id = ConsoleWorkspaceController._resolve_console_active_workspace_id(
            self
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
            self._switch_console_workspace(workspace_id)

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
        """Open the shared create dialog (spec 2026-08-17 §4.3)."""
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            self.app_instance.notify(
                "Workspace service is not ready.", severity="warning"
            )
            return
        from tldw_chatbook.Widgets.workspace_create_modal import WorkspaceCreateModal

        self.push_screen(
            WorkspaceCreateModal(
                registry_service=registry_service,
                description="Local workspace created from Console.",
            ),
            self._handle_workspace_create_result,
        )

    def _handle_workspace_create_result(
        self, result: WorkspaceCreateResult | None
    ) -> None:
        """Console-side post-create sync; the modal already created/bound."""
        if result is None:
            return
        if result.offer_profile_interview:
            from ...Personal_Context.interview_launch import (
                launch_workspace_profile_interview_after_commit,
            )

            launch_workspace_profile_interview_after_commit(
                self.app_instance,
                workspace_id=result.workspace_id,
                workspace_label=result.name,
                continuation=lambda: (
                    ConsoleWorkspaceController._continue_workspace_create_result(
                        self, result
                    )
                ),
            )
            return
        ConsoleWorkspaceController._continue_workspace_create_result(self, result)

    def _continue_workspace_create_result(self, result: WorkspaceCreateResult) -> None:
        """Run the established Console sync after the optional interview."""

        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            return
        for _folder, message in result.failed_folders:
            self.app_instance.notify(message, severity="warning")
        if not result.make_active:
            self._sync_console_workspace_context()
            self.app_instance.notify(f"Created {result.name}.", severity="information")
            if result.project_skills:
                maybe_offer_project_skills_import(
                    self.app_instance, result.project_skills
                )
            return
        try:
            registry_service.set_active_workspace(result.workspace_id)
        except WorkspaceRegistryServiceError:
            logger.opt(exception=True).warning(
                "Unable to activate new Console workspace"
            )
            self.app_instance.notify(
                "Workspace created but could not be activated.", severity="error"
            )
            if result.project_skills:
                maybe_offer_project_skills_import(
                    self.app_instance, result.project_skills
                )
            return
        self._sync_console_chat_core_state()
        self._activate_console_session_for_workspace(result.workspace_id)
        self._sync_console_workspace_context()
        self.run_worker(
            self._sync_native_console_chat_ui(), exclusive=True, group="console-sync"
        )
        # TASK-713: the whole sequence is invisible when the Workspace status
        # row is scrolled out of view -- keep the toast even with the modal.
        self.app_instance.notify(
            f"Created {result.name} and switched Console to it.",
            severity="information",
        )
        if result.project_skills:
            maybe_offer_project_skills_import(self.app_instance, result.project_skills)

    async def open_console_workspace_conversation(
        self,
        conversation_id: str,
        *,
        row_key: str = "",
        target_workspace_id: str | None = None,
    ) -> bool | None:
        """Open one saved conversation for both the flat browser and Tree."""

        conversation_id = str(conversation_id or "").strip()
        explicit_row_key = str(row_key or "").strip()
        browser_row = (
            self._find_console_browser_row(
                explicit_row_key,
                conversation_id=conversation_id,
            )
            if explicit_row_key
            else None
        )
        prior_browser_workspace_id: str | None = None
        if browser_row is not None:
            prior_browser_workspace_id = (
                self._active_console_workspace_id_for_conversation_search()
                or None
            )
            row_conversation_id = str(browser_row.conversation_id or "").strip()
            session_id = self._session_id_for_browser_row_fn(browser_row)
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
                return False
            self._set_conversation_row_loading_fn(row_conversation_id, True)
            try:
                resumed = await self._resume_console_workspace_conversation(
                    row_conversation_id,
                    target_scope_type=(
                        browser_row.scope_type
                        if browser_row is not None
                        else ("workspace" if target_workspace_id else None)
                    ),
                    target_workspace_id=(
                        browser_row.workspace_id
                        if browser_row is not None
                        else target_workspace_id
                    ),
                )
            finally:
                try:
                    self._set_conversation_row_loading_fn(
                        row_conversation_id, False
                    )
                except BaseException:
                    logger.opt(exception=True).warning(
                        "Unable to clear Console conversation-row loading state"
                    )
            if resumed:
                if browser_row is not None:
                    self._activate_console_workspace_for_browser_row(
                        browser_row,
                        previous_workspace_id=prior_browser_workspace_id,
                    )
                return True
            if resumed is None:
                return None
            self._mark_conversation_row_broken_fn(row_conversation_id)
            self.app_instance.notify(
                CONSOLE_SAVED_CONVERSATION_RESUME_FAILURE_COPY,
                severity="warning",
                timeout=15,
            )
            return False
        controller = self._ensure_chat_controller_fn()
        store = controller.store
        prior_active_session_id = store.active_session_id
        try:
            if prior_active_session_id != session_id:
                self._capture_console_draft_switch_snapshot()
                controller.switch_session(session_id)
            self._set_active_workspace_for_console_session(session_id)
            self._sync_console_chat_core_state()
            sync_result = self._sync_native_console_chat_ui_fn()
            if inspect.isawaitable(sync_result):
                await sync_result
            self._sync_temporary_chip_fn()
            self._focus_composer_if_needed_fn(force=True)
            await self._refresh_console_conversation_browser_after_selection()
        except asyncio.CancelledError:
            await self._restore_console_session_after_failed_open(
                store, prior_active_session_id
            )
            raise
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to present an already-open Console saved conversation"
            )
            await self._restore_console_session_after_failed_open(
                store, prior_active_session_id
            )
            self.app_instance.notify(
                CONSOLE_SAVED_CONVERSATION_RESUME_FAILURE_COPY,
                severity="error",
                timeout=15,
            )
            return None
        if browser_row is not None:
            self._activate_console_workspace_for_browser_row(
                browser_row,
                previous_workspace_id=prior_browser_workspace_id,
            )
        return True

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
        defaults = self._blank_console_session_settings()
        session = store.create_session(
            title=self._console_workspace_session_title(target_workspace_id),
            workspace_id=target_workspace_id,
            settings=defaults,
            canonical_settings_baseline=defaults,
        )
        session.new_chat_default_generation = (
            self._console_new_chat_default_generation()
        )
        # task-7 review: `create_session` activates the new (never
        # ephemeral -- no `ephemeral=` passed) session inline; same
        # staleness risk as the switch branch above if the workspace's
        # previous session was temporary.
        self._sync_console_temporary_chip()

    def _reconcile_console_session_with_registry(self) -> None:
        """Align the Console session with the registry's active workspace.

        TASK-18310: every IN-Console workspace-activation path -- the Alt+W
        switcher (``_open_console_workspace_switcher``'s ``_switch_to``), the
        shared create modal's Console handler
        (``_handle_workspace_create_result``), and conversation-browser
        row-open (``_activate_console_workspace_for_browser_row`` plus the
        session lookup around it) -- calls ``set_active_workspace`` on the
        registry AND ``_activate_console_session_for_workspace`` on this
        controller together, so the registry and the Console chat store's
        active session can never drift apart from any of those paths.
        Cross-screen activation -- Settings' create-modal ``_done``,
        Library's ``create_local_workspace`` ``_done``, and Settings' "Set
        active" button (Qodo finding 5 on PR #1809) -- only calls
        ``set_active_workspace`` on the registry, so registry/session
        misalignment can arise EXCLUSIVELY from a cross-screen change.

        Called from ``ChatScreen.on_screen_resume`` on every Console resume
        (including the mount's own -- the store is app-level and can carry
        a session that predates the first mount). When already aligned --
        the overwhelmingly common case, since every in-Console path keeps
        them in lockstep -- this is an O(1) early exit. Only on an actual
        cross-screen divergence does it re-run the same Console-side
        activation sequence the create handler uses, minus the registry
        write (already done by the other surface) and the toast (the
        originating surface already announced the switch).

        Deliberately conservative: any read of the registry is guarded, and
        a ``None`` registry or ``None`` active workspace returns quietly --
        this must never break screen resume. A store with no active session is
        different: TASK-2033 requires its first session to inherit the
        registry-active workspace instead of falling through to Default.

        Comparison is normalized, not a bare ``==``: a session's default
        ``workspace_id`` is the "no explicit workspace" sentinel
        (``CONSOLE_GLOBAL_WORKSPACE_ID``, `""` is treated the same), while
        the registry represents that identical state as its built-in
        Default workspace row (``DEFAULT_WORKSPACE_ID`` --
        ``ensure_default_workspace`` floors every context read to it). Per
        the task-15120 owner ruling (see
        ``_set_active_workspace_for_console_session``), those two spellings
        are THE SAME state on two layers, not a divergence -- comparing
        them raw here misfired on every plain/global mounted session
        whenever the registry's active workspace was Default (its ordinary
        resting state), tearing down a perfectly aligned session and
        replacing it with a fresh one on every resume. Caught by two
        existing tests going from GREEN to RED with the naive comparison
        (`test_mounted_first_chat_ack_exception_during_resume_restores_ui`,
        `test_mounted_console_unmount_times_out_hung_refresh_and_repairs_on_resume`).
        """
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        if registry_service is None:
            return
        try:
            active = registry_service.get_active_workspace()
        except Exception:
            logger.opt(exception=True).debug(
                "Unable to read active workspace during Console resume reconcile"
            )
            return
        if active is None:
            return
        store = self._ensure_console_chat_store()
        active_session = None
        stale_active_session_id = False
        if store.active_session_id is not None:
            # O(1) active-session lookup (Qodo, PR #1880): with an id present,
            # ensure_session() is a pure dict hit. Do not call it for a fresh
            # store: that would create a Default/global session before the
            # registry-active workspace is applied, reproducing TASK-2033 and
            # leaving an unnecessary empty tab behind.
            try:
                active_session = store.ensure_session()
            except KeyError:
                # A stale id is divergent and follows the normal repair path.
                active_session = None
                stale_active_session_id = True
        if active_session is not None and _normalized_console_workspace_id(
            active_session.workspace_id
        ) == _normalized_console_workspace_id(active.workspace_id):
            return
        if stale_active_session_id:
            # Core sync reads the active session. Repair the invalid identity
            # first so that read cannot abort the resume-time reconciliation.
            self._activate_console_session_for_workspace(active.workspace_id)
            self._sync_console_chat_core_state()
        else:
            self._sync_console_chat_core_state()
            self._activate_console_session_for_workspace(active.workspace_id)
        self._sync_console_workspace_context()
        self.run_worker(
            self._sync_native_console_chat_ui(), exclusive=True, group="console-sync"
        )

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
                # Default (established here and at the other ensure seams,
                # deliberately -- capability-less, safe; the read-only
                # context resolution floors to the same id, TASK-21118), so
                # a global conversation lands there, not on bare None.
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
        active_session = next(
            (
                session
                for session in store.sessions()
                if session.id == store.active_session_id
            ),
            None,
        )
        if (
            active_session is not None
            and str(active_session.persisted_conversation_id or "") == target
        ):
            return active_session.id
        for session in store.sessions():
            if str(session.persisted_conversation_id or "") == target:
                return session.id
        return None

    async def _restore_console_session_after_failed_open(
        self,
        store: Any,
        prior_active_session_id: str | None,
    ) -> None:
        """Best-effort repaint of the exact session active before an open."""
        if not any(
            session.id == prior_active_session_id for session in store.sessions()
        ):
            return
        try:
            store.switch_session(prior_active_session_id)
            self._set_active_workspace_for_console_session(prior_active_session_id)
            self._sync_console_chat_core_state()
            sync_result = self._sync_native_console_chat_ui_fn()
            if inspect.isawaitable(sync_result):
                await sync_result
            self._sync_temporary_chip_fn()
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to repaint the prior Console session after saved-chat open"
            )
        finally:
            try:
                self._focus_composer_if_needed_fn(force=True)
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to focus the prior Console composer after saved-chat open"
                )

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
        store = self._ensure_console_chat_store()
        prior_active_session_id = store.active_session_id
        # TASK-339: keystrokes typed while the conversation tree loads
        # belong to the resumed session — snapshot the composer now.
        self._capture_console_draft_switch_snapshot()
        # task-15860 Task 6: the tree load and the session build moved to
        # `Chat/console_conversation_hydration.py` -- the launch wake has to
        # hydrate a conversation with no screen in existence, and one policy
        # beats two. Everything BELOW the hydration call is this screen's own
        # work (marker overlay, scope warm, repaint, focus) and stays here;
        # so do both failure toasts, because the UX
        # for each failure is a view concern.
        try:
            tree = await load_console_conversation_tree(self.app_instance, target)
        except ConversationServiceUnavailable:
            await self._restore_console_session_after_failed_open(
                store, prior_active_session_id
            )
            self.app_instance.notify(
                "Saved conversation resume is unavailable in this build.",
                severity="warning",
            )
            return None
        except ConversationLoadFailed:
            logger.exception(
                f"Unable to resume Console saved conversation: conversation_id={target}"
            )
            await self._restore_console_session_after_failed_open(
                store, prior_active_session_id
            )
            self.app_instance.notify(
                CONSOLE_SAVED_CONVERSATION_RESUME_FAILURE_COPY,
                severity="error",
                timeout=15,
            )
            return None

        if tree is None:
            # TASK-717: missing record - the caller owns this failure's UX
            # (honest toast + marking the row visibly broken), so do not
            # stack a second notification here.
            await self._restore_console_session_after_failed_open(
                store, prior_active_session_id
            )
            return False

        conversation = tree.get("conversation")
        if not isinstance(conversation, dict):
            conversation = {}
        session = None
        hydration = self._console_session_settings_for_resume(conversation)
        if not isinstance(hydration, ConsoleGenerationSettingsHydration):
            raise TypeError(
                "Console resume settings must be ConsoleGenerationSettingsHydration"
            )
        try:
            session = await hydrate_console_session(
                app=self.app_instance,
                store=store,
                conversation_id=target,
                tree=tree,
                settings=hydration.settings,
                generation_durable_snapshot=hydration.durable_snapshot,
                generation_metadata_status=hydration.metadata_status,
                target_scope_type=target_scope_type,
                target_workspace_id=target_workspace_id,
                activate=False,
            )
            # Re-derive display-only agent TOOL markers from AgentRunsDB and
            # overlay them onto the restored active-path view.
            store.apply_resume_marker_overlay(
                session.id,
                self._inject_resume_agent_markers(
                    store.messages_for_session(session.id), target
                ),
            )
            # Warm the effective conversation/workspace scope before the final
            # activation commit so any failure leaves the prior session active.
            await self._resolve_console_effective_scope_state(session)
            store.switch_session(session.id)
            self._set_active_workspace_for_console_session(session.id)
            self._sync_console_retrieval_scope_row()
            self._console_agent_drilldown_run_id = None
            self._note_console_follow_intent()
            self._sync_console_chat_core_state()
            await self._sync_native_console_chat_ui()
            await self._refresh_console_conversation_browser_after_selection()
            self._focus_console_composer_if_needed(force=True)
            if callable(self._wake_retry_poke_fn):
                self._wake_retry_poke_fn()
        except asyncio.CancelledError:
            if session is not None:
                store.rollback_restored_session(
                    session.id,
                    expected_session=session,
                    prior_active_session_id=prior_active_session_id,
                )
            await self._restore_console_session_after_failed_open(
                store, prior_active_session_id
            )
            raise
        except Exception:
            logger.opt(exception=True).warning(
                "Unable to present Console saved conversation"
            )
            if session is not None:
                store.rollback_restored_session(
                    session.id,
                    expected_session=session,
                    prior_active_session_id=prior_active_session_id,
                )
            await self._restore_console_session_after_failed_open(
                store, prior_active_session_id
            )
            self.app_instance.notify(
                CONSOLE_SAVED_CONVERSATION_RESUME_FAILURE_COPY,
                severity="error",
                timeout=15,
            )
            return None
        return True

    # -- Workspace context state / grouped conversation rows -----------------

    #: The open run tick's shared build cache, or ``None`` outside a tick
    #: (TASK-22201). A CLASS attribute default, matching the screen's memo
    #: conventions, so hand-built fixtures that skip ``__init__`` still
    #: read a defined value.
    _console_tick_builds: "ConsoleTickWorkspaceBuilds | None" = None

    @contextmanager
    def tick_workspace_build_scope(self):
        """Share ONE fingerprint-validated context build across a run tick.

        Opened by ``_sync_native_console_chat_ui`` around its sync body
        (TASK-22201): every ``_build_console_workspace_context_state`` call
        the tick's own asyncio task performs -- directly or through the
        inspector/control-bar/agent-section legs -- is served from one
        :class:`ConsoleTickWorkspaceBuilds` cache. Deliberately opt-in and
        scoped, like the screen's ``_console_derivation_scope``
        (task-15452): outside a ``with`` block, and for any OTHER task
        interleaving during the tick's awaits, every build is live exactly
        as before. Re-entrant (an inner scope keeps the outer cache) and
        always torn down, so a raising tick cannot leave a stale build
        cached for the next one.
        """
        if getattr(self, "_console_tick_builds", None) is not None:
            yield
            return
        self._console_tick_builds = ConsoleTickWorkspaceBuilds(self)
        try:
            yield
        finally:
            self._console_tick_builds = None

    def _build_console_workspace_context_state(self) -> ConsoleWorkspaceContextState:
        builds = getattr(self, "_console_tick_builds", None)
        if builds is not None and builds.accepts_current_task():
            return builds.state()
        current_conversation = self._current_console_conversation_id()
        # The generation-keyed display-read view stands in for the raw
        # service (TASK-22201): the builder's read set (active workspace,
        # workspaces, runtime bindings, memberships) is served without SQL
        # while the registry is unchanged. ADR-028's from-disk binding
        # status recompute still runs per build inside the builder.
        state = build_console_workspace_state(
            registry_service=self._console_registry_reads_view(),
            current_conversation=current_conversation,
            conversations=(),
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
        state = self._with_console_conversation_browser_state(
            state,
            current_conversation_id=current_conversation,
        )
        registry = getattr(self.app_instance, "workspace_registry_service", None)
        availability: dict[str, bool] = {}
        if registry is not None:
            for workspace in self._console_browser_workspace_records():
                try:
                    availability[workspace.workspace_id] = any(
                        binding.binding_kind is RuntimeBindingKind.LOCAL_FILESYSTEM
                        and binding.status is RuntimeBindingStatus.READY
                        for binding in registry.list_folder_bindings(workspace.workspace_id)
                    )
                except Exception:
                    availability[workspace.workspace_id] = False
        return replace(state, workspace_files_available_by_id=availability)

    def _console_workspace_build_fingerprint(self) -> tuple | None:
        """Cheap change token over the context build's volatile inputs.

        Serves the run tick's build cache (TASK-22201): identical
        fingerprints between two reads WITHIN ONE TICK mean the cached
        build may be reused. Covered inputs -- registry identity +
        ``mutation_generation``, current conversation, store sessions
        (identity, title, timestamps, persistence, workspace) + active
        session, run status + per-session queued counts, canonical
        membership revision, persisted-rows cache token, both search
        lanes' generations/queries/errors, and the page-attempt shape --
        are exactly the ones the PR #660 / task-280 freshness rulings care
        about across the tick's awaits (session create/activate/persist).
        Deliberately NOT exhaustive: long-tail inputs (stars, unseen
        markers, collapse preferences) change only through paths that
        rebuild and push OUTSIDE the tick cache's lifetime, so a miss is
        bounded by one 0.2 s tick. Returns ``None`` (never reuse) when the
        registry lacks a real ``int`` generation or any component read
        fails.
        """
        try:
            registry_service = getattr(
                self.app_instance, "workspace_registry_service", None
            )
            generation = getattr(registry_service, "mutation_generation", None)
            if registry_service is not None and (
                isinstance(generation, bool) or not isinstance(generation, int)
            ):
                return None
            store = self._console_chat_store
            controller = self._console_chat_controller
            sessions_token: tuple = ()
            active_session_id = None
            queued_token: tuple = ()
            if store is not None:
                active_session_id = store.active_session_id
                sessions = tuple(store.sessions())
                sessions_token = tuple(
                    (
                        session.id,
                        str(session.title or ""),
                        str(session.updated_at or ""),
                        str(session.persisted_conversation_id or ""),
                        str(session.workspace_id or ""),
                    )
                    for session in sessions
                )
                if controller is not None:
                    queued_token = tuple(
                        controller.activity_for(session.id).queued_count
                        for session in sessions
                    )
            run_status = controller.run_state.status if controller is not None else None
            flat_lane = self._flat_conversation_search
            workspace_lane = self._workspace_tree_search
            attempts_token = tuple(
                (
                    workspace_id,
                    attempt.generation,
                    len(attempt.rows),
                    attempt.loading,
                    attempt.error,
                    attempt.next_cursor,
                    attempt.retry_cursor,
                    attempt.membership_unknown,
                )
                for workspace_id, attempt in self._workspace_page_attempts.items()
            )
            return (
                id(registry_service),
                generation,
                self._current_console_conversation_id(),
                active_session_id,
                sessions_token,
                queued_token,
                run_status,
                self._canonical_membership_revision,
                self._console_persisted_rows_cache_token,
                self._console_conversation_browser_query,
                (
                    flat_lane.generation,
                    flat_lane.query,
                    flat_lane.error,
                    flat_lane.settled_query,
                    flat_lane.request_key,
                ),
                (
                    workspace_lane.generation,
                    workspace_lane.query,
                    workspace_lane.error,
                    workspace_lane.settled_query,
                    workspace_lane.request_key,
                ),
                attempts_token,
                getattr(self.app_instance, "workspace_server_adapter_state", None),
                getattr(self.app_instance, "workspace_acp_handoff_state", None),
            )
        except Exception:
            logger.debug(
                "Console workspace build fingerprint unavailable; building live"
            )
            return None

    @staticmethod
    def _console_workspace_row_key(row: ConsoleWorkspaceConversationRow) -> str:
        return str(row.conversation_id or "").strip()

    def _activate_console_workspace_for_browser_row(
        self,
        row: ConsoleConversationBrowserRow,
        *,
        previous_workspace_id: str | None = None,
    ) -> None:
        """Align workspace context and announce a committed browser-row open."""
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
            workspace_changed = (
                active_workspace is None
                or active_workspace.workspace_id != workspace_id
            )
            if workspace_changed:
                registry_service.set_active_workspace(workspace_id)
            if (
                previous_workspace_id != workspace_id
                if previous_workspace_id is not None
                else workspace_changed
            ):
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

    #: Generation-keyed display-read view over the app's registry service,
    #: or ``None`` before the first read (TASK-22201). A CLASS attribute
    #: default, matching the screen's memo conventions, so hand-built
    #: fixtures that skip ``__init__`` still read a defined value.
    _console_registry_display_reads: "_ConsoleRegistryDisplayReads | None" = None

    def _console_registry_reads_view(self) -> "_ConsoleRegistryDisplayReads | None":
        """Return the display-read view bound to the CURRENT service instance.

        Rebound (dropping its cache) whenever the app swaps its registry
        service, mirroring the identity check in the TASK-21118 memo.
        """
        service = getattr(self.app_instance, "workspace_registry_service", None)
        if service is None:
            return None
        view = getattr(self, "_console_registry_display_reads", None)
        if view is None or view.service is not service:
            view = _ConsoleRegistryDisplayReads(service)
            self._console_registry_display_reads = view
        return view

    def _console_browser_workspace_records(self) -> tuple[WorkspaceRecord, ...]:
        """Return all local workspace records visible to the Console browser.

        Served from the generation-keyed display-read view (TASK-22201):
        the run tick reaches this method many times per 0.2 s (browser
        labels, the workspace tree projection, page-row labels), and each
        call used to run ``ensure_default_workspace()`` -- a write-capable
        REPAIR (SELECT + bindings probe + occasional DELETE transaction) --
        plus ``list_workspaces()``, synchronously on the event loop. The
        repair does not belong on a display path and was never this path's
        responsibility alone: boot wiring (``app.py``
        ``_wire_workspace_registry_services``), ``archive_workspace``,
        ``set_active_workspace``'s switch-to-Default repair, and
        ``_set_active_workspace_for_console_session``'s global branch all
        keep the registry resting on an active workspace. Display reads a
        degraded registry as degraded and repairs nothing.
        """
        view = self._console_registry_reads_view()
        if view is None:
            return ()
        list_workspaces = getattr(view, "list_workspaces", None)
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
            # The flat Conversations lane is Default/unassigned scope and is
            # independent of whichever named workspace becomes active.
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

    # ---- Row commands from the conversation action menu (TASK-23200) ----

    def set_console_conversation_state(
        self,
        conversation_id: str,
        state: str,
        *,
        conversation_title: str = "",
    ) -> None:
        """Persist a new conversation state and confirm it to the user.

        "Archive" is not a separate flag: it is the ``resolved`` state, the
        same mapping tldw_server's Sync v2 alias table uses. The write goes
        through ``update_conversation``, which normalizes the value against
        ``_ALLOWED_CONVERSATION_STATES`` and rejects anything else.

        Args:
            conversation_id: Persisted conversation to update.
            state: A canonical conversation state.
            conversation_title: Title for the confirmation toast.
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            self.app_instance.notify(
                "Conversation storage is unavailable.", severity="error"
            )
            return

        label = conversation_title or "conversation"
        self.run_worker(
            self._set_console_conversation_state_off_loop(
                db, conversation_id, state, label
            ),
            group="console-conversation-state",
            exit_on_error=False,
        )

    async def _set_console_conversation_state_off_loop(
        self,
        db: Any,
        conversation_id: str,
        state: str,
        label: str,
    ) -> None:
        """Read the current version, write the state, and report the outcome."""
        import asyncio

        from ...Chat.console_conversation_actions import conversation_state_label

        def _write() -> str:
            current = db.get_conversation_by_id(conversation_id)
            if not current:
                raise LookupError("conversation not found")
            db.update_conversation(
                conversation_id,
                {"state": state},
                expected_version=current["version"],
            )
            return state

        try:
            await asyncio.to_thread(_write)
        except LookupError:
            self.app_instance.notify(
                f"Could not find {label}; it may have been deleted.",
                severity="warning",
            )
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            logger.error(
                "Console conversation state change failed: exception_type={}",
                type(exc).__name__,
            )
            self.app_instance.notify(
                f"Could not change the status of {label}.", severity="error"
            )
            return

        self.app_instance.notify(f"{label} set to {conversation_state_label(state)}.")
        await self._refresh_console_conversation_browser_after_selection()

    def open_console_conversation_rename(
        self, conversation_id: str, current_title: str
    ) -> None:
        """Prompt for a new title and persist it.

        Args:
            conversation_id: Persisted conversation to rename.
            current_title: Title to seed the prompt with.
        """
        from ...Widgets.Console.console_rename_session_modal import (
            ConsoleRenameSessionModal,
        )

        def _apply(new_title: str | None) -> None:
            if not new_title:
                return
            candidate = new_title.strip()
            if not candidate or candidate == current_title.strip():
                return
            # Qodo review, PR #2233: user text reaching persistence must go
            # through the shared validator, not a bare strip(). Conversation
            # titles are rendered in the rail, the tab strip and the
            # transcript header, so length and content limits belong in one
            # place rather than being re-guessed per call site.
            if not validate_text_input(candidate, max_length=_CONVERSATION_TITLE_MAX):
                self.app_instance.notify(
                    "That title cannot be used. Keep it under "
                    f"{_CONVERSATION_TITLE_MAX} characters and free of markup.",
                    severity="warning",
                )
                return
            self._rename_console_conversation(
                conversation_id, sanitize_string(candidate, max_length=_CONVERSATION_TITLE_MAX)
            )

        self.app_instance.push_screen(ConsoleRenameSessionModal(title=current_title), _apply)

    def _rename_console_conversation(
        self, conversation_id: str, new_title: str
    ) -> None:
        """Write a new conversation title off the event loop."""
        import asyncio

        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            self.app_instance.notify("Conversation storage is unavailable.", severity="error")
            return

        async def _run() -> None:
            def _write() -> None:
                current = db.get_conversation_by_id(conversation_id)
                if not current:
                    raise LookupError("conversation not found")
                db.update_conversation(
                    conversation_id,
                    {"title": new_title},
                    expected_version=current["version"],
                )

            try:
                await asyncio.to_thread(_write)
            except LookupError:
                self.app_instance.notify(
                    "Could not find that conversation; it may have been deleted.",
                    severity="warning",
                )
                return
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                logger.error(
                    "Console conversation rename failed: exception_type={}",
                    type(exc).__name__,
                )
                self.app_instance.notify("Could not rename that conversation.", severity="error")
                return
            self.app_instance.notify(f"Renamed to {new_title}.")
            await self._refresh_console_conversation_browser_after_selection()

        self.run_worker(
            _run(), group="console-conversation-rename", exit_on_error=False
        )

    def confirm_console_conversation_delete(
        self, conversation_id: str, conversation_title: str
    ) -> None:
        """Ask before deleting, then soft-delete on confirmation.

        Soft delete, so the record is recoverable; the dialog says so rather
        than implying the chat is gone for good.

        Args:
            conversation_id: Persisted conversation to delete.
            conversation_title: Name shown in the confirmation.
        """
        from ...Widgets.delete_confirmation_dialog import DeleteConfirmationDialog

        def _confirmed(result: Any) -> None:
            if not result:
                return
            self._delete_console_conversation(conversation_id, conversation_title)

        self.app_instance.push_screen(
            DeleteConfirmationDialog(
                item_type="Conversation",
                item_name=conversation_title or "this conversation",
                permanent=False,
            ),
            _confirmed,
        )

    def _delete_console_conversation(
        self, conversation_id: str, conversation_title: str
    ) -> None:
        """Soft-delete one conversation off the event loop."""
        import asyncio

        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            self.app_instance.notify("Conversation storage is unavailable.", severity="error")
            return

        label = conversation_title or "Conversation"

        async def _run() -> None:
            def _write() -> None:
                current = db.get_conversation_by_id(conversation_id)
                if not current:
                    raise LookupError("conversation not found")
                db.soft_delete_conversation(
                    conversation_id, expected_version=current["version"]
                )

            try:
                await asyncio.to_thread(_write)
            except LookupError:
                self.app_instance.notify(
                    f"{label} was already removed.", severity="warning"
                )
                return
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                logger.error(
                    "Console conversation delete failed: exception_type={}",
                    type(exc).__name__,
                )
                self.app_instance.notify(f"Could not delete {label}.", severity="error")
                return
            self.app_instance.notify(f"Deleted {label}.")
            await self._refresh_console_conversation_browser_after_selection()

        self.run_worker(
            _run(), group="console-conversation-delete", exit_on_error=False
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
