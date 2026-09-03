"""Console session controller.

Extracted out of `ChatScreen` (wave-2 console decomposition, task 3): native
Console session lifecycle -- start/activate/swap/promote/rename -- plus
per-session settings, the Ctrl+K switcher's own choice handling, draft sync,
and screen-state (de)serialization for one `ConsoleChatSession`.

This module follows the SAME binding rule wave 1's `ConsoleDictationController`
established and wave 2's `ConsoleWorkspaceController`/`ConsoleHandsFreeController`
already applied (see `dictation.py`'s own docstring for the canonical
statement; restated briefly here):

1. **Framework services** (`run_worker`, `push_screen`, `push_screen_wait`) live-read from the
   screen via `@property` on every access -- never snapshotted.
2. **A sibling cluster's own attribute, not this cluster's business** is a
   disclosed, temporary live `@property` straight through `screen`:
   `_console_agent_drilldown_run_id` (the sub-agent drill-in cluster's own
   attribute, read+write -- same shape `ConsoleWorkspaceController` already
   uses for the identical attribute).
3. **Everything else this cluster depends on that is not its own state** is a
   NAMED keyword-only constructor callable, matching the design spec's rule
   that "a controller's dependencies are its signature". Each is a
   zero-arg (or narrowly-typed) callable the CALLER (`ChatScreen.__init__`)
   constructs as a late-binding lambda closing over `self` -- never a bound
   method passed directly, for the same instance-patch staleness reason
   `ConsoleDictationController.__init__`'s docstring gives in full. The
   controller's own property of the SAME NAME as the original private
   method/attribute is a thin wrapper around the stored callable.

**Controller-to-controller seam (session <-> workspace), the one new binding
shape this cluster needed**: five of the moved bodies below called straight
into `ConsoleWorkspaceController` in the pre-move source
(`self._workspace._set_active_workspace_for_console_session(...)`,
`.._resume_console_workspace_conversation(...)`,
`.._console_initial_session_title_for_workspace(...)` x3,
`.._merge_console_workspace_rows(...)`,
`.._console_session_id_for_workspace_conversation(...)`). Per the wave-2
Task 3 brief ("the two controllers may need a named callable between them;
design it deliberately, never a back-door through the screen"), those seven
call sites are the only INTENTIONAL, documented deviation from "byte-for-byte
moved" in this module: each drops its `self._workspace.` prefix in favour of
a same-named property here (`_set_active_workspace_for_console_session`,
`_resume_console_workspace_conversation`,
`_console_initial_session_title_for_workspace`, `_merge_console_workspace_
rows`, `_console_session_id_for_workspace_conversation`), each wrapping a
constructor-injected callable `ChatScreen.__init__` points at
`self._workspace.<same method>` -- Python resolves `self._workspace` at
CALL time inside that lambda, so construction order between the two
controllers does not matter. `ChatScreen` itself keeps calling both
controllers' methods directly by their original private names (ordinary
screen-calls-its-own-controller traffic, unchanged).

Moved: every `ChatScreen` method matching `*session*` whose body touched
only session-lifecycle state (or one of the callables above) and no DOM --
27 methods, plus the first-chat handoff family (eight methods and its
notification-revision state), and four small module-level pure helpers only
those methods (or
this cluster's own remaining screen-side callers) use:
`_has_selected_text`/`_is_empty_select_value` (Select-value predicates
`_default_console_session_settings` needs; `ChatScreen` imports both back --
`_is_empty_select_value` has its own independent `@on(Select.Changed)`
consumer, `_has_selected_text` a dozen more, none of them session-shaped),
and the character-handoff quintet `_canonical_card_character_id`/
`_canonical_character_id_text`/`_character_session_identity_from_handoff`/
`_character_session_prompt_seed`/`_SERVER_CHARACTER_AUTHORITY_PATTERN`
(`_start_character_console_session`'s own dependencies; the character
controller imports the card-id and prompt-seed helpers for its picker and
session-choice policy).

`_console_active_session_is_ephemeral` is the one exception to "moved for
real": its IMPLEMENTATION lives here, but `ChatScreen` keeps a one-line
delegation under the original name, because
`Widgets/Console/console_transcript.py`'s `_console_ephemeral_active` reaches
it by BARE NAME off `self.screen` (`getattr(screen,
"_console_active_session_is_ephemeral", None)`, failing toward "not
ephemeral" -- the unsafe direction -- when the name is missing). That is an
external, non-controller consumer probing the screen by name, not a sibling
controller's business, so it gets the same treatment as `ConsoleWorkspace
Controller`'s own DOM-reached members: a thin screen-side delegation.

`ChatScreen` also keeps these members this cluster's name pattern would
otherwise suggest belong here, for the reasons noted:

- Two `action_*` entry points: `action_open_console_session_switcher`
  (builds rows from THREE conversation-browser-cluster methods, not this
  cluster's state, before pushing the switcher modal) and
  `action_open_console_session_settings` (a bare guarded delegation to
  `_open_console_settings`, not part of this cluster at all).
- `_ensure_console_session_surface` (DOM: constructs/returns the
  `ConsoleSessionSurface` widget instance the screen owns -- "never store a
  widget instance the screen may replace").
- `_sync_console_native_session_tabs` (DOM: `query_one` for that surface).
- The seven `ConsoleRealtimeSession`-shaped `*session*` methods
  (`_start_console_realtime_tap`, `_build_console_realtime_session`,
  `_start_console_realtime_connect`, `_on_console_realtime_ready`,
  `_on_console_realtime_reply_done`, `_console_realtime_played_ms`,
  `_close_console_realtime_session`) -- the V4 realtime engine's own state,
  which `hands_free.py`'s own docstring already documents as staying
  screen-owned; a DIFFERENT meaning of "session" than this cluster's
  `ConsoleChatSession`.
- `_console_imagegen_inflight_sessions`/`_park_console_approval`/
  `_remember_console_impersonate` -- name-pattern false positives: in-flight
  image-generation bookkeeping, background-approval parking, and the
  impersonate-suggestion cache, none of which touch `ConsoleChatSession`
  state.
- `_serialize_native_console_state`/`_restore_native_console_state` -- the
  WHOLE native-Console screen-state (de)serializers (image view modes,
  library RAG source types, the pending live-work launch, task-resume
  state...). Since task-15860 Task 3 they carry VIEW state only: Console's
  sessions and transcripts live in the app-owned `ConsoleRuntime` store,
  which survives the navigation, so `_console_session_to_state`/
  `_console_session_from_state` below have **no production caller left**
  and are exercised only by their own tests. Their retirement is tracked
  as task-16520; nothing in the app reads them today.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
import json
from sqlite3 import Error as SQLiteError
from typing import Any, Optional, TYPE_CHECKING
import asyncio
import inspect
import re
import threading
import time
import uuid
import weakref

from loguru import logger
from loguru import logger as loguru_logger
from rich.text import Text
from textual.css.query import QueryError
from textual.widgets import Select

from ...Agents.session_todo_store import SessionTodoStore, TodoStoreError
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    DEFAULT_CONSOLE_SESSION_TITLE,
    ConsoleLifecycleImpact,
    ConsoleMessageRole,
    ConsoleChatMessage,
)
from ...Chat.console_chat_fork import (
    ConsoleChatForkSnapshot,
    ConsoleForkFence,
    ConsoleForkImageSelectionFence,
    default_fork_title,
)
from ...Chat.console_chat_store import (
    ConsoleChatSession,
    ConsoleChatStore,
    ConsoleSettingsComponent,
    ConsoleSettingsPersistenceFailure,
)
from ...Chat.console_chat_controller import (
    ProjectInstructionBindingRecovery,
    resolve_project_instruction_binding,
)
from ...Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextPolicyError,
)
from ...Chat.console_expression_state import (
    CharacterEmoteHistoryIdentity,
    resolve_console_expression_state,
)
from ...Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyDefaults,
)
from ...Chat.console_image_view import resolve_react_character_expressions
from ...Chat.console_roleplay_identity import (
    ChatDisplayNameError,
    effective_user_display_name,
    expand_character_template,
    normalize_chat_display_name,
)
from ...Chat.console_conversation_hydration import (
    ConsoleGenerationSettingsHydration,
    hydrate_console_generation_settings,
)
from ...Chat.console_display_state import (
    ConsoleProjectInstructionSourceRow,
    ConsoleProjectInstructionState,
    build_console_project_instruction_state,
)
from ...Chat.console_project_instructions import (
    ProjectInstructionControlState,
    decode_project_context_json,
    encode_project_context_json,
)
from ...Chat.console_session_settings import (
    ConsoleSessionSettings,
    blank_console_session_settings,
    build_default_console_session_settings,
    build_console_settings_readiness,
    default_console_session_settings,
)
from ...Chat.console_switcher_state import (
    ConsoleSwitcherEntry,
    SwitcherTargetKind,
    UnavailableSessionNotice,
)
from ...Chat.thinking_blocks import normalize_thinking_history_policy
from ...Chat.console_scratch_space import ConsoleScratchSnapshot
from ...Chat.console_turn_context import ConsoleTurnConfigurationSnapshot
from ...Chat.provider_readiness import provider_config_key
from ...Character_Chat.visual_identity import (
    VisualIdentityResolution,
    resolve_historical_visual_identity,
    resolve_visual_identity,
)
from ...Character_Chat.persona_visual_identity import (
    capture_local_persona_visual_identity,
    resolve_persona_visual_identity,
)
from ...DB.VisualIdentity_DB import VisualIdentityRepository
from ...config import (
    coerce_bool_setting,
    get_runtime_config_snapshot,
    run_if_runtime_config_generation_current,
)
from ..Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
    PendingHandoffStore,
)
from ...Widgets.Console import (
    ConsoleComposerUndoHistory,
    ConsoleProjectInstructionStatusRow,
    ConsoleRenameSessionModal,
    ConsoleForkChatModal,
    ConsoleForkDialogSummary,
    ConsoleForkSubmitResult,
    ProjectInstructionBindingOption,
    ProjectInstructionNoticeModal,
    ProjectInstructionSetupModal,
    ProjectInstructionSetupResult,
)
from ...Widgets.Console.console_reaction_picker_modal import (
    ConsoleReactionPickerModal,
    ReactionOption,
)
from ...Widgets.Console.console_session_switcher_modal import ConsoleSwitcherChoice
from ...Widgets.Console.console_activity_outcome_notice import (
    ConsoleActivityOutcomeNotice,
    ConsoleActivityOutcomePresentation,
)
from ...Workspaces import ConsoleConversationBrowserRow, DEFAULT_WORKSPACE_ID
from ...Workspaces.display_state import (
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationRow,
)
from .reaction_preview import ConsoleReactionPreviewCoordinator

if TYPE_CHECKING:
    from ..Screens.chat_screen import ChatScreen

# NOTE (boot budget, ADR-097): `Workspaces.assistant_defaults` is imported
# lazily at its per-turn use site (`_resolve_turn_persona_policy_rules`
# helpers) so it stays out of the UI-ready module census.

logger = logger.bind(module="ChatScreen")


def _safe_persona_lookup(service: Any, persona_id: str) -> Mapping[str, Any] | None:
    """Look a persona up through the app service; any failure means ``None``."""
    try:
        record = service.get_persona_profile(persona_id)
    except Exception:  # noqa: BLE001 -- workspace defaults degrade, never block
        return None
    return record if isinstance(record, Mapping) else None


def build_persona_agent_system_prompt(record: Mapping[str, Any]) -> str:
    """Compose a persona record into a Console system prompt (preview seam parity)."""
    from ...Character_Chat.Character_Chat_Lib import compose_character_card_text

    return (
        compose_character_card_text(
            name=str(record.get("name") or "Workspace Agent"),
            system_prompt=str(record.get("system_prompt") or ""),
            personality=str(record.get("personality") or ""),
            description=str(record.get("description") or ""),
            user_name="User",
        )
        or "Stay in character."
    )


_DEFAULT_PROJECT_INSTRUCTION_NOTICE_TIMEOUT_SECONDS = 120.0
_PROJECT_INSTRUCTION_NOTICE_POLL_SECONDS = 0.1
_PROJECT_INSTRUCTION_AUTHORITY_REFRESH_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class ConsoleSessionCloseImpact:
    """Content-free snapshot pinned to one requested Console session."""

    session_id: str
    transcript_message_count: int
    lifecycle: ConsoleLifecycleImpact

    @property
    def has_loss_risk(self) -> bool:
        return bool(self.transcript_message_count or self.lifecycle.has_loss_risk)


@dataclass(slots=True)
class _ConsoleForkRequest:
    """Controller-owned identity and recovery state for one fork dialog."""

    fence: ConsoleForkFence
    fork_session_id: str
    fork_conversation_id: str | None
    modal: ConsoleForkChatModal
    title: str
    snapshot: ConsoleChatForkSnapshot | None = None
    committed: bool = False
    registered: bool = False
    projection_pending: bool = False


# -- Module-level pure helpers this cluster owns (see module docstring) -----


def _is_empty_select_value(value: Any) -> bool:
    """Return True for Textual's blank/null select sentinels."""
    return value is None or value == Select.BLANK or str(value).startswith("Select.")


def _console_fork_excerpt(content: str, *, max_cells: int = 104) -> str:
    """Collapse and cell-truncate one untrusted message excerpt."""

    excerpt = " ".join(str(content or "").split()) or "(No text)"
    rendered = Text(excerpt)
    rendered.truncate(max_cells, overflow="ellipsis", pad=False)
    return rendered.plain


def _console_fork_copy_failure(error: ValueError) -> str:
    """Name safe content classes without exposing exception details."""

    lowered = str(error).lower()
    for content_class in ("image", "attachment", "citation", "video"):
        if content_class in lowered:
            return (
                f"The fork's {content_class} could not be copied safely. "
                f"Review the {content_class} and retry."
            )
    return "This fork cannot be copied safely. Close and review the source."


def _has_selected_text(value: Any) -> bool:
    """Return whether a provider/model value is meaningfully selected.

    Args:
        value: Value from Textual select state or app/default configuration.

    Returns:
        True when the value is not an empty Textual select sentinel and has
        non-whitespace text.
    """
    return not _is_empty_select_value(value) and bool(str(value).strip())


_MAX_CANONICAL_CHARACTER_ID = (1 << 63) - 1
_MAX_CANONICAL_CHARACTER_ID_TEXT = str(_MAX_CANONICAL_CHARACTER_ID)
_CANONICAL_CHARACTER_ID_PATTERN = re.compile(r"[1-9][0-9]{0,18}")
_SERVER_CHARACTER_AUTHORITY_PATTERN = re.compile(r"server-user-v1:[0-9a-f]{64}")
_CONSOLE_TODO_STATE_KEY = "todo_state"


def _canonical_character_id_text(value: Any) -> str | None:
    """Return an exact signed-64 positive decimal wire ID."""
    if type(value) is not str:
        return None
    if _CANONICAL_CHARACTER_ID_PATTERN.fullmatch(value) is None:
        return None
    if (
        len(value) == len(_MAX_CANONICAL_CHARACTER_ID_TEXT)
        and value > _MAX_CANONICAL_CHARACTER_ID_TEXT
    ):
        return None
    return value


def _canonical_card_character_id(value: Any) -> int | None:
    """Return a canonical signed-64 positive ID from a card response."""
    if type(value) is int:
        return value if 1 <= value <= _MAX_CANONICAL_CHARACTER_ID else None
    value_text = _canonical_character_id_text(value)
    return int(value_text) if value_text is not None else None


def _character_session_identity_from_handoff(
    payload: ChatHandoffPayload,
) -> tuple[str, int, str, str] | None:
    """Return character session identity for Personas Start Chat handoffs.

    Args:
        payload: Handoff payload staged by a source screen.

    Returns:
        A tuple of `(runtime_backend, character_id, character_name,
        assistant_id)` when the payload represents a source-aware Personas
        character Start Chat handoff; otherwise `None`.
    """
    metadata = payload.metadata
    if not isinstance(metadata, Mapping):
        return None
    if (
        payload.source != "personas"
        or payload.item_type != "character-card"
        or metadata.get("intent") != "start_chat"
        or metadata.get("selected_kind") != "character"
    ):
        return None
    runtime_backend = payload.runtime_backend
    if runtime_backend not in {"local", "server"}:
        return None
    if (
        payload.source_owner != runtime_backend
        or payload.source_selector_state != runtime_backend
        or metadata.get("backend") != runtime_backend
    ):
        return None

    character_id_text = _canonical_character_id_text(metadata.get("selected_record_id"))
    if character_id_text is None:
        return None
    if (
        metadata.get("selected_target_id")
        != f"{runtime_backend}:character:{character_id_text}"
    ):
        return None

    character_id = int(character_id_text)
    character_name = str(metadata.get("selected_name") or payload.title or "").strip()
    return runtime_backend, character_id, character_name, character_id_text


@dataclass(frozen=True, slots=True)
class CharacterSessionPromptSeed:
    """Trusted character sources and their current safe projections."""

    name: str
    system_template: str
    system_prompt: str
    greeting_template: str
    greeting: str


def _character_session_prompt_seed(
    card: Mapping[str, Any], name_hint: str = "", *, user_name: str = "User"
) -> CharacterSessionPromptSeed:
    """Return trusted sources and safe projections from a character card.

    Joins the card's prompt-bearing fields into the Console session's system
    prompt and picks the seeded greeting from ``first_message``.

    task-1744: the join itself (field order, labels, and macro resolution)
    is ``Character_Chat_Lib.compose_character_card_text`` -- the same
    function the character-probe eval engine uses
    (``Evals.character_probe.prompt.compose_system_prompt``), so a probe run
    predicts exactly what this seeds into a real Console session. This
    includes ``message_example`` and ``post_history_instructions``, which
    Console did not send before task-1744; that is a deliberate,
    user-visible change to every character session's system prompt, not an
    incidental refactor.

    Args:
        card: The character card record.
        name_hint: Fallback display name when the card has none.

    Returns:
        The character name, exact templates, and current safe projections.
    """
    # Local import matches this module's existing convention of deferring
    # Character_Chat submodule imports (they pull in Pillow and
    # CharactersRAGDB) rather than importing them at module scope.
    from ...Character_Chat.Character_Chat_Lib import (
        compose_character_card_template,
    )

    name = str(card.get("name") or name_hint or "").strip() or "Character"
    # Cards are written against SillyTavern-style macros; compose_character_card_text
    # resolves {{char}}/{{user}} (and aliases) before the text reaches
    # session settings, or they leak verbatim into every provider payload
    # (task-1530). "User" matches the greeting-display substitution used
    # across the Personas surfaces. A card with no prompt-bearing text at
    # all falls back to a fixed instruction -- Console, not the shared
    # composer, owns that fallback, since it is not meaningful to the eval
    # engine's own empty-card handling (an intentionally blank system
    # message, see compose_system_prompt).
    system_template = (
        compose_character_card_template(
            name=name,
            system_prompt=str(card.get("system_prompt") or ""),
            personality=str(card.get("personality") or ""),
            description=str(card.get("description") or ""),
            scenario=str(card.get("scenario") or ""),
            message_example=str(card.get("message_example") or ""),
            post_history_instructions=str(card.get("post_history_instructions") or ""),
        )
        or "Stay in character."
    )
    greeting_template = str(card.get("first_message") or "")
    return CharacterSessionPromptSeed(
        name=name,
        system_template=system_template,
        system_prompt=expand_character_template(
            system_template,
            user_name=user_name,
            character_name=name,
        ),
        greeting_template=greeting_template,
        greeting=expand_character_template(
            greeting_template,
            user_name=user_name,
            character_name=name,
        ),
    )


def _console_global_user_display_name(app_config: object) -> str:
    """Resolve the current global chat label before Task 5 adds its getter."""
    chat_defaults = (
        app_config.get("chat_defaults") if isinstance(app_config, Mapping) else None
    )
    raw_name = (
        chat_defaults.get("user_display_name")
        if isinstance(chat_defaults, Mapping)
        else None
    )
    try:
        return effective_user_display_name(None, raw_name)
    except ChatDisplayNameError:
        return "User"


def _resolve_visual_identity_for_db(
    db: Any,
    scope: tuple[str, str, str],
    requested_state: str,
    manual_expression_key: str | None,
    local_persona_service: object | None = None,
) -> VisualIdentityResolution | None:
    """Resolve one immutable preview request without retaining its screen."""

    _session_id, actor_kind, actor_id = scope
    try:
        if actor_kind == "persona":
            if local_persona_service is None:
                return None
            return resolve_persona_visual_identity(
                db,
                local_persona_service,
                persona_id=actor_id,
                requested_state=requested_state,
                manual_expression_key=manual_expression_key,
            )
        return resolve_visual_identity(
            db,
            actor_kind=actor_kind,
            actor_id=actor_id,
            requested_state=requested_state,
            manual_expression_key=manual_expression_key,
        )
    except (SQLiteError, TypeError, ValueError, OverflowError) as exc:
        logger.debug(  # noqa: PLE1205 - Loguru uses brace-style arguments.
            "Console reaction resolution failed for actor_kind={} actor_id={} "
            "error_type={}",
            actor_kind,
            actor_id,
            type(exc).__name__,
        )
        return None


def _visual_identity_options_for_db(
    db: Any,
    scope: tuple[str, str, str],
    local_persona_service: object | None = None,
) -> tuple[ReactionOption, ...]:
    """Read metadata-only preview options without retaining its screen."""

    _session_id, actor_kind, actor_id = scope
    try:
        persona_authority = None
        if actor_kind == "persona":
            if local_persona_service is None:
                return ()
            persona_authority = capture_local_persona_visual_identity(
                local_persona_service, actor_id
            )
            if persona_authority is None:
                return ()
        graph = VisualIdentityRepository(db).get_active_actor_pack(actor_kind, actor_id)
        if (
            actor_kind == "persona"
            and capture_local_persona_visual_identity(local_persona_service, actor_id)
            != persona_authority
        ):
            return ()
    except (SQLiteError, TypeError, ValueError, OverflowError) as exc:
        logger.debug(  # noqa: PLE1205 - Loguru uses brace-style arguments.
            "Console reaction inventory failed for actor_kind={} actor_id={} "
            "error_type={}",
            actor_kind,
            actor_id,
            type(exc).__name__,
        )
        return ()
    if graph is None:
        return ()
    return tuple(
        ReactionOption(
            expression_key=str(asset["expression_key"]),
            display_label=str(asset["display_label"]),
            content_type=str(asset["content_type"]),
            is_animated=bool(asset["is_animated"]),
        )
        for asset in graph["assets"]
    )


class ConsoleSessionController:
    """Owns the Console shell's native session lifecycle: start/activate/
    swap/promote/rename, per-session settings, the Ctrl+K switcher's choice
    handling, draft sync, and one-session (de)serialization.

    `ChatScreen` constructs exactly one of these, in `__init__`, and keeps a
    `self._session` reference plus the stay-behind members described in the
    module docstring.
    """

    # Readiness labels that stale-default session refresh may recover from:
    # credential/endpoint gaps a Settings save can fix. Provider-identity
    # blockers (Unknown/Pending WIP providers) are deliberate choices and are
    # never auto-replaced.
    _CONSOLE_REFRESHABLE_BLOCKED_LABELS = frozenset(
        {"Missing key", "Not ready", "Invalid URL", "Endpoint not saved"}
    )

    def __init__(
        self,
        screen: "ChatScreen",
        *,
        app_instance: Any,
        chat_store_accessor: Callable[[], ConsoleChatStore],
        current_chat_store_accessor: Callable[[], ConsoleChatStore | None],
        ensure_console_chat_controller: Callable[[], Any],
        composer_accessor: Callable[[], Any],
        restore_banked_raw_cli_stashes: Callable[[str, Any], int],
        effective_console_provider_model: Callable[[], tuple[Any, Any]],
        provider_readiness_app_config: Callable[[], Any],
        build_provider_selection: Callable[[str], Any],
        scratch_snapshot_provider: Callable[[str], ConsoleScratchSnapshot],
        rag_source_types_accessor: Callable[[], tuple[str, ...]],
        rag_top_k_accessor: Callable[[], int],
        sync_native_console_chat_ui: Callable[[], Any],
        sync_chat_core_state: Callable[[], Any],
        sync_temporary_chip: Callable[[], None],
        sync_settings_summary: Callable[[], None],
        sync_control_bar: Callable[[], None],
        sync_command_popup: Callable[[], None],
        note_follow_intent: Callable[[], None],
        focus_composer_if_needed: Callable[..., None],
        invalidate_persisted_rows_cache: Callable[[], None],
        mark_conversation_row_broken: Callable[[str], None],
        refresh_effective_scope_and_sync: Callable[[Any], Any],
        session_surface_accessor: Callable[[], Any | None],
        switcher_authority_accessor: Callable[[], tuple[str, str]],
        console_runtime_accessor: Callable[[], Any],
        set_active_workspace_for_session: Callable[[str], None],
        resume_workspace_conversation: Callable[..., Any],
        workspace_initial_session_title: Callable[[str | None], str],
        merge_workspace_rows: Callable[[list, tuple], list],
        session_id_for_workspace_conversation: Callable[[str], str | None],
        ensure_console_image_view: Callable[[], tuple[Any, Any]],
        visual_identity_db_accessor: Callable[[], Any | None],
        reaction_preview_coordinator_accessor: Callable[
            [], ConsoleReactionPreviewCoordinator
        ],
        refresh_character_avatar: Callable[..., Any],
        screen_mounted_accessor: Callable[[], bool],
        first_chat_presentation_snapshot: Callable[[], tuple[Any, Any, object | None]],
        apply_first_chat_control_selection: Callable[[Any, Any], None],
        restore_first_chat_focus: Callable[[object | None], None],
        capture_fork_image_selections: Callable[
            [Sequence[ConsoleChatMessage]], tuple[ConsoleForkImageSelectionFence, ...]
        ],
        validate_fork_image_selections: Callable[
            [
                Sequence[ConsoleChatMessage],
                Sequence[ConsoleForkImageSelectionFence],
            ],
            bool,
        ],
        workspace_display_name: Callable[[str], str],
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 27 method bodies below is a byte-for-byte copy of
        the pre-extraction `ChatScreen` method, EXCEPT the seven documented
        `self._workspace.X(...)` call sites (see the module docstring's
        "Controller-to-controller seam" section), which drop their
        `_workspace.` prefix in favour of a same-named property on this
        controller instead.

        The eight-method first-chat family is likewise copied with only its
        presentation edges adapted to the four exact late-bound callbacks:
        mounted state, provider/model/focus snapshot, control selection, and
        focus restoration. It retains the documented legacy `_screen`
        exception only for the framework services above; the moved family
        itself never reaches through `_screen` or into the DOM.

        Args:
            screen: The Console screen. Used ONLY for the framework
                services (`run_worker`, `push_screen`) and the one disclosed
                sibling-cluster reach-back (`_console_agent_drilldown_
                run_id`). Zero `query_one`/`query` traffic reaches through
                `screen` here.
            app_instance: For `notify()` and the character/handoff service
                lookups `_start_character_console_session` makes --
                unchanged from how the pre-extraction methods used
                `self.app_instance`. Snapshotted as a plain attribute: it
                does not change identity over the controller's life, and
                the pre-extraction methods never called it, only read it.
            chat_store_accessor: `ChatScreen._ensure_console_chat_store`
                (lazily creates the store) -- used where the original body
                called it as a method.
            current_chat_store_accessor: A DIFFERENT accessor for the SAME
                store, `lambda: self._console_chat_store` -- a bare
                attribute read that must NOT create the store as a side
                effect (several moved bodies branch on it still being
                `None`). Kept distinct from `chat_store_accessor` for the
                same reason `ConsoleWorkspaceController` keeps its own pair
                distinct.
            ensure_console_chat_controller: `ChatScreen._ensure_console_
                chat_controller` -- the (much larger-scoped) chat
                orchestration controller, used only by
                `_activate_native_console_session` for its shared
                `controller.store`/`controller.switch_session` sequence.
            composer_accessor: `ChatScreen._console_composer_or_none` (DOM);
                same shape as `dictation.py`'s/`hands_free.py`'s own
                `composer_accessor`.
            restore_banked_raw_cli_stashes: Late-bound raw CLI refusal restore
                after the origin session's composer has been reconciled.
            effective_console_provider_model: `ChatScreen._effective_
                console_provider_model`, used by `_default_console_session_
                settings`.
            provider_readiness_app_config: `ChatScreen._provider_readiness_
                app_config`.
            build_provider_selection: Owning-session provider selection builder.
            rag_source_types_accessor: Current normalized Console RAG source kinds.
            rag_top_k_accessor: Current active-profile RAG result count.
            sync_native_console_chat_ui: `ChatScreen._sync_native_console_
                chat_ui` -- a coroutine FUNCTION, called (not awaited
                directly) by every moved body exactly as the original did.
            sync_chat_core_state: `ChatScreen._sync_console_chat_core_
                state`.
            sync_temporary_chip: `ChatScreen._sync_console_temporary_chip`
                (DOM).
            sync_settings_summary: `ChatScreen._sync_console_settings_
                summary` (DOM).
            sync_control_bar: `ChatScreen._sync_console_control_bar` (DOM);
                the moved body calls it with no arguments, matching the
                original `self._sync_console_control_bar()` call shape.
            sync_command_popup: `ChatScreen._sync_console_command_popup`
                (DOM).
            note_follow_intent: `ChatScreen._note_console_follow_intent`
                (DOM); same name `ConsoleWorkspaceController` already binds
                independently.
            focus_composer_if_needed: `ChatScreen._focus_console_composer_
                if_needed`; the moved bodies call it with `force=True`
                exactly as before -- the stored callable accepts the same
                keyword, not a value baked in here.
            invalidate_persisted_rows_cache: `ChatScreen._invalidate_
                console_persisted_rows_cache` -- the conversation-browser
                cluster's own cache, not this cluster's state.
            mark_conversation_row_broken: `ChatScreen._mark_console_
                conversation_row_broken` -- same sibling cluster, used by
                `_apply_console_switcher_choice`'s TASK-717 branch.
            refresh_effective_scope_and_sync: `ChatScreen._refresh_console_
                effective_scope_and_sync`, async; same name
                `ConsoleWorkspaceController` already binds independently.
            set_active_workspace_for_session: `ConsoleWorkspaceController.
                _set_active_workspace_for_console_session` -- the
                session<->workspace seam (see module docstring); the ORIGINAL
                body called `self._workspace._set_active_workspace_for_
                console_session(...)`, dropped to `self._set_active_
                workspace_for_session(...)` here.
            resume_workspace_conversation: `ConsoleWorkspaceController.
                _resume_console_workspace_conversation` -- same seam,
                `_apply_console_switcher_choice`'s conversation-id branch.
            workspace_initial_session_title: `ConsoleWorkspaceController.
                _console_initial_session_title_for_workspace` -- same seam,
                used by three moved bodies
                (`_ensure_active_console_session_settings`, `_replace_
                active_console_session_settings`, `_sync_console_session_
                draft`).
            merge_workspace_rows: `ConsoleWorkspaceController._merge_
                console_workspace_rows` -- same seam, `_with_native_
                console_session_rows`.
            session_id_for_workspace_conversation: `ConsoleWorkspace
                Controller._console_session_id_for_workspace_conversation`
                -- same seam, `_console_session_id_for_browser_row`.
            ensure_console_image_view: `ChatScreen._ensure_console_image_
                view` -- returns the screen's ``(view state, render
                cache)`` pair, building it on first ask. The inline-image
                cluster stays screen-owned and is out of scope this wave,
                so `_close_console_session_tab` (wave-4 task 2) reaches it
                as a named callable rather than through `self._screen`.
                Not DOM: the pair is plain state plus a render cache, so
                the zero-DOM rule is untouched.
            visual_identity_db_accessor: Current profile-local character DB.
                Late-bound so tests and profile switches are observed.
            reaction_preview_coordinator_accessor: Current app's shared reaction
                preview single-flight coordinator. Late-bound so replacement
                Console screens cannot escape an older screen's draining work.
            refresh_character_avatar: Late-bound forced avatar refresh after
                a validated manual reaction change.
            screen_mounted_accessor: Late-bound presentation-only mounted state.
            first_chat_presentation_snapshot: Late-bound provider/model/focus
                snapshot used by first-chat rollback.
            apply_first_chat_control_selection: Late-bound projection of the
                first-chat provider/model selection onto screen controls.
            restore_first_chat_focus: Late-bound restoration of an opaque focus
                token after the native async projection is synchronized.
        """
        self._screen = screen
        self.app_instance = app_instance
        self._chat_store_accessor = chat_store_accessor
        self._current_chat_store_accessor = current_chat_store_accessor
        self._ensure_console_chat_controller_fn = ensure_console_chat_controller
        self._composer_accessor = composer_accessor
        self._restore_banked_raw_cli_stashes_fn = restore_banked_raw_cli_stashes
        self._effective_console_provider_model_fn = effective_console_provider_model
        self._provider_readiness_app_config_fn = provider_readiness_app_config
        self._build_provider_selection_fn = build_provider_selection
        self._scratch_snapshot_provider = scratch_snapshot_provider
        self._rag_source_types_accessor = rag_source_types_accessor
        self._rag_top_k_accessor = rag_top_k_accessor
        self._sync_native_console_chat_ui_fn = sync_native_console_chat_ui
        self._sync_chat_core_state_fn = sync_chat_core_state
        self._sync_temporary_chip_fn = sync_temporary_chip
        self._sync_settings_summary_fn = sync_settings_summary
        self._sync_control_bar_fn = sync_control_bar
        self._sync_command_popup_fn = sync_command_popup
        self._note_follow_intent_fn = note_follow_intent
        self._focus_composer_if_needed_fn = focus_composer_if_needed
        self._invalidate_persisted_rows_cache_fn = invalidate_persisted_rows_cache
        self._mark_conversation_row_broken_fn = mark_conversation_row_broken
        self._refresh_effective_scope_and_sync_fn = refresh_effective_scope_and_sync
        self._session_surface_accessor = session_surface_accessor
        self._switcher_authority_accessor = switcher_authority_accessor
        self._console_runtime_accessor = console_runtime_accessor
        self._set_active_workspace_for_session_fn = set_active_workspace_for_session
        self._resume_workspace_conversation_fn = resume_workspace_conversation
        self._workspace_initial_session_title_fn = workspace_initial_session_title
        self._merge_workspace_rows_fn = merge_workspace_rows
        self._session_id_for_workspace_conversation_fn = (
            session_id_for_workspace_conversation
        )
        self._ensure_console_image_view_fn = ensure_console_image_view
        self._visual_identity_db_accessor = visual_identity_db_accessor
        self._reaction_preview_coordinator_accessor = (
            reaction_preview_coordinator_accessor
        )
        self._refresh_character_avatar_fn = refresh_character_avatar
        self._screen_mounted_accessor = screen_mounted_accessor
        self._first_chat_presentation_snapshot_fn = first_chat_presentation_snapshot
        self._apply_first_chat_control_selection_fn = apply_first_chat_control_selection
        self._restore_first_chat_focus_fn = restore_first_chat_focus
        self._capture_fork_image_selections_fn = capture_fork_image_selections
        self._validate_fork_image_selections_fn = validate_fork_image_selections
        self._workspace_display_name_fn = workspace_display_name

        # This cluster's own state, moved verbatim from `ChatScreen.__init__`.
        self._console_visible_draft_session_id: str | None = None
        self._console_undo_histories: dict[str, ConsoleComposerUndoHistory] = {}
        self._console_draft_switch_snapshot: tuple[str | None, str, int] | None = None
        self._closing_session_requests: set[str] = set()
        self._manual_reaction_overrides: dict[tuple[str, str, str], str] = {}
        self._reaction_preview_generation = 0
        self._reaction_preview_worker: Any | None = None
        self._reaction_preview_target_ref: (
            weakref.ReferenceType[ConsoleReactionPickerModal] | None
        ) = None
        self._console_project_instruction_display_cache: dict[
            str, tuple[ProjectInstructionControlState, ConsoleProjectInstructionState]
        ] = {}
        self._console_project_instruction_refresh_inflight: dict[
            str, tuple[Any, Any]
        ] = {}
        self._console_project_instruction_refresh_completed: dict[
            str, tuple[tuple[Any, Any], float]
        ] = {}
        self._first_chat_handoff_notified_revision: int | None = None
        self._fork_validation_generation = 0
        self._active_fork_request: _ConsoleForkRequest | None = None

    # -- Framework services (live-read via `@property`) --------------------

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
    def push_screen_wait(self) -> Any:
        """`Screen.app.push_screen_wait`, bound. See `__init__`'s docstring."""

        return self._screen.app.push_screen_wait

    @property
    def run_app_worker(self) -> Any:
        """App worker service for work that must survive a modal dismissal."""

        return self._screen.app.run_worker

    # -- Sibling cluster's reach-back (disclosed) ---------------------------

    @property
    def _console_agent_drilldown_run_id(self) -> str | None:
        """The sub-agent drill-in cluster's own attribute. Read+write: the
        moved activation body clears it unconditionally on every switch.
        Same shape `ConsoleWorkspaceController` binds independently for the
        identical attribute."""
        return self._screen._console_agent_drilldown_run_id

    @_console_agent_drilldown_run_id.setter
    def _console_agent_drilldown_run_id(self, value: str | None) -> None:
        self._screen._console_agent_drilldown_run_id = value

    # -- Named constructor dependencies -------------------------------------
    #
    # Each property below is a thin wrapper around a stored callable, kept
    # under the SAME name the original `ChatScreen` method/attribute used --
    # see `__init__`'s docstring. `_console_chat_store` is the one
    # bare-attribute-read shape (calls the accessor immediately and returns
    # the value); every other property returns the callable itself.

    @property
    def _ensure_console_chat_store(self) -> Any:
        return self._chat_store_accessor

    @property
    def _console_chat_store(self) -> ConsoleChatStore | None:
        return self._current_chat_store_accessor()

    @property
    def _ensure_console_chat_controller(self) -> Any:
        return self._ensure_console_chat_controller_fn

    @property
    def _console_composer_or_none(self) -> Any:
        return self._composer_accessor

    @property
    def _effective_console_provider_model(self) -> Any:
        return self._effective_console_provider_model_fn

    @property
    def _provider_readiness_app_config(self) -> Any:
        return self._provider_readiness_app_config_fn

    @property
    def _sync_native_console_chat_ui(self) -> Any:
        return self._sync_native_console_chat_ui_fn

    @property
    def _sync_console_chat_core_state(self) -> Any:
        return self._sync_chat_core_state_fn

    @property
    def _sync_console_temporary_chip(self) -> Any:
        return self._sync_temporary_chip_fn

    @property
    def _sync_console_settings_summary(self) -> Any:
        return self._sync_settings_summary_fn

    @property
    def _sync_console_control_bar(self) -> Any:
        return self._sync_control_bar_fn

    @property
    def _sync_console_command_popup(self) -> Any:
        return self._sync_command_popup_fn

    @property
    def _note_console_follow_intent(self) -> Any:
        return self._note_follow_intent_fn

    @property
    def _focus_console_composer_if_needed(self) -> Any:
        return self._focus_composer_if_needed_fn

    @property
    def _invalidate_console_persisted_rows_cache(self) -> Any:
        return self._invalidate_persisted_rows_cache_fn

    @property
    def _mark_console_conversation_row_broken(self) -> Any:
        return self._mark_conversation_row_broken_fn

    @property
    def _refresh_console_effective_scope_and_sync(self) -> Any:
        return self._refresh_effective_scope_and_sync_fn

    # -- Session<->workspace seam (see module docstring) --------------------

    @property
    def _set_active_workspace_for_session(self) -> Any:
        """`ConsoleWorkspaceController._set_active_workspace_for_console_
        session`. The pre-move body called `self._workspace._set_active_
        workspace_for_console_session(...)`; the moved body drops the
        `_workspace.` prefix in favour of this property."""
        return self._set_active_workspace_for_session_fn

    @property
    def _resume_workspace_conversation(self) -> Any:
        """`ConsoleWorkspaceController._resume_console_workspace_
        conversation`. Same prefix-drop as above."""
        return self._resume_workspace_conversation_fn

    @property
    def _workspace_initial_session_title(self) -> Any:
        """`ConsoleWorkspaceController._console_initial_session_title_for_
        workspace`. Same prefix-drop as above."""
        return self._workspace_initial_session_title_fn

    @property
    def _merge_workspace_rows(self) -> Any:
        """`ConsoleWorkspaceController._merge_console_workspace_rows`. Same
        prefix-drop as above."""
        return self._merge_workspace_rows_fn

    @property
    def _ensure_console_image_view(self) -> Any:
        """The injected `ensure_console_image_view`. Stays on `ChatScreen`
        (the inline-image render cache is not this cluster's state). See
        `__init__`'s docstring."""
        return self._ensure_console_image_view_fn

    @property
    def _session_id_for_workspace_conversation(self) -> Any:
        """`ConsoleWorkspaceController._console_session_id_for_workspace_
        conversation`. Same prefix-drop as above."""
        return self._session_id_for_workspace_conversation_fn

    @staticmethod
    def _first_chat_defaults_match(
        intent: ConsoleFirstChatIntent,
        settings: ConsoleSessionSettings,
    ) -> bool:
        return (
            provider_config_key(settings.provider) == intent.provider
            and str(settings.model or "").strip() == intent.model
        )

    def _current_first_chat_defaults(
        self,
        *,
        provider: str,
        model: str,
        config_revision: int,
    ) -> ConsoleSessionSettings | None:
        """Resolve exact current defaults only while the config fence matches."""

        snapshot = get_runtime_config_snapshot()
        if snapshot.generation != config_revision:
            return None
        settings = build_default_console_session_settings(snapshot.values)
        if (
            provider_config_key(settings.provider) != provider_config_key(provider)
            or str(settings.model or "").strip() != str(model or "").strip()
        ):
            return None
        return settings

    def eligible_console_first_chat_session_id(self) -> str | None:
        """Return an exact untouched target without changing Console.

        Returns:
            str | None: The eligible active session ID, or ``None`` when the
                active session is not a pristine global Console target.
        """

        store = self._console_chat_store
        if store is None:
            return None
        active_id = store.active_session_id
        if active_id is None:
            return None
        active = next(
            (session for session in store.sessions() if session.id == active_id),
            None,
        )
        baseline = active.canonical_settings_baseline if active is not None else None
        if (
            baseline is None
            or active.workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID
            or not store.is_pristine_session(
                active_id,
                expected_settings=baseline,
            )
        ):
            return None
        return active_id

    def _release_first_chat_claim(self, claim, message: str) -> bool:
        """Release an exact claim without leaking failure-owned data."""

        handoffs = self.app_instance.pending_handoffs
        try:
            claim_is_current = handoffs.is_current_claim(claim)
        except Exception as exc:  # noqa: BLE001 - lifecycle boundary containment
            claim_is_current = False
            self._log_first_chat_handoff_exception("claim-current-check", exc)
        try:
            released = handoffs.release(claim)
        except Exception as exc:  # noqa: BLE001 - keep the channel retryable
            self._log_first_chat_handoff_exception("claim-release", exc)
            released = False
            if isinstance(handoffs, PendingHandoffStore):
                try:
                    # Bypass a failing instance wrapper while retaining the
                    # store's exact-claim and replacement invariants.
                    released = PendingHandoffStore.release(handoffs, claim)
                except Exception as fallback_exc:  # noqa: BLE001
                    self._log_first_chat_handoff_exception(
                        "claim-release-fallback",
                        fallback_exc,
                    )
        if not released:
            return False
        if not claim_is_current:
            if self._first_chat_handoff_notified_revision == claim.revision:
                self._first_chat_handoff_notified_revision = None
            return False
        if claim.revision != self._first_chat_handoff_notified_revision:
            self._first_chat_handoff_notified_revision = claim.revision
            try:
                self.app_instance.notify(message, severity="warning")
            except Exception as exc:  # noqa: BLE001 - lifecycle boundary containment
                self._log_first_chat_handoff_exception("notification", exc)
        return False

    @staticmethod
    def _log_first_chat_handoff_exception(category: str, exc: Exception) -> None:
        """Log only allowlisted failure classification, never exception content."""

        logger.warning(
            "First-chat handoff operation failed (category={}, error_type={})",
            category,
            type(exc).__name__,
        )

    async def _resync_console_after_first_chat_rollback(
        self,
        prior_focused_widget: object | None,
    ) -> None:
        """Re-render restored Console state, then restore still-mounted focus."""

        if not self._screen_mounted_accessor():
            return
        await self._sync_native_console_chat_ui()
        if not self._screen_mounted_accessor():
            return
        self._restore_first_chat_focus_fn(prior_focused_widget)

    def _resync_mounted_console_after_first_chat_rollback(
        self,
        *,
        prior_control_provider: str | None,
        prior_control_model: str | None,
        prior_focused_widget: object | None,
    ) -> None:
        """Restore first-chat-owned scalars and every mounted Console projection."""

        self._apply_first_chat_control_selection_fn(
            prior_control_provider,
            prior_control_model,
        )
        if not self._screen_mounted_accessor():
            return
        self._sync_console_chat_core_state()
        self._sync_console_settings_summary()
        self._sync_console_control_bar()
        self.run_worker(
            self._resync_console_after_first_chat_rollback(prior_focused_widget),
            group="console-first-chat-rollback",
            exit_on_error=False,
        )

    def consume_pending_console_first_chat_intent(
        self,
        *,
        defer_presentation: bool = False,
    ) -> bool:
        """Activate one exact first-run target without overwriting user state.

        Args:
            defer_presentation: Settle session and handoff ownership without
                projecting mounted controls or scheduling rollback focus. The
                ordered Resume opener will present the final target instead.

        Returns:
            bool: ``True`` only when the pending intent is applied and
                acknowledged; otherwise ``False``.
        """

        claim = self.app_instance.pending_handoffs.claim(
            HandoffChannel.CONSOLE_FIRST_CHAT
        )
        if claim is None:
            return False
        intent = claim.value
        if not isinstance(intent, ConsoleFirstChatIntent):
            return self._release_first_chat_claim(
                claim,
                "The first chat could not be opened yet; review provider setup.",
            )
        defaults = self._current_first_chat_defaults(
            provider=intent.provider,
            model=intent.model,
            config_revision=intent.config_revision,
        )
        if defaults is None:
            return self._release_first_chat_claim(
                claim,
                "Provider settings changed before Console opened. Review setup and try again.",
            )

        store = self._ensure_console_chat_store()
        prior_active_id = store.active_session_id
        prior_control_provider = None
        prior_control_model = None
        prior_focused_widget = None
        if not defer_presentation:
            (
                prior_control_provider,
                prior_control_model,
                prior_focused_widget,
            ) = self._first_chat_presentation_snapshot_fn()
        created_target = None
        refreshed_prior: (
            tuple[
                ConsoleSessionSettings,
                ConsoleSessionSettings,
                int,
                ConsoleSettingsPersistenceFailure | None,
                str,
            ]
            | None
        ) = None

        def rollback_mutation() -> None:
            if created_target is not None:
                store.rollback_created_pristine_session(
                    created_target.id,
                    expected_session=created_target,
                    expected_settings=defaults,
                    prior_active_session_id=prior_active_id,
                )
            elif refreshed_prior is not None:
                (
                    prior_settings,
                    prior_baseline,
                    prior_generation_revision,
                    prior_generation_failure,
                    prior_updated_at,
                ) = refreshed_prior
                store.rollback_pristine_session_refresh(
                    intent.session_id,
                    expected_current_settings=defaults,
                    prior_settings=prior_settings,
                    prior_canonical_settings=prior_baseline,
                    prior_generation_revision=prior_generation_revision,
                    prior_generation_failure=prior_generation_failure,
                    prior_updated_at=prior_updated_at,
                )
            if not defer_presentation:
                self._resync_mounted_console_after_first_chat_rollback(
                    prior_control_provider=prior_control_provider,
                    prior_control_model=prior_control_model,
                    prior_focused_widget=prior_focused_widget,
                )

        def rollback_and_release(message: str) -> bool:
            try:
                rollback_mutation()
            except Exception as exc:  # noqa: BLE001 - lifecycle boundary containment
                self._log_first_chat_handoff_exception("rollback", exc)
            return self._release_first_chat_claim(claim, message)

        def fence_matches(*, expected_active_id: str) -> bool:
            current = self._current_first_chat_defaults(
                provider=intent.provider,
                model=intent.model,
                config_revision=intent.config_revision,
            )
            return (
                current == defaults
                and store.active_session_id == expected_active_id
                and self.app_instance.pending_handoffs.is_current_claim(claim)
            )

        reserves_new_target = (
            self.app_instance.pending_handoffs.claim_reserves_new_console_session(claim)
        )
        target = next(
            (
                session
                for session in store.sessions()
                if session.id == intent.session_id
            ),
            None,
        )
        if target is None:
            if not reserves_new_target:
                return self._release_first_chat_claim(
                    claim,
                    "The intended Console session is no longer available. Review setup and try again.",
                )
            try:
                target = store.create_session(
                    session_id=intent.session_id,
                    workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
                    settings=defaults,
                    canonical_settings_baseline=defaults,
                    activate=False,
                )
            except ValueError:
                return self._release_first_chat_claim(
                    claim,
                    "The intended Console session was claimed before setup finished. It was left unchanged.",
                )
            created_target = target
            if not fence_matches(expected_active_id=prior_active_id):
                return rollback_and_release(
                    "Provider settings changed while Console prepared the first chat. It will retry.",
                )
            store.switch_session(intent.session_id)
            if not fence_matches(expected_active_id=intent.session_id):
                return rollback_and_release(
                    "Console changed while the first chat was opening. Your sessions were left unchanged.",
                )
        else:
            if reserves_new_target:
                return self._release_first_chat_claim(
                    claim,
                    "The intended Console session was claimed before setup finished. It was left unchanged.",
                )
            if store.active_session_id != intent.session_id:
                return self._release_first_chat_claim(
                    claim,
                    "Console changed sessions before setup finished. Your current session was left unchanged.",
                )
            baseline = target.canonical_settings_baseline
            if (
                baseline is None
                or target.workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID
                or not store.is_pristine_session(
                    intent.session_id,
                    expected_settings=baseline,
                )
            ):
                return self._release_first_chat_claim(
                    claim,
                    "The intended Console session now contains work. It was left unchanged.",
                )
            if baseline != defaults:
                refreshed_prior = (
                    target.settings,
                    baseline,
                    target.generation_settings_revision,
                    target.settings_persistence_failures.get(
                        ConsoleSettingsComponent.GENERATION_SETTINGS
                    ),
                    target.updated_at,
                )
                store.refresh_pristine_session_settings(
                    intent.session_id,
                    prior_canonical_settings=baseline,
                    current_canonical_settings=defaults,
                )
                target = next(
                    session
                    for session in store.sessions()
                    if session.id == intent.session_id
                )
                if not fence_matches(expected_active_id=intent.session_id):
                    return rollback_and_release(
                        "Provider settings changed while Console prepared the first chat. It will retry.",
                    )

        if (
            target.settings is None
            or not self._first_chat_defaults_match(intent, target.settings)
            or not fence_matches(expected_active_id=intent.session_id)
        ):
            return rollback_and_release(
                "The first chat target no longer matches provider setup. It was left unchanged.",
            )

        if not defer_presentation:
            self._apply_first_chat_control_selection_fn(
                target.settings.provider,
                target.settings.model,
            )
            if self._screen_mounted_accessor():
                self._sync_console_chat_core_state()
                self._sync_console_settings_summary()
                self._sync_console_control_bar()
        if not fence_matches(expected_active_id=intent.session_id):
            return rollback_and_release(
                "Console changed before the first chat finished opening. It will retry.",
            )
        try:
            acknowledged = run_if_runtime_config_generation_current(
                intent.config_revision,
                lambda: self.app_instance.pending_handoffs.acknowledge_current(claim),
            )
        except Exception as exc:  # noqa: BLE001 - mount/resume must not fail
            self._log_first_chat_handoff_exception("guarded-acknowledgement", exc)
            return rollback_and_release(
                "The first chat could not be acknowledged yet. It will retry.",
            )
        if not acknowledged:
            return rollback_and_release(
                "The first chat could not be acknowledged yet. It will retry.",
            )
        self._first_chat_handoff_notified_revision = None
        return True

    # -- Session-local character reactions ----------------------------------

    def _manual_reaction_key(self, scope: tuple[str, str, str]) -> str | None:
        """Return one session-and-actor-local manual reaction key."""

        return self._manual_reaction_overrides.get(scope)

    def _set_manual_reaction(
        self, scope: tuple[str, str, str], expression_key: str
    ) -> None:
        """Set one already-validated session-and-actor-local reaction key."""

        self._manual_reaction_overrides[scope] = expression_key

    def _clear_manual_reaction(self, scope: tuple[str, str, str]) -> None:
        """Clear the manual reaction for one exact session and actor."""

        self._manual_reaction_overrides.pop(scope, None)

    def _clear_session_manual_reactions(self, session_id: str) -> None:
        """Drop all in-memory reaction choices for one disposed session."""

        for scope in tuple(self._manual_reaction_overrides):
            if scope[0] == session_id:
                self._manual_reaction_overrides.pop(scope, None)

    def _clear_replaced_actor_reactions(
        self, session_id: str, *, actor_kind: str, actor_id: str
    ) -> None:
        """Drop old-actor overrides after a session is rebound successfully."""

        current = (session_id, actor_kind, actor_id)
        for scope in tuple(self._manual_reaction_overrides):
            if scope[0] == session_id and scope != current:
                self._manual_reaction_overrides.pop(scope, None)

    def _current_visual_identity_actor_scope(self) -> tuple[str, str, str] | None:
        """Return the active local Character or Persona actor scope."""

        session = self._active_native_console_session()
        if session is None or session.runtime_backend != "local":
            return None
        if session.assistant_kind == "persona":
            actor_id = session.assistant_id
            if type(actor_id) is not str or not actor_id or len(actor_id) > 200:
                return None
            return (session.id, "persona", actor_id)
        actor_id = session.local_character_id()
        return (
            (session.id, "character", str(actor_id)) if actor_id is not None else None
        )

    def _local_persona_visual_identity_service(self) -> object | None:
        """Return the current local Persona service without retaining it."""

        scope = getattr(self.app_instance, "character_persona_scope_service", None)
        return getattr(scope, "local_service", None)

    def _manual_reaction_label_for_current_actor(self) -> str | None:
        """Return a compact display label for the active manual reaction."""

        scope = self._current_visual_identity_actor_scope()
        key = self._manual_reaction_key(scope) if scope else None
        if not key:
            return None
        return key.rsplit(":", 1)[-1].replace("_", " ").replace("-", " ").title()

    async def invalidate_visual_identity_actor(
        self, actor_kind: str, actor_id: int | str
    ) -> None:
        """Invalidate and refresh one actor after Visual Identity publication."""

        await self._refresh_character_avatar_fn(
            invalidate_actor=(str(actor_kind), str(actor_id))
        )

    async def invalidate_persona_visual_identity(self, persona_id: str) -> None:
        """Invalidate one Persona after its operational runtime changes."""

        await self._refresh_character_avatar_fn(
            invalidate_actor=("persona", str(persona_id))
        )

    def _visual_identity_request_context(
        self,
    ) -> tuple[tuple[str, str, str] | None, str, str | None]:
        """Snapshot the active actor, operational state, and manual key."""

        scope = self._current_visual_identity_actor_scope()
        store = self._console_chat_store
        session_id = getattr(store, "active_session_id", None) if store else None
        react = resolve_react_character_expressions(
            getattr(self.app_instance, "app_config", {}) or {}
        )
        state = resolve_console_expression_state(store, session_id, react_enabled=react)
        manual = self._manual_reaction_key(scope) if scope else None
        return scope, state, manual

    def _resolve_visual_identity(
        self,
        scope: tuple[str, str, str],
        requested_state: str,
        manual_expression_key: str | None,
    ) -> VisualIdentityResolution | None:
        """Resolve one reaction synchronously for an off-thread caller."""

        db = self._visual_identity_db_accessor()
        if db is None:
            return None
        return _resolve_visual_identity_for_db(
            db,
            scope,
            requested_state,
            manual_expression_key,
            (
                self._local_persona_visual_identity_service()
                if scope[1] == "persona"
                else None
            ),
        )

    def _resolve_historical_visual_identity(
        self,
        scope: tuple[str, str, str],
        identity: CharacterEmoteHistoryIdentity,
    ) -> VisualIdentityResolution | None:
        """Resolve a message's exact immutable character expression."""

        _session_id, actor_kind, actor_id = scope
        db = self._visual_identity_db_accessor()
        if (
            db is None
            or actor_kind != "character"
            or str(identity.actor_id) != actor_id
        ):
            return None
        try:
            return resolve_historical_visual_identity(
                db,
                actor_id=identity.actor_id,
                pack_id=identity.pack_id,
                pack_version_id=identity.pack_version_id,
                expression_key=identity.expression_key,
                expression_id=identity.expression_id,
                asset_id=identity.asset_id,
            )
        except (SQLiteError, TypeError, ValueError, OverflowError):
            logger.debug(
                "Console historical reaction resolution failed actor_id={}",
                actor_id,
            )
            return None

    def _visual_identity_options(
        self, scope: tuple[str, str, str]
    ) -> tuple[ReactionOption, ...]:
        """Return metadata-only options for one active local actor pack."""

        db = self._visual_identity_db_accessor()
        if db is None:
            return ()
        return _visual_identity_options_for_db(
            db,
            scope,
            (
                self._local_persona_visual_identity_service()
                if scope[1] == "persona"
                else None
            ),
        )

    async def _open_console_reaction_picker(self) -> None:
        """Query reaction metadata off-thread and open the owned picker."""

        context = self._visual_identity_request_context()
        scope = context[0]
        if scope is None:
            self.app_instance.notify(
                "Choose a local Character or Persona before selecting a reaction.",
                severity="warning",
            )
            return
        db = self._visual_identity_db_accessor()
        persona_service = (
            self._local_persona_visual_identity_service()
            if scope[1] == "persona"
            else None
        )
        if db is None:
            options = ()
        else:
            args = (
                (db, scope, persona_service) if scope[1] == "persona" else (db, scope)
            )
            options = await asyncio.to_thread(_visual_identity_options_for_db, *args)
        if (
            self._visual_identity_db_accessor() is not db
            or self._visual_identity_request_context() != context
            or (
                scope[1] == "persona"
                and self._local_persona_visual_identity_service() is not persona_service
            )
        ):
            return
        if not options:
            self.app_instance.notify(
                "This actor has no reaction pack.", severity="information"
            )
            return
        self.push_screen(
            ConsoleReactionPickerModal(
                options=options,
                message_target=self._screen,
                preview_callback=self._dispatch_console_reaction_preview,
                preview_cancel_callback=self._cancel_console_reaction_preview,
                selection_callback=self._dispatch_console_reaction_selection,
                clear_callback=self._dispatch_console_reaction_clear,
            )
        )

    def _dispatch_console_reaction_preview(
        self, option: ReactionOption, picker: ConsoleReactionPickerModal
    ) -> None:
        generation = getattr(self, "_reaction_preview_generation", 0) + 1
        self._reaction_preview_generation = generation
        picker_ref = weakref.ref(picker)
        self._reaction_preview_target_ref = picker_ref
        self._reaction_preview_worker = self.run_worker(
            self._preview_console_reaction(option, picker_ref, generation),
            group="console-reaction-preview",
            exclusive=True,
            exit_on_error=False,
        )

    def _cancel_console_reaction_preview(
        self, picker: ConsoleReactionPickerModal
    ) -> None:
        """Cancel work only for the picker that originally requested it."""

        target_ref = getattr(self, "_reaction_preview_target_ref", None)
        if target_ref is None or target_ref() is not picker:
            return
        self._reaction_preview_generation = (
            getattr(self, "_reaction_preview_generation", 0) + 1
        )
        self._reaction_preview_target_ref = None
        worker = getattr(self, "_reaction_preview_worker", None)
        self._reaction_preview_worker = None
        if worker is not None:
            worker.cancel()

    def _dispatch_console_reaction_selection(self, option: ReactionOption) -> None:
        self.run_app_worker(
            self._apply_console_reaction_selection(option),
            group="console-reaction-selection",
            exclusive=True,
            exit_on_error=False,
        )

    async def _apply_console_reaction_selection(self, option: ReactionOption) -> None:
        if await self._select_console_reaction(option):
            await self._refresh_character_avatar_fn()

    def _dispatch_console_reaction_clear(self) -> None:
        if self._clear_current_console_reaction():
            self.run_app_worker(
                self._refresh_character_avatar_fn(),
                group="console-reaction-selection",
                exclusive=True,
                exit_on_error=False,
            )

    async def _select_console_reaction(self, option: ReactionOption) -> bool:
        """Validate then replace the active actor's manual reaction key."""

        context = self._visual_identity_request_context()
        scope = context[0]
        if scope is None:
            return False
        db = self._visual_identity_db_accessor()
        if db is None:
            return False
        persona_service = (
            self._local_persona_visual_identity_service()
            if scope[1] == "persona"
            else None
        )
        args = (db, scope, persona_service) if scope[1] == "persona" else (db, scope)
        options = await asyncio.to_thread(_visual_identity_options_for_db, *args)
        if (
            self._visual_identity_db_accessor() is not db
            or self._visual_identity_request_context() != context
            or (
                scope[1] == "persona"
                and self._local_persona_visual_identity_service() is not persona_service
            )
        ):
            return False
        if option.expression_key not in {
            candidate.expression_key for candidate in options
        }:
            self.app_instance.notify(
                "That reaction is no longer available.", severity="error"
            )
            return False
        self._set_manual_reaction(scope, option.expression_key)
        return True

    def _clear_current_console_reaction(self) -> bool:
        """Return the active actor to automatic operational reactions."""

        scope = self._current_visual_identity_actor_scope()
        if scope is None:
            return False
        self._clear_manual_reaction(scope)
        return True

    async def _run_serialized_preview_sync(
        self, function: Callable[..., Any], *args: Any
    ) -> Any:
        """Delegate sync work to the current app's shared single-flight."""

        return await self._reaction_preview_coordinator_accessor().run_sync(
            function, *args
        )

    def _preview_request_is_current(
        self,
        *,
        generation: int,
        context: tuple[tuple[str, str, str] | None, str, str | None],
        db: object,
        expression_key: str,
        picker_ref: weakref.ReferenceType[ConsoleReactionPickerModal],
        persona_service: object | None = None,
    ) -> bool:
        picker = picker_ref()
        return (
            generation == getattr(self, "_reaction_preview_generation", 0)
            and self._visual_identity_db_accessor() is db
            and self._visual_identity_request_context() == context
            and (
                context[0] is None
                or context[0][1] != "persona"
                or self._local_persona_visual_identity_service() is persona_service
            )
            and picker is not None
            and picker.is_preview_current(expression_key)
        )

    def _apply_console_reaction_preview(
        self,
        picker_ref: weakref.ReferenceType[ConsoleReactionPickerModal],
        expression_key: str,
        renderable: object,
    ) -> None:
        picker = picker_ref()
        if picker is not None:
            picker.update_preview(expression_key, renderable)

    async def _preview_console_reaction(
        self,
        option: ReactionOption,
        picker_ref: weakref.ReferenceType[ConsoleReactionPickerModal],
        generation: int,
    ) -> None:
        """Resolve and decode the latest preview with serialized sync work."""

        context = self._visual_identity_request_context()
        scope, state, _manual = context
        db = self._visual_identity_db_accessor()
        persona_service = (
            self._local_persona_visual_identity_service()
            if scope is not None and scope[1] == "persona"
            else None
        )
        if (
            scope is None
            or db is None
            or not self._preview_request_is_current(
                generation=generation,
                context=context,
                db=db,
                expression_key=option.expression_key,
                picker_ref=picker_ref,
                persona_service=persona_service,
            )
        ):
            return

        options_args = (
            (db, scope, persona_service) if scope[1] == "persona" else (db, scope)
        )
        options = await self._run_serialized_preview_sync(
            _visual_identity_options_for_db, *options_args
        )
        if not self._preview_request_is_current(
            generation=generation,
            context=context,
            db=db,
            expression_key=option.expression_key,
            picker_ref=picker_ref,
            persona_service=persona_service,
        ):
            return
        if option.expression_key not in {
            candidate.expression_key for candidate in options
        }:
            self._apply_console_reaction_preview(
                picker_ref, option.expression_key, "Preview unavailable."
            )
            return

        resolution_args = (
            (db, scope, state, option.expression_key, persona_service)
            if scope[1] == "persona"
            else (db, scope, state, option.expression_key)
        )
        resolution = await self._run_serialized_preview_sync(
            _resolve_visual_identity_for_db, *resolution_args
        )
        if not self._preview_request_is_current(
            generation=generation,
            context=context,
            db=db,
            expression_key=option.expression_key,
            picker_ref=picker_ref,
            persona_service=persona_service,
        ):
            return
        if (
            resolution is None
            or resolution.resolved_expression_key != option.expression_key
            or not resolution.image_bytes
        ):
            self._apply_console_reaction_preview(
                picker_ref, option.expression_key, "Preview unavailable."
            )
            return

        identity = resolution.cache_identity
        _view, cache = self._ensure_console_image_view()
        cache_key = "visual-identity-preview:" + "|".join(identity)
        prepared = await self._run_serialized_preview_sync(
            cache.prepare, cache_key, resolution.image_bytes
        )
        if not self._preview_request_is_current(
            generation=generation,
            context=context,
            db=db,
            expression_key=option.expression_key,
            picker_ref=picker_ref,
            persona_service=persona_service,
        ):
            return
        if not prepared:
            self._apply_console_reaction_preview(
                picker_ref, option.expression_key, "Preview unavailable."
            )
            return

        renderable = await self._run_serialized_preview_sync(
            cache.get_pixels, cache_key
        )
        if not self._preview_request_is_current(
            generation=generation,
            context=context,
            db=db,
            expression_key=option.expression_key,
            picker_ref=picker_ref,
            persona_service=persona_service,
        ):
            return
        current = await self._run_serialized_preview_sync(
            _resolve_visual_identity_for_db, *resolution_args
        )
        if (
            not self._preview_request_is_current(
                generation=generation,
                context=context,
                db=db,
                expression_key=option.expression_key,
                picker_ref=picker_ref,
                persona_service=persona_service,
            )
            or current is None
            or current.cache_identity != identity
        ):
            return
        self._apply_console_reaction_preview(
            picker_ref,
            option.expression_key,
            renderable or "Preview unavailable.",
        )

    # -- Session switcher / rename -------------------------------------------

    def _open_console_session_rename_modal(self, session_id: str) -> None:
        """Open a modal for viewing and editing the active Console tab title."""
        store = self._ensure_console_chat_store()
        session = next(
            (candidate for candidate in store.sessions() if candidate.id == session_id),
            None,
        )
        if session is None:
            self.app_instance.notify(
                "Console tab is no longer available.", severity="error"
            )
            return

        def _apply_rename(result: str | None) -> None:
            if result is None:
                return
            try:
                _renamed, persisted = store.rename_session(session_id, result)
            except ValueError as exc:
                self.app_instance.notify(str(exc), severity="warning")
                return
            except KeyError:
                self.app_instance.notify(
                    "Console tab is no longer available.", severity="error"
                )
                return
            if not persisted:
                self.app_instance.notify(
                    "Renamed this tab, but saving the conversation title "
                    "failed — the stored conversation keeps its old name.",
                    severity="warning",
                )
            # TASK-251: a renamed session's persisted conversation title
            # must appear in the browser on the very next sync.
            self._invalidate_console_persisted_rows_cache()
            self.run_worker(
                self._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )

        self.push_screen(
            ConsoleRenameSessionModal(title=session.title),
            callback=_apply_rename,
        )

    def _fork_prefix_messages(
        self,
        store: ConsoleChatStore,
        fence: ConsoleForkFence,
    ) -> tuple[ConsoleChatMessage, ...] | None:
        """Re-resolve the captured prefix by identity without changing source state."""

        try:
            return tuple(
                store.get_message(entry.native_message_id) for entry in fence.lineage
            )
        except KeyError:
            return None

    def _fork_dialog_summary(
        self,
        fence: ConsoleForkFence,
        messages: Sequence[ConsoleChatMessage],
    ) -> ConsoleForkDialogSummary:
        """Build bounded presentation facts from one already-captured fence."""

        boundary = fence.lineage[-1]
        role_label = "User" if boundary.role is ConsoleMessageRole.USER else "Assistant"
        boundary_label = f"Through {role_label} {len(fence.lineage)}"
        if boundary.role is ConsoleMessageRole.USER:
            boundary_label += " · No reply will be generated"
        elif boundary.status == "stopped":
            boundary_label += " · Partial response"
        elif boundary.status == "failed":
            boundary_label += " · Failed partial response"

        response_variant = None
        if boundary.visible_variant_id and len(boundary.sibling_identity) > 1:
            try:
                index = boundary.sibling_identity.index(boundary.visible_variant_id) + 1
            except ValueError:
                index = 1
            response_variant = (
                f"showing response {index} of {len(boundary.sibling_identity)}"
            )

        temporary = fence.source_durability == "temporary"
        source_session = next(
            session
            for session in self._ensure_console_chat_store().sessions()
            if session.id == fence.source_session_id
        )
        if temporary:
            destination = "Temporary chat · Save later to keep it"
        elif source_session.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID:
            destination = "Saved chat · Chats"
        else:
            destination = (
                "Saved chat · "
                f"{self._workspace_display_name_fn(source_session.workspace_id)}"
            )
        return ConsoleForkDialogSummary(
            default_title=default_fork_title(fence.source_title),
            boundary_label=boundary_label,
            boundary_excerpt=_console_fork_excerpt(boundary.visible_content),
            message_count=len(fence.lineage),
            response_variant=response_variant,
            destination=destination,
            temporary=temporary,
            includes_attachments=any(message.attachments for message in messages),
            includes_citations=any(
                message.citation_presentation is not None for message in messages
            ),
            contains_video=any(
                message.video_metadata is not None for message in messages
            ),
        )

    def request_console_chat_fork(self, message_id: str) -> None:
        """Capture one boundary and open its presentation-only naming dialog."""

        if self._active_fork_request is not None:
            self.app_instance.notify(
                "Finish or close the current fork dialog first.",
                severity="warning",
            )
            return
        store = self._ensure_console_chat_store()
        try:
            session_id = store.session_id_for_message(message_id)
            prefix_ids = store.active_path_message_ids(session_id)
            prefix_ids = prefix_ids[: prefix_ids.index(message_id) + 1]
            prefix = tuple(store.get_message(item) for item in prefix_ids)
            image_selections = self._capture_fork_image_selections_fn(prefix)
            fence = store.issue_fork_fence(
                message_id,
                image_selections=image_selections,
            )
            title = default_fork_title(fence.source_title)
        except (KeyError, TypeError, ValueError) as exc:
            self.app_instance.notify(str(exc), severity="warning")
            return

        modal: ConsoleForkChatModal

        def submit(result: ConsoleForkSubmitResult) -> None:
            self._submit_console_chat_fork(modal, result)

        def cancel() -> None:
            self._cancel_console_chat_fork(modal)

        def open_existing() -> None:
            self._open_created_console_chat_fork(modal)

        modal = ConsoleForkChatModal(
            self._fork_dialog_summary(fence, prefix),
            on_submit=submit,
            on_cancel=cancel,
            on_open=open_existing,
        )
        self._active_fork_request = _ConsoleForkRequest(
            fence=fence,
            fork_session_id=str(uuid.uuid4()),
            fork_conversation_id=(
                None if fence.source_durability == "temporary" else str(uuid.uuid4())
            ),
            modal=modal,
            title=title,
        )
        self.push_screen(modal)

    def _cancel_console_chat_fork(self, modal: ConsoleForkChatModal) -> None:
        request = self._active_fork_request
        if request is None or request.modal is not modal:
            return
        self._fork_validation_generation += 1
        self._active_fork_request = None

    def _submit_console_chat_fork(
        self,
        modal: ConsoleForkChatModal,
        result: ConsoleForkSubmitResult,
    ) -> None:
        request = self._active_fork_request
        if request is None or request.modal is not modal:
            return
        if (
            request.snapshot is not None
            and not request.committed
            and request.snapshot.title != result.title
        ):
            request.snapshot = None
        request.title = result.title
        self._fork_validation_generation += 1
        generation = self._fork_validation_generation
        self.run_app_worker(
            self._run_console_chat_fork(request, generation),
            group="console-chat-fork",
            exit_on_error=False,
        )

    def _open_created_console_chat_fork(self, modal: ConsoleForkChatModal) -> None:
        request = self._active_fork_request
        if request is None or request.modal is not modal:
            return
        self.run_app_worker(
            self._recover_created_console_chat_fork(request),
            group="console-chat-fork-open",
            exit_on_error=False,
        )

    def _fork_request_is_current(
        self,
        request: _ConsoleForkRequest,
        generation: int,
    ) -> bool:
        return (
            self._active_fork_request is request
            and self._fork_validation_generation == generation
            and request.modal.state == "validating"
        )

    async def _run_fork_io(
        self,
        operation: Callable[..., Any],
        /,
        *args: object,
        **kwargs: object,
    ) -> Any:
        """Keep production SQLite work off-loop while supporting memory fixtures."""

        store = self._ensure_console_chat_store()
        db = getattr(store.persistence, "db", None)
        call = partial(operation, *args, **kwargs)
        if bool(getattr(db, "is_memory_db", False)):
            return call()
        return await asyncio.to_thread(call)

    @staticmethod
    def _fork_conversation_kwargs(
        snapshot: ConsoleChatForkSnapshot,
    ) -> dict[str, object]:
        configuration = snapshot.configuration
        global_scope = configuration.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID
        return {
            "conversation_title": snapshot.title,
            "scope_type": "global" if global_scope else "workspace",
            "workspace_id": None if global_scope else configuration.workspace_id,
            "system_prompt": configuration.settings.system_prompt,
            "runtime_backend": configuration.runtime_backend,
            "assistant_kind": configuration.assistant_kind,
            "assistant_id": configuration.assistant_id,
            "assistant_authority_id": configuration.assistant_authority_id,
            "persona_memory_mode": configuration.persona_memory_mode,
            "character_id": configuration.character_id,
            "character_name": configuration.character_name,
            "speech_preferences": configuration.speech_preferences,
            "thinking_history_policy": configuration.thinking_history_policy,
        }

    def _registered_fork_exists(
        self,
        store: ConsoleChatStore,
        request: _ConsoleForkRequest,
    ) -> bool:
        return any(
            session.id == request.fork_session_id
            and session.persisted_conversation_id == request.fork_conversation_id
            for session in store.sessions()
        )

    async def _register_console_chat_fork(
        self,
        store: ConsoleChatStore,
        request: _ConsoleForkRequest,
    ) -> bool:
        if request.registered or self._registered_fork_exists(store, request):
            request.registered = True
            return True
        snapshot = request.snapshot
        if snapshot is None:
            return False
        try:
            store.register_fork_snapshot(snapshot, activate=False)
        except Exception:  # noqa: BLE001 -- controller recovery boundary
            return False
        request.registered = True
        if request.projection_pending and request.fork_conversation_id is not None:
            store._pending_workspace_projections[request.fork_session_id] = (  # noqa: SLF001
                request.fork_conversation_id
            )
        return True

    async def _commit_durable_console_chat_fork(
        self,
        store: ConsoleChatStore,
        request: _ConsoleForkRequest,
    ) -> bool:
        snapshot = request.snapshot
        if snapshot is None:
            return False
        if request.committed:
            return True
        persistence = store.persistence
        try:
            result = await self._run_fork_io(
                persistence.fork_console_conversation_bundle,
                snapshot=snapshot,
                conversation_kwargs=self._fork_conversation_kwargs(snapshot),
                policy_candidate=snapshot.configuration.library_policy,
                project_context_json=encode_project_context_json(
                    snapshot.configuration.project_instruction_state
                ),
            )
        except Exception:  # noqa: BLE001 -- every ambiguous write must reconcile
            try:
                result = await self._run_fork_io(
                    persistence.resolve_console_fork_commit,
                    snapshot,
                )
            except Exception as exc:  # noqa: BLE001 -- collision fails closed
                collision = "collision" in str(exc).lower()
                request.modal.show_precommit_error(
                    (
                        "Fork identity conflict. Close this dialog and choose Fork again."
                        if collision
                        else "Fork could not be verified. Close this dialog and try again."
                    ),
                    retryable=not collision,
                )
                return False
            if result is None:
                request.modal.show_precommit_error(
                    "Fork could not be created. Check storage and retry."
                )
                return False
        if result is None:
            request.modal.show_precommit_error(
                "Fork could not be created. Check storage and retry."
            )
            return False
        request.committed = True

        if snapshot.configuration.workspace_id != CONSOLE_GLOBAL_WORKSPACE_ID:
            try:
                await self._run_fork_io(
                    persistence.project_workspace_membership,
                    snapshot.fork_conversation_id,
                )
            except Exception:  # noqa: BLE001 -- durable row remains authoritative
                request.projection_pending = True
        return True

    def _show_created_not_opened(
        self,
        request: _ConsoleForkRequest,
        detail: str,
    ) -> None:
        identity = request.fork_conversation_id or request.fork_session_id
        request.modal.show_created_not_opened(
            title=request.title,
            identity=identity,
            detail=detail,
        )

    async def _run_console_chat_fork(
        self,
        request: _ConsoleForkRequest,
        generation: int,
    ) -> None:
        barrier = getattr(self, "_fork_validation_barrier", None)
        if callable(barrier):
            pending = barrier(generation)
            if inspect.isawaitable(pending):
                await pending
        if not self._fork_request_is_current(request, generation):
            return

        store = self._ensure_console_chat_store()
        prefix = self._fork_prefix_messages(store, request.fence)
        if (
            prefix is None
            or not self._validate_fork_image_selections_fn(
                prefix,
                request.fence.image_selections,
            )
            or not store.validate_fork_fence(
                request.fence,
                image_selections=request.fence.image_selections,
            )
        ):
            request.modal.show_stale_source()
            return
        if not self._fork_request_is_current(request, generation):
            return
        if request.snapshot is None:
            try:
                request.snapshot = store.stage_fork_snapshot(
                    request.fence,
                    title=request.title,
                    fork_session_id=request.fork_session_id,
                    fork_conversation_id=request.fork_conversation_id,
                )
            except ValueError as exc:
                if "source changed" in str(exc).lower():
                    request.modal.show_stale_source()
                else:
                    request.modal.show_precommit_error(_console_fork_copy_failure(exc))
                return
        if not self._fork_request_is_current(request, generation):
            return

        request.modal.show_committing()
        snapshot = request.snapshot
        assert snapshot is not None
        if snapshot.durable:
            if not await self._commit_durable_console_chat_fork(store, request):
                return
        if not await self._register_console_chat_fork(store, request):
            if snapshot.durable and request.committed:
                self._show_created_not_opened(
                    request,
                    "The fork was created but could not be opened.",
                )
            else:
                request.modal.show_precommit_error(
                    "The temporary fork could not be created. Retry."
                )
            return
        await self._finish_opening_console_chat_fork(store, request)

    async def _finish_opening_console_chat_fork(
        self,
        store: ConsoleChatStore,
        request: _ConsoleForkRequest,
    ) -> None:
        try:
            if request.snapshot is not None and request.snapshot.durable:
                await store.hydrate_session_library_policy(request.fork_session_id)
            await self._activate_native_console_session(request.fork_session_id)
        except Exception:  # noqa: BLE001 -- registered target stays recoverable
            self._show_created_not_opened(
                request,
                "The fork was created but could not be opened.",
            )
            return
        opening_copy = (
            "Fork created and opened."
            if request.fork_conversation_id
            else "Temporary fork created and opened."
        )
        projection_copy = (
            " Workspace placement is pending and will be retried."
            if request.projection_pending
            else ""
        )
        self.app_instance.notify(
            f"{opening_copy} The original chat is still open. "
            f'"{request.title}" is active.{projection_copy}'
        )
        request.modal.close_after_success()
        self._active_fork_request = None

    async def _recover_created_console_chat_fork(
        self,
        request: _ConsoleForkRequest,
    ) -> None:
        if self._active_fork_request is not request:
            return
        store = self._ensure_console_chat_store()
        if not await self._register_console_chat_fork(store, request):
            self._show_created_not_opened(
                request,
                "The fork exists but is not available in Console yet.",
            )
            return
        if request.projection_pending:
            if await store.reconcile_pending_workspace_projection(
                request.fork_session_id
            ):
                request.projection_pending = False
        await self._finish_opening_console_chat_fork(store, request)

    async def _activate_native_console_session(self, session_id: str) -> None:
        """Activate a native Console session through the shared activation sequence.

        Set the active workspace, switch the native session, refresh the
        retrieval-scope display, await the UI sync, then force composer
        focus. Shared by the session-tab click handler, the Ctrl+K switcher
        callback, and Alt+1..9 tab-jump so all three entry points follow
        one activation path.

        Args:
            session_id: Native Console session id to activate.
        """
        controller = self._ensure_console_chat_controller()
        if controller.store.active_session_id != session_id:
            self._hide_console_activity_notice()
            self._capture_console_draft_switch_snapshot()
            self._note_console_follow_intent()
            self._set_active_workspace_for_session(session_id)
            controller.switch_session(session_id)
            # Finding C: a sub-agent drill-in is scoped to the conversation
            # active when the user drilled in -- clear it immediately on
            # switch here (the shared activation path for tab clicks,
            # Ctrl+K, and Alt+1..9) rather than rely solely on the rail
            # render path's own defensive re-check on the next sync.
            self._console_agent_drilldown_run_id = None
            # Task-13 review finding 2: this path activates an ALREADY-
            # resumed native session (unlike `_resume_console_workspace_
            # conversation`, which warms the cache itself), so `_console_
            # effective_scope_cache` may hold a stale entry -- e.g. a
            # workspace-scope edit made via `_apply_console_workspace_
            # scope_save` while a DIFFERENT session's tab was active only
            # refreshes that other (active) session's row/chip. Without
            # this call the row/chip would keep rendering the stale
            # snapshot indefinitely: recompose reads the cache, never the
            # DB (`_build_console_retrieval_scope_state`'s own contract).
            new_session = self._active_native_console_session()
            if new_session is not None:
                try:
                    await self._refresh_console_effective_scope_and_sync(new_session)
                except Exception:
                    logger.opt(exception=True).warning(
                        "Failed to refresh retrieval scope display on session "
                        "activation: {}",
                        session_id,
                    )
            await self._sync_native_console_chat_ui()
        self._focus_console_composer_if_needed(force=True)

    # -- Tab-strip press handling (wave-4 task 2) ---------------------------
    #
    # Two of `ChatScreen.on_button_pressed`'s 19 branches mutated nothing
    # but this cluster's state (the chat controller's session set, the
    # per-session undo histories), so their bodies moved here whole and the
    # screen's branches became calls. Each takes the session id the pressed
    # button's own id encodes rather than the `Button.Pressed` event:
    # Textual's event object stays on the screen -- it is the screen that
    # must `event.stop()` -- and the id parsing is one `removeprefix` that
    # belongs with the id string it mirrors.

    async def _close_console_session_tab(self, session_id: str) -> None:
        """Close one native session after one revision-pinned confirmation.

        Moved verbatim out of `ChatScreen.on_button_pressed`'s
        `console-close-session-tab-` branch (wave-4 task 2), the
        third-largest of its 19, and byte-for-byte -- including the
        `_evict_closing_session_images` closure, whose
        `self._ensure_console_image_view()` is now the injected accessor
        under the same name because the inline-image cluster stays
        screen-owned.

        Args:
            session_id: The session behind the pressed ``×``, parsed by the
                screen from the button id.
        """

        async def _complete_close() -> None:
            store = self._ensure_console_chat_store()
            try:
                closing_ids = [
                    message.id for message in store.messages_for_session(session_id)
                ]
            except KeyError:
                return
            _state, cache = self._ensure_console_image_view()
            cache.evict_session(closing_ids)
            self._ensure_console_chat_controller().close_session(session_id)
            self._clear_session_manual_reactions(session_id)
            self._console_undo_histories.pop(session_id, None)
            self._console_project_instruction_display_cache.pop(session_id, None)
            self._console_project_instruction_refresh_inflight.pop(session_id, None)
            self._console_project_instruction_refresh_completed.pop(session_id, None)
            await self._sync_native_console_chat_ui()

        while True:
            impact = self._session_close_impact(session_id)
            if impact is None:
                return
            if not impact.has_loss_risk:
                await _complete_close()
                return
            if not await self._confirm_session_close(impact):
                return
            current = self._session_close_impact(session_id)
            if current is None:
                return
            if current == impact:
                await _complete_close()
                return
            self.app_instance.notify(
                "Session activity changed; review the updated close impact.",
                severity="warning",
            )

    def start_close_console_session_tab(self, session_id: str) -> None:
        """Dispatch one non-blocking confirmation flow for ``session_id``."""

        if not session_id or session_id in self._closing_session_requests:
            return
        self._closing_session_requests.add(session_id)

        async def _run() -> None:
            try:
                await self._close_console_session_tab(session_id)
            finally:
                self._closing_session_requests.discard(session_id)

        close_flow = _run()
        try:
            self.run_worker(
                close_flow,
                group=f"console-close-session:{session_id}",
                exclusive=True,
                exit_on_error=False,
            )
        except Exception:
            close_flow.close()
            self._closing_session_requests.discard(session_id)
            # TASK-15103: raw logger — the ledgered contract for this event
            # carries no bound fields, including the file-level module bind.
            loguru_logger.warning("Could not start Console session close flow")

    def _session_close_impact(
        self, session_id: str
    ) -> ConsoleSessionCloseImpact | None:
        """Capture tree-wide transcript and controller loss for one session."""

        store = self._ensure_console_chat_store()
        try:
            messages = store.all_messages_for_session(session_id)
        except KeyError:
            return None
        controller = self._ensure_console_chat_controller()
        return ConsoleSessionCloseImpact(
            session_id=session_id,
            transcript_message_count=sum(
                message.persisted_message_id is None for message in messages
            ),
            lifecycle=controller.lifecycle_impact(session_id=session_id),
        )

    async def _await_confirmation(self, dialog: Any) -> bool:
        """Await a modal from a worker so application input remains routable."""

        worker = self.run_worker(
            self.push_screen_wait(dialog),
            exclusive=False,
            exit_on_error=False,
        )
        return bool(await worker.wait())

    async def _confirm_session_close(self, impact: ConsoleSessionCloseImpact) -> bool:
        from ...Widgets.confirmation_dialog import ConfirmationDialog

        lifecycle = impact.lifecycle
        dialog = ConfirmationDialog(
            title="Close Console session?",
            message=(
                "Closing this session will discard or cancel:\n\n"
                f"Transcript messages: {impact.transcript_message_count}\n"
                f"Live agent turns: {lifecycle.live_run_count}\n"
                f"Unsent queued prompts: {lifecycle.unsent_prompt_count}\n\n"
                "Close this session?"
            ),
            confirm_label="Close",
            cancel_label="Stay",
        )
        return await self._await_confirmation(dialog)

    async def _confirm_fleet_loss(self, controller: Any, *, quitting: bool) -> bool:
        """Confirm one revision-stable Console fleet loss snapshot."""

        from ...Widgets.confirmation_dialog import ConfirmationDialog

        while True:
            impact = controller.lifecycle_impact()
            if not impact.has_loss_risk:
                return True
            action = "Quitting Chatbook" if quitting else "Leaving Console"
            question = "Quit Chatbook?" if quitting else "Leave Console?"
            confirm_label = "Quit" if quitting else "Leave"
            dialog = ConfirmationDialog(
                title=question,
                message=(
                    f"{action} will cancel or discard:\n\n"
                    f"Live agent runs: {impact.live_run_count}\n"
                    f"Sessions with queued prompts: {impact.queued_session_count}\n"
                    f"Unsent queued prompts: {impact.unsent_prompt_count}\n\n"
                    f"{question}"
                ),
                confirm_label=confirm_label,
                cancel_label="Stay",
            )
            if not await self._await_confirmation(dialog):
                return False
            current = controller.lifecycle_impact()
            if current == impact:
                return True
            self.app_instance.notify(
                "Console activity changed; review the updated impact.",
                severity="warning",
            )

    async def confirm_navigation(self, controller: Any) -> bool:
        """Confirm revision-stable Console loss before navigation."""

        return await self._confirm_fleet_loss(controller, quitting=False)

    async def confirm_quit(self, controller: Any) -> bool:
        """Confirm revision-stable Console loss before application quit."""

        return await self._confirm_fleet_loss(controller, quitting=True)

    async def _handle_console_session_tab_press(self, session_id: str) -> None:
        """Activate a tab, or rename it when it is already the active one.

        Moved verbatim out of `ChatScreen.on_button_pressed`'s
        `console-session-tab-` branch (wave-4 task 2). Small, but it
        reaches nothing outside this cluster: the store's active session,
        this controller's rename modal, and this controller's activation
        path.

        Args:
            session_id: The session behind the pressed tab, parsed by the
                screen from the button id.
        """
        controller = self._ensure_console_chat_controller()
        if controller.store.active_session_id == session_id:
            self._open_console_session_rename_modal(session_id)
            return
        await self._activate_native_console_session(session_id)

    async def _apply_console_switcher_choice(
        self, choice: ConsoleSwitcherChoice | None
    ) -> None:
        """Apply one authority-bound switcher selection without target inference.

        Args:
            choice: Switcher result, or ``None`` if the switcher was cancelled.
        """
        if choice is None:
            return
        entry = choice.entry
        if choice.kind == "mark_seen" and isinstance(entry, UnavailableSessionNotice):
            await self._mark_unavailable_switcher_notice_seen(entry)
            return
        if not isinstance(entry, ConsoleSwitcherEntry) or entry.target is None:
            self._notify_stale_switcher_target()
            return
        target = entry.target
        if not self._switcher_target_authority_is_current(target):
            self._notify_stale_switcher_target()
            return

        if choice.kind == "rename":
            if (
                target.kind is SwitcherTargetKind.NATIVE_SESSION
                and target.session_id == entry.native_session_id
                and self._native_switcher_destination_exists(target.session_id)
            ):
                self._open_console_session_rename_modal(target.session_id)
            else:
                self._notify_stale_switcher_target()
            return
        if choice.kind != "activate":
            return

        activated = False
        if target.kind is SwitcherTargetKind.NATIVE_SESSION:
            if (
                target.session_id != entry.native_session_id
                or not self._native_switcher_destination_exists(target.session_id)
            ):
                self._notify_stale_switcher_target()
                return
            await self._activate_native_console_session(target.session_id)
            activated = self._native_switcher_destination_is_current(target.session_id)
        elif target.kind is SwitcherTargetKind.PERSISTED_CONVERSATION:
            if (
                target.conversation_id != entry.conversation_id
                or not target.conversation_id
            ):
                self._notify_stale_switcher_target()
                return
            self._hide_console_activity_notice()
            resumed = await self._resume_workspace_conversation(
                target.conversation_id,
                target_scope_type=target.scope_type or None,
                target_workspace_id=target.workspace_id,
            )
            if resumed is False:
                # TASK-717: record missing - same honest feedback and broken
                # marking as the rail row path (resume no longer self-toasts
                # for this failure class).
                self._mark_console_conversation_row_broken(target.conversation_id)
                self.app_instance.notify(
                    "This saved conversation could not be loaded - "
                    "its record is missing.",
                    severity="warning",
                )
                return
            activated = resumed is True and (
                self._current_console_conversation_id() == target.conversation_id
            )
        if not activated:
            self._notify_stale_switcher_target()
            return
        self._show_console_activity_notice(entry)

    def _switcher_target_authority_is_current(self, target: Any) -> bool:
        """Return whether an immutable target still belongs to this runtime."""
        try:
            profile, token = self._switcher_authority_accessor()
        except Exception:  # noqa: BLE001 - stale selection must fail closed
            return False
        return bool(
            target.profile_authority == profile and target.authority_token == token
        )

    def _native_switcher_destination_exists(self, session_id: str | None) -> bool:
        """Return whether the exact native destination still exists."""
        if not session_id:
            return False
        store = self._console_chat_store
        return bool(
            store is not None
            and any(session.id == session_id for session in store.sessions())
        )

    def _native_switcher_destination_is_current(self, session_id: str) -> bool:
        """Return whether the exact native destination owns the visible surface."""
        store = self._console_chat_store
        return bool(store is not None and store.active_session_id == session_id)

    def _notify_stale_switcher_target(self) -> None:
        """Explain a failed-closed stale selection without guessing a fallback."""
        self.app_instance.notify(
            "This switcher result is no longer available. Reopen Ctrl+K to refresh.",
            severity="warning",
        )

    def _console_activity_notice(self) -> ConsoleActivityOutcomeNotice | None:
        """Return the mounted destination notice through the named DOM seam."""
        try:
            surface = self._session_surface_accessor()
            if surface is None or not surface.is_mounted:
                return None
            return surface.query_one(
                "#console-activity-outcome-notice",
                ConsoleActivityOutcomeNotice,
            )
        except (AttributeError, QueryError):
            return None

    def _hide_console_activity_notice(self) -> None:
        """Invalidate any destination evidence owned by the previous tab."""
        notice = self._console_activity_notice()
        if notice is not None and notice.presentation is not None:
            notice.hide()

    def _show_console_activity_notice(self, entry: ConsoleSwitcherEntry) -> None:
        """Show frozen result evidence and schedule exact success acknowledgement."""
        target = entry.target
        if target is None or not target.receipts:
            return
        store = self._console_chat_store
        active_session_id = store.active_session_id if store is not None else None
        if not active_session_id:
            return
        notice = self._console_activity_notice()
        if notice is None:
            return
        presentation = ConsoleActivityOutcomePresentation(
            title=entry.title,
            profile_authority=target.profile_authority,
            authority_token=target.authority_token,
            session_id=active_session_id,
            conversation_id=target.conversation_id,
            receipts=target.receipts,
        )
        notice.set_mark_seen_handler(self._mark_console_activity_seen)
        generation = notice.show(presentation)
        notice.call_after_refresh(
            self._acknowledge_painted_console_activity,
            presentation,
            generation,
        )

    def _console_activity_presentation_is_current(
        self,
        notice: ConsoleActivityOutcomeNotice,
        presentation: ConsoleActivityOutcomePresentation,
        generation: int,
    ) -> bool:
        """Revalidate authority, destination, mount, visibility, and generation."""
        if not notice.is_mounted or not notice.is_current(generation, presentation):
            return False
        try:
            profile, token = self._switcher_authority_accessor()
        except Exception:  # noqa: BLE001 - acknowledgement must fail closed
            return False
        if (
            presentation.profile_authority != profile
            or presentation.authority_token != token
            or not presentation.session_id
            or not self._native_switcher_destination_is_current(presentation.session_id)
        ):
            return False
        return bool(
            presentation.conversation_id is None
            or self._current_console_conversation_id() == presentation.conversation_id
        )

    def _acknowledge_painted_console_activity(
        self,
        presentation: ConsoleActivityOutcomePresentation,
        generation: int,
    ) -> None:
        """Acknowledge captured successes only after their exact notice paints."""
        notice = self._console_activity_notice()
        if notice is None or not self._console_activity_presentation_is_current(
            notice, presentation, generation
        ):
            return
        activity_ids = tuple(
            receipt.activity_id
            for receipt in presentation.receipts
            if receipt.status == "done"
        )
        if not activity_ids:
            return
        service = getattr(self._console_runtime_accessor(), "activity_receipts", None)

        async def acknowledge_after_paint() -> None:
            try:
                updated = (
                    await asyncio.to_thread(service.acknowledge, activity_ids)
                    if service is not None
                    else 0
                )
            except Exception:  # noqa: BLE001 - leave unseen and expose exact retry
                logger.opt(exception=True).warning(
                    "Failed to acknowledge painted Console activity"
                )
                updated = 0
            current_notice = self._console_activity_notice()
            if (
                current_notice is not notice
                or not self._console_activity_presentation_is_current(
                    notice, presentation, generation
                )
            ):
                return
            if (
                service is None
                or bool(getattr(service, "degraded", False))
                or updated < len(activity_ids)
            ):
                notice.require_mark_seen(generation)

        self.run_worker(
            acknowledge_after_paint(),
            exclusive=False,
            group=f"console-activity-ack:{generation}",
            exit_on_error=False,
        )

    async def _mark_console_activity_seen(
        self,
        presentation: ConsoleActivityOutcomePresentation,
        generation: int,
    ) -> bool:
        """Acknowledge only the explicit notice's frozen receipt identities."""
        notice = self._console_activity_notice()
        if notice is None or not self._console_activity_presentation_is_current(
            notice, presentation, generation
        ):
            return False
        service = getattr(self._console_runtime_accessor(), "activity_receipts", None)
        if service is None:
            return False
        activity_ids = tuple(
            receipt.activity_id
            for receipt in presentation.receipts
            if notice.should_retry_all(generation) or receipt.status != "done"
        )
        if not activity_ids:
            return False
        try:
            updated = await asyncio.to_thread(service.acknowledge, activity_ids)
        except Exception:  # noqa: BLE001 - explicit retry stays visible
            logger.opt(exception=True).warning("Failed to mark Console activity seen")
            return False
        current_notice = self._console_activity_notice()
        if (
            current_notice is not notice
            or not self._console_activity_presentation_is_current(
                notice, presentation, generation
            )
        ):
            return False
        return bool(
            updated >= len(activity_ids)
            and not bool(getattr(service, "degraded", False))
        )

    async def _mark_unavailable_switcher_notice_seen(
        self, notice: UnavailableSessionNotice
    ) -> None:
        """Acknowledge one unavailable destination's receipts without navigation."""
        try:
            profile, token = self._switcher_authority_accessor()
        except Exception:  # noqa: BLE001 - stale action must fail closed
            self._notify_stale_switcher_target()
            return
        if notice.profile_authority != profile or notice.authority_token != token:
            self._notify_stale_switcher_target()
            return
        service = getattr(self._console_runtime_accessor(), "activity_receipts", None)
        if service is None:
            self._notify_stale_switcher_target()
            return
        activity_ids = tuple(receipt.activity_id for receipt in notice.receipts)
        try:
            updated = await asyncio.to_thread(service.acknowledge, activity_ids)
        except Exception:  # noqa: BLE001 - receipt remains safely unseen
            logger.opt(exception=True).warning(
                "Failed to mark unavailable Console activity seen"
            )
            self.app_instance.notify(
                "Activity could not be marked seen. Reopen Ctrl+K and retry.",
                severity="warning",
            )
            return
        try:
            current_profile, current_token = self._switcher_authority_accessor()
        except Exception:  # noqa: BLE001 - no stale post-write UI
            return
        if (
            current_profile != notice.profile_authority
            or current_token != notice.authority_token
        ):
            return
        if updated < len(activity_ids) or bool(getattr(service, "degraded", False)):
            self.app_instance.notify(
                "Activity could not be marked seen. Reopen Ctrl+K and retry.",
                severity="warning",
            )

    def _refresh_console_library_policy_defaults(self) -> None:
        """Load the defaults captured by the next locally created session."""
        app_config = self._provider_readiness_app_config()
        console_config = (
            app_config.get("console", {}) if isinstance(app_config, Mapping) else {}
        )
        if not isinstance(console_config, Mapping):
            console_config = {}
        chat_defaults = (
            app_config.get("chat_defaults", {})
            if isinstance(app_config, Mapping)
            else {}
        )
        if not isinstance(chat_defaults, Mapping):
            chat_defaults = {}
        self._ensure_console_chat_store().set_library_policy_defaults(
            ConsoleLibraryPolicyDefaults(
                auto_retrieve=(
                    ConsoleAutoRetrieve.AUTOMATIC
                    if coerce_bool_setting(
                        chat_defaults.get("rag_auto_retrieve_on_send", False),
                        False,
                    )
                    else ConsoleAutoRetrieve.NEVER
                ),
                assistant_access=(
                    ConsoleAssistantLibraryAccess.ALLOWED
                    if coerce_bool_setting(
                        console_config.get(
                            "assistant_library_access_default",
                            False,
                        ),
                        False,
                    )
                    else ConsoleAssistantLibraryAccess.BLOCKED
                ),
            )
        )

    async def _create_native_console_session_from_active_context(
        self, *, ephemeral: bool = False
    ) -> None:
        """Create and focus a native Console session in the active workspace context.

        Args:
            ephemeral: Create the session temporary (never saved locally).
        """
        # TASK-339: new_session activates the fresh session inline; snapshot
        # first so the deferred draft swap attributes settle-window typing
        # to the new tab instead of clobbering it.
        self._capture_console_draft_switch_snapshot()
        self._refresh_console_library_policy_defaults()
        # Task 9 (workspace assistant defaults): a plain new tab in an
        # explicit workspace starts as the workspace default persona's
        # session. Settings/default-persona selection lives in the helper
        # below so published-default provenance is testable without a live
        # screen.
        settings, assistant_kwargs = self._new_session_startup_settings()
        self._ensure_console_chat_controller().new_session(
            settings=settings,
            canonical_settings_baseline=settings,
            new_chat_default_generation=(self._console_new_chat_default_generation()),
            ephemeral=ephemeral,
            **assistant_kwargs,
        )
        # TASK-251: new-chat-tab handler -- invalidate so the browser's
        # "selected" row indicator picks up the new active session promptly.
        self._invalidate_console_persisted_rows_cache()
        await self._sync_native_console_chat_ui()
        # Task-7: this is the shared activation path for every "new session"
        # entry point (plain Ctrl+T, "New Temporary" via the palette/tab-strip
        # button, and the other internal callers below) --
        # `_sync_native_console_chat_ui()` above
        # never touches `#console-temporary-chip` (same reason it never
        # touches the scope chip; see `_sync_console_retrieval_scope_row`).
        # Without this push a freshly created temporary tab would render
        # with no marker at all until some unrelated event (a tab switch
        # away and back) happened to resync it -- the chip's entire purpose
        # is to be visible the moment the tab exists. It also clears a
        # stale "Temporary" chip left over when a new ordinary chat is
        # created right after a temporary one.
        self._sync_console_temporary_chip()
        self._focus_console_composer_if_needed(force=True)

    # -- Per-session settings -------------------------------------------------

    def _active_console_session_settings(self) -> ConsoleSessionSettings | None:
        """Return settings for the active native Console session, if one exists."""
        store = self._console_chat_store
        if store is None or store.active_session_id is None:
            return None
        try:
            return store.session_settings(store.active_session_id)
        except KeyError:
            return None

    def _console_session_settings(
        self, session_id: str
    ) -> ConsoleSessionSettings | None:
        """Return settings for an owning session without switching the UI."""
        store = self._console_chat_store
        if store is None:
            return None
        try:
            return store.session_settings(session_id)
        except KeyError:
            return None

    def _resolve_turn_tool_policy_profile_id(self, workspace_id: str | None) -> str:
        """Resolve the workspace's named tool-permission profile id.

        Workspace assistant defaults (Task 7): the owning session's
        workspace may pin a ``tool_policy_profile_id`` in its assistant
        defaults; that profile is what THIS turn's tool gates resolve
        under. Any absence -- no workspace, no registry, no workspace
        record, no defaults, empty id -- degrades to ``"default"``, the
        single-profile behavior. Never raises.
        """
        try:
            registry_service = getattr(
                self.app_instance, "workspace_registry_service", None
            )
            if not workspace_id or registry_service is None:
                return "default"
            record = registry_service.get_workspace(workspace_id)
            defaults = getattr(record, "assistant_defaults", None) if record else None
            profile_id = getattr(defaults, "tool_policy_profile_id", None)
            if isinstance(profile_id, str) and profile_id.strip():
                return profile_id.strip()
        except Exception as exc:  # noqa: BLE001 -- posture degrades, never blocks
            logger.warning(
                "Console turn context: tool policy profile resolution failed; "
                "using the default profile; error_type={}",
                type(exc).__name__,
            )
        return "default"

    def _resolve_turn_persona_policy_rules(
        self, session_id: str
    ) -> tuple[Mapping[str, Any], ...]:
        """Resolve the owning session's persona policy rules.

        Workspace assistant defaults (Task 7): only a session whose durable
        assistant identity is a persona carries rules -- the session record's
        ``assistant_kind == "persona"`` resolves ``assistant_id`` through the
        app's local persona service (``get_persona_profile``), whose view
        already normalizes ``policy_rules``. Every failure (no store, no
        session, non-persona assistant, unknown persona, malformed rules)
        degrades to ``()`` -- the identity posture. Never raises.
        """
        try:
            store = self._console_chat_store
            if store is None:
                return ()
            session = next(
                (item for item in store.sessions() if item.id == session_id), None
            )
            if session is None:
                return ()
            if session.assistant_kind != "persona":
                return ()
            assistant_id = str(session.assistant_id or "").strip()
            if not assistant_id:
                return ()
            service = getattr(
                self.app_instance, "local_character_persona_service", None
            )
            if service is None:
                return ()
            profile = service.get_persona_profile(assistant_id)
            rules = (
                profile.get("policy_rules") if isinstance(profile, Mapping) else None
            )
            if isinstance(rules, (list, tuple)):
                return tuple(rule for rule in rules if isinstance(rule, Mapping))
        except Exception as exc:  # noqa: BLE001 -- posture degrades, never blocks
            logger.warning(
                "Console turn context: persona policy rules resolution failed; "
                "running with no persona rules; error_type={}",
                type(exc).__name__,
            )
        return ()

    def _workspace_default_for_new_session(
        self,
    ) -> tuple[str, str, str, str] | None:
        """Resolve the active workspace's default persona for a NEW session.

        Workspace assistant defaults (Task 9): explicit workspaces may pin a
        default persona; a plain new Console tab in such a workspace starts
        as that persona's session. Returns
        ``(assistant_id, label, system_prompt, memory_mode)`` only when the
        active workspace is explicit (never the global console workspace or
        the built-in Default workspace), not archived, and its effective
        default resolves ``available``. Every absence or failure degrades to
        ``None`` (plain session) -- never raises. Handoff/character paths do
        not consult this helper; their explicit choices outrank the default.
        """
        try:
            registry = getattr(self.app_instance, "workspace_registry_service", None)
            personas = getattr(
                self.app_instance, "local_character_persona_service", None
            )
            if registry is None or personas is None:
                return None
            store = self._ensure_console_chat_store()
            workspace_id = getattr(
                getattr(store, "workspace_context", None),
                "active_workspace_id",
                None,
            )
            if not workspace_id or workspace_id in (
                CONSOLE_GLOBAL_WORKSPACE_ID,
                DEFAULT_WORKSPACE_ID,
            ):
                return None
            workspace = registry.get_workspace(workspace_id)
            if workspace is None or getattr(workspace, "archived", False):
                return None
            # Lazy import (boot budget, ADR-097): per-turn resolution only.
            from ...Workspaces.assistant_defaults import (
                resolve_effective_assistant_default,
            )

            effective = resolve_effective_assistant_default(
                getattr(workspace, "assistant_defaults", None),
                lambda pid: _safe_persona_lookup(personas, pid),
            )
            if effective.status != "available" or not effective.assistant_id:
                return None
            record = _safe_persona_lookup(personas, effective.assistant_id)
            if record is None:
                return None
            prompt = build_persona_agent_system_prompt(record)
            return (
                effective.assistant_id,
                effective.label or "Workspace Agent",
                prompt,
                effective.persona_memory_mode or "read_only",
            )
        except Exception as exc:  # noqa: BLE001 -- workspace defaults degrade, never block
            logger.warning(
                "Console session startup: workspace default persona "
                "resolution failed; starting a plain session; error_type={}",
                type(exc).__name__,
            )
            return None

    def _new_session_startup_settings(
        self,
    ) -> "tuple[ConsoleSessionSettings | None, dict[str, str]]":
        """Pick the settings + assistant identity a plain new tab starts with.

        Per ADR-095, Ctrl+T and temporary chats are eligible blank-chat
        creation paths: they start from the app-published new-chat defaults
        and never clone the active session. An explicit workspace's available
        default persona is then stamped onto that snapshot per ADR-079.
        Duplicate, branch, continue, character, and handoff paths carry their
        own source settings without routing through this helper. Never raises.
        """
        try:
            settings = self._blank_console_session_settings()
            if (
                settings is not None
                and settings.system_prompt is None
                and not settings.character_label
                and settings.persona_memory_mode is None
            ):
                workspace_default = self._workspace_default_for_new_session()
                if workspace_default is not None:
                    (
                        default_assistant_id,
                        default_label,
                        default_prompt,
                        default_memory_mode,
                    ) = workspace_default
                    return (
                        replace(
                            settings,
                            system_prompt=default_prompt,
                            character_label=default_label,
                            persona_memory_mode=default_memory_mode,
                        ),
                        {
                            "assistant_kind": "persona",
                            "assistant_id": default_assistant_id,
                            "assistant_label": default_label,
                        },
                    )
            return settings, {}
        except Exception as exc:  # noqa: BLE001 -- startup degrades, never blocks
            logger.warning(
                "Console session startup: new-session settings selection "
                "failed; falling back to plain defaults; error_type={}",
                type(exc).__name__,
            )
            try:
                return self._blank_console_session_settings(), {}
            except Exception:  # noqa: BLE001 -- last-resort plain session
                return None, {}

    def _build_console_turn_execution_context(
        self, session_id: str
    ) -> ConsoleTurnConfigurationSnapshot:
        """Capture one detached configuration snapshot for an owning session."""
        from ...Chat.attachment_core import max_history_images
        from ..Screens.settings_library_rag_defaults import (
            load_direct_library_tools,
        )
        from ...model_capabilities import is_vision_capable

        app_config = self._provider_readiness_app_config()
        selection = self._build_provider_selection_fn(session_id)
        settings = self._console_session_settings(session_id)
        model = selection.explicit_model or selection.configured_model
        console_config = (
            app_config.get("console", {}) if isinstance(app_config, Mapping) else {}
        )
        if not isinstance(console_config, Mapping):
            console_config = {}
        workspace_id = self._ensure_console_chat_store().session_workspace_id(
            session_id
        )
        workspace_roots = ()
        ready_review_aliases = ()
        skipped_review_roots = ()
        consent_service = getattr(
            self.app_instance,
            "change_review_consent_service",
            None,
        )
        if consent_service is not None:
            try:
                admission = consent_service.admit_turn(workspace_id)
                workspace_roots = tuple(admission.ready_roots)
                ready_review_aliases = tuple(getattr(admission, "ready_aliases", ()))
                skipped_review_roots = tuple(admission.skipped_roots)
            except Exception:  # noqa: BLE001 -- review never blocks a send
                workspace_roots = ()
                ready_review_aliases = ()
                skipped_review_roots = ()
        # Workspace assistant defaults (Task 7): this turn's tool posture --
        # the workspace's named permission profile (absent/Default/global
        # defaults degrade to "default") and the owning session's persona
        # policy rules. Every failure degrades to the identity posture
        # rather than blocking the send; posture is narrowing-only, so a
        # degraded read can never widen access.
        tool_policy_profile_id = self._resolve_turn_tool_policy_profile_id(workspace_id)
        persona_policy_rules = self._resolve_turn_persona_policy_rules(session_id)

        return ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=selection,
            scratch_space=self._scratch_snapshot_provider(session_id),
            session_settings=settings,
            workspace_roots=workspace_roots,
            change_review_root_aliases=ready_review_aliases,
            change_review_skipped_roots=skipped_review_roots,
            persona_policy_rules=persona_policy_rules,
            tool_policy_profile_id=tool_policy_profile_id,
            capabilities={
                "vision": bool(model)
                and is_vision_capable(selection.provider, model or ""),
                "max_history_images": max_history_images(selection.provider, model),
            },
            rag_defaults={
                "source_types": tuple(self._rag_source_types_accessor()),
                "top_k": self._rag_top_k_accessor(),
            },
            tool_configuration={
                "agent_runtime_enabled": coerce_bool_setting(
                    console_config.get("agent_runtime", True),
                    True,
                ),
                "native_tool_calls_enabled": coerce_bool_setting(
                    console_config.get("native_tool_calls", True),
                    True,
                ),
                "local_tools_enabled": coerce_bool_setting(
                    console_config.get("local_tools_enabled", False),
                    False,
                ),
                "direct_library_tools": load_direct_library_tools(app_config),
            },
            provider_payload_settings={
                "streaming": selection.streaming,
                "temperature": selection.temperature,
                "top_p": selection.top_p,
                "min_p": selection.min_p,
                "top_k": selection.top_k,
                "max_tokens": selection.max_tokens,
                "seed": selection.seed,
                "presence_penalty": selection.presence_penalty,
                "frequency_penalty": selection.frequency_penalty,
                "reasoning_effort": selection.reasoning_effort,
                "reasoning_summary": selection.reasoning_summary,
                "verbosity": selection.verbosity,
                "thinking_effort": selection.thinking_effort,
                "thinking_budget_tokens": selection.thinking_budget_tokens,
            },
        )

    def _default_console_session_settings(self) -> ConsoleSessionSettings:
        """Build the default settings snapshot for a new native Console session."""
        provider, model = self._effective_console_provider_model()
        return default_console_session_settings(
            self._provider_readiness_app_config(),
            str(provider).strip() if _has_selected_text(provider) else None,
            str(model).strip() if _has_selected_text(model) else None,
        )

    def _blank_console_session_settings(self) -> ConsoleSessionSettings:
        """Build config-owned defaults for an eligible blank Console chat."""
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

    def _ensure_active_console_session_settings(self) -> ConsoleSessionSettings:
        """Ensure the active native Console session owns a settings snapshot."""
        store = self._ensure_console_chat_store()
        creating_blank_session = store.active_session_id is None
        defaults = self._blank_console_session_settings()
        # An ID-only saved-conversation resume is the authoritative startup
        # intent. Compose still needs settings to paint before its ordered
        # async opener runs, but creating a tab here would both leave an
        # orphan bootstrap session and let a global conversation inherit the
        # registry-active workspace through hydration's context fallback.
        if store.active_session_id is None and bool(
            getattr(
                self._screen,
                "_console_ordered_resume_pending",
                lambda: False,
            )()
        ):
            return defaults
        workspace_id = store.workspace_context.active_workspace_id
        # TASK-26839: `ensure_session` uses `title` only when it CREATES a
        # session; with one active the argument is discarded. Deriving the
        # workspace title is a synchronous registry `get_workspace` SQLite
        # query, and this method runs on every provider/model display
        # rebuild -- the in-terminal probe sampled that discarded lookup on
        # the main thread in three separate sessions. Compute it only for
        # the creation case this method already knows about.
        session = store.ensure_session(
            title=(
                self._workspace_initial_session_title(workspace_id)
                if creating_blank_session
                else DEFAULT_CONSOLE_SESSION_TITLE
            ),
            workspace_id=workspace_id,
            settings=defaults,
            canonical_settings_baseline=defaults,
        )
        if creating_blank_session:
            session.new_chat_default_generation = (
                self._console_new_chat_default_generation()
            )
        resolved = self._maybe_refresh_stale_default_console_settings(store, session)
        if resolved is None:
            raise RuntimeError(
                "Console session predates the published defaults without "
                "creation-time settings provenance."
            )
        return resolved

    def _maybe_refresh_stale_default_console_settings(
        self,
        store: ConsoleChatStore,
        session: ConsoleChatSession,
    ) -> ConsoleSessionSettings | None:
        """Re-derive default-sourced settings for blocked, never-used sessions.

        First-run sessions snapshot template defaults (e.g. OpenAI without a
        key) and that snapshot survives navigation via screen-state restore.
        When the user then configures a working provider in Settings, an empty
        session the user never explicitly configured must converge on the new
        defaults instead of keeping the setup card blocked until restart
        (task-177 live regression). Sessions with user work, settings that no
        longer equal their explicit canonical baseline, any messages, or
        already-sendable settings are never touched; stale defaults are only
        replaced when the re-derived defaults are actually send-capable.
        """
        settings = session.settings
        if (
            session.new_chat_default_generation
            < self._console_new_chat_default_generation()
        ):
            return settings
        if settings is None:
            settings = self._blank_console_session_settings()
            store.replace_session_settings(
                session.id,
                settings,
                mark_user_work=False,
                canonical_settings_baseline=settings,
            )
            return settings
        if session.has_user_work or session.canonical_settings_baseline != settings:
            return settings
        try:
            if store.messages_for_session(session.id):
                return settings
        except KeyError:
            return settings
        app_config = self._provider_readiness_app_config()
        current_readiness = build_console_settings_readiness(
            settings, app_config=app_config
        )
        if current_readiness.native_send_supported:
            return settings
        if current_readiness.label not in self._CONSOLE_REFRESHABLE_BLOCKED_LABELS:
            # Unknown/WIP providers are a provider *choice* problem, not a
            # config-fixable credential/endpoint gap; never override choice.
            return settings
        # Creation reads the app-owned published snapshot so an in-flight
        # Make Default cannot leak into a new chat before runtime publication.
        # This recovery path is different: full Settings may have updated the
        # config cache without replacing ``app.app_config``. Reuse the fresh
        # readiness mapping already resolved above so an eligible, unused,
        # blocked chat still converges without an app restart (task-177).
        fresh_defaults = blank_console_session_settings(app_config)
        if fresh_defaults == settings:
            return settings
        fresh_readiness = build_console_settings_readiness(
            fresh_defaults,
            app_config=app_config,
        )
        if not fresh_readiness.native_send_supported:
            return settings
        previous_provider_key = provider_config_key(settings.provider)
        next_provider_key = provider_config_key(fresh_defaults.provider)
        store.replace_session_settings(
            session.id,
            fresh_defaults,
            mark_user_work=False,
            canonical_settings_baseline=fresh_defaults,
        )
        if previous_provider_key and next_provider_key != previous_provider_key:
            # task-16475: the convergence itself is task-177 behavior, but a
            # session whose provider identity changes under it must not do so
            # silently -- from the user's seat the Provider chip just flipped
            # to a provider they never chose.
            self._notify_stale_default_provider_swap(
                previous_provider_key,
                next_provider_key or str(fresh_defaults.provider),
            )
        return fresh_defaults

    def _notify_stale_default_provider_swap(
        self,
        previous_provider_key: str,
        next_provider_key: str,
    ) -> None:
        """Announce one stale-default refresh that changed the provider.

        Best-effort: the refresh runs deep inside ensure/sync paths, so a
        notification failure must never break them.
        """
        copy = (
            f"Console provider changed {previous_provider_key} -> "
            f"{next_provider_key}: this unused session now follows your saved "
            "defaults (Settings > Providers & Models)."
        )
        logger.info(
            "Stale-default refresh swapped session provider ({} -> {})",
            previous_provider_key,
            next_provider_key,
        )
        try:
            self.app_instance.notify(copy, severity="warning")
        except Exception:  # noqa: BLE001 -- display-only signal
            logger.debug("Provider-swap notice could not be shown", exc_info=True)

    def _replace_active_console_session_settings(
        self,
        settings: ConsoleSessionSettings,
    ) -> None:
        """Replace settings for only the active native Console session."""
        store = self._ensure_console_chat_store()
        workspace_id = store.workspace_context.active_workspace_id
        session = store.ensure_session(
            title=self._workspace_initial_session_title(workspace_id),
            workspace_id=workspace_id,
            settings=self._default_console_session_settings(),
        )
        store.replace_session_settings(session.id, settings)
        self._sync_console_chat_core_state()
        self._sync_console_settings_summary()

    def _console_session_settings_for_resume(
        self,
        conversation: Mapping[str, Any],
    ) -> ConsoleGenerationSettingsHydration:
        """Hydrate resumed settings from current config and saved metadata."""
        return hydrate_console_generation_settings(
            self._provider_readiness_app_config(),
            conversation,
        )

    def _apply_console_session_system_prompt(
        self, system_prompt: Optional[str]
    ) -> None:
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
        self._sync_console_control_bar()

    # -- Temporary-session promotion ------------------------------------------

    def _dispatch_promote_console_temporary_session(self) -> None:
        """Kick off the temporary-chat save as its own worker (F5, final review).

        Both entry points (the composer menu row and the Temporary chip)
        land here, so the save behaves identically however it was reached,
        and both stay non-blocking: ``promote_ephemeral_session`` runs one
        DB transaction over every tree node including attachment BLOBs,
        which can visibly freeze the TUI for a temporary chat with several
        pasted images. ``group="console-promote"`` is its own family,
        distinct from ``console-sync``/``console-run-*``/``console-
        impersonate`` -- see ``Tests/UI/test_chat_screen_worker_groups.py``
        for why an ungrouped (or wrongly-grouped) exclusive worker is not a
        cosmetic bug on this screen: one already cancelled in-flight
        Console sends on this branch. ``exclusive=True`` collapses a
        double-activation (menu row AND chip clicked before the first save
        lands) into one save rather than two overlapping transactions on
        the same session.
        """
        self.run_worker(
            self._promote_console_temporary_session(),
            exclusive=True,
            group="console-promote",
        )

    async def _promote_console_temporary_session(self) -> None:
        """Save the active temporary chat, then refresh its marker and chip.

        The actual store call is offloaded to a thread
        (``asyncio.to_thread``) rather than run inline on the event loop --
        see ``_dispatch_promote_console_temporary_session`` for why blocking
        here is a real, user-visible freeze, not a theoretical one.
        """
        store = self._ensure_console_chat_store()
        session_id = getattr(store, "active_session_id", None)
        if not session_id:
            return
        try:
            conversation_id = await asyncio.to_thread(
                store.promote_ephemeral_session, session_id
            )
        except Exception:
            logger.opt(exception=True).warning("Saving the temporary chat failed")
            self.app_instance.notify(
                "Could not save this chat. It is still temporary.",
                severity="error",
            )
            return
        if conversation_id is None:
            # F6 (final review): every other outcome notifies -- a silent
            # return here left the user with no feedback at all on "Save
            # this chat". `promote_ephemeral_session` returns `None` for
            # two different reasons (its own docstring): the session was
            # already saved (idempotent, genuinely fine), or no
            # persistence adapter is configured at all (the chat is still
            # temporary and nothing happened) -- these read very
            # differently to the user, so distinguish them rather than
            # picking one generic sentence.
            session = next((s for s in store.sessions() if s.id == session_id), None)
            if session is not None and not session.ephemeral:
                # A cancelled-then-retried promote worker lands here: the
                # first (cancelled) run's `asyncio.to_thread` DB write still
                # completed, so the session really is saved, but that first
                # coroutine never reached the `_sync_console_temporary_chip()`
                # call below. Without refreshing here too, the chip and the
                # `◌` tab marker keep reading "Temporary" about a chat
                # that is, in fact, saved -- exactly the mismatch this
                # marker exists to prevent.
                self._sync_console_temporary_chip()
                self.app_instance.notify(
                    "This chat is already saved.", severity="information"
                )
            else:
                self.app_instance.notify(
                    "Could not save this chat right now. It is still temporary.",
                    severity="warning",
                )
            return
        self._invalidate_console_persisted_rows_cache()
        # `_sync_native_console_chat_ui()` below never touches the Temporary
        # chip (see `_sync_console_temporary_chip`'s docstring: it is pushed
        # explicitly at every session-creation/switch point, not folded into
        # the general sync tick), so without this call the chip would keep
        # showing "Temporary" on an already-saved conversation.
        self._sync_console_temporary_chip()
        self.run_worker(
            self._sync_native_console_chat_ui(), exclusive=False, group="console-sync"
        )
        self.app_instance.notify("Chat saved.", severity="information")

    # -- Character sessions -----------------------------------------------------

    def _swap_console_session_character(
        self,
        store: Any,
        character_id: int,
        seed: CharacterSessionPromptSeed,
        *,
        global_default: str,
    ) -> bool:
        """Rebind the active session to ``character_id`` in place.

        The greeting is only seeded into an EMPTY chat (user decision,
        2026-07-31): interrupting a conversation in progress with a
        greeting reads as the model talking to itself.

        Args:
            store: The Console chat store.
            character_id: The picked character's local id.
            seed: The character identity, trusted sources, and safe projections.
            global_default: Current global user display name.

        Returns:
            True when the active session was rebound.
        """
        session_id = getattr(store, "active_session_id", None)
        session = None
        if session_id:
            session = next((s for s in store.sessions() if s.id == session_id), None)
        if session is None:
            return False
        for field, value in (
            ("runtime_backend", "local"),
            ("assistant_kind", "character"),
            ("assistant_id", str(character_id)),
            ("assistant_authority_id", None),
            ("character_id", character_id),
        ):
            try:
                object.__setattr__(session, field, value)
            except Exception:
                logger.opt(exception=True).warning(
                    "Character swap: could not set {}.", field
                )
                return False
        try:
            greeting_template = (
                seed.greeting_template
                if not store.messages_for_session(session_id)
                else ""
            )
            _updated, _greeting, persisted = store.swap_session_character_roleplay(
                session_id,
                character_name=seed.name,
                system_template=seed.system_template,
                greeting_template=greeting_template,
                global_default=global_default,
            )
        except Exception as exc:
            logger.warning(
                "Character swap: roleplay template seed failed (error_type={}).",
                type(exc).__name__,
            )
            self.app_instance.notify(
                "Character changed for this session, but the change could not be saved.",
                severity="warning",
            )
            return False
        if not persisted:
            self.app_instance.notify(
                "Character changed for this session, but the change could not be saved.",
                severity="warning",
            )
            return False
        self._clear_replaced_actor_reactions(
            session_id, actor_kind="character", actor_id=str(character_id)
        )
        return True

    async def _start_character_console_session(
        self, payload: ChatHandoffPayload
    ) -> bool:
        """Build a dedicated character conversation from a Start-Chat handoff.

        Args:
            payload: The Personas Start-Chat handoff staged into the native
                Console (its metadata carries the character identity).

        Returns:
            ``True`` when a character session was created (the caller then
            marks the handoff consumed); ``False`` to let the caller fall back
            to the staged-context path (task-427).
        """
        identity = _character_session_identity_from_handoff(payload)
        if identity is None:
            return False
        runtime_backend, character_id, name_hint, assistant_id = identity
        scope_service = getattr(
            self.app_instance,
            "character_persona_scope_service",
            None,
        )
        get_character = getattr(scope_service, "get_character", None)
        if not callable(get_character):
            return False

        assistant_authority_id: str | None
        local_character_id: int | None
        server_context_capture: object | None = None
        server_context_is_current: Callable[[object], bool] | None = None

        def exact_server_context_is_current() -> bool:
            if server_context_capture is None or server_context_is_current is None:
                return False
            try:
                return server_context_is_current(server_context_capture) is True
            except Exception:
                return False

        if runtime_backend == "local":
            db = getattr(self.app_instance, "chachanotes_db", None)
            get_local_authority_id = getattr(db, "get_local_authority_id", None)
            if not callable(get_local_authority_id):
                return False
            try:
                local_authority_id = get_local_authority_id()
            except asyncio.CancelledError:
                raise
            except Exception:
                return False
            if (
                type(local_authority_id) is not str
                or not local_authority_id
                or local_authority_id != local_authority_id.strip()
            ):
                return False
            assistant_authority_id = local_authority_id
            local_character_id = character_id
        else:
            expected_server_id = payload.active_server_profile_id
            if (
                type(expected_server_id) is not str
                or not expected_server_id
                or expected_server_id != expected_server_id.strip()
            ):
                return False
            if (
                getattr(self.app_instance, "active_server_id", None)
                != expected_server_id
            ):
                return False

            assistant_authority_id = None
            provider = getattr(self.app_instance, "server_context_provider", None)
            capture_context = getattr(
                provider,
                "capture_character_authority_context",
                None,
            )
            server_context_is_current = getattr(
                provider,
                "is_character_authority_context_current",
                None,
            )
            resolver = getattr(provider, "resolve_character_authority_id", None)
            if not callable(capture_context) or not callable(server_context_is_current):
                return False
            try:
                server_context_capture = capture_context(
                    expected_server_id=expected_server_id
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return False
            if not exact_server_context_is_current():
                return False
            if callable(resolver):
                try:
                    resolved_authority_id = await resolver(
                        expected_server_id=expected_server_id,
                        context_capture=server_context_capture,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    resolved_authority_id = None
                if (
                    type(resolved_authority_id) is str
                    and _SERVER_CHARACTER_AUTHORITY_PATTERN.fullmatch(
                        resolved_authority_id
                    )
                    is not None
                ):
                    assistant_authority_id = resolved_authority_id

            # This is both the post-resolver fence and the immediately
            # pre-card-fetch fence. No card from a newly active target may be
            # used for the ID carried by the handoff.
            if (
                not exact_server_context_is_current()
                or getattr(self.app_instance, "active_server_id", None)
                != expected_server_id
            ):
                return False
            local_character_id = None

        try:
            card = await get_character(character_id, mode=runtime_backend)
            if hasattr(card, "model_dump"):
                card = card.model_dump(mode="json")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Start Chat: character card unavailable; staging context instead "
                "(source={}).",
                runtime_backend,
            )
            return False
        if not isinstance(card, Mapping) or not card:
            return False
        card = dict(card)
        if _canonical_card_character_id(card.get("id")) != character_id:
            return False

        if runtime_backend == "server":
            expected_server_id = payload.active_server_profile_id
            if (
                not exact_server_context_is_current()
                or getattr(self.app_instance, "active_server_id", None)
                != expected_server_id
            ):
                return False

        global_name = _console_global_user_display_name(
            self._provider_readiness_app_config()
        )
        seed = _character_session_prompt_seed(
            card,
            name_hint=str(name_hint or ""),
            user_name=global_name,
        )

        store = self._ensure_console_chat_store()
        canonical_defaults = self._default_console_session_settings()
        settings = replace(
            canonical_defaults,
            system_prompt=seed.system_prompt,
            character_label=seed.name,
        )
        if runtime_backend == "server" and (
            not exact_server_context_is_current()
            or getattr(self.app_instance, "active_server_id", None)
            != payload.active_server_profile_id
        ):
            return False
        active = next(
            (
                candidate
                for candidate in store.sessions()
                if candidate.id == store.active_session_id
            ),
            None,
        )
        if (
            active is not None
            and active.settings is not None
            and active.settings != canonical_defaults
            and active.canonical_settings_baseline == active.settings
        ):
            try:
                active = store.refresh_pristine_session_settings(
                    active.id,
                    prior_canonical_settings=active.settings,
                    current_canonical_settings=canonical_defaults,
                )
            except ValueError:
                pass
        active_messages = (
            store.messages_for_session(active.id) if active is not None else []
        )
        duplicate_handoff = bool(
            active is not None
            and active.settings == settings
            and active.runtime_backend == runtime_backend
            and active.assistant_kind == "character"
            and active.assistant_id == assistant_id
            and active.assistant_authority_id == assistant_authority_id
            and active.character_id == local_character_id
            and active.character_name == seed.name
            and active.character_system_template == seed.system_template
            and (
                (not seed.greeting_template.strip() and not active_messages)
                or (
                    bool(seed.greeting_template.strip())
                    and len(active_messages) == 1
                    and active_messages[0].role is ConsoleMessageRole.ASSISTANT
                    and active_messages[0].metadata is not None
                    and active_messages[0].metadata.template_kind
                    == "character_greeting"
                    and active_messages[0].metadata.template_source
                    == seed.greeting_template
                )
            )
        )
        if duplicate_handoff:
            session = active
        else:
            session = None
            if active is not None and store.is_pristine_session(
                active.id,
                expected_settings=canonical_defaults,
            ):
                try:
                    session = store.repurpose_pristine_session(
                        active.id,
                        canonical_settings=canonical_defaults,
                        trusted_system_prompt=seed.system_prompt,
                        title=f"Chat with {seed.name}",
                        settings=settings,
                        runtime_backend=runtime_backend,
                        assistant_kind="character",
                        assistant_id=assistant_id,
                        assistant_authority_id=assistant_authority_id,
                        character_id=local_character_id,
                        character_name=seed.name,
                    )
                except ValueError:
                    session = None
            if session is None:
                self._refresh_console_library_policy_defaults()
                session = store.create_session(
                    title=f"Chat with {seed.name}",
                    workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
                    settings=settings,
                    runtime_backend=runtime_backend,
                    assistant_kind="character",
                    assistant_id=assistant_id,
                    assistant_authority_id=assistant_authority_id,
                    character_id=local_character_id,
                    character_name=seed.name,
                )
            try:
                store.seed_character_roleplay(
                    session.id,
                    system_template=seed.system_template,
                    greeting_template=seed.greeting_template,
                    global_default=global_name,
                )
            except Exception as exc:
                logger.warning(
                    "Start Chat: roleplay template seed/persist failed; continuing "
                    "(error_type={}).",
                    type(exc).__name__,
                )
        store.switch_session(session.id)
        if not duplicate_handoff:
            if local_character_id is None:
                self._clear_session_manual_reactions(session.id)
            else:
                self._clear_replaced_actor_reactions(
                    session.id,
                    actor_kind="character",
                    actor_id=str(local_character_id),
                )
        try:
            await self._sync_native_console_chat_ui()
            self._focus_console_composer_if_needed(force=True)
        except asyncio.CancelledError:
            # The durable commit boundary is above. Report success so the
            # caller acknowledges this handoff instead of replaying it.
            return True
        except Exception:
            # The character session (and its greeting) is already durably
            # created above -- a UI-sync/focus failure here must not
            # propagate, or the caller would never clear
            # ``pending_chat_handoff`` and a later re-consume (e.g. a
            # screen re-mount timer) would build a SECOND durable
            # character session.
            logger.opt(exception=True).warning(
                "Start Chat: post-seed console sync/focus failed; character "
                "session was already created and the handoff is still "
                "considered consumed."
            )
        return True

    # -- Draft sync ---------------------------------------------------------

    def _capture_console_draft_switch_snapshot(self) -> None:
        """Record the composer state at the moment a session switch begins.

        TASK-339: the draft swap in ``_sync_console_session_draft`` can run
        a settle-window later (coalesced syncs, slow resume loads). Anything
        the user types in that window is intended for the NEW session; this
        snapshot lets the swap attribute it correctly instead of saving it
        to the old session and wiping it from the composer.
        """
        composer = self._console_composer_or_none()
        if composer is None:
            self._console_draft_switch_snapshot = None
            return
        self._console_draft_switch_snapshot = (
            self._console_visible_draft_session_id,
            composer.draft_text(),
            composer.edit_serial,
        )

    def _sync_console_session_draft(self) -> None:
        """Reconcile the composer draft with the active runtime Console session.

        Saves the visible draft back to the session that owns it, then loads
        the active session's draft when the active session changed. Runs
        inside the native Console sync pass so session transitions cannot
        lose drafts. Keystrokes typed between switch initiation (see
        ``_capture_console_draft_switch_snapshot``) and this swap carry
        forward into the new session, in order (TASK-339).
        """
        store = self._ensure_console_chat_store()
        creating_blank_session = store.active_session_id is None
        defaults = self._blank_console_session_settings()
        session = store.ensure_session(
            title=self._workspace_initial_session_title(
                store.workspace_context.active_workspace_id
            ),
            workspace_id=store.workspace_context.active_workspace_id,
            settings=defaults,
            canonical_settings_baseline=defaults,
        )
        if creating_blank_session:
            session.new_chat_default_generation = (
                self._console_new_chat_default_generation()
            )
        active_session_id = session.id
        composer = self._console_composer_or_none()
        if composer is None:
            return
        visible_session_id = self._console_visible_draft_session_id
        if visible_session_id == active_session_id:
            self._restore_banked_raw_cli_stashes_fn(active_session_id, composer)
            if visible_session_id is not None:
                try:
                    store.set_session_draft(visible_session_id, composer.draft_text())
                except KeyError:
                    pass
            return
        snapshot = self._console_draft_switch_snapshot
        self._console_draft_switch_snapshot = None
        live_text = composer.draft_text()
        save_text = live_text
        typed_suffix = ""
        if snapshot is not None and snapshot[0] == visible_session_id:
            snap_text, snap_serial = snapshot[1], snapshot[2]
            if composer.edit_serial != snap_serial and live_text.startswith(snap_text):
                # User typed during the settle window: the old session keeps
                # what it actually had at the keypress; the new typing rides
                # into the new session below. (Non-append edits — e.g.
                # backspacing into the old draft — fall back to today's
                # save-the-live-text semantics.)
                save_text = snap_text
                typed_suffix = live_text[len(snap_text) :]
        if visible_session_id is not None:
            try:
                store.set_session_draft(visible_session_id, save_text)
            except KeyError:
                pass
            # TASK-1281: this composer is about to start showing a DIFFERENT
            # session's draft -- bank its undo/redo history under the
            # session it actually belongs to before the swap below discards
            # it, so a later switch back can restore it.
            self._console_undo_histories[visible_session_id] = (
                composer.export_undo_history()
            )
        try:
            composer.load_draft(store.session_draft(active_session_id))
        except KeyError:
            composer.clear_draft()
        # TASK-1281: restore (or start fresh) BEFORE re-inserting any
        # settle-window keystrokes below, so those keystrokes land in --
        # and are recorded onto -- the session they actually typed into.
        composer.restore_undo_history(
            self._console_undo_histories.get(active_session_id)
        )
        self._restore_banked_raw_cli_stashes_fn(active_session_id, composer)
        if typed_suffix:
            composer.insert_text(typed_suffix)
        self._sync_console_command_popup()
        self._console_visible_draft_session_id = active_session_id

    # -- Session identity / state -------------------------------------------

    def _active_native_console_session(self) -> Any | None:
        """Return the active native Console session without creating the store."""
        console_store = self._console_chat_store
        active_session_id = (
            console_store.active_session_id if console_store is not None else None
        )
        if console_store is None or active_session_id is None:
            return None
        for console_session in console_store.sessions():
            if console_session.id == active_session_id:
                return console_session
        return None

    def _current_console_conversation_id(self) -> Optional[str]:
        """Return the active conversation id for Console context highlighting."""
        console_store = self._console_chat_store
        active_session_id = (
            console_store.active_session_id if console_store is not None else None
        )
        if console_store is not None and active_session_id is not None:
            for console_session in console_store.sessions():
                if console_session.id == active_session_id:
                    conversation_id = console_session.persisted_conversation_id
                    if conversation_id:
                        return str(conversation_id)
                    break
        return None

    def _current_console_session_id(self) -> Optional[str]:
        """Return a durable external Console session scope when one is available."""
        session_id = getattr(self.app_instance, "console_rail_session_id", None)
        if session_id:
            return str(session_id)
        console_store = self._console_chat_store
        if console_store is not None and console_store.active_session_id is not None:
            return str(console_store.active_session_id)
        return None

    def _console_active_session_is_ephemeral(self) -> bool:
        """Return whether the active Console session is temporary.

        Public-API only (`sessions()` + `active_session_id`): the store has
        no single-session getter that is not private. The scan is over open
        tabs, so it is a handful of items.
        """
        store = self._console_chat_store
        if store is None:
            return False
        active_id = store.active_session_id
        if not active_id:
            return False
        return any(
            session.id == active_id and session.ephemeral
            for session in store.sessions()
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
            return self._session_id_for_workspace_conversation(row_key)
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

        active_workspace_id = str(
            store.workspace_context.active_workspace_id or ""
        ).strip()
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
            conversation_rows=tuple(self._merge_workspace_rows(native_rows, rows)),
        )

    # -- Screen-state (de)serialization for one session ----------------------

    @staticmethod
    def _console_session_to_state(session: ConsoleChatSession) -> dict[str, Any]:
        """Serialize one ConsoleChatSession for screen-state restoration.

        Extracted from `_serialize_native_console_state` so the round trip
        is testable without a running app. This is an explicit field list:
        a field missing from it is silently dropped on the way back.
        """
        return {
            "id": session.id,
            "title": session.title,
            "workspace_id": session.workspace_id,
            "persisted_conversation_id": session.persisted_conversation_id,
            "draft": session.draft,
            "has_user_work": session.has_user_work,
            "settings": ConsoleSessionController._serialize_console_settings(
                session.settings
            ),
            "canonical_settings_baseline": (
                ConsoleSessionController._serialize_console_settings(
                    session.canonical_settings_baseline
                )
            ),
            "context_policy_overrides": session.context_policy_overrides.to_dict(),
            "thinking_history_policy": session.thinking_history_policy,
            "updated_at": session.updated_at,
            "runtime_backend": session.runtime_backend,
            "assistant_kind": session.assistant_kind,
            "assistant_id": session.assistant_id,
            "assistant_authority_id": session.assistant_authority_id,
            "persona_memory_mode": session.persona_memory_mode,
            "character_id": session.local_character_id(),
            "character_name": session.character_name,
            "user_display_name_override": session.user_display_name_override,
            "character_system_template": session.character_system_template,
            "identity_revision": session.identity_revision,
            # Temporary conversations: without this key a temporary chat
            # comes back as a persisting one after any screen navigation,
            # and the next send writes it to the DB.
            "ephemeral": session.ephemeral,
            _CONSOLE_TODO_STATE_KEY: session.todo_store.export_snapshot(),
            "project_instructions": json.loads(
                encode_project_context_json(session.project_instruction_state)
            ),
        }

    def _console_session_from_state(
        self, raw_session: dict[str, Any]
    ) -> ConsoleChatSession:
        """Rebuild one ConsoleChatSession from its serialized screen state.

        The mirror of `_console_session_to_state`. Every legacy-payload
        branch below exists because older saved states omit keys that newer
        ones carry -- keep them.
        """
        # `getattr` (not `self._ensure_console_chat_store()`) tolerates a
        # bare controller that never ran `__init__` (unit-level
        # serialize/restore round trips build one via `__new__`, same as
        # `ChatScreen.__new__` did pre-move). In the real restore path,
        # `_restore_native_console_state` already ensured the store before
        # this per-session helper runs, so this is the same object either way.
        store = getattr(self, "_console_chat_store", None)
        session_id = str(raw_session.get("id") or uuid.uuid4())
        session_kwargs: dict[str, Any] = dict(
            id=session_id,
            title=str(raw_session.get("title") or DEFAULT_CONSOLE_SESSION_TITLE),
            workspace_id=str(
                raw_session.get("workspace_id")
                or (
                    store.workspace_context.active_workspace_id
                    if store is not None
                    else None
                )
                or CONSOLE_GLOBAL_WORKSPACE_ID
            ),
            persisted_conversation_id=(
                str(raw_session["persisted_conversation_id"])
                if raw_session.get("persisted_conversation_id") is not None
                else None
            ),
            settings=self._restore_console_settings(raw_session.get("settings")),
            canonical_settings_baseline=self._restore_console_settings(
                raw_session.get("canonical_settings_baseline")
            ),
            draft=str(raw_session.get("draft") or ""),
            has_user_work=(
                raw_session.get("has_user_work") is True
                or bool(raw_session.get("draft"))
            ),
            thinking_history_policy=normalize_thinking_history_policy(
                raw_session.get("thinking_history_policy")
            ),
        )
        todo_store = SessionTodoStore()
        if _CONSOLE_TODO_STATE_KEY in raw_session:
            try:
                todo_store = SessionTodoStore.from_snapshot(
                    raw_session[_CONSOLE_TODO_STATE_KEY]
                )
            except TodoStoreError:
                logger.warning("Console task state invalid; starting empty.")
        session_kwargs["todo_store"] = todo_store
        try:
            session_kwargs["context_policy_overrides"] = (
                ConsoleContextPolicyOverrides.from_mapping(
                    raw_session.get("context_policy_overrides")
                )
            )
        except ContextPolicyError:
            session_kwargs["context_policy_overrides"] = ConsoleContextPolicyOverrides()
            session_kwargs["context_policy_error"] = "invalid_screen_context_policy"
        # Legacy payloads saved before `updated_at` was serialized omit the
        # key entirely; keep the ConsoleChatSession factory default (now)
        # for those instead of forcing an empty/invalid timestamp.
        raw_updated_at = raw_session.get("updated_at")
        if raw_updated_at:
            session_kwargs["updated_at"] = str(raw_updated_at)
        identity_keys = (
            "runtime_backend",
            "assistant_kind",
            "assistant_id",
            "assistant_authority_id",
        )
        has_source_aware_identity = any(key in raw_session for key in identity_keys)
        if has_source_aware_identity:
            raw_runtime_backend = raw_session.get("runtime_backend")
            session_kwargs["runtime_backend"] = (
                raw_runtime_backend if type(raw_runtime_backend) is str else ""
            )
            for key in (
                "assistant_kind",
                "assistant_id",
                "assistant_authority_id",
                "persona_memory_mode",
            ):
                value = raw_session.get(key)
                session_kwargs[key] = value if type(value) is str else None

        raw_character_id = raw_session.get("character_id")
        character_id: int | None = None
        if type(raw_character_id) is int and raw_character_id > 0:
            if not has_source_aware_identity:
                character_id = raw_character_id
            elif (
                session_kwargs.get("runtime_backend") == "local"
                and session_kwargs.get("assistant_kind") == "character"
                and session_kwargs.get("assistant_id") == str(raw_character_id)
            ):
                character_id = raw_character_id
        if character_id is not None:
            session_kwargs["character_id"] = character_id
        if not has_source_aware_identity and character_id is not None:
            # Pre-provenance screen state used the numeric local character
            # projection as its direct-chat routing marker. Preserve that
            # behavior without inventing a proven authority.
            session_kwargs.update(
                runtime_backend="local",
                assistant_kind="character",
                assistant_id=str(character_id),
                assistant_authority_id=None,
            )
        raw_character_name = raw_session.get("character_name")
        if raw_character_name is not None:
            session_kwargs["character_name"] = str(raw_character_name)
        raw_user_display_name_override = raw_session.get("user_display_name_override")
        try:
            session_kwargs["user_display_name_override"] = normalize_chat_display_name(
                raw_user_display_name_override,
                blank_means_none=True,
            )
        except ChatDisplayNameError:
            session_kwargs["user_display_name_override"] = None
        raw_character_system_template = raw_session.get("character_system_template")
        session_kwargs["character_system_template"] = (
            raw_character_system_template
            if isinstance(raw_character_system_template, str)
            else None
        )
        raw_identity_revision = raw_session.get("identity_revision")
        session_kwargs["identity_revision"] = (
            raw_identity_revision
            if isinstance(raw_identity_revision, int)
            and not isinstance(raw_identity_revision, bool)
            and raw_identity_revision >= 0
            else 0
        )
        # Legacy payloads predate the key; absent means saved, never temporary.
        session_kwargs["ephemeral"] = raw_session.get("ephemeral") is True
        raw_project_state = raw_session.get("project_instructions")
        try:
            encoded_project_state = json.dumps(raw_project_state)
        except (TypeError, ValueError):
            encoded_project_state = None
        session_kwargs["project_instruction_state"] = decode_project_context_json(
            encoded_project_state
        )
        return ConsoleChatSession(**session_kwargs)

    def _build_console_project_instruction_display_state(
        self,
        session_id: str | None = None,
    ) -> ConsoleProjectInstructionState:
        """Return cached content-free state without filesystem authority I/O."""
        store = self._console_chat_store
        session = None
        if store is not None:
            target_id = session_id or store.active_session_id
            session = next(
                (item for item in store.sessions() if item.id == target_id), None
            )
        control = (
            session.project_instruction_state
            if session is not None
            else ProjectInstructionControlState.legacy_disabled()
        )
        cache = getattr(self, "_console_project_instruction_display_cache", {})
        cached = cache.get(session.id) if session is not None else None
        if cached is not None and cached[0] == control:
            return cached[1]
        return build_console_project_instruction_state(
            control,
            binding_label=control.working_folder_binding_id or "",
        )

    async def _refresh_console_project_instruction_display_state(
        self, session_id: str
    ) -> ConsoleProjectInstructionState:
        """Resolve authority off-loop and publish only content-free state."""
        store = self._console_chat_store
        session = (
            next((item for item in store.sessions() if item.id == session_id), None)
            if store is not None
            else None
        )
        if session is None:
            return build_console_project_instruction_state(
                ProjectInstructionControlState.legacy_disabled()
            )
        control = session.project_instruction_state
        controller = self._ensure_console_chat_controller()
        registry = getattr(self.app_instance, "workspace_registry_service", None)

        def resolve_content_free() -> tuple[str, bool | None, tuple[str, ...]]:
            label = control.working_folder_binding_id or ""
            if not control.project_instructions_enabled or not label:
                return label, None, ()
            try:
                binding = registry.get_runtime_binding(
                    control.working_folder_binding_id
                )
                if binding is not None:
                    label = str(getattr(binding, "label", "") or label)
                resolve_project_instruction_binding(session, registry)
            except (AttributeError, ProjectInstructionBindingRecovery) as exc:
                return label, False, (str(exc) or "binding_unavailable",)
            return label, True, ()

        binding_label, locator_matches, authority_warnings = await asyncio.to_thread(
            resolve_content_free
        )
        current = next(
            (item for item in store.sessions() if item.id == session_id), None
        )
        if current is None or current.project_instruction_state != control:
            return self._build_console_project_instruction_display_state(session_id)
        metadata = controller.project_instruction_display_metadata(session_id)
        sources: tuple[ConsoleProjectInstructionSourceRow, ...] = ()
        warning_codes = authority_warnings
        if locator_matches is False:
            clear = getattr(controller, "_clear_project_instruction_delivery", None)
            if callable(clear):
                clear(session_id)
            metadata = None
        if metadata is not None:
            warning_codes = tuple(metadata.warning_codes)
            if metadata.relative_source:
                sources = (
                    ConsoleProjectInstructionSourceRow(
                        relative_source=str(metadata.relative_source),
                        scope=metadata.scope,
                        byte_count=metadata.byte_count,
                        outcome=metadata.outcome,
                        warning_code=(
                            metadata.warning_codes[0] if metadata.warning_codes else ""
                        ),
                    ),
                )
        state = build_console_project_instruction_state(
            control,
            binding_label=binding_label,
            locator_matches=locator_matches,
            sources=sources,
            warning_codes=warning_codes,
        )
        cache = getattr(self, "_console_project_instruction_display_cache", None)
        if cache is None:
            cache = self._console_project_instruction_display_cache = {}
        cache[session_id] = (control, state)
        return state

    def _request_console_project_instruction_display_refresh(
        self, session_id: str, *, force: bool = False
    ) -> None:
        """Start at most one authority refresh for a session/control snapshot."""
        store = self._console_chat_store
        session = (
            next((item for item in store.sessions() if item.id == session_id), None)
            if store is not None
            else None
        )
        if session is None:
            return
        control = session.project_instruction_state
        controller = self._ensure_console_chat_controller()
        metadata = controller.project_instruction_display_metadata(session_id)
        signature = (control, metadata)
        inflight = getattr(self, "_console_project_instruction_refresh_inflight", None)
        if inflight is None:
            inflight = self._console_project_instruction_refresh_inflight = {}
        if inflight.get(session_id) == signature:
            return
        completed = getattr(
            self, "_console_project_instruction_refresh_completed", None
        )
        if completed is None:
            completed = self._console_project_instruction_refresh_completed = {}
        previous = completed.get(session_id)
        now = time.monotonic()
        if (
            not force
            and previous is not None
            and previous[0] == signature
            and now - previous[1] < _PROJECT_INSTRUCTION_AUTHORITY_REFRESH_SECONDS
        ):
            return
        inflight[session_id] = signature

        async def refresh() -> None:
            try:
                state = await self._refresh_console_project_instruction_display_state(
                    session_id
                )
                active = self._active_native_console_session()
                if active is not None and active.id == session_id:
                    try:
                        row = self._screen.query_one(
                            "#console-project-instruction-status",
                            ConsoleProjectInstructionStatusRow,
                        )
                    except QueryError:
                        pass
                    else:
                        row.sync_state(state)
            finally:
                current_store = self._console_chat_store
                session_exists = current_store is not None and any(
                    item.id == session_id for item in current_store.sessions()
                )
                if session_exists:
                    completed[session_id] = (signature, time.monotonic())
                else:
                    completed.pop(session_id, None)
                if inflight.get(session_id) == signature:
                    inflight.pop(session_id, None)

        self.run_worker(
            refresh(),
            exclusive=True,
            group=f"console-project-instructions-{session_id}",
        )

    def _sync_console_project_instruction_status_row(self) -> None:
        """Refresh the mounted Inspector row from current session authority."""
        try:
            row = self._screen.query_one(
                "#console-project-instruction-status",
                ConsoleProjectInstructionStatusRow,
            )
        except QueryError:
            return
        row.sync_state(self._build_console_project_instruction_display_state())
        session = self._active_native_console_session()
        if session is not None:
            self._request_console_project_instruction_display_refresh(session.id)

    async def _select_project_instruction_binding(
        self,
        session_id: str,
        selections: tuple[Any, ...],
        recovery_code: str,
    ) -> tuple[str, str | None]:
        """Collect one setup decision on the Textual main loop."""

        def owning_session_exists() -> bool:
            store = self._console_chat_store
            return store is not None and any(
                session.id == session_id for session in store.sessions()
            )

        if not owning_session_exists():
            return "cancel", None
        options = tuple(
            ProjectInstructionBindingOption(
                binding_id=str(selection.binding.binding_id),
                label=str(
                    getattr(selection.binding, "display_name", "")
                    or getattr(selection.binding, "label", "")
                    or f"Folder {index + 1}"
                ),
                eligible=True,
            )
            for index, selection in enumerate(selections)
        )
        if not options:
            options = (
                ProjectInstructionBindingOption(
                    binding_id="",
                    label="No eligible folders",
                    eligible=False,
                    recovery=recovery_code,
                ),
            )
        result = await self._screen.app.push_screen_wait(
            ProjectInstructionSetupModal(options)
        )
        if not owning_session_exists() or not isinstance(
            result, ProjectInstructionSetupResult
        ):
            return "cancel", None
        return result.action, result.binding_id

    def _confirm_project_instruction_dispatch(self, notice: Any) -> str:
        """Marshal a worker-thread notice to Textual and wait fail-closed."""
        decided = threading.Event()
        result = {"decision": "cancel"}
        mounted_modal: list[ProjectInstructionNoticeModal] = []
        chat_controller = getattr(
            getattr(self, "_screen", None), "_console_chat_controller", None
        )
        active_cancel_events = getattr(chat_controller, "_active_cancel_events", {})
        owning_cancel_event = active_cancel_events.get(notice.session_id)

        def owning_session_exists() -> bool:
            sessions = (
                self._console_chat_store.sessions() if self._console_chat_store else ()
            )
            return any(session.id == notice.session_id for session in sessions)

        def owning_run_cancelled() -> bool:
            return bool(
                owning_cancel_event is not None and owning_cancel_event.is_set()
            )

        def finish(decision: str | None) -> None:
            if owning_session_exists():
                if decision in {"proceed", "cancel", "disable"}:
                    result["decision"] = decision
            decided.set()

        def mount() -> None:
            if not owning_session_exists():
                decided.set()
                return
            modal = ProjectInstructionNoticeModal(notice)
            mounted_modal.append(modal)
            self.push_screen(modal, callback=finish)

        def dismiss_owning_modal() -> None:
            if decided.is_set() or not mounted_modal:
                return
            try:
                mounted_modal[0].dismiss("cancel")
            except Exception:  # noqa: BLE001 - shutdown remains fail closed
                decided.set()

        call_from_thread = getattr(self.app_instance, "call_from_thread", None)
        if not callable(call_from_thread):
            return "cancel"
        try:
            call_from_thread(mount)
        except RuntimeError:
            return "cancel"
        timeout = max(
            0.0,
            float(
                getattr(
                    self,
                    "_project_instruction_notice_timeout_seconds",
                    _DEFAULT_PROJECT_INSTRUCTION_NOTICE_TIMEOUT_SECONDS,
                )
            ),
        )
        deadline = time.monotonic() + timeout
        while (
            not decided.is_set()
            and owning_session_exists()
            and not owning_run_cancelled()
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            decided.wait(min(_PROJECT_INSTRUCTION_NOTICE_POLL_SECONDS, remaining))
        if not decided.is_set():
            try:
                call_from_thread(dismiss_owning_modal)
            except RuntimeError:
                pass
        return result["decision"]

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
        values = dict(payload)
        values.pop("persona_label", None)
        values.pop("user_profile_label", None)
        valid_fields = set(ConsoleSessionSettings.__dataclass_fields__)
        values = {key: value for key, value in values.items() if key in valid_fields}
        provider = str(values.get("provider") or "").strip()
        if not provider:
            return None
        values["provider"] = provider
        try:
            return ConsoleSessionSettings(**values)
        except TypeError:
            logger.opt(exception=True).debug(
                "Skipping invalid Console session settings payload"
            )
            return None

    # -- Composer/session-draft coherence guard -------------------------------

    def _console_composer_history_session_synced(self) -> bool:
        """Return whether the composer's visible session matches the active one.

        TASK-1281 review F1 (HIGH): mirrors the guard `_insert_console_
        dictation` already carries for exactly this reason. During the
        session-switch settle window (TASK-339) -- `controller.switch_
        session(...)` runs synchronously and `store.active_session_id`
        changes immediately, but `_console_visible_draft_session_id` only
        catches up later, inside `_sync_console_session_draft`, once the
        deferred sync actually runs -- the composer can still be showing
        session A's draft while the store already considers session B
        active. Undo/redo must never run while that window is open: doing
        so would apply session A's history against the composer, then
        (via the re-persist below) write A's resulting text into the STORE
        under session B's id -- permanently destroying B's own draft the
        moment the deferred swap finally lands.

        Returns:
            True only when the composer is not None, an active session
            exists, and the composer is provably showing that exact
            session's draft right now.
        """
        composer = self._console_composer_or_none()
        if composer is None:
            return False
        store = self._ensure_console_chat_store()
        return (
            store.active_session_id is not None
            and self._console_visible_draft_session_id == store.active_session_id
        )
