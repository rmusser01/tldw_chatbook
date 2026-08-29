"""Console message controller.

Extracted out of `ChatScreen` (wave-3 console decomposition, task 1): the
native Console transcript MESSAGE cluster -- serialize/restore, resume-tree
flattening, screen-state rehydration, the per-message save-as/edit/retry/
continue/regenerate/variant-navigation helpers, and `handle_console_message_
action`, the 294-line dispatcher that routes every transcript row action
button (copy/edit/save-as/retry/regenerate/variant/keep/feedback/toggle-
image/save-image/delete/continue/speak/speak-stop/review-changes).

This module follows the SAME binding rule waves 1-2 established (see
`dictation.py`'s `ConsoleDictationController.__init__` docstring for the
canonical statement; restated briefly here):

1. **Framework services** (`run_worker`, `push_screen`) live-read from the
   screen via `@property` on every access -- never snapshotted.
2. **Everything else this cluster depends on that is not its own state** is a
   NAMED keyword-only constructor callable, matching the design spec's rule
   that "a controller's dependencies are its signature". Each is a
   zero-arg (or narrowly-typed) callable the CALLER (`ChatScreen.__init__`)
   constructs as a late-binding lambda closing over `self` -- never a bound
   method passed directly (the same instance-patch staleness reason
   `ConsoleDictationController.__init__`'s docstring gives in full). The
   controller's own property of the SAME NAME as the original private
   method/attribute is a thin wrapper around the stored callable.
3. `app_instance` is a plain snapshot, for the same reason `dictation.py`
   snapshots it: every read here is a bare-attribute/`notify()`/
   `post_message()` call in the pre-move source, never through a call that
   could go stale.

**Controller-to-controller seams this cluster needed** (session and
workspace; messages belong to a session, which lives in workspace context):
`active_native_console_session`/`current_console_conversation_id` reach
`ConsoleSessionController` directly (two moved bodies called
`self._session.X(...)` in the pre-move source -- each drops that prefix in
favour of a same-named property here, wrapping a constructor-injected
callable `ChatScreen.__init__` points at `self._session.X`, matching
`session.py`'s own documented seam to `ConsoleWorkspaceController`).
`active_session_is_ephemeral` is the one exception: it routes through the
SCREEN's own `_console_active_session_is_ephemeral` delegation instead
(session.py's disclosed exception for `console_transcript.py`'s bare-name
reach) because the pre-existing test suite ALSO monkeypatches that screen-
level name directly (see `__init__`'s own docstring for this parameter).
`console_initial_session_title_for_workspace` reaches
`ConsoleWorkspaceController` the same direct way (one moved body called
`self._workspace.X(...)`). The REVERSE seam -- `ConsoleWorkspaceController`
resuming a persisted conversation needs this cluster's own conversation-tree
flattener -- already existed as a named callable BEFORE this task
(`messages_from_conversation_tree_accessor`, wave-2 task 2); this task only
re-points that existing lambda at `self._message` instead of the screen.
Python resolves every one of these lambdas at CALL time, so construction
order between controllers never matters.

**`handle_console_message_action` is the wave's risk centre** (294 lines,
the largest constructor in this cluster as a direct consequence): it has NO
DOM access of its own, so nothing blocks the move, but it dispatches across
several clusters outside this controller (change-review, image-generation,
citation) -- each of those reaches becomes exactly one more named callable
here, never a back-door through `self.screen`.

**Dead bodies / delegation table**: this cluster's pre-move test suite
reaches an unusually large number of these methods directly by their
ORIGINAL private name -- `screen.handle_console_message_action(...)`,
`ChatScreen._serialize_console_message(...)`, `screen._select_console_
message_variant(...)`, etc, built up across many prior test-writing rounds
that predate this decomposition. Every one of those (17 of the 29 moved
`*message*`-named methods, plus none of the 7 non-matching helper functions
this cluster also owns exclusively) keeps a thin (<=3-line) delegation on
`ChatScreen` under its original name, forwarding to `self._message.X(...)`
(or, for the two state-free classmethods `_serialize_console_message`/
`_restore_console_message` and the staticmethod `_parse_console_message_
action_button_id`, to `ConsoleMessageController.X(...)` directly on the
class, since tests call those UNBOUND via `ChatScreen.X(...)` with no
instance). Three more moved methods keep their single in-repo staying
caller pointed at `self._message.X(...)` directly instead of a delegation
(`_active_console_transcript_has_messages`, `_console_message_excerpt`,
`_console_message_role_label` -- one staying caller each, no test
coupling, so a direct call-site edit was simpler than a delegation). The
full per-method table (moves clean / moves + delegation / moves + direct
call-site edit / stays, with reasons) is in the task-1 extraction report.

**Stays on `ChatScreen`, despite matching the `*message*` name pattern**:
- `_console_realtime_exit_message` -- the V4 realtime engine's own toast
  text, not `ConsoleChatMessage` state; `hands_free.py`/`session.py` already
  document the whole realtime engine as staying screen-owned this
  programme.
- `_console_imagegen_inflight_message_ids` -- image-generation in-flight
  bookkeeping keyed by message id, now owned with its session-keyed sibling
  by `ConsoleImageController`; it remains outside this message controller
  despite the name match.
- `_selected_console_message_inspector_rows` / `_clear_native_console_
  message_selection` -- real `query_one` DOM access.
- `handle_console_send_message` / `_send_console_message_from_visible_
  action` -- the latter does raw `query_one` DOM access (not through the
  established `_console_composer_or_none` accessor) and is composer/
  command-dispatch orchestration (command registry parsing, skill-blocked
  hints, the keyboard-capture draft stash shared with `on_key`, itself out
  of scope this wave) that only ever REACHES message creation by calling
  `_dispatch_console_draft_send` -> `_submit_console_native_draft` ->
  `ConsoleChatController.submit_draft` (business logic already outside
  `ChatScreen`), none of which are `*message*`-named or move this task.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING
import asyncio
import inspect
import os
import uuid

from loguru import logger
from rich.markup import escape as escape_markup
from textual.widgets import Button

from ...Chat.console_chat_controller import ConsoleChatController
from ...Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariant,
    ConsoleVariantSet,
    MessageAttachment,
)
from ...Chat.console_chat_store import ConsoleChatStore
from ...Chat.console_chat_fork import ConsoleForkEligibility
from ...Chat.console_conversation_hydration import (
    console_messages_from_conversation_tree,
)
from ...Chat.console_command_grammar import (
    GENERATE_IMAGE_COMMAND_HANDLER_ID,
    GENERATE_VIDEO_COMMAND_HANDLER_ID,
)
from ...Chat.console_roleplay_identity import ConsoleMessagePresentation
from ...Chat.console_ephemeral import blocked_reason
from ...Chat.console_image_view import IMAGE_CACHE_MAX_ENTRIES
from ...Chat.console_message_actions import (
    ConsoleActionResult,
    ConsoleMessageActionService,
)
from ...Chat.console_save_targets import (
    console_chatbook_artifact_payload,
    derive_console_save_title,
    resolve_console_artifact_owner_request,
)
from ...Chat.message_metadata import MessageMetadata
from ...Chat.provider_usage import ProviderUsage
from ...Video_Generation.video_metadata import VideoGenerationMetadata
from ...config import get_cli_setting
from ...Notes.notes_scope_service import ScopeType
from ...Widgets.Console import (
    ConsoleEditMessageModal,
    ConsoleEditResult,
    ConsoleSaveAsModal,
)

if TYPE_CHECKING:
    from ..Screens.chat_screen import ChatScreen

logger = logger.bind(module="ChatScreen")


# -- Module-level pure helpers this cluster owns exclusively (see module
# docstring) -- moved verbatim from `chat_screen.py`, which no longer
# references either name for itself.


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


class ConsoleMessageController:
    """Owns the Console shell's native message-transcript cluster:
    serialize/restore, resume-tree flattening, screen-state rehydration,
    per-message save-as/edit/retry/continue/regenerate/variant navigation,
    and `handle_console_message_action`'s full action dispatch.

    `ChatScreen` constructs exactly one of these, in `__init__`, and keeps a
    `self._message` reference plus the delegation/call-site-edit table
    described in the module docstring.
    """

    def __init__(
        self,
        screen: "ChatScreen",
        *,
        app_instance: Any,
        chat_store_accessor: Callable[[], ConsoleChatStore],
        current_chat_store_accessor: Callable[[], ConsoleChatStore | None],
        ensure_console_chat_controller: Callable[[], Any],
        current_chat_controller_accessor: Callable[[], Any | None],
        sync_native_console_chat_ui: Callable[[], Any],
        active_session_is_ephemeral: Callable[[], bool],
        active_native_console_session: Callable[[], Any],
        current_console_conversation_id: Callable[[], Any],
        active_console_provider_model_display: Callable[[], tuple],
        console_initial_session_title_for_workspace: Callable[[str | None], str],
        console_change_review_run_id: Callable[[ConsoleChatStore, str], str | None],
        open_change_review: Callable[[str | None], None],
        start_console_transcript_sync_timer: Callable[[], None],
        clear_native_console_message_selection: Callable[[], None],
        regenerate_console_generation_variant: Callable[[str], Any],
        select_console_generation_variant: Callable[[Any, str], None],
        keep_console_generation_variant: Callable[[Any], None],
        handle_console_toggle_image_view: Callable[[str], None],
        invalidate_console_persisted_rows_cache: Callable[[], None],
        invalidate_console_fork_image_selections: (
            Callable[[Sequence[str]], None] | None
        ) = None,
        play_console_video: Callable[[str], Any] | None = None,
        save_console_video_copy: Callable[[str], Any] | None = None,
        regenerate_console_video_message: Callable[[str], Any] | None = None,
        request_console_chat_fork: Callable[[str], Any] | None = None,
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 35 method bodies below (29 of the pre-move
        `*message*`-named cluster, plus 6 exclusively-owned helpers whose
        names don't match that pattern -- `_apply_console_message_
        attachments`,
        `_serialize_console_variants`, `_restore_console_variants`,
        `_console_save_as_destinations`, `_console_save_source_title`,
        `_clear_console_original_attempt_preview`) is a byte-for-byte copy
        of the pre-extraction `ChatScreen` method, EXCEPT: the four
        documented `self._session.X(...)`/`self._workspace.X(...)` seam
        call sites (dropping that prefix for a same-named property here,
        per the module docstring's "Controller-to-controller seams"
        section) and every `self.app.push_screen(...)` call site (dropping
        the `.app.` for the `push_screen` framework property below, the
        same transformation `session.py`'s own moved bodies already made).

        Args:
            screen: The Console screen. Used ONLY for the framework
                services (`run_worker`, `push_screen`) below. Zero
                `query_one`/`query` traffic reaches through `screen` here --
                every DOM-touching sibling of this cluster (`_selected_
                console_message_inspector_rows`, `_clear_native_console_
                message_selection`, `_send_console_message_from_visible_
                action`) stayed on `ChatScreen`; see the module docstring.
            app_instance: Snapshotted once, not re-read through `screen` --
                every reference in the moved bodies is a bare-attribute or
                `notify()`/`post_message()` call, never one that could
                observe a later `screen.app_instance` reassignment.
            chat_store_accessor: `ChatScreen._ensure_console_chat_store`,
                the general store accessor already shared by every other
                controller (see `session.py`'s identical parameter).
            current_chat_store_accessor: `ChatScreen._console_chat_store`'s
                bare-attribute-read shape (may be `None` pre-mount);
                distinct from `chat_store_accessor` for the same reason
                `session.py` keeps the two separate.
            ensure_console_chat_controller: `ChatScreen._ensure_console_
                chat_controller`.
            current_chat_controller_accessor: `ChatScreen._console_chat_
                controller`'s bare-attribute-read shape (may be `None`
                pre-mount, never lazily creates) -- used only by
                `_clear_console_original_attempt_preview`, distinct from
                `ensure_console_chat_controller` for the same reason
                `chat_store_accessor`/`current_chat_store_accessor` are
                kept separate.
            sync_native_console_chat_ui: `ChatScreen._sync_native_console_
                chat_ui`, the big render/inspector re-sync bridge -- stays
                screen-owned (DOM), reached here as an async callable.
            active_session_is_ephemeral: `ChatScreen._console_active_session_
                is_ephemeral`, the screen's OWN disclosed delegation to
                `ConsoleSessionController` (see `session.py`'s docstring for
                why that delegation exists: `console_transcript.py`'s
                bare-name reach) -- routed through the screen rather than
                straight to `self._session` because the pre-existing test
                suite also monkeypatches it at that name (4 sites in
                `test_console_native_chat_flow.py`).
            active_native_console_session: `ConsoleSessionController._active_
                native_console_session`, same seam, used only by
                `_console_save_source_title`.
            current_console_conversation_id: `ConsoleSessionController.
                _current_console_conversation_id`, same seam, used only by
                `_save_console_message_as_chatbook`.
            active_console_provider_model_display: `ChatScreen._active_
                console_provider_model_display`, a general screen helper
                (10 call sites across clusters) used only by
                `_save_console_message_as_chatbook` here.
            console_initial_session_title_for_workspace: `ConsoleWorkspace
                Controller._console_initial_session_title_for_workspace`
                (workspace <-> message seam), used only by `_append_native_
                console_system_message`.
            console_change_review_run_id: `ChatScreen._console_change_
                review_run_id` (DOM: reads the transcript's display model
                first) -- the change-review cluster stays screen-owned this
                wave.
            open_change_review: `ChatScreen._open_change_review` (`push_
                screen`) -- same cluster, same reason.
            start_console_transcript_sync_timer: `ChatScreen._start_console_
                transcript_sync_timer`, used by the four retry/continue/
                regenerate/edit-resend run launchers.
            clear_native_console_message_selection: `ChatScreen._clear_
                native_console_message_selection` (DOM), used only by
                `_continue_console_message`.
            regenerate_console_generation_variant: `ChatScreen._regenerate_
                console_generation_variant`, the image-generation cluster's
                own regenerate-append -- stays screen-owned this wave
                (image cluster explicitly out of scope), reached only from
                `handle_console_message_action`'s regenerate branch.
            select_console_generation_variant: `ChatScreen._select_console_
                generation_variant`, same cluster, same reason.
            keep_console_generation_variant: `ChatScreen._keep_console_
                generation_variant`, same cluster, same reason.
            handle_console_toggle_image_view: `ChatScreen._handle_console_
                toggle_image_view`, the image-VIEW cluster's own toggle --
                also out of scope this wave.
            invalidate_console_persisted_rows_cache: `ChatScreen._invalidate_
                console_persisted_rows_cache`, the conversation-browser
                cluster's cache invalidator (already a named callable on
                `session.py`), used only by `handle_console_message_
                action`'s delete branch.
            play_console_video: `ConsoleVideoController._play_console_video`,
                used by the video play action.
            save_console_video_copy: `ConsoleVideoController._save_console_
                video_copy`, used by the video save action.
            regenerate_console_video_message: `ConsoleVideoController._regenerate_
                console_video_message`, used by the video regenerate action.
        """
        self._screen = screen
        self.app_instance = app_instance
        self._chat_store_accessor = chat_store_accessor
        self._current_chat_store_accessor = current_chat_store_accessor
        self._ensure_console_chat_controller_fn = ensure_console_chat_controller
        self._current_chat_controller_accessor = current_chat_controller_accessor
        self._sync_native_console_chat_ui_fn = sync_native_console_chat_ui
        self._active_session_is_ephemeral_fn = active_session_is_ephemeral
        self._active_native_console_session_fn = active_native_console_session
        self._current_console_conversation_id_fn = current_console_conversation_id
        self._active_console_provider_model_display_fn = (
            active_console_provider_model_display
        )
        self._console_initial_session_title_for_workspace_fn = (
            console_initial_session_title_for_workspace
        )
        self._console_change_review_run_id_fn = console_change_review_run_id
        self._open_change_review_fn = open_change_review
        self._start_console_transcript_sync_timer_fn = (
            start_console_transcript_sync_timer
        )
        self._clear_native_console_message_selection_fn = (
            clear_native_console_message_selection
        )
        self._regenerate_console_generation_variant_fn = (
            regenerate_console_generation_variant
        )
        self._select_console_generation_variant_fn = select_console_generation_variant
        self._keep_console_generation_variant_fn = keep_console_generation_variant
        self._handle_console_toggle_image_view_fn = handle_console_toggle_image_view
        self._invalidate_console_persisted_rows_cache_fn = (
            invalidate_console_persisted_rows_cache
        )
        self._invalidate_console_fork_image_selections_fn = (
            invalidate_console_fork_image_selections or (lambda _message_ids: None)
        )
        self._play_console_video_fn = play_console_video
        self._save_console_video_copy_fn = save_console_video_copy
        self._regenerate_console_video_message_fn = regenerate_console_video_message
        self._request_console_chat_fork_fn = request_console_chat_fork or (
            lambda _message_id: None
        )

        # This cluster's own state, moved verbatim from `ChatScreen.__init__`.
        # `ChatScreen` keeps proxy properties under the original attribute
        # names for the members an external widget (`console_transcript.py`,
        # `_console_speaking_message_id`) or a staying screen method still
        # reads/writes -- see `chat_screen.py`'s own "Message cluster state"
        # comment block for the exact list.
        self._console_message_action_service = ConsoleMessageActionService()
        self._last_console_action: ConsoleActionResult | None = None
        self._pending_console_delete_message_id: str | None = None
        self._console_original_attempt_previews: Dict[str, str] = {}
        self._console_speaking_message_id: str | None = None
        self._console_speech_states: dict[str, str] = {}
        self._console_speech_request_generation = 0
        self._console_speech_lifetime_generation = 0
        self._console_speech_owner: Any | None = None
        self._console_speech_pending_stop: tuple[str, int] | None = None
        self._pending_console_swipe_selection: str | None = None

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

    # -- Named constructor dependencies -------------------------------------
    #
    # Each property below is a thin wrapper around a stored callable, kept
    # under the SAME name the original `ChatScreen` method/attribute used --
    # see `__init__`'s docstring. `_console_chat_store` is the one
    # bare-attribute-read shape (calls the accessor immediately and returns
    # the value); every other property returns the callable itself, matching
    # every other controller in this package.

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
    def _console_chat_controller(self) -> Any | None:
        return self._current_chat_controller_accessor()

    @property
    def _sync_native_console_chat_ui(self) -> Any:
        return self._sync_native_console_chat_ui_fn

    @property
    def _console_active_session_is_ephemeral(self) -> Any:
        return self._active_session_is_ephemeral_fn

    @property
    def _active_native_console_session(self) -> Any:
        return self._active_native_console_session_fn

    @property
    def _current_console_conversation_id(self) -> Any:
        return self._current_console_conversation_id_fn

    @property
    def _active_console_provider_model_display(self) -> Any:
        return self._active_console_provider_model_display_fn

    @property
    def _console_initial_session_title_for_workspace(self) -> Any:
        return self._console_initial_session_title_for_workspace_fn

    @property
    def _console_change_review_run_id(self) -> Any:
        return self._console_change_review_run_id_fn

    @property
    def _open_change_review(self) -> Any:
        return self._open_change_review_fn

    @property
    def _start_console_transcript_sync_timer(self) -> Any:
        return self._start_console_transcript_sync_timer_fn

    @property
    def _clear_native_console_message_selection(self) -> Any:
        return self._clear_native_console_message_selection_fn

    @property
    def _regenerate_console_generation_variant(self) -> Any:
        return self._regenerate_console_generation_variant_fn

    @property
    def _select_console_generation_variant(self) -> Any:
        return self._select_console_generation_variant_fn

    @property
    def _keep_console_generation_variant(self) -> Any:
        return self._keep_console_generation_variant_fn

    @property
    def _handle_console_toggle_image_view(self) -> Any:
        return self._handle_console_toggle_image_view_fn

    @property
    def _invalidate_console_persisted_rows_cache(self) -> Any:
        return self._invalidate_console_persisted_rows_cache_fn

    @property
    def _invalidate_console_fork_image_selections(self) -> Any:
        return self._invalidate_console_fork_image_selections_fn

    @property
    def _play_console_video(self) -> Any:
        if self._play_console_video_fn is None:
            raise RuntimeError("Console video play action is not wired")
        return self._play_console_video_fn

    @property
    def _save_console_video_copy(self) -> Any:
        if self._save_console_video_copy_fn is None:
            raise RuntimeError("Console video save action is not wired")
        return self._save_console_video_copy_fn

    @property
    def _regenerate_console_video_message(self) -> Any:
        if self._regenerate_console_video_message_fn is None:
            raise RuntimeError("Console video regenerate action is not wired")
        return self._regenerate_console_video_message_fn

    # -- Moved cluster methods (byte-for-byte except as documented above) --

    def _recent_console_image_messages(self, messages) -> list[Any]:
        """Return the most recent image-bearing messages, bounded to cache capacity.

        Mirrors the provider payload's most-recent-N image policy
        (``_provider_message_payloads``'s ``image_ids[-image_budget:]``).

        Excludes image-generation messages (non-empty ``generation_metadata``)
        -- those render as a ``"generation-card"`` row instead of the plain
        ``"image"`` row (see ``_build_generation_card_specs``), so including
        them here would double-render and burn a plain-image LRU slot under
        their bare message id for no row that ever reads it (TASK P2a-7).
        """
        # Bound the working set to the cache capacity so prep can never evict
        # what the transcript still shows (churn guard).
        image_messages = [
            message
            for message in messages
            if getattr(message, "image_data", None) is not None
            and not getattr(message, "generation_metadata", ())
        ]
        return image_messages[-IMAGE_CACHE_MAX_ENTRIES:]

    @staticmethod
    def _console_message_role_from_persisted(
        message: dict[str, Any],
    ) -> ConsoleMessageRole:
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
        """Build native Console messages from a persisted conversation tree.

        task-15860 Task 6: the walk itself moved to
        `Chat/console_conversation_hydration.py` so the launch wake -- which
        has to hydrate a conversation with no screen at all -- shares this
        policy instead of copying it. This method keeps its name and
        signature: eight test files call it directly.

        Task 8: flattens the ENTIRE tree (every node, all branches -- not just
        the ``children[-1]`` latest branch), each message carrying its
        ``persisted_message_id`` and persisted ``parent_message_id`` so the
        store can reconnect the full tree and pick the active branch from the
        stored active-leaf pointer.

        Parenthood is taken from the tree's own NESTING (the id of the node we
        recursed from), not the row's ``parent_message_id`` field: the real DB
        tree sets both consistently, but a node's structural position is the
        authoritative source and stays correct even for trees whose rows omit
        the field. A truly-empty node (no content and no image) is dropped but
        transparent to parenthood -- its children re-parent to the nearest kept
        ancestor -- so a skipped row never orphans a branch.
        """
        return console_messages_from_conversation_tree(
            tree, db=getattr(self.app_instance, "chachanotes_db", None)
        )

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
            "assistant_generation_state": getattr(
                message, "assistant_generation_state", None
            ),
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
            # Normalized provider usage (Console cost ticker): carried as the
            # same JSON string persistence uses, so a screen-state round trip
            # (navigate away and back) keeps a turn's real cost instead of
            # silently zeroing it. `getattr` tolerates plain-object stand-ins
            # that predate the field, like the neighbours above.
            "usage_json": (
                usage.to_json()
                if (usage := getattr(message, "usage", None)) is not None
                else None
            ),
            # Structured message metadata (task-2364): same reasoning as
            # `usage_json` above -- a screen-state round trip that dropped
            # it would lose the interrupted flag and a voice row's
            # transcript status, silently re-stranding what this field
            # exists to record. task-3401.4: a video row's payload lives in
            # ``video_metadata`` (mutually exclusive with ``metadata``) and
            # is preferred here so the round trip cannot strand it either.
            "metadata_json": (
                video_metadata.to_json()
                if (video_metadata := getattr(message, "video_metadata", None))
                is not None
                else (
                    metadata.to_json()
                    if (metadata := getattr(message, "metadata", None)) is not None
                    else None
                )
            ),
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
            str(payload["attachment_label"])
            if payload.get("attachment_label")
            else None
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
                str(payload["turn_id"]) if payload.get("turn_id") is not None else None
            ),
            status=status,  # type: ignore[arg-type]
            persisted_message_id=(
                str(payload["persisted_message_id"])
                if payload.get("persisted_message_id") is not None
                else None
            ),
            assistant_generation_state=(
                str(payload["assistant_generation_state"])
                if payload.get("assistant_generation_state") is not None
                else None
            ),
            variants=cls._restore_console_variants(payload.get("variants")),
            feedback=feedback,  # type: ignore[arg-type]
            image_mime_type=image_mime_type,
            attachment_label=attachment_label,
            attachments=attachments,
            # `from_json` returns None for missing/legacy/corrupt payloads,
            # which is exactly the "no usage known" state.
            usage=ProviderUsage.from_json(payload.get("usage_json")),
            # Same degrade-never-raise contract for structured metadata.
            # task-3401.4: video rows hydrate into video_metadata instead
            # (the two shapes never co-write one row).
            metadata=(
                None
                if (
                    video_metadata := VideoGenerationMetadata.from_json(
                        payload.get("metadata_json")
                    )
                )
                is not None
                else MessageMetadata.from_json(payload.get("metadata_json"))
            ),
            video_metadata=video_metadata,
        )

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
            rows_by_position = {int(row.get("position", 0)): row for row in extra_rows}
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

    def _rehydrate_console_message_generation_metadata(
        self,
        store: "ConsoleChatStore",
        restored_messages_by_session: Dict[str, list[ConsoleChatMessage]],
    ) -> None:
        """Batch-refill ``generation_metadata`` for messages restored from screen state.

        Counterpart of ``_rehydrate_console_message_attachments`` for the
        generation-metadata sidecar: `restore_state` -- the tab-switch
        (in-memory) restore path -- does not itself hydrate
        `generation_metadata`, unlike `restore_persisted_session` (the
        DB-resume path), which drives this same
        `get_generation_metadata_for_messages` +
        `ConsoleChatStore.hydrate_generation_metadata` seam internally. One
        batched call covers every restored message across every restored
        session in this pass, then `hydrate_generation_metadata` is invoked
        once per session (it filters by that session's own tree-node
        persisted ids, so handing it the whole merged mapping is safe and
        avoids a second per-session round trip). Must run AFTER
        `store.restore_state(...)`, which is what populates the store's
        tree nodes (and therefore `persisted_message_id` lookups)
        `hydrate_generation_metadata` needs. Any failure (missing DB,
        unreachable batch call) leaves messages metadata-only -- graceful
        degradation, matching `_rehydrate_console_message_attachments`'s
        own contract.

        Args:
            store: The Console chat store just populated by `restore_state`.
            restored_messages_by_session: The same per-session message lists
                passed to `store.restore_state(...)`.
        """
        persistence = getattr(store, "persistence", None)
        getter = getattr(persistence, "get_generation_metadata_for_messages", None)
        if not callable(getter):
            return
        ids = [
            message.persisted_message_id
            for messages in restored_messages_by_session.values()
            for message in messages
            if message.persisted_message_id
        ]
        if not ids:
            return
        try:
            rows_by_message = getter(ids)
        except Exception:
            logger.opt(exception=True).warning(
                "Console restore generation-metadata batch fetch failed."
            )
            return
        if not isinstance(rows_by_message, dict):
            return
        for session_id in restored_messages_by_session:
            store.hydrate_generation_metadata(session_id, rows_by_message)

    def _native_console_messages(self) -> list[Any]:
        """Return messages for the active native Console session."""
        store = self._ensure_console_chat_store()
        if store.active_session_id is None:
            return []
        return store.messages_for_session(store.active_session_id)

    @staticmethod
    def _console_citation_message_body(message: Any) -> str:
        """Return the exact currently selected body for one Console message."""
        variants = getattr(message, "variants", None)
        if variants is not None:
            try:
                body = variants.current.content
            except (AttributeError, IndexError):
                body = getattr(message, "content", "")
        else:
            body = getattr(message, "content", "")
        return body if isinstance(body, str) else ""

    async def _append_native_console_system_message(
        self, message: str, *, session_id: str | None = None
    ) -> None:
        """Append a system message to native Console state and refresh the bridge.

        Task 4 (background-write audit): most callers are synchronous
        command handlers with no ``await`` between "the user's intended
        session" and this call, so the default (``session_id=None``,
        resolving whichever session is active RIGHT NOW via
        ``store.ensure_session``) is safe -- there is no gap in which the
        active session could have changed underneath them.

        A handler that spans a real await gap while already anchored to a
        specific session (e.g. `/generate-image`'s in-flight batch, tracked
        per session in ``_console_imagegen_inflight_sessions``) must pass
        that session's id explicitly instead -- re-resolving "active" at
        append time would let a session switch during the awaited work
        misattribute the row to whatever the user is looking at NOW rather
        than the session that actually produced it. The resync below is
        unconditional either way and stays harmless: it only ever renders
        the store's CURRENTLY active session, so a background session's
        just-appended row simply doesn't show until the user visits it
        (store-first discipline; no view write needs gating here beyond
        that existing pull-based rebuild).
        """
        store = self._ensure_console_chat_store()
        if session_id is not None:
            try:
                store.append_message(
                    session_id,
                    role=ConsoleMessageRole.SYSTEM,
                    content=message,
                )
            except KeyError:
                # Session vanished (closed) before its background operation's
                # outcome could be attributed to it -- nothing to append to.
                pass
        else:
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

    async def request_console_message_speech(
        self,
        message_id: str,
        outcome_callback: Callable[[bool], None] | None = None,
        expected_destination_fingerprint: str | None = None,
        retry_failed_auto: bool = False,
    ) -> bool:
        """Dispatch Manual Speak's exact trusted snapshot/event path."""
        from tldw_chatbook.Chat.console_speech import ConsoleSpeechSnapshotRejected
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSMessageSpeechRequestEvent,
            TTSPlaybackEvent,
            TTSPlaybackLifecycle,
        )

        outcome_reported = False
        playback_lifecycle: TTSPlaybackLifecycle | None = None

        def report_outcome(ok: bool) -> None:
            nonlocal outcome_reported
            if outcome_reported:
                return
            outcome_reported = True
            if outcome_callback is None:
                return
            try:
                outcome_callback(ok is True)
            except Exception:
                return

        store = self._ensure_console_chat_store()
        try:
            speech_snapshot = store.issue_tts_message_speech_snapshot(
                message_id,
                presentation_context=self._screen._console_presentation_context(),
            )
        except ConsoleSpeechSnapshotRejected as error:
            self.app_instance.notify(str(error), severity="warning")
            report_outcome(False)
            return False
        except Exception:
            self.app_instance.notify(
                "Speech could not be requested. Try again.",
                severity="warning",
            )
            report_outcome(False)
            return False

        def validate_speech_snapshot(snapshot):
            return store.validate_tts_message_speech_snapshot(
                snapshot,
                presentation_context=self._screen._console_presentation_context(),
            )

        prior_message_id = self._console_speaking_message_id
        prior_owner = self._console_speech_owner
        if prior_message_id is not None:
            prior_generation = self._console_speech_request_generation
            prior_stop_outcome: bool | None = None
            self._console_speech_pending_stop = (
                prior_message_id,
                prior_generation,
            )

            def settle_prior_stop(accepted: bool) -> None:
                nonlocal prior_stop_outcome
                prior_stop_outcome = accepted is True
                pending = (prior_message_id, prior_generation)
                if self._console_speech_pending_stop != pending:
                    return
                self._console_speech_pending_stop = None
                if accepted and self._console_speech_states.get(prior_message_id) in {
                    "generating",
                    "playing",
                }:
                    self._settle_console_speech_presentation(
                        prior_message_id,
                        prior_generation,
                        state="stopped",
                    )

            stop_event = TTSPlaybackEvent(
                action="stop",
                message_id=prior_message_id,
                playback_lifecycle=prior_owner,
                outcome_callback=settle_prior_stop,
            )
            await self._dispatch_console_speech_stop_event(stop_event)
            if prior_stop_outcome is not True:
                self._console_speech_pending_stop = None
                report_outcome(False)
                return False

        request_generation = self._begin_console_speech_presentation(message_id)
        lifetime_generation = self._console_speech_lifetime_generation
        active_session_epoch = store.active_session_epoch()

        def playback_is_current() -> bool:
            if self._console_speech_request_generation != request_generation:
                return False
            if self._console_speech_lifetime_generation != lifetime_generation:
                return False
            if store.active_session_id != speech_snapshot.session_id:
                return False
            if store.active_session_epoch() != active_session_epoch:
                return False
            try:
                validate_speech_snapshot(speech_snapshot)
            except Exception:
                return False
            return True

        def report_playback(state: str) -> None:
            if playback_is_current():
                self._settle_console_speech_presentation(
                    message_id,
                    request_generation,
                    state=state,
                )
            if state == "stopped":
                report_outcome(True)
            elif state == "failed":
                report_outcome(False)

        playback_lifecycle = TTSPlaybackLifecycle(
            message_id=message_id,
            request_id=request_generation,
            validator=playback_is_current,
            callback=report_playback,
        )
        self._console_speech_owner = playback_lifecycle

        def report_generation_outcome(ok: bool) -> None:
            if ok is True:
                report_outcome(True)
            elif playback_lifecycle is not None:
                playback_lifecycle.report("failed")

        event = TTSMessageSpeechRequestEvent(
            speech_snapshot,
            validate_speech_snapshot,
            outcome_callback=report_generation_outcome,
            expected_destination_fingerprint=expected_destination_fingerprint,
            retry_failed_auto=retry_failed_auto,
            playback_lifecycle=playback_lifecycle,
        )
        try:
            posted = self.app_instance.post_message(event)
        except Exception:
            event.report_outcome(False)
            return False
        if posted is False:
            event.report_outcome(False)
            return False
        await self._sync_native_console_chat_ui()
        return True

    async def _dispatch_console_speech_stop_event(
        self,
        event: Any,
        *,
        post_first: bool = False,
    ) -> None:
        """Deliver one stop through the app handler with synchronous ack."""
        control = getattr(self.app_instance, "control_tts_playback", None)
        if not post_first and inspect.iscoroutinefunction(control):
            await control(event)
            return
        try:
            posted = self.app_instance.post_message(event)
        except Exception:
            posted = False
        if posted is not False:
            return
        handler = getattr(self.app_instance, "_tts_handler", None)
        handle = getattr(handler, "handle_tts_playback", None)
        if inspect.iscoroutinefunction(handle):
            try:
                await handle(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                event.report_outcome(False)
            return
        event.report_outcome(False)

    def _begin_console_speech_presentation(self, message_id: str) -> int:
        """Start one owned request and invalidate every older callback."""
        prior_message_id = self._console_speaking_message_id
        for other_id, state in tuple(self._console_speech_states.items()):
            if other_id == message_id:
                continue
            if other_id != prior_message_id or state in {"stopped", "failed"}:
                self._console_speech_states.pop(other_id, None)
        self._console_speech_request_generation += 1
        self._console_speech_states[message_id] = "generating"
        self._console_speaking_message_id = message_id
        return self._console_speech_request_generation

    def _settle_console_speech_presentation(
        self,
        message_id: str,
        generation: int,
        *,
        state: str,
    ) -> bool:
        """Accept one playback-owner state for the current screen request."""
        if self._console_speech_request_generation != generation:
            return False
        current = self._console_speech_states.get(message_id)
        allowed = {
            "generating": {"playing", "stopped", "failed"},
            "playing": {"stopped", "failed"},
        }
        if state not in allowed.get(current, set()):
            return False
        self._console_speech_states[message_id] = state
        if state == "playing":
            self._console_speaking_message_id = message_id
        elif self._console_speaking_message_id == message_id:
            self._console_speaking_message_id = None
            self._console_speech_owner = None
        self._schedule_console_speech_state_sync()
        return True

    def invalidate_console_speech_context(self) -> asyncio.Task[None] | None:
        """Fence callbacks while retaining audio ownership until stop ack."""
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSPlaybackEvent,
        )

        message_id = self._console_speaking_message_id
        owner = self._console_speech_owner
        self._console_speech_lifetime_generation += 1
        if message_id is None:
            self._console_speech_states.clear()
            return None
        generation = self._console_speech_request_generation
        store = self._ensure_console_chat_store()
        session_id = store.active_session_id
        session_epoch = store.active_session_epoch()
        lifetime_generation = self._console_speech_lifetime_generation
        pending = (message_id, generation)
        if self._console_speech_pending_stop == pending:
            return None
        self._console_speech_pending_stop = pending

        def context_is_current() -> bool:
            try:
                return bool(
                    self._ensure_console_chat_store() is store
                    and self._console_speech_lifetime_generation == lifetime_generation
                    and store.active_session_id == session_id
                    and store.active_session_epoch() == session_epoch
                )
            except Exception:
                return False

        def settle_invalidation_stop(accepted: bool) -> None:
            if self._console_speech_pending_stop != pending:
                return
            self._console_speech_pending_stop = None
            if self._console_speech_request_generation != generation:
                return
            if accepted:
                if context_is_current():
                    self._console_speech_states = {message_id: "stopped"}
                else:
                    self._console_speech_states.pop(message_id, None)
                if self._console_speech_owner is owner:
                    self._console_speech_owner = None
                if (
                    self._console_speaking_message_id == message_id
                    and self._console_speech_owner is None
                ):
                    self._console_speaking_message_id = None
            else:
                if context_is_current():
                    self._console_speech_states = {message_id: "failed"}
                else:
                    self._console_speech_states.pop(message_id, None)
                self._console_speech_owner = owner
                self._console_speaking_message_id = message_id
            self._schedule_console_speech_state_sync()

        stop_event = TTSPlaybackEvent(
            action="stop",
            message_id=message_id,
            playback_lifecycle=owner,
            outcome_callback=settle_invalidation_stop,
        )
        try:
            return asyncio.create_task(
                self._dispatch_console_speech_stop_event(
                    stop_event,
                    post_first=True,
                )
            )
        except RuntimeError:
            stop_event.report_outcome(False)
            return None

    def reconcile_console_speech_context(self) -> None:
        """Stop playback when the active store/screen owner became stale."""
        owner = self._console_speech_owner
        if owner is not None and not owner.is_current():
            self.invalidate_console_speech_context()

    def _schedule_console_speech_state_sync(self) -> None:
        """Repaint a callback-driven state change without blocking its reporter."""
        try:
            task = asyncio.create_task(self._sync_native_console_chat_ui())
        except RuntimeError:
            return

        def _consume_result(done: asyncio.Task) -> None:
            try:
                done.result()
            except Exception:
                logger.warning("Console speech-state repaint failed")

        task.add_done_callback(_consume_result)

    async def handle_console_message_action(self, event: Button.Pressed) -> bool:
        """Route a transcript message action through the native action service.

        The dispatcher for the transcript's per-message buttons: it decodes
        the button id, then routes to the cluster that owns that action. It
        stays one method because the routing table IS the unit -- splitting
        it would relocate the switch, not remove it.

        Args:
            event: The `Button.Pressed` from a transcript message action
                button, whose id encodes `<action>-<message id>`.

        Returns:
            bool: True when the id parsed and the action was dispatched
                (the event is then stopped); False when the button is not a
                message action, leaving the event to propagate.
        """
        button_id = event.button.id or ""
        action_id = getattr(event.button, "console_action_id", None)
        message_id = getattr(event.button, "console_message_id", None)
        if not isinstance(action_id, str) or not isinstance(message_id, str):
            action_id, message_id = self._parse_console_message_action_button_id(
                button_id
            )
        if action_id is None or message_id is None:
            return False

        event.stop()
        failed_cleared = False
        for failed_id, state in tuple(self._console_speech_states.items()):
            if state == "failed":
                self._console_speech_states.pop(failed_id, None)
                failed_cleared = True
        if failed_cleared:
            self._schedule_console_speech_state_sync()
        store = self._ensure_console_chat_store()

        if action_id == "review-changes":
            # TASK-2030 (live-UAT headline defect): the ✎ summary row is a
            # display-only TOOL marker -- deliberately NOT a tree node -- so
            # `store.get_message` can NEVER resolve it, and the pre-dispatch
            # lookup below killed the row's own advertised affordance
            # ("review with `v`") with the not-found toast on every press.
            # The run id is display data already ON the rendered row:
            # resolve it from the transcript's display model, falling back
            # to the store for tree-node rows. Every other action keeps the
            # store resolution (and its failure toast) untouched.
            self._pending_console_delete_message_id = None
            run_id = self._console_change_review_run_id(store, message_id)
            if run_id is None:
                self.app_instance.notify(
                    "Console message action target no longer exists.",
                    severity="warning",
                )
                return True
            self._open_change_review(run_id)
            return True

        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message action target no longer exists.", severity="warning"
            )
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

            await self.push_screen(
                ConsoleSaveAsModal(
                    destinations=destinations,
                    message_role=self._console_message_role_label(message),
                    message_excerpt=self._console_message_excerpt(message),
                    ephemeral=self._console_active_session_is_ephemeral(),
                ),
                callback=_apply_save_as,
            )
            self._last_console_action = ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Opened Save as destinations.",
            )
            return True

        if action_id == "fork":
            eligibility = self.console_fork_eligibility(message_id)
            if not eligibility.eligible:
                self.app_instance.notify(eligibility.reason, severity="warning")
                return True

        presentation = self._console_message_presentation(message)
        result = self._console_message_action_service.dispatch(action_id, message)
        if result.clipboard_text is not None:
            result = replace(result, clipboard_text=presentation.content)
        if (
            action_id in {"edit", "continue", "speak"}
            and result.target_content is not None
        ):
            result = replace(result, target_content=presentation.content)
        self._last_console_action = result
        if action_id == "fork" and result.status == "fork_requested":
            requested = self._request_console_chat_fork_fn(message_id)
            if inspect.isawaitable(requested):
                await requested
            return True
        if action_id == "view-original-attempt" and result.status == "completed":
            controller = self._ensure_console_chat_controller()
            original_attempt = controller.original_attempt_for_message(message_id)
            if original_attempt is None:
                self._console_original_attempt_previews.pop(message_id, None)
            elif message_id in self._console_original_attempt_previews:
                self._console_original_attempt_previews.pop(message_id, None)
            else:
                self._console_original_attempt_previews[message_id] = original_attempt
            await self._sync_native_console_chat_ui()
            return True
        if result.clipboard_text is not None:
            copy_to_clipboard = getattr(self.app_instance, "copy_to_clipboard", None)
            if callable(copy_to_clipboard):
                copy_to_clipboard(result.clipboard_text)
        if (
            action_id == "speak"
            and result.status == "completed"
            and not await self.request_console_message_speech(message.id)
        ):
            return True
        if action_id == "speak-stop" and result.status == "completed":
            # Reuses the legacy stop-button's exact plumbing (spec: "do not
            # invent a parallel audio-control path") -- safe to post
            # unconditionally, the app-level handler no-ops when nothing is
            # cached/playing for this message id.
            from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
                TTSPlaybackEvent,
            )

            was_speaking = self._console_speaking_message_id == message.id
            stop_generation = self._console_speech_request_generation
            stop_owner = self._console_speech_owner
            stop_pending = (message.id, stop_generation)
            if was_speaking:
                self._console_speech_pending_stop = stop_pending

            def settle_stop(accepted: bool) -> None:
                if not was_speaking:
                    return
                if self._console_speech_pending_stop != stop_pending:
                    return
                self._console_speech_pending_stop = None
                if self._console_speech_request_generation != stop_generation:
                    return
                retryable_owner = bool(
                    not accepted
                    and stop_owner is not None
                    and stop_owner.state not in {"stopped", "failed"}
                    and self._console_speech_owner is stop_owner
                )
                if not retryable_owner:
                    self._console_speech_request_generation += 1
                    if self._console_speech_owner is stop_owner:
                        self._console_speech_owner = None
                    if self._console_speaking_message_id == message.id:
                        self._console_speaking_message_id = None
                self._console_speech_states[message.id] = (
                    "stopped" if accepted else "failed"
                )
                self._schedule_console_speech_state_sync()
                if accepted:
                    self.app_instance.notify(
                        result.visible_copy,
                        severity="information",
                    )

            stop_event = TTSPlaybackEvent(
                action="stop",
                message_id=message.id,
                playback_lifecycle=stop_owner,
                outcome_callback=settle_stop if was_speaking else None,
            )
            await self._dispatch_console_speech_stop_event(stop_event)
            return True
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
            # rejection can run -- the screen must refuse, like the submit path.
            target_session_id = controller.store.active_session_id
            refusal = controller.send_refusal_copy(target_session_id)
            if refusal:
                self.app_instance.notify(refusal, severity="warning")
                return True
            self.run_worker(
                self._retry_console_message(controller, message_id),
                exclusive=True,
                group=f"console-run-{target_session_id}",
            )
            return True
        if action_id == "regenerate" and result.status == "wip":
            if message.generation_metadata:
                # A generation message's regenerate ALWAYS appends one new
                # image variant -- never an LLM text sibling -- so it skips
                # the controller/run-state gate entirely (that gate exists
                # for chat runs; image generation has its own in-flight
                # guard, checked inside this call).
                #
                # F5 follow-up (task-9 review): this is a fourth door onto
                # the same disk-writing sink /generate-image's dispatch
                # gate covers -- it calls `run_generation_batch` directly
                # and never passes through `_dispatch_console_command`, so
                # it needs its own check against the same registry entry.
                image_blocked = blocked_reason(
                    GENERATE_IMAGE_COMMAND_HANDLER_ID,
                    ephemeral=self._console_active_session_is_ephemeral(),
                )
                if image_blocked is not None:
                    self.app_instance.notify(image_blocked, severity="warning")
                    return True
                await self._regenerate_console_generation_variant(message_id)
                return True
            if getattr(message, "video_metadata", None) is not None:
                # task-3401.5: a video message's regenerate rebuilds the
                # request from the persisted video facts and appends a NEW
                # video message (there are no variants to swap). Same
                # ephemeral gate as /generate-video's command path.
                video_blocked = blocked_reason(
                    GENERATE_VIDEO_COMMAND_HANDLER_ID,
                    ephemeral=self._console_active_session_is_ephemeral(),
                )
                if video_blocked is not None:
                    self.app_instance.notify(video_blocked, severity="warning")
                    return True
                await self._regenerate_console_video_message(message_id)
                return True
            controller = self._ensure_console_chat_controller()
            target_session_id = controller.store.active_session_id
            refusal = controller.send_refusal_copy(target_session_id)
            if refusal:
                self.app_instance.notify(refusal, severity="warning")
                return True
            self.run_worker(
                self._regenerate_console_message(controller, message_id),
                exclusive=True,
                group=f"console-run-{target_session_id}",
            )
            return True
        if (
            action_id in {"variant-previous", "variant-next"}
            and result.status == "completed"
        ):
            landed_sibling_id: str | None = None
            if message.generation_metadata:
                self._select_console_generation_variant(message, direction=action_id)
            else:
                landed_sibling_id = self._select_console_message_variant(
                    message_id, direction=action_id
                )
            # task-501: keep the selection (and its action row) on the swapped
            # sibling so repeated `<`/`>` presses work without re-clicking the
            # row. The post-swipe view reaches the transcript asynchronously
            # (possibly coalesced), so hand the target off as a PENDING
            # selection the transcript applies at ingest time -- selecting
            # eagerly here would either miss its membership guard or be
            # cleared by reconciliation against the stale set. Other
            # selection-clearing actions ("continue" etc.) are untouched.
            if landed_sibling_id is not None:
                # Held on the screen (remount-proof); the sync below transfers
                # it onto whichever transcript instance receives the push.
                self._pending_console_swipe_selection = landed_sibling_id
            await self._sync_native_console_chat_ui()
            return True
        if action_id == "keep" and result.status == "completed":
            self._keep_console_generation_variant(message)
            await self._sync_native_console_chat_ui()
            self.app_instance.notify(result.visible_copy, severity="information")
            return True
        if (
            action_id in {"feedback-up", "feedback-down"}
            and result.status == "completed"
        ):
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
        if action_id == "video-play" and result.status == "completed":
            await self._play_console_video(message_id)
            return True
        if action_id == "video-save-copy" and result.status == "completed":
            self.run_worker(
                self._save_console_video_copy(message_id),
                exclusive=True,
                group="console-save-video",
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
            session_id = store.session_id_for_message(message_id)
            controller = self._ensure_console_chat_controller()
            # Deletion is subtree-wide, so clear the owning session while
            # descendant-to-session identity is still available.
            controller.clear_original_attempts_for_session(session_id)
            self._console_original_attempt_previews.clear()
            subtree_ids = store.subtree_message_ids(message_id)
            store.delete_message(message_id)
            self._invalidate_console_fork_image_selections(subtree_ids)
            # TASK-251: a deleted message can change what the browser row
            # shows for this conversation (title/updated_at) -- invalidate
            # so the next sync reflects it immediately.
            self._invalidate_console_persisted_rows_cache()
            await self._sync_native_console_chat_ui()
            self.app_instance.notify(result.visible_copy, severity="information")
            return True
        if action_id == "continue" and result.status == "continue_requested":
            controller = self._ensure_console_chat_controller()
            target_session_id = controller.store.active_session_id
            refusal = controller.send_refusal_copy(target_session_id)
            if refusal:
                self.app_instance.notify(refusal, severity="warning")
                return True
            self.run_worker(
                self._continue_console_message(controller, message_id),
                exclusive=True,
                group=f"console-run-{target_session_id}",
            )
            return True
        severity = "information" if result.status in {"completed", "wip"} else "warning"
        self.app_instance.notify(result.visible_copy, severity=severity)
        return True

    def console_fork_eligibility(self, message_id: str) -> ConsoleForkEligibility:
        """Return the store-owned frozen eligibility for one rendered boundary."""
        try:
            return self._ensure_console_chat_store().fork_eligibility(message_id)
        except (KeyError, ValueError):
            return ConsoleForkEligibility(False, "Message is not forkable.")

    def sync_selected_fork_eligibility(
        self, transcript: Any
    ) -> tuple[str | None, ConsoleForkEligibility | None]:
        """Push the selected row's current fork eligibility into a transcript."""
        selected_id = transcript.selected_message_id
        eligibility = (
            self.console_fork_eligibility(selected_id)
            if selected_id is not None
            else None
        )
        if selected_id is not None:
            transcript.set_fork_eligibilities({selected_id: eligibility})
        return selected_id, eligibility

    def _console_message_presentation(
        self, message: ConsoleChatMessage
    ) -> ConsoleMessagePresentation:
        """Delegate to the screen-owned active-session presentation resolver."""
        return self._screen._console_message_presentation(message)

    def _console_message_role_label(self, message: ConsoleChatMessage) -> str:
        """Return a user-facing role label for a Console transcript message."""
        return self._console_message_presentation(message).speaker_label

    def _console_message_content(self, message: ConsoleChatMessage) -> str:
        """Return the currently visible content for a Console transcript message."""
        return self._console_message_presentation(message).content

    def _console_message_excerpt(
        self,
        message: ConsoleChatMessage,
        *,
        max_length: int = 120,
    ) -> str:
        """Return a single-line excerpt for selected-message context surfaces."""
        normalized = " ".join(self._console_message_content(message).split())
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[: max(0, max_length - 1)].rstrip()}…"

    def _console_save_as_destinations(self, message: Any) -> list[Any]:
        """Return Save-as destinations available in the current app runtime.

        A temporary session blocks every destination outright: the write
        itself is the problem, so service readiness is moot and is never
        even checked in that case.
        """
        available_destinations: set[str] = set()
        unavailable_reasons: dict[str, str] = {}

        if self._console_active_session_is_ephemeral():
            # F4 (task-9 review): `blocked_reason` returns `str | None`;
            # every label above has a registry entry today, so this is
            # never None in practice, but `unavailable_reasons` is typed
            # `dict[str, str]` -- fall back to a generic sentence instead
            # of silently letting a future registry-key drift surface the
            # literal string "None" on the modal.
            for label in ("Chatbook", "Note", "Media", "Prompt"):
                unavailable_reasons[label] = (
                    blocked_reason(f"save-as-{label.lower()}", ephemeral=True)
                    or "Not available in a temporary chat."
                )
            return ConsoleMessageActionService(
                available_save_destinations=set(),
                unavailable_save_reasons=unavailable_reasons,
            ).save_as_destinations(message)

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
            unavailable_reasons["Prompt"] = (
                "Prompts service is not ready in this session."
            )

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

            def _fetch_persisted_attachment_data() -> dict[
                int, tuple[Any, Optional[str]]
            ]:
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
                        mime_type=fetched[attachment.position][1]
                        or attachment.mime_type,
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
                user_id=getattr(self.app_instance, "current_user", None)
                or "default_user",
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
        # FB-07 (TASK-2154.17): success confirmations read as success.
        self.app_instance.notify("Saved message as Note.", severity="success")

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
            severity="success",
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
            severity="success",
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
        coordinator = getattr(
            self.app_instance,
            "citation_artifact_ownership_coordinator",
            None,
        )
        owner_request = resolve_console_artifact_owner_request(
            coordinator=coordinator,
            persisted_message_id=message.persisted_message_id,
            message_text=content,
        )
        if owner_request is not None:
            payload["provenance_owner_request"] = owner_request
        try:
            result = create_chatbook(**payload)
            if inspect.isawaitable(result):
                await result
            if owner_request is not None and coordinator is not None:
                try:
                    await asyncio.to_thread(coordinator.reconcile_pending, limit=1)
                except Exception:
                    logger.warning(
                        "Citation artifact reconciliation deferred: "
                        "artifact_reconciliation_failed"
                    )
        except Exception as exc:
            logger.opt(exception=True).warning("Console save-as Chatbook failed.")
            self.app_instance.notify(
                f"Save as Chatbook failed: {exc}", severity="error"
            )
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
            # FB-07 (TASK-2154) moved every Save-as confirmation to "success";
            # Chatbook was missed and kept "information" for four days because
            # the test that pinned all four destinations called a seam
            # decomposition wave 3 had already moved, so it never ran
            # (task-14920).
            severity="success",
        )

    def _clear_console_original_attempt_preview(self, message_id: str) -> None:
        """Clear one screen preview and its controller-owned cached body."""
        self._console_original_attempt_previews.pop(message_id, None)
        controller = self._console_chat_controller
        if controller is not None:
            controller.clear_original_attempt(message_id)

    async def _open_console_message_edit_modal(
        self, *, message_id: str, content: str
    ) -> None:
        """Open the dedicated transcript edit modal for one Console message."""
        store = self._ensure_console_chat_store()
        try:
            message = store.get_message(message_id)
        except KeyError:
            self.app_instance.notify(
                "Console message action target no longer exists.",
                severity="error",
            )
            return
        can_resend = message.role is ConsoleMessageRole.USER
        clears_generation_provenance = bool(
            message.role is ConsoleMessageRole.ASSISTANT
            and (
                message.thinking is not None
                or message.opaque_thinking_json is not None
                or message.provider_continuation is not None
            )
        )

        def _apply_edit(result: ConsoleEditResult | None) -> None:
            if result is None:
                return
            self._clear_console_original_attempt_preview(message_id)
            if not result.resend:
                try:
                    store.update_message_content(message_id, result.text)
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
                    target_content=result.text,
                )
                self.run_worker(
                    self._sync_native_console_chat_ui(),
                    exclusive=True,
                    group="console-sync",
                )
                self.app_instance.notify("Edited message.", severity="information")
                return
            controller = self._ensure_console_chat_controller()
            # Gate BEFORE spawning: an exclusive console-run worker cancels the
            # in-flight run at creation time, before the controller's own
            # rejection can run -- the screen must refuse, like the submit path.
            target_session_id = controller.store.active_session_id
            refusal = controller.send_refusal_copy(target_session_id)
            if refusal:
                self.app_instance.notify(refusal, severity="warning")
                return
            self.run_worker(
                self._edit_resend_console_message(controller, message_id, result.text),
                exclusive=True,
                group=f"console-run-{target_session_id}",
            )

        await self.push_screen(
            ConsoleEditMessageModal(
                content=content,
                can_resend=can_resend,
                clears_generation_provenance=clears_generation_provenance,
            ),
            callback=_apply_edit,
        )

    @staticmethod
    def _parse_console_message_action_button_id(
        button_id: str,
    ) -> tuple[str | None, str | None]:
        prefixes = (
            (
                "console-message-action-view-original-attempt-",
                "view-original-attempt",
            ),
            ("console-message-action-feedback-up-", "feedback-up"),
            ("console-message-action-feedback-down-", "feedback-down"),
            ("console-message-action-variant-previous-", "variant-previous"),
            ("console-message-action-variant-next-", "variant-next"),
            ("console-message-action-keep-", "keep"),
            ("console-message-action-review-changes-", "review-changes"),
            ("console-message-action-save-as-", "save-as"),
            ("console-message-action-save-image-", "save-image"),
            ("console-message-action-video-play-", "video-play"),
            ("console-message-action-video-save-copy-", "video-save-copy"),
            ("console-message-action-toggle-image-view-", "toggle-image-view"),
            ("console-message-action-regenerate-", "regenerate"),
            ("console-message-action-continue-", "continue"),
            ("console-message-action-delete-", "delete"),
            ("console-message-action-retry-", "retry"),
            # speak-stop MUST be checked before speak -- "speak-" is itself
            # a prefix of "speak-stop-", so the more specific entry has to
            # win the ordered startswith() scan below (else a speak-stop
            # button id would mis-parse as action "speak" with message id
            # "stop-<real id>").
            ("console-message-action-speak-stop-", "speak-stop"),
            ("console-message-action-speak-", "speak"),
            ("console-message-action-copy-", "copy"),
            ("console-message-action-edit-", "edit"),
            ("console-message-action-fork-", "fork"),
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
        # TASK-343: without the sync timer these awaits run the whole
        # generation with zero on-screen feedback (the timer self-stops
        # when the run leaves an active status).
        self._start_console_transcript_sync_timer()
        result = await controller.retry_message(message_id)
        if result.accepted:
            # FB-07 (TASK-2154.17): a deliberate retry used to confirm at
            # "information" severity or stay silent; make the positive
            # confirmation explicit. The toast carries STATUS copy, never
            # `result.visible_copy` -- on an accepted retry that is the
            # recovered assistant text, which belongs in the transcript.
            # One toast per click on a failed row is not spam -- progress
            # then lives on the run-state surface.
            self.app_instance.notify(
                "Retrying failed response.",
                severity="success",
            )
        elif result.visible_copy:
            self.app_instance.notify(result.visible_copy, severity="warning")
        await self._sync_native_console_chat_ui()

    async def _continue_console_message(
        self,
        controller: ConsoleChatController,
        message_id: str,
    ) -> None:
        self._start_console_transcript_sync_timer()
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
        self._start_console_transcript_sync_timer()
        result = await controller.regenerate_message(message_id)
        if result.visible_copy and not result.accepted:
            self.app_instance.notify(result.visible_copy, severity="warning")
        await self._sync_native_console_chat_ui()

    async def _edit_resend_console_message(
        self,
        controller: ConsoleChatController,
        message_id: str,
        new_content: str,
    ) -> None:
        # TASK-343: without the sync timer these awaits run the whole
        # generation with zero on-screen feedback (the timer self-stops
        # when the run leaves an active status).
        self._start_console_transcript_sync_timer()
        result = await controller.edit_and_resend_message(message_id, new_content)
        if result.visible_copy and not result.accepted:
            self.app_instance.notify(result.visible_copy, severity="warning")
        await self._sync_native_console_chat_ui()

    def _select_console_message_variant(
        self, message_id: str, *, direction: str
    ) -> str | None:
        """Move the active leaf across ``message_id``'s persisted siblings.

        Returns the target sibling's native id when the swipe moved (so the
        caller can re-select that row after the UI sync -- task-501: repeated
        ``<``/``>`` presses must not require re-clicking the row), or ``None``
        on a no-op at either end of the sibling list.

        ``message_id`` identifies the transcript ROW the swipe control was
        clicked on -- this may be off the CURRENT active leaf's own subtree
        (e.g. after a previous swipe landed deep inside a sibling's branch),
        so sibling lookup always resolves from ``message_id`` itself via
        ``store.siblings_at`` (works for off-path nodes too), never from
        ``store.active_leaf``. The target sibling's own most-recent
        descendant (``store._leaf_under``) becomes the new active leaf, so
        swiping back into a branch that was mid-conversation resumes at its
        deepest turn rather than snapping back to the fork point. A no-op at
        either end of the sibling list (nothing before the first / after the
        last) leaves the active leaf untouched -- the caller re-syncs the UI
        either way, which is harmless when nothing moved.
        """
        store = self._ensure_console_chat_store()
        siblings, index, count = store.siblings_at(message_id)
        if direction == "variant-previous":
            target_index = index - 1
        elif direction == "variant-next":
            target_index = index + 1
        else:
            return None
        if target_index < 0 or target_index >= count:
            return None
        target_sibling_id = siblings[target_index].id
        session_id = store.session_id_for_message(message_id)
        store.set_active_leaf(session_id, store._leaf_under(target_sibling_id))
        return target_sibling_id

    def _console_transcript_has_messages(self) -> bool:
        """Return whether the active Console transcript has user/session content.

        Dead code, pre-existing (task-1 extraction audit, wave-3): zero
        callers anywhere in the repository. Moved verbatim rather than
        fixed/removed -- behaviour changes are out of scope for this
        extraction; see the task-1 report.
        """
        if self._console_chat_store is not None:
            session_id = self._console_chat_store.active_session_id
            if session_id is not None and self._console_chat_store.has_messages(
                session_id
            ):
                return True

        return False

    def _active_console_transcript_has_messages(self) -> bool:
        """Return whether the active Console session's store transcript has messages."""
        store = self._console_chat_store
        if store is None:
            return False
        session_id = store.active_session_id
        if session_id is None:
            return False
        # TASK-24300: emptiness only -- no snapshot of the transcript needed.
        return store.has_messages(session_id)
