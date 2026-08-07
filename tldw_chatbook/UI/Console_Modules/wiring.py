"""One place where the whole Console controller graph is wired.

Wave-4 console decomposition, task 1. `ChatScreen.__init__` had grown to 782
lines, of which 411 were the six `Console*Controller(...)` constructions
buried among ~250 unrelated attribute assignments -- so the one thing worth
reading as a whole (which controller depends on what, and in which order the
graph is built) was the hardest thing in the file to see.

This is a **move, not a redesign**. Every named keyword argument below is
character-for-character what `__init__` passed; only the call site changed
(`self` -> the `screen` parameter, a token-exact rename that touched no
comment and no string literal). A reviewer on PR #1408 proposed collapsing
this wiring into per-controller dependency objects; that was declined,
because a controller's dependencies being visible in its signature -- and at
its one call site -- is the entire point of the binding rule. Do not
factor out a shared helper, deduplicate similar lambdas, or "tidy" a kwarg
here: each of those hides exactly what this module exists to expose.

The binding rule itself is stated canonically in
`ConsoleDictationController.__init__`'s docstring (`dictation.py`). In
summary, as it applies to this file:

- App dependencies are **named keyword-only callables**, wired as
  **late-binding lambdas** (`lambda: screen.foo()`), never bound methods --
  a bound method captured here freezes the method as it was at construction
  time and stops observing a later `monkeypatch.setattr` on the screen
  instance.
- **Cross-controller** dependencies resolve the sibling at CALL time
  (`lambda: screen._session._x()`), which is why the build order below,
  although fixed and documented, is not load-bearing behaviour: `_workspace`
  legitimately names `_session` before `_session` exists.
- `app_instance` is the one justified snapshot (plain attribute, not a
  callable): it never changes identity over a controller's life.
- Controllers own no DOM.

`Screens/chat_screen.py` still imports all six controller CLASSES even though
it no longer names four of them in code. That is deliberate, and re-deleting
them is a regression: 18 sites across five test files reach them as
`chat_screen_module.ConsoleDictationController` /
`...ConsoleWorkspaceController` patch handles, and dropping the imports
turned 28 of those tests red during this very extraction. They are re-export
surface, in the same way as the ~30 other unused-but-imported names that file
already carries.
"""

from typing import TYPE_CHECKING

from .dictation import ConsoleDictationController
from .hands_free import ConsoleHandsFreeController
from .message import ConsoleMessageController
from .prompts import ConsolePromptsController
from .session import ConsoleSessionController
from .workspace import ConsoleWorkspaceController

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..Screens.chat_screen import ChatScreen

__all__ = ["build_console_controllers"]


def build_console_controllers(screen: "ChatScreen") -> None:
    """Construct the Console screen's six controllers and attach them.

    Assigns, in this order, `screen._workspace`, `screen._session`,
    `screen._dictation`, `screen._hands_free`, `screen._message` and
    `screen._prompts`. The order is documentation, not a constraint: every
    cross-controller dependency below is resolved at call time (see the
    module docstring), so no controller reads a sibling that does not exist
    yet.

    `ChatScreen.__init__` calls this at exactly the point the first
    construction used to occupy. That position matters: the ~250 attribute
    assignments around it in `__init__` include names these lambdas read, and
    none of the six constructors reads anything off `screen` eagerly (each
    stores `screen` and its callables and nothing else), so the call needs to
    sit where it can see everything the pre-move constructions could.

    Args:
        screen: The Console screen (`ChatScreen`) to wire. Mutated in place;
            taken as a parameter rather than imported so this module has no
            import cycle with `Screens/chat_screen.py`.

    Returns:
        None. The controllers are reachable as attributes of `screen`.
    """
    #: Workspace policy context, lifecycle, and resume-flow state and
    #: behaviour moved to `ConsoleWorkspaceController` (wave-2 console
    #: decomposition, task 2) -- the largest cluster in wave 2.
    #: `self._console_workspace_conversation_query`/`_search_timer`/
    #: `_search_token`/`_search_rows`/`_search_total`/`_search_error`
    #: stay readable/writable via the six proxy properties defined near
    #: `_console_composer_or_none`, so `on_console_workspace_
    #: conversation_search_changed` (a sibling conversation-browser
    #: handler that only mirrors three of them) and `on_button_pressed`'s
    #: workspace-search branches needed no change. See `workspace.py`'s
    #: module docstring for the full map of what moved and why.
    screen._workspace = ConsoleWorkspaceController(
        screen,
        app_instance=screen.app_instance,
        # Late-binding lambdas, not the bound methods directly -- same
        # staleness reason as `ConsoleDictationController`'s own wiring
        # just below: a bound method captured here would freeze the
        # CURRENT screen method, invisible to a later
        # `monkeypatch.setattr` on this screen instance.
        chat_store_accessor=lambda: screen._ensure_console_chat_store(),
        current_chat_store_accessor=lambda: screen._console_chat_store,
        current_conversation_id_accessor=(
            lambda: screen._session._current_console_conversation_id()
        ),
        native_session_rows_accessor=(
            lambda state: screen._session._with_native_console_session_rows(state)
        ),
        conversation_browser_state_accessor=(
            lambda state, current_conversation_id: (
                screen._with_console_conversation_browser_state(
                    state, current_conversation_id=current_conversation_id
                )
            )
        ),
        capture_draft_switch_snapshot=(
            lambda: screen._session._capture_console_draft_switch_snapshot()
        ),
        sync_chat_core_state=lambda: screen._sync_console_chat_core_state(),
        sync_native_console_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        sync_temporary_chip=lambda: screen._sync_console_temporary_chip(),
        default_session_settings_accessor=(
            lambda: screen._session._default_console_session_settings()
        ),
        scope_picker_listers_accessor=(
            lambda: screen._console_scope_picker_listers()
        ),
        active_native_session_accessor=(
            lambda: screen._session._active_native_console_session()
        ),
        refresh_effective_scope_and_sync=(
            lambda session: screen._refresh_console_effective_scope_and_sync(
                session
            )
        ),
        # Message <-> workspace seam (the reverse direction of the
        # session/message seams the message controller's own
        # constructor takes): the resume flow's tree flattener now
        # lives on `ConsoleMessageController` (wave-3 task 1). This
        # accessor already existed as a named callable before that
        # move (wave-2 task 2); only its target changed, from the
        # screen's own delegation to the controller directly. Python
        # resolves `self._message` at CALL time, so construction order
        # (workspace built before message) does not matter.
        messages_from_conversation_tree_accessor=(
            lambda tree: screen._message._console_messages_from_conversation_tree(
                tree
            )
        ),
        session_settings_for_resume_accessor=(
            lambda conversation: screen._session._console_session_settings_for_resume(
                conversation
            )
        ),
        resolve_resumed_character_name=(
            lambda character_id: screen._resolve_resumed_character_name(
                character_id
            )
        ),
        inject_resume_agent_markers_accessor=(
            lambda messages, conversation_id: screen._inject_resume_agent_markers(
                messages, conversation_id
            )
        ),
        resolve_effective_scope_state=(
            lambda session: screen._resolve_console_effective_scope_state(session)
        ),
        sync_retrieval_scope_row=(
            lambda: screen._sync_console_retrieval_scope_row()
        ),
        note_follow_intent=lambda: screen._note_console_follow_intent(),
        focus_composer_if_needed=(
            lambda **kwargs: screen._focus_console_composer_if_needed(**kwargs)
        ),
        conversation_section_config_accessor=(
            lambda: screen._console_conversation_section_config()
        ),
        # The grouped browser's collapse preferences live in the screen's
        # own `app_config` accessors (alongside rail-state and search
        # preferences that are not this cluster's), so the write stays
        # there and the two toggle branches moved in wave-4 task 2 reach
        # it by name.
        set_conversation_browser_group_collapsed=(
            lambda group_id, collapsed: (
                screen._set_console_conversation_browser_group_collapsed(
                    group_id, collapsed
                )
            )
        ),
        focus_conversation_search=(
            lambda: screen._focus_console_workspace_conversation_search()
        ),
        sync_workspace_context=lambda: screen._sync_console_workspace_context(),
    )
    #: Native session lifecycle -- start/activate/swap/promote/rename,
    #: per-session settings, the Ctrl+K switcher's choice handling,
    #: draft sync, and one-session (de)serialization -- moved to
    #: `ConsoleSessionController` (wave-2 console decomposition, task 3).
    #: `self._console_visible_draft_session_id`/`_console_undo_histories`
    #: stay readable/writable via the two proxy properties defined near
    #: `_console_composer_or_none`, so nothing outside this cluster
    #: (screen-state (de)serialization, the submit path, `on_button_
    #: pressed`'s tab-close branch, tests) had to change.
    #: `_console_active_session_is_ephemeral` keeps a one-line
    #: delegation instead of a proxy property (see that method's own
    #: docstring: an external, non-controller consumer reaches it by
    #: bare name off `self.screen`). See `session.py`'s module
    #: docstring for the full map of what moved and why.
    screen._session = ConsoleSessionController(
        screen,
        app_instance=screen.app_instance,
        chat_store_accessor=lambda: screen._ensure_console_chat_store(),
        current_chat_store_accessor=lambda: screen._console_chat_store,
        ensure_console_chat_controller=(
            lambda: screen._ensure_console_chat_controller()
        ),
        composer_accessor=lambda: screen._console_composer_or_none(),
        effective_console_provider_model=(
            lambda: screen._effective_console_provider_model()
        ),
        provider_readiness_app_config=(
            lambda: screen._provider_readiness_app_config()
        ),
        sync_native_console_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        sync_chat_core_state=lambda: screen._sync_console_chat_core_state(),
        sync_temporary_chip=lambda: screen._sync_console_temporary_chip(),
        sync_settings_summary=lambda: screen._sync_console_settings_summary(),
        sync_control_bar=lambda: screen._sync_console_control_bar(),
        sync_command_popup=lambda: screen._sync_console_command_popup(),
        note_follow_intent=lambda: screen._note_console_follow_intent(),
        focus_composer_if_needed=(
            lambda **kwargs: screen._focus_console_composer_if_needed(**kwargs)
        ),
        invalidate_persisted_rows_cache=(
            lambda: screen._invalidate_console_persisted_rows_cache()
        ),
        mark_conversation_row_broken=(
            lambda conversation_id: screen._mark_console_conversation_row_broken(
                conversation_id
            )
        ),
        refresh_effective_scope_and_sync=(
            lambda session: screen._refresh_console_effective_scope_and_sync(
                session
            )
        ),
        # Session<->workspace seam (design spec: "a named callable
        # between them; design it deliberately, never a back-door
        # through the screen"). `self._workspace` was constructed just
        # above; Python resolves it at CALL time inside these lambdas,
        # so construction order does not matter.
        set_active_workspace_for_session=(
            lambda session_id: (
                screen._workspace._set_active_workspace_for_console_session(
                    session_id
                )
            )
        ),
        resume_workspace_conversation=(
            lambda conversation_id, **kwargs: (
                screen._workspace._resume_console_workspace_conversation(
                    conversation_id, **kwargs
                )
            )
        ),
        workspace_initial_session_title=(
            lambda workspace_id: (
                screen._workspace._console_initial_session_title_for_workspace(
                    workspace_id
                )
            )
        ),
        merge_workspace_rows=(
            lambda native_rows, rows: screen._workspace._merge_console_workspace_rows(
                native_rows, rows
            )
        ),
        session_id_for_workspace_conversation=(
            lambda row_key: (
                screen._workspace._console_session_id_for_workspace_conversation(
                    row_key
                )
            )
        ),
        # The inline-image render cache stays screen-owned (out of scope
        # this wave); `_close_console_session_tab` (wave-4 task 2) drops a
        # closing tab's cached renders through this named callable rather
        # than reaching `screen._ensure_console_image_view` itself.
        evict_console_image_cache=(
            lambda message_ids: screen._evict_console_image_cache(message_ids)
        ),
    )
    #: Dictation's own state and lifecycle moved to
    #: `ConsoleDictationController` (wave-1 console decomposition,
    #: task 5) -- this is wave 1's proof of the controller collaborator
    #: kind. `self._console_dictation_*` / `self._console_pending_
    #: voice_action` stay readable/writable via the properties defined
    #: near `_console_composer_or_none`, so nothing outside this
    #: cluster (the hands-free wiring, `on_button_pressed`, tests) had
    #: to change. See `dictation.py`'s module docstring for the full
    #: map of what moved and why.
    screen._dictation = ConsoleDictationController(
        screen,
        app_instance=screen.app_instance,
        # Late-binding lambdas, not the bound methods directly: a
        # bound method captured here would freeze the CURRENT
        # `_console_composer_or_none`/`_ensure_console_chat_store`/
        # `_speak_status`, invisible to a later `monkeypatch.setattr`
        # on this screen instance. The lambda instead closes over
        # `self` (this screen, whose identity never changes) and does
        # the attribute lookup at CALL time, so a later instance-level
        # patch is still picked up -- see `ConsoleDictationController.
        # __init__`'s docstring, binding kind 3.
        composer_accessor=lambda: screen._console_composer_or_none(),
        chat_store_accessor=lambda: screen._ensure_console_chat_store(),
        speak_status=lambda reason: screen._speak_status(reason),
        # The four reach-backs wave 1 left as a disclosed, temporary
        # exception (bare `self._screen.X` properties) now that hands-
        # free has a controller of its own to hand a named dependency
        # to (wave-2 console decomposition, task 1) -- late-binding
        # lambdas onto `self._hands_free`, constructed just below.
        # Python resolves `self._hands_free` at CALL time, not here, so
        # construction order between the two controllers does not
        # matter.
        hands_free_session_accessor=lambda: screen._hands_free._console_hands_free,
        set_hands_free_vad_degraded=(
            lambda value: setattr(
                screen._hands_free, "_console_hands_free_vad_degraded", value
            )
        ),
        enter_hands_free_loop=(
            lambda **kwargs: screen._hands_free._enter_console_hands_free_loop(
                **kwargs
            )
        ),
        hands_free_force_immediate_send=(
            lambda: screen._hands_free._console_hands_free_force_immediate_send()
        ),
        deliver_hands_free_capture_ended=(
            lambda session, had_segments: (
                screen._hands_free._deliver_console_hands_free_capture_ended(
                    session, had_segments
                )
            )
        ),
        # Realtime stays screen-owned (not extracted this wave); this
        # was the same "temporary exception" shape as the four above,
        # for the same staleness reason, now closed out the same way.
        realtime_adopt_transcript=(
            lambda transcript: screen._console_realtime_adopt_transcript(
                transcript
            )
        ),
        # Same screen-owned realtime engine, read as a live session rather
        # than called: `_handle_console_dictation_button` (wave-4 task 2)
        # must see a loop that started AFTER this wiring ran, so this is a
        # late-binding accessor, never a snapshot. `ConsoleHandsFree
        # Controller` takes an identically-named, identically-shaped one.
        realtime_session_accessor=lambda: screen._console_realtime,
        run_pending_voice_action=(
            lambda session_id: screen._run_pending_console_voice_action(
                session_id
            )
        ),
        undo_histories_accessor=lambda: screen._console_undo_histories,
        visible_draft_session_id_accessor=(
            lambda: screen._console_visible_draft_session_id
        ),
    )
    #: The V3 pipeline hands-free loop's state and lifecycle, plus the
    #: two-engine fork/action coordination it shares with the V4
    #: realtime loop, moved to `ConsoleHandsFreeController` (wave-2
    #: console decomposition, task 1). `self._console_hands_free`/
    #: `_console_hands_free_vad_degraded` stay readable/writable via
    #: the properties defined near `_console_composer_or_none`, so
    #: nothing outside this cluster (`on_key`, `on_button_pressed`,
    #: `on_unmount`, the realtime engine's own loud fallback, tests)
    #: had to change. See `hands_free.py`'s module docstring for the
    #: full map of what moved and why, including the two-engine
    #: boundary this controller draws around the still-on-screen
    #: realtime engine.
    screen._hands_free = ConsoleHandsFreeController(
        screen,
        app_instance=screen.app_instance,
        composer_accessor=lambda: screen._console_composer_or_none(),
        chat_store_accessor=lambda: screen._ensure_console_chat_store(),
        dictation_state_accessor=lambda: screen._console_dictation_state,
        dictation_origin_session_id_accessor=(
            lambda: screen._console_dictation_origin_session_id
        ),
        set_pending_voice_action=(
            lambda value: setattr(screen, "_console_pending_voice_action", value)
        ),
        request_dictation_start=lambda: screen._request_console_dictation_start(),
        request_dictation_stop=lambda: screen._request_console_dictation_stop(),
        run_pending_voice_action=(
            lambda session_id: screen._run_pending_console_voice_action(
                session_id
            )
        ),
        realtime_session_accessor=lambda: screen._console_realtime,
        enter_realtime_loop=(
            lambda capture_live: screen._enter_console_realtime_loop(
                capture_live=capture_live
            )
        ),
    )
    #: The native message-transcript cluster -- serialize/restore,
    #: resume-tree flattening, screen-state rehydration, per-message
    #: save-as/edit/retry/continue/regenerate/variant navigation, and
    #: `handle_console_message_action`'s full dispatch -- moved to
    #: `ConsoleMessageController` (wave-3 console decomposition, task
    #: 1), the largest cluster in wave 3. `self._console_message_action_
    #: service`/`_last_console_action`/`_pending_console_delete_message_
    #: id`/`_console_original_attempt_previews`/`_console_speaking_
    #: message_id`/`_pending_console_swipe_selection` stay readable/
    #: writable via the proxy properties defined near
    #: `_console_composer_or_none`, so nothing outside this cluster (a
    #: few DOM-touching siblings, `console_transcript.py`'s bare-name
    #: reach for `_console_speaking_message_id`, tests) had to change.
    #: See `message.py`'s module docstring for the full map of what
    #: moved, the delegation table its own pre-move test coupling
    #: required, and why.
    screen._message = ConsoleMessageController(
        screen,
        app_instance=screen.app_instance,
        chat_store_accessor=lambda: screen._ensure_console_chat_store(),
        current_chat_store_accessor=lambda: screen._console_chat_store,
        ensure_console_chat_controller=(
            lambda: screen._ensure_console_chat_controller()
        ),
        current_chat_controller_accessor=(
            lambda: screen._console_chat_controller
        ),
        sync_native_console_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        # Session <-> message seam (design spec: "a named callable
        # between them; design it deliberately, never a back-door
        # through the screen"). `self._session` was constructed above;
        # Python resolves it at CALL time inside these lambdas, so
        # construction order does not matter.
        #
        # `active_session_is_ephemeral` is the one exception, routed
        # through the screen's OWN `_console_active_session_is_
        # ephemeral` delegation (session.py's disclosed exception for
        # `console_transcript.py`'s bare-name reach) rather than
        # straight to `self._session`: the pre-existing test suite
        # monkeypatches `screen._console_active_session_is_ephemeral`
        # directly (4 sites, `test_console_native_chat_flow.py`) to
        # stub ephemeral state for `_console_save_as_destinations`
        # scenarios -- reaching `self._session` directly here would
        # silently stop observing that patch.
        active_session_is_ephemeral=(
            lambda: screen._console_active_session_is_ephemeral()
        ),
        active_native_console_session=(
            lambda: screen._session._active_native_console_session()
        ),
        current_console_conversation_id=(
            lambda: screen._session._current_console_conversation_id()
        ),
        active_console_provider_model_display=(
            lambda: screen._active_console_provider_model_display()
        ),
        # Workspace <-> message seam, same shape.
        console_initial_session_title_for_workspace=(
            lambda workspace_id: (
                screen._workspace._console_initial_session_title_for_workspace(
                    workspace_id
                )
            )
        ),
        # The change-review, image-generation, and image-view clusters
        # stay screen-owned this wave (out of scope) -- each reach is a
        # named callable, never a back-door through screen attributes.
        console_change_review_run_id=(
            lambda store, message_id: screen._console_change_review_run_id(
                store, message_id
            )
        ),
        open_change_review=lambda run_id: screen._open_change_review(run_id),
        start_console_transcript_sync_timer=(
            lambda: screen._start_console_transcript_sync_timer()
        ),
        clear_native_console_message_selection=(
            lambda: screen._clear_native_console_message_selection()
        ),
        regenerate_console_generation_variant=(
            lambda message_id: screen._regenerate_console_generation_variant(
                message_id
            )
        ),
        select_console_generation_variant=(
            lambda message, direction: screen._select_console_generation_variant(
                message, direction=direction
            )
        ),
        keep_console_generation_variant=(
            lambda message: screen._keep_console_generation_variant(message)
        ),
        handle_console_toggle_image_view=(
            lambda message_id: screen._handle_console_toggle_image_view(message_id)
        ),
        invalidate_console_persisted_rows_cache=(
            lambda: screen._invalidate_console_persisted_rows_cache()
        ),
    )
    #: The prompt cluster -- Prompt Library modal, `/prompt` + `/system`
    #: resolution and their pickers, the system-prompt editor and its
    #: save-to-Library flow, the Library staged-insert handoff, and the
    #: shared prompt-history store (wave-3 console decomposition, task
    #: 3). Every dependency below is a LATE-BINDING lambda, never a
    #: bound method: eleven of the eighteen are replaced by name on the
    #: screen instance somewhere in the pre-existing suite, and a
    #: constructor snapshot would silently stop observing every one of
    #: those. See `Console_Modules/prompts.py`'s `__init__` docstring
    #: for the per-parameter rationale.
    screen._prompts = ConsolePromptsController(
        screen,
        app_instance=screen.app_instance,
        composer_accessor=lambda: screen._console_composer_or_none(),
        chat_store_accessor=lambda: screen._ensure_console_chat_store(),
        # Session <-> prompts seam (design spec: "a named callable
        # between them; design it deliberately, never a back-door
        # through the screen"). `self._session` was constructed above;
        # Python resolves it at CALL time inside these lambdas, so
        # construction order does not matter.
        ensure_active_console_session_settings=(
            lambda: screen._session._ensure_active_console_session_settings()
        ),
        apply_console_session_system_prompt=(
            lambda system_prompt: (
                screen._session._apply_console_session_system_prompt(system_prompt)
            )
        ),
        sync_console_session_draft=(
            lambda: screen._session._sync_console_session_draft()
        ),
        active_console_provider_model_display=(
            lambda: screen._active_console_provider_model_display()
        ),
        build_console_provider_selection=(
            lambda: screen._build_console_provider_selection()
        ),
        ensure_console_provider_gateway=(
            lambda: screen._ensure_console_provider_gateway()
        ),
        console_provider_blocker_copy=(
            lambda: screen._console_provider_blocker_copy()
        ),
        # A bare-attribute READ, not a call: the modal opener hands
        # this straight to `ConsolePromptsModal(configure_provider=...)`
        # without calling it, so the accessor must return the screen's
        # current attribute rather than a wrapper around it -- that is
        # what keeps `test_console_workbench_contract.py`'s
        # `console._open_console_provider_recovery = AsyncMock()`
        # reaching the modal exactly as it did pre-move.
        open_console_provider_recovery_accessor=(
            lambda: screen._open_console_provider_recovery
        ),
        console_setup_blocked_reason=(
            lambda: screen._console_setup_blocked_reason()
        ),
        focus_console_composer_if_needed=(
            lambda **kwargs: screen._focus_console_composer_if_needed(**kwargs)
        ),
        # DOM-touching, so it stays on the screen (`query_one` on the
        # native composer) -- and six pre-existing test sites replace it
        # there by name.
        insert_prompt_text_into_composer=(
            lambda text, *, replace: screen._insert_prompt_text_into_composer(
                text, replace=replace
            )
        ),
        clear_console_composer_draft=(
            lambda: screen._clear_console_composer_draft()
        ),
        append_native_console_system_message=(
            lambda text: screen._append_native_console_system_message(text)
        ),
        sync_console_chat_core_state=(
            lambda: screen._sync_console_chat_core_state()
        ),
        sync_console_rail_system_line=(
            lambda: screen._sync_console_rail_system_line()
        ),
        sync_console_settings_summary=(
            lambda: screen._sync_console_settings_summary()
        ),
    )
