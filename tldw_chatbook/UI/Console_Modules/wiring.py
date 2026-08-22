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

`Screens/chat_screen.py` no longer imports the controller CLASSES at all --
this module is the only place they are constructed. It used to keep them as
re-export surface because 32 sites across five test files reached them as
`chat_screen_module.ConsoleDictationController` / `...ConsoleWorkspaceController`
patch handles, and deleting the imports turned those tests red during this
very extraction. Task-3023 repointed every one of those sites at the defining
module, which is safe because they patch the CLASS OBJECT (or just read it),
never the screen module's namespace -- so the alias and the owning module
hand back the same object. Patch a controller on the module that defines it;
do not reintroduce a re-export in `chat_screen.py` to patch through.
"""

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from tldw_chatbook.Chat.console_fleet_attention import (
    FLEET_UNSEEN_REVISION_ATTR,
    clear_fleet_unseen_completion,
    fleet_unseen_conversation_ids,
)
from tldw_chatbook.Chat.console_runtime import leave_console_runtime
from tldw_chatbook.Widgets.Console.console_auto_speak_consent import (
    ConsoleAutoSpeakCoordinator,
)

from .agent import ConsoleAgentController
from .character import ConsoleCharacterController
from .dictation import ConsoleDictationController
from .fleet import ConsoleFleetLifecycleController
from .hands_free import ConsoleHandsFreeController
from .image import ConsoleImageController
from .message import ConsoleMessageController
from .prompt_queue import (
    ConsolePromptQueueUIController,
    commit_queued_draft_transaction,
)
from .prompts import ConsolePromptsController
from .reaction_preview import get_console_reaction_preview_coordinator
from .retrieval import ConsoleRetrievalController
from .session import ConsoleSessionController
from .skill import ConsoleSkillController
from .video import ConsoleVideoController
from .workspace import ConsoleWorkspaceController

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..Screens.chat_screen import ChatScreen

__all__ = ["build_console_controllers"]


def _displayed_console_composer_draft(screen: Any) -> str | None:
    """Return the draft from the Console composer receiving user input."""
    try:
        displayed = screen.app.screen
    except Exception:  # noqa: BLE001 -- no reachable app: use own composer
        displayed = None
    composer = None
    if displayed is not None and displayed is not screen:
        resolve = getattr(displayed, "_console_composer_or_none", None)
        if callable(resolve):
            try:
                composer = resolve()
            except Exception:  # noqa: BLE001 -- broken foreign resolver: fall back
                composer = None
    if composer is None:
        composer = screen._console_composer_or_none()
    return composer.draft_text() if composer is not None else None


def _console_screen_is_displayed(screen: Any) -> bool:
    """Return whether ``screen`` is displayed, preserving fixture fallback."""
    try:
        return screen.app.screen is screen
    except Exception:  # noqa: BLE001 -- no reachable app: unmounted fixtures
        return True


def build_console_controllers(
    screen: "ChatScreen",
    *,
    rag_source_types_accessor: Callable[[], tuple[str, ...]],
    rag_top_k_accessor: Callable[[], int],
) -> None:
    """Construct the Console screen's fourteen controllers and attach them.

    Assigns, in this order, `screen._image`, `screen._video`,
    `screen._retrieval`, `screen._skill`, `screen._workspace`,
    `screen._character`, `screen._fleet`, `screen._session`, `screen._dictation`,
    `screen._hands_free`, `screen._message`, `screen._prompts`, `screen._agent`, and
    `screen._prompt_queue`. The order is documentation, not a constraint:
    every cross-controller dependency below is resolved at call time (see the
    module docstring), so no controller reads a sibling that does not exist
    yet.

    `ChatScreen.__init__` calls this at exactly the point the first
    construction used to occupy. That position matters: the ~250 attribute
    assignments around it in `__init__` include names these lambdas read, and
    none of the fourteen constructors reads mutable state off `screen` eagerly
    (each stores its inputs and callables), so the call needs to sit where it
    can see everything the pre-move constructions could.

    Args:
        screen: The Console screen (`ChatScreen`) to wire. Mutated in place;
            taken as a parameter rather than imported so this module has no
            import cycle with `Screens/chat_screen.py`.

    Returns:
        None. The controllers are reachable as attributes of `screen`.
    """
    screen._image = ConsoleImageController(
        screen,
        app_instance=screen.app_instance,
        ensure_console_image_view=lambda: screen._ensure_console_image_view(),
        recent_console_image_messages=(
            lambda messages: screen._recent_console_image_messages(messages)
        ),
        console_image_default_mode=lambda: screen._console_image_default_mode,
        console_generation_browse=lambda: screen._console_generation_browse(),
        sync_native_console_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        ensure_console_chat_store=lambda: screen._ensure_console_chat_store(),
        build_console_provider_selection=(
            lambda: screen._build_console_provider_selection()
        ),
        ensure_console_provider_gateway=(
            lambda: screen._ensure_console_provider_gateway()
        ),
        console_image_preparing=(
            lambda: getattr(screen, "_console_image_preparing", None)
        ),
        current_console_chat_store=lambda: screen._console_chat_store,
        console_composer_or_none=lambda: screen._console_composer_or_none(),
        console_visible_draft_session_id=(
            lambda: screen._console_visible_draft_session_id
        ),
        append_native_console_system_message=(
            lambda *args, **kwargs: screen._append_native_console_system_message(
                *args, **kwargs
            )
        ),
        request_console_control_bar_sync=(
            lambda: screen._request_console_control_bar_sync()
        ),
        default_console_session_settings=(
            lambda: screen._session._default_console_session_settings()
        ),
        clear_console_composer_draft=(lambda: screen._clear_console_composer_draft()),
    )

    screen._video = ConsoleVideoController(
        app_instance=screen.app_instance,
        sync_native_console_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        ensure_console_chat_store=lambda: screen._ensure_console_chat_store(),
        wait_for_console_screen_result=(
            lambda modal: screen._wait_for_console_screen_result(modal)
        ),
        open_video_with_os=lambda path: screen._open_video_with_os(path),
        append_native_console_system_message=(
            lambda *args, **kwargs: screen._append_native_console_system_message(
                *args, **kwargs
            )
        ),
        default_console_session_settings=(
            lambda: screen._session._default_console_session_settings()
        ),
        console_composer_or_none=lambda: screen._console_composer_or_none(),
        clear_console_composer_draft=lambda: screen._clear_console_composer_draft(),
    )

    screen._retrieval = ConsoleRetrievalController(
        app_instance=screen.app_instance,
        active_native_session=(
            lambda: screen._session._active_native_console_session()
        ),
        current_conversation_id=(
            lambda: screen._character._current_console_rail_conversation_id()
        ),
        clear_evidence_sent_notice=(
            lambda: screen._clear_console_evidence_sent_notice()
        ),
        consume_pending_launch=lambda: screen._consume_pending_console_launch(),
        release_consumed_launch=(
            lambda launch, result: screen._release_consumed_console_launch(
                launch, result
            )
        ),
        is_mounted=lambda: screen.is_mounted,
        sync_retrieval_scope_row=(lambda: screen._sync_console_retrieval_scope_row()),
        sync_control_bar=lambda: screen._sync_console_control_bar(),
        request_control_bar_sync=(lambda: screen._request_console_control_bar_sync()),
        dictionary_scope_service=lambda: screen._dictionary_scope_service(),
        set_library_rag_source_scope=(
            lambda source_types: screen._set_console_library_rag_source_scope(
                source_types
            )
        ),
        set_library_rag_query=(
            lambda query: screen._set_console_library_rag_query(query)
        ),
        run_library_rag_action=(
            lambda: screen._run_console_library_rag_from_visible_action()
        ),
        library_rag_source_scope=rag_source_types_accessor,
        library_rag_top_k=rag_top_k_accessor,
        pending_launch=lambda: screen._pending_console_launch_context,
        set_pending_launch=(
            lambda launch: setattr(screen, "_pending_console_launch_context", launch)
        ),
        set_pending_auto_open=(
            lambda value: setattr(
                screen, "_pending_console_launch_auto_open_inspector", value
            )
        ),
        set_evidence_sent_notice=(
            lambda value: setattr(screen, "_console_evidence_sent_notice", value)
        ),
        sync_pending_launch_surfaces=(
            lambda: screen._sync_console_pending_launch_surfaces()
        ),
        refresh_screen=lambda: screen.refresh(recompose=True),
        has_staged_evidence=lambda: screen._has_staged_console_evidence(),
    )

    screen._skill = ConsoleSkillController(
        app_instance=screen.app_instance,
        append_native_console_system_message=(
            lambda message: screen._append_native_console_system_message(message)
        ),
        sync_console_command_popup=lambda: screen._sync_console_command_popup(),
        task_resume_state=lambda: screen._task_resume_state,
        set_task_resume_state=lambda state: screen.set_task_resume_state(state),
        current_chat_controller=lambda: screen._console_chat_controller,
    )

    #: Workspace policy, lifecycle, resume, scope, and conversation-browser
    #: behavior have one owner. The controller keeps canonical rich browser
    #: state and projects legacy Workspace rows through compatibility aliases.
    #: Its browser inputs are explicit late-bound callables; the screen keeps
    #: only the bounded plain-value input delegate and DOM edges.
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
            lambda: screen._retrieval._console_scope_picker_listers()
        ),
        active_native_session_accessor=(
            lambda: screen._session._active_native_console_session()
        ),
        refresh_effective_scope_and_sync=(
            lambda session: screen._retrieval._refresh_console_effective_scope_and_sync(
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
            lambda tree: screen._message._console_messages_from_conversation_tree(tree)
        ),
        session_settings_for_resume_accessor=(
            lambda conversation: screen._session._console_session_settings_for_resume(
                conversation
            )
        ),
        resolve_resumed_character_name=(
            lambda character_id: screen._resolve_resumed_character_name(character_id)
        ),
        # Agent <-> workspace seam, same shape as the message seam above:
        # the resume flow's TOOL-marker re-derivation moved to
        # `ConsoleAgentController` (wave-4 task 3). This accessor already
        # existed as a named callable (wave-2 task 2); only its target
        # changed, from the screen's own method to the controller.
        inject_resume_agent_markers_accessor=(
            lambda messages, conversation_id: (
                screen._agent._inject_resume_agent_markers(messages, conversation_id)
            )
        ),
        resolve_effective_scope_state=(
            lambda session: screen._retrieval._resolve_console_effective_scope_state(
                session
            )
        ),
        sync_retrieval_scope_row=(lambda: screen._sync_console_retrieval_scope_row()),
        note_follow_intent=lambda: screen._note_console_follow_intent(),
        focus_composer_if_needed=(
            lambda **kwargs: screen._focus_console_composer_if_needed(**kwargs)
        ),
        conversation_section_config_accessor=(
            lambda: screen._console_conversation_section_config()
        ),
        # Supply the screen-owned preference tree late-bound; Workspace owns
        # the grouped-browser collapse transition that mutates it.
        conversation_browser_config=(
            lambda: screen._console_conversation_browser_config()
        ),
        focus_conversation_search=(
            lambda: screen._focus_console_workspace_conversation_search()
        ),
        schedule_timer=lambda delay, callback: screen.set_timer(delay, callback),
        screen_running_accessor=lambda: screen.is_running,
        current_chat_controller_accessor=lambda: screen._console_chat_controller,
        fleet_unseen_ids_accessor=lambda: screen._fleet._console_fleet_unseen_ids(),
        run_marker_with_unseen=(
            lambda controller, session, unseen_ids: (
                screen._fleet._console_run_marker_with_unseen(
                    session,
                    unseen_ids,
                )
            )
        ),
        broken_conversation_ids_accessor=(
            lambda: getattr(screen, "_console_broken_conversation_ids", set())
        ),
        ensure_agent_bridge=lambda: screen._ensure_console_agent_bridge(),
        subagent_counts_for_rows=(
            lambda bridge, rows: screen._agent._console_subagent_counts_for_rows(
                bridge,
                rows,
            )
        ),
        conversation_browser_collapse_preferences=(
            lambda: screen._console_conversation_browser_collapse_preferences()
        ),
        # task-15864 AC#2: session-open (the resume flow) is a wake retry
        # trigger -- late-binding like every sibling above.
        wake_retry_poke=lambda: screen._fleet._poke_console_wake_retry(),
        sync_workspace_context=lambda: screen._sync_console_workspace_context(),
    )
    screen._character = ConsoleCharacterController(
        app_config_accessor=(
            lambda: getattr(screen.app_instance, "app_config", {}) or {}
        ),
        chat_store_accessor=(
            lambda: getattr(
                getattr(screen, "_console_chat_controller", None), "store", None
            )
        ),
        active_native_session_accessor=(
            lambda: screen._session._active_native_console_session()
        ),
        current_conversation_id_accessor=(
            lambda: screen._session._current_console_conversation_id()
        ),
        character_db_accessor=(
            lambda: getattr(screen.app_instance, "chachanotes_db", None)
        ),
        ensure_chat_store=lambda: screen._ensure_console_chat_store(),
        provider_readiness_config_accessor=(
            lambda: screen._provider_readiness_app_config()
        ),
        default_session_settings=(
            lambda: screen._session._default_console_session_settings()
        ),
        swap_session_character=(
            lambda store, character_id, seed, *, global_default: (
                screen._session._swap_console_session_character(
                    store,
                    character_id,
                    seed,
                    global_default=global_default,
                )
            )
        ),
        sync_temporary_chip=lambda: screen._sync_console_temporary_chip(),
        sync_native_chat_ui=lambda: screen._sync_native_console_chat_ui(),
        notify=(lambda message, **kwargs: screen.app.notify(message, **kwargs)),
        actor_scope_accessor=(
            lambda: screen._session._current_visual_identity_actor_scope()
        ),
        manual_reaction_key=lambda scope: screen._session._manual_reaction_key(scope),
        resolve_visual_identity=(
            lambda scope, state, manual: screen._session._resolve_visual_identity(
                scope, state, manual
            )
        ),
        ensure_console_image_view=lambda: screen._ensure_console_image_view(),
        console_image_default_mode=lambda: screen._console_image_default_mode,
        is_mounted=lambda: screen.is_mounted,
        render_character_avatar=(
            lambda **kwargs: screen._render_character_avatar_into_section(**kwargs)
        ),
    )

    screen._fleet = ConsoleFleetLifecycleController(
        pending_handoffs_accessor=lambda: screen.app_instance.pending_handoffs,
        ensure_chat_store=lambda: screen._ensure_console_chat_store(),
        ensure_chat_controller=lambda: screen._ensure_console_chat_controller(),
        activate_workspace_for_session=(
            lambda session_id: (
                screen._workspace._set_active_workspace_for_console_session(session_id)
            )
        ),
        switch_chat_session=(
            lambda session_id: screen._console_chat_controller.switch_session(
                session_id
            )
        ),
        schedule_native_console_sync=(
            lambda: screen.run_worker(
                screen._sync_native_console_chat_ui(),
                exclusive=True,
                group="console-sync",
            )
        ),
        ensure_agent_bridge=lambda: screen._ensure_console_agent_bridge(),
        wire_wake_coordinator=(
            lambda: (
                (wake.wire(app=screen.app_instance), True)[1]
                if (
                    wake := getattr(
                        screen._console_chat_controller
                        or screen._ensure_console_chat_controller(),
                        "fleet_wake",
                        None,
                    )
                )
                is not None
                else False
            )
        ),
        seed_wake_from_marks=(
            lambda: bool(
                wake.seed_from_marks()
                if (
                    wake := getattr(
                        screen._console_chat_controller,
                        "fleet_wake",
                        None,
                    )
                )
                is not None
                else False
            )
        ),
        retry_wake_soon=(
            lambda: (
                retry()
                if callable(
                    retry := getattr(
                        getattr(
                            screen._console_chat_controller,
                            "fleet_wake",
                            None,
                        ),
                        "retry_soon",
                        None,
                    )
                )
                else None
            )
        ),
        wake_has_pending=(
            lambda conversation_id: bool(
                has_pending(conversation_id)
                if callable(
                    has_pending := getattr(
                        getattr(
                            screen._console_chat_controller,
                            "fleet_wake",
                            None,
                        ),
                        "has_pending",
                        None,
                    )
                )
                else False
            )
        ),
        wake_delivering_conversation_id=(
            lambda: (
                delivering()
                if callable(
                    delivering := getattr(
                        getattr(
                            screen._console_chat_controller,
                            "fleet_wake",
                            None,
                        ),
                        "delivering_conversation_id",
                        None,
                    )
                )
                else None
            )
        ),
        displayed_composer_draft_accessor=(
            lambda: _displayed_console_composer_draft(screen)
        ),
        screen_displayed_accessor=lambda: _console_screen_is_displayed(screen),
        screen_mounted_accessor=lambda: screen.is_mounted,
        active_session_id_accessor=(
            lambda: getattr(screen._console_chat_store, "active_session_id", None)
        ),
        chat_sessions_accessor=(
            lambda: (
                tuple(screen._console_chat_store.sessions())
                if screen._console_chat_store is not None
                else ()
            )
        ),
        defer_on_message_pump=lambda callback: screen.call_later(callback),
        start_transcript_sync_timer=(
            lambda: screen._start_console_transcript_sync_timer()
        ),
        transcript_sync_timer_active=(
            lambda: screen._console_transcript_sync_timer is not None
        ),
        sync_native_console_ui=lambda: screen._sync_native_console_chat_ui(),
        create_interval=(
            lambda seconds, callback: screen.set_interval(seconds, callback)
        ),
        record_timer_created=lambda name: screen._record_ui_timer_created(name),
        record_timer_stopped=lambda name: screen._record_ui_timer_stopped(name),
        chat_controller_available=(lambda: screen._console_chat_controller is not None),
        fleet_has_unsettled_children=(
            lambda: (
                bool(checker())
                if callable(
                    checker := getattr(
                        screen._console_chat_controller,
                        "fleet_has_unsettled_children",
                        None,
                    )
                )
                else False
            )
        ),
        run_marker_for_session=(
            lambda session_id: screen._console_chat_controller.run_marker_for(
                session_id
            )
        ),
        fleet_teardown_split=(
            lambda: screen._console_chat_controller.fleet_teardown_split()
        ),
        leave_runtime=(lambda: leave_console_runtime(screen.app_instance, view=screen)),
        stage_teardown_notices=(
            lambda killed, surviving: (
                setattr(
                    screen.app_instance,
                    "_console_fleet_teardown_notice",
                    killed,
                )
                if killed
                else None,
                setattr(
                    screen.app_instance,
                    "_console_fleet_survivor_notice",
                    surviving,
                )
                if surviving
                else None,
            )
        ),
        fleet_unseen_revision_accessor=(
            lambda: getattr(screen.app_instance, FLEET_UNSEEN_REVISION_ATTR, 0)
        ),
        read_fleet_unseen_ids=(
            lambda: fleet_unseen_conversation_ids(screen.app_instance)
        ),
        clear_fleet_unseen=(
            lambda conversation_id: clear_fleet_unseen_completion(
                screen.app_instance,
                conversation_id,
            )
        ),
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
        provider_readiness_app_config=(lambda: screen._provider_readiness_app_config()),
        build_provider_selection=(
            lambda session_id: screen._build_console_provider_selection(session_id)
        ),
        rag_source_types_accessor=rag_source_types_accessor,
        rag_top_k_accessor=rag_top_k_accessor,
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
            lambda: screen._workspace._invalidate_console_persisted_rows_cache()
        ),
        mark_conversation_row_broken=(
            lambda conversation_id: screen._mark_console_conversation_row_broken(
                conversation_id
            )
        ),
        refresh_effective_scope_and_sync=(
            lambda session: screen._retrieval._refresh_console_effective_scope_and_sync(
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
                screen._workspace._set_active_workspace_for_console_session(session_id)
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
        # The inline-image cluster stays screen-owned (out of scope this
        # wave); `_close_console_session_tab` (wave-4 task 2) drops a
        # closing tab's cached renders through this named callable. It is
        # the screen's EXISTING lazy accessor, deliberately not a new
        # one-line screen method wrapping it: `chat_screen.py` is under a
        # method-count ratchet (Tests/Architecture/test_screen_size_
        # ratchet.py) that a convenience wrapper would push past, and the
        # controller's body is the pre-move closure unchanged either way.
        ensure_console_image_view=lambda: screen._ensure_console_image_view(),
        visual_identity_db_accessor=(
            lambda: getattr(screen.app_instance, "chachanotes_db", None)
        ),
        reaction_preview_coordinator_accessor=(
            lambda: get_console_reaction_preview_coordinator(screen.app_instance)
        ),
        refresh_character_avatar=(
            lambda **kwargs: (
                screen._character._refresh_active_character_avatar_if_scope_changed(
                    force=True, **kwargs
                )
            )
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
            lambda **kwargs: screen._hands_free._enter_console_hands_free_loop(**kwargs)
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
            lambda transcript: screen._console_realtime_adopt_transcript(transcript)
        ),
        # Same screen-owned realtime engine, read as a live session rather
        # than called: `_handle_console_dictation_button` (wave-4 task 2)
        # must see a loop that started AFTER this wiring ran, so this is a
        # late-binding accessor, never a snapshot. `ConsoleHandsFree
        # Controller` takes an identically-named, identically-shaped one.
        realtime_session_accessor=lambda: screen._console_realtime,
        run_pending_voice_action=(
            lambda session_id: screen._run_pending_console_voice_action(session_id)
        ),
        undo_histories_accessor=lambda: screen._console_undo_histories,
        visible_draft_session_id_accessor=(
            lambda: screen._console_visible_draft_session_id
        ),
        dictation_service_factory=lambda **kwargs: (
            screen.app_instance._create_console_dictation_service(**kwargs)
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
            lambda session_id: screen._run_pending_console_voice_action(session_id)
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
        current_chat_controller_accessor=(lambda: screen._console_chat_controller),
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
        # Change-review stays screen-owned; image-generation and image-view
        # are controller-owned. Each reach is a named callable, never a
        # back-door through screen attributes.
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
            lambda message_id: screen._image._regenerate_console_generation_variant(
                message_id
            )
        ),
        select_console_generation_variant=(
            lambda message, direction: screen._image._select_console_generation_variant(
                message, direction=direction
            )
        ),
        keep_console_generation_variant=(
            lambda message: screen._image._keep_console_generation_variant(message)
        ),
        handle_console_toggle_image_view=(
            lambda message_id: screen._image._handle_console_toggle_image_view(
                message_id
            )
        ),
        invalidate_console_persisted_rows_cache=(
            lambda: screen._workspace._invalidate_console_persisted_rows_cache()
        ),
        play_console_video=(
            lambda message_id: screen._video._play_console_video(message_id)
        ),
        save_console_video_copy=(
            lambda message_id: screen._video._save_console_video_copy(message_id)
        ),
        regenerate_console_video_message=(
            lambda message_id: screen._video._regenerate_console_video_message(
                message_id
            )
        ),
    )
    screen._console_auto_speak = ConsoleAutoSpeakCoordinator(
        store_accessor=lambda: screen._ensure_console_chat_store(),
        resolve_destination=(
            lambda assistant_kind, character_ref: (
                screen._resolve_console_auto_speak_destination(
                    assistant_kind,
                    character_ref,
                )
            )
        ),
        issue_message_speech=(
            lambda message_id, outcome_callback, expected_destination, retry_failed_auto: (
                screen._message.request_console_message_speech(
                    message_id,
                    outcome_callback,
                    expected_destination,
                    retry_failed_auto,
                )
            )
        ),
        open_consent=(
            lambda modal, callback: screen.app.push_screen(modal, callback=callback)
        ),
        hands_free_active=(
            lambda: (
                screen._console_hands_free is not None
                or screen._console_realtime is not None
            )
        ),
        sync_controls=lambda enabled, paused, retry_available: (
            screen._sync_console_auto_speak_controls(
                enabled,
                paused,
                retry_available,
            )
        ),
        notify=lambda copy, severity: screen.app_instance.notify(
            copy,
            severity=severity,
        ),
        schedule=lambda coroutine: screen.run_worker(
            coroutine,
            exclusive=False,
            group="console-auto-speak",
        ),
    )
    #: The prompt cluster -- Prompt Library modal, `/prompt` + `/system`
    #: resolution and their pickers, the system-prompt editor and its
    #: save-to-Library flow, the Library staged-insert handoff, and the
    #: shared prompt-history store (wave-3 console decomposition, task
    #: 3). Every dependency below is a LATE-BINDING lambda (or, for the
    #: post-apply re-sync trio, a nested function that reads the same
    #: way), never a bound method: twelve of the seventeen are replaced by
    #: name on the screen instance somewhere in the pre-existing suite,
    #: and a constructor snapshot would silently stop observing every one
    #: of those. See `Console_Modules/prompts.py`'s `__init__` docstring
    #: for the per-parameter rationale.

    def _sync_console_system_prompt_surfaces() -> None:
        """Re-sync the three surfaces that display the System prompt.

        One dependency rather than three because the prompt cluster only
        ever needs the trio, in this order, at the two moments the store
        accepted a new System prompt (task 2766). Each name is still
        resolved on the screen at CALL time, so the instance-level
        replacements the suite makes are observed exactly as before.
        """
        screen._sync_console_chat_core_state()
        screen._sync_console_rail_system_line()
        screen._sync_console_settings_summary()

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
            lambda system_prompt: screen._session._apply_console_session_system_prompt(
                system_prompt
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
        console_provider_blocker_copy=(lambda: screen._console_provider_blocker_copy()),
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
        console_setup_blocked_reason=(lambda: screen._console_setup_blocked_reason()),
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
        clear_console_composer_draft=(lambda: screen._clear_console_composer_draft()),
        append_native_console_system_message=(
            lambda text: screen._append_native_console_system_message(text)
        ),
        sync_console_system_prompt_surfaces=_sync_console_system_prompt_surfaces,
        sync_console_command_popup=lambda: screen._sync_console_command_popup(),
    )
    #: The agent runtime's screen-side cluster -- the lazily-built
    #: `ConsoleAgentBridge`, the Agent rail section's text derivation, the
    #: sub-agent drill-in cycle, the "View full log" target/probe/loader,
    #: the fleet auto-open override, the `[N Sub-Agents]` badge-count
    #: cache, and resume-time TOOL-marker re-derivation -- moved to
    #: `ConsoleAgentController` (wave-4 console decomposition, task 3).
    #: `self._console_agent_bridge`/`_console_agent_drilldown_run_id`/
    #: `_agent_section_user_dismissed_while_busy` stay readable/writable via
    #: THREE read-write proxy properties defined near
    #: `_console_composer_or_none`, so nothing outside this cluster (compose,
    #: `_toggle_console_rail_section`, the two sibling controllers' own
    #: drill-down clears, `on_button_pressed`, tests) had to change.
    #: `_console_agent_drilldown_conversation_id` is deliberately NOT among
    #: them: it had no reader outside the cluster, so it lives only on the
    #: controller and `screen._console_agent_drilldown_conversation_id`
    #: raises `AttributeError`. Reach it as `screen._agent.` + the name. `_sync_console_agent_section` stays on the
    #: screen (nine `query_one` calls); only its payload derivation moved.
    #: See `agent.py`'s module docstring for the full map of what moved and
    #: why, including the one method that name-matched but is not part of
    #: this cluster.
    screen._agent = ConsoleAgentController(
        screen,
        app_instance=screen.app_instance,
        chat_store_accessor=lambda: screen._ensure_console_chat_store(),
        provider_gateway_accessor=(lambda: screen._ensure_console_provider_gateway()),
        # A bare-attribute READ, not a call: `ConsoleAgentBridge` stores
        # this callable and calls it per run, so the accessor must return
        # the screen's method rather than today's answer -- the same shape
        # `ConsolePromptsController`'s `open_console_provider_recovery_
        # accessor` documents.
        native_tool_calls_enabled_accessor=(
            lambda: screen._console_native_tool_calls_enabled
        ),
        # These targets are replaced by name in
        # `Tests/UI/test_console_agent_rail.py`: the conversation accessor
        # on `screen._character` and the rail-state accessor on `screen`.
        # Bound methods captured here would stop observing those patches.
        current_rail_conversation_id=(
            lambda: screen._character._current_console_rail_conversation_id()
        ),
        current_rail_state_accessor=lambda: screen._current_console_rail_state(),
        # `getattr` with a default, matching the pre-move body: the fleet
        # summary line is reachable on a screen that has never built a chat
        # controller.
        chat_controller_accessor=(
            lambda: getattr(screen, "_console_chat_controller", None)
        ),
        # Another bare-attribute READ: the drill-in toggle hands this
        # straight to `run_worker` without calling it.
        sync_native_console_chat_ui_accessor=(
            lambda: screen._sync_native_console_chat_ui
        ),
    )
    screen._prompt_queue = ConsolePromptQueueUIController(
        chat_controller_accessor=(lambda: screen._ensure_console_chat_controller()),
        ensure_active_session=(
            lambda: screen._session._ensure_active_console_session_settings()
        ),
        blocked_reason_accessor=lambda: screen._console_send_blocked_reason(),
        setup_blocked_reason_accessor=(lambda: screen._console_setup_blocked_reason()),
        restore_stash=lambda stash: screen._restore_console_send_stash(stash),
        append_system_message=(
            lambda text: screen._append_native_console_system_message(text)
        ),
        notify=(
            lambda text, severity: screen.app_instance.notify(text, severity=severity)
        ),
        focus_composer=(lambda: screen._focus_console_composer_if_needed(force=True)),
        inflight_stashes_accessor=(lambda: screen._console_inflight_send_stashes),
        note_follow_intent=lambda: screen._note_console_follow_intent(),
        launch_chain=(
            lambda draft, session_id: screen.run_worker(
                screen._submit_console_native_draft(draft, session_id),
                exclusive=True,
                group=f"console-run-{session_id}",
            )
        ),
        commit_queued_draft=(
            lambda session_id, stash: commit_queued_draft_transaction(
                session_id,
                stash,
                composer=screen._console_composer_or_none(),
                visible_session_id=screen._console_visible_draft_session_id,
                undo_histories=screen._console_undo_histories,
                store=screen._ensure_console_chat_store(),
                sync_command_popup=screen._sync_console_command_popup,
            )
        ),
        edit_refusal=(
            lambda text: (
                "Slash commands cannot be queued."
                if screen._console_command_registry.parse(text).kind
                in {"command", "fallback"}
                else ""
            )
        ),
        sync_ui=lambda: screen._sync_native_console_chat_ui(),
    )
