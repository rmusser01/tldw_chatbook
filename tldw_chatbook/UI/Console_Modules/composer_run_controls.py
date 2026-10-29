"""Composer run controls and composer-bar keyboard ownership.

TASK-33625.1 and TASK-33622.2 (Console UX review 2026-09-29, G1-01/G1-07)
share one cause, so their fix shares this module:

* ``ChatScreen.on_key`` treated every composer descendant as the draft, so
  Enter on a keyboard-focused Menu, Dictate or Stop button went down the send
  path (a paid request) and Space typed into a draft that did not have focus.
  `route_composer_control_key` hands those keys back to the focused control,
  and moves focus to the draft before any other printable key is typed.
* Stop had no route but its button, and the action row clipped that button
  out entirely while a run was active. The stop key, the ``/stop`` command and
  the palette entries below give the viewed tab's run keyboard routes that do
  not depend on the row's geometry.

TASK-33620.15.17 adds the arrival case of the same key routing: keys typed
right after the Console opens, before its focus restore lands, reached the
nav bar's Home button and were dropped (`arm_arrival_focus`,
`claim_arrival_key`).

`ChatScreen` keeps only the wiring Textual resolves by name (the
``stop_console_run`` binding action, `check_action`, `on_key`); the policy
lives here.
"""

from __future__ import annotations

import sys
from functools import partial
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.css.query import QueryError
from textual.dom import DOMNode
from textual.widgets import Button

from ...Widgets.Console import console_composer_bar as composer_module
from ...Widgets.Console.console_composer_bar import ConsoleComposerBar
from ...Widgets.Console.console_composer_menu_modal import (
    ACTION_IMPROVE_CURRENT_DRAFT,
    build_composer_menu_entries,
)
from ...Widgets.Console.console_transcript import ConsoleTranscript

if TYPE_CHECKING:
    from textual.events import Key

    from ...Widgets.Console.console_composer_bar import ConsoleDraftStash

#: The stop key. Ctrl+G is the conventional "cancel" chord (Emacs'
#: keyboard-quit), arrives in every terminal as the C0 byte BEL without any
#: keyboard protocol, and nothing on the Console screen binds it. Ctrl+digit
#: and Alt+digit chords were rejected: many terminals never deliver them.
STOP_RUN_KEY = "ctrl+g"
STOP_RUN_KEY_LABEL = "Ctrl+G"
#: Footer hint, advertised only while the viewed tab's run is stoppable.
STOP_RUN_FOOTER_HINT = (STOP_RUN_KEY_LABEL, "stop run")

#: Chords that keep their composer-wide routing while a composer button holds
#: focus: transcript paging and draft undo/redo are deliberate chords, never
#: accidental typing (TASK-33622.2 keeps them "routed as intended").
_CONTROL_PASSTHROUGH_KEYS = frozenset(
    {"pageup", "pagedown", "ctrl+z", "ctrl+shift+z", "ctrl+shift+Z", "ctrl+y"}
)

#: Why a menu action that the Composer menu would not even list is refused.
_MENU_ACTION_ABSENT_COPY = {
    ACTION_IMPROVE_CURRENT_DRAFT: (
        "Write a draft first — Improve works on the unsent message."
    ),
}



def sync_dictation_presentation(screen: Any, composer: ConsoleComposerBar) -> None:
    """Keep a cold stock idle projection off first paint; actions remain live."""
    state = screen._console_dictation_state
    if (
        state == "idle"
        and type(composer) is ConsoleComposerBar
        and composer._dictation_state == "idle"
        and not getattr(screen.app, "_ui_ready", True)
        and "tldw_chatbook.Chat.console_voice_input" not in sys.modules
    ):
        callbacks = [
            (
                getattr(getattr(composer, name), "__func__", None),
                function,
                code,
                defaults,
                kwdefaults,
                values,
            )
            for name, function, code, defaults, kwdefaults, values in composer_module._IDLE_VOICE_CALLBACKS
        ]
        callbacks.append(
            (
                composer_module._load_console_voice_states,
                *composer_module._IDLE_VOICE_LOADER,
            )
        )
        if all(
            current is function
            and function.__code__ is code
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and len(function.__kwdefaults__ or {}) == len(values)
            and all(
                (function.__kwdefaults__ or {}).get(key) is value
                for key, value in values
            )
            for current, function, code, defaults, kwdefaults, values in callbacks
        ):
            # TASK22504's existing availability callback waits for actual ready
            # and replays through set_dictation_availability. No new timer.
            return
    composer.sync_dictation_state(state)


def _composer(screen: Any) -> ConsoleComposerBar | None:
    """Return the mounted Console composer, or None (class/stub safe).

    Args:
        screen: The Console screen, the ChatScreen class, or an inert stub.

    Returns:
        The mounted composer, or None when ``screen`` is not a mounted DOM
        node or holds no composer.
    """

    if not isinstance(screen, DOMNode):
        return None
    try:
        return screen.query_one("#console-native-composer", ConsoleComposerBar)
    except QueryError:
        return None


def route_composer_control_key(
    screen: Any, composer: ConsoleComposerBar, event: Key
) -> bool:
    """Let a keyboard-focused composer-bar control own its key.

    The draft surface is the composer itself; its Buttons are descendants.
    Only the draft is a text target. While a Button has focus, or nothing has:

    * Enter and Space fall through unconsumed to the Button's own bindings
      (`ComposerControlButton` binds Space, the web/ARIA convention): they
      press it and never type.
    * Any other printable key is type-to-compose (umbrella TASK-33622 AC#2):
      the draft takes focus first, then the screen types the key there, so
      it lands where the caret now shows it -- never in a caretless draft.
    * Every other key -- Tab, Escape, editing keys, bindings -- keeps its
      normal route and leaves the draft untouched; the draft-wide chords in
      `_CONTROL_PASSTHROUGH_KEYS` still reach the draft.

    Args:
        screen: The Console screen routing the key.
        composer: The mounted composer.
        event: The key being routed.

    Returns:
        True when the screen must not treat this key as draft input; False
        when the draft (focused now, or a passthrough chord) should handle it.
    """

    focused = screen.app.focused
    if focused is composer:
        return False
    if event.key in _CONTROL_PASSTHROUGH_KEYS:
        return False
    if event.key == "space" and isinstance(focused, Button):
        return True
    # With nothing focused, the same rule as a Button: a printable key takes
    # the draft's focus first, and an editing key alone (Backspace, Ctrl+W)
    # never edits a draft that shows no caret (PR #2934 round-2 review).
    if ConsoleComposerBar.is_text_entry_key(event):
        composer.focus_draft_from(focused)
        if composer.draft_has_focus:
            return False
        # The draft could not take focus: drop the key rather than edit a
        # draft that shows no caret.
        event.stop()
        event.prevent_default()
    return True


#: TASK-33622.4: the transcript is the Console's announced list mode (the
#: ADR-031 refinement of 2026-10-27): its single letters act on messages, so
#: while it holds focus the footer leads with that and with the way back.
TRANSCRIPT_KEYS_FOOTER = (
    ("", "Transcript keys · Esc to type"),
    ("j/k", "select"),
    ("Y", "trace"),
)
#: Hints that name something the key does not do while the transcript has
#: focus (there Enter shows the selected message's actions).
_NOT_TRANSCRIPT_HINTS = (("Enter", "send / queue"), ("Y", "trace"))


def focus_in_transcript(focused: Any) -> bool:
    """Whether ``focused`` is the transcript or one of its rows' controls.

    Args:
        focused: The focused widget, or None.

    Returns:
        True inside the transcript (its list mode), False anywhere else.
    """

    return focused is not None and any(
        isinstance(node, ConsoleTranscript)
        for node in getattr(focused, "ancestors_with_self", ())
    )


def transcript_focus_shortcuts(
    shortcuts: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Return the footer hints for focus inside the transcript (TASK-33622.4).

    Args:
        shortcuts: The footer's ``(key label, description)`` hints, in order.

    Returns:
        The transcript's announcement and keys first (the footer drops hints
        from the end as width runs out), then ``shortcuts`` without the hints
        that do not hold there.
    """

    rest = tuple(pair for pair in shortcuts if pair not in _NOT_TRANSCRIPT_HINTS)
    return (*TRANSCRIPT_KEYS_FOOTER, *rest)


def with_transcript_keys(
    screen: Any, shortcuts: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """Apply `transcript_focus_shortcuts` while the transcript holds focus.

    Records which variant the footer got, so `sync_transcript_keys_footer`
    re-registers the hints only when focus crosses the transcript's edge.

    Args:
        screen: The Console screen whose footer hints are being built.
        shortcuts: The footer's hints so far.

    Returns:
        The transcript variant inside the transcript, else ``shortcuts``.
    """

    inside = focus_in_transcript(screen.focused)
    screen._console_transcript_keys_footer = inside
    return transcript_focus_shortcuts(shortcuts) if inside else shortcuts


def sync_transcript_keys_footer(screen: Any, focused: Any) -> None:
    """Re-register the footer hints when focus enters or leaves the transcript.

    Args:
        screen: The Console screen.
        focused: The widget focus moved to, or None.
    """

    if focus_in_transcript(focused) != getattr(
        screen, "_console_transcript_keys_footer", False
    ):
        screen._register_console_footer_shortcuts()


def hand_paste_to_draft(screen: Any, composer: ConsoleComposerBar) -> None:
    """Focus the draft before a captured paste lands in it.

    The screen captures a paste while a composer button holds focus, or
    while nothing does (`_should_capture_console_input`). A paste is a
    deliberate gesture aimed at the draft, but it must not edit a draft that
    shows no caret (TASK-33622.2 AC#3/#4), so the draft takes focus first --
    in both cases -- and the paste lands where the caret shows it.

    Args:
        screen: The Console screen routing the paste.
        composer: The mounted composer the paste was captured for.
    """

    focused = screen.app.focused
    if focused is not composer:
        composer.focus_draft_from(focused)


#: The panes `ChatScreen._restore_console_workbench_focus` tries, after the
#: pane the user last focused, when focus sits outside every Console pane.
_RESTORE_FALLBACK_PANES = ("console-native-composer", "console-transcript-surface")


def workbench_restore_target(screen: Any) -> str | None:
    """Return the pane the focus restore picks for focus outside the panes.

    One source for the restore and for `claim_arrival_key`, which must make
    the choice the restore is about to make.

    Args:
        screen: The Console screen.

    Returns:
        The first displayed of the last-focused pane, the composer and the
        transcript, or None when none of them is displayed.
    """

    for widget_id in (
        screen._last_console_workbench_focus_id,
        *_RESTORE_FALLBACK_PANES,
    ):
        if widget_id and screen._is_console_widget_displayed(widget_id):
            return widget_id
    return None


def arm_arrival_focus(screen: Any) -> None:
    """Remember where an activation found focus, until focus first moves.

    TASK-33620.15.17: when the Console is pushed, Textual focuses the first
    focusable widget (the nav bar's Home button), and the activation's
    restore moves focus to the composer only after the first attach sync --
    live, 0.1-1.9 s after the first frame. Printable keys typed in between
    reached Home and were dropped. `claim_arrival_key` lets them land where
    that restore is about to put focus.

    Called from ChatScreen's own ``ScreenResume`` handler, so only a real
    activation arms -- never the attach reconcile's direct replay of
    ``on_screen_resume`` (by then any focus change is the user's). A focus
    already inside a Console pane is kept by the restore, so there is
    nothing to claim.

    Args:
        screen: The Console screen being activated.
    """

    focused = screen.focused
    if (
        focused is not None
        and screen._console_workbench_focus_id_for_widget(focused) is not None
    ):
        focused = None
    screen._console_arrival_focus = focused
    if focused is None or getattr(screen, "_console_arrival_watched", False):
        return
    screen._console_arrival_watched = True
    screen.watch(screen, "focused", partial(_end_arrival_focus, screen), init=False)


def _end_arrival_focus(screen: Any, focused: Any) -> None:
    """Close the arrival window once focus lands anywhere else."""
    if focused is not screen._console_arrival_focus:
        screen._console_arrival_focus = None


def claim_arrival_key(screen: Any, composer: ConsoleComposerBar, event: Key) -> None:
    """Focus the draft for a key typed before the activation restore lands.

    TASK-33620.15.17. Runs in `ChatScreen.on_key` before the capture check.
    When a printable key arrives while focus still sits where the activation
    found it, and the activation's restore would pick the composer, the draft
    takes focus now (synchronously), so this key and the ones queued behind
    it are typed there, in order. Everything else keeps its input: a focus
    the user moved, an explicit Resume (it owns its own focus), a collapsed
    composer, and the setup card -- also before its first sync has shown it.
    (A modal above the Console receives the keys itself.)

    Args:
        screen: The Console screen routing the key.
        composer: The mounted composer.
        event: The key being routed.
    """

    arrival = getattr(screen, "_console_arrival_focus", None)
    if arrival is None or screen.focused is not arrival:
        return
    if not ConsoleComposerBar.is_text_entry_key(event):
        return
    if screen._resume_navigation_startup_in_progress or composer.collapsed:
        return
    if workbench_restore_target(screen) != "console-native-composer":
        return
    try:
        blocked = screen._build_console_setup_card_state().mode == "card"
    except Exception as error:  # noqa: BLE001 - a key must not crash the app
        logger.debug(
            "Console arrival key left unclaimed: setup state unreadable "
            "(exception_type={})",
            type(error).__name__,
        )
        return
    if not blocked:
        composer.focus_draft_from(arrival)


def stop_available(screen: Any) -> bool:
    """Whether the stop key applies now -- exactly when Stop is shown.

    Args:
        screen: The Console screen (or a class/stub: then False).

    Returns:
        True while the viewed tab's run is active and the setup card does
        not block the Console; False otherwise.
    """

    composer = _composer(screen)
    if composer is None or not composer.run_active:
        return False
    return not screen._console_setup_modal_blocking()


def with_stop_shortcut(
    screen: Any, shortcuts: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """Prepend the stop-key hint while a run is stoppable.

    Prepended, not appended: the footer degrades by dropping hints from the
    END as width runs out, and the stop key is the one hint a running tab
    must never lose.

    Args:
        screen: The Console screen whose footer hints are being built.
        shortcuts: The footer's ``(key label, description)`` hints, in the
            order the footer shows them.

    Returns:
        ``shortcuts`` unchanged when `stop_available` is False; otherwise a
        new tuple with `STOP_RUN_FOOTER_HINT` first and ``shortcuts`` after
        it in their original order.
    """

    if not stop_available(screen):
        return shortcuts
    return (STOP_RUN_FOOTER_HINT, *shortcuts)


def sync_stop_affordances(screen: Any) -> None:
    """Re-advertise the stop key when its availability flips.

    Keyed on `stop_available` -- run state AND the setup-modal gate -- not
    on run state alone, so the footer hint never outlives the key.

    Args:
        screen: The Console screen whose footer hints and bindings follow
            the stop key's availability.
    """

    available = stop_available(screen)
    if getattr(screen, "_console_stop_key_advertised", False) == available:
        return
    screen._console_stop_key_advertised = available
    screen._register_console_footer_shortcuts()
    screen.refresh_bindings()


async def stop_this_tab_run(screen: Any) -> None:
    """Stop the viewed tab's run through the same path as the Stop button.

    Args:
        screen: The Console screen whose viewed tab's run is stopped.
    """

    if screen._console_setup_modal_blocking():
        return
    # A still-focused Stop (a palette closing restores focus to it) disables
    # itself below and hands focus to the draft (`ComposerControlButton`).
    await screen._stop_console_generation_from_visible_action()


async def stop_command(screen: Any, parse: Any) -> bool:
    """``/stop``: stop the viewed tab's run.

    Clears its own draft first, like ``/steer`` (review I-3 there): dispatch
    restores the stash before the handler, so ``/stop`` would otherwise sit
    in the composer after the run it stopped.

    Args:
        screen: The Console screen dispatching the command.
        parse: The command parse (unused: ``/stop`` takes no arguments).

    Returns:
        True: the command is always handled.
    """

    del parse  # takes no arguments
    screen._clear_console_composer_draft()
    await stop_this_tab_run(screen)
    return True


async def redirect_from_draft(screen: Any) -> None:
    """Redirect the running turn, taking the composer draft as the correction.

    Moved verbatim from the Redirect button's handler so the palette entry
    shares it. An empty draft is a prompt to type one, not a no-op.

    Args:
        screen: The Console screen whose viewed tab's run is redirected.
    """

    if screen._console_setup_modal_blocking():
        return
    composer = screen._console_composer_or_none()
    text = composer.draft_text().strip() if composer is not None else ""
    if not text:
        screen.app_instance.notify(
            "Type your correction in the composer, then press Redirect.",
            severity="warning",
        )
        return
    # TASK-33622.7: Redirect corrects the ACTIVE chat's run, so the draft must
    # be that chat's own -- not one left over from a chat being switched away.
    if screen._session.refuse_send_from_unbound_composer(
        strict=True, action="redirected"
    ):
        return
    controller = screen._ensure_console_chat_controller()
    refusal = controller.redirect_active_run(text)
    if refusal is not None:
        screen.app_instance.notify(f"Not redirected: {refusal}", severity="warning")
        return
    screen._clear_console_composer_draft()
    screen.app_instance.notify("Redirect sent — correcting the running turn.")


def composer_menu_state(screen: Any) -> dict[str, Any]:
    """Return the inputs the Composer menu renders its entries from.

    Args:
        screen: The Console screen the menu is opened (or consulted) for.

    Returns:
        Keyword arguments for `ConsoleComposerMenuModal` and
        `build_composer_menu_entries`.
    """

    composer = screen._console_composer_or_none()
    return {
        "attachment_kind": screen._console_pending_attachment_kind(),
        "ephemeral": screen._console_active_session_is_ephemeral(),
        # Same input the action-row button read before it moved to the menu,
        # so Save Chatbook's available/unavailable copy is unchanged.
        "can_save_chatbook": screen._console_chatbook_action_available(),
        "draft_available": bool(composer is not None and composer.draft_text().strip()),
        "improvement_undo_available": bool(
            composer is not None and composer.improvement_undo_available
        ),
    }


async def open_composer_menu(screen: Any) -> None:
    """Open the Composer actions menu (palette route, TASK-33622.2).

    Inert while the first-run setup card blocks the Console, like every
    other Console palette action: the card is embedded in the Console, not
    a pushed screen, so Ctrl+P still lists this entry under it.

    Args:
        screen: The Console screen the palette entry was built for.
    """

    if screen._console_setup_modal_blocking():
        return
    await screen._open_console_composer_menu()


def run_composer_menu_action(screen: Any, action_id: str) -> None:
    """Run one Composer-menu action from outside the menu.

    Honours the menu's own contract: an entry the menu would disable (or not
    list) is refused with the reason the menu would show, never run behind
    its back -- e.g. Save as Chatbook in a temporary chat. Inert while the
    setup card blocks the Console (see `open_composer_menu`).

    Args:
        screen: The Console screen the palette entry was built for.
        action_id: The Composer-menu action to run (``ACTION_*``).
    """

    if screen._console_setup_modal_blocking():
        return
    entries = {
        entry.action_id: entry
        for entry in build_composer_menu_entries(**composer_menu_state(screen))
    }
    entry = entries.get(action_id)
    if entry is None or not entry.enabled:
        reason = (
            entry.description
            if entry is not None
            else _MENU_ACTION_ABSENT_COPY.get(
                action_id, "That composer action is not available right now."
            )
        )
        screen.app_instance.notify(reason, severity="warning")
        return
    screen._handle_console_composer_menu_choice(action_id)


def _publish_console_authored_draft(self, draft: str, token: tuple[int, int]) -> bool:
    """Publish authored identity synchronously, before queued widget messages."""
    session_id = self._console_visible_draft_session_id
    store = self._console_chat_store
    if session_id is None or store is None:
        return False
    switch = self._session._console_draft_switch_snapshot
    if (
        switch is not None
        and switch[0] == session_id
        and store.active_session_id != session_id
    ):
        # The explicit switch owns these edits. Its normal draft sync
        # transfers the typed suffix without publishing it to the old chat.
        return False
    try:
        store.set_session_draft(session_id, draft, authored_token=token)
        self._session._console_visible_draft_revision = store.session_input_snapshot(
            session_id
        ).draft_revision
    except KeyError:
        return False
    return True


def _project_console_received_preparing(self, session_id: str) -> None:
    """Paint the resident admission without running configuration selectors."""
    if self._console_visible_draft_session_id != session_id:
        return
    composer = self._console_composer_or_none()
    if composer is not None:
        composer._authored_draft_observer = self._publish_console_authored_draft
        composer._send_label = "Preparing..."
        composer._send_blocked = True
        try:
            button = composer.query_one("#console-send-message", Button)
            button.label = "Preparing..."
            button.disabled = True
        except QueryError:
            pass
    self._start_console_transcript_sync_timer()
    from .agent import project_console_run_feedback

    project_console_run_feedback(self, "Sending…")


def _project_console_received_finished(self, session_id: str, reason: str = "") -> None:
    """Retire Preparing and show only bounded owner-selected refusal copy."""
    if self._console_visible_draft_session_id != session_id:
        return
    self._sync_console_control_bar()
    if reason:
        self.app_instance.notify(reason, severity="warning")
    from .agent import project_console_run_feedback

    project_console_run_feedback(self, self._console_active_run_copy())


def _project_console_received_input(self, session_id: str, *, _captured_stash=None, _captured_inputs=None) -> None:
    """Render the domain's accepted CAS without authoring another edit."""
    if _captured_stash is not None:
        from ..Console_Modules.wiring import _commit_sent_console_draft

        _commit_sent_console_draft(
            self, session_id, _captured_stash,
            _captured_inputs=_captured_inputs,
        )
        return
    if self._console_visible_draft_session_id != session_id:
        return
    composer = self._console_composer_or_none()
    store = self._console_chat_store
    if composer is None or store is None:
        return
    observer = composer._authored_draft_observer
    composer._authored_draft_observer = None
    try:
        # TASK-33620.15.8: this runs inside run start's post-commit step, so
        # Send's readiness restyles wait for the composer's next callback.
        composer.load_draft(store.session_draft(session_id), defer_action_state=True)
        self._session._console_visible_draft_revision = store.session_input_snapshot(
            session_id
        ).draft_revision
    finally:
        composer._authored_draft_observer = observer


async def _dispatch_console_draft_send(
    self,
    draft: str,
    stash: "ConsoleDraftStash | None" = None,
    *,
    session_id: str | None = None,
    _captured_inputs: Any | None = None,
) -> bool:
    """Compatibility delegate for the one typed queue-aware dispatcher."""

    import asyncio

    from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope

    self._console_draft_spend_refresh.stop()
    async with send_diagnostic_scope(
        "ui_dispatch", self._ui_responsiveness_monitor()
    ) as diagnostic:
        if session_id is None:
            session_id = self._console_visible_send_session_id()
        if stash is None and draft:  # An image-only send has no draft.
            composer = self._console_composer_or_none()
            stash = composer.capture_draft_for_send() if composer else None
        from ..Console_Modules.wiring import receive_console_visible_intent

        try:
            received = receive_console_visible_intent(
                self, draft, session_id, stash,
                _captured_inputs=_captured_inputs,
            )
        except (RuntimeError, ValueError):
            self.app_instance.notify(
                "Draft or chat changed; Send again.", severity="warning"
            )
            return False
        if received is not None:
            if not received:
                return False
            from ..Console_Modules.hooks import in_worker_task

            diagnostic.outcome = "awaiting_review"
            if in_worker_task():
                task = self._console_runtime()._turn_custody[received].task
                outcome = await asyncio.shield(task)
                return bool(
                    getattr(outcome, "accepted", False)
                    or getattr(outcome, "applied", False)
                )
            return False
        from ..Console_Modules.prompt_queue import (
            ConsolePromptDispatchResult,
            ConsolePromptDispatchStatus,
            ConsolePromptQueueUIController,
            _CONSOLE_PROMPT_QUEUE_DISPATCH,
        )

        fallback_kwargs = {}
        queue_owner = self._prompt_queue
        dispatch_function, dispatch_code = _CONSOLE_PROMPT_QUEUE_DISPATCH
        adapters = getattr(self, "_console_received_live_adapters", ())
        if (
            _captured_inputs is not None
            and type(self._prompt_queue) is ConsolePromptQueueUIController
            and getattr(self._prompt_queue.dispatch, "__func__", None)
            is dispatch_function
            and dispatch_function.__code__ is dispatch_code
            and any(
                target is self._prompt_queue
                and name == "_launch_chain_async"
                and getattr(target, name, None) is original
                for target, name, original in adapters
            )
            and all(
                getattr(target, name, None) is original
                for target, name, original in adapters
                if target is self._prompt_queue
            )
        ):
            fallback_kwargs["_captured_inputs"] = _captured_inputs

        async def dispatch_fallback():
            if fallback_kwargs and (
                self._prompt_queue is not queue_owner
                or getattr(self._prompt_queue.dispatch, "__func__", None)
                is not dispatch_function
                or dispatch_function.__code__ is not dispatch_code
                or not all(
                    getattr(target, name, None) is original
                    for target, name, original in adapters
                    if target is queue_owner
                )
            ):
                detail = "Draft or chat changed; Send again."
                self.app_instance.notify(detail, severity="warning")
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.REFUSED,
                    session_id=session_id,
                    detail=detail,
                )
            return await self._prompt_queue.dispatch(
                draft, session_id=session_id, stash=stash, **fallback_kwargs
            )

        result = await self._hooks.dispatch(
            draft,
            session_id=session_id,
            stash=stash,
            dispatch=dispatch_fallback,
        )
        diagnostic.outcome = result.status.value
        return result.accepted


def _stock_received_rag_sources(screen):
    from ..Screens.chat_screen import _console_library_rag_source_scope

    return _console_library_rag_source_scope(screen)


def _stock_received_rag_top_k():
    from ..Screens.chat_screen import _console_library_rag_profile_top_k

    return _console_library_rag_profile_top_k()


class PreparingPollChanged(RuntimeError):
    """The captured manual display must leave before ordinary FULL repair."""


_PREPARING_SCREEN_READERS = ()
_PREPARING_FULL_SYNC = None


def _preparing_methods_current(owner, records) -> bool:
    """Match only directly consumed defining callbacks for this display route."""
    from types import MethodType

    return all(
        isinstance(method := getattr(owner, name, None), MethodType)
        and method.__self__ is owner
        and method.__func__ is function
        and function.__code__ is code
        and function.__defaults__ is defaults
        and function.__kwdefaults__ is kwdefaults
        and tuple((kwdefaults or {}).items()) == keyword_values
        for name, function, code, defaults, kwdefaults, keyword_values in records
    )


def bind_preparing_screen_sources(screen_type) -> None:
    """Bind screen definitions at class publication, before lazy display use."""
    global _PREPARING_SCREEN_READERS, _PREPARING_FULL_SYNC

    names = (
        "_sync_console_chat_core_state",
        "_dispatch_active_console_roleplay_refresh",
        "_global_chat_display_name",
        "_console_agent_runtime_enabled",
        "_ensure_console_agent_bridge",
        "_ensure_console_chat_store",
        "_ensure_console_chat_controller",
        "_sync_console_native_session_tabs",
        "_sync_native_console_chat_ui",
    )
    records = tuple(
        (
            name,
            method,
            method.__code__,
            method.__defaults__,
            method.__kwdefaults__,
            tuple((method.__kwdefaults__ or {}).items()),
        )
        for name in names
        for method in (getattr(screen_type, name),)
    )
    _PREPARING_SCREEN_READERS, _PREPARING_FULL_SYNC = records[:-1], records[-1]


def _preparing_helpers_current() -> bool:
    return all(
        globals().get(name) is function
        and function.__code__ is code
        and function.__defaults__ is defaults
        and function.__kwdefaults__ is kwdefaults
        for name, function, code, defaults, kwdefaults in _PREPARING_HELPERS
    )


def preparing_display_read(self) -> bool:
    """Read established owners only inside the exact current display task."""
    import asyncio

    owner = getattr(self, "_console_preparing_display_task", None)
    if owner is None:
        return False
    try:
        current_task = asyncio.current_task()
    except RuntimeError:  # Ordinary thread callers never join this display task.
        return False
    if owner is not current_task:
        return False
    if preparing_poll_record(self) is not self._console_preparing_display_record:
        raise PreparingPollChanged
    return True


def poll_source(self) -> dict | None:
    """Observe established owners; this is never native/action authority."""
    from tldw_chatbook import config

    from ...Chat.console_chat_controller import ConsoleChatController
    from ...Chat.console_chat_store import ConsoleChatStore
    from ...Chat.console_runtime import ConsoleRuntime
    from ..Screens.chat_screen import ChatScreen, _console_screen_is_torn_down

    if type(self) is not ChatScreen or _console_screen_is_torn_down(self):
        return None
    runtime = self._console_runtime()
    if type(runtime) is not ConsoleRuntime:
        return None
    app = self.app_instance
    store, controller = runtime.chat_store, runtime.chat_controller
    if (
        type(store) is not ConsoleChatStore
        or type(controller) is not ConsoleChatController
        or controller.store is not store
        or controller.app is not app
        or runtime._app is not app
        or runtime._disposed
        or controller._disposed
        or controller._shutdown_requested.is_set()
    ):
        return None
    session = store._sessions.get(store.active_session_id)
    registry = vars(app).get("_workspace_registry_service")
    if registry is None:
        registry = vars(app).get("workspace_registry_service")
    generation = getattr(registry, "mutation_generation", None)
    app_config = getattr(app, "app_config", None)
    if (
        session is None
        or session.settings is None
        or type(app_config) is not dict  # Custom mappings stay FULL.
        or type(generation) is not int  # Booleans/custom generations stay FULL.
    ):
        return None
    return {
        "visit": self._current_console_attach_visit(),
        "config": config,
        "config_identity": config.current_config_identity(),
        "app": app,
        "app_config": app_config,
        "chat_database": vars(app).get("chachanotes_db"),
        "runtime": runtime,
        "store": store,
        "controller": controller,
        "registry": registry,
        "registry_generation": generation,
        "session": session,
        "settings": session.settings,
        "bridge": controller._agent_bridge,
        "runtime_bridge": runtime._agent_bridge,
        "membership": tuple(store.sessions()),
    }


def poll_source_is_current(self, captured: dict | None) -> bool:
    current = poll_source(self)
    if captured is None or current is None:
        return False
    return (
        captured["visit"].can_complete(current["visit"])
        and captured["config_identity"] == current["config_identity"]
        and captured["registry_generation"] == current["registry_generation"]
        and all(
            captured[name] is current[name]
            for name in (
                "config",
                "app",
                "app_config",
                "chat_database",
                "runtime",
                "store",
                "controller",
                "registry",
                "session",
                "settings",
                "bridge",
                "runtime_bridge",
            )
        )
        and len(captured["membership"]) == len(current["membership"])
        and all(
            before is after
            for before, after in zip(captured["membership"], current["membership"])
        )
    )


def preparing_poll_record(self):
    """Select only a sole, unchanged stock manual receipt for display work."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_received_configuration_preparation,
    )
    from tldw_chatbook.Chat.console_received_dispatch import stock_native_methods

    from ...Chat.console_chat_controller import _CONSOLE_PREPARING_CONTROLLER_READERS
    from ..Screens import chat_screen as screen_module
    from .session import _CONSOLE_PREPARING_SESSION_READERS, ConsoleSessionController
    from .turn_admission import console_live_runtime_enabled, policy_producers_current

    source = getattr(self, "_console_completed_full_source", None)
    if (
        not _preparing_helpers_current()
        or not policy_producers_current()
        or self._console_sync_requested
        or not self._console_attach_sync_complete
        or not self._console_attach_reconciled
        or self._console_attach_reconcile_running
        or self._console_ordered_resume_pending()
        or not poll_source_is_current(self, source)
        or not stock_native_methods(
            self, screen_module._CONSOLE_RECEIVED_SCREEN_SELECTORS
        )
        or not _preparing_methods_current(self, _PREPARING_SCREEN_READERS)
        or type(self._session) is not ConsoleSessionController
        or not _preparing_methods_current(
            self._session, _CONSOLE_PREPARING_SESSION_READERS
        )
        or any(
            getattr(screen_module, function.__name__, None) is not function
            or function.__code__ is not code
            for function, code in screen_module._CONSOLE_RECEIVED_RAG_READERS
        )
    ):
        return None
    runtime, store, controller = (
        source["runtime"],
        source["store"],
        source["controller"],
    )
    if (
        len(runtime._turn_custody) != 1
        or not _preparing_methods_current(
            controller, _CONSOLE_PREPARING_CONTROLLER_READERS
        )
        or any(
            store.dispatch_recovery_needs_queue_hydration(session.id)
            for session in source["membership"]
        )
        # The received claim itself occupies one slot. No other run may.
        or controller.in_flight_run_count() != 1
        or any(
            worker.group.startswith("console-run-") and not worker.is_finished
            for worker in self.workers
        )
        or not _preparing_methods_current(
            controller, _CONSOLE_PREPARING_CONTROLLER_READERS
        )
    ):
        return None
    # Existing Goal ownership and retained rule helpers never narrow here.
    if (
        getattr(controller, "_goal_coordinator", None) is not None
        or runtime.response_rule_cleanup_pending
    ):
        return None
    rules = controller.response_rules
    if rules is not None:
        from ...Chat.response_rules.runtime import (
            _PREPARING_STATE_READER,
            ResponseRuleRuntime,
        )

        if type(rules) is not ResponseRuleRuntime or not _preparing_methods_current(
            rules, (_PREPARING_STATE_READER,)
        ):
            return None
        if any(
            rules.state(session.id).phase != "idle" for session in source["membership"]
        ):
            return None
    record = next(iter(runtime._turn_custody.values()))
    intent, claim = record.received_intent, record.received_claim
    if (
        record.store is not store
        or record.session_id != source["session"].id
        or record.request is not None
        or record.task is None
        or record.task.done()
        or intent is None
        or intent.queue_revision is not None
        or claim is None
        or claim.origin is not ConsoleSubmissionOrigin.MANUAL
        or not store.received_turn_is_current(claim)
        or not store.session_inputs_are_current(intent.inputs)
        or intent.view_attachment_generation != source["visit"].generation
        or capture_console_received_configuration_preparation(
            source["app"], store, controller, session_id=record.session_id
        )
        is None
    ):
        return None
    context_provider = controller._turn_context_provider
    if context_provider is not None and (
        getattr(context_provider, "__self__", None) is not self._session
        or getattr(context_provider, "__func__", None)
        is not _CONSOLE_PREPARING_SESSION_READERS[0][1]
    ):
        return None
    adapters = getattr(self, "_console_received_live_adapters", ())
    if not adapters or any(
        getattr(target, name, None) is not original
        for target, name, original in adapters
    ):
        return None
    launch, launch_revision, _ = runtime.snapshot_console_staged_evidence()
    if (
        launch is not intent.staged_evidence_launch
        or launch_revision != intent.staged_evidence_revision
        or controller._provider_selection() != intent.selection.provider_selection
        # Stock provider-direct chats have no bridge without a chat DB.
        or source["bridge"] is None
        and (
            source["chat_database"] is not None or source["runtime_bridge"] is not None
        )
    ):
        return None
    console_config = source["app_config"].get("console")
    if (
        type(console_config) is not dict
        or controller._agent_runtime_enabled
        != (  # Custom mappings stay FULL.
            console_live_runtime_enabled(source["app_config"], console_config)
        )
    ):
        return None
    # Voice owns independent transitions; this first route is manual-only.
    missing = object()
    if self._dictation._console_dictation_state != "idle" or any(
        vars(owner).get(name, missing) is not None
        for owner, names in (
            (
                self._dictation,
                ("_console_dictation_session", "_console_pending_voice_action"),
            ),
            (
                self._hands_free,
                ("_console_hands_free", "_qualified_voice_startup_generation"),
            ),
            (self._realtime, ("session", "close_worker")),
            (
                self._message,
                (
                    "_console_speaking_message_id",
                    "_console_speech_owner",
                    "_console_speech_pending_stop",
                ),
            ),
        )
        for name in names
    ):
        return None
    wake = controller.fleet_wake
    coordinator = runtime.change_review_coordinator
    if (
        wake is not None
        and wake.delivering_conversation_ids()
        or coordinator is not None
        and coordinator.publication_signal.snapshot().pending
    ):
        return None
    if (
        any(
            getattr(self, name, None) is not None
            for name in (
                "_console_roleplay_pending_plan",
                "_console_roleplay_active_plan",
                "_console_roleplay_repair_plan",
            )
        )
        or self._console_roleplay_drain_scheduled
        or self._console_roleplay_tearing_down
        or self._console_roleplay_repair_inflight_generation > 0
        or self._console_roleplay_persistence_task is not None
        and not self._console_roleplay_persistence_task.done()
        or self._console_roleplay_writer_task is not None
        and not self._console_roleplay_writer_task.done()
        or getattr(source["app"], "_console_identity_refresh_generation", 0)
        > self._console_identity_refresh_generation
        or getattr(source["app"], "_console_roleplay_repair_generation", 0)
        > max(
            self._console_roleplay_repair_generation,
            getattr(source["app"], "_console_roleplay_repair_consumed_generation", 0),
        )
        or (record.session_id, self._global_chat_display_name())
        != self._last_console_roleplay_refresh_key
    ):
        return None
    return record


async def sync_poll_display_ui(self) -> bool:
    """Keep general polls FULL; only an unchanged manual Preparing pass narrows."""
    # Custom/legacy full callbacks retain their existing no-keyword contract.
    if not _preparing_methods_current(self, (_PREPARING_FULL_SYNC,)):
        return await self._sync_native_console_chat_ui()
    return await self._sync_native_console_chat_ui(_preparing_poll=True)


_PREPARING_HELPERS = tuple(
    (
        function.__name__,
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
    )
    for function in (
        preparing_display_read,
        poll_source,
        poll_source_is_current,
        preparing_poll_record,
        sync_poll_display_ui,
        _preparing_methods_current,
    )
)


def _project_console_saved_input(
    self, session_id: str, *, _captured_stash, _captured_inputs
) -> None:
    """Project a successful draft CAS after settling its current view owner."""
    store = self._console_chat_store
    if self._console_visible_draft_session_id != store.active_session_id:
        self._session._sync_console_session_draft()
    composer = self._console_composer_or_none()
    if (
        composer is not None
        and self._console_visible_draft_session_id == session_id
        and store.active_session_id == session_id
        and composer.draft_text() == _captured_inputs.draft
    ):
        # A tab reload creates a new capture of the same accepted draft.
        # Spend it before hydration so a held Enter cannot send it again.
        composer.spend_captured_draft(composer.capture_draft_for_send())
        self._session._console_undo_histories.pop(session_id, None)
        self._project_console_received_input(session_id)
    self._project_console_received_input(
        session_id,
        _captured_stash=_captured_stash,
        _captured_inputs=_captured_inputs,
    )
