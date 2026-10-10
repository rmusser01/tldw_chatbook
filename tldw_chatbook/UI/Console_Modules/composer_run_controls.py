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

`ChatScreen` keeps only the wiring Textual resolves by name (the
``stop_console_run`` binding action, `check_action`, `on_key`); the policy
lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.css.query import QueryError
from textual.dom import DOMNode
from textual.widgets import Button

from ...Widgets.Console.console_composer_bar import ConsoleComposerBar
from ...Widgets.Console.console_composer_menu_modal import (
    ACTION_IMPROVE_CURRENT_DRAFT,
    build_composer_menu_entries,
)

if TYPE_CHECKING:
    from textual.events import Key

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
        "draft_available": bool(
            composer is not None and composer.draft_text().strip()
        ),
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
