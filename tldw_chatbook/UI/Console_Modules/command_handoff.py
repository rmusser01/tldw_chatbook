"""Run a Console slash command off the pump that delivered its send.

TASK-33622.16. Every visible send -- Enter, the Send button, the Workbench
"send" action -- reaches ``ChatScreen._send_console_message_from_visible_action``
on a Textual message pump, and Textual awaits the handler there: Enter's
``app.call_later`` callback on the APP pump (``MessagePump.on_callback``), a
button press on the Console's own pump. A slash command used to be awaited
right there, and some await a modal and a long operation: ``/generate-video``
awaits its cost confirm, the paid generation and the storage choice. Parked
under it, the app pump delivered no key or click at all -- the confirm never
got the Escape that would have ended it, and Ctrl+Q and F1 were dead -- while
a parked Console pump could not handle the Stop button that cancels the run.

So every command runs in a Console worker and the send returns at once, the
same way from every route (a spoken "Console, send." too: a command is never
"sent", so it has no outcome to wait for). The one exception is an
argument-free ``/rewind``, which the send still runs itself: it opens its
picker with a ``push_screen`` callback and never waits on it. A command's captured draft stays in
the composer until the command itself takes it, as before -- and the worker
carries that capture, so the command takes only that revision
(``command_draft``).

Handing off lets a second press of the same draft arrive while the first
command is still running -- on a parked pump it queued behind it and found the
draft already taken. A repeat of the SAME captured draft for the same chat is
therefore dropped while its first run is in flight; any other command (``/stop``
during a generation, a retyped draft) runs.

It also lets the user switch chats before a command acts, and every command
reads "the active chat" -- written when nothing could change it under them. So
a command is bound to the chat it was sent from (``COMMAND_ORIGIN``): it does
not start unless that chat is still the one showing, a command that awaits
before changing that chat stops when it no longer is
(``refuse_if_chat_changed``), and one that awaits before answering answers in
its own chat (``append_command_output``).

Loaded on the first slash command, never on the boot path (ADR-097).
"""

from __future__ import annotations

import weakref
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.worker import WorkerCancelled

from .command_draft import CAPTURED_COMMAND_DRAFT

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_command_grammar import CommandParse
    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

COMMAND_WORKER_GROUP = "console-command"
"""Worker group of a handed-off command: non-exclusive, so a command never
cancels another (``/stop`` must run beside a generation)."""

_IN_FLIGHT: weakref.WeakKeyDictionary[Any, set[tuple]] = weakref.WeakKeyDictionary()


@dataclass(frozen=True)
class CommandOrigin:
    """The chat a handed-off command was sent from.

    Attributes:
        screen: The Console screen that sent it.
        session_id: The Console chat (session) it was sent from.
    """

    screen: Any
    session_id: str


COMMAND_ORIGIN: ContextVar[CommandOrigin | None] = ContextVar(
    "console_command_origin", default=None
)
"""Set inside a handed-off command's own worker; None for a direct call."""


def run_console_command(
    screen: Any,
    parse: CommandParse,
    session_id: str,
    captured: ConsoleDraftStash | str,
) -> None:
    """Start one parsed slash command in a worker; never await it here.

    Args:
        screen: The Console screen. Its ``_dispatch_console_command`` runs the
            command and its ``run_worker`` owns the hand-off, so a removed
            screen cancels its own commands.
        parse: The parsed command.
        session_id: The chat the send came from.
        captured: The draft the send captured -- its stash (text plus edit
            revision) or, with no composer, its text. With ``session_id`` it
            names a repeat press of the same draft.
    """
    in_flight = _IN_FLIGHT.setdefault(screen, set())
    key = _repeat_key(session_id, captured)
    if key in in_flight:
        logger.debug("Console command repeat dropped while its first run is in flight")
        return
    in_flight.add(key)
    command = _dispatch_then_release(
        screen,
        parse,
        CommandOrigin(screen, session_id),
        in_flight,
        key,
        None if isinstance(captured, str) else captured,
    )
    try:
        screen.run_worker(command, group=COMMAND_WORKER_GROUP, exclusive=False)
    except BaseException:
        command.close()
        in_flight.discard(key)
        raise


async def _dispatch_then_release(
    screen: Any,
    parse: CommandParse,
    origin: CommandOrigin,
    in_flight: set[tuple],
    key: tuple,
    stash: ConsoleDraftStash | None,
) -> None:
    # Set inside the worker's own task, so they are this command's alone.
    CAPTURED_COMMAND_DRAFT.set(stash)
    COMMAND_ORIGIN.set(origin)
    try:
        # The worker starts after its send returned; a chat switch in between
        # would point every handler's "active chat" at the wrong one.
        if refuse_if_chat_changed(parse.name):
            return
        await screen._dispatch_console_command(parse)
    except WorkerCancelled:
        # A modal the command waited on was torn down with its screen or the
        # app (Ctrl+Q under the cost confirm): the command is over, not
        # broken, so it must not surface as an unhandled app exception.
        logger.debug("Console command ended: the modal it waited on was cancelled")
    finally:
        in_flight.discard(key)


def command_origin_session() -> str | None:
    """The chat the running command was sent from.

    Returns:
        Its session id inside a handed-off command; None for a direct call,
        which acts on the active chat as it always did.
    """
    origin = COMMAND_ORIGIN.get()
    return origin.session_id if origin is not None else None


def refuse_if_chat_changed(name: str) -> bool:
    """Stop a command whose chat is no longer the one the Console shows.

    A command calls this after an await and before it acts on "the active
    chat" or the live composer, which by then may belong to another chat.

    Args:
        name: The command's name, for the warning.

    Returns:
        True -- after warning that nothing was done -- when the command must
        stop; False for a direct call, or while its own chat still shows.
    """
    origin = COMMAND_ORIGIN.get()
    if origin is None or _shows_chat(origin):
        return False
    logger.debug("Console command /{} refused: its chat is no longer shown", name)
    origin.screen.app_instance.notify(
        f"/{name} did nothing: you switched Console chats before it finished. "
        "Send it again from the chat it was meant for.",
        severity="warning",
    )
    return True


async def append_command_output(
    append: Callable[..., Awaitable[Any]], message: str
) -> None:
    """Post a command's answer in the chat the command was sent from.

    Args:
        append: The Console's system-row append
            (``_append_native_console_system_message`` or a controller's
            injected one); it takes ``session_id`` as a keyword.
        message: The row to post.
    """
    session_id = command_origin_session()
    if session_id is None:
        await append(message)
    else:
        await append(message, session_id=session_id)


def _shows_chat(origin: CommandOrigin) -> bool:
    screen = origin.screen
    store = screen._ensure_console_chat_store()
    return (
        store.active_session_id == origin.session_id
        and screen._console_visible_draft_session_id == origin.session_id
    )


def _repeat_key(session_id: str, captured: ConsoleDraftStash | str) -> tuple:
    if isinstance(captured, str):
        return (session_id, captured)
    return (session_id, captured.text, captured.edit_serial, captured.generation)
