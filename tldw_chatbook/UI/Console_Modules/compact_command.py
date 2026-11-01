"""`/compact` and the palette's Compact entry: run Compact now (TASK-33621.4).

Compact now used to live only at the bottom of Chat settings ▸ Context and
memory. The command runs the same controller action, so it compacts exactly
what that button does, and reports the outcome in the chat it ran for.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .command_handoff import COMMAND_ORIGIN, CommandOrigin, refuse_if_chat_changed

#: The toast shown while the summary call runs; the row reports the outcome.
COMPACTING_COPY = "Compacting older turns (one model call)…"


def capture_compact_origin(screen: Any) -> CommandOrigin | None:
    """Capture slash origin or palette chat before scheduling its worker."""
    origin = COMMAND_ORIGIN.get()
    if origin is not None:
        return origin
    session_id = screen._ensure_console_chat_controller().store.active_session_id
    return CommandOrigin(screen, session_id) if session_id is not None else None


async def run_compact_command(screen: Any, *, origin: CommandOrigin | None) -> bool:
    """Compact the admitted chat's older turns and post the outcome there.

    Args:
        screen: The Console screen (its controller, system-row append, sync
            and notify).
        origin: Chat captured by the action before its worker was scheduled.

    Returns:
        True when conversation memory was updated.
    """

    if origin is None:
        await screen._append_native_console_system_message("No chat to compact.")
        return False
    if origin.screen is not screen or refuse_if_chat_changed("compact", origin=origin):
        return False
    controller = screen._ensure_console_chat_controller()
    session_id = origin.session_id
    screen.notify(COMPACTING_COPY, markup=False)
    # Shielded like the settings button's run: leaving Console does not
    # abandon a summary call that is already billed.
    task = asyncio.ensure_future(controller.compact_context_now(session_id))
    try:
        compacted, copy = await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a command reports, never crashes
        # The type only: a provider error's text can carry request details.
        logger.warning(
            "Console /compact failed (exception_type={})", type(exc).__name__
        )
        compacted, copy = (
            False,
            "Conversation compaction failed before memory was updated.",
        )
    await screen._append_native_console_system_message(copy, session_id=session_id)
    await screen._sync_native_console_chat_ui()
    return bool(compacted)
