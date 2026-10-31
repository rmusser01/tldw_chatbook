"""`/compact` and the palette's Compact entry: run Compact now (TASK-33621.4).

Compact now used to live only at the bottom of Chat settings ▸ Context and
memory. The command runs the same controller action, so it compacts exactly
what that button does, and reports the outcome in the chat it ran for.
"""

from __future__ import annotations

import asyncio
from typing import Any

#: The toast shown while the summary call runs; the row reports the outcome.
COMPACTING_COPY = "Compacting older turns (one model call)…"


async def run_compact_command(screen: Any) -> bool:
    """Compact the active chat's older turns and post the outcome there.

    Args:
        screen: The Console screen (its controller, system-row append, sync
            and notify).

    Returns:
        True when conversation memory was updated.
    """

    controller = screen._ensure_console_chat_controller()
    session_id = controller.store.active_session_id
    if session_id is None:
        await screen._append_native_console_system_message("No chat to compact.")
        return False
    screen.notify(COMPACTING_COPY, markup=False)
    # Shielded like the settings button's run: leaving Console does not
    # abandon a summary call that is already billed.
    task = asyncio.ensure_future(controller.compact_context_now(session_id))
    try:
        compacted, copy = await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a command reports, never crashes the screen
        compacted, copy = (
            False,
            "Conversation compaction failed before memory was updated.",
        )
    await screen._append_native_console_system_message(copy, session_id=session_id)
    await screen._sync_native_console_chat_ui()
    return bool(compacted)
