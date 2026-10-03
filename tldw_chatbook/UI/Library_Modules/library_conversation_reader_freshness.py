"""Keep an open Conversations transcript current with the list beside it.

TASK-33628.10. Library is a reused screen, so every return visit re-reads
the Conversations list (``on_screen_resume`` -> page request -> apply ->
``_ensure_library_conversation_reader_selection``). The reader skipped its
own re-read whenever the list record's conversation ``version`` matched the
loaded one. A message delete, Undo, edit or send never changes that
version, so after a Console Delete the list showed the new count while the
reader kept its earlier load ("4 of 4 messages") until restart.

The saved transcript carries its own change token: ``message_epoch`` (a hash
of the conversation's message row count and version sum, which every
reader-visible message write moves) beside the exact ``message_total``. Once
per list read, a loaded transcript is re-checked with a one-message,
one-character read of those two values. When either moved, the reader's
existing fenced pipeline reloads it in place, keeping Read/Info and the Find
query. A load this list read started is not re-checked: it is already
current.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...Library.library_conversation_reader_state import ConversationReaderState
from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_CONVERSATIONS

#: Worker group for the re-check; a newer re-check replaces an older one.
RECHECK_WORKER_GROUP = "library_conversation_reader_recheck"


def ensure_reader_current(controller: Any, conversation_id: str) -> None:
    """Load the selected conversation, or re-check the transcript loaded for it.

    Args:
        controller: The ``LibraryConversationReaderController``.
        conversation_id: The list's selected conversation.
    """
    state = controller._library_conversation_reader_state
    record_version = controller._conversation_reader_record_version(
        controller._conversation_reader_record(conversation_id)
    )
    list_read = (conversation_id, controller._library_conversation_request_generation)
    if state.selected_id == conversation_id and (
        (
            state.loading
            and (record_version is None or state.selected_version == record_version)
        )
        or (
            state.loaded_actions_eligible
            and (record_version is None or state.loaded_version == record_version)
        )
    ):
        if (
            state.loaded_actions_eligible
            and controller._library_conversation_reader_checked_read != list_read
        ):
            controller._library_conversation_reader_checked_read = list_read
            controller.run_worker(
                recheck_loaded_transcript(controller, state),
                exclusive=True,
                group=RECHECK_WORKER_GROUP,
            )
        return
    controller._library_conversation_reader_checked_read = list_read
    controller._start_library_conversation_reader_selection(conversation_id)


async def recheck_loaded_transcript(
    controller: Any, checked: ConversationReaderState
) -> None:
    """Reload ``checked``'s transcript when its saved epoch or total moved.

    A failed or missing read keeps the transcript on screen and says
    nothing: the next list read re-checks, a deleted conversation is the
    list's own absence check to settle, and the reader reports failures of
    the reload it starts.

    Args:
        controller: The ``LibraryConversationReaderController``.
        checked: The eligible reader state the re-check was started for.
    """
    service = controller._conversation_reader_service()
    read = getattr(service, "get_library_conversation_messages", None)
    if not callable(read) or checked.loaded_id is None:
        return
    try:
        saved = await controller._run_library_service_call(
            read,
            checked.loaded_id,
            message_offset=0,
            message_limit=1,
            max_chars=1,
        )
    except Exception:  # noqa: BLE001 - the next list read re-checks again
        return
    current = controller._library_conversation_reader_state
    if not (
        controller._library_conversation_reader_mounted_authority
        and controller._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
        and not controller._library_conversations_select_mode
        and current.loaded_actions_eligible
        and current.loaded_id == checked.loaded_id
        and current.loaded_generation == checked.loaded_generation
        and isinstance(saved, Mapping)
    ):
        return
    if (
        saved.get("message_epoch") == current.message_epoch
        and saved.get("message_total") == current.message_total
    ):
        return
    controller._start_library_conversation_reader_selection(checked.loaded_id)
