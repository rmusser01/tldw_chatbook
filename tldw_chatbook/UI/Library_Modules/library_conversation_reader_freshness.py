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
query. A load started during this list read is normally not re-checked: it is
already current. (One narrow exception: a row press that lands while a version
bootstrap is still pending sits below that bootstrap's +2 mark, so it costs one
extra bounded re-check and no reload when nothing moved.) The mark is ``(list request generation, reader generation)``, and a
loaded generation at or past the mark's covers loads this module started and
loads a row press started straight through the reader pipeline alike. A load
already in flight when the list read applies may have read its pages before
the write, so the mark is set just past it and ``recheck_settled_load`` (which
the reader pipeline calls when a load completes) re-checks it once it settles.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger

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
    list_read = controller._library_conversation_request_generation
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
        checked = controller._library_conversation_reader_checked_read
        if state.loading and not (
            checked is not None
            and checked[0] == list_read
            and state.generation >= checked[1]
        ):
            # This load predates the list read, so its pages may have been
            # read before a Console write. ``recheck_settled_load`` re-checks
            # it when it settles; a version bootstrap selects once more first.
            controller._library_conversation_reader_checked_read = (
                list_read,
                state.generation + (1 if state.selected_version is not None else 2),
            )
        elif state.loaded_actions_eligible and not (
            checked is not None
            and checked[0] == list_read
            and state.loaded_generation >= checked[1]
        ):
            _start_recheck(controller, list_read, state)
        return
    controller._start_library_conversation_reader_selection(conversation_id)
    controller._library_conversation_reader_checked_read = (
        list_read,
        controller._library_conversation_reader_state.generation,
    )


def recheck_settled_load(controller: Any) -> None:
    """Re-check a just-settled load that predates the current list read.

    The reader pipeline calls this when a load completes. A load the current
    list read found already in flight sits below its mark; so does a row press
    that lands while a version bootstrap is still pending (the +2 mark), which
    costs one extra bounded re-check and no reload when nothing moved. Every
    other load is at or past the mark and needs nothing.

    Args:
        controller: The ``LibraryConversationReaderController``.
    """
    state = controller._library_conversation_reader_state
    checked = controller._library_conversation_reader_checked_read
    list_read = controller._library_conversation_request_generation
    if (
        checked is not None
        and checked[0] == list_read
        and state.loaded_actions_eligible
        and state.loaded_generation < checked[1]
    ):
        _start_recheck(controller, list_read, state)


def _start_recheck(
    controller: Any, list_read: int, state: ConversationReaderState
) -> None:
    """Mark ``state``'s transcript checked for ``list_read`` and re-check it."""
    controller._library_conversation_reader_checked_read = (
        list_read,
        state.loaded_generation,
    )
    controller.run_worker(
        recheck_loaded_transcript(controller, state),
        exclusive=True,
        group=RECHECK_WORKER_GROUP,
    )


async def recheck_loaded_transcript(
    controller: Any, checked: ConversationReaderState
) -> None:
    """Reload ``checked``'s transcript when its saved epoch or total moved.

    A failed or missing read keeps the transcript on screen and shows the
    user nothing: the next list read re-checks, a deleted conversation is
    the list's own absence check to settle, and the reader reports failures
    of the reload it starts. A failed read logs one warning naming only the
    exception type: storage errors can quote saved message text, so neither
    the exception's message nor its traceback is logged.

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
    except Exception as exc:  # noqa: BLE001 - the next list read re-checks again
        logger.warning(
            "Library conversation reader re-check failed; exception_type={}",
            type(exc).__name__,
        )
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
