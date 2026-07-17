"""Console inspector: attach/detach conversation-scoped chat dictionaries.

Roleplay P1g Task 5. Thin seam between the Console inspector's Attach/Detach
buttons (``ChatScreen``) and the P1e ``ChatDictionaryScopeService``. Mirrors
the conversation-attach pattern already proven by the Personas workbench
(``UI/Screens/personas_screen.py``'s ``DictionaryAttachRequested`` /
``DictionaryDetachRequested`` handlers) and the character-attach worker
pattern (P1f's ``_character_dictionary_attach_worker``), but targets the
native Console's active conversation instead of a Personas entity selection.

The two ``handle_console_dictionary_*`` functions own the write + error
handling only; the caller (``ChatScreen``) owns resolving the conversation
id and refreshing the cached "what's in play" summary after a successful
call (``ChatScreen.refresh_active_dictionaries_summary()``).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as cdl
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError


async def handle_console_dictionary_attach(app: Any, conversation_id: Any, dictionary_id: Any) -> bool:
    """Attach ``dictionary_id`` to ``conversation_id`` via the scope service.

    Never raises -- every failure path notifies the user and returns False.
    Returns True only once the write has actually succeeded.
    """
    if not conversation_id:
        app.notify("Start or load a conversation first.", severity="warning")
        return False
    service = getattr(app, "chat_dictionary_scope_service", None)
    if service is None:
        app.notify("Chat dictionaries are not available right now.", severity="warning")
        return False
    try:
        await service.attach_to_conversation(int(dictionary_id), str(conversation_id), mode="local")
    except ConflictError:
        app.notify("Dictionaries changed since loaded. Try again.", severity="warning")
        return False
    except Exception:
        logger.opt(exception=True).warning(
            f"Could not attach dictionary {dictionary_id!r} to conversation {conversation_id!r}."
        )
        app.notify("Could not attach the dictionary.", severity="warning")
        return False
    return True


async def handle_console_dictionary_detach(app: Any, conversation_id: Any, dictionary_id: Any) -> bool:
    """Detach ``dictionary_id`` from ``conversation_id`` via the scope service.

    Mirrors :func:`handle_console_dictionary_attach` exactly, over
    ``detach_from_conversation``.
    """
    if not conversation_id:
        app.notify("Start or load a conversation first.", severity="warning")
        return False
    service = getattr(app, "chat_dictionary_scope_service", None)
    if service is None:
        app.notify("Chat dictionaries are not available right now.", severity="warning")
        return False
    try:
        await service.detach_from_conversation(int(dictionary_id), str(conversation_id), mode="local")
    except ConflictError:
        app.notify("Dictionaries changed since loaded. Try again.", severity="warning")
        return False
    except Exception:
        logger.opt(exception=True).warning(
            f"Could not detach dictionary {dictionary_id!r} from conversation {conversation_id!r}."
        )
        app.notify("Could not detach the dictionary.", severity="warning")
        return False
    return True


def console_attachable_dictionaries(db: Any, conversation_id: Any) -> list[dict]:
    """Local dictionaries NOT yet attached to ``conversation_id`` (sync DB read).

    Character-source dictionaries are never included here -- the Console
    attach/detach flow targets only the conversation scope. Never raises: a
    bad row (or a DB error) degrades to ``[]`` rather than breaking the
    attach picker.
    """
    if db is None:
        return []
    try:
        attached = set(cdl.conversation_dictionary_ids(db, conversation_id))
        rows: list[dict] = []
        for row in cdl.list_chat_dictionaries(db, limit=1000, include_disabled=True) or []:
            try:
                dictionary_id = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            if dictionary_id in attached:
                continue
            rows.append({"dictionary_id": dictionary_id, "name": str(row.get("name") or "(unnamed)")})
        return rows
    except Exception:
        logger.opt(exception=True).warning("Could not list attachable dictionaries for the Console picker.")
        return []


def console_attached_dictionaries(db: Any, conversation_id: Any) -> list[dict]:
    """Dictionaries currently attached to ``conversation_id`` (sync DB read).

    Only conversation-source dictionaries are ever returned -- a character's
    embedded dictionaries are out of scope for Console conversation
    attach/detach and are never consulted here, even when one is active for
    the same chat. Never raises: an id that fails to load is skipped.
    """
    if db is None:
        return []
    try:
        rows: list[dict] = []
        for dictionary_id in cdl.conversation_dictionary_ids(db, conversation_id):
            try:
                record = cdl.load_chat_dictionary(db, dictionary_id)
            except Exception:
                continue
            if not record:
                continue
            rows.append({"dictionary_id": int(dictionary_id), "name": str(record.get("name") or "(unnamed)")})
        return rows
    except Exception:
        logger.opt(exception=True).warning("Could not list attached dictionaries for the Console picker.")
        return []


__all__ = [
    "handle_console_dictionary_attach",
    "handle_console_dictionary_detach",
    "console_attachable_dictionaries",
    "console_attached_dictionaries",
]
