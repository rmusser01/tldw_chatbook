"""Temporary (non-persisted) Console conversations: shared vocabulary.

A temporary session never acquires a ``persisted_conversation_id`` (see
``ConsoleChatStore.persist_session_if_needed``), so every durable write in
the store no-ops on its own. What this module owns is the OTHER half of the
guarantee: the UI actions that would write a derived artifact to disk even
though no conversation row exists.

The registry below is the single place that list lives. Adding a new
artifact-producing Console action means adding a row here -- the enumeration
test in ``Tests/Chat/test_console_ephemeral.py`` is what keeps that honest.

The promise is LOCAL DURABILITY only: "not saved locally". Nothing here may
imply privacy or provider-side behavior.

Sink audit (task 1): searching ``tldw_chatbook/Chat/`` and
``tldw_chatbook/Widgets/Console/`` for write/export patterns, and tracing
every ``action_id`` reachable from the Console message-action row and
composer menu in ``tldw_chatbook/UI/Screens/chat_screen.py``, surfaced four
per-message "Save as..." destinations and a message-level "Save Image"
action that write to local storage independently of conversation
persistence, plus a context-snapshot exporter reachable from any Console
session. None of these were named in the design spec's known-sinks list.
A review follow-up also traced the message-action row's ``speak`` entry to
a real file write (TTS playback audio); it is deliberately NOT in this
registry because the file is a transient OS-temp playback buffer,
secure-deleted within seconds and never exposed to the user, not a durable
artifact -- see the spec's audit table for the full reasoning.
See ``Docs/superpowers/specs/2026-07-31-temporary-conversations-design.md``
(``## Sink audit (task 1)``) for the full table and reasoning.
"""

from __future__ import annotations

#: Composer-menu action id for promoting a temporary chat ("Save this chat").
ACTION_SAVE_CHAT = "save-chat"

#: Chip label shown in the Console status strip while a chat is temporary.
TEMPORARY_LABEL = "Temporary — not saved"

#: Chip tooltip. Says what survives and what does not, without implying more.
TEMPORARY_TOOLTIP = (
    "This chat is not saved locally. It is lost when the tab closes or the "
    "app restarts. Activate to save it."
)

#: Action id -> why it is unavailable while the chat is temporary. Keyed by
#: the ids the Console workbench, composer menu, and message-action row
#: already use, so a lookup needs no translation layer.
EPHEMERAL_BLOCKED_ACTIONS: dict[str, str] = {
    "generate-image": (
        "Generating an image writes a file to disk — not available in a "
        "temporary chat."
    ),
    "save-chatbook": (
        "Saving a Chatbook exports a file to disk — not available in a "
        "temporary chat."
    ),
    "save-image": (
        "Saving the image writes a file to disk — not available in a "
        "temporary chat."
    ),
    "save-as-note": (
        "Saving as a Note writes it to the local Notes database — not "
        "available in a temporary chat."
    ),
    "save-as-media": (
        "Saving as Media writes it to the local Media library — not "
        "available in a temporary chat."
    ),
    "save-as-prompt": (
        "Saving as a Prompt writes it to the local Prompts library — not "
        "available in a temporary chat."
    ),
    "save-as-chatbook": (
        "Saving as a Chatbook artifact exports a file to disk — not "
        "available in a temporary chat."
    ),
    "save-context": (
        "Saving the context snapshot writes a JSON file to disk — not "
        "available in a temporary chat."
    ),
}


def blocked_reason(action_id: str, *, ephemeral: bool) -> str | None:
    """Return why ``action_id`` is unavailable, or ``None`` when it is available.

    Args:
        action_id: Console action id (workbench action, composer menu entry,
            or message-action row entry).
        ephemeral: Whether the active session is temporary.

    Returns:
        The reason sentence to show on the disabled control, or ``None`` when
        the action is available (which is always the case outside a temporary
        chat).
    """
    if not ephemeral:
        return None
    return EPHEMERAL_BLOCKED_ACTIONS.get(action_id)
