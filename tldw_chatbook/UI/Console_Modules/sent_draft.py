"""A sent Console draft leaves its chat's draft exactly once (TASK-33620.15.2).

Lead ruling (TASK-33620.15): what is sent is what was in the composer at the
press that sent it; keys typed afterwards stay as the next draft; nothing
typed is ever lost or silently sent. Keys flow while a send is admitted, so
by the time a send commits its capture the draft can have changed under it:

* A tab round-trip reloads the draft with a new generation, and the commit
  failed closed: the sent text stayed in the composer, and an Enter pressed
  on it sent it again. A reload with nothing typed since the capture still
  shows that capture, so the commit takes it out (``commit_capture``).
* A non-append edit (Home, then typing) leaves a draft the commit cannot
  take the sent text out of. The draft stays, as it must, and the user is
  now told that the message was sent; captures taken before that point are
  retired, so no held press can send the text again (``take_out_sent_draft``).
* A tab left during admission: the commit used to touch only the visible
  composer, so the saved draft of the chat the message was sent from kept
  it, and it came back when the user returned. It now comes off that saved
  draft, with anything typed after it kept.

A capture another send has already committed, or that a retire or a new
draft replaced, is superseded (``capture_superseded``): a press holding it
is dropped instead of sending the same text twice.

Imported on the first send only, so it adds nothing to the ADR-097 boot
census.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from tldw_chatbook.UI.character_display_text import sanitize_character_display_label
from tldw_chatbook.Utils.input_validation import escape_markup
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

#: A dispatched draft the composer still shows after its commit.
SENT_DRAFT_KEPT = (
    "Your message was {verb}, but the composer still shows it because the "
    "draft changed while sending. Clear or edit it before you send again."
)
#: The same, for a chat that is not on screen.
SENT_DRAFT_KEPT_IN = (
    "Your message in “{chat}” was {verb}, but that chat's draft still shows "
    "it because the draft changed while sending. Clear or edit it before you "
    "send again."
)
#: A press refused because a message held in another tab is being sent.
REFUSED_FOR_OTHER_CHAT = (
    "Not sent: your message in “{chat}” is still being sent. This text stays "
    "in the composer; send it again in a moment."
)
#: How much of a chat's title a notice names.
_LABEL_CHARACTERS = 60


def chat_label(store: Any, session_id: str) -> str:
    """Return a chat's title as a notice can show it: one line, escaped.

    Args:
        store: The Console chat store.
        session_id: The chat to name.

    Returns:
        The sanitized, bounded, markup-escaped title, or "Untitled".
    """
    title = next(
        (session.title for session in store.sessions() if session.id == session_id),
        "",
    )
    label = sanitize_character_display_label(title, max_characters=_LABEL_CHARACTERS)
    return escape_markup(label or "Untitled")


def _reloaded(composer: Any, stash: ConsoleDraftStash) -> bool:
    """Whether the composer shows ``stash`` again after a reload.

    A tab round-trip reloads the chat's saved draft with a new generation.
    With nothing typed since the capture (anywhere: the edit serial is the
    widget's), the reloaded draft is the captured one, plus nothing.
    """
    live = composer.capture_draft_snapshot()
    return (
        live.generation != stash.generation
        and live.edit_serial == stash.edit_serial
        and composer.draft_text().startswith(stash.text)
    )


def capture_superseded(composer: Any, stash: ConsoleDraftStash | None) -> bool:
    """Whether ``stash`` no longer describes the composer's draft.

    A commit, clear, retire or a different draft moved the draft generation
    on; a reload with nothing typed did not supersede it. An edit within the
    same generation does not either: under the lead ruling the capture is
    still what its press sends.

    Args:
        composer: The Console composer, showing the capture's chat.
        stash: A press's capture; ``None`` (nothing typed) is never stale.

    Returns:
        True when sending ``stash`` could repeat text that already left.
    """
    if stash is None:
        return False
    live = composer.capture_draft_snapshot()
    return live.generation != stash.generation and not _reloaded(composer, stash)


def commit_capture(composer: Any, stash: ConsoleDraftStash | None) -> bool:
    """Commit ``stash`` out of the composer, through a reload if need be.

    Args:
        composer: The Console composer, showing the capture's chat.
        stash: The dispatched capture.

    Returns:
        Whether the captured text left the composer.
    """
    if composer.commit_captured_draft(stash):
        return True
    if stash is None or not _reloaded(composer, stash):
        return False
    live = composer.capture_draft_snapshot()
    return composer.commit_captured_draft(replace(stash, generation=live.generation))


def superseded(screen: Any, session_id: str, stash: ConsoleDraftStash | None) -> bool:
    """Whether a press's capture can no longer be sent: its draft already left.

    Judged only while the capture's chat is on screen; elsewhere the send's
    own chat check refuses it.

    Args:
        screen: The Console ``ChatScreen``.
        session_id: The chat the press was made in.
        stash: The press's capture.

    Returns:
        True when another send took the draft, or a new draft replaced it.
    """
    composer = screen._console_composer_or_none()
    if composer is None or screen._console_visible_draft_session_id != session_id:
        return False
    return capture_superseded(composer, stash)


def refused_for(screen: Any, session_id: str) -> str:
    """The refusal copy naming the tab whose held message is being sent."""
    store = screen._ensure_console_chat_store()
    return REFUSED_FOR_OTHER_CHAT.format(chat=chat_label(store, session_id))


def take_out_sent_draft(
    session_id: str,
    stash: ConsoleDraftStash | None,
    *,
    composer: Any | None,
    visible_session_id: str | None,
    store: Any,
    undo_histories: dict[str, Any],
    notify: Callable[[str], None],
    verb: str = "sent",
) -> None:
    """Take a dispatched capture out of the draft of the chat it came from.

    Args:
        session_id: The chat the capture was sent (or queued) from.
        stash: The dispatched capture; ``None`` for an image-only send.
        composer: The Console composer, or ``None``.
        visible_session_id: The chat the composer is showing.
        store: The Console chat store.
        undo_histories: Each hidden chat's banked composer undo history.
        notify: Shows a warning to the user.
        verb: "sent" or "queued", for the notice.
    """
    if composer is not None and visible_session_id == session_id:
        if not commit_capture(composer, stash):
            # The user's edit stays; the text in it already went. Retire
            # every capture of it, so no held press sends it a second time.
            composer.retire_captured_drafts()
            notify(SENT_DRAFT_KEPT.format(verb=verb))
        _save_draft(store, session_id, composer.draft_text())
        return
    if stash is None:
        return
    try:
        draft = store.session_draft(session_id)
    except KeyError:
        return
    if draft.startswith(stash.text) or not draft:
        undo_histories.pop(session_id, None)
        _save_draft(store, session_id, draft[len(stash.text) :])
        return
    notify(SENT_DRAFT_KEPT_IN.format(verb=verb, chat=chat_label(store, session_id)))


def _save_draft(store: Any, session_id: str, text: str) -> None:
    try:
        store.set_session_draft(session_id, text)
    except KeyError:
        pass
