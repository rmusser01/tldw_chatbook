"""Pure menu model for the Console conversation action menu (TASK-23200).

The Context rail's conversation rows used to carry a star button that shipped
disabled on a fresh install, reserved the full height of a multi-line row, and
was explained by the developer-facing line "Local stars unavailable". A
2026-08-29 UX audit called it dead vertical space. It is replaced by an
asterisk that opens this menu, so a conversation can actually be managed from
the Console screen.

Everything here is pure: given what is true of one row, return the items to
paint. No DOM, no database, no service lookups -- so the menu's shape,
labelling and gating are testable without mounting an app.

Conversation state vocabulary is NOT invented here. It mirrors
``CharactersRAGDB._ALLOWED_CONVERSATION_STATES``, which in turn matches
tldw_server's ``_ALLOWED_CONVERSATION_STATES`` so the two never drift.
"Archive" is not a separate flag: it is the ``resolved`` state, the same
mapping tldw_server's Sync v2 alias table uses (``archived``/``closed`` ->
``resolved``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Canonical conversation states, mirroring CharactersRAGDB and tldw_server.
CONVERSATION_STATES: tuple[str, ...] = (
    "in-progress",
    "resolved",
    "backlog",
    "non-viable",
)

#: The state a persisted conversation carries unless told otherwise.
DEFAULT_CONVERSATION_STATE = "in-progress"

#: The state "Archive" sets, and the one "Unarchive" leaves.
ARCHIVED_STATE = "resolved"

#: Human labels for the raw state vocabulary. The raw values are storage
#: tokens; only these strings are ever shown to a person.
CONVERSATION_STATE_LABELS: dict[str, str] = {
    "in-progress": "In progress",
    "resolved": "Resolved",
    "backlog": "Backlog",
    "non-viable": "Not viable",
}

MenuPage = Literal["root", "status", "more", "copy"]

#: Action ids the menu can emit. Kept as plain strings so the widget, the
#: screen handler and the tests all name them the same way.
ACTION_FAVORITE = "favorite"
ACTION_UNFAVORITE = "unfavorite"
ACTION_ARCHIVE = "archive"
ACTION_UNARCHIVE = "unarchive"
ACTION_RENAME = "rename"
ACTION_DELETE = "delete"
ACTION_SET_STATE_PREFIX = "set-state:"
ACTION_PAGE_PREFIX = "page:"
ACTION_BACK = "page:root"
ACTION_COPY_CLEAN = "copy-markdown:clean"
ACTION_COPY_FULL = "copy-markdown:full"
ACTION_SAVE_MARKDOWN = "save-markdown"


@dataclass(frozen=True, slots=True)
class ConversationMenuItem:
    """One painted row of the conversation action menu.

    Attributes:
        action_id: Stable identifier emitted when the item is chosen.
        label: Text shown to the user.
        enabled: Whether the item may be chosen.
        disabled_reason: Why it may not be, shown as a tooltip. Only ever set
            when ``enabled`` is False -- a disabled control with no stated
            precondition is the defect this menu exists to remove.
        opens_page: The page this item navigates to, when it is a submenu
            opener rather than a command.
        is_current: Whether the item describes the row's present state, so the
            menu can mark it rather than implying choosing it does something.
    """

    action_id: str
    label: str
    enabled: bool = True
    disabled_reason: str = ""
    opens_page: MenuPage | None = None
    is_current: bool = False


@dataclass(frozen=True, slots=True)
class ConversationMenuTarget:
    """What the menu needs to know about the row it was opened from.

    Attributes:
        conversation_id: Persisted id, or None for a chat that has never been
            saved. Nearly every action needs one.
        title: Current title, used by the rename prompt.
        state: Current conversation state; unknown values are tolerated and
            treated as the default rather than raising at paint time.
        starred: Whether the conversation is locally favourited.
        favorites_available: Whether the local marks service can answer at
            all. False replaces the old "Local stars unavailable" line.
        native_session_id: The open native Console session behind this row,
            when there is one; Copy-as reads its messages live from the
            chat store instead of the database.
        has_messages: Whether any source reports messages for the row, so
            the copy entries can gate on it.
    """

    conversation_id: str | None
    title: str = ""
    state: str = DEFAULT_CONVERSATION_STATE
    starred: bool = False
    favorites_available: bool = True
    native_session_id: str = ""
    has_messages: bool = False

    @property
    def is_saved(self) -> bool:
        """Whether this row points at a persisted conversation."""
        return bool(self.conversation_id)

    @property
    def normalized_state(self) -> str:
        """The row's state, falling back to the default when unrecognised."""
        candidate = (self.state or "").strip().lower()
        return (
            candidate
            if candidate in CONVERSATION_STATES
            else (DEFAULT_CONVERSATION_STATE)
        )

    @property
    def is_archived(self) -> bool:
        """Whether the row is in the state Archive sets."""
        return self.normalized_state == ARCHIVED_STATE


_UNSAVED_REASON = "Send or save this chat first."
_FAVORITES_UNAVAILABLE_REASON = "Favourites are unavailable on this device."


def build_conversation_menu(
    target: ConversationMenuTarget,
    page: MenuPage = "root",
) -> tuple[ConversationMenuItem, ...]:
    """Return the items to paint for one page of the menu.

    Args:
        target: What is true of the row the menu was opened from.
        page: Which page to render.

    Returns:
        The ordered items for that page. Never empty: every non-root page
        carries a Back item even when it has nothing else to offer.
    """
    if page == "status":
        return _status_page(target)
    if page == "more":
        return _more_page(target)
    if page == "copy":
        return _copy_page(target)
    return _root_page(target)


def _root_page(
    target: ConversationMenuTarget,
) -> tuple[ConversationMenuItem, ...]:
    saved = target.is_saved
    if not target.favorites_available:
        favorite = ConversationMenuItem(
            action_id=ACTION_FAVORITE,
            label="Favourite",
            enabled=False,
            disabled_reason=_FAVORITES_UNAVAILABLE_REASON,
        )
    elif target.starred:
        favorite = ConversationMenuItem(
            action_id=ACTION_UNFAVORITE,
            label="Remove favourite",
            enabled=saved,
            disabled_reason="" if saved else _UNSAVED_REASON,
        )
    else:
        favorite = ConversationMenuItem(
            action_id=ACTION_FAVORITE,
            label="Favourite",
            enabled=saved,
            disabled_reason="" if saved else _UNSAVED_REASON,
        )

    if target.is_archived:
        archive = ConversationMenuItem(
            action_id=ACTION_UNARCHIVE,
            label="Unarchive",
            enabled=saved,
            disabled_reason="" if saved else _UNSAVED_REASON,
        )
    else:
        archive = ConversationMenuItem(
            action_id=ACTION_ARCHIVE,
            label="Archive",
            enabled=saved,
            disabled_reason="" if saved else _UNSAVED_REASON,
        )

    return (
        favorite,
        ConversationMenuItem(
            action_id=f"{ACTION_PAGE_PREFIX}status",
            label="Change status",
            enabled=saved,
            disabled_reason="" if saved else _UNSAVED_REASON,
            opens_page="status",
        ),
        archive,
        ConversationMenuItem(
            action_id=ACTION_RENAME,
            label="Rename…",
            enabled=saved,
            disabled_reason="" if saved else _UNSAVED_REASON,
        ),
        ConversationMenuItem(
            action_id=f"{ACTION_PAGE_PREFIX}copy",
            label="Copy as",
            opens_page="copy",
        ),
        ConversationMenuItem(
            action_id=f"{ACTION_PAGE_PREFIX}more",
            label="More",
            opens_page="more",
        ),
    )


def _status_page(
    target: ConversationMenuTarget,
) -> tuple[ConversationMenuItem, ...]:
    current = target.normalized_state
    items = [
        ConversationMenuItem(action_id=ACTION_BACK, label="‹ Back", opens_page="root")
    ]
    for state in CONVERSATION_STATES:
        items.append(
            ConversationMenuItem(
                action_id=f"{ACTION_SET_STATE_PREFIX}{state}",
                label=CONVERSATION_STATE_LABELS[state],
                # Choosing the state a row is already in is a no-op; mark it
                # rather than letting the user pick it and see nothing happen.
                enabled=state != current,
                disabled_reason=(
                    "This is the current status." if state == current else ""
                ),
                is_current=state == current,
            )
        )
    return tuple(items)


def _copy_page(
    target: ConversationMenuTarget,
) -> tuple[ConversationMenuItem, ...]:
    empty = not target.has_messages
    reason = "This chat has no messages yet." if empty else ""
    return (
        ConversationMenuItem(action_id=ACTION_BACK, label="‹ Back", opens_page="root"),
        ConversationMenuItem(
            action_id=ACTION_COPY_CLEAN,
            label="Clean markdown",
            enabled=not empty,
            disabled_reason=reason,
        ),
        ConversationMenuItem(
            action_id=ACTION_COPY_FULL,
            label="Full transcript",
            enabled=not empty,
            disabled_reason=reason,
        ),
        ConversationMenuItem(
            action_id=ACTION_SAVE_MARKDOWN,
            label="Save .md…",
            enabled=not empty,
            disabled_reason=reason,
        ),
    )


def _more_page(
    target: ConversationMenuTarget,
) -> tuple[ConversationMenuItem, ...]:
    saved = target.is_saved
    return (
        ConversationMenuItem(action_id=ACTION_BACK, label="‹ Back", opens_page="root"),
        ConversationMenuItem(
            action_id=ACTION_DELETE,
            label="Delete…",
            enabled=saved,
            disabled_reason="" if saved else _UNSAVED_REASON,
        ),
    )


def state_from_action(action_id: str) -> str | None:
    """Return the state a set-state action selects, or None.

    Args:
        action_id: An action id emitted by the menu.

    Returns:
        The canonical state name, or None when the action is not a state
        change. Unknown state names return None rather than being passed
        through to the database.
    """
    if not action_id.startswith(ACTION_SET_STATE_PREFIX):
        return None
    candidate = action_id[len(ACTION_SET_STATE_PREFIX) :]
    return candidate if candidate in CONVERSATION_STATES else None


def page_from_action(action_id: str) -> MenuPage | None:
    """Return the page an action navigates to, or None if it is a command."""
    if not action_id.startswith(ACTION_PAGE_PREFIX):
        return None
    candidate = action_id[len(ACTION_PAGE_PREFIX) :]
    return candidate if candidate in ("root", "status", "more", "copy") else None


def conversation_state_label(state: str | None) -> str:
    """Return the human label for a raw state token.

    Args:
        state: A raw state value, possibly unknown or blank.

    Returns:
        The friendly label, or the default state's label when unrecognised.
    """
    candidate = (state or "").strip().lower()
    return CONVERSATION_STATE_LABELS.get(
        candidate, CONVERSATION_STATE_LABELS[DEFAULT_CONVERSATION_STATE]
    )
