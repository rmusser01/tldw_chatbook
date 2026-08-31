"""Pure menu model for the Console workspace action menu (TASK-25712).

TASK-23200 gave the Context rail's conversation rows an asterisk that opens a
paged action menu; TASK-25709 made that menu dismissable everywhere. This
module extends the same pattern to the Workspaces tree's workspace nodes:
an asterisk that opens Activate / New chat / Rename… / RAG scope… / More ▸
Archive.

Everything here is pure: given what is true of one workspace, return the
items to paint. No DOM, no database, no service lookups -- so the menu's
shape, labelling and gating are testable without mounting an app.

Every command routes through code that already exists (activate_workspace_
id, the rename modal, the archive confirmation, the workspace scope picker,
and activate-then-create for "New chat" in a non-active workspace). RAG
scope is the one gated entry: the scope-picker seam is active-workspace
scoped, so a non-active workspace states that precondition rather than
silently editing the active workspace's scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MenuPage = Literal["root", "more"]

#: Action ids the menu can emit. Kept as plain strings so the widget, the
#: screen handler and the tests all name them the same way.
ACTION_ACTIVATE = "activate"
ACTION_NEW_CHAT = "new-chat"
ACTION_RENAME = "rename"
ACTION_RAG_SCOPE = "rag-scope"
ACTION_ARCHIVE = "archive"
ACTION_PAGE_PREFIX = "page:"
ACTION_BACK = "page:root"

_ALREADY_ACTIVE_REASON = "This workspace is already active."
_RAG_SCOPE_INACTIVE_REASON = "Activate this workspace to edit its RAG scope."


@dataclass(frozen=True, slots=True)
class WorkspaceMenuItem:
    """One painted row of the workspace action menu.

    Attributes:
        action_id: Stable identifier emitted when the item is chosen.
        label: Text shown to the user.
        enabled: Whether the item may be chosen.
        disabled_reason: Why it may not be, shown as a tooltip. Only ever set
            when ``enabled`` is False -- a disabled control with no stated
            precondition is the defect the sibling menu exists to remove.
        opens_page: The page this item navigates to, when it is a submenu
            opener rather than a command.
        is_current: Whether the item describes the workspace's present state,
            so the menu can mark it rather than implying choosing it does
            something.
    """

    action_id: str
    label: str
    enabled: bool = True
    disabled_reason: str = ""
    opens_page: MenuPage | None = None
    is_current: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceMenuTarget:
    """What the menu needs to know about the workspace it was opened from.

    Attributes:
        workspace_id: Registry id of the workspace.
        name: Current workspace name, for notifications.
        is_active: Whether this workspace is the active one. Drives the
            Activate entry's current-mark/gating and the RAG scope gate.
    """

    workspace_id: str
    name: str = ""
    is_active: bool = False


def build_workspace_menu(
    target: WorkspaceMenuTarget,
    page: MenuPage = "root",
) -> tuple[WorkspaceMenuItem, ...]:
    """Return the items to paint for one page of the menu.

    Args:
        target: What is true of the workspace the menu was opened from.
        page: Which page to render.

    Returns:
        The ordered items for that page. Never empty: the non-root page
        carries a Back item.
    """
    if page == "more":
        return _more_page()
    return _root_page(target)


def _root_page(
    target: WorkspaceMenuTarget,
) -> tuple[WorkspaceMenuItem, ...]:
    return (
        WorkspaceMenuItem(
            action_id=ACTION_ACTIVATE,
            label="Activate",
            enabled=not target.is_active,
            disabled_reason="" if not target.is_active else _ALREADY_ACTIVE_REASON,
            is_current=target.is_active,
        ),
        WorkspaceMenuItem(action_id=ACTION_NEW_CHAT, label="New chat"),
        WorkspaceMenuItem(action_id=ACTION_RENAME, label="Rename…"),
        WorkspaceMenuItem(
            action_id=ACTION_RAG_SCOPE,
            label="RAG scope…",
            enabled=target.is_active,
            disabled_reason="" if target.is_active else _RAG_SCOPE_INACTIVE_REASON,
        ),
        WorkspaceMenuItem(
            action_id=f"{ACTION_PAGE_PREFIX}more",
            label="More",
            opens_page="more",
        ),
    )


def _more_page() -> tuple[WorkspaceMenuItem, ...]:
    return (
        WorkspaceMenuItem(action_id=ACTION_BACK, label="‹ Back", opens_page="root"),
        WorkspaceMenuItem(action_id=ACTION_ARCHIVE, label="Archive"),
    )


def page_from_action(action_id: str) -> MenuPage | None:
    """Return the page an action navigates to, or None if it is a command.

    Args:
        action_id: An action id emitted by the menu.

    Returns:
        The page name, or None for commands and for page ids this menu
        cannot navigate to (the sibling conversation menu's pages do not
        leak in).
    """
    if not action_id.startswith(ACTION_PAGE_PREFIX):
        return None
    candidate = action_id[len(ACTION_PAGE_PREFIX) :]
    return candidate if candidate in ("root", "more") else None
