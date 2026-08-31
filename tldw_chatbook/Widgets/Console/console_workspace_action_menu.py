"""Anchored action menu for one Console workspace tree node (TASK-25710).

Opened by the trailing asterisk on a workspace row in the Workspaces tree
(pointer) or the ``m`` binding (keyboard). Mirrors
``ConsoleConversationActionMenu``: an absolutely-positioned ``Vertical``
overlaying the screen, keyboard navigable, paged in place ("More" swaps
contents rather than cascading -- a 27-column rail has no room for a second
popup), dismissed with Escape and by the screen's outside-click pass.

Disposal follows the TASK-25709 contract exactly: a constructor-registered
``WeakSet`` registry backs ``workspace_action_menus_on_screen`` so the
screen's per-press dismissal checks pay no DOM walk; ``remove()`` is
memoized single-shot with an awaitable detach so a same-id remount can
never race an un-awaited prune into ``DuplicateIds``; and the opener
focus-restore is skipped for outside-click / stranded-Escape dismissals
because those causes already express where the user wants focus.

The menu holds no policy: what to paint comes from
``Chat.console_workspace_actions``, which is pure and separately tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakSet

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.dom import NoScreen
from textual.events import Key
from textual.geometry import Offset
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button

from tldw_chatbook.Chat.console_workspace_actions import (
    WorkspaceMenuTarget,
    build_workspace_menu,
    page_from_action,
)

if TYPE_CHECKING:  # pragma: no cover
    from textual.app import AwaitRemove
    from textual.screen import Screen

MENU_ID = "console-workspace-action-menu"
MENU_ITEM_PREFIX = "console-workspace-action-"

#: TASK-25710: every mounted menu registers itself here so the screen's
#: per-press dismissal checks never pay a full-DOM ``query`` walk (the
#: TASK-21119 rule). Mirrors the conversation menu's registry.
_LIVE_MENUS: "WeakSet[ConsoleWorkspaceActionMenu]" = WeakSet()


def workspace_action_menus_on_screen(
    screen: "Screen[object]",
) -> list["ConsoleWorkspaceActionMenu"]:
    """Return the workspace action menus currently attached to ``screen``.

    Args:
        screen: The screen whose mounted menus are wanted (usually the
            Console ``ChatScreen``).

    Returns:
        Every registry-registered menu still attached to that screen, in
        registry iteration order. Detached menus (``parent is None`` or no
        resolvable screen) are skipped.
    """
    menus: list[ConsoleWorkspaceActionMenu] = []
    for menu in _LIVE_MENUS:
        if menu.parent is None:
            continue
        try:
            if menu.screen is screen:
                menus.append(menu)
        except NoScreen:
            continue
    return menus


class WorkspaceActionChosen(Message):
    """A command item was chosen from the workspace action menu."""

    def __init__(self, action_id: str, target: WorkspaceMenuTarget) -> None:
        """Create the message.

        Args:
            action_id: The chosen command's stable identifier. Never a
                navigation id -- page moves are handled inside the menu.
            target: The workspace the menu was opened from, captured at open
                time so a later tree refresh cannot redirect the action.
        """
        super().__init__()
        self.action_id = action_id
        self.target = target


class WorkspaceActionMenuDismissed(Message):
    """The menu closed without choosing a command.

    ``restore_focus`` is False for dismissals where the user has already
    expressed a focus intent elsewhere -- an outside click (Textual moves
    focus to the clicked widget before the press reaches the screen) or an
    Escape issued while focus sits outside the menu. The opener-restore is
    only correct when the menu itself held focus (its own Escape path).
    """

    def __init__(self, opener_id: str, *, restore_focus: bool = True) -> None:
        """Create the message.

        Args:
            opener_id: DOM id of the opener (the tree) that opened the
                menu, so focus can be restored deterministically.
            restore_focus: Whether the screen handler should return focus
                to the opener.
        """
        super().__init__()
        self.opener_id = opener_id
        self.restore_focus = restore_focus


class ConsoleWorkspaceActionMenu(Vertical):
    """Paged, keyboard-operable action menu bound to one workspace row."""

    can_focus = True

    #: Painted height of the menu's tallest page (the root: five one-row
    #: buttons plus the rounded border), used by the screen's anchor
    #: clamping. Keep in lockstep with the root page's item count.
    ROOT_PAGE_HEIGHT = 7


    #: Anchoring clamps against this; the stylesheet below must declare the
    #: same width or viewport clamping drifts from what is painted. Pinned
    #: together by test the same way the conversation menu's pair is.
    MENU_WIDTH = 26

    BUNDLED_CSS = """
    ConsoleWorkspaceActionMenu {
        position: absolute;
        overlay: screen;
        width: 26;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    ConsoleWorkspaceActionMenu Button {
        width: 100%;
        height: 1 !important;
        min-height: 1 !important;
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        padding: 0 1 !important;
        text-align: left;
    }
    ConsoleWorkspaceActionMenu Button:focus {
        outline: heavy $accent;
    }
    """

    def __init__(
        self,
        *,
        target: WorkspaceMenuTarget,
        opener_id: str,
        screen_x: int,
        screen_y: int,
    ) -> None:
        """Create a menu bound immutably to one workspace.

        Args:
            target: What is true of the workspace, captured at open time.
            opener_id: DOM id of the opener used for deterministic focus
                restoration (the Workspaces tree).
            screen_x: Absolute column to anchor at.
            screen_y: Absolute row to anchor at.
        """
        super().__init__(id=MENU_ID)
        self._target = target
        self._opener_id = opener_id
        self._anchor = (screen_x, screen_y)
        self._page = "root"
        self._claimed = False
        #: Memoized ``remove()`` awaitable; see the module docstring for why
        #: every dismissal path funnels through it.
        self._removal: AwaitRemove | None = None
        _LIVE_MENUS.add(self)

    @property
    def target(self) -> WorkspaceMenuTarget:
        """Return the workspace captured when the menu opened."""
        return self._target

    @property
    def opener_id(self) -> str:
        """Return the opener used for deterministic focus restoration."""
        return self._opener_id

    @property
    def page(self) -> str:
        """Return the page currently painted."""
        return self._page

    def compose(self) -> ComposeResult:
        """Build one ``Button`` per entry on the menu's current page."""
        for item in build_workspace_menu(self._target, self._page):
            label = (
                f"{item.label} ▸"
                if item.opens_page and item.action_id.endswith("more")
                else item.label
            )
            if item.is_current:
                label = f"• {label}"
            button = Button(
                label,
                id=f"{MENU_ITEM_PREFIX}{_slug(item.action_id)}",
                disabled=not item.enabled,
            )
            button.workspace_action_id = item.action_id
            if item.disabled_reason:
                button.tooltip = item.disabled_reason
            yield button

    def on_mount(self) -> None:
        """Anchor at the captured screen cell and seat focus on row one."""
        self.absolute_offset = Offset(*self._anchor)
        self._focus_first_enabled()

    def _focus_first_enabled(self) -> None:
        """Focus the first enabled entry, skipping gated ones.

        Focusing a disabled button drops focus entirely, which would strand
        keyboard navigation, so disabled entries are skipped rather than
        visited.
        """
        for button in self.query(Button):
            if not button.disabled:
                button.focus(scroll_visible=False)
                return

    async def _show_page(self, page: str) -> None:
        """Swap the menu's contents in place and re-seat focus."""
        self._page = page
        await self.recompose()
        self._focus_first_enabled()

    def dismiss_menu(self, *, restore_focus: bool = True) -> None:
        """Close the menu, optionally handing focus back to the opener.

        Args:
            restore_focus: Post the opener-restore message. False when the
                dismissal cause already expresses the user's focus intent
                (outside click, Escape from elsewhere, replacement).
        """
        if self._claimed:
            return
        self._claimed = True
        if restore_focus:
            self.post_message(WorkspaceActionMenuDismissed(self._opener_id))
        self._detach()

    async def await_detachment(self) -> None:
        """Claim and fully detach this menu, awaiting the DOM removal.

        The replace-on-reopen path calls this before mounting a new menu:
        ``remove()`` only posts a Prune message, and mounting the same DOM
        id while the old menu is still attached raises Textual's app-fatal
        ``DuplicateIds``. No dismissal message is posted -- the new menu
        takes focus anyway.
        """
        self._claimed = True
        await self._detach()

    def _detach(self) -> "AwaitRemove":
        """Return the memoized single removal awaitable for this menu."""
        if self._removal is None:
            self._removal = self.remove()
        return self._removal

    def on_key(self, event: Key) -> None:
        """Handle the menu's keyboard contract: cycle, page, or dismiss.

        Args:
            event: The key event delivered while the menu (or one of its
                buttons) holds focus.
        """
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            # Escape steps back out of a submenu before closing the menu,
            # mirroring the conversation menu's contract.
            if self._page != "root":
                self.run_worker(
                    self._show_page("root"),
                    group="console-workspace-action-menu-page",
                    exclusive=True,
                )
                return
            self.dismiss_menu()
            return
        if event.key not in {"up", "down", "tab", "shift+tab"}:
            return
        event.stop()
        event.prevent_default()
        buttons = [button for button in self.query(Button) if not button.disabled]
        if not buttons:
            return
        focused = next((button for button in buttons if button.has_focus), buttons[0])
        step = 1 if event.key in {"down", "tab"} else -1
        buttons[(buttons.index(focused) + step) % len(buttons)].focus()

    @on(Button.Pressed)
    def _choose(self, event: Button.Pressed) -> None:
        event.stop()
        action_id = getattr(event.button, "workspace_action_id", None)
        if self._claimed or not isinstance(action_id, str):
            return
        page = page_from_action(action_id)
        if page is not None:
            self.run_worker(
                self._show_page(page),
                group="console-workspace-action-menu-page",
                exclusive=True,
            )
            return
        self._claimed = True
        self.post_message(WorkspaceActionChosen(action_id, self._target))
        self._detach()


def _slug(action_id: str) -> str:
    """Return a DOM-id-safe form of an action id."""
    return action_id.replace(":", "-")


def dismiss_workspace_action_menus(
    screen: Widget, *, restore_focus: bool = True
) -> int:
    """Remove any open workspace action menu on ``screen``.

    Args:
        screen: The screen (or any widget on it) to search.
        restore_focus: Passed through to each menu's dismissal; False for
            outside-click and stranded-Escape dismissals, where the user
            has already moved focus somewhere deliberate.

    Returns:
        How many menus were open and dismissed.
    """
    menus = workspace_action_menus_on_screen(screen.screen)
    for menu in menus:
        menu.dismiss_menu(restore_focus=restore_focus)
    return len(menus)
