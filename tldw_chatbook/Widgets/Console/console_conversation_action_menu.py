"""Anchored action menu for one Console conversation row (TASK-23200).

Opened by the asterisk on a conversation row in the Context rail. Replaces a
star button that shipped disabled, reserved the full height of a multi-line
row, and was explained by the developer-facing line "Local stars unavailable"
-- dead vertical space, per the 2026-08-29 UX audit.

Structure follows ``ConsoleMessageMoreMenu``: an absolutely-positioned
``Vertical`` overlaying the screen, keyboard navigable, dismissed with Escape,
restoring focus to its opener. It differs in one way -- it PAGES. "Change
status" and "More" swap the menu's contents in place instead of stacking a
second popup, which keeps one widget, one lifecycle and one focus owner in a
27-column rail that has no room for cascading submenus.

TASK-25709 extends the dismissal contract (ADR-068) to the two paths the
original shipped without: a click outside the menu folds it from the screen's
per-press dismissal pass, and Escape reaches a stranded menu whose focus has
moved elsewhere. Both skip the opener focus-restore -- the dismissal cause
already tells us where the user wants focus. Reopening a row also awaits the
old menu's DOM detachment first, because ``remove()`` only schedules the
prune and a same-id remount over a still-attached menu crashes the app with
``DuplicateIds``.

The menu holds no policy: what to paint comes from
``Chat.console_conversation_actions``, which is pure and separately tested.
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

from tldw_chatbook.Chat.console_conversation_actions import (
    ConversationMenuTarget,
    MenuPage,
    build_conversation_menu,
    page_from_action,
)

if TYPE_CHECKING:  # pragma: no cover
    from textual.app import AwaitRemove
    from textual.screen import Screen

MENU_ID = "console-conversation-action-menu"
MENU_ITEM_PREFIX = "console-conversation-action-"

#: TASK-25709: every mounted menu registers itself here so the screen's
#: per-press dismissal checks never pay a full-DOM ``query`` walk (the
#: TASK-21119 rule). Mirrors ``console_message_more_menu``'s registry.
_LIVE_MENUS: "WeakSet[ConsoleConversationActionMenu]" = WeakSet()


def conversation_action_menus_on_screen(
    screen: "Screen[object]",
) -> list["ConsoleConversationActionMenu"]:
    """Return the conversation action menus currently attached to ``screen``.

    Args:
        screen: The screen whose mounted menus are wanted (usually the
            Console ``ChatScreen``).

    Returns:
        Every registry-registered menu still attached to that screen, in
        registry iteration order. Detached menus (``parent is None`` or no
        resolvable screen) are skipped.
    """
    menus: list[ConsoleConversationActionMenu] = []
    for menu in _LIVE_MENUS:
        if menu.parent is None:
            continue
        try:
            if menu.screen is screen:
                menus.append(menu)
        except NoScreen:
            continue
    return menus


class ConversationActionChosen(Message):
    """A command item was chosen from the conversation action menu."""

    def __init__(self, action_id: str, target: ConversationMenuTarget) -> None:
        """Create the message.

        Args:
            action_id: The chosen command's stable identifier. Never a
                navigation id -- page moves are handled inside the menu.
            target: The row the menu was opened from, captured at open time
                so a later rail refresh cannot redirect the action.
        """
        super().__init__()
        self.action_id = action_id
        self.target = target


class ConversationActionMenuDismissed(Message):
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
            opener_id: DOM id of the asterisk that opened the menu, so focus
                can be restored deterministically.
            restore_focus: Whether the screen handler should return focus to
                the opener.
        """
        super().__init__()
        self.opener_id = opener_id
        self.restore_focus = restore_focus


class ConsoleConversationActionMenu(Vertical):
    """Paged, keyboard-operable action menu bound to one conversation row."""

    can_focus = True

    #: Painted height of the menu's tallest page (the root: six one-row
    #: buttons plus the rounded border -- Copy as joined in TASK-25886),
    #: used by the screen's anchor clamping. Keep in lockstep with the
    #: root page's item count.
    ROOT_PAGE_HEIGHT = 8


    #: Anchoring clamps against this; the stylesheet below must declare the
    #: same width or viewport clamping drifts from what is painted (Qodo
    #: review, PR #2233). It CANNOT be interpolated: `css/build_css.py`
    #: lifts `BUNDLED_CSS` into the built stylesheet statically and rejects
    #: anything that is not a plain string literal. The two are pinned
    #: together by a test instead --
    #: Tests/UI/test_console_conversation_action_menu.py.
    MENU_WIDTH = 26

    BUNDLED_CSS = """
    ConsoleConversationActionMenu {
        position: absolute;
        overlay: screen;
        width: 26;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    ConsoleConversationActionMenu Button {
        width: 100%;
        height: 1 !important;
        min-height: 1 !important;
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        padding: 0 1 !important;
        text-align: left;
    }
    ConsoleConversationActionMenu Button:focus {
        outline: heavy $accent;
    }
    """

    def __init__(
        self,
        *,
        target: ConversationMenuTarget,
        opener_id: str,
        screen_x: int,
        screen_y: int,
    ) -> None:
        """Create a menu bound immutably to one row.

        Args:
            target: What is true of the conversation row, captured at open
                time.
            opener_id: DOM id of the asterisk that opened this menu.
            screen_x: Absolute column to anchor at.
            screen_y: Absolute row to anchor at.
        """
        super().__init__(id=MENU_ID)
        self._target = target
        self._opener_id = opener_id
        self._anchor = (screen_x, screen_y)
        self._page: MenuPage = "root"
        self._claimed = False
        #: Memoized ``remove()`` awaitable. Textual detaches a removed widget
        #: only when its Prune message is processed, so a same-id remount
        #: races an un-awaited removal into ``DuplicateIds``; every dismissal
        #: path funnels through this so ``remove()`` stays single-shot and
        #: the open path can await the detach it guards against.
        self._removal: AwaitRemove | None = None
        _LIVE_MENUS.add(self)

    @property
    def target(self) -> ConversationMenuTarget:
        """Return the row captured when the menu opened."""
        return self._target

    @property
    def opener_id(self) -> str:
        """Return the opener used for deterministic focus restoration."""
        return self._opener_id

    @property
    def page(self) -> MenuPage:
        """Return the page currently painted."""
        return self._page

    def compose(self) -> ComposeResult:
        """Build one button per entry on the menu's current page.

        Returns:
            One ``Button`` per item from ``build_conversation_menu`` for the
            page in view, in order. Submenu openers are suffixed with a
            disclosure glyph and the row's present state is bulleted, so a
            keyboard user can tell navigation from commands without colour.
        """
        for item in build_conversation_menu(self._target, self._page):
            label = f"{item.label} ▸" if item.opens_page and item.action_id.endswith(
                ("status", "more", "copy")
            ) else item.label
            if item.is_current:
                label = f"• {label}"
            button = Button(
                label,
                id=f"{MENU_ITEM_PREFIX}{_slug(item.action_id)}",
                disabled=not item.enabled,
            )
            button.console_action_id = item.action_id
            if item.disabled_reason:
                button.tooltip = item.disabled_reason
            yield button

    def on_mount(self) -> None:
        self.absolute_offset = Offset(*self._anchor)
        self._focus_first_enabled()

    def _focus_first_enabled(self) -> None:
        for button in self.query(Button):
            if not button.disabled:
                button.focus(scroll_visible=False)
                return

    async def _show_page(self, page: MenuPage) -> None:
        """Swap the menu's contents in place and re-seat focus."""
        self._page = page
        await self.recompose()
        self._focus_first_enabled()

    def dismiss_menu(self, *, restore_focus: bool = True) -> None:
        """Close the menu, optionally handing focus back to the opener.

        Args:
            restore_focus: Post the opener-restore message. False when the
                dismissal cause already expresses the user's focus intent
                (outside click, Escape from elsewhere, replacement by a new
                menu).
        """
        if self._claimed:
            return
        self._claimed = True
        if restore_focus:
            self.post_message(ConversationActionMenuDismissed(self._opener_id))
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
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            # Escape steps back out of a submenu before closing the menu, so
            # a user who opened "Change status" by accident is not thrown all
            # the way out of the row they were managing.
            if self._page != "root":
                self.run_worker(
                    self._show_page("root"),
                    group="console-conversation-action-menu-page",
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
        action_id = getattr(event.button, "console_action_id", None)
        if self._claimed or not isinstance(action_id, str):
            return
        page = page_from_action(action_id)
        if page is not None:
            self.run_worker(
                self._show_page(page),
                group="console-conversation-action-menu-page",
                exclusive=True,
            )
            return
        self._claimed = True
        self.post_message(ConversationActionChosen(action_id, self._target))
        self._detach()


def _slug(action_id: str) -> str:
    """Return a DOM-id-safe form of an action id."""
    return action_id.replace(":", "-")


def dismiss_conversation_action_menus(
    screen: Widget, *, restore_focus: bool = True
) -> int:
    """Remove any open conversation action menu on ``screen``.

    Args:
        screen: The screen (or any widget on it) to search.
        restore_focus: Passed through to each menu's dismissal; False for
            outside-click and stranded-Escape dismissals, where the user
            has already moved focus somewhere deliberate.

    Returns:
        How many menus were open and dismissed.
    """
    menus = conversation_action_menus_on_screen(screen.screen)
    for menu in menus:
        menu.dismiss_menu(restore_focus=restore_focus)
    return len(menus)
