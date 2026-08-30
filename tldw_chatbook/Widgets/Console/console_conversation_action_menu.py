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

The menu holds no policy: what to paint comes from
``Chat.console_conversation_actions``, which is pure and separately tested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
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
    pass

MENU_ID = "console-conversation-action-menu"
MENU_ITEM_PREFIX = "console-conversation-action-"


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
    """The menu closed without choosing a command."""

    def __init__(self, opener_id: str) -> None:
        """Create the message.

        Args:
            opener_id: DOM id of the asterisk that opened the menu, so focus
                can be restored deterministically.
        """
        super().__init__()
        self.opener_id = opener_id


class ConsoleConversationActionMenu(Vertical):
    """Paged, keyboard-operable action menu bound to one conversation row."""

    can_focus = True

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
                ("status", "more")
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

    def dismiss_menu(self) -> None:
        """Close the menu and hand focus back to the opener."""
        if self._claimed:
            return
        self._claimed = True
        self.post_message(ConversationActionMenuDismissed(self._opener_id))
        self.remove()

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
        self.remove()


def _slug(action_id: str) -> str:
    """Return a DOM-id-safe form of an action id."""
    return action_id.replace(":", "-")


def dismiss_conversation_action_menus(screen: Widget) -> None:
    """Remove any open conversation action menu on ``screen``.

    Args:
        screen: The screen (or any widget on it) to search.
    """
    for menu in screen.query(ConsoleConversationActionMenu):
        menu.dismiss_menu()
