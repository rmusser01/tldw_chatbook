"""Captured-target overflow menu for selected Console messages."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from weakref import WeakSet

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.dom import NoScreen
from textual.events import Click, Key, Resize
from textual.geometry import Offset
from textual.widget import Widget
from textual.widgets import Button

from tldw_chatbook.Chat.console_message_actions import ConsoleMessageAction

if TYPE_CHECKING:  # pragma: no cover
    from textual.screen import Screen


_LIVE_MORE_MENUS: "WeakSet[ConsoleMessageMoreMenu]" = WeakSet()


def message_more_menus_on_screen(
    screen: "Screen[object]",
) -> list["ConsoleMessageMoreMenu"]:
    """Return attached message overflow menus owned by ``screen``."""
    menus: list[ConsoleMessageMoreMenu] = []
    for menu in _LIVE_MORE_MENUS:
        if menu.parent is None:
            continue
        try:
            if menu.screen is screen:
                menus.append(menu)
        except NoScreen:
            continue
    return menus


def dismiss_message_more_menus(
    menus: Iterable["ConsoleMessageMoreMenu"],
) -> None:
    """Dismiss overflow menus and restore focus to each captured opener."""
    for menu in menus:
        menu.dismiss_menu()


class ConsoleMessageMoreMenu(Vertical):
    """Small labelled menu bound immutably to one message and action set."""

    can_focus = True
    MENU_WIDTH = 24

    BUNDLED_CSS = """
    ConsoleMessageMoreMenu {
        position: absolute;
        overlay: screen;
        width: 24;
        height: auto;
        max-height: 100%;
        overflow-y: auto;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    ConsoleMessageMoreMenu Button {
        width: 100%;
        height: 1 !important;
        min-height: 1 !important;
        border: none !important;
        border-top: none !important;
        border-bottom: none !important;
        padding: 0 1 !important;
        text-align: left;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    def __init__(
        self,
        *,
        message_id: str,
        actions: Iterable[ConsoleMessageAction],
        owner: Widget,
        opener_button_id: str,
        screen_x: int,
        screen_y: int,
    ) -> None:
        super().__init__(id="console-message-more-menu")
        self._message_id = message_id
        self._actions = tuple(actions)
        self._action_ids = tuple(action.action_id for action in self._actions)
        self._opener_button_id = opener_button_id
        self._owner = owner
        self._anchor = (screen_x, screen_y)
        self._claimed = False
        self._layout_refresh_signal = None
        self._owner_bounds_size: tuple[int, int] | None = None
        _LIVE_MORE_MENUS.add(self)

    @property
    def message_id(self) -> str:
        """Return the message target captured when the menu opened."""
        return self._message_id

    @property
    def actions(self) -> tuple[ConsoleMessageAction, ...]:
        """Return the captured, ordered overflow action snapshot."""
        return self._actions

    @property
    def action_ids(self) -> tuple[str, ...]:
        """Return the captured action IDs used to reject stale presses."""
        return self._action_ids

    @property
    def opener_button_id(self) -> str:
        """Return the exact opener used for deterministic focus fallback."""
        return self._opener_button_id

    def compose(self) -> ComposeResult:
        for action in self.actions:
            button = Button(
                action.label,
                id=f"console-message-more-{action.action_id}",
                disabled=not action.enabled,
            )
            button.console_action_id = action.action_id
            button.console_message_id = self.message_id
            if action.disabled_reason:
                button.tooltip = action.disabled_reason
            yield button

    def on_mount(self) -> None:
        self.absolute_offset = Offset(*self._anchor)
        self._layout_refresh_signal = self.screen.screen_layout_refresh_signal
        self._layout_refresh_signal.subscribe(self, self._on_screen_layout_refresh)
        self.call_after_refresh(self._clamp_within_owner)
        buttons = [button for button in self.query(Button) if not button.disabled]
        if buttons:
            buttons[0].focus(scroll_visible=False)

    def on_resize(self, _event: Resize) -> None:
        self.call_after_refresh(self._clamp_within_owner)

    def _on_screen_layout_refresh(self, _screen: Screen) -> None:
        self._clamp_within_owner()

    def _clamp_within_owner(self) -> None:
        """Keep the scrollable action viewport inside the transcript."""
        if not self.is_mounted:
            return
        bounds = self._owner.region.intersection(self.screen.region)
        if not bounds:
            bounds = self.screen.region
        bounds_size = (bounds.width, bounds.height)
        if self._owner_bounds_size != bounds_size:
            self._owner_bounds_size = bounds_size
            self.styles.max_width = bounds.width
            self.styles.max_height = bounds.height
        width = min(self.outer_size.width, bounds.width)
        height = min(self.outer_size.height, bounds.height)
        offset = Offset(
            max(bounds.x, min(self._anchor[0], bounds.right - width)),
            max(bounds.y, min(self._anchor[1], bounds.bottom - height)),
        )
        if self.absolute_offset != offset:
            self.absolute_offset = offset
            self.refresh(layout=True)
        focused = self.screen.focused
        if (
            isinstance(focused, Button)
            and focused.parent is self
            and not self.region.contains_region(focused.region)
        ):
            focused.scroll_visible(animate=False, immediate=True)

    @property
    def owner(self) -> Widget:
        """Return the transcript that owns lifecycle and focus restoration."""
        return self._owner

    def dismiss_menu(self, *, restore_focus: bool = True) -> None:
        """Close the menu without dispatching an action; single-shot.

        Same keyword as the row menus' ``dismiss_menu``, so a caller can fold
        any focus-taking Console menu the same way.

        Args:
            restore_focus: Hand focus back to the captured opener. False when
                focus has already moved on deliberately -- a typed letter took
                it to the draft (TASK-33622.3).
        """
        if getattr(self, "_pruning", False):
            return
        if restore_focus:
            self._owner._restore_message_action_focus(self._opener_button_id)
        self.remove()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self._owner.run_worker(
                self._owner.dismiss_message_more_menu(),
                group="console-message-more-menu",
                exclusive=True,
            )
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
        if (
            self._claimed
            or not isinstance(action_id, str)
            or action_id not in self.action_ids
        ):
            return
        self._claimed = True
        for button in self.query(Button):
            button.disabled = True
        self._owner.call_later(
            self._owner.choose_captured_message_more_action,
            self.message_id,
            action_id,
            opener_button_id=self.opener_button_id,
        )

    def _on_click(self, event: Click) -> None:
        event.stop()

    def _on_unmount(self) -> None:
        if self._layout_refresh_signal is not None:
            self._layout_refresh_signal.unsubscribe(self)
            self._layout_refresh_signal = None
        _LIVE_MORE_MENUS.discard(self)
