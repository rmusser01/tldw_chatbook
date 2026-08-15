"""Floating selection-action menu (console selection phase 1).

Mounted on the owning SCREEN at an absolute offset (the same anchoring
mechanism Textual's own tooltips use); NOT a ModalScreen (modals are
layer-centered and cannot anchor at a cell) and NOT a docked transcript
child. Live-spike round 3: the previous ``dock: top`` + ``styles.offset``
combination painted the menu translated by the offset while clipping it to
the un-translated dock slot -- the user saw one button, and hit-tests used
the un-translated region so the other buttons were unclickable. Mounting on
the screen with ``absolute_offset`` folds the position into the widget's
region, so paint, clipping, and hit-testing all agree.

Escape and click-outside dismiss with no side effects (task-16211 modal
contract, recorded by ADR-066). Keyboard navigation: up/down cycles the
action buttons, Enter activates the focused one, Escape closes.
"""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Click, Key
from textual.geometry import Offset
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button


class ConsoleSelectionQuoteRequested(Message):
    """Bubbled when the user adds the active selection as a quote to the chat."""

    def __init__(self, quote: str) -> None:
        super().__init__()
        self.quote = quote


class ConsoleSideChatRequested(Message):
    """Bubbled when the user opens an ephemeral side chat about a selection.

    Console selection phase 2: the transcript posts this after the menu's
    "More Details" (``mode="more-details"``) or "Ask in Side Chat"
    (``mode="ask"``) action; the owning ``ChatScreen`` resolves config +
    gateway and pushes ``ConsoleSideChatModal``.
    """

    MODE_MORE_DETAILS = "more-details"
    MODE_ASK = "ask"

    def __init__(self, quote: str, mode: str) -> None:
        super().__init__()
        self.quote = quote
        self.mode = mode


class ConsoleSelectionMenu(Vertical):
    """Floating stacked menu anchored at the selection release cell."""

    can_focus = True

    DEFAULT_CSS = """
    ConsoleSelectionMenu {
        position: absolute;
        width: auto;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "dismiss", show=False)]

    class AddToChat(Message):
        """User chose 'Add to chat' for the active selection."""

    class MoreDetails(Message):
        """User chose 'More Details' (auto-send side chat) for the selection."""

    class AskInSideChat(Message):
        """User chose 'Ask in Side Chat' (freeform) for the selection."""

    def __init__(
        self,
        *,
        screen_x: int,
        screen_y: int,
        has_add_to_chat: bool = True,
        owner: Widget | None = None,
    ) -> None:
        """Anchor the menu at screen coordinates.

        Args:
            screen_x: X anchor (in cells) in SCREEN coordinates -- the
                release column of the drag.
            screen_y: Y anchor in screen coordinates; the caller passes
                ``release_row + 1`` so the menu sits just below the release
                cell.
            has_add_to_chat: Whether to offer the "Add to chat" action.
            owner: The widget that owns the selection lifecycle (the
                ``ConsoleTranscript``). The menu mounts on the SCREEN, so
                its action/dismissal messages are POSTED DIRECTLY to the
                owner instead of bubbling -- screen-level bubbling would
                never reach the transcript's handlers. ``None`` falls back
                to normal bubbling (bare test harnesses).
        """
        super().__init__(id="console-selection-menu", classes="console-selection-menu")
        self._anchor = (screen_x, screen_y)
        self._owner = owner
        self._has_add_to_chat = has_add_to_chat
        #: Widget holding focus before the menu grabbed it (captured in
        #: ``on_mount`` BEFORE focusing the first button); ``None`` =
        #: nothing was focused (or the capture raced teardown), so unmount
        #: falls back to the composer.
        self._previous_focus: Widget | None = None

    def compose(self):
        if self._has_add_to_chat:
            yield Button("Add to chat", id="console-selection-add-to-chat", variant="primary")
        yield Button("More Details", id="console-selection-more-details")
        yield Button("Ask in Side Chat", id="console-selection-ask-side-chat")

    def on_mount(self) -> None:
        # Capture the pre-mount focus holder BEFORE focusing a menu button:
        # a drag that started from a focused transcript must return focus
        # there on dismissal, not be pulled into the composer (final review).
        try:
            self._previous_focus = self.screen.focused
        except Exception:  # noqa: BLE001 - capture is best-effort during odd teardown
            self._previous_focus = None
        self.absolute_offset = Offset(*self._anchor)
        # Only this widget knows its real extent (border + padding +
        # buttons); pull the anchor back inside the screen once layout has
        # measured it, or a release near the bottom edge anchors the lower
        # buttons off-screen (pilot/OutOfBounds; real terminal: unreachable).
        self.call_after_refresh(self._clamp_within_screen)
        buttons = list(self.query(Button))
        if buttons:
            # scroll_visible=False: focusing must not scroll the screen to
            # the menu (it shifted the transcript out from under the
            # selection when the menu mounted).
            buttons[0].focus(scroll_visible=False)
        else:
            self.focus(scroll_visible=False)

    def _clamp_within_screen(self) -> None:
        """Shift the anchor so the measured menu fits the screen."""
        if self.parent is None or not self.is_attached:
            return
        region = self.region
        screen_size = self.screen.size
        if not region:
            return
        shift_x = max(0, region.right - screen_size.width)
        shift_y = max(0, region.bottom - screen_size.height)
        if not shift_x and not shift_y:
            return
        x, y = self._anchor
        self._anchor = (max(0, x - shift_x), max(0, y - shift_y))
        self.absolute_offset = Offset(*self._anchor)

    def on_key(self, event: Key) -> None:
        """Keyboard navigation: arrows cycle actions; Escape closes.

        Escape is handled here (not only via BINDINGS) because ancestor
        widgets -- ``ConsoleTranscript.on_key`` stops Escape for its
        clear-selection action during bubbling, before binding dispatch
        would consult this menu's BINDINGS. The focused widget's
        ``on_key`` runs first in the bubble chain (a menu button or the
        menu itself), so handling the key here is what actually fires in
        the real transcript context.
        """
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_dismiss()
            return
        if event.key in ("up", "down"):
            event.stop()
            event.prevent_default()
            buttons = [b for b in self.query(Button) if b.display]
            if not buttons:
                return
            focused = next((b for b in buttons if b.has_focus), buttons[0])
            index = buttons.index(focused)
            step = 1 if event.key == "down" else -1
            buttons[(index + step) % len(buttons)].focus()

    def _post(self, message: Message) -> None:
        (self._owner if self._owner is not None else self).post_message(message)

    @on(Button.Pressed, "#console-selection-add-to-chat")
    def _add_to_chat(self) -> None:
        self._post(self.AddToChat())

    @on(Button.Pressed, "#console-selection-more-details")
    def _more_details(self) -> None:
        self._post(self.MoreDetails())

    @on(Button.Pressed, "#console-selection-ask-side-chat")
    def _ask_side_chat(self) -> None:
        self._post(self.AskInSideChat())

    class Dismissed(Message):
        """Escape dismissal: the owning transcript clears the selection UI."""

    def action_dismiss(self) -> None:
        self._post(self.Dismissed())
        self.remove()

    def _on_click(self, event: Click) -> None:
        event.stop()  # clicks inside the menu must not clear anything
        # A click on the menu that did NOT land on one of its action
        # buttons (border/padding/label areas) is a popover dismissal:
        # clear the selection UI so the next click reaches the row.
        node = event.control
        while node is not None and node is not self:
            if isinstance(node, Button):
                return
            node = node.parent
        self._post(self.Dismissed())
        self.remove()

    def _on_unmount(self) -> None:
        self._restore_previous_focus()

    def _restore_previous_focus(self) -> None:
        """Return focus to the widget that held it before the menu mounted.

        ``on_mount`` captured ``screen.focused`` before the menu grabbed
        focus; every dismissal path (escape, click-outside, add-to-chat
        cleanup) funnels through removal, so unmount is the single restore
        seam. The captured widget is restored when it is still mounted on
        the same screen -- a drag that started from a focused transcript
        must return focus there on Escape, not be pulled into the
        composer. Otherwise focus falls back to the console composer.
        Skips quietly when the fallback finds no composer (bare
        transcript/test harnesses) or when the screen is already gone
        during teardown (``self.screen`` raises NoScreen).
        """
        try:
            screen = self.screen
        except Exception:  # noqa: BLE001 - focus restore is best-effort during teardown
            return
        previous = self._previous_focus
        if previous is not None and previous is not self and previous.is_mounted:
            try:
                if previous.screen is screen:
                    previous.focus(scroll_visible=False)
                    return
            except Exception:  # noqa: BLE001, S110 - previous detached during teardown
                pass
        for composer in screen.query("#console-native-composer"):
            composer.focus(scroll_visible=False)
            return
