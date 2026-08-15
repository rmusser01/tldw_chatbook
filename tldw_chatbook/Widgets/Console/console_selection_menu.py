"""Floating selection-action menu (console selection phase 1).

Mounted inside ``ConsoleTranscript``; NOT a ModalScreen (modals are
layer-centered and cannot anchor at a cell). Escape and click-outside
dismiss with no side effects (task-16211 modal contract, recorded by
ADR-066). Docked out of the transcript's scroll flow so ``styles.offset``
positions it relative to the transcript container rather than the end of
the scroll content.
"""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Click, Key
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
        dock: top;
        layer: overlay;
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

    def __init__(self, *, local_x: int, local_y: int, has_add_to_chat: bool = True) -> None:
        """Anchor the menu at transcript-local coordinates.

        Args:
            local_x: X offset (in cells) relative to the owning transcript.
            local_y: Y offset relative to the owning transcript; the caller
                passes ``release_row + 1`` so the menu sits just below the
                release cell.
            has_add_to_chat: Whether to offer the "Add to chat" action.
        """
        super().__init__(id="console-selection-menu", classes="console-selection-menu")
        self._anchor = (local_x, local_y)
        self._has_add_to_chat = has_add_to_chat
        #: Widget holding focus before the menu grabbed it (captured in
        #: ``on_mount`` BEFORE ``self.focus()``); ``None`` = nothing was
        #: focused (or the capture raced teardown), so unmount falls back
        #: to the composer.
        self._previous_focus: Widget | None = None

    def compose(self):
        if self._has_add_to_chat:
            yield Button("Add to chat", id="console-selection-add-to-chat", variant="primary")
        yield Button("More Details", id="console-selection-more-details")
        yield Button("Ask in Side Chat", id="console-selection-ask-side-chat")

    def on_mount(self) -> None:
        x, y = self._anchor
        self.styles.offset = (x, y)
        # Capture the pre-mount focus holder BEFORE self.focus(): a drag
        # that started from a focused transcript must return focus there
        # on dismissal, not be pulled into the composer (final review).
        try:
            self._previous_focus = self.screen.focused
        except Exception:  # noqa: BLE001 - capture is best-effort during odd teardown
            self._previous_focus = None
        self.focus()
        # The transcript pre-clamps the anchor with a small fixed margin, but
        # only this widget knows its real extent (border + padding + button);
        # correct the offset once the layout has measured it.
        self.call_after_refresh(self._clamp_within_parent)

    def on_key(self, event: Key) -> None:
        """Dismiss on Escape before the transcript can claim the key.

        The menu is a child of ``ConsoleTranscript``, whose own ``on_key``
        stops Escape for its clear-selection action during bubbling -- before
        binding dispatch would ever consult this menu's BINDINGS. The focused
        widget's ``on_key`` runs first in the bubble chain, so handling the
        key here is what actually fires in the real transcript context.
        """
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_dismiss()

    def _clamp_within_parent(self) -> None:
        """Pull the measured menu back inside the owning transcript.

        A release near the transcript's right (or bottom) edge anchored the
        menu so its action button overhung the edge -- unreachable. Runs
        after the first layout (via ``call_after_refresh``) so the real cell
        extents are known; the anchor's non-negative offsets mean only the
        right/bottom edges can overflow.
        """
        parent = self.parent
        if parent is None or not self.is_attached:
            return
        region = self.region
        parent_region = parent.region
        if not region or not parent_region:
            return
        shift_x = max(0, region.right - parent_region.right)
        shift_y = max(0, region.bottom - parent_region.bottom)
        if not shift_x and not shift_y:
            return
        x, y = self._anchor
        self.styles.offset = (max(0, x - shift_x), max(0, y - shift_y))

    @on(Button.Pressed, "#console-selection-add-to-chat")
    def _add_to_chat(self) -> None:
        self.post_message(self.AddToChat())

    @on(Button.Pressed, "#console-selection-more-details")
    def _more_details(self) -> None:
        self.post_message(self.MoreDetails())

    @on(Button.Pressed, "#console-selection-ask-side-chat")
    def _ask_side_chat(self) -> None:
        self.post_message(self.AskInSideChat())

    class Dismissed(Message):
        """Escape dismissal: the owning transcript clears the selection UI."""

    def action_dismiss(self) -> None:
        self.post_message(self.Dismissed())
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
        self.post_message(self.Dismissed())
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
                    previous.focus()
                    return
            except Exception:  # noqa: BLE001, S110 - previous detached during teardown
                pass
        for composer in screen.query("#console-native-composer"):
            composer.focus()
            return
