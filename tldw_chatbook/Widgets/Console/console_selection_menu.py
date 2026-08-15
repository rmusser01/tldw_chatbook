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
from textual.widgets import Button


class ConsoleSelectionQuoteRequested(Message):
    """Bubbled when the user adds the active selection as a quote to the chat."""

    def __init__(self, quote: str) -> None:
        super().__init__()
        self.quote = quote


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

    def compose(self):
        if self._has_add_to_chat:
            yield Button("Add to chat", id="console-selection-add-to-chat", variant="primary")

    def on_mount(self) -> None:
        x, y = self._anchor
        self.styles.offset = (x, y)
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

    def action_dismiss(self) -> None:
        self.remove()

    def _on_click(self, event: Click) -> None:
        event.stop()  # clicks inside the menu must not clear anything

    def _on_unmount(self) -> None:
        self._restore_composer_focus()

    def _restore_composer_focus(self) -> None:
        """Return focus to the console composer after the menu goes away.

        The menu grabs focus on mount; every dismissal path (escape,
        click-outside, add-to-chat cleanup) funnels through removal, so
        unmount is the single restore seam. Skips quietly when no composer
        is mounted (bare transcript/test harnesses) or when the screen is
        already gone during teardown (``self.screen`` raises NoScreen).
        """
        try:
            matches = self.screen.query("#console-native-composer")
        except Exception:  # noqa: BLE001 - focus restore is best-effort during teardown
            return
        for composer in matches:
            composer.focus()
            return
