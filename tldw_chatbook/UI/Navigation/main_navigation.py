"""Main navigation bar for screen-based navigation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Static
from textual.message import Message
from textual import on

from .shell_destinations import (
    SHELL_DESTINATION_ORDER,
    get_shell_destination,
    resolve_shell_route,
)

if TYPE_CHECKING:
    pass


#: Hotkey digits for the nav keyboard layer: ctrl+1..ctrl+9 select the first
#: nine destinations in SHELL_DESTINATION_ORDER and ctrl+0 selects the tenth.
#: The remaining destinations get F7/F8/F9 (see app.py SHELL_DESTINATION_FKEYS);
#: their labels carry the key name so the bar stays truthful.
NAV_HOTKEY_DIGITS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
NAV_FKEY_LABELS: tuple[str, ...] = ("F7", "F8", "F9")

#: F-002: the tab labels used to read "1 Home", implying a bare-digit key
#: while the actual binding (app.py SHELL_DESTINATION_HOTKEYS) is ctrl+digit.
#: The UP ARROWHEAD glyph (⌃, the macOS control convention) makes the label
#: honest at zero extra width per tab -- "⌃1" reads "ctrl+1".
NAV_HOTKEY_GLYPH = "⌃"


def nav_button_label(index: int, label: str) -> str:
    """Prefix a destination label with its hotkey affordance when it has one.

    The destination hotkey layer is ``ctrl+<digit>`` for the first ten
    destinations and F7/F8/F9 for the rest (see ``app.py``), so the label
    must say so: rendering a bare "1 Home" taught users a key that does
    nothing.

    Args:
        index: Position of the destination in SHELL_DESTINATION_ORDER.
        label: Compact destination label from the shell destination model.

    Returns:
        ``"⌃<digit> <label>"`` for the first ten destinations,
        ``"F<n> <label>"`` for the next ones with an F-key route,
        else the bare ``label``.
    """
    if 0 <= index < len(NAV_HOTKEY_DIGITS):
        return f"{NAV_HOTKEY_GLYPH}{NAV_HOTKEY_DIGITS[index]} {label}"
    fkey_index = index - len(NAV_HOTKEY_DIGITS)
    if 0 <= fkey_index < len(NAV_FKEY_LABELS):
        return f"{NAV_FKEY_LABELS[fkey_index]} {label}"
    return label


class NavigateToScreen(Message):
    """Message to request navigation to a specific screen."""

    def __init__(
        self, screen_name: str, screen_context: dict[str, object] | None = None
    ):
        super().__init__()
        self.screen_name = screen_name
        self.screen_context = dict(screen_context or {})


class NavigationButton(Button):
    """Navigation button that remains pressable when mounted in hidden chrome."""

    def __init__(self, *args, target_route: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_route = target_route

    def press(self):
        if not self.display:
            self.app.post_message(NavigateToScreen(self._target_route))
            return self
        return super().press()


class MainNavigationBar(Container):
    """
    Main navigation bar for the application.
    Replaces the tab-based navigation with screen-based navigation.
    """

    DEFAULT_CSS = """
    MainNavigationBar {
        height: 3;
        min-height: 3;
        width: 100%;
        dock: top;
        background: $background;
        border-bottom: solid $surface-lighten-2;
        layout: horizontal;
        overflow: hidden;
    }

    .main-nav {
        height: 100%;
        width: 1fr;
        layout: horizontal;
        align: left middle;
        padding: 0;
        margin: 0;
        overflow-x: auto;
        scrollbar-size-horizontal: 0;
    }

    .nav-button {
        margin: 0;
        padding: 0;
        min-width: 4;
        background: $surface-darken-1;
        border: solid $surface-lighten-2;
        height: 3;
        min-height: 3;
        content-align: center middle;
    }

    .nav-button:hover {
        background: $surface;
        border: solid $primary-lighten-1;
        text-style: bold;
    }

    .nav-button:focus {
        background: $surface;
        border: solid $primary;
        text-style: bold underline;
        color: $text;
        outline: none;
    }

    .nav-button.is-active {
        background: $primary-darken-1;
        border: solid $primary;
        text-style: bold;
        color: $text;
    }

    .nav-button.is-active:focus {
        background: $primary-darken-1;
        border: solid $primary;
        text-style: bold underline;
        color: $text;
        outline: none;
    }

    .nav-group-separator {
        margin: 0;
        padding: 0 1;
        color: $accent;
        width: auto;
        text-style: bold;
    }

    .nav-overflow-hint {
        width: auto;
        min-width: 0;
        padding: 0 1;
        height: 3;
        min-height: 3;
        content-align: center middle;
        color: $text-muted;
        background: transparent;
        border: none;
    }

    .nav-overflow-hint:hover {
        color: $text;
        background: $surface;
        text-style: bold;
    }
    """

    def __init__(self, active: str = "chat", active_route: str | None = None, **kwargs):
        """Initialize the navigation bar with destination and route state.

        Args:
            active: Current screen or destination used to highlight the owning
                top-level destination.
            active_route: Canonical active route when the highlighted
                destination owns a subroute. When omitted, `active` is used.
            **kwargs: Additional Textual container keyword arguments.
        """
        super().__init__(**kwargs)
        resolved_active = resolve_shell_route(active)
        self.active_destination_id = resolved_active.destination_id
        self.active_route = resolve_shell_route(active_route or active).canonical_route
        self.active_screen = self.active_destination_id

    def compose(self) -> ComposeResult:
        """Compose the navigation bar from master-shell destination metadata."""
        # Left overflow indicator: visible only when the strip is scrolled
        # right, so off-screen destinations on the left stay discoverable.
        left_hint = Static("‹", id="nav-overflow-hint-left", classes="nav-overflow-hint")
        left_hint.tooltip = "More destinations to the left — scroll back"
        left_hint.display = False
        yield left_hint
        with Horizontal(id="nav-destination-strip", classes="main-nav"):
            for index, destination in enumerate(SHELL_DESTINATION_ORDER):
                button = NavigationButton(
                    nav_button_label(index, destination.label),
                    id=f"nav-{destination.destination_id}",
                    classes="nav-button ascii-nav-tab",
                    tooltip=destination.tooltip,
                    target_route=destination.primary_route,
                )
                if destination.destination_id == self.active_destination_id:
                    button.add_class("is-active")
                yield button
        # Docked outside the scrollable strip so the affordance stays at the
        # right edge exactly when the destinations overflow. F-001: it is a
        # real control now, not just a hint -- pressing it pages the strip
        # right (wrapping at the far end) so every destination stays
        # mouse/keyboard-reachable at narrow widths, and it hides when all
        # destinations fit instead of crowding the edge. When the bar has
        # the cells to spare, its label spells out the F-key legend for the
        # overflow destinations (`_HINT_WIDE`); otherwise "More ›".
        overflow_hint = Button(
            "More ›",
            id="nav-overflow-hint",
            classes="nav-overflow-hint",
            compact=True,
            tooltip="Show more destinations (Ctrl+P for the full list)",
        )
        # Hidden until `_sync_overflow_hint` (post-layout) knows whether the
        # strip actually overflows -- never a flash of affordance chrome on
        # a bar where every destination fits.
        overflow_hint.display = False
        yield overflow_hint

    def on_mount(self) -> None:
        """Scroll the initially active destination's button into view."""
        # Order matters: settle the overflow indicators (which change the
        # strip's width) before aligning the active button.
        self.call_after_refresh(self._update_overflow_hints)
        self.call_after_refresh(self._scroll_active_destination_into_view)
        self.set_interval(0.5, self._update_overflow_hints)

    #: Overflow hint label by available width: the full F-key legend when the
    #: bar is wide enough to spare the cells, the compact affordance otherwise.
    _HINT_WIDE = "F7 Lab · F8 Logs · F9 Settings · More ›"
    _HINT_NARROW = "More ›"

    def _update_overflow_hints(self) -> None:
        """Toggle the ‹ / More indicators and their text from real state."""
        # Skip while the screen/tab is inactive so hidden tabs burn no CPU.
        if not self.is_attached or not self.screen.is_active:
            return
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
            left_hint = self.query_one("#nav-overflow-hint-left", Static)
            right_hint = self.query_one("#nav-overflow-hint", Button)
        except Exception:
            return
        try:
            max_scroll_x = strip.max_scroll_x
            scroll_x = strip.scroll_x
        except Exception:
            return
        # Left hint tracks position (more destinations hidden on the left);
        # the right hint marks that overflow exists at all — the palette
        # offers every destination regardless of scroll position.
        left_hint.display = scroll_x > 0
        new_right = max_scroll_x > 0
        right_hint.display = new_right
        if new_right:
            wide_text = self.size.width >= 110
            right_hint.label = self._HINT_WIDE if wide_text else self._HINT_NARROW
        # Layout settles asynchronously (hint toggles change the strip's
        # width, fonts finish, etc.), so keep the active destination pinned
        # every tick instead of only when a hint changed state — the call is
        # idempotent and cheap.
        self.call_after_refresh(self._scroll_active_destination_into_view)
        self.call_after_refresh(self._sync_overflow_hint)

    def on_resize(self) -> None:
        """Re-sync the overflow affordance when the bar's width changes.

        The strip's overflow (``max_scroll_x``) is a function of the bar's
        rendered width, so every resize re-evaluates whether the
        "More ›" control shows (F-001).
        """
        self._sync_overflow_hint()

    def _sync_overflow_hint(self) -> None:
        """Show the "More ›" affordance exactly when the strip overflows."""
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
            hint = self.query_one("#nav-overflow-hint", Button)
        except Exception:
            return
        hint.display = strip.max_scroll_x > 0

    @on(Button.Pressed, "#nav-overflow-hint")
    def _page_destination_overflow(self, event: Button.Pressed) -> None:
        """Page the strip right; at the far end, wrap back to the start."""
        event.stop()
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
        except Exception:
            return
        max_scroll = strip.max_scroll_x
        if max_scroll <= 0:
            return
        if strip.scroll_offset.x >= max_scroll - 1:
            # Already at the far end: wrap so the control keeps working.
            strip.scroll_to(x=0, animate=False)
            return
        visible_width = max(strip.scrollable_content_region.width, 1)
        # NOTE: `scrollable_content_region` reads like "the full scrollable
        # content" but is the VISIBLE viewport (region minus gutter and
        # scrollbar) -- the correct page increment. The virtual content is
        # wider by exactly max_scroll (PR #1322 review).
        strip.scroll_to(
            x=min(strip.scroll_offset.x + visible_width, max_scroll),
            animate=False,
        )

    def _scroll_active_destination_into_view(self) -> None:
        """Bring the active destination's button into the strip's visible scroll window."""
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
            button = strip.query_one(
                f"#nav-{self.active_destination_id}", NavigationButton
            )
        except Exception:
            return
        try:
            strip.scroll_to_widget(button, animate=False)
        except Exception:
            return

    @on(Button.Pressed, ".nav-button")
    def handle_navigation(self, event: Button.Pressed) -> None:
        """Handle navigation button clicks."""
        self._activate_navigation_button(event.button)

    def on_click(self, event) -> None:
        """Route clicks on a tab's visible border back to the owning button."""
        try:
            clicked_widget = self.app.get_widget_at(event.screen_x, event.screen_y)[0]
        except Exception:
            return
        if clicked_widget is not self and not (
            isinstance(clicked_widget, Horizontal)
            and clicked_widget.has_class("main-nav")
        ):
            return

        click_point = (event.screen_x, event.screen_y)
        for button in self.query(NavigationButton):
            if button.region.contains_point(click_point):
                if self._activate_navigation_button(button):
                    event.stop()
                return

    def _activate_navigation_button(self, button: Button) -> bool:
        """Activate a navigation button and return whether navigation was requested."""
        button_id = button.id
        if not button_id:
            return False

        destination_id = button_id.replace("nav-", "")
        destination = get_shell_destination(destination_id)
        screen_name = destination.primary_route

        # A destination-owned subroute may highlight the same top-level destination;
        # clicking the destination should still return to its primary route.
        if (
            destination.destination_id == self.active_destination_id
            and screen_name == self.active_route
        ):
            return False

        # Update active state
        for nav_button in self.query(".nav-button"):
            nav_button.remove_class("is-active")
        button.add_class("is-active")
        self.active_destination_id = destination.destination_id
        self.active_route = screen_name
        self.active_screen = self.active_destination_id
        self.call_after_refresh(self._scroll_active_destination_into_view)

        # Post navigation message to app
        self.post_message(NavigateToScreen(screen_name))

        logger.info(f"Navigation requested to screen: {screen_name}")
        return True
