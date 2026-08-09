"""Main navigation bar for screen-based navigation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.events import DescendantFocus
from textual.geometry import Region
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


def _straddles_viewport(region: Region, viewport: Region) -> bool:
    """True if `region` is PARTIALLY (not fully in, not fully out) within
    `viewport` on the horizontal axis -- the geometric definition of
    task-3200's "partially-clipped label" straddle, used by
    `MainNavigationBar._ghost_clipped_buttons` to decide which button to
    ghost (and, paired with `disabled`, which button Tab may land on --
    see that method's docstring for why straddle-rejection was NOT put
    directly on a `NavigationButton.allow_focus` override: an earlier
    attempt at exactly that broke Tab cycling and was reverted).
    """
    if region.width <= 0 or viewport.width <= 0:
        return False
    return (
        region.x < viewport.x < region.right
        or region.x < viewport.right < region.right
    )


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
        # `display` is never set False by anything in this module today
        # (task-3200's clip-straddle handling ghosts via CSS + `disabled`
        # instead -- see `_ghost_clipped_buttons`), so this branch is
        # currently unreachable in practice; kept as-is since some other
        # future caller could still legitimately hide a nav button. A
        # ghosted (task-3200) button does NOT hit this branch either way:
        # it stays `display=True`, so `super().press()` runs, and
        # `Button.press()` already no-ops when `self.disabled` is set --
        # no separate `disabled` check is needed here.
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

    /* task-3200: a destination button whose render straddles the strip's
       scroll viewport edge gets this class instead of `display: none` --
       it keeps its normal layout box (so `max_scroll_x` is unaffected --
       see `_ghost_clipped_buttons`), but every visible surface matches
       the bar's own background, so whatever
       sliver of it happens to be on-screen reads as empty space rather
       than a mid-word cut ("Watchlists" -> "Watc"). Listed after (and so
       overriding) hover/focus/active so a straddling button can never
       flash real content via those states. `_ghost_clipped_buttons` pairs
       this class with `disabled = True` (review finding: a ghosted
       button was otherwise still fully interactive -- Tab-reachable with
       no visible focus ring, clickable/Enter-navigable while invisible),
       so `opacity`/`text-opacity` are pinned to 100% here to cancel
       Textual's own built-in disabled-dimming (`App`'s global
       `*:disabled:can-focus { opacity: 0.7; }` and, every `Button`'s own
       default variant class, `Button.-style-default:disabled {
       text-opacity: 0.6; }`) -- without this the ghosted color blends
       toward transparent instead of staying pixel-exact on `$background`.
       `!important` is required, not decorative: `Button.-style-default:
       disabled` carries an extra TYPE-selector component (`Button`) that
       `.nav-button.nav-button-clip-ghost:disabled` (two classes, no type)
       does not, so by standard CSS specificity comparison Button's own
       rule wins even though it is declared earlier and matches every
       `NavigationButton` (a `Button` subclass that never opts out of the
       default `-style-default` class) -- confirmed by direct tmux capture
       (`capture-pane -e`) BEFORE `!important` was added: the ghosted
       "Watc" fragment rendered as foreground `38;2;43;43;43` against
       background `48;2;16;16;16` -- visibly distinct, not the intended
       pixel-exact match. IMPORTANT CAVEAT: this `!important` alone is
       NOT sufficient in the real running app -- `tldw_chatbook/css/
       components/_buttons.tcss`'s app-wide `Button:disabled { opacity:
       50%; }`, loaded via `App.CSS_PATH`, outranks ANY widget
       `DEFAULT_CSS` rule as a TIER, `!important` or not (Textual gives
       `CSS_PATH` stylesheets priority over widget `DEFAULT_CSS`
       independent of specificity). The rule that actually wins live is
       `tldw_chatbook/css/components/_navigation.tcss`'s
       `.nav-button.nav-button-clip-ghost:disabled` override, in the SAME
       `CSS_PATH` tier -- a known, precedented pattern in this codebase
       (see `Tests/UI/test_mcp_inspector.py`'s
       `test_disabled_action_buttons_stay_legible_with_bundled_css` for
       the MCP inspector's identical fix). This `!important` block stays
       as defense-in-depth for the `DEFAULT_CSS` tier itself (e.g.
       against some future widget-level rule), not as the actual fix for
       the app-wide `Button:disabled` opacity. */
    .nav-button.nav-button-clip-ghost,
    .nav-button.nav-button-clip-ghost:hover,
    .nav-button.nav-button-clip-ghost:focus,
    .nav-button.nav-button-clip-ghost.is-active,
    .nav-button.nav-button-clip-ghost.is-active:focus,
    .nav-button.nav-button-clip-ghost:disabled {
        background: $background !important;
        border: solid $background !important;
        color: $background !important;
        text-style: none;
        opacity: 100% !important;
        text-opacity: 100% !important;
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

    .nav-overflow-hint:focus {
        background: $surface;
        color: $text;
        text-style: bold underline;
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
        # Docked outside the scrollable strip so the affordance stays visible
        # at the right edge exactly when the destinations overflow -- and ONLY
        # then: when every destination fits (e.g. 140 cols, where the buttons
        # need 134 cells) the button hides so it never re-clips the strip
        # itself (NV-01, TASK-2154.21; the old 14-cell static hint is what
        # truncated "Settings" to "Set").
        overflow_hint = Button(
            "More ▾", id="nav-overflow-hint", classes="nav-overflow-hint", compact=True
        )
        overflow_hint.tooltip = "All destinations"
        # Hidden until `_refresh_overflow_hint_visibility` (post-layout) knows
        # whether the strip actually overflows -- never a flash of affordance
        # chrome on a bar where every destination fits.
        overflow_hint.display = False
        yield overflow_hint

    def on_mount(self) -> None:
        """Scroll the initially active destination's button into view."""
        # Order matters: settle the overflow indicators (which change the
        # strip's width) before aligning the active button.
        self.call_after_refresh(self._update_overflow_hints)
        self.call_after_refresh(self._scroll_active_destination_into_view)
        self.set_interval(0.5, self._update_overflow_hints)
        # The strip's virtual_size only settles once its scrollable content
        # has laid out, which is AFTER the first call_after_refresh tick (an
        # early check measures a zero-width region and pins the hint visible).
        # Re-check on short timers and on every later resize.
        self.set_timer(0.05, self._refresh_overflow_hint_visibility)
        self.set_timer(0.25, self._refresh_overflow_hint_visibility)

    def _update_overflow_hints(self) -> None:
        """Toggle the ‹ indicator from real scroll state."""
        # Skip while the screen/tab is inactive so hidden tabs burn no CPU.
        if not self.is_attached or not self.screen.is_active:
            return
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
            left_hint = self.query_one("#nav-overflow-hint-left", Static)
        except Exception:
            return
        try:
            scroll_x = strip.scroll_x
        except Exception:
            return
        # Left hint tracks position (more destinations hidden on the left);
        # the right affordance's visibility is width-driven, re-evaluated by
        # `_refresh_overflow_hint_visibility` (NV-01 reclaimable-space math).
        left_hint.display = scroll_x > 0
        self._refresh_overflow_hint_visibility()
        # Layout settles asynchronously (hint toggles change the strip's
        # width, fonts finish, etc.), so keep the active destination pinned
        # every tick instead of only when a hint changed state — the call is
        # idempotent and cheap.
        self.call_after_refresh(self._scroll_active_destination_into_view)

    def on_resize(self) -> None:
        """Re-sync the overflow affordance when the bar's width changes.

        The strip's overflow is a function of the bar's rendered width, so
        every resize re-evaluates whether the "More ▾" control shows, and
        (task-3200) re-measures which buttons clip -- a resize can
        un-straddle (or newly straddle) a destination at the edge.

        Both calls below are deferred (`call_after_refresh`); the first,
        `_update_overflow_hints`, itself unconditionally ends by chaining
        into `_scroll_active_destination_into_view` (re-scroll, THEN
        ghost-check via `_ghost_clipped_buttons`) -- so a resize reaches
        the ghost re-check through that existing chain rather than calling
        `_ghost_clipped_buttons` directly. That matters for the same
        reason it did before this rename: a screen transition fires
        several resizes while content is still settling, and ghost-
        checking against whatever `scroll_x` happens to be at THIS
        particular resize, without first re-scrolling for the CURRENT
        viewport width, can check a stale position. If a later resize is
        the last one that actually lands before the widget goes idle, its
        ghost state persists uncorrected (live-reproduced: navigating to a
        destination that needs scrolling left a straddling neighbor
        un-ghosted because the final settle's ghost-check ran against an
        intermediate, not the final, scroll_x).
        """
        self.call_after_refresh(self._update_overflow_hints)
        self.call_after_refresh(self._refresh_overflow_hint_visibility)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Re-scroll to the newly-focused button, then ghost-check
        (task-3200 review finding).

        `Widget.focus()` defaults to `scroll_visible=True`: Tab landing
        on an off-screen-but-focusable button auto-scrolls the strip to
        reveal it, same as a resize -- but that happens via Textual's OWN
        internal, independently-scheduled `call_later(scroll_to_center,
        ...)` (`Screen.set_focus`), which nothing here waits on or drives
        -- so the ghost/disabled state stayed pinned to whatever scroll
        position was current when it was last computed. Live-reproduced:
        Tab-cycling through the bar eventually left a genuinely
        straddling button un-ghosted AND un-disabled, reachable by Tab.

        Rather than react to (and race) Textual's own scroll, this drives
        an equivalent scroll itself -- `strip.scroll_to_widget(event.
        widget)`, the same call `_scroll_active_destination_into_view`
        already makes for the active destination -- then chains the
        ghost-check the same proven way every other trigger in this
        class does. A first attempt reacted to `DescendantFocus` by only
        re-running the ghost-check (not re-scrolling): it raced Textual's
        scroll closely enough to occasionally starve it entirely,
        observed live as Tab getting stuck oscillating between two
        buttons instead of cycling the whole bar; a second attempt made
        the button reject focus based on a LIVE geometry check instead of
        `disabled` (`NavigationButton.allow_focus`) -- also reverted, it
        broke Tab in the same "stuck between two buttons" way, most
        likely because Textual's own focus-chain walk snapshots
        candidates' regions once per Tab press rather than re-measuring
        as scroll changes, so evaluating straddle status against
        already-stale regions rejected far more candidates than
        intended. Doing OWN scroll first (idempotent alongside Textual's,
        since `scroll_to_center` no-ops once the widget is already fully
        visible) sidesteps both failure modes.
        """
        widget = event.widget
        if not isinstance(widget, NavigationButton):
            return
        self.call_after_refresh(self._scroll_to_focused_then_ghost_check, widget)

    def _scroll_to_focused_then_ghost_check(self, widget: "NavigationButton") -> None:
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
            strip.scroll_to_widget(widget, animate=False)
        except Exception:
            return
        self.call_after_refresh(self._ghost_clipped_buttons)

    def _refresh_overflow_hint_visibility(self) -> None:
        """Show the overflow menu button only when the strip actually clips.

        Hiding the button widens the strip by the button's own width, so the
        check adds that reclaimable space back when the button is displayed --
        otherwise the hint would only ever hide at widths where the strip
        already fits WITH the hint still docked (>=145 cols instead of 134).
        """
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
            hint = self.query_one("#nav-overflow-hint", Button)
        except Exception:
            return
        strip_virtual = strip.virtual_size.width
        strip_width = strip.region.width
        if strip_virtual == 0 or strip_width == 0:
            # Layout has not settled yet (a zero-width virtual size would
            # wrongly read as "everything fits" and hide the button); keep
            # the current state and let the next timer/resize re-check.
            return
        reclaimable = hint.outer_size.width if hint.display else 0
        hint.display = strip_virtual > strip_width + reclaimable

    def _scroll_active_destination_into_view(self) -> None:
        """Bring the active destination's button into the strip's visible
        scroll window, then re-check for a straddling neighbor."""
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
        self.call_after_refresh(self._ghost_clipped_buttons)

    def _ghost_clipped_buttons(self) -> None:
        """Visually blank (never partially render) any nav button whose
        current position straddles either edge of the strip's scroll
        viewport (task-3200: a destination label must never be cut
        mid-word, e.g. "Watchlists" -> "Watc").

        This is purely cosmetic (`.nav-button-clip-ghost`, defined in
        `DEFAULT_CSS`, colors every surface to match the bar's own
        background) -- it does NOT touch `display`, so the button keeps
        its normal layout box. That distinction is what makes this safe
        to call from every settle path (mount, resize, activating a
        destination) with no iteration and no ordering constraints:
        earlier attempts hid straddling buttons via `display: none`,
        which shrinks the strip's virtual size and therefore
        `max_scroll_x` -- that broke reachability of destinations further
        along the strip (nothing left to scroll to) and, when the active
        destination sat near the end of the strip, cascaded into hiding
        nearly every other destination (removing a LEADING straddler reflows
        everything after it, including the active button itself, which
        can newly expose a different straddler; for some
        active-destination/viewport-width combinations no scroll offset
        can make the active destination fully visible AND land flush on a
        button boundary, so that cascade never converged on a clean
        state). Ghosting sidesteps all of it: geometry is never touched,
        so there is nothing to iterate or race.

        A ghosted button is also made ``disabled`` (review finding: color
        alone left it fully interactive -- Tab could land on an invisible
        button with no focus ring, and a click or Enter would silently
        navigate). Textual's own `disabled` semantics do exactly what's
        needed here for free: `Widget.focusable` excludes disabled
        widgets from Tab order, `watch_disabled` blurs it immediately if
        it currently holds focus, and `Button.press()` already no-ops
        when `self.disabled` is set -- `NavigationButton.press()`'s own
        `display`-hidden-chrome branch doesn't need touching, since that
        check runs first and only ever applies when `display` is False,
        which ghosting never sets. The active destination is exempt from
        `should_ghost` below, so it is always re-enabled in the same pass
        that clears its ghost class -- never a visible-but-disabled tab.
        """
        if not self.is_attached or not self.screen.is_active:
            return
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
        except Exception:
            return
        strip_region = strip.region
        if strip_region.width <= 0:
            return
        active_id = f"nav-{self.active_destination_id}"
        # Review-round regression fix: the button that currently HOLDS
        # keyboard focus must never be ghosted/disabled, exactly like the
        # active destination is exempt below. `_scroll_to_focused_then_
        # ghost_check` (called from `on_descendant_focus`) always scrolls
        # this button fully into view immediately before this method runs
        # -- but `scroll_to_widget` targets a fractional `scroll_x` and a
        # subsequent layout pass can still measure this button's region as
        # straddling by a hair (rounding at the cell boundary). Ghosting a
        # focused button sets `disabled = True`, and Textual's own
        # `watch_disabled` immediately blurs a focused widget when it
        # becomes disabled -- observed live via a direct reproduction: Tab
        # landing on `nav-lab` triggered exactly this, blurring it before
        # the next Tab press, which then computed "next focusable after
        # nothing" and wrapped all the way back to the first button in the
        # bar -- Tab cycling never escaped the nav bar within the
        # `test_tab_order_reaches_visible_primary_action` budget (a real,
        # reproduced regression, not a hypothetical). Exempting the
        # focused button the same way the active destination is exempted
        # fixes it without touching scroll or focus-rejection logic (the
        # two approaches already reverted elsewhere in this file for
        # breaking Tab in a similar way).
        focused = self.screen.focused
        focused_id = focused.id if isinstance(focused, NavigationButton) else None
        for button in strip.query(NavigationButton):
            region = button.region
            if region.width <= 0:
                continue
            straddles = _straddles_viewport(region, strip_region)
            # The active destination is guaranteed fully visible by
            # `_scroll_active_destination_into_view` and must never be
            # ghosted, even transiently; same guarantee for whichever
            # button currently holds focus (see docstring above).
            should_ghost = (
                straddles and button.id != active_id and button.id != focused_id
            )
            button.set_class(should_ghost, "nav-button-clip-ghost")
            button.disabled = should_ghost

    @on(Button.Pressed, "#nav-overflow-hint")
    def handle_overflow_hint(self, event: Button.Pressed) -> None:
        """Open the overflow menu listing every destination (NV-01)."""
        event.stop()
        # Local import: nav_overflow_menu imports NavigateToScreen/nav_button_label
        # from this module, so a top-level import here would be circular.
        from .nav_overflow_menu import NavOverflowMenu

        self.app.push_screen(
            NavOverflowMenu(active_destination_id=self.active_destination_id)
        )

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

    def restore_active(self, route: str) -> None:
        """Reset the highlight to ``route``, undoing a click's optimism.

        A click activates its destination here before the navigation worker
        has run; when that navigation fails, the app calls this with the
        route still on the screen stack so the highlight matches reality and
        the failed destination stays re-clickable — the already-active check
        in `_activate_navigation_button` would otherwise swallow every retry
        (task-2720, observed live: one transient error made a tab
        unreachable for the rest of the session).

        Args:
            route: Screen or destination route actually on the screen stack.
        """
        resolved = resolve_shell_route(route)
        self.active_destination_id = resolved.destination_id
        self.active_route = resolved.canonical_route
        self.active_screen = self.active_destination_id
        for nav_button in self.query(".nav-button"):
            nav_button.set_class(
                nav_button.id == f"nav-{self.active_destination_id}",
                "is-active",
            )
        # The restored destination may have been clip-ghosted (task-3200)
        # while it wasn't active -- re-scroll so it's guaranteed visible
        # now that it is, same as a normal click-driven activation.
        self.call_after_refresh(self._scroll_active_destination_into_view)
