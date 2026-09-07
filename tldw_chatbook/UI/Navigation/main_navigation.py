"""Main navigation bar for screen-based navigation."""

from typing import TYPE_CHECKING, Any, Callable
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.events import DescendantFocus
from textual.geometry import Region
from textual.css.query import NoMatches
from textual.widgets import Button, Static
from textual.message import Message
from textual import on

from .shell_destinations import (
    SHELL_DESTINATION_ORDER,
    SHELL_DESTINATION_SHORTCUTS,
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


#: F-002: the tab labels used to read "1 Home", implying a bare-digit key
#: while the actual binding (app.py SHELL_DESTINATION_HOTKEYS) is ctrl+digit.
#: The UP ARROWHEAD glyph (⌃, the macOS control convention) makes the label
#: honest at zero extra width per tab -- "⌃1" reads "ctrl+1".
NAV_HOTKEY_GLYPH = "⌃"


def nav_button_label(destination_id: str, label: str) -> str:
    """Prefix a destination label with its hotkey affordance when it has one.

    Shortcut ownership comes from the stable destination-ID mapping, so the
    label cannot drift when navigation order changes.

    Args:
        destination_id: Stable shell destination ID.
        label: Compact destination label from the shell destination model.

    Returns:
        The shortcut-prefixed label.
    """
    shortcut = SHELL_DESTINATION_SHORTCUTS[destination_id]
    if shortcut.startswith("ctrl+"):
        return f"{NAV_HOTKEY_GLYPH}{shortcut.removeprefix('ctrl+')} {label}"
    return f"{shortcut.upper()} {label}"


#: task-31385: where the app remembers how many Console interrupt rounds
#: are pending, so a navigation bar composed AFTER a round armed (every
#: screen composes its own bar) still shows the badge on mount.
CONSOLE_ATTENTION_ATTR = "_console_pending_interrupts"
#: The pending-interrupt badge on the Console nav button; the same glyph
#: the session tabs use for "needs approval".
CONSOLE_ATTENTION_GLYPH = "◆"


def set_console_attention(app: Any, pending: int) -> None:
    """UI THREAD: remember ``pending`` on the app and repaint every mounted bar.

    Args:
        app: The running app (a test double is fine; it only needs to
            accept the attribute).
        pending: How many Console interrupt rounds are registered now.
    """
    setattr(app, CONSOLE_ATTENTION_ATTR, int(pending))
    stack = getattr(app, "screen_stack", None)
    if not isinstance(stack, (list, tuple)):
        return
    for screen in stack:
        for bar in screen.query(MainNavigationBar):
            bar.apply_console_attention()


class NavigateToScreen(Message):
    """Message to request navigation to a specific screen."""

    def __init__(
        self,
        screen_name: str,
        screen_context: dict[str, object] | None = None,
        *,
        on_completion: Callable[[bool], None] | None = None,
        require_character_inspection_admission: bool = False,
        is_current: Callable[[], bool] | None = None,
        on_commit_started: Callable[[], bool] | None = None,
    ):
        super().__init__()
        self.screen_name = screen_name
        self.screen_context = dict(screen_context or {})
        self._on_completion = on_completion
        self.require_character_inspection_admission = require_character_inspection_admission
        self.is_current = is_current
        self.on_commit_started = on_commit_started
        self._completion_reported = False
        self._target_ownership_committed = False

    @property
    def target_ownership_committed(self) -> bool:
        """Whether the destination has synchronously taken the Textual stack."""
        return self._target_ownership_committed

    def commit_target_ownership(self) -> None:
        """Commit successful navigation when the destination owns the stack."""
        if self._target_ownership_committed:
            return
        self._target_ownership_committed = True
        self.report_completion(True)

    def report_completion(self, succeeded: bool) -> None:
        """Settle one optional source callback after the route reaches a terminal state."""
        if self._completion_reported:
            return
        self._completion_reported = True
        callback = self._on_completion
        self._on_completion = None
        if callback is None:
            return
        try:
            callback(bool(succeeded))
        except Exception:
            logger.debug("Navigation completion callback failed.", exc_info=True)


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

    BUNDLED_CSS = """
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
       the app-wide `Button:disabled` opacity.

       REVIEW ROUND 4 (task-3225), THE GEOMETRY BUG THIS RULE USED TO HAVE:
       this rule declared `border: solid $background !important`, which was
       NOT geometry-neutral. Textual's own `Button.-style-default` gives a
       nav button `border: none; border-top: tall ...; border-bottom: tall
       ...` -- i.e. ZERO horizontal border cells -- so switching it to a
       four-edge `solid` border added one cell of border on the left AND the
       right: a ghosted button measured 2 cells WIDER than the same button
       un-ghosted (directly measured: `#nav-workflows` 14 -> 16). Because
       the strip is a horizontal layout, that reflowed every button after
       it 2 cells to the right, which could push a previously fully-visible
       button (including the deliberately-focused one) into a straddling
       position AFTER the ghost pass that was supposed to settle the strip
       -- and nothing re-checks after a ghost pass, because the whole point
       of ghosting over `display: none` is that it is supposed to leave
       geometry untouched. That is what produced the ~0.3s "drift-back"
       task-3225 filed: the corrective scroll landed, then the ghost pass's
       own reflow undid it one layout pass later.

       The fix is that this rule now declares NO box-model property at all
       -- colors and text style only -- so a ghosted button's box is,
       by construction, byte-identical to its un-ghosted box in whichever
       CSS tier is actually winning (bare widget harness: `border-top/
       bottom: tall`, zero horizontal cells; the real app: `Button {
       border: none }` from `components/_buttons.tcss`, also zero). A
       ghosted button therefore cannot reflow its neighbours, which is the
       invariant this whole approach was chosen over `display: none` for
       in the first place.

       `visibility: hidden` was tried first (it is the Textual primitive
       that means "invisible but still occupies space") and REJECTED:
       `Widget.region` returns an EMPTY region for an invisible widget
       (measured: `outer_size` stays 14, `region.width` drops to 0), and
       `_ghost_clipped_buttons` decides straddle from `button.region` and
       skips any button with `region.width <= 0` -- so a once-ghosted
       button could never be measured again, i.e. never un-ghosted. Left
       out rather than reworking every geometry read onto a different
       property. */
    .nav-button.nav-button-clip-ghost,
    .nav-button.nav-button-clip-ghost:hover,
    .nav-button.nav-button-clip-ghost:focus,
    .nav-button.nav-button-clip-ghost.is-active,
    .nav-button.nav-button-clip-ghost.is-active:focus,
    .nav-button.nav-button-clip-ghost:disabled {
        background: $background !important;
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
        # Review round 2: distinguishes a genuine, later Tab press from
        # Textual's own initial `AUTO_FOCUS` landing on the first nav
        # button the instant the bar mounts -- see `_mark_mount_settled`
        # and `on_descendant_focus`.
        self._mount_settled = False
        # task-4024: armed by `_mark_mount_settled` when it runs before the
        # screen's automatic focus placement has landed anywhere (the
        # post-recompose ordering) -- the next `DescendantFocus` is then
        # consumed as that automatic landing, never recorded as deliberate.
        self._settle_after_next_focus = False
        self._deliberate_focus_id: str | None = None
        # task-15473: cheap fingerprint of everything `_update_overflow_hints`'
        # downstream work (hint toggles, `_recenter_strip` scheduling, the
        # ghost-check it chains into) depends on -- when the periodic 0.5s
        # tick recomputes the same signature it stored last pass, nothing
        # has moved and the whole pipeline is skipped. `None` never equals a
        # real signature tuple, so the very first pass after mount always
        # runs in full.
        self._overflow_signature: tuple[object, ...] | None = None

    def compose(self) -> ComposeResult:
        """Compose the navigation bar from master-shell destination metadata."""
        # Left overflow indicator: visible only when the strip is scrolled
        # right, so off-screen destinations on the left stay discoverable.
        left_hint = Static("‹", id="nav-overflow-hint-left", classes="nav-overflow-hint")
        left_hint.tooltip = "More destinations to the left — scroll back"
        left_hint.display = False
        yield left_hint
        with Horizontal(id="nav-destination-strip", classes="main-nav"):
            for destination in SHELL_DESTINATION_ORDER:
                button = NavigationButton(
                    nav_button_label(
                        destination.destination_id,
                        destination.label,
                    ),
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

    def apply_console_attention(self) -> None:
        """task-31385: badge the Console button while interrupt rounds are pending."""
        try:
            button = self.query_one("#nav-console", Button)
        except NoMatches:
            return
        base = nav_button_label("console", get_shell_destination("console").label)
        pending = int(getattr(self.app, CONSOLE_ATTENTION_ATTR, 0) or 0)
        label = f"{base} {CONSOLE_ATTENTION_GLYPH}" if pending else base
        if str(button.label) != label:
            button.label = label

    def on_mount(self) -> None:
        """Scroll the initially active destination's button into view."""
        self.apply_console_attention()
        # Order matters: settle the overflow indicators (which change the
        # strip's width) before aligning the active button.
        self.call_after_refresh(self._update_overflow_hints)
        self.call_after_refresh(self._recenter_strip)
        # Review round 2: marks the point after which a `DescendantFocus`
        # is trusted as a genuine, later Tab press rather than Textual's
        # own initial `AUTO_FOCUS` (which lands on the first focusable
        # widget -- empirically confirmed to fire between `on_mount`'s own
        # synchronous return and this bar's first `call_after_refresh`
        # callback, i.e. strictly BEFORE this marker runs). See
        # `on_descendant_focus` and `_recenter_strip`. Because
        # `_deliberate_focus_id` can only ever be set AFTER this marker
        # fires, `_recenter_strip` is always equivalent to the plain
        # active-only scroll during this entire mount-settle window --
        # routing this call through it (round 3) rather than the
        # lower-level `_scroll_active_destination_into_view` directly is
        # therefore behavior-neutral here, and keeps every settle path
        # going through the one shared, focus-aware entry point.
        self.call_after_refresh(self._mark_mount_settled)
        self.set_interval(0.5, self._update_overflow_hints)
        # The strip's virtual_size only settles once its scrollable content
        # has laid out, which is AFTER the first call_after_refresh tick (an
        # early check measures a zero-width region and pins the hint visible).
        # Re-check on short timers and on every later resize.
        self.set_timer(0.05, self._refresh_overflow_hint_visibility)
        # The overflow hint can reduce the strip only after the first active
        # scroll. Recenter once that final width is known.
        self.set_timer(0.06, self._recenter_strip)
        self.set_timer(0.25, self._refresh_overflow_hint_visibility)
        self.set_timer(0.26, self._recenter_strip)

    def _mark_mount_settled(self) -> None:
        """Close the mount-settle window -- but only once the screen's
        AUTOMATIC focus placement has verifiably landed (task-4024).

        The original marker was purely time-based (this method runs on
        `on_mount`'s first `call_after_refresh` tick) and relied on the
        empirical first-mount ordering: `AUTO_FOCUS` lands strictly BEFORE
        this tick, so any `DescendantFocus` arriving after it must be a
        genuine user Tab press. A screen-level recompose breaks that
        ordering: `SettingsScreen.on_mount` -> `_refresh_sync_rows()` sets
        `recompose=True` reactives that recompose the whole screen and mint
        a REPLACEMENT bar, and for that bar the screen is already laid out,
        so this tick fires a few ms BEFORE the post-recompose focus
        placement (Textual's refocus after the focused widget was removed,
        or `SettingsScreen._restore_focus_after_sync_rows`) lands on the
        bar's first button. That automatic landing then got recorded as a
        DELIBERATE focus, and every later `_recenter_strip` pass recentered
        on always-visible `nav-home` instead of the active destination --
        live-observed as Settings' active highlight pinned off-screen at
        `scroll_x=0` indefinitely, with manual `scroll_to_widget` calls
        snapped back within one 0.5s interval tick (task-4024).

        So: if nothing on the screen holds focus yet when this tick runs,
        the automatic placement has NOT landed -- stay unsettled and let
        `on_descendant_focus` consume the next focus event as that
        automatic landing, closing the window right after it. If focus HAS
        landed (first-mount `AUTO_FOCUS`, or a screen that focuses its own
        content on mount), settle immediately -- identical to the old
        behavior in every previously-working ordering.
        """
        focused = None
        if self.is_attached:
            try:
                focused = self.screen.focused
            except Exception:
                focused = None
        if focused is None:
            self._settle_after_next_focus = True
            return
        self._mount_settled = True

    def _update_overflow_hints(self) -> None:
        """Toggle the ‹ indicator from real scroll state.

        task-15473: gated behind a cheap signature of everything the
        downstream work actually depends on -- scroll position, the
        strip's rendered and virtual (content) widths, and the set of
        button ids currently in the strip. This is the callback for the
        periodic 0.5s interval (`on_mount`), the first post-mount pass,
        and `on_resize`'s deferred call; before this it ran the full
        measure/toggle/recenter/ghost pipeline unconditionally on every
        one of those triggers, forever, on every screen (verified: a
        no-op tick used to still call `_refresh_overflow_hint_visibility`
        and schedule `_recenter_strip` every 0.5s with nothing having
        moved). When the signature is unchanged since the last full pass,
        every one of those is skipped -- geometry that has not moved
        cannot need a different hint state or scroll position, so this
        is behavior-neutral, not merely "probably fine": any scroll, any
        resize, and any button being added or removed changes the
        signature and forces a full pass on the very next tick.
        """
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
        signature = (
            scroll_x,
            strip.region.width,
            strip.virtual_size.width,
            tuple(child.id for child in strip.children),
        )
        if signature == self._overflow_signature:
            return
        self._overflow_signature = signature
        # Left hint tracks position (more destinations hidden on the left);
        # the right affordance's visibility is width-driven, re-evaluated by
        # `_refresh_overflow_hint_visibility` (NV-01 reclaimable-space math).
        left_hint.display = scroll_x > 0
        self._refresh_overflow_hint_visibility()
        # Layout settles asynchronously (hint toggles change the strip's
        # width, fonts finish, etc.), so keep the active destination (or,
        # while it differs, the keyboard-focused button -- see
        # `_recenter_strip`, review rounds 2-3) pinned every time the
        # signature above moves — the call is idempotent and cheap, and
        # the signature already changing here is what used to be covered
        # by calling it unconditionally every tick.
        self.call_after_refresh(self._recenter_strip)

    def on_resize(self) -> None:
        """Re-sync the overflow affordance when the bar's width changes.

        The strip's overflow is a function of the bar's rendered width, so
        every resize re-evaluates whether the "More ▾" control shows, and
        (task-3200) re-measures which buttons clip -- a resize can
        un-straddle (or newly straddle) a destination at the edge.

        Both calls below are deferred (`call_after_refresh`); the first,
        `_update_overflow_hints`, itself unconditionally ends by chaining
        into `_recenter_strip` (re-scroll to the active destination, or
        the deliberately-focused button if one differs, THEN ghost-check
        via `_ghost_clipped_buttons`) -- so a resize reaches the ghost
        re-check through that existing chain rather than calling
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

        Review round 3: routing through `_recenter_strip` (rather than
        the lower-level `_scroll_active_destination_into_view`) also
        matters here specifically: a resize used to drag the strip back
        toward the active destination indifferently to focus -- live-
        reproduced as a second instance of the interval's exact defect
        class: `active="schedules"`, Tab to `nav-settings`, then a resize
        (80 -> 90 cols) dragged the strip back toward active the same way
        the un-fixed interval once did, leaving `nav-settings` straddling
        (`Region(x=66, width=15)` vs `strip.region.right == 80`),
        un-ghosted, enabled, and still focused. `_update_overflow_hints`
        already chains into `_recenter_strip` (not the lower-level,
        active-only method), so this indirect path carries the fix
        without on_resize needing its own direct call.
        """
        self.call_after_refresh(self._update_overflow_hints)
        self.call_after_refresh(self._refresh_overflow_hint_visibility)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        """Re-scroll to the newly-focused button, then ghost-check
        (task-3200 review finding).

        Args:
            event: Textual's focus notification; ``event.widget`` is the
                descendant that just received focus. Only
                ``NavigationButton`` descendants trigger the re-scroll —
                anything else falls through untouched.

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

        Review round 2: also records `widget.id` as the "deliberate focus"
        target (`_recenter_strip` reads it) -- but ONLY once `self.
        _mount_settled` is True. Textual's own `AUTO_FOCUS` (`App.
        AUTO_FOCUS = "*"` by default, not overridden anywhere in this
        app) posts exactly this same `DescendantFocus` event for whichever
        button happens to be first in the bar the instant it mounts --
        empirically confirmed to arrive strictly BEFORE this bar's own
        first `call_after_refresh` callback, i.e. before `_mount_settled`
        is set. Recording that as "deliberate" would make the periodic
        interval fight the initial mount-time scroll-to-active forever
        (nothing ever moves that stray auto-focus away on its own) --
        reproduced live, it broke `test_master_shell_navigation_keeps_
        active_destination_visible_on_mount`. A genuine Tab press always
        arrives well after mount settles (call_after_refresh chains
        resolve within a frame or two of real time, an eternity before any
        user could press a key), so gating on `_mount_settled` cleanly
        separates the two without depending on exact event ordering
        beyond what was already confirmed.
        """
        widget = event.widget
        if not isinstance(widget, NavigationButton):
            return
        if self._mount_settled and widget.id:
            self._deliberate_focus_id = widget.id
        elif self._settle_after_next_focus:
            # task-4024: `_mark_mount_settled` ran before the screen's
            # automatic focus placement landed (the post-recompose
            # ordering) -- THIS event is that automatic landing. Consume it
            # without recording it as deliberate, and close the settle
            # window so the NEXT focus change is trusted as a user's.
            self._settle_after_next_focus = False
            self._mount_settled = True
        self.call_after_refresh(self._scroll_to_focused_then_ghost_check, widget)

    def _scroll_to_focused_then_ghost_check(self, widget: "NavigationButton") -> None:
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
            strip.scroll_to_widget(widget, animate=False, immediate=True)
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
        # (rebase note) Review round 3's `_defer_focus_release_check`/
        # `_release_focus_if_left_straddling` pair -- keeping a
        # deliberately-focused nav button from being left straddling by a
        # "More ›" pager press -- is not carried forward here: dev's
        # NV-01/TASK-2154.21 rework (merged in parallel with the whole
        # task-3200 series) replaced the in-strip paging control with
        # `handle_overflow_hint` opening a real `NavOverflowMenu` screen
        # (see that method below) instead of scrolling the strip. There is
        # no more `_page_destination_overflow`/`event.button` press to
        # hang this fix off of, and the menu makes every destination
        # reachable directly rather than by paging a scroll viewport, so
        # the defect class this fix closed (paging strands a focused
        # button mid-straddle) cannot recur.


    def _focused_strip_button(self) -> "NavigationButton | None":
        """The nav button currently holding keyboard focus, or ``None``.

        `NavigationButton` is constructed only in this class's own
        `compose()` (confirmed by an exhaustive codebase grep -- nothing
        else builds one), so an `isinstance` check against the screen's
        `focused` widget is a sufficient and sole test. Note this can be
        true for reasons OTHER than a deliberate Tab press -- e.g.
        Textual's own default `AUTO_FOCUS` lands on the first focusable
        widget (`nav-home` in a bare test) the instant the app mounts, well
        before any user interaction -- so callers must not treat "a strip
        button is focused" as "the user is mid-Tab-interaction" (see
        `_recenter_strip`'s docstring for the regression that
        distinction fixed).

        Review round 2 crash fix: `self.screen` raises `NoScreen` once this
        widget is no longer attached to an active screen (a real, live
        crash caught only by a full-app regression sweep, not the bare-
        widget nav tests -- a deferred `call_after_refresh` callback
        reaching `_recenter_strip` after a screen swap had already
        unmounted this bar took the WHOLE app down mid-test, `NoScreen:
        node has no screen`, and every subsequent `Tab` press in that test
        silently did nothing because the app had already exited). Every
        OTHER `self.screen` access in this class is guarded by `self.
        is_attached` first (see `_update_overflow_hints`, `_ghost_clipped_
        buttons`) -- this one needed the same guard.
        """
        if not self.is_attached:
            return None
        try:
            focused = self.screen.focused
        except Exception:
            return None
        return focused if isinstance(focused, NavigationButton) else None

    def _scroll_active_destination_into_view(self) -> None:
        """Bring the active destination's button into the strip's visible
        scroll window, then re-check for a straddling neighbor.

        Always targets the ACTIVE destination, unconditionally, with no
        awareness of focus -- the low-level primitive `_recenter_strip`
        builds on. Nothing outside this class, and no other method in it,
        should call this directly anymore (review round 3): every
        settle-triggering event (mount, the interval, `on_resize`,
        `restore_active`) now goes through `_recenter_strip` instead, so
        that "is a deliberately-focused button different from active"
        gets asked in exactly ONE place rather than re-implemented, or
        forgotten, per call site -- which is exactly what round 2's fix
        (scoped only to the interval) missed: the identical defect was
        independently live-reproduced through `on_resize` and
        `restore_active` at HEAD after round 2 shipped, because those two
        still called this method directly. `_activate_navigation_button`
        (a click) is the one remaining direct caller, deliberately not
        migrated -- see that method for why.
        """
        try:
            strip = self.query_one("#nav-destination-strip", Horizontal)
            button = strip.query_one(
                f"#nav-{self.active_destination_id}", NavigationButton
            )
        except Exception:
            return
        try:
            strip.scroll_to_widget(button, animate=False, immediate=True)
            # Textual's widget helper can settle one cell short after the
            # docked overflow control narrows the strip. Finish the tiny
            # horizontal correction so the active button never straddles.
            if button.region.right > strip.region.right:
                strip.scroll_to(
                    x=strip.scroll_x + button.region.right - strip.region.right,
                    animate=False,
                    immediate=True,
                )
        except Exception:
            return
        self.call_after_refresh(self._ghost_clipped_buttons)

    def _recenter_strip(self) -> None:
        """The ONE focus-aware recenter every settle trigger in this class
        (mount, the periodic interval, `on_resize`, `restore_active`)
        funnels through (review round 3 generalization).

        Round 2 fixed this exact defect class for the periodic interval
        ONLY (as `_recenter_periodic`, this method's prior name), because
        that was the trigger the round-2 review's probe demonstrated. The
        round-3 re-review live-reproduced the IDENTICAL stranding --
        genuinely straddling, un-ghosted, enabled, still-focused button --
        through the three OTHER active-only recenter triggers that
        round 2 left untouched:
        - `on_resize`: `active="schedules"`, Tab to `nav-settings`, resize
          80 -> 90 cols -> `nav-settings` measured `Region(x=66, width=
          15)` against `strip.region.right == 80`.
        - `restore_active`: Tab to `nav-settings` (`active="schedules"`),
          an optimistic click-activate to `console`, then `restore_active
          ("schedules")` -> same stranding.
        - the "More ›" pager: this class no longer pages the strip at all
          (see `handle_overflow_hint` below) -- dev's parallel NV-01/
          TASK-2154.21 rework replaced in-strip paging with a real
          `NavOverflowMenu` screen listing every destination, so the
          pager-specific stranding this generalization originally also
          had to cover (a paged-away, deliberately-focused button left
          straddling) cannot recur; there is no scroll-viewport position
          for a menu row to strand against.

        This is why round 2's "closed at the source" framing (this
        report, task-3200's notes) OVERSTATED coverage: the source was
        "every caller of a active-only recenter", not "the interval", and
        patching each remaining call site individually would have been
        the same mistake a third and fourth time. Generalizing HERE, once,
        and switching every non-pager caller to call this method instead
        of `_scroll_active_destination_into_view` directly, closes the
        whole class in one place.

        Behavior: while a DIFFERENT nav-strip button than the active
        destination currently holds DELIBERATE focus, this recenters on
        THAT button instead, matching what the user's own Tab press
        already asked for; with no such conflict (nothing deliberately
        focused in the strip, or the focused button IS the active one --
        including the entire mount-settle window, when `_deliberate_
        focus_id` cannot yet be set at all, see `on_mount`'s comment), it
        defers to the normal, unconditional `_scroll_active_destination_
        into_view`. Only ever one target per call -- never active and
        focused both fighting for scroll position in the same pass, which
        is what would risk a ping-pong (this file's history already shows
        two earlier, unrelated scroll/focus attempts broke Tab that way --
        this does not touch scroll TARGETING based on live geometry or
        focus REJECTION, only which existing button `scroll_to_widget` is
        pointed at).

        "Currently holds focus" specifically means `self._deliberate_
        focus_id` (set by `on_descendant_focus`, only once `self.
        _mount_settled`) matches the LIVE focused widget -- not merely
        "some nav button is focused" (`_focused_strip_button()` alone).
        The live widget is checked too, not just the id, so focus that has
        since moved to page content (or blurred entirely) correctly falls
        through to the active-destination default; a bare id string could
        not distinguish "still focused" from "was focused once, isn't
        anymore".
        """
        focused = self._focused_strip_button()
        if (
            focused is not None
            and focused.id
            and focused.id == self._deliberate_focus_id
            and focused.id != f"nav-{self.active_destination_id}"
        ):
            try:
                strip = self.query_one("#nav-destination-strip", Horizontal)
                strip.scroll_to_widget(focused, animate=False, immediate=True)
            except Exception:
                return
            self.call_after_refresh(self._ghost_clipped_buttons)
            return
        self._scroll_active_destination_into_view()

    def _ghost_clipped_buttons(self) -> None:
        """Visually blank (never partially render) any nav button whose
        current position straddles either edge of the strip's scroll
        viewport (task-3200: a destination label must never be cut
        mid-word, e.g. "Watchlists" -> "Watc").

        This is purely cosmetic (`.nav-button-clip-ghost`, defined in
        `BUNDLED_CSS`, colors every surface to match the bar's own
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
        which ghosting never sets.

        The active destination is UNCONDITIONALLY exempt from
        `should_ghost` below (never ghosted, even transiently). Every
        settle trigger (mount, the interval, `on_resize`, `restore_active`)
        now funnels through `_recenter_strip` (review round 3), which
        guarantees active fully visible UNLESS a deliberately-focused
        different button currently takes priority -- in that one case,
        active itself is not re-guaranteed visible by this pass, but stays
        exempt from ghosting regardless (the same accepted trade-off round
        2 already made for the interval, now applying uniformly instead of
        to one trigger only: a transiently non-guaranteed-visible active
        destination is still never actively hidden, and the next pass
        without a focus conflict restores the guarantee). `
        _activate_navigation_button` (a click) is the one caller that
        still goes straight to the lower-level, always-active `_scroll_
        active_destination_into_view` (see that method's docstring for
        why).

        The FOCUSED button (if it differs from active, and only when it is
        a DELIBERATE focus -- see `_recenter_strip`'s docstring) is
        UNCONDITIONALLY exempt too, the same way active is -- NOT a
        "retry the scroll, then judge by a synchronous re-measurement"
        guard. A round 2 attempt at exactly that guard was reverted: a
        `Region` only reflects a NEW `scroll_x` after the NEXT layout
        pass, so a `scroll_to_widget` call and an immediate, SAME-call
        `button.region` re-check inside this one synchronous method can
        never see the fix take effect -- the retry always "failed" against
        stale geometry, so the focused button got ghosted AND disabled
        anyway, immediately blurring itself (`watch_disabled`) and
        reintroducing the exact "Tab jumps back to the first button" bug
        Important #1's original focused-exemption fixed (regression
        caught live: a full-suite regression sweep, not the bare-widget
        nav tests, which never exercise a real screen switch or the
        multi-lap Tab cycling needed to trigger it -- `test_tab_order_
        reaches_visible_primary_action` went from a clean, monotonic
        13-press traversal of the bar back to a 2-lap, 23-press one).
        `scroll_to_widget` is still called here for the focused button (a
        harmless, one-line best-effort nudge, and it satisfies the review
        round 2 "drive scroll_to_widget(focused) in the same chain"
        request literally) -- it just does not gate the ghost/disable
        decision, which stays unconditional like active's. The actual,
        properly-deferred guarantee that a focused button becomes (and
        stays) non-straddling comes from `on_descendant_focus` (fires once
        per real focus change, always followed by its OWN `call_after_
        refresh`-deferred re-scroll-then-ghost-check) and `_recenter_
        strip` (the interval, via `_update_overflow_hints`, re-affirming it
        every 0.5s -- `_recenter_strip` was named `_recenter_periodic`
        before review round 3 generalized it to every settle trigger) --
        both already correctly deferred, unlike a synchronous retry here could
        ever be. A focused button stuck straddling despite both of those
        is the same already-acknowledged, out-of-scope degenerate case
        (wider than the entire viewport) the active exemption already
        accepts.
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
        focused = self._focused_strip_button()
        # Same "deliberate, not merely auto-focused" gate as
        # `_recenter_strip` (see that method and `on_descendant_focus`
        # for why): a straddling AUTO_FOCUS target from before the bar
        # settled must not earn an exemption from active either.
        focused_id = (
            focused.id
            if focused is not None and focused.id == self._deliberate_focus_id
            else None
        )
        for button in strip.query(NavigationButton):
            region = button.region
            if region.width <= 0:
                continue
            straddles = _straddles_viewport(region, strip_region)
            if straddles and focused_id is not None and button.id == focused_id:
                # Best-effort nudge only -- does NOT gate `should_ghost`
                # below (see docstring: a synchronous re-measurement here
                # cannot be trusted).
                try:
                    strip.scroll_to_widget(button, animate=False, immediate=True)
                except Exception:
                    pass
            should_ghost = straddles and button.id != active_id and button.id != focused_id
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
        """Activate a navigation button and return whether navigation was requested.

        Guards `button.disabled` here, at the one shared entry point, rather
        than only in a caller: `handle_navigation` (via `Button.Pressed`)
        already can't reach a disabled button -- `Button.press()` no-ops on
        `disabled` before the event ever posts -- but `on_click`'s
        border-click router (below) resolves a click landing on the bar's
        own chrome to ANY `NavigationButton` whose `region.contains_point`
        matches, then used to call this method directly, bypassing `Button.
        press()`'s disabled check entirely (task-3200 review round 5). A
        ghosted (task-3200/3225) button's region is real, unshrunk geometry
        -- ghosting is purely cosmetic -- so a click on the blank-looking
        chrome over a ghosted tab silently navigated to an invisible
        destination. Guarding the shared method, not just `on_click`,
        also covers any future direct caller of this method for free.
        """
        if button.disabled:
            return False
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
        # Deliberately still the plain, active-only
        # `_scroll_active_destination_into_view`, NOT `_recenter_strip`
        # (review round 3): a click/Enter-press on `button` here is the
        # thing that just SET `active_destination_id` to `button`'s own
        # destination, and a real mouse click (or Enter while `button`
        # holds focus) also focuses `button` itself via Textual's normal
        # click-handling -- so the "deliberate focus" and "active" targets
        # are the SAME button in the by-far-common case, making this a
        # no-op distinction there. The risk of routing through
        # `_recenter_strip` here instead: if some STALE, unrelated
        # `_deliberate_focus_id` from an earlier Tab press happens to
        # still be the LIVE focused widget when a DIFFERENT button gets
        # clicked (e.g. a click path that does not itself move focus),
        # `_recenter_strip` would scroll back to that stale target instead
        # of the button the user just activated -- the opposite of what a
        # click-driven activation should ever do. Not exercised by any
        # review finding (round 3's three repros are `on_resize`,
        # `restore_active`, and the pager only), so left as the
        # unconditional, already-correct-for-its-purpose primitive.
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
        # Review round 3: routes through `_recenter_strip`, not the plain
        # active-only primitive -- live-reproduced as a stranding: Tab to
        # `nav-settings` (active="schedules"), an optimistic click-activate
        # to `console` (`_activate_navigation_button`, before its
        # navigation actually completes), then `restore_active
        # ("schedules")` left `nav-settings` genuinely straddling,
        # un-ghosted, enabled, and still focused -- this call used to
        # unconditionally re-target `schedules`, indifferent to the
        # keyboard focus that had moved to `nav-settings` in the
        # meantime, the identical defect class round 2 fixed only for the
        # periodic interval.
        self.call_after_refresh(self._recenter_strip)
