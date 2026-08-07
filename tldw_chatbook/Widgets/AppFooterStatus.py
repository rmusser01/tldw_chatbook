# tldw_chatbook/Widgets/AppFooterStatus.py
#
# Imports
#
# 3rd-party Libraries
from rich.cells import cell_len
from textual.app import ComposeResult
from textual.events import Resize
from textual.widget import Widget
from textual.widgets import Static

#
# Local Imports
from ..UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
#
########################################################################################################################
#
# AppFooterStatus

#: TASK-451: cells reserved for the footer's padding/margins plus a gap so the
#: key hints don't sit flush against the debug memory stats at the boundary.
#: Below `hints + word + token + stats + this`, the memory stats hide.
_FOOTER_STATS_HEADROOM = 10


class AppFooterStatus(Widget):
    """Per-screen footer: screen hint context + protected global hints.

    Layout contract (see the UX critique, UX-006/UX-041):
    * The app-global hints (F1 help, F6 panes, Ctrl+P palette, Ctrl+Q quit)
      are ALWAYS present — a screen's shortcut context may prepend its own
      hints but never replaces the globals.
    * When width runs out, the screen-context hints drop first (leaving an
      ellipsis marker), then the right cluster (token/word/DB sizes) hides
      progressively; nothing ever clips mid-word.
    """

    GLOBAL_HINTS = "F1 help · F6 panes · Ctrl+P palette · Ctrl+Q quit"
    GLOBAL_HINTS_COMPACT = "F1 · Ctrl+P · Ctrl+Q"
    GLOBAL_HINTS_MIN = "Ctrl+Q"
    DEFAULT_SHORTCUT_TEXT = GLOBAL_HINTS

    #: Keys owned by the app-global layer (ADR-031); context hints that
    #: repeat them are filtered so the footer never says the same key twice.
    _RESERVED_GLOBAL_KEYS = frozenset({"f1", "f6", "ctrl+p", "ctrl+q"})

    # Right-cluster hiding thresholds (terminal columns).
    _TOKEN_MIN_WIDTH = 110
    _WORD_MIN_WIDTH = 100
    _DB_MIN_WIDTH = 80

    # task-264: this widget used to be mounted exactly once, directly by
    # `TldwCli.compose()` -- which always loads the app's full CSS bundle
    # (`Constants.py`'s `AppFooterStatus { ... }` type-selector rule below),
    # so the layout this widget needs to actually look like a footer (docked
    # to the bottom, 1 row tall, children arranged left/right) never had to
    # be self-contained. Now that `BaseAppScreen.compose()` mounts one of
    # these on every screen, it can be exercised by lightweight test
    # harnesses (or, in principle, any future host) that never load that
    # bundle. Without SOME baked-in layout, the un-styled `Widget` defaults
    # (block layout, unconstrained height) let its children -- notably the
    # empty ``#footer-spacer`` -- balloon to cover most of the screen and
    # silently swallow clicks meant for whatever's actually on screen.
    # Mirroring Textual's own built-in `Footer` widget (which ships its own
    # `DEFAULT_CSS` for exactly this reason), this repeats a SUBSET of the
    # bundle's rules -- the core layout ones -- so they always apply, with
    # or without that bundle loaded. The bundle carries extras (word/token
    # count ids, per-child heights) and wins by origin when both are
    # present.
    # KEEP IN SYNC with the live bundle source
    # css/components/_widgets.tcss ("Window Footer Widget" block, built
    # into tldw_cli_modular.tcss -- NOT Constants.py's css_content, which
    # has no consumers): DEFAULT_CSS covers stylesheet-less harnesses; the
    # app bundle wins by origin in production. A bundle-only edit would
    # silently diverge harness geometry from production (task-264 review).
    DEFAULT_CSS = """
    AppFooterStatus {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-1;
        width: 100%;
        layout: horizontal;
        padding: 0 1;
    }

    AppFooterStatus #footer-key-quit {
        width: auto;
        padding: 0 1;
        color: $text-muted;
        dock: left;
    }

    AppFooterStatus #footer-spacer {
        width: 1fr;
    }

    AppFooterStatus #internal-db-size-indicator {
        width: auto;
        color: $text-muted;
        dock: right;
        padding: 0 1;
        margin-left: 2;
    }
    """

    def __init__(self, show_token_count: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        #: Token counts only mean something on chat/console screens; other
        #: destinations hide the dead "Tokens: --" chrome (UX-076).
        self._show_token_count = show_token_count
        self._shortcut_text = self.DEFAULT_SHORTCUT_TEXT
        #: Rendered screen-context hints, or ``None`` for the default footer.
        self._context_text: str | None = None
        #: Source of the active shortcut context (e.g. "personas"); ``None``
        #: when the default shortcuts are shown.
        self._shortcut_source: str | None = None
        self._shortcut_display = Static(self._shortcut_text, id="footer-key-quit")
        self._word_count_display: Static = Static("", id="footer-word-count")
        self._token_count_display: Static = Static("", id="footer-token-count")
        # F-003: the Tokens chip is meaningful only where token counts exist
        # (chat contexts). It starts empty and hidden -- no "Tokens: --"
        # placeholder -- and `update_token_count` reveals it once a real
        # count lands (the periodic updater writes "" on non-chat tabs, so
        # authoring/config destinations never render dead chrome).
        self._token_count_display.display = False
        self._db_status_display: Static = Static("", id="internal-db-size-indicator")
        # F-014: the DB-size readout left user chrome (telemetry lives in
        # the Library Details disclosure and the logs now), so the indicator
        # starts collapsed and only takes space while it has content.
        self._db_status_display.display = False
        # task-1714: labels are spelled in the readout itself now; the
        # tooltip only adds the "local database file sizes" context.
        self._db_status_display.tooltip = "Local database file sizes"
        #: TASK-451: last known footer width, so a content change (new shortcut
        #: context / DB stats) can re-run the priority reflow without a resize.
        self._last_footer_width = 0

    def compose(self) -> ComposeResult:
        yield self._shortcut_display
        yield Static(id="footer-spacer")  # This will push items to the right
        yield self._word_count_display  # Word count display
        yield self._token_count_display  # Token count display
        yield self._db_status_display  # This is the existing DB size display

    def on_resize(self, event: Resize) -> None:
        """Reprioritise the footer when its width changes (TASK-451).

        Runs both responsive pipelines: the shortcut-context ladder (hint
        variants + right-cluster visibility) and the priority reflow, which
        gets the final say on the debug memory stats.

        Args:
            event: The resize event; its ``size.width`` becomes the width the
                priority reflow measures against.
        """
        self._last_footer_width = event.size.width
        self._apply_responsive_footer()
        self._reflow_footer_priority()

    def _reflow_footer_priority(self) -> None:
        """Preserve the left key hints; the right debug memory stats yield.

        On a narrow footer the right-docked memory stats (``P:/C/N:/M:`` file
        sizes -- debug telemetry) would otherwise keep full width and squeeze
        the left-docked key hints (navigation the user needs). When there is not
        room for the hints AND every right-side item, the memory stats hide
        (TASK-451). Recomputed from the raw renderables, so the decision is
        stable regardless of the stats' current visibility (no flicker).

        F-014: an EMPTY indicator (the normal state now that DB sizes live
        in the Library Details disclosure) stays collapsed regardless of
        width -- the reflow must never resurrect blank chrome.
        """
        width = self._last_footer_width or self.size.width
        if width <= 0:
            return
        stats_text = str(self._db_status_display.renderable)
        # Measure the DISPLAYED hint variant (the responsive ladder may have
        # shrunk `self._shortcut_text` to a compact form), not the stored
        # full text -- otherwise the stats would yield to hints that are not
        # actually taking the cells.
        needed = (
            cell_len(str(self._shortcut_display.renderable))
            + cell_len(str(self._word_count_display.renderable))
            + cell_len(str(self._token_count_display.renderable))
            + cell_len(stats_text)
            + _FOOTER_STATS_HEADROOM
        )
        self._db_status_display.display = bool(stats_text) and width >= needed

    @property
    def shortcut_text(self) -> str:
        return self._shortcut_text

    def _full_text(self) -> str:
        """Screen context followed by the always-present global hints."""
        if self._context_text:
            return f"{self._context_text} | {self.GLOBAL_HINTS}"
        return self.GLOBAL_HINTS

    def _set_shortcut_text(self, text: str) -> None:
        self._shortcut_text = text
        # The responsive ladder owns the hint text (it may render a shrunken
        # variant for the current width); the TASK-451 reflow then gets the
        # final say on the debug memory stats, since a new shortcut context
        # changes how much room the hints need.
        self._apply_responsive_footer()
        self._reflow_footer_priority()

    def set_shortcut_context(self, context: ShortcutContext) -> None:
        # Drop hints that duplicate the always-present global keys.
        filtered_actions = tuple(
            action
            for action in context.actions
            if action.key.lower() not in self._RESERVED_GLOBAL_KEYS
        )
        rendered = ShortcutContext(
            source=context.source, actions=filtered_actions
        ).render()
        self._shortcut_source = context.source
        self._context_text = rendered or None
        self._set_shortcut_text(self._full_text())

    def set_workbench_shortcuts(
        self,
        *,
        source: str,
        shortcuts: tuple[tuple[str, str], ...],
    ) -> None:
        """Render Workbench shortcut hints through the footer context model."""
        context = ShortcutContext(
            source=source,
            actions=tuple(ShortcutAction(key, label) for key, label in shortcuts),
        )
        self.set_shortcut_context(context)

    def clear_shortcut_context(self, source: str | None = None) -> None:
        """Reset the footer to the default shortcuts.

        Textual's ``switch_screen`` mounts the incoming screen before
        unmounting the outgoing one, so an unmount-time clear can race a
        just-registered context. Passing ``source`` makes the clear a no-op
        unless that source still owns the context; calling with no argument
        clears unconditionally (backward compatible).
        """
        if source is not None and source != self._shortcut_source:
            return
        self._shortcut_source = None
        self._context_text = None
        self._set_shortcut_text(self._full_text())

    # ------------------------------------------------------------------
    # Responsive behavior
    # ------------------------------------------------------------------
    def _right_cluster_text_len(self) -> int:
        """Rendered width of the visible right-cluster displays."""
        total = 0
        for display in (
            self._word_count_display,
            self._token_count_display,
            self._db_status_display,
        ):
            if display.display:
                total += len(str(display.render()))
        return total

    def _apply_responsive_footer(self) -> None:
        """Pick the honest hint variant that fits; never clip mid-word.

        Degradation order with a screen context registered (discoverability
        outranks metrics chrome): shrink the right cluster (DB sizes, then
        word, then token counts) BEFORE eliding the screen's own hints;
        globals stay to the last row. With no context the full and compact
        variants advertise the same global keys, so the hints shrink to
        compact first and the DB stats chip keeps its cells until even that
        overflows (matching the TASK-451 reflow's geometry).

        LIB-18: once the right cluster is fully hidden and ``full`` still
        does not fit, the screen's own hints used to drop to a bare
        ellipsis immediately (``"… F1 help · F6 panes · ..."``) -- at
        Library's real ~100-column footer width this meant the screen-
        specific keys (``/ focus search``, ``i import content``, ...) the
        user actually came here to discover vanished behind that leading
        "…", while the always-present globals (F1/Ctrl+P/Ctrl+Q, muscle-
        memory keys most users already know) stayed in full. A step in
        between now compacts the GLOBAL half first (reusing
        ``GLOBAL_HINTS_COMPACT``, the same constant the no-context branch
        below already leans on) while the screen's own hints stay intact --
        ordering the screen-specific keys ahead of the globals in practice,
        without touching ``set_shortcut_context``'s reserved-key filtering
        (task-2860's own, separate seam).
        """
        width = self.size.width
        if width <= 0:
            # Pre-layout: show the full text; on_resize will refine.
            self._shortcut_display.update(self._shortcut_text)
            return

        hard_token = width >= self._TOKEN_MIN_WIDTH and self._show_token_count
        hard_word = width >= self._WORD_MIN_WIDTH
        hard_db = width >= self._DB_MIN_WIDTH

        if self._context_text:
            full = f"{self._context_text} | {self.GLOBAL_HINTS}"
            context_compact_globals = (
                f"{self._context_text} | {self.GLOBAL_HINTS_COMPACT}"
            )
            ellipsis = f"… {self.GLOBAL_HINTS}"
            compact = f"… {self.GLOBAL_HINTS_COMPACT}"
        else:
            full = self.GLOBAL_HINTS
            context_compact_globals = full
            ellipsis = self.GLOBAL_HINTS_COMPACT
            compact = self.GLOBAL_HINTS_COMPACT

        if self._context_text:
            # (text, show_token, show_word, show_db) in degradation order.
            steps = [
                (full, True, True, True),
                (full, True, True, False),
                (full, True, False, False),
                (full, False, False, False),
                (context_compact_globals, False, False, False),
                (ellipsis, False, False, False),
                (compact, False, False, False),
                (self.GLOBAL_HINTS_MIN, False, False, False),
            ]
        else:
            # No screen context: the full and compact variants advertise the
            # same global keys, so shrink the hints to compact and keep the
            # DB stats chip until even that overflows (the TASK-451 reflow's
            # geometry assumes the stats can coexist with short hints).
            steps = [
                (full, True, True, True),
                (compact, False, False, True),
                (full, True, True, False),
                (full, True, False, False),
                (full, False, False, False),
                (compact, False, False, False),
                (self.GLOBAL_HINTS_MIN, False, False, False),
            ]
        for text, token_flag, word_flag, db_flag in steps:
            token_vis = token_flag and hard_token
            word_vis = word_flag and hard_word
            db_vis = db_flag and hard_db
            right_len = 0
            if word_vis:
                right_len += len(str(self._word_count_display.render()))
            if token_vis:
                right_len += len(str(self._token_count_display.render()))
            if db_vis:
                right_len += len(str(self._db_status_display.render()))
            available = max(width - right_len - 6, 8)
            if len(text) <= available:
                self._token_count_display.display = token_vis
                self._word_count_display.display = word_vis
                self._db_status_display.display = db_vis
                self._shortcut_display.update(text)
                return

    # ------------------------------------------------------------------
    # Right-cluster updaters
    # ------------------------------------------------------------------
    def update_db_sizes_display(self, status_string: str) -> None:
        try:
            self._db_status_display.update(status_string)
            # F-014: the indicator takes footer space only while it has
            # content -- an empty string collapses it (the reflow keeps it
            # down; see `_reflow_footer_priority`).
            self._db_status_display.display = bool(status_string)
            self._apply_responsive_footer()
            self._reflow_footer_priority()
        except Exception as e:
            # If the app is shutting down, the widget might be gone
            # In a real scenario, you'd use self.log from the widget
            print(f"Error updating AppFooterStatus display: {e}")

    def update_word_count(self, word_count: int) -> None:
        """Update the word count display in the footer."""
        try:
            if word_count > 0:
                self._word_count_display.update(f"Words: {word_count} | ")
            else:
                self._word_count_display.update("")
            # The count's width feeds the priority threshold, so re-run the
            # reflow when it changes without a resize (Qodo #834).
            self._reflow_footer_priority()
        except Exception as e:
            print(f"Error updating word count display: {e}")

    def update_token_count(self, display_text: str) -> None:
        """Update the token count display in the footer."""
        try:
            if display_text:
                self._token_count_display.update(f"{display_text} | ")
            else:
                self._token_count_display.update("")
            # F-003: the chip takes footer space only while it has content.
            self._token_count_display.display = bool(display_text)
            self._reflow_footer_priority()
        except Exception as e:
            print(f"Error updating token count display: {e}")


#
# End of AppFooterStatus.py
########################################################################################################################
