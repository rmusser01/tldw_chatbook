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
    * Every app-global key (F1 help, F6 next pane, Ctrl+P palette, Ctrl+Q
      quit)
      is ALWAYS represented somewhere in the footer, but not always by the
      generic global label: a screen's shortcut context renders UNFILTERED,
      and the global cluster then excludes whichever of those four keys the
      screen's own context already covers (task-2860's per-key dedup, see
      `_remaining_global_text`) so the same key never shows twice. A key
      the screen doesn't mention still gets its generic global hint; a key
      the screen DOES mention shows only the screen's own (more specific)
      copy, never both -- so a screen's context routinely supersedes a
      global key's generic label, it just can never make the key vanish
      from the footer entirely.
    * When width runs out, the screen-context hints drop first (leaving an
      ellipsis marker), then the right cluster (token/word/DB sizes) hides
      progressively; nothing ever clips mid-word.
    """

    #: (key, label) pairs backing each width tier of the app-global hint
    #: cluster (ADR-031: f1/f6/ctrl+p/ctrl+q are app-global keys screens
    #: must not rebind). Single source of truth for both the joined
    #: class-level strings below (still referenced directly by several
    #: tests/screens) and the per-key dedup in `_remaining_global_text`.
    #: task-2860: keeping two separate representations of these four keys
    #: in sync (a joined string here, a hardcoded key-only filter that used
    #: to live in `set_shortcut_context`) is exactly how a screen's own F6
    #: ("next pane") hint went silently missing -- the old filter dropped the
    #: screen's copy unconditionally, leaving the key advertised nowhere
    #: in the historical compact tier. The screen's context now renders
    #: UNFILTERED (see `set_shortcut_context`), and this global half
    #: instead excludes whichever keys the context already covers -- see
    #: `_remaining_global_text`. A key the screen does not mention still
    #: gets its generic global hint; a key the screen does mention shows
    #: only the screen's own (more specific) copy, never both.
    _GLOBAL_HINT_ITEMS_FULL = (
        ("f1", "F1 help"),
        # task-4023 AC#5: "next pane" -- the same name the screens' own
        # context sets use for this key (Library said "F6 next pane" while
        # this cluster said "F6 panes" on the SAME footer line's other
        # half, two names for one key).
        ("f6", "F6 next pane"),
        ("ctrl+p", "Ctrl+P palette"),
        ("ctrl+q", "Ctrl+Q quit"),
    )
    _GLOBAL_HINT_ITEMS_COMPACT = (
        ("f1", "F1"),
        ("f6", "F6"),
        ("ctrl+p", "Ctrl+P"),
        ("ctrl+q", "Ctrl+Q"),
    )
    _GLOBAL_HINT_ITEMS_MIN = (("ctrl+q", "Ctrl+Q"),)

    GLOBAL_HINTS = " · ".join(label for _key, label in _GLOBAL_HINT_ITEMS_FULL)
    GLOBAL_HINTS_COMPACT = " · ".join(
        label for _key, label in _GLOBAL_HINT_ITEMS_COMPACT
    )
    GLOBAL_HINTS_MIN = " · ".join(label for _key, label in _GLOBAL_HINT_ITEMS_MIN)
    DEFAULT_SHORTCUT_TEXT = GLOBAL_HINTS

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
    # has no consumers): BUNDLED_CSS covers stylesheet-less harnesses; the
    # app bundle wins by origin in production. A bundle-only edit would
    # silently diverge harness geometry from production (task-264 review).
    BUNDLED_CSS = """
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
        #: The active context's raw (unfiltered) actions -- task-2860: kept
        #: around so `_remaining_global_text` can tell which reserved keys
        #: the screen already covers, at whatever width tier is rendering.
        self._context_actions: tuple[ShortcutAction, ...] = ()
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

    def _remaining_global_text(
        self,
        items: tuple[tuple[str, str], ...],
        actions: tuple[ShortcutAction, ...] | None = None,
    ) -> str:
        """Join a global-hint tier, dropping keys the screen already covers.

        Args:
            items: ``(key, label)`` pairs for one width tier (see
                ``_GLOBAL_HINT_ITEMS_FULL``/``_COMPACT``/``_MIN``).

        Returns:
            The tier's hints, minus any whose key the active context
            already advertises under its own (available) label -- task-2860.
        """
        visible_actions = self._context_actions if actions is None else actions
        covered = {
            action.key.lower()
            for action in visible_actions
            if action.available
        }
        return " · ".join(label for key, label in items if key not in covered)

    @staticmethod
    def _render_actions(actions: tuple[ShortcutAction, ...]) -> str:
        """Render an ordered subset of available workflow hints."""
        return " | ".join(
            action.render() for action in actions if action.available
        )

    @staticmethod
    def _combine(context_text: str, globals_text: str) -> str:
        """Join the context and global halves, tolerating an empty globals
        half (every reserved key the tier would show is already covered by
        the context itself)."""
        if not globals_text:
            return context_text
        return f"{context_text} | {globals_text}"

    def _full_text(self) -> str:
        """Screen context followed by the always-present global hints.

        Any reserved global key (f1/f6/ctrl+p/ctrl+q) the context already
        advertises under its own label is excluded from the global half
        here instead of being dropped from the context -- see
        ``_remaining_global_text`` (task-2860).
        """
        if self._context_text:
            remaining = self._remaining_global_text(self._GLOBAL_HINT_ITEMS_FULL)
            return self._combine(self._context_text, remaining)
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
        # task-2860: the context renders UNFILTERED now -- a screen's own
        # hint for a reserved global key (e.g. F6 "next pane") is real
        # content the user came here to discover, not noise to censor. The
        # always-present global cluster instead excludes whatever the
        # context already covers (see `_remaining_global_text`), so the key
        # is still never shown twice.
        self._context_actions = context.actions
        self._shortcut_source = context.source
        self._context_text = context.render() or None
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
        self._context_actions = ()
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
        ellipsis immediately (``"… F1 help · F6 next pane · ..."``) -- at
        Library's real ~100-column footer width this meant the screen-
        specific keys (``/ focus search``, ``i import content``, ...) the
        user actually came here to discover vanished behind that leading
        "…", while the always-present globals (F1/Ctrl+P/Ctrl+Q, muscle-
        memory keys most users already know) stayed in full. A step in
        between now compacts the GLOBAL half first (reusing
        ``GLOBAL_HINTS_COMPACT``, the same constant the no-context branch
        below already leans on) while the screen's own hints stay intact --
        ordering the screen-specific keys ahead of the globals in practice.

        task-2860: the global half at every tier below is built via
        ``_remaining_global_text``, which excludes whichever reserved keys
        (f1/f6/ctrl+p/ctrl+q) the context already advertises under its own
        label -- so a screen's own F6 ("next pane") hint, say, survives
        even the compact tier instead of being silently dropped everywhere.
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
            full = self._combine(
                self._context_text,
                self._remaining_global_text(self._GLOBAL_HINT_ITEMS_FULL),
            )
            context_compact_globals = self._combine(
                self._context_text,
                self._remaining_global_text(self._GLOBAL_HINT_ITEMS_COMPACT),
            )
            # Once the context is dropped entirely (ellipsis/compact below),
            # there is nothing left to dedupe against -- fall back to the
            # plain, undeduped global constants.
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
            ]
            # TASK-15702: when the full workflow context still does not
            # fit, retain its highest-priority prefix before falling back
            # to a global-only ellipsis. Screens order actions primary,
            # recovery, then navigation. Build globals against the prefix
            # actually shown so a truncated F6 reappears in the compact
            # global cluster rather than vanishing.
            available_actions = tuple(
                action for action in self._context_actions if action.available
            )
            for count in range(len(available_actions) - 1, 0, -1):
                prefix = available_actions[:count]
                prefix_text = self._render_actions(prefix)
                prefix_globals = self._remaining_global_text(
                    self._GLOBAL_HINT_ITEMS_COMPACT,
                    prefix,
                )
                steps.append(
                    (
                        self._combine(prefix_text, prefix_globals),
                        False,
                        False,
                        False,
                    )
                )
            steps.extend(
                (
                    (ellipsis, False, False, False),
                    (compact, False, False, False),
                    (self.GLOBAL_HINTS_MIN, False, False, False),
                )
            )
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
