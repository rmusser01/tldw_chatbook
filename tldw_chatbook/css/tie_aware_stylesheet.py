# tie_aware_stylesheet.py
# Description: Stylesheet that reparses after a source's tie-breaker is lowered
# (TASK-21115).
#
# Textual registers one stylesheet source per widget class with `DEFAULT_CSS`,
# and a class's *base* classes are offered at tie-breaker `-(MRO position)`
# every time any widget instance registers (`Widget._post_register` ->
# `DOMNode._get_default_css`). `Stylesheet.add_source` keeps the LOWEST
# tie-breaker ever offered for a source -- but when it lowers a stored
# tie-breaker it does not set `_require_parse` (textual 8.2.8,
# `css/stylesheet.py`), so the already-parsed rules keep the OLD value until
# something else forces a reparse.
#
# Before the BUNDLED_CSS consolidation that staleness was unobservable: a
# class's own `DEFAULT_CSS` was a NEW source at its first mount, which set
# `_require_parse` itself, so the very next `apply()` reparsed with every
# lowered tie-breaker in effect. A consolidated class adds NO source at first
# mount, so a *dynamic* first mount (after boot) can resolve against a stale
# parse. Measured failure shape (this task's harness A/B): a bare `Vertical`
# mounts at boot, registering Textual's `Vertical { width: 1fr; height: 1fr }`
# at tie-breaker 0 (it is that widget's OWN class); a consolidated
# Vertical-subclass then first-mounts dynamically -- its registration lowers
# the stored `Vertical` tie-breaker to -1, but with no reparse the parsed
# rules still carry 0, which exactly ties the consolidated sheet's
# `<Class> { width: auto; ... }` rule (specificity (0,0,1), tie-breaker 0) and
# wins on source order. The widget mounts full-size instead of its own
# geometry. With per-class `DEFAULT_CSS` the same mount produced
# `width: auto` -- so this subclass restores the exact pre-consolidation
# cascade, it does not invent a new one.
#
# Cost: one extra reparse per *lowering* of a stored tie-breaker (bounded per
# base class per session, and warm reparses hit Textual's parse cache -- the
# cache the consolidation exists to keep under its 64-source cliff).
#
# Known gap: Textual's dev-mode CSS hot-reload (`App._on_css_change`) swaps in
# a plain `Stylesheet`, dropping this subclass until the next app start. That
# path only runs with `watch_css`/dev tooling, never in production or tests.

from __future__ import annotations

from textual.css.stylesheet import CssSource, Stylesheet
from textual.css.types import CSSLocation


class TieAwareStylesheet(Stylesheet):
    """`Stylesheet` that treats a lowered source tie-breaker as a CSS change.

    See the module docstring for why: without this, a widget class whose CSS
    rides the consolidated `BUNDLED_CSS` sheets can first-mount against a
    stale parse in which a base class's default rules still carry a
    tie-breaker of 0 and shadow the sheet's rules for that class.
    """

    def add_source(
        self,
        css: str,
        read_from: CSSLocation | None = None,
        is_default_css: bool = False,
        tie_breaker: int = 0,
        scope: str = "",
    ) -> None:
        """Add a CSS source, arming a reparse if its tie-breaker lowered.

        Args:
            css: String with CSS source.
            read_from: The original source location of the CSS.
            is_default_css: True for widget-level (default-tier) CSS.
            tie_breaker: Priority of this source; the lowest offer is kept.
            scope: CSS type name to limit scope, or empty for no scope.

        Raises:
            StylesheetError: If the CSS could not be read.
            StylesheetParseError: If the CSS is invalid.
        """
        key = read_from if read_from is not None else ("", str(hash(css)))
        existing: CssSource | None = self.source.get(key)
        super().add_source(
            css,
            read_from=read_from,
            is_default_css=is_default_css,
            tie_breaker=tie_breaker,
            scope=scope,
        )
        if (
            existing is not None
            and existing.content == css
            and tie_breaker < existing.tie_breaker
        ):
            # Upstream lowered the stored tie-breaker (it keeps the minimum)
            # without flagging a reparse; the parsed rules would keep the old
            # value until an unrelated new source forces one.
            self._require_parse = True
