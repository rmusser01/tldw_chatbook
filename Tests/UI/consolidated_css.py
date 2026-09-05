# consolidated_css.py
# Description: Test-harness support for the consolidated widget CSS (TASK-15450).
#
# Widget CSS that used to live in a class's `DEFAULT_CSS` now lives in
# `BUNDLED_CSS` and is lifted into `css/widget_defaults_{self,scoped}.tcss` at
# build time, which the real app registers as two stylesheet sources. A harness
# App that mounts one of those widgets therefore has to register them too, or
# the widget mounts unstyled and geometry/colour assertions measure nothing.
#
# The same applies to the seven screen/modal classes whose class-level `CSS`
# became `BUNDLED_SCREEN_CSS`: Textual used to register that automatically when
# the screen was pushed, so a harness that pushes one now gets a modal with no
# CSS at all unless the screen sheets are loaded too. `CSS_PATH` carries those.
#
# A subclass that declares its own `CSS_PATH` (most do, pointing at the app
# bundle) would ordinarily shadow that class attribute outright via normal
# Python attribute lookup, and Textual's own `App.__init__` resolves a
# `css_path=` constructor kwarg the same way (`css_path or self.CSS_PATH`) --
# either form drops the screen sheets entirely, unlike before TASK-15450 where
# a pushed screen's own CSS registered itself regardless of the harness's
# `CSS_PATH` (TASK-15995). `__init__` below merges the screen sheets around
# whichever form a subclass uses, so both keep working.
#
# Inherit `ConsolidatedCSSApp` instead of `App` to get exactly what production
# gets -- same sheets, same tie-breakers, same cascade position -- via the same
# `build_css` helpers `TldwCli` uses.

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual._path import CSSPathType, _css_path_type_as_list
from textual.app import App

from tldw_chatbook.css import build_css
from tldw_chatbook.css.tie_aware_stylesheet import TieAwareStylesheet

CSS_DIR = Path(build_css.__file__).parent

#: The app bundle, for harnesses that want the app-CSS tier as well.
BUNDLED_STYLESHEET = CSS_DIR / "tldw_cli_modular.tcss"

#: TASK-25812/TASK-24459: the per-screen sheets split out of the
#: screen-owned modules (agentic terminal, evals, scheduling). The REAL app
#: parses these lazily, on first visit to the owning screen
#: (`App._load_screen_css` via each screen's `CSS_PATH`), so at steady
#: state they are part of the app's styling exactly as the bundle is. A
#: harness that pins `CSS_PATH` to the bundle alone silently loses every
#: moved rule; harnesses and contracts that mean "the app's styling"
#: should use `APP_STYLESHEETS` / `app_css_text()` instead.
AGENTIC_SPLIT_STYLESHEETS = tuple(
    CSS_DIR / name for name in build_css.AGENTIC_SPLIT_SHEETS.values()
)

SCREEN_OWNED_SPLIT_STYLESHEETS = tuple(
    CSS_DIR / filename
    for split in build_css.SCREEN_OWNED_SPLITS
    for filename in split.sheets.values()
)

#: Every app-tier stylesheet the running app ends up with: the boot bundle
#: plus the lazily-loaded split sheets.
APP_STYLESHEETS = (BUNDLED_STYLESHEET, *SCREEN_OWNED_SPLIT_STYLESHEETS)


def app_css_text() -> str:
    """Union text of the bundle and the split sheets.

    For text-level contracts ("this rule is styled somewhere the app
    loads"). Asserting on the bundle alone re-encodes the pre-split
    packaging, which TASK-25812 deliberately changed.

    Returns:
        The concatenated text of every path in ``APP_STYLESHEETS``, joined
        with newlines, in load order.
    """
    return "\n".join(
        path.read_text(encoding="utf-8") for path in APP_STYLESHEETS
    )

#: The screen/modal sheets (TASK-15450), scope-prefixed stream first and self
#: stream last -- the exact pair and order `TldwCli.CSS_PATH` brackets its app
#: bundle with. `screen_css_paths` is the single ordering source; nothing else
#: in this module hardcodes the pair.
_SCREEN_CSS_SCOPED, _SCREEN_CSS_SELF = build_css.screen_css_paths(CSS_DIR)


def _merge_screen_css_paths(css_path: CSSPathType | None) -> list[str]:
    """Bracket ``css_path`` with the screen/modal sheets, production order.

    ``css_path`` is whatever a subclass/instance ultimately supplied -- a
    ``CSS_PATH`` class attribute or a ``css_path=`` constructor kwarg, either
    of which replaces `ConsolidatedCSSApp.CSS_PATH` wholesale on its own.
    Re-adds the two screen sheets around it without duplicating them if they
    are already present (e.g. a subclass that does not override ``CSS_PATH``
    at all, where ``css_path`` already *is* the pair).

    Args:
        css_path: The effective CSS_PATH a subclass/instance supplied, or
            ``None``.

    Returns:
        ``[scoped_sheet, *own_entries, self_sheet]``, as strings.
    """
    own_paths = (
        [Path(path) for path in _css_path_type_as_list(css_path)] if css_path else []
    )
    middle = [
        path for path in own_paths if path not in (_SCREEN_CSS_SCOPED, _SCREEN_CSS_SELF)
    ]
    return [
        str(_SCREEN_CSS_SCOPED),
        *(str(path) for path in middle),
        str(_SCREEN_CSS_SELF),
    ]


class ConsolidatedCSSApp(App):
    """An ``App`` that loads the consolidated widget CSS, as the real app does."""

    CSS_PATH = [str(_SCREEN_CSS_SCOPED), str(_SCREEN_CSS_SELF)]

    def __init__(
        self, *args: Any, css_path: CSSPathType | None = None, **kwargs: Any
    ) -> None:
        # Mirrors Textual's own `css_path or self.CSS_PATH` resolution so the
        # merge sees whichever form actually won -- the kwarg if the caller
        # passed one, else whatever `CSS_PATH` resolves to on this instance's
        # (possibly subclassed) type.
        effective = css_path if css_path is not None else self.CSS_PATH
        super().__init__(*args, css_path=_merge_screen_css_paths(effective), **kwargs)
        # TASK-21115: same stylesheet the real app uses -- a consolidated
        # class first-mounted DYNAMICALLY (post-boot) otherwise resolves
        # against a stale parse where a base class's defaults still hold
        # tie-breaker 0 and shadow the sheet's rules for that class. See
        # `tldw_chatbook/css/tie_aware_stylesheet.py`.
        self.stylesheet = TieAwareStylesheet(variables=self.get_css_variables())

    def _get_default_css(self):  # noqa: D102 - see module docstring
        return build_css.widget_defaults_sources(CSS_DIR) + super()._get_default_css()
