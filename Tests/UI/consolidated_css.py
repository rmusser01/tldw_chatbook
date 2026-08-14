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
# A subclass that declares its own `CSS_PATH` (most do, pointing at the bundle)
# overrides it -- which matches what those harnesses had before.
#
# Inherit `ConsolidatedCSSApp` instead of `App` to get exactly what production
# gets -- same sheets, same tie-breakers, same cascade position -- via the same
# `build_css` helpers `TldwCli` uses.

from __future__ import annotations

from pathlib import Path

from textual.app import App

from tldw_chatbook.css import build_css

CSS_DIR = Path(build_css.__file__).parent

#: The app bundle, for harnesses that want the app-CSS tier as well.
BUNDLED_STYLESHEET = CSS_DIR / "tldw_cli_modular.tcss"


class ConsolidatedCSSApp(App):
    """An ``App`` that loads the consolidated widget CSS, as the real app does."""

    CSS_PATH = [str(path) for path in build_css.screen_css_paths(CSS_DIR)]

    def _get_default_css(self):  # noqa: D102 - see module docstring
        return build_css.widget_defaults_sources(CSS_DIR) + super()._get_default_css()
