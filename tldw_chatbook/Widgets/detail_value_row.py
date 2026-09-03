"""DetailValueRow + DetailGroup: reusable label/value detail-pane row grammar.

schedules-redesign PR-1, Task 1 (`.superpowers/sdd/plan-2026-09-03-
schedules-redesign-pr1/task-1-brief.md`, the approved spec's section 5 row
grammar). A row renders one field as ``label`` (muted, left) and ``value``
(right-aligned, ellipsized rather than wrapped when it overflows). An
optional dimmed ``▾`` affordance glyph marks a row whose value will become
interactive in a later PR -- it is purely decorative here, not clickable or
focusable. A hidden error-line slot below the row is implemented and tested
(``show_error``/``clear_error``) but unused by any PR-1 caller; PR-3 wires a
caller to it.

``DetailGroup`` is a thin ``Collapsible`` subclass (the house idiom already
used directly across this codebase -- see
``redesign-pr1-survey.md`` section 3 -- rather than a bespoke toggle-button
section like ``ConsoleInspectorSection``, which exists for a different row
shape). It gains click/Enter-to-toggle and the chevron for free from
``Collapsible`` itself.

Values may be a plain ``str`` or a pre-built ``rich.text.Text``. Both render
literally: a ``str`` is never interpreted as Rich markup. Callers that want
markup build their own ``Text`` and pass it in (last program's escaping
lesson -- server/user-derived strings must never be treated as markup).

CSS lives as ``BUNDLED_CSS`` on these classes rather than a ``.tcss``
source file: reusable, non-screen-specific ``Widgets/`` components style
themselves this way in this codebase (``TaskDetail``'s own label/value rows
are the precedent). Run ``python tldw_chatbook/css/build_css.py`` after
editing either class's ``BUNDLED_CSS`` block and commit the regenerated
``css/widget_defaults_self.tcss`` / ``css/widget_defaults_scoped.tcss``
alongside the source -- the app only ever reads those generated files, not
the class attribute directly (`Tests/UI/consolidated_css.py`).
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Collapsible, Static


def _literal(value: "str | Text") -> Text:
    """Return ``value`` as a ``Text`` that will render with no markup parsing."""
    return value if isinstance(value, Text) else Text(str(value))


class DetailValueRow(Vertical):
    """One label/value detail-pane field, plus a dormant affordance and error slot."""

    BUNDLED_CSS = """
    /* Local pass-through of the $ds-* tokens this block uses, aliased to
       the exact same underlying values `css/core/_variables.tcss` defines.
       The widget-defaults CSS tier (BUNDLED_CSS, TASK-15450) is parsed as
       its own stylesheet source before the app bundle that defines the
       real $ds-* tokens, so a bare reference is an unresolved-variable
       parse error in *any* host, including the real app -- see
       `TldwCli.CSS_PATH`'s own comment on this exact hazard. Not a
       distinct palette (contrast `PromptBlockEditor`'s intentionally
       shadowed one); keep these identical to their source of truth. */
    $ds-text-muted: $text-muted;
    $ds-text-primary: $text;
    $ds-text-disabled-readable: #8a8a8a;
    $ds-status-error-readable: #ff8fa3;

    DetailValueRow {
        height: auto;
        min-height: 0;
    }

    DetailValueRow .detail-value-row-line {
        height: 1;
        min-height: 1;
    }

    DetailValueRow .detail-value-row-label {
        color: $ds-text-muted;
        width: auto;
        min-width: 0;
        height: 1;
        padding: 0 1 0 0;
    }

    DetailValueRow .detail-value-row-value {
        color: $ds-text-primary;
        width: 1fr;
        min-width: 0;
        height: 1;
        text-align: right;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    DetailValueRow .detail-value-row-affordance {
        color: $ds-text-disabled-readable;
        width: 2;
        min-width: 2;
        height: 1;
        text-align: right;
        padding: 0 0 0 1;
    }

    DetailValueRow .detail-value-row-error {
        color: $ds-status-error-readable;
        width: 1fr;
        height: 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    def __init__(
        self,
        label: str,
        value: "str | Text",
        *,
        affordance: bool = False,
        value_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._initial_value = value
        self._affordance = affordance
        self._value_id = value_id
        self._error_id = f"{value_id}-error" if value_id else None
        self._value_static: Static | None = None
        self._error_static: Static | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="detail-value-row-line"):
            yield Static(self._label, classes="detail-value-row-label", markup=False)
            self._value_static = Static(
                _literal(self._initial_value),
                classes="detail-value-row-value",
                markup=False,
                id=self._value_id,
            )
            yield self._value_static
            if self._affordance:
                yield Static(
                    "▾", classes="detail-value-row-affordance", markup=False
                )
        self._error_static = Static(
            "", classes="detail-value-row-error", markup=False, id=self._error_id
        )
        self._error_static.styles.display = "none"
        yield self._error_static

    def update_value(self, value: "str | Text") -> None:
        """Refresh the painted value in place -- no recompose."""
        assert self._value_static is not None, "update_value called before mount"
        self._value_static.update(_literal(value))

    def show_error(self, msg: str) -> None:
        """Reveal the hidden error-line slot with ``msg``, rendered literally."""
        assert self._error_static is not None, "show_error called before mount"
        self._error_static.update(_literal(msg))
        self._error_static.styles.display = "block"

    def clear_error(self) -> None:
        """Hide the error-line slot again and drop its text."""
        assert self._error_static is not None, "clear_error called before mount"
        self._error_static.styles.display = "none"
        self._error_static.update("")


class DetailGroup(Collapsible):
    """Titled, collapsible container composing ``DetailValueRow`` children.

    A thin subclass of Textual's own ``Collapsible`` -- click the title (or
    press Enter while it has focus) to toggle. Pass rows as constructor
    children, e.g. ``DetailGroup(DetailValueRow(...), title="Schedule")``,
    or with the ``with``-block compose idiom.
    """

    BUNDLED_CSS = """
    /* Local pass-through fallback -- see `DetailValueRow.BUNDLED_CSS`. */
    $ds-surface-panel: $panel;
    $ds-grid-line: $surface-lighten-1;

    DetailGroup {
        background: $ds-surface-panel;
        border-top: hkey $ds-grid-line;
    }
    """

    def __init__(self, *children, title: str, collapsed: bool = False, **kwargs) -> None:
        super().__init__(*children, title=title, collapsed=collapsed, **kwargs)
        self.add_class("detail-group")
