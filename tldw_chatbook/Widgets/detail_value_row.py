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

CSS for both classes lives in ``css/features/_scheduling.tcss`` (fix round
1: not ``BUNDLED_CSS`` on the class, as first shipped), per the brief's
file list, with plain type/class selectors (``DetailValueRow``,
``DetailGroup``, ``.detail-value-row-*``) rather than a
``scheduling-``-prefixed ones, so the widgets stay reusable outside
Scheduling -- graduate them to a shared features file if/when a
non-scheduling consumer appears. That module is concatenated into the
app's boot-loaded CSS bundle (``tldw_cli_modular.tcss``, via
``build_css.py``'s ``CSS_MODULES``) in the same parse pass as
``css/core/_variables.tcss``, so ``$ds-*`` design tokens resolve directly
there with no alias workaround (the widget-defaults ``BUNDLED_CSS`` tier
this widget used at first is parsed as a separate stylesheet source,
*before* ``_variables.tcss`` -- a bare ``$ds-*`` reference there is an
unresolved-variable error). Run ``python tldw_chatbook/css/build_css.py``
after editing ``_scheduling.tcss`` and commit the regenerated
``tldw_cli_modular.tcss`` alongside the source.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Collapsible, Static


def _literal(value: str | Text) -> Text:
    """Return ``value`` as a ``Text`` that will render with no markup parsing."""
    return value if isinstance(value, Text) else Text(str(value))


class DetailValueRow(Vertical):
    """One label/value detail-pane field, plus a dormant affordance and error slot."""

    def __init__(
        self,
        label: str,
        value: str | Text,
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

    def update_value(self, value: str | Text) -> None:
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
    press Enter while it has focus) to toggle; the chevron and ``.collapsed``
    reactive come for free from the base class.

    **``title`` is keyword-only** (``DetailGroup(*children, title, collapsed
    = False)``), unlike the brief's originally-drafted fully-positional
    ``DetailGroup(title, *, collapsed=False)`` -- a direct, unavoidable
    consequence of subclassing ``Collapsible``, whose own ``__init__`` is
    ``(*children, title="Toggle", collapsed=True, ...)``: with ``*children``
    first, ``title`` cannot also be positional. Call it as
    ``DetailGroup(row_1, row_2, title="Schedule")``, never
    ``DetailGroup("Schedule", row_1)`` -- the latter is accepted by the
    signature but silently treats the title string as a child widget
    instead. Rows may also be added with the ``with``-block compose idiom
    (``Collapsible.compose_add_child`` routes them into the body either
    way).
    """

    def __init__(self, *children, title: str, collapsed: bool = False, **kwargs) -> None:
        super().__init__(*children, title=title, collapsed=collapsed, **kwargs)
