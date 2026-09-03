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

Three PR-3 seams are dormant in PR-1 (final review F13, fixed while the row
had only two consumers): ``affordance`` is a settable property, not a
construction-only flag (the glyph is always mounted and toggled with
``display``, so flipping a row read-only<->editable never means a remount);
``row_key`` gives the row an identity of its own to carry on whatever
message PR-3 posts; and ``can_focus`` is a constructor flag (default
``False``, as every PR-1 caller leaves it) so spec §12's Up/Down row
traversal needs no subclass.

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
        row_key: str | None = None,
        can_focus: bool = False,
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
        self._affordance_static: Static | None = None
        #: Stable field identity for the row itself (final-review F13.2).
        #: PR-3 addresses the ROW (open its editor, route its error) and
        #: must not reach through `static.parent.parent` to find it.
        self.row_key = row_key
        #: Focusable-ready (final-review F13.3): spec §12 wants Up/Down
        #: traversal of detail rows. `Vertical.can_focus` is False and
        #: every PR-1 caller leaves it there; PR-3 flips it per row.
        self.can_focus = can_focus

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
            # Always mounted, shown/hidden by the `affordance` property
            # (final-review F13.1): PR-3 flips a row between read-only and
            # editable in place, and rebuilding the row would mean a
            # remount -- both consumers hold hard refs assigned in their
            # own `compose()`.
            self._affordance_static = Static(
                "▾", classes="detail-value-row-affordance", markup=False
            )
            self._affordance_static.styles.display = (
                "block" if self._affordance else "none"
            )
            yield self._affordance_static
        self._error_static = Static(
            "", classes="detail-value-row-error", markup=False, id=self._error_id
        )
        self._error_static.styles.display = "none"
        yield self._error_static

    @property
    def affordance(self) -> bool:
        """Whether the ``▾`` affordance glyph is showing."""
        return self._affordance

    @affordance.setter
    def affordance(self, value: bool) -> None:
        self._affordance = bool(value)
        if self._affordance_static is not None:
            self._affordance_static.styles.display = (
                "block" if self._affordance else "none"
            )

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
