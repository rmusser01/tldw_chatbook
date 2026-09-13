"""DetailValueRow + DetailGroup: reusable label/value detail-pane row grammar.

schedules-redesign PR-1, Task 1 (`.superpowers/sdd/plan-2026-09-03-
schedules-redesign-pr1/task-1-brief.md`, the approved spec's section 5 row
grammar). A row renders one field as ``label`` (muted, left) and ``value``
(right-aligned, ellipsized rather than wrapped when it overflows). An
optional dimmed ``▾`` affordance glyph marks a row whose value is editable.
A hidden error-line slot below the row is implemented and tested
(``show_error``/``clear_error``) but unused by any PR-1 caller; PR-3 wires a
caller to it.

schedules-redesign PR-3, Task 1 activates the three dormant PR-1 seams
below: a row with ``affordance=True`` now posts ``DetailValueRow.Activated``
on click or Enter, but only while no editor is open -- a dormant row
(``affordance`` left at its ``False`` default, as every PR-1/PR-2 caller
still does) stays a complete no-op, exactly as before. ``begin_edit``/
``end_edit`` swap the read-only value ``Static`` for a caller-built editor
widget in place (no recompose); ``begin_edit`` is a guarded no-op while an
editor is already open, and the error slot coexists with an open editor.
The affordance glyph also gains a live visual state -- it un-dims while the
row (or its open editor) has focus, or on hover.

schedules-redesign PR-3, Task 3 adds one more general behavior: Escape
while an editor is open closes it via ``end_edit()`` without notifying the
caller (no ``Activated`` re-fire, no commit) -- a plain cancel, available
to every `DetailValueRow` consumer, not just the reminder pane that first
wires a caller through it.

Three PR-3 seams were left dormant in PR-1 (final review F13, fixed while
the row had only two consumers) and are wired up starting with PR-3 Task 1
above: ``affordance`` is a settable property, not a construction-only flag
(the glyph is always mounted and toggled with ``display``, so flipping a row
read-only<->editable never means a remount); ``row_key`` gives the row an
identity of its own, carried on ``Activated``; and ``can_focus`` is a
constructor flag (default ``False``, as every PR-1/PR-2 caller still leaves
it) so spec §12's Up/Down row traversal needs no subclass.

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
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Collapsible, Static


def _literal(value: str | Text) -> Text:
    """Return ``value`` as a ``Text`` that will render with no markup parsing."""
    return value if isinstance(value, Text) else Text(str(value))


class DetailValueRow(Vertical):
    """One label/value detail-pane field, with an editable affordance and error slot.

    ``affordance=True`` makes the row post ``Activated`` on click OR Enter,
    but Enter only ever reaches a widget that has focus: a caller that wants
    an editable row to be keyboard-reachable (not just clickable) must also
    pass ``can_focus=True`` (review finding 2, task-1-review.md) -- the two
    flags are independent by design, so ``affordance=True`` alone is a
    click-only, keyboard-unreachable row.
    """

    class Activated(Message):
        """Posted when the row's affordance is triggered by click or Enter.

        Only posted while ``affordance`` is true and no editor is
        currently open (``begin_edit`` not yet called, or ``end_edit``
        already closed it) -- a dormant row (``affordance`` left at its
        ``False`` default) never posts this message, so every PR-1/PR-2
        read-only consumer stays untouched.

        Attributes:
            row: The ``DetailValueRow`` that was activated. Carries its own
                ``row_key`` for a handler to route on.
        """

        def __init__(self, row: "DetailValueRow") -> None:
            self.row = row
            super().__init__()

    def __init__(
        self,
        label: str,
        value: str | Text,
        *,
        affordance: bool = False,
        value_id: str | None = None,
        row_key: str | None = None,
        can_focus: bool = False,
        tooltip: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._initial_value = value
        self._affordance = affordance
        self._value_id = value_id
        #: Hover explanation for the value (31712 AC#1): a permanently
        #: read-only row with no affordance glyph otherwise looks like a
        #: silently-inconsistent sibling of an editable row of the same
        #: name elsewhere in the pane -- a caller passes this to say why.
        #: Applied to the value `Static` in `compose()` below (Textual
        #: resolves a hovered widget's OWN `tooltip`, never an ancestor's).
        self._tooltip_text = tooltip
        self._error_id = f"{value_id}-error" if value_id else None
        self._value_static: Static | None = None
        self._error_static: Static | None = None
        self._affordance_static: Static | None = None
        self._line: Horizontal | None = None
        #: The open editor widget, if any (PR-3 Task 1's edit-swap API).
        #: ``None`` means the row is showing its read-only value.
        self._editor: Widget | None = None
        #: Stable field identity for the row itself (final-review F13.2).
        #: PR-3 addresses the ROW (open its editor, route its error) and
        #: must not reach through `static.parent.parent` to find it.
        self.row_key = row_key
        #: Focusable-ready (final-review F13.3): spec §12 wants Up/Down
        #: traversal of detail rows. `Vertical.can_focus` is False and
        #: every PR-1 caller leaves it there; PR-3 flips it per row.
        self.can_focus = can_focus

    def compose(self) -> ComposeResult:
        line = Horizontal(classes="detail-value-row-line")
        self._line = line
        with line:
            yield Static(self._label, classes="detail-value-row-label", markup=False)
            self._value_static = Static(
                _literal(self._initial_value),
                classes="detail-value-row-value",
                markup=False,
                id=self._value_id,
            )
            if self._tooltip_text:
                self._value_static.tooltip = self._tooltip_text
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
        """Refresh the painted value in place -- no recompose.

        A no-op while an editor is open (PR-3 final review I2): the panes
        that own these rows repaint on a timer, and the value region
        belongs to the editor for as long as it is mounted. The write
        would land on the hidden `Static` and only surface as a flash of
        a value the user did not choose, the moment `end_edit` restores
        it. A repaint that changes the ROW's identity closes the editor
        first (`TaskDetail._reset_row_editing` and its `DefinitionDetail`
        twin), so the next `update_value` after that lands normally.
        """
        assert self._value_static is not None, "update_value called before mount"
        if self._editor is not None:
            return
        self._value_static.update(_literal(value), layout=False)

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

    def _on_click(self, event: events.Click) -> None:
        """Activate on click, per PR-3 Task 1's edit-swap API (module docstring).

        A dormant row (``affordance`` False) never intercepts the click --
        it bubbles up untouched, same as before this row had a handler at
        all. Only the activation case (no editor open yet) stops the event
        and posts ``Activated``. Once an editor is open the row must NOT
        stop the event -- review finding 1 (task-1-review.md): Textual only
        resolves a BINDINGS-driven editor's own key/click handling once the
        raw event bubbles unstopped up to ``App``, so an unconditional stop
        here silently ate a mounted ``Select``'s Enter-to-open even though
        that is `begin_edit`'s own named typical case. While editing, the
        event passes through untouched and the editor/App own it.
        """
        if not self._affordance or self._editor is not None:
            return
        event.stop()
        self.post_message(self.Activated(self))

    def _on_key(self, event: events.Key) -> None:
        """Activate on Enter, cancel an open editor on Escape, or
        traverse to the next/previous row on Up/Down.

        PR-3 Task 3: Escape while editing closes the editor without
        committing (``end_edit()``, no caller notified) -- checked FIRST
        and independently of ``self._affordance`` so it works the same
        whether or not the row's glyph is currently shown. Neither
        `Select` nor `Input` binds Escape for anything while collapsed/
        idle (verified against Textual's own `Select`/`Input`/
        `SelectOverlay` BINDINGS -- only `SelectOverlay` binds it, and
        only to dismiss its own open dropdown), so the raw key event
        bubbles here untouched exactly like Task 1's Enter fix relies on
        for the OPEN case; stopping it here for the EDITING case is safe
        and deliberate, unlike the open-Select-overlay's own Enter, which
        must keep bubbling to resolve via `App`'s BINDINGS chain.

        redesign PR-4, task 4 (spec §12's "Up/Down traverse detail rows
        when the pane has focus"): guarded by ``self._editor is None``,
        the SAME editor-open-ownership rule Enter/Escape already use --
        an open editor (e.g. a mounted `Select`, which binds Up/Down
        itself to `show_overlay`) owns the arrow keys, not the row. When
        no editor is open, Up/Down move focus to the previous/next
        `DetailValueRow` using Textual's OWN focus-chain machinery
        (`Screen.focus_previous`/`focus_next`, selector-scoped to this
        widget class) rather than a hand-rolled row registry: `focus_
        chain` already excludes non-focusable rows (`can_focus=False`)
        and anything inside a `display:none` ancestor -- a hidden sibling
        pane (`.pane-hidden`) or a collapsed `DetailGroup` -- for free, so
        traversal is naturally scoped to THIS pane's currently-visible,
        focusable rows without this widget needing to know its own
        container. Wraps at the ends (`_move_focus`'s own modulo
        wrap-around) rather than stopping -- picked over stopping since
        it is what the native mechanism gives for free and is the more
        common list-traversal convention.
        """
        if event.key == "escape" and self._editor is not None:
            event.stop()
            event.prevent_default()
            self.end_edit()
            return
        if event.key in ("up", "down") and self._editor is None:
            event.stop()
            event.prevent_default()
            if event.key == "up":
                self.screen.focus_previous(DetailValueRow)
            else:
                self.screen.focus_next(DetailValueRow)
            return
        if event.key != "enter" or not self._affordance or self._editor is not None:
            return
        event.stop()
        event.prevent_default()
        self.post_message(self.Activated(self))

    def begin_edit(self, editor: Widget) -> None:
        """Swap the read-only value for ``editor``, in place, and focus it.

        Hides the value ``Static`` and mounts ``editor`` where it sat, no
        recompose. Guarded: a no-op while an editor is already open -- one
        editor at a time. The error slot (`show_error`/`clear_error`) is
        untouched and keeps working alongside an open editor.

        Args:
            editor: The widget to mount in the value's place -- typically a
                ``Select`` or a small composite editor built by the caller.
        """
        if self._editor is not None:
            return
        assert self._value_static is not None and self._line is not None, (
            "begin_edit called before mount"
        )
        self._editor = editor
        self._value_static.styles.display = "none"
        self._line.mount(editor, before=self._value_static)
        self.call_after_refresh(editor.focus)

    def end_edit(self, *, restore_focus: bool = True) -> None:
        """Close the open editor and restore the read-only value display.

        A no-op if no editor is currently open.

        Args:
            restore_focus: When ``True`` (default), give focus back to the
                row itself once the editor is removed.
        """
        if self._editor is None:
            return
        editor, self._editor = self._editor, None
        editor.remove()
        assert self._value_static is not None, "end_edit called before mount"
        self._value_static.styles.display = "block"
        if restore_focus:
            self.call_after_refresh(self.focus)


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

    def __init__(self, *children: Widget, title: str, collapsed: bool = False, **kwargs: object) -> None:
        super().__init__(*children, title=title, collapsed=collapsed, **kwargs)
