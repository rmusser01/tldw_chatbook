"""Searchable multi-select over character cards.

Receives already-fetched rows rather than opening a database: cards live in
``ChaChaNotes_DB`` while this slice's own handle is ``EvalsDB``, and keeping
the fetch outside means this widget is testable without either -- see
``evals_state.py``'s ``EvalsViewModel.character_cards()`` for the read side
that supplies these rows.

Card ids are INTEGERs (``character_cards.id``) while every eval id in this
slice is TEXT. They are deliberately never normalised to strings -- the
engine's ``CharacterProbeConfig`` (``Evals/character_probe/models.py``)
rejects any non-``int`` element of ``character_ids`` at construction, so
stringifying an id here would only move that failure somewhere harder to
diagnose.

**Rebuilding the row list must not touch the search box.** An earlier draft
tore down and remounted this whole widget's ``compose()`` output --
including the ``#evals-card-search`` `Input` -- on every keystroke, which
drops focus and the character just typed before the next one lands. The row
list instead lives in its own nested container (``#evals-card-picker-rows``)
that ``_refresh_rows`` clears and refills; the search `Input` is mounted
exactly once, in ``compose()``, and never removed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Button, Input, Static

#: Debounce for the search `Input` -- mirrors the console picker family's
#: 0.2 s shape (`console_prompt_picker_modal.py`). A full refresh removes
#: and remounts up to `evals_state.py`'s ``_LIST_LIMIT`` (500) card rows,
#: which should not happen on every keystroke (task-15476).
SEARCH_DEBOUNCE_SECONDS = 0.2


class CardRow(Button):
    """One selectable card. The label is a pre-built Rich ``Text`` (not a
    markup string) so a card name containing bracket markup (``"Vex[/]v2"``)
    renders literally instead of being parsed -- Textual's ``Content.
    from_text`` special-cases a ``Text`` instance via ``from_rich_text``,
    bypassing markup parsing entirely (confirmed against the installed
    Textual: ``textual.widgets._button.Button.validate_label`` ->
    ``Content.from_text``)."""

    def __init__(self, card_id: int, card_name: str, selected: bool, index: int) -> None:
        self.card_id = card_id
        self.card_name = card_name
        self._selected = selected
        super().__init__(
            self._compose_label(card_name, selected),
            id=f"evals-card-row-{index}",
            classes="evals-card-row",
            compact=True,
        )

    @staticmethod
    def _compose_label(card_name: str, selected: bool) -> Text:
        return Text(f"{'✓' if selected else ' '} {card_name}")

    def render_label(self) -> Text:
        """The row's rendered label -- used both by ``set_selected`` (to
        refresh ``self.label`` after a toggle) and by tests asserting a
        markup-hazard name rendered literally."""
        return self._compose_label(self.card_name, self._selected)

    def set_selected(self, selected: bool) -> None:
        """Update this row's own check-glyph in place. Never touches the
        picker's row list -- toggling one row does not need (and must not
        trigger) a rebuild of every other row."""
        self._selected = selected
        self.label = self.render_label()


class CardPicker(Vertical):
    """Search box plus one toggle row per matching card.

    Mounts a fixed ``#evals-card-search`` `Input` once, and a nested
    ``#evals-card-picker-rows`` container holding the current rows (or an
    empty-state `Static`) -- only the latter is torn down and rebuilt when
    the search filter changes, so the `Input` never loses focus or its
    in-progress text.

    ``#evals-card-picker-rows`` is bounded and independently scrollable in
    ``_evals.tcss`` (``max-height: 10`` / ``overflow-y: auto``) so a list up
    to ``evals_state.py``'s ``_LIST_LIMIT`` (500 cards) stays reachable by
    scrolling rather than silently clipped, without this widget itself
    growing to consume unbounded screen space.
    """

    class SelectionChanged(Message, namespace="card_picker"):
        """Posted whenever the selected set changes.

        Args:
            selected_ids: Every currently-selected card id, in card order.
        """

        def __init__(self, selected_ids: tuple[int, ...]) -> None:
            self.selected_ids = selected_ids
            super().__init__()

    def __init__(
        self,
        cards: Sequence[Mapping[str, Any]],
        selected_ids: Sequence[int] = (),
        **kwargs: Any,
    ) -> None:
        # `.evals-card-picker` is always applied, regardless of whatever a
        # caller's own `classes=` supplies -- `_evals.tcss` targets this
        # class (not a bare `CardPicker` type selector; no other editor
        # widget in this file uses one) to bound this widget's height to
        # its own content.
        caller_classes = kwargs.pop("classes", "")
        merged_classes = " ".join(
            part for part in ("evals-card-picker", caller_classes) if part
        )
        super().__init__(classes=merged_classes, **kwargs)
        self._cards = [dict(card) for card in cards]
        self._selected: set[int] = {int(cid) for cid in selected_ids}
        self._filter = ""
        self._refresh_timer: Timer | None = None

    def selected_ids(self) -> tuple[int, ...]:
        """Selected card ids, in the order the cards were supplied.

        Returns:
            tuple[int, ...]: Ids of every selected card, including any
            currently filtered out of view -- filtering hides rows, it
            never deselects.
        """
        return tuple(
            int(card["id"]) for card in self._cards if int(card["id"]) in self._selected
        )

    def _visible(self) -> list[dict[str, Any]]:
        needle = self._filter.strip().lower()
        if not needle:
            return self._cards
        return [c for c in self._cards if needle in str(c.get("name") or "").lower()]

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search characters", id="evals-card-search")
        with Vertical(id="evals-card-picker-rows", classes="evals-card-picker-rows"):
            yield from self._compose_rows()

    def _compose_rows(self) -> ComposeResult:
        """The row list's content for the CURRENT filter -- shared by
        ``compose()`` (first mount) and ``_refresh_rows`` (every
        subsequent search change), so the two can never drift apart into
        two different renderings of the same state."""
        if not self._cards:
            yield Static(
                "No character cards yet — create one in Roleplay first.",
                id="evals-card-picker-empty",
                markup=False,
            )
            return
        visible = self._visible()
        if not visible:
            yield Static(
                "No character cards match your search.",
                id="evals-card-picker-no-matches",
                markup=False,
            )
            return
        for index, card in enumerate(visible):
            yield CardRow(
                int(card["id"]),
                str(card.get("name") or ""),
                int(card["id"]) in self._selected,
                index,
            )

    async def _refresh_rows(self) -> None:
        """Rebuild ONLY ``#evals-card-picker-rows``'s children. The search
        `Input` is a sibling of this container (see ``compose()``), never a
        descendant of it, so it is never part of what gets torn down here --
        this is the fix for the dropped-keystroke defect described in this
        module's own docstring."""
        rows_container = self.query_one("#evals-card-picker-rows")
        await rows_container.remove_children()
        await rows_container.mount_all(list(self._compose_rows()))

    def on_input_changed(self, event: Input.Changed) -> None:
        """Re-filters the row list as the search box is typed into.

        Guarded on ``#evals-card-search``'s own id even though it is the
        only `Input` this widget currently mounts, so a future sibling
        `Input` added to this widget can never be mistaken for the search
        box. ``event.stop()`` once that guard passes: this widget owns
        searching its own cards end to end, so the event must not keep
        bubbling past it to ``CharacterBenchEditor`` (which mounts this
        picker) or any other ancestor that has no business reacting to a
        keystroke in this search box.

        Args:
            event: The search `Input`'s own ``Changed`` message; only its
                ``input.id``/``value`` are read.
        """
        if event.input.id != "evals-card-search":
            return
        event.stop()
        self._filter = event.value
        # Debounced (task-15476): with up to `_LIST_LIMIT` (500) cards, a
        # full remove_children()+mount_all() of the row list on every
        # keystroke is the exact defect this task fixes. `call_after_refresh`
        # alone (the previous approach) only deferred one frame -- it still
        # rebuilt on every keystroke, just slightly later. A 0.2 s timer that
        # re-arms on each keystroke settles once typing pauses instead.
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_timer(
            SEARCH_DEBOUNCE_SECONDS, self._debounced_refresh_rows
        )

    def _debounced_refresh_rows(self) -> None:
        self._refresh_timer = None
        # `call_after_refresh`, not an immediate `await` here (this runs
        # from a timer callback, not the Input's own Changed handler): lets
        # Textual finish any in-flight layout pass before the row list
        # underneath the search box is torn down and remounted.
        self.call_after_refresh(self._refresh_rows)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Toggles a card row's selection when ITS button is pressed.

        Guarded on ``isinstance(row, CardRow)`` since ``Button.Pressed``
        bubbles from any button, not only a card row -- there are none
        today (this widget mounts only the search `Input` and its own
        `CardRow` buttons; see ``compose()``), but the guard keeps this
        handler correct if one is ever added. ``event.stop()`` once that
        guard passes: a card-row selection is this widget's own concern,
        so once handled here the event must not keep bubbling to
        ``CharacterBenchEditor`` or any other ancestor.

        Args:
            event: The pressed button's own message; ``event.button`` is
                checked against ``CardRow`` to find the selected card.
        """
        row = event.button
        if not isinstance(row, CardRow):
            return
        event.stop()
        if row.card_id in self._selected:
            self._selected.discard(row.card_id)
        else:
            self._selected.add(row.card_id)
        # Only this one row's own glyph is refreshed -- selecting a card
        # must not rebuild (and so must not disturb) every other row or the
        # search box.
        row.set_selected(row.card_id in self._selected)
        self.post_message(self.SelectionChanged(self.selected_ids()))
