# Character Probe Evals — Phase 2 (Authoring UI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a character-probe bench creatable and runnable entirely through the UI — import a probe set, pick character cards, create the bench, and run it — so phase 1's engine becomes reachable without a Python prompt.

**Architecture:** Extends the existing Evals slice rather than adding a screen. The Catalog rail gains probe-set import and a "+ New character bench" affordance; the detail pane gains a character-probe editor (distinct from the word-bench editor, selected by `bench_type`); the existing primary action runs it. Review and summary are Phase 3 and 4 — this phase stops when a run produces conversations.

**Tech Stack:** Python ≥3.11, Textual, pytest. Engine from phase 1 (`tldw_chatbook/Evals/character_probe/`), unchanged by this phase except where noted.

## Global Constraints

Copied from `Docs/superpowers/specs/2026-08-01-character-probe-eval-design.md` and phase 1's established conventions. Every task's requirements implicitly include this section.

- **Both bench types share the rail's Benches section, so a character-probe bench needs a visible marker** distinguishing it from a word bench at a glance — without one, selecting a bench is a guess about which detail surface appears.
- **The two bench types never share a detail surface.** Selecting a character-probe bench renders its own editor in the slot the word-bench editor occupies.
- **No logprobs / top-K / normalizer / canary vocabulary** anywhere in character-probe UI. That vocabulary judges distributions; this eval reads generated text. A degenerate-canary warning must never appear on this bench type.
- **Cost is shown before running.** Total calls = cards × probes × targets × samples × turns-per-probe. The Estimate must reflect it before Run is pressed.
- **User-authored text is a markup hazard.** Any Static carrying a card name, probe text, or model output takes `markup=False`; any Button label or tooltip interpolating it uses `escape_markup`. Whitespace shown with the `␣` convention via `snippet_editor.render_snippet_cell`; newlines guarded to one line with `⏎`.
- **Fail loudly, never silently default** — the engine's convention. A corrupt row or missing record raises a named error identifying it; a write affecting no rows raises rather than reporting success.
- **`character_ids` are ints** (`character_cards.id`); every eval id is a str. Do not normalise them.
- **Tests must drive real widgets.** Setting a widget's `.value` programmatically and asserting it "works" is what let phase 1 ship a checkbox no user could toggle and steering that never reached the model. Every behavioural UI test presses or types through `pilot`, and asserts the persisted result, not the widget.
- **Painted geometry is the arbiter.** This pane has pushed a control out of reach three times (task-1764). Any task adding rows asserts the controls below it stay hit-testable — `screen.get_widget_at(*control.region.center)` resolves to the control — at 160x45 AND 235x52.
- Google-style docstrings (Args/Returns/Raises) on public callables; parameterized SQL only; CSS in `css/features/_evals.tcss` regenerated via `build_css.py`, never hand-edited.
- Run tests foreground: `/private/tmp/tldw-venv/bin/python -m pytest <paths> -p no:randomly` from the clone root. Never `-q`.

## File Structure

- `tldw_chatbook/UI/Evals/library_rail.py` — add probe-set import and the "+ New character bench" affordance to the existing Datasets and Benches sections; mark probe-set dataset rows and character-probe bench rows.
- `tldw_chatbook/UI/Evals/character_bench_editor.py` (new) — the character-probe detail pane: name/description, probe set, card selection, targets, sampler, and the read-only probe listing. Sibling to `bench_editor.py`, never merged into it.
- `tldw_chatbook/UI/Evals/card_picker.py` (new) — searchable multi-select over `character_cards`, used by the editor. Its own file because it crosses a DB boundary and will be reused by Phase 3.
- `tldw_chatbook/UI/Screens/evals_screen.py` — route `bench_type` to the right detail pane; own the cross-DB handle; wire the run action.
- `tldw_chatbook/UI/Evals/evals_state.py` — read-side helpers for probe sets and character benches.
- Tests mirror each under `Tests/UI/`.

---

### Task 1: Rail marks probe sets and character benches

**Files:**
- Modify: `tldw_chatbook/UI/Evals/library_rail.py`
- Modify: `tldw_chatbook/UI/Evals/evals_state.py`
- Test: `Tests/UI/test_evals_empty_states.py`

**Interfaces:**
- Consumes: `character_probe.storage.is_probe_set(dataset_row)`, `is_character_bench(task_row)` (phase 1, already on dev).
- Produces: `EvalsViewModel.character_benches() -> list[dict]`, `EvalsViewModel.probe_sets() -> list[dict]`; rail row labels prefixed `"◆ "` for character-probe benches and probe-set datasets.

Both bench types share the Benches section and both dataset types share Datasets. Today a probe set and a snippet set look identical in the rail, and selecting either kind of bench is a guess about which detail pane appears. A single-width marker glyph is the minimum honest fix.

- [ ] **Step 1: Write the failing test**

```python
def test_a_character_bench_row_is_marked_in_the_rail(evals_app, evals_db):
    from tldw_chatbook.Evals.character_probe.models import CharacterProbeConfig
    from tldw_chatbook.Evals.character_probe.storage import save_character_bench

    save_character_bench(
        evals_db,
        CharacterProbeConfig(
            name="villain probes",
            probe_set_id="ps-1",
            character_ids=(1,),
            target_ids=("t-1",),
        ),
    )
    async def _check(pilot):
        rail = pilot.app.screen.query_one(LibraryRail)
        labels = [b.label.plain for b in rail.query(Button) if "evals-rail-row" in b.classes]
        assert any(label.startswith("◆ ") and "villain probes" in label for label in labels)
    run_evals(evals_app, _check)


def test_a_word_bench_row_is_not_marked(evals_app, seeded_bench):
    async def _check(pilot):
        rail = pilot.app.screen.query_one(LibraryRail)
        labels = [b.label.plain for b in rail.query(Button) if "evals-rail-row" in b.classes]
        word_rows = [l for l in labels if "loaded-nouns" in l]
        assert word_rows and not any(l.startswith("◆ ") for l in word_rows)
    run_evals(evals_app, _check)


def test_a_probe_set_dataset_row_is_marked(evals_app, evals_db):
    from tldw_chatbook.Evals.character_probe.models import Probe, ProbeSet
    from tldw_chatbook.Evals.character_probe.storage import save_probe_set

    save_probe_set(evals_db, "starter probes", ProbeSet(probes=(Probe(turns=("Hi",)),)))
    async def _check(pilot):
        rail = pilot.app.screen.query_one(LibraryRail)
        labels = [b.label.plain for b in rail.query(Button) if "evals-rail-row" in b.classes]
        assert any(label.startswith("◆ ") and "starter probes" in label for label in labels)
    run_evals(evals_app, _check)


def test_the_marker_glyph_is_single_width():
    """A double-width glyph would shift every rail row's alignment."""
    from rich.cells import cell_len
    from tldw_chatbook.UI.Evals.library_rail import CHARACTER_PROBE_MARKER
    assert cell_len(CHARACTER_PROBE_MARKER.strip()) == 1
```

Use the file's existing `evals_app`/`evals_db`/`seeded_bench` fixtures and its `run_evals` helper — grep the top of `Tests/UI/test_evals_empty_states.py` and reuse them verbatim rather than inventing new ones.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_empty_states.py -k "marked or marker_glyph" -p no:randomly`
Expected: FAIL — `ImportError: cannot import name 'CHARACTER_PROBE_MARKER'`

- [ ] **Step 3: Write minimal implementation**

In `evals_state.py`, beside the existing `benches()`/`datasets()`:

```python
    def character_benches(self) -> list[dict[str, Any]]:
        """Character-probe benches: ``eval_tasks`` rows tagged
        ``bench_type == "character_probe"``.

        Returns:
            list[dict[str, Any]]: Matching rows, newest first, or an empty
            list when the evaluation service is unavailable.
        """
        from ...Evals.character_probe.storage import is_character_bench

        return [task for task in self._all_tasks() if is_character_bench(task)]

    def probe_sets(self) -> list[dict[str, Any]]:
        """Datasets holding probes rather than snippets.

        Returns:
            list[dict[str, Any]]: Matching dataset rows, or an empty list
            when the evaluation service is unavailable.
        """
        from ...Evals.character_probe.storage import is_probe_set

        return [row for row in self.datasets() if is_probe_set(row)]
```

In `library_rail.py`, near the other row-label helpers:

```python
#: Prefixes a rail row whose bench or dataset belongs to the character-probe
#: eval, so the two kinds sharing one section are distinguishable at a
#: glance. Single-width by construction -- a double-width glyph would shift
#: every row's alignment (the ␣/⏎/✓✗ markers elsewhere follow the same rule).
CHARACTER_PROBE_MARKER = "◆ "
```

and apply it in the bench and dataset row-label functions, after `escape_markup`:

```python
def _bench_row_label(row: Mapping[str, Any]) -> str:
    from ...Evals.character_probe.storage import is_character_bench

    name = escape_markup(str(row.get("name") or "Untitled bench"))
    return f"{CHARACTER_PROBE_MARKER}{name}" if is_character_bench(row) else name
```

Mirror that in `_dataset_row_label` using `is_probe_set`. The classic-task label is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_empty_states.py -p no:randomly`
Expected: PASS (existing tests plus the 4 new)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/library_rail.py tldw_chatbook/UI/Evals/evals_state.py Tests/UI/test_evals_empty_states.py
git commit -m "feat(evals): rail marks character-probe benches and probe sets (task-1691 phase 2)"
```

---

### Task 2: Import a probe set from a file

**Files:**
- Modify: `tldw_chatbook/UI/Evals/library_rail.py`
- Test: `Tests/UI/test_evals_empty_states.py`

**Interfaces:**
- Consumes: `character_probe.probe_format.parse_probe_text(text) -> ProbeSet` (raises `ValueError` naming the 1-based probe on malformed input); `character_probe.storage.save_probe_set(db, name, probe_set) -> str`.
- Produces: `#evals-rail-import-probes` Button in the Datasets actions row; `LibraryRail._handle_probe_import_file_selected(path)` — public-shaped so tests bypass the file dialog, exactly as `_handle_import_file_selected` already does for snippets.

Mirror the existing snippet import (`library_rail.py:753` `evals-rail-import-dataset` and its `FileOpen` handler) rather than inventing a second pattern.

- [ ] **Step 1: Write the failing test**

```python
def test_importing_a_probe_file_creates_a_marked_probe_set(evals_app, evals_db, tmp_path):
    probe_file = tmp_path / "starter.txt"
    probe_file.write_text("What do you think about lying?\n---\nAnd to protect someone?\n===\nDescribe your earliest memory.")

    async def _check(pilot):
        rail = pilot.app.screen.query_one(LibraryRail)
        rail._handle_probe_import_file_selected(probe_file)
        await pilot.pause()
        from tldw_chatbook.Evals.character_probe.storage import is_probe_set, load_probe_set
        rows = [r for r in evals_db.list_datasets() if is_probe_set(r)]
        assert len(rows) == 1
        probes = load_probe_set(evals_db, rows[0]["id"]).probes
        assert len(probes) == 2
        assert probes[0].turns == ("What do you think about lying?", "And to protect someone?")
    run_evals(evals_app, _check)


def test_importing_a_malformed_probe_file_notifies_and_creates_nothing(evals_app, evals_db, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("Real probe\n===\n   \n===\nAnother")

    async def _check(pilot):
        rail = pilot.app.screen.query_one(LibraryRail)
        rail._handle_probe_import_file_selected(bad)
        await pilot.pause()
        from tldw_chatbook.Evals.character_probe.storage import is_probe_set
        assert [r for r in evals_db.list_datasets() if is_probe_set(r)] == []
        assert any("probe 2" in str(m) for m in pilot.app.app_instance.notifications)
    run_evals(evals_app, _check)


def test_importing_a_file_that_cannot_be_read_notifies(evals_app, evals_db, tmp_path):
    async def _check(pilot):
        rail = pilot.app.screen.query_one(LibraryRail)
        rail._handle_probe_import_file_selected(tmp_path / "missing.txt")
        await pilot.pause()
        from tldw_chatbook.Evals.character_probe.storage import is_probe_set
        assert [r for r in evals_db.list_datasets() if is_probe_set(r)] == []
        assert pilot.app.app_instance.notifications
    run_evals(evals_app, _check)


def test_the_import_probes_button_is_present_in_the_dataset_actions(evals_app):
    async def _check(pilot):
        assert pilot.app.screen.query_one("#evals-rail-import-probes")
    run_evals(evals_app, _check)
```

`pilot.app.app_instance.notifications` is the fake app's recorded-toast list — check the fixture's actual attribute name in `Tests/UI/test_evals_screen.py`'s `_FakeAppInstance` and use the real one.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_empty_states.py -k probe_import -p no:randomly`
Expected: FAIL — `AttributeError: 'LibraryRail' object has no attribute '_handle_probe_import_file_selected'`

- [ ] **Step 3: Write minimal implementation**

Add the button to `_dataset_actions`' Horizontal, beside the existing two:

```python
            Button("Import probes…", id="evals-rail-import-probes", compact=True),
```

Add its branch to `on_button_pressed`, mirroring the snippet import's `FileOpen` push:

```python
        if button_id == "evals-rail-import-probes":
            event.stop()
            self.app.push_screen(
                FileOpen(title="Import probe set"),
                self._handle_probe_import_file_selected,
            )
            return
```

and the handler:

```python
    def _handle_probe_import_file_selected(self, path: Optional[Path]) -> None:
        """Import a probe file as a new probe set.

        Public-shaped (not ``_on_...``) so tests can drive the import
        without standing up the file dialog -- the same convention
        ``_handle_import_file_selected`` uses for snippets.

        Args:
            path: The chosen file, or None when the dialog was cancelled.
        """
        if path is None:
            return
        db = self.view_model.db
        if db is None:
            self._notify("The evaluation service is unavailable.", severity="error")
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            self._notify(f"Could not read {Path(path).name}: {exc}", severity="error")
            return
        try:
            probe_set = parse_probe_text(text)
        except ValueError as exc:
            self._notify(f"That file is not a valid probe set: {exc}", severity="error")
            return
        dataset_id = save_probe_set(db, Path(path).stem, probe_set)
        count = len(probe_set.probes)
        self._notify(f"Imported {count} probe(s) into a new probe set.")
        self.post_message(
            self.EvalsSelectionChanged(EvalsSelection(kind="dataset", id=dataset_id))
        )
```

Import `parse_probe_text` and `save_probe_set` at module scope from `...Evals.character_probe`; they pull no UI and no provider code.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_empty_states.py -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/library_rail.py Tests/UI/test_evals_empty_states.py
git commit -m "feat(evals): import a probe set from a file (task-1691 phase 2)"
```

---

### Task 3: The character card picker

**Files:**
- Create: `tldw_chatbook/UI/Evals/card_picker.py`
- Modify: `tldw_chatbook/UI/Evals/evals_state.py`
- Test: `Tests/UI/test_evals_card_picker.py`

**Interfaces:**
- Consumes: `CharactersRAGDB.list_character_cards(limit, offset)` returning rows with int `id` and str `name`.
- Produces: `CardPicker(cards: Sequence[Mapping], selected_ids: Sequence[int], id=...)` — a `Vertical` widget with `#evals-card-search` (Input) and one `#evals-card-row-{index}` toggle per visible card; `CardPicker.selected_ids() -> tuple[int, ...]`; message `CardPicker.SelectionChanged(selected_ids: tuple[int, ...])`.

Cards live in `ChaChaNotes_DB`, a different database from `EvalsDB`. This widget receives already-fetched rows — it never opens a database itself, so it stays testable without one and the screen owns the cross-DB handle.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input

from tldw_chatbook.UI.Evals.card_picker import CardPicker

CARDS = [
    {"id": 3, "name": "Vex"},
    {"id": 7, "name": "Marlow"},
    {"id": 9, "name": "vexing puzzle"},
]


class _Host(App):
    def __init__(self, cards, selected=()):
        super().__init__()
        self._cards = cards
        self._selected = selected

    def compose(self) -> ComposeResult:
        yield CardPicker(self._cards, self._selected, id="picker")


@pytest.mark.asyncio
async def test_every_card_renders_a_row():
    async with _Host(CARDS).run_test() as pilot:
        picker = pilot.app.query_one(CardPicker)
        assert len(picker.query(".evals-card-row")) == 3


@pytest.mark.asyncio
async def test_clicking_a_row_selects_that_card_by_int_id():
    async with _Host(CARDS).run_test() as pilot:
        await pilot.click("#evals-card-row-0")
        picker = pilot.app.query_one(CardPicker)
        assert picker.selected_ids() == (3,)
        assert all(isinstance(i, int) for i in picker.selected_ids())


@pytest.mark.asyncio
async def test_clicking_a_selected_row_deselects_it():
    async with _Host(CARDS, selected=(3,)).run_test() as pilot:
        await pilot.click("#evals-card-row-0")
        assert pilot.app.query_one(CardPicker).selected_ids() == ()


@pytest.mark.asyncio
async def test_search_filters_rows_case_insensitively():
    async with _Host(CARDS).run_test() as pilot:
        picker = pilot.app.query_one(CardPicker)
        await pilot.click("#evals-card-search")
        await pilot.press(*"vex")
        await pilot.pause()
        shown = [w.card_name for w in picker.query(".evals-card-row")]
        assert shown == ["Vex", "vexing puzzle"]


@pytest.mark.asyncio
async def test_filtering_does_not_drop_a_selection_that_is_hidden():
    """A card selected then filtered out of view is still selected."""
    async with _Host(CARDS, selected=(7,)).run_test() as pilot:
        picker = pilot.app.query_one(CardPicker)
        await pilot.click("#evals-card-search")
        await pilot.press(*"vex")
        await pilot.pause()
        assert 7 in picker.selected_ids()


@pytest.mark.asyncio
async def test_a_markup_hazard_card_name_renders_literally():
    async with _Host([{"id": 1, "name": "Vex[/]v2"}]).run_test() as pilot:
        row = pilot.app.query_one("#evals-card-row-0")
        assert "[/]" in row.render_label().plain


@pytest.mark.asyncio
async def test_selection_change_posts_a_message():
    async with _Host(CARDS).run_test() as pilot:
        seen = []
        pilot.app.query_one(CardPicker).post_message = lambda m: seen.append(m)
        await pilot.click("#evals-card-row-1")
        await pilot.pause()
        assert any(getattr(m, "selected_ids", None) == (7,) for m in seen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_card_picker.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...card_picker'`

- [ ] **Step 3: Write minimal implementation**

`tldw_chatbook/UI/Evals/card_picker.py`:

```python
"""Searchable multi-select over character cards.

Receives already-fetched rows rather than opening a database: cards live in
``ChaChaNotes_DB`` while this slice's own handle is ``EvalsDB``, and keeping
the fetch outside means this widget is testable without either.

Card ids are INTEGERs (``character_cards.id``) while every eval id in this
slice is TEXT. They are deliberately never normalised to strings -- the
engine's ``CharacterProbeConfig`` rejects non-int ids at construction.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static


class CardRow(Button):
    """One selectable card. Label is a pre-built ``Text`` so a card name
    containing markup renders literally rather than raising."""

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
        """The row's rendered label, for tests and for refresh after a toggle."""
        return self._compose_label(self.card_name, self._selected)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.label = self.render_label()


class CardPicker(Vertical):
    """Search box plus one toggle row per matching card."""

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
        super().__init__(**kwargs)
        self._cards = [dict(card) for card in cards]
        self._selected: set[int] = {int(cid) for cid in selected_ids}
        self._filter = ""

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
        if not self._cards:
            yield Static(
                "No character cards yet — create one in Roleplay first.",
                id="evals-card-picker-empty",
                markup=False,
            )
            return
        for index, card in enumerate(self._visible()):
            yield CardRow(
                int(card["id"]),
                str(card.get("name") or ""),
                int(card["id"]) in self._selected,
                index,
            )

    async def _rebuild(self) -> None:
        await self.remove_children()
        await self.mount_all(list(self.compose()))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "evals-card-search":
            return
        event.stop()
        self._filter = event.value
        self.call_after_refresh(self._rebuild)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        row = event.button
        if not isinstance(row, CardRow):
            return
        event.stop()
        if row.card_id in self._selected:
            self._selected.discard(row.card_id)
        else:
            self._selected.add(row.card_id)
        row.set_selected(row.card_id in self._selected)
        self.post_message(self.SelectionChanged(self.selected_ids()))
```

Add to `evals_state.py`:

```python
    def character_cards(self, chacha_db: Any) -> list[dict[str, Any]]:
        """Character cards for the picker, ordered by name.

        Args:
            chacha_db: A ``CharactersRAGDB``-shaped handle, or None when the
                character database is unavailable.

        Returns:
            list[dict[str, Any]]: Card rows, or an empty list when no handle
            was supplied.
        """
        if chacha_db is None:
            return []
        return list(chacha_db.list_character_cards(limit=_LIST_LIMIT))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_card_picker.py -p no:randomly`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/card_picker.py tldw_chatbook/UI/Evals/evals_state.py Tests/UI/test_evals_card_picker.py
git commit -m "feat(evals): searchable multi-select character card picker (task-1691 phase 2)"
```

---

### Task 4: The character bench editor

**Files:**
- Create: `tldw_chatbook/UI/Evals/character_bench_editor.py`
- Modify: `tldw_chatbook/css/features/_evals.tcss` (+ regenerate the bundle)
- Test: `Tests/UI/test_evals_character_bench_editor.py`

**Interfaces:**
- Consumes: `CardPicker` (Task 3); `character_probe.storage.load_character_bench(db, task_id)`, `save_character_bench(db, config, task_id)`, `load_probe_set(db, dataset_id)`; `CharacterProbeConfig`.
- Produces: `CharacterBenchEditor(view_model, bench_id, cards, id=...)`; widget ids `#evals-cb-name`, `#evals-cb-description`, `#evals-cb-samples`, `#evals-cb-seed`, `#evals-cb-temperature`, `#evals-cb-max-tokens`, `#evals-cb-save`, `#evals-cb-revert`, `#evals-cb-form-error`; message `CharacterBenchEditor.Saved(bench_id: str)`.

Mirrors `bench_editor.py`'s established form contract: editing is display-only until Save; a failed Save renders in `#evals-cb-form-error` and does NOT recompose, so typed state survives; a successful Save posts `Saved` and the screen re-selects. It is a SEPARATE widget — the two bench types never share a detail surface.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_editor_renders_the_stored_bench(character_app, saved_bench_id):
    async with character_app.run_test(size=(160, 45)) as pilot:
        select_bench(pilot, saved_bench_id)
        await pilot.pause()
        assert pilot.app.screen.query_one("#evals-cb-name", Input).value == "villain probes"
        assert pilot.app.screen.query_one("#evals-cb-samples", Input).value == "1"


@pytest.mark.asyncio
async def test_saving_persists_every_edited_field(character_app, saved_bench_id, evals_db):
    async with character_app.run_test(size=(160, 45)) as pilot:
        select_bench(pilot, saved_bench_id)
        await pilot.pause()
        name = pilot.app.screen.query_one("#evals-cb-name", Input)
        await pilot.click("#evals-cb-name")
        name.value = ""
        await pilot.press(*"renamed")
        samples = pilot.app.screen.query_one("#evals-cb-samples", Input)
        samples.value = ""
        await pilot.click("#evals-cb-samples")
        await pilot.press("3")
        await pilot.click("#evals-cb-save")
        await pilot.pause()
        from tldw_chatbook.Evals.character_probe.storage import load_character_bench
        stored = load_character_bench(evals_db, saved_bench_id)
        assert stored.name == "renamed"
        assert stored.samples_per_cell == 3


@pytest.mark.asyncio
async def test_an_invalid_samples_value_renders_the_error_and_keeps_typed_state(
    character_app, saved_bench_id
):
    async with character_app.run_test(size=(160, 45)) as pilot:
        select_bench(pilot, saved_bench_id)
        await pilot.pause()
        samples = pilot.app.screen.query_one("#evals-cb-samples", Input)
        samples.value = "0"
        name = pilot.app.screen.query_one("#evals-cb-name", Input)
        name.value = "typed-but-not-saved"
        await pilot.click("#evals-cb-save")
        await pilot.pause()
        assert pilot.app.screen.query_one("#evals-cb-form-error").visible
        assert pilot.app.screen.query_one("#evals-cb-name", Input).value == "typed-but-not-saved"


@pytest.mark.asyncio
async def test_selecting_cards_in_the_picker_persists_on_save(
    character_app, saved_bench_id, evals_db
):
    async with character_app.run_test(size=(160, 45)) as pilot:
        select_bench(pilot, saved_bench_id)
        await pilot.pause()
        await pilot.click("#evals-card-row-1")
        await pilot.click("#evals-cb-save")
        await pilot.pause()
        from tldw_chatbook.Evals.character_probe.storage import load_character_bench
        assert 7 in load_character_bench(evals_db, saved_bench_id).character_ids


@pytest.mark.asyncio
async def test_the_probe_listing_shows_whitespace_markers(character_app, saved_bench_id):
    """Probe turns are byte-exact prompts; leading spaces must be visible."""
    async with character_app.run_test(size=(160, 45)) as pilot:
        select_bench(pilot, saved_bench_id)
        await pilot.pause()
        listing = pilot.app.screen.query_one("#evals-cb-probes").render()
        assert "␣" in str(listing)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52)])
async def test_save_stays_hit_testable_at_realistic_sizes(character_app, saved_bench_id, size):
    """This pane has pushed a control out of reach three times (task-1764)."""
    async with character_app.run_test(size=size) as pilot:
        select_bench(pilot, saved_bench_id)
        await pilot.pause()
        editor = pilot.app.screen.query_one("#evals-character-bench-editor")
        editor.scroll_end(animate=False)
        await pilot.pause()
        save = pilot.app.screen.query_one("#evals-cb-save")
        hit = pilot.app.screen.get_widget_at(*save.region.center)[0]
        assert hit is save or save in hit.ancestors
```

Build `character_app`/`saved_bench_id`/`select_bench` on `Tests/UI/test_evals_screen.py`'s `EvalsHarness` and `_FakeAppInstance` — import them rather than writing new harness code, and seed a bench whose probe set contains a turn with a leading space so the marker test has something to find.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_character_bench_editor.py -p no:randomly`
Expected: FAIL — `ModuleNotFoundError: No module named '...character_bench_editor'`

- [ ] **Step 3: Write minimal implementation**

Create `character_bench_editor.py` following `bench_editor.py`'s structure: a `Vertical` whose `compose()` yields the name and description Inputs, a read-only probe-set line, the `CardPicker`, a read-only targets listing, the four sampler Inputs, `#evals-cb-form-error` (a `Static` with `markup=False`, no `.ds-recovery-callout` class until an error fires), and the Save/Revert Buttons. `_on_save_pressed` reads every widget fresh, builds a `CharacterProbeConfig`, and calls `save_character_bench`; `ValueError` and `ConflictError` render in the error Static without recomposing; success posts `Saved(bench_id)`.

Probe turns render through `snippet_editor.render_snippet_cell` for `␣` markers, with newlines replaced by `⏎` so one probe stays one line — reuse `bench_editor.py`'s existing preview helper rather than writing a second one.

CSS: give `#evals-character-bench-editor` `height: auto` and `overflow-y: auto` — the same pairing `#evals-inspector-bench` needed, for the same reason (a `Vertical` defaults to `height: 1fr` and starves its siblings). Regenerate the bundle with `/private/tmp/tldw-venv/bin/python tldw_chatbook/css/build_css.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_character_bench_editor.py -p no:randomly`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Evals/character_bench_editor.py tldw_chatbook/css Tests/UI/test_evals_character_bench_editor.py
git commit -m "feat(evals): character bench editor with card selection (task-1691 phase 2)"
```

---

### Task 5: Route the detail pane and create a bench from the rail

**Files:**
- Modify: `tldw_chatbook/UI/Screens/evals_screen.py`
- Modify: `tldw_chatbook/UI/Evals/library_rail.py`
- Test: `Tests/UI/test_evals_screen.py`

**Interfaces:**
- Consumes: `CharacterBenchEditor` (Task 4), `EvalsViewModel.character_benches()`/`probe_sets()`/`character_cards(chacha_db)` (Tasks 1, 3), `save_character_bench`.
- Produces: `#evals-rail-new-character-bench` Button; `EvalsScreen._chacha_db` (the cross-DB handle, resolved once, `None` when unavailable).

Selecting a bench must render the editor matching its `bench_type`. The screen owns the `ChaChaNotes_DB` handle because the widgets must not open databases themselves.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_selecting_a_character_bench_renders_its_own_editor(evals_app, character_bench_id):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="bench", id=character_bench_id)
        await pilot.pause()
        assert pilot.app.screen.query("#evals-character-bench-editor")
        assert not pilot.app.screen.query("#evals-bench-editor")


@pytest.mark.asyncio
async def test_selecting_a_word_bench_still_renders_the_word_editor(evals_app, seeded_bench):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()
        assert pilot.app.screen.query("#evals-bench-editor")
        assert not pilot.app.screen.query("#evals-character-bench-editor")


@pytest.mark.asyncio
async def test_new_character_bench_requires_a_probe_set(evals_app, evals_db):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        button = pilot.app.screen.query_one("#evals-rail-new-character-bench")
        assert button.disabled
        assert "probe set" in str(button.tooltip)


@pytest.mark.asyncio
async def test_new_character_bench_creates_and_selects_a_draft(evals_app, evals_db, probe_set_id):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.click("#evals-rail-new-character-bench")
        await pilot.pause()
        from tldw_chatbook.Evals.character_probe.storage import is_character_bench
        benches = [t for t in evals_db.list_tasks() if is_character_bench(t)]
        assert len(benches) == 1
        assert benches[0]["config_data"]["probe_set_id"] == probe_set_id
        assert pilot.app.screen._selection.id == benches[0]["id"]


@pytest.mark.asyncio
async def test_a_character_bench_with_no_cards_cannot_be_run(evals_app, character_bench_id):
    """A run with no cards would produce an empty grid, not a result."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="bench", id=character_bench_id)
        await pilot.pause()
        action = pilot.app.screen.query_one("#evals-primary-action")
        assert action.disabled
        assert "card" in str(action.tooltip)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_screen.py -k character -p no:randomly`
Expected: FAIL — `NoMatches: #evals-rail-new-character-bench`

- [ ] **Step 3: Write minimal implementation**

In `library_rail.py`, add the affordance beside "+ New bench" in the Benches section body, disabled with a tooltip when `view_model.probe_sets()` is empty (a character bench without a probe set has nothing to run), and post a `NewCharacterBenchRequested` message the screen handles.

In `evals_screen.py`:
- resolve `self._chacha_db` once, tolerating absence (`None`);
- in `_compose_detail_pane`'s bench branch, choose the editor by `bench_type`: `is_character_bench(bench)` → `CharacterBenchEditor(self._view_model, selection.id, self._view_model.character_cards(self._chacha_db), id="evals-character-bench-editor")`, else the existing `BenchEditor`;
- handle `NewCharacterBenchRequested` by creating a `CharacterProbeConfig` bound to the newest probe set with `_unique_name("Untitled character bench")`, empty `character_ids`, and the configured llama.cpp target if one exists, then selecting it;
- extend `_primary_action_state`'s bench branch: for a character bench with no `character_ids`, disabled with `"This bench has no characters yet; pick at least one in the editor."`; with no `target_ids`, reuse the existing no-targets reason.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_screen.py Tests/UI/test_evals_character_bench_editor.py -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/evals_screen.py tldw_chatbook/UI/Evals/library_rail.py Tests/UI/test_evals_screen.py
git commit -m "feat(evals): route the detail pane by bench type and create character benches (task-1691 phase 2)"
```

---

### Task 6: Run a character bench, with its cost shown first

**Files:**
- Modify: `tldw_chatbook/UI/Screens/evals_screen.py`
- Modify: `tldw_chatbook/UI/Evals/inspector.py`
- Test: `Tests/UI/test_evals_screen.py`, `Tests/UI/test_evals_character_run_e2e.py`

**Interfaces:**
- Consumes: `CharacterProbeRunner(chat_fn, cancel_token)`, `create_probe_run_group`, `save_conversations`, `snapshot_cards`, `resolve_targets`, `load_probe_set`, `load_character_bench` (all phase 1).
- Produces: `EvalsScreen._character_probe_chat_factory` — the injectable provider seam, mirroring `_sample_bench_client_factory`; `EvalsScreen._run_character_bench_worker`.

The chat callable is SYNCHRONOUS. The runner already dispatches it through `asyncio.to_thread`; the screen must pass a plain callable and must not await it itself.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_the_estimate_counts_cards_probes_targets_samples_and_turns(
    evals_app, runnable_character_bench
):
    """2 cards x 2 probes (2 turns, 1 turn) x 1 target x 1 sample = 6 calls."""
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="bench", id=runnable_character_bench)
        await pilot.pause()
        estimate = pilot.app.screen.query_one("#evals-estimate-calls").render()
        assert str(estimate).startswith("6 calls")


@pytest.mark.asyncio
async def test_running_a_character_bench_persists_conversations(
    evals_app, runnable_character_bench, evals_db
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="bench", id=runnable_character_bench)
        await pilot.pause()
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: pilot.app.screen._selection.kind == "run_group")
        from tldw_chatbook.Evals.character_probe.storage import load_conversations
        conversations = load_conversations(evals_db, pilot.app.screen._selection.id)
        assert len(conversations) == 4
        assert all(c.turns for c in conversations)


@pytest.mark.asyncio
async def test_a_failing_provider_leaves_the_rest_of_the_grid_intact(
    evals_app, runnable_character_bench, evals_db, failing_once_chat
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen._character_probe_chat_factory = lambda cfg: failing_once_chat
        pilot.app.screen.select(kind="bench", id=runnable_character_bench)
        await pilot.pause()
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: pilot.app.screen._selection.kind == "run_group")
        from tldw_chatbook.Evals.character_probe.storage import load_conversations
        conversations = load_conversations(evals_db, pilot.app.screen._selection.id)
        assert any(c.error for c in conversations)
        assert any(not c.error and c.turns for c in conversations)


@pytest.mark.asyncio
async def test_the_run_snapshot_records_card_text_and_sampler(
    evals_app, runnable_character_bench, evals_db
):
    async with evals_app.run_test(size=(160, 45)) as pilot:
        pilot.app.screen.select(kind="bench", id=runnable_character_bench)
        await pilot.pause()
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: pilot.app.screen._selection.kind == "run_group")
        from tldw_chatbook.Evals.character_probe.storage import load_probe_run_snapshot
        snapshot = load_probe_run_snapshot(evals_db, pilot.app.screen._selection.id)
        assert snapshot["cards"]
        assert snapshot["sampler"]["samples_per_cell"] == 1
```

Confirm the real names of `load_probe_run_snapshot`, `resolve_targets`, and `create_probe_run_group` against `tldw_chatbook/Evals/character_probe/` before writing — phase 1 shipped them, but verify the signatures rather than trusting this plan.

- [ ] **Step 2: Run test to verify it fails**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_character_run_e2e.py -p no:randomly`
Expected: FAIL — the estimate still reports the word-bench count, and no character-run worker exists.

- [ ] **Step 3: Write minimal implementation**

In `inspector.py`, branch the estimate by bench type: for a character bench, `sum(len(p.turns) for p in probe_set.probes) * len(cards) * len(targets) * samples_per_cell`. Keep the word-bench arithmetic untouched.

In `evals_screen.py`, add `_run_character_bench_worker` mirroring `_run_bench_worker`: guard against a run already in flight, snapshot cards, resolve targets, `create_probe_run_group`, run `CharacterProbeRunner` with the factory's callable, `save_conversations`, then select the new run group through the existing dirty-editor-aware guard. Errors notify with `markup=False`; the finally-block restores the button under a `QueryError` guard.

- [ ] **Step 4: Run test to verify it passes**

Run: `/private/tmp/tldw-venv/bin/python -m pytest Tests/UI/test_evals_character_run_e2e.py Tests/UI/test_evals_screen.py -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/evals_screen.py tldw_chatbook/UI/Evals/inspector.py Tests/UI/test_evals_character_run_e2e.py Tests/UI/test_evals_screen.py
git commit -m "feat(evals): run a character bench from the UI (task-1691 phase 2)"
```

---

## Phase 2 exit criteria

- A user can import a probe set, create a character bench, pick cards, save, and run — entirely through the UI, with no Python.
- The Estimate states the true call count before Run is pressed.
- Selecting either bench type renders its own editor; neither leaks into the other's pane.
- Every new control is hit-testable at 160x45 and 235x52 after the rows above it grow.
- `Tests/Evals/character_probe` and `Tests/Evals/word_bench` remain green and behaviourally untouched.

## Not in Phase 2 (deliberate)

The review queue, the conversation view, tag application, review state, ordering hints, and the summary are Phase 3 and Phase 4. This phase produces conversations; reading them is next. A starter probe set shipped with the app is also deferred — Phase 2 imports from a file, which is enough to reach a first run.
