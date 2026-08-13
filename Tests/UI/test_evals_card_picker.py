"""Character card picker (Phase 2 Task 3, task-1691).

``CardPicker`` is a standalone, searchable multi-select over already-fetched
character-card rows (``id``: int, ``name``: str). It never opens a database
itself -- see ``card_picker.py``'s own module docstring for why -- so every
test here mounts it directly in a lightweight host ``App`` rather than the
full ``EvalsScreen`` harness ``test_evals_bench_editor.py`` and friends use;
there is no ``EvalsDB``/``ChaChaNotes_DB`` dependency to fake.

**The search box must survive a rebuild.** An earlier draft of this widget's
``_rebuild()`` tore down and remounted its ENTIRE ``compose()`` output
(including the ``#evals-card-search`` `Input` itself) on every keystroke --
which drops focus and the character just typed the instant the next
keystroke's `Input.Changed` fires the next rebuild. ``CardPicker`` instead
rebuilds only the row list, nested under its own ``#evals-card-picker-rows``
container that never contains the search `Input`.
``test_search_filters_rows_case_insensitively`` types "vex" one character at
a time through ``pilot.press`` specifically to catch a regression of that
shape -- a rebuild-everything implementation drops keystrokes and this test
never converges on the full needle.

``test_selection_change_posts_a_message`` uses a real host-`App` handler
(``on_card_picker_selection_changed``), not a patched ``post_message``: the
message is asserted the way a real parent (Task 4's bench editor) would
actually receive it, per this phase's own "tests must drive real widgets"
rule.

The two ``test_*_stays_hit_testable_*`` cases are this task's share of the
phase-wide "painted geometry is the arbiter" rule -- any task that adds rows
must prove a control below them stays clickable, at both 160x45 and 235x52,
not merely present in the DOM.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.errors import NoWidget
from textual.widgets import Button, Input

from tldw_chatbook.UI.Evals.card_picker import SEARCH_DEBOUNCE_SECONDS, CardPicker
from tldw_chatbook.UI.Evals.evals_state import _LIST_LIMIT, EvalsViewModel

CARDS = [
    {"id": 3, "name": "Vex"},
    {"id": 7, "name": "Marlow"},
    {"id": 9, "name": "vexing puzzle"},
]

_SIZES = [(160, 45), (235, 52)]


class _FakeChaChaDB:
    """Duck-typed ``CharactersRAGDB`` stand-in for
    ``EvalsViewModel.character_cards()`` -- CardPicker itself never opens a
    database (see ``card_picker.py``'s own module docstring), so this is
    the read side's own unit coverage, not something any widget test above
    exercises."""

    def __init__(self, rows):
        self._rows = rows
        self.received_kwargs: dict | None = None

    def list_character_cards(self, **kwargs):
        self.received_kwargs = kwargs
        return self._rows


def test_character_cards_returns_empty_list_without_a_handle():
    """Empty-safe like every other read on ``EvalsViewModel`` -- a caller
    (Task 4's bench editor) needs no ``chacha_db is None`` branch of its
    own before calling this."""
    view_model = EvalsViewModel(None)
    assert view_model.character_cards(None) == []


def test_character_cards_delegates_to_the_chacha_handle_with_the_shared_list_limit():
    """Uses the module's existing ``_LIST_LIMIT`` (500), not
    ``list_character_cards``'s own default of 100 -- every other read on
    this class already makes that same choice (see e.g. ``llama_targets``'s
    own comment) so an install with more than 100 character cards doesn't
    silently lose the rest from the picker."""
    fake_db = _FakeChaChaDB([{"id": 3, "name": "Vex"}])
    view_model = EvalsViewModel(None)
    result = view_model.character_cards(fake_db)
    assert result == [{"id": 3, "name": "Vex"}]
    assert fake_db.received_kwargs == {"limit": _LIST_LIMIT}


class _Host(App):
    """Bare host: just the picker, no CSS bundle -- matches how the other
    functional tests below only care about widget-tree behaviour, not
    painted geometry (see ``_GeometryHost`` for the tests that do)."""

    def __init__(self, cards, selected=()):
        super().__init__()
        self._cards = cards
        self._selected = selected

    def compose(self) -> ComposeResult:
        yield CardPicker(self._cards, self._selected, id="picker")


class _MessageCaptureHost(App):
    """A real ancestor with a real ``on_card_picker_selection_changed``
    handler -- ``CardPicker.SelectionChanged`` is declared with
    ``namespace="card_picker"``, so this is the exact handler name Textual's
    own bubbling dispatch will call on a parent (see
    ``textual.message.Message.__init_subclass__``: ``namespace`` becomes
    ``on_<namespace>_<message>``)."""

    def __init__(self, cards):
        super().__init__()
        self._cards = cards
        self.received: list[CardPicker.SelectionChanged] = []

    def compose(self) -> ComposeResult:
        yield CardPicker(self._cards, id="picker")

    def on_card_picker_selection_changed(
        self, message: CardPicker.SelectionChanged
    ) -> None:
        self.received.append(message)


class _GeometryHost(App):
    """A picker plus a real sibling control below it, at the REAL bundled
    stylesheet -- painted-geometry checks are only meaningful against the
    CSS that will actually ship (``#evals-card-picker-rows``'s bounded,
    scrollable ``max-height`` in ``_evals.tcss``), not Textual's unstyled
    defaults."""

    CSS_PATH = None  # set per-instance below (path resolved at import time)

    def __init__(self, cards):
        super().__init__()
        self._cards = cards

    def compose(self) -> ComposeResult:
        yield CardPicker(self._cards, id="picker")
        yield Button("Done", id="picker-host-done")


def _bundled_css_path() -> str:
    import tldw_chatbook
    from pathlib import Path

    return str(Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss")


def _hit_widget(screen, expected):
    """``screen.get_widget_at`` at ``expected``'s own painted center, or
    ``None`` if nothing is painted there at all (``NoWidget`` -- the case
    for a row scrolled fully out of its bounded, clipped container, which a
    bare ``get_widget_at`` call would otherwise raise out of an assertion
    instead of failing it with a readable message)."""
    center = expected.region.center
    try:
        hit, _ = screen.get_widget_at(int(center[0]), int(center[1]))
    except NoWidget:
        return None
    return hit


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
        # Debounced (task-15476): the row list only rebuilds once the
        # filter settles, not on every keystroke.
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        shown = [w.card_name for w in picker.query(".evals-card-row")]
        assert shown == ["Vex", "vexing puzzle"]


@pytest.mark.asyncio
async def test_search_box_keeps_focus_and_full_text_across_keystrokes():
    """TASK-1691 phase 2 T3 regression: a rebuild that remounts the search
    `Input` itself (rather than only the row list below it) drops focus and
    truncates the typed text after the first keystroke-triggered rebuild --
    the exact defect flagged in this task's brief. Typing "vex" one
    character at a time and then asserting the Input's OWN final value and
    focus state (not just the filtered rows, which the previous test
    already covers) catches a rebuild-everything regression directly rather
    than only inferring it from row output."""
    async with _Host(CARDS).run_test() as pilot:
        await pilot.click("#evals-card-search")
        await pilot.press(*"vex")
        await pilot.pause()
        search = pilot.app.query_one("#evals-card-search", Input)
        assert search.value == "vex"
        assert search.has_focus is True


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
    """Real message-capture, not a patched ``post_message``: asserts the
    message a parent widget would genuinely receive via Textual's own
    bubbling dispatch (see ``_MessageCaptureHost``'s docstring) -- setting a
    widget's internals directly and asserting they changed is exactly the
    shape of test that let phase 1 ship a checkbox no user could toggle."""
    async with _MessageCaptureHost(CARDS).run_test() as pilot:
        await pilot.click("#evals-card-row-1")
        await pilot.pause()
        assert any(m.selected_ids == (7,) for m in pilot.app.received)


@pytest.mark.asyncio
async def test_no_cards_renders_an_explicit_empty_state():
    """Fail loudly / never silently blank: zero cards must render visible
    guidance, not an empty row list a user could mistake for "still
    loading"."""
    async with _Host([]).run_test() as pilot:
        picker = pilot.app.query_one(CardPicker)
        assert picker.query_one("#evals-card-picker-empty") is not None
        assert len(picker.query(".evals-card-row")) == 0


@pytest.mark.asyncio
async def test_a_search_with_no_matches_renders_an_explicit_empty_state():
    async with _Host(CARDS).run_test() as pilot:
        picker = pilot.app.query_one(CardPicker)
        await pilot.click("#evals-card-search")
        await pilot.press(*"zzz-no-such-card")
        # Debounced (task-15476): the empty state only appears once the
        # filter settles.
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        assert len(picker.query(".evals-card-row")) == 0
        assert picker.query_one("#evals-card-picker-no-matches") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("size", _SIZES)
async def test_control_below_the_picker_stays_hit_testable_with_many_cards(size):
    """TASK-1691 phase 2 T3: a sibling control below the picker (Task 4's
    real Add/Cancel controls will occupy exactly this position) must stay
    hit-testable regardless of how many cards the picker is given --
    ``evals_state.py``'s ``character_cards()`` can return up to 500 rows
    (``_LIST_LIMIT``). Proven at both realistic viewports this codebase's
    other painted-geometry checks use (``test_evals_bench_editor.py``'s
    ``_REALISTIC_SIZE`` and its 235x52 companion)."""
    cards = [{"id": i, "name": f"Card {i}"} for i in range(60)]
    host = _GeometryHost(cards)
    host.CSS_PATH = _bundled_css_path()
    async with host.run_test(size=size) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        control = screen.query_one("#picker-host-done", Button)
        hit = _hit_widget(screen, control)
        assert hit is control, (
            f"'Done' control not hit-testable at {size} with 60 cards -- "
            f"landed on {hit!r} instead"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", _SIZES)
async def test_control_below_the_picker_stays_hit_testable_after_search(size):
    """Same guarantee, after the row list has been rebuilt by a search
    filter -- proves ``_refresh_rows`` doesn't leave the sibling control's
    painted region stale relative to the (now different) row count."""
    cards = [{"id": i, "name": f"Card {i}"} for i in range(60)]
    host = _GeometryHost(cards)
    host.CSS_PATH = _bundled_css_path()
    async with host.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.click("#evals-card-search")
        await pilot.press(*"Card 1")
        # Debounced (task-15476): the row list only rebuilds once the
        # filter settles -- wait for it so this actually exercises the
        # post-rebuild geometry the test's docstring describes.
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        screen = pilot.app.screen
        control = screen.query_one("#picker-host-done", Button)
        hit = _hit_widget(screen, control)
        assert hit is control, (
            f"'Done' control not hit-testable at {size} after filtering -- "
            f"landed on {hit!r} instead"
        )


@pytest.mark.asyncio
async def test_a_card_beyond_the_bounded_row_list_is_reachable_by_scrolling():
    """Bounding ``#evals-card-picker-rows`` (``_evals.tcss``'s ``max-height:
    10``) must not turn into silent clipping: `Vertical`'s own DEFAULT_CSS
    is ``overflow: hidden hidden``, so without the ``overflow-y: auto`` this
    rule also carries, every row past whatever fits the bounded box would
    be gone with no scrollbar and no scroll action able to reach it --
    not merely off-screen, permanently unselectable. Card 59 (the last of
    60) is not hit-testable at the initial scroll position, and IS after
    scrolling the row container to its end -- proving the list is actually
    scrollable, not just short."""
    cards = [{"id": i, "name": f"Card {i}"} for i in range(60)]
    host = _GeometryHost(cards)
    host.CSS_PATH = _bundled_css_path()
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        last_row = screen.query_one("#evals-card-row-59", Button)
        assert _hit_widget(screen, last_row) is not last_row, (
            "expected card 59 to start OUT of view (not hit-testable) so "
            "this test can prove scrolling, not mere visibility, reaches it"
        )

        rows_container = screen.query_one("#evals-card-picker-rows")
        rows_container.scroll_end(animate=False, force=True)
        await pilot.pause()

        hit = _hit_widget(screen, last_row)
        assert hit is last_row, (
            f"card 59 still not hit-testable after scrolling the row list "
            f"to its end -- landed on {hit!r} instead"
        )
