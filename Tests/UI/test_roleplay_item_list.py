"""Roleplay frame B2a: the items list (spec 1.5).

Pure contracts of ``UI/Persona_Modules/roleplay_frame_state.py``'s items-list
section, and (from Task 3 of the B2a plan) pane-level contracts of
``RoleplayItemList`` on a small styled host. Nothing here mounts
``PersonasScreen``, so this file runs in the UI Fast Lane; the mounted
contracts live in ``test_roleplay_item_list_app.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from rich.cells import cell_len

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.Persona_Modules.roleplay_frame_state as frame_state
from textual.app import ComposeResult
from textual.widgets import Input

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.roleplay_frame_harness import (
    click_mark_column,
    first_list_item,
    highlighted_item_id,
    item_at,
    select_roleplay_item,
    visible_item_ids,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    LibraryRow,
    PersonasLibraryPane,
    RoleplayItemList,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaEntitySelected,
    PersonaMarksChanged,
)
from tldw_chatbook.Widgets.glyph_fallback import set_ascii_glyph_mode

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
HOSTILE_NAMES = ("[/]", "[b]x", "[@click=app.record('x')]N[/]", "[/", "[TODO] y")


@pytest.fixture
def ascii_mode():
    set_ascii_glyph_mode(True)
    try:
        yield
    finally:
        set_ascii_glyph_mode(False)


def _plain(prompt: frame_state.ItemPrompt) -> list[str]:
    return ["".join(text for text, _style in line) for line in prompt.lines]


# ---- plurals and ages (RP-070) ----


@pytest.mark.parametrize(
    ("count", "singular", "plural_form", "expected"),
    [
        (1, "entry", "entries", "1 entry"),
        (0, "entry", "entries", "0 entries"),
        (2, "rule", None, "2 rules"),
        (1, "rule", None, "1 rule"),
    ],
)
def test_plural_matches_the_count(count, singular, plural_form, expected):
    assert frame_state.plural(count, singular, plural_form) == expected


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(seconds=30), "now"),
        (timedelta(minutes=5), "5m"),
        (timedelta(hours=3), "3h"),
        (timedelta(days=3), "3d"),
        (timedelta(days=65), "2mo"),
        (timedelta(days=800), "2y"),
        (timedelta(days=-2), "now"),
    ],
)
def test_relative_age_reads_like_a_person(delta, expected):
    stamp = (NOW - delta).replace(tzinfo=None).isoformat()
    assert frame_state.relative_age(stamp, now=NOW) == expected


@pytest.mark.parametrize("value", [None, "", "not a date", 17])
def test_relative_age_is_absent_without_a_date(value):
    assert frame_state.relative_age(value, now=NOW) is None


# ---- middle ellipsis (spec 1.5.2 "Names"; Review Focus 3) ----


def test_middle_ellipsis_keeps_the_last_ten_cells():
    name = "Highspire Gazette (imported from card) 2"
    cut = frame_state.middle_ellipsize_cells(name, 28)
    assert cut == "Highspire Gazette…om card) 2"
    assert cell_len(cut) == 28


@pytest.mark.parametrize(
    "name",
    [
        "Captain Isolde Varga of the Deep-Survey Frigate",
        "東京の夜の物語を語る長い名前のキャラクター",
        "Ana‍maria​ with zero width joiners and spaces",
        "🐉🐉🐉🐉🐉🐉🐉🐉🐉🐉🐉🐉🐉🐉🐉🐉",
        "龍",
    ],
)
@pytest.mark.parametrize("budget", [1, 2, 3, 5, 9, 14, 23, 40])
def test_middle_ellipsis_never_exceeds_its_budget(name, budget):
    cut = frame_state.middle_ellipsize_cells(name, budget)
    assert cell_len(cut) <= budget
    if cell_len(name) <= budget:
        assert cut == name


def test_middle_ellipsis_uses_the_ascii_marker_in_ascii_mode(ascii_mode):
    cut = frame_state.middle_ellipsize_cells(
        "Highspire Gazette (imported from card) 2", 28
    )
    assert "..." in cut and "…" not in cut
    assert cell_len(cut) == 28


# ---- gutter and density ----


def test_unicode_gutter_is_three_cells_and_marks_column_two():
    assert frame_state.item_gutter(loaded=True, marked=True) == "▸✓ "
    assert frame_state.item_gutter(loaded=False, marked=False) == "   "
    assert frame_state.gutter_mark_columns() == range(1, 2)


def test_ascii_mode_gutter_is_five_cells_and_the_mark_column_moves(ascii_mode):
    assert frame_state.item_gutter(loaded=True, marked=True) == ">[x] "
    assert frame_state.item_gutter(loaded=False, marked=False) == "     "
    assert frame_state.gutter_mark_columns() == range(1, 4)


@pytest.mark.parametrize(
    ("inner_width", "kind_total", "expected"),
    [(23, 12, 1), (39, 12, 1), (40, 40, 2), (49, 41, 1), (49, 28, 2)],
)
def test_density_is_two_lines_only_when_wide_and_short(
    inner_width, kind_total, expected
):
    assert (
        frame_state.item_density(inner_width=inner_width, kind_total=kind_total)
        == expected
    )


# ---- the prompt builder ----


def test_a_loaded_marked_lore_row_keeps_its_state_word_when_cut():
    prompt = frame_state.build_item_prompt(
        name="The Codex of the Drowned Archive",
        state="off",
        meta="250 entries",
        short="250",
        loaded=True,
        marked=True,
        width=24,
        density=1,
    )
    (line,) = _plain(prompt)
    assert line.startswith("▸✓ ") and line.endswith(" · off")
    assert cell_len(line) <= 24
    assert prompt.cut and prompt.name_cut
    styles = [style for _text, style in prompt.lines[0]]
    assert "bold" in styles  # loaded: bold name (spec 1.5.3)


def test_one_line_lore_adds_the_count_only_when_the_full_name_fits():
    fits = frame_state.build_item_prompt(
        name="Lore A",
        state="on",
        meta="250 entries",
        short="250",
        loaded=False,
        marked=False,
        width=40,
        density=1,
    )
    assert _plain(fits) == ["   Lore A · 250 · on"]
    tight = frame_state.build_item_prompt(
        name="A much longer lore book name",
        state="on",
        meta="250 entries",
        short="250",
        loaded=False,
        marked=False,
        width=30,
        density=1,
    )
    assert _plain(tight)[0].endswith(" · on") and " · 250 · " not in _plain(tight)[0]


def test_two_line_rows_carry_a_dim_scent_and_a_blank_second_line_without_one():
    with_scent = frame_state.build_item_prompt(
        name="Medical Abbrev",
        state="on",
        meta="2 rules",
        short=None,
        loaded=False,
        marked=False,
        width=50,
        density=2,
    )
    assert _plain(with_scent) == ["   Medical Abbrev · on", "   2 rules"]
    assert with_scent.lines[1][-1] == ("2 rules", "dim")
    without = frame_state.build_item_prompt(
        name="Detective Sam",
        state=None,
        meta=None,
        short=None,
        loaded=False,
        marked=False,
        width=50,
        density=2,
    )
    assert _plain(without) == ["   Detective Sam", "   "]


def test_an_off_book_paints_its_name_muted():
    prompt = frame_state.build_item_prompt(
        name="Chat-Speak",
        state="off",
        meta=None,
        short=None,
        loaded=False,
        marked=False,
        width=40,
        density=1,
    )
    assert ("Chat-Speak", "dim") in prompt.lines[0]


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_markup_shaped_names_stay_plain_text_in_the_spans(name):
    prompt = frame_state.build_item_prompt(
        name=name,
        state=None,
        meta=name,
        short=None,
        loaded=False,
        marked=False,
        width=60,
        density=2,
    )
    assert _plain(prompt) == [f"   {name}", f"   {name}"]


def test_a_name_with_a_newline_paints_on_one_line():
    prompt = frame_state.build_item_prompt(
        name="home\nevil",
        state=None,
        meta=None,
        short=None,
        loaded=False,
        marked=False,
        width=60,
        density=1,
    )
    assert _plain(prompt) == ["   home evil"]


@pytest.mark.parametrize(
    ("built_width", "name", "new_width", "expected"),
    [
        (40, "Short", 30, False),  # still fits: never rebuilt
        (40, "Short", 120, False),
        (40, "A name that is long enough to be cut", 30, True),  # cut: rebuilt
        (60, "A name that is long enough to be cut", 20, True),  # newly too narrow
        (
            60,
            "A name that is long enough to be cut",
            80,
            False,
        ),  # fitted and still fits
    ],
)
def test_only_prompts_the_new_width_changes_are_rebuilt(
    built_width, name, new_width, expected
):
    prompt = frame_state.build_item_prompt(
        name=name,
        state=None,
        meta=None,
        short=None,
        loaded=False,
        marked=False,
        width=built_width,
        density=1,
    )
    assert prompt.needs_rebuild(width=new_width, density=1) is expected
    assert prompt.needs_rebuild(width=built_width, density=2) is True


# ---- scents (spec 1.5.2 v1) ----


def test_character_scent_is_age_tag_and_snippet_and_skips_what_is_missing():
    record = {
        "last_modified": (NOW - timedelta(days=3)).isoformat(),
        "tags": ["sci-fi", "captain"],
        "description": "Commands the deep-survey frigate Meridian's Wake.",
    }
    assert frame_state.character_scent(record, now=NOW) == (
        "3d · sci-fi · Commands the deep-survey frigate Meridian's Wake."
    )
    assert frame_state.character_scent({"description": "Noir"}, now=NOW) == "Noir"
    assert frame_state.character_scent({}, now=NOW) is None


def test_character_scent_is_one_bounded_line():
    scent = frame_state.character_scent(
        {"description": "word\n" * 200, "tags": ["t" * 60]}, now=NOW
    )
    assert "\n" not in scent and cell_len(scent) <= frame_state.SCENT_MAX_CELLS


def test_persona_scent_is_the_system_prompts_first_words():
    assert frame_state.persona_scent({"system_prompt": "You are terse.\nAlways."}) == (
        "You are terse. Always."
    )
    assert frame_state.persona_scent({"system_prompt": ""}) is None


def test_book_scents_use_plural_words():
    assert frame_state.lore_scent(1) == "1 entry"
    assert frame_state.lore_scent(250) == "250 entries"
    assert frame_state.dictionary_scent(1) == "1 rule"
    assert frame_state.dictionary_scent(150) == "150 rules"


# ---- border titles and the loaded row's position (spec 1.5.1, RC-10) ----


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"kind": "character", "total": 28}, "Characters 28"),
        ({"kind": "character", "total": 28, "matched": 3}, "Characters 3 of 28"),
        ({"kind": "character", "total": None}, "Characters (…)"),
        (
            {"kind": "character", "total": 348, "matched": 20, "tag": "fantasy"},
            "Characters 20 of 348 · tag fantasy",
        ),
        ({"kind": "character", "total": 100, "total_known": False}, "Characters 100+"),
        ({"kind": "character", "total": 0, "failed": True}, "Characters (?)"),
        ({"kind": "persona", "total": 114}, "Personas 114"),
        ({"kind": "lore", "total": 3, "enabled": 2}, "Lore books 3 · 2 on"),
        (
            {"kind": "dictionary", "total": 3, "matched": 1, "enabled": 2},
            "Chat dictionaries 1 of 3 · 2 on",
        ),
    ],
)
def test_border_title_names_the_kind_and_its_count(kwargs, expected):
    kind = kwargs.pop("kind")
    assert frame_state.items_border_title(kind, **kwargs) == expected


def test_position_is_blank_without_a_loaded_row():
    assert frame_state.loaded_position(3, 28) == "#4 of 28"
    assert frame_state.loaded_position(None, 28) == ""
    assert frame_state.loaded_position(3, None) == ""


def test_hidden_marks_count_only_while_a_filter_narrows_the_list():
    marked = {"character:1", "character:2", "character:3"}
    loaded = {"character:1"}
    assert frame_state.hidden_mark_count(marked, loaded, filtered=True) == 2
    assert frame_state.hidden_mark_count(marked, loaded, filtered=False) == 0


# ---- the marks bar (spec 1.5.5) ----


@pytest.mark.parametrize(
    ("width", "expected_text", "expected_labels", "expected_separator"),
    [
        (60, "✓ 3 marked (2 hidden)", ("Export…", "Delete…", "Clear esc"), " · "),
        (41, "✓ 3 marked (2 hidden)", ("Export", "Delete", "Clear"), " "),
        (35, "✓ 3 (2 hidden)", ("Export", "Delete", "Clear"), " "),
        (24, "✓ 3", ("Export", "Delete", "Clear"), " "),
        (12, "✓ 3", ("Export", "Delete", "Clear"), " "),
    ],
)
def test_marks_bar_degrades_in_a_fixed_order(
    width, expected_text, expected_labels, expected_separator
):
    view = frame_state.marks_bar_view(3, 2, width, can_export=True)
    assert (view.count_text, view.labels, view.separator) == (
        expected_text,
        expected_labels,
        expected_separator,
    )
    assert view.count_tooltip == "3 marked, 2 hidden by the search or tag"


def test_marks_bar_is_absent_without_marks_and_drops_export_for_books():
    assert frame_state.marks_bar_view(0, 0, 60, can_export=True) is None
    view = frame_state.marks_bar_view(2, 0, 60, can_export=False)
    assert view.count_text == "✓ 2 marked"
    assert view.labels == ("Delete…", "Clear esc")


# ---- the one delete modal (R32) ----


def test_single_delete_copy_is_unchanged_from_dev():
    copy = frame_state.delete_modal_copy("character", ["Detective Sam"], bulk=False)
    assert copy == frame_state.DeleteModalCopy(
        title="Delete",
        message="Delete Detective Sam? This cannot be undone here.",
        confirm_label="Delete",
        cancel_label="Cancel",
    )


def test_bulk_delete_names_five_and_counts_the_rest():
    names = [f"Character {n}" for n in range(1, 8)]
    copy = frame_state.delete_modal_copy("character", names, bulk=True)
    assert copy.title == "Delete 7 marked"
    assert copy.confirm_label == "Delete 7 marked"
    assert copy.message.splitlines() == [
        "Delete these 7 characters? This cannot be undone here.",
        "",
        "  Character 1",
        "  Character 2",
        "  Character 3",
        "  Character 4",
        "  Character 5",
        "  and 2 more",
    ]


def test_bulk_delete_of_one_reads_as_one():
    copy = frame_state.delete_modal_copy("lore", ["Codex"], bulk=True)
    assert (
        copy.message.splitlines()[0]
        == "Delete this lore book? This cannot be undone here."
    )
    assert copy.confirm_label == "Delete 1 marked"


def test_the_toggle_toast_is_the_one_undo_string():
    assert (
        frame_state.turned_off_toast("Chat-Speak")
        == "Chat-Speak turned off · space to undo"
    )


# ---- windows (spec 1.5.6) ----


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"loaded": 200, "last_window": 200, "total": 348},
            False,
        ),  # one window per keystroke
        ({"loaded": 200, "last_window": 200, "total": 348, "to_end": True}, True),
        ({"loaded": 348, "last_window": 148, "total": 348, "to_end": True}, False),
        ({"loaded": 200, "last_window": 200, "total": 348, "target_index": 250}, True),
        (
            {"loaded": 400, "last_window": 200, "total": 5000, "target_index": 250},
            False,
        ),
        (
            {"loaded": 200, "last_window": 200, "total": 348, "target_found": False},
            True,
        ),
        (
            {"loaded": 200, "last_window": 120, "total": None, "target_found": False},
            False,
        ),
    ],
)
def test_the_window_worker_loads_through_only_to_its_target(kwargs, expected):
    assert frame_state.window_wants_more(window_rows=200, **kwargs) is expected


def test_new_option_ids_keeps_the_first_copy_and_drops_present_ids():
    keep, dropped = frame_state.new_option_ids(
        {"character:1"}, ["character:1", "character:2", "character:2", "character:3"]
    )
    assert keep == (1, 3)
    assert dropped == ("character:1", "character:2")


def test_a_fence_differs_when_any_field_differs():
    fence = frame_state.ListFence(
        "characters", "isolde", "relevance", None, "local", None, 7
    )
    assert fence == frame_state.ListFence(
        "characters", "isolde", "relevance", None, "local", None, 7
    )
    for changed in (
        frame_state.ListFence(
            "characters", "isold", "relevance", None, "local", None, 7
        ),
        frame_state.ListFence(
            "characters", "isolde", "name_asc", None, "local", None, 7
        ),
        frame_state.ListFence(
            "characters", "isolde", "relevance", "x", "local", None, 7
        ),
        frame_state.ListFence(
            "characters", "isolde", "relevance", None, "server", "s1", 7
        ),
        frame_state.ListFence(
            "characters", "isolde", "relevance", None, "local", None, 8
        ),
    ):
        assert changed != fence


# ==== The pane half (B2a plan Task 3): RoleplayItemList on a styled host ====
#
# The host composes the pane alone (no PersonasScreen, so no bootstrap
# profile) under the app's stylesheets, Roleplay's lazy sheet included, so
# the geometry assertions below are evidence (spec 5.7.1).


class _PaneHost(ConsolidatedCSSApp):
    """``PersonasLibraryPane`` alone, styled like the app."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]
    # The pane's width follows the terminal, as in the workbench: about 23
    # list cells at 120 columns, 31 at 160 (1-line rows at both, spec 1.5.2).
    CSS = "#personas-library-pane { width: 20%; }"

    def __init__(self) -> None:
        super().__init__()
        self.selected: list[str] = []
        self.marks: list[tuple[tuple[str, str, str], ...]] = []
        self.wanted: list[tuple[str, str | None]] = []

    def compose(self) -> ComposeResult:
        yield Input(id="host-input")
        yield PersonasLibraryPane(id="personas-library-pane")

    def on_persona_entity_selected(self, message: PersonaEntitySelected) -> None:
        self.selected.append(message.entity_id)

    def on_persona_marks_changed(self, message: PersonaMarksChanged) -> None:
        self.marks.append(message.marks)

    def on_roleplay_item_list_window_wanted(
        self, message: RoleplayItemList.WindowWanted
    ) -> None:
        self.wanted.append((message.reason, message.target_id))


def _rows(first: int, last: int, kind: str = "character") -> tuple[LibraryRow, ...]:
    return tuple(
        LibraryRow(item_id=str(n), kind=kind, name=f"Row {n:04d}")
        for n in range(first, last + 1)
    )


async def _host(pilot, rows: tuple[LibraryRow, ...], *, total: int | None = None):
    pane = pilot.app.query_one(PersonasLibraryPane)
    await pane.update_rows(
        rows, total=len(rows) if total is None else total, noun="characters"
    )
    await pilot.pause()
    return pane, pane.item_list


async def test_options_are_kind_and_id_and_prompts_stay_literal():
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(
            pilot, (LibraryRow(item_id="7", kind="character", name="[b]x[/b]"),)
        )
        assert visible_item_ids(pilot.app.screen) == ("character:7",)
        assert item_at(pilot.app.screen, 0).lines == ("[b]x[/b]",)
        assert item_list.get_option_at_index(0).prompt.plain == "   [b]x[/b]"


async def test_duplicate_ids_are_dropped_and_never_raised():
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 3) + _rows(2, 2))
        assert visible_item_ids(pilot.app.screen) == (
            "character:1",
            "character:2",
            "character:3",
        )
        dropped = item_list.append_rows(_rows(3, 5))
        await pilot.pause()
        assert dropped == ("character:3",)
        assert visible_item_ids(pilot.app.screen)[-3:] == (
            "character:3",
            "character:4",
            "character:5",
        )


async def test_a_click_on_row_two_of_an_unfocused_list_loads_row_two():
    """Review Focus 5 / census F1: focus must not move the rows under the click."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 3))
        pilot.app.query_one("#host-input", Input).focus()
        await pilot.pause()
        assert not item_list.has_focus
        await select_roleplay_item(pilot, pilot.app.screen, "character:2", via="click")
        assert pilot.app.selected == ["2"]
        assert item_list.has_focus


async def test_focus_never_moves_the_first_row():
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 3))
        before = first_list_item(pilot.app.screen).region
        item_list.focus()
        await pilot.pause()
        assert first_list_item(pilot.app.screen).region == before


async def test_a_click_on_the_mark_column_marks_and_never_loads():
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        await _host(pilot, _rows(1, 3))
        await click_mark_column(pilot, pilot.app.screen, "character:2")
        assert item_at(pilot.app.screen, "character:2").marked
        assert item_at(pilot.app.screen, "character:2").painted_mark
        assert not item_at(pilot.app.screen, "character:1").painted_mark
        assert pilot.app.selected == []
        assert pilot.app.marks[-1] == (("character", "2", "Row 0002"),)


async def test_a_click_on_the_mark_column_marks_in_ascii_mode_too(ascii_mode):
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        await _host(pilot, _rows(1, 3))
        assert item_at(pilot.app.screen, 0).option_id == "character:1"
        await click_mark_column(pilot, pilot.app.screen, "character:3")
        assert item_at(pilot.app.screen, "character:3").marked
        assert item_at(pilot.app.screen, "character:3").painted_mark
        assert not item_at(pilot.app.screen, "character:2").painted_mark
        assert pilot.app.selected == []


async def test_m_and_space_act_only_on_the_focused_list():
    """RP-033: a letter on a toolbar button is inert; on the list it marks."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 3))
        pilot.app.query_one("#personas-library-new").focus()
        await pilot.press("m")
        await pilot.pause()
        assert not item_list.marked_ids()
        assert not item_at(pilot.app.screen, 0).painted_mark
        item_list.focus()
        await select_roleplay_item(pilot, pilot.app.screen, index=0, load=False)
        await pilot.press("m")
        await pilot.pause()
        assert item_list.marked_ids() == frozenset({"character:1"})
        assert item_at(pilot.app.screen, 0).painted_mark


async def test_the_cursor_comes_back_by_id_after_set_options():
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 50))
        await select_roleplay_item(pilot, pilot.app.screen, "character:30", load=False)
        await pane.update_rows(_rows(10, 60), total=51, noun="characters")
        await pilot.pause()
        assert highlighted_item_id(pilot.app.screen) == "character:30"


async def test_a_cut_name_shows_whole_in_the_hover_tooltip():
    name = "Captain Isolde Varga of the Deep-Survey Frigate Meridian's Wake"
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(
            pilot, (LibraryRow(item_id="1", kind="character", name=name),)
        )
        assert item_list.prompt_for("character:1").name_cut
        await pilot.hover(item_list, offset=(6, 0))
        await pilot.pause()
        assert str(item_list.tooltip) == name
        await pilot.hover(item_list, offset=(2, 0))
        await pilot.pause()
        assert str(item_list.tooltip) == "Click to mark"


async def test_end_past_the_loaded_rows_asks_for_the_rest_of_the_list():
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 200), total=348)
        item_list.set_more_available(True)
        item_list.focus()
        await pilot.press("end")
        await pilot.pause()
        assert ("end", None) in pilot.app.wanted
        assert highlighted_item_id(pilot.app.screen) != "character:200"


async def test_near_the_loaded_end_the_list_asks_for_one_window_once():
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 200), total=348)
        item_list.set_more_available(True)
        await select_roleplay_item(pilot, pilot.app.screen, "character:195", load=False)
        await select_roleplay_item(pilot, pilot.app.screen, "character:196", load=False)
        assert pilot.app.wanted == [("near_end", None)]


async def test_a_row_past_the_loaded_end_loads_through_then_shows_as_loaded():
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 200), total=348)
        item_list.set_more_available(True)
        assert pane.mark_active_row("character", "300") is False
        await pilot.pause()
        assert pilot.app.wanted == [("id", "character:300")]
        item_list.append_rows(_rows(201, 348))
        item_list.set_more_available(False)
        await pilot.pause()
        assert item_list.active_id == "character:300"
        assert highlighted_item_id(pilot.app.screen) == "character:300"
        assert item_at(pilot.app.screen, "character:300").active


async def test_prompts_are_rebuilt_once_per_settled_width_and_only_where_changed(
    monkeypatch,
):
    """RC-6: a resize rebuilds only the prompts it changes, once; height alone, none."""
    long_name = "A character whose name is far too long for a twenty-cell list"
    rows = _rows(1, 30) + (LibraryRow(item_id="99", kind="character", name=long_name),)
    built: list[str] = []
    original = frame_state.build_item_prompt

    def counting(**kwargs):
        built.append(kwargs["name"])
        return original(**kwargs)

    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, rows)
        assert item_list.prompt_for("character:99").cut
        monkeypatch.setattr(frame_state, "build_item_prompt", counting)
        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        assert built == []  # a height-only resize rebuilds nothing
        await pilot.resize_terminal(160, 40)
        await pilot.pause()
        await pilot.pause()
        assert built == [long_name]  # the one cut row, once


# ==== Moved from test_personas_library_pane_paging.py (deleted with paging, B2a) ====
# The toolbar's Sort and Tag stay until frame slice B2b; their pager
# assertions left with the pager.


class _ToolbarHost(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, None]] = []

    def compose(self) -> ComposeResult:
        yield PersonasLibraryPane(id="pane")

    def on_persona_sort_cycle_requested(self, message) -> None:
        self.events.append(("sort", None))

    def on_persona_tag_filter_requested(self, message) -> None:
        self.events.append(("tag", None))


async def test_sort_and_tag_buttons_post_intents():
    app = _ToolbarHost()
    async with app.run_test() as pilot:
        app.query_one(PersonasLibraryPane).set_mode("characters")
        await pilot.pause()
        await pilot.click("#personas-library-sort")
        await pilot.click("#personas-library-tag")
        await pilot.pause()
        assert ("sort", None) in app.events and ("tag", None) in app.events


async def test_tag_button_hidden_for_personas_visible_for_characters():
    app = _ToolbarHost()
    async with app.run_test():
        pane = app.query_one(PersonasLibraryPane)
        pane.set_mode("personas")
        assert app.query_one("#personas-library-tag").display is False
        assert app.query_one("#personas-library-sort").display is True
        pane.set_mode("characters")
        assert app.query_one("#personas-library-tag").display is True


async def test_sort_and_tag_hidden_for_lore():
    app = _ToolbarHost()
    async with app.run_test():
        pane = app.query_one(PersonasLibraryPane)
        pane.set_mode("lore")
        assert app.query_one("#personas-library-sort").display is False
        assert app.query_one("#personas-library-tag").display is False


async def test_set_sort_label_and_set_tag_label():
    app = _ToolbarHost()
    async with app.run_test():
        pane = app.query_one(PersonasLibraryPane)
        pane.set_sort_label("Sort: Recent")
        pane.set_tag_label("Tag: villain")
        assert str(app.query_one("#personas-library-sort").label) == "Sort: Recent"
        assert str(app.query_one("#personas-library-tag").label) == "Tag: villain"
