"""Roleplay frame B2a: the items list (spec 1.5).

Pure contracts of ``UI/Persona_Modules/roleplay_frame_state.py``'s items-list
section, and (from Task 3 of the B2a plan) pane-level contracts of
``RoleplayItemList`` on a small styled host. Nothing here mounts
``PersonasScreen``, so this file runs in the UI Fast Lane; the mounted
contracts live in ``test_roleplay_item_list_app.py``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from rich.cells import cell_len

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.Persona_Modules.roleplay_frame_state as frame_state
import tldw_chatbook.UI.Screens.personas_screen as personas_screen_module
import tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane as pane_module
from rich.style import Style
from textual import events
from textual.app import ComposeResult
from textual.widgets import Input

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.roleplay_frame_harness import (
    click_mark_column,
    first_list_item,
    highlighted_item_id,
    item_at,
    painted_option_rows,
    painted_rows,
    select_roleplay_item,
    visible_item_ids,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_library_pane import (
    LibraryRow,
    PersonasLibraryPane,
    RoleplayItemList,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    PersonaActionRequested,
    PersonaEntitySelected,
    PersonaMarksChanged,
)
from tldw_chatbook.Character_Chat import Character_Chat_Lib
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


@pytest.mark.parametrize("builder", ["_dictionary_row", "_lore_row"])
def test_a_book_with_a_count_that_is_not_a_number_still_lists(builder):
    """B2a's final review (M10): dev only formatted ``entry_count``; B2a's
    plural scent took ``int()`` of it, so one odd value failed the whole book
    render. New with B2a; fixed here: the row lists without a count."""
    row_for = getattr(personas_screen_module.PersonasScreen, builder)
    odd = row_for({"id": 7, "name": "Odd Book", "entry_count": "n/a"})
    assert (odd.item_id, odd.name, odd.meta) == ("7", "Odd Book", None)
    assert odd.short is None
    fine = row_for({"id": 8, "name": "Fine Book", "entry_count": "3"})
    assert fine.meta in ("3 rules", "3 entries")


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
        (22, "✓ 3", ("Export", "Delete"), " "),
        (12, "✓ 3", ("Delete",), " "),
        (9, "", ("Delete",), ""),
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
    assert view.actions == tuple(
        label.split()[0].rstrip("…").lower() for label in expected_labels
    )
    assert view.count_tooltip == "3 marked, 2 hidden by the search or tag"


@pytest.mark.parametrize("can_export", [True, False])
def test_marks_bar_never_needs_more_cells_than_it_has(can_export):
    """Spec 1.5.5: the bar degrades by width; an action it shows is never cut.

    Below the narrowest three-action form whole actions drop, Clear first
    (Escape on the list clears too), then Export; then the count yields, so
    Delete never leaves the bar while marks exist. No form is the count
    alone: that left no bulk path at all at 80x24 (Task 5 review).
    """
    every = ("export", "delete", "clear") if can_export else ("delete", "clear")
    for width in range(cell_len("Delete"), 61):
        view = frame_state.marks_bar_view(12, 0, width, can_export=can_export)
        cells = cell_len(view.count_text) + sum(
            cell_len(view.separator + label) for label in view.labels
        )
        assert cells <= width, (width, view)
        assert len(view.actions) == len(view.labels)
        assert "delete" in view.actions, (width, view)
        if "clear" in view.actions:
            assert view.actions == every
    narrowest = frame_state.marks_bar_view(12, 0, 11, can_export=can_export)
    assert (narrowest.count_text, narrowest.actions) == ("✓ 12", ("delete",))


@pytest.mark.parametrize("count", [10, 12, 100])
def test_the_count_yields_before_delete_at_ten_cells(count):
    """80x24 gives the bar 10 content cells, and ``✓ 10 Delete`` needs 11.

    The count moves into Delete's tooltip, so bulk Delete stays reachable
    (the R32 path). The Task 5 review found the bar showing the count alone
    at 10 marks and more, with no action.
    """
    view = frame_state.marks_bar_view(count, 0, 10, can_export=True)
    assert (view.count_text, view.labels, view.separator, view.actions) == (
        "",
        ("Delete",),
        "",
        ("delete",),
    )
    assert view.count_tooltip == f"{count} marked"
    # Given exactly the cells the count and Delete need, the count comes back.
    roomy = frame_state.marks_bar_view(
        count, 0, cell_len(f"✓ {count} Delete"), can_export=True
    )
    assert (roomy.count_text, roomy.actions) == (f"✓ {count}", ("delete",))


def test_the_count_yields_before_delete_in_ascii_mode(ascii_mode):
    """ASCII mode's mark is ``[x]``: ``[x] 2 Delete`` needs 12 cells, not 10."""
    assert frame_state.marks_bar_view(2, 0, 12, can_export=True).count_text == "[x] 2"
    view = frame_state.marks_bar_view(2, 1, 10, can_export=True)
    assert (view.count_text, view.labels, view.actions) == (
        "",
        ("Delete",),
        ("delete",),
    )
    assert view.count_tooltip == "2 marked, 1 hidden by the search or tag"


def test_delete_never_drops_while_marks_exist():
    """Narrower than ``Delete`` itself the bar still offers Delete alone."""
    for width in range(cell_len("Delete")):
        view = frame_state.marks_bar_view(3, 0, width, can_export=True)
        assert (view.count_text, view.actions) == ("", ("delete",)), (width, view)


def test_marks_bar_is_absent_without_marks_and_drops_export_for_books():
    assert frame_state.marks_bar_view(0, 0, 60, can_export=True) is None
    view = frame_state.marks_bar_view(2, 0, 60, can_export=False)
    assert view.count_text == "✓ 2 marked"
    assert view.labels == ("Delete…", "Clear esc")
    assert view.actions == ("delete", "clear")


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


def test_the_screen_keeps_the_list_seams_tests_patch():
    """Spec 1.5.6: tests patch these names on ``personas_screen``, so they stay
    importable there as the library's own objects (a re-export, never a copy
    or a wrapper; the AC#8 counts patch ``get_character_page_for_ui`` there and
    see every window read). The page size is now the window size."""
    assert (
        personas_screen_module.count_character_page
        is Character_Chat_Lib.count_character_page
    )
    assert (
        personas_screen_module.get_character_page_for_ui
        is Character_Chat_Lib.get_character_page_for_ui
    )
    assert personas_screen_module.PERSONAS_LIBRARY_PAGE_SIZE == 200


def test_the_items_list_uses_only_public_option_list_api():
    """Spec 1.5.6: public ``OptionList`` API only. Textual's private line and
    render caches (``_lines``, ``_clear_caches``) change between releases."""
    source = Path(pane_module.__file__).read_text(encoding="utf-8")
    for private in ("._lines", "._clear_caches", "._option_render_cache"):
        assert private not in source, private


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


# ---- The window state's one owner (TASK-33910.34) ----


def _fence(query: str = "", generation: int = 1) -> frame_state.ListFence:
    return frame_state.ListFence(
        "characters", query, "name_asc", None, "local", None, generation
    )


def test_same_list_ignores_only_the_generation():
    assert frame_state.same_list(_fence(generation=1), _fence(generation=9))
    assert not frame_state.same_list(_fence("a"), _fence("b"))


def test_a_leave_keeps_landed_rows_usable_and_forgets_the_owed_move():
    window = frame_state.ListWindowState()
    window.landing(_fence(generation=0))
    window.want("end", read=True)
    window.leave(_fence(generation=0))
    assert window.generation == 1
    assert window.rows_current(_fence(generation=1))
    assert window.intent is None and not window.requested
    assert window.returned() is False


@pytest.mark.parametrize("stage", ["reading", "painting"])
def test_a_leave_before_a_replace_has_landed_and_painted_reloads_on_the_return(
    stage,
):
    window = frame_state.ListWindowState()
    window.landing(_fence(generation=0))
    if stage == "reading":
        window.reload_started()
        window.leave(_fence(generation=1))
    else:
        with window.painting():
            window.leave(_fence(generation=0))
    assert window.returned() is True
    assert window.returned() is False


def test_a_landing_drops_a_move_owed_on_another_list_and_replays_one_owed_on_its_own():
    window = frame_state.ListWindowState()
    window.landing(_fence("old", 1))
    window.asked_for(window.want("end", read=True), _fence("old", 1))
    window.landing(_fence("new", 2))
    assert window.intent is None
    owed = window.want("id", "character:7", read=True)
    window.asked_for(owed, _fence("new", 3))
    window.landing(_fence("new", 4))
    assert window.take_replay(_fence("new", 4)) is owed
    assert window.intent is None


def test_a_move_the_screen_has_not_seen_or_that_read_nothing_is_never_replayed():
    window = frame_state.ListWindowState()
    unseen = window.want("end", read=True)
    window.landing(_fence())
    assert window.take_replay(_fence()) is None and window.intent is unseen
    no_read = window.want("id", "character:7", read=False)
    window.asked_for(no_read, _fence())
    assert window.take_replay(_fence()) is None and window.intent is no_read


def test_a_newer_move_forgets_the_owed_one_and_a_late_request_finds_none():
    window = frame_state.ListWindowState()
    stale = window.want("end", read=True)
    window.user_moved()
    window.asked_for(stale, _fence())
    assert window.intent is None and stale.fence is None
    assert window.take_end() is False


def test_the_owed_row_is_met_once_it_arrives():
    window = frame_state.ListWindowState()
    window.want("id", "character:300", read=True)
    assert window.arrived({"character:1"}) is None
    assert window.arrived({"character:300"}) == "character:300"
    assert window.intent is None


def test_a_failed_read_answers_the_request_and_drops_only_a_move_that_read():
    """Fix round 1 (review finding 1): a failed read owes no move any more, and
    the list may ask again; a move that asked for no read (a row waiting for a
    later list) stays."""
    window = frame_state.ListWindowState()
    window.asked_for(window.want("end", read=True), _fence())
    window.read_failed()
    assert window.intent is None and not window.requested
    waiting = window.want("id", "character:7", read=False)
    window.ask()
    window.read_failed()
    assert window.intent is waiting and not window.requested


def test_a_blank_and_a_kind_switch_leave_no_rows_to_append_to():
    window = frame_state.ListWindowState()
    window.landing(_fence())
    window.forget_rows()
    assert not window.rows_current(_fence())
    window.landing(_fence())
    window.set_more(True)
    window.want("end", read=True)
    window.kind_switched()
    assert window.rows_fence is None and window.intent is None
    assert not window.more_available and not window.requested


def test_the_window_state_has_no_second_copy_in_the_list_or_the_screen():
    """AC#1: the five fields B2a split between the list and the screen, the
    reload-on-return flag and the page generation live only in
    ``ListWindowState`` (a name is matched whole, so ``_apply_pending_active``
    is not ``_pending_active``)."""
    names = (
        "_more_available",
        "_window_requested",
        "_pending_active",
        "_character_rows_fence",
        "_deferred_window_wanted",
        "_reload_on_return",
        "_character_page_generation",
    )
    for module in (pane_module, personas_screen_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for name in names:
            assert not re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", source), (
                module.__name__,
                name,
            )


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
        self.actions: list[str] = []
        self.toggled: list[tuple[str | None, str | None]] = []

    def compose(self) -> ComposeResult:
        yield Input(id="host-input")
        yield PersonasLibraryPane(id="personas-library-pane")

    def on_persona_entity_selected(self, message: PersonaEntitySelected) -> None:
        self.selected.append(message.entity_id)

    def on_persona_action_requested(self, message: PersonaActionRequested) -> None:
        self.actions.append(message.action)
        if message.action == "toggle_enabled":
            self.toggled.append((message.entity_kind, message.entity_id))

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


def _painted_name_style(item_list: RoleplayItemList, index: int) -> tuple:
    """The background and emphasis the compositor paints on one row's first name cell."""
    y = painted_option_rows(item_list)[index][0]
    x = item_list.content_region.x + len(
        frame_state.item_gutter(loaded=False, marked=False)
    )
    style = item_list.screen.get_style_at(x, y)
    return style.bgcolor, bool(style.bold), bool(style.underline)


async def test_the_cursor_is_painted_only_while_the_list_has_focus():
    """Spec 1.5.3 (RP-062): "Cursor, list not focused: not painted". The app's
    global OptionList rule paints the highlighted row bold and underlined
    whatever has focus; the Roleplay sheet's rest rule clears that, and only
    the focused list paints its cursor (its own background, bold underline)."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 3))
        pilot.app.query_one("#host-input", Input).focus()
        await select_roleplay_item(pilot, pilot.app.screen, "character:2", load=False)
        assert not item_list.has_focus
        assert _painted_name_style(item_list, 1) == _painted_name_style(item_list, 0)
        item_list.focus()
        await pilot.pause()
        cursor = _painted_name_style(item_list, 1)
        assert cursor[0] != _painted_name_style(item_list, 0)[0]
        assert cursor[1:] == (True, True)


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


async def test_a_mark_column_click_on_a_row_that_just_left_does_nothing():
    """B2a's final review (M5): a click's ``option`` meta can be a frame stale
    after ``set_options`` shrank the list. Reading that index raised
    ``OptionDoesNotExist``, which Textual turns into an app exit. New with
    B2a (dev's ListView had no mark column); fixed here."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 3))
        await pane.update_rows(_rows(1, 1), total=1, noun="characters")
        await pilot.pause()
        x = item_list.gutter.left + frame_state.gutter_mark_columns().start
        y = item_list.gutter.top
        stale = events.Click(
            item_list,
            x,
            y,
            0,
            0,
            1,
            False,
            False,
            False,
            screen_x=item_list.region.x + x,
            screen_y=item_list.region.y + y,
            style=Style(meta={"option": 2}),
        )
        item_list.on_click(stale)
        await pilot.pause()
        assert item_list.marked_ids() == frozenset()
        assert pilot.app.selected == []


async def test_arrows_move_the_cursor_and_never_load():
    """I6: highlight never loads; only Enter or a click does."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 10))
        item_list.focus()
        await select_roleplay_item(pilot, pilot.app.screen, index=0, load=False)
        await pilot.press("down", "down", "down", "up")
        await pilot.pause()
        assert highlighted_item_id(pilot.app.screen) == "character:3"
        assert pilot.app.selected == []
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.selected == ["3"]


async def test_hovering_a_row_moves_no_row():
    """I10 / spec 5.7.2 item 5: hover changes colour only, focused or not."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 10))
        before = painted_option_rows(item_list)
        content = item_list.content_region
        for focused in (False, True):
            if focused:
                item_list.focus()
            await pilot.hover(
                item_list,
                offset=(
                    content.x - item_list.region.x + 6,
                    before[2][0] - item_list.region.y,
                ),
            )
            await pilot.pause()
            assert item_list.is_mouse_over
            assert painted_option_rows(item_list) == before, focused
            assert first_list_item(pilot.app.screen).region.y == before[0][0]


async def test_m_and_space_act_only_on_the_focused_list():
    """RP-033: ``m`` and Space on a toolbar button are inert; on the list ``m``
    marks the cursor row and Space asks to turn its book on or off."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        pane.set_mode("lore")
        _pane, item_list = await _host(pilot, _rows(1, 3, kind="lore"))
        await select_roleplay_item(pilot, pilot.app.screen, index=0, load=False)
        pilot.app.query_one("#personas-library-new").focus()
        await pilot.press("m", "space")
        await pilot.pause()
        assert not item_list.marked_ids()
        assert not item_at(pilot.app.screen, 0).painted_mark
        assert pilot.app.actions == []
        item_list.focus()
        await pilot.pause()
        assert highlighted_item_id(pilot.app.screen) == "lore:1"
        await pilot.press("m")
        await pilot.pause()
        assert item_list.marked_ids() == frozenset({"lore:1"})
        assert item_at(pilot.app.screen, 0).painted_mark
        await pilot.press("space")
        await pilot.pause()
        assert pilot.app.actions == ["toggle_enabled"]
        assert pilot.app.toggled == [("lore", "1")]


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


async def test_a_kind_switch_forgets_the_window_state():
    """B2a's final review (I1): Characters' partial window left "more rows"
    set, so End in Lore books asked for a window the screen drops there and
    never moved. New with B2a (dev's list had no End); fixed here. The
    pending load-through goes too, so it cannot move the cursor later."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 200), total=348)
        item_list.set_more_available(True)
        assert pane.mark_active_row("character", "300") is False
        pane.set_mode("lore")
        await pane.update_rows(_rows(1, 5, kind="lore"), total=5, noun="lore books")
        await pilot.pause()
        assert not item_list.more_available
        item_list.focus()
        await select_roleplay_item(pilot, pilot.app.screen, "lore:1", load=False)
        pilot.app.wanted.clear()
        await pilot.press("end")
        await pilot.pause()
        assert highlighted_item_id(pilot.app.screen) == "lore:5"
        assert pilot.app.wanted == []
        pane.set_mode("characters")
        await pane.update_rows(_rows(1, 200), total=348, noun="characters")
        item_list.append_rows(_rows(201, 348))
        await pilot.pause()
        assert highlighted_item_id(pilot.app.screen) != "character:300"


@pytest.mark.parametrize(
    "move",
    ["home", "up", "down", "pageup", "pagedown", "mark", "selection", "enter"],
)
async def test_a_newer_cursor_move_or_selection_forgets_an_owed_end(move):
    """AC#2: End past the loaded rows owes the user the last row; each newer
    cursor move or selection takes that back (``enter`` is Enter in the
    search box, which puts the cursor on the first match)."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 200), total=348)
        item_list.set_more_available(True)
        item_list.focus()
        await select_roleplay_item(pilot, pilot.app.screen, "character:5", load=False)
        await pilot.press("end")
        await pilot.pause()
        assert item_list.window_state.intent is not None
        if move == "mark":
            await click_mark_column(pilot, pilot.app.screen, "character:5")
        elif move == "selection":
            pane.mark_active_row("character", "50")
        elif move == "enter":
            # Enter in the search box owes the first match: a newer move than
            # the End it replaces (TASK-33910.35).
            item_list.window_state.owe_first(_search_fence())
        else:
            await pilot.press(move)
        await pilot.pause()
        if move == "enter":
            assert item_list.window_state.landing_owed()
        else:
            assert item_list.window_state.intent is None


async def test_a_re_mark_of_the_open_row_keeps_an_owed_end_and_enter_on_it_does_not():
    """Fix round 1 (review finding 2): the screen re-marks the open row after a
    background refresh (a Console save of the open card). That is no user
    move, so End stays owed; Enter on that same row is one."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 200), total=348)
        item_list.set_more_available(True)
        item_list.focus()
        pane.mark_active_row("character", "5")
        await pilot.pause()
        await pilot.press("end")
        await pilot.pause()
        owed = item_list.window_state.intent
        assert owed is not None and owed.reason == "end"
        pane.mark_active_row("character", "5")
        assert item_list.window_state.intent is owed
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.selected[-1:] == ["5"]
        assert item_list.window_state.intent is None


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


@pytest.mark.parametrize("mode", ["characters", "personas"])
async def test_a_longer_label_paints_whole_on_a_one_row_bar(mode):
    """Button.label never re-measured an auto-width button, so a label that
    grew wrapped onto an unpainted second line while the bar was one row.
    Exists on dev in Personas (whose bar is one row at 220); frame slice
    B2b's New chooser makes Characters' bar one row too, and fixes it."""
    app = _ToolbarHost()
    async with app.run_test(size=(120, 36)) as pilot:
        pane = app.query_one(PersonasLibraryPane)
        pane.set_mode(mode)
        await pilot.pause()
        assert not pane.has_class("personas-library-stacked-controls")
        pane.set_sort_label("Sort: Recent edit")
        if mode == "characters":
            pane.set_tag_label("Tag: villains")
        await pilot.pause()
        painted = "\n".join(painted_rows(app.screen))
        assert "Sort: Recent edit" in painted
        if mode == "characters":
            assert "Tag: villains" in painted


# ==== Marks by id and the marks bar (spec 1.5.5; B2a plan Task 5) ====


class _MarksHost(_PaneHost):
    """The pane wide enough for the marks bar's full form (spec 1.5.5)."""

    CSS = "#personas-library-pane { width: 60; }"


def _bar_text(pilot) -> str:
    """The marks bar's row exactly as painted ("" while it is hidden)."""
    bar = pilot.app.query_one("#personas-library-marks")
    if not bar.display:
        return ""
    region = bar.content_region
    row = painted_rows(pilot.app.screen)[region.y]
    return row[region.x : region.right].rstrip()


async def _mark(pilot, *option_ids: str) -> None:
    for option_id in option_ids:
        await select_roleplay_item(pilot, pilot.app.screen, option_id, load=False)
        await pilot.press("m")
    await pilot.pause()


async def test_the_marks_bar_shows_only_while_marks_exist():
    async with _MarksHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 10))
        assert _bar_text(pilot) == ""
        item_list.focus()
        await _mark(pilot, "character:2")
        assert _bar_text(pilot) == "✓ 1 marked · Export… · Delete… · Clear esc"
        await _mark(pilot, "character:2")
        assert _bar_text(pilot) == ""


async def test_marks_survive_a_search_and_count_the_hidden_ones():
    """AC#5: marks are kept by id; a filter that hides two says so."""
    async with _MarksHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 10))
        item_list.focus()
        await _mark(pilot, "character:1", "character:2", "character:3")
        await pane.update_rows(_rows(3, 10), total=10, noun="characters", filtered=True)
        await pilot.pause()
        assert _bar_text(pilot).startswith("✓ 3 marked (2 hidden) · Export…")
        assert item_at(pilot.app.screen, "character:3").marked
        await pane.update_rows(_rows(1, 10), total=10, noun="characters")
        await pilot.pause()
        assert _bar_text(pilot).startswith("✓ 3 marked · Export…")
        assert [item_at(pilot.app.screen, i).marked for i in range(4)] == [
            True,
            True,
            True,
            False,
        ]


async def test_the_marks_bar_takes_the_bottom_row_and_moves_no_row_above_it():
    """Spec 1.5.5 / I10: the bar appears below the list; no row above it moves."""
    async with _MarksHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 10))
        before = painted_option_rows(item_list)
        item_list.focus()
        await _mark(pilot, "character:1")
        bar = pilot.app.query_one("#personas-library-marks")
        assert bar.region.y == item_list.region.bottom
        assert bar.region.height == 1
        assert bar.region.bottom <= pane.content_region.bottom
        after = painted_option_rows(item_list)
        assert {i: before[i] for i in after} == after


async def test_marks_are_kept_per_kind_across_a_kind_switch():
    async with _MarksHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 10))
        item_list.focus()
        await _mark(pilot, "character:4")
        pane.set_mode("lore")
        await pane.update_rows(_rows(1, 3, kind="lore"), total=3, noun="lore books")
        await pilot.pause()
        assert _bar_text(pilot) == ""
        assert pilot.app.marks[-1] == ()
        pane.set_mode("characters")
        await pane.update_rows(_rows(1, 10), total=10, noun="characters")
        await pilot.pause()
        assert _bar_text(pilot).startswith("✓ 1 marked")
        assert pilot.app.marks[-1] == (("character", "4", "Row 0004"),)


async def test_books_offer_no_bulk_export():
    async with _MarksHost().run_test(size=(120, 36)) as pilot:
        pane = pilot.app.query_one(PersonasLibraryPane)
        pane.set_mode("lore")
        _pane, item_list = await _host(pilot, _rows(1, 3, kind="lore"))
        item_list.focus()
        await _mark(pilot, "lore:2")
        assert _bar_text(pilot) == "✓ 1 marked · Delete… · Clear esc"


async def test_export_and_delete_post_their_intents():
    async with _MarksHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 10))
        item_list.focus()
        await _mark(pilot, "character:1")
        assert await pilot.click("#personas-library-marks-export")
        assert await pilot.click("#personas-library-marks-delete")
        await pilot.pause()
        assert pilot.app.actions == ["export_marked", "delete_marked"]


async def test_clear_hides_the_bar_and_focus_lands_on_the_list():
    """I1: the bar hides under its own focused button; focus goes to the list."""
    async with _MarksHost().run_test(size=(120, 36)) as pilot:
        _pane, item_list = await _host(pilot, _rows(1, 10))
        item_list.focus()
        await _mark(pilot, "character:1", "character:2")
        clear = pilot.app.query_one("#personas-library-marks-clear")
        clear.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert _bar_text(pilot) == ""
        assert not item_list.marked_ids()
        assert pilot.app.focused is item_list


# ---- The first-match landing (TASK-33910.35) ----


def _search_fence(**changes):
    base = dict(
        mode="characters",
        query="isolde",
        sort_key="relevance",
        tag=None,
        runtime_source="local",
        server_id=None,
        generation=3,
    )
    base.update(changes)
    return frame_state.ListFence(**base)


def test_the_first_match_landing_belongs_to_its_list():
    window = frame_state.ListWindowState()
    window.owe_first(_search_fence())
    assert window.landing_owed()
    assert not window.take_first(_search_fence(query="other"))
    assert window.landing_owed()
    intent = window.intent
    assert window.take_first(_search_fence(generation=9)) is intent  # any generation
    assert not window.landing_owed()
    assert not window.take_first(_search_fence())


def test_landing_another_list_or_a_failed_read_or_a_newer_move_drops_the_first_match():
    window = frame_state.ListWindowState()
    for change in (
        {"sort_key": "name_asc"}, {"tag": "other"}, {"mode": "lore"},
        {"runtime_source": "server", "server_id": "other"},
    ):
        window.owe_first(_search_fence())
        window.defer_first_activation()
        window.landing(_search_fence(**change))
        assert window.intent is None
    window.owe_first(_search_fence())
    window.defer_first_activation()
    window.read_failed()
    assert not window.landing_owed()
    window.owe_first(_search_fence())
    window.user_moved()
    assert not window.landing_owed()
    window.owe_first(_search_fence())
    window.leave(_search_fence())
    assert not window.landing_owed()
    window.owe_first(_search_fence())
    window.want("end", read=False)  # a newer move than the owed landing
    assert not window.landing_owed()


async def test_enter_on_the_list_is_inert_while_the_first_match_is_owed():
    """Enter on a row opens it; while the searched list's rows are pending the
    cursor is on rows they replace, so Enter opens nothing and the landing
    stays owed (it is not a newer move)."""
    async with _PaneHost().run_test(size=(120, 36)) as pilot:
        pane, item_list = await _host(pilot, _rows(1, 20), total=20)
        item_list.focus()
        item_list.highlighted = 3
        item_list.window_state.owe_first(_search_fence())
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.selected == []
        assert item_list.window_state.landing_owed()
        item_list.window_state.user_moved()
        await pilot.press("enter")
        await pilot.pause()
        assert len(pilot.app.selected) == 1


def test_a_landing_owed_on_another_list_is_dropped_when_that_list_can_never_land():
    """A non-character render ends without landing(): ``landed_elsewhere`` drops
    a first-match landing owed on a superseded list, keeps one owed on this
    list, and leaves an owed End alone."""
    window = frame_state.ListWindowState()
    window.owe_first(_search_fence())
    window.landed_elsewhere(_search_fence(generation=9))
    assert window.landing_owed()
    window.landed_elsewhere(_search_fence(query="newer"))
    assert not window.landing_owed()
    window.want("end", read=False)
    window.landed_elsewhere(_search_fence(query="newer"))
    assert window.intent is not None and window.intent.reason == "end"


# ---- The New chooser (frame slice B2b, spec 1.4.1 "New…", 1.4.6) ----

_CHOOSER_ORDER = [
    "character:blank",
    "character:concept",
    "character:portrait",
    "persona:blank",
    "persona:portrait",
    "lore",
    "dictionary",
]


@pytest.mark.parametrize(
    ("mode", "blank"),
    [
        ("characters", "character:blank"),
        ("personas", "persona:blank"),
        ("lore", "lore"),
        ("dictionaries", "dictionary"),
    ],
)
def test_new_chooser_lists_every_create_variant_with_the_current_blank_preselected(
    mode, blank
):
    options = frame_state.new_chooser_options(
        mode, local_characters=True, local_runtime=True
    )
    assert [option.choice_id for option in options] == _CHOOSER_ORDER
    assert [option.choice_id for option in options if option.preselected] == [blank]
    assert all(option.enabled and option.reason is None for option in options)
    by_id = {option.choice_id: option for option in options}
    assert (
        by_id["character:portrait"].label == "Character · From a portrait (Actor Pack)"
    )
    assert by_id["persona:portrait"].mode == "personas"
    assert (by_id["lore"].mode, by_id["dictionary"].mode) == ("lore", "dictionaries")


def test_new_chooser_disables_local_only_variants_under_the_server_runtime_and_says_why():
    options = frame_state.new_chooser_options(
        "personas", local_characters=False, local_runtime=False
    )
    off = {option.choice_id: option.reason for option in options if not option.enabled}
    assert off == {
        "character:blank": frame_state.SERVER_CHARACTERS_READ_ONLY,
        "character:concept": frame_state.SERVER_CHARACTERS_READ_ONLY,
        "character:portrait": frame_state.SERVER_CHARACTERS_READ_ONLY,
        "persona:portrait": frame_state.PORTRAIT_NEEDS_LOCAL,
    }
    assert [option.choice_id for option in options if option.preselected] == [
        "persona:blank"
    ]
    # The screen's server read-only tooltip is the same words.
    assert (
        personas_screen_module._SERVER_READ_ONLY_TOOLTIP
        == frame_state.SERVER_CHARACTERS_READ_ONLY
    )
    # Characters' own Blank is off there: the first row that runs is picked.
    options = frame_state.new_chooser_options(
        "characters", local_characters=False, local_runtime=False
    )
    assert [option.choice_id for option in options if option.preselected] == [
        "persona:blank"
    ]


def test_new_chooser_rows_follow_the_ascii_glyph_mode(ascii_mode):
    options = frame_state.new_chooser_options(
        "characters", local_characters=True, local_runtime=True
    )
    assert options[0].label == "Character - Blank"


class _ChooserHost(ConsolidatedCSSApp):
    """Pushes one ``RoleplayChoiceModal`` and keeps what it returns."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, modal) -> None:
        super().__init__()
        self._modal = modal
        self.results: list[str | None] = []

    def on_mount(self) -> None:
        self.push_screen(self._modal, self.results.append)


def _chooser(title: str, name: str):
    from tldw_chatbook.Widgets.Persona_Widgets.roleplay_choice_modal import (
        ChoiceOption,
        RoleplayChoiceModal,
    )

    # Each row fits the 50-cell box on one line, so a row reads back whole.
    return RoleplayChoiceModal(
        title,
        [
            ChoiceOption("card", f"Card {name}"),
            ChoiceOption("hinted", "Hinted", hint=name),
            ChoiceOption("lore", "Off", enabled=False, reason=name),
            ChoiceOption("dictionary", "Chat dictionary"),
        ],
        preselect="dictionary",
    )


@pytest.mark.parametrize("name", HOSTILE_NAMES)
async def test_the_choice_modal_paints_literally_and_escape_returns_none(name):
    app = _ChooserHost(_chooser(f"Import {name} as:", name))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        rows = painted_rows(pilot.app.screen)
        for text in (
            f"Import {name} as:",
            f"Card {name}",
            f"Hinted  {name}",
            f"Off  {name}",
            "enter picks · esc cancels",
        ):
            assert any(text in row for row in rows), (text, rows)
        await pilot.press("escape")
        await pilot.pause()
        assert app.results == [None]


async def test_the_choice_modal_opens_on_its_preselect_and_enter_returns_that_id():
    app = _ChooserHost(_chooser("New", "x"))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.results == ["dictionary"]


async def test_a_disabled_choice_cannot_be_picked():
    app = _ChooserHost(_chooser("New", "x"))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        await pilot.press("up", "enter")
        await pilot.pause()
        # Up from Chat dictionary skips the disabled row.
        assert app.results == ["hinted"]
