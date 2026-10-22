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
