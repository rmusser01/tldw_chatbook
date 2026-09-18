"""TASK-32752: empty Notes actions fit the actual production pane."""

from dataclasses import replace

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_library_notes_w4_editor import _build_notes_host
from Tests.UI.test_library_notes_w4_layout import (
    _note_selected_app,
    _note_selected_projection,
)
from Tests.UI.test_library_notes_wave_list import _list_state
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas


def _assert_tree_actions_painted(screen, canvas):
    actions = list(canvas.query(".library-notes-tree-action-row Button"))
    assert actions
    for button in actions:
        assert button.region.width > 0
        assert button.region.x >= canvas.content_region.x
        assert button.region.right <= canvas.content_region.right, repr(
            {
                "action": button.id,
                "button": button.region,
                "canvas": canvas.content_region,
                "contract": canvas.pane_width,
                "measured": canvas._measured_width,
            }
        )
        painted = "\n".join(
            strip.text
            for strip in screen._compositor.render_strips()[
                button.region.y : button.region.bottom
            ]
        )
        assert str(button.label) in painted
    return actions


@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_empty_notes_actions_fit_the_actual_pane(request, size, theme):
    app = _build_notes_host(notes=[])
    async with app.run_test(size=size) as pilot:
        app.theme = theme
        screen = _active_library_screen(app)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("browse-notes")
        await _wait_for_selector(screen, pilot, "#library-notes-canvas")
        await pilot.pause()
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        actions = _assert_tree_actions_painted(screen, canvas)
        assert (
            canvas._tree_toolbar_width(canvas._effective_pane_width())
            == canvas.content_region.width
        )
        assert len(actions) == (1 if canvas.compact else 4)
        if not canvas.compact:
            remove = canvas.query_one("#library-notes-placement-remove", Button)
            assert remove.disabled
            assert (
                "select a note"
                in "\n".join(
                    strip.text for strip in screen._compositor.render_strips()
                ).lower()
            )


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_empty_notes_actions_repack_on_shrink_without_churning_on_growth(
    request, theme
):
    app = _build_notes_host(notes=[])
    async with app.run_test(size=(190, 48)) as pilot:
        app.theme = theme
        screen = _active_library_screen(app)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("browse-notes")
        await _wait_for_selector(screen, pilot, "#library-notes-canvas")
        await pilot.pause()
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        _assert_tree_actions_painted(screen, canvas)
        for size in ((170, 48), (80, 24), (170, 48)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            await pilot.pause()
            _assert_tree_actions_painted(screen, canvas)
        field = canvas.query_one("#library-notes-filter")
        await pilot.resize_terminal(190, 48)
        await pilot.pause()
        await pilot.pause()
        assert canvas.query_one("#library-notes-filter") is field
        _assert_tree_actions_painted(screen, canvas)


@pytest.mark.parametrize("selection", ["note", "protected", "folder"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_selected_note_and_folder_actions_keep_complete_labels(
    request, selection, theme
):
    projection = _note_selected_projection()
    folder, note = projection.rows
    if selection == "protected":
        note = replace(note, protected=True)
    projection = replace(projection, rows=(folder, note))
    selected = folder if selection == "folder" else note
    app = _note_selected_app(
        64,
        tree_projection=projection,
        tree_selected_placement_id=selected.placement_id,
    )
    app.theme = theme
    async with app.run_test(size=(170, 48)) as pilot:
        await pilot.pause()
        canvas = app.query_one("#library-notes-canvas", LibraryNotesCanvas)
        actions = _assert_tree_actions_painted(app.screen, canvas)
        assert len(actions) == 4
        if selection == "protected":
            assert canvas.query_one("#library-notes-placement-remove", Button).disabled
        # The measured fallback already excludes host chrome; it must still
        # produce complete labels when no screen contract has arrived.
        canvas.pane_width = 0
        canvas.refresh(recompose=True)
        await pilot.pause()
        _assert_tree_actions_painted(app.screen, canvas)


@private_profile_test
async def test_legacy_list_resize_does_not_repack_a_previous_tree(request):
    app = _note_selected_app(86)
    async with app.run_test(size=(170, 48)) as pilot:
        await pilot.pause()
        canvas = app.query_one("#library-notes-canvas", LibraryNotesCanvas)
        assert canvas.query(".library-notes-tree-action-row")
        canvas.sync_state(
            list_state=_list_state(),
            sort_mode="newest",
            filter_value="",
            mode="list",
            presentation_state=None,
            tree_projection=None,
            tree_selected_placement_id="",
            tree_deleted_folder_available=False,
            title_placeholder_only=canvas.title_placeholder_only,
            compact=canvas.compact,
            pane_width=86,
            create_running=False,
            create_status="",
            load_state=canvas.load_state,
            load_message="",
        )
        await pilot.pause()
        assert not canvas.query(".library-notes-tree-action-row")
        field = canvas.query_one("#library-notes-filter")
        canvas.styles.width = 64
        canvas.styles.max_width = 64
        canvas.apply_pane_width(64)
        await pilot.pause()
        assert canvas.query_one("#library-notes-filter") is field
