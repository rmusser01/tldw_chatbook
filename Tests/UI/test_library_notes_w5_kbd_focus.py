"""Library ▸ Notes critique #4, wave 5 -- keyboard and focus group.

Tasks 32607, 32608, 32609 and 32613: the footer describes a keyboard model
the screen must actually implement. See ``backlog/tasks/task-<id>*.md`` for
the acceptance criteria; each test names the task it pins.

Everything here reads production output -- the real
``_library_notes_footer_shortcuts`` tier, the real ``check_action`` gate, the
real mounted widgets and their COMPUTED styles -- never a value the test just
handed the code.

The focus-indicator tests compare a ``(text-style, outermost edge type)``
pair, deliberately dropping colour: that pair is what a monochrome capture
keeps. An accent-coloured ``solid`` border over a ``solid`` border is a
colour-only cue and fails here, which is how the two reading regions were
found.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from tldw_chatbook.css import build_css
from Tests.UI.test_library_file_notes_git import (
    _PanelHarness,
    _commit_draft_projection,
)
from Tests.UI.test_library_notes_w4_editor import (
    _build_notes_host,
    _chips,
    _open_first_note,
    _open_notes_list,
    _open_preview,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (
    LibraryFileNotesGitPanel,
)


# --- shared: the monochrome-visible half of a focus cue -------------------


def _focus_shape(widget) -> tuple[str, tuple[str, ...]]:
    """The part of a widget's current styling a monochrome dump keeps.

    ``text-style`` plus the OUTERMOST painted edge type per side -- the
    outline when one is set, else the border, because an outline is drawn
    over the border's own cells. Colours are dropped on purpose.
    """
    styles = widget.styles
    edges = tuple(
        (outline[0] or border[0])
        for border, outline in zip(tuple(styles.border), tuple(styles.outline))
    )
    return (str(styles.text_style), edges)


async def _shape_change_on_focus(screen, pilot, widget, park) -> bool:
    """Whether focusing ``widget`` changes anything colour-blind readers see."""
    park.focus()
    await pilot.pause()
    blurred = _focus_shape(widget)
    widget.focus()
    await pilot.pause()
    return blurred != _focus_shape(widget)


def _work_pane_stops(screen) -> list:
    """Every tab stop inside the open note pane, in the screen's own order."""
    return [
        widget
        for widget in screen.focus_chain
        if any(
            node.id == "library-note-work-pane"
            for node in widget.ancestors_with_self
        )
    ]


async def _open_info(screen, pilot) -> None:
    screen.query_one("#library-note-context", Button).press()
    await _wait_for_selector(screen, pilot, "#library-note-context-delete")
    await pilot.pause()


# --- task-32607 AC#1/#4: Info names the control Enter would fire ----------


#: Every Info control that owns Enter, and the label the footer must show.
#: The labels are production's own (``_LIBRARY_NOTE_EDITOR_ENTER_LABELS``);
#: what this pins is that the CONTEXT tier reaches them, which before the
#: fix it never did -- it rendered its literal "run action" on all of them.
_INFO_ENTER_STOPS = (
    ("library-note-context-back", "back to list"),
    ("library-note-edit", "edit note"),
    ("library-note-preview", "preview note"),
    ("library-note-context", "show info"),
    ("library-note-save", "save note"),
    ("library-note-use-in-console", "use in Console"),
    ("library-note-context-copy", "copy note"),
    ("library-note-context-export-md", "export Markdown"),
    ("library-note-context-export-txt", "export text"),
    ("library-note-context-delete", "delete note"),
)


@pytest.mark.asyncio
async def test_info_footer_names_the_focused_control_instead_of_run_action():
    """task-32607 AC#1/#4.

    Assessor A pressed Tab eleven times inside Info and read the identical
    "enter run action" chip on '‹ Notes', Keywords, Copy, Export Markdown,
    Export text and Delete -- so Enter was a coin toss between "copy to
    clipboard" and "delete this note". Of the four Notes footer tiers this
    was the only one not wrapped in the shared focus chip.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        assert screen._library_notes_focus_region() == "context"

        for widget_id, expected in _INFO_ENTER_STOPS:
            screen.query_one(f"#{widget_id}", Button).focus()
            await pilot.pause()
            chips = _chips(screen)
            assert chips.get("enter") == expected, (
                f"Info footer with #{widget_id} focused: {chips}"
            )
            assert chips.get("esc") == "back to note", chips


@pytest.mark.asyncio
async def test_info_drops_the_enter_chip_where_enter_does_nothing():
    """task-32607 AC#1: the keywords field owns no Enter.

    There is no ``Input.Submitted`` handler for
    ``#library-note-context-keywords``, so "enter run action" was the same
    lie one control smaller. The tier keeps the grammar the editor tier
    uses: a text field gets no Enter chip (nor does ``#library-note-title``
    in Edit), and the exit chip survives.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        screen.query_one("#library-note-context-keywords", Input).focus()
        await pilot.pause()
        chips = _chips(screen)
        assert "enter" not in chips, chips
        assert chips.get("esc") == "back to note", chips


@pytest.mark.asyncio
async def test_a_backlink_row_names_itself_even_though_it_has_no_dom_id():
    """task-32607 AC#1/#4: Info's backlink rows are identified by class.

    ``LibraryNotesCanvas._backlink_buttons`` gives them a ``note_id``
    attribute and ``library-note-backlink``, no ``id`` -- so the id-keyed
    label table could never name them and they fell back to the tier's
    literal. Asserted through the production resolver itself.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        backlink = Button("Some other note", classes="library-note-backlink")
        assert screen._library_focus_enter_label(backlink) == "open linked note"


# --- task-32607 AC#2 + task-32613 AC#1/#4: every stop paints -------------


@pytest.mark.parametrize("mode", ("editor", "preview", "context"))
@pytest.mark.asyncio
async def test_every_note_pane_tab_stop_paints_a_shape_change_on_focus(mode: str):
    """task-32613 AC#1/#4 and task-32607 AC#2.

    Enumerates the pane's own stops from the SCREEN's focus chain (so a new
    control cannot ship without an indicator) and fails on any whose
    ``(text-style, outermost edge type)`` is identical focused and blurred.

    Before the fix this failed on ``#library-note-preview-region`` and
    ``#library-note-context-region`` -- both kept a ``solid`` border and
    only swapped its colour, with the reset's ``*:focus { outline: solid }``
    repainting the same glyphs -- and on ``#library-note-context-keywords``,
    whose field styling was being spent on ``#library-note-keywords``, an
    undisplayed twin.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        if mode == "preview":
            await _open_preview(screen, pilot)
        elif mode == "context":
            await _open_info(screen, pilot)

        stops = _work_pane_stops(screen)
        assert len(stops) >= 7, [widget.id for widget in stops]

        park = screen.query_one("#library-notes-new", Button)
        flat = []
        for widget in stops:
            if not await _shape_change_on_focus(screen, pilot, widget, park):
                flat.append(widget.id or type(widget).__name__)
        assert not flat, (
            f"{mode}: these tab stops paint no monochrome-visible focus cue: "
            f"{flat}"
        )


@pytest.mark.asyncio
async def test_info_delete_reads_as_destructive_rather_than_disabled():
    """task-32607 AC#3.

    The ANSI decode showed Delete's label at #a5a5a5 against its siblings'
    #e1e1e1 -- ``.library-media-action-danger``'s only treatment is
    ``color: $ds-text-muted``, which in a group whose other members are
    enabled reads as THE disabled one. Compared against a real sibling and
    against the muted role, at runtime, not in the sheet.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        delete = screen.query_one("#library-note-context-delete", Button)
        copy = screen.query_one("#library-note-context-copy", Button)
        # Info's own metadata line is the pane's reference muted role.
        muted_line = screen.query_one("#library-note-context-meta")

        assert delete.styles.color != muted_line.styles.color, (
            "Delete still paints in the muted role its neighbours' inert "
            "copy uses"
        )
        assert delete.styles.color != copy.styles.color, (
            "Delete is indistinguishable from the neutral actions beside it"
        )

        # And the role is the readable error token, not the 3.06:1 one --
        # asserted in the built sheet because the token resolves per theme.
        sheet = (
            BUNDLED_STYLESHEET.parent / build_css.AGENTIC_SPLIT_SHEETS["library"]
        ).read_text(encoding="utf-8")
        block = sheet.split("#library-note-context-delete {", 1)
        assert len(block) == 2, "no #library-note-context-delete rule in the sheet"
        assert "color: $ds-status-error-readable;" in block[1].split("}", 1)[0]


# --- task-32607 AC#5: the list footer's map is true, then extended -------


@pytest.mark.asyncio
async def test_the_navigator_map_advertises_only_keys_that_fire():
    """task-32607 AC#5 and task-32609 AC#1/#2.

    Every printable key in this tier is swallowed by a focused text field,
    and the tier is chosen by REGION -- which still says "navigator" while
    focus sits in the filter box or, after Escape, in the rail's own
    "Search Library…" input. That is assessor A's walk: the footer read
    "/ find note" the whole time while "/" landed as a literal character
    and "abc" went into the rail search.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        screen.query_one("#library-notes-new", Button).focus()
        await pilot.pause()
        chips = _chips(screen)
        assert chips.get("n") == "new note", chips
        assert chips.get("/") == "find note", chips
        assert chips.get("g") == "go to folder", chips
        assert chips.get("esc") == "focus rail", chips
        for key in ("n", "/", "g"):
            assert screen.check_action(
                {
                    "n": "library_notes_new",
                    "/": "library_notes_focus_filter",
                    "g": "library_notes_focus_folders",
                }[key],
                (),
            ), f"advertised {key!r} but its own gate refuses it"

        for selector in ("#library-notes-filter", "#library-search-input"):
            screen.query_one(selector, Input).focus()
            await pilot.pause()
            typing_chips = _chips(screen)
            assert "/" not in typing_chips, (selector, typing_chips)
            assert "n" not in typing_chips, (selector, typing_chips)
            assert "g" not in typing_chips, (selector, typing_chips)
            assert typing_chips.get("esc") == "focus rail", typing_chips


@pytest.mark.asyncio
async def test_g_jumps_to_the_folder_tree_from_the_notes_toolbar():
    """task-32607 AC#5: "g go to folder" is a real key, not new copy."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        screen.query_one("#library-notes-new", Button).focus()
        await pilot.pause()
        await pilot.press("g")
        await pilot.pause()

        assert screen.focused is not None
        assert screen.focused.has_class("library-notes-folder-row"), (
            screen.focused.id
        )


@pytest.mark.asyncio
async def test_e_runs_export_selected_only_once_something_is_selected():
    """task-32607 AC#5: "e export selected" appears with the live key.

    Export selected is composed by the select strip alone and is disabled at
    zero selected, so the chip must not appear before a row is checked --
    the dimmed-but-advertised shape this group exists to remove.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        screen.query_one("#library-notes-select-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-export-selected")
        await pilot.pause()
        assert screen._notes_state.select_mode
        assert "e" not in _chips(screen), _chips(screen)
        assert not screen.check_action("library_notes_export_selected", ())

        screen.query_one("#library-notes-select-all", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.row_selection.count > 0,
            message="Select all never selected a row",
        )
        screen.query_one("#library-notes-select-all", Button).focus()
        await pilot.pause()
        assert _chips(screen).get("e") == "export selected", _chips(screen)
        assert screen.check_action("library_notes_export_selected", ())


# --- task-32609 AC#1/#3/#4: where "/" goes -------------------------------


@pytest.mark.asyncio
async def test_slash_from_a_notes_toolbar_button_reaches_the_notes_filter():
    """task-32609 AC#1/#3: not the rail search, and not as a character."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        notes_filter = screen.query_one("#library-notes-filter", Input)
        rail_search = screen.query_one("#library-search-input", Input)
        screen.query_one("#library-notes-sort", Button).focus()
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()

        assert screen.focused is notes_filter, screen.focused
        assert notes_filter.value == ""
        assert rail_search.value == ""


@pytest.mark.asyncio
async def test_slash_in_the_editor_is_not_advertised_by_the_notes_footer():
    """task-32609 AC#2/#4: the editor tier never claimed "/", and must not.

    Inside the open note "/" belongs to the body text; on a toolbar button
    it falls through to the screen-wide rail-search grab. Either way the
    Notes footer says nothing about it, which is the honest half of the
    contradiction the two assessors reported.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        for selector in ("#library-note-save", "#library-note-body"):
            screen.query_one(selector).focus()
            await pilot.pause()
            assert "/" not in _chips(screen), (selector, _chips(screen))
        assert not screen.check_action("library_notes_focus_filter", ())


# --- task-32613 AC#2/#3: the order out of a reading pane -----------------


@pytest.mark.asyncio
async def test_tab_out_of_previews_reading_region_reaches_the_mode_strip():
    """task-32613 AC#3.

    Textual orders the focus chain by screen POSITION, so the heading's
    "‹ Notes" is the pane's first stop and Preview's reading region is its
    LAST: one Tab wrapped the cycle onto the exit and a blind Enter closed
    the note with no warning. Reproduced headlessly before the fix -- Tab
    from ``#library-note-preview-region`` focused ``#library-note-back``.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_preview(screen, pilot)

        stops = _work_pane_stops(screen)
        assert stops[-1].id == "library-note-preview-region", [
            stop.id for stop in stops
        ]

        stops[-1].focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()

        assert screen.focused is not None
        assert screen.focused.id == "library-note-edit", screen.focused.id
        # The note is still open: the pane did not fall back to its
        # "Select a note to edit it here" empty state.
        assert screen._notes_state.view == "editor"
        assert not screen.query("#library-note-work-empty")


@pytest.mark.asyncio
async def test_tab_walks_every_stop_of_the_info_pane():
    """task-32613 AC#3, fix round 1 -- the half the first cut broke.

    Info's reading region is stop 7 of 12, not the last, so the Preview
    override must NOT fire there. The first cut applied it to both regions
    and the forward ring collapsed to six stops: Keywords, Copy, Export
    Markdown, Export text, Delete and "‹ Notes" became Shift+Tab-or-mouse
    only, while task-32607's footer went on naming "enter delete note" on
    them. Entry focus in Info lands on that region, so the very first
    forward Tab a reader pressed was the broken one.

    This walks with ``pilot.press("tab")`` rather than ``widget.focus()``:
    a focus-order pin that sets focus directly cannot see a redirect.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        stops = _work_pane_stops(screen)
        expected = [stop.id for stop in stops]
        assert "library-note-context-region" in expected, expected
        assert expected[-1] != "library-note-context-region", (
            "this pin only means anything while Info's region is NOT the "
            f"last stop: {expected}"
        )

        stops[0].focus()
        await pilot.pause()

        visited = [screen.focused.id]
        for _ in range(len(stops) - 1):
            await pilot.press("tab")
            await pilot.pause()
            focused = screen.focused
            assert focused is not None
            assert any(
                node.id == "library-note-work-pane"
                for node in focused.ancestors_with_self
            ), f"Tab left the note pane and landed on {focused.id!r}"
            visited.append(focused.id)

        missing = [stop for stop in expected if stop not in visited]
        assert not missing, (
            f"forward Tab never reached {missing}; walk was {visited}"
        )


@pytest.mark.asyncio
async def test_the_note_pane_toolbar_prefix_is_the_same_in_every_mode():
    """task-32613 AC#2 (the part that is achievable).

    A control's distance from the mode strip must not depend on which mode
    you arrived from. The pane's first six stops are the same controls in
    the same order in Edit, Preview and Info -- Back (whichever of its two
    ids that mode displays), then Edit / Preview / Info / Save / Use in
    Console -- so Shift+Tab counts from the strip are stable.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        def prefix() -> list[str]:
            return [
                "back" if widget.id.endswith("-back") else widget.id
                for widget in _work_pane_stops(screen)[:6]
            ]

        editor_prefix = prefix()
        assert editor_prefix == [
            "back",
            "library-note-edit",
            "library-note-preview",
            "library-note-context",
            "library-note-save",
            "library-note-use-in-console",
        ], editor_prefix

        await _open_preview(screen, pilot)
        assert prefix() == editor_prefix, prefix()
        await _open_info(screen, pilot)
        assert prefix() == editor_prefix, prefix()


# --- task-32608: Tab reaches the terminal action of a full-pane task -----


@pytest.mark.asyncio
async def test_tab_stays_inside_the_lasting_sync_pane():
    """task-32608 AC#1/#4.

    The lasting-sync canvas mounts in the same ``#library-note-work-pane``
    as the note editor but was not in the pane's closed Tab cycle, so Tab
    ran past the review pane's terminal action into the rail -- where Enter
    navigates the app to Conversations and the root was never activated.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        screen.query_one("#library-notes-add-from-files", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.view in {"lasting_add", "lasting_roots"},
            message=lambda: (
                "Add from files never entered a lasting view: "
                f"{screen._notes_state.view!r}"
            ),
        )
        await pilot.pause()

        stops = _work_pane_stops(screen)
        assert stops, "the lasting-sync canvas mounted no tab stops"
        stops[0].focus()
        await pilot.pause()

        visited = []
        for _ in range(len(stops) + 3):
            await pilot.press("tab")
            await pilot.pause()
            focused = screen.focused
            assert focused is not None
            assert any(
                node.id == "library-note-work-pane"
                for node in focused.ancestors_with_self
            ), f"Tab left the sync pane and landed on {focused.id!r}"
            visited.append(focused.id)
        assert len(set(visited)) == len(stops), visited

        # task-32613 AC#1: and every one of those stops paints. B reported
        # the sync form's first four stops showing no indicator at all.
        park = screen.query_one("#library-notes-new", Button)
        flat = [
            widget.id or type(widget).__name__
            for widget in stops
            if not await _shape_change_on_focus(screen, pilot, widget, park)
        ]
        assert not flat, f"sync-pane stops with no visible focus: {flat}"


def test_the_lasting_sync_views_are_in_the_work_pane_tab_cycle():
    """task-32608 AC#1/#4: the gate itself, so a later view cannot drop out.

    ``_library_note_work_pane_owns_tab`` keys on the VIEW, not the sync
    phase, so this covers the review phase the walk above cannot reach
    without a real vault.
    """
    from tldw_chatbook.UI.Library_Modules.screen_constants import (
        LIBRARY_NOTES_FULL_CANVAS_VIEWS,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    assert LIBRARY_NOTES_FULL_CANVAS_VIEWS <= set(
        LibraryScreen._LIBRARY_WORK_PANE_TAB_VIEWS
    )


class _BundledPanelHarness(_PanelHarness):
    """The Git panel under the app stylesheet the real screen loads.

    ``ConsolidatedCSSApp`` alone carries only the screen-scoped sheets, so a
    panel-only harness misses ``Input:focus``/``TextArea:focus`` and reports
    focus cues the real app does paint. Measured both ways before writing
    this: without the bundle the commit form's two fields looked cue-less;
    with it they go ``round`` -> ``solid``.
    """

    CSS_PATH = [str(BUNDLED_STYLESHEET)]


@pytest.mark.asyncio
async def test_the_commit_form_traps_tab_and_reaches_its_terminal_actions():
    """task-32608 AC#2/#4 and task-32613 AC#1.

    From the subject field every action of the form is reachable, focus
    never leaves the form, and each stop paints a monochrome-visible cue.
    Without the trap, Tab out of the subject walked the whole Library
    screen's position-ordered chain and the form's own footer -- 30 rows
    below -- was reached by nothing short of a computed mouse click.
    """
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    app = _BundledPanelHarness(panel)
    async with app.run_test(size=(120, 40)) as pilot:
        panel.render_commit_availability(
            _commit_draft_projection(
                binding_key=object(), staged_note_count=2, body="Preserved body"
            )
        )
        panel.query_one("#file-notes-git-commit-staged", Button).press()
        await pilot.pause()
        assert panel.commit_phase == "form"

        workflow = panel.query_one("#file-notes-git-commit-workflow")
        assert workflow.display
        assert workflow._trap_focus, "the open commit form does not trap Tab"

        subject = panel.query_one("#file-notes-git-commit-subject", Input)
        subject.focus()
        await pilot.pause()

        stops = list(app.screen.focus_chain)
        reached = []
        for _ in range(len(stops)):
            await pilot.press("tab")
            await pilot.pause()
            focused = app.screen.focused
            assert focused is not None
            assert panel in focused.ancestors_with_self
            reached.append(focused.id)
        for terminal in (
            "file-notes-git-commit-cancel",
            "file-notes-git-commit-review",
        ):
            assert terminal in reached, reached

        park = panel.query_one("#file-notes-git-commit-cancel", Button)
        flat = []
        for widget in stops:
            other = subject if widget is not subject else park
            other.focus()
            await pilot.pause()
            blurred = _focus_shape(widget)
            widget.focus()
            await pilot.pause()
            if blurred == _focus_shape(widget):
                flat.append(widget.id)
        assert not flat, f"commit-form stops with no visible focus: {flat}"

        # task-32608 AC#3: and the walk COMPLETES from the keyboard -- subject,
        # Tab to Review commit, Enter -- with no pointer anywhere in it.
        subject.value = "Keyboard-only commit"
        subject.focus()
        await pilot.pause()
        for _ in range(len(stops)):
            await pilot.press("tab")
            await pilot.pause()
            if app.screen.focused is not None and (
                app.screen.focused.id == "file-notes-git-commit-review"
            ):
                break
        else:
            raise AssertionError("Tab never reached Review commit")
        await pilot.press("enter")
        await pilot.pause()
        # The panel-only harness has no service behind it, so the draft stops
        # at the checking phase the real workspace resolves into "review";
        # what is pinned here is that the KEYPRESS ran the action at all.
        assert panel.commit_phase not in {"form", "list"}, panel.commit_phase
