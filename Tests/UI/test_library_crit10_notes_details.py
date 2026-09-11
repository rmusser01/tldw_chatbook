"""Library ▸ Notes create/draft and the Details rail panel -- critique #10.

Group `notes-details` of the critique-10 fix wave:

- task-32356: `ctrl+n` (and its `n` twin, which shares the same action) must
  land in a note, not in a nine-option chooser. The create canvas keeps the
  templates, behind one `From a template…` row.
- task-32358: the editor's chip must agree with the list and the rail count
  standing next to it. The blank row IS committed when it is created, so
  "Draft — not saved yet" was the one thing on screen that was false.
- task-32357: the Details panel speaks outcomes -- no "WIP", no bare
  eligible/blocked arithmetic, and the DB sizes behind a Diagnostics
  disclosure.
- task-32360 AC#2, handed to this group by the coordinator because both
  measured losses are composed in this group's canvas: below 64 columns the
  Notes status line and the browse toolbar must not be cut mid-word.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _notes_host(*, notes: int = 1) -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[
            {"title": f"Research Note {index}", "id": f"note-{index}"}
            for index in range(1, notes + 1)
        ],
    )
    return LibraryHarness(app)


async def _open_notes_list(screen, pilot) -> None:
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-filter")
    await pilot.pause()


# --- task-32356: ctrl+n creates, the chooser is the template path ----------


@pytest.mark.asyncio
async def test_ctrl_n_opens_a_blank_note_straight_away():
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("ctrl+n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        assert not screen.query(
            "#library-notes-create-blank"
        ), "no chooser between ctrl+n and the editor"
        # The plan's sketch asserted body focus; the create seam's own pin
        # ("Successful Create never focused the title field",
        # test_library_shell.py) has focused the title since LIB-14, which
        # is also what the placeholder-only title exists for. The pin wins:
        # what this task changes is the chooser, not where typing lands.
        assert getattr(screen.focused, "id", "") == "library-note-title"


@pytest.mark.asyncio
async def test_bare_n_opens_a_blank_note_straight_away():
    """`n` and `ctrl+n` share ``library_notes_new`` (task-32138) -- one key
    creating and the other choosing would be the drift that pin prevented."""
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        assert not screen.query("#library-notes-create-blank")


@pytest.mark.asyncio
async def test_templates_are_one_row_on_the_create_canvas():
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        screen.query_one("#library-notes-new", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        assert not screen.query(".library-notes-template-row"), (
            "the create canvas opens with the templates folded away"
        )
        opener = screen.query_one("#library-note-from-template", Button)
        assert str(opener.label) == "From a template…"

        opener.press()
        await pilot.pause()
        await pilot.pause()
        assert len(screen.query(".library-notes-template-row")) == 8


# --- task-32358: the chip agrees with the list -----------------------------


@pytest.mark.asyncio
async def test_the_draft_chip_and_the_list_agree_at_every_moment():
    host = _notes_host(notes=0)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("ctrl+n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        # The chip the reader is looking at, not the projection behind it
        # (`_library_note_status_line` is controller-owned now).
        status = str(screen.query_one("#library-note-status", Static).renderable)
        listed = bool(screen.query(".library-notes-row"))
        assert (status == "Empty note — type to keep it") is listed, (status, listed)


# --- task-32357 AC#1: the Details panel speaks outcomes --------------------


def _handoff_state(*, blocked: int):
    """A depth state carrying ``blocked`` blocked conversation rows."""
    from tldw_chatbook.Workspaces.display_state import (
        LibraryWorkspaceDepthState,
        LibraryWorkspaceSourceRow,
    )

    rows = tuple(
        LibraryWorkspaceSourceRow(
            item_type="conversation",
            item_id=f"chat-{index}",
            title=f"Conversation {index}",
            workspace_ids=(),
            workspace_label="Unscoped",
            visible=True,
            active_context_eligible=False,
            authority_label="",
            context_label="",
            recovery_copy=(
                "Copy or link this conversation into workspace w-1 before "
                "using it in Console."
            ),
            reason_code="not_in_active_workspace",
        )
        for index in range(blocked)
    )
    return LibraryWorkspaceDepthState(
        heading="Workspaces",
        workspace_label="Workspace: Local Default",
        workspace_name="Local Default",
        visibility_label="",
        handoff_label=f"Console/RAG handoff: 2 eligible, {blocked} blocked",
        context_handoff_enabled=False,
        context_handoff_tooltip="",
        source_authority_label="",
        collections_membership_label="",
        import_export_label="",
        source_rows=rows,
    )


def test_the_handoff_row_names_the_task_not_the_arithmetic():
    """task-32357 AC#1: the counts and the reason are task-32230's; only the
    sentence changes, from a status report into something to do."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    label = LibraryScreen._workspace_handoff_summary_label(
        None, _handoff_state(blocked=1)
    )
    assert label == (
        "1 item can't be used in Console yet · not in this workspace · "
        "Link it from the conversation's header"
    ), label

    unblocked = LibraryScreen._workspace_handoff_summary_label(
        None, _handoff_state(blocked=0)
    )
    assert unblocked == "2 eligible, 0 blocked", unblocked


@pytest.mark.asyncio
async def test_the_details_panel_says_where_the_content_lives():
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._set_library_rail_section("details", True)
        await pilot.pause()

        note = screen.query_one("#library-workspace-create-local-copy", Static)
        assert str(note.renderable) == (
            "Everything here is stored on this machine · syncing to a server "
            "isn't available yet."
        )
        rendered = " ".join(
            str(getattr(widget, "renderable", ""))
            for widget in screen.query(Static)
        )
        tooltips = " ".join(
            str(getattr(widget, "tooltip", "") or "") for widget in screen.query(Button)
        )
        assert "WIP" not in rendered and "WIP" not in tooltips


# --- task-32357 AC#2: DB sizes behind a diagnostics disclosure -------------


@pytest.mark.asyncio
async def test_db_sizes_live_behind_a_closed_diagnostics_disclosure():
    from tldw_chatbook.Widgets.destination_rail import DestinationRailSectionHeader

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.db_sizes_status = {
        "prompts": "180.0KB",
        "chachanotes": "1.1MB",
        "media": "508.0KB",
    }
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._set_library_rail_section("details", True)
        await pilot.pause()
        sizes = await _wait_for_selector(screen, pilot, "#library-details-db-sizes")

        header = screen.query_one(
            "#library-rail-section-header-diagnostics", DestinationRailSectionHeader
        )
        assert header.open is False
        body = screen.query_one("#library-rail-section-body-details-diagnostics")
        assert body.display is False
        assert sizes in body.walk_children(), "the sizes moved under Diagnostics"

        screen.query_one(
            "#console-rail-section-toggle-library-details-diagnostics", Button
        ).press()
        await pilot.pause()
        assert body.display is True
        assert header.open is True


@pytest.mark.asyncio
async def test_a_first_reading_mounts_the_disclosure_it_belongs_in():
    """Recompose discipline: the rail composes the Diagnostics disclosure
    only when there are sizes to put in it (an empty one costs a rail row
    -- it pushed Details ▸ Actions out of an 18-row rail), so the
    Details-open refresh must mount the disclosure too, not just the rows,
    when the first reading lands on a rail that composed without one."""
    from tldw_chatbook.Widgets.destination_rail import DestinationRailSectionHeader

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    if hasattr(app, "db_sizes_status"):
        delattr(app, "db_sizes_status")

    class _StubManager:
        async def update_db_sizes(self) -> None:
            app.db_sizes_status = {
                "prompts": "10.0KB",
                "chachanotes": "20.0KB",
                "media": "30.0KB",
            }

    app.db_status_manager = _StubManager()
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._set_library_rail_section("details", False)
        await pilot.pause()
        assert not screen.query("#library-rail-section-header-diagnostics")

        screen._set_library_rail_section("details", True)
        sizes = await _wait_for_selector(screen, pilot, "#library-details-db-sizes")
        header = screen.query_one(
            "#library-rail-section-header-diagnostics", DestinationRailSectionHeader
        )
        assert header.open is False
        body = screen.query_one("#library-rail-section-body-details-diagnostics")
        assert body.display is False
        assert sizes in body.walk_children()
        assert "10.0KB" in str(sizes.render())


# --- task-32360 AC#2: the compact Notes copy is never cut mid-word -------


@pytest.mark.asyncio
async def test_notes_copy_is_not_clipped_mid_word_below_64_columns():
    """task-32360 AC#2 (handed over from the layout branch, reproduced live
    at 60x24 and in this harness): the Notes status line lost "files." to a
    HEIGHT clip (the compact sheet caps it at two rows; the full line needs
    three in a 32-cell pane) and the browse toolbar painted "Sel" where
    "Select" belongs (three actions need 33 cells). Both are asserted here
    as what the reader sees -- painted text and pane geometry -- not as the
    class or the width that happens to produce them today.
    """
    host = _notes_host(notes=7)
    async with host.run_test(size=(60, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        for _ in range(10):
            await pilot.pause()

        assert screen._notes_state.compact, "60 columns must measure compact"
        canvas = screen.query_one("#library-notes-canvas")
        painted = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in screen._compositor.render_strips()
        )

        # The status line keeps its last word.
        authority = screen.query_one("#library-notes-authority", Static)
        assert "add from files." in painted, painted
        assert authority.region.height <= 2, authority.region

        # Every action is painted whole, inside the pane it belongs to.
        offenders = [
            (action.id, action.region, str(action.label))
            for action in screen.query("#library-notes-canvas .library-canvas-action")
            if action.display
            and (
                action.region.right > canvas.region.right
                or action.region.x < canvas.region.x
            )
        ]
        assert not offenders, (
            f"actions painted off the {canvas.region.width}-column pane: "
            f"{offenders}"
        )
        # Review F11: the crop shows as a MISSING word, not as a "Sel" run
        # followed by a newline (trailing spaces pad the row), so "Select"
        # in the paint is the whole check -- it is the clause that fails in
        # the reverted run.
        assert "Select" in painted, painted


@pytest.mark.parametrize(
    ("size", "keeps_prefix"), [((100, 30), True), ((60, 24), False)]
)
@pytest.mark.asyncio
async def test_the_authority_noun_survives_above_the_narrow_stage(size, keeps_prefix):
    """Review F1: the prefix was dropped on `compact`, which is every
    terminal under 120 columns -- so 64..119 lost the noun where nothing
    was clipping, and the guide's own "narrower than 64 columns" became
    false. It is the narrow STAGE that has no room, not compact.
    """
    host = _notes_host(notes=7)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        for _ in range(10):
            await pilot.pause()

        authority = str(
            screen.query_one("#library-notes-authority", Static).renderable
        )
        assert authority.startswith("Library notes · ") is keeps_prefix, (
            size,
            authority,
        )


def test_the_split_browse_row_cannot_oscillate():
    """Review F5: splitting the row costs a line, which can bring a
    scrollbar into the pane and shave it -- with a bare `needed > width`
    test that shaving flips the answer straight back and `on_resize`
    recomposes for ever. A split row rejoins only with cells to spare.
    """
    from tldw_chatbook.Widgets.Library.library_notes_canvas import (
        browse_row_overflows,
        browse_row_width,
    )

    needed = browse_row_width(("New", "Sort: Newest", "Select"))
    assert needed == 33, needed

    assert browse_row_overflows(32, needed, already_split=False) is True
    # the split's own line came back to shave the pane: stay split
    assert browse_row_overflows(33, needed, already_split=True) is True
    # real room again: rejoin
    assert browse_row_overflows(35, needed, already_split=True) is False
    # and an unsplit row that fits exactly stays unsplit
    assert browse_row_overflows(33, needed, already_split=False) is False
    # unmeasured panes keep the shape they have always had
    assert browse_row_overflows(0, needed, already_split=False) is False


# --- review F4/F7 round 2: copy names what is on the surface it paints on ---


@pytest.mark.asyncio
async def test_each_next_step_names_a_control_that_is_on_that_surface():
    """Review F4: the create canvas said "Next: Start typing" while its only
    controls were Blank note and From a template… -- the brief's string,
    written for the editor the key now goes to. Each surface names its own.
    """
    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        screen.query_one("#library-notes-new", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        await pilot.pause()
        # The create canvas paints in the WORK pane; the list pane keeps its
        # own authority line beside it.
        create_line = str(
            screen.query_one("#library-note-work-authority", Static).renderable
        )
        assert create_line.endswith(
            "Next: Press Blank note, or choose a template."
        ), create_line

        # …and the editor a fresh note lands in, whose body IS the surface,
        # keeps the brief's wording.
        await pilot.press("escape")
        await _wait_for_selector(screen, pilot, "#library-notes-filter")
        await pilot.press("ctrl+n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()
        editor_line = str(
            screen.query_one("#library-note-work-authority", Static).renderable
        )
        assert editor_line.endswith("Next: Start typing."), editor_line


@pytest.mark.asyncio
async def test_the_details_actions_carry_no_bare_acronym():
    """Review F7: AC#1 retired "WIP" from this panel and left "ACP handoff"
    standing, which is the same class of internal vocabulary for a
    first-time reader. Spell the concept, keep the acronym in parentheses.
    """
    import re

    host = _notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._set_library_rail_section("details", True)
        await pilot.pause()

        tooltip = str(
            screen.query_one("#library-create-local-workspace", Button).tooltip or ""
        )
        bare = [
            match.group()
            for match in re.finditer(r"\b[A-Z]{2,}\b", tooltip)
            if f"({match.group()})" not in tooltip
        ]
        assert not bare, (bare, tooltip)


@pytest.mark.asyncio
async def test_opening_diagnostics_refreshes_the_reading_it_reveals():
    """Qodo #2: task-4023 AC#3 made opening a disclosure the refresh trigger
    for the DB sizes, because a `display` toggle never recomposes and the
    line otherwise keeps whatever the cache held (measured live: 180.0KB
    against 4.8MB on disk). Nesting those rows one disclosure deeper put
    them back behind a toggle that refreshed nothing: with Details already
    open, opening Diagnostics revealed a cached reading.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.db_sizes_status = {
        "prompts": "180.0KB",
        "chachanotes": "1.1MB",
        "media": "508.0KB",
    }

    class _StubManager:
        """Writes whatever "disk" currently says, once per refresh."""

        def __init__(self) -> None:
            self.calls = 0
            self.on_disk = dict(app.db_sizes_status)

        async def update_db_sizes(self) -> None:
            self.calls += 1
            app.db_sizes_status = dict(self.on_disk)

    manager = _StubManager()
    app.db_status_manager = manager
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen._set_library_rail_section("details", True)
        sizes = await _wait_for_selector(screen, pilot, "#library-details-db-sizes")
        for _ in range(60):
            if "180.0KB" in str(sizes.render()):
                break
            await pilot.pause(0.02)
        assert manager.calls >= 1

        # The file grew while Details stayed open.
        manager.on_disk["prompts"] = "9.9MB"
        screen.query_one(
            "#console-rail-section-toggle-library-details-diagnostics", Button
        ).press()
        for _ in range(120):
            if "9.9MB" in str(
                screen.query_one("#library-details-db-sizes", Static).render()
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Opening Diagnostics revealed a cached reading: "
                f"{screen.query_one('#library-details-db-sizes', Static).render()!r} "
                f"after {manager.calls} refresh(es)"
            )
