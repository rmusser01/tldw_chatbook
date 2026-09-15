"""Library ▸ Notes critique-#4 wave 5 -- import & preview correctness group.

Tasks 32612, 32616, 32618, 32620, 32621, 32622. Four of the six are the
screen asserting something that is not true, so each pin is taken at the
PRODUCER of the claim (the pure projection, the status resolver, the save
path) rather than at a renderer handed a constructed state.

See ``backlog/tasks/task-<id>*.md`` for the acceptance criteria; every test
below names the task and the AC it pins.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Markdown, Static, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _open_note_editor,
    _seed_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Utils.markdown_parsing import (
    front_matter_parser_factory,
    render_obsidian_callouts,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import render_preview_source
from tldw_chatbook.app import TldwCli


# --- shared harnesses -----------------------------------------------------


class _ChooserHost(App[None]):
    """The Add-from-files canvas under the two stylesheets the app parses."""

    CSS_PATH: ClassVar[list[str]] = [*TldwCli.CSS_PATH, *LibraryScreen.CSS_PATH]

    def __init__(self, snapshot) -> None:
        super().__init__()
        self.snapshot = snapshot

    def compose(self) -> ComposeResult:
        yield LibraryNotesAddFromFilesCanvas(self.snapshot)


class _PreviewHost(App[None]):
    """The exact Preview pair: the title Static above its rendered body."""

    CSS_PATH: ClassVar[list[str]] = [*TldwCli.CSS_PATH, *LibraryScreen.CSS_PATH]

    def __init__(self, body: str, title: str) -> None:
        super().__init__()
        self._body = body
        self._title = title

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="library-note-preview-region", can_focus=True):
            yield Static(
                self._title,
                id="library-note-preview-body-title",
                classes="destination-section",
                markup=False,
            )
            yield Markdown(
                render_preview_source(self._body, title=self._title),
                id="library-note-preview-body",
                parser_factory=front_matter_parser_factory(),
            )


def _frame(app: App) -> str:
    """Render what a reader would see, one line per row."""
    return "\n".join(
        "".join(segment.text for segment in strip).rstrip()
        for strip in app.screen._compositor.render_strips()
    )


def _build_notes_host(notes=None) -> LibraryHarness:
    """A Library harness over versioned seeded notes.

    ``version`` and ``last_modified`` are not decoration: a note seeded
    without them makes the first explicit save land as a CONFLICT, and the
    rename pin below then measures the conflict path instead of the save.
    """
    app = _build_test_app()
    _seed_conversations(app, [], notes=notes or _two_notes())
    return LibraryHarness(app)


# --- import-workflow fixtures (kept here: one group, one file) ------------

_APPROVAL_ID = "00000000-0000-4000-8000-000000000042"


def _preview_item(body: str, *, action=None, replace_content: bool = False):
    """One reviewed source whose single payload carries ``body``."""
    from tldw_chatbook.Notes.note_import_plan_models import (
        ImportAction,
        ImportClassification,
        ImportMatch,
        ImportMatchKind,
        ImportPreviewItem,
        ImportSource,
        ImportSourceKind,
        ParsedNotePayload,
        ProposedFolderMembership,
    )

    from pathlib import Path

    selected = action or ImportAction.CREATE_NEW
    updating = selected is ImportAction.UPDATE_EXISTING
    return ImportPreviewItem(
        item_id="item-1",
        source=ImportSource(
            kind=ImportSourceKind.SELECTED_FILE,
            display_path="vault/Daily/2026-09-14.md",
            source_path=Path("/vault/Daily/2026-09-14.md"),
        ),
        payloads=(ParsedNotePayload(title="Daily note", content=body),),
        memberships=(
            ProposedFolderMembership(payload_index=0, folder_segments=("Imported",)),
        ),
        classification=(
            ImportClassification.CHANGED_REPEAT
            if updating
            else ImportClassification.NEW
        ),
        reason="Ready after review.",
        default_action=selected,
        selected_action=selected,
        allowed_actions=(ImportAction.SKIP, selected),
        match=(
            ImportMatch(
                kind=ImportMatchKind.EXACT, note_id="note-1", note_version=7
            )
            if updating
            else None
        ),
        replace_content=replace_content,
        add_membership=not updating,
    )


def _receipt(*, imported: int = 1, updated: int = 0):
    from tldw_chatbook.Notes.note_import_execution_models import (
        ImportExecutionReceipt,
        ImportSessionState,
    )

    total = imported + updated
    return ImportExecutionReceipt(
        approval_id=_APPROVAL_ID,
        state=ImportSessionState.COMPLETED,
        total=total,
        completed=total,
        imported=imported,
        updated=updated,
        skipped=0,
        failed=0,
        retryable=0,
    )


def _receipt_snapshot(*, notes_written: int):
    from tldw_chatbook.Library.library_note_import_state import (
        LibraryNoteImportSnapshot,
    )

    return LibraryNoteImportSnapshot(
        phase="receipt",
        selected_names=("vault",),
        selection_kind="folder",
        destination="Imported",
        status_line="Import completed.",
        preview_items=(),
        page=1,
        page_count=1,
        can_check=False,
        check_disabled_reason="",
        can_import=False,
        import_disabled_reason="",
        receipt_line=f"{notes_written} notes created",
        notes_written=notes_written,
    )


class _ImportCanvasHost(App[None]):
    """The import canvas with the production stylesheets and a message log."""

    CSS_PATH: ClassVar[list[str]] = [*TldwCli.CSS_PATH, *LibraryScreen.CSS_PATH]

    def __init__(self, snapshot) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.messages: list[object] = []

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
            LibraryNoteImportCanvas,
        )

        yield LibraryNoteImportCanvas(self.snapshot, id="library-note-import-canvas")

    def on_library_note_import_canvas_view_imported_notes_requested(
        self, message
    ) -> None:
        self.messages.append(message)


class _NavigationHost(App[None]):
    """The picker's progressive directory listing, rooted at one folder."""

    def __init__(self, location) -> None:
        super().__init__()
        self._location = location

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Third_Party.textual_fspicker.parts import (
            DirectoryNavigation,
        )

        yield DirectoryNavigation(self._location)


async def _wait_for_loaded_listing(app, pilot, attempts: int = 200):
    from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation

    navigation = app.query_one(DirectoryNavigation)
    for _ in range(attempts):
        await pilot.pause()
        if navigation.listing_status.startswith("Loaded"):
            return navigation
    raise AssertionError(
        f"The listing never settled: {navigation.listing_status!r}"
    )


# --- task-32612: the chooser names the structural difference --------------


@pytest.mark.asyncio
async def test_the_chooser_asks_its_question_once():
    """AC#1: three near-identical header lines stacked over two buttons --
    the pane's "Add files to Library notes.", the status line's "Choose how
    files should relate to Library notes." and the body's "Choose the
    relationship before selecting a file or folder." (A cap 17)."""
    app = _ChooserHost(initial_lasting_sync_snapshot(lasting_available=True))
    async with app.run_test(size=(190, 40)) as pilot:
        await pilot.pause()
        frame = _frame(app)

    assert "Add files to Library notes." in frame, frame
    assert "Choose how files should relate" not in frame, frame
    assert "Choose the relationship before selecting" not in frame, frame


@pytest.mark.asyncio
async def test_each_relationship_names_what_happens_to_the_folder_structure():
    """AC#2: the largest consequence of this choice -- Import once reproduced
    the vault tree (A cap 23) while lasting sync put all 54 notes flat under
    one managed folder (A cap 51) -- was named by neither option."""
    app = _ChooserHost(initial_lasting_sync_snapshot(lasting_available=True))
    async with app.run_test(size=(190, 40)) as pilot:
        await pilot.pause()
        frame = " ".join(_frame(app).split())

    assert "reproducing your folder structure as Library folders" in frame, frame
    assert "collected in one managed Library folder" in frame, frame


@pytest.mark.asyncio
async def test_the_chooser_points_at_folder_files_as_the_third_answer():
    """AC#3: the option that actually matches "edit my vault where it is" is
    a MODE of this same screen, and the chooser never mentioned it."""
    app = _ChooserHost(initial_lasting_sync_snapshot(lasting_available=True))
    async with app.run_test(size=(190, 40)) as pilot:
        await pilot.pause()
        frame = " ".join(_frame(app).split())
        pointer = app.query_one("#notes-add-folder-files-pointer", Static)
        assert pointer.region.width > 0

    assert "Folder files" in frame, frame
    assert "imports nothing" in frame, frame


@pytest.mark.asyncio
async def test_the_chooser_renders_its_documented_back_control():
    """AC#4, taken on the REAL screen rather than the canvas in isolation:
    notes.md promises "the bar below holds only ‹ Notes" and critique #4
    reported nothing rendered there, with only Escape working."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, ".library-notes-row")
        screen.query_one("#library-notes-add-from-files", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.view == "lasting_add",
            message="Add from files never took over the view.",
        )
        back = await _wait_for_selector(screen, pilot, "#notes-sync-back")
        pinned = screen.query_one("#notes-sync-pinned-actions")

        assert back in screen._compositor.visible_widgets, (
            "The documented ‹ Notes control is not on screen"
        )
        assert pinned.region.contains_region(back.region)
        assert "Notes" in str(back.label)


# --- task-32616: the open note's row, and one honest Next -----------------


@pytest.mark.asyncio
async def test_a_rename_repaints_the_open_notes_row_without_losing_the_title():
    """AC#1 and AC#2 together. A caps 06/07: the editor heading read the new
    title and the status read "Saved", while the row beside them -- the only
    durable artefact on screen -- still read the old one. AC#2 is the other
    half: the task-32062 focus guard must still hold, so the reader's own
    title field keeps focus AND its text across the repaint."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot, "n-1")

        title_input = screen.query_one("#library-note-title", Input)
        title_input.focus()
        await pilot.pause()
        title_input.value = "My first note"
        await pilot.pause()
        screen.query_one("#library-note-save", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: any(
                "My first note" in str(row.label)
                for row in screen.query(".library-notes-row")
            ),
            message="The open note's row never caught up with its saved title.",
        )

        # AC#2: the editor was NOT rebuilt underneath the reader.
        assert screen.query_one("#library-note-title", Input) is title_input
        assert title_input.has_focus
        assert title_input.value == "My first note"


def test_a_body_only_save_does_not_ask_the_items_pane_to_repaint():
    """AC#2's cost control: the repaint above is gated on a genuine rename,
    so a body-only autosave -- one per debounce tick while typing -- costs
    the tree nothing. Pinned at the producer: the patch reports whether the
    label the list caches were rendering actually changed."""
    import inspect

    source = inspect.getsource(LibraryScreen._patch_library_note_list_from_session)
    assert "return title_changed" in source
    assert "return False" in source


def _next_instructions(screen) -> list[tuple[str, str]]:
    """Every "Next:" instruction the screen is painting, with its owner."""
    return [
        (static.id or "", str(static.renderable))
        for static in screen.query(Static)
        if "Next:" in str(static.renderable)
    ]


@pytest.mark.asyncio
async def test_only_one_pane_at_a_time_issues_a_next_instruction():
    """AC#3. A cap 04: the list's "Next: Create a note or add from files."
    beside the editor's "Next: Start typing." -- two instructions at once,
    and the list's is advice against what the reader is already doing.

    Re-derived live at dev 3b26c66ce0 (both lines present, verbatim). The
    finding's OTHER half -- that the editor's line goes stale after typing
    (cap 05) -- did NOT reproduce: after a title-only edit the body really is
    empty, so "Start typing." is the true next step, and it changes to "Keep
    editing" the moment the body has words. That half is recorded as
    not-a-defect on the task rather than "fixed"."""
    empty = dict(_two_notes()[0], content="")
    host = _build_notes_host(notes=[empty])
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot, "n-1")

        instructions = _next_instructions(screen)
        assert len(instructions) == 1, instructions
        assert instructions[0][0] == "library-note-work-authority", instructions

        # The list pane takes its instruction back the moment it is alone.
        title = screen.query_one("#library-note-title", Input)
        title.focus()
        await pilot.pause()
        body = screen.query_one("#library-note-body", TextArea)
        body.focus()
        await pilot.pause()
        body.text = "Some words that are definitely not nothing."
        await _wait_for_condition(
            pilot,
            lambda: "Keep editing" in str(
                screen.query_one("#library-note-work-authority", Static).renderable
            ),
            message="the work pane never moved past 'Start typing'",
        )
        assert len(_next_instructions(screen)) == 1, _next_instructions(screen)


# --- task-32618: an embed becomes dead text, and the review says so -------


def test_a_review_row_says_an_embed_will_show_as_text():
    """AC#1/AC#3: the review classifies the embedded PNG as "Unsupported ·
    Image — not a note" and never said the notes embedding it keep the raw
    syntax. The wording names what the reader will SEE, not the internal
    classification."""
    from tldw_chatbook.Library.library_note_import_state import _effect_summary
    summary = _effect_summary(_preview_item("Daily note\n\n![[diagram.png]]\n"))
    assert "shows as ![[…]] text, not the file" in summary, summary

    plain = _effect_summary(_preview_item("Daily note\n\nNo embeds.\n"))
    assert "![[" not in plain, plain


def test_an_embed_inside_a_code_span_is_not_counted():
    """A note that DOCUMENTS embed syntax is not a note with a dead embed --
    the same reason ``WIKILINK_SCAN`` matches code spans first."""
    from tldw_chatbook.Notes.note_import_plan_models import embedded_file_count

    assert embedded_file_count("Use `![[file.png]]` to embed.\n") == 0
    assert embedded_file_count("```\n![[file.png]]\n```\n") == 0
    assert embedded_file_count("![[a.png]] and ![[b.png]]\n") == 2
    # A plain wikilink is a link, not an embed.
    assert embedded_file_count("[[Another note]]\n") == 0


def test_the_receipt_repeats_the_embed_count():
    """AC#2: a user who skipped the review still has to learn it."""
    from tldw_chatbook.Library.library_note_import_state import _receipt_outcome

    line = _receipt_outcome(_receipt(imported=54), resolved_links=0, dead_embeds=3)
    assert "3 embedded files left as text" in line, line
    assert "1 embedded file left as text" in _receipt_outcome(
        _receipt(imported=1), dead_embeds=1
    )
    assert "embedded file" not in _receipt_outcome(_receipt(imported=1))


# --- task-32620: Preview shows one title, and a headed callout ------------


def test_a_callout_header_is_broken_from_its_body():
    """AC#3. The lines under a callout header are a lazy paragraph
    continuation, so "**Warning**" and the first body line rendered as one
    run of text (A cap 25, B cap 27) where the guide promises a quoted block
    headed by its type. Two trailing spaces are CommonMark's hard break."""
    rendered = render_obsidian_callouts("> [!warning]\n> Body follows.\n")
    assert rendered == "> **Warning**  \n> Body follows.\n", repr(rendered)


@pytest.mark.asyncio
async def test_preview_renders_a_callout_type_on_its_own_line():
    """The same fix through the widget that actually paints it."""
    app = _PreviewHost(
        "Intro.\n\n> [!warning]\n> The preview and the editor disagree.\n",
        "Markdown showcase",
    )
    async with app.run_test(size=(80, 16)) as pilot:
        await pilot.pause()
        lines = [line.strip("│ ") for line in _frame(app).splitlines()]

    assert any(line.endswith("Warning") for line in lines), lines
    assert not any("Warning The preview" in line for line in lines), lines


@pytest.mark.asyncio
async def test_a_frontmatter_titled_note_does_not_paint_a_second_title():
    """AC#1/AC#2. A cap 25: the note title left-aligned directly above a
    CENTRED body H1 -- two title-shaped lines, the second one a page banner.
    task-32551's de-duplication only fires on an exact match, which a note
    titled from Obsidian frontmatter never produces."""
    app = _PreviewHost(
        "# Library review\n\nBody paragraph.\n", "Q3 planning — library review"
    )
    async with app.run_test(size=(80, 16)) as pilot:
        await pilot.pause()
        lines = _frame(app).splitlines()

    heading = next(
        line for line in lines if "Library review" in line and "Q3 planning" not in line
    )
    paragraph = next(line for line in lines if "Body paragraph." in line)
    # The heading begins where the rest of the body does, rather than
    # floating to the middle of the reading measure like a page banner.
    # (Not compared against the title Static above it: the Markdown widget
    # carries its own one-cell padding, which is not what this pins.)
    assert heading.index("Library review") == paragraph.index("Body paragraph."), (
        f"heading {heading!r} is not aligned with the body {paragraph!r}"
    )


# --- task-32621: Folder files states what it lists, and claims no save ----


def test_no_file_selected_asserts_no_save_state():
    """AC#1. A cap 29: the right pane showed the chip "Saved" beside the
    words "No file selected" -- every save-state input is False when nothing
    is open, so the resolver's final ``else`` claimed a save for a file that
    does not exist."""
    from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
        resolve_file_note_status_channels,
    )

    empty = resolve_file_note_status_channels(root="/notes/vault", file_open=False)
    assert empty.content_recovery == "No file open."
    assert empty.safe_next_action == "Choose a file in the tree"

    opened = resolve_file_note_status_channels(root="/notes/vault", file_open=True)
    assert opened.content_recovery == "Saved"


def test_the_folder_files_tree_states_what_it_lists():
    """AC#2: the tree silently omitted notes.csv, meta.yaml, a Canvas folder
    and an attachments folder (A cap 29, B cap 35). The extensions named here
    are ``file_notes_service.SUPPORTED_EXTENSIONS``; if that set changes, this
    sentence is what has to change with it."""
    import inspect

    from tldw_chatbook.Notes.file_notes_service import SUPPORTED_EXTENSIONS
    from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
        LibraryFileNotesWorkspace,
    )

    source = inspect.getsource(LibraryFileNotesWorkspace._build_reader_items_pane)
    assert 'id="file-notes-tree-scope"' in source
    for extension in SUPPORTED_EXTENSIONS:
        assert extension in source, f"{extension} missing from the tree's legend"


def test_the_lasting_sync_review_says_which_sources_it_reads():
    """AC#3: lasting sync silently ignores the same .csv and .yaml sources
    Import once reports under Failed and Skipped, and "0 need attention" hid
    it. Neither path changes what it does; each now says which it does. The
    extensions are ``notes_sync_runtime._SYNC_FILE_EXTENSIONS``."""
    import inspect

    from tldw_chatbook.Notes.notes_sync_runtime import _SYNC_FILE_EXTENSIONS
    from tldw_chatbook.Widgets.Library import library_notes_add_from_files_canvas

    source = inspect.getsource(
        library_notes_add_from_files_canvas.LibraryNotesAddFromFilesCanvas
        ._compose_phase
    )
    assert 'id="notes-sync-review-scope"' in source
    for extension in _SYNC_FILE_EXTENSIONS:
        assert extension in source, f"{extension} missing from the review's scope line"


