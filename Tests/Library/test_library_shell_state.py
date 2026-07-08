from tldw_chatbook.Constants import TAB_INGEST
from tldw_chatbook.Library.library_shell_state import (
    LibraryShellInput,
    build_library_shell_state,
)


def test_shell_sections_rows_and_targets_are_fixed():
    shell = build_library_shell_state(LibraryShellInput(
        media_count=17, conversations_count=128, notes_count=42, collections_count=5,
    ))
    assert shell.header_line == "Library | Local"
    assert [s.section_id for s in shell.sections] == ["browse", "create", "ingest"]
    assert [s.title for s in shell.sections] == ["Browse", "Create", "Ingest"]
    browse = shell.sections[0]
    assert [r.row_id for r in browse.rows] == [
        "browse-media", "browse-conversations", "browse-notes", "browse-collections",
        "browse-search",
    ]
    assert [r.title for r in browse.rows] == [
        "Media", "Conversations", "Notes", "Collections", "Search / RAG"
    ]
    assert browse.rows[4].target_kind == "mode" and browse.rows[4].target_id == "search"
    assert browse.rows[4].count is None
    conv = browse.rows[1]
    assert (conv.target_kind, conv.target_id, conv.count) == ("canvas", "conversations", 128)
    media = browse.rows[0]
    assert (media.target_kind, media.target_id) == ("canvas", "media")
    notes = browse.rows[2]
    assert (notes.target_kind, notes.target_id, notes.count) == ("canvas", "notes", 42)
    create_ids = [r.row_id for r in shell.sections[1].rows]
    assert create_ids == ["create-note", "create-study", "create-flashcards", "create-quizzes"]
    assert [r.title for r in shell.sections[1].rows] == [
        "New note", "Study decks", "Flashcards", "Quizzes"
    ]
    assert (shell.sections[1].rows[0].target_kind, shell.sections[1].rows[0].target_id) == (
        "canvas", "notes-create",
    )
    ingest = shell.sections[2]
    assert [r.title for r in ingest.rows] == ["Import media", "Import / Export"]
    assert ingest.rows[0].target_id == TAB_INGEST
    assert (ingest.rows[1].target_kind, ingest.rows[1].target_id) == ("mode", "import-export")
    assert all(r.count is None for r in shell.sections[1].rows)


def test_empty_selection_yields_landing_canvas():
    shell = build_library_shell_state(LibraryShellInput())
    assert shell.canvas_kind == "empty"
    assert shell.canvas_empty_copy == "Search, pick a content type, or ingest something new."


def test_conversations_selection_yields_conversations_canvas():
    shell = build_library_shell_state(
        LibraryShellInput(conversations_count=3), selected_row_id="browse-conversations"
    )
    assert (shell.canvas_kind, shell.canvas_target) == ("conversations", "")
    assert shell.selected_row_id == "browse-conversations"


def test_media_selection_yields_media_canvas():
    shell = build_library_shell_state(
        LibraryShellInput(media_count=5), selected_row_id="browse-media"
    )
    assert (shell.canvas_kind, shell.canvas_target) == ("media", "")
    assert shell.selected_row_id == "browse-media"


def test_mode_selection_yields_mode_canvas():
    shell = build_library_shell_state(LibraryShellInput(), selected_row_id="create-flashcards")
    assert (shell.canvas_kind, shell.canvas_target) == ("mode", "flashcards")


def test_browse_notes_row_targets_notes_canvas():
    shell = build_library_shell_state(
        LibraryShellInput(notes_count=42), selected_row_id="browse-notes"
    )
    notes_row = next(
        r for section in shell.sections for r in section.rows if r.row_id == "browse-notes"
    )
    assert notes_row.target_kind == "canvas"
    assert notes_row.target_id == "notes"
    assert (shell.canvas_kind, shell.canvas_target) == ("notes", "")
    assert shell.selected_row_id == "browse-notes"


def test_create_note_row_targets_notes_create_canvas():
    shell = build_library_shell_state(LibraryShellInput(), selected_row_id="create-note")
    row = next(
        r for section in shell.sections for r in section.rows if r.row_id == "create-note"
    )
    assert row.target_kind == "canvas"
    assert row.target_id == "notes-create"
    assert (shell.canvas_kind, shell.canvas_target) == ("notes-create", "")
    assert shell.selected_row_id == "create-note"


def test_screen_and_unknown_rows_resolve_to_empty_canvas():
    # browse-notes and create-note used to be "screen" rows resolving to an
    # empty canvas; they now target real canvases (see the two tests above).
    # ingest-import-media remains the surviving "screen" row.
    for row_id in ("ingest-import-media", "nope", ""):
        shell = build_library_shell_state(LibraryShellInput(), selected_row_id=row_id)
        assert shell.canvas_kind == "empty", row_id


def test_server_header_line():
    shell = build_library_shell_state(
        LibraryShellInput(runtime_source="server", server_label="lab-box")
    )
    assert shell.header_line == "Library | Server: lab-box"
