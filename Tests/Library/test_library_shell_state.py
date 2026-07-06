from tldw_chatbook.Constants import TAB_INGEST, TAB_MEDIA, TAB_NOTES
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
    create_ids = [r.row_id for r in shell.sections[1].rows]
    assert create_ids == ["create-note", "create-study", "create-flashcards", "create-quizzes"]
    assert [r.title for r in shell.sections[1].rows] == [
        "New note", "Study decks", "Flashcards", "Quizzes"
    ]
    assert shell.sections[1].rows[0].target_id == TAB_NOTES
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


def test_screen_and_unknown_rows_resolve_to_empty_canvas():
    for row_id in ("create-note", "nope", ""):
        shell = build_library_shell_state(LibraryShellInput(), selected_row_id=row_id)
        assert shell.canvas_kind == "empty", row_id


def test_server_header_line():
    shell = build_library_shell_state(
        LibraryShellInput(runtime_source="server", server_label="lab-box")
    )
    assert shell.header_line == "Library | Server: lab-box"
