import pytest

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_CANVAS_KIND_NOTES_CREATE,
    LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP,
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_BROWSE_SKILLS,
    LIBRARY_ROW_CREATE_PROMPT,
    LIBRARY_ROW_CREATE_SKILL,
    LIBRARY_ROW_INGEST_EXPORT,
    LibraryShellInput,
    build_library_shell_state,
)


def test_shell_sections_rows_and_targets_are_fixed():
    shell = build_library_shell_state(
        LibraryShellInput(
            media_count=17,
            conversations_count=128,
            notes_count=42,
            collections_count=5,
            prompts_count=9,
            skills_count=3,
        )
    )
    assert shell.header_line == "Library | Local"
    assert [s.section_id for s in shell.sections] == [
        "browse",
        "create",
        "study",
        "ingest",
    ]
    assert [s.title for s in shell.sections] == [
        "Browse",
        "Create",
        "Study",
        "Import / Export",
    ]
    browse = shell.sections[0]
    assert [r.row_id for r in browse.rows] == [
        "browse-media",
        "browse-conversations",
        "browse-notes",
        LIBRARY_ROW_BROWSE_PROMPTS,
        LIBRARY_ROW_BROWSE_SKILLS,
        "browse-collections",
        "browse-search",
    ]
    assert [r.title for r in browse.rows] == [
        "Media",
        "Conversations",
        "Notes",
        "Prompts",
        "Skills",
        "Collections",
        "Search / RAG",
    ]
    assert (
        browse.rows[6].target_kind == "canvas" and browse.rows[6].target_id == "search"
    )
    assert browse.rows[6].count is None
    assert (
        browse.rows[5].target_kind == "canvas"
        and browse.rows[5].target_id == "collections"
    )
    conv = browse.rows[1]
    assert (conv.target_kind, conv.target_id, conv.count) == (
        "canvas",
        "conversations",
        128,
    )
    media = browse.rows[0]
    assert (media.target_kind, media.target_id) == ("canvas", "media")
    notes = browse.rows[2]
    assert (notes.target_kind, notes.target_id, notes.count) == ("canvas", "notes", 42)
    prompts = browse.rows[3]
    assert (prompts.target_kind, prompts.target_id, prompts.count) == (
        "canvas",
        "prompts",
        9,
    )
    skills = browse.rows[4]
    assert (skills.target_kind, skills.target_id, skills.count) == (
        "canvas",
        "skills",
        3,
    )
    create_ids = [r.row_id for r in shell.sections[1].rows]
    assert create_ids == [
        "create-note",
        LIBRARY_ROW_CREATE_PROMPT,
        LIBRARY_ROW_CREATE_SKILL,
    ]
    assert [r.title for r in shell.sections[1].rows] == [
        "New note",
        "New prompt",
        "New skill",
    ]
    assert (
        shell.sections[1].rows[0].target_kind,
        shell.sections[1].rows[0].target_id,
    ) == (
        "canvas",
        LIBRARY_CANVAS_KIND_NOTES_CREATE,
    )
    # F-017: the Study rows are handoffs ("Continue in Study"), grouped in
    # their own section -- not under Create, which is for making things.
    study = shell.sections[2]
    assert [r.row_id for r in study.rows] == [
        "create-study",
        "create-flashcards",
        "create-quizzes",
    ]
    assert [r.title for r in study.rows] == ["Study decks", "Flashcards", "Quizzes"]
    assert all(r.target_kind == "handoff" for r in study.rows)
    assert [r.target_id for r in study.rows] == ["study", "flashcards", "quizzes"]
    ingest = shell.sections[3]
    assert [r.title for r in ingest.rows] == ["Import…", "Export"]
    assert (ingest.rows[0].target_kind, ingest.rows[0].target_id) == (
        "canvas",
        "ingest-media",
    )
    assert (ingest.rows[1].target_kind, ingest.rows[1].target_id) == (
        "canvas",
        "export",
    )
    assert ingest.rows[1].disabled is False
    assert all(r.count is None for r in shell.sections[1].rows)


def test_empty_selection_yields_landing_canvas():
    shell = build_library_shell_state(LibraryShellInput())
    assert shell.canvas_kind == "empty"
    assert (
        shell.canvas_empty_copy
        == "Search everything, pick a section, or add something new."
    )


def test_jargon_rows_carry_plain_language_subtitles():
    """F-013: the load-bearing jargon rows get a one-line plain-language
    gloss; already-plain rows stay unglossed so the rail doesn't stutter.
    task-2236 (R2): every gloss fits the rail's realistic width budget
    (<=25 content cells at 170x50, title and count included)."""
    shell = build_library_shell_state(LibraryShellInput())
    subtitles = {
        row.row_id: row.subtitle
        for section in shell.sections
        for row in section.rows
    }
    assert subtitles["browse-media"] == "your files"
    assert subtitles[LIBRARY_ROW_BROWSE_PROMPTS] == "reuse"
    assert subtitles[LIBRARY_ROW_BROWSE_SKILLS] == "AI add-ons"
    assert subtitles["browse-collections"] == "saved captures"
    assert subtitles["browse-search"] == "find all"
    # Plain-language rows carry no gloss.
    assert subtitles["browse-conversations"] == ""
    assert subtitles["browse-notes"] == ""
    assert subtitles["create-note"] == ""
    assert subtitles["ingest-import-media"] == ""
    assert subtitles["ingest-export"] == ""


def test_collections_row_is_count_pending_not_count_loading():
    """LIB-15: Collections is neither "loading" (F-014's dim placeholder,
    which would misrepresent an idle row as actively fetching) nor "off"
    (Search/RAG's permanent no-count case) -- it is ``count_pending``, a
    distinct state the gloss-fit rule uses to reserve stable width for the
    count that has not arrived yet. This must hold regardless of
    ``counts_loading`` (Collections' own fetch is a separate, lazy,
    first-visit-triggered lifecycle)."""
    for counts_loading in (True, False):
        shell = build_library_shell_state(
            LibraryShellInput(counts_loading=counts_loading)
        )
        rows = {
            row.row_id: row for section in shell.sections for row in section.rows
        }
        collections = rows["browse-collections"]
        assert collections.count_pending is True
        assert collections.count_loading is False
        # Every other row never flags count_pending -- Collections is the
        # sole exception this state exists for.
        for row_id, row in rows.items():
            if row_id == "browse-collections":
                continue
            assert row.count_pending is False, row_id


def test_short_title_fallbacks_are_set_for_the_rows_that_need_them():
    """LIB-18: rows whose title risks a mid-word ellipsis cut at the
        rail's real minimum width (Conversations/Flashcards/Collections) carry
    a ``short_title`` fallback; every other row's title is short enough
    that it does not need one."""
    shell = build_library_shell_state(LibraryShellInput())
    rows = {
        row.row_id: row for section in shell.sections for row in section.rows
    }
    assert rows["browse-conversations"].short_title == "Chats"
    assert rows["browse-collections"].short_title == "Captures"
    assert rows["create-flashcards"].short_title == "Cards"
    no_short_title_ids = {
        "browse-media",
        "browse-notes",
        LIBRARY_ROW_BROWSE_PROMPTS,
        LIBRARY_ROW_BROWSE_SKILLS,
        "browse-search",
        "create-note",
        LIBRARY_ROW_CREATE_PROMPT,
        LIBRARY_ROW_CREATE_SKILL,
        "create-study",
        "create-quizzes",
        "ingest-import-media",
        LIBRARY_ROW_INGEST_EXPORT,
    }
    for row_id in no_short_title_ids:
        assert rows[row_id].short_title == "", row_id


def test_counts_loading_marks_snapshot_backed_rows_but_not_collections():
    """F-014: while the source snapshot is in flight, every row whose count
    rides that snapshot shows the loading placeholder. Collections is the
    one exception: its count is fetched lazily (first canvas visit), so a
    placeholder would sit there indefinitely on the landing screen."""
    shell = build_library_shell_state(LibraryShellInput(counts_loading=True))
    rows = {
        row.row_id: row for section in shell.sections for row in section.rows
    }
    loading_ids = {
        "browse-media",
        "browse-conversations",
        "browse-notes",
        LIBRARY_ROW_BROWSE_PROMPTS,
        LIBRARY_ROW_BROWSE_SKILLS,
        "create-study",
        "create-flashcards",
        "create-quizzes",
    }
    for row_id, row in rows.items():
        assert row.count_loading is (row_id in loading_ids), row_id
    assert rows["browse-collections"].count_loading is False
    # Search, Create-verb and Import/Export rows carry no count at all.
    assert rows["browse-search"].count_loading is False
    assert rows["create-note"].count_loading is False
    assert rows["ingest-import-media"].count_loading is False


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


def test_handoff_selection_yields_handoff_canvas():
    # create-study/-flashcards/-quizzes are "handoff" rows (L3b Task 8), not
    # legacy "mode" rows: their canvas_kind is the dedicated "handoff" kind,
    # with canvas_target carrying which of the three handoffs is active.
    shell = build_library_shell_state(
        LibraryShellInput(), selected_row_id="create-flashcards"
    )
    assert (shell.canvas_kind, shell.canvas_target) == ("handoff", "flashcards")


@pytest.mark.parametrize(
    ("row_id", "expected_target"),
    [
        ("create-study", "study"),
        ("create-flashcards", "flashcards"),
        ("create-quizzes", "quizzes"),
    ],
)
def test_handoff_rows_target_handoff_kind_and_carry_their_target_id(
    row_id, expected_target
):
    shell = build_library_shell_state(LibraryShellInput(), selected_row_id=row_id)
    row = next(r for r in shell.sections[2].rows if r.row_id == row_id)
    assert row.target_kind == "handoff"
    assert row.target_id == expected_target
    assert (shell.canvas_kind, shell.canvas_target) == ("handoff", expected_target)


def test_browse_search_selection_yields_search_canvas():
    # browse-search is a first-class canvas row (not a legacy "mode" row):
    # its canvas_kind is the target_id itself, matching browse-media/-notes.
    shell = build_library_shell_state(
        LibraryShellInput(), selected_row_id="browse-search"
    )
    assert (shell.canvas_kind, shell.canvas_target) == ("search", "")
    assert shell.selected_row_id == "browse-search"


def test_browse_collections_selection_yields_collections_canvas():
    shell = build_library_shell_state(
        LibraryShellInput(collections_count=5), selected_row_id="browse-collections"
    )
    assert (shell.canvas_kind, shell.canvas_target) == ("collections", "")
    assert shell.selected_row_id == "browse-collections"


def test_browse_notes_row_targets_notes_canvas():
    shell = build_library_shell_state(
        LibraryShellInput(notes_count=42), selected_row_id="browse-notes"
    )
    notes_row = next(
        r
        for section in shell.sections
        for r in section.rows
        if r.row_id == "browse-notes"
    )
    assert notes_row.target_kind == "canvas"
    assert notes_row.target_id == "notes"
    assert (shell.canvas_kind, shell.canvas_target) == ("notes", "")
    assert shell.selected_row_id == "browse-notes"


def test_browse_prompts_row_targets_prompts_canvas():
    # Task 1 (count seam + rail row): the row is inert-but-selectable --
    # canvas_kind resolves to "prompts" via the generic "canvas" target_kind
    # branch, but the screen has no dedicated "prompts" render branch yet,
    # so it falls through to the placeholder-empty canvas (a later task
    # adds the real list canvas).
    shell = build_library_shell_state(
        LibraryShellInput(prompts_count=9), selected_row_id=LIBRARY_ROW_BROWSE_PROMPTS
    )
    prompts_row = next(
        r
        for section in shell.sections
        for r in section.rows
        if r.row_id == LIBRARY_ROW_BROWSE_PROMPTS
    )
    assert prompts_row.target_kind == "canvas"
    assert prompts_row.target_id == "prompts"
    assert (shell.canvas_kind, shell.canvas_target) == ("prompts", "")
    assert shell.selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS


def test_create_note_row_targets_notes_create_canvas():
    assert LIBRARY_CANVAS_KIND_NOTES_CREATE == "notes-create"
    shell = build_library_shell_state(
        LibraryShellInput(), selected_row_id="create-note"
    )
    row = next(
        r
        for section in shell.sections
        for r in section.rows
        if r.row_id == "create-note"
    )
    assert row.target_kind == "canvas"
    assert row.target_id == LIBRARY_CANVAS_KIND_NOTES_CREATE
    assert (shell.canvas_kind, shell.canvas_target) == (
        LIBRARY_CANVAS_KIND_NOTES_CREATE,
        "",
    )
    assert shell.selected_row_id == "create-note"


def test_create_prompt_row_targets_prompts_canvas():
    # Task 8b D1: the "New prompt" row reuses the SAME canvas kind
    # ("prompts") Browse > Prompts targets -- the screen distinguishes the
    # two by view state (`_prompts_state.view`), not by a separate canvas
    # kind, mirroring the brief's "a `prompts` editor view ... prompt_id=None
    # sentinel" design.
    shell = build_library_shell_state(
        LibraryShellInput(), selected_row_id=LIBRARY_ROW_CREATE_PROMPT
    )
    row = next(
        r
        for section in shell.sections
        for r in section.rows
        if r.row_id == LIBRARY_ROW_CREATE_PROMPT
    )
    assert row.target_kind == "canvas"
    assert row.target_id == "prompts"
    assert (shell.canvas_kind, shell.canvas_target) == ("prompts", "")
    assert shell.selected_row_id == LIBRARY_ROW_CREATE_PROMPT


def test_create_skill_row_targets_skills_canvas():
    # Skills sub-project (skills-200 spec, "Create > New skill"): mirrors
    # ``test_create_prompt_row_targets_prompts_canvas`` exactly -- the row
    # reuses the SAME canvas kind ("skills") Browse > Skills targets, the
    # screen distinguishes the two by ``_selected_skill_name`` being empty
    # (view state), not by a separate canvas kind.
    shell = build_library_shell_state(
        LibraryShellInput(), selected_row_id=LIBRARY_ROW_CREATE_SKILL
    )
    row = next(
        r
        for section in shell.sections
        for r in section.rows
        if r.row_id == LIBRARY_ROW_CREATE_SKILL
    )
    assert row.target_kind == "canvas"
    assert row.target_id == "skills"
    assert row.title == "New skill"
    assert (shell.canvas_kind, shell.canvas_target) == ("skills", "")
    assert shell.selected_row_id == LIBRARY_ROW_CREATE_SKILL


def test_unknown_rows_resolve_to_empty_canvas():
    # browse-notes and create-note used to be "screen" rows resolving to an
    # empty canvas; they now target real canvases (see the two tests above).
    # ingest-import-media is likewise now a "canvas" row (see the test
    # below) -- no row remains "screen"-kind, so only unknown/empty
    # selections still land on the empty landing canvas.
    for row_id in ("nope", ""):
        shell = build_library_shell_state(LibraryShellInput(), selected_row_id=row_id)
        assert shell.canvas_kind == "empty", row_id


def test_ingest_import_media_selection_yields_ingest_media_canvas():
    shell = build_library_shell_state(
        LibraryShellInput(), selected_row_id="ingest-import-media"
    )
    assert (shell.canvas_kind, shell.canvas_target) == ("ingest-media", "")
    assert shell.selected_row_id == "ingest-import-media"


def test_ingest_export_selection_yields_export_canvas():
    shell = build_library_shell_state(
        LibraryShellInput(), selected_row_id=LIBRARY_ROW_INGEST_EXPORT
    )
    assert (shell.canvas_kind, shell.canvas_target) == ("export", "")
    assert shell.selected_row_id == LIBRARY_ROW_INGEST_EXPORT


def test_export_row_enabled_when_runtime_source_is_local():
    shell = build_library_shell_state(LibraryShellInput(runtime_source="local"))
    export_row = next(
        r
        for section in shell.sections
        for r in section.rows
        if r.row_id == LIBRARY_ROW_INGEST_EXPORT
    )
    assert export_row.disabled is False
    assert export_row.disabled_tooltip == ""


def test_export_row_disabled_with_tooltip_when_runtime_source_is_server():
    shell = build_library_shell_state(
        LibraryShellInput(runtime_source="server", server_label="lab-box")
    )
    export_row = next(
        r
        for section in shell.sections
        for r in section.rows
        if r.row_id == LIBRARY_ROW_INGEST_EXPORT
    )
    assert export_row.disabled is True
    assert export_row.disabled_tooltip == LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP
    # Every other row stays enabled -- the gate is Export-specific, not a
    # blanket server-mode lockout of the whole rail.
    other_rows = [
        r
        for section in shell.sections
        for r in section.rows
        if r.row_id != LIBRARY_ROW_INGEST_EXPORT
    ]
    assert all(r.disabled is False for r in other_rows)


def test_server_header_line():
    shell = build_library_shell_state(
        LibraryShellInput(runtime_source="server", server_label="lab-box")
    )
    assert shell.header_line == "Library | Server: lab-box"


def _study_row(shell, row_id):
    # F-017: the study handoff rows live in the "study" section (index 2),
    # not under Create.
    return next(r for r in shell.sections[2].rows if r.row_id == row_id)


def test_flashcards_due_count_renders_bright_when_positive():
    shell = build_library_shell_state(LibraryShellInput(flashcards_due_count=12))
    row = _study_row(shell, "create-flashcards")
    assert row.count_display == " due: 12"
    assert row.count_emphasis == "bright"
    # count/count_known are untouched by the flashcards due copy -- the row
    # renders exclusively via count_display.
    assert row.count is None


def test_flashcards_due_count_renders_dim_when_zero():
    shell = build_library_shell_state(LibraryShellInput(flashcards_due_count=0))
    row = _study_row(shell, "create-flashcards")
    assert row.count_display == " due: 0"
    assert row.count_emphasis == "dim"


def test_flashcards_due_count_none_yields_no_display_or_emphasis():
    shell = build_library_shell_state(LibraryShellInput(flashcards_due_count=None))
    row = _study_row(shell, "create-flashcards")
    assert row.count_display == ""
    assert row.count_emphasis == ""


def test_study_decks_and_quizzes_counts_land_in_row_count():
    shell = build_library_shell_state(
        LibraryShellInput(study_decks_count=3, quizzes_count=2)
    )
    decks_row = _study_row(shell, "create-study")
    quizzes_row = _study_row(shell, "create-quizzes")
    assert (decks_row.count, decks_row.count_known) == (3, True)
    assert (quizzes_row.count, quizzes_row.count_known) == (2, True)
    # Neither the study-decks nor the quizzes row uses the count_display
    # override -- that's exclusive to the flashcards-due copy contract.
    assert decks_row.count_display == ""
    assert quizzes_row.count_display == ""


def test_study_decks_and_quizzes_counts_default_to_none():
    shell = build_library_shell_state(LibraryShellInput())
    decks_row = _study_row(shell, "create-study")
    quizzes_row = _study_row(shell, "create-quizzes")
    assert decks_row.count is None
    assert quizzes_row.count is None
