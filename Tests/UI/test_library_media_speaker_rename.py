"""Task 8: rename meeting speakers after the meeting, on the Library media item."""

import pytest
from textual.widgets import Input, Static


def test_rename_after_rewrites_content_and_reindexes(tmp_media_db, meeting_folder_media_item):
    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker
    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")
    row = tmp_media_db.get_media_by_id(media_id)
    assert "Alice:" in row["content"]
    hits, _total = tmp_media_db.search_media_db(search_query="Alice")
    assert any(h["id"] == media_id for h in hits)


def test_rename_survives_a_soft_deleted_prior_transcript_row(tmp_media_db, meeting_folder_media_item):
    """M2: a soft-deleted prior Transcripts row must not collide the rename's
    INSERT -- UNIQUE(media_id, whisper_model) counts deleted rows, so the old
    `WHERE deleted = 0` lookup missed it and the INSERT hit a caught
    IntegrityError that silently dropped the rename."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")   # creates the Transcripts row

    # Soft-delete that row (as a prior delete would), keeping the sync trigger
    # happy: version increments by exactly 1, client_id stays non-empty.
    with tmp_media_db.transaction() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, version, client_id FROM Transcripts WHERE media_id=?", (media_id,))
        prior = cur.fetchone()
        cur.execute(
            "UPDATE Transcripts SET deleted=1, version=?, client_id=? WHERE id=?",
            (prior["version"] + 1, prior["client_id"], prior["id"]),
        )

    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Bob")     # must not silently fail
    row = tmp_media_db.get_media_by_id(media_id)
    assert "Bob:" in row["content"]
    hits, _total = tmp_media_db.search_media_db(search_query="Bob")
    assert any(h["id"] == media_id for h in hits)


def test_rename_after_disabled_when_folder_gone(tmp_media_db, meeting_folder_media_item):
    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hi")])
    import shutil; shutil.rmtree(folder)
    from tldw_chatbook.Widgets.Library.library_media_canvas import can_rename_meeting_speakers
    assert can_rename_meeting_speakers(tmp_media_db, media_id) is False


def test_rename_to_empty_name_removes_map_entry_and_drops_the_label(
    tmp_media_db, meeting_folder_media_item
):
    """Empty-name removal via the public path (review item 4)."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker

    media_id, folder = meeting_folder_media_item(names={"S1": "Alice"}, segments=[("S1", "hello")])
    rename_meeting_speaker(tmp_media_db, media_id, "S1", "")

    row = tmp_media_db.get_media_by_id(media_id)
    assert "Alice" not in row["content"]
    assert "Speaker 1:" in row["content"]

    import json
    meeting = json.loads((folder / "meeting.json").read_text())
    assert "S1" not in meeting["speaker_names"]


def test_render_uses_the_meetings_stored_display_name(tmp_media_db, meeting_folder_media_item):
    """task 31746: the Library item's own render reads `meeting.json`'s
    `user_display_name`, not a hardcoded "You" -- so it agrees with what the
    live Meetings screen showed for this recording."""
    media_id, _folder = meeting_folder_media_item(
        names={}, segments=[(None, "hello", "you")], user_display_name="Alice",
    )
    row = tmp_media_db.get_media_by_id(media_id)
    assert "Alice:" in row["content"]


def test_presentation_reachability_reflects_meeting_folder(
    tmp_media_db, meeting_folder_media_item
):
    """Review item 1: ``_library_media_canvas_presentation`` actually drives
    ``can_rename_speakers`` from the real DB/selection, not a stub default."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from Tests.UI.app_factory import _build_test_app

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hi")])
    other_id, _uuid, _msg = tmp_media_db.add_media_with_keywords(
        title="A plain document", media_type="article", content="not a meeting",
    )

    app = _build_test_app()
    app.media_db = tmp_media_db
    screen = LibraryScreen(app)

    screen._selected_media_id = f"local:media:{media_id}"
    presentation = screen._library_media_canvas_presentation()
    assert presentation["can_rename_speakers"] is True
    assert presentation["speaker_rename_media_id"] == media_id

    screen._selected_media_id = f"local:media:{other_id}"
    assert screen._library_media_canvas_presentation()["can_rename_speakers"] is False


@pytest.mark.asyncio
async def test_speaker_legend_submit_renames_and_refreshes_preview(
    tmp_media_db, meeting_folder_media_item
):
    """Review item 2: submitting a legend rename input calls
    ``rename_meeting_speaker`` and the canvas's own shown preview text
    picks up the new name (no page reload needed)."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
    from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState, LibraryMediaRow
    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    row = tmp_media_db.get_media_by_id(media_id)
    state = LibraryMediaCanvasState(
        rows=(
            LibraryMediaRow(
                media_id=str(media_id), title="Meeting", media_type="audio",
                secondary="audio · today", selected=True,
            ),
        ),
        type_options=(None, "audio"), active_type=None, status_copy="",
        empty_copy="", selected_id=str(media_id),
        preview_lines=tuple(row["content"].splitlines()), count=1,
    )

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaCanvas(
                canvas=state, can_rename_speakers=True, media_db=tmp_media_db,
                speaker_rename_media_id=media_id, id="canvas",
            )

    app = _App()
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        input_widget = app.query_one("#library-media-speaker-input-S1", Input)
        input_widget.post_message(Input.Submitted(input_widget, "Alice"))
        await pilot.pause()
        # Qodo Q3: the write now runs on a thread worker, so wait for it
        # rather than racing the assertions against it.
        await app.workers.wait_for_complete()
        await pilot.pause()

        preview = app.query_one("#library-media-preview-lines", Static)
        assert "Alice" in str(preview.renderable)
        label = app.query_one("#library-media-speaker-label-S1", Static)
        assert str(label.renderable) == "Alice"

    updated = tmp_media_db.get_media_by_id(media_id)
    assert "Alice:" in updated["content"]


@pytest.mark.asyncio
async def test_rename_failure_log_carries_no_path(
    tmp_media_db, meeting_folder_media_item, monkeypatch, captured_lines
):
    """TASK-31748: `rename_meeting_speaker` reads/writes `meeting.json`, and
    a filesystem failure's `str()` embeds the meeting folder path -- the
    canvas's rename-failure log must redact it."""
    import tldw_chatbook.Widgets.Library.library_media_canvas as canvas_module
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
    from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState, LibraryMediaRow
    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    def boom(*a, **k):
        raise OSError("/Users/alice/meeting.json: denied")

    monkeypatch.setattr(canvas_module, "update_meeting_json", boom)

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    row = tmp_media_db.get_media_by_id(media_id)
    state = LibraryMediaCanvasState(
        rows=(
            LibraryMediaRow(
                media_id=str(media_id), title="Meeting", media_type="audio",
                secondary="audio · today", selected=True,
            ),
        ),
        type_options=(None, "audio"), active_type=None, status_copy="",
        empty_copy="", selected_id=str(media_id),
        preview_lines=tuple(row["content"].splitlines()), count=1,
    )

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaCanvas(
                canvas=state, can_rename_speakers=True, media_db=tmp_media_db,
                speaker_rename_media_id=media_id, id="canvas",
            )

    app = _App()
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        input_widget = app.query_one("#library-media-speaker-input-S1", Input)
        input_widget.post_message(Input.Submitted(input_widget, "Alice"))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    joined = "\n".join(captured_lines)
    assert "/Users/alice" not in joined and "alice" not in joined


# ---- C2: never replace ingest-produced Library content ---------------------

def test_rename_refuses_when_the_library_content_is_not_the_meeting_render(
    tmp_media_db, meeting_folder_media_item
):
    """Fix C2: with the shipped `post_transcribe = true` default the Library
    item's content is the INGEST's own offline transcription of mixed.wav, and
    the rename used to overwrite it wholesale with the near-live render (no
    version row, no rollback). Prove it refuses and touches nothing."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import (
        RENAME_REFUSED_NOT_MEETING_CONTENT,
        rename_meeting_speaker,
    )

    media_id, folder = meeting_folder_media_item(
        names={}, segments=[("S1", "hello")],
        content="A much better offline transcription from the ingest pipeline.",
    )
    before = tmp_media_db.get_media_by_id(media_id)

    outcome = rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")

    assert outcome.ok is False
    assert outcome.reason == RENAME_REFUSED_NOT_MEETING_CONTENT
    after = tmp_media_db.get_media_by_id(media_id)
    assert after["content"] == before["content"]
    assert after["version"] == before["version"]
    import json
    assert json.loads((folder / "meeting.json").read_text())["speaker_names"] == {}


def test_rename_refuses_when_the_transcript_is_missing(tmp_media_db, meeting_folder_media_item):
    """Qodo Q16: a missing/empty transcript.jsonl rendered as "" and the
    rename wrote that empty string over `Media.content` AND its FTS row."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import (
        RENAME_REFUSED_EMPTY_TRANSCRIPT,
        rename_meeting_speaker,
    )

    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    (folder / "transcript.jsonl").unlink()
    before = tmp_media_db.get_media_by_id(media_id)

    outcome = rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")

    assert outcome.ok is False
    assert outcome.reason == RENAME_REFUSED_EMPTY_TRANSCRIPT
    assert tmp_media_db.get_media_by_id(media_id)["content"] == before["content"]


def test_successful_rename_records_a_reversible_document_version(
    tmp_media_db, meeting_folder_media_item
):
    """C2(c): the content swap goes through `create_document_version` so it
    can be rolled back -- including the FIRST rename, whose pre-state has to
    be seeded (rollback refuses the latest version number)."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    original = tmp_media_db.get_media_by_id(media_id)["content"]

    assert rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice").ok is True

    with tmp_media_db.transaction() as conn:
        rows = conn.execute(
            "SELECT version_number, content FROM DocumentVersions WHERE media_id=? ORDER BY version_number",
            (media_id,),
        ).fetchall()
    assert [r["content"] for r in rows][0] == original          # the replaced state
    assert "Alice:" in rows[-1]["content"]                      # ... and the new one

    rolled = tmp_media_db.rollback_to_version(media_id, rows[0]["version_number"])
    assert "error" not in rolled
    assert tmp_media_db.get_media_by_id(media_id)["content"] == original


# ---- I3: the non-meeting hot path -----------------------------------------

def test_presentation_never_fetches_content_and_memoizes_per_selection(
    tmp_media_db, meeting_folder_media_item
):
    """Fix I3: `_library_media_canvas_presentation` runs on EVERY media
    selection, filter and page change; `can_rename_meeting_speakers` used to
    do a `SELECT *` (dragging the whole `content` blob) plus a filesystem
    check on the UI thread for every item."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from Tests.UI.app_factory import _build_test_app

    _meeting_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hi")])
    other_id, _uuid, _msg = tmp_media_db.add_media_with_keywords(
        title="A plain document", media_type="article", content="not a meeting",
    )

    class _SpyDB:
        def __init__(self, real):
            self._real = real
            self.queries: list[str] = []
            self.full_row_reads = 0

        def execute_query(self, query, params=None, **kwargs):
            self.queries.append(query)
            return self._real.execute_query(query, params, **kwargs)

        def get_media_by_id(self, *args, **kwargs):
            self.full_row_reads += 1
            return self._real.get_media_by_id(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    spy = _SpyDB(tmp_media_db)
    app = _build_test_app()
    app.media_db = spy
    screen = LibraryScreen(app)
    screen._selected_media_id = f"local:media:{other_id}"

    assert screen._library_media_canvas_presentation()["can_rename_speakers"] is False
    assert spy.full_row_reads == 0                       # no whole-row/content fetch
    assert all("SELECT url" in q for q in spy.queries)   # ... just the url column
    queries_after_first = len(spy.queries)

    for _ in range(5):
        screen._library_media_canvas_presentation()
    assert len(spy.queries) == queries_after_first       # memoized for this selection


# ---- Q15 / MINOR: failure surfacing, hostile ids, bounded names ------------

@pytest.mark.asyncio
async def test_a_refused_rename_is_reported_to_the_user(tmp_media_db, meeting_folder_media_item):
    """Qodo Q15: a rename that changed nothing left only a debug log."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
    from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState, LibraryMediaRow
    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    media_id, _folder = meeting_folder_media_item(
        names={}, segments=[("S1", "hello")], content="not the meeting's own render",
    )
    row = tmp_media_db.get_media_by_id(media_id)
    state = LibraryMediaCanvasState(
        rows=(
            LibraryMediaRow(
                media_id=str(media_id), title="Meeting", media_type="audio",
                secondary="audio · today", selected=True,
            ),
        ),
        type_options=(None, "audio"), active_type=None, status_copy="",
        empty_copy="", selected_id=str(media_id),
        preview_lines=tuple(row["content"].splitlines()), count=1,
    )

    notices: list[str] = []

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaCanvas(
                canvas=state, can_rename_speakers=True, media_db=tmp_media_db,
                speaker_rename_media_id=media_id, id="canvas",
            )

        def notify(self, message, **kwargs):
            notices.append(str(message))

    app = _App()
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        input_widget = app.query_one("#library-media-speaker-input-S1", Input)
        input_widget.post_message(Input.Submitted(input_widget, "Alice"))
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    assert notices and "Couldn't rename this speaker" in notices[0]
    assert tmp_media_db.get_media_by_id(media_id)["content"] == row["content"]


def test_a_hostile_speaker_id_is_skipped_rather_than_taking_compose_down(
    tmp_media_db, meeting_folder_media_item
):
    """Final review MINOR: `speaker_id` is interpolated into a widget id, and
    transcript.jsonl is a plain file a user can edit."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import _meeting_speaker_legend_rows

    media_id, _folder = meeting_folder_media_item(
        names={}, segments=[("S1", "hi"), ("bad id #1", "yo"), ("", "hm")],
    )
    assert [cid for cid, _label in _meeting_speaker_legend_rows(tmp_media_db, media_id)] == ["S1"]


def test_a_submitted_name_is_bounded(tmp_media_db, meeting_folder_media_item):
    """Qodo Q2: names reach meeting.json, the transcript, `Media.content` and
    FTS -- bound them at the boundary both rename paths share."""
    from tldw_chatbook.Audio.meeting_session import MAX_SPEAKER_NAME_CHARS
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker

    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    assert rename_meeting_speaker(tmp_media_db, media_id, "S1", "  " + "A" * 500 + "  ").ok

    import json
    stored = json.loads((folder / "meeting.json").read_text())["speaker_names"]["S1"]
    assert stored == "A" * MAX_SPEAKER_NAME_CHARS


@pytest.mark.asyncio
async def test_a_name_with_rich_markup_renders_literally(tmp_media_db, meeting_folder_media_item):
    """Fix I2 (Qodo Q2): the live legend's `Static` was markup-enabled, so a
    name containing "[/]" raised out of Rich and took the app down."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
    from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState, LibraryMediaRow
    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    media_id, _folder = meeting_folder_media_item(
        names={"S1": "Alice [/]"}, segments=[("S1", "hello")],
    )
    row = tmp_media_db.get_media_by_id(media_id)
    state = LibraryMediaCanvasState(
        rows=(
            LibraryMediaRow(
                media_id=str(media_id), title="Meeting", media_type="audio",
                secondary="audio · today", selected=True,
            ),
        ),
        type_options=(None, "audio"), active_type=None, status_copy="",
        empty_copy="", selected_id=str(media_id),
        preview_lines=tuple(row["content"].splitlines()), count=1,
    )

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaCanvas(
                canvas=state, can_rename_speakers=True, media_db=tmp_media_db,
                speaker_rename_media_id=media_id, id="canvas",
            )

    app = _App()
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        label = app.query_one("#library-media-speaker-label-S1", Static)
        assert str(label.renderable) == "Alice [/]"


@pytest.mark.asyncio
async def test_legend_rows_are_not_built_while_the_preview_pane_is_hidden(
    tmp_media_db, meeting_folder_media_item
):
    """Final review MINOR: building the rows parses the WHOLE transcript.jsonl
    during compose(), and the Library shell composes this canvas with its
    preview pane hidden (it uses its own reader pane instead)."""
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
    from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState, LibraryMediaRow
    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    media_id, _folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    row = tmp_media_db.get_media_by_id(media_id)

    class _SpyDB:
        def __init__(self, real):
            self._real = real
            self.reads = 0

        def get_media_by_id(self, *args, **kwargs):
            self.reads += 1
            return self._real.get_media_by_id(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    spy = _SpyDB(tmp_media_db)
    state = LibraryMediaCanvasState(
        rows=(
            LibraryMediaRow(
                media_id=str(media_id), title="Meeting", media_type="audio",
                secondary="audio · today", selected=True,
            ),
        ),
        type_options=(None, "audio"), active_type=None, status_copy="",
        empty_copy="", selected_id=str(media_id),
        preview_lines=tuple(row["content"].splitlines()), count=1,
    )

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaCanvas(
                canvas=state, can_rename_speakers=True, media_db=spy,
                speaker_rename_media_id=media_id, show_preview=False, id="canvas",
            )

    app = _App()
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        assert spy.reads == 0                                    # transcript never parsed
        assert not app.query("#library-media-speaker-legend")     # ... and no rows mounted
