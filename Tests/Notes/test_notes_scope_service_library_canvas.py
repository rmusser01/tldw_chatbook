"""Real-DB coverage for ``NotesScopeService.save_note`` as consumed by the
Library screen's in-canvas notes editor (save + autosave + conflict
policy).

Constructs the real ``NotesScopeService`` over the real
``NotesInteropService`` backed by a real (temp-file) ``CharactersRAGDB``,
mirroring how ``app.py`` wires ``self.notes_scope_service`` in
``app.py:1579``/``app.py:2851`` (``NotesInteropService(base_db_directory=...,
api_client_id=..., global_db_to_use=<CharactersRAGDB>)``). The Library
screen's save/conflict handling makes assumptions about this seam's real
return shapes (dict vs. bool, raised ``ConflictError`` vs. falsy return)
that a hand-rolled fake could get wrong silently; these tests pin the real
behavior down.
"""

from datetime import datetime

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

USER_ID = "library-canvas-user"


@pytest.fixture
def notes_scope_service(tmp_path):
    """Real ``NotesScopeService`` over a real (temp-file) ChaChaNotes DB.

    Mirrors ``app.py``'s construction: a ``NotesInteropService`` given a
    unified DB "template" instance (``global_db_to_use``) whose file path
    every per-user ``CharactersRAGDB`` instance it creates points at,
    wrapped by a ``NotesScopeService`` with no policy enforcer/server
    service (these tests exercise the local-note save seam directly, not
    the runtime policy layer or server routing).
    """
    db_dir = tmp_path / "chachanotes"
    db_dir.mkdir()
    global_db = CharactersRAGDB(str(db_dir / "unified.db"), client_id="library-canvas-app")
    interop = NotesInteropService(
        base_db_directory=db_dir,
        api_client_id="library-canvas-client",
        global_db_to_use=global_db,
    )
    return NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        policy_enforcer=None,
    )


@pytest.mark.asyncio
async def test_create_then_update_round_trip_bumps_version(notes_scope_service):
    """Create (no keywords) returns the raw note id; a correct-version
    update (no keywords) returns the real backend's plain ``True`` and the
    stored version advances by one -- the shape the Library screen's
    "truthy but not a dict -> bump by 1" branch depends on.
    """
    created = await notes_scope_service.save_note(
        scope="local_note", title="Original", content="v1 body", user_id=USER_ID,
    )
    assert isinstance(created, str) and created

    updated = await notes_scope_service.save_note(
        scope="local_note",
        title="Updated",
        content="v2 body",
        note_id=created,
        version=1,
        user_id=USER_ID,
    )
    assert updated is True

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=created, user_id=USER_ID
    )
    assert detail["title"] == "Updated"
    assert detail["content"] == "v2 body"
    assert detail["version"] == 2


@pytest.mark.asyncio
async def test_update_with_stale_version_raises_conflict_and_does_not_mutate(
    notes_scope_service,
):
    """A stale ``version`` on update does not return a falsy value -- the
    real local backend (``CharactersRAGDB.update_note``) raises
    ``ConflictError`` on the optimistic-lock mismatch, and the stored
    content/version are left exactly as they were before the attempt. The
    Library screen's ``_save_library_note`` must catch this exception and
    treat it the same as a falsy result to reach its conflict UI.
    """
    created = await notes_scope_service.save_note(
        scope="local_note", title="Original", content="body", user_id=USER_ID,
    )
    await notes_scope_service.save_note(
        scope="local_note",
        title="Second",
        content="body2",
        note_id=created,
        version=1,
        user_id=USER_ID,
    )

    with pytest.raises(ConflictError):
        await notes_scope_service.save_note(
            scope="local_note",
            title="Stale writer",
            content="clobber attempt",
            note_id=created,
            version=1,  # stale: the note is already at version 2
            user_id=USER_ID,
        )

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=created, user_id=USER_ID
    )
    assert detail["title"] == "Second"
    assert detail["content"] == "body2"
    assert detail["version"] == 2


@pytest.mark.asyncio
async def test_save_with_keywords_returns_dict_with_bumped_version_and_persists(
    notes_scope_service,
):
    """Passing ``keywords`` changes the success return shape to a dict
    carrying the bumped ``version`` (the shape the Library screen reads
    ``result["version"]`` from). ``get_note_detail``'s local-scope shape is
    the raw ``notes`` table row and does NOT include a ``keywords`` key --
    keyword persistence can only be observed via the local notes service's
    own ``get_keywords_for_note``, which is also what
    ``NotesScopeService`` uses internally to sync keyword links on save.
    """
    created = await notes_scope_service.save_note(
        scope="local_note",
        title="Original",
        content="body",
        user_id=USER_ID,
        keywords=["a", "b"],
    )
    assert isinstance(created, dict)
    assert created["version"] == 1
    note_id = created["id"]

    updated = await notes_scope_service.save_note(
        scope="local_note",
        title="Updated",
        content="body2",
        note_id=note_id,
        version=created["version"],
        user_id=USER_ID,
        keywords=["a", "c"],
    )
    assert isinstance(updated, dict)
    assert updated["version"] == 2

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=note_id, user_id=USER_ID
    )
    assert "keywords" not in detail
    assert detail["title"] == "Updated"
    assert detail["version"] == 2

    keyword_rows = notes_scope_service.local_notes_service.get_keywords_for_note(
        USER_ID, note_id
    )
    persisted_keywords = sorted(row["keyword"] for row in keyword_rows)
    assert persisted_keywords == ["a", "c"]


@pytest.mark.asyncio
async def test_create_from_note_template_round_trips_title_and_content(notes_scope_service):
    """The Library screen's in-canvas Create view (task 6) resolves a
    ``NOTE_TEMPLATES`` entry's ``{date}`` placeholders before ever calling
    this seam (see ``LibraryScreen._library_note_template_fields``), so the
    title/content it hands to ``save_note`` already have the current date
    substituted in -- no literal ``{date}`` survives. This pins the seam's
    call shape (no keywords, no version: returned id is a plain string, not
    a dict -- the create path only becomes a dict when ``keywords`` is
    passed) and confirms ``get_note_detail`` round-trips the resolved
    title/content unchanged.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    resolved_title = f"Meeting Notes - {today}"
    resolved_content = f"## Meeting Notes\n\n**Date:** {today}\n**Attendees:**\n"

    created_id = await notes_scope_service.save_note(
        scope="local_note",
        title=resolved_title,
        content=resolved_content,
        note_id=None,
        user_id=USER_ID,
    )
    assert isinstance(created_id, str) and created_id

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=created_id, user_id=USER_ID
    )
    assert "{date}" not in detail["title"]
    assert "{date}" not in detail["content"]
    assert detail["title"] == resolved_title
    assert detail["content"] == resolved_content
    assert detail["version"] == 1


@pytest.mark.asyncio
async def test_delete_with_correct_version_removes_the_note(notes_scope_service):
    """A correct-version delete returns the real local backend's plain
    ``True`` (``CharactersRAGDB.soft_delete_note``'s success return), and
    the note is gone from both ``get_note_detail`` (soft-deleted rows are
    filtered out, so this is None/missing exactly like a never-existing
    id) and ``list_notes``. The Library screen's delete confirm handler
    depends on this truthy/removed shape.
    """
    created = await notes_scope_service.save_note(
        scope="local_note", title="To delete", content="body", user_id=USER_ID,
    )

    deleted = await notes_scope_service.delete_note(
        scope="local_note", note_id=created, version=1, user_id=USER_ID,
    )
    assert deleted is True

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=created, user_id=USER_ID
    )
    assert detail is None

    remaining = notes_scope_service.local_notes_service.list_notes(USER_ID, limit=100)
    assert all(note["id"] != created for note in remaining)


@pytest.mark.asyncio
async def test_delete_with_stale_version_raises_conflict_and_does_not_remove(
    notes_scope_service,
):
    """A stale ``version`` on delete does not return a falsy value -- the
    real local backend (``CharactersRAGDB.soft_delete_note``) raises
    ``ConflictError`` on the optimistic-lock mismatch, exactly like
    ``update_note``, and the note is left in place. The Library screen's
    delete confirm handler must catch this exception and treat it the same
    as a falsy result (quiet warning, stay in the editor) rather than
    letting it propagate.
    """
    created = await notes_scope_service.save_note(
        scope="local_note", title="Original", content="body", user_id=USER_ID,
    )
    await notes_scope_service.save_note(
        scope="local_note",
        title="Updated",
        content="body2",
        note_id=created,
        version=1,
        user_id=USER_ID,
    )  # note is now at version 2

    with pytest.raises(ConflictError):
        await notes_scope_service.delete_note(
            scope="local_note",
            note_id=created,
            version=1,  # stale: the note is already at version 2
            user_id=USER_ID,
        )

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=created, user_id=USER_ID
    )
    assert detail is not None
    assert detail["title"] == "Updated"
    assert detail["version"] == 2


@pytest.mark.asyncio
async def test_create_with_keywords_returns_dict_and_persists_keywords(
    notes_scope_service,
):
    """The CREATE path with ``keywords`` returns the real backend's dict
    (``{"id": ..., "version": 1, ...}``) and persists the keywords -- the
    shape the Library create-from-template flow depends on (templates carry
    keywords, and the standalone screen applies them on create).
    """
    result = await notes_scope_service.save_note(
        scope="local_note",
        title="Meeting Notes - 2026-07-08",
        content="body",
        user_id=USER_ID,
        keywords=["meeting", "notes"],
    )
    assert isinstance(result, dict)
    created_id = str(result.get("id") or "")
    assert created_id
    assert result.get("version") == 1

    stored = notes_scope_service.local_notes_service.get_keywords_for_note(
        USER_ID, created_id
    )
    stored_texts = {
        str(k.get("keyword") if isinstance(k, dict) else k) for k in stored
    }
    assert {"meeting", "notes"} <= stored_texts


@pytest.mark.asyncio
async def test_get_note_detail_enriched_with_keywords_round_trips_through_editor_state(
    notes_scope_service,
):
    """``get_note_detail``'s local-scope shape omits ``keywords`` (pinned by
    ``test_save_with_keywords_returns_dict_with_bumped_version_and_persists``
    above); the Library screen's ``_refresh_library_note_detail`` enriches
    the fetched detail with ``notes_service.get_keywords_for_note`` (the
    same ``local_notes_service`` seam used here) before building editor
    state. This simulates that enrichment against the real seam and
    confirms the enriched detail round-trips through
    ``build_library_note_editor_state`` the way the screen renders it.
    """
    from tldw_chatbook.Library.library_notes_state import build_library_note_editor_state

    created = await notes_scope_service.save_note(
        scope="local_note",
        title="Keyworded",
        content="body",
        user_id=USER_ID,
        keywords=["alpha", "beta"],
    )
    note_id = created["id"] if isinstance(created, dict) else created

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=note_id, user_id=USER_ID
    )
    assert "keywords" not in detail

    keyword_rows = notes_scope_service.local_notes_service.get_keywords_for_note(
        USER_ID, note_id
    )
    enriched = dict(detail)
    enriched["keywords"] = keyword_rows

    editor_state = build_library_note_editor_state(enriched)
    assert editor_state.keywords_text == "alpha, beta"
