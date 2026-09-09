"""Real-DB coverage for the note-backlink query (task-32145).

The Obsidian importer rewrites a resolvable ``[[wikilink]]`` as
``[label](note://<note_id>)`` (``note_import_plan_models.rewrite_wikilinks``),
so "which notes link to this one" is a containment search for that exact
token. These tests run the query against a real ``CharactersRAGDB`` through
the real ``NotesInteropService``/``NotesScopeService`` wiring, because the
properties that matter -- parameterisation, LIKE-wildcard escaping, the row
bound, and soft-deleted exclusion -- are all properties of the SQL, which a
fake cannot have.
"""

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

USER_ID = "backlink-user"


@pytest.fixture
def notes_scope_service(tmp_path):
    """Real ``NotesScopeService`` over a real (temp-file) ChaChaNotes DB."""
    db_dir = tmp_path / "chachanotes"
    db_dir.mkdir()
    global_db = CharactersRAGDB(str(db_dir / "unified.db"), client_id="backlink-app")
    interop = NotesInteropService(
        base_db_directory=db_dir,
        api_client_id="backlink-client",
        global_db_to_use=global_db,
    )
    return NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        policy_enforcer=None,
    )


async def _add(service, title, content):
    return await service.save_note(
        scope="local_note", title=title, content=content, user_id=USER_ID
    )


@pytest.mark.asyncio
async def test_backlinks_list_every_note_whose_body_links_to_the_target(
    notes_scope_service,
):
    target = await _add(notes_scope_service, "Zettelkasten — overview", "hub")
    await _add(
        notes_scope_service,
        "Library review",
        f"See [Zettelkasten](note://{target}) for the method.",
    )
    await _add(
        notes_scope_service,
        "Daily 2026-09-07",
        f"Follow-up: [overview](note://{target})",
    )
    await _add(notes_scope_service, "Unrelated", "no links here")

    backlinks = await notes_scope_service.list_note_backlinks(
        scope="local_note", note_id=target, user_id=USER_ID
    )

    assert [row["title"] for row in backlinks] == [
        "Daily 2026-09-07",
        "Library review",
    ]
    assert all(row["id"] != target for row in backlinks)


@pytest.mark.asyncio
async def test_backlinks_exclude_deleted_notes(notes_scope_service):
    target = await _add(notes_scope_service, "Target", "hub")
    linker = await _add(
        notes_scope_service, "Linker", f"[t](note://{target}) still here"
    )
    doomed = await _add(
        notes_scope_service, "Doomed", f"[t](note://{target}) going away"
    )
    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=doomed, user_id=USER_ID
    )
    await notes_scope_service.delete_note(
        scope="local_note",
        note_id=doomed,
        version=detail["version"],
        user_id=USER_ID,
    )

    backlinks = await notes_scope_service.list_note_backlinks(
        scope="local_note", note_id=target, user_id=USER_ID
    )

    assert [row["id"] for row in backlinks] == [linker]


@pytest.mark.asyncio
async def test_backlinks_are_bounded_by_the_requested_limit(notes_scope_service):
    target = await _add(notes_scope_service, "Target", "hub")
    for index in range(6):
        await _add(
            notes_scope_service, f"Linker {index}", f"[t](note://{target}) body"
        )

    backlinks = await notes_scope_service.list_note_backlinks(
        scope="local_note", note_id=target, user_id=USER_ID, limit=4
    )

    assert len(backlinks) == 4


@pytest.mark.asyncio
async def test_a_wildcard_in_the_note_id_is_matched_literally(notes_scope_service):
    """The id reaches SQL as a bound parameter with LIKE wildcards escaped.

    A ``%``/``_`` reaching an unescaped LIKE pattern would turn "notes that
    link here" into "every note that links anywhere".
    """
    real = await _add(notes_scope_service, "Real target", "hub")
    await _add(notes_scope_service, "Links elsewhere", f"[x](note://{real})")

    backlinks = await notes_scope_service.list_note_backlinks(
        scope="local_note", note_id="%", user_id=USER_ID
    )

    assert backlinks == []


@pytest.mark.asyncio
async def test_a_prefix_of_another_note_id_is_not_a_backlink(notes_scope_service):
    """``note://abc`` must not match a link to ``note://abcdef``."""
    target = await _add(notes_scope_service, "Target", "hub")
    await _add(
        notes_scope_service, "Longer link", f"[x](note://{target}0123456789)"
    )

    backlinks = await notes_scope_service.list_note_backlinks(
        scope="local_note", note_id=target, user_id=USER_ID
    )

    assert backlinks == []
