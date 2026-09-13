"""Real-DB coverage for the note-backlink query (task-32145).

The Obsidian importer rewrites a resolvable ``[[wikilink]]`` as
``[[target|title]](note://<note_id>)`` (``note_import_plan_models.
rewrite_wikilinks``; task-32263 moved the display text back inside the
wikilink and kept the ``(note://<id>)`` tail this query needs),
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
async def test_backlinks_refuse_every_scope_but_local(notes_scope_service):
    """A scope with no ``note://`` links must refuse, not answer zero.

    Only local notes carry the importer's link form. Answering ``[]`` would
    reach Info as "Linked from (0) — no notes link here yet", a claim nothing
    checked; the raise lands in the controller's ``failed`` path, whose
    header copy is pinned by
    ``Tests/UI/test_library_notes_riders_backlinks.py::
    test_a_failed_backlink_lookup_does_not_claim_there_are_none``. Same
    refusal ``list_deleted_notes`` gives for the same condition.
    """
    for scope in ("server_note", "workspace"):
        with pytest.raises(ValueError):
            await notes_scope_service.list_note_backlinks(
                scope=scope, note_id="anything", user_id=USER_ID
            )


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


# --- task-32186: the answer comes from the persisted note_links relation ---
#
# The query above used to read every active body. These pin the relation's
# side of the contract: it tracks edits, it survives the round trip through
# Trash, and the lookup no longer scans the corpus.


def _connection(notes_scope_service):
    return notes_scope_service.local_notes_service._get_db(USER_ID).get_connection()


@pytest.mark.asyncio
async def test_editing_a_body_adds_and_removes_its_backlinks(notes_scope_service):
    """An edit is reflected the next time Info is opened (AC #3)."""
    first = await _add(notes_scope_service, "First target", "hub one")
    second = await _add(notes_scope_service, "Second target", "hub two")
    linker = await _add(notes_scope_service, "Linker", f"[a](note://{first})")

    assert [row["id"] for row in await _backlinks(notes_scope_service, first)] == [
        linker
    ]

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=linker, user_id=USER_ID
    )
    await notes_scope_service.save_note(
        scope="local_note",
        title="Linker",
        content=f"[b](note://{second})",
        note_id=linker,
        version=detail["version"],
        user_id=USER_ID,
    )

    assert await _backlinks(notes_scope_service, first) == []
    assert [row["id"] for row in await _backlinks(notes_scope_service, second)] == [
        linker
    ]


@pytest.mark.asyncio
async def test_a_restored_note_brings_its_backlinks_back(notes_scope_service):
    """Trash hides a linker's backlink; restoring it returns the row."""
    target = await _add(notes_scope_service, "Target", "hub")
    linker = await _add(notes_scope_service, "Linker", f"[t](note://{target})")

    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=linker, user_id=USER_ID
    )
    await notes_scope_service.delete_note(
        scope="local_note",
        note_id=linker,
        version=detail["version"],
        user_id=USER_ID,
    )
    assert await _backlinks(notes_scope_service, target) == []

    await notes_scope_service.restore_note(
        scope="local_note",
        note_id=linker,
        version=detail["version"] + 1,
        user_id=USER_ID,
    )

    assert [row["id"] for row in await _backlinks(notes_scope_service, target)] == [
        linker
    ]


@pytest.mark.asyncio
async def test_the_lookup_does_not_scan_the_note_corpus(notes_scope_service):
    """The budget: an indexed search on both tables, never a SCAN of notes.

    A wall-clock budget would be a flake on shared CI; the query plan is the
    durable form of "does not read every body". ``SCAN notes`` reappearing
    here is exactly the regression task-32186 fixed.
    """
    target = await _add(notes_scope_service, "Target", "hub")
    await _add(notes_scope_service, "Linker", f"[t](note://{target})")

    # The plan is only evidence when captured the way production runs it:
    # ChaChaNotes_DB.py never runs ANALYZE, so no user's database has a
    # sqlite_stat1 and the planner works from default estimates
    # (scripts/check_index_plan_pins.py, TASK-21593).
    assert _connection(notes_scope_service).execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'"
    ).fetchone() is None, (
        "this fixture must reproduce the no-stats production state; a plan "
        "captured with sqlite_stat1 present is not the plan users run"
    )

    plan = [
        str(row["detail"])
        for row in _connection(notes_scope_service).execute(
            "EXPLAIN QUERY PLAN " + CharactersRAGDB._BACKLINK_SOURCES_SQL,
            (target, 50),
        )
    ]

    assert any("note_links" in step and "idx_note_links_target" in step for step in plan)
    assert not any(step.startswith("SCAN notes") for step in plan), plan


@pytest.mark.asyncio
async def test_a_title_only_save_leaves_the_backlinks_alone(notes_scope_service):
    """An update that carries no body must not touch the relation.

    Pins the ``if "content" in update_data`` around the link rewrite in
    ``_update_note_with_cursor``: a rename (or any metadata-only update)
    neither clears the linker's edges nor fails for want of a body.
    """
    target = await _add(notes_scope_service, "Target", "hub")
    linker = await _add(notes_scope_service, "Linker", f"[t](note://{target})")

    db = notes_scope_service.local_notes_service._get_db(USER_ID)
    detail = await notes_scope_service.get_note_detail(
        scope="local_note", note_id=linker, user_id=USER_ID
    )
    assert db.update_note(
        linker, {"title": "Linker, renamed"}, expected_version=detail["version"]
    ) is True

    assert [row["id"] for row in await _backlinks(notes_scope_service, target)] == [
        linker
    ]


async def _backlinks(service, note_id):
    return await service.list_note_backlinks(
        scope="local_note", note_id=note_id, user_id=USER_ID
    )
@pytest.mark.asyncio
async def test_backlinks_find_the_display_text_link_form_the_importer_writes(
    notes_scope_service,
):
    """task-32263: the stored spelling changed; the containment probe must not.

    The link the importer actually writes is built here by the production
    rewrite, not typed by hand, so a future change to that spelling fails
    this test instead of silently emptying every note's Backlinks panel.
    """
    from tldw_chatbook.Notes.note_import_plan_models import (
        ParsedNotePayload,
        rewrite_wikilinks,
    )

    target = await _add(notes_scope_service, "Zettelkasten — overview", "hub")
    linked = rewrite_wikilinks(
        ParsedNotePayload(
            title="Library review",
            content="Related: [[Reading/Zettelkasten]].",
            wikilinks=("Reading/Zettelkasten",),
        ),
        {"reading/zettelkasten": target},
        titles={"reading/zettelkasten": "Zettelkasten — overview"},
    )
    await _add(notes_scope_service, "Library review", linked.content)

    backlinks = await notes_scope_service.list_note_backlinks(
        scope="local_note", note_id=target, user_id=USER_ID
    )

    assert [row["title"] for row in backlinks] == ["Library review"]
