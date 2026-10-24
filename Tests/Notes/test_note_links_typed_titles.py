"""TASK-34000.11: a typed ``[[Title]]`` records a backlink; the ``[[`` picker's
candidates and pure helpers.

Review finding N-08 (notes-library-ux-review-2026-10-02): a power user typed
``See [[Trip planning: Lisbon]]`` and Lisbon's Info kept reading
"Linked from (1)". Only the ``(note://<id>)`` tail ever recorded an edge.

The edge tests run the real ``CharactersRAGDB`` through the real
``NotesInteropService``/``NotesScopeService`` wiring over a temp file, the way
``test_note_backlink_query.py`` does, because the rule under test -- exactly
one LIVE title match, case-insensitive, the body byte-identical (ADR-027) --
is a property of the SQL ``replace_note_links`` runs, which a fake cannot
have. Pure helpers get pure tests; the Hypothesis properties live in the
sibling ``test_note_links_typed_titles_properties.py`` (outside the fast
lane, which installs no Hypothesis).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

# The real CharactersRAGDB opens through the profile bootstrap; outside a
# kept profile it raises RecoveryRequired(raw_source_selection_changed).
# PR #3234's PR Fast Lane failed exactly that way -- the same marker the
# FAST_LANE chatbook round trip carries (test_chatbook_keywords_round_trip).
pytestmark = pytest.mark.bootstrap_profile

USER_ID = "typed-link-user"
LISBON = "Trip planning: Lisbon"
TYPED = "See [[Trip planning: Lisbon]] for the itinerary."


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    db_dir = tmp_path / "chachanotes"
    db_dir.mkdir()
    return CharactersRAGDB(str(db_dir / "unified.db"), client_id="typed-link-app")


@pytest.fixture
def notes_scope_service(tmp_path, db) -> NotesScopeService:
    """Real ``NotesScopeService`` over a real (temp-file) ChaChaNotes DB."""
    interop = NotesInteropService(
        base_db_directory=tmp_path / "chachanotes",
        api_client_id="typed-link-client",
        global_db_to_use=db,
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


async def _backlink_titles(service, note_id) -> list[str]:
    rows = await service.list_note_backlinks(
        scope="local_note", note_id=note_id, user_id=USER_ID
    )
    return [row["title"] for row in rows]


async def _update(service, db, note_id, *, title=None, content=None):
    row = db.get_note_by_id(note_id)
    assert row is not None
    return await service.save_note(
        scope="local_note",
        title=row["title"] if title is None else title,
        content=row["content"] if content is None else content,
        note_id=note_id,
        version=row["version"],
        user_id=USER_ID,
    )


@pytest.mark.asyncio
async def test_a_typed_title_that_matches_exactly_one_note_records_a_backlink(
    notes_scope_service, db
):
    """AC#2 / AC#5: the review's exact body links Lisbon, and the body is
    stored byte-for-byte as typed (ADR-027: no rewrite at persistence)."""
    lisbon = await _add(notes_scope_service, LISBON, "itinerary")
    await _add(notes_scope_service, "Index", "hub")

    linker = await _add(notes_scope_service, "Power user note", TYPED)

    assert await _backlink_titles(notes_scope_service, lisbon) == ["Power user note"]
    assert db.get_note_by_id(linker)["content"] == TYPED


@pytest.mark.asyncio
async def test_zero_or_two_title_matches_record_nothing_and_leave_the_text_as_typed(
    notes_scope_service, db
):
    """AC#2 / AC#5 and Review Focus 4.

    Arms, in order: no such title (negative control on the base, green
    there too); two LIVE "Weekly" notes (negative control); a deleted
    "Zebra" twin beside a live one -> ONE edge (RED on the base: the live
    twin is the only live match); an alias (RED on the base); a code span
    (negative control); an embed (negative control); a self-link (negative
    control).
    """
    lisbon = await _add(notes_scope_service, LISBON, "itinerary")
    nobody = await _add(notes_scope_service, "Nobody links here", "[[Nobody]] is no note")
    assert db.get_note_by_id(nobody)["content"] == "[[Nobody]] is no note"
    # Nothing is titled "Nobody": no edge from anywhere.
    for note_id in (lisbon, nobody):
        assert await _backlink_titles(notes_scope_service, note_id) == []

    weekly_a = await _add(notes_scope_service, "Weekly", "a")
    weekly_b = await _add(notes_scope_service, "weekly", "b")
    ambiguous = await _add(notes_scope_service, "Ambiguous", "See [[Weekly]].")
    assert await _backlink_titles(notes_scope_service, weekly_a) == []
    assert await _backlink_titles(notes_scope_service, weekly_b) == []
    assert db.get_note_by_id(ambiguous)["content"] == "See [[Weekly]]."

    zebra_gone = await _add(notes_scope_service, "Zebra", "old")
    gone_row = db.get_note_by_id(zebra_gone)
    assert db.soft_delete_note(zebra_gone, gone_row["version"]) is True
    zebra_live = await _add(notes_scope_service, "Zebra", "new")
    await _add(notes_scope_service, "Stripes", "Case-folded: [[zebra]].")
    assert await _backlink_titles(notes_scope_service, zebra_live) == ["Stripes"]

    await _add(notes_scope_service, "Alias user", "Read [[Trip planning: Lisbon|the trip]].")
    assert await _backlink_titles(notes_scope_service, lisbon) == ["Alias user"]

    await _add(notes_scope_service, "Code sample", "`[[Trip planning: Lisbon]]` is sample text")
    await _add(notes_scope_service, "Embed user", "![[Trip planning: Lisbon]]")
    assert await _backlink_titles(notes_scope_service, lisbon) == ["Alias user"]

    selfie = await _add(notes_scope_service, "Loop", "I link [[Loop]] to myself")
    assert await _backlink_titles(notes_scope_service, selfie) == []


@pytest.mark.asyncio
async def test_the_title_match_folds_case_for_ascii_letters_only(notes_scope_service):
    """Fix round 1, M-3 (ruling: document, don't widen): ``COLLATE NOCASE``
    folds A-Z only. ``[[WEEKLY]]`` finds "Weekly"; ``[[élan]]`` does NOT find
    "Élan" (Unicode case folding is TASK-34000.74). Pinned so the guide's
    wording and the behaviour cannot drift apart silently."""
    weekly = await _add(notes_scope_service, "Weekly", "a")
    await _add(notes_scope_service, "Shouter", "see [[WEEKLY]]")
    assert await _backlink_titles(notes_scope_service, weekly) == ["Shouter"]
    elan = await _add(notes_scope_service, "Élan", "x")
    await _add(notes_scope_service, "Accented", "see [[élan]]")
    assert await _backlink_titles(notes_scope_service, elan) == []
    await _add(notes_scope_service, "Exact", "see [[Élan]]")
    assert await _backlink_titles(notes_scope_service, elan) == ["Exact"]


@pytest.mark.asyncio
async def test_editing_away_a_typed_title_removes_its_backlink_and_a_title_only_save_leaves_it(
    notes_scope_service, db
):
    """The task-32186 contract extended to typed titles: the edges follow the
    body on every save; a save that changes only the title keeps them."""
    lisbon = await _add(notes_scope_service, LISBON, "itinerary")
    linker = await _add(notes_scope_service, "Linker", TYPED)
    assert await _backlink_titles(notes_scope_service, lisbon) == ["Linker"]

    await _update(notes_scope_service, db, linker, title="Linker renamed")
    assert await _backlink_titles(notes_scope_service, lisbon) == ["Linker renamed"]

    await _update(notes_scope_service, db, linker, content="No link any more.")
    assert await _backlink_titles(notes_scope_service, lisbon) == []


def test_bare_wikilink_titles_and_the_completion_context_are_pure():
    """The four pure helpers behind the picker and the edge writer."""
    from tldw_chatbook.Notes.note_import_plan_models import (
        WIKILINK_SCAN,
        bare_wikilink_titles,
        note_link_entries,
        stored_note_link,
        wikilink_completion_context,
    )

    # bare_wikilink_titles: trimmed, de-duplicated as written, no code spans,
    # no tailed links, alias ignored, heading suffix outside the target.
    body = (
        "[[ A ]] and [[A]] and [[B|alias]] and [[C#heading]] and `[[D]]` and "
        "[[E]](note://e) and ![[F]] and [[]] and [[ ]]\n```\n[[G]]\n```\n[[a]]"
    )
    assert bare_wikilink_titles(body) == ["A", "B", "C", "a"]
    assert bare_wikilink_titles("") == []

    # wikilink_completion_context: on the cursor's line only.
    assert wikilink_completion_context("plain text", (0, 5)) is None
    assert wikilink_completion_context("See [[", (0, 6)) == ("", (0, 4))
    assert wikilink_completion_context("See [[Lis", (0, 9)) == ("Lis", (0, 4))
    assert wikilink_completion_context("x\nSee [[Lis", (1, 9)) == ("Lis", (1, 4))
    # Closed, aliased, or a fresh `[` after the opener: no context.
    assert wikilink_completion_context("[[Lis]]", (0, 7)) is None
    assert wikilink_completion_context("[[Lis|al", (0, 8)) is None
    assert wikilink_completion_context("[[Lis[", (0, 6)) is None
    # Final review C-1(a): a caret INSIDE an already-closed link -- bare or
    # stored -- is no completion, so navigation keys can never open the
    # picker over an existing link and rewrite it to another note.
    stored = "See [[Trip planning: Lisbon]](note://x) today."
    assert wikilink_completion_context(stored, (0, 10)) is None
    assert wikilink_completion_context(stored, (0, 30)) is None
    assert wikilink_completion_context("See [[Lisbon]] now", (0, 9)) is None
    assert wikilink_completion_context("[[Lis]] then [[Tri", (0, 18)) == ("Tri", (0, 13))
    # Not an embed, not inside backticks on that line, not inside an open fence.
    assert wikilink_completion_context("![[Lis", (0, 6)) is None
    assert wikilink_completion_context("`[[Lis", (0, 6)) is None
    assert wikilink_completion_context("`code` [[Lis", (0, 12)) == ("Lis", (0, 7))
    assert wikilink_completion_context("```\n[[Lis", (1, 5)) is None
    assert wikilink_completion_context("```\n```\n[[Lis", (2, 5)) == ("Lis", (2, 0))
    # The cursor before the opener sees nothing.
    assert wikilink_completion_context("[[Lis", (0, 1)) is None
    # Out-of-range cursors never raise.
    assert wikilink_completion_context("[[", (5, 99)) is None
    assert wikilink_completion_context("[[", (0, 99)) == ("", (0, 0))

    # stored_note_link: always ONE WIKILINK_SCAN match whose tail is the id.
    link = stored_note_link("A [b] | c", "id-1")
    matches = [m for m in WIKILINK_SCAN.finditer(link) if m.group("code") is None]
    assert len(matches) == 1 and matches[0].group(0) == link
    assert matches[0].group("link") == "(note://id-1)"
    assert matches[0].group("alias") is None
    assert stored_note_link("Trip planning: Lisbon", "x") == "[[Trip planning: Lisbon]](note://x)"
    assert stored_note_link("back\\slash", "x").startswith("[[back\\\\slash]]")
    for hostile in ("a]]b", "a#b", "a^b", "a|b", "a[b"):
        hostile_link = stored_note_link(hostile, "x")
        found = [m for m in WIKILINK_SCAN.finditer(hostile_link) if m.group("code") is None]
        assert len(found) == 1 and found[0].group(0) == hostile_link, hostile

    # note_link_entries: (label, id) in document order, de-duplicated by id,
    # never from a code span.
    stored = (
        f"{stored_note_link('Lisbon', 'l')} then [[Idx|Index]](note://i) then "
        f"{stored_note_link('Lisbon again', 'l')} and `[[Code]](note://c)`"
    )
    assert note_link_entries(stored) == (("Lisbon", "l"), ("Index", "i"))
    assert note_link_entries("no links") == ()


@pytest.mark.asyncio
async def test_list_note_link_candidates_narrows_by_title_and_lists_recent_notes_for_an_empty_query(
    notes_scope_service, db
):
    """The picker's rows: eight most recent for ``""`` (never the open note),
    a title prefix, any word order through Task 3's rule, and a query the
    rule refuses (``"``) lists nothing -- no SQL runs for it."""
    import time

    ids = []
    for index in range(10):
        ids.append(await _add(notes_scope_service, f"Filler {index:02d}", "x"))
        time.sleep(0.002)
    lisbon = await _add(notes_scope_service, LISBON, "itinerary")
    time.sleep(0.002)
    open_note = await _add(notes_scope_service, "The open note", "typing here")

    async def candidates(query):
        rows = await notes_scope_service.list_note_link_candidates(
            scope="local_note",
            query=query,
            exclude_note_id=open_note,
            user_id=USER_ID,
        )
        return [row["title"] for row in rows]

    recent = await candidates("")
    assert len(recent) == 8
    assert recent[0] == LISBON
    assert "The open note" not in recent
    assert recent[1:] == [f"Filler {index:02d}" for index in range(9, 2, -1)]

    assert await candidates("Lis") == [LISBON]
    # Any order, LAST word a prefix (Task 3's rule): "trip lisb" finds
    # Lisbon; "lisb trip" does not, because a non-last word is matched whole.
    assert await candidates("trip lisb") == [LISBON]
    assert await candidates("lisbon trip") == [LISBON]
    assert await candidates("lisb trip") == []
    # Ten matches, eight rows: rank ties, then title order (the LIMIT).
    assert await candidates("Fill") == [f"Filler {index:02d}" for index in range(8)]
    # A one-character last word is matched whole (Task 3's rule), so it
    # cannot fan out over "00".."09".
    assert await candidates("Filler 0") == []
    assert await candidates('"') == []
    assert await candidates("zzzz") == []
